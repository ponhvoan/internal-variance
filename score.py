import numpy as np
from scipy.stats import directional_stats

import torch
import torch.nn.functional as F
from utils.misc import to_cpu

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CoEScore():
    allowed_token = {'last', 'mean'}
    def __init__(self, hidden_states, which='mean'):
        self.hidden_states = hidden_states
        self.which = which
        try:
            np.ndim(hidden_states[0])
        except TypeError:
            hidden_states = to_cpu(hidden_states)
        if np.ndim(hidden_states[0])==4:
            self.num_tokens = len(hidden_states)
            self.num_layers = len(hidden_states[0])
        elif np.ndim(hidden_states[0])==3:
            self.num_tokens = len(hidden_states[0][0]) - 1
            self.num_layers = len(hidden_states)
        self.hs_layer = self._extract_hs()
            
    
    def _extract_hs(self):
        if self.which=='last':
            layer_last = (torch.stack(self.hidden_states[-1]).squeeze(1).squeeze(1)) 
            hs_layer = layer_last.cpu().numpy().astype(np.float32)
            hs_layer = hs_layer[None,...]
            return hs_layer
        
        elif self.which=='mean':
            layer_means = (
                torch.stack([torch.stack(hs).squeeze(1).squeeze(1) if i>=1 else torch.stack(hs)[:,:,-1,:].squeeze(1) 
                            for i, hs in enumerate(self.hidden_states)],
                            dim=0)
                .mean(dim=0)
            ) 
            hs_layer = layer_means.cpu().numpy().astype(np.float32)
            hs_layer = hs_layer[None,...]
            return hs_layer
        
        elif self.which not in self.__class__.allowed_token:
            raise ValueError(f"This token selection {self.which} is not supported for CoE methods.")
    
    # Implement CoE features https://github.com/Alsace08/Chain-of-Embedding/blob/master/score.py
    def coe_ang(self):
        eps = 1e-8
        hs = self.hs_layer

        num_last_first = (hs[:, -1] * hs[:, 0]).sum(axis=-1)
        denom_last_first = (np.linalg.norm(hs[:, -1], axis=-1, ord=2) *
                            np.linalg.norm(hs[:,  0], axis=-1, ord=2) + eps)
        cos_beta = np.clip(num_last_first / denom_last_first, -1.0, 1.0)
        norm_denominator = np.arccos(cos_beta)

        a, b = hs[:, 1:], hs[:, :-1]
        num_ab = (a * b).sum(axis=-1)
        denom_ab = (np.linalg.norm(a, axis=-1, ord=2) * 
                    np.linalg.norm(b, axis=-1, ord=2) + eps)
        cos_alpha = np.clip(num_ab / denom_ab, -1.0, 1.0)
        alpha = np.arccos(cos_alpha)

        al_semdiff_norm = alpha / norm_denominator[:, None]
        al_semdiff_ave  = al_semdiff_norm.mean(axis=-1)
            
        return al_semdiff_norm, al_semdiff_ave


    def coe_mag(self):
        hs = self.hs_layer
        eps = 1e-8
            
        norm_denominator = np.linalg.norm(hs[:, -1] - hs[:, 0], axis=-1, ord=2) + eps
        diff = hs[:, 1:] - hs[:, :-1]
        diff_norm = np.linalg.norm(diff, axis=-1, ord=2)
        
        al_repdiff_norm = diff_norm / norm_denominator[:, None]
        al_repdiff_ave  = al_repdiff_norm.mean(axis=-1)

        return al_repdiff_norm, al_repdiff_ave
    
    def compute_CoE_R(self):
        _, ang_ave = self.coe_ang()
        _, diff_ave = self.coe_mag()
        if self.which!='per_token':
            ang_ave, diff_ave = np.asarray(ang_ave)[None,...], np.asarray(diff_ave)[None,...]
        coe_r = diff_ave - ang_ave
        return coe_r
    
    def compute_CoE_C(self):
        ang_norm, _ = self.coe_ang()
        diff_norm, _ = self.coe_mag()
        if self.which!='per_token':
            ang_norm, diff_norm = ang_norm[None,...], diff_norm[None, ...]
        x = diff_norm * np.cos(ang_norm)
        y = diff_norm * np.sin(ang_norm)
        x_ave = np.mean(x, axis=-1)
        y_ave = np.mean(y, axis=-1)
        coe_c = np.sqrt(x_ave**2 + y_ave**2)
        return coe_c

class VarianceScore(CoEScore):
    allowed_token = CoEScore.allowed_token | {'per_token', 'first'}
    def __init__(self, hidden_states, which='per_token'):
        super().__init__(hidden_states, which)

    def _extract_hs(self):
        if self.which=='last':
            layer_last = (torch.stack(self.hidden_states[-1]).squeeze(1).squeeze(1)) 
            hs_layer = layer_last.cpu().numpy().astype(np.float32)
            hs_layer = hs_layer[None,...]
        
        elif self.which=='first':
            layer_first = (torch.stack(self.hidden_states[0]).squeeze(1))[:,-1,:]
            hs_layer = layer_first.cpu().numpy().astype(np.float32)
            hs_layer = hs_layer[None,...]
        
        elif self.which=='mean':
            layer_means = (
                torch.stack([torch.stack(hs).squeeze(1).squeeze(1) if i>=1 else torch.stack(hs)[:,:,-1,:].squeeze(1) 
                            for i, hs in enumerate(self.hidden_states)],
                            dim=0)
                .mean(dim=0)
            ) 
            hs_layer = layer_means.cpu().numpy().astype(np.float32)
            hs_layer = hs_layer[None,...]
        
        elif self.which=='per_token':
            layer_all = torch.stack([torch.stack(hs).squeeze(1).squeeze(1) if i>=1 else torch.stack(hs)[:,:,-1,:].squeeze(1) 
                         for i, hs in enumerate(self.hidden_states)],
                        dim=0)
            hs_layer = layer_all.cpu().numpy().astype(np.float32)
            
        return hs_layer
    
    def circ_variance(self):
        # Compute circular variance
        layer_mat = self.hs_layer
        dirstats = directional_stats(layer_mat, axis=1, normalize=True)
        var = 1 - dirstats.mean_resultant_length
        
        return var
     
    def covdet(self):
        embeddings = self.hs_layer
        centred = embeddings - embeddings.mean(axis=-1, keepdims=True)
        cov = centred @ centred.transpose(0, 2, 1) / (centred.shape[-1] - 1)
        reg_cov = cov + 1e-3 * np.stack([np.eye(cov.shape[1])]*cov.shape[0], axis=0)
        eigvals = np.linalg.eigvalsh(reg_cov)
        covdet = np.log(np.clip(eigvals, 1e-8, None)).mean(-1)
        
        return covdet
    
# Adapted from https://github.com/Alsace08/Chain-of-Embedding/blob/master/score.py, and vectorised for parallel computation
class OutputScore():
    def __init__(self, logits, per_token=False):
        self.logits = torch.stack(
            [torch.as_tensor(logits[t][0], device=device, dtype=torch.float32)
             for t in range(len(logits))],
            dim=0
        )
        self.probs = F.softmax(self.logits, dim=1)
        self.per_token = per_token
        
    def compute_maxprob(self):
        max_prob = torch.mean(torch.max(self.probs, dim=1)[0]).item()
        return max_prob

    def compute_ppl(self):
        seq_ppl = torch.log(torch.clip(torch.max(self.probs, dim=1)[0], min=1e-8))
        ppl = -torch.mean(seq_ppl).item()
        return ppl

    def compute_entropy(self):
        seq_entropy = torch.sum(-self.probs * torch.log(torch.clip(self.probs, min=1e-8)), dim=1)
        if self.per_token:
            seq_entropy = seq_entropy.detach().cpu().numpy()
        else:
            seq_entropy = torch.mean(seq_entropy).item()
        return seq_entropy
    
    def compute_tempscale(self, temperature = 0.7):
        probs = F.softmax(self.logits / temperature, dim=-1)
        max_scaled_prob = probs.max(dim=-1).values.mean().item()
        return max_scaled_prob
    
    def compute_energy(self, temperature=0.7):
        energy_per_token = -temperature * torch.logsumexp(self.logits / temperature, dim=-1)
        return float(energy_per_token.mean().item())
    