import torch
import math
import torch.nn as nn
import torch.nn.functional as F

""" RNN Classifier """

class RNNClassifier(nn.Module):
    def __init__(self, input_dim: int = 1, hidden_dim: int = 64, n_layers: int = 1):
        super().__init__()
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers=n_layers, batch_first=True)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, x, mask):
        lengths = torch.sum(~mask, axis=1)
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, h_n = self.rnn(packed)
        logits = self.out(h_n[-1])  # last layer's hidden state
        return logits.squeeze(1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) *
                        -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


""" Transformer """
class MaskedMeanPool(nn.Module):
    def forward(self, h, pad_mask):
        valid = ~pad_mask
        m = valid.unsqueeze(-1).float()
        return (h * m).sum(1) / m.sum(1).clamp_min(1e-6)

class AttnPool(nn.Module):
    def __init__(self, d, hidden=128):
        super().__init__()
        self.scorer = nn.Sequential(nn.Linear(d, hidden), nn.Tanh(), nn.Linear(hidden, 1))
    def forward(self, h, pad_mask):
        s = self.scorer(h).squeeze(-1) 
        s = s.masked_fill(pad_mask, -1e9)
        a = s.softmax(dim=1)
        z = (h * a.unsqueeze(-1)).sum(1)
        return z, a

class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 4,
        num_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
        pooled: str = "attn",  # "mean" | "cls"
    ):
        super().__init__()
        self.d_model = d_model
        self.input_proj = nn.Linear(input_dim, d_model)
        self.posenc = PositionalEncoding(d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.pooled = pooled
        if pooled == "cls":
            self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))
        self.head = nn.Linear(d_model, 1)

        nn.init.xavier_uniform_(self.head.weight)
        
    def forward(self, x, mask):
        B, L, _ = x.shape
        h = self.input_proj(x)
        h = self.posenc(h)

        if self.pooled == "cls":
            cls = self.cls_token.expand(B, -1, -1)
            h = torch.cat([cls, h], dim=1)
            mask = F.pad(mask, (0, 1), value=False)

        h = self.encoder(h, src_key_padding_mask=mask)

        if self.pooled == "cls":
            seq_emb = h[:, 0]
        elif self.pooled == "attn":
            seq_emb, _ = AttnPool(self.d_model).to(h.device)(h, mask)
        else:
            lengths = (~mask).sum(1, keepdim=True)
            seq_emb = (h * (~mask).unsqueeze(-1)).sum(1) / lengths

        logit = self.head(seq_emb).squeeze(-1)  
        return logit
