import os
import pickle
import argparse
from tqdm import tqdm

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from utils.dataset import prepare_dataset
from utils.generate import Inference
from utils.misc import fpr_at_95_tpr
from score import CoEScore, VarianceScore, OutputScore

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--dataset_name', type=str, default='math')
    parser.add_argument('--topic', type=str, default=None)
    parser.add_argument('--max_new_tokens', type=int, default=1024)
    parser.add_argument('--data_portion', type=float, default=0.01)
    parser.add_argument('--prompt_type', type=str, default='math')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    args = parser.parse_args()
    np.random.seed(0)

    # Config
    max_new_tokens = args.max_new_tokens
    out_dir = f"outputs/{args.dataset_name}/{args.topic}/{args.model}" if args.save else None
    out_dir = out_dir.replace('/None', '') if 'None' in out_dir else out_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    dataset, formatter = prepare_dataset(args.dataset_name, args.topic)
    prompts, gt = formatter(args.dataset_name, dataset, args.prompt_type)
    gt = 1 - np.array(gt) # flip label -> 0: true, 1: false
    
    data_len = int(args.data_portion*len(gt))
    prompts = prompts[:data_len]
    gt = gt[:data_len]
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    # ensure a pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Evaluate 
    inference = Inference(model, tokenizer, args.dataset_name, prompts, gt, out_dir, args.prompt_type, max_new_tokens)
    all_hs, all_logits, labels, gt, _ = inference.single_inference()
    all_hs = all_hs[:data_len]
    all_logits = all_logits[:data_len]                    
    
    scores = {'max_prob': [],
              'perplexity': [],
              'entropy': [],
              'tempscaled': [],
              'energy': [],
              'pairwise_dissimilarity': [],
              'circ_variance':[],
              'eigenscore': [],
              'coe_angles': [],
              'coe_mag': [],
              'coe_r': [],
              'coe_c': []
              }
    
    for idx in tqdm(range(len(all_hs)), desc='Computing scores'):
        
        output_scorer = OutputScore(all_logits[idx])
        ppl = output_scorer.compute_ppl()
        ent = output_scorer.compute_entropy()
        maxp = output_scorer.compute_maxprob()
        max_scaledp = output_scorer.compute_tempscale()
        energy = output_scorer.compute_energy()
        scores['perplexity'].append(np.mean(ppl))
        scores['entropy'].append(np.mean(ent))
        scores['max_prob'].append(-np.mean(maxp))
        scores['tempscaled'].append(-np.mean(max_scaledp))
        scores['energy'].append(np.mean(energy))
        
        var_scorer = VarianceScore(all_hs[idx], which='first')
        dissim = var_scorer.pairwise_dissimilarities()
        var = var_scorer.circ_variance()
        eigenscore = var_scorer.eigenscore()
        scores['pairwise_dissimilarity'].append(np.mean(dissim))
        scores['circ_variance'].append(-np.mean(var))
        scores['eigenscore'].append(-np.mean(eigenscore))
        
        coe_scorer = CoEScore(all_hs[idx], which='mean')
        ang = coe_scorer.coe_ang()[-1]
        mag = -coe_scorer.coe_mag()[-1]
        coe_c = coe_scorer.compute_CoE_C()
        coe_r = coe_scorer.compute_CoE_R()
        scores['coe_angles'].append(np.mean(ang))
        scores['coe_mag'].append(np.mean(mag))
        scores['coe_r'].append(-np.mean(coe_r))
        scores['coe_c'].append(-np.mean(coe_c))
        
    # Compute AUC, FPR@95 scores
    print(f'######### {args.model} ----- {args.dataset_name}/{args.topic} #########')
    results = {'auc': [],
               'fpr@95': [],
               'aupr': []}
    for k in scores.keys():
        scores[k] = np.array(scores[k])
        auc = roc_auc_score(labels, scores[k])
        fpr95 = fpr_at_95_tpr(labels, scores[k])
        aupr = average_precision_score(labels, scores[k])
        results['auc'].append(auc)
        results['fpr@95'].append(fpr95)
        results['aupr'].append(aupr)

    results = pd.DataFrame(results, index=list(scores.keys()))
    print(results)
    
    if args.save:
        hs_tokens = []
        for i in tqdm(range(len(all_hs))):
            curr_hs = np.stack([all_hs[i][j][-1].squeeze().detach().cpu().numpy() if j!=0 else all_hs[i][j][-1][-1][-1].squeeze().detach().cpu().numpy() for j in range(len(all_hs[i]))])
            hs_tokens.append(curr_hs)
        
        lens = [hs.shape[0] for hs in hs_tokens]
        hs_tokens = np.vstack(hs_tokens)
        scaler = StandardScaler()
        hs_scaled = scaler.fit_transform(hs_tokens)
        pca = PCA(n_components=10, random_state=42)
        hs_pca = pca.fit_transform(hs_scaled)
        hs_pca = np.split(hs_pca, np.cumsum(lens)[:-1])
        
        token_vars = []
        for idx in range(len(all_hs)):
            var_scorer = VarianceScore(all_hs[idx], which='per_token', weights=None)
            var = var_scorer.circ_variance()
            token_vars.append(var)
        
        with open(os.path.join(out_dir, f"hsPCA_{args.prompt_type}.pkl"), "wb") as f:
            pickle.dump(hs_pca, f)
        with open(os.path.join(out_dir, f"tokenDict_{args.prompt_type}.pkl"), "wb") as f:
            pickle.dump(token_vars, f)
        np.save(f"{out_dir}/labels", labels)