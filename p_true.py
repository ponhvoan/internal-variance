#!/usr/bin/env python3
import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from utils.dataset import prepare_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

from sklearn.metrics import roc_auc_score, average_precision_score
from utils.misc import fpr_at_95_tpr

def candidate_first_token_ids(tokenizer, letter="A"):
    variants = (letter, f" {letter}", f"\n{letter}", f"({letter})", f" ({letter})", f"\n({letter})")
    # variants = (letter, f" {letter}", f"\n{letter}")
    ids = set()
    for v in variants:
        toks = tokenizer.encode(v, add_special_tokens=False)
        if toks:
            ids.add(toks[0])
    return sorted(ids)


@torch.no_grad()
def p_true_for_prompts(model, tokenizer, prompts, temperature=1.0):
    device = next(model.parameters()).device
    enc = tokenizer(prompts, padding=True, return_tensors="pt")
    enc = {k: v.to(device) for k, v in enc.items()}  # input_ids, attention_mask
    logits = model(**enc).logits

    # Next-token logits occur at the last non-pad position in each sequence
    last_idx = enc["attention_mask"].sum(dim=1) - 1
    B = logits.size(0)
    next_logits = logits[torch.arange(B, device=device), last_idx]

    probs = F.softmax(next_logits / temperature, dim=-1)

    # Sum probability mass over plausible first-token IDs for 'A'
    a_ids = torch.tensor(candidate_first_token_ids(tokenizer, "A"), device=device)
    p_true = probs.index_select(1, a_ids).sum(dim=1)
    return p_true.detach().cpu().numpy() 


def main():
    parser = argparse.ArgumentParser(description="Compute P(True) from p_true.txt prompts.")
    parser.add_argument("--model", type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--dataset_name', type=str, default='true_false')
    parser.add_argument('--topic', type=str, default='animals')
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    # Load model/tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto",
    ).eval()
    # ensure a pad token exists
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.generation_config.pad_token_id = tokenizer.pad_token_id

    # Read data
    dataset, formatter = prepare_dataset(args.dataset_name, args.topic)
    prompts, gt = formatter(args.dataset_name, dataset, 'p_true')
    

    # Batched scoring
    scores = []
    for i in range(0, len(prompts), args.batch_size):
        batch_prompts = prompts[i:i + args.batch_size]
        s = p_true_for_prompts(model, tokenizer, batch_prompts, temperature=args.temperature)
        scores.append(s)
    scores = np.concatenate(scores, axis=0)  # shape: (N_statements,)

    # Save
    if args.save:
        out_dir = f"outputs/{args.dataset_name}/{args.topic}/{args.model}"
        os.makedirs(out_dir, exist_ok=True)
        np.save(f"{out_dir}/p_true", scores)
        print(f"Saved {len(scores)} P(True) scores to: {out_dir}")
    
    # Show metrics
    auc = roc_auc_score(gt, scores)
    fpr95 = fpr_at_95_tpr(gt, scores)
    aupr = average_precision_score(gt, scores)
    print(f'######### {args.model} ----- {args.dataset_name}/{args.topic} #########\n'
          f'auc: {auc}, fpr@95: {fpr95}, aupr: {aupr}')

if __name__ == "__main__":
    out_dir = f"outputs/{args.dataset_name}/{args.topic}/{args.model}"
    main()
