import os
import re
import pickle
import random

import argparse
import numpy as np

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import roc_auc_score, average_precision_score

from utils.dataset import prepare_dataset
from utils.generate import Inference
from utils.misc import fpr_at_95_tpr

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--dataset_name', type=str, default='true_false')
    parser.add_argument('--topic', type=str, default='animals')
    parser.add_argument('--max_new_tokens', type=int, default=128)
    parser.add_argument('--data_portion', type=float, default=1.0)
    parser.add_argument('--prompt_type', type=str, default='verb')
    parser.add_argument('--save', action='store_true')
    args = parser.parse_args()

    args = parser.parse_args()
    np.random.seed(0)

    # Config
    max_new_tokens = args.max_new_tokens

    dataset, formatter = prepare_dataset(args.dataset_name, args.topic)
    prompts, gt = formatter(args.dataset_name, dataset, args.prompt_type)
    
    data_len = int(args.data_portion*len(prompts))
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
    inference = Inference(model, tokenizer, args.dataset_name, prompts, gt, None, args.prompt_type, max_new_tokens)
    qa_sets = inference.single_inference()[-1]
    
    verbs, labels = [], []
    for idx, pair in enumerate(qa_sets):
        ans = pair['response']
        verb = re.findall(r"\d+\.+\d*", ans)
        if len(verb)>0:
            verb = float(verb[-1])
        else:
            continue
        verbs.append(verb)
        labels.append(gt[idx])
    
    if args.save:
        out_dir = f"outputs/{args.dataset_name}/{args.topic}/{args.model}"
        os.makedirs(out_dir, exist_ok=True)
        np.save(f"{out_dir}/verbs", verbs)
        print(f"Saved {len(verbs)} verb confidence to: {out_dir}")
    # Show metrics
    auc = roc_auc_score(labels, verbs)
    fpr95 = fpr_at_95_tpr(labels, verbs)
    aupr = average_precision_score(labels, verbs)
    print(f'######### {args.model} ----- {args.dataset_name}/{args.topic} #########\n'
          f'Result on {len(labels)} instance.\n'
          f'auc: {auc}, fpr@95: {fpr95}, aupr: {aupr}')