import json 
from tqdm import tqdm

import torch
from sentence_transformers.cross_encoder import CrossEncoder
from transformers import AutoTokenizer


def get_tokenwise_importance(args):
    measure_model = CrossEncoder('cross-encoder/stsb-roberta-large', num_labels=1)

    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    generations = []
    if args.topic!='None':
        file_dir = f'outputs/{args.dataset_name}/{args.topic}/{args.model}'
    else:
        file_dir = f'outputs/{args.dataset_name}/{args.model}'
    with open(f'{file_dir}/responses_{args.prompt_type}.jsonl',
              'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                generations.append(json.loads(line))

    scores = []
    token_importance_list = []
    for sample_idx, gen in tqdm(enumerate(generations), total=len(generations), desc='Getting token importance'):
        generated_text = gen['response']
        
        # tokenized = torch.tensor(tokenizer.encode(generated_text, add_special_tokens=False))
        tokenized = torch.tensor(gen['response_id'])
        token_importance = []
        # measure cosine similarity by removing each token and compare the similarity
        for token in tokenized:
            similarity_to_original = measure_model.predict([generated_text,
                                                            generated_text.replace(
                                                                tokenizer.decode(token, skip_special_tokens=True),
                                                                '')])
            token_importance.append(1 - torch.tensor(similarity_to_original))

        token_importance = torch.tensor(token_importance).reshape(-1)
        token_importance_list.append(token_importance)

    scores = torch.tensor(scores)
    if torch.isnan(scores).sum() > 0:
        scores[torch.isnan(scores).nonzero(as_tuple=True)] = 0

   
    return token_importance_list

