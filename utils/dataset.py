import re
import io
import json
import pickle
from typing import List, Sequence
from collections.abc import Sequence, Mapping

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from datasets import load_dataset, Value
import evaluate

##### Data preprocessing and misc utils #######

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else: return super().find_class(module, name)

@torch.no_grad()
def fit_scaler(train_loader, device="cpu"):
    total_sum = None
    total_sq  = None
    total_cnt = 0

    for x, pad_mask, _ in train_loader:
        x = x.to(device)
        pad_mask = pad_mask.to(device)

        valid = (~pad_mask).unsqueeze(-1)
        x_valid = x * valid

        s  = x_valid.sum(dim=(0,1))
        s2 = (x_valid * x_valid).sum(dim=(0,1))
        n  = valid.sum().item()

        if total_sum is None:
            total_sum = s
            total_sq  = s2
        else:
            total_sum += s
            total_sq  += s2
        total_cnt += n

    mu  = total_sum / total_cnt
    ex2 = total_sq  / total_cnt
    var = (ex2 - mu**2).clamp_min(1e-12)
    std = var.sqrt()

    return mu, std

# Normalise
def preprocess(seqs, pad_mask, mu, std):
    x = (seqs - mu.view(1,1,-1)) / std.view(1,1,-1)
    x = x.masked_fill(pad_mask.unsqueeze(-1), 0.0)
    return x

####### Dataset utils #######
class SequenceDataset(Dataset):
    def __init__(self, sequences: List[Sequence[float]], labels):
        assert len(sequences) == len(labels)
        self.sequences = [torch.tensor(s, dtype=torch.float32) for s in sequences]
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

def collate_fn(batch):
    """Pads to the longest sequence in the batch and returns (padded, lengths, labels)"""
    sequences, labels = zip(*batch)
    lengths = torch.tensor([len(s) for s in sequences], dtype=torch.int64)
    padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=0.)
    L = padded.size(1)
    mask   = torch.arange(L)[None, :].expand(len(lengths), -1) >= lengths[:, None]
    
    return padded, mask, torch.tensor(labels, dtype=torch.float)

def permute_sequence(seq):
    idx = torch.randperm(seq.size(0))
    return seq[idx]

def reverse_sequence(seq):
    return torch.flip(seq, dims=[0])


def to_cpu(obj):
    """Recursively detach + move tensors to CPU, preserve structure."""
    if torch.is_tensor(obj):
        return obj.detach().cpu()        # or .float().cpu() if you want fp32
    elif isinstance(obj, Mapping):
        return {k: to_cpu(v) for k, v in obj.items()}
    elif isinstance(obj, Sequence):
        return type(obj)(to_cpu(x) for x in obj)
    else:
        return obj      

def binarize(batch):
    # FEVER labels are usually 'SUPPORTS', 'REFUTES'
    labs = [str(x).strip().upper() for x in batch["label"]]
    y = [0 if l.startswith("REFUTES") else 1 for l in labs]
    return {"label": y}

def format_qa(question, choice):
    return f"Q: {question} A: {choice}"

def format_prompt(dataset_name, dataset):
    if dataset_name in ['true_false', 'fever']:
        prompt_type = 'fact'
    else:
        prompt_type = dataset_name
    all_prompts = []
    all_gt = []
    
    if dataset_name in ['true_false', 'mmlu', 'commonsenseqa', 'math', 'fever']:
        for i in range(len(dataset)):
            gt = dataset[i]['answer']
            query= dataset[i]['question']
            if prompt_type is None:
                prompt = query
            else:
                with open(f'prompts/{prompt_type}.txt', 'r', encoding='utf-8') as f:
                    prompt = f.read()
            prompt = prompt.format(query=query)
            all_prompts.append(prompt)
            all_gt.append(gt)
    
    elif dataset_name=='gsm':
        languages = ['bn', 'en', 'ja', 'th']
        for lang in languages:
            for i in range(len(dataset)):
                gt = dataset[i]['answer']
                query = dataset[i][lang]
                if prompt_type is None:
                    prompt = query
                else:
                    with open(f'prompts/{prompt_type}.txt', 'r', encoding='utf-8') as f:
                        prompt = f.read()
                prompt = prompt.format(query=query)
                all_prompts.append(prompt)
                all_gt.append(gt)
    
    elif dataset_name=='trivia':
        for i in range(len(dataset)):
            gt = dataset[i]['answer']['value']
            query = dataset[i]['question'] + ' Be concise, and output only the final answer.\n'
            all_prompts.append(query)
            all_gt.append(gt)
            
    elif dataset_name=='sciq':
        for i in range(len(dataset)):
            gt = dataset[i]['correct_answer']
            query = dataset[i]['question'] + ' Be concise, and output only the final answer.\n'
            all_prompts.append(query)
            all_gt.append(gt)
            
    elif dataset_name=='medmcqa':
        for i in range(len(dataset)):
            gt = dataset[i]['cop']
            query= dataset[i]['question']
        
            with open(f'prompts/{prompt_type}.txt', 'r', encoding='utf-8') as f:
                prompt = f.read()
            prompt = prompt.format(query=query, opa=dataset[i]['opa'],
                                   opb=dataset[i]['opb'], opc=dataset[i]['opc'], opd=dataset[i]['opd'])
            all_prompts.append(prompt)
            all_gt.append(gt)
    return all_prompts, all_gt

def prepare_dataset(dataset_name, subdataset):
    
    if dataset_name == 'true_false':
        dataset = load_dataset('csv', data_files=f'data/true_false_data/{subdataset}.csv')['train']
        dataset = dataset.rename_columns({'statement': 'question', 'label': 'answer'})
    elif dataset_name == 'mmlu':
        dataset = load_dataset('cais/mmlu', 'all', split='validation')
    elif dataset_name == 'gsm':
        dataset = load_dataset('json', data_files='data/mgsm.jsonl')['train']
    elif dataset_name == 'commonsenseqa':
        dataset = load_dataset('json', data_files='data/commonsenseqa.jsonl')['train']
        letter_to_int = {"A": 0, "B": 1, "C": 2, "D": 3, "E": 4}
        dataset = dataset.map(lambda x: {'answer': letter_to_int[x['answer']]})
        dataset = dataset.rename_column('en', 'question')
    elif dataset_name == 'math':
        dataset = load_dataset('json', data_files='data/math.jsonl')['train']
        dataset = dataset.shuffle(seed=42).select(range(1000))
        # dataset = dataset.train_test_split(test_size=0.2, seed=42)['test']
        dataset = dataset.rename_column('en', 'question')
    elif dataset_name == 'fever':
        dataset = load_dataset('fever', 'v1.0', trust_remote_code=True)
        keep = ['claim', 'label']
        remove_cols = [c for c in dataset['labelled_dev'].column_names if c not in keep]
        dataset = dataset.filter(lambda ex: str(ex['label']).strip().upper() in {'SUPPORTS', 'REFUTES'})
        dataset = dataset.map(binarize, batched=True, remove_columns=remove_cols)
        dataset = dataset.rename_columns({'claim': 'question', 'label': 'answer'})
        dataset = dataset.cast_column('answer', Value('int64'))
        dataset = dataset['labelled_dev'].shuffle(seed=42).select(range(1000))
    elif dataset_name == 'trivia':
        dataset = load_dataset('mandarjoshi/trivia_qa', 'rc', split='validation')
        dataset = dataset.shuffle(seed=42).select(range(1000))
    elif dataset_name == 'sciq':
        dataset = load_dataset('allenai/sciq', split='test')
    elif dataset_name == 'medmcqa':
        dataset = load_dataset('openlifescienceai/medmcqa', split='validation')
        dataset = dataset.shuffle(seed=42).select(range(1000))
    formatter = format_prompt
    return dataset, formatter 

def load_generations(dataset, args):
    fp = f"outputs/{args.dataset_name}/{args.subdataset}/{args.model}"
    fp = fp.replace("/None", "") if "None" in fp else fp
        
    statements = []
    gt = []
    for i in range(len(dataset)):
        gt.append(int(dataset[i]['answer']))
        statements.append(dataset[i]['question'])
    labels = []
    prompts = []
    with open(f"{fp}/responses_cot_zero.jsonl", "r", encoding="utf-8") as f:
        for i, l in enumerate(f):
            data = json.loads(l)
            with open(f'prompts/p_true_f.txt', "r", encoding="utf-8") as f:
                prompt = f.read()
            if "true" in data["response"].lower():
                labels.append(1 if gt[i]==1 else 0)
                prompt = prompt.format(query=statements[i], answer=data["response"].replace("<|eot_id|>", ""))
                prompts.append(prompt)
            elif "false" in data["response"].lower():
                labels.append(1 if gt[i]==0 else 0)
                
                prompt = prompt.format(query=statements[i], answer=data["response"].replace("<|eot_id|>", ""))
                prompts.append(prompt)
            else:
                continue
    return prompts, labels

def parse_answer(answer_txt, dataset_name):

    if dataset_name in ['true_false', 'fever']:
        if 'true' in answer_txt.lower():
            return 1
        elif 'false' in answer_txt.lower():
            return 0
        else:
            return None
        
    elif dataset_name in ['mmlu', 'medmcqa']:
        map = {'a': 0, 'b': 1, 'c': 2, 'd':3}
        match = re.search(r"(?i)Answer\s*:\s*([A-D])", answer_txt)
        if match:
            key = match.group(1).lower()
            gen_ans = map[key]
            return gen_ans
        else:
            return None
    
    elif dataset_name == 'commonsenseqa':
        map = {'a': 0, 'b': 1, 'c': 2, 'd':3, 'e':4}
        match = re.search(r"(?i)Answer\s*:\s*([A-E])", answer_txt)
        if match:
            key = match.group(1).lower()
            gen_ans = map[key]
            return gen_ans
        else:
            return None
        
    elif dataset_name == 'gsm':
        ANSWER_PREFIX = {
            "en": "Answer",
            "bn": "উত্তর",
            "de": "Antwort",
            "es": "Respuesta",
            "fr": "Réponse",
            "ja": "答え",
            "ru": "Ответ",
            "sw": "Jibu",
            "te": "సమాధానం",
            "th": "คำตอบ",
            "zh": "答案",
        }
        for _, prefix in ANSWER_PREFIX.items():
            if prefix in answer_txt:
                answer_txt = answer_txt.split(prefix)[-1].strip()
                break
            else:
                return None
        match = re.search(r"\d+\.?\d*", answer_txt.replace(",", ""))
        if match:
            # gen_ans = float(match.group(1))
            gen_ans = float(match.group())
            return gen_ans
        else:
            return None
    
    elif dataset_name == 'math':
        def extract_boxed_content(text):
            pattern = re.compile(r'\\boxed{')
            matches = pattern.finditer(text)
            results = []
            for match in matches:
                start_pos = match.end()
                brace_count = 1
                i = start_pos
                while i < len(text) and brace_count > 0:
                    if text[i] == '{':
                        brace_count += 1
                    elif text[i] == '}':
                        brace_count -= 1
                    i += 1
                if brace_count == 0:
                    results.append(text[start_pos:i-1])
            return results
        extracted_ans = extract_boxed_content(answer_txt)
        extracted_ans = extracted_ans[0] if extracted_ans else None
        if extracted_ans:
            extracted_ans = extracted_ans.replace(" ", "")
            return extracted_ans
        else:
            return None
    
    else:
        return answer_txt

def append_answer(labels, gen_ans, gt, dataset_name):
    if dataset_name in ['true_false', 'fever', 'mmlu', 'medmcqa', 'commonsenseqa']:
       labels.append(0 if gen_ans==gt else 1)
    elif dataset_name == 'gsm':
       labels.append(0 if gen_ans==float(gt.replace(',', '').strip()) else 1)
    elif dataset_name == 'math':
        labels.append(0 if gen_ans==gt.replace(' ', '') else 1)
    elif dataset_name in ['trivia', 'sciq']:
        rouge = evaluate.load('rouge')
        rouge_score = rouge.compute(predictions=[gen_ans], references=[gt])['rougeL']
        threshold = 0.7
        labels.append(0 if rouge_score > threshold else 1)
    return labels

def extract_labels(dataset_name, gen_ans, ref_ans):
    if dataset_name in ['true_false', 'fever', 'mmlu', 'medmcqa', 'commonsenseqa']:
       gen_ans = [1 if 'true' in label.lower() else 0 for label in gen_ans]
       labels = [0 if ans==ref else 1 for ans, ref in zip(gen_ans, ref_ans)]

    elif dataset_name == 'gsm':
       labels = [0 if gen==float(gt.replace(',', '').strip()) else 1 for gen, gt in zip(gen_ans, ref_ans)]
    elif dataset_name == 'math':
        labels = [0 if gen==gt.replace(' ', '') else 1 for gen, gt in zip(gen_ans, ref_ans)]
    elif dataset_name in ['trivia', 'sciq']:
        rouge = evaluate.load('rouge')
        rouge_scores = rouge.compute(predictions=gen_ans, references=ref_ans, use_aggregator=False)['rougeL']
        threshold = 0.7
        labels = [0 if rouge_score >= threshold else 1 for rouge_score in list(rouge_scores)]
    return labels

####### Visualisation utils #######

def load_hidden_states(dataset, subdataset, model_name, prompt_only=False):
    if not prompt_only:
        hs_path = f'outputs/{dataset}/{subdataset}/{model_name}/hidden_states.pkl'
    else:
        hs_path = f'outputs/{dataset}/{subdataset}/{model_name}/prompt_hidden_states.pkl'
    with open(hs_path, 'rb') as f:
        if torch.cuda.is_available():
            hs_all_responses = pickle.load(f)
        else:
            hs_all_responses = CPU_Unpickler(f).load()
    return hs_all_responses

# Extract hidden states for UMAP
def extract_hidden_states(hs_all_responses, response_idx):
    NUM_LAYERS_PLUS_ONE = len(hs_all_responses[0][0])
    collected_hidden_states = []
    token_indices = [] 

    # for i, response_idx in enumerate(response_indices_to_plot):
    if response_idx >= len(hs_all_responses):
        print(f"Warning: Response index {response_idx} is out of bounds. Skipping.")
        return None, None

    response_token_steps = hs_all_responses[response_idx]
    num_tokens_in_response = len(response_token_steps)

    print(f"\nExtracting states for Response {response_idx}:")
    print(f"  Number of token steps (tokens in response): {num_tokens_in_response}")
    print(f"  Number of layers per step: {NUM_LAYERS_PLUS_ONE}")

    for token_step_idx in range(num_tokens_in_response):
        layer_states_for_token_step = response_token_steps[token_step_idx]
        for layer_idx in range(NUM_LAYERS_PLUS_ONE):
            tensor = layer_states_for_token_step[layer_idx]
            
            # Ensure tensor is on CPU and convert to NumPy
            tensor_np = tensor.cpu().numpy()

            if token_step_idx == 0:
                if tensor_np.shape[1] > 0:
                    state_vector = tensor_np[0, -1, :]
                else:
                    print(f"Warning: Prompt length is 0 for response {response_idx}, layer {layer_idx}. Skipping.")
                    continue
            else:
                # For subsequent token steps, shape is (1, 1, hidden_dim)
                state_vector = tensor_np[0, 0, :]
            
            collected_hidden_states.append(state_vector)
            token_indices.append(token_step_idx)
        

    if not collected_hidden_states:
        print("No hidden states were collected. Cannot proceed with UMAP.")
        return None, None
        
    return np.array(collected_hidden_states), np.array(token_indices)


def tok(tokenizer, dataset, subdataset, model_name, response_idx):
    responses = [] 
    with open(f'outputs/{dataset}/{subdataset}/{model_name}/responses.jsonl', 'r', encoding='utf-8') as f:
        for l in f:
            data = json.loads(l)
            responses.append(data['response'])
    response = responses[response_idx]

    enc = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True
    )

    offsets = enc["offset_mapping"] 
    vis_offsets  = [(s, e) for (s, e) in offsets if e > s]   # keep only real text spans
    tokenized = []
    for idx, (start, end) in enumerate(vis_offsets):
        token_text  = response[start:end]
        tokenized.append(token_text)
        
    return tokenized

def tok_input(tokenizer, dataset, subdataset, response_idx):
    data = pd.read_csv(f"data/{dataset}_data/{subdataset}.csv")
    response = data.iloc[response_idx]['statement']
    enc = tokenizer(
        response,
        add_special_tokens=False,
        return_offsets_mapping=True
    )

    offsets = enc["offset_mapping"] 
    vis_offsets  = [(s, e) for (s, e) in offsets if e > s] # keep only real text spans
    tokenized = []
    for idx, (start, end) in enumerate(vis_offsets):
        token_text  = response[start:end]
        tokenized.append(token_text)
    return tokenized
