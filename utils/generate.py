import os
from tqdm import tqdm
import json
import numpy as np
import torch
from utils.misc import to_cpu
from utils.dataset import parse_answer, extract_labels, append_answer

class Inference():
    def __init__(self, model, tokenizer, dataset_name, prompts, gt, out_dir=None, max_tokens=128):
        self.model = model
        self.tokenizer = tokenizer
        self.dataset_name = dataset_name
        self.prompts = prompts
        self.gt = gt
        self.max_tokens = max_tokens
        self.out_dir = out_dir
        if dataset_name in ['true_false', 'halueval', 'fever']:
            self.prompt_type = 'fact'
        else:
            self.prompt_type = dataset_name
            
    def _to_tuple(self, obj):
        # Convert lists/tuples to tuples (and recurse)
        if isinstance(obj, (list, tuple)):
            return tuple(self._to_tuple(x) for x in obj)

        if isinstance(obj, torch.Tensor):
            if obj.ndim > 3:
                return tuple(self._to_tuple(x) for x in obj)
            else:
                return obj
        # Fallback: return as-is
        return obj
    
    def _compute_lengths(self, attn_mask, sequences, eos_token_id):
        """Per-sample (prompt_len, gen_len_trimmed, total_len_trimmed)."""
        B, max_prompt_len = attn_mask.shape
        eos_set = set(eos_token_id if isinstance(eos_token_id, (list, tuple)) else
                    ([] if eos_token_id is None else [eos_token_id]))
        prompt_lens = attn_mask.sum(dim=1).tolist()
        gen_lens = []
        for i in range(B):
            gen_ids = sequences[i, max_prompt_len:].tolist()  # generated tokens
            eos_pos = None
            if eos_set:
                for k, t in enumerate(gen_ids):
                    if t in eos_set:
                        eos_pos = k
                        break
            gen_len = len(gen_ids) if eos_pos is None else eos_pos + 1
            gen_lens.append(gen_len)
        totals = [p + g for p, g in zip(prompt_lens, gen_lens)]
        return prompt_lens, gen_lens, totals

    def extract_internal(self, hidden_states, logits,
        attention_mask,
        sequences,
        eos_token_id=None,
    ):
        # which layers to keep (HF exposes embeddings + L transformer layers)
        L_total = len(hidden_states[0])
        layer_ids = range(L_total)

        # lengths and trimming
        prompt_lens, gen_lens, totals = self._compute_lengths(
            attention_mask, sequences, eos_token_id
        )
        B = attention_mask.size(0)
        hs_list = []

        layers_resp = []
        for l in layer_ids:
            pieces = [hs_t[l][:, -1:, :] for hs_t in hidden_states[1:]]
            layers_resp.append(torch.cat(pieces, dim=1))

        for i in range(B):
            gl = gen_lens[i]
            per_layer = [L[i, :gl, :] for L in layers_resp]
            mat = torch.stack(per_layer, dim=1)
            hs_list.append(mat)
            
        scores_full = torch.stack(list(logits), dim=1)
        logits_list = []
        for i in range(B):
            gl = gen_lens[i]
            mat = scores_full[i, :gl, :]
            logits_list.append(mat)
        
        return hs_list, logits_list

    
    def batch_inference(self):
        batch_size = 16
        # Generate responses to prompts
        all_hs = []
        all_logits = []
        all_pairs = []
        labels = []
        gt = []
        
        for idx in tqdm(range(0, len(self.prompts), batch_size),  total=int(len(self.prompts)//batch_size+1), desc='Generating responses'):
            batch_prompts = self.prompts[idx: idx+batch_size]
            input, output = self.batch_generate(batch_prompts)
            response_ids = output.sequences[:, input.input_ids.shape[1]:]
            answer_txts = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
            
            excl_indices = []
            for i, answer in enumerate(answer_txts):
                ans = parse_answer(answer, self.dataset_name)
                if ans is None:
                    excl_indices.append(i)
                    continue
                labels = append_answer(labels, ans, self.gt[idx],  self.dataset_name)
                gt.append(self.gt[idx])
    
                all_pairs.append({"prompt": batch_prompts[i],
                            "response": answer_txts[i],
                            "response_id": response_ids[i].detach().cpu().numpy().tolist()})
            hs_list, logits_list = self.extract_internal(output.hidden_states, 
                                                         output.scores, 
                                                         input["attention_mask"], 
                                                         output.sequences,
                                                         self.tokenizer.eos_token_id)    
            hs_list, logits_list = [v for i, v in enumerate(hs_list) if i not in excl_indices], [v for i, v in enumerate(logits_list) if i not in excl_indices]
            all_hs.extend(hs_list), all_logits.extend(logits_list)
            
            if idx==0:
                print(f'Prompt: {self.prompts[0]}\nAnswer: {answer_txts[0]}')
                
            del output
            torch.cuda.empty_cache()
        print(f'{len(labels)}/{len(self.prompts)} answered.')
        if self.out_dir:
            self.save(all_pairs)
        
        all_logits = [logits.reshape(logits.shape[0], 1, logits.shape[1]) for logits in all_logits]
        all_logits = [self._to_tuple(logits) for logits in all_logits]
        all_hs = [hs.reshape(hs.shape[0], hs.shape[1], 1, 1, hs.shape[2]) for hs in all_hs]
        all_hs = [tuple(
                    tuple(hs[i, j] for j in range(hs.shape[1]))
                    for i in range(hs.shape[0])
                ) for hs in all_hs]
        return all_hs, all_logits, labels, gt, all_pairs
                
    def batch_generate(self, batch_prompts):
        
        batch_prompts = [self.tokenizer.apply_chat_template([
                                {"role": "system",
                                "content": "You are a helpful assistant, providing accurate and concise information without overthinking."},
                                {"role": "user",
                                "content": f"{query}"}],
                        tokenize=False,
                        add_generation_prompt=True,
                        enable_thinking=False) for query in batch_prompts]
        input = self.tokenizer(batch_prompts, return_tensors="pt", padding=True).to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **input,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
                top_p = None,
                temperature = None,
            )
            
        return input, output
    
    def extract_hs(self):
        all_hs = []
        all_logits = []
        for idx, query in tqdm(enumerate(self.prompts), total=len(self.prompts), desc='Extracting hidden states'):
            with torch.inference_mode():
                messages = [
                    {"role": "system",
                    "content": "You are a helpful assistant, providing accurate and concise information without overthinking."},
                    {"role": "user",
                    "content": f"{query}"}
                    ]   
                prompt = self.tokenizer.apply_chat_template(
                            messages,
                            tokenize=False,
                            add_generation_prompt=True,
                            enable_thinking=False)
                input = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
                output = self.model(**input, 
                                    return_dict_in_generate=True,
                                    output_hidden_states=True, 
                                    output_scores=True)
            all_hs.append(torch.stack(output.hidden_states).unsqueeze(3).permute(2, 0, 1, 3, 4).cpu())
            all_logits.append(output.logits.permute(1, 0, 2).cpu())
            del output; del input
            torch.cuda.empty_cache()
            
        all_logits = [self._to_tuple(logits) for logits in all_logits]
        all_hs = [self._to_tuple(hs) for hs in all_hs]
        return all_hs, all_logits, None, self.gt, None, list(np.arange(len(self.prompts)))
    
    def data_inference(self):
        # Generate responses to prompts
        all_hs = []
        all_logits = []
        all_pairs = []
        gen_ans = []
        labels = []
        gt = []
        ids = []
        
        for idx, query in tqdm(enumerate(self.prompts), total=len(self.prompts), desc='Generating responses'):
            input, output = self.generate(query)
            sequences = output.sequences
            response_id = sequences[0,input.input_ids.shape[-1]:]
            answer_txt = self.tokenizer.decode(response_id, skip_special_tokens=True)            
            ans = parse_answer(answer_txt, self.dataset_name)
            
            if idx==0:
                print(f'Prompt: {query}\nAnswer: {answer_txt}')
            
            if ans is None:
                continue
            
            all_pairs.append({"prompt": query,
                            "response": answer_txt,
                            "response_id": response_id.detach().cpu().numpy().tolist()})
            
            gen_ans.append(ans)
            gt.append(self.gt[idx])
            ids.append(idx)
            
            all_hs.append(to_cpu(output.hidden_states))
            all_logits.append(to_cpu(output.scores))
        
            del output
            torch.cuda.empty_cache()
        labels = extract_labels(self.dataset_name, gen_ans, gt)
        print(f'{len(labels)}/{len(self.prompts)} extracted.')
        
        if self.out_dir:
            self.save(all_pairs)
                
        return all_hs, all_logits, labels, gt, all_pairs, ids
                
    def generate(self, query):
        messages = [
                {"role": "system",
                "content": "You are a helpful assistant, providing accurate and concise information without overthinking."},
                {"role": "user",
                "content": f"{query}"}
            ]
        prompt = self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False)
        input = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        with torch.inference_mode():
            output = self.model.generate(
                **input,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                return_dict_in_generate=True,
                output_hidden_states=True,
                output_scores=True,
                top_p = None,
                temperature = None
            )
            
        return input, output
    
    def save(self, responses):
        with open(os.path.join(self.out_dir, f"responses_{self.prompt_type}.jsonl"), "w", encoding="utf-8") as f:
            for pair in responses:
                f.write(json.dumps(pair, ensure_ascii=True) + "\n")
                