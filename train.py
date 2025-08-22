import os
import random
import argparse
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from utils.models import TransformerClassifier
from utils.dataset import prepare_dataset, SequenceDataset, collate_fn
from utils.generate import Inference
from utils.misc import fpr_at_95_tpr, preprocess


def train(model, optim, criterion, device, mu, std, train_loader):
    model.train()
    loss_sum = 0
    for x, lengths, y in train_loader:
        x, lengths, y = x.to(device), lengths.to(device), y.to(device)
        x = preprocess(x, mu.to(device), std.to(device))
        # x = [reverse_sequence(s) for s in x]
        # x = torch.stack(x)
        optim.zero_grad()
        logits = model(x, lengths)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        loss_sum += loss.item()*len(y)
    train_loss = loss_sum / len(train_set)
    
    return train_loss

def validate(model, device, mu, std, val_loader):
    model.eval(); preds, targets = [], []
    with torch.no_grad():
        for x, lengths, y in val_loader:
            x, lengths = x.to(device), lengths.to(device)
            x = preprocess(x, mu.to(device), std.to(device))
            logits = model(x, lengths)
            prob = torch.sigmoid(logits).cpu().numpy()
            preds.extend(prob)
            targets.extend(y.numpy())
    acc = accuracy_score(targets, np.array(preds) > 0.5)
    auc = roc_auc_score(targets, preds)
    aupr = average_precision_score(targets, preds)
    fpr95 = fpr_at_95_tpr(targets, preds)
    return acc, auc, aupr, fpr95

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='meta-llama/Llama-3.2-3B-Instruct')
    parser.add_argument('--dataset_name', type=str, default='gsm')
    parser.add_argument('--topic', type=str, default=None)
    parser.add_argument('--data_portion', type=float, default=1.0)
    parser.add_argument('--prompt_type', type=str, default='gsm')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()
    
    # Load dataset and prepare dataloaders
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    if args.topic is None:
        fp = f"outputs/{args.dataset_name}/{args.model}"
    else:
        fp = f"outputs/{args.dataset_name}/{args.topic}/{args.model}"
    with open(os.path.join(fp, f"tokenDict_{args.prompt_type}.pkl"), "rb") as f:
        scores = pickle.load(f)
    with open(os.path.join(fp, f"hsPCA_{args.prompt_type}.pkl"), "rb") as f:
        hs = pickle.load(f)
        
    # seqs = hs
    seqs = [
            np.concatenate((a, b[:, None]), axis=1)
            for a, b in zip(hs, scores)
            ]
    dataset, formatter = prepare_dataset(args.dataset_name.replace('_data', ''), fp.split('/')[2])
    _, labels = formatter(args.dataset_name, dataset, args.prompt_type)
    # labels = np.load(f'{fp}/labels.npy')
    data_len = int(args.data_portion*len(labels))
    labels = labels[:data_len]
    seqs = seqs[:data_len]
    labels = 1 - np.array(labels)

    dataset = SequenceDataset(seqs, labels)
    train_set, val_set = random_split(dataset, [int(0.8*len(seqs)), len(seqs)-int(0.8*len(seqs))])
    all_values = torch.cat([seq for seq, _ in train_set])
    mu  = all_values.mean(0)
    std = all_values.std(0).clamp_min(1e-6)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader   = DataLoader(val_set,   batch_size=args.batch_size, collate_fn=collate_fn)
    from sklearn.metrics import accuracy_score, roc_auc_score

    # --------------- model / optim ---------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransformerClassifier(input_dim=11).to(device)
    optim  = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.BCEWithLogitsLoss()

    # --------------- training loop ---------------
    patience = 100
    best_auc, count = 0, 0
    for epoch in range(args.num_epochs+1):
        train_loss = train(model, optim, criterion, device, mu, std, train_loader)
        acc, auc, aupr, fpr95 = validate(model, device, mu, std, val_loader)
        
        if auc > best_auc:
            best_auc = auc
            curr_acc, curr_fpr, curr_aupr = acc, fpr95, aupr
            count = 0
        else:
            count += 1
            
        # print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val acc {acc*100:.2f} | val AUC {auc*100:.2f} | val FPR@95 {fpr95*100:.2f} | val AUPR {aupr*100:.2f} ")
        if epoch%25==0 and count<patience and epoch<args.num_epochs:
            print(f"Epoch {epoch:02d} | train loss {train_loss:.4f} | val acc {acc*100:.2f} | val AUC {auc*100:.2f} | val FPR@95 {fpr95*100:.2f} | val AUPR {aupr*100:.2f} ")
            
        if count>=patience or epoch==args.num_epochs:
            if epoch<100:    
                print(f"Early stopping triggered at epoch {epoch}.\n"
                        f"Best val AUC {best_auc*100:.2f} with val acc {curr_acc*100:.2f}, val FPR@95 {fpr95*100:.2f}, val AUPR {aupr*100:.2f}")
            else:
                print(f"Best val AUC {best_auc*100:.2f} with val acc {curr_acc*100:.2f}, val FPR@95 {fpr95*100:.2f}, val AUPR {aupr*100:.2f}")
            
            break