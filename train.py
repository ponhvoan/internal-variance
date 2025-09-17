import os
import random
import argparse
import pickle

import torch
from torch.utils.data import DataLoader, random_split
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

from utils.models import TransformerClassifier, RNNClassifier
from utils.dataset import SequenceDataset, collate_fn, preprocess
from utils.misc import fpr_at_95_tpr


def train(model, optim, criterion, device, mu, std, train_loader):
    model.train()
    loss_sum = 0
    for x, mask, y in train_loader:
        x, mask, y = x.to(device), mask.to(device), y.to(device)
        x = preprocess(x, mu.to(device), std.to(device))
        # x = [reverse_sequence(s) for s in x]
        # x = torch.stack(x)
        optim.zero_grad()
        logits = model(x, mask)
        loss = criterion(logits, y)
        loss.backward()
        optim.step()
        loss_sum += loss.item()*len(y)
    train_loss = loss_sum / len(train_set)
    
    return train_loss

def validate(model, device, mu, std, val_loader):
    model.eval(); preds, targets = [], []
    with torch.no_grad():
        for x, mask, y in val_loader:
            x, mask = x.to(device), mask.to(device)
            x = preprocess(x, mu.to(device), std.to(device))
            logits = model(x, mask)
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
    parser.add_argument('--dataset_name', type=str, default='true_false')
    parser.add_argument('--subdataset', type=str, default='animals')
    parser.add_argument('--arch', type=str, default='rnn', choices=['rnn', 'transformer'])
    parser.add_argument('--features', type=str, default='var+impt', choices=['hs', 'var', 'var+impt', 'all'])
    parser.add_argument('--data_portion', type=float, default=1.0)
    parser.add_argument('--prompt_type', type=str, default='cot_zero')
    parser.add_argument('--num_epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=128)

    args = parser.parse_args()
    
    # Load dataset and prepare dataloaders
    seed = 42
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)

    fp = f"outputs/{args.dataset_name}/{args.subdataset}/{args.model}"
    fp = fp.replace("/None", "") if "None" in fp else fp
    with open(os.path.join(fp, f"tokenDict.pkl"), "rb") as f:
        scores = pickle.load(f)
        scores = [score.T for score in scores] if np.ndim(scores[0])>=2 else [np.expand_dims(s, 1) for s in scores]
    with open(os.path.join(fp, f"tokenImpt.pkl"), "rb") as f:
        token_importance = pickle.load(f)
        # token_importace = [np.expand_dims(i, 1) for i in token_importance]
    with open(os.path.join(fp, f"hsPCA.pkl"), "rb") as f:
        hs = pickle.load(f)
        
    if args.features=='var':
        # seqs = [np.expand_dims(s, 1) for s in scores]
        seqs = scores
    elif args.features=='hs':
        seqs = hs
    elif args.features=='var+impt':
        seqs = [
                np.concatenate((a, b[:,None]), axis=1)
                for a, b in zip(scores, token_importance)
                ]
    elif args.features=='all':
        seqs = [
                np.concatenate((a, b), axis=1)
                for a, b in zip(hs, scores)
                ]
    
    labels = np.load(f'{fp}/labels.npy')
    data_len = int(args.data_portion*len(labels))
    labels = labels[:data_len]
    seqs = seqs[:data_len]

    dataset = SequenceDataset(seqs, labels)
    train_set, val_set = random_split(dataset, [int(0.8*len(seqs)), len(seqs)-int(0.8*len(seqs))])
    train_values = torch.cat([seq for seq, _ in train_set])
    mu  = train_values.mean(0)
    std = train_values.std(0).clamp_min(1e-6)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_set,   batch_size=args.batch_size, collate_fn=collate_fn)

    # --------------- model / optim ---------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.arch=='transformer':
        model = TransformerClassifier(input_dim=seqs[0].shape[-1]).to(device)
    elif args.arch=='rnn':
        model = RNNClassifier(input_dim=seqs[0].shape[-1]).to(device)

    optim  = torch.optim.Adam(model.parameters(), lr=1e-4)
    # criterion = torch.nn.BCEWithLogitsLoss()
    criterion = torch.nn.CrossEntropyLoss()

    # --------------- training loop ---------------
    print(f"-----------{args.model.split('/')[-1]}: {args.dataset_name}/{args.subdataset}-----------")
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