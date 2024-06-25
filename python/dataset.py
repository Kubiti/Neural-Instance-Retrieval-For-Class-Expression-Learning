import torch
import numpy as np
import helper as h
import json
from ontolearn.knowledge_base import KnowledgeBase


class NRDataset(torch.utils.data.Dataset):
    def __init__(self, data, embedding):
        self.data = data
        self.embedding = embedding
        
        
    def __len__(self):
        return len(self.data)

    def neg(self, exp):
        exp_b = exp.replace('¬', '')
        b_emb = self.embedding.loc[exp_b]
        return torch.tensor(b_emb, dtype=torch.float32)
    
    def process(self, ele):
            ele = ele.strip().replace('(', '').replace(')', '')
            exp_emb = self.neg(ele) if '¬' in ele else self.embedding.loc[ele]
            return torch.tensor(exp_emb, dtype=torch.float32)

       
    def __getitem__(self, idx):
        (exp, label) = self.data[idx]
        label = torch.tensor(label)
        if ' ' not in exp and '¬' in exp:
        # condition for negation
        #    if :
                # print(f'exp: {exp}')
            exp_b = exp.replace('¬', '')
            exp_emb = torch.tensor(self.embedding.loc[exp_b], dtype=torch.float32)
            label = torch.tensor(label)

            return (exp_emb, label)
                
        elif (h.quant(exp) or h.const(exp)) and h.top_bot(exp):
            exp_emb = []
            if (' ⊔ ' in exp):
                parts = exp.split(' ⊔ ')
                parts = [self.process(part) for part in parts]
                # for part in parts:
                #     print("ele: ", exp, torch.max(part))
                return (parts, label)
                
            elif (' ⊓ ' in exp):
                parts = exp.split(' ⊓ ')
                parts = [self.process(part) for part in parts]
                return (parts, label)
                
            else:
                if (exp.startswith('∀')):
                    parts = exp.split(' ')[1].split('.')
                    parts = [self.process(part) for part in parts]
                    return (parts, label)
                    
                if (exp.startswith('∃')):
                    parts = exp.split(' ')[1].split('.')
                    parts = [self.process(part) for part in parts]
                    return (parts, label)
                else:
                    print('Failed', exp)
                
        #elif ' ⊔ ' in exp:
        #    a,b = exp.split(' ⊔ ')
        #    a = torch.tensor(self.embedding.loc[a],dtype=torch.float32)
        #    b = torch.tensor(self.embedding.loc[b],dtype=torch.float32)
        #    return ([a,b], label)
        #
        #elif ' ⊓ ' in exp:
        #    a,b = exp.split(' ⊓ ')
        #    a = torch.tensor(self.embedding.loc[a],dtype=torch.float32)
        #    b = torch.tensor(self.embedding.loc[b],dtype=torch.float32)
        #    return ([a,b], label)
        
        else:
            print('None')
            print(exp, label)
            return None
        #     label = torch.tensor(label)
        #     return (exp_emb, label)
         
   
        