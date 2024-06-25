import torch
import numpy as np
from sklearn.metrics import accuracy_score, f1_score
import pandas as pd
import json
from ontolearn.knowledge_base import KnowledgeBase
from model import *
import helper as h
kb = ['animals', 'family', 'lymphography',   'nctrer', 'suramin']

def convert_to_labels(kb_path, expr):
    # datapoint = []
    KB = KnowledgeBase(path=kb_path) # update file path Done :)
    L = sorted([ind.str.split("/")[-1] for ind in KB.individuals()])
    datapoints = [] 
    name_to_ids = {name: idx for idx, name in enumerate(L)} 
    

    y_C = np.zeros(len(L))
    # print(""expr)
    pos = expr[1]['positive examples']
    Ind_pos = list(map(name_to_ids.get, pos))
    y_C[Ind_pos] = 1
    datapoint = (expr[0], list(y_C))
    return datapoint

def neg_func(exp, models, emb, len_1=False):
        exp_b = exp.replace('¬', '')
        b_emb = emb.loc[exp_b].values
        return models['neg'](torch.FloatTensor(b_emb)) if len_1 else models['neg'].encode(torch.FloatTensor(b_emb).unsqueeze(0))
    
def process(ele, eval, emb):
        ele = ele.strip().replace('(', '').replace(')', '')
        if ele in eval:
            exp_emb = eval[ele]
        else:
            exp_emb = neg_func(ele) if '¬' in ele else torch.FloatTensor(emb.loc[ele].values).unsqueeze(0)
        return exp_emb

def exp_three(exp, eval, emb, models, last=False):
    if ' ' not in exp:
        if '¬' in exp:
            # print('neg', exp)
            return neg_func(exp, models, emb)
        else:
            return torch.FloatTensor(emb.loc[exp].values).unsqueeze(0)
        
    elif (h.quant(exp) or h.const(exp)) and h.top_bot(exp):
        if (' ⊔ ' in exp):
            parts = exp.split(' ⊔ ')
            parts = [process(part, eval, emb) for part in parts]
            enc = models['disj'](parts[0], parts[1]) if last else models['disj'].encode(parts[0], parts[1]) 
            return enc
                
        elif (' ⊓ ' in exp):
            parts = exp.split(' ⊓ ')
            parts = [process(part, eval, emb) for part in parts]
            enc = models['conj'](parts[0], parts[1]) if last else models['conj'].encode(parts[0], parts[1])
            return enc
            
        else:
            if (exp.startswith('∀')):
                parts = exp.split(' ')[1].split('.')
                parts = [process(part, eval, emb) for part in parts]
                enc = models['uni'](parts[0], torch.FloatTensor(parts[1])) if last else models['uni'].encode(parts[0], parts[1])
                return enc
                
            if (exp.startswith('∃')):
                parts = exp.split(' ')[1].split('.')
                parts = [process(part, eval, emb) for part in parts]
                enc = models['exist'](parts[0], torch.FloatTensor(parts[1])) if last else models['exist'].encode(parts[0], parts[1])
                return enc
            else:
                print('Failed', exp)
    else:
        print('comp', exp)

def evaluate_parentheses(expression, emb, models):
    # exp_list = list(expression)
    count_exp = 0
    print('expression', expression)
    computed = {}
    i = 0
    while '(' in expression:
        open_idx = None
        close_idx = None

        for idx, item in enumerate(expression):
            if item == '(':
                open_idx = idx
            if item == ')':
                close_idx = idx
                break
            
        # print('evaluating', expression[open_idx + 1: close_idx])
        inner_results = exp_three(expression[open_idx + 1: close_idx], computed, emb, models)  


        expression = list(expression)
        computed[str(i)] = inner_results
        expression[open_idx: close_idx + 1] = str(i)
        expression = "".join(expression)
        
        i+= 1

    # print('inner_results', inner_results)
    print('expression', expression)
    if (not (h.quant(expression) or h.const(expression))) and h.top_bot(expression):
        print('complex', expression)
        if " ⊓ " in expression:
            # return "Contains ⊓"
            parts = expression.split(" ⊓ ")
            parts = expression.split(" ⊓ ")
            first = " ⊓ ".join(parts[:2])
        elif " ⊔ " in expression:
            # return "Contains ⊔"
            parts = expression.split(" ⊔ ")
            parts = expression.split(" ⊔ ")
            first = " ⊔ ".join(parts[:2])
        else:
            print('super complex', expression)
        
        
        len_first = len(first)
        inner_results = exp_three(first, computed, emb, models)
        expression = list(expression)
        computed[str(i)] = inner_results
        expression[:len_first] = str(i)
        expression = "".join(expression)
        print(expression)
    
    final = exp_three(expression, computed, emb, models, True)
    # print('computed', computed)
    return final

# parts = input_string.split(" ⊓ ")
# result = " ⊓ ".join(parts[:2])

# print(result) Grandmother ⊓ 1 ⊓ 3


def test_run(kb):
    for data_name in kb:
        print(f'In {data_name}')
        # path = f'./testing_data/{data_name}/Data.json'
        path = f'./testing_data/test_data_{data_name}.json'
        kb_path = f'../NCESData/{data_name}/{data_name}.owl'
        with open(path, 'r') as f:
            data = json.load(f)
        emb = pd.read_csv(f'../generated_data/{data_name}/{data_name}_emb.csv', index_col = 0)
        neg_path = f'./trained_models/{data_name}/{data_name}_model_Negation.pth'
        conj_path = f'./trained_models/{data_name}/{data_name}_model_Conjunction.pth'
        disj_path = f'./trained_models/{data_name}/{data_name}_model_Disjunction.pth'
        exist_path = f'./trained_models/{data_name}/{data_name}_model_Existential.pth'
        uni_path = f'./trained_models/{data_name}/{data_name}_model_Universal.pth'

        KB = KnowledgeBase(path=kb_path) # update file path Done :)
        L = sorted([ind.str.split("/")[-1] for ind in KB.individuals()])
        neg = Negation(40, 40, len(L), 16)
        neg.load_state_dict(torch.load(neg_path))
        neg.eval()

        conj = Conjunction(40, 40, len(L), 16)
        conj.load_state_dict(torch.load(conj_path))
        conj.eval()

        disj = Disjunction(40, 40, len(L), 16)
        disj.load_state_dict(torch.load(disj_path))
        disj.eval()

        if data_name not in ['animals', 'lymphography']:
            exist = Existential(40, 40, len(L), 16)
            exist.load_state_dict(torch.load(exist_path))
            exist.eval()

            uni = Universal(40, 40, len(L), 16)
            uni.load_state_dict(torch.load(uni_path))
            uni.eval()


        models = {
            'neg': neg,
            'conj': conj,
            'disj': disj,
            'exist': exist,
            'uni': uni
        } if data_name not in ['animals', 'lymphography'] else {
            'neg': neg,
            'conj': conj,
            'disj': disj,
        }


        test_acc = []
        test_f1 = []
        for pair in data:    
            converted = convert_to_labels(kb_path, pair)
            expression = converted[0]
            label = converted[1]
            # print(expression)
            if not h.top_bot(expression):
                print("skipping", expression)
                continue
            if "(" in expression or expression.count(' ') > 2:
                print('evalulating', expression)
                y_pred = evaluate_parentheses(expression, emb, models)
                # print('output', y_pred[:2])
            else:
                print('evalulating', expression)
                if (' ' not in expression) and ('¬' in expression):
                    y_pred = neg_func(expression, models, emb, True)
                else:
                    y_pred = exp_three(expression, {}, emb, models, True)
                # print('output', y_pred.squeeze(), 'len', len(y_pred.squeeze()))
                # print('label', label, 'len', len(label))
            pred = (y_pred > 0.5).numpy().astype(int)
            # pred
            test_acc.append(accuracy_score(pred.squeeze(), label))
            test_f1.append(f1_score(pred.squeeze(), label))
        print("\n ############ Done ############### \n")

        with open(f"./metrics/{data_name}/{data_name}_testing_accuracies_today.json", "w") as file:
            json.dump([('testing accuracy', test_acc)] , file)
        with open(f"./metrics/{data_name}/{data_name}_testing_f1_today.json", "w") as file:
            json.dump([('testing f1', test_f1)], file)


test_run(kb)