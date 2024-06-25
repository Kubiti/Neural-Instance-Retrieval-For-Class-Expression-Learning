import numpy as np
import pandas as pd
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, f1_score
from sklearn import preprocessing
from dataset import NRDataset #, atomic_check, three_check
import random
import warnings
warnings.filterwarnings('ignore')
kb = ['animals', 'family', 'lymphography',   'nctrer', 'suramin']
import helper as h


from model import Negation, Conjunction, Disjunction, Existential, Universal


best_accuracies = {}

hidden_size = 40
learning_rate = 0.001
num_epochs = 500
batch_size = 16


def train(model_in, model_path=None):
    model_name = str(model_in).split('.')[-1].split('\'')[0]

    for data_name in kb:
        # if data_name in ['animals', 'lymphography'] and (model_in == Existential or model_in == Universal):
        #     print(f"skipping {data_name} and {model_name} ")
        #     continue
        

        print(f'In {data_name}, {model_name}')
        data_path = f'../generated_data/{data_name}/train_data_{data_name}.json'
        emb_path = f'../generated_data/{data_name}/{data_name}_emb.csv'
        inst_emb = pd.read_csv(emb_path)
        #if data_name != 'animals':
        #    inst_emb['Unnamed: 0'] = inst_emb['Unnamed: 0'].apply(lambda x: x.replace(f"{'NTNames' if data_name == 'semantic_bible' else ('ontology' if data_name == 'vicodi' else data_name)}#", ''))
        inst_emb.set_index(inst_emb.columns[0], inplace=True)  
        
        
        input_size = inst_emb.shape[1] 
        
        
        def read_json(path):
            with open(path, 'r') as f:
                data = json.load(f)
            data = [(exp, label) for (exp, label) in data]
            # print('data', data[:10][0])
            
            if model_in == Negation:
                check = h.neg_check
            elif model_in == Conjunction:
                check = h.inter_check
            elif model_in == Disjunction:
                check = h.union_check
            elif model_in == Existential:# and (data_name not in ['animals', 'lymphography']):
                check = h.exist_check
            elif model_in == Universal:# and (data_name not in ['animals', 'lymphography']):
                check = h.forall_check
            else:
                check = h.null
                print(f"skipping {data_name}")
                

            data = [(exp, label) for (exp, label) in data if check(exp)]
            # print('data', len(data))
            # if len(data) < 5:
            #     print(f"skipping {data_name}")
            #     continue
            # print([exp for (exp, label) in data][:10])
            # print('')
    #         print('atomic_data', len(atomic_data))
    #         print('three_data', len(three_data))
            # print(data[0])
        
            # output_size = len(data[1][1]) # todo uncomment output_size
            if len(data) < 3:
                output_size = 0
            else:
                output_size = len(data[1][1])
            # print('output', output_size)
            random.shuffle(data)
            length = len(data)
            train_len = int(0.8 * length)
            train = data[:train_len]
            validation = data[train_len:]
            return (train, validation, output_size) # todo add output size to return
        
        
        
        # read_json(data_path)
        # # print('data', len(data))
        # if len(data) < 5:
        #         print(f"skipping {data_name}")
        train_dataset, validation_dataset, output_size = read_json(data_path)
    #     print('train_dataset', len(train_dataset))

        if output_size == 0:
            print(f"skipping {data_name}")
            continue
        
        train_dataset = NRDataset(train_dataset, inst_emb)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=3, shuffle=True)


        validation_dataset = NRDataset(validation_dataset, inst_emb)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=3, shuffle=True)


        model = model_in(input_size, hidden_size, output_size, batch_size)        
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        criterion = nn.BCELoss()
        mod = num_epochs // 10
        best_val_score = 0
        best_weights = model.state_dict()

        f1 =[]
        acc = []
        
        avg_train_lossarr = []
        avg_validation_lossarr = []
        avg_train_accarr = []
        avg_validation_accarr = []
        avg_train_f1arr = []
        avg_validation_f1arr = []
        for e in range(num_epochs):
            train_loss = []
            validation_loss = []
            train_acc = []
            validation_acc = []
            train_f1 = []
            validation_f1 = []
            
            for x, y in train_dataloader:
                # print('shape', x.shape if model_name == 'Negation' else (x[0].shape, x[1].shape))
                # x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
                
                optimizer.zero_grad()
                # print('first', torch.max(x[0]), torch.min(x[0]))
                # print('second', torch.max(x[1]), torch.min(x[1]))
                
                outputs = model(x) if model_name == 'Negation' else model(x[0], x[1])
                # print('x shape', x.shape)
    #             print('y shape', y.shape)
    #             print('outputs shape', outputs.shape)
                loss = criterion(outputs, y)
                train_loss.append(loss.item())
                loss.backward()
                optimizer.step()

    #             # compute accuracy
                y_pred = np.zeros((len(outputs), outputs.shape[1]))
                y_pred[outputs.detach().numpy() > 0.5] = 1
                accuracy = accuracy_score(y_pred.flatten(), y.flatten())
                train_acc.append(accuracy)
                
                # computer f1-score
                f1_score_ = f1_score(y_pred.flatten(), y.flatten())
                train_f1.append(f1_score_)

            for x, y in validation_dataloader:
    #             print(x.shape)
                # x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
                
                outputs = model(x) if model_name == 'Negation' else model(x[0], x[1])
                loss_val = criterion(outputs, y)
                validation_loss.append(loss_val.item())

                # compute accuracy
                y_pred = np.zeros((len(outputs), outputs.shape[1]))
                y_pred[outputs.detach().numpy() > 0.5] = 1
                accuracy = accuracy_score(y_pred.flatten(), y.flatten())
                if accuracy > best_val_score:
                    best_val_score = accuracy  
                    best_weights = model.state_dict()
                    print(f'Best acc: {best_val_score}')
                validation_acc.append(accuracy)
                
                # computer f1-score
                f1_score_ = f1_score(y_pred.flatten(), y.flatten())
                validation_f1.append(f1_score_)

            avg_train_loss = np.mean(train_loss)
            avg_train_lossarr.append(avg_train_loss)
            
            avg_validation_loss = np.mean(validation_loss)
            avg_validation_lossarr.append(avg_validation_loss)

            avg_train_acc = np.mean(train_acc)
            avg_train_accarr.append(avg_train_acc)
            
            avg_validation_acc = np.mean(validation_acc)
            avg_validation_accarr.append(avg_validation_acc)
            
            avg_train_f1 = np.mean(train_f1)
            avg_train_f1arr.append(avg_train_f1)
            
            avg_validation_f1 = np.mean(validation_f1)
            avg_validation_f1arr.append(avg_validation_f1)
    

            if e % 5 == 0:
                    print("Epoch: %d, Training loss: %1.3f" % (e, avg_train_loss), "Validation loss: %1.3f" % (avg_validation_loss))
                    print("Epoch: %d, Training acc: %1.3f" % (e, avg_train_acc), "Validation acc: %1.3f" % (avg_validation_acc))
                    print("Epoch: %d, Training F1-score: %1.3f" % (e, avg_train_f1), "Validation F1-score: %1.3f" % (avg_validation_f1))

        with open(f"./metrics/{data_name}/{data_name}_accuracies_{model_name}.json", "w") as file:
            json.dump([('train accuracy', avg_train_accarr), ('validation_accuracy', avg_validation_accarr)] , file)
        with open(f"./metrics/{data_name}/{data_name}_f1_{model_name}.json", "w") as file:
            json.dump([('train f1', avg_train_f1arr), ('validation f1', avg_validation_f1arr)], file)


        torch.save(model.state_dict(), f'./trained_models/{data_name}/{data_name}_model_{model_name}.pth')
        best_accuracies[data_name] = best_val_score
        print(f'Best acc: {best_val_score}') 
        
    # for key, val in best_accuracies.items():
    #     print(f"{key}: {val}")
            
    with open(f"./trained_models/best_accuracies.json", "w") as file:
        json.dump(best_accuracies, file)
        
train(Negation)   
train(Conjunction)
train(Disjunction)
train(Existential)
train(Universal)
