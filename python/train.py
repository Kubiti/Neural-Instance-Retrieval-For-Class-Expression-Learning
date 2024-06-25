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
# from traindata import kb
# kb = ['carcinogenesis', 'vicodi', 'mutagenesis']
# kb = ['animals', 'family', 'lymphography',   'mutagenesis', 'nctrer', 'suramin']
kb = ['mutagenesis']
import helper as h


from model import Negation, Conjunction, Disjunction, Existential, Universal


best_accuracies = {}

hidden_size = 60
# num_layers = 4
learning_rate = 0.001
num_epochs = 500
batch_size = 64


def train(model):

    for data_name in kb:
        print(f'In {data_name}')
        data_path = f'./training_data/train_data_{data_name}.json'
        emb_path = f'./training_data/{data_name}_emb.csv'
        inst_emb = pd.read_csv(emb_path)
        if data_name != 'animals' and data_name != 'mutagenesis':
            inst_emb['Unnamed: 0'] = inst_emb['Unnamed: 0'].apply(lambda x: x.replace(f"{'NTNames' if data_name == 'semantic_bible' else ('ontology' if data_name == 'vicodi' else data_name)}#", ''))
        inst_emb.set_index(inst_emb.columns[0], inplace=True)
        # if data_name == 'semantic_bible':


        
        input_size = inst_emb.shape[1]
        
        
        def read_json(path):
            with open(path, 'r') as f:
                data = json.load(f)
            data = [(exp, label) for (exp, label) in data]
    #         print('data', len(data))
            data = [(exp, label) for (exp, label) in data if h.neg_check(exp)]
            # data = [(exp, label) for (exp, label) in data if (h.three_check(exp) or h.atomic_check(exp)) and h.top_bot(exp)]
            print('data', len(data))
    #         print('atomic_data', len(atomic_data))
    #         print('three_data', len(three_data))
            output_size = len(data[1][1])
            random.shuffle(data)
            length = len(data)
            train_len = int(0.8 * length)
            train = data[:train_len]
            validation = data[train_len:]
            print('output size', output_size)
            return (train, validation, output_size)
        
        
        

        train_dataset, validation_dataset, output_size = read_json(data_path)
    #     print('train_dataset', len(train_dataset))
        

        train_dataset = NRDataset(train_dataset, inst_emb)
    #     print('train_dataset', len(train_dataset))
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=3, shuffle=True)
    #     print('len train_dataset', len(train_dataset))
    #     print('len(loader)', len(train_dataloader))

        validation_dataset = NRDataset(validation_dataset, inst_emb)
        validation_dataloader = DataLoader(validation_dataset, batch_size=batch_size, num_workers=3, shuffle=True)


        model = Negation(input_size, hidden_size, output_size, batch_size)        
        optimizer = optim.Adam(model.parameters(), lr=0.001)
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
    #             print(x.shape)
                # x = torch.reshape(x, (x.shape[0], 1, x.shape[1]))
                
                optimizer.zero_grad()
                outputs = model(x)
    #             print('x shape', x.shape)
    #             print('y shape', y.shape)
    #             print('outputs shape', outputs.shape)
                print('min', torch.max(outputs))
                print('max', torch.min(outputs))
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
                
                outputs = model(x)
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
            avg_validation_lossarr.append(validation_loss)

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

        with open(f"./metrics/{data_name}/{data_name}_accuracies_neg.json", "w") as file:
            json.dump([('train accuracy', avg_train_accarr), ('validation_accuracy', avg_validation_accarr)] , file)
        with open(f"./metrics/{data_name}/{data_name}_f1_neg.json", "w") as file:
            json.dump([('train f1', avg_train_f1arr), ('validation f1', avg_validation_f1arr)], file)


        torch.save(model.state_dict(), f'./trained_models/{data_name}/{data_name}_model_neg.pth')
        best_accuracies[data_name] = best_val_score
        print(f'Best acc: {best_val_score}') 
        
    # for key, val in best_accuracies.items():
    #     print(f"{key}: {val}")
            
    with open(f"./trained_models/best_accuracies.json", "w") as file:
        json.dump(best_accuracies, file)
        
            
    # 
train(Negation)