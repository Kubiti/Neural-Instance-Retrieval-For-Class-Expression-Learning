# import libraries
from ontolearn.knowledge_base import KnowledgeBase
import json
import numpy as np
import pandas as pd
import os

# filepath = '/home/jupyter-kubiti/python/NCESData/'
# filepath = f'../generated_data/'
filepath = f'../NCESData/'

# kb = ['animals', 'family', 'lymphography', 'nctrer', 'suramin']
kb = ['lymphography']


def traindata(kb):
    for kb_name in kb:
        kb_path = filepath + kb_name

        # load datasets
        KB = KnowledgeBase(path=kb_path + '/' + kb_name + '.owl') # update file path Done :)
        L = sorted([ind.str.split("/")[-1] for ind in KB.individuals()])
        print("Generating data for", kb_name)
        # with open(kb_path + '/LPs.json', 'r') as f:
        with open(kb_path + '/training_data/Data.json', 'r') as f:
            data = json.load(f)
        print(len(data))

        # Create list of y vectors for each expressions
        keys = data.keys()
        datapoints = [] 
        name_to_ids = {name: idx for idx, name in enumerate(L)} 

        for exp in keys:
            if 'None' in exp:
                continue
            y_C = np.zeros(len(L))
            pos = data[exp]['positive examples']
            Ind_pos = list(map(name_to_ids.get, pos))
            y_C[Ind_pos] = 1
            datapoints.append((exp, list(y_C)))
        # with open(f"{filepath}{kb_name}/train_data_{kb_name}.json", "w") as file:
        with open(f"./training_data/prev/train_data_{kb_name}.json", "w") as file:
            json.dump(datapoints, file)


        
traindata(kb)

# data_path = f'./NCESData/family/training_data/Data.json'

def map_new_name(name):
    if name in Names:
        return name.split('#')[-1]
    return name

def dataset_gen(kb):
    for kb_name in kb:
        print(f'In {kb_name}')
        with open(f'{filepath}{kb_name}/LPs.json', 'r')  as f:
            data = json.load(f)
        kb = KnowledgeBase(path=f'./NCESData/{kb_name}/{kb_name}.owl')
        # first = list(kb.ontology.classes_in_signature())[0]
        # c.to_string_id().split("#")[-1]

        emb = pd.read_csv(f'./NCESData/{kb_name}/embeddings/ConEx_entity_embeddings.csv', index_col=0)
        

        classes = [c.to_string_id().split("/")[-1] for c in kb.ontology.classes_in_signature()]
        properties = [r.to_string_id().split("/")[-1] for r in kb.ontology.object_properties_in_signature()]
        classtoinstance = {C.to_string_id().split("/")[-1]: [ind.str.split("/")[-1] for ind in kb.individuals(C)] for C in kb.ontology.classes_in_signature()}
        classrename = {c:c.split('#')[-1] for c in classtoinstance.keys()}
        emb_copy = emb.copy()
        #emb_copy = emb_copy.reset_index()
        old_index = emb_copy.index
        global Names
        Names = classes
        #print('old', old_index.rename(classrename).tolist())
        emb_copy.index = list(map(map_new_name, old_index))
        old_index = emb_copy.index
        Names = properties
        emb_copy.index = list(map(map_new_name, old_index))
        #emb_copy.drop(columns=['Unnamed: 0'], inplace=True)

        for idx in range(len(emb_copy)):
            if emb_copy.index[idx] in classes and len(classtoinstance[emb_copy.index[idx]])>0:
                emb_copy.loc[emb_copy.index[idx], :] = emb_copy.loc[classtoinstance[emb_copy.index[idx]]].mean(axis=0) # compute embedding of a class as average embedding of its instances
                #if len(classtoinstance[emb_copy.index[idx]]) == 0:
                    # print(emb_copy.index[idx], classtoinstance[emb_copy.index[idx]])
                    # print(emb_copy.index[idx], emb.loc[emb_copy.index[idx]])
                    #emb_copy.loc[emb_copy.index[idx], :] = emb.loc[emb_copy.index[idx]]
                #    print('Failed:', emb_copy.index[idx])
                # compute embedding of a class as average embedding of its instances
                
                
        emb_copy.to_csv(f'{filepath}{kb_name}/{kb_name}_emb.csv') # save for all datasets

        # print(f'In {kb_name}')
        print(emb.head(3))
        print(emb_copy.head(3))

# dataset_gen(kb)
# if __name__ == '__main__':
#     dataset_gen(kb)