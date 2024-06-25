# import libraries
from ontolearn.concept_learner import NCES
from ontolearn.knowledge_base import KnowledgeBase
from ontolearn.metrics import F1
from owlapy.parser import DLSyntaxParser
import json
import numpy as np


# load datasets
KB = KnowledgeBase(path='../../../NCESData/family/family.owl')
L = sorted([ind.str.split("/")[3] for ind in KB.individuals()])

filepath = '/home/student117/Documents/aims/project/NCESData/family/training_data/Data.json'
with open(filepath, 'r') as f:
    data = json.load(f)
# data


# Create list of y vectors for each expressions
keys = data.keys()
datapoints = []
# i = 0
# for exp in keys:
#     expressions[exp] = np.zeros(len(L))
#     # print(exp)
#     # i+= 1
#     # j = 0
#     for j in range(len(L)):
#         expressions[exp][j] = int(L[j] in data[exp]['positive examples'])

# print(expressions)
name_to_ids = {name: idx for idx, name in enumerate(L)} 

for exp in keys:
    y_C = np.zeros(len(L))
    pos = data[exp]['positive examples']
    Ind_pos = list(map(name_to_ids.get, pos))
    y_C[Ind_pos] = 1
    print(exp, y_C)
    datapoints.append((exp, y_C))











