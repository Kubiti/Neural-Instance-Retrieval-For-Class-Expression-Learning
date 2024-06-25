import numpy as np
import json
from collections import defaultdict


def decompose(concept_name: str) -> list:
        list_ordered_pieces = []
        i = 0
        while i < len(concept_name):
            concept = ''
            while i < len(concept_name) and not concept_name[i] in ['(', ')', '⊔', '⊓', '∃', '∀', '¬', '.', ' ']:
                concept += concept_name[i]
                i += 1
            if concept:
                list_ordered_pieces.append(concept)
            i += 1
        return list_ordered_pieces

def concept_length(concept_string):
    spec_chars = ["∃", "∀", "¬", "⊔", "⊓"]
    return len(decompose(concept_string)) + sum(map(concept_string.count, spec_chars))


kb = ['animals', 'family', 'lymphography', 'suramin', 'nctrer']

for data_name in kb:
    f1_path = f'./metrics/{data_name}/{data_name}_testing_f1_today.json'
    accuracy_path = f'./metrics/{data_name}/{data_name}_testing_accuracies_today.json'
    exp_path = f'./testing_data/test_data_{data_name}.json'
    save_path = f'./metrics/{data_name}/{data_name}_performance_per_len_updated.json'

    with open(f1_path, 'r') as f:
        f1_data = json.load(f)
    # f1_scores = []
    # print(f1_data)
    with open(accuracy_path, 'r') as f:
        acc_data = json.load(f)

    with open(exp_path, 'r') as f:
        data = json.load(f)
    expressions = [expr for (expr, label) in data]

    performance_per_len = defaultdict(lambda: defaultdict(list))

    for acc, f1, expr in zip(acc_data[0][1], f1_data[0][1], expressions):
        performance_per_len[concept_length(expr)]['accuracy'].append(acc)
        performance_per_len[concept_length(expr)]['f1'].append(f1)

    avg_per_len = defaultdict(lambda: defaultdict())
    for key in performance_per_len:
        avg_per_len[key]['accuracy'] = np.mean(performance_per_len[key]['accuracy'])
        avg_per_len[key]['f1'] = np.mean(performance_per_len[key]['f1'])

    print(avg_per_len)


    with open(save_path, 'w') as f:
        json.dump(avg_per_len, f)