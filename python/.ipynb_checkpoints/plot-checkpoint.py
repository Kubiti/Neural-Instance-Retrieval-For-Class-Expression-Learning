import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
# from traindata import kb
kb = ['animals', 'family', 'lymphography',   'mutagenesis', 'nctrer', 'suramin', 'vicodi', 'carcinogenesis']


fig, ax = plt.subplots(3, 3, figsize=(15,10))

path = f"./trained_models/animals_accuracies.json"
i = 0
for name in kb:
    row, col = i // 3, i % 3
    path = f"./trained_models/{name}_accuracies.json"
    acc = []
    with open(path, 'r') as f:
    #     for line in f:
    #         acc.append(json.loads(line))
        data = json.load(f)
    train_acc = data[0]['train_accuracy'][:1000]
    validation_acc = data[1]['validation_accuracy'][:1000]
    x = np.arange(1000)
    # y = np.random.randn(1000)     # 1000 random values from a normal distribution

    # Create the plot
    ax[row, col].plot(x, train_acc,  'b--o', markersize=0.1, label='Train Accuracy',)
    ax[row, col].plot(x, validation_acc,  'g--o', markersize=0.1, label='Validation Accuracy')

    ax[row, col].set_title(f'Plot of Accuracies for {name}')
    ax[row, col].set_xlabel('Epochs')
    ax[row, col].set_ylabel('Accuracy')
    ax[row, col].legend()

    i += 1
plt.subplots_adjust(wspace=0.5, hspace=0.5)
plt.show()
