import json


kb = ['animals', 'family', 'lymphography', 'suramin', 'nctrer']

for data_name in kb:
    file_path = f'./training_data/prev/train_data_{data_name}.json'

    with open(file_path, 'r') as f:
        data = json.load(f)

    data = [exp for (exp, label) in data if ' ' in exp]
    # data = [exp for (exp, label) in data]
    # print(data[:5])
    print("In ", data_name, len(data))
    with open(f"./testing_data/comp_{data_name}.json", "w") as file:
            json.dump(data, file)