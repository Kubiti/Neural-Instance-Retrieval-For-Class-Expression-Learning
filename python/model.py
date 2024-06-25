import torch
import torch.nn as nn

class Negation(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(Negation, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def encode(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        # x = self.activation(x)
        return x

    def forward(self, x):
        x = self.encode(x)
        x = self.head(x)
        return torch.sigmoid(x)


class Conjunction(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(Conjunction, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def encode(self, x1, x2):
        x = self.fc1(torch.cat([x1, x2], dim=1))
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        # x = self.activation(x)
        return x

    def forward(self, x1, x2):
        x = self.encode(x1, x2)
        x = self.head(x)
        return torch.sigmoid(x)

class Disjunction(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(Disjunction, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def encode(self, x1, x2):
        x = self.fc1(torch.cat([x1, x2], dim=1))
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        # x = self.activation(x)
        return x

    def forward(self, x1, x2):
        x = self.encode(x1, x2)
        x = self.head(x)
        return torch.sigmoid(x)

class Existential(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(Existential, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def encode(self, x1, x2):
        x = self.fc1(torch.cat([x1, x2], dim=1))
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        # x = self.activation(x)
        return x

    def forward(self, x1, x2):
        x = self.encode(x1, x2)
        x = self.head(x)
        return torch.sigmoid(x)

class Universal(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(Universal, self).__init__()
        self.fc1 = nn.Linear(input_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.head = nn.Linear(hidden_size, output_size)
        self.activation = nn.ReLU()

    def encode(self, x1, x2):
        x = self.fc1(torch.cat([x1, x2], dim=1))
        x = self.activation(x)
        x = self.fc2(x)
        x = self.activation(x)
        x = self.fc3(x)
        x = self.activation(x)
        return x

    def forward(self, x1, x2):
        x = self.encode(x1, x2)
        x = self.head(x)
        return torch.sigmoid(x)
        # out = self.fc(out[:, -1, :])
        
        # return torch.sigmoid(out)
    

