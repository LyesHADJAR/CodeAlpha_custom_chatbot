import torch
import torch.nn as nn


class neural_network(nn.Module):
    def __init__(self, inpu_size, hidden_size, num_classes):
        super(neural_network, self).__init__()
        self.l1 = nn.Linear(inpu_size, hidden_size)
        self.l2 = nn.Linear(hidden_size, hidden_size)
        self.l3 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()   
    
    def forward(self, x):
        output = self.l1(x)
        output = self.relu(output)
        output = self.l2(output)
        output = self.relu(output)
        output = self.l3(output)
        return output
