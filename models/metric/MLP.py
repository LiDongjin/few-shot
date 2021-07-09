import torch
import torch.nn as nn


class LR(nn.Module):

    def __init__(self, num_classes):
        super(LR, self).__init__()
        self.fc1 = nn.Linear(768, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        output = torch.softmax(x, dim=1)
        return output


class MLP(nn.Module):

    def __init__(self, num_classes):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(768, 100)
        self.relu1 = nn.Tanh()
        self.fc2 = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        output = torch.softmax(x, dim=1)
        return output