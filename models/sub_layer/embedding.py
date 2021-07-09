import torch
import torch.nn as nn


# BERT encoding 之后的两层全连接
class EmbeddingNet(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.0, layernorm=False, batchnorm=False):
        super(EmbeddingNet, self).__init__()
        self.dropout1 = nn.Dropout(p=dropout_p)
        self.dropout2 = nn.Dropout(p=0.4)
        self.Tanh = nn.Tanh()  ##nn.Tanh() / nn.ReLU() etc
        self.fc1 = nn.Linear(hidden_size, 200)
        if batchnorm:
            self.bn1 = nn.BatchNorm1d(num_features=200)
        else:
            self.bn1 = None
        if layernorm:
            self.ln1 = nn.LayerNorm(200)
        else:
            self.ln1 = None
        self.fc2 = nn.Linear(200, output_size)

    def forward(self, x):
        output = self.dropout1(x)
        output = self.fc1(output)
        if not self.bn1 is None:
            output = self.bn1(output)
        if not self.ln1 is None:
            output = self.ln1(output)
        output = self.Tanh(output)
        output = self.dropout2(output)
        output = self.fc2(output)

        return output

    def get_embedding(self, x):
        return self.forward(x)