import torch.nn as nn


class MatchingNetwork(nn.Module):
    def __init__(self, embedding_net):
        super(MatchingNetwork, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x):
        return self.embedding_net(x)

    def get_embedding(self, x):
        return self.forward(x)
