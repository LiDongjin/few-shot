import torch.nn as nn


class ProtoNet(nn.Module):
    def __init__(self, embedding_net, embedding_size):
        super(ProtoNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x):
        return self.embedding_net(x)

    def get_embedding(self, x):
        return self.embedding_net(x)


def prototypical_loss(input, target):



