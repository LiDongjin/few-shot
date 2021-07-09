import torch.nn as nn


class SiameseNet(nn.Module):
    def __init__(self, embedding_net, embedding_size):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net
        self.alpha = nn.Linear(embedding_size, 1)   # additional parameters weighting the importance of the component-wise distance.
        self.loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, x1, x2, label):
        output1 = self.embedding_net(x1)    # [batch, embedding_size]
        output2 = self.embedding_net(x2)    # [batch, embedding_size]
        l1 = abs(output2 - output1)
        weighted_distance = self.alpha(l1)
        loss = self.loss_fn(weighted_distance, label)
        return loss

    def get_embedding(self, x):
        return self.embedding_net(x)


