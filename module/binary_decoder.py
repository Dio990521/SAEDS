import torch.nn as nn
import torch

class BinaryDecoder(nn.Module):
    def __init__(self, hidden_dim, num_label):
        super(BinaryDecoder, self).__init__()
        self.num_label = num_label
        self.linear_layer = nn.Linear(hidden_dim, self.num_label)

    def forward(self, out):
        pred = self.linear_layer(out)
        return pred
