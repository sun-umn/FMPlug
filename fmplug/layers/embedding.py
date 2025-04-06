# third party
import torch
import torch.nn as nn


class ConstantEmbedding(nn.Module):
    """
    Class that creates a constant embedding layer.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.embedding_table = nn.Parameter(torch.empty((1, out_channels)))
        nn.init.uniform_(self.embedding_table, -(in_channels**0.5), in_channels**0.5)

    def forward(self, emb):
        return self.embedding_table.repeat(emb.shape[0], 1)
