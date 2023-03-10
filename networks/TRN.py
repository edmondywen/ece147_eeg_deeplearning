import torch
import torch.nn as nn


class TRN(torch.nn.Module):
    """
    Basic logistic regression on 224x224x3 images.
    """

    def __init__(self):
        super().__init__()
        self.transformer = nn.Transformer(nhead=16, num_encoder_layers=12, num_decoder_layers=12, batch_first=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(224 * 224 * 3, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.transformer(x) 
        x = self.flatten(x)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x
