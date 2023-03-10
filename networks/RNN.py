import torch
import torch.nn as nn

class RNN(torch.nn.Module):
    def __init__(
        self,
        batch_size,
        n_layers=1,
    ):
        
        input_size=1000
        hidden_size=128
        num_classes = 4

        super(RNN, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        self.rnn = nn.GRU(
            input_size,
            hidden_size,
            num_layers=n_layers,
            batch_first=True,
            dropout=0.2,
        )
        self.decoder = nn.Linear(hidden_size*22, num_classes)
        
    def init_hidden(self):
        return torch.randn(self.n_layers, self.batch_size, self.hidden_size)
    
    def forward(self, inputs):
        # Avoid breaking if the last batch has a different size
        batch_size = inputs.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        
        inputs = inputs.to(torch.float32)
        output, hidden = self.rnn(inputs, self.init_hidden())
        # 32, 22, 128 (BATCH_SIZE, NUM_NODES, HIDDEN_SIZE)

        output = self.decoder(output.reshape(self.batch_size, -1))
        return output