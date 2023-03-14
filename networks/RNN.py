import torch
import torch.nn as nn
import constants

class RNN(torch.nn.Module):
    def __init__(
        self,
        batch_size,
        n_layers=10,
    ):
        
        input_size=250
        hidden_size=128
        num_classes = 4

        super(RNN, self).__init__()
        self.n_layers = 1
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        self.rnn = nn.GRU(
            input_size,
            hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=0.2,
        )

        self.fc1 = nn.Linear(self.hidden_size * constants.NUM_NODES, 500)
        self.fc2 = nn.Linear(500, 1000)
        self.fc3 = nn.Linear(1000, 200)
        self.fc4 = nn.Linear(200, num_classes)
        self.relu = nn.ReLU()

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

        x = self.relu(self.fc1(output.reshape(self.batch_size, -1)))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        output = self.fc4(x)
        return output