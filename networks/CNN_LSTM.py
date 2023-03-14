import torch
import torch.nn as nn
import constants

class CNN_LSTM(torch.nn.Module):
    def __init__(
        self,
        batch_size,
        n_layers=1,
    ):
        
        hidden_size= 8
        num_classes = 4

        super(CNN_LSTM, self).__init__()
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        
        self.conv1 = nn.Conv1d(in_channels=22, out_channels=128, kernel_size=5)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=192, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=192, out_channels=128, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        
        self.rnn = nn.GRU(
            27,
            hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=0.5,
        )

        # self.fc1 = nn.Linear(self.hidden_size * constants.NUM_NODES, 500)
        # self.fc2 = nn.Linear(500, 1000)
        # self.fc3 = nn.Linear(1000, 200)
        # self.fc4 = nn.Linear(200, num_classes)
        # self.relu = nn.ReLU()
        self.decoder = nn.Linear(128 * 8, num_classes)
    def init_hidden(self):
        return torch.randn(self.n_layers, self.batch_size, self.hidden_size)
    
    def forward(self, inputs):
        # Avoid breaking if the last batch has a different size
        # print("input type: ", inputs.dtype)

        batch_size = inputs.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
       
        inputs = inputs.float()
        x = self.dropout(self.maxpool(self.conv1(inputs)))
        x = self.dropout(self.maxpool(self.conv2(x)))
        x = self.dropout(self.maxpool(self.conv3(x)))
        
        output, hidden = self.rnn(x, self.init_hidden())
        # 32, 22, 128 (BATCH_SIZE, NUM_NODES, HIDDEN_SIZE)
        
        # x = self.relu(self.fc1(output.reshape(self.batch_size, -1)))
        # x = self.relu(self.fc2(x))
        # x = self.relu(self.fc3(x))
        # output = self.fc4(x)
        output = self.decoder(output.reshape(self.batch_size, -1))
        return output