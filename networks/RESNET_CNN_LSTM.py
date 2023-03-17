
import torch
import torch.nn as nn
import constants
from torchvision import models
 
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
        
        # initialize resnet for transfer learning
        self.resnet18 = models.resnet18(pretrained=True) # i don't think this will work because resnet uses 2d convolutions and I don't want to implement a 1d resnet
        self.resnet18 = nn.Sequential(*list(self.resnet18.modules())[:-1])

        # init remaining layers 
        self.conv1 = nn.Conv1d(in_channels=22, out_channels=128, kernel_size=5)
        self.maxpool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=192, kernel_size=5)
        self.conv3 = nn.Conv1d(in_channels=192, out_channels=128, kernel_size=5)
        self.dropout = nn.Dropout(0.5)
        # self.batchnorm1 = nn.BatchNorm1d(num_features=128)
        # self.batchnorm2 = nn.BatchNorm1d(num_features=192)
        # self.batchnorm3 = nn.BatchNorm1d(num_features=128)

        self.rnn = nn.GRU(
            27,
            hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
            dropout=0.5,
        )

        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(128*8, num_classes)
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

        output = self.fc1(output.reshape(self.batch_size, -1));
        return output