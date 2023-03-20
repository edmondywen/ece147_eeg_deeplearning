import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, models, transforms


class Resnet_Transfer(torch.nn.Module):
    def __init__(
        self,
        batch_size,
        n_layers=1,
    ):
        hidden_size= 64
        num_classes = 4
        super(Resnet_Transfer, self).__init__()
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.n_layers = n_layers

        #self.input_cnn = nn.Conv2d(in_channels=1, out_channels=64, kernel_size=(7, 7), padding=(3, 3), bias='False')
        
        # model = models.resnet18(pretrained=True)
        # self.resnet = nn.Sequential(*list(model.children())[:-2])
        # print(self.resnet[0].weight.size())
        # self.resnet[0].weight = nn.Parameter(self.resnet[0].weight.sum(dim=1, keepdim=True))
        
        
        # #Freeze weights
        # for param in self.resnet.parameters():
        #     param.requires_grad = False

        self.rnn = nn.GRU(
            72,
            hidden_size,
            num_layers=self.n_layers,
            batch_first=True,
        )
        #12 * 250
        self.conv1 = nn.Conv2d(in_channels=22, out_channels=64, kernel_size=(3, 7), padding=(1, 3), bias='True')
        self.maxpool1 = nn.MaxPool2d(kernel_size=(2, 10), stride=(1, 2)) #11 * 121
        self.conv2 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(3, 7), padding=(1, 3), bias='True') #11 x 121
        self.maxpool2 = nn.MaxPool2d(kernel_size=(2, 10), stride = (1, 3)) #10 x 38
        self.conv3 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=(3, 7), padding=(1, 3), bias='True')
        self.maxpool3 = nn.MaxPool2d(kernel_size=(2, 10), stride=(1, 4)) #9 x 8

        self.fc1 = nn.Linear(256 * 9 * 8, 4)
        self.fc2 = nn.Linear(1000, 200)
        self.fc3 = nn.Linear(200, num_classes)
        self.dropout = nn.Dropout(0.4)

    def init_hidden(self):
        return torch.randn(self.n_layers, self.batch_size, self.hidden_size)
    
    def forward(self, inputs):
        # Avoid breaking if the last batch has a different size
        # print("input type: ", inputs.dtype)

        batch_size = inputs.size(0)
        if batch_size != self.batch_size:
            self.batch_size = batch_size
        inputs = inputs.float()
        x = self.dropout(self.maxpool1(self.conv1(inputs)))
        x = self.dropout(self.maxpool2(self.conv2(x)))
        x = self.dropout(self.maxpool3(self.conv3(x)))
        x = x.reshape(self.batch_size, -1)
        output = self.fc1(x)
        return output