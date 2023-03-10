import torch
import numpy as np

DATASET_PATH = "project_data/"

class StartingDataset(torch.utils.data.Dataset):
    def __init__(self, split, train_val_split=0.2):
        if split == "train":
            self.X = np.load(DATASET_PATH + "X_train_valid.npy") # (2115, 22, 1000)
            self.X = self.X[int(train_val_split*len(self.X)):]
            #self.X = self.X.astype(np.float128)
            self.X = torch.from_numpy(self.X).double() 

            labels = np.load(DATASET_PATH + "y_train_valid.npy") - 769 # (2115,)
            labels = labels[int(train_val_split*len(labels)):]
            self.y = labels.astype(np.int64) 
            
            # self.y = np.zeros((labels.size, labels.max() + 1))
            # self.y[np.arange(labels.size), labels] = 1 # (2115, 4); converted into one-hot
            self.person = np.load(DATASET_PATH + "person_train_valid.npy") # (2115, 1)


        elif split == "val":
            self.X = np.load(DATASET_PATH + "X_train_valid.npy") # (2115, 22, 1000)
            self.X = self.X[:int(train_val_split*len(self.X))]
            self.X = torch.from_numpy(self.X).double() 


            labels = np.load(DATASET_PATH + "y_train_valid.npy") - 769 # (2115,)
            labels = labels[:int(train_val_split*len(labels))]
            self.y = labels.astype(np.int64)
            
            # self.y = np.zeros((labels.size, labels.max() + 1))
            # self.y[np.arange(labels.size), labels] = 1 # (2115, 4); converted into one-hot
            self.person = np.load(DATASET_PATH + "person_train_valid.npy") # (2115, 1)


        elif split == "test":
            self.X = np.load(DATASET_PATH + "X_test.npy") # (443, 22, 1000)
            self.X = torch.from_numpy(self.X).double() 

            labels = np.load(DATASET_PATH + "y_test.npy") - 769 # (443,)
            self.y = np.zeros((labels.size, labels.max() + 1))
            self.y[np.arange(labels.size), labels] = 1 # (443, 4); converted into one-hot
            self.person = np.load(DATASET_PATH + "person_test.npy") # (443, 1)
        else:
            raise Exception("Invalid split name")
        self.length = self.X.shape[0]

    def __getitem__(self, index):
        inputs = self.X[index] 
        label = self.y[index]
        return inputs, label

    def __len__(self):
        return self.length

