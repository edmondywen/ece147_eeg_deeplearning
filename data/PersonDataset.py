import torch
import numpy as np

DATASET_PATH = "project_data/"

class PersonDataset(torch.utils.data.Dataset):
    def data_prep(self, X,y,sub_sample,average,noise):
            total_X = None
            total_y = None

            # Trimming the data (sample,22,1000) -> (sample,22,500)
            X = X[:,:,0:500]
            # print('Shape of X after trimming:',X.shape)

            # Maxpooling the data (sample,22,1000) -> (sample,22,500/sub_sample)
            X_max = np.max(X.reshape(X.shape[0], X.shape[1], -1, sub_sample), axis=3)
            total_X = X_max
            total_y = y
            # print('Shape of X after maxpooling:',total_X.shape)

            # Averaging + noise 
            X_average = np.mean(X.reshape(X.shape[0], X.shape[1], -1, average),axis=3)
            X_average = X_average + np.random.normal(0.0, 0.5, X_average.shape)
            total_X = np.vstack((total_X, X_average))
            total_y = np.hstack((total_y, y))
            # print('Shape of X after averaging+noise and concatenating:',total_X.shape)

            # Subsampling
            for i in range(sub_sample):
                X_subsample = X[:, :, i::sub_sample] + \
                                    (np.random.normal(0.0, 0.5, X[:, :,i::sub_sample].shape) if noise else 0.0)
                total_X = np.vstack((total_X, X_subsample))
                total_y = np.hstack((total_y, y))
            # print('Shape of X after subsampling and concatenating:',total_X.shape)
            return total_X,total_y


    def __init__(self, split, v_index = None, train_val_split=0.2):

        if split == "train":
            self.X = np.load(DATASET_PATH + "X_train_valid.npy") # (2115, 22, 1000)
            labels = np.load(DATASET_PATH + "y_train_valid.npy") - 769
            people = np.load(DATASET_PATH + "person_train_valid.npy")
            self.train_indices = np.where(people != 0)[0] # remove person 0 from training
            self.X = self.X[self.train_indices]
            self.y = labels[self.train_indices]
            self.X, self.y = self.data_prep(self.X, self.y, 2, 2,True)
            self.X = torch.from_numpy(self.X).double() 
            self.y = self.y.astype(np.int64) 
            print("Train Shape is: ", self.X.size(), " labels ", self.y.size)
            self.person = np.load(DATASET_PATH + "person_train_valid.npy") # (2115, 1)

        elif split == "val":
            self.X = np.load(DATASET_PATH + "X_train_valid.npy") # (2115, 22, 1000)
            labels = np.load(DATASET_PATH + "y_train_valid.npy") - 769
            people = np.load(DATASET_PATH + "person_train_valid.npy")
            self.val_indices = np.where(people == 0)[0] # use only person 0 for validation
            self.X = self.X[self.val_indices]
            self.y = labels[self.val_indices]
            self.X, self.y = self.data_prep(self.X, self.y, 2, 2,True)
            self.X = torch.from_numpy(self.X).double() 
            self.y = self.y.astype(np.int64) 
            print("Valid Shape is: ", self.X.size(), " labels ", self.y.size)

        # used to test if inference samples come from people in the training set
        elif split == "test_same":
            self.X = np.load(DATASET_PATH + "X_test.npy") # (443, 22, 1000)
            labels = np.load(DATASET_PATH + "y_test.npy") - 769 # (443,)
            people = np.load(DATASET_PATH + "person_test.npy")
            indices = np.where(people != 0)[0]
            print(indices)
            self.X = self.X[indices]
            labels = labels[indices]

            # preprocessing
            self.X, self.y = self.data_prep(self.X, labels, 2, 2,True)
            self.X = torch.from_numpy(self.X).double() 
            self.y = self.y.astype(np.int64)
            print("Test Shape is: ", self.X.size(), " labels ", self.y.size)

        # used to test if inference samples do not come from people in the training set
        elif split == "test_new":
            self.X = np.load(DATASET_PATH + "X_test.npy") # (443, 22, 1000)
            labels = np.load(DATASET_PATH + "y_test.npy") - 769 # (443,)
            people = np.load(DATASET_PATH + "person_test.npy")
            indices = np.where(people == 0)[0]
            self.X = self.X[indices]
            labels = labels[indices]

            # preprocessing
            self.X, self.y = self.data_prep(self.X, labels, 2, 2,True)
            self.X = torch.from_numpy(self.X).double() 
            self.y = self.y.astype(np.int64)
            self.person = np.load(DATASET_PATH + "person_test.npy") # (443, 1)
            print("Test Shape is: ", self.X.size(), " labels ", self.y.size)

        else:
            raise Exception("Invalid split name")
        self.length = self.X.shape[0]

    def __getitem__(self, index):
        inputs = self.X[index] 
        label = self.y[index]
        return inputs, label

    def __len__(self):
        return self.length

