from StartingDataset import StartingDataset

train_sd = StartingDataset("train")
test_sd = StartingDataset("test")
print("Train X shape: {}, y shape: {}, y element: {}".format(train_sd.X.shape, train_sd.y.shape, train_sd.y[0]))
print("Test X shape: {}, y shape: {}, y element: {}".format(test_sd.X.shape, test_sd.y.shape, test_sd.y[0]))

print(test_sd.X.dtype)
