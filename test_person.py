import torch
import torch.nn as nn
from tqdm import tqdm
from data.PersonDataset import PersonDataset
global device 
import constants
import argparse
import time
device = torch.device('cpu') 

def test(model_path):
    #1. load model and set to evaluation mode
    model = torch.load(model_path)
    model.eval()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    #2. get test data
    test_dataset_same = PersonDataset('test_same')
    test_dataset_new = PersonDataset('test_new')
    test_loader_same = torch.utils.data.DataLoader(
        test_dataset_same, batch_size=constants.TEST_BATCH_SIZE
    )
    test_loader_new = torch.utils.data.DataLoader(
        test_dataset_new, batch_size=constants.TEST_BATCH_SIZE
    )
    num_test = len(test_dataset_same)

    #3. test same
    print("\n---Testing Same People---")
    loss, correct, count = 0, 0, 0
    forward_time = 0
    with torch.no_grad():
        for batch in tqdm(test_loader_same):
            input_data, label_data = batch
            # Move both images and labels to GPU, if available
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            
            start_time = time.time()
            pred = model(input_data)
            end_time = time.time()
            forward_time += end_time - start_time
            loss += loss_fn(pred, label_data).mean().item()

            # Update both correct and count (use metrics for tensorboard)
            correct += (torch.argmax(pred, dim=1) == label_data).sum().item()
            count += len(label_data)
    
    print("Time spent in forward prop (sec): ", forward_time, "Number test entries: ", num_test, "Inference time: ", forward_time/num_test)
    print(loss, correct/count)
       
    #4. test new
    print("\n---Testing New People---")
    loss, correct, count = 0, 0, 0
    forward_time = 0
    with torch.no_grad():
        for batch in tqdm(test_loader_new):
            input_data, label_data = batch
            # Move both images and labels to GPU, if available
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            
            start_time = time.time()
            pred = model(input_data)
            end_time = time.time()
            forward_time += end_time - start_time
            loss += loss_fn(pred, label_data).mean().item()

            # Update both correct and count (use metrics for tensorboard)
            correct += (torch.argmax(pred, dim=1) == label_data).sum().item()
            count += len(label_data)
    
    print("Time spent in forward prop (sec): ", forward_time, "Number test entries: ", num_test, "Inference time: ", forward_time/num_test)
    print(loss, correct/count)
    return None

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str)
args = parser.parse_args()
test(args.model_path)