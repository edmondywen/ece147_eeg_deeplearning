import torch
import torch.nn as nn
from tqdm import tqdm
from data.StartingDataset import StartingDataset
global device 
import constants
import argparse
device = torch.device('cpu') 

def test(model_path):
    #1. load model and set to evaluation mode
    model = torch.load(model_path)
    model.eval()
    model = model.to(device)
    loss_fn = nn.CrossEntropyLoss()

    #2. get test data
    test_dataset = StartingDataset('test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=constants.TEST_BATCH_SIZE
    )

    #3. test
    loss, correct, count = 0, 0, 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            input_data, label_data = batch
            # Move both images and labels to GPU, if available
            input_data = input_data.to(device)
            label_data = label_data.to(device)

            pred = model(input_data)
            loss += loss_fn(pred, label_data).mean().item()

            # Update both correct and count (use metrics for tensorboard)
            correct += (torch.argmax(pred, dim=1) == label_data).sum().item()
            count += len(label_data)
    
    print(loss, correct/count)
    return loss, correct/count

parser = argparse.ArgumentParser()
parser.add_argument('model_path', type=str)
args = parser.parse_args()
test(args.model_path)