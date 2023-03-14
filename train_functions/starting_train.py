import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
import os
global device 
device = torch.device('cpu') 

if torch.cuda.is_available():
    device = torch.device('cuda:0')


def starting_train(train_dataset, val_dataset, model, hyperparameters, n_eval, curr_model_type):
    """
    Trains and evaluates a model.

    Args:
        train_dataset:   PyTorch dataset containing training data.
        val_dataset:     PyTorch dataset containing validation data.
        model:           PyTorch model to be trained.
        hyperparameters: Dictionary containing hyperparameters.
        n_eval:          Interval at which we evaluate our model.
        curr_model_type: String of our model type 
    """

    # Get keyword arguments
    batch_size, epochs = hyperparameters["batch_size"], hyperparameters["epochs"]

    # Initialize dataloaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    ) 
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True
    )

    # Initalize optimizer (for gradient descent) and loss function
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-2) # increase weight decay
    loss_fn = nn.CrossEntropyLoss()

    writer = SummaryWriter()
    # append the model type inside runs to the writing path
    writing_path = f"runs/{curr_model_type}/{writer.log_dir.split('/')[1]}"
    writer = SummaryWriter(writing_path)
    model = model.to(device)

    # Initialize summary writer (for logging)
    # summary_path = "" # <-- todo! add this in CONSTANTS.py for tensorboard gen 
    # tb_summary = None
    # if summary_path is not None:
    #     tb_summary = torch.utils.tensorboard.SummaryWriter(summary_path)
    best_valid_acc = -1


    step = 0
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1} of {epochs}")

        # Loop over each batch in the dataset
        for batch in tqdm(train_loader):

            input_data, label_data = batch
            input_data = input_data.to(device)
            label_data = label_data.to(device)
            
            pred = model(input_data)

            # Prediction, label data have same shape
            loss = loss_fn(pred, label_data) 
            pred = pred.argmax(axis=1)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print(f"\n    Train Loss: {loss.item()}")



            # Periodically evaluate our model + log to Tensorboard
            if step % n_eval == 0:

                # Compute training loss and accuracy.
                train_accuracy = compute_accuracy(pred, label_data)
                print(f"    Train Accu: {train_accuracy}")




                # Compute validation loss and accuracy.
                valid_loss, valid_accuracy = evaluate(val_loader, model, loss_fn)
                if valid_accuracy > best_valid_acc:
                    print(f"Best validation accuracy of {valid_accuracy} found!")
                    best_valid_acc = valid_accuracy
                    print("Saving the model!")
                    cwd = os.getcwd()
                    save_path = f"{writing_path}/best_model.pt"
                    torch.save(model.state_dict(),save_path)
                

                # # Log the results to Tensorboard.
                if writer:
                    writer.add_scalars('losses', {
                        'Validation' : valid_loss,
                        'Train': loss},
                        epoch)
                    writer.add_scalars('Accuracy',{
                        'Validation' : valid_accuracy,
                        'Train' : train_accuracy},
                        epoch)

                print(f"    Valid Loss: {valid_loss}")
                print(f"    Valid Accu: {valid_accuracy}")

                model.train()

            step += 1

        print()
    print("Saving final model")
    save_path = f"{writing_path}/final_model.pt"
    torch.save(model.state_dict(),save_path)


    


def compute_accuracy(outputs, labels):
    """
    Computes the accuracy of a model's predictions.

    Example input:
        outputs: [0.7, 0.9, 0.3, 0.2]
        labels:  [1, 1, 0, 1]

    Example output:
        0.75
    """
    n_correct = (outputs == labels).sum().item()
    n_total = len(outputs)
    return n_correct / n_total


def evaluate(val_loader, model, loss_fn):
    """
    Computes the loss and accuracy of a model on the validation dataset!
    """
    model.eval()
    model = model.to(device)

    loss, correct, count = 0, 0, 0
    with torch.no_grad(): 
        for batch in val_loader:
            input_data, label_data = batch

            # Move both images and labels to GPU, if available
            input_data = input_data.to(device)
            label_data = label_data.to(device)

            pred = model(input_data)
            loss += loss_fn(pred, label_data).mean().item()

            # Update both correct and count (use metrics for tensorboard)
            correct += (torch.argmax(pred, dim=1) == label_data).sum().item()
            count += len(label_data)

    return loss, correct/count
