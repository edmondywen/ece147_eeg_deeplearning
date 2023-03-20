import os
import argparse
import constants
from data.StartingDataset import StartingDataset
from data.PersonDataset import PersonDataset
from train_functions.starting_train import starting_train
from importlib import import_module
from torchsummary import summary

# TODO: make a conda env to run everything 

def main(args):
    # Get command line arguments

    curr_model = args.model
    params = constants.params[curr_model] 
    epochs, batch_size, n_eval = params['EPOCHS'], params['BATCH_SIZE'], params['N_EVAL']
        
    hyperparameters = {"epochs": epochs, "batch_size": batch_size}

    print(f"Epochs: {epochs}\n Batch size: {batch_size}")

    # Initalize dataset 
    if args.person:
        val_dataset = PersonDataset("val")
        train_dataset = PersonDataset("train", v_index = val_dataset.val_indices)
    else: 
        val_dataset = StartingDataset("val")
        train_dataset = StartingDataset("train", v_index=val_dataset.val_indices)

    # Initialize model 
    network_class = import_module("networks." + curr_model).__getattribute__(curr_model)
    model = network_class(batch_size)
    model = model.float()

    # summary(model, input_size=(22, 250))

    starting_train(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        model=model,
        hyperparameters=hyperparameters,
        n_eval=n_eval,
        curr_model_type=curr_model
    )


if __name__ == "__main__":
    # Arg Parsing
    parser = argparse.ArgumentParser()
    parser.add_argument('model', type=str)
    parser.add_argument('--person', action='store_true')
    args = parser.parse_args()

    main(args)
