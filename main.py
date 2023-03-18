import os
import argparse
import constants
from data.StartingDataset import StartingDataset
from train_functions.starting_train import starting_train
from importlib import import_module
from torchsummary import summary

# TODO: make a conda env to run everything 

def main(curr_model):
    # Get command line arguments

    args = constants.params[curr_model] 
    epochs, batch_size, n_eval = args['EPOCHS'], args['BATCH_SIZE'], args['N_EVAL']
        
    hyperparameters = {"epochs": epochs, "batch_size": batch_size}

    print(f"Epochs: {epochs}\n Batch size: {batch_size}")

    # Initalize dataset 
    val_dataset = StartingDataset("val")
    train_dataset = StartingDataset("train", v_index=val_dataset.val_indices)
    

    # Initialize model 
    network_class = import_module("networks." + curr_model).__getattribute__(curr_model)
    model = network_class(batch_size)
    model = model.float()
    # print("MODEL TYPE: ", model.dtype)

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
    # parser.add_argument('-t', '--test', action='store_true')
    args = parser.parse_args()
    curr_model= args.model
    # is_test = args.test

    main(curr_model)
