import os

import constants
from train_functions.starting_train import starting_train
from importlib import import_module
from torchsummary import summary

args = constants.params[constants.CURR_MODEL] 
epochs, batch_size, n_eval = args['EPOCHS'], args['BATCH_SIZE'], args['N_EVAL']

network_class = import_module("networks." + constants.CURR_MODEL).__getattribute__(constants.CURR_MODEL)
model = network_class(batch_size)
model = model.float()

summary(model, input_size=(22, 250))