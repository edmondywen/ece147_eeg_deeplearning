DATASET_PATH = "../../project/"
NUM_NODES = 22
CURR_MODEL = "CNN_LSTM"

params = {
    "RNN" : {
        "EPOCHS" : 100, 
        "BATCH_SIZE": 32, 
        "N_EVAL": 10 
    }, 
    "TRN": {
        "EPOCHS" : 100, 
        "BATCH_SIZE" : 32, 
        "N_EVAL" : 10,
        "NUM_HEADS" : 16,
    },
    'CNN_LSTM': {
        "EPOCHS": 100,
        "BATCH_SIZE": 32,
        "N_EVAL": 20
    }
}
