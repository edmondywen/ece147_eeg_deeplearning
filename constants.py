DATASET_PATH = "../../project/"

CURR_MODEL = "RNN"

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
    }
}
