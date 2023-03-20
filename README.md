# EEG Project

## Running the Code Locally

### After activating your conda environment, run the following command:

```
python main.py <model name>
```
Valid model names can be found as the keys for the params dictionary in constants.py. 

### To run on test code, run the following command instead:
```
python test.py <path to model weights>
```

### To run on test code which isolates an experimental subject, run the following command instead:
```
python test_person.py <path to model weights>
```

When training, model runs will be saved to /runs/model_name


