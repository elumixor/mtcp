# ttH

## Plan

1. Reproduce the results
2. Include systematic uncertainties

## Code architecture

On the basic level, we just need to train a simple NN to classify the event.

### Data processing
Before we can train the network, we need to *preprocess* the data from the `*.root` files.
Data processing is done in the corresponding folder `data_processing`.
The processed data will be in the `processed_data` folder and will contain the following numpy arrays:
1. `features.npy` - the input features, shape `(n_events, n_features)`
2. `labels.npy` - the labels (e.g. ttH, ttZ, ttW, [VV](VV), other), shape `(n_events, n_classes)`
3. `weights.npy` - the [event weights](Event Weights), shape `(n_events, 1)`

It is done in two steps:
1. Convert the `*.root` files to `*.csv` files
2. Convert the `*.csv` files to `*.pkl` files
