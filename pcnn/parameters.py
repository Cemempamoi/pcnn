"""
File defining all the parameters.

If you want to modify any of these default values, don't forget to change them both in `create_parameters` and
in the individual parameters below
"""

import os
import torch.nn.functional as F


DATA_SAVE_PATH = os.path.join("saves", "data")
MODEL_SAVE_PATH = os.path.join("saves", "models")

# Create missing directories
for path in [DATA_SAVE_PATH, MODEL_SAVE_PATH]:
    if not os.path.isdir(path):
        os.mkdir(path)

def parameters(name: str = "Default_model", save_path: str = MODEL_SAVE_PATH,
                seed: int = 0, batch_size: int = 128, shuffle: bool = True, n_epochs: int = 20,
                learning_rate: float = 0.05, decrease_learning_rate:bool = True,
                heating: bool = True, cooling: bool = True, loss = F.mse_loss,
                warm_start_length: int = 12, minimum_sequence_length: int = 5, maximum_sequence_length: int = 240,
                overlapping_distance: int = 4, validation_percentage: float = 0.2, test_percentage: float = 0.1,
                learn_initial_hidden_states: bool = True, feed_input_through_nn: bool = True,
                input_nn_hidden_sizes: list = [128], lstm_hidden_size: int = 256,
                layer_norm: bool = True, lstm_num_layers: int = 1,
                output_nn_hidden_sizes: list = [128, 128], division_factor: float = 10.,
                device: str = None, verbose: int = 2, eps: float = 1e-6):
    """
    Parameters of the models

    Returns:
        name:                       Name of the model
        save_path:                  Where to save models
        seed:                       To fix the seed for reproducibility
        heating:                    Whether to use the model for the heating season
        cooling:                    Whether to use the model for the cooling season
        room_models:                Which rooms to model
        learn_initial_hidden_states:Whether to learn the initial hidden and cell states
        warm_start_length:          Length of data to warm start the model (autoregression terms required to
                                      initialize hidden and cell states
        minimum_sequence_length:    Minimum length of a prediction sequence (forward)
        maximum_sequence_length:    Maximum length of a sequence to predict
        overlapping_distance:       Distance between overlapping sequences to predict
        batch_size:                 Batch size for the training of models
        shuffle:                    Flag to shuffle the data in training or testing procedure
        n_epochs:                   Number of epochs to train the model
        learning_rate:              Learning rate of the models
        decrease_learning_rate:     Flag to adjust the learning rate while training models
        validation_percentage:      Percentage of the data to put in the validation set
        test_percentage:            Percentage of the data to put in the test set
        feed_input_through_nn:      Flag whether to preprocess the input before the LSTM
        input_nn_hidden_sizes:      Hidden sizes of the NNs processing the input
        lstm_hidden_size:           Hidden size of the LSTMs processing the input
        lstm_num_layers:            Number of layers for the LSTMs
        layer_norm:                 Flag whether to put a normalization layer after the LSTMs
        output_nn_hidden_sizes:     Hidden sizes of the NNs processing the output
        division_factor:            Factor to scale the output of the networks to ease learning
        verbose:                    Verbose of the models
        eps:                        Small value used for precision
    """

    assert cooling | heating, "At least heating or cooling needs to be true, otherwise nothing can be done"

    if feed_input_through_nn:
        assert len(input_nn_hidden_sizes) > 0, "You need to provide some hidden sizes for the input NN."
        if type(input_nn_hidden_sizes) == int:
            input_nn_hidden_sizes = list(input_nn_hidden_sizes)

    assert len(output_nn_hidden_sizes) > 0, "You need to provide some hidden sizes for the output NN."
    if type(output_nn_hidden_sizes) == int:
        output_nn_hidden_sizes = list(output_nn_hidden_sizes)

    if not isinstance(division_factor, list):
        division_factor = [division_factor]

    return dict(name=name,
                save_path=save_path,
                seed=seed,
                heating=heating,
                cooling=cooling,
                loss=loss,
                learn_initial_hidden_states=learn_initial_hidden_states,
                warm_start_length=warm_start_length,
                minimum_sequence_length=minimum_sequence_length,
                maximum_sequence_length=maximum_sequence_length,
                overlapping_distance=overlapping_distance,
                batch_size=batch_size,
                shuffle=shuffle,
                n_epochs=n_epochs,
                learning_rate=learning_rate,
                decrease_learning_rate=decrease_learning_rate,
                validation_percentage=validation_percentage,
                test_percentage=test_percentage,
                feed_input_through_nn=feed_input_through_nn,
                input_nn_hidden_sizes=input_nn_hidden_sizes,
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                layer_norm=layer_norm,
                output_nn_hidden_sizes=output_nn_hidden_sizes,
                division_factor=division_factor,
                device=device,
                verbose=verbose,
                eps=eps)

