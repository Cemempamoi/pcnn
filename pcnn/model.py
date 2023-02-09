"""
File containing the base class of models, with general functions
"""

import os
import pandas as pd
import math
import time
from typing import Union

import random
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.nn.utils.rnn import pad_sequence
from torch import optim
import torch.nn.functional as F

from pcnn.module import PCNN, S_PCNN, M_PCNN, LSTM
from pcnn.data import prepare_data
from pcnn.util import model_save_name_factory, format_elapsed_time, inverse_normalize, inverse_standardize


class Model:
    """
    Class of models using PyTorch
    """

    def __init__(self, data: pd.DataFrame, interval: int, model_kwargs: dict, inputs_D: list, 
                module, rooms, case_column, out_column, neigh_column, temperature_column, power_column,
                Y_columns: list, X_columns: list = None, topology: dict = None, load_last: bool = False,
                load: bool = True):
        """
        Initialize a model.

        Args:
            model_kwargs:   Parameters of the models, see 'parameters.py'
            Y_columns:      Name of the columns that are to be predicted
            X_columns:      Sensors (columns) of the input data
        """


        assert module in ['PCNN', 'S_PCNN', 'M_PCNN', 'LSTM'],\
            f"The provided model type {module} does not exist, please chose among `'PCNN', 'S_PCNN', 'M_PCNN', 'LSTM'`."

        # Define the main attributes
        self.name = model_kwargs["name"]
        self.model_kwargs = model_kwargs
        self.rooms = rooms if isinstance(rooms, list) else [rooms]

        # Create the name associated to the model
        self.save_name = model_save_name_factory(rooms=self.rooms, module=module, model_kwargs=model_kwargs)

        if not os.path.isdir(self.save_name):
            os.mkdir(self.save_name)

        # Fix the seeds for reproduction
        self._fix_seeds(seed=model_kwargs["seed"])

        self.case_column = case_column
        self.out_column = out_column
        if neigh_column is not None:
            self.neigh_column = neigh_column if isinstance(neigh_column, list) else [neigh_column]
        else:
            self.neigh_column = neigh_column
        self.temperature_column = temperature_column if isinstance(temperature_column, list) else [temperature_column]
        self.power_column = power_column if isinstance(power_column, list) else [power_column]
        self.inputs_D = inputs_D
        self.topology = topology
        self.module = module

        self.unit = model_kwargs['unit']
        self.batch_size = model_kwargs["batch_size"]
        self.shuffle = model_kwargs["shuffle"]
        self.n_epochs = model_kwargs["n_epochs"]
        self.verbose = model_kwargs["verbose"]
        self.learning_rate = model_kwargs["learning_rate"]
        self.decrease_learning_rate = model_kwargs["decrease_learning_rate"]
        self.heating = model_kwargs["heating"]
        self.cooling = model_kwargs["cooling"]
        self.warm_start_length = model_kwargs["warm_start_length"]
        self.minimum_sequence_length = model_kwargs["minimum_sequence_length"]
        self.maximum_sequence_length = model_kwargs["maximum_sequence_length"]
        self.overlapping_distance = model_kwargs["overlapping_distance"]
        self.validation_percentage = model_kwargs["validation_percentage"]
        self.test_percentage = model_kwargs["test_percentage"]
        self.feed_input_through_nn = model_kwargs["feed_input_through_nn"]
        self.input_nn_hidden_sizes = model_kwargs["input_nn_hidden_sizes"]
        self.lstm_hidden_size = model_kwargs["lstm_hidden_size"]
        self.lstm_num_layers = model_kwargs["lstm_num_layers"]
        self.layer_norm = model_kwargs["layer_norm"]
        self.output_nn_hidden_sizes = model_kwargs["output_nn_hidden_sizes"]
        self.learn_initial_hidden_states = model_kwargs["learn_initial_hidden_states"]
        self.division_factor = model_kwargs['division_factor']
        self.model_kwargs = model_kwargs
   
        # Prepare the data
        self.dataset = prepare_data(data=data, interval=interval, model_kwargs=model_kwargs, 
                                    Y_columns=Y_columns, X_columns=X_columns, verbose=self.verbose)

        self.model = None
        self.optimizer = None
        self.loss = None
        self.train_losses = []
        self.validation_losses = []
        self._validation_losses = []
        self.test_losses = []
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        self.times = []
        self.heating_sequences, self.cooling_sequences = None, None
        self.train_sequences = None
        self.validation_sequences = None
        self.test_sequences = None

        # Sanity check
        if self.neigh_column is None:
            print('\nSanity check of the columns:\n', [(w, [self.dataset.X_columns[i] for i in x]) 
                    for w, x in zip(['Case', 'Room temp', 'Room power', 'Out temp'],
                                    [[self.case_column], self.temperature_column, self.power_column,
                                     [self.out_column]])])
        else:
            print('\nSanity check of the columns:\n', [(w, [self.dataset.X_columns[i] for i in x]) 
                    for w, x in zip(['Case', 'Room temp', 'Room power', 'Out temp', 'Neigh temp'],
                                    [[self.case_column], self.temperature_column, self.power_column,
                                     [self.out_column], self.neigh_column])])

        print("Inputs used in D:\n", np.array(self.dataset.X_columns)[inputs_D])

        # To use the GPU when available
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            print("\nGPU acceleration on!")
        else:
            self.device = "cpu"
            self.save_path = model_kwargs["save_path"]

        # Compute the scaled zero power points and the division factors to use in ResNet-like
        # modules
        self.zero_power = self.compute_zero_power()
        self.normalization_variables = self.get_normalization_variables()
        self.parameter_scalings = self.create_scalings()

        # Prepare the torch module
        if self.module == "PCNN":
            self.model = PCNN(
                device=self.device,
                inputs_D=self.inputs_D,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                out_column=self.out_column,
                neigh_column=self.neigh_column,
                zero_power=self.zero_power,
                division_factor=self.division_factor,
                normalization_variables=self.normalization_variables,
                parameter_scalings=self.parameter_scalings,
            )
        
        elif self.module == "S_PCNN":
            self.model = S_PCNN(
                device=self.device,
                inputs_D=self.inputs_D,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                out_column=self.out_column,
                zero_power=self.zero_power,
                division_factor=self.division_factor,
                normalization_variables=self.normalization_variables,
                parameter_scalings=self.parameter_scalings,
                topology=self.topology,
            )

        elif self.module == "M_PCNN":
            self.model = M_PCNN(
                device=self.device,
                inputs_D=self.inputs_D,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                case_column=self.case_column,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                out_column=self.out_column,
                zero_power=self.zero_power,
                division_factor=self.division_factor,
                normalization_variables=self.normalization_variables,
                parameter_scalings=self.parameter_scalings,
                topology=self.topology,
            )

        elif self.module == "LSTM":
            self.model = LSTM(
                device=self.device,
                rooms=self.rooms,
                inputs_D=self.inputs_D,
                learn_initial_hidden_states=self.learn_initial_hidden_states,
                feed_input_through_nn=self.feed_input_through_nn,
                input_nn_hidden_sizes=self.input_nn_hidden_sizes,
                lstm_hidden_size=self.lstm_hidden_size,
                lstm_num_layers=self.lstm_num_layers,
                layer_norm=self.layer_norm,
                output_nn_hidden_sizes=self.output_nn_hidden_sizes,
                temperature_column=self.temperature_column,
                power_column=self.power_column,
                division_factor=self.division_factor
            )

        self.optimizer = optim.Adam(self.model.parameters(), lr=model_kwargs["learning_rate"])
        self.loss = F.mse_loss

        # Load the model if it exists
        if load:
            self.load(load_last=load_last)

        # if the model doesn't exist, the sequences were not loaded
        if self.train_sequences is None:
            self.heating_sequences, self.cooling_sequences = self.get_sequences()
            self.train_test_validation_separation(validation_percentage=self.validation_percentage,
                                                  test_percentage=self.test_percentage)

        self.model = self.model.to(self.device)

    @property
    def X(self):
        return self.dataset.X

    @property
    def Y(self):
        return self.dataset.Y

    @property
    def columns(self):
        return self.dataset.data.columns

    @property
    def differences_Y(self):
        return self.dataset.differences_Y

    def _fix_seeds(self, seed: int = None):
        """
        Function fixing the seeds for reproducibility.

        Args:
            seed:   Seed to fix everything
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _create_sequences(self, X: pd.DataFrame = None, Y: pd.DataFrame = None, inplace: bool = False):
        """
        Function to create tuple designing the beginning and end of sequences of data we can predict.
        This is needed because PyTorch models don't work with NaN values, so we need to only path
        sequences of data that don't contain any.

        Args:
            X:          input data
            Y:          output data, i.e. labels
            inplace:    Flag whether to do it in place or not

        Returns:
            The created sequences if not inplace.
        """

        # Take the data of the current model if nothing is given
        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y

        # Get the places of NaN values not supported by PyTorch models
        nans = list(set(np.where(np.isnan(X))[0]) | set(np.where(np.isnan(Y))[0]))

        # List of indices that present no nans
        indices = np.arange(len(X))
        not_nans_indices = np.delete(indices, nans)
        last = len(indices) - 1

        sequences = []

        if len(not_nans_indices) > 0:
            # Get the "jumps", i.e. where the the nan values appear
            jumps = np.concatenate([[True], np.diff(not_nans_indices) != 1, [True]])

            # Get the beginnings of all the sequences, correcting extreme values and adding 0 if needed
            beginnings = list(not_nans_indices[np.where(jumps[:-1])[0]])
            if 0 in beginnings:
                beginnings = beginnings[1:]
            if last in beginnings:
                beginnings = beginnings[:-1]
            if (0 in not_nans_indices) and (1 in not_nans_indices):
                beginnings = [0] + beginnings

            # Get the ends of all the sequences, correcting extreme values and adding the last value if needed
            ends = list(not_nans_indices[np.where(jumps[1:])[0]])
            if 0 in ends:
                ends = ends[1:]
            if last in ends:
                ends = ends[:-1]
            if (last in not_nans_indices) and (last - 1 in not_nans_indices):
                ends = ends + [last]

            # We should have the same number of series beginning and ending
            assert len(beginnings) == len(ends), "Something went wrong"

            # Bulk of the work: create starts and ends of sequences tuples
            for beginning, end in zip(beginnings, ends):
                # Add sequences from the start to the end, jumping with the wanted overlapping distance and ensuring
                # the required warm start length and minimum sequence length are respected
                sequences += [(beginning + self.overlapping_distance * x,
                               min(beginning + self.warm_start_length + self.maximum_sequence_length
                                   + self.overlapping_distance * x, end))
                    for x in range(math.ceil((end - beginning - self.warm_start_length
                                              - self.minimum_sequence_length) / self.overlapping_distance))]

        if inplace:
            self.sequences = sequences
        else:
            return sequences

    def get_sequences(self, X: pd.DataFrame = None, Y: pd.DataFrame = None) -> list:
        """
        Function to get tuple designing the beginning and end of sequences of data we can predict.
        This is needed because PyTorch models don't work with NaN values, so we need to only path
        sequences of data that don't contain any.

        If no sequences exist, it creates them.

        Args:
            X:          input data
            Y:          output data, i.e. labels

        Returns:
            All the sequences we can predict
        """

        # Create the corresponding name
        name = os.path.join(self.save_name, "sequences.pt")

        if X is None:
            X = self.X
        if Y is None:
            Y = self.Y

        if self.verbose > 0:
            print("\nTrying to load the predictable sequences, where the data has no missing values...")

        try:
            # Check the existence of the model
            assert os.path.exists(name), f"The file {name} doesn't exist."
            # Load the checkpoint
            checkpoint = torch.load(name)
            # Put it into the model
            heating_sequences = checkpoint["heating_sequences"]
            cooling_sequences = checkpoint["cooling_sequences"]

            if self.verbose > 0:
                print("Found!")

        except AssertionError:
            if self.verbose > 0:
                print("Nothing found, building the sequences...")

            # Create the sequences
            if self.heating:
                X_ = X.copy()
                X_[np.where(X_[:, self.case_column] < 0.5)[0]] = np.nan
                heating_sequences = self._create_sequences(X=X_, Y=Y)
            else:
                heating_sequences = []

            if self.cooling:
                X_ = X.copy()
                X_[np.where(X_[:, self.case_column] > 0.5)[0]] = np.nan
                cooling_sequences = self._create_sequences(X=X_, Y=Y)
            else:
                cooling_sequences = []

            # Save the built list to be able to load it later and avoid the computation
            torch.save({"heating_sequences": heating_sequences, "cooling_sequences": cooling_sequences}, name)

        if self.verbose > 0:
            print(f"Number of sequences for the model {self.name}: {len(heating_sequences)} heating sequences and " f"{len(cooling_sequences)} cooling sequences.")

        # Return the sequences
        return heating_sequences, cooling_sequences

    def train_test_validation_separation(self, validation_percentage: float = 0.2, test_percentage: float = 0.0) -> None:
        """
        Function to separate the data into training and testing parts. The trick here is that
        the data is not actually split - this function actually defines the sequences of
        data points that are in the training/testing part.

        Args:
            validation_percentage:  Percentage of the data to keep out of the training process
                                    for validation
            test_percentage:        Percentage of data to keep out of the training process for
                                    the testing

        Returns:
            Nothing, in place definition of all the indices
        """

        # Sanity checks: the given inputs are given as percentage between 0 and 1
        if 1 <= validation_percentage <= 100:
            validation_percentage /= 100
            print("The train-test-validation separation rescaled the validation_percentage between 0 and 1")
        if 1 <= test_percentage <= 100:
            test_percentage /= 100
            print("The train-test-validation separation rescaled the test_percentage between 0 and 1")

        # Prepare the lists
        self.train_sequences = []
        self.validation_sequences = []
        self.test_sequences = []

        if self.verbose > 0:
            print("Creating training, validation and testing data")

        for sequences in [self.heating_sequences, self.cooling_sequences]:
            if len(sequences) > 0:
                # Given the total number of sequences, define aproximate separations between training
                # validation and testing sets
                train_validation_sep = int((1 - test_percentage - validation_percentage) * len(sequences))
                validation_test_sep = int((1 - test_percentage) * len(sequences))

                # Little trick to ensure training, validation and test sequences are completely distinct
                while True:
                    if (sequences[train_validation_sep - 1][1] < sequences[train_validation_sep][0]) | (train_validation_sep == 1):
                        break
                    train_validation_sep -= 1
                if test_percentage > 0.:
                    while True:
                        if (sequences[validation_test_sep - 1][1] < sequences[validation_test_sep][0]) | (validation_test_sep == 1):
                            break
                        validation_test_sep -= 1

                # Prepare the lists
                self.train_sequences += sequences[:train_validation_sep]
                self.validation_sequences += sequences[train_validation_sep:validation_test_sep]
                self.test_sequences += sequences[validation_test_sep:]

    def compute_zero_power(self):
        """
        Small helper function to compute the scaled value of zero power
        """

        # Scale the zero
        if self.dataset.is_normalized:
            min_ = self.dataset.min_[self.power_column]
            max_ = self.dataset.max_[self.power_column]
            zero = 0.8 * (0.0 - min_) / (max_ - min_) + 0.1

        elif self.dataset.is_standardized:
            mean = self.dataset.mean[self.power_column]
            std = self.dataset.std[self.power_column]
            zero = (0.0 - mean) / std

        else:
            zero = np.array([0.0] * len(self.rooms))

        return np.array(zero)

    def create_scalings(self):
        """
        Function to initialize good parameters for a, b, c and d, the key parameters of the structure.
        Intuition:
          - The room loses 1.5 degrees in 6h when the outside temperature is 25 degrees lower than
              the inside one (and some for losses to the neighboring room)
          - The room gains 2 degrees in 4h of heating

        Returns:
            The scaling parameters according to the data
        """

        parameter_scalings = {}

        if self.unit == 'DFAB':
            # DFAB power is in kW
            parameter_scalings['b'] = [1 / (2 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column]
                                           * 0.8 / 25 / 6 / 60 * self.dataset.interval).mean()]
            parameter_scalings['c'] = [1 / (2.5 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column]
                                           * 0.8 / 25 / 6 / 60 * self.dataset.interval).mean() / 10]

            parameter_scalings['a'] = [
                1 / (1 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column[i]] * 0.8  # Gain 2 degrees
                     / (self.dataset.max_['Thermal total power'] / (self.dataset.max_ - self.dataset.min_)[
                            'Thermal total power'] * 0.8)  # With average power of 1500 W
                     / (10 * 60 / self.dataset.interval)) for i in range(len(self.rooms))]  # In 4h
            parameter_scalings['d'] = [
                1 / (1 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column[i]] * 0.8  # Gain 2 degrees
                     / (- self.dataset.min_['Thermal total power'] / (self.dataset.max_ - self.dataset.min_)[
                            'Thermal total power'] * 0.8)  # With average power of 1000 W
                     / (10 * 60 / self.dataset.interval)) for i in range(len(self.rooms))]  # In 4h

        else:
            parameter_scalings['b'] = [1 / (1.5 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column]
                                           * 0.8 / 25 / 6 / 60 * self.dataset.interval).mean()]
            parameter_scalings['c'] = [1 / (1.5 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column]
                                           * 0.8 / 25 / 6 / 60 * self.dataset.interval).mean() / 10]

            # needed condition to make sure to deal with data in Watts and kWs
            if (self.dataset.max_ - self.dataset.min_)[self.power_column[0]] > 100:
                parameter_scalings['a'] = [
                    1 / (2 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column[i]] * 0.8  # Gain 2 degrees
                         / (1000 / (self.dataset.max_ - self.dataset.min_)[
                                self.power_column[i]] * 0.8)  # With average power of 1.5 kW
                         / (4 * 60 / self.dataset.interval)) for i in range(len(self.rooms))] # in 4h
                parameter_scalings['d'] = [
                    1 / (2 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column[i]] * 0.8  # Gain 2 degrees
                         / (1000 / (self.dataset.max_ - self.dataset.min_)[
                                self.power_column[i]] * 0.8)  # With average power of 1.5 kW
                         / (4 * 60 / self.dataset.interval)) for i in range(len(self.rooms))] # in 4h
            else:
                parameter_scalings['a'] = [
                    1 / (2 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column[i]] * 0.8  # Gain 2 degrees
                         / (self.dataset.max_[self.power_column[i]] * 3/2 / (self.dataset.max_ - self.dataset.min_)[
                                self.power_column[i]] * 0.8)  # With average power of 1.5 kW
                         / (4 * 60 / self.dataset.interval)) for i in range(len(self.rooms))]  # in 4h
                parameter_scalings['d'] = [
                    1 / (2 / (self.dataset.max_ - self.dataset.min_)[self.temperature_column[i]] * 0.8  # Gain 2 degrees
                         / (self.dataset.max_[self.power_column[i]] * 3/2 / (self.dataset.max_ - self.dataset.min_)[
                                self.power_column[i]] * 0.8)  # With average power of 1.5 kW
                         / (4 * 60 / self.dataset.interval)) for i in range(len(self.rooms))]  # in 4h

        return parameter_scalings

    def get_normalization_variables(self):
        """
        Function to get the minimum and the amplitude of some variables in the data. In particular, we need
        that for the room temperature, the outside temperature and the neighboring room temperature.
        This is used by the physics-inspired network to unnormalize the predictions.
        """
        normalization_variables = {}
        normalization_variables['Room'] = [torch.Tensor(self.dataset.min_[self.temperature_column].values).to(self.device),
                                           torch.Tensor((self.dataset.max_ - self.dataset.min_)[self.temperature_column].values).to(self.device)]
        if self.neigh_column is not None:
            normalization_variables['Neigh'] = [torch.Tensor(self.dataset.min_[self.neigh_column].values).to(self.device),
                                           torch.Tensor((self.dataset.max_ - self.dataset.min_)[self.neigh_column].values).to(self.device)]
        normalization_variables['Out'] = [torch.Tensor([self.dataset.min_[self.out_column]]).to(self.device),
                                          torch.Tensor([(self.dataset.max_ - self.dataset.min_)[self.out_column]]).to(self.device)]
        return normalization_variables

    def batch_iterator(self, iterator_type: str = "train", batch_size: int = None, shuffle: bool = True) -> None:
        """
        Function to create batches of the data with the wanted size, either for training,
        validation, or testing

        Args:
            iterator_type:  To know if this should handle training, validation or testing data
            batch_size:     Size of the batches
            shuffle:        Flag to shuffle the data before the batches creation

        Returns:
            Nothing, yields the batches to be then used in an iterator
        """

        # Firstly control that the training sequences exist - create them otherwise
        if self.train_sequences is None:
            self.train_test_validation_separation()
            print("The Data was not separated in train, validation and test --> the default 70%-20%-10% was used")

        # If no batch size is given, define it as the default one
        if batch_size is None:
            batch_size = self.batch_size

        # Copy the indices of the correct type (without first letter in case of caps)
        if "rain" in iterator_type:
            sequences = self.train_sequences
        elif "alidation" in iterator_type:
            sequences = self.validation_sequences
        elif "est" in iterator_type:
            sequences = self.test_sequences
        else:
            raise ValueError(f"Unknown type of batch creation {iterator_type}")

        # Shuffle them if wanted
        if shuffle:
            np.random.shuffle(sequences)

        # Define the right number of batches according to the wanted batch_size - taking care of the
        # special case where the indicies ae exactly divisible by the batch size, which can induce
        # an additional empty batch breaking the simulation down the line
        n_batches = int(np.ceil(len(sequences) / batch_size))

        # Iterate to yield the right batches with the wanted size
        for batch in range(n_batches):
            yield sequences[batch * batch_size: (batch + 1) * batch_size]

    def build_input_output_from_sequences(self, sequences: list):
        """
        Input and output generator from given sequences of indices corresponding to a batch.

        Args:
            sequences: sequences of the batch to prepare

        Returns:
            batch_x:    Batch input of the model
            batch_y:    Targets of the model, the temperature and the power
        """

        # Ensure the given sequences are a list of list, not only one list
        if type(sequences) == tuple:
            sequences = [sequences]

        # Iterate over the sequences to build the input in the right form
        input_tensor_list = [torch.Tensor(self.X[sequence[0]: sequence[1], :].copy()) for sequence in sequences]

        # Prepare the output for the temperature and power consumption
        output_tensor_list = [torch.Tensor(self.Y[sequence[0]: sequence[1], :].copy()) for sequence in
                                    sequences]


        # Build the final results by taking care of the batch_size=1 case
        if len(sequences) > 1:
            batch_x = pad_sequence(input_tensor_list, batch_first=True, padding_value=0)
            batch_y = pad_sequence(output_tensor_list, batch_first=True, padding_value=0)
        else:
            batch_x = input_tensor_list[0].view(1, input_tensor_list[0].shape[0], -1)
            batch_y = output_tensor_list[0].view(1, output_tensor_list[0].shape[0], -1)

        # Return everything
        return batch_x.to(self.device), batch_y.to(self.device)

    def predict(self, sequences: Union[list, int] = None, data: torch.Tensor = None, mpc_mode: bool = False):
        """
        Function to predict batches of "sequences", i.e. it creates batches of input and output of the
        given sequences you want to predict and forwards them through the network

        Args:
            sequences:  Sequences of the data to predict
            data:       Alternatively, data to predict, a tuple of tensors with the X and Y (if there is no
                          Y just put a vector of zeros with the right output size)
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predictions and the true output
        """

        if sequences is not None:
            # Ensure the given sequences are a list of list, not only one list
            if type(sequences) == tuple:
                sequences = [sequences]

            # Build the input and output
            batch_x, batch_y = self.build_input_output_from_sequences(sequences=sequences)

        elif data is not None:
            batch_x = data[0].reshape(data[0].shape[0], data[0].shape[1], -1)
            batch_y = data[1].reshape(data[0].shape[0], data[0].shape[1], len(self.rooms))

        else:
            raise ValueError("Either sequences or data must be provided to the `predict` function")

        predictions = torch.zeros_like(batch_y).to(self.device)
        states = None

        # Iterate through the sequences of data to predict each step, replacing the true power and temperature
        # values with the predicted ones each time
        for i in range(batch_x.shape[1]):
            # Predict the next output and store it
            pred, states = self.model(batch_x[:, i, :], states, warm_start=i<self.warm_start_length)
            predictions[:, i, :] = pred

        return predictions, batch_y

    def scale_back_predictions(self, sequences: Union[list, int] = None, data: torch.Tensor = None):
        """
        Function preparing the data for analyses: it predicts the wanted sequences and returns the scaled
        predictions and true_data

        Args:
            sequences:  Sequences to predict
            data:       Alternatively, data to predict, a tuple of tensors with the X and Y (if there is no
                          Y just put a vector of zeros with the right output size)

        Returns:
            The predictions and the true data
        """

        # Compute the predictions and get the true data out of the GPU
        predictions, true_data = self.predict(sequences=sequences, data=data)
        predictions = predictions.cpu().detach().numpy()
        true_data = true_data.cpu().detach().numpy()

        # Reshape the data for consistency with the next part of the code if only one sequence is given
        if sequences is not None:
            # Reshape when only 1 sequence given
            if type(sequences) == tuple:
                sequences = [sequences]

        elif data is not None:
            sequences = [0]

        else:
            raise ValueError("Either sequences or data must be provided to the `scale_back_predictions` function")

        if len(predictions.shape) == 2:
            predictions = predictions.reshape(1, predictions.shape[0], -1)
            true_data = true_data.reshape(1, true_data.shape[0], -1)

        # Scale the data back
        cols = self.dataset.Y_columns[:-1] if self.predict_power else self.dataset.Y_columns[:-2]
        truth = true_data.reshape(true_data.shape[0], true_data.shape[1], -1)
        true = np.zeros_like(predictions)

        if self.dataset.is_normalized:
            for i, sequence in enumerate(sequences):
                predictions[i, :, :] = inverse_normalize(data=predictions[i, :, :],
                                                         min_=self.dataset.min_[self.dataset.Y_columns],
                                                         max_=self.dataset.max_[self.dataset.Y_columns])
                true[i, :, :] = inverse_normalize(data=truth[i, :, :],
                                                       min_=self.dataset.min_[self.dataset.Y_columns],
                                                       max_=self.dataset.max_[self.dataset.Y_columns])
        elif self.dataset.is_standardized:
            for i, sequence in enumerate(sequences):
                predictions[i, :, :] = inverse_standardize(data=predictions[i, :, :],
                                                           mean=self.dataset.mean[self.dataset.Y_columns],
                                                           std=self.dataset.std[self.dataset.Y_columns])
                true[i, :, :] = inverse_standardize(data=truth[i, :, :],
                                                         mean=self.dataset.mean[self.dataset.Y_columns],
                                                         std=self.dataset.std[self.dataset.Y_columns])

        return predictions, true

    def fit(self, n_epochs: int = None, print_each: int = 5) -> None:
        """
        General function fitting a model for several epochs, training and evaluating it on the data.

        Args:
            n_epochs:         Number of epochs to fit the model, if None this takes the default number
                                defined by the parameters
            n_batches_print:  Control how many batches to print per epoch

        Returns:
            Nothing
        """

        self.times.append(time.time())

        if self.verbose > 0:
            print("\nTraining starts!")

        # If no special number of epochs is given, take the default one
        if n_epochs is None:
            n_epochs = self.n_epochs

        # Define the best loss, taking the best existing one or a very high loss
        best_loss = np.min(self.validation_losses) if len(self.validation_losses) > 0 else np.inf

        # Assess the number of epochs the model was already trained on to get nice prints
        trained_epochs = len(self.train_losses)

        for epoch in range(trained_epochs, trained_epochs + n_epochs):

            if self.verbose > 0:
                print(f"\nTraining epoch {epoch + 1}...")

            # Start the training, define a list to retain the training losses along the way
            self.model.train()
            train_losses = []

            # Adjust the learning rate if wanted
            if self.decrease_learning_rate:
                self.adjust_learning_rate(epoch=epoch)

            # Create training batches and run through them, using the batch_iterator function, which has to be defined
            # independently for each subclass, as different types of data are handled differently
            for num_batch, batch_sequences in enumerate(self.batch_iterator(iterator_type="train")):

                # Compute the loss of the batch and store it
                loss = self.compute_loss(batch_sequences)

                # Compute the gradients and take one step using the optimizer
                loss.backward()
                #for p in self.model.named_parameters():
                #    if (p[1].grad is not None):
                #        print(p[0], ":", p[1].grad.norm())
                self.optimizer.step()
                self.optimizer.zero_grad()
                train_losses.append(float(loss))

                # Regularly print the current state of things
                if (self.verbose > 1) & (num_batch % print_each == print_each - 1):
                    print(f"Loss batch {num_batch + 1}: {float(loss): .5f}")

            # Compute the average loss of the training epoch and print it
            train_loss = sum(train_losses) / len(train_losses)
            print(f"Average training loss after {epoch + 1} epochs: {train_loss}")
            self.train_losses.append(train_loss)

            # Start the validation, again defining a list to recall the losses
            if self.verbose > 0:
                print(f"Validation epoch {epoch + 1}...")
            validation_losses = []
            _validation_losses = []

            # Create validation batches and run through them. Note that we use larger batches
            # to accelerate it a bit, and there is no need to shuffle the indices
            for num_batch, batch_indices in enumerate(self.batch_iterator(iterator_type="validation", batch_size=2 * self.batch_size, shuffle=False)):

                # Compute the loss, in the torch.no_grad setting: we don't need the model to
                # compute and use gradients here, we are not training
                if 'PiNN' not in self.name:
                    self.model.eval()
                    with torch.no_grad():
                        loss = self.compute_loss(batch_indices)
                        validation_losses.append(float(loss))
                        # Regularly print the current state of things
                        if (self.verbose > 1) & (num_batch % (print_each//2) == (print_each//2) - 1):
                            print(f"Loss batch {num_batch + 1}: {float(loss): .5f}")

                else:
                    self.model.train()
                    loss = self.compute_loss(batch_indices)
                    validation_losses.append(float(loss))
                    # Regularly print the current state of things
                    if (self.verbose > 1) & (num_batch % (print_each//2) == (print_each//2) - 1):
                        print(f"Loss batch {num_batch + 1}: {float(loss): .5f}")
                    self.model.eval()
                    with torch.no_grad():
                        loss = self._compute_loss(batch_indices)
                        _validation_losses.append(float(loss))
                        # Regularly print the current state of things
                        if (self.verbose > 1) & (num_batch % (print_each//2) == (print_each//2) - 1):
                            print(f"Loss batch {num_batch + 1}: {float(loss): .5f}")

            # Compute the average validation loss of the epoch and print it
            validation_loss = sum(validation_losses) / len(validation_losses)
            self.validation_losses.append(validation_loss)
            print(f"Average validation loss after {epoch + 1} epochs: {validation_loss}")

            if 'PiNN' in self.name:
                _validation_loss = sum(_validation_losses) / len(_validation_losses)
                self._validation_losses.append(_validation_loss)
                if self.verbose > 0:
                    print(f"Average accuracy validation loss after {epoch + 1} epochs: {_validation_loss}")

            # Timing information
            self.times.append(time.time())
            if self.verbose > 0:
                print(f"Time elapsed for the epoch: {format_elapsed_time(self.times[-2], self.times[-1])}"
                      f" - for a total training time of {format_elapsed_time(self.times[0], self.times[-1])}")

            # Save parameters
            if 'PCNN' in self.module:
                p = self.model.E_parameters
                self.a.append(p[0])
                self.b.append(p[1])
                self.c.append(p[2])
                self.d.append(p[3])

            # Save last and possibly best model
            self.save(name_to_add="last", verbose=0)

            if validation_loss < best_loss:
                self.save(name_to_add="best", verbose=1)
                best_loss = validation_loss

        if self.verbose > 0:
            best_epoch = np.argmin([x for x in self.validation_losses])
            print(f"\nThe best model was obtained at epoch {best_epoch + 1} after training for " f"{trained_epochs + n_epochs} epochs")

    def adjust_learning_rate(self, epoch: int) -> None:
        """
        Custom function to decrease the learning rate along the training

        Args:
            epoch:  Epoch of the training

        Returns:
            Nothing, modifies the optimizer in place
        """

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = param_group["lr"] * 0.997

    def compute_loss(self, sequences: list):
        """
        Custom function to compute the loss of a batch of sequences.

        Args:
            sequences: The sequences in the batch

        Returns:
            The loss
        """

        predictions, batch_y = self.predict(sequences=sequences)
        return self.loss(predictions, batch_y)

    def save(self, name_to_add: str = None, verbose: int = 0):
        """
        Function to save a PyTorch model: Save the state of all parameters, as well as the one of the
        optimizer. We also recall the losses for ease of analysis.

        Args:
            name_to_add:    Something to save a unique model

        Returns
            Nothing, everything is done in place and stored in the parameters
        """

        if verbose > 0:
            print(f"\nSaving the new {name_to_add} model!")

        if name_to_add is not None:
            save_name = os.path.join(self.save_name, f"{name_to_add}_model.pt")
        else:
            save_name = os.path.join(self.save_name, "model.pt")

        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "train_sequences": self.train_sequences,
                "validation_sequences": self.validation_sequences,
                "test_sequences": self.test_sequences,
                "train_losses": self.train_losses,
                "validation_losses": self.validation_losses,
                "_validation_losses": self._validation_losses,
                "test_losses": self.test_losses,
                "times": self.times,
                "a": self.a,
                "b": self.b,
                "c": self.c,
                "d": self.d,
                "warm_start_length": self.warm_start_length,
                "maximum_sequence_length": self.maximum_sequence_length,
                "feed_input_through_nn": self.feed_input_through_nn,
                "input_nn_hidden_sizes": self.input_nn_hidden_sizes,
                "lstm_hidden_size": self.lstm_hidden_size,
                "lstm_num_layers": self.lstm_num_layers,
                "output_nn_hidden_sizes": self.output_nn_hidden_sizes,
            },
            save_name,
        )

    def load(self, load_last: bool = False):
        """
        Function trying to load an existing model, by default the best one if it exists. But for training purposes,
        it is possible to load the last state of the model instead.

        Args:
            load_last:  Flag to set to True if the last model checkpoint is wanted, instead of the best one

        Returns:
             Nothing, everything is done in place and stored in the parameters.
        """

        if load_last:
            save_name = os.path.join(self.save_name, "last_model.pt")
        else:
            save_name = os.path.join(self.save_name, "best_model.pt")

        if self.verbose > 0:
            print("\nTrying to load a trained model...")
        try:
            # Build the full path to the model and check its existence

            assert os.path.exists(save_name), f"The file {save_name} doesn't exist."

            # Load the checkpoint
            checkpoint = torch.load(save_name, map_location=lambda storage, loc: storage)

            # Put it into the model
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if torch.cuda.is_available():
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.cuda()
            self.train_sequences = checkpoint["train_sequences"]
            self.validation_sequences = checkpoint["validation_sequences"]
            self.test_sequences = checkpoint["test_sequences"]
            self.train_losses = checkpoint["train_losses"]
            self.validation_losses = checkpoint["validation_losses"]
            self.test_losses = checkpoint["test_losses"]
            self.times = checkpoint["times"]
            self.a = checkpoint["a"]
            self.b = checkpoint["b"]
            self.c = checkpoint["c"]
            self.d = checkpoint["d"]
            self.warm_start_length = checkpoint["warm_start_length"]
            self.maximum_sequence_length = checkpoint["maximum_sequence_length"]
            self.feed_input_through_nn = checkpoint["feed_input_through_nn"]
            self.input_nn_hidden_sizes = checkpoint["input_nn_hidden_sizes"]
            self.lstm_hidden_size = checkpoint["lstm_hidden_size"]
            self.lstm_num_layers = checkpoint["lstm_num_layers"]
            self.output_nn_hidden_sizes = checkpoint["output_nn_hidden_sizes"]

            # Print the current status of the found model
            if self.verbose > 0:
                print(f"Found!\nThe model has been fitted for {len(self.train_losses)} epochs already, "
                      f"with loss {np.min(self.validation_losses): .5f}.")
                print(f"It contains {len(self.train_sequences)} training sequences and "
                      f"{len(self.validation_sequences)} validation sequences.\n")

        # Otherwise, keep an empty model, return nothing
        except AssertionError:
            print("\nNo existing model was found!\n")
