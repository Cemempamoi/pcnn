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
from loguru import logger
from timeit import default_timer

import torch
from torch.nn.utils.rnn import pad_sequence
from torch import optim

from pcnn.module import PCNN, S_PCNN, M_PCNN, LSTM
from pcnn.data import prepare_data
from pcnn.util import model_save_name_factory, format_elapsed_time, inverse_normalize, check_GPU_availability, \
    elapsed_timer, ensure_list, check_initialization_physical_parameters


class Model:
    """
    Class of models using PyTorch
    """

    def __init__(self, data: pd.DataFrame, module: str, model_params: dict, data_params: dict, 
                 load: bool = True, load_last: bool = False):
        """
        Initialize a model.

        Args:
            data:           DataFrame to use
            module:         Module to use
            model_kwargs:   Parameters of the models, see 'parameters.py'
            data_kwargs:    Parameters of the data
            load:           Whether to load a model or not
            load_last:      Whether to load the last model or not
        """

        data_kwargs = data_params.copy()
        model_kwargs = model_params.copy()

        assert module in ['PCNN', 'S_PCNN', 'M_PCNN', 'LSTM'],\
            f"The provided model type {module} does not exist, please chose among `'PCNN', 'S_PCNN', 'M_PCNN', 'LSTM'`."

        # Define the main attributes
        self.module = module
        self.number_rooms = len(ensure_list(data_kwargs['temperature_column']))
        model_kwargs['number_rooms'] = self.number_rooms
        self.verbose = model_kwargs["verbose"]

        # Prepare the data
        self.dataset = prepare_data(data=data, data_kwargs=data_kwargs, verbose=self.verbose)
        # Recover the updated data kwargs
        data_kwargs = self.dataset.data_kwargs

        # Compute the temperature range 
        model_kwargs['temperature_min'], model_kwargs['temperature_range'] = self.dataset.get_temperarature_min_and_range()

        # Special LSTM case
        if module != 'LSTM':
            for key in model_kwargs['initial_values_physical_parameters']:
                model_kwargs['initial_values_physical_parameters'][key] = ensure_list(model_kwargs['initial_values_physical_parameters'][key])
            # Snaity check
            check_initialization_physical_parameters(initial_values_physical_parameters=model_kwargs['initial_values_physical_parameters'],
                                                    data_params=data_kwargs)
        else:
            model_kwargs['number_inputs'] = len(self.dataset.X_columns)

        # Create the name associated to the model
        self.name = model_kwargs["name"]
        self.save_name = model_save_name_factory(module=module, model_kwargs=model_kwargs)

        # Fix the seeds for reproduction
        self._fix_seeds(seed=model_kwargs["seed"])

        # To use the GPU when available
        if model_kwargs['device'] is None:
            self.device = check_GPU_availability()
            model_kwargs['device'] = self.device
        else:
            self.device = model_kwargs['device']
        
        # Push everything to the right device
        self.X = torch.from_numpy(self.dataset.X).to(torch.float32).to(self.device) 
        self.Y = torch.from_numpy(self.dataset.Y).to(torch.float32).to(self.device)

        # Store needed parameters
        self.save_model = model_kwargs["save"]
        self.batch_size = model_kwargs["batch_size"]
        self.shuffle = model_kwargs["shuffle"]
        self.n_epochs = model_kwargs["n_epochs"]
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

        self.case_column = data_kwargs['case_column']
        data_kwargs['outside_walls'] =  ensure_list(data_kwargs['outside_walls'])
        data_kwargs['neighboring_rooms'] = ensure_list(data_kwargs['neighboring_rooms'])

        # Prepare the torch module
        # Group parameters for simplicity
        kwargs = model_kwargs | data_kwargs
        if self.module == "PCNN":
            self.model = PCNN(kwargs=kwargs)
        elif self.module == "S_PCNN":
            self.model = S_PCNN(kwargs=kwargs)
        elif self.module == "M_PCNN":
            self.model = M_PCNN(kwargs=kwargs)
        elif self.module == "LSTM":
            self.model = LSTM(kwargs=kwargs)

        # define the optimizer and the loss
        self.optimizer = optim.Adam(self.model.parameters(), lr=model_kwargs["learning_rate"])
        self.loss = model_kwargs['loss']

        # Save the updated parameters
        self.model_kwargs = model_kwargs
        self.data_kwargs = data_kwargs

        # Prepare the lists to store progress
        self.train_losses = []
        self.validation_losses = []
        self.test_losses = []
        self.a = []
        self.b = []
        self.c = []
        self.d = []
        self.times = [0.]

        # Load the model if it exists
        self.train_sequences = None
        if load:
            self.load(load_last=load_last)

        # if the model doesn't exist, the sequences were not loaded
        if self.train_sequences is None:
            self.heating_sequences, self.cooling_sequences = self.get_sequences()
            self.train_test_validation_separation(validation_percentage=self.validation_percentage,
                                                  test_percentage=self.test_percentage)

        self.model = self.model.to(self.device)

    @property
    def columns(self):
        return self.dataset.data.columns

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

            # Shuffle the sequences to ensure training/validation/test are spread over the entire data
            seqs = np.arange(len(ends))

            # For reproducibility - different models trained on the same data should have the same sequences
            np.random.seed(self.data_kwargs['seed'])
            np.random.shuffle(seqs)
            np.random.seed(self.model_kwargs['seed'])
            beginnings = list(np.array(beginnings)[seqs])
            ends = list(np.array(ends)[seqs])

            # Bulk of the work: create starts and ends of sequences tuples, and store which group they belong to
            # (the group is used when separating train, validation, and test sequences)
            for group, (beginning, end) in enumerate(zip(beginnings, ends)):
                # Add sequences from the start to the end, jumping with the wanted overlapping distance and ensuring
                # the required warm start length and minimum sequence length are respected
                sequences += [(beginning + self.overlapping_distance * x,
                               min(beginning + self.warm_start_length + self.maximum_sequence_length
                                   + self.overlapping_distance * x, end), group)
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
            logger.info("Trying to load the predictable sequences, where the data has no missing values...")

        try:
            # Check the existence of the model
            assert os.path.exists(name), f"The file {name} doesn't exist."
            # Load the checkpoint
            checkpoint = torch.load(name, weights_only=False)
            # Put it into the model
            heating_sequences = checkpoint["heating_sequences"]
            cooling_sequences = checkpoint["cooling_sequences"]

            if self.verbose > 0:
                logger.info("Found!")

        except AssertionError:
            if self.verbose > 0:
                logger.info("Nothing found, building the sequences...")

            # Create the sequences
            if self.heating:
                X_ = X.clone()
                X_[np.where(X_[:, self.case_column[0]] < 0.5)[0]] = np.nan
                heating_sequences = self._create_sequences(X=X_, Y=Y)
            else:
                heating_sequences = []

            if self.cooling:
                X_ = X.clone()
                X_[np.where(X_[:, self.case_column[0]] > 0.5)[0]] = np.nan
                cooling_sequences = self._create_sequences(X=X_, Y=Y)
            else:
                cooling_sequences = []

            # Save the built list to be able to load it later and avoid the computation
            if self.save_model:
                torch.save({"heating_sequences": heating_sequences, "cooling_sequences": cooling_sequences}, name)

        if self.verbose > 0:
            logger.info(f"Number of sequences for the model {self.name}: {len(heating_sequences)} heating sequences and " f"{len(cooling_sequences)} cooling sequences.")

        # Return the sequences
        return heating_sequences, cooling_sequences

    def train_test_validation_separation(self, validation_percentage: float = 0.2, test_percentage: float = 0.1) -> None:
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
            logger.info("The train-test-validation separation rescaled the validation_percentage between 0 and 1")
        if 1 <= test_percentage <= 100:
            test_percentage /= 100
            logger.info("The train-test-validation separation rescaled the test_percentage between 0 and 1")

        # Prepare the lists
        self.train_sequences = []
        self.validation_sequences = []
        self.test_sequences = []

        if self.verbose > 0:
            logger.info("Creating training, validation and testing data...\n")

        for i, sequences in enumerate([self.heating_sequences, self.cooling_sequences]):
            if len(sequences) > 0:
                # Given the total number of sequences, define approximate separations between training
                # validation and testing sets
                train_validation_sep_start = train_validation_sep_end = int((1 - test_percentage - validation_percentage) * len(sequences))
                validation_test_sep_start = validation_test_sep_end = int((1 - test_percentage) * len(sequences))

                # Little trick to ensure training, validation and test sequences are completely distinct
                # works by decreasing/increasing the start/end of each split until total separation is guaranteee
                # This happens either if the end of the last training sequence is before the start of the first validation sequence,
                # or if the group is different(sequences are not from the same part of the data, as defined in '_create_sequences')

                # Train/validation separation
                increase_blocked = False
                decrease_blocked = False
                while True:
                    # If the end of the last training sequence is before the start of the first validation sequence, the sets are distinct
                    if sequences[train_validation_sep_start-1][1] < sequences[train_validation_sep_end][0]:
                        break
                    # Or, if the group is different, the sets are distinct 
                    if sequences[train_validation_sep_start-1][2] != sequences[train_validation_sep_end][2]:
                        # In that case, we can take the entire original group either 
                        # in the validation set (if we were decreasing the start of the split)
                        if not decrease_blocked:
                            train_validation_sep_end = train_validation_sep_start + 1
                        # or in the traning set (if we were increasing the end of the split)
                        else:
                            train_validation_sep_start = train_validation_sep_end - 1
                        break
                    # Otherwise, while they are not distinct, try decreasing the start of the split if not yet blocked
                    if not decrease_blocked:
                        if train_validation_sep_start > 1:
                            train_validation_sep_start -= 1
                        # If reached the beginning, block the decrease
                        else:
                            decrease_blocked = True
                    # If decreasing is blocked, try increasing the end of the split if not yet blocked
                    elif not increase_blocked:
                        if validation_test_sep_end < validation_test_sep_start - 1:
                            validation_test_sep_end += 1 
                        # If reached the end, block the increase
                        else:
                            increase_blocked = True
                    # If both are blocked, we can't separate the sequences, log a warning and break
                    else:
                        logger.warning(f"Could not separate train and validation {'heating' if i==0 else 'cooling'} sequences, not enough data!")
                        logger.warning(f"There is only one train and one validation {'heating' if i==0 else 'cooling'} sequence, and overlapping.")
                        break
                    
                # Validation/test separation (works the same way as above)
                increase_blocked = False
                decrease_blocked = False
                while True:
                    if sequences[validation_test_sep_start-1][1] < sequences[validation_test_sep_end][0]:
                        break
                    if sequences[validation_test_sep_start-1][2] != sequences[validation_test_sep_end][2]:
                        if decrease_blocked:
                            validation_test_sep_end = validation_test_sep_start + 1
                        else:
                            validation_test_sep_start = validation_test_sep_end - 1
                        break
                    if not decrease_blocked:
                        if validation_test_sep_start > train_validation_sep_end + 1:
                            validation_test_sep_start -= 1
                        else:
                            decrease_blocked = True
                    elif not increase_blocked:
                        if validation_test_sep_end < len(sequences) - 1:
                            validation_test_sep_end += 1 
                        else:
                            increase_blocked = True
                    else:
                        logger.warning(f"Could not separate validation and test {'heating' if i==0 else 'cooling'} sequences, not enough data!")
                        logger.warning(f"There is only one validation and one test {'heating' if i==0 else 'cooling'} sequence, and overlapping.")
                        break
                                                
                # Sanity check to ensure sets are not empty
                if train_validation_sep_start == 0:
                   logger.warning(f"No training {'heating' if i==0 else 'cooling'} sequences could not be created without overlapping!")
                if train_validation_sep_end == validation_test_sep_start:
                   logger.warning(f"No validation {'heating' if i==0 else 'cooling'} sequences could not be created without overlapping!")
                if validation_test_sep_end == len(sequences):
                   logger.warning(f"No test {'heating' if i==0 else 'cooling'} sequences could not be created without overlapping!")

                # Prepare the lists according to the defined splits
                self.train_sequences += sequences[:train_validation_sep_start+1]
                self.validation_sequences += sequences[train_validation_sep_end:validation_test_sep_start+1]
                self.test_sequences += sequences[validation_test_sep_end:]

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
        if isinstance(sequences, tuple):
            sequences = [sequences]

        # Iterate over the sequences to build the input in the right form
        input_tensor_list = [self.X[sequence[0]: sequence[1], :].clone() for sequence in sequences]

        # Prepare the output for the temperature and power consumption
        output_tensor_list = [self.Y[sequence[0]: sequence[1], :].clone() for sequence in sequences]


        # Build the final results by taking care of the batch_size=1 case
        if len(sequences) > 1:
            batch_x = pad_sequence(input_tensor_list, batch_first=True, padding_value=0)
            batch_y = pad_sequence(output_tensor_list, batch_first=True, padding_value=0)
        else:
            batch_x = input_tensor_list[0].view(1, input_tensor_list[0].shape[0], -1)
            batch_y = output_tensor_list[0].view(1, output_tensor_list[0].shape[0], -1)

        # Return everything
        return batch_x, batch_y

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

        return_y = True
        if sequences is not None:
            # Ensure the given sequences are a list of list, not only one list
            if isinstance(sequences, tuple):
                sequences = [sequences]

            # Build the input and output
            batch_x, batch_y = self.build_input_output_from_sequences(sequences=sequences)

        elif data is not None:
            if isinstance(data, tuple):
                if len(data[0].shape) == 3:
                    batch_x = data[0].reshape(data[0].shape[0], data[0].shape[1], -1)
                    batch_y = data[1].reshape(data[0].shape[0], data[0].shape[1], self.number_rooms)
                else:
                    batch_x = data[0].reshape(1, data[0].shape[0], -1)
                    batch_y = data[1].reshape(1, data[0].shape[0], self.number_rooms)
            else:
                if len(data.shape) == 3:
                    batch_x = data.reshape(data.shape[0], data.shape[1], -1)
                else:
                    batch_x = data.reshape(1, data.shape[0], -1)
                return_y = False

        else:
            raise ValueError("Either sequences or data must be provided to the `predict` function")

        predictions = torch.zeros((batch_x.shape[0], batch_x.shape[1], self.number_rooms)).to(self.device)
        states = None

        # Iterate through the sequences of data to predict each step, replacing the true power and temperature
        # values with the predicted ones each time
        for i in range(batch_x.shape[1]):
            # Predict the next output and store it
            pred, states = self.model(batch_x[:, i, :], states, warm_start=i<self.warm_start_length)
            predictions[:, i, :] = pred

        if return_y:
            return predictions, batch_y
        else:
            return predictions

    def predict_temperature(self, sequences: Union[list, int] = None, data: torch.Tensor = None):
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
            if isinstance(sequences, tuple):
                sequences = [sequences]

        elif data is not None:
            sequences = [0]

        else:
            raise ValueError("Either sequences or data must be provided to the `scale_back_predictions` function")

        if len(predictions.shape) == 2:
            predictions = predictions.reshape(1, predictions.shape[0], -1)
            true_data = true_data.reshape(1, true_data.shape[0], -1)

        # Scale the data back
        if self.dataset.is_normalized:
            predictions = inverse_normalize(data=predictions,
                                            min_=self.model_kwargs['temperature_min'],
                                            max_=self.model_kwargs['temperature_min'] + self.model_kwargs['temperature_range'])
            true_data = inverse_normalize(data=true_data,
                                          min_=self.model_kwargs['temperature_min'],
                                          max_=self.model_kwargs['temperature_min'] + self.model_kwargs['temperature_range'])
            
        # Remove padded values
        predictions[true_data < self.model_kwargs['temperature_min'] + 1e-6] = np.nan
        true_data[true_data < self.model_kwargs['temperature_min'] + 1e-6] = np.nan

        return predictions, true_data

    def fit(self, n_epochs: int = None, number_sequences: int = None, print_each: int = 1,
            output_best: bool = True) -> None:
        """
        General function fitting a model for several epochs, training and evaluating it on the data.

        Args:
            n_epochs:         Number of epochs to fit the model, if None this takes the default number
                                defined by the parameters
            number_sequences: Number of sequences to use for training, validation and testing
            print_each:       Number of epochs to print the loss
            output_best:      Whether to output the best model

        Returns:
            Nothing
        """

        if number_sequences is not None:
            self.train_sequences = self.train_sequences[:number_sequences]
            self.validation_sequences = self.validation_sequences[:number_sequences]
            self.test_sequences = self.test_sequences[:number_sequences]
        
        if self.verbose > 0:
            logger.info(f"Training starts! {len(self.train_sequences)} train, {len(self.validation_sequences)} validation, and {len(self.test_sequences)} test sequences.\n")

        # If no special number of epochs is given, take the default one
        if n_epochs is None:
            n_epochs = self.n_epochs

        # Define the best loss, taking the best existing one or a very high loss
        best_loss = np.min(self.validation_losses) if len(self.validation_losses) > 0 else np.inf

        # Assess the number of epochs the model was already trained on to get nice prints
        trained_epochs = len(self.train_losses)

        start_time = self.times[-1]
        if self.verbose > 0:
            print('Epoch\tTrain loss\tVal loss\tTest loss\tTime')

        with elapsed_timer() as elapsed:

            for epoch in range(trained_epochs, trained_epochs + n_epochs):

                # Start the training, define a list to retain the training losses along the way
                if epoch > 0:
                    self.model.train()
                else:
                    self.model.eval()
                train_losses = []
                train_sizes = []

                # Adjust the learning rate if wanted
                if self.decrease_learning_rate:
                    self.adjust_learning_rate(epoch=epoch)

                # Create training batches and run through them, using the batch_iterator function, which has to be defined
                # independently for each subclass, as different types of data are handled differently
                for num_batch, batch_sequences in enumerate(self.batch_iterator(iterator_type="train")):
                    if epoch > 0:
                        # Compute the loss of the batch and store it
                        loss = self.compute_loss(batch_sequences)
                        # Compute the gradients and take one step using the optimizer
                        loss.backward()
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        train_losses.append(float(loss))
                        train_sizes.append(len(batch_sequences))
                    else:
                        # Compute the loss, in the torch.no_grad setting: we don't need the model to
                        # compute and use gradients here, we are not training                        
                        with torch.no_grad():
                            loss = self.compute_loss(batch_sequences)
                            train_losses.append(float(loss))
                            train_sizes.append(len(batch_sequences))

                # Compute the average loss of the training epoch and print it
                train_loss = sum([loss*size for loss,size in zip(train_losses, train_sizes)]) / sum(train_sizes)
                self.train_losses.append(train_loss)

                # Start the validation, again defining a list to recall the losses
                self.model.eval()
                validation_losses = []
                validation_sizes = []

                # Create validation batches and run through them. Note that we use larger batches
                # to accelerate it a bit, and there is no need to shuffle the indices
                for num_batch, batch_sequences in enumerate(self.batch_iterator(iterator_type="validation", batch_size=2 * self.batch_size, shuffle=False)):

                    # Compute the loss, in the torch.no_grad setting: we don't need the model to
                    # compute and use gradients here, we are not training                        
                    with torch.no_grad():
                        loss = self.compute_loss(batch_sequences)
                        validation_losses.append(float(loss))
                        validation_sizes.append(len(batch_sequences))

                # Compute the average validation loss of the epoch and print it
                validation_loss = sum([loss*size for loss,size in zip(validation_losses, train_sizes)]) / sum(validation_sizes)
                self.validation_losses.append(validation_loss)                    

                # Start the test, again defining a list to recall the losses
                test_losses = []
                test_sizes = []

                # Create validation batches and run through them. Note that we use larger batches
                # to accelerate it a bit, and there is no need to shuffle the indices
                for num_batch, batch_sequences in enumerate(self.batch_iterator(iterator_type="test", batch_size=2 * self.batch_size, shuffle=False)):

                    # Compute the loss, in the torch.no_grad setting: we don't need the model to
                    # compute and use gradients here, we are not training
                    with torch.no_grad():
                        loss = self.compute_loss(sequences=batch_sequences)
                        test_losses.append(float(loss))
                        test_sizes.append(len(batch_sequences))
                
                # Compute the average test loss of the epoch and print it
                test_loss = sum([loss*size for loss,size in zip(test_losses, test_sizes)]) / sum(test_sizes)
                self.test_losses.append(test_loss)

                # Timing information
                self.times.append(elapsed() + start_time)

                # Prints
                if ((self.verbose > 0) and (epoch % print_each == 0)) or (validation_loss < best_loss):
                    print(f'{epoch}\t{train_loss:.2E}\t{validation_loss:.2E}\t{test_loss:.2E}', end='\t')
                    print(f'{format_elapsed_time(0, self.times[-1])}', end='\t' if (validation_loss < best_loss) and (epoch > 0) else '\n')

                # Save parameters
                if 'PCNN' in self.module:
                    p = self.model.E_parameters
                    self.a.append(p[0])
                    self.b.append(p[1])
                    self.c.append(p[2])
                    self.d.append(p[3])

                # Save last and possibly best model
                if self.save_model:
                    self.save(name_to_add="last", verbose=0)
                    if validation_loss < best_loss:
                        self.save(name_to_add="best", verbose=0)
                    
                # Store the best loss
                if validation_loss < best_loss:
                    best_loss = validation_loss
                    if epoch > 0:
                        print('New best!')

            if self.verbose > 0:
                time.sleep(0.7) # For clean printing, let the last test losses be printed before going ahead
                best_epoch = np.argmin([x for x in self.validation_losses])
                logger.info(f"The best model was obtained at epoch {best_epoch + 1} after training for " f"{trained_epochs + n_epochs} epochs in {format_elapsed_time(0, self.times[-1])}")
                logger.info(f"Train loss:\t{self.train_losses[best_epoch]:.2E}")
                logger.info(f"Val loss:\t{self.validation_losses[best_epoch]:.2E}")
                logger.info(f"Test loss:\t{self.test_losses[best_epoch]:.2E}")

            if output_best:
                self.load(load_last=False, verbose=0)
                logger.info('Loaded and returned the best obtained model!')


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
            print(f'New {name_to_add}!')

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
                "test_losses": self.test_losses,
                "times": self.times,
                "a": self.a,
                "b": self.b,
                "c": self.c,
                "d": self.d,
                "model_kwargs": self.model_kwargs,
                "data_kwargs": self.data_kwargs
            },
            save_name,
        )

    def load(self, load_last: bool = False, verbose: int = None):
        """
        Function trying to load an existing model, by default the best one if it exists. But for training purposes,
        it is possible to load the last state of the model instead.

        Args:
            load_last:  Flag to set to True if the last model checkpoint is wanted, instead of the best one

        Returns:
             Nothing, everything is done in place and stored in the parameters.
        """

        if verbose is None:
            verbose = self.verbose

        if load_last:
            save_name = os.path.join(self.save_name, "last_model.pt")
        else:
            save_name = os.path.join(self.save_name, "best_model.pt")

        if verbose > 0:
            logger.info("Trying to load a trained model...")
        try:
            # Build the full path to the model and check its existence

            assert os.path.exists(save_name), f"The file {save_name} doesn't exist."

            # Load the checkpoint
            checkpoint = torch.load(save_name, map_location=lambda storage, loc: storage, weights_only=False)

            # Put it into the model
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            if self.device != 'cpu':
                for state in self.optimizer.state.values():
                    for k, v in state.items():
                        if isinstance(v, torch.Tensor):
                            state[k] = v.to(self.device)
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
            for key in self.model_kwargs:
                if key in checkpoint['model_kwargs']:
                    if isinstance(self.model_kwargs[key], dict):
                        for x,y in zip(self.model_kwargs[key].keys(), checkpoint['model_kwargs'][key].keys()):
                            if isinstance(self.model_kwargs[key][x], list):
                                for a,b in zip(self.model_kwargs[key][x], checkpoint['model_kwargs'][key][y]):
                                    if a != b:
                                        logger.warning(f"The parameter {key} was found in the checkpoint as {checkpoint['model_kwargs'][key]} "
                                            f"while you passed {self.model_kwargs[key]}.")
                    elif np.any(self.model_kwargs[key] != checkpoint['model_kwargs'][key]):
                        logger.warning(f"The parameter {key} was found in the checkpoint as {checkpoint['model_kwargs'][key]} "
                                       f"while you passed {self.model_kwargs[key]}.")
            for key in self.data_kwargs:
                if key in checkpoint['data_kwargs']:
                    if np.any(self.data_kwargs[key] != checkpoint['data_kwargs'][key]):  
                        logger.warning(f"The parameter {key} was found in the checkpoint as {checkpoint['data_kwargs'][key]} "
                                       f"while you passed {self.data_kwargs[key]}.")
            self.model_kwargs = checkpoint["model_kwargs"]
            self.data_kwargs = checkpoint["data_kwargs"]

            # Print the current status of the found model
            if verbose > 0:
                logger.info(f"Found! It contains {len(self.train_sequences)} training, {len(self.validation_sequences)} validation, and {len(self.test_sequences)} test sequences.")
                if load_last:
                    logger.info(f"The model has been fitted for {len(self.train_losses)} epochs already. Last checkpoint:")
                else:
                    logger.info(f"The model achieved its best performance after {len(self.train_losses)} epochs:")
                logger.info(f"Train loss:\t{self.train_losses[-1]:.2E}")
                logger.info(f"Val loss:\t{self.validation_losses[-1]:.2E}")
                logger.info(f"Test loss:\t{self.test_losses[-1]:.2E}")

        # Otherwise, keep an empty model, return nothing
        except AssertionError:
            logger.info("No existing model was found!")
