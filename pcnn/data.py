from loguru import logger
import pandas as pd
import numpy as np
import torch

from pcnn.util import normalize, inverse_normalize


class DataSet:
    def __init__(self, data: pd.DataFrame, data_kwargs: dict) -> None:

        if data_kwargs['verbose'] > 0:
            logger.info("Preparing the data...")

        # Only keep useful columns
        X_columns = data_kwargs['X_columns'] 
        if X_columns is not None:
            to_keep = [x for x in data.columns if x in list(set(X_columns) | set(data_kwargs['Y_columns']))]
            data.drop(columns=[x for x in data.columns if x not in to_keep], inplace=True)
            # Reorder the columns to ensure they are in the right order
            self.X_columns = [x for x in data.columns if x in X_columns]
        else:
            self.X_columns = data.columns

        self.Y_columns = data_kwargs['Y_columns']
        
        # Define all needed columns
        self.case_column = data_kwargs['case_column'] if isinstance(data_kwargs['case_column'], list) else [data_kwargs['case_column']]
        self.out_column = data_kwargs['out_column']
        if data_kwargs['neigh_column'] is not None:
            self.neigh_column = data_kwargs['neigh_column'] if isinstance(data_kwargs['neigh_column'], list) else [data_kwargs['neigh_column']]
        else:
            self.neigh_column = data_kwargs['neigh_column']
        self.temperature_column = data_kwargs['temperature_column'] if isinstance(data_kwargs['temperature_column'], list) else [data_kwargs['temperature_column']]
        self.power_column = data_kwargs['power_column'] if isinstance(data_kwargs['power_column'], list) else [data_kwargs['power_column']]
        self.inputs_D = data_kwargs['inputs_D']
        self.topology = data_kwargs['topology']

        # Sanity check
        if self.neigh_column is None:
            logger.info(f'Sanity check of the columns:\n{[(w, [self.X_columns[i] for i in x]) 
                                                          for w, x in zip(['Case', 'Room temp', 'Room power', 'Out temp'],
                                    [self.case_column, self.temperature_column, self.power_column, [self.out_column]])]}')
        else:
            logger.info(f'Sanity check of the columns:\n{[(w, [self.X_columns[i] for i in x]) 
                                                          for w, x in zip(['Case', 'Room temp', 'Room power', 'Out temp', 'Neigh temp'],
                                    [self.case_column, self.temperature_column, self.power_column, [self.out_column], self.neigh_column])]}')

        logger.info(f"Inputs used in D:\n{np.array(self.X_columns)[self.inputs_D]}")

        # Define inputs and labels
        self.data = data
        self.X = self.data[self.X_columns].iloc[:-1, :].copy().values
        self.Y = self.data[self.Y_columns].iloc[1:, :].copy().values
        self.interval = (data.index[1] - data.index[0]).seconds / 60

        self.is_normalized = False
        self.min_ = None
        self.max_ = None

    def normalize(self, data=None):
        """
        Function to normalize the dataset, i.e. scale it by the min and max values
        The min and max of each column (sensor) is kept in memory to reverse the
        operation and to be able to apply them to other datasets

        There is an additional trick here, the data is actually scaled between
        0.1 and 0.9 instead of the classical 0-1 scaling to avoid saturation
        This is supposed to help the learning
        """
        inplace = False

        # If no data is provided, take the entire data
        if data is None:
            data = self.data

            # First, check that no normalization or standardization was already performed
            assert not self.is_normalized, "The data is already normalized!"

            # Set the normalized flag to true
            self.is_normalized = True
            inplace = True

        # Normalize the data and recall mins and maxes
        data, min_, max_ = normalize(data)

        if inplace:
            self.data = data
            self.min_ = min_
            self.max_ = max_
        else:
            return data, min_, max_

    def inverse_normalize(self, data=None, min_=None, max_=None, inplace: bool = False):
        """
        Function to reverse the normalization to get the original scales back. If no data is provided,
        The entire data is scaled back.

        Args:
            data:       Data to inverse normalize
            min_:       Min to use (takes the one of the dataset by default)
            max_:       Max to use (takes the one of the dataset by default)
            inplace:    Flag to do the operation in place or not

        Returns:
            Normalized data if wanted
        """

        # If no data is provided, take the entire DataFrame of the DataSet
        if data is None:
            # First sanity check: the data is already normalized
            assert self.is_normalized, "The data is not normalized!"
            if inplace:
                self.is_normalized = False
                data_ = self.data
            else:
                data_ = self.data.copy()

        # Otherwise, copy the data to avoid issues
        else:
            if type(data) == pd.Series:
                data_ = pd.DataFrame(data=data, index=data.index, columns=[data.name])
            else:
                data_ = data.copy()

        if min_ is None:
            min_ = self.min_[data_.columns]
        if max_ is None:
            max_ = self.max_[data_.columns]

        # Inverse the normalization using the needed mins and maxes
        data = inverse_normalize(data=data_, min_=min_, max_=max_)

        # Return the scaled data if wanted
        if inplace:
            self.data = data
        else:
            return data
        
    def get_normalization_variables(self):
        """
        Function to get the minimum and the amplitude of some variables in the data. In particular, we need
        that for the room temperature, the outside temperature and the neighboring room temperature.
        This is used by the physics-inspired network to unnormalize the predictions.
        """
        normalization_variables = {}
        normalization_variables['Room'] = [self.min_.iloc[self.temperature_column].values,
                                           (self.max_ - self.min_).iloc[self.temperature_column].values]
        if self.neigh_column is not None:
            normalization_variables['Neigh'] = [self.min_.iloc[self.neigh_column].values,
                                           (self.max_ - self.min_).iloc[self.neigh_column].values]
        normalization_variables['Out'] = [[self.min_.iloc[self.out_column]],
                                          [(self.max_ - self.min_).iloc[self.out_column]]]
        return normalization_variables
    
    def compute_zero_power(self):
        """
        Small helper function to compute the scaled value of zero power
        """

        # Scale the zero
        if self.is_normalized:
            min_ = self.min_.iloc[self.power_column]
            max_ = self.max_.iloc[self.power_column]
            zero = 0.8 * (0.0 - min_) / (max_ - min_) + 0.1

        else:
            zero = np.array([0.0] * len(self.power_column))

        return np.array(zero)


def prepare_data(data: pd.DataFrame, data_kwargs: dict, verbose: int = 2):
    """
    Pipeline of actions to take to prepare the data for some model.

    Args:
        data:                   DataFrame to use
        data_kwargs:            Parameters of the data
        verbose:                Verbose of the function

    Returns:
        A preprocessed dataset ready to be put into a model
    """

    # Use the custom function to load and prepare the full dataset 
    data_kwargs['verbose'] = verbose
    dataset = DataSet(data=data.copy(), data_kwargs=data_kwargs)

    # If normalization is wanted
    if data_kwargs['to_normalize']:
        if verbose > 0:
            logger.info("Normalizing the data...")
        dataset.normalize()

    if dataset.data.min().min() < 0.05:
        raise ValueError("The data needs to be normalized between 0.1 and 0.9. If it is not already the case, set `to_normalize=True` in the parameters to rescale it accordingly.")

    assert len(dataset.X) == len(dataset.Y), "Something weird happened!"

    # Print the result and return it
    if verbose > 0:
        logger.info("Data ready!\n")

    return dataset
