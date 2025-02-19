from loguru import logger
import pandas as pd
import numpy as np

from pcnn.util import normalize, inverse_normalize, ensure_list


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

        # Data arguments
        self.data = data
        self.interval = (data.index[1] - data.index[0]).seconds / 60

        self.is_normalized = False
        self.min_ = None
        self.max_ = None

        data_kwargs = self.ensure_columns_list(data_kwargs)

        # Need to retain the names of the columns for later use
        self.temperature_column_name = data_kwargs['temperature_column']

        self.normalized_columns = self.get_columns_to_normalize(data_kwargs)
        data_kwargs = self.get_columns_placements(data_kwargs)

        # Sanity check
        self.check_columns(data_kwargs)
        
        # Normalizing the inputs to D and outputs
        if data_kwargs['verbose'] > 0:
            logger.info("Normalizing the data...")
        # Order might matter
        self.normalize(columns=self.normalized_columns)

        # Define inputs and labels    
        self.X = self.data[self.X_columns].iloc[:-1, :].copy().values
        self.Y = self.data[self.Y_columns].iloc[1:, :].copy().values

        self.data_kwargs = data_kwargs

    def ensure_columns_list(self, data_kwargs: dict):
        data_kwargs['case_column'] = ensure_list(data_kwargs['case_column'])
        data_kwargs['neigh_column'] = ensure_list(data_kwargs['neigh_column'])
        data_kwargs['temperature_column'] = ensure_list(data_kwargs['temperature_column']) 
        data_kwargs['power_column'] = ensure_list(data_kwargs['power_column'])
        return data_kwargs
    
    def get_columns_to_normalize(self, data_kwargs: dict):
        """
        Need to normalize the inputs to D, the outputs (temperatures), 
        and the "case" column defining heating or cooling, as this is used
        in module.py and expected to be between 0.1 (cooling) or 0.9 (heating)
        """
        return [x for x in self.data.columns if x in 
                data_kwargs['inputs_D'] + data_kwargs['temperature_column'] + data_kwargs['case_column']]

    def get_columns_placements(self, data_kwargs: dict):
        """
        Torch works with tensors, so we need to know which columns is where, i.e., get their index
        """
        data_kwargs['case_column'] = [i for i,x in enumerate(self.X_columns) if x in data_kwargs['case_column']]
        data_kwargs['out_column'] = [i for i,x in enumerate(self.X_columns) if x == data_kwargs['out_column']][0]
        data_kwargs['neigh_column'] = [i for i,x in enumerate(self.X_columns) if x in data_kwargs['neigh_column']]
        data_kwargs['temperature_column'] = [i for i,x in enumerate(self.X_columns) if x in data_kwargs['temperature_column']]
        data_kwargs['power_column'] = [i for i,x in enumerate(self.X_columns) if x in data_kwargs['power_column']]
        data_kwargs['inputs_D'] = [i for i,x in enumerate(self.X_columns) if x in data_kwargs['inputs_D']]
        return data_kwargs
    
    def check_columns(self, data_kwargs: dict):
        """
        Sanity check of the columns
        """
        if data_kwargs['neigh_column'] is None:
            logger.info(f'Sanity check of the columns:\n{[(w, [self.X_columns[i] for i in x]) 
                                                          for w, x in zip(['Case', 'Room temp', 'Room power', 'Out temp'],
                                    [data_kwargs['case_column'], data_kwargs['temperature_column'], 
                                     data_kwargs['power_column'], [data_kwargs['out_column']]])]}')
        else:
            logger.info(f'Sanity check of the columns:\n{[(w, [self.X_columns[i] for i in x]) 
                                                          for w, x in zip(['Case', 'Room temp', 'Room power', 'Out temp', 'Neigh temp'],
                                    [data_kwargs['case_column'], data_kwargs['temperature_column'], 
                                     data_kwargs['power_column'], [data_kwargs['out_column']], data_kwargs['neigh_column']])]}')

        logger.info(f"Inputs used in D:\n{np.array(self.X_columns)[data_kwargs['inputs_D']]}")

    def normalize(self, data=None, columns=None):
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

        if columns is None:
            columns = data.columns

        # Normalize the data and recall mins and maxes
        data[columns], min_, max_ = normalize(data[columns])

        if inplace:
            self.data = data
            self.min_ = min_
            self.max_ = max_
        else:
            return data, min_, max_

    def inverse_normalize(self, data=None, min_=None, max_=None, inplace: bool = False, columns=None):
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

        if columns is None:
            columns = self.normalized_columns

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
        data[columns] = inverse_normalize(data=data_[columns], min_=min_, max_=max_)

        # Return the scaled data if wanted
        if inplace:
            self.data = data
        else:
            return data
        
    def get_temperarature_min_and_range(self):
        """
        Function to get the minimum and the amplitude of room temperatures.
        This is used by the physics-inspired network to unnormalize the predictions.
        """
        return self.min_.loc[self.temperature_column_name].values,\
            (self.max_ - self.min_).loc[self.temperature_column_name].values


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

    assert len(dataset.X) == len(dataset.Y), "Something weird happened!"

    # Print the result and return it
    if verbose > 0:
        logger.info("Data ready!\n")

    return dataset
