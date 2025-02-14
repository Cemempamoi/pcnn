from loguru import logger
import pandas as pd
from pcnn.util import standardize, inverse_standardize, normalize, inverse_normalize

class DataSet:
    def __init__(self, data, interval, to_normalize, to_standardize) -> None:

        self.data = data
        self.interval = interval
        self.to_normalize = to_normalize
        self.to_standardize = to_standardize

        self.is_normalized = False
        self.is_standardized = False

        self.X = None
        self.Y = None
        self.min_ = None
        self.max_ = None
        self.mean = None
        self.std = None

    def standardize(self, data=None):
        """
        Function to standardize the dataset, i.e. put it to zero mean and 1 std
        The mean and std of each column (sensor) is kept in memory to reverse the
        operation and to be able to apply them to other datasets
        """

        inplace = False

        # If no data is provided, take the entire data of the class instance
        if data is None:
            data = self.data

            # First, check that no normalization or standardization was already performed
            assert not self.is_normalized, "The data is already normalized!"
            assert not self.is_standardized, "The data is already standardized!"

            # Set the standardized flag to true
            self.is_standardized = True
            inplace = True

        # Standardize the data, recall the mean and std
        data, mean, std = standardize(data)

        if inplace:
            self.data = data
            self.mean = mean
            self.std = std
        else:
            return data, mean, std

    def inverse_standardize(self, data=None, mean=None, std=None, inplace: bool = False):
        """
        Function to reverse the standardization to get the original scales back. If no data is provided,
        the entire dataset is scaled back.

        Args:
            data:       Data to inverse standardize
            mean:       Mean to use (takes the one of the dataset by default)
            std:        Std to use (takes the one of the dataset by default)
            inplace:    Flag to do the operation in place or not

        Returns:
            Standardized data if wanted
        """

        # First sanity check: the data is already standardized
        assert self.is_standardized, "The data is not standardized!"

        # If no data is provided, take the entire DataFrame of the DataSet
        if data is None:
            # First sanity check: the data is already normalized
            assert self.is_normalized, "The data is not normalized!"
            if inplace:
                self.is_standardized = False
                data_ = self.data
            else:
                data_ = self.data.copy()

        # Otherwise, copy the data to avoid issues
        else:
            if type(data) == pd.Series:
                data_ = pd.DataFrame(data=data, index=data.index, columns=[data.name])
            else:
                data_ = data.copy()

        if mean is None:
            mean = self.mean[data_.columns]
        if std is None:
            std = self.std[data_.columns]

        # Inverse the standardization with the wanted means and stds
        data = inverse_standardize(data=data_, mean=mean, std=std)

        # Return the scaled data if wanted
        if inplace:
            self.data = data
        else:
            return data

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
            assert not self.is_standardized, "The data is already standardized!"
            assert not self.is_normalized, "The data is already normalized!"

            # Set the standardized flag to true
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

def prepare_data(data: pd.DataFrame, interval: int, model_kwargs: dict, Y_columns: list, X_columns: list = None, verbose: int = 2):
    """
    Pipeline of actions to take to prepare the data for some model.

    Args:
        data:                   DataFrame to use
        interval:               Sampling time (in minutes) of the data
        model_kwargs:           Model arguments (start and end date, interval, ...), see 'parameters.py'
        predict_differences:    Whether to predict differences in outputs
        Y_columns:              Name of the columns that are to be predicted
        X_columns:              Sensors (columns) of the input data, if None all columns are kept
        verbose:                Verbose of the function

    Returns:
        A preprocessed dataset ready to be put into a model
    """

    # Use the custom function to load and prepare the full dataset from the NEST data
    dataset = DataSet(data=data.copy(), interval=interval, to_normalize=model_kwargs['to_normalize'], 
                        to_standardize=model_kwargs['to_standardize'])

    if verbose > 0:
        logger.info("Preparing the data...")

    # Reorder the columns to ensure they are in the right order
    dataset.X_columns = [x for x in dataset.data.columns if x in X_columns]
    dataset.Y_columns = Y_columns # [y for y in dataset.data.columns if y in Y_columns]

    # Only keep useful columns
    if X_columns is not None:
        to_keep = [x for x in dataset.data.columns if x in list(set(X_columns) | set(Y_columns))]
        dataset.data.drop(columns=[x for x in dataset.data.columns if x not in to_keep], inplace=True)

    # If standardization is wanted
    if dataset.to_standardize:
        if verbose > 0:
            logger.info("Standardizing the data...")
        dataset.standardize()

    # Else, normalization is usually done
    elif dataset.to_normalize:
        if verbose > 0:
            logger.info("Normalizing the data...")
        dataset.normalize()

    # Define inputs and labels
    if X_columns is not None:
        dataset.X = dataset.data[dataset.X_columns].iloc[:-1, :].copy().values
    else:
        dataset.X = dataset.data.iloc[:-1, :].copy().values
    dataset.Y = dataset.data[dataset.Y_columns].iloc[1:, :].copy().values

    assert len(dataset.X) == len(dataset.Y), "Something weird happened!"

    # Print the result and return it
    if verbose > 0:
        logger.info("Data ready!\n")

    return dataset
