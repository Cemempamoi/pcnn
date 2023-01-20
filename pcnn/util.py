import os
import pandas as pd
import numpy as np
from typing import Union

from parameters import DATA_SAVE_PATH

def load_data(save_name: str, save_path: str = DATA_SAVE_PATH) -> pd.DataFrame:
    """
    Function to load a dataframe if it exists

    Args:
        save_name:  Name of the file to load
        save_path:  Where to load it
    """

    # Build the full path and check its existence
    full_path = os.path.join(save_path, save_name + ".csv")
    assert os.path.exists(full_path), f"The file {full_path} doesn't exist."

    # Load the data and change the index to timestamps
    data = pd.read_csv(full_path, index_col=[0])
    data.index = pd.to_datetime(data.index)

    return data

def standardize(data: pd.DataFrame):
    """
    function to standardize the columns of a DataFrame.

    Args:
        data: The DataFrame to standardize

    Returns:
        the standardized data, the means and the stds
    """

    # Define and keep the means and stds in memory
    mean = data.mean()
    std = data.std()

    # Substract the mean
    data = data.subtract(mean)

    # Little trick to handle constant data (zero std --> cannot divide)
    # Only consider non zero variance columns
    non_zero_std = np.where(data.std().values > 1e-10)[0]
    # If there is a zero variance column, it is actually useless since it is constant
    if len(non_zero_std) < len(data.columns):
        print(f"Warning, 0 std for columns {data.columns[np.where(data.std().values < 1e-10)[0]].values}, really useful?")

    # Divide by the std where possible
    data.iloc[:, non_zero_std] = data.iloc[:, non_zero_std].divide(std[non_zero_std])

    return data, mean, std


def inverse_standardize(data: pd.DataFrame, mean: pd.Series, std: pd.Series):
    """
    function to inverse the standardization of the columns of a DataFrame.

    Args:
        data:   The DataFrame to inverse standardize
        mean:   The mean values of the columns
        std:    The standard deviations of the columns

    Returns:
        The original data
    """

    # If the given data is a number
    if isinstance(data, float):
        # Can get back to the original scale back
        data = mean + (data * std)

    else:
        # Get the places where the variance is not zero and multiply back
        # (the other columns were ignored)
        non_zero_std = np.where(std.values > 1e-10)[0]

        # In the case of an array
        if isinstance(data, np.ndarray):
            data[:, non_zero_std] = data[:, non_zero_std] * std[non_zero_std].values.reshape(1, -1)
            data += mean

        # If the given data is an DataFrame
        elif isinstance(data, pd.DataFrame):
            data.iloc[:, non_zero_std] = data.iloc[:, non_zero_std].multiply(std[non_zero_std])
            # Add the mean back
            data = data.add(mean)

        else:
            raise ValueError(f"Unexpected data type {type(data)}")

    return data


def normalize(data: pd.DataFrame):
    """
    Function to normalize the columns of a DataFrame to 0.1-0.9

    Args:
        data: The DataFrame to standardize

    Returns:
        the normalized data, the mins and the maxs
    """

    # Define and save the min and max of each column
    max_ = data.max()
    min_ = data.min()

    # Little trick to handle constant data (zero std --> cannot divide)
    # Only consider non zero variance columns
    non_zero_div = np.where(max_ - min_ > 1e-10)[0]
    # If there is a zero variance column, it is actually useless since it is constant
    if len(non_zero_div) < len(data.columns):
        print(f"Warning, columns {data.columns[np.where(max_ - min_ < 1e-10)[0]].values} are constant, really useful?")

    # Scale the data between 0.1 and 0.9
    data.iloc[:, non_zero_div] = 0.8 * (data.iloc[:, non_zero_div] - min_[non_zero_div]) / (max_[non_zero_div] - min_[non_zero_div]) + 0.1

    data.iloc[:, np.where(max_ - min_ <= 1e-10)[0]] = 0.5

    return data, min_, max_


def inverse_normalize(data: Union[np.ndarray, pd.DataFrame, float], min_: Union[pd.Series, float], max_: Union[pd.Series, float]):
    """
    Function to inverse the normalization of the columns of a DataFrame to 0.1-0.9

    Args:
        data:   The DataFrame to inverse standardize
        min_:   The min values of the columns
        max_:   The max values of the columns

    Returns:
        The original data
    """

    # If the given data is a number
    if isinstance(data, float):
        # Can get back to the original scale back
        data = min_ + (data - 0.1) * (max_ - min_) / 0.8

    else:
        # Get the places where the variance is not zero and multiply back
        # (the other columns were ignored)
        non_zero_div = np.where(max_ - min_ > 1e-10)[0]

        # If the given data is an array
        if isinstance(data, np.ndarray):
            # Can get back to the original scale back
            data[:, non_zero_div] = (data[:, non_zero_div] - 0.1) * (max_[non_zero_div] - min_[non_zero_div]).values.reshape(1, -1) / 0.8
            data[:, non_zero_div] = data[:, non_zero_div] + min_[non_zero_div].values.reshape(1, -1)

        # If the given data is a DataFrame
        elif isinstance(data, pd.DataFrame):
            # Can get back to the original scale back
            data.iloc[:, non_zero_div] = (data.iloc[:, non_zero_div] - 0.1).multiply(max_[non_zero_div] - min_[non_zero_div]) / 0.8
            data.iloc[:, non_zero_div] = data.iloc[:, non_zero_div].add(min_[non_zero_div])

        else:
            raise ValueError(f"Unexpected data type {type(data)}")

    return data

def model_save_name_factory(rooms, module, model_kwargs):
    """
    Function to create helpful and somewhat unique names to easily save and load the wanted models
    This uses the starting and ending date of the data used to fit the model, as well as another
    part in "model_name" that is specific to each model, representing the model type and possibly
    some hyperparameters' choices

    Args:
        rooms:          List of rooms modeled
        module:         Module used in the model
        model_kwargs:   Parameters of the model, see 'parameters.py'

    Returns:
        A full name to save the model as
    """

    name = ''
    for room in rooms:
        name += f"{room}_"
    name = os.path.join(model_kwargs["save_path"], f"{model_kwargs['name']}_{model_kwargs['seed']}_"
                                                   f"{module}_{name}_")

    name += f"_{model_kwargs['warm_start_length']}_{model_kwargs['maximum_sequence_length']}"

    if (model_kwargs["heating"]) & (not model_kwargs["cooling"]):
        name += "_heating"
    elif (model_kwargs["cooling"]) & (not model_kwargs["heating"]):
        name += "_cooling"

    return name


def format_elapsed_time(tic, toc):
    """
    Small function to print the time elapsed between tic and toc in a nice manner
    """

    diff = toc - tic
    hours = int(diff // 3600)
    minutes = int((diff - 3600 * hours) // 60)
    seconds = str(int(diff - 3600 * hours - 60 * minutes))

    # Put everything in strings
    hours = str(hours)
    minutes = str(minutes)

    # Add a zero for one digit numbers for consistency
    if len(hours) == 1:
        hours = '0' + hours
    if len(minutes) == 1:
        minutes = '0' + minutes
    if len(seconds) == 1:
        seconds = '0' + seconds

    # Final nice looking print
    return f"{hours}:{minutes}:{seconds}"

def load_data(save_name: str, save_path: str = DATA_SAVE_PATH) -> pd.DataFrame:
    """
    Function to load a dataframe if it exists

    Args:
        save_name:  Name of the file to load
        save_path:  Where to load it
    """

    # Build the full path and check its existence
    full_path = os.path.join(save_path, save_name + ".csv")
    assert os.path.exists(full_path), f"The file {full_path} doesn't exist."

    # Load the data and change the index to timestamps
    data = pd.read_csv(full_path, index_col=[0])
    data.index = pd.to_datetime(data.index)

    return data