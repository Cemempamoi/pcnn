import os
import pandas as pd
import numpy as np
from typing import Union
from loguru import logger
import torch
from contextlib import contextmanager
from timeit import default_timer

from pcnn.parameters import DATA_SAVE_PATH


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


def normalize(data: pd.DataFrame):
    """
    Function to normalize the columns of a DataFrame to 0.1-0.9

    Args:
        data: The DataFrame to normalize

    Returns:
        the normalized data, the mins and the maxs
    """

    data = data.astype(float)

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
    data.iloc[:, non_zero_div] = 0.8 * (data.iloc[:, non_zero_div] - min_.iloc[non_zero_div]) / (max_.iloc[non_zero_div] - min_.iloc[non_zero_div]) + 0.1

    data.iloc[:, np.where(max_ - min_ <= 1e-10)[0]] = 0.5

    return data, min_, max_


def inverse_normalize(data: Union[np.ndarray, pd.DataFrame, float], min_: Union[pd.Series, float], max_: Union[pd.Series, float]):
    """
    Function to inverse the normalization of the columns of a DataFrame to 0.1-0.9

    Args:
        data:   The DataFrame to inverse normalize
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
            data.iloc[:, non_zero_div] = (data.iloc[:, non_zero_div] - 0.1).multiply(max_.iloc[non_zero_div] - min_.iloc[non_zero_div]) / 0.8
            data.iloc[:, non_zero_div] = data.iloc[:, non_zero_div].add(min_.iloc[non_zero_div])

        else:
            raise ValueError(f"Unexpected data type {type(data)}")

    return data


def model_save_name_factory(module, model_kwargs):
    """
    Function to create helpful and somewhat unique names to easily save and load the wanted models
    This uses the starting and ending date of the data used to fit the model, as well as another
    part in "model_name" that is specific to each model, representing the model type and possibly
    some hyperparameters' choices

    Args:
        module:         Module used in the model
        model_kwargs:   Parameters of the model, see 'parameters.py'

    Returns:
        A full name to save the model as
    """

    name = os.path.join(model_kwargs["save_path"], f"{model_kwargs['name']}_{model_kwargs['seed']}_"
                                                   f"{module}")

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


def check_GPU_availability():
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        logger.info("GPU acceleration on!")
    elif torch.backends.mps.is_available():
        device = 'mps'
        logger.info("Using mps.")
    else:
        device = "cpu"
        logger.info("Using CPU.")
    return device


@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start


def ensure_list(value):
    if value is not None:
        if isinstance(value, list):
            return value
        else:
            return [value]
    else:
        return None


def initialize_heat_losses_from_temperature_differences(degrees_lost: list, temperature_difference: list, time_elapsed_hours: list, parameters: dict):
    """
    Function to compute the heat losses from temperature differences

    Args:
        degrees_lost:             List of the degrees lost per room
        temperature_differences:  List of the temperature differences per room
        time_elasped:             List of the time elapsed per room (in hours)
        interval:                 Interval of the data (in hours)
        min_temperature:          Minimum room temperature of the data
        max_temperature:          Maximum room temperature of the data

    Returns:
        initial values for 'b' or 'c'
    """

    # Need the output as an array
    if isinstance(degrees_lost, float) or isinstance(degrees_lost, int):
        degrees_lost = [degrees_lost]
    if isinstance(temperature_difference, float) or isinstance(temperature_difference, int):
        temperature_difference = [temperature_difference]

    # Heat losses approximation
    # T_diff_inside ~ b * T_diff_outside * time_elapsed  (or c * T_diff_neighboring room * time_elapsed)
    # --> b ~ T_diff_inside / T_diff_outside / time_elapsed (or c ~ T_diff_neighboring_room / T_diff_outside / time_elapsed)
    initial_values = np.array(degrees_lost) / np.array(temperature_difference) / np.array(time_elapsed_hours)

    # Discretization to the right interval
    initial_values = initial_values / 60 * parameters['interval_minutes'] 

    # Rescale to work with normalized data since PCNN predictions are betwween 0.1 and 0.9
    # Note: there is no need to apply normalization to the temperature differnces here since
    # the data is "unnormalized" in the PCNN modules during computation
    initial_values = initial_values/ (parameters['max_temperature'] - parameters['min_temperature']) * 0.8

    return initial_values


def initialize_heat_gains_from_heating_cooling(degrees_difference: list, power: list, time_elapsed_hours: list, parameters: dict):
    """
    Function to compute the heat gains from power inputs

    Args:
        degrees_lost:             List of the degrees lost per room
        temperature_differences:  List of the temperature differences per room
        time_interval:            List of the time elapsed per room (in hours)
        interval:                 Interval of the data (in minutes)
        min_temperature:          Minimum room temperature of the data
        max_temperature:          Maximum room temperature of the data
        min_power:                Minimum power of the data
        max_power:                Maximum power of the data

    Returns:
        initial values for 'a' or 'd'
    """

    # Need the output as an array
    if isinstance(degrees_difference, float) or isinstance(degrees_difference, int):
        degrees_difference = [degrees_difference]
    if isinstance(power, float) or isinstance(power, int):
        power = [power]

    # Normalized power
    zero_power = 0. - parameters['min_power'] / (parameters['max_power'] - parameters['min_power']) * 0.8 + 0.1
    # Subtract the zero power tas this is what is actually input in E in the PCNN modules
    power = np.array(power) - parameters['min_power'] / (parameters['max_power'] - parameters['min_power']) * 0.8 + 0.1 - zero_power

    # Heat losses approximation
    # T_diff_inside ~ a * power * time_elapsed 
    # --> a ~ T_diff_inside / power / time_elapsed 
    initial_values = np.array(degrees_difference) / power / np.array(time_elapsed_hours)

    # Discretization to the right interval
    initial_values = initial_values / 60 * parameters['interval_minutes'] 

    # Rescale to work with normalized data since PCNN predictions and power inputs are betwween 0.1 and 0.9
    initial_values = initial_values / (parameters['max_temperature'] - parameters['min_temperature']) * 0.8

    # Cooling parameters must also be positive
    return np.abs(initial_values)