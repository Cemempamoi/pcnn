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
        if isinstance(value, float) or isinstance(value, int) or isinstance(value, str):
            return [value]
        else:
            if isinstance(value, pd.Series):
                return value.values
            else:
                return value
    else:
        return None


def common_initialization(temperature_difference, factor, time_elapsed_hours, parameters):
    """
    Function to compute the common initialization of the physical parameters

    Args:
        temperature_difference:         List of the temperature differences per room
        factor:                         Factor causing the observed differences, also as a lit
        time_elasped:                   List of the time elapsed per room (in hours)
        max_temperature:                Maximum room temperature of the data
        min_temperature:                Minimum room temperature of the data

    Returns:
        basic initialization values to be standardized
    """

    # Need the output as an array
    temperature_difference = ensure_list(temperature_difference)
    factor = ensure_list(factor)
    # For robustness in case a Series is passed
    max_temperature = np.array(ensure_list(parameters['max_temperature'])) 
    min_temperature = np.array(ensure_list(parameters['min_temperature'])) 

    ## Heat losses approximation
    
    # T_diff_inside ~ b * T_diff_outside * time_elapsed  (or c * T_diff_neighboring_oom * time_elapsed)
    # --> b ~ T_diff_inside / T_diff_outside / time_elapsed (or c ~ T_diff_neighboring_room / T_diff_outside / time_elapsed)
    
    # or # T_diff_inside ~ a * power * time_elapsed (or T_diff_inside ~ d * power * time_elapsed)
    # --> a ~ T_diff_inside / power / time_elapsed (or d ~ T_diff_inside / power / time_elapsed)
    initial_values = np.array(temperature_difference) / np.array(factor) / np.array(time_elapsed_hours)

    # Discretization to the right interval
    initial_values = initial_values / 60 * parameters['interval_minutes'] 

    return initial_values, max_temperature, min_temperature


def initialize_heat_losses_to_outside(temperature_difference: list, temperature_difference_to_outsie: list, time_elapsed_hours: list, parameters: dict):
    """
    Function to compute the heat losses from temperature differences

    Args:
        temperature_difference:             List of the temperature differences per room
        temperature_difference_to_outsie:   List of the temperature differences per room
        time_elasped:                       List of the time elapsed per room (in hours)
        interval:                           Interval of the data (in hours)
        min_temperature:                    Minimum room temperature of the data        
        max_temperature:                    Maximum room temperature of the data

    Returns:
        initial values for 'b' 
    """

    initial_values, max_temperature, min_temperature = common_initialization(temperature_difference=temperature_difference, 
                                                                             factor=temperature_difference_to_outsie, 
                                                                             time_elapsed_hours=time_elapsed_hours, 
                                                                             parameters=parameters)
     
    # Rescale to work with normalized data since PCNN predictions are betwween 0.1 and 0.9
    initial_values = initial_values / (max_temperature - min_temperature) * 0.8

    return ensure_list(initial_values)
    

def initialize_heat_losses_to_neighbors(neighboring_rooms: list, temperature_difference: list, temperature_difference_to_neighbors: list, time_elapsed_hours: list, parameters: dict):
    """
    Function to compute the heat losses from temperature differences

    Args:
        neighboring_rooms:                      List of the pairs of neighboring rooms - if None then assumes single-zone PCNN
        temperature_difference:                 List of the degrees lost per room
        temperature_difference_to_neighbors:    List of the temperature differences per room
        time_elasped:                           List of the time elapsed per room (in hours)
        interval:                               Interval of the data (in hours)
        min_temperature:                        Minimum room temperature of the data
        max_temperature:                        Maximum room temperature of the data

    Returns:
        initial values for 'b' 
    """

    initial_values, max_temperature, min_temperature = common_initialization(temperature_difference=temperature_difference,
                                                                             factor=temperature_difference_to_neighbors,
                                                                             time_elapsed_hours=time_elapsed_hours,
                                                                             parameters=parameters)

    # Need to ensure right normalization for neighboring rooms having different temperature ranges
    # To correctly initialize 'c'
    if neighboring_rooms is not None:
        initial_values_split = [[], []]
        # Ensure we have the right number of initial values (if all the same)
        if len(initial_values) == 1:
            initial_values = list(initial_values) * len(neighboring_rooms)
        # Loop over the pairs of rooms and create two lists to have the right normalization for each room
        for i, (room1, room2) in enumerate(neighboring_rooms):
            initial_values_split[0].append(initial_values[i] / (max_temperature[room1] - min_temperature[room1]) * 0.8)
            initial_values_split[1].append(initial_values[i] / (max_temperature[room2] - min_temperature[room2]) * 0.8)
        initial_values = initial_values_split

    else:
        initial_values = initial_values / (max_temperature - min_temperature) * 0.8

    return ensure_list(initial_values)


def initialize_heat_gains_from_heating_cooling(temperature_difference: list, power: list, time_elapsed_hours: list, parameters: dict):
    """
    Function to compute the heat gains from power inputs

    Args:
        degrees_lost:             List of the degrees lost per room
        power:                    List of the power consumption per room during that time
        time_interval:            List of the time elapsed per room (in hours)
        interval:                 Interval of the data (in minutes)
        min_temperature:          Minimum room temperature of the data
        max_temperature:          Maximum room temperature of the data

    Returns:
        initial values for 'a' or 'd'
    """

    initial_values, max_temperature, min_temperature = common_initialization(temperature_difference=temperature_difference, 
                                                                              factor=power, 
                                                                              time_elapsed_hours=time_elapsed_hours, 
                                                                              parameters=parameters)

    # Rescale to work with normalized data since PCNN predictions are betwween 0.1 and 0.9
    initial_values = initial_values / (max_temperature - min_temperature) * 0.8

    # Cooling parameters must also be positive
    return np.abs(ensure_list(initial_values))


def check_initialization_physical_parameters(initial_values_physical_parameters, data_params):
    """
    Function to check the initialization of the physical parameters
    """
    assert len(initial_values_physical_parameters['a']) == len(data_params['temperature_column']), \
        f"The initial value of a is not the right size! You have {len(data_params['temperature_column'])} rooms but {len(initial_values_physical_parameters['a'])} initial values for 'a'."
    
    if data_params['outside_walls'] is not None:
        assert len(initial_values_physical_parameters['b']) == len(data_params['outside_walls']), \
            f"The initial value of b is not the right size! You have {len(data_params['outside_walls'])} external walls but {len(initial_values_physical_parameters['b'])} initial values for 'b'."
    # Single-zone PCNNs have only one 'b' parameters
    else:
        assert len(initial_values_physical_parameters['b']) == 1, \
            f"The initial value of b is not the right size! You have 1 external walls but {len(initial_values_physical_parameters['b'])} initial values for 'b'."
    
    if data_params['neighboring_rooms'] is not None:
        assert len(initial_values_physical_parameters['c']) == len(data_params['neighboring_rooms']), \
            f"The initial value of c is not the right size! You have {len(data_params['neighboring_rooms'])} pairs of rooms but {len(initial_values_physical_parameters['c'])} initial values for 'c'."
    # Single-zone PCNNs don't use 'neighboring_rooms'
    elif data_params['neigh_column'] is not None:
        assert len(initial_values_physical_parameters['c']) == len(data_params['neigh_column']), \
            f"The initial value of c is not the right size! You have {len(data_params['neigh_column'])} pairs of rooms but {len(initial_values_physical_parameters['c'])} initial values for 'c'."
    
    assert len(initial_values_physical_parameters['d']) == len(data_params['temperature_column']), \
        f"The initial value of d is not the right size! You have {len(data_params['temperature_column'])} rooms but {len(initial_values_physical_parameters['d'])} initial values for 'd'."