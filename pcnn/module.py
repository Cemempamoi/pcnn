import torch
from torch import nn
import numpy as np


class PositiveLinear(nn.Module):
    """
    https://discuss.pytorch.org/t/positive-weights/19701/7
    """
    def __init__(self, in_features, out_features, require_bias=True):
        super(PositiveLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.require_bias = require_bias
        self.log_weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if self.require_bias:
            self.bias = nn.Parameter(torch.Tensor(1, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.log_weight, 0.)

    def forward(self, input):
        if self.require_bias:
            return nn.functional.linear(input, self.log_weight.exp()) + self.bias
        else:
            return nn.functional.linear(input, self.log_weight.exp())


class PCNN(nn.Module):
    """
    S-PCNN model, from the paper
    `Towards Scalable Physically Consistent Neural Networks: an Application to Data-driven Multi-zone Thermal Building Models`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland

    The PCNN has two parts to predict the temperature evolution in a zone of a building
      - A base module `D` that computes unknown effect using neural networks
      - A linear module, the energy accumulator `E`, which includes prior physical knowledge:
        - Energy inputs from the HVAC system increase (heating) or decrease (cooling) `E`
        - Heat losses are proportional to temperature gradients, both to the outside and neighboring zone

    The final temperature prediction is `T=D+E` at each step.
    `D` and `E` are then carried on to the next step and updated.
    """

    def __init__(
            self,
            device,
            inputs_D: list,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            temperature_column: int,
            out_column: int,
            neigh_column: int,
            power_column: int,
            zero_power: float,
            division_factor: list,
            normalization_variables: dict,
            parameter_scalings: dict,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            inputs_D:                   Input features used to predict the unforce dynamics (D)
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            neigh_column:               Index of the column corresponding to the neighboring room temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            division_factor:            Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.inputs_D = inputs_D
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.neigh_column = neigh_column
        self.zero_power = torch.Tensor(zero_power).to(self.device)
        self.division_factor = torch.Tensor(division_factor).to(self.device)

        # Define latent variables
        self.last_D = None  ## D
        self.last_E = None  ## E

        # Recall normalization constants
        self.room_min = normalization_variables['Room'][0]
        self.room_diff = normalization_variables['Room'][1]
        self.neigh_min = normalization_variables['Neigh'][0]
        self.neigh_diff = normalization_variables['Neigh'][1]
        self.out_min = normalization_variables['Out'][0]
        self.out_diff = normalization_variables['Out'][1]

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = torch.Tensor(parameter_scalings['a']).to(self.device)
        self.b_scaling = torch.Tensor(parameter_scalings['b']).to(self.device)
        self.c_scaling = torch.Tensor(parameter_scalings['c']).to(self.device)
        self.d_scaling = torch.Tensor(parameter_scalings['d']).to(self.device)

        # Build the models
        self._build_model()

    def _build_model(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        ## Initialization of the parameters of `E`
        self.a = PositiveLinear(1, 1, require_bias=False)
        self.b = PositiveLinear(1, 1, require_bias=False)
        self.c = PositiveLinear(1, 1, require_bias=False)
        self.d = PositiveLinear(1, 1, require_bias=False)

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            size = [len(self.inputs_D)] + self.input_nn_hidden_sizes
            self.input_nn = nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), nn.ReLU())
                                                for i in range(0, len(size) - 1)])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.inputs_D)
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.lstm_num_layers, batch_first=True)
        if self.layer_norm:
            self.norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules 
        # ensure the last layer has size 1 since we only model one zone
        sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
        self.output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh())
                                                for i in range(0, len(sizes) - 1)])

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if ('bias' in name) or ('log_weight' in name):
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)


    def forward(self, x_: torch.Tensor, states=None, warm_start: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (h, c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                h = torch.stack([self.initial_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                c = torch.stack([self.initial_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last_D = torch.zeros((x.shape[0], 1)).to(self.device)  ## D
            self.last_E = torch.zeros((x.shape[0], 1)).to(self.device)  ## E

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, self.temperature_column] = self.last_D
        else:
            x[:, -1, self.temperature_column] = x[:, -1, self.temperature_column].clone() - self.last_E

        ## Forward 'D'
        # Input embedding when wanted
        if self.feed_input_through_nn:
            D_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.inputs_D]
                for layer in self.input_nn:
                    temp = layer(temp)
                D_embedding[:, time_step, :] = temp
        else:
            D_embedding = x[:, :, self.inputs_D]

        # LSTM prediction for the base temperature
        lstm_output, (h, c) = self.lstm(D_embedding, (h, c))

        if self.layer_norm:
            lstm_output = self.norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        # Put the data is the form needed for the neural net
        temp = lstm_output[:, -1, :]
        # Go through the input layer of the NN
        for layer in self.output_nn:
            temp = layer(temp)
        D = temp / self.division_factor + x[:, -1, self.temperature_column]
        
        ## Heat losses computation in 'E'
        E = self.last_E.clone()

        # Loss to the outside is b*(T_k-T^out_k)
        if self.out_column is not None:
            E = E.clone() - self.b(
                ((x[:, -1, self.temperature_column] + self.last_E.clone() - 0.1) / 0.8
                  * self.room_diff + self.room_min)
                 - ((x[:, -1, [self.out_column]] - 0.1) / 0.8
                    * self.out_diff + self.out_min)) / self.b_scaling

        # Loss to the neighboring zone is c*(T_k-T^neigh_k)
        for i in range(len(self.neigh_column)):
            E = E.clone() - self.c(
                (((x[:, -1, self.temperature_column]+ self.last_E.clone() - 0.1) / 0.8
                  * self.room_diff + self.room_min)
                - ((x[:, -1, [self.neigh_column[i]]] - 0.1) / 0.8
                    * self.neigh_diff[i] + self.neigh_min[i]))) / self.c_scaling

        ## Heating/cooling effect of HVAC on 'E'

        # Find sequences in the batch where there actually is heating/cooling
        # Trick needed to put padded values back to zero power before detecting power intputs
        temp = torch.where(x[:, :, self.power_column] > 0.05, x[:, :, self.power_column], self.zero_power)
        mask = torch.where(torch.abs(temp - self.zero_power) > 1e-6, True, False).squeeze(1)

        if len(mask) > 0:

            # Find heating and cooling sequences
            heating = x[mask, 0, self.case_column] > 0.5
            cooling = x[mask, 0, self.case_column] < 0.5

            temp = x[mask, -1, :].clone()
            # Substract the 'zero power' to get negative values for cooling power
            power = temp[:, self.power_column] - self.zero_power

            if sum(heating) > 0:
                # Heating effect: add a*u to 'E'
                E[mask][heating, :] = E[mask][heating, :].clone() + self.a(power[heating]) / self.a_scaling

            if sum(cooling) > 0:
                # Cooling effect: add d*u (where u<0 now, so we actually subtract energy) to 'E'
                E[mask][cooling, :] = E[mask][cooling, :].clone() + self.d(power[cooling]) / self.d_scaling

        # Recall 'D' and 'E' for the next time step
        self.last_D = D.clone()
        self.last_E = E.clone()

        # Final computation of the result 'T=D+E'
        output = D + E
        # Trick needed since some sequences are padded
        output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.
        # Return the predictions and states of the model
        return output, (h, c)

    @property
    def E_parameters(self):
        return [float(x._parameters['log_weight']) for x in [self.a, self.b, self.c, self.d]]


class S_PCNN(nn.Module):
    """
    S-PCNN model, from the paper
    `Towards Scalable Physically Consistent Neural Networks: an Application to Data-driven Multi-zone Thermal Building Models`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland

    The PCNN has two parts to predict the temperature evolution in a zone of a building
      - A base module `D` that computes unknown effect using neural networks
      - A linear module, the energy accumulator `E`, which includes prior physical knowledge:
        - Energy inputs from the HVAC system increase (heating) or decrease (cooling) `E`
        - Heat losses are proportional to temperature gradients, both to the outside and neighboring zone

    The final temperature prediction is `T=D+E` at each step.
    `D` and `E` are then carried on to the next step and updated.
    """

    def __init__(
            self,
            device,
            inputs_D: list,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            temperature_column: int,
            out_column: int,
            power_column: int,
            zero_power: float,
            division_factor: list,
            normalization_variables: dict,
            parameter_scalings: dict,
            topology: dict,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            inputs_D:                   Input features used to predict the unforce dynamics (D)
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            division_factor:            Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
            topology:                   Topology of the building to model
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.inputs_D = inputs_D
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.zero_power = torch.Tensor(zero_power).to(self.device)
        self.division_factor = torch.Tensor(division_factor).to(self.device)
        self.topology = topology

        # Define latent variables
        self.last_D = None  ## D
        self.last_E = None  ## E

        # Recall normalization constants
        self.room_min = normalization_variables['Room'][0]
        self.room_diff = normalization_variables['Room'][1]
        self.out_min = normalization_variables['Out'][0]
        self.out_diff = normalization_variables['Out'][1]

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = torch.Tensor(parameter_scalings['a']).to(self.device)
        self.b_scaling = torch.Tensor(parameter_scalings['b']).to(self.device)
        self.c_scaling = torch.Tensor(parameter_scalings['c']).to(self.device)
        self.d_scaling = torch.Tensor(parameter_scalings['d']).to(self.device)

        # Build the models
        self._build_model()

    def _build_model(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        ## Initialization of the parameters of `E`
        self.a = nn.ModuleList(
            [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Rooms']))])
        self.b = nn.ModuleList(
            [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Outside']))])
        self.c = nn.ModuleList(
            [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Neighbors']))])
        self.d = nn.ModuleList(
            [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Rooms']))])

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            size = [len(self.inputs_D)] + self.input_nn_hidden_sizes
            self.input_nn = nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), nn.ReLU())
                                                for i in range(0, len(size) - 1)])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.inputs_D)
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.lstm_num_layers, batch_first=True)
        if self.layer_norm:
            self.norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [len(self.topology['Rooms'])]
        self.output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh())
                                                for i in range(0, len(sizes) - 1)])

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if ('bias' in name) or ('log_weight' in name):
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)


    def forward(self, x_: torch.Tensor, states=None, warm_start: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (h, c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                h = torch.stack([self.initial_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                c = torch.stack([self.initial_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last_D = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## D
            self.last_E = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## E

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, self.temperature_column] = self.last_D
        else:
            x[:, -1, self.temperature_column] = x[:, -1, self.temperature_column].clone() - self.last_E

        ## Forward 'D'
        # Input embedding when wanted
        if self.feed_input_through_nn:
            D_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.inputs_D]
                for layer in self.input_nn:
                    temp = layer(temp)
                D_embedding[:, time_step, :] = temp
        else:
            D_embedding = x[:, :, self.inputs_D]

        # LSTM prediction for the base temperature
        lstm_output, (h, c) = self.lstm(D_embedding, (h, c))

        if self.layer_norm:
            lstm_output = self.norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        # Put the data is the form needed for the neural net
        temp = lstm_output[:, -1, :]
        # Go through the input layer of the NN
        for layer in self.output_nn:
            temp = layer(temp)
        D = temp / self.division_factor + x[:, -1, self.temperature_column]
        
        ## Heat losses computation in 'E'
        E = self.last_E.clone()  

        # Loss to the outside is b*(T_k-T^out_k)
        for i, room in enumerate(self.topology['Outside']):
            E[:, room] = E[:, room].clone() - self.b[i](
                (((x[:, -1, self.temperature_column[room]]
                   + self.last_E[:, room].clone() - 0.1) / 0.8
                  * self.room_diff[room] + self.room_min[room])
                 - ((x[:, -1, self.out_column] - 0.1) / 0.8
                    * self.out_diff + self.out_min)).
                    reshape(-1, 1)).squeeze() / self.b_scaling

        # Loss to the neighboring zone is c*(T_k-T^neigh_k)
        for i, (rooma, roomb) in enumerate(self.topology['Neighbors']):
            for room1, room2 in zip([rooma, roomb], [roomb, rooma]):
                E[:, room1] = E[:, room1].clone() - self.c[i](
                    (((x[:, -1, self.temperature_column[room1]]
                    + self.last_E[:, room1].clone() - 0.1) / 0.8
                    * self.room_diff[room1] + self.room_min[room1])
                    - ((x[:, -1, self.temperature_column[room2]]
                        + self.last_E[:, room2].clone() - 0.1) / 0.8
                        * self.room_diff[room2] + self.room_min[room2])).
                        reshape(-1, 1)).squeeze() / self.c_scaling

        ## Heating/cooling effect of HVAC on 'E'
        
        # Find sequences in the batch where there actually is heating/cooling
        # Trick needed to put padded values back to zero power before detecting power intputs
        temp = torch.where(x[:, :, self.power_column] > 0.05, x[:, :, self.power_column], self.zero_power)
        mask = torch.where(torch.abs(temp - self.zero_power) > 1e-6, True, False).squeeze(1).sum(axis=-1) > 0
        
        if len(mask) > 0:

            # Find heating and cooling sequences
            heating = x[mask, 0, self.case_column] > 0.5
            cooling = x[mask, 0, self.case_column] < 0.5

            temp = x[mask, -1, :].clone()
            # Substract the 'zero power' to get negative values for cooling power
            power = temp[:, self.power_column] - self.zero_power

            if sum(heating) > 0:
                # Heating effect: add a*u to 'E'
                for i in range(len(self.topology['Rooms'])):
                    E[mask][heating, i] = E[mask][heating, i].clone() + self.a[i](
                        power[heating, i].unsqueeze(-1)).squeeze() / self.a_scaling[i]

            if sum(cooling) > 0:
                # Cooling effect: add d*u (where u<0 now, so we actually subtract energy) to 'E'
                for i in range(len(self.topology['Rooms'])):
                    E[mask][cooling, i] = E[mask][cooling, i].clone() + self.d[i](
                        power[cooling, i].unsqueeze(-1)).squeeze() / self.d_scaling[i]

        # Recall 'D' and 'E' for the next time step
        self.last_D = D.clone()
        self.last_E = E.clone()

        # Final computation of the result 'T=D+E'
        output = D + E
        # Trick needed since some sequences are padded
        output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.

        # Return the predictions and states of the model
        return output, (h, c)

    @property
    def E_parameters(self):
        return [[float(y._parameters['log_weight']) for y in x] for x in [self.a, self.b, self.c, self.d]]



class M_PCNN(nn.Module):
    """
    S-PCNN model, from the paper
    `Towards Scalable Physically Consistent Neural Networks: an Application to Data-driven Multi-zone Thermal Building Models`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland

    The PCNN has two parts to predict the temperature evolution in a zone of a building
      - A base module `D` that computes unknown effect using neural networks
      - A linear module, the energy accumulator `E`, which includes prior physical knowledge:
        - Energy inputs from the HVAC system increase (heating) or decrease (cooling) `E`
        - Heat losses are proportional to temperature gradients, both to the outside and neighboring zone

    The final temperature prediction is `T=D+E` at each step.
    `D` and `E` are then carried on to the next step and updated.
    """

    def __init__(
            self,
            device,
            inputs_D: list,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            output_nn_hidden_sizes: list,
            case_column: int,
            temperature_column: int,
            out_column: int,
            power_column: int,
            zero_power: float,
            division_factor: list,
            normalization_variables: dict,
            parameter_scalings: dict,
            topology: dict,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            inputs_D:                   Input features used to predict the unforce dynamics (D)
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            case_column:                Index of the column corresponding to the case
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            out_column:                 Index of the column corresponding to the outside temperature
            zero_power:                 Scaled value of zero heating/cooling power. This is needed since the
                                          model should predict no power consumption if the valves stay closed
            division_factor:            Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
            normalization_variables:    Minimum and differences of various parameters in the data, so that we can
                                          unnormalize data when needed
            parameter_scalings:         Scalings of the parameters a, b, c, d in the network for a better learning
                                          Corresponds to a good initialization
            topology:                   Topology of the building to model
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.inputs_D = inputs_D
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.case_column = case_column
        self.temperature_column = temperature_column
        self.power_column = power_column
        self.out_column = out_column
        self.zero_power = torch.Tensor(zero_power).to(self.device)
        self.division_factor = torch.Tensor(division_factor).to(self.device)
        self.topology = topology

        # Define latent variables
        self.last_D = None  ## D
        self.last_E = None  ## E

        # Recall normalization constants
        self.room_min = normalization_variables['Room'][0]
        self.room_diff = normalization_variables['Room'][1]
        self.out_min = normalization_variables['Out'][0]
        self.out_diff = normalization_variables['Out'][1]

        # Specific scalings for the physical parameters `a`, `b`, `c`, `d`
        self.a_scaling = torch.Tensor(parameter_scalings['a']).to(self.device)
        self.b_scaling = torch.Tensor(parameter_scalings['b']).to(self.device)
        self.c_scaling = torch.Tensor(parameter_scalings['c']).to(self.device)
        self.d_scaling = torch.Tensor(parameter_scalings['d']).to(self.device)

        # Build the models
        self._build_model()

    def _build_model(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        ## Initialization of the parameters of `E`
        self.a = nn.ModuleList(
            [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Rooms']))])
        self.b = nn.ModuleList(
            [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Outside']))])
        self.c = nn.ModuleList(
            [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Neighbors']))])
        self.d = nn.ModuleList(
            [PositiveLinear(1, 1, require_bias=False) for _ in range(len(self.topology['Rooms']))])

        ## Initialization of `D`
        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_h = nn.ParameterList([nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size)) for _ in range(len(self.topology['Rooms']))])
            self.initial_c = nn.ParameterList([nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size)) for _ in range(len(self.topology['Rooms']))])

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            sizes = [[len(input_D)] + self.input_nn_hidden_sizes for input_D in self.inputs_D]
            self.input_nn = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), nn.ReLU())
                                                               for i in range(0, len(size) - 1)]) for size in sizes])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_sizes = [self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(input_D) for
                           input_D in self.inputs_D] 
        self.lstm = nn.ModuleList([nn.LSTM(input_size=lstm_input_sizes[i], hidden_size=self.lstm_hidden_size,
                                                num_layers=self.lstm_num_layers, batch_first=True) for i in
                                        range(len(self.topology['Rooms']))])
        if self.layer_norm:
            self.norm = nn.ModuleList(
                [nn.LayerNorm(normalized_shape=self.lstm_hidden_size) for _ in range(len(self.topology['Rooms']))])

        # Create the NNs to process the output of the LSTMs for each modules if wanted
        sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [1]
        self.output_nn = nn.ModuleList([nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh())
                                                            for i in range(0, len(sizes) - 1)]) for _ in
                                                range(len(self.topology['Rooms']))])

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if ('bias' in name) or ('log_weight' in name):
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)


    def forward(self, x_: torch.Tensor, states=None, warm_start: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (h, c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                h = [torch.stack([initial_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device) for
                          initial_h in self.initial_h]
                c = [torch.stack([initial_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device) for
                          initial_c in self.initial_c]
            else:
                h = [torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device) for _
                          in range(len(self.topology['Rooms']))]
                c = [torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device) for _
                          in range(len(self.topology['Rooms']))]

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last_D = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## D
            self.last_E = torch.zeros((x.shape[0], len(self.topology['Rooms']))).to(self.device)  ## E

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, self.temperature_column] = self.last_D
        else:
            x[:, -1, self.temperature_column] = x[:, -1, self.temperature_column].clone() - self.last_E

        ## Forward 'D'
        D = torch.zeros(x.shape[0], len(self.topology['Rooms'])).to(self.device) 

        # Input embedding when wanted
        if self.feed_input_through_nn:
            embeddings = []
            for i, input_nn in enumerate(self.input_nn):
                D_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
                for time_step in range(x.shape[1]):
                    temp = x[:, time_step, self.inputs_D[i]]
                    for layer in input_nn:
                        temp = layer(temp)
                    D_embedding[:, time_step, :] = temp
                embeddings.append(D_embedding)
        else:
            D_embedding = [x[:, :, input_D] for input_D in self.inputs_D]

        # LSTM prediction for the base temperature
        lstm_outputs = []
        for i, lstm in enumerate(self.lstm):
            lstm_output, (h[i], c[i]) = lstm(embeddings[i], (h[i], c[i]))
            lstm_outputs.append(lstm_output)

        if self.layer_norm:
            for i in range(len(self.norm)):
                lstm_outputs[i] = self.norm[i](lstm_outputs[i])

        # Some manipulations are needed to feed the output through the neural network
        for j, output_nn in enumerate(self.output_nn):
            # Put the data is the form needed for the neural net
            temp = lstm_outputs[j][:, -1, :]
            # Go through the input layer of the NN
            for layer in output_nn:
                temp = layer(temp)
            D[:, j] = temp.squeeze() / self.division_factor + x[:, -1, self.temperature_column[j]]

        ## Heat losses computation in 'E'
        E = self.last_E.clone()

        # Loss to the outside is b*(T_k-T^out_k)
        for i, room in enumerate(self.topology['Outside']):
            E[:, room] = E[:, room].clone() - self.b[i](
                (((x[:, -1, self.temperature_column[room]]
                   + self.last_E[:, room].clone() - 0.1) / 0.8
                  * self.room_diff[room] + self.room_min[room])
                 - ((x[:, -1, self.out_column] - 0.1) / 0.8
                    * self.out_diff + self.out_min)).
                    reshape(-1, 1)).squeeze() / self.b_scaling

        # Loss to the neighboring zone is c*(T_k-T^neigh_k)
        for i, (rooma, roomb) in enumerate(self.topology['Neighbors']):
            for room1, room2 in zip([rooma, roomb], [roomb, rooma]):
                E[:, room1] = E[:, room1].clone() - self.c[i](
                    (((x[:, -1, self.temperature_column[room1]]
                    + self.last_E[:, room1].clone() - 0.1) / 0.8
                    * self.room_diff[room1] + self.room_min[room1])
                    - ((x[:, -1, self.temperature_column[room2]]
                        + self.last_E[:, room2].clone() - 0.1) / 0.8
                        * self.room_diff[room2] + self.room_min[room2])).
                        reshape(-1, 1)).squeeze() / self.c_scaling

        ## Heating/cooling effect of HVAC on 'E'
        # Find sequences in the batch where there actually is heating/cooling
        # Trick needed to put padded values back to zero power before detecting power intputs
        temp = torch.where(x[:, :, self.power_column] > 0.05, x[:, :, self.power_column], self.zero_power)
        mask = torch.where(torch.abs(temp - self.zero_power) > 1e-6, True, False).squeeze(1).sum(axis=-1) > 0

        if len(mask) > 0:

            # Find heating and cooling sequences
            heating = x[mask, 0, self.case_column] > 0.5
            cooling = x[mask, 0, self.case_column] < 0.5

            temp = x[mask, -1, :].clone()
            # Substract the 'zero power' to get negative values for cooling power
            power = temp[:, self.power_column] - self.zero_power

            if sum(heating) > 0:
                # Heating effect: add a*u to 'E'
                for i in range(len(self.topology['Rooms'])):
                    E[mask][heating, i] = E[mask][heating, i].clone() \
                                                  + self.a[i](power[heating, i].unsqueeze(-1)).squeeze() \
                                                  / self.a_scaling[i]

            if sum(cooling) > 0:
                # Cooling effect: add d*u (where u<0 now, so we actually subtract energy) to 'E'
                for i in range(len(self.topology['Rooms'])):
                    E[mask][cooling, i] = E[mask][cooling, i].clone() \
                                                  + self.d[i](power[cooling, i].unsqueeze(-1)).squeeze() \
                                                  / self.d_scaling[i]

        # Recall 'D' and 'E' for the next time step
        self.last_D = D.clone()
        self.last_E = E.clone()

        # Final computation of the result 'T=D+E'
        output = D + E
        # Trick needed since some sequences are padded
        output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.

        # Return the predictions and states of the model
        return output, (h, c)

    @property
    def E_parameters(self):
        return [[float(y._parameters['log_weight']) for y in x] for x in [self.a, self.b, self.c, self.d]]

class LSTM(nn.Module):
    """
    Classical LSTM model for comparison (with an encoder and decoder), from
    `Towards Scalable Physically Consistent Neural Networks: an Application to Data-driven Multi-zone Thermal Building Models`
    L. Di Natale (1,2), B. Svetozarevic (1), P. Heer (1) and C.N. Jones (2)
    1 - Urban Energy Systems, Empa, Switzerland
    2 - Laboratoire d'Automatique 3, EPFL, Switzerland
    """

    def __init__(
            self,
            device,
            rooms: list,
            inputs_D: list,
            learn_initial_hidden_states: bool,
            feed_input_through_nn: bool,
            input_nn_hidden_sizes: list,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            layer_norm: bool,
            output_nn_hidden_sizes: list,
            temperature_column: list,
            power_column: list,
            division_factor: list,
    ):
        """
        Function to build the models.

        Args:
            device:                     CPU or GPU
            inputs_D:                   Input features used to predict the unforce dynamics (D)
            learn_initial_hidden_states:Whether to learn the initial hidden and cell states
            feed_input_through_nn:      Flag whether or not to preprocess the inputs before the LSTMs
            input_nn_hidden_sizes:      Hidden sizes of the input processing NNs
            lstm_hidden_size:           Hidden size of the LSTMs processing the input
            lstm_num_layers:            Number of layers for the LSTMs
            layer_norm:                 Flag whether to include a layer normalization layer after the LSTM
            output_nn_hidden_sizes:     Hidden sizes of the NNs
            temperature_column:         Index of the column corresponding to the room temperature
            power_column:               Index of the column corresponding to the power consumption
            division_factor:            Factors to scale the base predictions and ensure not too big differences
                                          between timesteps
        """

        # Start by initializing a PyTorch module
        super().__init__()

        # Recall the parameters for further use
        self.device = device
        self.rooms = rooms
        self.inputs_D = inputs_D
        self.learn_initial_hidden_states = learn_initial_hidden_states
        self.feed_input_through_nn = feed_input_through_nn
        self.input_nn_hidden_sizes = input_nn_hidden_sizes
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.layer_norm = layer_norm
        self.output_nn_hidden_sizes = output_nn_hidden_sizes
        self.temperature_column = temperature_column
        self.power_columns = power_column
        self.division_factor = torch.Tensor(division_factor)

        # Define latent variables
        self.last = None  

        # Build the models
        self._build_model()

    def _build_model(self) -> None:
        """
        Function to build the model, i.e. a base temperature prediction `D` and then the added effect of
        heating/cooling and heat losses to the outside and neighboring zone in `E`.
            - `D` is composed of a LSTM, possibly followed by a NN and preceeded by another one,
                with a skip connection (ResNet)
            - `E` is a linear module with parameters `a`, `b`, `c`, `d`
        """

        # Hidden and cell state initialization
        if self.learn_initial_hidden_states:
            self.initial_h = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))
            self.initial_c = nn.Parameter(data=torch.zeros(self.lstm_num_layers, self.lstm_hidden_size))

        # Process the input by a NN if wanted
        if self.feed_input_through_nn:
            size = [len(self.inputs_D)] + self.input_nn_hidden_sizes
            self.input_nn = nn.ModuleList([nn.Sequential(nn.Linear(size[i], size[i + 1]), nn.ReLU())
                                                for i in range(0, len(size) - 1)])

        # Create the LSTMs at the core of `D`, with normalization layers
        lstm_input_size = self.input_nn_hidden_sizes[-1] if self.feed_input_through_nn else len(self.inputs_D)
        self.lstm = nn.LSTM(input_size=lstm_input_size, hidden_size=self.lstm_hidden_size,
                                 num_layers=self.lstm_num_layers, batch_first=True)
        if self.layer_norm:
            self.norm = nn.LayerNorm(normalized_shape=self.lstm_hidden_size)

        # Create the NNs to process the output of the LSTMs for each modules
        sizes = [self.lstm_hidden_size] + self.output_nn_hidden_sizes + [len(self.rooms)]
        self.output_nn = nn.ModuleList([nn.Sequential(nn.Linear(sizes[i], sizes[i + 1]), nn.Tanh())
                                                for i in range(0, len(sizes) - 1)])

        # Xavier initialization of all the weights of NNs, parameters `a`, `b`, `c`, `d` are set to 1
        for name, param in self.named_parameters():
            if 'norm' not in name:
                if 'bias' in name:
                    nn.init.constant_(param, 0.0)
                elif 'weight' in name:
                    nn.init.xavier_normal_(param)


    def forward(self, x_: torch.Tensor, states=None, warm_start: bool = False) -> dict:  # noqa: C901
        """
        Custom redefinition of the forward path. The only sepcifc thing is the warm start, where we feed
        true temperatures back to the model.
        Another trick: we actually predict the power along the temperature. This is a generalization, in the
        paper the power is known, so we just store one input as output. This is to be used when there is
        no direct control over the power input and it needs preprocessing through some 'g'.

        Args:
            x_:         Input
            states:     Original hidden and cell states if known
                          (for all LSTMs, i.e. the base, the heating and the cooling)
            warm_start: Whether we are warm starting the model
            mpc_mode:   Flag to pass to the MPC mode and return D and E separately

        Returns:
            The predicted temperature and power, and a tuple containing the hidden and cell states of
              the LSTMs
        """

        x = x_.clone()

        if len(x.shape) == 2:
            x = x.unsqueeze(1)

        # Take the given hidden and cell states of the LSTMs if they exist
        if states is not None:
            (h, c) = states

        # Otherwise, the sequence of data is new, so we need to initialize the hidden and cell states
        else:
            if self.learn_initial_hidden_states:
                h = torch.stack([self.initial_h.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
                c = torch.stack([self.initial_c.clone() for _ in range(x.shape[0])], dim=1).to(self.device)
            else:
                h = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)
                c = torch.zeros((self.lstm_num_layers, x.shape[0], self.lstm_hidden_size)).to(self.device)

            # Define vectors to store 'D' and 'E', which will evolve through time
            self.last = torch.zeros((x.shape[0], len(self.temperature_column))).to(self.device)  ## D

        # Use the last base predictions as input when not warm starting, otherwise we keep the true temperature
        if not warm_start:
            x[:, -1, self.temperature_column] = self.last

        ## Forward 'D'
        # Input embedding when wanted
        if self.feed_input_through_nn:
            D_embedding = torch.zeros(x.shape[0], x.shape[1], self.input_nn_hidden_sizes[-1]).to(self.device)
            for time_step in range(x.shape[1]):
                temp = x[:, time_step, self.inputs_D]
                for layer in self.input_nn:
                    temp = layer(temp)
                D_embedding[:, time_step, :] = temp
        else:
            D_embedding = x[:, :, self.inputs_D]

        # LSTM prediction for the base temperature
        lstm_output, (h, c) = self.lstm(D_embedding, (h, c))

        if self.layer_norm:
            lstm_output = self.norm(lstm_output)

        # Some manipulations are needed to feed the output through the neural network if wanted
        # Put the data is the form needed for the neural net
        temp = lstm_output[:, -1, :]
        # Go through the input layer of the NN
        for layer in self.output_nn:
            temp = layer(temp)
        output = temp.squeeze() / self.division_factor + x[:, -1, self.temperature_column]

        # Recall 'D' and 'E' for the next time step
        self.last = output.clone()

        # Trick needed since some sequences are padded
        output[torch.where(x[:, -1, 0] < 1e-6)[0], :] = 0.

        # Return the predictions and states of the model
        return output, (h, c)
