import numpy as np
from . Base import BaseLayer
from . FullyConnected import FullyConnected
from . Sigmoid import Sigmoid
from . TanH import TanH


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.trainable = True
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(shape=(1, self.hidden_size))
        self._memorize = False
        self._weights = np.random.uniform(0, 1, size=(self.hidden_size + self.input_size + 1, self.hidden_size))
        self.weights_output = np.random.uniform(0, 1, size=(self.hidden_size + 1, self.output_size))
        self.fc_hidden = None
        self.fc_output = None
        self.sigmoid = None
        self.tanh = None
        self.create_embedded_layers()
        self.input_tensor = None
        self.current_hidden_error = np.zeros(shape=(1, self.hidden_size))
        self.tanh_activations = None
        self.sigmoid_activations = None
        self.hidden_inputs = None
        self.output_inputs = None
        self._optimizer = None
        self._gradient_weights = None
        self.gradient_weights_output = None

    @property
    def optimizer(self):
        """
        Getter for attribute _optimizer.

        Returns:
            Sgd: Optimizer of the layer.
        """
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        """
        Setter for attribute _optimizer.

        Args:
            optimizer(Sgd): New optimizer for the layer.

        Returns:
            None
        """
        self._optimizer = optimizer

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @gradient_weights.setter
    def gradient_weights(self, weights):
        self._gradient_weights = weights

    @property
    def weights(self):
        return self._weights

    @weights.setter
    def weights(self, weights):
        self._weights = weights

    def create_embedded_layers(self):
        self.fc_hidden = FullyConnected(self.hidden_size + self.input_size, self.hidden_size)
        self.fc_output = FullyConnected(self.hidden_size, self.output_size)
        self.sigmoid = Sigmoid()
        self.tanh = TanH()

    def initialize(self, weight_initializer, bias_initializer):
        fan_in = self.hidden_size + self.input_size + 1
        fan_out = self.hidden_size
        self.weights = weight_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.weights_output = weight_initializer.initialize(self.weights_output.shape, fan_in, fan_out)
        self.fc_hidden.weights = self.weights
        self.fc_output.weights = self.weights_output

    def forward(self, input_tensor):

        self.input_tensor = input_tensor

        sigmoid_activations = np.zeros(shape=(self.input_tensor.shape[0], self.output_size))
        tanh_activations = np.zeros(shape=(self.input_tensor.shape[0], self.hidden_size))
        output_tensor = np.zeros(shape=(self.input_tensor.shape[0], self.output_size))

        hidden_inputs = np.zeros(shape=(self.input_tensor.shape[0], self.input_size + self.hidden_size + 1))
        output_inputs = np.zeros(shape=(self.input_tensor.shape[0], self.hidden_size + 1))

        if not self.memorize:
            self.hidden_state = np.zeros(shape=(1, self.hidden_size))

        for i in range(self.input_tensor.shape[0]):

            # Prepare input for the first pass.
            current_input_tensor = input_tensor[i, :]
            current_input_tensor = np.transpose(current_input_tensor.reshape(-1, 1))
            input_hidden = np.concatenate((self.hidden_state, current_input_tensor), axis=1)

            # Compute hidden state.
            hidden_state = self.tanh.forward(self.fc_hidden.forward(input_hidden))
            self.hidden_state = hidden_state

            # Compute output state.
            # input_output = np.append(self.hidden_state, np.ones(shape=(input_hidden.shape[0], 1)), axis=1)
            input_output = hidden_state
            output = self.sigmoid.forward(self.fc_output.forward(input_output))
            output_tensor[i, :] = output

            # Save activations.
            tanh_activations[i, :] = hidden_state
            sigmoid_activations[i, :] = output

            # Save inputs for FC layers.
            hidden_inputs[i, :] = self.fc_hidden.input_tensor
            output_inputs[i, :] = self.fc_output.input_tensor

        self.tanh_activations = tanh_activations
        self.sigmoid_activations = sigmoid_activations
        self.hidden_inputs = hidden_inputs
        self.output_inputs = output_inputs

        return output_tensor

    def backward(self, error_tensor):

        accumulated_weights_gradient = np.zeros_like(self.weights)
        accumulated_output_gradient = np.zeros_like(self.weights_output)

        previous_error_tensor = np.zeros_like(self.input_tensor)
        self.current_hidden_error = 0
        for i in reversed(range(len(self.input_tensor))):
            # Set the activations of the activation layers.
            current_activation = self.tanh_activations[i, :]
            current_activation = np.transpose(current_activation.reshape(-1, 1))
            self.tanh.activations = current_activation

            current_activation = self.sigmoid_activations[i, :]
            current_activation = np.transpose(current_activation.reshape(-1, 1))
            self.sigmoid.activations = current_activation

            # Set the inputs of the FC layers.
            current_hidden_input = self.hidden_inputs[i, :]
            # current_hidden_input = np.append(current_hidden_input, 1)
            current_hidden_input = np.transpose(current_hidden_input.reshape(-1, 1))
            self.fc_hidden.input_tensor = current_hidden_input

            current_output_input = self.output_inputs[i, :]
            # current_output_input = np.append(current_output_input, 1)
            current_output_input = np.transpose(current_output_input.reshape(-1, 1))
            self.fc_output.input_tensor = current_output_input

            # Propagate error from yt to xt.
            previous_error = self.sigmoid.backward(error_tensor[i])
            previous_error = self.fc_output.backward(previous_error)
            # previous_error = previous_error[:, :-1]                                   # remove the bias from tensor
            previous_error += self.current_hidden_error                               # BP of copy
            previous_error = self.tanh.backward(previous_error)
            previous_error = self.fc_hidden.backward(previous_error)

            # Unpack all gradient.
            gradient_weights = self.fc_hidden.gradient_weights
            gradient_output = self.fc_output.gradient_weights

            # Accumulate the gradients.
            accumulated_weights_gradient += gradient_weights
            accumulated_output_gradient += gradient_output

            # Save gradient w.r.t. ht for the computations for sample t-1 (we need it for BP for copy function).
            self.current_hidden_error = previous_error[:, :self.hidden_size]

            # Save error tensor.
            previous_error_tensor[i, :] = previous_error[:, self.hidden_size:self.hidden_size + self.input_size]

        self.gradient_weights = accumulated_weights_gradient
        self.fc_hidden.gradient_weights = accumulated_weights_gradient
        self.gradient_weights_output = accumulated_output_gradient
        self.fc_output.gradient_weights = accumulated_output_gradient

        # Update weights in FC layers.

        if self.optimizer:
            updated_weight_tensor = self.optimizer.calculate_update(self.fc_hidden.weights, accumulated_weights_gradient)
            self.weights = updated_weight_tensor
            self.fc_hidden.weights = updated_weight_tensor
            updated_weight_tensor = self.optimizer.calculate_update(self.fc_output.weights, accumulated_output_gradient)
            self.fc_output.weights = updated_weight_tensor
            self.weights_output = updated_weight_tensor

        return previous_error_tensor
