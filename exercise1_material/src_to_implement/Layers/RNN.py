import numpy as np
from . Base import BaseLayer
from . FullyConnected import FullyConnected
from . Sigmoid import Sigmoid
from . TanH import TanH
import sys
import copy
sys.path.append('C:/Users/Admin/Documents/GitHub/DeepLearningFAU')
from exercise1_material.src_to_implement.Optimization.Optimizers import Adam, Sgd


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
        self.weights_output = np.random.uniform(0, 1, size=(self.output_size + 1, self.hidden_size))
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

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memorize):
        self._memorize = memorize

    @property
    def gradient_weights(self):
        return self.fc_hidden.gradient_weights

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
            # input_hidden = np.append(input_hidden, np.ones((input_hidden.shape[0], 1)), axis=1)
            # TODO: Mozda treba izbaciti posl liniju!!

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

        accumulated_weights_gradient = 0
        accumulated_output_gradient = 0

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
            gradient_weights = self.gradient_weights
            gradient_output = self.fc_output.gradient_weights

            # Accumulate the gradients.
            accumulated_weights_gradient += gradient_weights
            accumulated_output_gradient += gradient_output

            # Save gradient w.r.t. ht for the computations for sample t-1 (we need it for BP for copy function).
            self.current_hidden_error = previous_error[:, :self.hidden_size]

            # Save error tensor.
            previous_error_tensor[i, :] = previous_error[:, self.hidden_size:self.hidden_size + self.input_size]

        # Update weights in FC layers.
        optimizer_hidden = Sgd(1e-3)
        self.fc_hidden.optimizer = optimizer_hidden
        self.fc_hidden.optimize(accumulated_weights_gradient)
        self.fc_hidden.optimizer = None

        optimizer_output = Sgd(1e-3)
        self.fc_output.optimizer = optimizer_output
        self.fc_output.optimize(accumulated_output_gradient)
        self.fc_output.optimizer = None

        # Set weights of RNN to the updated values.
        self.weights = self.fc_hidden.weights
        self.weights_output = self.fc_output.weights

        return previous_error_tensor
