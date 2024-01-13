import numpy as np
from . Base import BaseLayer
from . FullyConnected import FullyConnected
from . Sigmoid import Sigmoid
from . TanH import TanH
import sys

# Add the absolute path of the 'DeepLearningFAU' directory to sys.path
sys.path.append('C:/Users/Admin/Documents/GitHub/DeepLearningFAU')

# Now you can import the Adam optimizer
from exercise1_material.src_to_implement.Optimization.Optimizers import Adam


class RNN(BaseLayer):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.hidden_state = np.zeros(shape=hidden_size)
        self._memorize = False
        self._weights = np.random.uniform(0, 1, size=(self.hidden_size + self.input_size + 1, self.hidden_size))
        self._weights_output = np.random.uniform(0, 1, size=(self.output_size + 1, self.hidden_size))
        self.fc_hidden = None
        self.fc_output = None
        self.tanh = None
        self.sigmoid = None
        self.input_tensor = None
        self.current_hidden_gradient = 0
        self.create_embedded_layers()

    @property
    def memorize(self):
        return self._memorize

    @memorize.setter
    def memorize(self, memory):
        self._memorize = memory

    @property
    def gradient_weights(self):
        return self.fc_hidden.gradient_weights

    @property
    def weights(self):
        return self.weights

    @weights.setter
    def weights(self, weight):
        self._weights = weight

    @property
    def weights_output(self):
        return self._weights_output

    @weights_output.setter
    def weights_output(self, weight):
        self._weights_output = weight

    def create_embedded_layers(self):
        self.fc_hidden = FullyConnected(self.hidden_size + self.input_size + 1, self.hidden_size)
        self.fc_output = FullyConnected(self.hidden_size + 1, self.output_size)
        self.tanh = TanH()
        self.sigmoid = Sigmoid()

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        output_tensor = np.zeros(shape=(self.input_tensor.shape[0], self.output_size))
        if not self.memorize:
            self.hidden_state = np.zeros(self.hidden_size)
        for i in range(self.input_tensor.shape[0]):

            input_hidden = np.concatenate((self.hidden_state, input_tensor[i,:]), axis=0)
            input_hidden = np.append(input_hidden,1)
            input_hidden = np.transpose(input_hidden.reshape(-1, 1))
            hidden_state = self.tanh.forward(self.fc_hidden.forward(input_hidden))
            self.hidden_state = hidden_state
            input_output = np.append(self.hidden_state,1)
            output_tensor[i, :] = self.sigmoid.forward(self.fc_output.forward(input_output))
        return output_tensor

    def backward(self, error_tensor):
        accumulated_weights_gradient = 0
        accumulated_error = 0
        accumulated_output_gradient = 0
        for i in range(self.input_tensor.shape[0]-1,0):

            # Propagate error tensor
            previous_error_tensor = self.sigmoid.backward(error_tensor[i])
            previous_error_tensor = self.fc_output.backward(previous_error_tensor)
            #BP copy
            previous_error_tensor += self.current_hidden_gradient
            previous_error_tensor = self.tanh.backward(previous_error_tensor)
            previous_error_tensor = self.fc_hidden.backward(previous_error_tensor)

            # Calculate gradients
            gradient_weights = self.gradient_weights
            hh_gradient = gradient_weights[0:self.hidden_size]

            gradient_output = self.fc_output.gradient_weights

            self.current_hidden_gradient = hh_gradient
            accumulated_weights_gradient += gradient_weights
            accumulated_output_gradient += gradient_output
            accumulated_error += previous_error_tensor

        # Optimizing fully connected layers using Adam optimizer and acumulated errors
        optimizer_output = Adam(1e-3, 0.9, 0.999)
        optimizer_hidden = Adam(1e-3, 0.9, 0.999)
        self.fc_output.optimizer(optimizer_output)
        self.fc_hidden.optimizer(optimizer_hidden)
        self.fc_output.optimize(accumulated_output_gradient)
        self.fc_hidden.optimize(accumulated_weights_gradient)
        self.fc_output.optimizer(None)
        self.fc_hidden.optimizer(None)

        # Setting weights of RNN
        self.weights(self.fc_hidden.weights)
        self.weights_output(self.fc_output.weights)
        return accumulated_error



