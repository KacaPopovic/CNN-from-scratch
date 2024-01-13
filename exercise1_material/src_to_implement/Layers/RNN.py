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
        self.hidden_state = np.zeros(shape=(1,hidden_size))
        self._memorize = False
        self._weights = np.random.uniform(0, 1, size=(self.hidden_size + self.input_size + 1, self.hidden_size))
        self._weights_output = np.random.uniform(0, 1, size=(self.output_size + 1, self.hidden_size))
        self.fc_hidden = FullyConnected(self.hidden_size + self.input_size + 1, self.hidden_size)
        self.fc_output = FullyConnected(self.hidden_size + 1, self.output_size)
        self.tanh = TanH()
        self.sigmoid = Sigmoid()
        self.input_tensor = None
        self.current_hidden_error = np.zeros(shape=(1,self.hidden_size))
        self.trainable = True
        self.tanh_activations = None
        self.sigmoid_activations = None

    def initialize(self, weight_initializer, bias_initializer):
        fan_in = self.hidden_size + self.input_size + 1
        fan_out = self.hidden_size
        self.weights = weight_initializer.initialize(self.weights.shape, fan_in, fan_out)

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
        return self._weights

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
        self.sigmoid_activations = np.zeros(shape=(self.input_tensor.shape[0], self.output_size))
        self.tanh_activations = np.zeros(shape=(self.input_tensor.shape[0], self.hidden_size))
        output_tensor = np.zeros(shape=(self.input_tensor.shape[0], self.output_size))
        if not self.memorize:
            #hidden_state = np.zeros(shape=(1,self.hidden_size))
            self.hidden_state = np.zeros(shape=(1,self.hidden_size))
        for i in range(self.input_tensor.shape[0]):
            current_input_tensor = input_tensor[i,:]
            current_input_tensor = np.transpose(current_input_tensor.reshape(-1,1))
            input_hidden = np.concatenate((self.hidden_state, current_input_tensor), axis=1)
            input_hidden = np.append(input_hidden, np.ones((input_hidden.shape[0], 1)), axis=1)
            #input_hidden = np.transpose(input_hidden.reshape(-1, 1))
            hidden_state = self.tanh.forward(self.fc_hidden.forward(input_hidden))
            self.tanh_activations[i,:] = hidden_state
            self.hidden_state = hidden_state
            input_output = np.append(self.hidden_state,np.ones((input_hidden.shape[0], 1)), axis=1)
            #input_output = np.transpose(input_output.reshape(-1, 1))
            output_sigmoid = self.sigmoid.forward(self.fc_output.forward(input_output))
            output_tensor[i, :] = output_sigmoid
            self.sigmoid_activations[i,:] = output_sigmoid
        return output_tensor

    def backward(self, error_tensor):
        accumulated_weights_gradient = 0
        accumulated_output_gradient = 0
        previous_error_tensor_complete = np.zeros_like(self.input_tensor)
        self.current_hidden_error = 0
        for i in range(self.input_tensor.shape[0]-1,0, -1):
            current_activation = self.tanh_activations[i,:]
            current_activation = np.transpose(current_activation.reshape(-1,1))
            self.tanh.activations = current_activation

            current_activation = self.sigmoid_activations[i, :]
            current_activation = np.transpose(current_activation.reshape(-1, 1))
            self.sigmoid.activations = current_activation

            # Propagate error tensor
            previous_error_tensor = self.sigmoid.backward(error_tensor[i])
            previous_error_tensor = self.fc_output.backward(previous_error_tensor)
            #BP copy - remove bias, add current hidden gradient
            previous_error_tensor = previous_error_tensor[:,:-1]
            previous_error_tensor += self.current_hidden_error #this should be tensor 1xhidden_size
            previous_error_tensor = self.tanh.backward(previous_error_tensor)
            previous_error_tensor = self.fc_hidden.backward(previous_error_tensor)

            # Calculate gradients
            gradient_weights = self.gradient_weights
            gradient_output = self.fc_output.gradient_weights

            accumulated_weights_gradient += gradient_weights
            accumulated_output_gradient += gradient_output

            previous_error_tensor_complete[i,:] = previous_error_tensor[:,self.hidden_size:self.hidden_size+self.input_size]
            self.current_hidden_error = previous_error_tensor[:, 0:self.hidden_size]

        # Optimizing fully connected layers using Adam optimizer and acumulated errors
        optimizer_output = Adam(1e-3, 0.9, 0.999)
        optimizer_hidden = Adam(1e-3, 0.9, 0.999)
        self.fc_output.optimizer = optimizer_output
        self.fc_hidden.optimizer = optimizer_hidden
        self.fc_output.optimize(accumulated_output_gradient)
        self.fc_hidden.optimize(accumulated_weights_gradient)
        self.fc_output.optimizer = None
        self.fc_hidden.optimizer = None

        # Setting weights of RNN
        self.weights = self.fc_hidden.weights
        self.weights_output = self.fc_output.weights
        return previous_error_tensor_complete



