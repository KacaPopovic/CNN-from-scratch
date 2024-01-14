import numpy as np

from . Base import BaseLayer
from . Helpers import compute_bn_gradients

class BatchNormalization(BaseLayer):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        self.trainable = True
        self.weights = np.ones(self.channels)
        self.bias = np.zeros(self.channels)
        self.mean = None
        self.variance = None
        self.normalized_tensor = None
        self.input_tensor = None
        self._optimizer = None
        self.moving_mean = None
        self.moving_variance = None
        self.alpha = 0.8
        self.original_shape = None
        self.gradient_bias = None
        self.gradient_weights = None


    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.channels
        fan_out = self.channels
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in,fan_out)

    @property
    def optimizer(self):
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._optimizer = optimizer

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        reshape_flag = False
        if input_tensor.ndim == 4:
            input_tensor = self.reformat(input_tensor)
            reshape_flag = True
        self.mean = np.mean(input_tensor, axis=0)
        self.variance = np.var(input_tensor, axis=0)
        if not self.testing_phase:
            if self.moving_mean is None:
                self.moving_mean = self.mean
                self.moving_variance = self.variance
            self.moving_mean = self.alpha * self.moving_mean + (1 - self.alpha) * self.mean
            self.moving_variance = self.alpha * self.moving_variance + ( 1 - self.alpha) * self.variance
            self.normalized_tensor = (input_tensor - self.mean) / np.sqrt(self.variance + np.finfo(float).eps)
        else:
            self.normalized_tensor = (input_tensor - self.moving_mean) / np.sqrt(self.moving_variance + np.finfo(float).eps)
        # Scale and shift
        output_tensor = self.weights * self.normalized_tensor + self.bias
        if reshape_flag:
            output_tensor = self.reformat(output_tensor)
        return output_tensor



    def backward(self, error_tensor):
        reshape_flag = False
        if error_tensor.ndim == 4:
            error_tensor = self.reformat(error_tensor)
            input_tensor = self.reformat(self.input_tensor)
            reshape_flag = True
        else:
            input_tensor = self.input_tensor
        grad_bias = np.sum(error_tensor, axis=0)
        self.gradient_bias = grad_bias
        grad_weights = np.sum(error_tensor*self.normalized_tensor, axis=0)
        self.gradient_weights = grad_weights
        if self.optimizer:
            updated_weights = self.optimizer.calculate_update(self.weights, grad_weights)
            updated_bias = self.optimizer.calculate_update(self.bias, grad_bias)
            self.weights = updated_weights
            self.bias = updated_bias
        grad_input = compute_bn_gradients(error_tensor, input_tensor, self.weights, self.mean, self.variance)
        if reshape_flag:
            grad_input = self.reformat(grad_input)
        return grad_input


    def reformat(self, input_tensor):
        if input_tensor.ndim == 4:
            self.original_shape = input_tensor.shape
            b,h,m,n = input_tensor.shape
            reshaped_tensor = np.reshape(input_tensor, newshape=(b,h,m*n))
            reshaped_tensor = reshaped_tensor.transpose(0,2,1)
            output_tensor = reshaped_tensor.reshape(b*m*n,h)
            return output_tensor
        elif input_tensor.ndim == 2:
            b,h,m,n = self.original_shape
            reshaped_tensor = input_tensor.reshape(b,m*n, h)
            reshaped_tensor = reshaped_tensor.transpose(0,2,1)
            output_tensor = reshaped_tensor.reshape(b,h,m,n)
            return output_tensor
        else:
            return input_tensor




