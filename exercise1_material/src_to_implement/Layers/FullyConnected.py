from . Base import *
import numpy as np


class FullyConnected(BaseLayer):
    """
    Layer that performs linear operation on its input. It represents backbone for Neural Network framework.

    Attributes:
        trainable(bool): Flag implying if the layer will be trained.
        _optimizer(Sgd): Optimizer for weights tensor updating. It is a protected member of the class.
        input_tensor(np.ndarray): Tensor with features of the whole batch. Its size is batch_size * (input_size + 1)
        because of the added bias
        gradient_tensor(np.ndarray): Tensor with calculated gradient of the layer.
        weights(np.ndarray): Tensor of weights for this layer.
    """
    def __init__(self, input_size, output_size):
        super().__init__()
        self.trainable = True
        # bias is included in the weights, so the size is increased by 1
        self.weights = None
        self._optimizer = None
        self.input_tensor = None
        self.gradient_tensor = None

    def forward(self, input_tensor):
        """
        Calculates the output of the layer in the forward pass, by multiplying input tensor and weights of the layer.

        Args:
            input_tensor(np.ndarray): Tensor with inputs for the layer.

        Returns:
            np.ndarray: Output tensor of the layer.
        """
        # We add one column of ones in the input tensor to multiply them with biases in weights tensor.
        bias = np.ones(shape=(input_tensor.shape[0], 1))
        self.input_tensor = np.append(input_tensor, bias, axis=1)
        output_tensor = np.dot(self.input_tensor, self.weights)
        return output_tensor

    def backward(self, error_tensor):
        """
        Calculates the error for the previous layer and the gradient, in the backward pass.

        Args:
            error_tensor(np.ndarray): Tensor with errors the layer after this one generates.

        Returns:
            np.ndarray: Tensor with errors of the layer before this one.
        """
        # We calculate the error tensor for the previous error.
        previous_error_tensor = np.dot(error_tensor, self.weights[:-1, :].T)
        # We calculate the gradient of the layer and remove added column for biases.
        gradient_tensor = np.dot(self.input_tensor.transpose(), error_tensor)
        self.gradient_tensor = gradient_tensor
        # If we have the optimizer, we update the weights.
        if self.optimizer:
            updated_weight_tensor = self.optimizer.calculate_update(self.weights, gradient_tensor)
            self.weights = updated_weight_tensor
        return previous_error_tensor

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
    def gradient_weights(self):
        """
        Getter for attribute gradient_weights.

        Returns:
            np.ndarray: Tensor with calculated gradient of the layer.
        """
        # TODO check if this makes sense
        return self.gradient_tensor

    def initialize(self, weights_initializer, bias_initializer):
        weights = weights_initializer.initialize()
        bias = bias_initializer.initialize()
        self.weights = np.vstack(weights, bias)

