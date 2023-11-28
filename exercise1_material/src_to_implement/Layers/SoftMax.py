import numpy as np
from . Base import BaseLayer


class SoftMax(BaseLayer):
    """
    Layer implementing SoftMax activation function, that transforms output of the network to a probability distribution.

    Attributes:
        trainable(bool): Flag implying if the layer will be trained.
        output_tensor(np.ndarray): Tensor with the computed probabilities.
    """
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        """
        Calculates the output of the layer in the forward pass, as a probability for each class.

        Args:
            input_tensor(np.ndarray): Tensor with inputs for the layer.

        Returns:
            np.ndarray: Output tensor of the layer.
        """
        # We shift our input by the mean, to increase numerical stability, as exp(input) can be very large.
        input_tensor = input_tensor - np.max(input_tensor, axis=1).reshape(-1, 1)
        # We divide the exponent of each input, by the sum of all exponents, and get probabilities for all classes.
        exp_tensor = np.exp(input_tensor)
        sums = np.sum(exp_tensor, axis=1)
        sums = np.tile(sums, (exp_tensor.shape[1], 1)).T
        probabilities = exp_tensor / sums
        self.output_tensor = probabilities
        return probabilities

    def backward(self, error_tensor):
        """
        Calculates the error for the previous layer, in the backward pass.

        Args:
            error_tensor(np.ndarray): Tensor with errors the layer after this one generates.

        Returns:
            np.ndarray: Tensor with errors of the layer before this one.
        """
        # We multiply generated output tensor with difference between error tensor and sum of error * prediction,
        # for all the classes.
        inner_sum = np.sum(error_tensor*self.output_tensor, axis=1, keepdims=True)
        previous_error_tensor = self.output_tensor * (error_tensor-inner_sum)
        return previous_error_tensor
