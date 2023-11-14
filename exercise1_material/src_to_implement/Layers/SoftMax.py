import numpy as np
from . Base import BaseLayer


class SoftMax(BaseLayer):
    def __init__(self):
        super().__init__()
        self.output_tensor = None

    def forward(self, input_tensor):
        input_tensor = input_tensor - np.max(input_tensor, axis=1).reshape(-1, 1)
        exp_tensor = np.exp(input_tensor)
        sums = np.sum(exp_tensor, axis=1)
        sums = np.tile(sums, (exp_tensor.shape[1], 1)).T
        probabilities = exp_tensor / sums
        self.output_tensor = probabilities
        return probabilities

    def backward(self, error_tensor):

        # Todo check if this makes sense
        jacobian_matrix = self.output_tensor * (np.eye(self.output_tensor.shape[1]) - self.output_tensor.T)
        prev_error_tensor = np.dot(error_tensor, jacobian_matrix)
        return prev_error_tensor
