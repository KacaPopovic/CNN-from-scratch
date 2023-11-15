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
        #temp = np.diagflat(self.output_tensor) - self.output_tensor.T
        #jacobian_matrix = np.dot(self.output_tensor,temp)
        enum = np.arange(error_tensor.shape[0])
        z= np.zeros_like(error_tensor)

        for i in enum:
            x=error_tensor[i,:]
            y= self.output_tensor[i,:].T
            z[i,:] = x*y
        prev_error_tensor = self.output_tensor * (error_tensor - z)
        return prev_error_tensor
