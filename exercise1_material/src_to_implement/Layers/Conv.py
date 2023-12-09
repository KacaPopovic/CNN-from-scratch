from .Base import BaseLayer
import numpy as np
from scipy.signal import correlate


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape[1:]))
        self.bias = np.random.uniform(0, 1, num_kernels)
        self._gradient_weights = None
        self._gradient_bias = None
        self.padding_shape = None
        self._weights_optimizer = None
        self._bias_optimizer = None

    @property
    def gradient_weights(self):
        return self.gradient_weights

    @property
    def gradient_bias(self):
        return self.gradient_bias

    def forward(self, input_tensor):

        # TODO proveri jel ovo sme
        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, axis=3)
        b, c, y, x = input_tensor.shape
        if len(self.convolution_shape) == 2:
            convolution_shape = np.expand_dims(self.convolution_shape, axis=2)
        else:
            convolution_shape = self.convolution_shape
        if len(self.stride_shape) == 1:
            self.stride_shape = (self.stride_shape,)
        padding_y = ((y-1)*self.stride_shape[0]+self.convolution_shape[1] - y) / 2
        padding_x = ((x-1)*self.stride_shape[1]+self.convolution_shape[2]-x)/2
        self.padding_shape = [padding_y, padding_x]
        y_output = np.floor((y + 2*self.padding_shape[0] - convolution_shape[1]) / self.stride_shape[0] + 1)
        x_output = np.floor((x + 2*self.padding_shape[1] - convolution_shape[2]) / self.stride_shape[1] + 1)
        output_shape = [b, self.num_kernels, y_output, x_output]
        # todo make this work for even number kernel size
        padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (self.padding_shape[0], self.padding_shape[0]), (self.padding_shape[1], self.padding_shape[1])), mode='constant')
        output = np.zeros(output_shape)
        for i in range(b):
            for j in range(self.num_kernels):
                output[i,j,:,:] = correlate(padded_input[i,:,:,:], self.weights[j,:,:,:], mode='valid') + self.bias[j]




    def backward(self, error_tensor):
        pass
