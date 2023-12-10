from .Base import BaseLayer
import numpy as np
from scipy.signal import correlate
import copy


class Conv(BaseLayer):
    def __init__(self, stride_shape, convolution_shape, num_kernels):
        super().__init__()
        self.stride_shape = stride_shape
        self.convolution_shape = convolution_shape
        self.num_kernels = num_kernels
        self.trainable = True
        self.weights = np.random.uniform(0, 1, (num_kernels, *convolution_shape))
        self.bias = np.random.uniform(0, 1, num_kernels)
        self._gradient_weights = None
        self._gradient_bias = None
        self.padding_shape = None
        self._weights_optimizer = None
        self._bias_optimizer = None

    @property
    def optimizer(self):
        return self._weights_optimizer, self._bias_optimizer

    @optimizer.setter
    def optimizer(self, optimizer):
        self._weights_optimizer = copy.deepcopy(optimizer)
        self._bias_optimizer = copy.deepcopy(optimizer)

    @property
    def gradient_weights(self):
        return self._gradient_weights

    @property
    def gradient_bias(self):
        return self._gradient_bias

    def forward1(self, input_tensor):
        # input tensor dimensions: B x C x Y x X
        # convolution shape = C x M x N
        # stride shape = single val or tuple

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
        padding_y = int((self.stride_shape[0] * (y - 1) + convolution_shape[1] - y) / 2)
        padding_x = int((self.stride_shape[1] * (x - 1) + convolution_shape[2] - x) / 2)

        # Calculate output dimensions
        y_output = int((y + 2 * padding_y - self.convolution_shape[1]) / self.stride_shape[0] + 1)
        x_output = int((x + 2 * padding_x - self.convolution_shape[2]) / self.stride_shape[1] + 1)

        output_shape = (b, self.num_kernels, y_output, x_output)

        self.padding_shape = [padding_y, padding_x]
        # todo make this work for even number kernel size
        additional_padding_y = 0
        additional_padding_x = 0
        if self.convolution_shape[1]%2==0:
            additional_padding_y = 1
        if self.convolution_shape[2]%2==0:
            additional_padding_x = 1
        padded_input = np.pad(input_tensor, ((0, 0), (0, 0), (self.padding_shape[0], self.padding_shape[0] + additional_padding_y),(self.padding_shape[1], self.padding_shape[1] + additional_padding_x)), mode='constant')
        output = np.zeros(output_shape)
        for i in range(b):
            for j in range(self.num_kernels):
                output[i, j, :, :] = correlate(padded_input[i, :, :, :], self.weights[j, :, :, :],
                                               mode='valid')[:,::self.stride_shape[0],::self.stride_shape[1]] + self.bias[j]
        return output

    def forward(self, input_tensor):
        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, axis=3)
        b, c, y, x = input_tensor.shape
        if len(self.convolution_shape) == 2:
            convolution_shape = np.expand_dims(self.convolution_shape, axis=2)
        else:
            convolution_shape = self.convolution_shape
        if len(self.stride_shape) == 1:
            self.stride_shape = (self.stride_shape,)

        output_tensor = np.zeros([b, self.num_kernels, y, x])

        for i in range(b):
            for k in range(self.num_kernels):
                for channel in range(c):
                    curr_input_tensor = input_tensor[i, channel]
                    curr_weights = self.weights[k, channel]
                    curr_output_tensor = output_tensor[i,k,:,:]
                    output_tensor[i, k, :, :] = correlate(input_tensor[i, channel, :, :], self.weights[k, channel, :, :], mode="same", method="direct")
        return output_tensor

    def initialize(self, weight_initializer, bias_initializer):
        self.weights = weight_initializer.initialize(self.weights.shape, self.convolution_shape[0], self.num_kernels)
        self.bias = bias_initializer.initialize(self.bias.shape, self.convolution_shape[0], self.num_kernels)

    def backward(self, error_tensor):
        pass
