from .Base import BaseLayer

class Flatten(BaseLayer):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape
        batch_size, width, height, channel_number = input_tensor.shape
        input_tesnor = input_tensor.reshape(batch_size, width*height*channel_number)
        return input_tesnor

    def backward(self, error_tensor):
        error_tensor = error_tensor.reshape(self.input_shape)
        return error_tensor
