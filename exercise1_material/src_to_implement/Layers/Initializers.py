import numpy as np


class Constant:
    def __init__(self, initialization_value=0.1):
        self.initialization_value = initialization_value

    def initialize(self, weights_shape, fan_in, fan_out):
        init_weights = np.full(weights_shape,self.initialization_value)
        return init_weights


class UniformRandom:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        init_weights = np.random.uniform(0, 1, weights_shape)
        return init_weights


class Xavier:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        std = np.sqrt(2/(fan_in + fan_out))
        init_weights = np.random.normal(0, std, weights_shape)
        return init_weights


class He:
    def __init__(self):
        pass

    def initialize(self, weights_shape, fan_in, fan_out):
        std = np.sqrt(2 / fan_in)
        init_weights = np.random.normal(0, std, weights_shape)
        return init_weights
