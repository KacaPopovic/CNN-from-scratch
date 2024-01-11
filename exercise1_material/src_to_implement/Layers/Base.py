class BaseLayer:
    """
    Base class that will be inherited by every layer in our neural Network framework.

    Attributes:
        trainable(bool): Flag implying if the layer will be trained.
        weights(np.ndarray): Tensor of weights for this layer.
        testing_phase(bool): Flag implying if we are in testing or training phasr
    """

    def __init__(self):
        self.trainable = False
        self.weights = None
        self.testing_phase = False
