import copy

class NeuralNetwork:
    def __init__(self, optimizer):
        self.optimizer = optimizer
        self.loss = None
        self.layers = None
        self.data_layer = None
        self.loss_layer = None
        self.label_tensor = None

    def forward(self):
        input_tensor, label_tensor = self.data_layer.next()
        self.label_tensor = label_tensor
        for i in range(self.layers.len()-1):
            input_tensor = self.layers[i].forward(input_tensor)
        loss = self.layers[-1].forward(input_tensor, label_tensor)
        output_tensor = input_tensor
        return loss

    def backward(self):
        error_tensor = self.layers[-1].backward(self.label_tensor)
        for i in range(self.layers.len()-2,-1):
            error_tensor = self.layers[i].backward(error_tensor)
        return error_tensor

    def append_layer(self, layer):
        if layer.trainable:
            optimizor_copy = copy.deepcopy(self.optimizer)
            layer.optimizer(optimizor_copy)
        self.layers.append(layer)

    def train(self, iterations):
        for i in range(iterations):
            loss = self.forward()
            self.loss.append(loss)

    def test(self, input_tensor):
        for i in range(self.layers.len()-1):
            input_tensor = self.layers[i].forward(input_tensor)
        probabilities = input_tensor
        return probabilities


