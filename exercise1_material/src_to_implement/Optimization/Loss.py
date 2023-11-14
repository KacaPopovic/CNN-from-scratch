import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.prediction_tensor = None
        self.loss = None

    def forward(self, prediction_tensor, label_tensor):
        self.prediction_tensor = prediction_tensor
        eps = np.finfo(float).eps

        labels_indexes = np.argmax(label_tensor, axis=1)
        selected_probs = prediction_tensor[np.arange(prediction_tensor.shape[0]), labels_indexes]
        loss = - np.sum(np.log(selected_probs + eps))
        self.loss = loss

        # todo check if you need to return it
        return loss


    def backward(self, label_tensor):
        eps = np.finfo(float).eps
        prev_error_tensor = -label_tensor/(self.prediction_tensor+eps)
        return prev_error_tensor
