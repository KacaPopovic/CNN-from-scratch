import os.path
import json
import scipy.misc
import numpy as np
import matplotlib.pyplot as plt

# In this exercise task you will implement an image generator. Generator objects in python are defined as having a next function.
# This next function returns the next generated object. In our case it returns the input of a neural network each time it gets called.
# This input consists of a batch of images and its corresponding labels.

class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):

        # Also depending on the size of your data-set you can consider loading all images into memory here already.
        # The labels are stored in json format and can be directly loaded as dictionary.
        # Note that the file names correspond to the dicts of the label dictionary.
        # file_path - to the directionary of images
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.epoch = 0
        self.index = 0
        self.labels = self.load_labels()
        self.image_paths = self.load_image_paths()

    def load_labels(self):
        with open(self.label_path, 'r') as f:
            labels = json.load(f)
        return labels

    def load_image_paths(self):
        data = []
        for filename, label in self.labels.items():
            image_path = os.path.join(self.file_path, f"{filename}.npy")
            data.append(image_path)
        return np.array(data)

    def load_images(self, start, end):
        #if resizing is specified, it is done here
        images = [scipy.misc.imresize(np.load(image_path), self.image_size[0:2]) for image_path in self.image_paths[start:end]]
        return np.array(images)

    def next(self):
        # This function creates a batch of images and corresponding labels and returns them.
        # In this context a "batch" of images just means a bunch, say 10 images that are forwarded at once.
        # Note that your amount of total data might not be divisible without remainder with the batch_size.
        # Think about how to handle such cases
        start = self.index
        end = min(self.index + self.batch_size, len(self.labels))
        images = self.load_images(start, end)
        labels = np.array([self.labels[str(index)] for index in range(start, end)])

        if end < self.index+self.batch_size:
            #Todo call shuffle
            self.epoch += 1
            start = 0
            end = self.batch_size + self.index - len(self.labels)
            images = np.append(images, self.load_images(start, end), axis=0)
            labels = np.append(labels, np.array([self.labels[str(index)] for index in range(start, end)]))

        self.index = (self.index + self.batch_size) % len(self.labels)
        return images, labels

    def augment(self,img):
        # this function takes a single image as an input and performs a random transformation
        # (mirroring and/or rotation) on it and outputs the transformed image
        #TODO: implement augmentation function

        return img

    def current_epoch(self):
        # return the current epoch number
        return 0

    def class_name(self, x):
        # This function returns the class name for a specific input
        #TODO: implement class name function
        return
    def show(self):
        # In order to verify that the generator creates batches as required, this functions calls next to get a
        # batch of images and labels and visualizes it.
        #TODO: implement show method
        pass

