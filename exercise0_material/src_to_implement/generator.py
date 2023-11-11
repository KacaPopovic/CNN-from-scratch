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
        self.index = -1
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
        # resizing is done here
        images = []
        for image_path in self.image_paths[start:end]:
            image = np.load(image_path)
            image = scipy.misc.imresize(image, self.image_size[0:2])
            image = self.augment(image)
            images.append(image)
        return np.array(images)

    def shuffle_data(self):
        keys = list(self.labels.keys())
        np.random.shuffle(keys)
        shuffled_labels = {key: self.labels[key] for key in keys}
        self.labels = shuffled_labels
        self.image_paths = self.load_image_paths()

    def next(self):
        start = self.index
        end = min(self.index + self.batch_size, len(self.labels))

        if self.index == 0:
            self.epoch += 1
            self.shuffle_data()

        images = self.load_images(start, end)
        values = list(self.labels.values())
        labels = np.array(values[start:end])

        if end < self.index + self.batch_size:
            self.shuffle_data()
            self.epoch += 1
            start = 0
            end = self.batch_size + self.index - len(self.labels)
            images = np.append(images, self.load_images(start, end), axis=0)
            values = list(self.labels.values())
            labels = np.append(labels, np.array(values[start:end]), axis=0)

        self.index = (self.index + self.batch_size) % len(self.labels)
        return images, labels

    def augment(self,img):
        if self.mirroring:
            flipping_methods = ['none', 'horizontal', 'vertical', 'both']
            chosen_method = np.random.choice(flipping_methods)
            if chosen_method == 'horizontal':
                img = np.fliplr(img)
            elif chosen_method == 'vertical':
                img = np.flipud(img)
            elif chosen_method == 'both':
                img = np.flip(img, (0, 1))
            elif chosen_method == 'none':
                pass
        if self.rotation:
            rotations_number = np.random.randint(0, 4)
            img = np.rot90(img, rotations_number)

        return img

    def current_epoch(self):
        return self.epoch

    def class_name(self, x):
        return self.class_dict[x]

    def show(self):
        images, labels = self.next()
        image_number = len(images)
        row_number = np.ceil(np.sqrt(image_number))
        col_number = image_number // row_number

        plt.figure()
        for i in range(row_number):
            for j in range(col_number):
                count = i*row_number + j
                plt.subplot(i, j, count)
                plt.imshow(images[count])
                title = self.class_name(labels[count])
                plt.title(title)
        plt.show()
