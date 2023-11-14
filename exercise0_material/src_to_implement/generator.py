import os.path
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


class ImageGenerator:
    def __init__(self, file_path, label_path, batch_size, image_size, rotation=False, mirroring=False, shuffle=False):
        """
        Initializes an ImageGenerator object.

        Args:
            file_path (str): The path to the directory containing image data files.
            label_path (str): The path to the file containing image labels json.
            batch_size (int): The size of each batch during data generation.
            image_size (tuple): A tuple representing the desired size of the images (height, width).
            rotation (bool): Whether to apply random rotation to the images.
            mirroring (bool): Whether to apply random mirroring (flipping) to the images.
            shuffle (bool): Whether to shuffle the data during each epoch.
        """
        self.class_dict = {0: 'airplane', 1: 'automobile', 2: 'bird', 3: 'cat', 4: 'deer', 5: 'dog', 6: 'frog',
                           7: 'horse', 8: 'ship', 9: 'truck'}
        self.file_path = file_path
        self.label_path = label_path
        self.batch_size = batch_size
        self.image_size = image_size
        self.rotation = rotation
        self.mirroring = mirroring
        self.shuffle = shuffle
        self.labels = self.load_labels()
        self.image_paths = self.load_image_paths()
        self.epoch = -1
        self.index = 0

    def load_labels(self):
        """
        Loads image labels from the json file located in label_path

        Returns:
            dict: A dictionary mapping image names with their class
        """
        with open(self.label_path, 'r') as f:
            labels = json.load(f)
        return labels

    def load_image_paths(self):
        """
        Creates a numpy array of file paths for the images located in file_path based on the loaded labels.

        Returns:
            numpy.ndarray: An array containing file paths for the images.
        """
        data = []
        for filename, label in self.labels.items():
            image_path = os.path.join(self.file_path, f"{filename}.npy")
            data.append(image_path)
        return np.array(data)

    def load_images(self, start, end):
        """
        Loads, resizes and preprocesses images within the specified range. Function augment is
        called here and it does all the augmentation that was specified in constructor.

        Args:
            start (int): The starting index for loading images.
            end (int): The ending index for loading images.

        Returns:
            numpy.ndarray: An array containing the loaded and preprocessed images.
        """
        images = []
        for image_path in self.image_paths[start:end]:
            image = np.load(image_path)
            # Resizing is done here
            image = np.array(Image.fromarray(image).resize((self.image_size[0], self.image_size[1])))
            # Preprocessing is done in fuction augment
            image = self.augment(image)
            images.append(image)
        return np.array(images)

    def shuffle_data(self):
        """
            Shuffles the data by randomly rearranging the order of labels and updating image paths.
        """
        if self.shuffle:
            keys = list(self.labels.keys())
            np.random.shuffle(keys)
            shuffled_labels = {key: self.labels[key] for key in keys}
            self.labels = shuffled_labels
            self.image_paths = self.load_image_paths()

    def next(self):
        """
        Generates the next batch of images and labels.

        Returns:
            tuple: A tuple containing images (numpy.ndarray) and labels (numpy.ndarray).
        """

        # Defining beginning and end of the batch. Takes into account the posibility
        # of the end of the epoch
        start = self.index
        end = min(self.index + self.batch_size, len(self.labels))

        # Updating the epoch if it happened at the end of the batch
        if start == 0:
            self.epoch += 1
            self.shuffle_data()

        # Loading necessary data
        images = self.load_images(start, end)
        values = list(self.labels.values())
        labels = np.array(values[start:end])

        # Shuffling and updating when end of epoch happens in the middle of the batch
        if (self.index + self.batch_size) > len(self.labels):
            self.epoch += 1
            self.shuffle_data()
            start = 0
            end = self.index + self.batch_size - len(self.labels)
            images = np.append(images, self.load_images(start, end), axis=0)
            values = list(self.labels.values())
            labels = np.append(labels, np.array(values[start:end]), axis=0)

        self.index = (self.index + self.batch_size) % len(self.labels)

        return images, labels

    def augment(self, image):
        """
        Applies data augmentation to the given image. Which augmentation is done
        depends on the flags in contructor. Possible options are mirroring and rotation

        Args:
            image (numpy.ndarray): The input image.

        Returns:
            numpy.ndarray: The augmented image.
        """
        if self.mirroring:
            flipping_methods = ['none', 'horizontal', 'vertical', 'both']
            chosen_method = np.random.choice(flipping_methods)
            if chosen_method == 'horizontal':
                image = np.fliplr(image)
            elif chosen_method == 'vertical':
                image = np.flipud(image)
            elif chosen_method == 'both':
                image = np.flip(image, (0, 1))
            elif chosen_method == 'none':
                pass
        if self.rotation:
            rotation_number = np.random.randint(0, 4)
            image = np.rot90(image, rotation_number)

        return image

    def current_epoch(self):
        """
        Returns the current epoch number.

        Returns:
            int: The current epoch number.
        """
        return self.epoch

    def class_name(self, x):
        """
        Maps class index to class name.

        Args:
            x (int): The class index.

        Returns:
            str: The corresponding class name.
        """
        return self.class_dict[x]

    def show(self):
        """
        Displays a grid of images and their corresponding labels for visualization.
        """
        images, labels = self.next()
        image_number = len(images)
        row_number =int( np.ceil(np.sqrt(image_number)))
        col_number = int(image_number // row_number)
        plt.figure()
        count = 1
        for i in range(row_number):
            for j in range(col_number):
                plt.subplot(row_number, col_number, count)
                plt.imshow(images[count])
                title = self.class_name(labels[count])
                plt.title(title)
                count += 1
        plt.tight_layout()
        plt.show()
