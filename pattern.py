import copy
import numpy as np
import matplotlib.pyplot as plt


class Checker():
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        black_tiles = np.zeros(shape=(self.tile_size, self.tile_size))
        white_tiles = np.ones(shape=(self.tile_size, self.tile_size))
        tile = np.concatenate((black_tiles, white_tiles), axis=0)
        tile = np.concatenate((tile, np.flip(tile)), axis=1)

        tile_number = int(self.resolution / self.tile_size/2)
        tile = np.tile(tile, (tile_number, tile_number))
        self.output = tile

        return copy.deepcopy(tile)

    def show(self):
        plt.figure()
        plt.imshow(self.output, cmap='gray')
        plt.title('Checkerboard')
        plt.show()


class Circle():
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        self.output = np.zeros(shape=(self.resolution, self.resolution))

        x_grid, y_grid = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
        x_grid_centered = x_grid - self.position[0]
        y_grid_centered = y_grid - self.position[1]

        distances = np.sqrt(x_grid_centered**2 + y_grid_centered**2)
        self.output[distances <= self.radius] = 1

        return copy.deepcopy(self.output)

    def show(self):
        plt.figure()
        plt.imshow(self.output, cmap='gray')
        plt.title('Circle')
        plt.show()


class Spectrum():
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        self.output = np.zeros(shape=(self.resolution, self.resolution, 3))
        # CH0 -> RED, CH1 -> GREEN, CH3 -> BLUE

        space = np.linspace(0, 1, self.resolution)
        red_channel = np.tile(space, (self.resolution, 1))
        blue_channel = np.flip(red_channel, axis=1)
        green_channel = np.rot90(blue_channel)

        self.output[:, :, 0] = red_channel
        self.output[:, :, 1] = green_channel
        self.output[:, :, 2] = blue_channel

        return copy.deepcopy(self.output)

    def show(self):
        plt.figure()
        plt.imshow(self.output)
        plt.title('Spectrum')
        plt.show()
