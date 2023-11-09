import copy
import numpy as np
import matplotlib.pyplot as plt


class Checker:

    """
    A class for generating and displaying a checkerboard pattern.

    Args:
        resolution (int): The resolution of the checkerboard pattern.
        tile_size (int): The size of each individual tile in the checkerboard.

    Attributes:
        resolution (int): The resolution of the checkerboard pattern.
        tile_size (int): The size of each individual tile in the checkerboard.
        output (numpy.ndarray): The generated checkerboard pattern.

    Methods:
        draw():
            Generates a checkerboard pattern based on the provided resolution and tile size.

        show():
            Displays the generated checkerboard pattern using matplotlib.

    Example:
        # Initialize Checker object with resolution 800 and tile size 20
        checkerboard = Checker(resolution=800, tile_size=20)
        # Generate the checkerboard pattern
        checkerboard.draw()
        # Display the checkerboard pattern
        checkerboard.show()
    """
    def __init__(self, resolution, tile_size):
        self.resolution = resolution
        self.tile_size = tile_size
        self.output = None

    def draw(self):
        """
        Generates a checkerboard pattern based on the provided resolution and tile size.

        Returns:
            numpy.ndarray: The generated checkerboard pattern.

        Raises:
            None
        """
        black_tiles = np.zeros(shape=(self.tile_size, self.tile_size))
        white_tiles = np.ones(shape=(self.tile_size, self.tile_size))
        tile = np.concatenate((black_tiles, white_tiles), axis=0)
        tile = np.concatenate((tile, np.flip(tile)), axis=1)

        tile_number = int(self.resolution / self.tile_size/2)
        tile = np.tile(tile, (tile_number, tile_number))
        self.output = tile

        return copy.deepcopy(tile)

    def show(self):
        """
        Displays the generated checkerboard pattern using matplotlib.

        Returns:
            None

        Raises:
            None
        """
        plt.figure()
        plt.imshow(self.output, cmap='gray')
        plt.title('Checkerboard')
        plt.show()


class Circle:

    """
    A class for generating and displaying a filled circle on a grid.

    Args:
        resolution (int): The resolution of the grid (both width and height).
        radius (float): The radius of the circle.
        position (tuple): The position (x, y) of the center of the circle in the grid.

    Attributes:
        resolution (int): The resolution of the grid (both width and height).
        radius (float): The radius of the circle.
        position (tuple): The position (x, y) of the center of the circle in the grid.
        output (numpy.ndarray): The grid with the filled circle drawn on it.

    Methods:
        draw():
            Draws a filled circle on the grid based on the provided radius and position.

        show():
            Displays the grid with the filled circle using matplotlib.

    Example:
        # Initialize Circle object with resolution 800, radius 50, and position (400, 400)
        circle = Circle(resolution=800, radius=50, position=(400, 400))
        # Draw the filled circle on the grid
        circle.draw()
        # Display the grid with the filled circle
        circle.show()

    """
    def __init__(self, resolution, radius, position):
        self.resolution = resolution
        self.radius = radius
        self.position = position
        self.output = None

    def draw(self):
        """
        Draws a filled circle on the grid based on the provided radius and position.

        Returns:
            numpy.ndarray: The grid with the filled circle drawn on it.

        Raises:
            None
        """
        self.output = np.zeros(shape=(self.resolution, self.resolution))

        x_grid, y_grid = np.meshgrid(np.arange(self.resolution), np.arange(self.resolution))
        x_grid_centered = x_grid - self.position[0]
        y_grid_centered = y_grid - self.position[1]

        distances = np.sqrt(x_grid_centered**2 + y_grid_centered**2)
        self.output[distances <= self.radius] = 1

        return copy.deepcopy(self.output)

    def show(self):
        """
        Displays the grid with the filled circle using matplotlib.

        Returns:
            None

        Raises:
            None
        """
        plt.figure()
        plt.imshow(self.output, cmap='gray')
        plt.title('Circle')
        plt.show()


class Spectrum:

    """
    A class for generating and displaying a color spectrum image.

    Args:
        resolution (int): The resolution of the spectrum image (both width and height).

    Attributes:
        resolution (int): The resolution of the spectrum image (both width and height).
        output (numpy.ndarray): The color spectrum image represented as a NumPy array.

    Methods:
        draw():
            Generates a color spectrum image based on the provided resolution.

        show():
            Displays the color spectrum image using matplotlib.

    Example:
        # Initialize Spectrum object with resolution 800
        spectrum = Spectrum(resolution=800)
        # Generate the color spectrum image
        spectrum.draw()
        # Display the color spectrum image
        spectrum.show()
    """
    def __init__(self, resolution):
        self.resolution = resolution
        self.output = None

    def draw(self):
        """
        Generates a color spectrum image based on the provided resolution.

        Returns:
            numpy.ndarray: The color spectrum image represented as a NumPy array.

        Raises:
            None
        """
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

        """
        Displays the color spectrum image using matplotlib.

        Returns:
            None

        Raises:
            None
        """
        plt.figure()
        plt.imshow(self.output)
        plt.title('Spectrum')
        plt.show()
