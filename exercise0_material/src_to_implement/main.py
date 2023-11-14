from pattern import Checker, Circle, Spectrum
from generator import ImageGenerator


def main():

    # Tests to check if pattern functions work
    checker = Checker(12, 3)
    checker.draw()
    checker.show()

    circle = Circle(1000, 300, (350, 450))
    circle.draw()
    circle.show()

    spectrum = Spectrum(500)
    spectrum.draw()
    spectrum.show()
    file_path = "./data/exercise_data"
    label_path = "./data/labels.json"
    generator = ImageGenerator(file_path, label_path, 10, (300, 300))
    generator.show()


if __name__ == '__main__':
    main()
