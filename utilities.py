import numpy as np
from numpy.random import rand


def tanh(x: np.float64):
    return np.tanh(x)


def sigmoid(x: np.float64):
    return 1/(1 + np.exp(-x))


def sigmoid_grad(x: np.float64):
    return x * (1.0 - x)


def tanh_grad(x: np.float64):
    return 1 - x ** 2


def initalize(dim: tuple, init_range: float):
    return rand(*dim) * init_range


def zeros(*dim: int):
    return np.zeros(dim)


def ones(*dim: int):
    return np.ones(dim)


def generate_data(length: int, maximum_array_length: int):
    """
    length: number of samples in your dataset, should be > 1000

    maximum_array_length: dataset will contain arrays with length
    [2, max_array_length]
    """
    X = []
    y = []
    for _ in range(length):
        size = np.random.randint(2, maximum_array_length + 1)
        array = np.random.random_integers(1, 10, size)

        original_array = list(array)
        sorted_array = sorted(original_array)

        X.append(original_array)
        y.append(sorted_array)

    return X, y
