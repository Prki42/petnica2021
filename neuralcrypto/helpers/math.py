import numpy as np


def theta(a: int, b: int) -> int:
    return 0 if a == b else 1


vtheta = np.vectorize(theta)


def sgn(x: int) -> int:
    return 1 if x > 0 else -1


vsgn = np.vectorize(sgn)


def g(x: int, l: int) -> int:
    if x < -l:
        return -l
    if x > l:
        return l
    return x


vg = np.vectorize(g)


def relu(x: float) -> float:
    if x > 0:
        return x
    return 0


v_relu = np.vectorize(relu)


def relu_derivative(x: float) -> float:
    if x > 0:
        return 1
    return 0


v_relu_derivative = np.vectorize(relu_derivative)


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


v_sigmoid = np.vectorize(sigmoid)


def sigmoid_derivative(x: float) -> float:
    return sigmoid(x) * (1.0 - sigmoid(x))


v_sigmoid_derivative = np.vectorize(sigmoid_derivative)


def custom_activation(x: int) -> int:
    return 1 if x > 0 else -1


v_custom_activation = np.vectorize(custom_activation)
