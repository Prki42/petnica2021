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


def relu(x: int) -> float:
    if x > 0:
        return x
    return 0


def relu_derivative(x: int) -> float:
    if x > 0:
        return 1
    return 0
