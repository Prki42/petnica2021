import numpy as np
from typing import Tuple
from neuralcrypto.helpers.math import vsgn, vg, vtheta


class TPM:
    def __init__(self, k: int, n: int, l: int):
        self.k = k
        self.n = n
        self.l = l
        # W (k, n)
        self.W = np.random.randint(low=-l, high=l + 1, size=(k, n))
        # sigma (k, 1)
        self.sigma = np.ones(shape=(k, 1))
        self.tau: int = None

    def get_properties(self) -> Tuple[int, int, int]:
        return (self.k, self.n, self.l)

    def forward(self, X: np.ndarray) -> int:
        self.sigma = vsgn(np.sum(self.W * X, axis=1))
        self.tau = np.prod(self.sigma)
        return self.tau

    def __call__(self, X: np.ndarray) -> int:
        return self.forward(X)

    def hebbian_learning(self, tau_b: int, X: np.ndarray) -> None:
        if self.tau != tau_b:
            return
        k, _, l = self.get_properties()
        self.W = vg(
            self.W
            + (self.sigma * X.T).T
            * vtheta(self.sigma, self.tau).reshape(k, 1),
            l,
        )

    def anti_hebbian_learning(self, tau_b: int, X: np.ndarray) -> None:
        if self.tau != tau_b:
            return
        k, _, l = self.get_properties()
        self.W = vg(
            self.W
            - (self.sigma * X.T).T
            * vtheta(self.sigma, self.tau).reshape(k, 1),
            l,
        )

    def random_walk(self, tau_b: int, X: np.ndarray) -> None:
        if self.tau != tau_b:
            return
        k, _, l = self.get_properties()
        self.W = vg(
            self.W
            + (self.sigma * X.T).T
            * vtheta(self.sigma, self.tau).reshape(k, 1),
            l,
        )


def synchronize(
    a: TPM,
    b: TPM,
    learning: str = "hebbian",
    maxIters: int = 1000,
    return_weights: bool = False,
) -> np.ndarray:
    progress = np.zeros(shape=(maxIters), dtype=float)
    num = 0
    k, n, l = a.get_properties()
    for i in range(maxIters):
        progress[i] = synched_percent(a, b)
        if np.array_equal(a.W, b.W):
            break
        num += 1
        X = np.random.choice([-1, 1], size=(k, n))
        tauA = a(X)
        tauB = b(X)
        if learning == "anti_hebbian":
            a.anti_hebbian_learning(tauB, X)
            b.anti_hebbian_learning(tauA, X)
        elif learning == "random_walk":
            a.random_walk(tauB, X)
            b.random_walk(tauA, X)
        else:
            a.hebbian_learning(tauB, X)
            b.hebbian_learning(tauA, X)
    progress = progress[0 : num + 1]
    if return_weights:
        return a.W
    return progress


def synched_percent(a: TPM, b: TPM) -> float:
    k, n, l = a.get_properties()
    percent = (
        1.0
        - np.mean(np.abs(a.W.reshape(k * n) - b.W.reshape(k * n)) / (2 * l))
    ) * 100
    return percent
