import numpy as np
from typing import Dict, Tuple
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
        # h (k, 1)
        self.h = np.ones(shape=(k, 1))
        self.tau: int = None

    def get_properties(self) -> Tuple[int, int, int]:
        return (self.k, self.n, self.l)

    def forward(self, X: np.ndarray) -> int:
        # h - used as the "confidence factor"
        self.h = np.sum(self.W * X, axis=1)
        self.sigma = vsgn(self.h)
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


class TPMChromosome:
    def __init__(self, *args) -> None:
        self.network: TPM = None
        args_len = len(args)
        if args_len == 2:
            if isinstance(args[0], TPM) and isinstance(args[1], TPM):
                self.network = TPM(args[0].k, args[0].n, args[0].l)
                self.crossover(args[0], args[1])
            else:
                raise Exception("los ulaz")
        elif args_len == 3:
            if False not in [isinstance(args[i], int) for i in range(3)]:
                self.network = TPM(args[0], args[1], args[2])
            else:
                Exception("los ulaz")
        else:
            raise Exception("los ulaz")

    def mutate(self):
        # TODO: mozda
        pass

    def crossover(self, a: TPM, b: TPM):
        hs = np.concatenate((a.h, b.h), axis=None)
        weights = np.concatenate((a.W, b.W), axis=0)
        sorted = [(hs[i], weights[i]) for i in range(2 * a.k)]
        sorted.sort(key=lambda x: x[0])
        self.network.W = np.array([s[1] for s in sorted[0 : a.k]]).reshape(
            (a.k, a.n)
        )
        self.mutate()


def synchronize(
    a: TPM,
    b: TPM,
    learning: str = "hebbian",
    maxIters: int = 1000,
) -> Dict[str, np.ndarray]:
    if learning not in ["hebbian", "anti_hebbian"]:
        raise Exception(f"{learning} ???")

    num = 0
    k, n, l = a.get_properties()
    result = {}
    result["parameters"] = (a.k, a.n, a.l, learning)
    result["progress"] = []
    result["X"] = []
    result["tauA"] = []
    result["tauB"] = []

    for _ in range(maxIters):
        result["progress"].append(synched_percent(a, b))
        if np.array_equal(a.W, b.W):
            break
        num += 1
        X = np.random.choice([-1, 1], size=(k, n))
        result["X"].append(X)
        tauA = a(X)
        tauB = b(X)
        result["tauA"].append(tauA)
        result["tauB"].append(tauB)
        if learning == "anti_hebbian":
            a.anti_hebbian_learning(tauB, X)
            b.anti_hebbian_learning(tauA, X)
        else:
            a.hebbian_learning(tauB, X)
            b.hebbian_learning(tauA, X)

    result["W"] = a.W
    result["progress"] = np.array(result["progress"])
    result["X"] = np.array(result["X"])
    result["tauA"] = np.array(result["tauA"])
    result["tauB"] = np.array(result["tauB"])

    return result


def synched_percent(a: TPM, b: TPM) -> float:
    k, n, l = a.get_properties()
    percent = (
        1.0
        - np.mean(np.abs(a.W.reshape(k * n) - b.W.reshape(k * n)) / (2 * l))
    ) * 100
    return percent


def synched_percent_numpy(
    a: np.ndarray, b: np.ndarray, k: int, n: int, l: int
) -> float:
    percent = (
        1.0 - np.mean(np.abs(a.reshape(k * n) - b.reshape(k * n)) / (2 * l))
    ) * 100
    return percent
