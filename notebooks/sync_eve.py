# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.3
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# # Eve synchronization

# +
import os
import sys

module_path = ""
try:
    module_path = os.path.join(globals()["_dh"][0], "..")
except KeyError:
    module_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".."
    )
module_path = os.path.abspath(module_path)
if module_path not in sys.path:
    sys.path.append(module_path)

import matplotlib.pyplot as plt


# %matplotlib inline
# %load_ext lab_black


# +
import numpy as np
from neuralcrypto.models.tpm import TPM, synched_percent
from typing import Tuple


def synchronize_eve(
    a: TPM, b: TPM, e: TPM, learning: str = "hebbian", maxIters: int = 5000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    progress = np.zeros(shape=(maxIters), dtype=float)
    progress_eve = np.zeros(shape=(maxIters), dtype=float)
    num = 0
    k, n, l = a.get_properties()
    for i in range(maxIters):
        progress[i] = synched_percent(a, b)
        progress_eve[i] = synched_percent(a, e)
        if np.array_equal(a.W, b.W):
            break
        num += 1
        X = np.random.choice([-1, 1], size=(k, n))
        tauA = a(X)
        tauB = b(X)
        _ = e(X)
        if learning == "anti_hebbian":
            a.anti_hebbian_learning(tauB, X)
            b.anti_hebbian_learning(tauA, X)
            e.anti_hebbian_learning(tauA, X)
        elif learning == "random_walk":
            a.random_walk(tauB, X)
            b.random_walk(tauA, X)
            e.random_walk(tauA, X)
        else:
            a.hebbian_learning(tauB, X)
            b.hebbian_learning(tauA, X)
            e.hebbian_learning(tauA, X)
    progress = progress[0 : num + 1]
    progress_eve = progress_eve[0 : num + 1]
    return (progress, progress_eve, a.W, e.W)


# +
from multiprocessing import Pool, cpu_count
from random import randint


k = 3
n = 101
l = 3
learning_rule = "hebbian"

iter_count = 1000

p = Pool(cpu_count())
results = p.starmap(
    synchronize_eve,
    iter(
        [
            (TPM(k, n, l), TPM(k, n, l), TPM(k, n, l), learning_rule, 5000)
            for _ in range(iter_count)
        ]
    ),
)

synch_count = len([[1] for p in results if p[1][-1] == 100])

print(
    f"Eve synch: {synch_count}/{iter_count} - {(synch_count/iter_count):.3f}%"
)

# +
progress, progress_eve, _, _ = results[randint(0, len(results))]

steps = list(range(1, len(progress) + 1))

plt.plot(steps, progress, label="sinh a i b")
plt.plot(steps, progress_eve, label="ucenje e od a")
plt.xlabel("iteracija")
plt.ylabel("% sinhronizacije")
plt.title(f"K={k}, N={n}, L={l}; {learning_rule}")
plt.legend()
plt.show()

# +
eve_weights = [r[3] for r in results]
eve_histograms = [np.histogram(w, 2 * l + 1)[0] for w in eve_weights]
eve_average_hist = np.sum(eve_histograms, axis=0) / (iter_count * k * n) * 100

print(eve_average_hist)
print(sum(eve_average_hist))
# -

plt.bar(np.arange(-k, k + 1), eve_average_hist, color="red")
plt.title("Prosecna distribucija vrednosti tezista kod trece strane")
plt.ylabel("Ucestalost (%)")
plt.show()
