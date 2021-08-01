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

# # Synch weights histogram

# +
# %load_ext lab_black
import matplotlib.pyplot as plt

# %matplotlib inline

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

# +
from neuralcrypto.models.tpm import TPM
from neuralcrypto.models.tpm import synchronize
from multiprocessing import Pool, cpu_count
import numpy as np

k = 3
n = 101
l = 3
learning_rule = "hebbian"

it_count = 1000

p = Pool(cpu_count())
weights = p.starmap(
    synchronize,
    iter(
        [
            (TPM(k, n, l), TPM(k, n, l), learning_rule, 5000, True)
            for _ in range(it_count)
        ]
    ),
)

histograms = [np.histogram(w, 2 * l + 1)[0] for w in weights]
average_hist = np.sum(histograms, axis=0) / (it_count * k * n) * 100
# -

plt.bar(np.arange(-k, k + 1), average_hist, color="blue")
plt.title(f"Prosecna distribucija vrednosti tezista\nK={k}, N={n}, L={l}")
plt.ylabel("Ucestalost (%)")
plt.show()

# +
from random import randint

plt.bar(
    np.arange(-k, k + 1),
    histograms[randint(0, it_count - 1)] / (k * n) * 100,
    color="blue",
)
plt.title(
    f"Distribucija vrednosti tezistia u random slucaju\nK={k}, N={n}, L={l}"
)
plt.ylabel("Ucestalost (%)")
plt.show()
