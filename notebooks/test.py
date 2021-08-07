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

# + [markdown] tags=[]
# # Test
# -

# Formatiranje koda

# %load_ext lab_black

# +
import matplotlib.pyplot as plt

# %matplotlib inline
# -

# Dodavanje putanje za lokalne module

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
# -

# ## Distribucija broja iteracija do sinhronizacije

# +
from neuralcrypto.models.tpm import TPM
from neuralcrypto.models.tpm import synchronize
from multiprocessing import Pool
import numpy as np
import time

k = 3
n = 1001
l = 5

it = 100
learning_rule = "hebbian"

start_time = time.perf_counter()
p = Pool(4)
results = p.starmap(
    synchronize,
    iter(
        [(TPM(k, n, l), TPM(k, n, l), learning_rule, 10000) for _ in range(it)]
    ),
)
elapsed = time.perf_counter() - start_time

print(f"Vreme: {elapsed:.3f}s")

# +
s = np.array([x["X"].shape[0] for x in results])

plt.hist(
    s,
    color="blue",
    edgecolor="black",
    bins=30,
)
plt.xlabel("Brot iteracija")
plt.title(f"K={k}, N={n}, L={l}; {learning_rule}")
plt.show()
# -

worst_index = np.argmax(s)
print(np.max(s))
print(worst_index)
progress = results[worst_index]["progress"]
plt.plot(list(range(1, len(progress) + 1)), progress)
plt.xlabel("iteracija")
plt.ylabel("% sinhronizacije")
plt.title("Najsporija sinhronizacija")
plt.show()
