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
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] tags=[]
# # Petnica 2021
# -

# Formatiranje koda

# %load_ext lab_black

# Dodavanje putanje za lokalne module

# +
import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)
# -

# Test sinhronizacije

# +
from neuralcrypto.models.tpm import TPM
from neuralcrypto.models.tpm import synchronize

k = 3
n = 101
l = 3

s = 0
it = 100
for i in range(it):
    a = TPM(k, n, l)
    b = TPM(k, n, l)
    progress = synchronize(a, b)
    s += progress.shape[0]
s /= it

print(f"Prosecan broj iteracija u {it} ponavljanja: {s:.3f}")
