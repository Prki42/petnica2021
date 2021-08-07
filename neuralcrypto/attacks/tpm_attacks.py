from typing import Any, Dict
from neuralcrypto.models.tpm import TPM
import numpy as np


def naive_attack(synch_result: Dict[str, Any]) -> Dict[str, Any]:
    k, n, l, learning = synch_result["params"]
    eve = TPM(k, n, l)
    res = {"success": False}
    for iteration, X in enumerate(synch_result["X"]):
        tauE = eve(X)
        tauA = synch_result["tauA"][iteration]
        tauB = synch_result["tauB"][iteration]
        if tauA == tauB and tauA == tauE:
            if learning == "hebbian":
                eve.hebbian_learning(tauA, X)
            else:
                eve.anti_hebbian_learning(tauA, X)

    if np.array_equal(synch_result["W"], eve.W):
        res["success"] = True
    return res


def geometric_attack(synch_result: Dict[str, Any]) -> Dict[str, Any]:
    k, n, l, learning = synch_result["params"]
    eve = TPM(k, n, l)
    res = {"success": False}
    for iteration, X in enumerate(synch_result["X"]):
        tauE = eve(X)
        tauA = synch_result["tauA"][iteration]
        tauB = synch_result["tauB"][iteration]
        # Skip iterations when A and B stay unchanged
        if tauA != tauB:
            continue
        if tauA != tauE:
            min_sigma_index = np.argmin(np.abs(eve.h))
            # Invert sigma with smallest pre-activated value
            eve.sigma[min_sigma_index] *= -1
            eve.tau *= -1
        # Now E should be able to make coordinated move with A
        if learning == "hebbian":
            eve.hebbian_learning(tauA, X)
        else:
            eve.anti_hebbian_learning(tauA, X)

    # Check if attack was successful
    if np.array_equal(synch_result["W"], eve.W):
        res["success"] = True
    return res


def majority_attack(
    synch_result: Dict[str, Any], population_size: int
) -> Dict[str, Any]:
    k, n, l, learning = synch_result["params"]
    eve = TPM(k, n, l)
    res = {"success_count": 0, "success": False}

    population = [[TPM(k, n, l), 0] for _ in range(population_size)]

    for i in range(len(synch_result["X"])):
        X = synch_result["X"][i]
        tauA = synch_result["tauA"][i]
        tauB = synch_result["tauB"][i]
        # Skip iterations when A and B stay unchanged
        if tauA != tauB:
            continue
        if i % 2 == 0:
            # Geometric atttack
            for idx in range(len(population)):
                eve = population[idx][0]
                tauE = eve.forward(X)
                if tauA != tauE:
                    min_sigma_index = np.argmin(np.abs(eve.h))
                    eve.sigma[min_sigma_index] *= -1
                    eve.tau *= -1
                if learning == "hebbian":
                    eve.hebbian_learning(tauA, X)
                else:
                    eve.anti_hebbian_learning(tauA, X)
        else:
            # Majority attack + geometric atttack
            configs = {}
            # Get map of all hidden layer configurations
            for idx in range(len(population)):
                eve = population[idx][0]
                tauE = eve.forward(X)
                if tauA != tauE:
                    min_sigma_index = np.argmin(np.abs(eve.h))
                    eve.sigma[min_sigma_index] *= -1
                    eve.tau *= -1
                config = "|".join([str(s) for s in eve.sigma])
                try:
                    configs[config] += 1
                except KeyError:
                    configs[config] = 1

            # Extract configuration which majority of networks use
            configs = sorted(configs.items(), key=lambda x: x[1], reverse=True)
            config = [int(c) for c in configs[0][0].split("|")]

            # Apply it to every network and do coordinated move
            for idx in range(len(population)):
                eve = population[idx][0]
                eve.sigma = np.array(config)
                if learning == "hebbian":
                    eve.hebbian_learning(tauA, X)
                else:
                    eve.anti_hebbian_learning(tauA, X)

    # Diagnose population after attack
    for p in population:
        if np.array_equal(p[0].W, synch_result["W"]):
            res["success_count"] += 1
    if res["success_count"] > 0:
        res["success"] = True
    return res
