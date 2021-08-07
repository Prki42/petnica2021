from multiprocessing import Pool, cpu_count
from typing import Dict, List
from neuralcrypto.models.tpm import TPM, synchronize
import numpy as np


def simulate_tpm(
    number_of_sims: int,
    k: int,
    n: int,
    l: int,
    learning_rule: str,
    max_iters: int = 2000,
) -> List[Dict[str, np.ndarray]]:
    p = Pool(cpu_count())
    return p.starmap(
        synchronize,
        iter(
            [
                (TPM(k, n, l), TPM(k, n, l), learning_rule, max_iters)
                for _ in range(number_of_sims)
            ]
        ),
    )
