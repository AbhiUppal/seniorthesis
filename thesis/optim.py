import numpy as np
import pandas as pd
import scipy as sp
import time

from math import pi
from typing import Callable


def likelihood_t(A: np.array_equiv, B: np.array_equiv, x: np.array_equiv, mu: Callable):
    if x.shape[0] == 1:
        x = x.T
    n = len(x)
    BBT = np.matmul(B, B.T)
    Ax = np.matmul(A, x)

    prod1 = np.matmul((x - mu(Ax)).T, BBT.inv)
    prod2 = np.matmul(prod1, x - mu(Ax))
    L = n * np.log(2 * pi) + np.log(np.linalg.det(BBT)) + prod2

    return L

