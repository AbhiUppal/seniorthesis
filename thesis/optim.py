import numpy as np
import pandas as pd
import time

from math import pi
from typing import Callable

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


def likelihood_t(A: np.array_equiv, B: np.array_equiv, x: np.array_equiv, mu: Callable):
    if x.shape[0] == 1:
        x = x.T
    n = len(x)
    BBT = np.matmul(B, B.T)
    Ax = np.matmul(A, x)

    prod1 = np.matmul((x - mu(Ax)).T, np.linalg.inv(BBT))
    prod2 = np.matmul(prod1, x - mu(Ax))
    L = n * np.log(2 * pi) + np.log(np.linalg.det(BBT)) + prod2

    return L


if __name__ == "__main__":
    AB = np.load("data/testdata/AB_1.npz")
    A = AB["A"]
    B = AB["B"]
    data = pd.read_csv("data/testdata/data.csv")

    x = np.array(data.iloc[1, 1:]).T

    print(likelihood_t(A, B, x, mu=np.tanh))
