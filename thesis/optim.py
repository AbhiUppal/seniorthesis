import numpy as np
import pandas as pd
import time

from math import pi
from typing import Callable
from utils import timer

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources


def likelihood_t(
    A: np.array_equiv,
    B: np.array_equiv,
    data: pd.DataFrame,
    t: int,
    mu: Callable = np.tanh,
    cache: dict = None,
    debug: bool = False,
) -> float:
    """likelihood_t Computes the likelhihood function at a single point in time. This is intended to be summed over

    Parameters
    ----------
    A : np.array_equiv
        Adjacency matrix describing the system
    B : np.array_equiv
        Covariance matrix describing the system
    data : pd.DataFrame
        Data to get x_t and x_{t+1} from
    t : int
        Time to evaluate the likelihood function at. Must be at least two less than the length of the data
    mu : Callable
        Function from the positive real numbers to [0, 1] such as np.tanh
    debug : bool, optional
        Whether or not to print extra debug information, by default False

    Returns
    -------
    float
        log probability evaluated at this point in time
    """
    # x_now = x_t
    # x_next = x_{t+1}

    x_now = data.iloc[t, 1:]
    x_next = data.iloc[t + 1, 1:]
    x_now = x_now.T if x_now.shape[0] == 1 else x_now
    x_next = x_next.T if x_next.shape[0] == 1 else x_next

    n = len(x_now)
    Ax_now = np.matmul(A, x_now)

    if cache is None:
        BBT = np.matmul(B, B.T)
        prod1 = np.matmul((x_next - mu(Ax_now)).T, np.linalg.inv(BBT))
        prod2 = np.matmul(prod1, x_next - mu(Ax_now))

        L = -0.5 * (n * np.log(2 * pi) + np.log(np.linalg.det(BBT)) + prod2)

    else:
        prod1 = np.matmul((x_next - mu(Ax_now)).T, cache["BBT_inv"])
        prod2 = np.matmul(prod1, x_next - mu(Ax_now))

        L = -0.5 * (n * np.log(2 * pi) + cache["BBT_log_det"] + prod2)

    if debug:
        print(n * np.log(2 * pi))
        print(np.log(np.linalg.det(BBT)))
        print(prod2)

    return L


@timer
def likelihood(
    data,
    A: np.array_equiv,
    B: np.array_equiv,
    mu: Callable,
    debug: bool = False,
    t0: int = None,
    T: int = None,
):
    t0 = 0 if t0 is None else t0
    T = len(data) if T is None else T

    BBT = np.matmul(B, B.T)
    BBT_inv = np.linalg.inv(BBT)
    BBT_log_det = np.log(np.linalg.det(BBT))

    cache = {  # Values we want to look up instead of calculating multiple times
        "BBT": BBT,
        "BBT_inv": BBT_inv,
        "BBT_log_det": BBT_log_det,
    }

    timerange = range(t0, T - 1)
    likelihoods = [
        likelihood_t(A=A, B=B, data=data, t=t, mu=mu, cache=cache) for t in timerange
    ]

    return sum(likelihoods)

def main():
    AB = np.load("data/testdata/AB_1.npz")
    A = AB["A"]
    B = AB["B"]
    data = pd.read_csv("data/testdata/data.csv")

    test = likelihood(data=data, A=A, B=B, mu=np.tanh, t0=0, T=1000)
    print(test)


if __name__ == "__main__":
    main()
