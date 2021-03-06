import numpy as np
import pandas as pd
import time

from math import pi
from multiprocessing import cpu_count, Pool
from typing import Callable, Union
from utils import timer

try:
    import importlib.resources as pkg_resources
except ImportError:
    import importlib_resources as pkg_resources

# -------------------------
# Functions and derivatives
# -------------------------


def sigmoid(x: Union[int, float, np.array_equiv]):
    return 1 / (1 + np.exp(-x))


def sigmoid_deriv(x: Union[int, float, np.array_equiv]):
    return sigmoid(x) * (1 - sigmoid(x))


def tanh_deriv(x: Union[int, float, np.array_equiv]):
    return 1 - np.tanh(np.tanh(x))


# -----------------------------
# Callables for Multiprocessing
# -----------------------------


class CopierShell:
    def __init__(
        self,
        A: np.array_equiv,
        B: np.array_equiv,
        data: pd.DataFrame,
        t0: int = 0,
        T: int = None,
        mu: Callable = np.tanh,
        cache: dict = None,
    ):
        self.A = A
        self.B = B
        self.data = data
        self.t0 = t0
        self.T = T
        self.mu = mu
        self.cache = cache


class CopierGradA(CopierShell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, ij: tuple):
        return gradient_ij(
            A=self.A,
            B=self.B,
            data=self.data,
            ij=ij,
            t0=self.t0,
            T=self.T,
            mu=self.mu,
            cache=self.cache,
        )["A"]


class CopierGradB(CopierShell):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __call__(self, ij: tuple):
        return gradient_ij(
            A=self.A,
            B=self.B,
            data=self.data,
            ij=ij,
            t0=self.t0,
            T=self.T,
            mu=self.mu,
            cache=self.cache,
        )["B"]


# -----------------------
# Likelihood Calculations
# -----------------------


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


# ---------------------
# Gradient Calculations
# ---------------------


def gradient_ij(
    A: np.array_equiv,
    B: np.array_equiv,
    data: pd.DataFrame,
    ij: tuple,
    t0: int = 0,
    T: int = None,
    mu: Callable = np.tanh,
    cache: dict = None,
) -> float:

    T = len(data) - 1 if T is None else T
    t0 = 0 if t0 is None else t0

    i = ij[0]
    j = ij[1]

    derivative = cache.get("derivative", None)
    BBT_inv = cache.get("BBT_inv", None)
    B_inv = cache.get("B_inv", None)
    BT_inv = cache.get("BT_inv", None)

    # Caclulating the gradients
    A_grad_ij = 0
    B_grad_ij = 0
    n = A.shape[0]
    for t in range(t0, T - 1):
        # Terms to pre-compute
        if len(data.iloc[t, :]) > n:
            x_now = data.iloc[t, 1:]
            x_next = data.iloc[t + 1, 1:]
        else:
            x_now = data.iloc[t, :]
            x_next = data.iloc[t + 1, :]
        Ax = np.matmul(A, x_now)
        predicted_term = x_next - mu(Ax)  # x_{t+1} - \mu(Ax_t)

        # Calculate A gradient
        grad_A_ij_t = (
            x_now[i] * derivative(Ax)[j] * np.matmul(BBT_inv, predicted_term)[j]
        )

        A_grad_ij += grad_A_ij_t

        # Calculate B gradient
        grad_B_ij_t = 0
        for l in range(n):
            for k in range(n):
                grad_B_ij_t += (
                    predicted_term[k]
                    * predicted_term[l]
                    * (BBT_inv[k, i] * B_inv[j, l] + BBT_inv[i, l] * BT_inv[k, j])
                )
        B_grad_ij += grad_B_ij_t

    B_grad_ij = -0.5 * B_grad_ij - T * B[i, j] * B[j, j]
    A_grad_ij = -A_grad_ij

    # Calculating the B gradient

    grad = {"A": A_grad_ij, "B": B_grad_ij}

    return grad


@timer
def gradient(
    A: np.array_equiv,
    B: np.array_equiv,
    data: pd.DataFrame,
    t0: int = 0,
    T: int = None,
    mu: Callable = np.tanh,
    parallel: bool = False,
) -> np.array_equiv:

    assert mu in [np.tanh, sigmoid], "mu must be either np.tanh or sigmoid"

    t0 = 0 if t0 is None else t0
    T = len(data) if T is None else T

    BBT = np.matmul(B, B.T)
    B_inv = np.linalg.inv(B)
    BT_inv = np.linalg.inv(B.T)
    BBT_inv = np.linalg.inv(BBT)
    BBT_log_det = np.log(np.linalg.det(BBT))

    shape = A.shape

    derivative = tanh_deriv if mu == np.tanh else sigmoid_deriv

    cache = {  # Values we want to look up instead of calculating multiple times
        "BBT": BBT,
        "B_inv": B_inv,
        "BT_inv": BT_inv,
        "BBT_inv": BBT_inv,
        "BBT_log_det": BBT_log_det,
        "derivative": derivative,
    }

    grad_a = np.zeros(shape)
    grad_b = np.zeros(shape)

    if not parallel:
        for i in range(shape[0]):
            for j in range(shape[1]):
                grad_a[i, j] = gradient_ij(
                    A=A, B=B, data=data, ij=(i, j), t0=t0, T=T, cache=cache
                )["A"]
                grad_b[i, j] = gradient_ij(
                    A=A, B=B, data=data, ij=(i, j), t0=t0, T=T, cache=cache
                )["B"]
    else:
        n = shape[0]
        indices = [(i, j) for i in range(n) for j in range(n)]

        with Pool(cpu_count() - 2) as p:
            grad_a_list = p.map(
                CopierGradA(A=A, B=B, data=data, t0=t0, T=T, mu=mu, cache=cache),
                indices,
            )
            grad_b_list = p.map(
                CopierGradB(A=A, B=B, data=data, t0=t0, T=T, mu=mu, cache=cache),
                indices,
            )

        grad_a = np.reshape(np.array(grad_a_list), (-1, n))
        grad_b = np.reshape(np.array(grad_b_list), (-1, n))

    res = {"A": grad_a, "B": grad_b}

    return res


# -----------------
# Optimization Code
# -----------------


def optim(
    A_guess: np.array_equiv,
    B_guess: np.array_equiv,
    data: pd.DataFrame,
    epsilon: float = 0.01,
    lr: float = 1e-4,
    max_steps: int = 5,
    t0: int = 0,
    T: int = None,
    mu: Callable = np.tanh,
    print_diffs: bool = True,
    parallel: bool = True,
):
    assert mu in [np.tanh, sigmoid], "mu must be either np.tanh or sigmoid"

    t0 = 0 if t0 is None else t0
    T = len(data) if T is None else T

    A_prev = 0
    B_prev = 0
    A = A_guess
    B = B_guess

    step = 0

    shape = A.shape
    expected_diff_norm = epsilon / np.sqrt(shape[0])
    likelihood_step = likelihood(data=data, A=A, B=B, mu=mu, debug=False)

    print(f"Starting optimization. Initial likelihood {likelihood_step:,.1f}")
    while (
        np.any(A - A_prev) > epsilon or np.any(B - A_prev) > epsilon
    ) and step <= max_steps:
        grads = gradient(A=A, B=B, data=data, t0=t0, T=T, mu=mu, parallel=parallel)
        grad_a = grads["A"]
        grad_b = grads["B"]

        A_prev = A
        B_prev = B

        A = A_prev - lr * grad_a
        B = B_prev - lr * grad_b

        step += 1

        if print_diffs:
            likelihood_step = likelihood(data=data, A=A, B=B, mu=mu, debug=False)
            A_diff = A - A_prev
            B_diff = B - B_prev

            A_diff_norm = np.linalg.norm(A_diff)
            B_diff_norm = np.linalg.norm(B_diff)

            print(
                f"Step {step} | A_diff norm: {A_diff_norm:.5f} | B_diff norm: {B_diff_norm:.5f} | Expected Norm: {expected_diff_norm} | Likelihood: {likelihood_step:,.1f}"
            )

    res = {"A": A_guess, "B": B_guess, "A_optim": A, "B_optim": B}

    return res


# -----------------
# Questions
# -----------------

# How to use scipy optimization in this context?

# TODO: The B gradient seems to be blowing up. Determine why.


def main():
    AB = np.load("data/testdata/small_AB.npz")
    A = AB["A"]
    B = AB["B"]
    data = pd.read_csv("data/testdata/small_data.csv")

    test_grad = gradient(A=A, B=B, data=data, t0=0, mu=np.tanh, parallel=True)

    test_optim = optim(
        A_guess=A,
        B_guess=B,
        data=data,
        epsilon=0.01,
        lr=1e-2,
        max_steps=5,
        t0=0,
        T=None,
        mu=np.tanh,
        print_diffs=True,
        parallel=True,
    )

    print(test_optim)


if __name__ == "__main__":
    main()
