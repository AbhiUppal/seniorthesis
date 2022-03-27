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


# ----------------------------
# Callable for Multiprocessing
# ----------------------------


class Copier:
    def __init__(
        self,
        A: np.array_equiv,
        c: Union[float, int],
        data: pd.DataFrame,
        t0: int = 0,
        T: int = None,
        mu: Callable = np.tanh,
        derivative: Callable = None,
    ):
        self.A = A
        self.c = c
        self.data = data
        self.t0 = t0
        self.T = T
        self.mu = mu
        self.derivative = derivative

    def __call__(self, ij: tuple):
        return gradient_ij(
            A=self.A,
            c=self.c,
            data=self.data,
            ij=ij,
            t0=self.t0,
            T=self.T,
            mu=self.mu,
            derivative=self.derivative,
        )


# -----------------------
# Likelihood Calculations
# -----------------------


def likelihood(
    data,
    A: np.array_equiv,
    c: Union[float, int],
    mu: Callable = np.tanh,
    debug: bool = False,
    t0: int = None,
    T: int = None,
):
    t0 = 0 if t0 is None else t0
    T = len(data) if T is None else T
    timerange = range(t0, T - 1)

    sumterm = 0

    n = A.shape[0]

    for t in timerange:
        x_now = data.iloc[t, 1:]
        x_next = data.iloc[t + 1, 1:]
        x_now = x_now.T if x_now.shape[0] == 1 else x_now
        x_next = x_next.T if x_next.shape[0] == 1 else x_next

        Ax_t = np.matmul(A, x_now)
        predicted = x_next - mu(Ax_t)

        sumterm += np.matmul(predicted.T, predicted)

    L_term_1 = -T * n * np.log(2 * pi) / 2
    L_term_2 = -T * n * np.log(c)
    L_term_3 = -sumterm / (c ** 2)

    L = L_term_1 + L_term_2 + L_term_3

    timerange = range(t0, T - 1)

    return L


# ---------------------
# Gradient Calculations
# ---------------------


def gradient_ij(
    A: np.array_equiv,
    c: Union[float, int],
    data: pd.DataFrame,
    ij: tuple,
    t0: int = 0,
    T: int = None,
    mu: Callable = np.tanh,
    derivative: Callable = None,
) -> float:

    T = len(data) - 1 if T is None else T
    t0 = 0 if t0 is None else t0

    i = ij[0]
    j = ij[1]

    # Caclulating the gradients
    A_grad_ij = 0
    B_grad_ij = 0
    n = A.shape[0]
    for t in range(t0, T - 1):
        # Terms to pre-compute
        x_now = data.iloc[t, 1:]
        x_next = data.iloc[t + 1, 1:]

        Ax = np.matmul(A, x_now)
        predicted_term = x_next - mu(Ax)  # x_{t+1} - \mu(Ax_t)

        # Calculate A gradient
        grad_A_ij_t = x_now[i] * derivative(Ax)[j] * predicted_term[j]

        A_grad_ij += grad_A_ij_t

    A_grad_ij = -A_grad_ij / (c ** 2)

    return A_grad_ij


def gradient(
    A: np.array_equiv,
    c: Union[float, int],
    data: pd.DataFrame,
    t0: int = 0,
    T: int = None,
    mu: Callable = np.tanh,
    parallel: bool = False,
) -> np.array_equiv:

    assert mu in [np.tanh, sigmoid], "mu must be either np.tanh or sigmoid"

    t0 = 0 if t0 is None else t0
    T = len(data) if T is None else T

    shape = A.shape

    derivative = tanh_deriv if mu == np.tanh else sigmoid_deriv

    grad_a = np.zeros(shape)

    if not parallel:
        for i in range(shape[0]):
            for j in range(shape[1]):
                grad_a[i, j] = gradient_ij(
                    A=A, c=c, data=data, ij=(i, j), t0=t0, T=T, derivative=derivative
                )
    else:
        n = shape[0]
        indices = [(i, j) for i in range(n) for j in range(n)]

        with Pool(cpu_count() - 2) as p:
            grad_a_list = p.map(
                Copier(A=A, c=c, data=data, t0=t0, T=T, mu=mu, derivative=derivative),
                indices,
            )

        grad_a = np.reshape(np.array(grad_a_list), (-1, n))

    return grad_a


# -----------------
# Optimization Code
# -----------------


def optim(
    A_guess: np.array_equiv,
    c: Union[float, int],
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
    A = A_guess
    shape = A.shape

    step = 0

    expected_diff_norm = epsilon / np.sqrt(shape[0])
    likelihood_step = likelihood(data=data, A=A, c=c, mu=mu, debug=False)
    likelihood_prev = 0

    print(f"Starting optimization. Initial likelihood {likelihood_step:,.1f}")
    while (np.abs(likelihood_step - likelihood_prev) > epsilon) and step <= max_steps:
        grad = gradient(A=A, c=c, data=data, t0=t0, T=T, mu=mu, parallel=parallel)

        A_prev = A
        A = A_prev - lr * grad

        step += 1

        if print_diffs:
            likelihood_prev = likelihood_step
            likelihood_step = likelihood(data=data, A=A, c=c, mu=mu, debug=False)
            A_diff = A - A_prev

            A_diff_norm = np.linalg.norm(A_diff)

            print(
                f"Step {step} | A_diff norm: {A_diff_norm:.5f} | Expected Norm: {expected_diff_norm} | Likelihood: {likelihood_step:,.1f}"
            )

    res = {
        "A": A_guess,
        "A_optim": A,
    }

    return res


# -----------------
# Questions
# -----------------


def main():
    np.random.seed(123456)
    Ac = np.load("data/testdata/small_const_Ac.npz")
    A_true = Ac["A"]
    shape = A_true.shape
    A_test = np.random.rand(*shape)
    data = pd.read_csv("data/testdata/small_const_data.csv")

    test_optim = optim(
        A_guess=A_test, c=Ac["c"], data=data, lr=1e-4, max_steps=50, epsilon=0.1
    )


if __name__ == "__main__":
    main()
