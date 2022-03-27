import numpy as np
from numpy.lib.npyio import save
import pandas as pd
import plotly.express as px

from typing import Callable, Union


def _rowNorm(matrix: Union[np.array, np.array_equiv]):
    """Normalize the rows of a 2d numpy array (matrix)"""
    row_sums = matrix.sum(axis=1)
    new_matrix = matrix / row_sums[:, np.newaxis]
    return new_matrix


def _step(
    x0: Union[np.array, np.array_equiv],
    A: Union[np.array, np.array_equiv],
    B: Union[np.array, np.array_equiv],
    f: Callable[[np.array], np.array] = None,
) -> np.array:
    """_step advances the system forward one step in time via the equation
    x(t+1) = f(Ax(t)) + B n(t)

    Parameters
    ----------
    x0 : Union[np.array, np.array_equiv]
        initial state
    A : Union[np.array, np.array_equiv]
        Adjaency matrix
    B : Union[np.array, np.array_equiv]
        Matrix representing the relationship between the noises of the different variables
    f : Callable[[np.array], np.array], optional
        function to apply to Ax as a transformation, by default None

    Returns
    -------
    np.array
        State at the next point in time
    """
    n = len(x0)
    I = np.identity(n)
    eta = np.random.multivariate_normal(mean=np.zeros(n), cov=I)
    if f is None:
        x1 = np.matmul(A, x0) + np.matmul(B, eta)
    else:
        x1 = f(np.matmul(A, x0)) + np.matmul(B, eta)
    return x1


def _generate_time_series(
    A: np.array, B: np.array, T: int, N: int, f: Callable = None, save_path: str = None
):
    """_generate_time_series generates N different time series of length T based on two NxN matrices, A and B

    Parameters
    ----------
    A : np.array
        N x N adjacency matrix
    B : np.array
        N x N matrix representing how the noise of all of the variables are related
    T : int
        Length of the time series to generate
    N : int
        Number of time series to generate
    f : Callable, optional
        Function to transform the Ax term by, by default None
        If A is not row-normalized, then this should be some sigmoid function like tanh
        Otherwise, the series will all blow up and overflow errors will occur
    save_path : str, optional
        FOLDER to save the resulting time series to, by default None
        If None, does not save

    Returns
    -------
    pd.DataFrame
        dataframe with indices
    """
    x = []
    x.append(np.random.rand(N))
    for i in range(1, T):
        x.append(_step(x[i - 1], A, B, f))

    cols = ["x" + str(i + 1) for i in range(N)]
    arr = np.array(x)
    df = pd.DataFrame(arr, columns=cols)

    if save_path is not None:
        df.to_csv(save_path + "_data.csv")
        np.savez(save_path + "_AB", A=A, B=B)

    return df, A, B


def _step_constant(
    x0: Union[np.array, np.array_equiv],
    A: Union[np.array, np.array_equiv],
    c: Union[float, int],
    f: Callable[[np.array], np.array] = None,
) -> np.array:
    n = len(x0)
    I = np.identity(n)
    eta = np.random.multivariate_normal(mean=np.zeros(n), cov=I)
    if f is None:
        x1 = np.matmul(A, x0) + c * eta
    else:
        x1 = f(np.matmul(A, x0)) + c * eta
    return x1


def _generate_time_series_constant(
    A: np.array,
    c: Union[float, int],
    N: int,
    T: int,
    mu: Callable = None,
    save_path: str = None,
):
    x = []
    x.append(np.random.rand(N))
    for i in range(1, T):
        x.append(_step_constant(x0=x[i - 1], A=A, c=c, f=mu))

    cols = ["x" + str(i + 1) for i in range(N)]
    arr = np.array(x)
    df = pd.DataFrame(arr, columns=cols)

    if save_path is not None:
        df.to_csv(save_path + "_data.csv")
        np.savez(save_path + "_Ac", A=A, c=c)

    return df, A, c


def main():
    np.random.seed = 123456
    A = np.random.rand(10, 10)
    c = 1
    df = _generate_time_series_constant(
        A=A, c=c, T=1000, N=A.shape[0], mu=np.tanh, save_path="testdata/small_const"
    )


if __name__ == "__main__":
    main()
