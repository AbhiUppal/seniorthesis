import numpy as np
import pandas as pd
import plotly.express as px

from typing import Callable, Union


def _rowNorm(matrix: Union[np.array, np.array_equiv]):
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
        save_path += (
            "/" if save_path[-1] != "/" else ""
        )  # Make sure we save into the folder
        df.to_csv(save_path + "data.csv")
        np.save(save_path + "A.npy", A)
        np.save(save_path + "B.npy", B)

    return df, A, B


# Okay, so this is a bit of a hunch, but I think making the adj. matrix a contraction mapping
# by normalizing the rows is a reasonable assumption. For example, assuming there is a stationary distribution
# then the contraction would represent convergence to the stationary distribution
# plus some noise, which should generate the behavior we're looking for
# Essentially, we need constraints on the adj. matrix to ensure stationarity
# Or, use a sigmoid function to constrain like tanh

if __name__ == "__main__":
    np.random.seed = 123456
    x0 = np.array([1, 3, 4, 5, 6])
    A = np.random.rand(50, 50)
    B = np.random.rand(50, 50)
    df = _generate_time_series(A, B, T=1000, N=50, f=np.tanh, save_path="testdata")

    testDict = {"A": A, "B": B, "dataset": df}

    print(type(testDict["A"]))
    print(type(testDict["B"]))
    print(type(testDict["dataset"]))

    print(testDict)

    np.save("test_A.npy", A)
    np.save("test_B.npy", B)

    load_A = np.load("test_A.npy")
    load_B = np.load("test_B.npy")

    assert A.all() == load_A.all()
    assert B.all() == load_B.all()

    fig = px.line(df, x=df.index, y=[df["x1"], df["x50"]])
    fig.show()
