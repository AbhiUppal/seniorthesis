# Utilities for generating data and running experiments

import glob
import numpy as np


from thesis.data.generation import _generate_time_series_constant as generate
from typing import Callable

# -----------------------------
# File Generation and Searching
# -----------------------------


def experiment_data_generation(
    c_values: list,
    T_values: list,
    N_values: list,
    num_each: int,
    prefix: str = None,
    mu: Callable = np.tanh,
) -> None:
    """experiment_generation generates datasets

    Parameters
    ----------
    c_values : list
        values of the constant to generate data with
    T_values : list
        values of T (dataset lengths) to generate data with
    N_values : list
        number of variables to generate data with
    num_each: int
        Number of each dataset to produce
    prefix: str
        Identifier for the batch of data generation, for the purpose of segrating different experiments
    """
    for N in N_values:
        for T in T_values:
            for c in c_values:
                A = np.random.rand(N, N)

                dfs = [
                    generate(A=A, c=c, N=N, T=T, save_path=None, mu=mu)
                    for i in range(num_each)
                ]
                cstr = str(c).replace(".", "-")

                for i, df in enumerate(dfs):
                    fname_df = (
                        f"{prefix}_N{N}_c{cstr}_T{T}_data_{i}.csv"
                        if prefix is not None
                        else f"N{N}_c{cstr}_T{T}_{i}_data.csv"
                    )
                    df.to_csv(f"experiment-data/{fname_df}")

                fname_ac = (
                    f"{prefix}_N{N}_c{cstr}_T{T}_Ac"
                    if prefix is not None
                    else f"N{N}_c{cstr}_T{T}_{i}_Ac"
                )
                np.savez(f"experiment-data/{fname_ac}", A=A, c=c)


def experiment_file_eunumeration(
    c_values: list, T_values: list, N_values: list, prefix: str = None
) -> list:
    """Probably going to deprecate this. It's overengineered.
    experiment_file_eunumeration gets the filenames for a given experiment by first validating the existence
    of the files and creating them if they do not exist, based on the values of c, T, and N.

    Parameters
    ----------
    c_values : list
        constant values of interest
    T_values : list
        dataset length values of interest
    N_values : list
        number of nodes values of interest

    Returns
    -------
    list
        A list of filenames to consider for the given experiment.
    """
    filenames = []
    for N in N_values:
        for T in T_values:
            for c in c_values:
                cstr = str(c).replace(".", "-")
                if prefix is not None:
                    match = f"{prefix}_N{N}_c{cstr}_T{T}_*"
                else:
                    match = f"N{N}_c{cstr}_T{T}_*"
                matches = glob.glob(f"experiment-data/{match}")
                filenames.append(matches)


# --------------------
# Code for experiments
# --------------------


def main():
    pass


if __name__ == "__main__":
    main()
