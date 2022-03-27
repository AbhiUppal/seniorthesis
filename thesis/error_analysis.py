# Script for performing error analysis on optimized matrices

# For this experiment, I want to try the following values:
# c: [0.25, 0.5, 0.75, 1, 1.5, 2]
# N: 10
# T: [1000, 2000]
# 10 datasets each

# This script will be used to generate the error bar plots as well as
# the heatmaps for inference error

# TODO: Function that takes the range of parameters, and computes optim_single for each
#   With said output, compute the frobenius norm of each error matrix and store it
#   Store frobenius norms in DF along with the other parameters
#   Output it all to a CSV.
# TODO: Function that computes, for any one run, the number of

import json
import numpy as np
import pandas as pd

from experiment import experiment_data_generation
from glob import glob
from optim_single import optim, likelihood
from typing import Union, Callable
from utils import timer


def generate_data():
    c = [0.25, 0.5, 0.75, 1, 1.5, 2]
    N = [10]
    T = [1000]
    neach = 10

    experiment_data_generation(
        c_values=c, N_values=N, T_values=T, num_each=neach, prefix="error"
    )


# ---------------------------
# Computing the Optimizations
# ---------------------------


@timer
def one_optim(
    c: Union[float, int],
    N: int,
    T: int,
    mu: Callable = np.tanh,
    set_number: int = 1,
    save_output: bool = False,
    epsilon: float = 0.1,
    lr: float = 1e-4,
    max_steps: int = 50,
    print_steps: bool = False,
):
    cstr = str(c).replace(".", "-")

    base = f"error_N{N}_c{cstr}_T{T}"
    fname_df = f"{base}_data_{set_number}.csv"
    fname_ac = f"{base}_Ac.npz"

    Ac = np.load(f"experiment-data/{fname_ac}")
    df = pd.read_csv(f"experiment-data/{fname_df}")

    true_A = Ac["A"]
    guess_A = np.random.rand(N, N)

    optim_values = optim(
        A_guess=guess_A,
        c=c,
        data=df,
        epsilon=epsilon,
        lr=lr,
        max_steps=max_steps,
        parallel=True,
        print_diffs=print_steps,
    )

    optim_A = optim_values["A_optim"]

    error_guess = true_A - guess_A
    error_optim = true_A - optim_A

    true_likelihood = likelihood(
        data=df,
        A=true_A,
        c=c,
        mu=mu,
    )

    optim_likelihood = likelihood(
        data=df,
        A=optim_A,
        c=c,
        mu=mu,
    )

    guess_likelihood = likelihood(
        data=df,
        A=guess_A,
        c=c,
        mu=mu,
    )

    res = {
        "guess_A": guess_A,
        "optim_A": optim_A,
        "true_A": true_A,
        "true_likelihood": true_likelihood,
        "optim_likelihood": optim_likelihood,
        "guess_likelihood": guess_likelihood,
        "error_guess": error_guess,
        "error_optim": error_optim,
    }

    if save_output:
        outname = f"{base}_Ac_{set_number}_result"
        np.savez(f"experiment-outputs/{outname}", **res)

    return res


def optimizations(
    mu: Callable = np.tanh,
    epsilon: float = 0.1,
    lr: float = 1e-4,
    max_steps: int = 50,
    print_steps: bool = False,
):
    c_values = [0.25, 0.5, 0.75, 1, 1.5, 2]
    N_values = [10]
    T_values = [1000]
    set_numbers = range(1, 11)

    for set_number in set_numbers:
        for c in c_values:
            for N in N_values:
                for T in T_values:
                    # Check to see if this has already been computed
                    cstr = str(c).replace(".", "-")
                    base = f"error_N{N}_c{cstr}_T{T}"
                    outname = f"{base}_Ac_{set_number}_result"
                    if len(glob(f"experiment-data/{outname}.npz")) != 0:
                        continue
                    else:
                        one_optim(
                            c=c,
                            N=N,
                            T=T,
                            mu=mu,
                            epsilon=epsilon,
                            lr=lr,
                            max_steps=max_steps,
                            print_steps=print_steps,
                        )


# ------------------
# Analyzing the Data
# ------------------


# ----
# Main
# ----


def main():
    cNT = {"c": 1, "N": 10, "T": 1000}
    one_optim(cNT=cNT, print_steps=True, set_number=1, save_output=True)


if __name__ == "__main__":
    main()
