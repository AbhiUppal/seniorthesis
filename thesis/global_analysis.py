# This script contains code to experiment on whether we achieve the same estimates with different guesses
# We can test this with c = 1, N = 10, T = 1000
# For this experiment, we just use the data from one of the runs of the error analysis code
# File of choice: experiment-data/error_N10_c1_T1000_1.npz

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns

from glob import glob
from optim_single import optim, likelihood
from typing import Union, Callable

sns.set(font_scale=1.2)


def one_optim(
    c: Union[float, int],
    N: int,
    T: int,
    mu: Callable = np.tanh,
    run_number: int = 1,
    save_output: bool = False,
    epsilon: float = 0.1,
    lr: float = 1e-4,
    max_steps: int = 50,
    print_steps: bool = False,
):
    """This is similar to one_optim from error_analysis.py with some small modifications
    for the right file I/O.

    set_number is replaced with run_number. They serve the same purpose, but remember that
    a 'run' in this experiment constitutes doing the same optimization, only with a different
    initial guess."""

    # Since we're just using the data from the other experiment
    cstr = str(c).replace(".", "-")
    base = f"error_N{N}_c{cstr}_T{T}"
    base_global = f"global_N{N}_c{cstr}_T{T}"
    fname_df = f"{base}_data_1.csv"
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
        outname = f"{base_global}_{run_number}_result"
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
    T = 1000
    N = 10
    run_numbers = range(0, 10)

    for c in c_values:
        for run_number in run_numbers:
            # Check to see if this has already been computed
            cstr = str(c).replace(".", "-")
            base = f"global_N{N}_c{cstr}_T{T}"
            outname = f"{base}_{run_number}_result"

            if len(glob(f"experiment-outputs/{outname}.npz")) != 0:
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
                    save_output=True,
                    run_number=run_number,
                )


# --------------------
# Aggregating the Data
# --------------------


def data_to_npz():
    # Things to pull:
    #   The True A value
    # Things to store:
    #   A list of each of the 10 inferred values of A
    #   Use that list to compute the element-wise mean and sd
    # Parameter values: c = 1, N = 10, T = 1000

    fnames = [
        f"experiment-outputs/global_N10_c1_T1000_{i}_result.npz" for i in range(10)
    ]

    true_A = None
    A_guesses = []

    for fname in fnames:
        results = np.load(fname)
        true_A = results["true_A"] if true_A is None else true_A
        A_guesses.append(results["optim_A"])

    sum_A = 0
    for A in A_guesses:
        sum_A += A
    mean_A = sum_A / len(A_guesses)  # Mean optimized A value, element-wise

    sum_sq = 0
    for A in A_guesses:
        diff = A - mean_A
        sum_sq += diff * diff

    std_A = np.sqrt(sum_sq / (len(A_guesses) - 1))  # Sample std.dev, element-wise

    np.savez(
        "experiment-results/global_hm_matrices",
        std_A=std_A,
        mean_A=mean_A,
        true_A=true_A,
    )

    return true_A, mean_A, std_A


# ------------------
# Analyzing the Data
# ------------------


def heatmap_sd_guesses(
    show: bool = False, save_path: str = None, seaborn: bool = False
):
    fname = "experiment-results/global_hm_matrices.npz"
    result = np.load(fname)
    std_A = result["std_A"]

    if not seaborn:
        hm = go.Figure(data=go.Heatmap(z=std_A))
        if show:
            hm.show()
        if save_path is not None:
            hm.write_image(save_path, height=1080, width=1920, scale=2, format="png")
    else:
        colorbar = {
            "label": "Parameter Standard Deviation",
            "orientation": "vertical",
        }
        hm_sns = sns.heatmap(
            data=std_A,
            vmin=0,
            vmax=0.5,
            linewidths=0.5,
            cmap="twilight",
            cbar_kws=colorbar,
        )
        if show:
            plt.show()
        if save_path is not None:
            hm_sns.figure.savefig(save_path, dpi=300, pad_inches=0)

    return hm if not seaborn else hm_sns


def heatmap_mean_minus_true(
    show: bool = False, save_path: str = None, seaborn: bool = False
):
    fname = "experiment-results/global_hm_matrices.npz"
    result = np.load(fname)
    diff = result["mean_A"] - result["true_A"]

    if not seaborn:
        hm = go.Figure(data=go.Heatmap(z=diff))
        if show:
            hm.show()
        if save_path is not None:
            hm.write_image(save_path, height=1080, width=1920, scale=2, format="png")
    else:
        colorbar = {
            "label": "Error in Mean Inferred Parameter",
            "orientation": "vertical",
        }
        hm_sns = sns.heatmap(
            data=diff,
            vmin=-0.7,
            vmax=0.7,
            linewidths=0.5,
            cmap="twilight",
            cbar_kws=colorbar,
        )
        if show:
            plt.show()
        if save_path is not None:
            hm_sns.figure.savefig(save_path, dpi=300, pad_inches=0)

    return hm if not seaborn else hm_sns


# ----
# Main
# ----


def main():
    heatmap_sd_guesses(
        show=True, save_path="figures/heatmap_sd_guesses.png", seaborn=True
    )
    heatmap_mean_minus_true(
        show=True, save_path="figures/heatmap_average_minus_true.png", seaborn=True
    )


if __name__ == "__main__":
    main()
