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

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from experiment import experiment_data_generation
from glob import glob
from optim_single import optim, likelihood
from scipy.stats import t
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
    set_numbers = range(0, 10)

    for set_number in set_numbers:
        for c in c_values:
            for N in N_values:
                for T in T_values:
                    # Check to see if this has already been computed
                    cstr = str(c).replace(".", "-")
                    base = f"error_N{N}_c{cstr}_T{T}"
                    outname = f"{base}_Ac_{set_number}_result"
                    if len(glob(f"experiment-outputs/{outname}.npz")) != 0:
                        continue
                    else:
                        one_optim(
                            c=c,
                            N=N,
                            T=T,
                            mu=mu,
                            epsilon=epsilon * c,
                            lr=lr,
                            max_steps=max_steps,
                            print_steps=print_steps,
                            save_output=True,
                            set_number=set_number,
                        )


# --------------------
# Aggregating the Data
# --------------------


def data_to_csv():
    c_values = [0.25, 0.5, 0.75, 1, 1.5, 2]
    # N = 10, T = 1000

    # Things to store:
    # Frobenius norms of error_guess and error_optim
    # sum of bool(\hat{E} < E_0), the number of parameters that improved
    # N
    # The run number, since we need to aggregate by those
    # initial likelihood
    # optimized likelihood
    # true likelihood

    error_guess_norms = []
    error_optim_norms = []
    improved_params = []
    set_numbers = []
    initial_likelihoods = []
    optimized_likelihoods = []
    true_likelihoods = []
    cs = []

    # Also, just compute the summary stats straight up as well
    # mean of frobenius norms
    # sd of frobenius norms
    # Number of trials for each (10)

    cs_agg = []

    mean_error_guess_norm = []
    sd_error_guess_norm = []
    mean_error_guess = []
    sd_error_guess = []

    mean_error_optim_norm = []
    sd_error_optim_norm = []
    mean_error_optim = []
    sd_error_optim = []

    mean_improved = []
    sd_improved = []

    mean_likelihood_guess = []
    sd_likelihood_guess = []

    mean_likelihood_optim = []
    sd_likelihood_optim = []

    true_likelihoods_agg = []

    for c in c_values:
        error_guess_norms_temp = []
        error_guess_temp = []
        error_optim_norms_temp = []
        error_optim_temp = []
        num_improved_temp = []
        likelihood_guess_temp = []
        likelihood_optim_temp = []
        likelihood_true_temp = []
        for set_number in range(0, 10):
            cstr = str(c).replace(".", "-")
            fname = (
                f"experiment-outputs/error_N10_c{cstr}_T1000_Ac_{set_number}_result.npz"
            )
            results = np.load(fname)

            error_guess = results["error_guess"]
            error_optim = results["error_optim"]
            init_likelihood = results["guess_likelihood"]
            optim_likelihood = results["optim_likelihood"]
            true_likelihood = results["true_likelihood"]

            set_numbers.append(set_number)
            initial_likelihoods.append(init_likelihood)
            likelihood_guess_temp.append(init_likelihood)
            optimized_likelihoods.append(optim_likelihood)
            likelihood_optim_temp.append(optim_likelihood)
            true_likelihoods.append(true_likelihood)
            likelihood_true_temp.append(true_likelihood)

            num_improved = np.sum(error_optim < error_guess)
            error_guess_norm = np.linalg.norm(error_guess)
            error_optim_norm = np.linalg.norm(error_optim)
            error_guess_temp.append(np.sum(error_guess))
            error_optim_temp.append(np.sum(error_optim))

            improved_params.append(num_improved)
            num_improved_temp.append(num_improved)
            error_guess_norms.append(error_guess_norm)
            error_guess_norms_temp.append(error_guess_norm)
            error_optim_norms.append(error_optim_norm)
            error_optim_norms_temp.append(error_optim_norm)
            cs.append(c)

        mean_error_guess_norm.append(np.mean(error_guess_norms_temp))
        sd_error_guess_norm.append(np.std(error_guess_norms_temp))

        mean_error_guess.append(np.mean(error_guess_temp))
        sd_error_guess.append(np.std(error_guess_temp))

        mean_error_optim_norm.append(np.mean(error_optim_norms_temp))
        sd_error_optim_norm.append(np.std(error_optim_norms_temp))

        mean_error_optim.append(np.mean(error_optim_temp))
        sd_error_optim.append(np.std(error_optim_temp))

        mean_improved.append(np.mean(num_improved_temp))
        sd_improved.append(np.std(num_improved_temp))

        mean_likelihood_guess.append(np.mean(likelihood_guess_temp))
        sd_likelihood_guess.append(np.std(likelihood_guess_temp))

        mean_likelihood_optim.append(np.mean(likelihood_optim_temp))
        sd_likelihood_optim.append(np.std(likelihood_optim_temp))

        true_likelihoods_agg.append(true_likelihood)

        cs_agg.append(c)

    df_each_run = pd.DataFrame(
        {
            "error_guess_norm": error_guess_norms,
            "error_optim_norm": error_optim_norms,
            "improved_params": improved_params,
            "set_number": set_numbers,
            "initial_likelihood": initial_likelihoods,
            "optimized_likelihood": optimized_likelihoods,
            "true_likelihood": true_likelihoods,
            "N": 10,
            "T": 1000,
            "c": cs,
        }
    )

    df_aggregated = pd.DataFrame(
        {
            "mean_error_guess_norm": mean_error_guess_norm,
            "sd_error_guess_norm": sd_error_guess_norm,
            "mean_error_guess": mean_error_guess,
            "sd_error_guess": sd_error_guess,
            "mean_likelihood_guess": mean_likelihood_guess,
            "sd_likelihood_guess": sd_likelihood_guess,
            "mean_error_optim_norm": mean_error_optim_norm,
            "sd_error_optim_norm": sd_error_optim_norm,
            "mean_error_optim": mean_error_optim,
            "sd_error_optim": sd_error_optim,
            "mean_likelihood_optim": mean_likelihood_optim,
            "sd_likelihood_optim": sd_likelihood_optim,
            "true_likelihood": true_likelihoods_agg,
            "mean_improved": mean_improved,
            "sd_improved": sd_improved,
            "c": cs_agg,
        }
    )

    df_each_run.to_csv("experiment-results/error_all_runs.csv")
    df_aggregated.to_csv("experiment-results/error_aggregated.csv")

    return df_each_run, df_aggregated


# ------------------
# Analyzing the Data
# ------------------


def error_bars_optim(show: bool = False, save_path: str = None):
    """Frobenius norm of the error matrix for each value of c, error bar plot"""
    number_trials = 10
    critical_t = t.ppf(0.975, df=number_trials - 1)

    fname = "experiment-results/error_aggregated.csv"
    data = pd.read_csv(fname).drop(columns=["Unnamed: 0"])

    data["margin_error_optim"] = (
        critical_t * data["sd_error_optim"] / np.sqrt(number_trials - 1)
    )

    fig = go.Figure(
        data=go.Scatter(
            x=data["c"],
            y=data["mean_error_optim"],
            error_y=dict(type="data", array=data["margin_error_optim"], visible=True),
            line=dict(color="black", width=2, dash="dash"),
        )
    )

    fig.update_layout(
        font_family="Glacial Indifference",
        font_color="black",
        font_size=18,
        yaxis_title="Mean Error of Inferred Matrix",
        xaxis_title="Value of c",
        legend_title_font_color="black",
        template="plotly_white",
    )

    if show:
        fig.show()

    if save_path is not None:
        fig.write_image(save_path, height=1080, width=1920, scale=2, format="png")

    return fig


def error_bars_params_improved(show: bool = False, save_path: str = None):
    """Number of improved parameters for each value of c, error bars"""
    number_trials = 10
    critical_t = t.ppf(0.975, df=number_trials - 1)

    fname = "experiment-results/error_aggregated.csv"
    data = pd.read_csv(fname).drop(columns=["Unnamed: 0"])

    data["margin_error_improved"] = (
        critical_t * data["sd_improved"] / np.sqrt(number_trials - 1)
    )

    fig = go.Figure(
        data=go.Scatter(
            x=data["c"],
            y=data["mean_improved"],
            error_y=dict(
                type="data", array=data["margin_error_improved"], visible=True
            ),
            line=dict(color="black", width=2, dash="dash"),
            marker_size=10,
        )
    )

    fig.update_layout(
        font_family="Glacial Indifference",
        font_color="black",
        font_size=18,
        yaxis_title="Number of parameters improved",
        xaxis_title="Value of c",
        legend_title_font_color="black",
        template="plotly_white",
    )

    if show:
        fig.show()

    if save_path is not None:
        fig.write_image(save_path, height=1080, width=1920, scale=2, format="png")

    return fig


def heatmap_errors(show: bool = False, save_path: str = None):
    # Load results from experiment c=1, T=1000, N=10
    # Trial number 2 (index 1)
    fname = "experiment-outputs/error_N10_c1_T1000_Ac_3_result.npz"
    result = np.load(fname)

    error_guess = result["error_guess"]
    error_optim = result["error_optim"]

    hm1 = go.Figure(data=go.Heatmap(z=error_guess))

    hm1.update_layout(
        coloraxis_colorbar=dict(title="Error", tickvals=[-1, -0.5, 0, 0.5, 1])
    )

    hm2 = go.Figure(data=go.Heatmap(z=error_optim))

    hm2.update_layout(
        coloraxis_colorbar=dict(title="Error", tickvals=[-1, -0.5, 0, 0.5, 1])
    )

    hm3 = go.Figure(data=go.Heatmap(z=error_optim - error_guess))

    if show:
        hm1.show()
        hm2.show()
        hm3.show()

    if save_path is not None:
        hm3.write_image(save_path, height=1080, width=1920, scale=2, format="png")

    return hm1, hm2, hm3


# ----
# Main
# ----


def main():
    error_bars_optim(show=False, save_path="figures/error_bars_optim.png")
    error_bars_params_improved(
        show=False, save_path="figures/error_bars_num_params_improved.png"
    )
    heatmap_errors(show=False, save_path="figures/heatmap_improved_params.png")


if __name__ == "__main__":
    main()
