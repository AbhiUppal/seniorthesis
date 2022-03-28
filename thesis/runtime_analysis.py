import numpy as np
import pandas as pd
import time

from data.generation import _generate_time_series
from experiment import experiment_data_generation
from optim import gradient
from utils import timer

# ---------------
# Data Generation
# ---------------


def generate_data():
    c_values = [1]
    N_values = list(range(5, 13))
    T_values = [1000]

    experiment_data_generation(
        c_values=c_values,
        T_values=T_values,
        N_values=N_values,
        num_each=1,
        prefix="runtime",
        mu=np.tanh,
    )

    # File output: runtime_N{N}_c1_T1000_data_0.csv
    # NPZ output:  runtime_N{N}_c1_T1000_Ac.npz


# -----------------
# Experimental Data
# -----------------


@timer
def gradient_timer(*args, **kwargs):
    return gradient(*args, **kwargs)


def compute_runtimes() -> pd.DataFrame:
    """Collects summary statistics on runtime for gradient calculations for N = 5, 6, ... 12
    Outputs a dataframe at experiment-results/runtimes.csv"""
    means = []
    sds = []
    log_means = []
    log_sds = []
    Ns = []
    for N in range(5, 13):
        # Generate a dataset
        A = np.random.rand(N, N)
        B = np.random.rand(N, N)
        data = _generate_time_series(A=A, B=B, T=1000, N=N, f=np.tanh)[0]
        times = []
        log_times = []
        for i in range(25):
            A = np.random.rand(N, N)
            B = np.random.rand(N, N)

            grad, time = gradient(A=A, B=B, data=data, mu=np.tanh, parallel=True)
            times.append(time)
            log_times.append(np.log(time))
        means.append(np.mean(times))
        sds.append(np.std(times))
        log_means.append(np.mean(log_times))
        log_sds.append(np.mean(log_times))
        Ns.append(N)

    output_df = pd.DataFrame(
        {
            "N": Ns,
            "mean_runtime": means,
            "sd_runtime": sds,
            "mean_log_runtime": log_means,
            "sd_log_runtime": log_sds,
        }
    )

    output_df.to_csv("experiment-results/runtimes.csv", index=False)

    return output_df


# ----
# Main
# ----


def main():
    compute_runtimes()


if __name__ == "__main__":
    main()
