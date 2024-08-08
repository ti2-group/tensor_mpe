import pickle

import matplotlib.pyplot as plt
import numpy as np


def main():
    with open("measurements/runtime.pkl", "rb") as f:
        n_conditional_variables_to_runtimes, n_conditional_variabales_to_chain_lengths, n_conditional_to_median_runtime = pickle.load(f)

    # plot the results
    total_num_variables = lambda chain_length: 2 * chain_length - 1
    save_file = "graphics/runtimes.png"
    plt.title("Runtime of MPE on random tensor train graphical models")
    plt.xlabel("Total number of variables")
    plt.ylabel("Mean runtime (s)")
    markers = ["o", "s", "D", "x", "^"]
    for n_variables_set, marker in zip(n_conditional_variables_to_runtimes.keys(), markers):
        plt.scatter(total_num_variables(np.array(n_conditional_variabales_to_chain_lengths[n_variables_set])), n_conditional_variables_to_runtimes[n_variables_set], label=f"{n_variables_set} conditional variables", marker=marker)
        sorted_chain_lengths, medians = n_conditional_to_median_runtime[n_variables_set]
        plt.plot(total_num_variables(np.array(sorted_chain_lengths)), medians)
    plt.legend()
    plt.savefig(save_file)
    print(f"Saved plot to {save_file}")


if __name__ == "__main__":
    main()
