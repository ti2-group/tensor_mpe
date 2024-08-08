import pickle
import time
from collections import defaultdict

import numpy as np
from tqdm import tqdm

from graphical_model import GraphicalModel
from random_benchmarks import Query, gen_benchmarks

RUN_N_TIMES = 10


def measure_queries(model: GraphicalModel, queries: list[Query]) -> list[float]:
    runtimes = []
    for variables, evidence in tqdm(queries, leave=False, desc="running queries on model"):
        start = time.time()
        for _ in range(RUN_N_TIMES):
            model.mpe(variables, evidence)
        end = time.time()
        runtimes.append((end - start) / RUN_N_TIMES)
    return runtimes


def benchmark_mpe(models: list[GraphicalModel], queries_per_model: list[list[Query]]) -> list[list[int]]:
    n_models = len(models)
    return [measure_queries(models[i], queries_per_model[i]) for i in tqdm(range(n_models), desc="benchmarking all models", leave=False)]


def compute_runtimes(runtimes: list[list[int]], models: list[GraphicalModel], queries_per_model: list[list[Query]], save_file: str):
    chain_lengths = [model.n_variables // 2 for model in models]
    # match how many variables each query has set as evidence vs runtime
    n_conditional_variables_to_runtimes: dict[int, list[int]] = defaultdict(list)
    n_conditional_variabales_to_chain_lengths: dict[int, list[int]] = defaultdict(list)
    for i in range(len(models)):
        for j in range(len(queries_per_model[i])):
            chain_length = chain_lengths[i]
            n_variables_set = len(queries_per_model[i][j][1])
            runtime = runtimes[i][j]
            n_conditional_variables_to_runtimes[n_variables_set].append(runtime)
            n_conditional_variabales_to_chain_lengths[n_variables_set].append(chain_length)
    # get the median of the runtimes for each chain length and number of variables set
    n_conditional_to_median_runtime: dict[int, tuple[list[int], list[float]]] = {}
    for n_variables_set in n_conditional_variables_to_runtimes.keys():
        runtimes_per_chain_length = defaultdict(list)
        for chain_length, runtime in zip(n_conditional_variabales_to_chain_lengths[n_variables_set], n_conditional_variables_to_runtimes[n_variables_set]):
            runtimes_per_chain_length[chain_length].append(runtime)
        sorted_chain_lengths = sorted(runtimes_per_chain_length.keys())
        medians = [np.median(runtimes_per_chain_length[chain_length]) for chain_length in sorted_chain_lengths]
        n_conditional_to_median_runtime[n_variables_set] = (sorted_chain_lengths, medians)

    with open(save_file, "wb") as f:
        pickle.dump((n_conditional_variables_to_runtimes, n_conditional_variabales_to_chain_lengths, n_conditional_to_median_runtime), f)


def main():
    models, queries_per_model = gen_benchmarks()
    runtimes = benchmark_mpe(models, queries_per_model)
    compute_runtimes(runtimes, models, queries_per_model, "measurements/runtime.pkl")


if __name__ == "__main__":
    main()
