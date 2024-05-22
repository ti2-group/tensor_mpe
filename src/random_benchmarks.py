import torch
from tqdm import tqdm

from graphical_model import GraphicalModel, Variables, Evidence
from my_bitsets import Bitset


Query = tuple[Variables, Evidence]
Benchmarks = tuple[list[GraphicalModel], list[list[Query]]]


SEED = 0

DEFAULT_MIN_CHAIN_LENGTH = 20
DEFAULT_MAX_CHAIN_LENGTH = 400
DEFAULT_CHAIN_STEP = 20

DEFAULT_AXIS_SIZE = 64
DEFAULT_MIN_N_OBSERVED = 0
DEFAULT_MAX_N_OBSERVED = 200
DEFAULT_N_OBSERVED_STEP = 50
# number of queries that are randomly chosen for each possible number of variables
DEFAULT_QUERIES_PER_NUMBER_OF_VARIABLES = 3


def gen_hidden_markov_model(chain_length: int, axis_size: int = DEFAULT_AXIS_SIZE, auto_seed=True) -> GraphicalModel:
    if auto_seed:
        torch.manual_seed(SEED)
    # n - 1 hidden variables: 0 ... chain_length - 2
    # n observed variables: chain_length - 1 ... 2 * chain_length - 2
    used_variables_per_tensor = []
    for i in range(chain_length):
        # lower tensor
        lower_interaction_variables = [chain_length + i - 1]
        used_variables_per_tensor.append(Bitset(lower_interaction_variables))
        # upper tensor
        upper_interaction_variables = [chain_length + i - 1]
        if i > 0:
            upper_interaction_variables.append(i - 1)
        if i < chain_length - 1:
            upper_interaction_variables.append(i)
        used_variables_per_tensor.append(Bitset(upper_interaction_variables))
    interaction_parameters = {}
    for interaction_variables in used_variables_per_tensor:
        interaction_tensor = torch.rand((axis_size,) * len(interaction_variables))
        interaction_parameters[interaction_variables] = interaction_tensor
    return GraphicalModel(2 * chain_length - 1, interaction_parameters)


def gen_queries(model: GraphicalModel, axis_size: int = DEFAULT_AXIS_SIZE, min_n_observed: int = DEFAULT_MIN_N_OBSERVED, max_n_observed: int = DEFAULT_MAX_N_OBSERVED, n_observed_step: int = DEFAULT_N_OBSERVED_STEP, queries_per_number_of_variables: int = DEFAULT_QUERIES_PER_NUMBER_OF_VARIABLES, auto_seed=True) -> list[Query]:
    """Generates queries for a specific model. A query consists of a set of observed variables and corresponding evidence. Also computes the maximum probability for that query for a sanity check."""

    if auto_seed:
        torch.manual_seed(SEED)
    queries = []
    max_n_observed = min(max_n_observed, model.n_variables)
    for n_observed in range(min_n_observed, max_n_observed + 1, n_observed_step):
        for _ in range(queries_per_number_of_variables):
            # choose random observed variables
            observed_variables = Bitset(torch.randperm(model.n_variables)[:n_observed].tolist())
            # assign these random variables with random values
            evidence = tuple(torch.randint(0, axis_size - 1, (n_observed,)).tolist()) if n_observed > 0 else ()
            queries.append((observed_variables, evidence))
    return queries


def query_to_list(query: Query) -> list[int]:
    variables, evidence = query
    return [variables, *evidence]


def list_to_query(query_list: list[int]) -> Query:
    return tuple(query_list[0], tuple(query_list[1:]))


def gen_benchmarks(
        min_chain_length: int = DEFAULT_MIN_CHAIN_LENGTH,
        max_chain_length: int = DEFAULT_MAX_CHAIN_LENGTH,
        chain_step: int = DEFAULT_CHAIN_STEP,
        axis_size: int = DEFAULT_AXIS_SIZE,
        min_n_observed: int = DEFAULT_MIN_N_OBSERVED,
        max_n_observed: int = DEFAULT_MAX_N_OBSERVED,
        n_observed_step: int = DEFAULT_N_OBSERVED_STEP,
        queries_per_number_of_variables: int = DEFAULT_QUERIES_PER_NUMBER_OF_VARIABLES,
        auto_seed=True) -> Benchmarks:
    chain_lengths = range(min_chain_length, max_chain_length + 1, chain_step)
    models = [gen_hidden_markov_model(chain_length, axis_size=axis_size, auto_seed=auto_seed) for chain_length in tqdm(chain_lengths, desc="generating models", leave=False)]
    queries_per_model = [gen_queries(
        model, axis_size=axis_size,
        min_n_observed=min_n_observed,
        max_n_observed=max_n_observed,
        n_observed_step=n_observed_step,
        queries_per_number_of_variables=queries_per_number_of_variables
    ) for model in tqdm(models, desc="generating queries", leave=False)]
    return models, queries_per_model
