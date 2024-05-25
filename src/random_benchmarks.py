import torch
from tqdm import tqdm

from graphical_model import Evidence, GraphicalModel, Variables
from my_bitsets import Bitset

Query = tuple[Variables, Evidence]


SEED = 0

DEFAULT_MIN_CHAIN_LENGTH = 20
DEFAULT_MAX_CHAIN_LENGTH = 400
DEFAULT_CHAIN_LENGTH_STEP = 20

DEFAULT_AXIS_SIZE = 64
DEFAULT_MIN_N_GIVEN = 0
DEFAULT_MAX_N_GIVEN = 200
DEFAULT_N_GIVEN_STEP = 50
# number of queries that are randomly chosen for each possible number of variables
DEFAULT_QUERIES_PER_NUMBER_OF_VARIABLES = 3


def gen_hidden_markov_model(chain_length: int, axis_size: int = DEFAULT_AXIS_SIZE, auto_seed=True) -> GraphicalModel:
    """Generates a tensor train hidden markov model. The model consists of n - 1 hidden variables and n observed variables. The hidden variables are connected in a chain, and the observed variables are connected two hidden variables each. The interaction parameters are randomly initialized.

    Parameters
    ----------
    chain_length : int
        Number of tensors in the chain.
    axis_size : int, optional
        Axis size that is common for all the tensors, by default `DEFAULT_AXIS_SIZE`.
    auto_seed : bool, optional
        Whether to automatically seed with `SEED` before generating the model, by default True.

    Returns
    -------
    GraphicalModel
        Tensor train hidden markov model.
    """

    if auto_seed:
        torch.manual_seed(SEED)
    # tensor train looks like this:
    # o - o - o
    # |   |   |
    # where o is a tensor, - is a hidden variable shared between neighbouring tensors, and | is an observed variable
    # n - 1 hidden variables: 0 ... chain_length - 2
    # n observed variables: chain_length - 1 ... 2 * chain_length - 2
    interaction_parameters = {}
    for i in range(chain_length):
        # left hidden variable, observed variable, right hidden variable
        if i == 0:
            variables = Bitset([chain_length - 1, i])
        elif i == chain_length - 1:
            variables = Bitset([i - 1, 2 * chain_length - 2])
        else:
            variables = Bitset([i - 1, chain_length + i - 1, i])
        interaction_parameters[variables] = torch.rand((axis_size,) * len(variables))
    return GraphicalModel(2 * chain_length - 1, interaction_parameters)


def gen_queries(model: GraphicalModel, min_n_given: int = DEFAULT_MIN_N_GIVEN, max_n_given: int = DEFAULT_MAX_N_GIVEN, n_given_step: int = DEFAULT_N_GIVEN_STEP, queries_per_number_of_variables: int = DEFAULT_QUERIES_PER_NUMBER_OF_VARIABLES, auto_seed=True) -> list[Query]:
    """Generates queries for a specific model. A query consists of a set of given variables and corresponding evidence.

    Parameters
    ----------
    model : GraphicalModel
        Graphical model for which the queries are generated.
    min_n_given : int, optional
        Minimum number of variables that is conditioned on, by default DEFAULT_MIN_N_GIVEN.
    max_n_given : int, optional
        Maximum number of variables that is conditioned on, by default DEFAULT_MAX_N_GIVEN.
    n_given_step : int, optional
        Step size for picking the number of variables that is conditioned on, by default DEFAULT_N_GIVEN_STEP.
    queries_per_number_of_variables : int, optional
        Number of random queries generated for each picked number of variables, by default DEFAULT_QUERIES_PER_NUMBER_OF_VARIABLES.
    auto_seed : bool, optional
        Whether to automatically seed with `SEED` once before generating the queries, by default True.

    Returns
    -------
    list[Query]
        Random tuples of given variables and evidence.
    """

    if auto_seed:
        torch.manual_seed(SEED)
    queries = []
    max_n_given = min(max_n_given, model.n_variables)
    for n_given in range(min_n_given, max_n_given + 1, n_given_step):
        for _ in range(queries_per_number_of_variables):
            # choose random given variables
            given_variables = Bitset(torch.randperm(model.n_variables)[:n_given].tolist())
            # assign these random variables with random values
            axis_sizes = [model.axis_sizes[variable] for variable in given_variables]
            evidence = tuple([torch.randint(0, axis_size, (1, )).item() for axis_size in axis_sizes])
            queries.append((given_variables, evidence))
    return queries


def gen_benchmarks(min_chain_length: int = DEFAULT_MIN_CHAIN_LENGTH, max_chain_length: int = DEFAULT_MAX_CHAIN_LENGTH, chain_length_step: int = DEFAULT_CHAIN_LENGTH_STEP, axis_size: int = DEFAULT_AXIS_SIZE, min_n_given: int = DEFAULT_MIN_N_GIVEN, max_n_given: int = DEFAULT_MAX_N_GIVEN, n_given_step: int = DEFAULT_N_GIVEN_STEP, queries_per_number_of_variables: int = DEFAULT_QUERIES_PER_NUMBER_OF_VARIABLES, auto_seed=True) -> tuple[list[GraphicalModel], list[list[Query]]]:
    """Generates random tensor train graphical models and random queries for each model.

    Parameters
    ----------
    min_chain_length : int, optional
        Minimum number of tensors in a model, by default DEFAULT_MIN_CHAIN_LENGTH.
    max_chain_length : int, optional
        Maximum number of tensors in a model, by default DEFAULT_MAX_CHAIN_LENGTH.
    chain_length_step : int, optional
        Step size for the number of tensors in the models, by default DEFAULT_CHAIN_STEP.
    axis_size : int, optional
        Axis size that is common for all tensors in a model, by default `DEFAULT_AXIS_SIZE`.
    min_n_given : int, optional
        Minimum number of variables that is conditioned on, by default DEFAULT_MIN_N_GIVEN.
    max_n_given : int, optional
        Maximum number of variables that is conditioned on, by default DEFAULT_MAX_N_GIVEN.
    n_given_step : int, optional
        Step size for picking the number of variables that is conditioned on, by default DEFAULT_N_GIVEN_STEP.
    queries_per_number_of_variables : int, optional
        Number of random queries generated for each picked number of variables, by default DEFAULT_QUERIES_PER_NUMBER_OF_VARIABLES.
    auto_seed : bool, optional
        Whether to automatically seed with `SEED` before every generated model and before generating the queries for that model, by default True.

    Returns
    -------
    tuple[list[GraphicalModel], list[list[Query]]]
        List of generated graphical models and a list of queries for each model.
    """

    chain_lengths = range(min_chain_length, max_chain_length + 1, chain_length_step)
    models = [gen_hidden_markov_model(chain_length, axis_size=axis_size, auto_seed=auto_seed) for chain_length in tqdm(chain_lengths, desc="generating models", leave=False)]
    queries_per_model = [gen_queries(model, min_n_given=min_n_given, max_n_given=max_n_given, n_given_step=n_given_step, queries_per_number_of_variables=queries_per_number_of_variables) for model in tqdm(models, desc="generating queries", leave=False)]
    return models, queries_per_model
