import functools

import numpy as np
import opt_einsum as oe
import torch
from torch import Tensor

from contractions import max_plus_einsum, slice_tensor
# we're using bitsets because we want hashable, ordered sets of small integers
from my_bitsets import Bitset

Variables = Bitset
Evidence = tuple[int, ...]


class GraphicalModel:
    def __init__(self, n_variables: int, interaction_parameters: dict[Variables, Tensor]):
        """Graphical model class for inference with mpe queries.

        Parameters
        ----------
        n_variables : int
            Number of variables in the graphical model.
        interaction_parameters : dict[Variables, Tensor]
            Map from set of variables to log interaction tensor. The set of variables is represented as a bitset.
        """

        self.n_variables = n_variables
        self.interaction_parameters = interaction_parameters
        self.all_variables = Bitset.full_set(n_variables)
        self.symbols = [oe.get_symbol(i) for i in range(n_variables)]
        assert len(interaction_parameters) > 0, "There must be at least one interaction parameter."
        # figure out how big the axes are and if there are any incosistencies
        self.axis_sizes: dict[int, int] = {}
        for variables, tensor in interaction_parameters.items():
            axis_variables = list(variables)
            for axis, axis_size in enumerate(tensor.shape):
                axis_variable = axis_variables[axis]
                if axis_variable in self.axis_sizes:
                    assert self.axis_sizes[axis_variable] == axis_size, f"Axis {axis} of tensor {bin(variables)} has size {axis_size}, but the size of this variable was previously set to {self.axis_sizes[axis_variable]}."
                else:
                    self.axis_sizes[axis_variables[axis]] = axis_size
        # assert that every variable is actually used in the interaction parameters
        missing_variables = self.all_variables - functools.reduce(Bitset.union, interaction_parameters.keys())
        for variable in missing_variables:
            self.axis_sizes[variable] = 1

    def log_probability(self, full_evidence: Evidence) -> float:
        """Calculates the log probability of the evidence.

        Parameters
        ----------
        full_evidence : Evidence
            Evidence over all variables.

        Returns
        -------
        float
            Log probability of the evidence.
        """

        assert len(full_evidence) == self.n_variables, f"Expected evidence for all {self.n_variables} variables, but got {len(full_evidence)}."
        probability = 0
        for interaction_variables, interaction_tensor in self.interaction_parameters.items():
            sliced_evidence = tuple(full_evidence[i] for i in interaction_variables)
            probability += interaction_tensor[sliced_evidence].item()
        return probability

    def index_string(self, variables: Variables) -> str:
        return "".join(self.symbols[i] for i in range(self.n_variables) if i in variables)

    def build_maximizing_einsum(self, given_variables: Variables, given_evidence: Evidence) -> tuple[str, list[Tensor]]:
        """Builds the format string and the arguments for the einsum computation that maximizes the log probability over the unconditioned variables.

        Parameters
        ----------
        given_variables : Variables
            Variables fixed in the MPE query.
        given_evidence : Evidence
            Evidence that the variables are fixed to.

        Returns
        -------
        tuple[str, list[Tensor]]
            Format string and list of tensors to be used in the maximizing einsum.
        """

        assert len(given_variables) == len(given_evidence), f"Expected evidence to have the same length as given variables ({len(given_variables)}), but found {len(given_evidence)}."
        index_strings = []
        sliced_tensors = []
        for interaction_variables, interaction_tensor in self.interaction_parameters.items():
            sliced_variables = interaction_variables & given_variables
            # get the axes of the interaction tensor that correspond to the sliced variables
            sliced_axes = Bitset.indices(interaction_variables, sliced_variables)
            # get the evidence that corresponds to the sliced variables
            sliced_evidence_indices = Bitset.indices(given_variables, sliced_variables)
            sliced_evidence = tuple(given_evidence[i] for i in sliced_evidence_indices)
            # add the sliced tensor
            sliced_tensors.append(slice_tensor(interaction_tensor, sliced_axes, sliced_evidence))
            # add the corresponding index string
            index_strings.append(self.index_string(interaction_variables - sliced_variables))
        # add the axis tensors
        searched_variables = self.all_variables - given_variables
        axis_tensors = [torch.zeros(self.axis_sizes[variable], requires_grad=True) for variable in searched_variables]
        sliced_tensors.extend(axis_tensors)
        index_strings.extend(self.symbols[variable] for variable in searched_variables)
        # multiply the sliced tensors and maximize over the variables that are not observed
        format_string = ",".join(index_strings) + "->"
        return format_string, sliced_tensors

    def max_log_probability(self, given_variables: Variables, given_evidence: Evidence) -> float:
        """Computes the maximum log probability over the unconditioned variables, without computing the corresponding evidence.

        Parameters
        ----------
        given_variables : Variables
            Variables fixed in the MPE query.
        given_evidence : Evidence
            Evidence that the variables are fixed to.

        Returns
        -------
        float
            Maximum log probability over the unconditioned variables.
        """

        format_string, sliced_tensors = self.build_maximizing_einsum(given_variables, given_evidence)
        return max_plus_einsum(format_string, *sliced_tensors).item()

    def mpe(self, given_variables: Variables, given_evidence: Evidence) -> Evidence:
        """Answers the most probable explanation (MPE) query.

        Parameters
        ----------
        given_variables : Variables
            Variables fixed in the MPE query.
        given_evidence : Evidence
            Evidence that the variables are fixed to.

        Returns
        -------
        Evidence
            Complete evidence over all variables, where the given variables are given to the observed evidence, that has the highest (log) probability.
        """

        assert len(given_variables) == len(given_evidence), f"Expected evidence to have the same length as given variables ({len(given_variables)}), but found {len(given_evidence)}."
        # if there is nothing to optimize, the given evidense is the most probable evidence.
        if len(given_variables) == self.n_variables:
            return given_evidence
        format_string, sliced_tensors = self.build_maximizing_einsum(given_variables, given_evidence)
        max_log_probability = max_plus_einsum(format_string, *sliced_tensors)
        # differentiate the max probability to get the MPE
        max_log_probability.backward()
        searched_variables = searched_variables = self.all_variables - given_variables
        axis_tensors = sliced_tensors[-len(searched_variables):]
        argmax_indices = [torch.argmax(tensor.grad).item() for tensor in axis_tensors]
        # complete the argmax with the observed evidence
        argmax_list = [0] * self.n_variables
        for i, variable in enumerate(searched_variables):
            argmax_list[variable] = argmax_indices[i]
        for i, variable in enumerate(given_variables):
            argmax_list[variable] = given_evidence[i]
        argmax = tuple(argmax_list)
        assert np.isclose(self.log_probability(argmax), max_log_probability.item()), f"Probability of the MPE is not equal to the maximum probability: {self.log_probability(argmax)} (probability at argmax) != {max_log_probability.item()} (max probability). found argmax: {argmax}"
        return argmax


def main():
    # make a hidden markov model:
    # o - o - o
    # |   |   |
    # o   o   o
    # variables:
    # 0 - 1 - 2
    # |   |   |
    # 3   4   5
    # interaction parameters:
    # o 1 o 2 o
    # 3   4   5
    # o   o   o
    axis_sizes = [3, 4, 5, 6, 7, 8]
    tensor_axes = [
        (0, 1),
        (1, 2),
        (0, 3),
        (1, 4),
        (2, 5)
    ]
    tensor_shapes = [(axis_sizes[i], axis_sizes[j]) for i, j in tensor_axes]
    torch.manual_seed(0)
    tensors = [torch.rand(*shape) for shape in tensor_shapes]
    n_variables = len(axis_sizes)
    interaction_parameters = {Bitset(axes): tensor for axes, tensor in zip(tensor_axes, tensors)}
    gm = GraphicalModel(n_variables, interaction_parameters)
    observed_variables = Bitset([0, 5])
    evidence = (2, 6)
    mpe = gm.mpe(observed_variables, evidence)
    # compare the mpe result to the torch argmax
    combination = torch.einsum("ij,jk,il,jm,kn->ijklmn", *tensors)[2, :, :, :, :, 6]
    torch_argmax_incomplete = (combination == combination.max()).nonzero()[0]
    torch_argmax = [None] * n_variables
    searched_variables = gm.all_variables - observed_variables
    for i, variable in enumerate(searched_variables):
        torch_argmax[variable] = torch_argmax_incomplete[i]
    for i, variable in enumerate(observed_variables):
        torch_argmax[variable] = evidence[i]
    assert mpe == tuple(torch_argmax)


if __name__ == "__main__":
    main()
