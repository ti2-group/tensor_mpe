import torch
from torch import Tensor
from collections import Counter
from typing import Callable
import opt_einsum as oe
from tropical_bmm import tropical_pairwise_einsum


def slice_tensor(tensor: Tensor, axes: list[int], slice_multi_index: tuple[int, ...]) -> Tensor:
    assert len(axes) == len(slice_multi_index)
    slicing_expression = [slice(None)] * len(tensor.shape)
    for axis, value in zip(axes, slice_multi_index):
        slicing_expression[axis] = value
    return tensor[tuple(slicing_expression)]


def standard_pairwise_einsum(input_string_1: str, input_string_2: str, output_string: str, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    return torch.einsum(f"{input_string_1},{input_string_2}->{output_string}", tensor_1, tensor_2)


def contract_along_path(format_string: str, *args: Tensor, pairwise_einsum: Callable[[str, str, str, Tensor, Tensor], Tensor] = standard_pairwise_einsum, path=None) -> Tensor:
    """Computes einsum with a contraction path given by opt_einsum, but a custom pairwise einsum can be given, in order to support different semirings.

    Parameters
    ----------
    format_string : str
        Einsum format string.
    pairwise_einsum : Callable[[str, str, str, str, Tensor, Tensor], Tensor], optional
        An efficient implementation of einsum over 2 input tensors.

    Returns
    -------
    Tensor
        Result of the einsum operation (depends on the semiring used in `pairwise_einsum`).
    """

    assert len(args) >= 2, "At least 2 tensors are needed for my einsum."
    # prepare intermediate tensors and intermediate index strings
    intermediate_tensors = list(args)
    input_strings, output_string = format_string.split("->")
    intermediate_index_strings = input_strings.split(",")
    # prepare symbol histogram
    symbol_histogram: Counter[str] = Counter("".join(intermediate_index_strings))
    symbol_histogram.update(output_string)
    # find the optimal contraction path
    if path is None:
        path, _ = oe.contract_path(format_string, *args)
    # contract the tensors in the computed order (except for the last contraction)
    for contracted_elements in path[:-1]:
        # pop the contracted tensors and index strings in reverse order so it doesn't mess up the indices
        contracted_tensors = [intermediate_tensors[i] for i in contracted_elements]
        intermediate_tensors = [intermediate_tensor for i, intermediate_tensor in enumerate(intermediate_tensors) if i not in contracted_elements]
        contracted_index_strings = [intermediate_index_strings[i] for i in contracted_elements]
        intermediate_index_strings = [intermediate_index_string for i, intermediate_index_string in enumerate(intermediate_index_strings) if i not in contracted_elements]
        # look what symbols are left after contracting these tensors
        contracted_symbol_histogram = Counter("".join(contracted_index_strings))
        # substract the contracted symbols from the symbol histogram, and aggregate over the symbols which aren't found anywhere else
        symbol_histogram.subtract(contracted_symbol_histogram)
        kept_symbols = "".join(symbol for symbol in contracted_symbol_histogram.keys() if symbol_histogram[symbol] > 0)
        # add the kept symbols to the symbol histogram again
        symbol_histogram.update(kept_symbols)
        # combine the tensors and then aggregate over the aggregated symbols
        contracted_tensor = pairwise_einsum(*contracted_index_strings, kept_symbols, *contracted_tensors)
        # add the contracted tensor and index string to the intermediates
        intermediate_tensors.append(contracted_tensor)
        intermediate_index_strings.append(kept_symbols)
    # contract the last tensors
    return pairwise_einsum(*intermediate_index_strings, output_string, *intermediate_tensors)


def max_plus_einsum(format_string: str, *args: Tensor, path=None) -> Tensor:
    return contract_along_path(format_string, *args, pairwise_einsum=tropical_pairwise_einsum, path=path)


def main():
    # have some fun with the gradients on max
    torch.manual_seed(0)
    x = torch.rand(2, 3)
    y = torch.rand(3, 4)
    axis_1 = torch.zeros(2, requires_grad=True)
    axis_2 = torch.zeros(3, requires_grad=True)
    axis_3 = torch.zeros(4, requires_grad=True)
    max_element = contract_along_path("ik,kj,i,k,j->", x, y, axis_1, axis_2, axis_3, pairwise_einsum=tropical_pairwise_einsum)
    max_element.backward()
    combination = contract_along_path("ik,kj->ikj", x, y, pairwise_einsum=tropical_pairwise_einsum)
    print((combination == max_element).nonzero())
    print(combination)
    print(axis_1.grad)
    print(axis_2.grad)
    print(axis_3.grad)
    print(max_element)


if __name__ == "__main__":
    main()
