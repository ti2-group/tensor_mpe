import numpy as np
import torch
from torch import Tensor

from bmm_interface import backward_wrapper, forward_wrapper


class _TropicalBatchMatrixMultiplcation(torch.autograd.Function):
    """Computes the bmm (bik, bkj -> bij; a, b) over the tropical (max-plus) semiring."""

    @staticmethod
    def forward(ctx, tensor_1: Tensor, tensor_2: Tensor):
        # the tensors are not always contiguous because of the permutations etc. we need to make them contiguous for the c++ code.
        result, max_k = forward_wrapper(np.ascontiguousarray(tensor_1.numpy()), np.ascontiguousarray(tensor_2.numpy()))
        ctx.k_size = tensor_1.shape[2]
        ctx.save_for_backward(torch.from_numpy(max_k))
        return torch.from_numpy(result)

    @staticmethod
    def backward(ctx, grad_output):
        max_k = ctx.saved_tensors[0]
        # we need to make grad_output contiguous because all the permutations etc mess with its strides. max_k is made by us and therefore already contiguous.
        grad_1, grad_2 = backward_wrapper(np.ascontiguousarray(grad_output.numpy()), max_k.numpy(), ctx.k_size)
        return torch.from_numpy(grad_1), torch.from_numpy(grad_2)


def _to_bmm(input_string_1: str, input_string_2: str, output_string: str, tensor_1: Tensor, tensor_2: Tensor) -> tuple[Tensor, Tensor, str]:
    """Maps any tensor contraction to a batch matrix multiplication. Returns *transposed* and reshaped tensors and the output string of the transposed result tensor before reshaping."""

    # figure out the types of symbols
    input_set_1 = set(input_string_1)
    input_set_2 = set(input_string_2)
    ouput_set = set(output_string)
    batch_symbols = input_set_1 & input_set_2 & ouput_set
    contracted_symbols_common = (input_set_1 & input_set_2) - ouput_set
    contracted_symbols_tensor_1 = input_set_1 - (input_set_2 | ouput_set)
    contracted_symbols_tensor_2 = input_set_2 - (input_set_1 | ouput_set)
    kept_symbols_tensor_1 = (input_set_1 & ouput_set) - batch_symbols
    kept_symbols_tensor_2 = (input_set_2 & ouput_set) - batch_symbols
    # assert there are no symbols which could have easily been contracted earlier (if an input symbol is only contracted in one tensor, then it should have been contracted before this bmm)
    assert len(contracted_symbols_tensor_1) == 0, f"found symbols which should have been contracted earlier: {contracted_symbols_tensor_1}"
    assert len(contracted_symbols_tensor_2) == 0, f"found symbols which should have been contracted earlier: {contracted_symbols_tensor_2}"
    # *transpose* the first tensor to "batch_symbols kept_symbols contracted_symbols" and the second to "batch_symbols contracted_symbols kept_symbols"
    # first step: find the permutations necessary to transform the old input strings to the bmm input strings
    bmm_input_string_1 = "".join(batch_symbols) + "".join(kept_symbols_tensor_1) + "".join(contracted_symbols_common)
    bmm_input_string_2 = "".join(batch_symbols) + "".join(contracted_symbols_common) + "".join(kept_symbols_tensor_2)
    bmm_output_string = "".join(batch_symbols) + "".join(kept_symbols_tensor_1) + "".join(kept_symbols_tensor_2)
    axis_permutation_tensor_1 = [input_string_1.index(symbol) for symbol in bmm_input_string_1]
    axis_permutation_tensor_2 = [input_string_2.index(symbol) for symbol in bmm_input_string_2]
    tensor_1_new = tensor_1
    tensor_2_new = tensor_2
    # second step: use the permutations found for the input string to permute the axes
    if len(axis_permutation_tensor_1) > 1:
        tensor_1_new = tensor_1_new.permute(*axis_permutation_tensor_1)
    if len(axis_permutation_tensor_2) > 1:
        tensor_2_new = tensor_2_new.permute(*axis_permutation_tensor_2)
    # third step: flatten the axes used for batch index, the contracted index, and the kept axis. if one of them didn't exist before, add a new axis of size one
    # flatten batch dimension
    if len(batch_symbols) > 0:
        tensor_1_new = tensor_1_new.flatten(start_dim=0, end_dim=len(batch_symbols) - 1)
        tensor_2_new = tensor_2_new.flatten(start_dim=0, end_dim=len(batch_symbols) - 1)
    else:
        tensor_1_new = tensor_1_new.unsqueeze(0)
        tensor_2_new = tensor_2_new.unsqueeze(0)
    # for tensor_1: the kept axis comes before the contracted axis
    if len(kept_symbols_tensor_1) > 0:
        tensor_1_new = tensor_1_new.flatten(start_dim=1, end_dim=len(kept_symbols_tensor_1))
    else:
        tensor_1_new = tensor_1_new.unsqueeze(1)
    if len(contracted_symbols_common) > 0:
        tensor_1_new = tensor_1_new.flatten(start_dim=2, end_dim=1 + len(contracted_symbols_common))
    else:
        tensor_1_new = tensor_1_new.unsqueeze(2)
    # for tensor_2: the kept axis comes after the contracted axis
    if len(contracted_symbols_common) > 0:
        tensor_2_new = tensor_2_new.flatten(start_dim=1, end_dim=len(contracted_symbols_common))
    else:
        tensor_2_new = tensor_2_new.unsqueeze(1)
    if len(kept_symbols_tensor_2) > 0:
        tensor_2_new = tensor_2_new.flatten(start_dim=2, end_dim=1 + len(kept_symbols_tensor_2))
    else:
        tensor_2_new = tensor_2_new.unsqueeze(2)
    assert len(tensor_1_new.shape) == len(tensor_2_new.shape) == 3, f"Expected 3 dimensions, got {tensor_1_new.shape} and {tensor_2_new.shape}, originally {tensor_1.shape} and {tensor_2.shape}, with index strings ({input_string_1},{input_string_2}->{output_string}), lengths: ({len(input_string_1)}, {len(input_string_2)} -> {len(output_string)})."
    return tensor_1_new, tensor_2_new, bmm_output_string


def _from_bmm_result(wanted_index_string: str, bmm_output_string: str, bmm_output: Tensor, size_dict: dict[str, int]) -> Tensor:
    """Unflattens the tensor to transposed index string, and then transposes it to the wanted index string."""

    unflattened_shape = tuple([size_dict[symbol] for symbol in bmm_output_string])
    if len(unflattened_shape) == 0:
        return bmm_output.view(())
    unflattened_bmm_output = bmm_output.view(*unflattened_shape)
    wanted_permutation = [bmm_output_string.index(symbol) for symbol in wanted_index_string]
    return unflattened_bmm_output.permute(*wanted_permutation)


def tropical_pairwise_einsum(input_string_1: str, input_string_2: str, output_string: str, tensor_1: Tensor, tensor_2: Tensor) -> Tensor:
    """Computes the tensor contraction of two tensors with einsum over the tropical (max-plus) semiring."""

    size_dict = {symbol: size for symbol, size in zip(input_string_1 + input_string_2, tensor_1.shape + tensor_2.shape)}
    reshaped_tensor_1, reshaped_tensor_2, bmm_output_string = _to_bmm(input_string_1, input_string_2, output_string, tensor_1, tensor_2)
    bmm_output = _TropicalBatchMatrixMultiplcation.apply(reshaped_tensor_1, reshaped_tensor_2)
    return _from_bmm_result(output_string, bmm_output_string, bmm_output, size_dict)
