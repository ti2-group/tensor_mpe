#pragma once

// result of (bik, bkj -> bij)
struct forward_result {
    // shape: bij
    const float* matrix_batch;
    // shape: bij
    const int* max_k;
};

struct backward_result {
    // shape: bik
    const float* grad_tensor_1;
    // shape: bkj
    const float* grad_tensor_2;
};

void free_forward_result(forward_result* result);

void free_backward_result(backward_result* result);

forward_result* forward(const float* tensor_1, const float* tensor_2, const int b_size, const int i_size, const int j_size, const int k_size);

backward_result* backward(const float* grad_output, const int* max_k, const int b_size, const int i_size, const int j_size, const int k_size);