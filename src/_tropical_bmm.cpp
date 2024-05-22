#include "_tropical_bmm.h"
#include <cstdlib>
#include <omp.h>
#include <limits>

void free_forward_result(forward_result* result) {
    free(const_cast<float*> (result->matrix_batch));
    free(const_cast<int*> (result->max_k));
    delete result;
}

void free_backward_result(backward_result* result) {
    free(const_cast<float*> (result->grad_tensor_1));
    free(const_cast<float*> (result->grad_tensor_2));
    delete result;
}

constexpr float LOWEST_VALUE = -std::numeric_limits<float>::max();

forward_result* forward(const float* tensor_1, const float* tensor_2, const int b_size, const int i_size, const int j_size, const int k_size) {
    auto matrix_batch = (float*) calloc(b_size * i_size * j_size, sizeof(float));
    auto max_k = (int*) calloc(b_size * i_size * j_size, sizeof(int));
    bool is_big_enough = b_size * i_size * j_size > 10'000;
#pragma omp parallel for shared(matrix_batch, max_k, tensor_1, tensor_2) collapse(2) if(is_big_enough)
    for (int b = 0; b < b_size; ++b) {
        for (int i = 0; i < i_size; ++i) {
            for (int j = 0; j < j_size; ++j) {
                float max_value = LOWEST_VALUE;
                int max_index = 0;
                for (int k = 0; k < k_size; ++k) {
                    float new_value = tensor_1[b * i_size * k_size + i * k_size + k] + tensor_2[b * k_size * j_size + k * j_size + j];
                    int is_bigger = new_value > max_value;
                    max_value = is_bigger * new_value + (1 - is_bigger) * max_value;
                    max_index = is_bigger * k + (1 - is_bigger) * max_index;
                }
                matrix_batch[b * i_size * j_size + i * j_size + j] = max_value;
                max_k[b * i_size * j_size + i * j_size + j] = max_index;
            }
        }
    }
    return new forward_result{ matrix_batch, max_k };
}

backward_result* backward(const float* grad_output, const int* max_k, const int b_size, const int i_size, const int j_size, const int k_size) {
    auto grad_tensor_1 = (float*) calloc(b_size * i_size * k_size, sizeof(float));
    auto grad_tensor_2 = (float*) calloc(b_size * k_size * j_size, sizeof(float));
#pragma omp parallel for shared(max_k, grad_tensor_1, grad_tensor_2)
    for (int b = 0; b < b_size; ++b) {
        for (int i = 0; i < i_size; ++i) {
            for (int j = 0; j < j_size; ++j) {
                int k = max_k[b * i_size * j_size + i * j_size + j];
                grad_tensor_1[b * i_size * k_size + i * k_size + k] += grad_output[b * i_size * j_size + i * j_size + j];
                grad_tensor_2[b * k_size * j_size + k * j_size + j] += grad_output[b * i_size * j_size + i * j_size + j];
            }
        }
    }
    return new backward_result{ grad_tensor_1, grad_tensor_2 };
}