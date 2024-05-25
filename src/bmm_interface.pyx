# linux version (compiler directives for cython compiler)
# distutils: language = c++
# distutils: sources = tropical_bmm.cpp
# distutils: extra_compile_args = -O3 -ffast-math -march=native -fopenmp
# distutils: extra_link_args = -fopenmp
# cython: language_level = 3


import numpy as np

cdef extern from "tropical_bmm.hpp":
    struct forward_result:
        const float* matrix_batch
        const int* max_k
        const int b_size
        const int i_size
        const int j_size
        const int k_size
    
    struct backward_result:
        const float* grad_tensor_1
        const float* grad_tensor_2

    void free_forward_result(forward_result* result)

    void free_backward_result(backward_result* result)

    forward_result* forward(const float* tensor_1, const float* tensor_2, const int b_size, const int i_size, const int j_size, const int k_size)

    backward_result* backward(const float* grad_output, const int* max_k, const int b_size, const int i_size, const int j_size, const int k_size)

# (bik, bkj -> bij) in max-plus semiring
def forward_wrapper(float[:, :, :] tensor_1, float[:, :, :] tensor_2):
    assert tensor_1.shape[0] == tensor_2.shape[0]
    assert tensor_1.shape[2] == tensor_2.shape[1]
    cdef Py_ssize_t b_size = tensor_1.shape[0]
    cdef Py_ssize_t i_size = tensor_1.shape[1]
    cdef Py_ssize_t k_size = tensor_1.shape[2]
    cdef Py_ssize_t j_size = tensor_2.shape[2]
    cdef forward_result* result = forward(&tensor_1[0, 0, 0], &tensor_2[0, 0, 0], b_size, i_size, j_size, k_size)
    matrix_batch = np.empty((b_size, i_size, j_size), dtype=np.float32)
    max_k = np.empty((b_size, i_size, j_size), dtype=np.int32)
    matrix_batch[:, :, :] = <float[:b_size, :i_size, :j_size]>result[0].matrix_batch
    max_k[:, :, :] = <int[:b_size, :i_size, :j_size]>result[0].max_k
    free_forward_result(result)
    return matrix_batch, max_k


def backward_wrapper(float[:, :, :] grad_output, int[:, :, :] max_k, int k_size):
    cdef Py_ssize_t b_size = max_k.shape[0]
    cdef Py_ssize_t i_size = max_k.shape[1]
    cdef Py_ssize_t j_size = max_k.shape[2]

    grad_1 = np.zeros((b_size, i_size, k_size), dtype=np.float32)
    grad_2 = np.zeros((b_size, k_size, j_size), dtype=np.float32)
    cdef backward_result* result = backward(&grad_output[0, 0, 0], &max_k[0, 0, 0], b_size, i_size, j_size, k_size)
    grad_1 = np.empty((b_size, i_size, k_size), dtype=np.float32)
    grad_2 = np.empty((b_size, k_size, j_size), dtype=np.float32)
    grad_1[:, :, :] = <float[:b_size, :i_size, :k_size]>result[0].grad_tensor_1
    grad_2[:, :, :] = <float[:b_size, :k_size, :j_size]>result[0].grad_tensor_2
    free_backward_result(result)
    return grad_1, grad_2