/*
The GPT-2 Encoder, which combines two encodings: token and position
In the forward pass, both encodings are added together
In the backward pass, the gradients flow to both, handled by different kernels
*/
#include <assert.h>
#include <stdint.h>
#include <utility>              // std::pair
#include <vector>
#include <algorithm>
#include <unordered_map>
// llmc internal imports
#include "cuda_common.h"
#include "cuda_utils.cuh"

// ----------------------------------------------------------------------------
// CUDA kernels

// ----------------------------------------------------------------------------
// kernel launchers
// mapping cls token to classes

void encoder_forward(floatX* out,
                     const int* inp, const floatX* classifierw, const floatX* classifierb,
                     int B, int C, int num_classes, cudaStream_t stream)
{
    // inp: (B, 1, C) -> (B, 1, num_classes)
    matmul_forward_cublaslt(out, inp, classifierw, classifierb, B, 1, C, num_classes, stream);
}