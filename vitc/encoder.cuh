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

// embeddings = inp + position_embedding
__global__ void encoder_forward_kernel3(floatX* out,
                                        const floatX* inp,
                                        const floatX* ppe,
                                        int B, int T, int C) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int N = B * T * C;
    if(idx >= N)return;

    int bt = idx / C;
    int t = bt % T;
    int c = idx % C;

    floatX* out_ptr = out + idx;
    const floatX* inp_ptr = inp + idx;
    const floatX* ppe_ptr = ppe + t * C + c;

    x128 inp_vec = load128cs(inp_ptr);   // Load input embeddings
    x128 ppe_vec = load128cs(ppe_ptr);   // Load position embeddings
    x128 out_vec;

    // Compute out = inp + ppe
#pragma unroll
    for(int k = 0; k < x128::size; k++){
        out_vec[k] = (floatX)((float)inp_vec[k] + (float)ppe_vec[k]);
    }

    store128(out_ptr, out_vec);  // Store the result
    
}

// Modified PTE backward kernel to compute projection matrix gradient
__global__ void pte_backward_kernel(floatX* dpte,
                                    const floatX* dout,  // Gradient from the next layer
                                    const floatX* inp,   // Input patches (floating-point)
                                    int B, int P, int C, unsigned int seed) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size; // Global thread index
    if (idx >= C * C) return;

    int i = idx / C; // Output channel index (0 <= i < C)
    int j = idx % C; // Input channel index (0 <= j < C)
    float accum[x128::size] = {0.0f};

    for (int b = 0; b < B; b++) {
        for (int p = 0; p < P; p++) {
            // dout has shape [B, P+1, C], skip the CLS token at position 0
            x128 packed_dout = load128cs(dout + b * (P + 1) * C + (p + 1) * C + i);
            x128 packed_inp = load128cs(inp + b * P * C + p * C + j);
            // float dout_val = static_cast<float>(dout[b * (P + 1) * C + (p + 1) * C + i]);
            // float inp_val = static_cast<float>(inp[b * P * C + p * C + j]);
            // accum += inp_val * dout_val;
            for (int k = 0; k < x128::size; k++) {
                accum[k] += static_cast<float>(packed_inp[k]) * static_cast<float>(packed_dout[k]);
            }
        }
    }
    
    floatX* dpte_ij = dpte + (i * C) + j;
    x128 packed_dwpe = load128(dpte_ij);

    for (int k = 0; k < x128::size; k++) {
        stochastic_rounding(accum[k] + static_cast<float>(packed_dwpe[k]), &packed_dwpe[k], seed + idx + k);
    }

    store128(dpte_ij, packed_dwpe);
}

__global__ void cls_backward_kernel(floatX* dcls,
                               const floatX* dout,
                               int B, int P, int C) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x; // Global thread index
    int total_elements = C; // Total elements in dcls

    if (idx >= total_elements) return;

    int i = idx; // Output channel index (0 <= i < C)

    float accum = 0.0f;

    for (int b = 0; b < B; b++) {
        for (int p = 0; p < P; p++) {
            // dout has shape [B, P+1, C], skip the CLS token at position 0
            float dout_val = static_cast<float>(dout[b * (P + 1) * C + (p + 1) * C + i]);
            accum += dout_val;
        }
    }

    // Update dcls[i]
    dcls[i] = (floatX)((float)dcls[i] + accum);
}

// Modified PPE backward kernel to exclude CLS token if needed
__global__ void ppe_backward_kernel(floatX* dppe,
                                    const floatX* dout,
                                    int B, int P, int C, unsigned int seed) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * x128::size;
    int T = P + 1; // Total tokens including CLS token
    if (idx >= P * C) { return; } // Exclude CLS token if not using position embedding for CLS

    int p = idx / C; // Patch index (excluding CLS token)
    int c = idx % C;

    float accum[x128::size] = {0.0f};

    for (int b = 0; b < B; b++) {
        // Offset by +1 to skip CLS token
        x128 dout_vec = load128cs(dout + b * T * C + (p + 1) * C + c);
        for (int k = 0; k < x128::size; k++) {
            accum[k] += static_cast<float>(dout_vec[k]);
        }
    }

    // Update dppe (gradient w.r.t. positional embeddings)
    floatX* dppe_ptr = dppe + (p + 1) * C + c; // Offset by +1 if CLS token does not use positional embedding
    x128 dppe_vec = load128(dppe_ptr);
    for (unsigned int k = 0; k < x128::size; k++) {
        stochastic_rounding(accum[k] + static_cast<float>(dppe_vec[k]), &dppe_vec[k], seed + idx + k);
    }
    store128(dppe_ptr, dppe_vec);
}

// ----------------------------------------------------------------------------
// kernel launchers
// encoder_forward(acts.encoded, model->inputs, params.ptew, params.pteb, params.ppe, B, P, C, main_stream); // encoding goes into residual[0]

void encoder_forward(floatX* out,
                     const floatX* inp, const floatX* cls, const floatX* ppe, // gpu inputs
                     int B, int P, int C, cudaStream_t stream) {
    NVTX_RANGE_FN();

    // embeddings = cat(cls_tokens, embeddings)
    cudaCheck(cudaMemcpyAsync(out, cls, B * 1 * C * sizeof(floatX), cudaMemcpyDeviceToDevice, stream));

    // embeddings = embeddings + position_embedding
    const int block_size = 256;
    const int grid_size = CEIL_DIV(B * (P + 1) * C, (int)(block_size * x128::size));
    encoder_forward_kernel3<<<grid_size, block_size, 0, stream>>>(out, out, ppe, B, P + 1, C);
    cudaCheck(cudaGetLastError());
}


void encoder_backward(floatX* dcls, floatX* dpte, floatX* dppe, floatX* scratch, // GPU outputs & scratch
                      const floatX* dout, const floatX* inp,       // GPU inputs
                      int B, int P, int C, unsigned int seed, cudaStream_t stream) {
    NVTX_RANGE_FN();

    const int block_size = 256;
    
    // Launch PTE backward kernel
    int N_pte = CEIL_DIV(C * C, x128::size);
    int grid_size_pte = CEIL_DIV(N_pte, block_size);
    pte_backward_kernel<<<grid_size_pte, block_size, 0, stream>>>(dpte, dout, inp, B, P, C, seed);

    // Launch CLS backward kernel
    int grid_size_cls = CEIL_DIV(C, block_size);
    cls_backward_kernel<<<grid_size_cls, block_size, 0, stream>>>(dcls, dout, B, P, C);

    // Launch PPE backward kernel
    int N_ppe = CEIL_DIV(P * C, x128::size);
    int grid_size_ppe = CEIL_DIV(N_ppe, block_size);
    ppe_backward_kernel<<<grid_size_ppe, block_size, 0, stream>>>(dppe, dout, B, P, C, seed);
    cudaCheck(cudaGetLastError());
}