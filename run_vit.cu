/*
Vision Transformer Neural Net training loop. See README.md for usage.
*/
#include <unistd.h>
#include <cassert>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string>
#include <string_view>
#include <sys/stat.h>
#include <sys/types.h>

// ----------- CPU utilities -----------
// defines: fopenCheck, freadCheck, fcloseCheck, fseekCheck, mallocCheck
// defines: create_dir_if_not_exists, find_max_step, ends_with_bin
#include "vitc/utils.h"
// defines: tokenizer_init, tokenizer_decode, tokenizer_free
#include "vitc/tokenizer.h"
// defines: dataloader_init, dataloader_reset, dataloader_next_batch, dataloader_free
// defines: evalloader_init, evalloader_reset, evalloader_next_batch, evalloader_free
#include "vitc/dataloader.h"
// defines: manual_seed, normal_ (same as torch.manual_seed and torch.normal)
#include "vitc/rand.h"
// defines: lr_scheduler_init, get_learning_rate
#include "vitc/schedulers.h"
// defines: sample_softmax, random_f32
#include "vitc/sampler.h"
// defines: logger_init, logger_log_eval, logger_log_val, logger_log_train
#include "vitc/logger.h"
// defines: get_flops_promised
#include "vitc/mfu.h"
// defines: OutlierDetector, init_detector, update_detector
#include "vitc/outlier_detector.h"
// ----------- GPU utilities -----------
// defines:
// WARP_SIZE, MAX_1024_THREADS_BLOCKS, CEIL_DIV, cudaCheck, PRECISION_MODE
// NVTX_RANGE_FN
#include "vitc/cuda_common.h"
// defines:
// Packed128, f128, x128
// warpReduceSum, warpReduceMax, blockReduce, copy_and_cast_kernel, cudaMallocConditionallyManaged
#include "vitc/cuda_utils.cuh"
// defines: CUBLAS_LOWP, cublasCheck, cublaslt_workspace_size, cublaslt_workspace
// defines: cublas_compute, cublaslt_handle, cublas_handle
#include "vitc/cublas_common.h"
// ----------- Layer implementations in CUDA -----------
// defines: encoder_forward, encoder_backward
#include "vitc/encoder.cuh"
// defines: layernorm_forward, residual_forward, fused_residual_forward5, layernorm_backward
#include "vitc/layernorm.cuh"
// defines: matmul_cublaslt, matmul_forward, matmul_backward, gelu_forward, gelu_backward_inplace
#include "vitc/matmul.cuh"
#ifdef ENABLE_CUDNN
// defines: create_cudnn, destroy_cudnn, attention_forward_cudnn, attention_backward_cudnn
#include "vitc/cudnn_att.h"
#else
// defines: attention_forward, attention_backward
#include "vitc/attention.cuh"
#endif
// defines: fused_classifier
#include "vitc/fused_classifier.cuh"
// defines: adamw_kernel3
#include "vitc/adamw.cuh"
// defines: global_norm_squared
#include "vitc/global_norm.cuh"

// ----------- Multi-GPU support -----------
// defines: ncclFloatX, ncclCheck, MultiGpuConfig, ShardInfo
// defines: printf0, multi_gpu_config
// defines: multi_gpu_config_init, multi_gpu_config_free
// defines: set_zero_configs, multi_gpu_cpu_float_sum, multi_gpu_barrier
// defines: multi_gpu_get_shard_offset, multi_gpu_async_reduce_gradient
#include "vitc/zero.cuh"

// ----------------------------------------------------------------------------
// global vars for I/O
char filename_buffer[512];

// ----------------------------------------------------------------------------
// global vars containing information about the GPU this process is running on
cudaDeviceProp deviceProp; // fills in common_start()
cudaStream_t main_stream;
// buffer size to use for device <-> disk io
constexpr const size_t IO_BUF_SIZE = 32 * 1024 * 1024;

// ----------------------------------------------------------------------------
// Vision Transformer model definition

typedef struct {
    int num_classes; // num_classes, e.g. 1000
    int num_patches; // number of patches, e.g. 14*14
    int num_layers; // number of layers, e.g. 12
    int num_heads; // number of heads in attention, e.g. 12
    int channels; // number of channels, e.g. 768
} ViTConfig;

// the parameters of the model
constexpr const int NUM_PARAMETER_TENSORS = 20;
typedef struct {
    floatX* cls; // (C)
    floatX* ppe; // (P+1, C) // patch position embedding // No transpose?
    floatX* ptew; // (C, C) // patch token embedding weight
    floatX* pteb; // (C) // patch token embedding bias
    floatX* ln1w; // (L, C) // layernorm_before weight
    floatX* ln1b; // (L, C) // layernorm_before bias
    floatX* qkvw; // (L, 3*C, C)
    floatX* qkvb; // (L, 3*C)
    floatX* attprojw; // (L, C, C) attention output dense weight
    floatX* attprojb; // (L, C) attention output dense bias
    floatX* ln2w; // (L, C) layernorm_after weight
    floatX* ln2b; // (L, C) layernorm_after bias
    floatX* fcw; // (L, 4*C, C) intermediate dense weight
    floatX* fcb; // (L, 4*C) intermediate dense bias
    floatX* fcprojw; // (L, C, 4*C) output dense weight
    floatX* fcprojb; // (L, C) output dense bias
    floatX* lnfw; // (C) layernorm final weight
    floatX* lnfb; // (C) layernorm final bias
    floatX* classifierw; // (num_classes, C) classifier weight
    floatX* classifierb; // (num_classes) classifier bias
} ParameterTensors;
static_assert(sizeof(ParameterTensors) == NUM_PARAMETER_TENSORS * sizeof(void*), "Inconsistent sizes!");

void fill_in_parameter_sizes(size_t* param_sizes, size_t* param_sizeof, ViTConfig config) {
    size_t P = config.num_patches;
    size_t C = config.channels;
    size_t L = config.num_layers;
    size_t num_classes = config.num_classes;
    // printf0("fill_in_parameter_sizes: P=%d, C=%d, L=%d, num_classes=%d\n", P, C, L, num_classes);
    param_sizes[0] = C; // cls
    param_sizes[1] = (P + 1) * C; // ppe
    param_sizes[2] = C * C; // ptew
    param_sizes[3] = C; // pteb
    param_sizes[4] = L * C; // ln1w
    param_sizes[5] = L * C; // ln1b
    param_sizes[6] = L * 3 * C * C; // qkvw
    param_sizes[7] = L * 3 * C; // qkvb
    param_sizes[8] = L * C * C; // attprojw
    param_sizes[9] = L * C; // attprojb
    param_sizes[10] = L * C; // ln2w
    param_sizes[11] = L * C; // ln2b
    param_sizes[12] = L * 4 * C * C; // fcw
    param_sizes[13] = L * 4 * C; // fcb
    param_sizes[14] = L * C * 4 * C; // fcprojw
    param_sizes[15] = L * C; // fcprojb
    param_sizes[16] = C; // lnfw
    param_sizes[17] = C; // lnfb
    param_sizes[18] = num_classes * C; // classifierw
    param_sizes[19] = num_classes; // classifierb

    // populate the parameter sizes in bytes (all the same for now, keeping for future use)
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        param_sizeof[i] = sizeof(floatX);
    }
}

// allocate memory for the parameters and point the individual tensors to the right places
void* malloc_and_point_parameters(ParameterTensors* params, size_t* param_elements, size_t *param_sizeof) {
    // calculate the total number of parameters and bytes across all tensors
    size_t num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        num_parameters_bytes += param_elements[i] * param_sizeof[i];
    }

    printf0("allocating %d MB for parameters\n", (int)round(num_parameters_bytes / 1024 / 1024));
    
    // malloc all parameters all at once on the device
    void* params_memory;
    cudaCheck(cudaMalloc((void**)&params_memory, num_parameters_bytes));
    // assign all the tensors their place in the array
    floatX** ptrs[] = {
        &params->cls, &params->ppe, &params->ptew, &params->pteb, &params->ln1w, &params->ln1b, &params->qkvw, &params->qkvb,
        &params->attprojw, &params->attprojb, &params->ln2w, &params->ln2b, &params->fcw, &params->fcb,
        &params->fcprojw, &params->fcprojb, &params->lnfw, &params->lnfb, &params->classifierw, &params->classifierb
    };
    char* params_memory_iterator = (char*)params_memory;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        *(ptrs[i]) = (floatX*)params_memory_iterator;
        params_memory_iterator += param_elements[i] * param_sizeof[i];
    }
    return params_memory;
}

constexpr int NUM_ACTIVATION_TENSORS = 22;
typedef struct {
    floatX* encoded;     // (B, P + 1, C)
    floatX* ln1; // (L, P + 1, , C)
    float* ln1_mean; // (L, B, P + 1)
    float* ln1_rstd; // (L, B, P + 1)
    floatX* atty; // (L, B, P + 1, C)
    // cuDNN saves only some statistics information
#if ENABLE_CUDNN
    float* att;  // (L, B, NH, P + 1)
#else
    floatX* att; // (L, B, NH, P + 1, P + 1)
#endif

    floatX* residual2; // (L, B, P + 1, C)
    floatX* ln2; // (L, B, P + 1, C)
    float* ln2_mean; // (L, B, P + 1)
    float* ln2_rstd; // (L, B, P + 1)
    floatX* fch; // (L, B, P + 1, 4*C)
    floatX* fch_gelu; // (L, B, P + 1, 4*C)
    floatX* residual3; // (L, B, P + 1, C)
    floatX* lnf; // (B, P + 1, C);   if LN recomputation is enabled (-r 2 and above), will be used for _all_ layernorms
    float* lnf_mean; // (B, P + 1)
    float* lnf_rstd; // (B, P + 1)
    float* losses; // (B)
    floatX* padding_memory; // (B) 
    floatX* qkvr; // (L, B, P + 1, 3*C)

    floatX* cls_last; // (B, C)
    floatX* scratch; // (B, (P + 1) * 3*C)

    // Final logits
    floatX* output;      // (B, num_classes)
} ActivationTensors;



struct TensorSpec {
    void** ptr;
    size_t size;
    DType type;
};


#define TENSOR_SPEC(pointer, size) TensorSpec{(void**)(&pointer), (size), dtype_of(pointer)};

void fill_in_activation_sizes(const ActivationTensors* data, TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS], size_t B, ViTConfig config, int recompute) {
    size_t L = config.num_layers;
    size_t NH = config.num_heads;
    size_t C = config.channels;
    size_t num_classes = config.num_classes;
    size_t P = config.num_patches;

    tensors[0] = TENSOR_SPEC(data->encoded, B * (P + 1) * C);
    // if recompute >= 1 then we will recompute the layernorm forward activation during backward pass
    tensors[1] = TENSOR_SPEC(data->ln1,  (recompute < 2) ? L * B * (P + 1) * C : 0);
    tensors[2] = TENSOR_SPEC(data->ln1_mean, L * B * (P + 1));
    tensors[3] = TENSOR_SPEC(data->ln1_rstd, L * B * (P + 1));
    tensors[4] = TENSOR_SPEC(data->atty, L * B * (P + 1) * C);
    #ifdef ENABLE_CUDNN
    // FP32 stats tensor for cuDNN to be passed to backward pass
    tensors[5] = TENSOR_SPEC(data->att, L * B * NH * (P + 1));
    #else
    tensors[5] = TENSOR_SPEC(data->att, L * B * NH * (P + 1) * (P + 1));
    #endif
    tensors[6] = TENSOR_SPEC(data->residual2, L * B * (P + 1) * C);
    // if recompute >= 1 then we will recompute the layernorm forward activation during backward pass
    tensors[7] = TENSOR_SPEC(data->ln2, (recompute < 2) ? L * B * (P + 1) * C : 0);
    tensors[8] = TENSOR_SPEC(data->ln2_mean, L * B * (P + 1));
    tensors[9] = TENSOR_SPEC(data->ln2_rstd, L * B * (P + 1));
    tensors[10] = TENSOR_SPEC(data->fch, L * B * (P + 1) * 4*C);
    // if recompute >= 1 then we will recompute gelu_forward during backward and use this as scratch buffer
    tensors[11] = TENSOR_SPEC(data->fch_gelu, (recompute < 1) ? L * B * (P + 1) * 4*C : B * (P + 1) * 4*C);
    tensors[12] = TENSOR_SPEC(data->residual3, L * B * (P + 1) * C);
    tensors[13] = TENSOR_SPEC(data->lnf, B * (P + 1) * C);
    tensors[14] = TENSOR_SPEC(data->lnf_mean, B * (P + 1));
    tensors[15] = TENSOR_SPEC(data->lnf_rstd, B * (P + 1));
    tensors[16] = TENSOR_SPEC(data->losses, B);
    tensors[17] = TENSOR_SPEC(data->padding_memory, B);
    tensors[18] = TENSOR_SPEC(data->qkvr, L * B * (P + 1) * 3*C);
    tensors[19] = TENSOR_SPEC(data->cls_last, B * C);
    tensors[20] = TENSOR_SPEC(data->scratch, B * (P + 1) * 3*C);
    tensors[21] = TENSOR_SPEC(data->output, B * num_classes);
}

void* malloc_and_point_activations(TensorSpec (&tensors)[NUM_ACTIVATION_TENSORS]) {
    size_t bytes = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        bytes += tensors[i].size * sizeof_dtype(tensors[i].type);
    }

    printf0("allocating %d MB for activations\n", (int)round(bytes / 1024 / 1024));

    void* acts_memory;
    cudaCheck(cudaMalloc((void**)&acts_memory, bytes));

    // cudaMalloc does not guarantee initial memory values so we memset the allocation here
    // this matters because e.g. non-cuDNN attention assumes the attention buffer is zeroed
    // todo - up to ~100ms on slow GPUs, could theoretically be more selective, but this is safer
    cudaCheck(cudaMemset(acts_memory, 0, bytes));

    char* acts_memory_iterator = (char*)acts_memory;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        // extra protection so we don't accidentally use an empty buffer
        if(tensors[i].size == 0) {
            *(tensors[i].ptr) = NULL;
        }else {
            *(tensors[i].ptr) = acts_memory_iterator;
            acts_memory_iterator += tensors[i].size * sizeof_dtype(tensors[i].type);
        }
    }
    return acts_memory;
}

typedef struct {
    ViTConfig config;
    // the weights of the model, and their sizes
    ParameterTensors params;
    size_t param_elements[NUM_PARAMETER_TENSORS];
    size_t param_sizeof[NUM_PARAMETER_TENSORS];
    void* params_memory;
    size_t num_parameters;
    size_t num_parameters_bytes;
    // gradients of the weights
    ParameterTensors grads;
    void* grads_memory;
    // buffers for the AdamW optimizer
    float* m_memory;
    float* v_memory;
    float* master_weights;     // is NULL unless fp32 weights is enabled.
    // the activations of the model, and their sizes
    ActivationTensors acts;
    TensorSpec acts_specs[NUM_ACTIVATION_TENSORS];
    void* acts_memory;
    // other run state configuration
    int batch_size; // the batch size (B) of current forward pass
    floatX* inputs; // the input vectors for the current forward pass
    int* classes; // the classes for the current forward pass
    float mean_loss; // after the last backward micro-batch, will be populated with mean loss across all GPUs and micro-steps
    float* accumulated_mean_loss; // GPU buffer used to accumulate loss across micro-steps
    float* cpu_losses; // CPU buffer to copy the losses to, allocated with cudaMallocHost
    unsigned long long rng_state; // the RNG state for seeding stochastic rounding etc.
    unsigned long long rng_state_last_update; // RNG before last vit_update() to re-round identically from master weights
    int use_master_weights; // keep master weights copy in float for optim update? 0|1
    bool init_state;   // set to true if master weights need to be initialized
    int gelu_fusion; // fuse gelu via cuBLASLt (0=none, 1=forward, 2=forward+backward)
    int recompute; // recompute gelu | layernorm forward during model backward? 0|1|2
    // todo - if other functions need cpu scratch buffers in the future, reuse as generic scratch?
    int* workload_indices; // encoder_backward, B*T*num_c_groups (int)
    int4* bucket_info;     // encoder_backward, B*T*num_c_groups (int4) - size for worst case
} ViT;

void vit_init_common(ViT *model) {
    // common inits outside of the model weights
    // memory lazily initialized in forward()
    model->acts_memory = NULL;
    model->inputs = NULL; // the input tokens for the current forward pass
    model->classes = NULL; // the target tokens for the current forward pass
    model->accumulated_mean_loss = NULL;
    model->cpu_losses = NULL;
    // the B,T params are determined and set, fixed on first batch in forward()
    model->batch_size = 0;
    model->mean_loss = -1.0f; // -1.0f designates no loss, set at end of forward()
    model->params_memory = NULL;
    // memory lazily initialized in backward()
    model->grads_memory = NULL;
    model->workload_indices = NULL; // on cpu, for encoder_backward
    model->bucket_info = NULL; // on cpu, for encoder_backward
    // memory lazily initialized in update()
    model->m_memory = NULL;
    model->v_memory = NULL;
    model->master_weights = NULL;
    // other default settings
    model->rng_state = 13371337 + multi_gpu_config.process_rank; // used in stochastic rounding
    model->use_master_weights = 1; // safe default: do keep master weights in fp32
    model->init_state = true;
    model->recompute = 1; // good default: recompute gelu but not layernorm
    model->gelu_fusion = 0; //deviceProp.major >= 9 ? 2 : 0; // default: off for now (default must match main())
}

void vit_allocate_weights(ViT *model) {
    // fill in all the parameter tensor dimensions and types
    fill_in_parameter_sizes(model->param_elements, model->param_sizeof, model->config);
    model->num_parameters = 0;
    model->num_parameters_bytes = 0;
    for (int i = 0; i < NUM_PARAMETER_TENSORS; i++) {
        model->num_parameters += model->param_elements[i];
        model->num_parameters_bytes += model->param_elements[i] * model->param_sizeof[i];
    }
    // create memory for model parameters on the device
    assert(model->params_memory == nullptr);
    model->params_memory = malloc_and_point_parameters(&model->params, model->param_elements, model->param_sizeof);
}

void vit_allocate_state(ViT *model, int B) {
    assert(model->grads_memory == nullptr);
    model->grads_memory = malloc_and_point_parameters(&model->grads, model->param_elements, model->param_sizeof);

    // record the current B,Np as well
    model->batch_size = B;
    int P = model->config.num_patches;
    int channels = model->config.channels;

    // allocate the space
    fill_in_activation_sizes(&model->acts, model->acts_specs, B, model->config, model->recompute);
    model->acts_memory = malloc_and_point_activations(model->acts_specs);
    // also create memory for caching inputs and targets
    // printf0("allocating %d B for inputs\n", (int)round(B * P * channels * sizeof(floatX)));
    cudaCheck(cudaMalloc((void**)&model->inputs, B * P * channels * sizeof(floatX)));
    cudaCheck(cudaMalloc((void**)&model->classes, B * sizeof(int)));
    cudaCheck(cudaMalloc(((void**)&model->accumulated_mean_loss), sizeof(float)));
    cudaCheck(cudaMallocHost((void**)&model->cpu_losses, B * sizeof(float)));

    // initialise cpu scratch buffers for encoder backward
    size_t num_c_groups = CEIL_DIV(model->config.channels, (WARP_SIZE * x128::size));
    assert((size_t)(model->batch_size * P) * num_c_groups < (1ULL<<31ULL)); // todo - maybe an issue for llama3-400B(?)
    model->workload_indices = (int*)mallocCheck(sizeof(int) * model->batch_size * P * num_c_groups);
    model->bucket_info = (int4*)mallocCheck(sizeof(int4) * model->batch_size * P * num_c_groups);

    // cudaMallocConditionallyManaged can fall back to cudaMallocManaged if not enough memory on device
    // and returns a status code of 1 if it had to fall back, in that case we want to print warning.
    int memory_status = 0;

    // we will now init the optimizer states and master weights
    // this is usually a substantial amount of memory allocation right here.
    size_t shard_num_parameters = multi_gpu_config.shard_num_parameters; // num parameters we are responsible for
    printf0("allocating %zu MiB for AdamW optimizer state m\n", (shard_num_parameters * sizeof(float)) >> 20);
    printf0("allocating %zu MiB for AdamW optimizer state v\n", (shard_num_parameters * sizeof(float)) >> 20);
    assert(model->m_memory == nullptr);
    assert(model->v_memory == nullptr);
    memory_status |= cudaMallocConditionallyManaged((void**)&model->m_memory, shard_num_parameters * sizeof(float));
    memory_status |= cudaMallocConditionallyManaged((void**)&model->v_memory, shard_num_parameters * sizeof(float));

    if (model->use_master_weights == 1) {
        assert(model->master_weights == nullptr);
        printf0("allocating %zu MiB for master copy of params\n", (shard_num_parameters * sizeof(float)) >> 20);
        memory_status |= cudaMallocConditionallyManaged((void**) &model->master_weights, shard_num_parameters * sizeof(float));
    }

    // report on mixed memory allocation status (re-using our float reduce function, bit awk ok)
    int reduced_memory_status = (int) multi_gpu_cpu_float_sum((float)memory_status, &multi_gpu_config);
    if (reduced_memory_status >= 1) {
        printf0("WARNING: Fell back to cudaMallocManaged when initializing m,v,master_weights on %d GPUs\n", reduced_memory_status);
        printf0("         Prevents an OOM, but code may run much slower due to device <-> host memory movement\n");
    }
    // report on device memory usage
    size_t free, total;
    cudaCheck(cudaMemGetInfo(&free, &total));
    printf0("device memory usage: %zd MiB / %zd MiB\n", (total-free) / 1024 / 1024, total / 1024 / 1024);
    // give an estimate of the maximum batch size
    size_t bytes_per_sequence = 0;
    for (size_t i = 0; i < NUM_ACTIVATION_TENSORS; i++) {
        bytes_per_sequence += model->acts_specs[i].size * sizeof_dtype(model->acts_specs[i].type) / B;
    }
    printf0("memory per sequence: %zu MiB\n", bytes_per_sequence / 1024 / 1024);
    printf0(" -> estimated maximum batch size: %zu\n", B + free / bytes_per_sequence);
}

void vit_write_to_checkpoint(ViT *model, const char* checkpoint_path) {
    // write the model to a checkpoint file
    printf0("Writing model to %s\n", checkpoint_path);
    FILE *model_file = fopenCheck(checkpoint_path, "wb");
    // write the header first
    int model_header[256];
    memset(model_header, 0, sizeof(model_header));
    model_header[0] = 20240326; // magic number
    assert(PRECISION_MODE == PRECISION_FP32 || PRECISION_MODE == PRECISION_BF16);
    model_header[1] = PRECISION_MODE == PRECISION_FP32 ? 3 : 5; // version
    model_header[2] = (int)sqrt(model->config.num_patches)*16; // image size
    model_header[3] = model->config.channels; // hidden size
    model_header[4] = model->config.num_layers; // num_layers
    model_header[5] = model->config.num_heads; // num_heads
    model_header[6] = 4 * model->config.channels; // intermediate size
    fwriteCheck(model_header, sizeof(int), 256, model_file);
    // write the parameters
    device_to_file(model_file, model->params_memory, model->num_parameters_bytes,
                   IO_BUF_SIZE, main_stream);
    // close file, we're done
    fcloseCheck(model_file);
}

void vit_build_from_checkpoint(ViT *model, const char* checkpoint_path, bool weight_init=true) {
    // If weight_init is true, we will load the weights from this checkpoint .bin file
    // We sometimes want this to be false, if we are going to initialize these weights from
    // the master weights that are instead stored in the state .bin file.
    // In that case, this function mostly loads the model hyperparameters from the header.

    if (PRECISION_MODE == PRECISION_FP16) {
        // TODO for later perhaps, would require us dynamically converting the
        // model weights from fp32 to fp16 online, here in this function, or writing
        // the fp16 weights directly from Python, which we only do for fp32/bf16 atm.
        fprintf(stderr, "build_from_checkpoint() does not support fp16 right now.\n");
        exit(EXIT_FAILURE);
    }

    // read in model from a checkpoint file
    FILE *model_file = fopenCheck(checkpoint_path, "rb");
    int model_header[256];
    freadCheck(model_header, sizeof(int), 256, model_file);
    if (model_header[0] != 20240326) { printf("Bad magic model file\n"); exit(EXIT_FAILURE); }
    int version = model_header[1];
    if (!(version == 3 || version == 5)) {
        // 3 = fp32, padded vocab
        // 5 = bf16, padded vocab, layernorms also in bf16
        fprintf(stderr, "Bad version in model file\n");
        fprintf(stderr, "---> HINT: try to re-run `python train_vit.py`\n");
        exit(EXIT_FAILURE);
    }

    // check if the precision mode of the checkpoing matches the model precision
    if (weight_init) {
        if (PRECISION_MODE == PRECISION_BF16 && version != 5) {
            fprintf(stderr, "Precision is configured as BF16 but model at %s is not.\n", checkpoint_path);
            fprintf(stderr, "---> HINT: are you sure you're loading a _bf16.bin file?\n");
            exit(EXIT_FAILURE);
        }
        if (PRECISION_MODE == PRECISION_FP32 && version != 3) {
            fprintf(stderr, "Precision is configured as FP32 but model at %s is not.\n", checkpoint_path);
            fprintf(stderr, "---> HINT: to turn on FP32 you have to compile like: `make train_vit.cu PRECISION=FP32`\n");
            fprintf(stderr, "---> HINT: are you sure you're loading a .bin file without any _bf16 in the name?\n");
            exit(EXIT_FAILURE);
        }
    }

    // read in hyperparameters
    model->config.num_patches = (model_header[2]/16)*(model_header[2]/16);
    model->config.channels = model_header[3];
    model->config.num_layers = model_header[4];
    model->config.num_heads = model_header[5];
    model->config.num_classes = 1000;

    // allocate memory for the model parameters
    vit_allocate_weights(model);

    // read in the parameters if weight_init is true
    if (weight_init) {
        assert(model->params_memory != NULL);
        printf0("Reading model from %s %d\n", checkpoint_path, model->num_parameters_bytes);
        file_to_device(model->params_memory, model_file, model->num_parameters_bytes, IO_BUF_SIZE, main_stream);
    }
    fcloseCheck(model_file);

    // only return from this function once we are certain the params are ready on the GPU
    cudaCheck(cudaDeviceSynchronize());
}

// void vit_set_hyperparameters(ViTConfig* config, const char* depth_str) {
//     int depth = atoi(depth_str);
//     assert(depth > 0); // atoi returns 0 if not a number
//     int channels, num_heads;
//     if      (depth == 12)  { channels = 768; num_heads = 12; }   // ViT base (86M)
//     else if (depth == 24) { channels = 1024; num_heads = 16; }  // ViT (307M)
//     else if (depth == 32) { channels = 1280; num_heads = 16; } // ViT-medium (632M)
//     else { fprintf(stderr, "Unsupported ViT depth: %d\n", depth); exit(EXIT_FAILURE); }
//     config->num_layers = depth;
//     config->channels = channels;
//     config->num_heads = num_heads;
//     config->num_patches = 14 * 14; // 224x224 images
//     config->num_classes = 1000; // ImageNet
// }

// propagate inputs through the network to produce logits.
// right now, this function is fully synchronous with the host
void vit_forward(ViT *model, const float* inputs, size_t B) {
    NVTX_RANGE_FN();
    // we must be careful and use size_t instead of int, otherwise
    // we could overflow int. E.g. l * B * NH * (P + 1) * (P + 1) overflows int at B 16.

    // ensure the model was initialized or error out
    if (model->params_memory == NULL) {
        printf("Error: model was not initialized properly.\n");
        exit(EXIT_FAILURE);
    }

    // convenience parameters
    const size_t P = model->config.num_patches;
    const size_t num_classes = model->config.num_classes;
    const size_t L = model->config.num_layers;
    const size_t NH = model->config.num_heads;
    const size_t C = model->config.channels;

    cudaCheck(cudaMemcpy(model->inputs, inputs, B * P * C * sizeof(floatX), cudaMemcpyHostToDevice));

    // forward pass
    ParameterTensors params = model->params; // for brevity
    ActivationTensors acts = model->acts;
    
    for (int i = 0; i < B; i++) {
        matmul_forward_cublaslt(acts.encoded + i * P * C + 1 * 1 * C, model->inputs + i * P * C, params.ptew, params.pteb, 1, P, C, C, main_stream);

        encoder_forward(acts.encoded  + i * P * C, acts.encoded + i * P * C + 1 * 1 * C, params.cls, params.ppe, 1, P, C, main_stream); // encoding goes into residual[0]
    }

    // first layernorm isn't fused
    layernorm_forward((model->recompute < 2) ? acts.ln1 : acts.lnf, acts.ln1_mean, acts.ln1_rstd, acts.encoded, params.ln1w, params.ln1b, B, (P + 1), C, main_stream);

    for (int l = 0; l < L; l++) {
        NvtxRange layer_range("Layer", l);

        floatX* residual = l == 0 ? acts.encoded : acts.residual3 + (l-1) * B * (P + 1) * C;

        // get the pointers of the weights for this layer
        floatX* l_qkvw = params.qkvw + l * 3*C * C;
        floatX* l_qkvb = params.qkvb + l * 3*C;
        floatX* l_attprojw = params.attprojw + l * C * C;
        floatX* l_attprojb = params.attprojb + l * C;
        floatX* l_ln2w = params.ln2w + l * C;
        floatX* l_ln2b = params.ln2b + l * C;
        floatX* l_fcw = params.fcw + l * 4*C * C;
        floatX* l_fcb = params.fcb + l * 4*C;
        floatX* l_fcprojw = params.fcprojw + l * C * 4*C;
        floatX* l_fcprojb = params.fcprojb + l * C;

        // get the pointers of the activations for this layer
        floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + l * B * (P + 1) * C : acts.lnf;
        floatX* l_qkvr = acts.qkvr + l * B * (P + 1) * 3*C;
        floatX* l_atty = acts.atty + l * B * (P + 1) * C;
        floatX* l_residual2 = acts.residual2 + l * B * (P + 1) * C;
        floatX* l_ln2 = (model->recompute < 2) ? acts.ln2 + l * B * (P + 1) * C : acts.lnf;
        float* l_ln2_mean = acts.ln2_mean + l * B * (P + 1);
        float* l_ln2_rstd = acts.ln2_rstd + l * B * (P + 1);
        floatX* l_fch = acts.fch + l * B * (P + 1) * 4*C;
        // reuse the same activation buffer at each layer, as we'll re-compute the gelu during backward
        // very useful because we dramatically reduce VRAM usage, and may be able to fit larger batch size
        floatX* l_fch_gelu = (model->recompute < 1) ? acts.fch_gelu + l * B * (P + 1) * 4*C : acts.fch_gelu;
        floatX* l_residual3 = acts.residual3 + l * B * (P + 1) * C;
        floatX* scratch = (floatX*)acts.scratch; // used for non-cudnn attention, fcproj, attproj, etc.

        // now do the forward pass
        #ifdef ENABLE_CUDNN
        float* l_att = (float*)acts.att + l * B * NH * (P + 1); // cuDNN needs a smaller FP32 tensor
        matmul_forward_cublaslt(l_qkvr, l_ln1, l_qkvw, l_qkvb, B, (P + 1), C, 3*C, main_stream);
        attention_forward_cudnn(l_atty, (float*)l_att, l_qkvr, B, (P + 1), NH, C, main_stream);
        #else
        floatX* l_att = acts.att + l * B * NH * (P + 1) * (P + 1);

        // these are only needed as scratchpads for the forward pass, but
        // need not be stored for backward
        matmul_forward_cublaslt(scratch, l_ln1, l_qkvw, l_qkvb, B, (P + 1), C, 3*C, main_stream);

        attention_forward(l_atty, l_qkvr, l_att, scratch, B, (P + 1), C, NH, main_stream);
        #endif

        matmul_forward_cublaslt(scratch, l_atty, l_attprojw, l_attprojb, B, (P + 1), C, C, main_stream);

        fused_residual_forward5(l_residual2, l_ln2, l_ln2_mean, l_ln2_rstd, residual, scratch, l_ln2w, l_ln2b, B*(P + 1), C, main_stream);

        matmul_forward_cublaslt(l_fch_gelu, l_ln2, l_fcw, l_fcb, B, (P + 1), C, 4*C, main_stream, l_fch, model->gelu_fusion);

        matmul_forward_cublaslt(scratch, l_fch_gelu, l_fcprojw, l_fcprojb, B, (P + 1), 4*C, C, main_stream);
        // OK, fusion across blocks.
        if(l+1 != L) {
            floatX* l_ln1 = (model->recompute < 2) ? acts.ln1 + (l + 1) * B * (P + 1) * C : acts.lnf;
            float* l_ln1_mean = acts.ln1_mean + (l + 1) * B * (P + 1);
            float* l_ln1_rstd = acts.ln1_rstd + (l + 1) * B * (P + 1);
            const floatX* l_ln1w = params.ln1w + (l + 1) * C;
            const floatX* l_ln1b = params.ln1b + (l + 1) * C;
            fused_residual_forward5(l_residual3, l_ln1, l_ln1_mean, l_ln1_rstd, l_residual2, scratch, l_ln1w, l_ln1b,
                                    B * (P + 1), C, main_stream);
        } else {
            fused_residual_forward5(l_residual3, acts.lnf, acts.lnf_mean, acts.lnf_rstd, l_residual2, scratch,
                                    params.lnfw, params.lnfb,
                                    B * (P + 1), C, main_stream);
        }
    }
    // do layernorm once more at the end
    layernorm_forward(acts.residual3, acts.lnf_mean, acts.lnf_rstd, acts.lnf, params.lnfw, params.lnfb, B, (P + 1), C, main_stream);

    for (int i = 0; i < B; i++) {
        // copy (B, (P + 1), C) to (B, C)
        cudaCheck(cudaMemcpy(acts.cls_last + i * C, acts.residual3 + i * (P + 1) * C, C * sizeof(floatX), cudaMemcpyDeviceToDevice));
    }
    // Map the cls token to class logits
    matmul_forward_cublaslt(acts.output, acts.cls_last, params.classifierw, params.classifierb, B, 1, C, num_classes, main_stream);
    cudaCheck(cudaDeviceSynchronize());
}

void vit_free(ViT *model) {
    cudaFreeCheck(&model->params_memory);
    cudaFreeCheck(&model->grads_memory);
    cudaFreeCheck(&model->m_memory);
    cudaFreeCheck(&model->v_memory);
    cudaFreeCheck(&model->master_weights);
    cudaFreeCheck(&model->acts_memory);
    cudaFreeCheck(&model->inputs);
    cudaFreeCheck(&model->accumulated_mean_loss);
    cudaCheck(cudaFreeHost(model->cpu_losses));
    free(model->workload_indices);
    free(model->bucket_info);
}

// ----------------------------------------------------------------------------
// common init & free code for all of train/test/profile

void common_start(bool override_enable_tf32 = true, bool print_device_info = true) {

    // get CUDA device infos
    cudaCheck(cudaGetDeviceProperties(&deviceProp, multi_gpu_config.local_device_idx));
    if (print_device_info) {
        printf("[System]\n");
        printf("Device %d: %s\n", multi_gpu_config.local_device_idx, deviceProp.name);
    }

    // set up the cuda streams. atm everything is on the single main stream
    cudaCheck(cudaStreamCreate(&main_stream));
    nvtxNameCudaStreamA(main_stream, "main stream");

    // set up cuBLAS and cuBLASLt
    cublasCheck(cublasLtCreate(&cublaslt_handle));
    cudaCheck(cudaMalloc(&cublaslt_workspace, cublaslt_workspace_size));

    // TF32 precision is equivalent to torch.set_float32_matmul_precision('high')
    bool enable_tf32 = PRECISION_MODE == PRECISION_FP32 && deviceProp.major >= 8 && override_enable_tf32;
    cublas_compute = enable_tf32 ? CUBLAS_COMPUTE_32F_FAST_TF32 : CUBLAS_COMPUTE_32F;

    #ifdef ENABLE_CUDNN
    create_cudnn();
    #endif
}

void common_free(ViT &model) {
    cudaCheck(cudaStreamDestroy(main_stream));
    cudaCheck(cudaFree(cublaslt_workspace));
    cublasCheck(cublasLtDestroy(cublaslt_handle));
    #ifdef ENABLE_CUDNN
    destroy_cudnn();
    #endif
}

void read_floats_from_file(const char* filename, int N, float* output) {
    // Open the file for reading
    FILE* file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        return;
    }

    // Read floats from the file
    for (int i = 0; i < N; i++) {
        if (fscanf(file, "%f", &output[i]) != 1) {
            fprintf(stderr, "Error reading float at index %d\n", i);
            fclose(file);
            return;
        }
    }

    // Close the file
    fclose(file);
    return;
}

void generate_filename(int n, char* buffer, size_t buffer_size, int flag) {
    if(flag)
        snprintf(buffer, buffer_size, "info/ILSVRC2012_test_%08d.txt", n);
    else
        snprintf(buffer, buffer_size, "ILSVRC2012_test_%08d", n);
}

void read_classes(const char *filename, char classes[1000][256]) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("Error opening file");
        exit(EXIT_FAILURE);
    }

    char line[512]; // Buffer to hold each line
    while (fgets(line, sizeof(line), file)) {
        int index;
        char label[256];
        // Parse each line (format: "index<tab>label")
        if (sscanf(line, "%d %[^\n]", &index, label) == 2) {
            if (index >= 0 && index < 1000) {
                strncpy(classes[index], label, 255);
                classes[index][255] = '\0'; // Ensure null-termination
            } else {
                fprintf(stderr, "Index out of bounds: %d\n", index);
            }
        } else {
            fprintf(stderr, "Malformed line: %s\n", line);
        }
    }

    fclose(file);
}

// ----------------------------------------------------------------------------
// main inference loop

int main(int argc, char *argv[]) {
    // Sample images
    int max_avail = 10; // change this when new image was added
    int S = 1;
    int N;         // Number of sample images
    int B;
    int C = 3;     // Number of channels (RGB)
    int H = 224;   // Image height
    int W = 224;   // Image width

    int opt;
    bool isNSet = false, isBSet = false;
    printf("Usage: -N <number of images> -B <batch size> -S <start location>\n");
    while ((opt = getopt(argc, argv, "N:B:S:")) != -1) {
        switch (opt) {
            case 'N':
                N = std::atoi(optarg);
                isNSet = true;
                break;
            case 'B':
                B = std::atoi(optarg);
                isBSet = true;
                break;
            case 'S':
                S = std::atoi(optarg);
                break;
            default:
                printf("Error: Wrong argument\n");
                return 1;
        }
    }

    assert(isNSet && "Error: -N <number of images> is required.");
    assert(isBSet && "Error: -B <batch size> is required.");
    assert(N > 0 && "Error: -N must be greater than 0.");
    assert(B > 0 && "Error: -B must be greater than 0.");
    assert(S > 0 && "Error: -S must be greater than 0.");
    assert(N % B == 0 && "Error: -N must be divisible by -B.");

    if((S+N-1) > max_avail){
        printf("Error: Not enough images\n");
        return 1;
    }

    // Load the ViT model from the binary file
    const char* load_filename = "vit_base_patch16_224.bin";

    // Initialize ViT model structure
    ViT model;
    vit_init_common(&model);

    // Initialize common resources (e.g., CUDA streams, cuBLAS handles)
    common_start(false, false);

    // Build the ViT model from the binary checkpoint
    vit_build_from_checkpoint(&model, load_filename);

    // Allocate memory for input images and output predictions
    size_t input_size = N * C * H * W * sizeof(float);
    size_t output_size = N * model.config.num_classes * sizeof(float);

    // Allocate host memory for input images
    float* host_input_images = (float*)malloc(input_size);

    // Load and preprocess images
    char image_path[100];
    for (int i = 0; i < N; ++i) {
        // format:
        // [[],....,[]]
        // size: (196,768)
        
        // Load the image data
        generate_filename((S+i), image_path, sizeof(image_path), 1);
        read_floats_from_file(image_path, 3*224*224, host_input_images + i * 3*224*224);
        // print the image data for debugging
        // for(int k=0; k<196*768; k++){
        //      printf("%f ", host_input_images[i*N + k]);
        // }
        // printf("\n");
    }

    vit_allocate_state(&model, B);

    // Run the forward pass
    vit_forward(&model, host_input_images, B);
    free(host_input_images);


    // Process the output logits
    float* host_logits = (float*)malloc(output_size);
    // cuda copy model.acts.output to host_logits
    cudaCheck(cudaMemcpy(host_logits, model.acts.output, output_size, cudaMemcpyDeviceToHost));
    
    char classes[1000][256];
    read_classes("classes.txt", classes);

    // Post-processing: Find the predicted class for each image
    for (int i = 0; i < N; ++i) {
        int max_idx = -1;
        float max_val = -340282346638528859811704183484516925440.0f;
        for (int j = 0; j < model.config.num_classes; ++j) {
            float logit = host_logits[i * model.config.num_classes + j];
            if (logit > max_val) {
                max_val = logit;
                max_idx = j;
            }
        }
        generate_filename((S+i), image_path, sizeof(image_path), 0);
        printf("Image %s: Predicted class %s\n", image_path, classes[max_idx]);
    }

    // Cleanup
    free(host_logits);

    // Free the model resources
    vit_free(&model);

    // Free common resources
    common_free(model);

    return 0;
}