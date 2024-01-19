#include <cuda_runtime.h>
#include "CUDA_acc.cuh"

#define BNUM 1024

__global__ void kernel_sum(float *data1, float *data2)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    data1[idx] += data2[idx];
}

__global__ void logDataVSPrior(const float dat_comp, const float dat_real, const float pri_comp, const float pri_real, 
                const float ctf, const float sigRcp, float *ans_dev, const float *disturb_dev, const int K)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    float comp = dat_comp - disturb_dev[idx] * ctf * pri_comp;
    float real = dat_real - disturb_dev[idx] * ctf * pri_real;

    ans_dev[idx] += (real * real + comp * comp) * sigRcp;
}

extern "C"
void use_GPU(const float* dat_comp, const float* dat_real, const float* pri_comp, const float* pri_real, 
                const float* ctf, const float* sigRcp, const float* disturb, float* ans, const int m, const int K)
{
    int Ksize = sizeof(float) * K;

    float *dev1_host = (float*)malloc(Ksize);

    float *disturb_dev[2];
    float *ans_dev[2];
    float *dev1_dev;
    cudaStream_t stream[2];

    cudaSetDevice(0);
    cudaMalloc((void**)&disturb_dev[0], Ksize);
    cudaMalloc((void**)&ans_dev[0], Ksize);
    cudaMalloc((void**)&dev1_dev, Ksize);
    cudaStreamCreate(&stream[0]);

    cudaSetDevice(4);
    cudaMalloc((void**)&disturb_dev[1], Ksize);
    cudaMalloc((void**)&ans_dev[1], Ksize);
    cudaStreamCreate(&stream[1]);

    cudaMemcpyAsync(disturb_dev[0], disturb, Ksize, cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(disturb_dev[1], disturb, Ksize, cudaMemcpyHostToDevice, stream[1]);

    dim3 block_size(BNUM);
    dim3 grid_size((K - 1) / block_size.x + 1);

    for (unsigned int t = 0; t < m; t += 2)
    {
        cudaSetDevice(0);
        logDataVSPrior<<< grid_size, block_size, 0, stream[0] >>>(dat_comp[t], dat_real[t], pri_comp[t], 
                pri_real[t], ctf[t], sigRcp[t], ans_dev[0], disturb_dev[0], K);
        
        cudaSetDevice(4);
        logDataVSPrior<<< grid_size, block_size, 0, stream[1] >>>(dat_comp[t + 1], dat_real[t + 1], pri_comp[t + 1], 
                pri_real[t + 1], ctf[t + 1], sigRcp[t + 1], ans_dev[1], disturb_dev[1], K);

    }

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[1]);

    cudaMemcpyAsync(dev1_host, ans_dev[1], Ksize, cudaMemcpyDeviceToHost, stream[1]);

    cudaSetDevice(0);
    cudaMemcpyAsync(dev1_dev, dev1_host, Ksize, cudaMemcpyHostToDevice, stream[0]);

    kernel_sum<<< grid_size, block_size, 0, stream[0] >>>(ans_dev[0], dev1_dev);
    cudaMemcpyAsync(ans, ans_dev[0], Ksize, cudaMemcpyDeviceToHost, stream[0]);

    cudaStreamSynchronize(stream[0]);

    free(dev1_host);
    cudaFree(dev1_dev);
    cudaFree(disturb_dev[0]);
    cudaFree(disturb_dev[1]);
    cudaFree(ans_dev[0]);
    cudaFree(ans_dev[1]);
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
}