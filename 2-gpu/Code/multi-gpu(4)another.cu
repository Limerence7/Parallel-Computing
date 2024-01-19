#include <cuda_runtime.h>
#include "CUDA_acc.cuh"

#define BNUM 1024

__global__ void kernel_sum(float *data1, const float *data2)
{
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    data1[idx] += data2[idx];
}

__global__ void logDataVSPrior(const float dat_comp, const float dat_real, const float pri_comp, const float pri_real, 
                const float ctf, const float sigRcp, float *ans_dev, const float *disturb_dev)
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

    float *dev_host[3];
    dev_host[0] = (float*)malloc(Ksize);
    dev_host[1] = (float*)malloc(Ksize);
    dev_host[2] = (float*)malloc(Ksize);

    float *disturb_dev[4];
    float *ans_dev[4];
    float *dev_dev[3];
    cudaStream_t stream[4];

    cudaSetDevice(0);
    cudaMalloc((void**)&disturb_dev[0], Ksize);
    cudaMalloc((void**)&ans_dev[0], Ksize);
    cudaStreamCreate(&stream[0]);

    cudaMalloc((void**)&dev_dev[0], Ksize);
    cudaMalloc((void**)&dev_dev[1], Ksize);

    cudaSetDevice(1);
    cudaMalloc((void**)&disturb_dev[1], Ksize);
    cudaMalloc((void**)&ans_dev[1], Ksize);
    cudaStreamCreate(&stream[1]);

    cudaSetDevice(2);
    cudaMalloc((void**)&disturb_dev[2], Ksize);
    cudaMalloc((void**)&ans_dev[2], Ksize);
    cudaStreamCreate(&stream[2]);

    cudaMalloc((void**)&dev_dev[2], Ksize);

    cudaSetDevice(3);
    cudaMalloc((void**)&disturb_dev[3], Ksize);
    cudaMalloc((void**)&ans_dev[3], Ksize);
    cudaStreamCreate(&stream[3]);

    cudaMemcpyAsync(disturb_dev[0], disturb, Ksize, cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(disturb_dev[1], disturb, Ksize, cudaMemcpyHostToDevice, stream[1]);
    cudaMemcpyAsync(disturb_dev[2], disturb, Ksize, cudaMemcpyHostToDevice, stream[2]);
    cudaMemcpyAsync(disturb_dev[3], disturb, Ksize, cudaMemcpyHostToDevice, stream[3]);

    dim3 block_size(BNUM);
    dim3 grid_size((K - 1) / block_size.x + 1);

    for (unsigned int t = 0; t < m; t += 4)
    {
        cudaSetDevice(0);
        logDataVSPrior<<< grid_size, block_size, 0, stream[0] >>>(dat_comp[t], dat_real[t], pri_comp[t], 
                pri_real[t], ctf[t], sigRcp[t], ans_dev[0], disturb_dev[0]);
        
        cudaSetDevice(1);
        logDataVSPrior<<< grid_size, block_size, 0, stream[1] >>>(dat_comp[t + 1], dat_real[t + 1], pri_comp[t + 1], 
                pri_real[t + 1], ctf[t + 1], sigRcp[t + 1], ans_dev[1], disturb_dev[1]);

        cudaSetDevice(2);
        logDataVSPrior<<< grid_size, block_size, 0, stream[2] >>>(dat_comp[t + 2], dat_real[t + 2], pri_comp[t + 2], 
                pri_real[t + 2], ctf[t + 2], sigRcp[t + 2], ans_dev[2], disturb_dev[2]);

        cudaSetDevice(3);
        logDataVSPrior<<< grid_size, block_size, 0, stream[3] >>>(dat_comp[t + 3], dat_real[t + 3], pri_comp[t + 3], 
                pri_real[t + 3], ctf[t + 3], sigRcp[t + 3], ans_dev[3], disturb_dev[3]);

    }

    for (int i = 0; i < 4; i++)
        cudaStreamSynchronize(stream[i]);

    cudaMemcpyAsync(dev_host[0], ans_dev[1], Ksize, cudaMemcpyDeviceToHost, stream[1]);
    cudaMemcpyAsync(dev_host[2], ans_dev[3], Ksize, cudaMemcpyDeviceToHost, stream[3]);
    
    cudaMemcpyAsync(dev_dev[0], dev_host[0], Ksize, cudaMemcpyHostToDevice, stream[0]);
    cudaMemcpyAsync(dev_dev[2], dev_host[2], Ksize, cudaMemcpyHostToDevice, stream[2]);

    cudaSetDevice(0);
    kernel_sum<<< grid_size, block_size, 0, stream[0] >>>(ans_dev[0], dev_dev[0]);

    cudaSetDevice(2);
    kernel_sum<<< grid_size, block_size, 0, stream[2] >>>(ans_dev[2], dev_dev[2]);

    cudaStreamSynchronize(stream[0]);
    cudaStreamSynchronize(stream[2]);

    cudaMemcpyAsync(dev_host[1], ans_dev[2], Ksize, cudaMemcpyDeviceToHost, stream[2]);
    cudaMemcpyAsync(dev_dev[1], dev_host[1], Ksize, cudaMemcpyHostToDevice, stream[0]);

    cudaSetDevice(0);
    kernel_sum<<< grid_size, block_size, 0, stream[0] >>>(ans_dev[0], dev_dev[1]);

    cudaMemcpyAsync(ans, ans_dev[0], Ksize, cudaMemcpyDeviceToHost, stream[0]);
    

    //free(dev_host);
    //cudaFree(dev_dev);
    cudaFree(disturb_dev[0]);
    cudaFree(disturb_dev[1]);
    cudaFree(ans_dev[0]);
    cudaFree(ans_dev[1]);
    cudaStreamDestroy(stream[0]);
    cudaStreamDestroy(stream[1]);
}