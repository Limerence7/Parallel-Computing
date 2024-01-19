/*****************************************************************************
 * File:        simpleDivergence.cu
 * Description: Measure the performance of some kernels.
 *              One has warp divergence and others doesn't have warp divergence.
 *              
 * Compile:     nvcc -g -G -arch=sm_75 -o simpleDivergence simpleDivergence.cu -I..
 * Run:         ./simpleDivergence
 * Argument:    n.a
 *****************************************************************************/
#include <stdio.h>
#include <cuda_runtime.h>

#include "call.h"

__global__ void kernel(int* pivots, const int* com, const int npoints, const int n, const int k)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= npoints) return;
    int i = 1, j = k - 1, sum = idx + 1;
    idx *= k;
    while (j)
    {
        int cur = com[(n - i) * k + j];
        if (sum > cur)
            sum -= cur;
        else
        {
            pivots[idx + j] = i;
            j--;
        }
        i++;
    }
    pivots[idx] = sum + pivots[idx + 1];
}

int main()
{
    int n = 98, k = 5;
    int npoints = 1;
    for (int i = 1; i <= k; i++)
        npoints = npoints * (n - i + 1) / i;

    int nBytes = sizeof(int) * (n + 1) * k;
    int *com_host = new int[(n + 1) * k];
    int *pivots_host = new int[npoints * k];

    for (int i = 0; i <= n; i++)
    {
        com_host[i * k] = 1;
        for (int j = 1; j < k && j <= i; j++)
        {
            com_host[i * k + j] = com_host[(i - 1) * k + j] + com_host[(i - 1) * k + j - 1];
        }
    }

    dim3 block(1024);
    dim3 grid((npoints - 1) / block.x + 1);

    printf("%d\n", grid.x);

    int *pivots = NULL, *com = NULL;
    cudaMalloc((void**)&pivots, sizeof(int) * npoints * k);
    cudaMalloc((void**)&com, nBytes);

    cudaMemcpy(com, com_host, nBytes, cudaMemcpyHostToDevice);

    kernel<<< grid, block >>>(pivots, com, npoints, n, k);

    cudaMemcpy(pivots_host, pivots, sizeof(int) * npoints * k, cudaMemcpyDeviceToHost);
    FILE *fp;
    fp=fopen("test.txt","w");
    for (int i = 0; i < npoints; i++)
    {
        for (int j = k - 1; j >= 0; j--)
            fprintf(fp, "%d ", pivots_host[i * k + j]);
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
}