#include <algorithm>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "CUDA_acc.cuh"

#define BNUM 1024

using namespace std;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

struct Cheby
{
    int pivot;
    double sum;
};

bool min_cmp(const Cheby& a, const Cheby& b)
{
    if (a.sum == b.sum) return a.pivot < b.pivot;
    else return a.sum < b.sum;
}

bool max_cmp(const Cheby& a, const Cheby& b)
{
    if (a.sum == b.sum) return a.pivot < b.pivot;
    else return a.sum > b.sum;
}

__global__ void SumDistance(Cheby* Dis_dev, const int* com_dev, const double* rebuilt_dev, const int npoints, 
                        const int n, const int k)
{
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx >= npoints) return;
    int ord = threadIdx.x;
    int sum = idx + 1, cur = n - 1, ki = k - 1;
    extern __shared__ int pivots[];
    while (ki)
    {
        int com = com_dev[cur * k + ki];
        if (sum > com)
            sum -= com;
        else
        {
            pivots[ord * k + ki] = n - cur - 1;
            ki--;
        }
        cur--;
    }
    pivots[ord * k] = pivots[ord * k + 1] + sum;
    double chebyshevSum = 0;
    for(int i = 0; i < n; i++){
        for(int j = i + 1; j < n; j++){
            double chebyshev = 0;
            for(int ki = 0; ki < k; ki++){
                double dis = fabs(rebuilt_dev[i * n + pivots[ord * k + ki]] - rebuilt_dev[j * n + pivots[ord * k + ki]]);
                if (chebyshev < dis)
                    chebyshev = dis;
            }
            chebyshevSum += 2 * chebyshev;
        }
    }

    Dis_dev[idx].pivot = idx;
    Dis_dev[idx].sum = chebyshevSum;
}

// Recursive function Combination() : combine pivots and calculate the sum of distance while combining different pivots.
// ki  : current depth of the recursion
// k   : number of pivots
// n   : number of points
// dim : dimension of metric space
// M   : number of combinations to store
// coord  : coordinates of points
// pivots : indexes of pivots
// maxDistanceSum  : the largest M distance sum
// maxDisSumPivots : the top M pivots combinations
// minDistanceSum  : the smallest M distance sum
// minDisSumPivots : the bottom M pivots combinations
extern "C"
void Combination(const int k, const int n, const int dim, const int M, double* coord, int* pivots,
                 double* maxDistanceSum, int* maxDisSumPivots, double* minDistanceSum, int* minDisSumPivots){
    double iStart, iElaps;

    iStart = cpuSecond();

    int npoints = 1;
    for (int i = 1; i <= k; i++)
        npoints = npoints * (n - i + 1) / i;

    int *com_host = (int*)malloc(sizeof(int) * n * k);
    double *rebuilt_host = (double*)malloc(sizeof(double) * n * n);
    Cheby *Dis_host = (Cheby*)malloc(sizeof(Cheby) * npoints);

    for (int i = 0; i < n; i++)
    {
        com_host[i * k] = 1;
        for (int j = 1; j < k && j <= i; j++)
        {
            com_host[i * k + j] = com_host[(i - 1) * k + j] + com_host[(i - 1) * k + j - 1];
        }
    }

    iElaps = cpuSecond() - iStart;
    printf("com and npoints: %lf\n", iElaps);

    iStart = cpuSecond();

    for(int i = 0; i < n; i++){
        rebuilt_host[i * n + i] = 0;
        for(int j = i + 1; j < n; j++){
            double distance = 0;
            for (int k = 0; k < dim; k++){
                distance += pow(coord[i*dim + k] - coord[j*dim + k], 2);
            }
            distance = sqrt(distance);
            rebuilt_host[i * n + j] = distance;
            rebuilt_host[j * n + i] = distance;
        }
    }
    
    int *com_dev = NULL;
    double *rebuilt_dev = NULL;
    Cheby *Dis_dev = NULL;
    cudaMalloc((void**)&com_dev, sizeof(int) * n * k);
    cudaMalloc((void**)&rebuilt_dev, sizeof(double) * n * n);
    cudaMalloc((void**)&Dis_dev, sizeof(Cheby) * npoints);
    cudaMemcpy(com_dev, com_host, sizeof(int) * n * k, cudaMemcpyHostToDevice);
    cudaMemcpy(rebuilt_dev, rebuilt_host, sizeof(double) * n * n, cudaMemcpyHostToDevice);

    iElaps = cpuSecond() - iStart;
    printf("rebuilt: %lf\n", iElaps);

    iStart = cpuSecond();

    dim3 block(BNUM);
    dim3 grid((npoints - 1) / block.x + 1);
    SumDistance<<< grid, block, BNUM * k * sizeof(int) >>>(Dis_dev, com_dev, rebuilt_dev, npoints, n, k);
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(Dis_host, Dis_dev, sizeof(Cheby) * npoints, cudaMemcpyDeviceToHost);

    iElaps = cpuSecond() - iStart;
    printf("kernel function: %lf\n", iElaps);

    iStart = cpuSecond();

    int piv, cur, ki, com;
    sort(Dis_host, Dis_host + npoints, max_cmp);

    iElaps = cpuSecond() - iStart;
    printf("sort1: %lf\n", iElaps);

    iStart = cpuSecond();

    for (int i = 0; i < M; i++)
    {
        maxDistanceSum[i] = Dis_host[i].sum;
        piv = Dis_host[i].pivot + 1; cur = n - 1; ki = k - 1;
        while (ki)
        {
            com = com_host[cur * k + ki];
            if (piv > com)
                piv -= com;
            else
            {
                maxDisSumPivots[i * k + k - ki - 1] = n - cur - 1;
                ki--;
            }
            cur--;
        }
        maxDisSumPivots[i * k + k - 1] = maxDisSumPivots[i * k + k - 2] + piv;
    }

    iElaps = cpuSecond() - iStart;
    printf("restore1: %lf\n", iElaps);

    iStart = cpuSecond();

    sort(Dis_host, Dis_host + npoints, min_cmp);

    iElaps = cpuSecond() - iStart;
    printf("sort1: %lf\n", iElaps);

    iStart = cpuSecond();

    for (int i = 0; i < M; i++)
    {
        minDistanceSum[i] = Dis_host[i].sum;
        piv = Dis_host[i].pivot + 1; cur = n - 1; ki = k - 1;
        while (ki)
        {
            com = com_host[cur * k + ki];
            if (piv > com)
                piv -= com;
            else
            {
                minDisSumPivots[i * k + k - ki - 1] = n - cur - 1;
                ki--;
            }
            cur--;
        }
        minDisSumPivots[i * k + k - 1] = minDisSumPivots[i * k + k - 2] + piv;
    }

    iElaps = cpuSecond() - iStart;
    printf("restore2: %lf\n", iElaps);

    free(com_host);
    free(rebuilt_host);
    free(Dis_host);
    cudaFree(com_dev);
    cudaFree(rebuilt_dev);
    cudaFree(Dis_dev);
    return;
}