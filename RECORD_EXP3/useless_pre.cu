#include <algorithm>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "CUDA_acc.cuh"

#define BREB 32
#define BNUM 1024

using namespace std;

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

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

__global__ void Cal_rebuilt(const double* coord_dev, double* rebuilt_dev, int dim, const int n)
{
    unsigned int idx = (threadIdx.x + blockDim.x * blockIdx.x) / dim;
    unsigned int i = idx / n, j = idx % n, tid = threadIdx.x, pos = tid % dim;
    if (i >= n) return;
    extern __shared__ double distance[];
    distance[tid] = pow(coord_dev[i*dim + pos] - coord_dev[j*dim + pos], 2);
    if (pos < 2 && dim >= 4)
        distance[tid] += distance[tid + 2];
    if (pos == 0 && dim >= 2)
    {
        distance[tid] += distance[tid + 1];
        rebuilt_dev[idx] = sqrt(distance[tid]);
    }
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
void Combination(const int k, const int n, const int dim, const int M, double* coord,
                 double* maxDistanceSum, int* maxDisSumPivots, double* minDistanceSum, int* minDisSumPivots){
    int npoints = 1;
    for (int i = 1; i <= k; i++)
        npoints = npoints * (n - i + 1) / i;

    int *com_host = (int*)malloc(sizeof(int) * n * k);
    Cheby *Dis_host = (Cheby*)malloc(sizeof(Cheby) * npoints);

    for (int i = 0; i < n; i++)
    {
        com_host[i * k] = 1;
        for (int j = 1; j < k && j <= i; j++)
        {
            com_host[i * k + j] = com_host[(i - 1) * k + j] + com_host[(i - 1) * k + j - 1];
        }
    }
    double iStart = cpuSecond();
    int *com_dev = NULL;
    double *coord_dev = NULL;
    double *rebuilt_dev = NULL;
    Cheby *Dis_dev = NULL;
    cudaMalloc((void**)&com_dev, sizeof(int) * n * k);
    cudaMalloc((void**)&coord_dev, sizeof(double) * dim * n);
    cudaMalloc((void**)&rebuilt_dev, sizeof(double) * n * n);
    cudaMalloc((void**)&Dis_dev, sizeof(Cheby) * npoints);
    double iElaps = cpuSecond() - iStart;
    printf("%lf\n", iElaps);

    cudaMemcpy(com_dev, com_host, sizeof(int) * n * k, cudaMemcpyHostToDevice);

    iStart = cpuSecond();
    cudaMemcpy(coord_dev, coord, sizeof(double) * dim * n, cudaMemcpyHostToDevice);
    iElaps = cpuSecond() - iStart;
    printf("%lf\n", iElaps);
    iStart = cpuSecond();

    dim3 block_re(BREB);
    dim3 grid_re((n * n * dim - 1) / block_re.x + 1);
    Cal_rebuilt<<< grid_re, block_re, BREB >>>(coord_dev, rebuilt_dev, dim, n);
    CHECK(cudaDeviceSynchronize());
    iElaps = cpuSecond() - iStart;
    printf("%lf\n", iElaps);

    dim3 block(BNUM);
    dim3 grid((npoints - 1) / block.x + 1);
    SumDistance<<< grid, block, BNUM * k * sizeof(int) >>>(Dis_dev, com_dev, rebuilt_dev, npoints, n, k);
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(Dis_host, Dis_dev, sizeof(Cheby) * npoints, cudaMemcpyDeviceToHost);

    int piv, cur, ki, com;
    sort(Dis_host, Dis_host + npoints, max_cmp);
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

    sort(Dis_host, Dis_host + npoints, min_cmp);
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

    free(com_host);
    free(Dis_host);
    cudaFree(com_dev);
    cudaFree(coord_dev);
    cudaFree(rebuilt_dev);
    cudaFree(Dis_dev);
}