#include <algorithm>
#include <sys/time.h>
#include <cuda_runtime.h>
#include "CUDA_acc.cuh"

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

__global__ void naive_bitonic_sort_decre(Cheby *data,int i,int j){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int neighour_data = tid ^ j;//find the pair data
	if(neighour_data > tid){//exchange data by low thread
		if(((tid / i) % 2) == 0){//sort ascending
			if((data[tid].sum < data[neighour_data].sum) || 
                    (data[tid].sum == data[neighour_data].sum && data[tid].pivot > data[neighour_data].pivot)){
				Cheby temp = data[tid];
				data[tid] = data[neighour_data];
				data[neighour_data] = temp;
			}
		}
		else if(((tid / i) % 2) == 1){//sort decending,exist the same data of same position
			if(data[tid].sum > data[neighour_data].sum || 
                    (data[tid].sum == data[neighour_data].sum && data[tid].pivot > data[neighour_data].pivot)){
				Cheby temp = data[tid];
				data[tid] = data[neighour_data];
				data[neighour_data] = temp;
			}
		}
	}
}

__global__ void naive_bitonic_sort_incre(Cheby *data,int i,int j){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int neighour_data = tid ^ j;//find the pair data
	if(neighour_data > tid){//exchange data by low thread
		if(((tid / i) % 2) == 0){//sort ascending
			if(data[tid].sum > data[neighour_data].sum || 
                    (data[tid].sum == data[neighour_data].sum && data[tid].pivot > data[neighour_data].pivot)){
				Cheby temp = data[tid];
				data[tid] = data[neighour_data];
				data[neighour_data] = temp;
			}
		}
		else if(((tid / i) % 2) == 1){//sort decending,exist the same data of same position
			if(data[tid].sum < data[neighour_data].sum || 
                    (data[tid].sum == data[neighour_data].sum && data[tid].pivot > data[neighour_data].pivot)){
				Cheby temp = data[tid];
				data[tid] = data[neighour_data];
				data[neighour_data] = temp;
			}
		}
	}
}
__host__ void naive_call(int data_size, Cheby *cuda_data, int block_size, bool ls) {
    if (ls)
    {
        for(int i = 2; i <= data_size; i = i * 2){//stride_len
            for(int j = i/2;j > 0;j = j/2){//calc for the neighborhood
                naive_bitonic_sort_decre<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1, block_size>>>(cuda_data,i,j);
            }
        }
    }
    else
    {
        for(int i = 2; i <= data_size; i = i * 2){//stride_len
            for(int j = i/2;j > 0;j = j/2){//calc for the neighborhood
                naive_bitonic_sort_incre<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1, block_size>>>(cuda_data,i,j);
            }
        }
    }
	
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
    cudaSetDevice(5);
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

    dim3 block(BNUM);
    dim3 grid((npoints - 1) / block.x + 1);
    SumDistance<<< grid, block, BNUM * k * sizeof(int) >>>(Dis_dev, com_dev, rebuilt_dev, npoints, n, k);
    CHECK(cudaDeviceSynchronize());

    cudaMemcpy(Dis_host, Dis_dev, sizeof(Cheby) * npoints, cudaMemcpyDeviceToHost);

    int bin_size = 1;
    while (bin_size <= npoints)
        bin_size *= 2;
    bin_size /= 2;

    sort(Dis_host + bin_size - M, Dis_host + npoints, max_cmp);

    cudaMemcpy(Dis_dev, Dis_host, sizeof(Cheby) * bin_size, cudaMemcpyHostToDevice);
    naive_call(bin_size, Dis_dev, BNUM, true);
    cudaDeviceSynchronize();
    cudaMemcpy(Dis_host, Dis_dev, sizeof(Cheby) * bin_size, cudaMemcpyDeviceToHost);

    int piv, cur, ki, com;
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

    sort(Dis_host + bin_size - M, Dis_host + npoints, min_cmp);

    cudaMemcpy(Dis_dev, Dis_host, sizeof(Cheby) * bin_size, cudaMemcpyHostToDevice);
    naive_call(bin_size, Dis_dev, BNUM, false);
    cudaDeviceSynchronize();
    cudaMemcpy(Dis_host, Dis_dev, sizeof(Cheby) * M, cudaMemcpyDeviceToHost);

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
    free(rebuilt_host);
    free(Dis_host);
    cudaFree(com_dev);
    cudaFree(rebuilt_dev);
    cudaFree(Dis_dev);
}