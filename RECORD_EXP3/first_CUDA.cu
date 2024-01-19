#include <cuda_runtime.h>
#include "CUDA_acc.cuh"

__global__ void kernel_chebyshev(const double* rebuilt_dev, double* result, const int n, const int k)
{
    unsigned int idx = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int i = idx / n, j = idx % n;
    if (i >= n) return;
    for(unsigned int ki = 0; ki < k; ki++)
    {
        double dis = fabs(rebuilt_dev[i * k + ki] - rebuilt_dev[j * k + ki]);
        result[idx] = dis > result[idx] ? dis : result[idx];
    }
}

__global__ void reduceCompleteUnrollWarp8(double *result, double *data_dev, unsigned int n)
{
	//set thread ID
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x * blockIdx.x * 8 + threadIdx.x;
	//boundary check
	if (tid >= n) return;
	//convert global data pointer to the
	double *idata = result + blockIdx.x*blockDim.x*8;
	if(idx+7 * blockDim.x<n)
	{
		double a1=result[idx];
		double a2=result[idx+blockDim.x];
		double a3=result[idx+2*blockDim.x];
		double a4=result[idx+3*blockDim.x];
		double a5=result[idx+4*blockDim.x];
		double a6=result[idx+5*blockDim.x];
		double a7=result[idx+6*blockDim.x];
		double a8=result[idx+7*blockDim.x];
		result[idx]=a1+a2+a3+a4+a5+a6+a7+a8;

	}
	__syncthreads();
	//in-place reduction in global memory
	if(blockDim.x>=1024 && tid <512)
		idata[tid]+=idata[tid+512];
	__syncthreads();
	if(blockDim.x>=512 && tid <256)
		idata[tid]+=idata[tid+256];
	__syncthreads();
	if(blockDim.x>=256 && tid <128)
		idata[tid]+=idata[tid+128];
	__syncthreads();
	if(blockDim.x>=128 && tid <64)
		idata[tid]+=idata[tid+64];
	__syncthreads();
	//write result for this block to global mem
	if(tid<32)
	{
		volatile double *vmem = idata;
		vmem[tid]+=vmem[tid+32];
		vmem[tid]+=vmem[tid+16];
		vmem[tid]+=vmem[tid+8];
		vmem[tid]+=vmem[tid+4];
		vmem[tid]+=vmem[tid+2];
		vmem[tid]+=vmem[tid+1];

	}

	if (tid == 0)
		data_dev[blockIdx.x] = idata[0];

}

// Calculate sum of distance while combining different pivots. Complexity : O( n^2 )
double SumDistance(const int k, const int n, const int dim, double* coord, int* pivots){
    double* rebuiltCoord = (double*)malloc(sizeof(double) * n * k);
    double *data_host = (double*)malloc(sizeof(double) * 512);

    int i;
    for(i=0; i<n*k; i++){
        rebuiltCoord[i] = 0;
    }

    double *rebuilt_dev = NULL, *result = NULL, *data_dev = NULL;
    cudaMalloc((void**)&rebuilt_dev, sizeof(double) * n * k);
    cudaMalloc((void**)&result, sizeof(double) * 262144);
    cudaMalloc((void**)&data_dev, sizeof(double) * 512);

    // Rebuild coordinates. New coordinate of one point is its distance to each pivot.
    for(i=0; i<n; i++){
        int ki;
        for(ki=0; ki<k; ki++){
            double distance = 0;
            int pivoti = pivots[ki];
            int j;
            for(j=0; j<dim; j++){
                distance += pow(coord[pivoti*dim + j] - coord[i*dim + j], 2);
            }
            rebuiltCoord[i*k + ki] = sqrt(distance);
        }
    }
    cudaMemcpy(rebuilt_dev, rebuiltCoord, sizeof(double) * n * k, cudaMemcpyHostToDevice);

    // Calculate the sum of Chebyshev distance with rebuilt coordinates between every points
    dim3 block(512);
    dim3 grid(512);
    kernel_chebyshev<<< grid, block >>>(rebuilt_dev, result, n, k);
    cudaDeviceSynchronize();

    reduceCompleteUnrollWarp8 <<< grid.x / 8, block >>>(result, data_dev, 262144);
    cudaDeviceSynchronize();
    cudaMemcpy(data_host, data_dev, grid.x * sizeof(double), cudaMemcpyDeviceToHost);

	double chebyshevSum = 0;
	for (int i = 0; i < grid.x/8; i++)
		chebyshevSum += data_host[i];
    
    free(rebuiltCoord);
    free(data_host);
    cudaFree(rebuilt_dev);
    cudaFree(result);
    cudaFree(data_dev);

    return chebyshevSum;
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
    int cnt = -1;
    int *limit = new int[k];
    for (int i = 0; i < k; i++)
    {
        limit[i] = n - (k - i);
        pivots[i] = i;
    }
    pivots[k - 1] = k - 2;

    int num = 0;
    int npoints = 1;
    for (int i = 1; i <= k; i++)
        npoints = npoints * (n - i + 1) / i;

    for (unsigned int i = 1; i <= npoints; i++)
    {
        int pos = k - 1;
        pivots[pos]++;
        while (pivots[pos] > limit[pos])
            pivots[--pos]++;
        for (unsigned int j = pos + 1; j < k; j++)
            pivots[j] = pivots[j - 1] + 1;

        // Calculate sum of distance while combining different pivots.
        double distanceSum = SumDistance(k, n, dim, coord, pivots);

        // put data at the end of array
        maxDistanceSum[num] = distanceSum;
        minDistanceSum[num] = distanceSum;
        int kj;
        for(kj=0; kj<k; kj++){
            maxDisSumPivots[num*k + kj] = pivots[kj];
        }
        for(kj=0; kj<k; kj++){
            minDisSumPivots[num*k + kj] = pivots[kj];
        }
        // sort
        int a;
        for(a=num; a>0; a--){
            if(maxDistanceSum[a] > maxDistanceSum[a-1]){
                double temp = maxDistanceSum[a];
                maxDistanceSum[a] = maxDistanceSum[a-1];
                maxDistanceSum[a-1] = temp;
                int kj;
                for(kj=0; kj<k; kj++){
                    int temp = maxDisSumPivots[a*k + kj];
                    maxDisSumPivots[a*k + kj] = maxDisSumPivots[(a-1)*k + kj];
                    maxDisSumPivots[(a-1)*k + kj] = temp;
                }
            }
            else
                break;
        }
        for(a=num; a>0; a--){
            if(minDistanceSum[a] < minDistanceSum[a-1]){
                double temp = minDistanceSum[a];
                minDistanceSum[a] = minDistanceSum[a-1];
                minDistanceSum[a-1] = temp;
                int kj;
                for(kj=0; kj<k; kj++){
                    int temp = minDisSumPivots[a*k + kj];
                    minDisSumPivots[a*k + kj] = minDisSumPivots[(a-1)*k + kj];
                    minDisSumPivots[(a-1)*k + kj] = temp;
                }
            }
            else
                break;
        }
        if (num < M)
            num++;

        // if(pivots[0] != cnt){
        //     cnt++;
        //     int kj;
        //     for(kj=0; kj<k; kj++){
        //         printf("%d ", pivots[kj]);
        //     }
        //     putchar('\t');
        //     for(kj=0; kj<k; kj++){
        //         printf("%d ", maxDisSumPivots[kj]);
        //     }
        //     printf("%lf\t", maxDistanceSum[0]);
        //     for(kj=0; kj<k; kj++){
        //         printf("%d ", minDisSumPivots[kj]);
        //     }
        //     printf("%lf\n", minDistanceSum[0]);
        // }
    }
}