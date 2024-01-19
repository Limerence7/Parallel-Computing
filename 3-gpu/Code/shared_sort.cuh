#ifndef __SHARED_SORT_
#define __SHARED_SORT_

struct Cheby
{
    int pivot;
    double sum;
};

#include "shared_sort.cuh"

__host__ void shared_call_incre(int data_size, Cheby *cuda_data, int block_size);
__global__ void shared_sort_incre(Cheby *data, int data_size);
__global__ void naive_bitonic_sort_incre(Cheby *data,int i,int j);
__global__ void shared_merge2bitonic_sort_incre(Cheby *data,int data_size,int i,int block_size);

__host__ void shared_call_decre(int data_size, Cheby *cuda_data, int block_size);
__global__ void shared_sort_decre(Cheby *data, int data_size);
__global__ void naive_bitonic_sort_decre(Cheby *data,int i,int j);
__global__ void shared_merge2bitonic_sort_decre(Cheby *data,int data_size,int i,int block_size);

#endif
