#include "shared_sort.cuh"

__host__ void shared_call_incre(int data_size, Cheby *cuda_data, int block_size){
    shared_sort_incre<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1,block_size,block_size*sizeof(Cheby)>>>(cuda_data,data_size);
    if(block_size < data_size){
        for(int i = block_size*2 ;i <= data_size;i *=2){
            for(int j = i/2;j >= block_size;j = j/2){//calc for the neighborhood
                naive_bitonic_sort_incre<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1, block_size>>>(cuda_data,i,j);
			}
			shared_merge2bitonic_sort_incre<<<(data_size%(block_size) == 0)?data_size/(block_size):data_size/(block_size)+1,(block_size),block_size*sizeof(Cheby)>>>(cuda_data,data_size,i,block_size);
        }
    }

}

__global__ void shared_sort_incre(Cheby *data, int data_size){
	extern __shared__ Cheby smem[];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	smem[threadIdx.x] = data[tid];
    int end = min(data_size,blockDim.x);
    int neighour_data;
    Cheby temp;
    for(int i = 2;i <= end;i = i * 2){
        for(int j = i/2;j > 0;j = j/2){
            neighour_data = threadIdx.x ^ j;//find the pair data
            if(neighour_data > threadIdx.x){//exchange data by low thread
                if(((tid / i) % 2) == 0){//sort ascending
                    if(smem[threadIdx.x].sum > smem[neighour_data].sum || 
                    (smem[threadIdx.x].sum == smem[neighour_data].sum && smem[threadIdx.x].pivot > smem[neighour_data].pivot)){
                        temp = smem[threadIdx.x];
                        smem[threadIdx.x] = smem[neighour_data];
                        smem[neighour_data] = temp;
                    }
                }
                else if(((tid / i) % 2) == 1){//sort decending,exist the same smem of same position
                    if(smem[threadIdx.x].sum < smem[neighour_data].sum || 
                    (smem[threadIdx.x].sum == smem[neighour_data].sum && smem[threadIdx.x].pivot > smem[neighour_data].pivot)){
                        temp = smem[threadIdx.x];
                        smem[threadIdx.x] = smem[neighour_data];
                        smem[neighour_data] = temp;
                    }
                }
            }
            __syncthreads();
        }
	}
	data[tid] = smem[threadIdx.x];
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

__global__ void shared_merge2bitonic_sort_incre(Cheby *data,int data_size,int i,int block_size){
	extern __shared__ Cheby smem[];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	smem[threadIdx.x] = data[tid];
	__syncthreads();
    int neighour_data;
	Cheby temp;
	if(((tid / i) % 2) == 0) {
		for(int j = block_size/2;j > 0;j = j/2){
			neighour_data = threadIdx.x ^ j;//find the pair data
			if(neighour_data > threadIdx.x && (smem[threadIdx.x].sum > smem[neighour_data].sum || 
                    (smem[threadIdx.x].sum == smem[neighour_data].sum && smem[threadIdx.x].pivot > smem[neighour_data].pivot))){//exchange data by low thread
				temp = smem[threadIdx.x];
				smem[threadIdx.x] = smem[neighour_data];
				smem[neighour_data] = temp;
			}
			__syncthreads();
		}
	}
	if(((tid / i) % 2) == 1) {
		for(int j = block_size/2;j > 0;j = j/2){
			neighour_data = threadIdx.x ^ j;//find the pair data
			if(neighour_data > threadIdx.x && (smem[threadIdx.x].sum < smem[neighour_data].sum || 
                    (smem[threadIdx.x].sum == smem[neighour_data].sum && smem[threadIdx.x].pivot > smem[neighour_data].pivot))){//exchange data by low thread
				temp = smem[threadIdx.x];
				smem[threadIdx.x] = smem[neighour_data];
				smem[neighour_data] = temp;
			}
			__syncthreads();
		}
	}
	data[tid] = smem[threadIdx.x];
}


__host__ void shared_call_decre(int data_size, Cheby *cuda_data, int block_size){
    shared_sort_decre<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1,block_size,block_size*sizeof(Cheby)>>>(cuda_data,data_size);
    if(block_size < data_size){
        for(int i = block_size*2 ;i <= data_size;i *=2){
            for(int j = i/2;j >= block_size;j = j/2){//calc for the neighborhood
                naive_bitonic_sort_decre<<<(data_size%block_size == 0) ? data_size/block_size : data_size/block_size + 1, block_size>>>(cuda_data,i,j);
			}
			shared_merge2bitonic_sort_decre<<<(data_size%(block_size) == 0)?data_size/(block_size):data_size/(block_size)+1,(block_size),block_size*sizeof(Cheby)>>>(cuda_data,data_size,i,block_size);
        }
    }

}

__global__ void shared_sort_decre(Cheby *data, int data_size){
	extern __shared__ Cheby smem[];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	smem[threadIdx.x] = data[tid];
    int end = min(data_size,blockDim.x);
    int neighour_data;
    Cheby temp;
    for(int i = 2;i <= end;i = i * 2){
        for(int j = i/2;j > 0;j = j/2){
            neighour_data = threadIdx.x ^ j;//find the pair data
            if(neighour_data > threadIdx.x){//exchange data by low thread
                if(((tid / i) % 2) == 0){//sort ascending
                    if(smem[threadIdx.x].sum < smem[neighour_data].sum || 
                    (smem[threadIdx.x].sum == smem[neighour_data].sum && smem[threadIdx.x].pivot > smem[neighour_data].pivot)){
                        temp = smem[threadIdx.x];
                        smem[threadIdx.x] = smem[neighour_data];
                        smem[neighour_data] = temp;
                    }
                }
                else if(((tid / i) % 2) == 1){//sort decending,exist the same smem of same position
                    if(smem[threadIdx.x].sum > smem[neighour_data].sum || 
                    (smem[threadIdx.x].sum == smem[neighour_data].sum && smem[threadIdx.x].pivot > smem[neighour_data].pivot)){
                        temp = smem[threadIdx.x];
                        smem[threadIdx.x] = smem[neighour_data];
                        smem[neighour_data] = temp;
                    }
                }
            }
            __syncthreads();
        }
	}
	data[tid] = smem[threadIdx.x];
}

__global__ void naive_bitonic_sort_decre(Cheby *data,int i,int j){
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	int neighour_data = tid ^ j;//find the pair data
	if(neighour_data > tid){//exchange data by low thread
		if(((tid / i) % 2) == 0){//sort ascending
			if(data[tid].sum < data[neighour_data].sum || 
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

__global__ void shared_merge2bitonic_sort_decre(Cheby *data,int data_size,int i,int block_size){
	extern __shared__ Cheby smem[];
	int tid = threadIdx.x + blockDim.x * blockIdx.x;
	smem[threadIdx.x] = data[tid];
	__syncthreads();
    int neighour_data;
	Cheby temp;
	if(((tid / i) % 2) == 0) {
		for(int j = block_size/2;j > 0;j = j/2){
			neighour_data = threadIdx.x ^ j;//find the pair data
			if(neighour_data > threadIdx.x && (smem[threadIdx.x].sum < smem[neighour_data].sum || 
                    (smem[threadIdx.x].sum == smem[neighour_data].sum && smem[threadIdx.x].pivot > smem[neighour_data].pivot))){//exchange data by low thread
				temp = smem[threadIdx.x];
				smem[threadIdx.x] = smem[neighour_data];
				smem[neighour_data] = temp;
			}
			__syncthreads();
		}
	}
	if(((tid / i) % 2) == 1) {
		for(int j = block_size/2;j > 0;j = j/2){
			neighour_data = threadIdx.x ^ j;//find the pair data
			if(neighour_data > threadIdx.x && (smem[threadIdx.x].sum > smem[neighour_data].sum || 
                    (smem[threadIdx.x].sum == smem[neighour_data].sum && smem[threadIdx.x].pivot > smem[neighour_data].pivot))){//exchange data by low thread
				temp = smem[threadIdx.x];
				smem[threadIdx.x] = smem[neighour_data];
				smem[neighour_data] = temp;
			}
			__syncthreads();
		}
	}
	data[tid] = smem[threadIdx.x];
}