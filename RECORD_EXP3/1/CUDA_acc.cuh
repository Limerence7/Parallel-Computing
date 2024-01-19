#include <iostream>
#include <math.h>
#include <sys/time.h>

#define CHECK(call)                                                     \
do                                                                      \
{                                                                       \
    const cudaError_t error_code = call;                                \
    if (error_code != cudaSuccess)                                      \
    {                                                                   \
        printf("CUDA Error:\n");                                        \
        printf("    File:   %s\n", __FILE__);                           \
        printf("    Line:   %d\n", __LINE__);                           \
        printf("    Error code: %d\n", error_code);                     \
        printf("    Error text: %s\n", cudaGetErrorString(error_code)); \
        exit(1);                                                        \
    }                                                                   \
}                                                                       \
while(0);                                                               \

extern "C"
void Combination(int ki, const int k, const int n, const int dim, const int M, double* coord, int* pivots,
                 double* maxDistanceSum, int* maxDisSumPivots, double* minDistanceSum, int* minDisSumPivots);

double SumDistance(const int k, const int n, const int dim, double* coord, int* pivots);