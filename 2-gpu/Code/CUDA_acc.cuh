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
void use_GPU(const float* dat_comp, const float* dat_real, const float* pri_comp, const float* pri_real, 
                const float* ctf, const float* sigRcp, const float* disturb, float* ans, const int m, const int K);