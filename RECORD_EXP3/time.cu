#include <cuda_runtime.h>
#include "CUDA_acc.cuh"

__device__ int min_pos, max_pos;

double cpuSecond() {
    struct timeval tp;
    gettimeofday(&tp,NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}  

// Calculate sum of distance while combining different pivots. Complexity : O( n^2 )
double SumDistance(const int k, const int n, const int dim, double* coord, int* pivots){
    static double iStart, rebuild = 0.0, chebyshev = 0.0;
    static int cnt = 0;

    iStart = cpuSecond();

    double* rebuiltCoord = (double*)malloc(sizeof(double) * n * k);
    int i;
    for(i=0; i<n*k; i++){
        rebuiltCoord[i] = 0;
    }

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

    rebuild += cpuSecond() - iStart;

    iStart = cpuSecond();

    // Calculate the sum of Chebyshev distance with rebuilt coordinates between every points
    double chebyshevSum = 0;
    for(i=0; i<n; i++){
        int j;
        for(j=i+1; j<n; j++){
            double chebyshev = 0;
            int ki;
            for(ki=0; ki<k; ki++){
                double dis = fabs(rebuiltCoord[i*k + ki] - rebuiltCoord[j*k + ki]);
                chebyshev = dis>chebyshev ? dis : chebyshev;
            }
            chebyshevSum += 2 * chebyshev;
        }
    }

    free(rebuiltCoord);

    chebyshev += cpuSecond() - iStart;

    cnt++;
    if (cnt == 124750)
    {
        printf("rebuild part: %lf\n", rebuild);
        printf("chebyshevSum part: %lf\n", chebyshev);
    }

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
    double time1 = 0, time2 = 0, time3 = 0, time4 = 0;
    double iStart;

    iStart = cpuSecond();

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

    time1 = cpuSecond() - iStart;

    for (unsigned int i = 1; i <= npoints; i++)
    {
        iStart = cpuSecond();

        int pos = k - 1;
        pivots[pos]++;
        while (pivots[pos] > limit[pos])
            pivots[--pos]++;
        for (unsigned int j = pos + 1; j < k; j++)
            pivots[j] = pivots[j - 1] + 1;

        time2 += cpuSecond() - iStart;

        iStart = cpuSecond();
        // Calculate sum of distance while combining different pivots.
        double distanceSum = SumDistance(k, n, dim, coord, pivots);

        time3 += cpuSecond() - iStart;

        iStart = cpuSecond();

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

        time4 += cpuSecond() - iStart;

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
    printf("time1: %lf\n", time1);
    printf("time2: %lf\n", time2);
    printf("time3: %lf\n", time3);
    printf("time4: %lf\n", time4);
}