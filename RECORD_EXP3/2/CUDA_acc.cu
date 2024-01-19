#include <cuda_runtime.h>
#include "CUDA_acc.cuh"

// Calculate sum of distance while combining different pivots. Complexity : O( n^2 )
double SumDistance(const int k, const int n, const int dim, double* coord, int* pivots){
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

        if(pivots[0] != cnt){
            cnt++;
            int kj;
            for(kj=0; kj<k; kj++){
                printf("%d ", pivots[kj]);
            }
            putchar('\t');
            for(kj=0; kj<k; kj++){
                printf("%d ", maxDisSumPivots[kj]);
            }
            printf("%lf\t", maxDistanceSum[0]);
            for(kj=0; kj<k; kj++){
                printf("%d ", minDisSumPivots[kj]);
            }
            printf("%lf\n", minDistanceSum[0]);
        }
    }
}