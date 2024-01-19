#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <sys/time.h>

#include "CUDA_acc.cuh"

int main(int argc, char* argv[]){
    // filename : input file namespace
    char* filename = (char*)"uniformvector-2dim-5h.txt";
    if( argc==2 ) {
        filename = argv[1];
    }  else if(argc != 1) {
        printf("Usage: ./pivot <filename>\n");
        return -1;
    }
    // M : number of combinations to store
    const int M = 1000;
    // dim : dimension of metric space
    int dim;
    // n : number of points
    int n;
    // k : number of pivots
    int k;

    // Read parameter
    FILE* file = fopen(filename, "r");
    if( file == NULL ) {
        printf("%s file not found.\n", filename);
        return -1;
    }
    fscanf(file, "%d", &dim);
    fscanf(file, "%d", &n);
    fscanf(file, "%d", &k);
    printf("dim = %d, n = %d, k = %d\n", dim, n, k);

    // Start timing
    struct timeval start;

    // Read Data
    double* coord = (double*)malloc(sizeof(double) * dim * n);
    int i;
    for(i=0; i<n; i++){
        int j;
        for(j=0; j<dim; j++){
            fscanf(file, "%lf", &coord[i*dim + j]);
        }
    }
    fclose(file);
    gettimeofday(&start, NULL);

    // maxDistanceSum : the largest M distance sum
    double* maxDistanceSum = (double*)malloc(sizeof(double) * (M+1));
    for(i=0; i<M; i++){
        maxDistanceSum[i] = 0;
    }
    // maxDisSumPivots : the top M pivots combinations
    int* maxDisSumPivots = (int*)malloc(sizeof(int) * k * (M+1));
    for(i=0; i<M; i++){
        int ki;
        for(ki=0; ki<k; ki++){
            maxDisSumPivots[i*k + ki] = 0;
        }
    }
    // minDistanceSum : the smallest M distance sum
    double* minDistanceSum = (double*)malloc(sizeof(double) * (M+1));
    for(i=0; i<M; i++){
        minDistanceSum[i] = __DBL_MAX__;
    }
    // minDisSumPivots : the bottom M pivots combinations
    int* minDisSumPivots = (int*)malloc(sizeof(int) * k * (M+1));
    for(i=0; i<M; i++){
        int ki;
        for(ki=0; ki<k; ki++){
            minDisSumPivots[i*k + ki] = 0;
        }
    }

    // temp : indexes of pivots with dummy array head
    int* temp = (int*)malloc(sizeof(int) * (k+1));
    temp[0] = -1;

    // Main loop. Combine different pivots with recursive function and evaluate them. Complexity : O( n^(k+2) )
    Combination(k, n, dim, M, coord, &temp[1], maxDistanceSum, maxDisSumPivots, minDistanceSum, minDisSumPivots);

    // End timing
    struct timeval end;
    gettimeofday (&end, NULL);
    printf("Using time : %f ms\n", (end.tv_sec-start.tv_sec)*1000.0+(end.tv_usec-start.tv_usec)/1000.0);

    // Store the result
    FILE* out = fopen("result.txt", "w");
    for(i=0; i<M; i++){
        int ki;
        for(ki=0; ki<k-1; ki++){
            fprintf(out, "%d ", maxDisSumPivots[i*k + ki]);
        }
        fprintf(out, "%d\n", maxDisSumPivots[i*k + k-1]);
    }
    for(i=0; i<M; i++){
        int ki;
        for(ki=0; ki<k-1; ki++){
            fprintf(out, "%d ", minDisSumPivots[i*k + ki]);
        }
        fprintf(out, "%d\n", minDisSumPivots[i*k + k-1]);
    }
    fclose(out);

    // Log
    int ki;
    printf("max : ");
    for(ki=0; ki<k; ki++){
        printf("%d ", maxDisSumPivots[ki]);
    }
    printf("%lf\n", maxDistanceSum[0]);
    printf("min : ");
    for(ki=0; ki<k; ki++){
        printf("%d ", minDisSumPivots[ki]);
    }
    printf("%lf\n", minDistanceSum[0]);
    // for(i=0; i<M; i++){
        // int ki;
        // for(ki=0; ki<k; ki++){
            // printf("%d\t", maxDisSumPivots[i*k + ki]);
        // }
        // printf("%lf\n", maxDistanceSum[i]);
    // }
    // for(i=0; i<M; i++){
        // int ki;
        // for(ki=0; ki<k; ki++){
            // printf("%d\t", minDisSumPivots[i*k + ki]);
        // }
        // printf("%lf\n", minDistanceSum[i]);
    // }

    return 0;
}
