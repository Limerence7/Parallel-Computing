#include<cstdio>
#include<algorithm>
#include<cmath>
#include<sys/time.h>

#include"solver.h"

#define INT int
#define DOU double

using namespace std;

int main(int argc , char ** argv)
{
    //------------------calculation iterations------------------//
    char * filename;
    if(argc > 1)
    {
            filename = argv[1];
    }
    else
    {
            printf("Error!\n");
    }
    //----------------------------------------------------------//
    FILE * fp_mtx;
    fp_mtx = fopen(filename,"r+");

    INT  * row_ptr;
    INT  * col_idx;
    DOU  * mtx_val;
    DOU  * mtx_ans;
    DOU  * vec_val;
    INT    row;
    INT    col;
    INT    nnz;
    INT    row_ptr_num;

    fscanf(fp_mtx, "%d %d %d",&row, &col, &nnz);
    row_ptr_num = row + 1;

    row_ptr = (INT *)aligned_alloc(64, sizeof(INT) * row_ptr_num);
    col_idx = (INT *)aligned_alloc(64, sizeof(INT) * nnz);
    mtx_val = (DOU *)aligned_alloc(64, sizeof(DOU) * nnz);
    vec_val = (DOU *)aligned_alloc(64, sizeof(DOU) * col);
    mtx_ans = (DOU *)aligned_alloc(64, sizeof(DOU) * row);
    
    for(int i = 0; i < row_ptr_num; i++)
        fscanf(fp_mtx, "%d", &row_ptr[i]);

    for(int i = 0; i < nnz; i++)
        fscanf(fp_mtx, "%d", &col_idx[i]);

    for(int i = 0; i < nnz; i++)
        fscanf(fp_mtx, "%lf", &mtx_val[i]);

    for(int i = 0; i < col; i++)
        fscanf(fp_mtx, "%lf", &vec_val[i]);

    int ite = min(200000ull, max(100ull, ((16ull << 30) / nnz)));
    
    struct timeval startTime, endTime;

    //-----------------------------Warm Up---------------------------//
    for(INT i = 0; i < ite; i++)
    {
        solver_fun(row_ptr,col_idx,mtx_val,mtx_ans,vec_val,row_ptr_num,row,col,nnz);
    }
    //---------------------------------------------------------------//
    
    DOU Timeuse=0;

    gettimeofday(&startTime, NULL);
    for(INT i = 0; i < ite; i++)
    {
        solver_fun(row_ptr,col_idx,mtx_val,mtx_ans,vec_val,row_ptr_num,row,col,nnz);
    }
    gettimeofday(&endTime, NULL);

    Timeuse = (endTime.tv_sec - startTime.tv_sec) + (endTime.tv_usec - startTime.tv_usec)/1000000.0;
    printf("time = %lf ms , Gflops = %lf Gflops\n", Timeuse * 1000 / ite * 1.0, (nnz / (Timeuse / ite * 1.0)/1000000000) * 2.0);

    FILE * ans;
    ans = fopen("result.txt","w+");

    for(int i = 0; i < row; i++)
        fprintf(ans, "%.6lf\n", mtx_ans[i]);
    
    free(row_ptr);
    free(col_idx);
    free(mtx_val);
    free(mtx_ans);
    free(vec_val);
    return 0;
}
