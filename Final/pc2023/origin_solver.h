#define INT int
#define DOU double

void solver_fun(INT * row_ptr, INT * col_idx, DOU * mtx_val, DOU * mtx_ans, DOU * vec_val, 
           INT row_ptr_num,INT row, INT col, INT nnz)
{
    for(int i = 0; i < row; i++)
    {
        mtx_ans[i] = 0.0;
        for(int j = row_ptr[i]; j < row_ptr[i + 1]; j++)
        {
            mtx_ans[i] += mtx_val[j] * vec_val[col_idx[j]];
        }
    }
    return ;
}