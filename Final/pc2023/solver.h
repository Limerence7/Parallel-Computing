#include <iostream>
#include <stdio.h>
#include <math.h>
#include <time.h>
#include <omp.h>
#include <immintrin.h>
#include <cstring>
#include <sys/time.h>
#include <stdlib.h>
using namespace std;

#define INT int
#define DOU double
#define thread_nums 8

INT *start, *start1, *end, *end1;
DOU *mid_ans;

inline void thread_block(INT thread_id, INT start, INT end, INT start2, INT end2, INT *__restrict row_ptr, INT *__restrict col_idx, DOU *__restrict mtx_val, DOU *__restrict mtx_ans, DOU *__restrict mid_ans, DOU *__restrict vec_val)
{
    register INT start1, end1, num, Thread, i, j;
    register DOU sum;
    if (start < end)                                    // 该线程的负载不知在一行中
    {
        mtx_ans[start] = 0.0;
        mtx_ans[end] = 0.0;
        start1 = row_ptr[start] + start2;               // 分别获取首元素
        start++;
        end1 = row_ptr[start];                          // 和末元素
        Thread = thread_id << 1;                        // 左移×2，因为每个线程都需要左右边界
        sum = 0.0;
        #pragma unroll(8)                               // 展开循环
        for (j = start1; j < end1; j++)                 // 第一个单独计算
        {
            sum += mtx_val[j] * vec_val[col_idx[j]];
        }

        mid_ans[Thread] = sum;                          // 存放左边界结果
        start1 = end1;

        for (i = start; i < end; ++i)                   // 中间的都是独占一行的，不需要判断边界
        {
            end1 = row_ptr[i + 1];
            sum = 0.0;
            #pragma simd                                // 强制性的让编译器做自动并行化      
            for (j = start1; j < end1; j++)
            {
                sum += mtx_val[j] * vec_val[col_idx[j]];
            }
            mtx_ans[i] = sum;                           // 独占的直接存放到最终的结果数组
            start1 = end1;
        }
        start1 = row_ptr[end];                          // 最后的也单独处理
        end1 = start1 + end2;
        sum = 0.0;
        #pragma unroll(8)
        for (j = start1; j < end1; j++)
        {
            sum += mtx_val[j] * vec_val[col_idx[j]];
        }
        mid_ans[Thread | 1] = sum;                      // 存放到右边界
        return;
    }
    else                                                // 所有数据负载只在一行
    {
        mtx_ans[start] = 0.0;
        sum = 0.0;
        Thread = thread_id << 1;
        start1 = row_ptr[start] + start2;
        end1 = row_ptr[end] + end2;
        #pragma unroll(8)
        for (j = start1; j < end1; j++)
        {
            sum += mtx_val[j] * vec_val[col_idx[j]];
        }
        mid_ans[Thread] = sum;                          // 同理经过计算后，按照规定存放在左边界
        mid_ans[Thread | 1] = 0.0;
        return;
    }
}

inline INT binary_search(INT *&row_ptr, INT num, INT end)
{
    INT l, r, mid, t = 0;
    l = 0, r = end;
    while (l <= r)
    {
        mid = (l + r) >> 1;
        if (row_ptr[mid] >= num)        // 如果大于等于，利用mid缩小范围
            r = mid - 1;
        else                            // 如果小于，则缩小范围并记录结果
        {
            l = mid + 1;
            t = mid;
        }
    }
    return t;                           // 返回最后一次记录的结果
}

inline void albus_balance(INT *row_ptr, INT *&start, INT *&end, INT *&start1, INT *&end1, DOU *&mid_ans, INT row, INT nnz)
{
    register int tmp;
    // 头尾两个线程的start(1)和end(1)特殊处理
    start[0] = 0;
    start1[0] = 0;
    end[thread_nums - 1] = row;
    end1[thread_nums - 1] = 0;
    INT tt = nnz / thread_nums;                         // nnz为所有非零元素个数，均分给所有线程
    for (INT i = 1; i < thread_nums; i++)           
    {
        tmp = tt * i;                                   // 当前累计的负载数
        start[i] = binary_search(row_ptr, tmp, row);    // 找到恰好小于累计负载数的行序列号
        start1[i] = tmp - row_ptr[start[i]];            // 表示从当前行需要额外多少个才能刚好满足平均负载数
        end[i - 1] = start[i];
        end1[i - 1] = start1[i];
    }
}

inline void solver_fun(INT *__restrict row_ptr, INT *__restrict col_idx, DOU *__restrict mtx_val, DOU *__restrict mtx_ans, DOU *__restrict vec_val, INT row_ptr_num, INT row, INT col, INT nnz)
{
    omp_set_num_threads(8);                 // 指定线程数
    static bool if_balanced = false;        // 检验是否预处理(负载平衡)过
    if (!if_balanced)
    {
        start = (INT *)aligned_alloc(64, sizeof(INT) * thread_nums);
        start1 = (INT *)aligned_alloc(64, sizeof(INT) * thread_nums);
        end = (INT *)aligned_alloc(64, sizeof(INT) * thread_nums);
        end1 = (INT *)aligned_alloc(64, sizeof(INT) * thread_nums);
        mid_ans = (DOU *)aligned_alloc(64, sizeof(DOU) * thread_nums * 2);
        albus_balance(row_ptr, start, end, start1, end1, mid_ans, row, nnz);    // 分配负载到每一个线程

        if_balanced = true;
    }

    register INT i;
    #pragma omp parallel private(i)                 // 创建openmp parallel 并行域
    {
        #pragma omp for schedule(static) nowait     // omp for循环执行
        for (i = 0; i < thread_nums; ++i)
        {
            thread_block(i, start[i], end[i], start1[i], end1[i], row_ptr, col_idx, mtx_val, mtx_ans, mid_ans, vec_val);
        }
    }

    mtx_ans[0] = mid_ans[0];                        // 0号线程一定是从第零个元素开始的，直接将左边界赋给答案数组
    INT sub;
    #pragma unroll(32)                              // 展开循环（最大为32）
    for (i = 1; i < thread_nums; ++i)               
    {
        sub = i << 1;                               // 和Thread变量同理，区分左右边界
        register INT tmp1 = start[i];               // 开始元素的行序列号
        register INT tmp2 = end[i - 1];             // 结束元素的行序列号
        if (tmp1 == tmp2)                           // 若在同一行则加上左右边界
        {
            mtx_ans[tmp1] += (mid_ans[sub - 1] + mid_ans[sub]);
        }
        else                                        // 不在同一行则分别将不同行加上对应的左或右边界对应存储值
        {
            mtx_ans[tmp1] += mid_ans[sub];
            mtx_ans[tmp2] += mid_ans[sub - 1];
        }
    }
}