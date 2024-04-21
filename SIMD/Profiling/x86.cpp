// g++ x86.cpp -march=corei7 -march=corei7-avx -march=native
// vtune -collect performance-snapshot -result-dir r001hs -quiet /home/u220187/aaa
// ./a.out
#include <iostream>
#include <stdlib.h>
#include <sys/time.h>
#include <iomanip>
#include <nmmintrin.h>
#include <immintrin.h>
using namespace std;

const int N = 64;
float m[N][N];

// 初始化矩阵
void init()
{
    srand(2112495);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
            m[i][j] = 0;
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            m[i][j] = rand();
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                m[i][j] += m[k][j];
}

// 串行
void serial()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
            m[k][j] = m[k][j] / m[k][k];
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1; j < N; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            m[i][k] = 0;
        }
    }
}

// 对两部分做SSE并行化(不对齐)
void parallel_SSE_twoparts()
{
    for (int k = 0; k < N; k++)
    {
        float tmp[4] = {m[k][k], m[k][k], m[k][k], m[k][k]};
        __m128 vt = _mm_loadu_ps(tmp);
        int a = k + 1;
        for (int j = k + 1; j + 4 <= N; j += 4, a = j)
        {
            __m128 va;
            va = _mm_loadu_ps(m[k] + j);
            va = _mm_div_ps(va, vt);
            _mm_storeu_ps(m[k] + j, va);
        }
        for (int j = a; j < N; j++)
            m[k][j] = m[k][j] / m[k][k];
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            __m128 vaik;
            float tmp[4] = {m[i][k], m[i][k], m[i][k], m[i][k]};
            vaik = _mm_loadu_ps(tmp);
            int a = k + 1;
            for (int j = k + 1; j + 4 <= N; j += 4, a = j)
            {
                __m128 vakj = _mm_loadu_ps(m[k] + j);
                __m128 vaij = _mm_loadu_ps(m[i] + j);
                __m128 vx = _mm_mul_ps(vakj, vaik);
                vaij = _mm_sub_ps(vaij, vx);
                _mm_storeu_ps(m[i] + j, vaij);
            }
            for (int j = a; j < N; j++)
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            m[i][k] = 0;
        }
    }
}

// 对两部分做AVX并行化(不对齐)
void parallel_AVX_twoparts()
{
    for (int k = 0; k < N; k++)
    {
        float tmp[8] = {m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k]};
        __m256 vt = _mm256_loadu_ps(tmp);
        int a = k + 1;
        for (int j = k + 1; j + 8 <= N; j += 8, a = j)
        {
            __m256 va;
            va = _mm256_loadu_ps(m[k] + j);
            va = _mm256_div_ps(va, vt);
            _mm256_storeu_ps(m[k] + j, va);
        }
        for (int j = a; j < N; j++)
            m[k][j] = m[k][j] / m[k][k];
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            __m256 vaik;
            float tmp[8] = {m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k]};
            vaik = _mm256_loadu_ps(tmp);
            int a = k + 1;
            for (int j = k + 1; j + 8 <= N; j += 8, a = j)
            {
                __m256 vakj = _mm256_loadu_ps(m[k] + j);
                __m256 vaij = _mm256_loadu_ps(m[i] + j);
                __m256 vx = _mm256_mul_ps(vakj, vaik);
                vaij = _mm256_sub_ps(vaij, vx);
                _mm256_storeu_ps(m[i] + j, vaij);
            }
            for (int j = a; j < N; j++)
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            m[i][k] = 0;
        }
    }
}

int main()
{
    init();
    serial();
    parallel_SSE_twoparts();
    parallel_AVX_twoparts();

    return 0;
}