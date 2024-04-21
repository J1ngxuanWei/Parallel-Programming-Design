// g++ AVX512_align.cpp -march=corei7 -march=corei7-avx -march=native
// vtune -collect performance-snapshot -result-dir r001hs -quiet /home/u220187/aaa
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

// 对两部分做AVX-512并行化(不对齐)
void parallel_AVX512_twoparts()
{
    for (int k = 0; k < N; k++)
    {
        float tmp[16] = {m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k]};
        __m512 vt = _mm512_loadu_ps(tmp);
        int a = k + 1;
        for (int j = k + 1; j + 16 <= N; j += 16, a = j)
        {
            __m512 va;
            va = _mm512_loadu_ps(m[k] + j);
            va = _mm512_div_ps(va, vt);
            _mm512_storeu_ps(m[k] + j, va);
        }
        for (int j = a; j < N; j++)
            m[k][j] = m[k][j] / m[k][k];
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            __m512 vaik;
            float tmp[16] = {m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k]};
            vaik = _mm512_loadu_ps(tmp);
            int a = k + 1;
            for (int j = k + 1; j + 16 <= N; j += 16, a = j)
            {
                __m512 vakj = _mm512_loadu_ps(m[k] + j);
                __m512 vaij = _mm512_loadu_ps(m[i] + j);
                __m512 vx = _mm512_mul_ps(vakj, vaik);
                vaij = _mm512_sub_ps(vaij, vx);
                _mm512_storeu_ps(m[i] + j, vaij);
            }
            for (int j = a; j < N; j++)
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            m[i][k] = 0;
        }
    }
}

// 对两部分做AVX-512并行化(对齐)
void parallel_AVX512_twoparts_align()
{
    for (int k = 0; k < N; k++)
    {
        int pre = 16 - (k + 1) % 16;
        for (int j = k + 1; j < k + 1 + pre; j++)
            m[k][j] = m[k][j] / m[k][k];
        float tmp[16] = {m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k], m[k][k]};
        __m512 vt = _mm512_loadu_ps(tmp);
        int a = k + 1 + pre;
        for (int j = k + 1 + pre; j + 16 <= N; j += 16, a = j)
        {
            __m512 va;
            va = _mm512_loadu_ps(m[k] + j);
            va = _mm512_div_ps(va, vt);
            _mm512_storeu_ps(m[k] + j, va);
        }
        for (int j = a; j < N; j++)
            m[k][j] = m[k][j] / m[k][k];
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            int pre = 16 - (k + 1) % 16;
            for (int j = k + 1; j < k + 1 + pre; j++)
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            __m512 vaik;
            float tmp[16] = {m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k], m[i][k]};
            vaik = _mm512_loadu_ps(tmp);
            int a = k + 1 + pre;
            for (int j = k + 1 + pre; j + 16 <= N; j += 16, a = j)
            {
                __m512 vakj = _mm512_loadu_ps(m[k] + j);
                __m512 vaij = _mm512_loadu_ps(m[i] + j);
                __m512 vx = _mm512_mul_ps(vakj, vaik);
                vaij = _mm512_sub_ps(vaij, vx);
                _mm512_storeu_ps(m[i] + j, vaij);
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
    parallel_AVX512_twoparts();
    // parallel_AVX512_twoparts_align();
    return 0;
}