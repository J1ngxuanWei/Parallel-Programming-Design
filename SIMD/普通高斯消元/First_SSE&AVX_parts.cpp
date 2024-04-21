// g++ First_SSE&AVX_parts.cpp -march=corei7 -march=corei7-avx -march=native
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
const int epochs = 100;

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

// 对第一部分做SSE并行化
void parallel_SSE_partOne()
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
            for (int j = k + 1; j < N; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            m[i][k] = 0;
        }
    }
}

// 对第二部分做SSE并行化
void parallel_SSE_partTwo()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
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

// 对第一部分做AVX并行化
void parallel_AVX_partOne()
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
            for (int j = k + 1; j < N; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            m[i][k] = 0;
        }
    }
}

// 对第二部分做AVX并行化
void parallel_AVX_partTwo()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
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

// 对第一部分做AVX-512并行化
void parallel_AVX512_partOne()
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
            for (int j = k + 1; j < N; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            m[i][k] = 0;
        }
    }
}

// 对第二部分做AVX-512并行化
void parallel_AVX512_partTwo()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
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

int main()
{
    struct timeval start, end;

    double time1 = 0;
    for (int i = 0; i < epochs; i++)
    {
        init();
        gettimeofday(&start, NULL);
        serial();
        gettimeofday(&end, NULL);
        time1 += ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    }
    cout << "N=" << N << " serial:" << time1 / epochs << endl;

    double time2 = 0;
    for (int i = 0; i < epochs; i++)
    {
        init();
        gettimeofday(&start, NULL);
        parallel_SSE_partOne();
        gettimeofday(&end, NULL);
        time2 += ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    }
    cout << "N=" << N << " SSE_partOne:" << time2 / epochs << endl;

    double time3 = 0;
    for (int i = 0; i < epochs; i++)
    {
        init();
        gettimeofday(&start, NULL);
        parallel_SSE_partTwo();
        gettimeofday(&end, NULL);
        time3 += ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    }
    cout << "N=" << N << " SSE_partTwo:" << time3 / epochs << endl;

    double time4 = 0;
    for (int i = 0; i < epochs; i++)
    {
        init();
        gettimeofday(&start, NULL);
        parallel_SSE_twoparts();
        gettimeofday(&end, NULL);
        time4 += ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    }
    cout << "N=" << N << " SSE_twoparts:" << time4 / epochs << endl;

    double time5;
    for (int i = 0; i < epochs; i++)
    {
        init();
        gettimeofday(&start, NULL);
        parallel_AVX_partOne();
        gettimeofday(&end, NULL);
        time5 += ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    }
    cout << "N=" << N << " AVX_partOne:" << time5 / epochs << endl;

    double time6 = 0;
    for (int i = 0; i < epochs; i++)
    {
        init();
        gettimeofday(&start, NULL);
        parallel_AVX_partTwo();
        gettimeofday(&end, NULL);
        time6 += ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    }
    cout << "N=" << N << " AVX_partTwo:" << time6 / epochs << endl;

    double time7 = 0;
    for (int i = 0; i < epochs; i++)
    {
        init();
        gettimeofday(&start, NULL);
        parallel_AVX_twoparts();
        gettimeofday(&end, NULL);
        time7 += ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    }
    cout << "N=" << N << " AVX_twoparts:" << time7 / epochs << endl;

    double time8;
    for (int i = 0; i < epochs; i++)
    {
        init();
        gettimeofday(&start, NULL);
        parallel_AVX512_partOne();
        gettimeofday(&end, NULL);
        time8 += ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    }
    cout << "N=" << N << " AVX-512_partOne:" << time8 / epochs << endl;

    double time9 = 0;
    for (int i = 0; i < epochs; i++)
    {
        init();
        gettimeofday(&start, NULL);
        parallel_AVX512_partTwo();
        gettimeofday(&end, NULL);
        time9 += ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    }
    cout << "N=" << N << " AVX-512_partTwo:" << time9 / epochs << endl;

    double time10 = 0;
    for (int i = 0; i < epochs; i++)
    {
        init();
        gettimeofday(&start, NULL);
        parallel_AVX512_twoparts();
        gettimeofday(&end, NULL);
        time10 += ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    }
    cout << "N=" << N << " AVX-512_twoparts:" << time10 / epochs << endl;

    cout << "------------------------------------------------------------------------" << endl;

    // 加速比
    cout << "SSE_part1:" << fixed << setprecision(2) << time1 / time2 << endl;
    cout << "SSE_part2:" << fixed << setprecision(2) << time1 / time3 << endl;
    cout << "SSE_2parts:" << fixed << setprecision(2) << time1 / time4 << endl;
    cout << "AVX_part1:" << fixed << setprecision(2) << time1 / time5 << endl;
    cout << "AVX_part2:" << fixed << setprecision(2) << time1 / time6 << endl;
    cout << "AVX_2parts:" << fixed << setprecision(2) << time1 / time7 << endl;
    cout << "AVX-512_part1:" << fixed << setprecision(2) << time1 / time8 << endl;
    cout << "AVX-512_part2:" << fixed << setprecision(2) << time1 / time9 << endl;
    cout << "AVX-512_2parts:" << fixed << setprecision(2) << time1 / time10 << endl;

    return 0;
}