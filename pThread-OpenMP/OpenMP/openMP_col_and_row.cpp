// g++ −g −march=native openMP_col_and_row.cpp -o pp -fopenmp
#include <iostream>
#include <sys/time.h>
#include <pthread.h>
#include <semaphore.h>
#include <nmmintrin.h>
#include <omp.h>
using namespace std;

const int NUM_THREADS = 2;

// 生成测试用例
void m_reset(int N, float **m)
{
    srand(2112495);
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < i; j++)
            m[i][j] = 0;
        m[i][i] = 1.0;
        for (int j = i + 1; j < N; j++)
            m[i][j] = rand() % 100;
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                m[i][j] += m[k][j];
}

void openMPROW(int N, float **m)
{
    int i, j, k;
    float temp;
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, temp)
    for (k = 0; k < N; k++)
    {
        // 串行部分
#pragma omp single
        {
            temp = m[k][k];
            for (j = k + 1; j < N; j++)
                m[k][j] = m[k][j] / temp;
            m[k][k] = 1.0;
        }
        // 并行部分，使用行划分
#pragma omp for
        for (i = k + 1; i < N; i++)
        {
            temp = m[i][k];
            for (j = k + 1; j < N; j++)
                m[i][j] = m[i][j] - temp * m[k][j];
            m[i][k] = 0.0;
        }
    }
}

void openMPCOL(int N, float **m)
{
    int i, j, k;
    float temp;
#pragma omp parallel num_threads(NUM_THREADS), private(i, j, k, temp)
    for (k = 0; k < N; k++)
    {
        // 串行部分
#pragma omp single
        {
            temp = m[k][k];
            for (j = k + 1; j < N; j++)
                m[k][j] = m[k][j] / temp;
            m[k][k] = 1.0;
        }
        // 并行部分，使用列划分
#pragma omp for
        for (j = k + 1; j < N; j++)
        {
            for (i = k + 1; i < N; i++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
        }

        for (int i = k + 1; i < N; i++)
            m[i][k] = 0.0;
    }
}

int main()
{
    for (int N = 64; N <= 4096; N *= 2)
    {
        float **m = new float *[N];
        for (int i = 0; i < N; i++)
            m[i] = new float[N];

        struct timeval start, end;

        double timeuse1 = 0;
        m_reset(N, m);
        gettimeofday(&start, NULL);
        openMPROW(N, m);
        gettimeofday(&end, NULL);
        timeuse1 = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
        cout << "N=" << N << " openMP(row):" << timeuse1 << endl;

        double timeuse2 = 0;
        m_reset(N, m);
        gettimeofday(&start, NULL);
        openMPCOL(N, m);
        gettimeofday(&end, NULL);
        timeuse2 = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
        cout << "N=" << N << " openMP(col):" << timeuse2 << endl;

        for (int i = 0; i < N; i++)
            delete[] m[i];
        delete[] m;
    }
}
