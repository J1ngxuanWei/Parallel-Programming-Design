// g++ −g −march=native First_NEON_parts.cpp -o simd
#include <arm_neon.h>
#include <iostream>
#include <stdlib.h>
#include <iomanip>
#include <sys/time.h>
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

// 串行计算
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

// Neon并行化第一部分
void parallel_Neon_partOne()
{
    for (int k = 0; k < N; k++)
    {
        float tmp[4] = {m[k][k], m[k][k], m[k][k], m[k][k]};
        float32x4_t vt = vld1q_f32(tmp);
        vrecpeq_f32(vt);
        int a = k + 1;
        for (int j = k + 1; j + 4 <= N; j += 4, a = j)
        {
            float32x4_t va;
            va = vld1q_f32(m[k] + j);
            va = vmulq_f32(va, vt);
            vst1q_f32(m[k] + j, va);
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

// Neon并行化第二部分
void parallel_Neon_partTwo()
{
    for (int k = 0; k < N; k++)
    {
        for (int j = k + 1; j < N; j++)
            m[k][j] = m[k][j] / m[k][k];
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float32x4_t vaik;
            float tmp[4] = {m[i][k], m[i][k], m[i][k], m[i][k]};
            vaik = vld1q_f32(tmp);
            int a = k + 1;
            for (int j = k + 1; j + 4 <= N; j += 4, a = j)
            {
                float32x4_t vakj = vld1q_f32(m[k] + j);
                float32x4_t vaij = vld1q_f32(m[i] + j);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(m[i] + j, vaij);
            }
            for (int j = a; j < N; j++)
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            m[i][k] = 0;
        }
    }
}

// Neon并行化两部分
void parallel_Neon_twoparts()
{
    for (int k = 0; k < N; k++)
    {
        float tmp[4] = {m[k][k], m[k][k], m[k][k], m[k][k]};
        float32x4_t vt = vld1q_f32(tmp);
        vrecpeq_f32(vt);
        int a = k + 1;
        for (int j = k + 1; j + 4 <= N; j += 4, a = j)
        {
            float32x4_t va;
            va = vld1q_f32(m[k] + j);
            va = vmulq_f32(va, vt);
            vst1q_f32(m[k] + j, va);
        }
        for (int j = a; j < N; j++)
            m[k][j] = m[k][j] / m[k][k];
        m[k][k] = 1.0;
        for (int i = k + 1; i < N; i++)
        {
            float32x4_t vaik;
            float tmp[4] = {m[i][k], m[i][k], m[i][k], m[i][k]};
            vaik = vld1q_f32(tmp);
            int a = k + 1;
            for (int j = k + 1; j + 4 <= N; j += 4, a = j)
            {

                float32x4_t vakj = vld1q_f32(m[k] + j);
                float32x4_t vaij = vld1q_f32(m[i] + j);
                float32x4_t vx = vmulq_f32(vakj, vaik);
                vaij = vsubq_f32(vaij, vx);
                vst1q_f32(m[i] + j, vaij);
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
        parallel_Neon_partOne();
        gettimeofday(&end, NULL);
        time2 += ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    }
    cout << "N=" << N << "  Neon_partOne::" << time2 / epochs << endl;

    double time3 = 0;
    for (int i = 0; i < epochs; i++)
    {
        init();
        gettimeofday(&start, NULL);
        parallel_Neon_partTwo();
        gettimeofday(&end, NULL);
        time3 += ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    }
    cout << "N=" << N << " Neon_partTwo:" << time3 / epochs << endl;

    double time4 = 0;
    for (int i = 0; i < epochs; i++)
    {
        init();
        gettimeofday(&start, NULL);
        parallel_Neon_twoparts();
        gettimeofday(&end, NULL);
        time4 += ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    }
    cout << "N=" << N << " Neon_twoparts:" << time4 / epochs << endl;

    cout<<"---------------------------------------------------------------------"<<endl;

    // 加速比
    cout << "NEON_part1:" << fixed << setprecision(2) << time1 / time2 << endl;
    cout << "NEON_part2:" << fixed << setprecision(2) << time1 / time3 << endl;
    cout << "NEON_2parts:" << fixed << setprecision(2) << time1 / time4 << endl;
}