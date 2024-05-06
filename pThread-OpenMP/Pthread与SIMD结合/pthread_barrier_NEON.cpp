// g++ −g −march=native pthread_barrier_NEON.cpp -o pp -lpthread
#include <iostream>
#include <sys/time.h>
#include <pthread.h>
#include <semaphore.h>
#include <arm_neon.h>
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
            m[i][j] = rand() % 10;
    }
    for (int k = 0; k < N; k++)
        for (int i = k + 1; i < N; i++)
            for (int j = 0; j < N; j++)
                m[i][j] += m[k][j];
}

typedef struct
{
    int N;     // 矩阵规模
    float **m; // 矩阵
    int t_id;  // 线程id
} threadParam_t;

pthread_barrier_t barrier_Divsion;
pthread_barrier_t barrier_Elimination;

void *threadFuncROW(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int N = p->N;
    float **m = p->m;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++)
    {
        if (t_id == 0)
        {
            for (int j = k + 1; j < N; j++)
                m[k][j] = m[k][j] / m[k][k];
            m[k][k] = 1.0;
        }
        // 第一个同步点
        pthread_barrier_wait(&barrier_Divsion);

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
        {
            for (int j = k + 1; j < N; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            m[i][k] = 0.0;
        }
        // 第二个同步点
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return NULL;
}

void mainfuncROW(int N, float **m)
{
    pthread_barrier_init(&barrier_Divsion, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);
    // 创建线程
    pthread_t handles[NUM_THREADS];   // 创 建 对 应 的 Handle
    threadParam_t param[NUM_THREADS]; // 创建对应的线程数据结构
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].N = N;
        param[t_id].m = m;
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFuncROW, (void *)&param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        pthread_join(handles[t_id], NULL);

    // 销毁所有的 barrier
    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrier_destroy(&barrier_Elimination);
}

void *threadFuncNEON(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int N = p->N;
    float **m = p->m;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++)
    {
        if (t_id == 0)
        {
            for (int j = k + 1; j < N; j++)
                m[k][j] = m[k][j] / m[k][k];
            m[k][k] = 1.0;
        }
        // 第一个同步点
        pthread_barrier_wait(&barrier_Divsion);

        // 循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
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
        // 第二个同步点
        pthread_barrier_wait(&barrier_Elimination);
    }
    pthread_exit(NULL);
    return NULL;
}

void mainfuncNEON(int N, float **m)
{
    pthread_barrier_init(&barrier_Divsion, NULL, NUM_THREADS);
    pthread_barrier_init(&barrier_Elimination, NULL, NUM_THREADS);
    // 创建线程
    pthread_t handles[NUM_THREADS];   // 创 建 对 应 的 Handle
    threadParam_t param[NUM_THREADS]; // 创建对应的线程数据结构
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
    {
        param[t_id].N = N;
        param[t_id].m = m;
        param[t_id].t_id = t_id;
        pthread_create(&handles[t_id], NULL, threadFuncNEON, (void *)&param[t_id]);
    }
    for (int t_id = 0; t_id < NUM_THREADS; t_id++)
        pthread_join(handles[t_id], NULL);

    // 销毁所有的 barrier
    pthread_barrier_destroy(&barrier_Divsion);
    pthread_barrier_destroy(&barrier_Elimination);
}

int main()
{
    for (int N = 64; N <= 4096; N *= 2)
    {
        float **m = new float *[N];
        for (int i = 0; i < N; i++)
            m[i] = new float[N];

        struct timeval start, end;

        /*
        double timeuse1 = 0;
        m_reset(N, m);
        gettimeofday(&start, NULL);
        mainfuncROW(N, m);
        gettimeofday(&end, NULL);
        timeuse1 = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
        cout << "N=" << N << " 静态+barrier(row):" << timeuse1 << " ";
        */

        double timeuse2 = 0;
        m_reset(N, m);
        gettimeofday(&start, NULL);
        mainfuncNEON(N, m);
        gettimeofday(&end, NULL);
        timeuse2 = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
        cout << "N=" << N << " 静态+barrier(NEON):" << timeuse2 << endl;

        for (int i = 0; i < N; i++)
            delete[] m[i];
        delete[] m;
    }
}
