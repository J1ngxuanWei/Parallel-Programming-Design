// g++ −g −march=native Pthread3.cpp -o pp -lpthread
#include <iostream>
#include <sys/time.h>
#include <pthread.h>
#include <semaphore.h>
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

// 信号量
sem_t sem_leader;
sem_t sem_Divsion[NUM_THREADS - 1];
sem_t sem_Elimination[NUM_THREADS - 1];

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
        else
            sem_wait(&sem_Divsion[t_id - 1]);

        if (t_id == 0)
            for (int i = 0; i < NUM_THREADS - 1; i++)
                sem_post(&sem_Divsion[i]);

        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
        {
            for (int j = k + 1; j < N; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            m[i][k] = 0.0;
        }
        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++)
                sem_wait(&sem_leader);
            for (int i = 0; i < NUM_THREADS - 1; i++)
                sem_post(&sem_Elimination[i]);
        }
        else
        {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
    }
    pthread_exit(NULL);
}
void mainfuncROW(int N, float **m)
{
    sem_init(&sem_leader, 0, 0);

    for (int i = 0; i < NUM_THREADS - 1; i++)
    {
        sem_init(&sem_Divsion[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }

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

    sem_destroy(&sem_leader);
    for (int id = 0; id < NUM_THREADS - 1; id++)
        sem_destroy(&sem_Divsion[id]);
    for (int id = 0; id < NUM_THREADS - 1; id++)
        sem_destroy(&sem_Elimination[id]);
}
void *threadFuncCOL(void *param)
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
        else
            sem_wait(&sem_Divsion[t_id - 1]);

        if (t_id == 0)
            for (int i = 0; i < NUM_THREADS - 1; i++)
                sem_post(&sem_Divsion[i]);

        for (int i = k + 1; i < N; i++)
            for (int j = k + 1 + t_id; j < N; j += NUM_THREADS)
                m[i][j] = m[i][j] - m[k][j] * m[i][k];

        if (t_id == 0)
        {
            for (int i = 0; i < NUM_THREADS - 1; i++)
                sem_wait(&sem_leader);
            for (int i = 0; i < NUM_THREADS - 1; i++)
                sem_post(&sem_Elimination[i]);
        }
        else
        {
            sem_post(&sem_leader);
            sem_wait(&sem_Elimination[t_id - 1]);
        }
        for (int i = k + 1; i < N; i++)
            m[i][k] = 0;
    }
    pthread_exit(NULL);
}

void mainfuncCOL(int N, float **m)
{
    sem_init(&sem_leader, 0, 0);

    for (int i = 0; i < NUM_THREADS - 1; i++)
    {
        sem_init(&sem_Divsion[i], 0, 0);
        sem_init(&sem_Elimination[i], 0, 0);
    }

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

    sem_destroy(&sem_leader);
    for (int id = 0; id < NUM_THREADS - 1; id++)
        sem_destroy(&sem_Divsion[id]);
    for (int id = 0; id < NUM_THREADS - 1; id++)
        sem_destroy(&sem_Elimination[id]);
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
        mainfuncROW(N, m);
        gettimeofday(&end, NULL);
        timeuse1 = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
        cout << "N=" << N << " 静态+信号同步+三重循环全部纳入线程(row):" << timeuse1 << endl;

        double timeuse2 = 0;
        m_reset(N, m);
        gettimeofday(&start, NULL);
        mainfuncCOL(N, m);
        gettimeofday(&end, NULL);
        timeuse2 = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
        cout << "N=" << N << " 静态+信号同步+三重循环全部纳入线程(col):" << timeuse2 << endl;

        for (int i = 0; i < N; i++)
            delete[] m[i];
        delete[] m;
    }
}