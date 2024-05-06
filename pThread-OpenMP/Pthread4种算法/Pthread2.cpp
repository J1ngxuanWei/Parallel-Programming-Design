// g++ −g −march=native Pthread2.cpp -o pp -lpthread
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
sem_t sem_main;
sem_t sem_workerstart[NUM_THREADS];
sem_t sem_workerend[NUM_THREADS];

void *threadFuncROW(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int N = p->N;
    float **m = p->m;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++)
    {
        sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作（操作自己专属的信号量）

        // 循环划分任务
        for (int i = k + 1 + t_id; i < N; i += NUM_THREADS)
        {
            for (int j = k + 1; j < N; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            m[i][k] = 0.0;
        }
        sem_post(&sem_main);            // 唤醒主线程
        sem_wait(&sem_workerend[t_id]); // 阻塞，等待主线程唤醒进入下一轮
    }
    pthread_exit(NULL);
    return 0;
}

void mainfuncROW(int N, float **m)
{
    sem_init(&sem_main, 0, 0);
    for (int id = 0; id < NUM_THREADS; id++)
        sem_init(&sem_workerstart[id], 0, 0);
    for (int id = 0; id < NUM_THREADS; id++)
        sem_init(&sem_workerend[id], 0, 0);
    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];

    for (int id = 0; id < NUM_THREADS; id++)
    {
        param[id].N = N;
        param[id].m = m;
        param[id].t_id = id;
        pthread_create(&handles[id], NULL, threadFuncROW, (void *)&param[id]);
    }
    for (int k = 0; k < N; k++)
    {
        int j;
        for (j = k + 1; j < N; j++)
            m[k][j] /= m[k][k];
        m[k][k] = 1.0;
        for (int id = 0; id < NUM_THREADS; id++)
            sem_post(&sem_workerstart[id]);
        for (int id = 0; id < NUM_THREADS; id++)
            sem_wait(&sem_main);
        for (int id = 0; id < NUM_THREADS; id++)
            sem_post(&sem_workerend[id]);
    }
    for (int id = 0; id < NUM_THREADS; id++)
        pthread_join(handles[id], NULL);
    sem_destroy(&sem_main);
    for (int id = 0; id < NUM_THREADS; id++)
        sem_destroy(&sem_workerstart[id]);
    for (int id = 0; id < NUM_THREADS; id++)
        sem_destroy(&sem_workerend[id]);
}

void *threadFuncCOL(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int N = p->N;
    float **m = p->m;
    int t_id = p->t_id;
    for (int k = 0; k < N; k++)
    {
        sem_wait(&sem_workerstart[t_id]); // 阻塞，等待主线完成除法操作（操作自己专属的信号量）

        // 循环划分任务
        for (int i = k + 1; i < N; i++)
        {
            for (int j = k + 1 + t_id; j < N; j += NUM_THREADS)
            {
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            }
        }

        sem_post(&sem_main);            // 唤醒主线程
        sem_wait(&sem_workerend[t_id]); // 阻塞，等待主线程唤醒进入下一轮
    }
    pthread_exit(NULL);
    return 0;
}
void mainfuncCOL(int N, float **m)
{
    sem_init(&sem_main, 0, 0);
    for (int id = 0; id < NUM_THREADS; id++)
        sem_init(&sem_workerstart[id], 0, 0);
    for (int id = 0; id < NUM_THREADS; id++)
        sem_init(&sem_workerend[id], 0, 0);
    pthread_t handles[NUM_THREADS];
    threadParam_t param[NUM_THREADS];

    for (int id = 0; id < NUM_THREADS; id++)
    {
        param[id].N = N;
        param[id].m = m;
        param[id].t_id = id;
        pthread_create(&handles[id], NULL, threadFuncCOL, (void *)&param[id]);
    }
    for (int k = 0; k < N; k++)
    {
        int j;
        for (j = k + 1; j < N; j++)
            m[k][j] /= m[k][k];
        m[k][k] = 1.0;
        for (int id = 0; id < NUM_THREADS; id++)
            sem_post(&sem_workerstart[id]);
        for (int id = 0; id < NUM_THREADS; id++)
            sem_wait(&sem_main);
        for (int id = 0; id < NUM_THREADS; id++)
            sem_post(&sem_workerend[id]);
        for (int i = k + 1; i < N; i++)
            m[i][k] = 0;
    }
    for (int id = 0; id < NUM_THREADS; id++)
        pthread_join(handles[id], NULL);
    sem_destroy(&sem_main);
    for (int id = 0; id < NUM_THREADS; id++)
        sem_destroy(&sem_workerstart[id]);
    for (int id = 0; id < NUM_THREADS; id++)
        sem_destroy(&sem_workerend[id]);
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
        cout << "N=" << N << " 静态+信号量同步（row）:" << timeuse1 << endl;

        double timeuse2 = 0;
        m_reset(N, m);
        gettimeofday(&start, NULL);
        mainfuncCOL(N, m);
        gettimeofday(&end, NULL);
        timeuse2 = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
        cout << "N=" << N << " 静态+信号量同步（col）:" << timeuse2 << endl;

        for (int i = 0; i < N; i++)
            delete[] m[i];
        delete[] m;
    }
}