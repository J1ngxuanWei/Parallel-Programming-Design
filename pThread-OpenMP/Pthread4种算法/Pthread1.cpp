// g++ −g −march=native Pthread1.cpp -o pp -lpthread
#include <iostream>
#include <sys/time.h>
#include <pthread.h>
using namespace std;

int worker_count = 2;

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
    int k;     // 消去轮次
    int t_id;  // 线程id
} threadParam_t;

void *threadFuncROW(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int N = p->N;
    float **m = p->m;
    int k = p->k;
    int t_id = p->t_id;
    // 获取自己的计算任务
    for (int i = k + t_id + 1; i < N; i += worker_count)
    {
        for (int j = k + 1; j < N; j++)
            m[i][j] = m[i][j] - m[k][j] * m[i][k];
        m[i][k] = 0;
    }
    pthread_exit(NULL);
    return 0;
}
void mainfuncROW(int N, float **m)
{
    for (int k = 0; k < N; k++)
    {
        // 主线程做除法操作
        for (int j = k + 1; j < N; j++)
            m[k][j] = m[k][j] / m[k][k];
        m[k][k] = 1.0;

        // 创建工作线程，进行消去操作
        // 工作线程数量
        pthread_t *handles = (pthread_t *)malloc(worker_count * sizeof(pthread_t));           // 创建对应的 Handle
        threadParam_t *param = (threadParam_t *)malloc(worker_count * sizeof(threadParam_t)); // 创建对应的线程数据结构
        // 分配任务
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            param[t_id].N = N;
            param[t_id].m = m;
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        // 创建线程
        for (int t_id = 0; t_id < worker_count; t_id++)
            pthread_create(&handles[t_id], NULL, threadFuncROW, (void *)&param[t_id]);

        // 主线程挂起等待所有的工作线程完成此轮消去工作
        for (int t_id = 0; t_id < worker_count; t_id++)
            pthread_join(handles[t_id], NULL);
        free(handles);
        free(param);
    }
}
void *threadFuncCOL(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int N = p->N;
    float **m = p->m;
    int k = p->k;
    int t_id = p->t_id;
    // 获取自己的计算任务
    for (int j = k + 1 + t_id; j < N; j += worker_count)
        for (int i = k + 1; i < N; i++)
            m[i][j] = m[i][j] - m[k][j] * m[i][k];
    pthread_exit(NULL);
    return 0;
}
void mainfuncCOL(int N, float **m)
{
    for (int k = 0; k < N; k++)
    {
        // 主线程做除法操作
        for (int j = k + 1; j < N; j++)
            m[k][j] = m[k][j] / m[k][k];
        m[k][k] = 1.0;

        // 创建工作线程，进行消去操作
        // 工作线程数量
        pthread_t *handles = (pthread_t *)malloc(worker_count * sizeof(pthread_t));           // 创建对应的 Handle
        threadParam_t *param = (threadParam_t *)malloc(worker_count * sizeof(threadParam_t)); // 创建对应的线程数据结构
        // 分配任务
        for (int t_id = 0; t_id < worker_count; t_id++)
        {
            param[t_id].N = N;
            param[t_id].m = m;
            param[t_id].k = k;
            param[t_id].t_id = t_id;
        }
        // 创建线程
        for (int t_id = 0; t_id < worker_count; t_id++)
            pthread_create(&handles[t_id], NULL, threadFuncCOL, (void *)&param[t_id]);

        // 主线程挂起等待所有的工作线程完成此轮消去工作
        for (int t_id = 0; t_id < worker_count; t_id++)
            pthread_join(handles[t_id], NULL);

        for (int i = k + 1; i < N; i++)
            m[i][k] = 0;

        free(handles);
        free(param);
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
        mainfuncROW(N, m);
        gettimeofday(&end, NULL);
        timeuse1 = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
        cout << "N=" << N << " 动态（row）:" << timeuse1 << "   ";

        double timeuse2 = 0;
        m_reset(N, m);
        gettimeofday(&start, NULL);
        mainfuncCOL(N, m);
        gettimeofday(&end, NULL);
        timeuse2 = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
        cout << "N=" << N << " 动态（col）:" << timeuse2 << endl;

        for (int i = 0; i < N; i++)
            delete[] m[i];
        delete[] m;
    }
}
