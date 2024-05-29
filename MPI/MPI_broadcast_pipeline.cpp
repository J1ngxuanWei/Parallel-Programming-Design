#include <iostream>
#include <mpi.h>
#include <sys/time.h>
#include <cmath>
using namespace std;

// 数据规模
int N;
// 数据矩阵
float **m;

// 生成测试用例
void m_reset()
{
    srand(2112495);
    m = new float *[N];
    for (int i = 0; i < N; i++)
        m[i] = new float[N];

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

// MPI:广播方式
// 使用上一个问题中的循环划分策略的实现
void MPI_broadcast()
{
    double Tstart, Tend;

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Tstart = MPI_Wtime();
    int tasks; // 每个进程的任务量
    // 因为不一定正好成倍数，所以并不是完全平均
    if (rank < N % size)
        tasks = N / size + 1;
    else
        tasks = N / size;
    // 0号进程负责任务的初始分发工作
    // buff用来暂时存放要分发给某个进程的任务
    float *buff = new float[tasks * N];
    if (rank == 0)
    {
        for (int p = 1; p < size; p++)
        {
            for (int i = p; i < N; i += size)
                for (int j = 0; j < N; j++)
                    buff[i / size * N + j] = m[i][j];
            int ptasks = p < N % size ? N / size + 1 : N / size; // 待接收进程的任务行数
            MPI_Send(buff, ptasks * N, MPI_FLOAT, ptasks, 0, MPI_COMM_WORLD);
        }
    }
    // 非0号进程负责任务的接收工作
    else
    {
        MPI_Recv(&m[rank][0], tasks * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // 将接收到的数据调整到对应的位置上
        for (int i = 0; i < tasks; i++)
            for (int j = 0; j < N; j++)
                m[rank + i * size][j] = m[rank + i][j];
    }
    // 做消元运算
    for (int k = 0; k < N; k++)
    {
        // 如果除法操作是本进程负责的任务，并将除法结果广播
        if (k % size == rank)
        {
            for (int j = k + 1; j < N; j++)
                m[k][j] /= m[k][k];
            m[k][k] = 1;
            for (int i = 0; i < size; i++)
                if (i != rank)
                    MPI_Send(&m[k][0], N, MPI_FLOAT, i, 1, MPI_COMM_WORLD);
        }
        // 其余进程接收除法行的结果
        else
            MPI_Recv(&m[k][0], N, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // 减法消元
        int begin = N / size * size + rank < N ? N / size * size + rank : N / size * size + rank - size;
        for (int i = begin; i > k; i -= size)
        {
            for (int j = k + 1; j < N; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            m[i][k] = 0;
        }
    }
    Tend = MPI_Wtime();
    if (rank == 0)
        cout << "N=" << N << " MPI_broadcast：" << (Tend - Tstart) * 1000 << "ms" << endl;
    return;
}

// MPI:流水线方式
void MPI_pipeline()
{
    double Tstart, Tend;

    int rank;
    int size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    Tstart = MPI_Wtime();

    int tasks; // 记录每个进程的任务量，在循环划分策略下最多相差一行
    if (rank < N % size)
        tasks = N / size + 1;
    else
        tasks = N / size;
    // 0号进程负责任务的初始分发工作
    float *buff = new float[tasks * N];
    if (rank == 0)
    {
        for (int p = 1; p < size; p++)
        {
            for (int i = p; i < N; i += size)
                for (int j = 0; j < N; j++)
                    buff[i / size * N + j] = m[i][j];
            int ptasks = p < N % size ? N / size + 1 : N / size;
            MPI_Send(buff, ptasks * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    }
    // 非0号进程负责任务的接收工作
    else
    {
        MPI_Recv(&m[rank][0], tasks * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // 将接收到的数据调整到对应的位置上
        for (int i = 0; i < tasks; i++)
            for (int j = 0; j < N; j++)
                m[rank + i * size][j] = m[rank + i][j];
    }
    // 做消元运算
    int pre = (rank + (size - 1)) % size;
    int next = (rank + 1) % size;
    for (int k = 0; k < N; k++)
    {
        // 如果除法操作是本进程负责的任务，并将除法结果广播
        if (k % size == rank)
        {
            for (int j = k + 1; j < N; j++)
                m[k][j] /= m[k][k];
            m[k][k] = 1;
            MPI_Send(&m[k][0], N, MPI_FLOAT, next, 1, MPI_COMM_WORLD);
        }
        // 其余进程接收除法行的结果
        else
        {
            // 从上一个进程接收数据集
            MPI_Recv(&m[k][0], N, MPI_FLOAT, pre, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            // 如果下一个进程是最先发出（做除法）的那个线程，不用继续发，否则就发消息）
            if (next != k % size)
                MPI_Send(&m[k][0], N, MPI_FLOAT, next, 1, MPI_COMM_WORLD);
        }
        // 进行消元操作
        int begin = N / size * size + rank < N ? N / size * size + rank : N / size * size + rank - size;
        for (int i = begin; i > k; i -= size)
        {
            for (int j = k + 1; j < N; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            m[i][k] = 0;
        }
    }
    Tend = MPI_Wtime();
    if (rank == 0)
        cout << "N=" << N << " MPI_pineline：" << (Tend - Tstart) * 1000 << "ms" << endl;
    return;
}

int main()
{
    MPI_Init(nullptr, nullptr);
    // 1.MPI_broadcast测试
    for (int i = 100; i < 400; i += 100)
    {
        m_reset();
        MPI_broadcast();
    }
    for (int i = 400; i < 1000; i += 200)
    {
        m_reset();
        MPI_broadcast();
    }
    for (int i = 1000; i <= 3000; i += 500)
    {
        m_reset();
        MPI_broadcast();
    }
    // 2.MPI_pineline测试
    for (int i = 100; i < 400; i += 100)
    {
        m_reset();
        MPI_pipeline();
    }
    for (int i = 400; i < 1000; i += 200)
    {
        m_reset();
        MPI_pipeline();
    }
    for (int i = 1000; i <= 3000; i += 500)
    {
        m_reset();
        MPI_pipeline();
    }
    return 0;
}
