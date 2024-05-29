#include <iostream>
#include <mpi.h>
#include <cmath>
#include <sys/time.h>
#include <arm_neon.h>
#include <omp.h>
using namespace std;

// 数据规模
int N;
// 数据矩阵
float **m;
// 线程数
int NUM_THREADS = 8;

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

// MPI:未与SIMD、OpenMP结合
// 使用前面问题中实现的循环数据划分
void MPI_cycle()
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
        cout << "N=" << N << " MPI_cycle：" << (Tend - Tstart) * 1000 << "ms" << endl;
    return;
}

// MPI+NEON
void MPI_NEON()
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
            int ptasks = p < N % size ? N / size + 1 : N / size;
            MPI_Send(buff, ptasks * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    }
    // 非0号进程负责任务的接收工作
    else
    {
        MPI_Recv(&m[rank][0], tasks * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
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
            for (int j = a; j < N; j++) // 剩余不足4个数部分处理
                m[k][j] = m[k][j] / m[k][k];
            m[k][k] = 1.0;
            for (int p = 0; p < size; p++)
                if (p != rank)
                    MPI_Send(&m[k][0], N, MPI_FLOAT, p, 1, MPI_COMM_WORLD);
        }
        // 其余进程接收除法行的结果
        else
            MPI_Recv(&m[k][0], N, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        // 进行消元操作
        int begin = k + 1;
        while (begin % size != rank) // 找到进行减法的开始任务行
            begin++;
        for (int i = begin; i < N; i += size)
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
            for (int j = a; j < N; j++) // 串行处理不足4个数
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            m[i][k] = 0;
        }
    }
    Tend = MPI_Wtime();
    if (rank == 0)
        cout << "N=" << N << " MPI_SIMD_NEON：" << (Tend - Tstart) * 1000 << "ms" << endl;
    return;
}

// MPI+OMP
void MPI_OMP()
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
            int ptasks = p < N % size ? N / size + 1 : N / size;
            MPI_Send(buff, ptasks * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    }
    // 非0号进程负责任务的接收工作
    else
    {
        MPI_Recv(&m[rank][0], tasks * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < tasks; i++)
            for (int j = 0; j < N; j++)
                m[rank + i * size][j] = m[rank + i][j];
    }
    // 做消元运算
    int i, j, k;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k) shared(m, N, size, rank)
    for (k = 0; k < N; k++)
    {
        // 如果除法操作是本进程负责的任务，并将除法结果广播
#pragma omp single
        {
            if (k % size == rank)
            {
                for (j = k + 1; j < N; j++)
                    m[k][j] /= m[k][k];
                m[k][k] = 1;
                for (int p = 0; p < size; p++)
                    if (p != rank)
                        MPI_Send(&m[k][0], N, MPI_FLOAT, p, 1, MPI_COMM_WORLD);
            }
            // 其余进程接收除法行的结果
            else
                MPI_Recv(&m[k][0], N, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // 进行消元操作
        int begin = k + 1;
        while (begin % size != rank) // 找到进行减法的开始任务行
            begin++;
#pragma omp for schedule(simd : guided)
        for (int i = begin; i < N; i += size)
        {
            for (j = k + 1; j < N; j++)
                m[i][j] = m[i][j] - m[i][k] * m[k][j];
            m[i][k] = 0;
        }
    }
    Tend = MPI_Wtime();
    if (rank == 0)
        cout << "N=" << N << " MPI_OMP " << (Tend - Tstart) * 1000 << "ms" << endl;
}

// MPI+NEON+OMP
void MPI_NEON_OMP()
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
            int ptasks = p < N % size ? N / size + 1 : N / size;
            MPI_Send(buff, ptasks * N, MPI_FLOAT, p, 0, MPI_COMM_WORLD);
        }
    }
    // 非0号进程负责任务的接收工作
    else
    {
        MPI_Recv(&m[rank][0], tasks * N, MPI_FLOAT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        for (int i = 0; i < tasks; i++)
            for (int j = 0; j < N; j++)
                m[rank + i * size][j] = m[rank + i][j];
    }
    // 做消元运算
    int i, j, k;
#pragma omp parallel num_threads(NUM_THREADS) default(none) private(i, j, k) shared(m, N, size, rank)
    for (k = 0; k < N; k++)
    {
        // 如果除法操作是本进程负责的任务，并将除法结果广播
#pragma omp single
        {
            if (k % size == rank)
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
                for (int j = a; j < N; j++) // 剩余不足4个数部分处理
                    m[k][j] = m[k][j] / m[k][k];
                m[k][k] = 1.0;
                for (int p = 0; p < size; p++)
                    if (p != rank)
                        MPI_Send(&m[k][0], N, MPI_FLOAT, p, 1, MPI_COMM_WORLD);
            }
            // 其余进程接收除法行的结果
            else
                MPI_Recv(&m[k][0], N, MPI_FLOAT, k % size, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
        // 进行消元操作
        int begin = k + 1;
        while (begin % size != rank) // 找到进行减法的开始任务行
            begin++;
#pragma omp for schedule(simd : guided)
        for (int i = begin; i < N; i += size)
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
            for (int j = a; j < N; j++) // 串行处理不足4个数
                m[i][j] = m[i][j] - m[k][j] * m[i][k];
            m[i][k] = 0;
        }
    }
    Tend = MPI_Wtime();
    if (rank == 0)
        cout << "N=" << N << " MPI_NEON_OMP" << (Tend - Tstart) * 1000 << "ms" << endl;
}

int main()
{
    MPI_Init(nullptr, nullptr);
    // 1.MPI测试
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
    // 2.MPI+NEON测试
    for (int i = 100; i < 400; i += 100)
    {
        m_reset();
        MPI_NEON();
    }
    for (int i = 400; i < 1000; i += 200)
    {
        m_reset();
        MPI_NEON();
    }
    for (int i = 1000; i <= 3000; i += 500)
    {
        m_reset();
        MPI_NEON();
    }
    // 3.MPI+OMP测试
    for (int i = 100; i < 400; i += 100)
    {
        m_reset();
        MPI_OMP();
    }
    for (int i = 400; i < 1000; i += 200)
    {
        m_reset();
        MPI_OMP();
    }
    for (int i = 1000; i <= 3000; i += 500)
    {
        m_reset();
        MPI_OMP();
    }
    // 4.MPI+NEON+OMP测试
    for (int i = 100; i < 400; i += 100)
    {
        m_reset();
        MPI_NEON_OMP();
    }
    for (int i = 400; i < 1000; i += 200)
    {
        m_reset();
        MPI_NEON_OMP();
    }
    for (int i = 1000; i <= 3000; i += 500)
    {
        m_reset();
        MPI_NEON_OMP();
    }
    return 0;
}
