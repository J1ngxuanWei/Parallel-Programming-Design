#include <iostream>
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

int main()
{
    struct timeval start, end;
    double timeuse = 0;
    for (int i = 100; i < 400; i += 100)
    {
        timeuse = 0;
        m_reset();
        gettimeofday(&start, NULL);
        serial();
        gettimeofday(&end, NULL);
        timeuse = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
        cout << "N=" << N << " serial:" << timeuse << "ms" << endl;
    }
    for (int i = 400; i < 1000; i += 200)
    {
        timeuse = 0;
        m_reset();
        gettimeofday(&start, NULL);
        serial();
        gettimeofday(&end, NULL);
        timeuse = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
        cout << "N=" << N << " serial:" << timeuse << "ms" << endl;
    }
    for (int i = 1000; i <= 3000; i += 500)
    {
        timeuse = 0;
        m_reset();
        gettimeofday(&start, NULL);
        serial();
        gettimeofday(&end, NULL);
        timeuse = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
        cout << "N=" << N << " serial:" << timeuse << "ms" << endl;
    }
    return 0;
}