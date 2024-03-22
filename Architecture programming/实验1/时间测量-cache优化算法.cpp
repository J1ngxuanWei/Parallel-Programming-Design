#include <iostream>
#include <vector>
#include<windows.h>
using namespace std;



int n = 10;
vector<float> sum(n);                         // 保存结果
vector<vector<float>> b(n, vector<float>(n)); // 5行10列的矩阵
vector<float> a(n);                           // 5列的行向量



void ini()
{
        for (int i = 0; i < n; i++)
    {
        a[i] = (float)i / 4;
        for (int j = 0; j < n; j++)
            b[i][j] = ((i + 1) * (j + 1) / 4) * 0.1 + j;
    }

}

int main()
{
    // 算法主体
    // 改为逐行访问矩阵元素：一步外层循环计算不出任何一个内积，只是向每个内积累加一个乘法结果



    long long head,tail,freq;
    int counter=0;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq );

    //开始
    QueryPerformanceCounter((LARGE_INTEGER *)&head);

    while(counter<1000000)
    {
       ini();
       for (int i = 0; i < n; i++)
        sum[i] = 0.0;
        for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            sum[i] += b[j][i] * a[j];

        counter++;
    }

    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    //结束

    // 输出
    cout << "Time: " << ( tail - head) * 1000.0 / freq << "ms" << endl ;


    return 0;
}


