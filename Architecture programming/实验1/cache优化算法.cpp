#include <iostream>
#include <vector>
using namespace std;
int main()
{
    int n = 5;
    vector<float> sum(n);                         // 保存结果
    vector<vector<float>> b(n, vector<float>(n)); // 5行10列的矩阵
    vector<float> a(n);                           // 5列的行向量

    // 初始化
    for (int i = 0; i < n; i++)
    {
        a[i] = (float)i / 4;
        for (int j = 0; j < n; j++)
            b[i][j] = ((i + 1) * (j + 1) / 4) * 0.1 + j;
    }

    // 算法主体
    // 改为逐行访问矩阵元素：一步外层循环计算不出任何一个内积，只是向每个内积累加一个乘法结果
    for (int i = 0; i < n; i++)
        sum[i] = 0.0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            sum[i] += b[j][i] * a[j];

    // 输出
    for (int i = 0; i < n; i++)
        cout << sum[i] << " ";

    return 0;
}