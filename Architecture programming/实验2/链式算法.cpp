#include <iostream>
#include <vector>
#include <windows.h>
using namespace std;
int main()
{
    int n = 8;
    int sum = 0;
    vector<int> a(n); // n个数

    // 初始化
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
    }

    // 算法主体
    // 链式：将给定元素依次累加到结果变量即可
    for (int i = 0; i < n; i++)
        sum += a[i];

    cout << sum << endl;

    return 0;
}