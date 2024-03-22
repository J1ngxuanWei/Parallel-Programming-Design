#include <iostream>
#include <vector>
#include <windows.h>
using namespace std;

int n = 1000;
int sum = 0;
int sum1 = 0;
int sum2 = 0;
vector<int> a(n); // n个数

void ini()
{
    sum=0;
    sum1=0;
    sum2=0;
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
    }
}

int main()
{
    // 算法主体
    // 多链路式
    long long head,tail,freq;
    int counter=0;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq );

    //开始
    QueryPerformanceCounter((LARGE_INTEGER *)&head);

    while(counter<1000000)
    {
       ini();
       for (int i = 0; i < n; i += 2)
    {
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    sum = sum1 + sum2;

        counter++;
    }

    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    //结束

    // 输出
    cout << "Time: " << ( tail - head) * 1000.0 / freq << "ms" << endl ;
    cout << sum;

    return 0;
}
