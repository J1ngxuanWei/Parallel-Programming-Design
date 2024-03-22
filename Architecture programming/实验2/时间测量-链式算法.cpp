#include <iostream>
#include <vector>
#include <windows.h>
using namespace std;

int n=8;
int sum=0;
vector<int> a(n); // n个数

void ini()
{
    sum=0;
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
    }
}

int main()
{
    //初始化
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
    }
    // 算法主体
    // 链式：将给定元素依次累加到结果变量即可
    long long head,tail,freq;
    int counter=0;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq );

    //开始
    QueryPerformanceCounter((LARGE_INTEGER *)&head);

    while(counter<1000000)
    {
        ini();
        for (int i = 0; i < n; i++)
        sum += a[i];
        counter++;
    }

    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    //结束

    // 输出
    cout << "Time: " << ( tail - head) * 1000.0 / freq << "ms" << endl ;
    cout<<sum;


    return 0;
}