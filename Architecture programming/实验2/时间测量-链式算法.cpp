#include <iostream>
#include <vector>
#include <windows.h>
using namespace std;

int n=8;
int sum=0;
vector<int> a(n); // n����

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
    //��ʼ��
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
    }
    // �㷨����
    // ��ʽ��������Ԫ�������ۼӵ������������
    long long head,tail,freq;
    int counter=0;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq );

    //��ʼ
    QueryPerformanceCounter((LARGE_INTEGER *)&head);

    while(counter<1000000)
    {
        ini();
        for (int i = 0; i < n; i++)
        sum += a[i];
        counter++;
    }

    QueryPerformanceCounter((LARGE_INTEGER *)&tail );
    //����

    // ���
    cout << "Time: " << ( tail - head) * 1000.0 / freq << "ms" << endl ;
    cout<<sum;


    return 0;
}