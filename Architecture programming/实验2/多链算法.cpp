#include <iostream>
#include <vector>
#include <windows.h>
using namespace std;
int main()
{
    int n = 8;
    int sum = 0;
    int sum1 = 0;
    int sum2 = 0;
    vector<int> a(n); // n����

    // ��ʼ��
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
    }

    // �㷨����
    // ����·ʽ
    for (int i = 0; i < n; i += 2)
    {
        sum1 += a[i];
        sum2 += a[i + 1];
    }
    sum = sum1 + sum2;

    cout << sum;

    return 0;
}