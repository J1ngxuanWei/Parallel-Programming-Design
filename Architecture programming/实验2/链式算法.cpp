#include <iostream>
#include <vector>
#include <windows.h>
using namespace std;
int main()
{
    int n = 8;
    int sum = 0;
    vector<int> a(n); // n����

    // ��ʼ��
    for (int i = 0; i < n; i++)
    {
        a[i] = i;
    }

    // �㷨����
    // ��ʽ��������Ԫ�������ۼӵ������������
    for (int i = 0; i < n; i++)
        sum += a[i];

    cout << sum << endl;

    return 0;
}