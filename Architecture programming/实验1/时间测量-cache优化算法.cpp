#include <iostream>
#include <vector>
#include<windows.h>
using namespace std;



int n = 10;
vector<float> sum(n);                         // ������
vector<vector<float>> b(n, vector<float>(n)); // 5��10�еľ���
vector<float> a(n);                           // 5�е�������



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
    // �㷨����
    // ��Ϊ���з��ʾ���Ԫ�أ�һ�����ѭ�����㲻���κ�һ���ڻ���ֻ����ÿ���ڻ��ۼ�һ���˷����



    long long head,tail,freq;
    int counter=0;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq );

    //��ʼ
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
    //����

    // ���
    cout << "Time: " << ( tail - head) * 1000.0 / freq << "ms" << endl ;


    return 0;
}


