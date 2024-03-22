#include <iostream>
#include <vector>
using namespace std;
int main()
{
    int n = 5;
    vector<float> sum(n);                         // ������
    vector<vector<float>> b(n, vector<float>(n)); // 5��10�еľ���
    vector<float> a(n);                           // 5�е�������

    // ��ʼ��
    for (int i = 0; i < n; i++)
    {
        a[i] = (float)i / 4;
        for (int j = 0; j < n; j++)
            b[i][j] = ((i + 1) * (j + 1) / 4) * 0.1 + j;
    }

    // �㷨����
    // ��Ϊ���з��ʾ���Ԫ�أ�һ�����ѭ�����㲻���κ�һ���ڻ���ֻ����ÿ���ڻ��ۼ�һ���˷����
    for (int i = 0; i < n; i++)
        sum[i] = 0.0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++)
            sum[i] += b[j][i] * a[j];

    // ���
    for (int i = 0; i < n; i++)
        cout << sum[i] << " ";

    return 0;
}