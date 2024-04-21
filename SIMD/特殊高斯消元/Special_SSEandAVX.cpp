// g++ Special_SSEandAVX.cpp -march=corei7 -march=corei7-avx -march=native
#include <iostream>
#include <fstream>
#include <sstream>
#include <bitset>
#include <cmath>
#include <stdlib.h>
#include <sys/time.h>
#include <nmmintrin.h>
#include <immintrin.h>
using namespace std;

const int test = 0; // 测试的序号

string test_num[6][2] = {
    {"/home/u220187/data/130_1.txt", "/home/u220187/data/130_2.txt"},
    {"/home/u220187/data/254_1.txt", "/home/u220187/data/254_2.txt"},
    {"/home/u220187/data/562_1.txt", "/home/u220187/data/562_2.txt"},
    {"/home/u220187/data/1011_1.txt", "/home/u220187/data/1011_2.txt"},
    {"/home/u220187/data/2362_1.txt", "/home/u220187/data/2362_2.txt"},
    {"/home/u220187/data/3799_1.txt", "/home/u220187/data/3799_2.txt"},
};

int test_nub[6] = {130, 254, 562, 1011, 2362, 3799};

int colNum = test_nub[test]; // 矩阵的列数
const int R_lineNum = colNum;
const int actual_colNum = ceil(colNum * 1.0 / 32);

void Init_Zero(unsigned int **m, int line, int col)
{
    for (int i = 0; i < line; i++)
        for (int j = 0; j < col; j++)
            m[i][j] = 0;
}

// 初始化被消元行矩阵
void Init_E(unsigned int **E, int *First)
{
    unsigned int StringToNum;
    ifstream infile(test_num[test][1]);

    char fin[20000] = {0};
    int index = 0;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int IsFirst = 0; // 用于判断是不是首项
        while (line >> StringToNum)
        { // 是首项则存入First数组
            if (IsFirst == 0)
            {
                First[index] = StringToNum;
                IsFirst = 1;
            }
            int offset = StringToNum % 32;
            int post = StringToNum / 32;
            int temp = 1 << offset;
            E[index][actual_colNum - 1 - post] += temp;
        }
        index++;
    }
    infile.close();
}

// 初始化消元子矩阵
void Init_R(unsigned int **R)
{
    unsigned int StringToNum;
    ifstream infile(test_num[test][0]);

    char fin[20000] = {0};
    int index = 0;
    while (infile.getline(fin, sizeof(fin)))
    {
        std::stringstream line(fin);
        int IsFirst = 0; // 用于判断是不是首项
        while (line >> StringToNum)
        {
            if (IsFirst == 0)
            {
                index = StringToNum; // 对于消元子来说首项是谁就放到哪一行里面，有很多空行
                IsFirst = 1;
            }
            int offset = StringToNum % 32;
            int post = StringToNum / 32;
            int temp = 1 << offset;
            R[index][actual_colNum - 1 - post] += temp;
        }
    }
    infile.close();
}

// 判断消元子矩阵中某一行是否为空
bool Is_R_Null(unsigned int **R, int index)
{
    for (int i = 0; i < actual_colNum; i++)
        if (R[index][i] != 0)
            return false;
    return true;
}

// 将E矩阵的某一行赋值给R矩阵的某一行
void Set_EtoR(unsigned int **R, unsigned int **E, int Eindex, int Rindex)
{
    for (int i = 0; i < actual_colNum; i++)
        R[Rindex][i] = E[Eindex][i];
}

// 重置E矩阵的某一行的首项
void Reset_Efirst(unsigned int **E, int index, int *First)
{
    int i = 0;
    while (E[index][i] == 0 && i < actual_colNum)
        i++;
    if (i == actual_colNum)
    {
        First[index] = -1;
        return;
    }
    unsigned int temp = E[index][i];
    int j = 0;
    while (temp != 0)
    {
        temp = temp >> 1;
        j++;
    }
    First[index] = actual_colNum * 32 - (i + 1) * 32 + j - 1;
}

// 串行异或操作
void ExorR(unsigned int **R, unsigned int **E, int Eindex, int Rindex)
{
    for (int i = 0; i < actual_colNum; i++)
        E[Eindex][i] = E[Eindex][i] ^ R[Rindex][i];
}

// 串行算法
void serial(int eNum, int *First, unsigned int **R, unsigned int **E)
{
    for (int i = 0; i < eNum; i++)
        while (First[i] != -1)          // 当前E被消元行还未完成消元
            if (Is_R_Null(R, First[i])) // 如果E当前行首项对应的R的消元子为空
            {
                Set_EtoR(R, E, i, First[i]);
                break;
            }
            else // 如果E当前行首项对应的R的消元子为空
            {
                ExorR(R, E, i, First[i]);
                Reset_Efirst(E, i, First);
            }
}

// SSE异或算法
void SSE_ExorR(unsigned int **R, unsigned int **E, int Eindex, int Rindex)
{
    int i = 0;
    for (; i + 4 <= actual_colNum; i += 4)
    {
        unsigned int *tmp1 = &(E[Eindex][i]);
        unsigned int *tmp2 = &(R[Rindex][i]);
        __m128i vE = _mm_loadu_si128((__m128i *)tmp1);
        __m128i vR = _mm_loadu_si128((__m128i *)tmp2);
        vE = _mm_xor_si128(vE, vR);
        _mm_storeu_si128((__m128i *)tmp1, vE);
    }
    for (; i < actual_colNum; i++)
        E[Eindex][i] = E[Eindex][i] ^ R[Rindex][i];
}

// SSE算法
void SSE(int eNum, int *First, unsigned int **R, unsigned int **E)
{
    for (int i = 0; i < eNum; i++)
        while (First[i] != -1)          // 当前E被消元行还未完成消元
            if (Is_R_Null(R, First[i])) // 如果E当前行首项对应的R的消元子为空
            {
                Set_EtoR(R, E, i, First[i]);
                break;
            }
            else // 如果E当前行首项对应的R的消元子为空
            {
                SSE_ExorR(R, E, i, First[i]);
                Reset_Efirst(E, i, First);
            }
}

// AVX异或算法
void AVX_ExorR(unsigned int **R, unsigned int **E, int Eindex, int Rindex)
{
    int i = 0;
    for (; i + 8 <= actual_colNum; i += 8)
    {
        unsigned int *tmp1 = &(E[Eindex][i]);
        unsigned int *tmp2 = &(R[Rindex][i]);
        __m256i vE = _mm256_loadu_si256((__m256i *)tmp1);
        __m256i vR = _mm256_loadu_si256((__m256i *)tmp2);
        vE = _mm256_xor_si256(vE, vR);
        _mm256_storeu_si256((__m256i *)tmp1, vE);
    }
    for (; i < actual_colNum; i++)
        E[Eindex][i] = E[Eindex][i] ^ R[Rindex][i];
}

// AVX算法
void AVX(int eNum, int *First, unsigned int **R, unsigned int **E)
{
    for (int i = 0; i < eNum; i++)
        while (First[i] != -1)          // 当前E被消元行还未完成消元
            if (Is_R_Null(R, First[i])) // 如果E当前行首项对应的R的消元子为空
            {
                Set_EtoR(R, E, i, First[i]);
                break;
            }
            else // 如果E当前行首项对应的R的消元子为空
            {
                AVX_ExorR(R, E, i, First[i]);
                Reset_Efirst(E, i, First);
            }
}

// AVX512异或算法
void AVX512_ExorR(unsigned int **R, unsigned int **E, int Eindex, int Rindex)
{
    int i = 0;
    for (; i + 16 <= actual_colNum; i += 16)
    {
        unsigned int *tmp1 = &(E[Eindex][i]);
        unsigned int *tmp2 = &(R[Rindex][i]);
        __m512i vE = _mm512_loadu_si512((__m512i *)tmp1);
        __m512i vR = _mm512_loadu_si512((__m512i *)tmp2);
        vE = _mm512_xor_si512(vE, vR);
        _mm512_storeu_si512((__m512i *)tmp1, vE);
    }
    for (; i < actual_colNum; i++)
        E[Eindex][i] = E[Eindex][i] ^ R[Rindex][i];
}

// AVX512算法
void AVX512(int eNum, int *First, unsigned int **R, unsigned int **E)
{
    for (int i = 0; i < eNum; i++)
        while (First[i] != -1)          // 当前E被消元行还未完成消元
            if (Is_R_Null(R, First[i])) // 如果E当前行首项对应的R的消元子为空
            {
                Set_EtoR(R, E, i, First[i]);
                break;
            }
            else // 如果E当前行首项对应的R的消元子为空
            {
                AVX512_ExorR(R, E, i, First[i]);
                Reset_Efirst(E, i, First);
            }
}

int main()
{
    ifstream infile(test_num[test][1]);

    char fin[20000] = {0};
    int eNum = 0;
    while (infile.getline(fin, sizeof(fin)))
        eNum++;
    infile.close();

    // 存E矩阵的首项
    int *First = new int[eNum];

    // 创建被消元行矩阵并初始化
    unsigned int **E = new unsigned int *[eNum];
    for (int i = 0; i < eNum; ++i)
        E[i] = new unsigned int[actual_colNum];
    Init_Zero(E, eNum, actual_colNum);
    Init_E(E, First);

    // 创建消元子矩阵并初始化
    unsigned int **R = new unsigned int *[R_lineNum];
    for (int i = 0; i < R_lineNum; ++i)
        R[i] = new unsigned int[actual_colNum];
    Init_Zero(R, R_lineNum, actual_colNum);
    Init_R(R);

    struct timeval start, end;
    double timeuse;

    gettimeofday(&start, NULL);
    serial(eNum, First, R, E);
    gettimeofday(&end, NULL);
    timeuse = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    cout << "Col=" << colNum << " serial:" << timeuse << endl;

    Init_Zero(E, eNum, actual_colNum);
    Init_E(E, First);
    Init_Zero(R, R_lineNum, actual_colNum);
    Init_R(R);
    gettimeofday(&start, NULL);
    SSE(eNum, First, R, E);
    gettimeofday(&end, NULL);
    timeuse = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    cout << "Col=" << colNum << " SSE:" << timeuse << endl;

    Init_Zero(E, eNum, actual_colNum);
    Init_E(E, First);
    Init_Zero(R, R_lineNum, actual_colNum);
    Init_R(R);
    gettimeofday(&start, NULL);
    AVX(eNum, First, R, E);
    gettimeofday(&end, NULL);
    timeuse = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    cout << "Col=" << colNum << " AVX:" << timeuse << endl;

    Init_Zero(E, eNum, actual_colNum);
    Init_E(E, First);
    Init_Zero(R, R_lineNum, actual_colNum);
    Init_R(R);
    gettimeofday(&start, NULL);
    AVX512(eNum, First, R, E);
    gettimeofday(&end, NULL);
    timeuse = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    cout << "Col=" << colNum << " AVX-512:" << timeuse << endl;

    return 0;
}