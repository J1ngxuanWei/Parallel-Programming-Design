// g++ -march=native special_x86.cpp -o pp -lpthread -fopenmp
#include <iostream>
#include <fstream>
#include <sstream>
#include <bitset>
#include <cmath>
#include <sys/time.h>
#include <semaphore.h>
#include <omp.h>
#include <cstring>
using namespace std;

const int NUM_THREADS = 2;

const int test = 0; // 测试的序号

string test_num[2][2] = {
    {"/home/u220187/data/254_1.txt", "/home/u220187/data/254_2.txt"},
    {"/home/u220187/data/2362_1.txt", "/home/u220187/data/2362_2.txt"},
};

int test_nub[2] = {254, 2362};

int colNum = test_nub[test]; // 矩阵的列数
const int R_lineNum = colNum;
const int actual_colNum = ceil(colNum * 1.0 / 32); // 数组的列数

pthread_barrier_t xor_barrier;
pthread_barrier_t setr_barrier;
int pthreadcol = 0;
int openMPcol = 0;
typedef struct
{
    int eNum;
    int *First;
    unsigned int **R;
    unsigned int **E;
    int t_id;
} threadParam_t;

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
            int offset = StringToNum % 32; //+1的原因是txt中给的列数是从0开始计数而不是1
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

// 重置E矩阵的首项
void Reset_Efirst(unsigned int **E, int index, int *First)
{
    int i = 0;
    while (E[index][i] == 0 && i < actual_colNum)
        i++;
    if (i == actual_colNum)
    {
        First[index] = -1;
        pthreadcol--;
        openMPcol--;
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

// 异或操作
void ExorR(unsigned int **R, unsigned int **E, int Eindex, int Rindex)
{
    for (int i = 0; i < actual_colNum; i++)
        E[Eindex][i] = E[Eindex][i] ^ R[Rindex][i];
}

// 串行算法
void serial(int eNum, int *First, unsigned int **R, unsigned int **E)
{
    for (int i = 0; i < eNum; i++)
    {
        while (First[i] != -1) // 当前被消元行是否已完成消元
        {
            if (Is_R_Null(R, First[i])) // 判断是否存在首项对应的消元行
            {
                Set_EtoR(R, E, i, First[i]); // 当前行消元结束,并设置为消元子
                First[i] = -1;               // 修改首项为-1表示消元结束
                break;
            }
            else // 如果E当前行首项对应的R的消元子不为空
            {
                ExorR(R, E, i, First[i]);  // 消元:被消元行与对应的消元行异或
                Reset_Efirst(E, i, First); // 某行被消元后首项要进行调整
            }
        }
    }
}

// Pthread算法
void *threadFunc_Pthread(void *param)
{
    threadParam_t *p = (threadParam_t *)param;
    int t_id = p->t_id;
    int eNum = p->eNum;
    int *First = p->First;
    unsigned int **R = p->R;
    unsigned int **E = p->E;

    while (1)
    {
        for (int i = t_id; i < eNum; i += NUM_THREADS)
            while (First[i] != -1)
                if (!Is_R_Null(R, First[i]))
                {                              // 对应的R的行非空
                    ExorR(R, E, i, First[i]);  // 消去
                    Reset_Efirst(E, i, First); // 重置消去后的首项
                }
                else
                    break;

        pthread_barrier_wait(&xor_barrier); // 第一个同步点

        if (t_id == 0) // 只0号线程执行
            for (int i = 0; i < eNum; i++)
                if (First[i] != -1) // 找到第一个对应消元子行为空的被消元子行
                    if (Is_R_Null(R, First[i]))
                    {
                        Set_EtoR(R, E, i, First[i]); // 将找到的被消元子设置为消元子
                        First[i] = -1;               // 标记当前行消元结束
                        pthreadcol--;
                        break;
                    }

        pthread_barrier_wait(&setr_barrier); // 第二个同步点
        if (pthreadcol == 0)                 // 判断整个消元是否结束
            break;
    }
    pthread_exit(NULL);
}

void mainfunc_Pthread(int eNum, int *First, unsigned int **R, unsigned int **E)
{
    // 初始化barrier
    pthread_barrier_init(&xor_barrier, NULL, NUM_THREADS);
    pthread_barrier_init(&setr_barrier, NULL, NUM_THREADS);
    // 创建线程
    pthread_t handles[NUM_THREADS];   // 创建对应的handle
    threadParam_t param[NUM_THREADS]; // 创建对应的线程数据结构

    for (int t_id = 0; t_id < NUM_THREADS; ++t_id)
    {
        param[t_id].t_id = t_id;
        param[t_id].eNum = eNum;
        param[t_id].First = First;
        param[t_id].R = R;
        param[t_id].E = E;
        pthread_create(&handles[t_id], NULL, threadFunc_Pthread, (void *)&param[t_id]);
    }

    for (int t_id = 0; t_id < NUM_THREADS; ++t_id)
        pthread_join(handles[t_id], NULL);

    pthread_barrier_destroy(&xor_barrier);
    pthread_barrier_destroy(&setr_barrier);
}

// openMP算法
void mainfunc_openMP(int eNum, int *First, unsigned int **R, unsigned int **E)
{
    int i;
#pragma omp parallel num_threads(NUM_THREADS), private(i), shared(eNum)
    while (1)
    {
#pragma omp for private(i)
        for (i = 0; i < eNum; i++)
        {
            while (First[i] != -1)
                if (!Is_R_Null(R, First[i]))
                {                              // 对应的R的行非空
                    ExorR(R, E, i, First[i]);  // 消去
                    Reset_Efirst(E, i, First); // 重置消去后的首项
                }
                else
                    break;
        }
#pragma omp single
        {
            for (i = 0; i < eNum; i++)
            {
                if (First[i] != -1)
                { // 找到第一个对应消元子行为空的被消元子行
                    if (Is_R_Null(R, First[i]))
                    {
                        Set_EtoR(R, E, i, First[i]); // 将找到的被消元子设置为消元子
                        First[i] = -1;               // 标记当前行消元结束
                        openMPcol--;
                        break;
                    }
                }
            }
        }
        if (openMPcol == 0)
            break;
    }
}

int main()
{
    /*eNum中存放被消元行的行数*/
    ifstream infile(test_num[test][1]);

    char fin[20000] = {0};
    int eNum = 0;
    while (infile.getline(fin, sizeof(fin)))
        eNum++;
    infile.close();

    pthreadcol = eNum;
    openMPcol = eNum;

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

    Init_Zero(E, eNum, actual_colNum);
    Init_E(E, First);
    Init_Zero(R, R_lineNum, actual_colNum);
    Init_R(R);
    gettimeofday(&start, NULL);
    serial(eNum, First, R, E);
    gettimeofday(&end, NULL);
    timeuse = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    cout << "Threads:" << NUM_THREADS << "Col:" << colNum << " serial:" << timeuse << " ";

    Init_Zero(E, eNum, actual_colNum);
    Init_E(E, First);
    Init_Zero(R, R_lineNum, actual_colNum);
    Init_R(R);
    gettimeofday(&start, NULL);
    mainfunc_Pthread(eNum, First, R, E);
    gettimeofday(&end, NULL);
    timeuse = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    cout << "Threads:" << NUM_THREADS << "Col:" << colNum << " pthread:" << timeuse << " ";

    Init_Zero(E, eNum, actual_colNum);
    Init_E(E, First);
    Init_Zero(R, R_lineNum, actual_colNum);
    Init_R(R);
    gettimeofday(&start, NULL);
    mainfunc_openMP(eNum, First, R, E);
    gettimeofday(&end, NULL);
    timeuse = ((end.tv_sec - start.tv_sec) + (double)(end.tv_usec - start.tv_usec) / 1000000.0) * 1000;
    cout << "Threads:" << NUM_THREADS << "Col:" << colNum << " openMP:" << timeuse << " ";

    return 0;
}
