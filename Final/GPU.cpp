#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <algorithm>
#include <CL/sycl.hpp>

#define MAXSIZE 2000

using namespace std;
using namespace sycl;

class index
{
public:
    int len = 0;
    vector<unsigned int> order;
};

bool operator<(const index &s1, const index &s2)
{
    return s1.len < s2.len;
}

class BitMap
{
public:
    BitMap(int range)
    {
        this->m_bits.resize(range / 32 + 1);
        this->first_index.resize(range / 1024 + 1);
        this->second_index.resize(range / 32768 + 1);
    }

    void set_value(int data)
    {
        int index0 = data / 32;
        int index1 = index0 / 32;
        int index2 = index1 / 32;
        int tmp0 = data % 32;
        int tmp1 = index0 % 32;
        int tmp2 = index1 % 32;

        this->m_bits[index0] |= (1 << tmp0);
        this->first_index[index1] |= (1 << tmp1);
        this->second_index[index2] |= (1 << tmp2);
    }

    void reset(int data)
    {
        int index = data / 32;
        int tmp = data % 32;
        this->m_bits[index] &= ~(1 << tmp);
    }
    vector<int> m_bits;
    vector<int> first_index;
    vector<int> second_index;
};

index t_index;
index n_index;
vector<index> idx;
BitMap n_bit(30000000);

void search_list_bit(queue &q, int *query, vector<index> &idx, int num)
{
    vector<index> t_idx;
    for (int i = 0; i < num; i++)
    {
        t_idx.push_back(idx[query[i]]);
    }
    sort(t_idx.begin(), t_idx.end());

    vector<BitMap> bitmap;
    for (int i = 0; i < num; i++)
    {
        bitmap.push_back(30000000);
        for (int j = 0; j < t_idx[i].len; j++)
        {
            bitmap[i].set_value(t_idx[i].order[j]);
        }
    }

    n_bit = bitmap[0];

    buffer<int, 1> buf_second_index(n_bit.second_index.data(), range<1>(n_bit.second_index.size()));
    buffer<int, 1> buf_first_index(n_bit.first_index.data(), range<1>(n_bit.first_index.size()));
    buffer<int, 1> buf_m_bits(n_bit.m_bits.data(), range<1>(n_bit.m_bits.size()));
    vector<int> result_second_index(n_bit.second_index.size());
    vector<int> result_first_index(n_bit.first_index.size());
    vector<int> result_m_bits(n_bit.m_bits.size());

    for (int i = 1; i < num; i++)
    {
        q.submit([&](handler &h) {
            auto acc_second_index = buf_second_index.get_access<access::mode::read_write>(h);
            auto acc_first_index = buf_first_index.get_access<access::mode::read_write>(h);
            auto acc_m_bits = buf_m_bits.get_access<access::mode::read_write>(h);

            h.parallel_for(range<1>(n_bit.second_index.size()), [=](id<1> j) {
                bool judge = false;
                acc_second_index[j] &= bitmap[i].second_index[j];
                if (acc_second_index[j] != 0)
                {
                    for (int t = j * 32; t < j * 32 + 32; t++)
                    {
                        acc_first_index[t] &= bitmap[i].first_index[t];
                        if (acc_first_index[t] != 0)
                        {
                            for (int l = t * 32; l < t * 32 + 32; l++)
                            {
                                acc_m_bits[l] &= bitmap[i].m_bits[l];
                                if (acc_m_bits[l] != 0)
                                {
                                    judge = true;
                                }
                            }
                        }
                    }
                }
                if (judge == false)
                {
                    acc_second_index[j] = 0;
                }
            });
        });
    }

    q.wait();

    q.submit([&](handler &h) {
        auto acc_second_index = buf_second_index.get_access<access::mode::read>(h);
        auto acc_first_index = buf_first_index.get_access<access::mode::read>(h);
        auto acc_m_bits = buf_m_bits.get_access<access::mode::read>(h);

        h.copy(acc_second_index, result_second_index.data());
        h.copy(acc_first_index, result_first_index.data());
        h.copy(acc_m_bits, result_m_bits.data());
    }).wait();

    n_bit.second_index = result_second_index;
    n_bit.first_index = result_first_index;
    n_bit.m_bits = result_m_bits;
}

void gettime(queue &q, void (*func)(queue &, int *query, vector<index> &idx, int num), int t_query[1000][5], vector<index> &idx)
{
    long long head, tail, freq;
    QueryPerformanceFrequency((LARGE_INTEGER *)&freq);
    QueryPerformanceCounter((LARGE_INTEGER *)&head);
    for (int i = 0; i < 1000; i++)
    {
        int num = 0;
        for (int j = 0; j < 5; j++)
        {
            if (t_query[i][j] != 0)
            {
                num++;
            }
        }
        int *query = new int[num];
        for (int j = 0; j < num; j++)
        {
            query[j] = t_query[i][j];
        }
        func(q, query, idx, num);
        delete[] query;
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << ((tail - head) * 1000.0 / freq) * 1000.0 << "ms" << '\n';
}

int main()
{
    queue q;

    fstream outfile;
    outfile.open("ExpIndex", ios::binary | ios::in);
    for (int i = 0; i < 2000; i++)
    {
        index tmp;
        outfile.read((char *)&tmp.len, sizeof(tmp.len));
        for (int j = 0; j < (tmp.len); j++)
        {
            unsigned int n_tmp;
            outfile.read((char *)&n_tmp, sizeof(n_tmp));
            tmp.order.push_back(n_tmp);
        }
        idx.push_back(tmp);
    }
    outfile.close();
    outfile.open("ExpQuery", ios::in);
    int t_query[1000][5] = {0};
    string line;
    int n_count = 0;
    while (getline(outfile, line))
    {
        stringstream ss(line);
        int addr = 0;
        while (!ss.eof())
        {
            int tmp;
            ss >> tmp;
            t_query[n_count][addr] = tmp;
            addr++;
        }
        n_count++;
    }
    outfile.close();
    cout << "按表求交(位图):";
    gettime(q, search_list_bit, t_query, idx);
    return 0;
}
