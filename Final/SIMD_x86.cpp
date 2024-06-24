#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <iostream>
#include <fstream>
#include <vector>
#include <windows.h>
#include <ctime>
#include <cmath>
#include <math.h>
#include <string>
#include <string.h>
#include <sstream>
#include <algorithm>
#define MAXSIZE 2000
using namespace std;
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

void search_list_bit_SIMD_SSE(int *query, vector<index> &idx, int num)
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
    for (int i = 1; i < num; i++)
    {
        for (int j = 0; j < n_bit.second_index.size(); j++)
        {
            bool judge = false;
            n_bit.second_index[j] &= bitmap[i].second_index[j];
            if (n_bit.second_index[j] != 0)
            {
                for (int t = j * 32; t < j * 32 + 32; t += 4)
                {
                    __m128i var, var0, var1;
                    var0 = _mm_set_epi32(n_bit.first_index[t], n_bit.first_index[t + 1], n_bit.first_index[t + 2], n_bit.first_index[t + 3]);
                    var1 = _mm_set_epi32(bitmap[i].first_index[t], bitmap[i].first_index[t + 1], bitmap[i].first_index[t + 2], bitmap[i].first_index[t + 3]);
                    var = _mm_and_si128(var0, var1);
                    int compare[4] = {0};
                    _mm_storeu_epi32(compare, var);
                    for (int r = 0; r < 4; r++)
                    {
                        if (compare[r] != 0)
                        {
                            for (int l = (t + r) * 32; l < (t + r) * 32 + 32; l += 4)
                            {
                                __m128i tmp, tmp0, tmp1;
                                tmp0 = _mm_set_epi32(n_bit.m_bits[l], n_bit.m_bits[l + 1], n_bit.m_bits[l + 2], n_bit.m_bits[l + 3]);
                                tmp1 = _mm_set_epi32(bitmap[i].m_bits[l], bitmap[i].m_bits[l + 1], bitmap[i].m_bits[l + 2], bitmap[i].m_bits[l + 3]);
                                tmp = _mm_and_si128(tmp0, tmp1);
                                int compare0[4] = {0};
                                _mm_storeu_epi32(compare0, tmp);
                                for (int s = 0; s < 4; s++)
                                {
                                    if (compare0[s] != 0)
                                    {
                                        judge = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (judge == false)
            {
                n_bit.second_index[j] = 0;
            }
        }
    }
}

void search_list_bit_SIMD_AVX(int *query, vector<index> &idx, int num)
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
    for (int i = 1; i < num; i++)
    {
        for (int j = 0; j < n_bit.second_index.size(); j++)
        {
            bool judge = false;
            n_bit.second_index[j] &= bitmap[i].second_index[j];
            if (n_bit.second_index[j] != 0)
            {
                for (int t = j * 32; t < j * 32 + 32; t += 8)
                {
                    __m256i var, var0, var1;
                    var0 = _mm256_set_epi32(n_bit.first_index[t], n_bit.first_index[t + 1], n_bit.first_index[t + 2], n_bit.first_index[t + 3], n_bit.first_index[t + 4], n_bit.first_index[t + 5], n_bit.first_index[t + 6], n_bit.first_index[t + 7]);
                    var1 = _mm256_set_epi32(bitmap[i].first_index[t], bitmap[i].first_index[t + 1], bitmap[i].first_index[t + 2], bitmap[i].first_index[t + 3], bitmap[i].first_index[t + 4], bitmap[i].first_index[t + 5], bitmap[i].first_index[t + 6], bitmap[i].first_index[t + 7]);
                    var = _mm256_and_si256(var0, var1);
                    int compare[8] = {0};
                    _mm256_storeu_epi32(compare, var);
                    for (int r = 0; r < 8; r++)
                    {
                        if (compare[r] != 0)
                        {
                            for (int l = (t + r) * 32; l < (t + r) * 32 + 32; l += 8)
                            {
                                __m256i tmp, tmp0, tmp1;
                                tmp0 = _mm256_set_epi32(n_bit.m_bits[l], n_bit.m_bits[l + 1], n_bit.m_bits[l + 2], n_bit.m_bits[l + 3], n_bit.m_bits[l + 4], n_bit.m_bits[l + 5], n_bit.m_bits[l + 6], n_bit.m_bits[l + 7]);
                                tmp1 = _mm256_set_epi32(bitmap[i].m_bits[l], bitmap[i].m_bits[l + 1], bitmap[i].m_bits[l + 2], bitmap[i].m_bits[l + 3], bitmap[i].m_bits[l + 4], bitmap[i].m_bits[l + 5], bitmap[i].m_bits[l + 6], bitmap[i].m_bits[l + 7]);
                                tmp = _mm256_and_si256(tmp0, tmp1);
                                int compare0[8] = {0};
                                _mm256_storeu_epi32(compare0, tmp);
                                for (int s = 0; s < 8; s++)
                                {
                                    if (compare0[s] != 0)
                                    {
                                        judge = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            if (judge == false)
            {
                n_bit.second_index[j] = 0;
            }
        }
    }
}

void search_element_bit_SIMD_SSE(int *query, vector<index> &idx, int num)
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
    for (int i = 0; i < n_bit.second_index.size(); i++)
    {
        bool judge = false;
        for (int j = 1; j < num; j++)
        {
            n_bit.second_index[i] &= bitmap[j].second_index[i];
            if (n_bit.second_index[i] != 0)
            {
                for (int t = i * 32; t < i * 32 + 32; t += 4)
                {
                    __m128i var, var0, var1;
                    var0 = _mm_set_epi32(n_bit.first_index[t], n_bit.first_index[t + 1], n_bit.first_index[t + 2], n_bit.first_index[t + 3]);
                    var1 = _mm_set_epi32(bitmap[i].first_index[t], bitmap[i].first_index[t + 1], bitmap[i].first_index[t + 2], bitmap[i].first_index[t + 3]);
                    var = _mm_and_si128(var0, var1);
                    int compare[4] = {0};
                    _mm_storeu_epi32(compare, var);
                    for (int r = 0; r < 4; r++)
                    {
                        if (compare[r] != 0)
                        {
                            for (int l = (t + r) * 32; l < (t + r) * 32 + 32; l += 4)
                            {
                                __m128i tmp, tmp0, tmp1;
                                tmp0 = _mm_set_epi32(n_bit.m_bits[l], n_bit.m_bits[l + 1], n_bit.m_bits[l + 2], n_bit.m_bits[l + 3]);
                                tmp1 = _mm_set_epi32(bitmap[i].m_bits[l], bitmap[i].m_bits[l + 1], bitmap[i].m_bits[l + 2], bitmap[i].m_bits[l + 3]);
                                tmp = _mm_and_si128(tmp0, tmp1);
                                int compare0[4] = {0};
                                _mm_storeu_epi32(compare0, tmp);
                                for (int s = 0; s < 4; s++)
                                {
                                    if (compare0[s] != 0)
                                    {
                                        judge = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                break;
            }
            if (judge == false)
            {
                n_bit.second_index[i] = 0;
                break;
            }
        }
    }
}

void search_element_bit_SIMD_AVX(int *query, vector<index> &idx, int num)
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
    for (int i = 0; i < n_bit.second_index.size(); i++)
    {
        bool judge = false;
        for (int j = 1; j < num; j++)
        {
            n_bit.second_index[i] &= bitmap[j].second_index[i];
            if (n_bit.second_index[i] != 0)
            {
                for (int t = i * 32; t < i * 32 + 32; t += 8)
                {
                    __m256i var, var0, var1;
                    var0 = _mm256_set_epi32(n_bit.first_index[t], n_bit.first_index[t + 1], n_bit.first_index[t + 2], n_bit.first_index[t + 3], n_bit.first_index[t + 4], n_bit.first_index[t + 5], n_bit.first_index[t + 6], n_bit.first_index[t + 7]);
                    var1 = _mm256_set_epi32(bitmap[i].first_index[t], bitmap[i].first_index[t + 1], bitmap[i].first_index[t + 2], bitmap[i].first_index[t + 3], bitmap[i].first_index[t + 4], bitmap[i].first_index[t + 5], bitmap[i].first_index[t + 6], bitmap[i].first_index[t + 7]);
                    var = _mm256_and_si256(var0, var1);
                    int compare[8] = {0};
                    _mm256_storeu_epi32(compare, var);
                    for (int r = 0; r < 8; r++)
                    {
                        if (compare[r] != 0)
                        {
                            for (int l = (t + r) * 32; l < (t + r) * 32 + 32; l += 8)
                            {
                                __m256i tmp, tmp0, tmp1;
                                tmp0 = _mm256_set_epi32(n_bit.m_bits[l], n_bit.m_bits[l + 1], n_bit.m_bits[l + 2], n_bit.m_bits[l + 3], n_bit.m_bits[l + 4], n_bit.m_bits[l + 5], n_bit.m_bits[l + 6], n_bit.m_bits[l + 7]);
                                tmp1 = _mm256_set_epi32(bitmap[i].m_bits[l], bitmap[i].m_bits[l + 1], bitmap[i].m_bits[l + 2], bitmap[i].m_bits[l + 3], bitmap[i].m_bits[l + 4], bitmap[i].m_bits[l + 5], bitmap[i].m_bits[l + 6], bitmap[i].m_bits[l + 7]);
                                tmp = _mm256_and_si256(tmp0, tmp1);
                                int compare0[8] = {0};
                                _mm256_storeu_epi32(compare0, tmp);
                                for (int s = 0; s < 8; s++)
                                {
                                    if (compare0[s] != 0)
                                    {
                                        judge = true;
                                    }
                                }
                            }
                        }
                    }
                }
            }
            else
            {
                break;
            }
            if (judge == false)
            {
                n_bit.second_index[i] = 0;
                break;
            }
        }
    }
}

void search_list_SIMD_SSE(int *query, vector<index> &idx, int num)
{
    vector<index> t_idx;
    for (int i = 0; i < num; i++)
    {
        t_idx.push_back(idx[query[i]]);
    }
    sort(t_idx.begin(), t_idx.end());
    n_index = t_idx[0];
    for (int i = 1; i < num; i++)
    {
        int t_count = 0;
        for (int j = 0; j < n_index.len; j++)
        {
            __m128i var, var0;
            var0 = _mm_set1_epi32(n_index.order[t_count]);
            bool judge = false;
            int o;
            for (o = 0; o < t_idx[i].len && ((t_idx[i].len - o) & 3); o++)
            {
                if (n_index.order[j] == t_idx[i].order[o])
                {
                    judge = true;
                    break;
                }
            }
            for (; o < t_idx[i].len; o += 4)
            {
                __m128i var1;
                var1 = _mm_set_epi32(t_idx[i].order[o], t_idx[i].order[o + 1], t_idx[i].order[o + 2], t_idx[i].order[o + 3]);
                var = _mm_set1_epi32(0);
                var = _mm_sub_epi32(var0, var1);
                int compare[4] = {0};
                _mm_storeu_epi32(compare, var);
                for (int r = 0; r < 4; r++)
                {
                    if (compare[r] == 0)
                    {
                        judge = true;
                        break;
                    }
                }
            }
            if (judge == false)
            {
                n_index.len--;
                n_index.order.erase(n_index.order.begin() + t_count);
            }
            else
            {
                t_count++;
            }
        }
    }
}

void search_list_SIMD_AVX(int *query, vector<index> &idx, int num)
{
    vector<index> t_idx;
    for (int i = 0; i < num; i++)
    {
        t_idx.push_back(idx[query[i]]);
    }
    sort(t_idx.begin(), t_idx.end());
    n_index = t_idx[0];
    for (int i = 1; i < num; i++)
    {
        int t_count = 0;
        for (int j = 0; j < n_index.len; j++)
        {
            __m256i var, var0;
            var0 = _mm256_set1_epi32(n_index.order[t_count]);
            bool judge = false;
            int o;
            for (o = 0; o < t_idx[i].len && ((t_idx[i].len - o) & 7); o++)
            {
                if (n_index.order[j] == t_idx[i].order[o])
                {
                    judge = true;
                    break;
                }
            }
            for (; o < t_idx[i].len; o += 8)
            {
                __m256i var1;
                var1 = _mm256_set_epi32(t_idx[i].order[o], t_idx[i].order[o + 1], t_idx[i].order[o + 2], t_idx[i].order[o + 3], t_idx[i].order[o + 4], t_idx[i].order[o + 5], t_idx[i].order[o + 6], t_idx[i].order[o + 7]);
                var = _mm256_set1_epi32(0);
                var = _mm256_sub_epi32(var0, var1);
                int compare[8] = {0};
                _mm256_storeu_epi32(compare, var);
                for (int r = 0; r < 8; r++)
                {
                    if (compare[r] == 0)
                    {
                        judge = true;
                        break;
                    }
                }
            }
            if (judge == false)
            {
                n_index.len--;
                n_index.order.erase(n_index.order.begin() + t_count);
            }
            else
            {
                t_count++;
            }
        }
    }
}

void search_element_SIMD_SSE(int *query, vector<index> &idx, int num)
{
    vector<index> t_idx;
    for (int i = 0; i < num; i++)
    {
        t_idx.push_back(idx[query[i]]);
    }
    sort(t_idx.begin(), t_idx.end());
    n_index = t_idx[0];
    for (int i = 0; i < n_index.len; i++)
    {
        int t_count = 0;
        __m128i var0;
        var0 = _mm_set1_epi32(n_index.order[i]);
        for (int j = 1; j < num; j++)
        {
            bool judge = false;
            int o;
            for (o = 0; o < t_idx[j].len && ((t_idx[j].len - o) & 3); o++)
            {
                if (t_idx[j].order[o] == n_index.order[i])
                {
                    t_count++;
                    judge = true;
                    break;
                }
            }
            if (judge)
            {
                continue;
            }
            for (; o < t_idx[j].len; o += 4)
            {
                __m128i var1, var;
                var1 = _mm_set_epi32(t_idx[j].order[o], t_idx[j].order[o + 1], t_idx[j].order[o + 2], t_idx[j].order[o + 3]);
                var = _mm_set1_epi32(0);
                var = _mm_sub_epi32(var0, var1);
                int compare[4] = {0};
                _mm_storeu_epi32(compare, var);
                for (int r = 0; r < 4; r++)
                {
                    if (compare[r] == 0)
                    {
                        t_count++;
                        judge = true;
                        break;
                    }
                }
                if (judge)
                {
                    break;
                }
            }
        }
        if (t_count == num - 1)
        {
            t_index.len++;
            t_index.order.push_back(n_index.order[i]);
        }
    }
}

void search_element_SIMD_AVX(int *query, vector<index> &idx, int num)
{
    vector<index> t_idx;
    for (int i = 0; i < num; i++)
    {
        t_idx.push_back(idx[query[i]]);
    }
    sort(t_idx.begin(), t_idx.end());
    n_index = t_idx[0];
    for (int i = 0; i < n_index.len; i++)
    {
        int t_count = 0;
        __m256i var0;
        var0 = _mm256_set1_epi32(n_index.order[i]);
        for (int j = 1; j < num; j++)
        {
            bool judge = false;
            int o;
            for (o = 0; o < t_idx[j].len && ((t_idx[j].len - o) & 7); o++)
            {
                if (t_idx[j].order[o] == n_index.order[i])
                {
                    t_count++;
                    judge = true;
                    break;
                }
            }
            if (judge)
            {
                continue;
            }
            for (; o < t_idx[j].len; o += 4)
            {
                __m256i var1, var;
                var1 = _mm256_set_epi32(t_idx[i].order[o], t_idx[i].order[o + 1], t_idx[i].order[o + 2], t_idx[i].order[o + 3], t_idx[i].order[o + 4], t_idx[i].order[o + 5], t_idx[i].order[o + 6], t_idx[i].order[o + 7]);
                var = _mm256_set1_epi32(0);
                var = _mm256_sub_epi32(var0, var1);
                int compare[8] = {0};
                _mm256_storeu_epi32(compare, var);
                for (int r = 0; r < 8; r++)
                {
                    if (compare[r] == 0)
                    {
                        t_count++;
                        judge = true;
                        break;
                    }
                }
                if (judge)
                {
                    break;
                }
            }
        }
        if (t_count == num - 1)
        {
            t_index.len++;
            t_index.order.push_back(n_index.order[i]);
        }
    }
}

void gettime(void (*func)(int *query, vector<index> &idx, int num), int t_query[1000][5], vector<index> &idx)
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
        func(query, idx, num);
        delete query;
    }
    QueryPerformanceCounter((LARGE_INTEGER *)&tail);
    cout << ((tail - head) * 1000.0 / freq) * 1000.0 << "ms" << '\n';
}

int main()
{
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
    return 0;
}
