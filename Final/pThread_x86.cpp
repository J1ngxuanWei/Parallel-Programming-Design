#include <xmmintrin.h>
#include <emmintrin.h>
#include <pmmintrin.h>
#include <tmmintrin.h>
#include <smmintrin.h>
#include <nmmintrin.h>
#include <immintrin.h>
#include <pthread.h>
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
#define NUM_THREADS 4
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

vector<Bitmap> bm;

BitMap n_bit(30000000);

pthread_t thread[NUM_THREADS];

struct threadParam_t
{
	int t_id;
	int num_of_query;
	int tmp;
	index n_idx;
	bool isfound = false;
	bool judge;
};
threadParam_t param[NUM_THREADS];

struct threadParam_bitamp
{
};
typedef struct NamedType : threadParam_bitamp
{
	int t_id;
	bool isfound = false;
	bool judge;
} threadParam_bitmap;
threadParam_bitmap param_bitmap[NUM_THREADS];

vector<unsigned int> n_tmp[NUM_THREADS];

void *threadFunc_search_list_bit(void *param_bit)
{
	threadParam_bitmap *p = (threadParam_bit *)param_bit;
	int isfound = p->isfound;
	int id = p->t_id;
	for (unsigned int j = 0; j < bm[id].second_index.size(); j++)
	{
		n_bit.second_index[j] &= bm[id].second_index[j];
		if (n_bit.second_index[j] != 0)
		{
			for (int t = j * 32; t < j * 32 + 32; t++)
			{
				n_bit.first_index[t] &= bm[id].first_index[t];
				if (n_bit.first_index[t] != 0)
				{
					for (int l = t * 32; l < t * 32 + 32; l++)
					{
						n_bit.m_bits[l] &= bm[id].m_bits[l] if (n_bit.m_bits[l] != 0)
						{
							isfound = true;
						}
					}
				}
			}
			if (isfound == false)
			{
				n_bit.second_index[j] = 0;
			}
		}
	}
	return p;
}

void search_list_bit_SIMD_SSE_P(int *query, vector<index> &idx, int num)
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
	for (int i = 1; i < num; i += 4)
	{
		int tmp = i;
		for (int o = 0; o < 4; o++)
		{
			bm.push_back(30000000);
		}
		for (int o = 0; o < 4; o++)
		{

			if (tmp < num)
			{
				bm[o] = bitmap[tmp];
				param_bitmap[o].t_id = o;
				param_bitmap[o].judge = true;
				pthread_create(&thread[o], NULL, threadFunc_search_list_bit, &param_bitmap[o]);
			}
			else
			{
				param_bitmap[k].judge = false;
			}
			tmp++;
		}
		bm.clear();
	}
}

void *threadFunc_search_element_d(void *param)
{
	threadParam_t *p = (threadParam_t *)param;

	unsigned int m = p->tmp;
	__m128i var0;
	var0 = _mm_set1_epi32(m);

	int length = ceil(p->n_idx.len / 4) * 4;
	for (int m = p->s0.len; m < length; m++)
	{
		p->n_idx.order[m] = 0;
	}

	if (p->judge == true)
	{
		int t_id = p->t_id;
		int num_of_query = p->num_of_query;
		bool found = p->isfound;

		for (int i = 0; i < p->s0.len; i += 4)
		{
			found = false;
			__m128i var1, var;
			var1 = _mm_loadu_epi32(&p->n_idx.order[i]);
			var = _mm_set1_epi32(0);
			var = _mm_sub_epi32(var0, var1);
			int compare[4] = {0};
			_mm_storeu_epi32(&compare[0], var);
			_mm_storeu_epi32(&compare[2], var);
			for (int i = 0; i < 4; i++)
			{
				if (compare[i] == 0)
				{
					found = true;
					break;
				}
			}
		}
		p->isfound = found;
	}
}

void search_element_SIMD_SSE_P_d(int *query, vector<index> &idx, int num)
{
	vector<index> t_idx;
	for (int i = 0; i < num; i++)
	{
		t_idx.push_back(idx[query[i]]);
	}
	sort(t_idx.begin(), t_idx.end());
	n_index = t_idx[0];
	vector<unsigned int> nn;
	for (int i = 0; i < n_index.len; i++)
	{
		int t_count = 0;
		int var = n_index.order[i];
		int j;
		for (j = 1; j <= num;)
		{
			for (int o = 0; o < 4; o++, j++)
			{
				if (j < num)
				{
					param[o].num_of_query = query[j];
					param[o].t_id = o;
					param[o].tmp = var;
					param[o].n_idx = t_idx[j];
					param[o].isfound = found;
					param[o].judge = true;
					pthread_create(&thread[o], NULL, threadFunc_search_element, &param[o]);
				}
				else
				{
					param[o + 1].judge = false;
					break;
				}
			}
			void *compare;
			for (int o = 0; o < num - 1; o++)
			{
				pthread_join(thread[o], &compare);
				param[o] = *(threadParam_t *)compare;
			}
			for (int o = 0; o < 4; o++)
			{
				if (param[o].isfound == false)
				{
					found = false;
				}
			}
		}
		if (j >= num && found == true)
		{
			nn.push_back(var);
		}
	}
}

void *threadFunc_search_element_s(void *param)
{
	threadParam_t *p = (threadParam_t *)param;
	bool found = true;
	for (int i = 0; i < p->n_idx.order.size(); i++)
	{
		int s = 1;
		unsigned int var = idx0[0].order[t];
		while (s != p->num && found == true)
		{
			for (int i = 0; i < idx0[s].len; i++)
			{
				found = false;
				if (var == idx0[s].order[i])
				{
					found = true;
					break;
				}
			}
			s = s + 1;
		}
		if (s == p->num && found == true)
		{
			n_tmp[p->t_id].push_back(e);
		}
	}
	sem_post(&sem_main);
	return p;
}

vector<index> idx0;

void search_element_SIMD_SSE_P_s(int *query, vector<index> &idx, int num)
{
	vector<unsigned int> nn;
	sem_init(&sem_main, 0, 0);
	for (int i = 0; i < NUM_THREADS; i++)
	{
		sem_init(&sem_workerstart[i], 0, 0);
		sem_init(&sem_workerend[i], 0, 0);
	}
	for (int i = 0; i < num; i++)
	{
		idx0.push_back(idx[query[i]]);
	}
	sort(idx0.begin(), idx0.end());
	n_index = idx0[0];
	int spare = n_index.order.size() / 4;
	int endd = 0;
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		param[t_id].t_id = t_id;
		param[t_id].num_of_query = query;
		param[t_id].num = num;
		int startt = spare * t_id;
		if (t_id == 3)
		{
			endd = n_index.order.size();
		}
		else
		{
			endd = startt + spare;
		}
		param[t_id].n_idx.order.assign(n_index.order.begin() + startt, n_index.order.begin() + endd);
		pthread_create(&thread[t_id], NULL, threadFunc_search_element_s, &param[t_id]);
	}
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		sem_post(&sem_workerstart[t_id]);
	}
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		sem_wait(&sem_main);
	}
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		nn.insert(nn.end(), n_tmp[t_id].begin(), n_tmp[t_id].end());
	}
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		pthread_join(thread[t_id], NULL);
	}
	sem_destroy(&sem_main);
	for (int t_id = 0; t_id < NUM_THREADS; t_id++)
	{
		sem_destroy(&sem_workerstart[t_id]);
		sem_destroy(&sem_workerend[t_id]);
	}
	return nn;
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
