#include <iostream>
#include <fstream>
#include <vector>
#include <sys/time.h>
#include <arm_neon.h>
#include <ctime>
#include <cmath>
#include <math.h>
#include <string>
#include <string.h>
#include <sstream>
#include <algorithm>
#include <pthread.h>
#define MAXSIZE 2000
#define NUM_THREADS 4
typedef long long ll;
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

void *threadFunc_search_list_d(void *param)
{
	threadParam_t *p = (threadParam_t *)param;
	int t_id = p->t_id;
	int num_of_query = p->num_of_query;
	int startt = (n_index.order.size() / 4) * t_id;
	int endd;
	int count = 0;
	if (t_id == 3)
	{
		endd = n_index.order.size();
	}
	else
	{
		endd = startt + (n_index.order.size()) / 4;
	}
	index n_idx;
	n_index.order.assign(n_index.order.begin() + startt, n_index.order.begin() + endd);

	uint32_t length = ceil((endd - startt) / 4) * 4;
	for (int m = endd; m < length; m++)
	{
		n_idx.order[m] = 0;
		idx[num_of_list].order[m] = 0;
	}

	for (int i = 0; i < endd - startt; i++)
	{
		bool isexit = false;
		for (int t = 0; t < length; t += 4)
		{
			unsigned int compare[4] = {0};
			uint32x4_t tmp0 = vmovq_n_u32(n_idx.order[count]);
			uint32x4_t tmp1 = vld1q_u32(&idx[num_of_list].order[t]);
			uint32x4_t tmp = vceqq_u32(tmp0, tmp1);
			vst1q_u32(compare, tmp);
			if (compare[0] == 1 || compare[1] == 1 || compare[2] == 1 || compare[3] == 1)
			{
				isexit = true;
				break;
			};
		}
		if (isexit == false)
		{
			n_idx.order[count] = -1;
		}
		count++;
	}
	vector<unsigned int>::iterator newEnd(remove(n_idx.order.begin(), n_idx.order.end(), -1));
	n_idx.order.erase(newEnd);
	p->n_idx = n_idx;
	return p;
}

void search_list_d(int *query, vector<index> &idx, int num)
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
		int length = n_index.len;
		for (int j = 0; j < 4; j++)
		{
			param[j].num_of_query = query[i];
			param[j].t_id = j;
			pthread_create(&thread[j], NULL, threadFunc_search_list_d, &param[j]);
		}
		void *ttmp;
		for (int i = 0; i < 4; i++)
		{
			pthread_join(thread[i], &ttmp);
			param[i] = *(threadParam_t *)ttmp;
		}
		index n_index0;
		for (int i = 0; i < 4; i++)
		{
			n_index0.order.insert(n_index0.order.end(), param[i].n_idx.order.begin(), param[i].n_idx.order.end());
		}
		n_index0.len = n_index0.order.size();
		n_index = s2;
	}
}

void gettime(void (*func)(int *query, vector<index> &idx, int num), int t_query[1000][5], vector<index> &idx)
{
	timeval tv_begin, tv_end;
	gettimeofday(&tv_begin, 0);
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
	gettimeofday(&tv_begin, 0);
	cout << ((ll)tv_end.tv_sec - (ll)tv_begin.tv_sec) * 1000.0 + ((ll)tv_end.tv_usec - (ll)tv_begin.tv_usec) / 1000.0 << "ms" << '\n';
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
