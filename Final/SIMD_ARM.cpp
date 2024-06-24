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
#define MAXSIZE 2000
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

void search_list_bit_SIMD(int *query, vector<index> &idx, int num)
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
					uint32x4_t zero, var, var0, var1;
					zero = vmovq_n_u32(0);
					var0 = vmovq_n_u32(n_bit.first_index[t]);
					var1 = vld1q_u32(&bitmap[i].first_index[t]);
					var = vornq_u32(var0, var1);
					int compare[4] = {0};
					vst1q_u32(compare, var);
					vector<int> constract = search(compare);
					int t_count = 0;
					while (!constract.empty() && n_bit.first_index[t] != 0)
					{
						int p = constract[t_count];
						for (int l = p * 32; l < p * 32 + 32; l += 4)
						{
							uint32x4_t tmp, tmp0, tmp1;
							tmp0 = vmovq_n_u32(n_bit.m_bits[l]);
							tmp1 = vld1q_u32(&bitmap[i].m_bits[l]);
							tmp = vld1q_u32(tmp0, tmp1);
							int compare0[4] = {0};
							vst1q_u32(compare0, tmp);
							vector<int> constract0 = search(compare0);
							int n_count = 0;
							while (!constract0.empty() && n_bit.m_bits[l] != 0)
							{
								int q = constract0[n_count];
								q *= 32;
								n_bit.m_bits[q] &= bitmap.m_bits[q];
								if (n_bit.m_bits[q] != 0)
								{
									judge = true;
								}
								n_count++;
								constract0.pop_back();
							}
						}
					}
					t_count++;
					constract.pop_back();
				}
			}
			if (judge == false)
			{
				n_bit.second_index[j] = 0;
			}
		}
	}
}

void search_list_SIMD(int *query, vector<index> &idx, int num)
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
			uint32x4_t var, var0;
			var0 = vmovq_n_u32(n_index.order[t_count]);
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
				uint32x4_t var1;
				var1 = vld1q_u32(&t_idx[i].order[o]);
				var = vceqq_u32(var0, var1);
				int compare[4] = {0};
				vst1q_u32(compare, var);
				if (compare[0] == 1 && compare[1] == 1 && compare[2] == 1 && compare[3] == 1)
				{
					judge = true;
					break;
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

void search_element_SIMD(int *query, vector<index> &idx, int num)
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
		uint32x4_t var0;
		var0 = vmovq_n_u32(n_index.order[i]);
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
				uint32x4_t var1, var;
				var1 = vld1q_u32(&t_idx[j].order[o]);
				var = vceqq_u32(var0, var1);
				int compare[4] = {0};
				vst1q_u32(compare, var);
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
