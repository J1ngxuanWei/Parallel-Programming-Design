#include <iostream>
#include <fstream>
#include <vector>
#include <sys/time.h>
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

void search_list_bit(int *query, vector<index> &idx, int num)
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
				for (int t = j * 32; t < j * 32 + 32; t++)
				{
					n_bit.first_index[t] &= bitmap[i].first_index[t];
					if (n_bit.first_index[t] != 0)
					{
						for (int l = t * 32; l < t * 32 + 32; l++)
						{
							n_bit.m_bits[l] &= bitmap[i].m_bits[l];
							if (n_bit.m_bits[l] != 0)
							{
								judge = true;
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

void search_element_bit(int *query, vector<index> &idx, int num)
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
	bool judge = false;
	n_bit = bitmap[0];
	for (int i = 0; i < n_bit.second_index.size(); i++)
	{
		for (int j = 1; j < num; j++)
		{
			n_bit.second_index[i] &= bitmap[j].second_index[i];
			if (n_bit.second_index[i] != 0)
			{
				for (int t = i * 32; t < i * 32 + 32; t++)
				{
					n_bit.first_index[t] &= bitmap[j].first_index[t];
					if (n_bit.first_index[t] != 0)
					{
						for (int l = t * 32; l < t * 32 + 32; l++)
						{
							n_bit.m_bits[l] &= bitmap[j].m_bits[l];
							if (n_bit.m_bits[l] != 0)
							{
								judge = true;
							}
						}
					}
				}
			}
			else
			{
				break;
			}
		}
	}
}

void search_list(int *query, vector<index> &idx, int num)
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
			bool judge = false;
			for (int o = 0; o < t_idx[i].len; o++)
			{
				if (n_index.order[t_count] == t_idx[i].order[o])
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

void search_element(int *query, vector<index> &idx, int num)
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
		for (int j = 1; j < num; j++)
		{
			for (int o = 0; o < t_idx[j].len; o++)
			{
				if (t_idx[j].order[o] == n_index.order[i])
				{
					t_count++;
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
	gettimeofday(&tv_end, 0);
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
	cout << "按表求交(平凡):";
	gettime(search_list, t_query, idx);
	cout << "按元素求交(平凡):";
	gettime(search_element, t_query, idx);
	cout << "按表求交(位图):";
	gettime(search_list_bit, t_query, idx);
	cout << "按元素求交(位图):";
	gettime(search_element_bit, t_query, idx);
	return 0;
}
