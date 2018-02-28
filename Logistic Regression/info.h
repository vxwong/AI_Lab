#include <bits/stdc++.h>
using namespace std;

/*******
	储存训练过程及验证过程中的数据
*******/

#define max_row 8050	//最大数据集行数
#define max_col 45		//最大数据集列数

class Info{			
public:
	double  **set;	//数据矩阵
	int row,		//行数
		col;		//列数
	Info() = default;	
	Info(int, int);

};

Info::Info(int row, int col){
	this->row = row;
	this->col = col;
	set = new double *[row];
	//初始化数据集
	for (int i=0; i<row; ++i){
		set[i] = new double [col];
		set[i][0] = 1;	//验收要求偏置位为1
	}
}

double tmp_set[max_row][max_col]; 
Info read_file (string f){
	//读取数据
	fstream in(f.c_str());
	if (!in.is_open()) {cout << f << " open failed\n"; exit(1);}
	const char *p = ",?";
	string raw_data;
	int row = 0, col = 0;
	while (getline(in, raw_data)){
		int _col = 1;
		char *tmp = strtok((char *)raw_data.data(), p);
		while (tmp){
			tmp_set[row][_col++] = atof(tmp);
			tmp = strtok(NULL, p);
		}
		row ++;
		col = _col;
	}
	in.close();

	//把数据存到Info
	Info info(row, col);

	for (int i=0; i<row; ++i)
		for (int j=1; j<col; ++j)
			info.set[i][j] = tmp_set[i][j];

	return info;
}

