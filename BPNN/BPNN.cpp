#include <bits/stdc++.h>
using namespace std;

#define max_row 7000
#define max_col 13
#define neuron 70
#define itr_times 1000
#define learn_rate 0.00001
/*
| season | mnt | hr | holiday | weekday | workingday | wether(1-4) | temp | atemp|  humidity |windspeed |cnt
*/
int train_row = 0, train_col = 0;
double train_set[max_row][max_col] = {0};		//输入层
double input_to_hide[neuron][max_col] = {0};	//输入层到隐藏层权重
double hide_to_output[neuron] = {0};			//隐藏层到输出层权重
double hide_output[neuron] ={0};				//隐藏层输出
double output[max_row] = {0};					//输出层输出
double hide_b[neuron] = {0};					//隐藏层输入偏置
double ouput_b = 0;								//输出层输入偏置 	
double val_set[max_row][max_col]={0};			//验证集
// double val_set[max_row][max_col];		

void read_file (string f, int & row, int & col, double set[][max_col])
{
	fstream in(f.c_str());
	if (!in.is_open()) {cout << f << " open failed\n"; exit(1);}
	const char *p = ",";
	string raw_data;
	while (getline(in, raw_data))
	{
		int _col = 0;
		char *tmp = strtok((char *)raw_data.data(), p);
		while (tmp)
		{
			set[row][_col++] = atof(tmp);
			tmp = strtok(NULL, p);
		}
		row ++;
		col = _col;
	}

	in.close();
}

void initial()
{
	memset(train_set, 0, sizeof(train_set));
	memset(input_to_hide, 0, sizeof(input_to_hide));
	memset(hide_to_output, 0, sizeof(hide_to_output));
	memset(hide_output, 0, sizeof(hide_output));
	memset(output, 0, sizeof(output));
}

/*******归一化*******/
// void data_process(double set[][max_col], int row, int col)
// {
// 	double mx[col] = {0};	
// 	double mn[col] ;
// 	for (int i=0; i<col; i++)
// 		mn[i] = 999;
// 	for (int j=0; j<col-1; j++)
// 	{
// 		for (int i=0; i<row; i++)
// 		{
// 			mn[j] = set[i][j] < mn[j] ? set[i][j] : mn[j];
// 			mx[j] = set[i][j] > mx[j] ? set[i][j] : mx[j];
// 		}
// 	}
// 	for (int j=0; j<col-1; j++)
// 	{
// 		for (int i=0; i<row; i++)
// 		{
// 			train_set[i][j] = (train_set[i][j] - mn[j] ) / (mx[j] - mn[j]);
// 		}
// 	}
// }

void initialize_weight()
{
	//初始化输入层=》隐藏层权重
	for (int i=0; i<neuron; i++)
		for (int j=0; j<max_col; j++)
			input_to_hide[i][j] = (double)rand()/RAND_MAX; 
			// input_to_hide[i][j] = (rand()*2.0/RAND_MAX-1)/2; 

	//初始化隐藏层=》输出层权重
	for (int i=0; i<neuron; i++)
	{
		// hide_to_output[i] = (rand()*2.0/RAND_MAX-1)/2; 
		hide_to_output[i] = (double)rand()/RAND_MAX; 
		hide_b[i] = (double)rand()/RAND_MAX;
	}
	ouput_b = (double)rand()/RAND_MAX;

}


void forward_pass(int r, int col, double set[max_row][max_col])//col为特征个数(包结果) r为dir个样本
{
	//计算隐藏层输出
	double sum;
	for (int n=0; n<neuron; n++)//遍历隐藏层结点
	{
		sum = 0;
		for (int j=0; j<col-1; j++)				//遍历每个输入层结点
			sum += input_to_hide[n][j]*set[r][j];
		hide_output[n] = 1.0 / (1+exp(-sum));	//隐藏层输出
	}

	//输出层输出 y = h(x) = x
	sum = 0;
	for (int n=0; n<neuron; n++)
	{
		sum += hide_to_output[n] * hide_output[n];
	}
	output[r] = sum;//当前样本输出
}

void backward_pass(int r)
{
	//计算输出层error,out = in
	//output是预测的
	double output_err = train_set[r][train_col-1] - output[r];
	ouput_b += learn_rate*output_err;
	//计算隐藏层梯度	& 更新权重
	double hide_err[neuron] = {0};
	for (int i=0; i<neuron; i++)
	{	
		hide_err[i] = output_err*hide_to_output[i]*(hide_output[i])*(1-hide_output[i]);
		hide_to_output[i] += learn_rate * output_err * hide_output[i];//更新权重
	}
	//计算输入层->隐藏层权重 w[i][j] += learn_rate * errj * Oi
	for (int i=0; i<train_col-1; i++)
	{
		for (int j=0; j<neuron; j++)
		{
			input_to_hide[j][i] += learn_rate * hide_err[j] * train_set[r][i];
		}

	}
}

void train_nn()
{
	int cnt = 0;

	while (cnt < itr_times)
	{
		// double re = 0;	//for debug
		for (int i=0; i<train_row; i++)//遍历样本
		{
			forward_pass(i, train_col, train_set);
			// re += pow((train_set[i][train_col-1]-output[i]),2) / (double)train_row;	//for debug
			backward_pass(i); 
		}			
		// out << re << endl;	//for debug
		cnt ++;
	}
}

void test()
{
	int val_row = 0, val_col = 0;
	read_file("test.csv", val_row, val_col, val_set);
	ofstream vout("ans.txt");
	for (int i=0; i<val_row; i++)
	{
		double h_op[neuron]={0};
		double val_output=0;
		for (int n=0; n<neuron; n++)
		{
			int tmp = 0;
			for (int j=0; j<val_col-1; j++)
			{
				tmp += val_set[i][j]*input_to_hide[n][j];	
			}
			h_op[n] = 1.0/(1+exp(-tmp));
			val_output += hide_to_output[n]*h_op[n];
		}
		vout << val_output << endl;
	}
	vout.close();
}

int main()
{
	initial();
	initialize_weight();
	read_file("train0.csv", train_row, train_col, train_set);	//读取训练集
	train_nn();
	test();														//读测试集预测结果
}