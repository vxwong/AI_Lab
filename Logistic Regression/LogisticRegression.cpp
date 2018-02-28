#include "info.h"
#define learning_rate 1	//学习率
#define itr_times 100	//迭代次数

class Training{		//训练过程
	public:
		Info train_info;
		double *w;							//权重
		void init_w();						//初始化权重
		double * find_w();					//求出最终权重
		void update_w(int, int, double **);	//更新权重
};

class Testing{		//测试过程
	public:
		Info test_info;
		int predict(int, double *);			//预测结果
};

void Training::init_w(){
	w = new double [train_info.col];
	for (int j=0; j<train_info.col; ++j)
		w[j] = 0;
}

void Training::update_w(int row, int col, double **set){
	//每个样例的权重分数
	double s[row] = {0};
	for (int i=0; i<row; i++)
		for (int j=0; j<col-1; j++)
			s[i] += w[j]*set[i][j];

	//每一维的梯度计算
	double cost[col] = {0};
	for (int j=0; j<col-1; j++)
		for (int i=0; i<row; i++)
			cost[j] += ((1/(1+exp(-s[i])))-set[i][col-1])*set[i][j];

	//更新每一维度权重
	for (int j=0; j<col-1; j++)
		w[j] -= learning_rate * cost[j];
}

double * Training::find_w(){
	int cnt = 0;
	init_w();
	while (cnt < itr_times){	
		update_w(train_info.row, train_info.col, train_info.set);
		cnt++;
	}
	return w;
}


int Testing::predict(int row, double *w)
{
	double s = 0;
	for (int j=0; j<test_info.col-1; j++)
		s += w[j]*test_info.set[row][j];
	if (1/(1+exp(-s)) > 0.5) return 1;	
	else return 0;
}


int main(){
	Training train_case;
	Testing test_case;	
	train_case.train_info = read_file("train.csv");		//输入训练集
	test_case.test_info = read_file("test.csv");		//输入测试集
	double * w = train_case.find_w();					//训练ing
	//训练完毕,输出答案
	ofstream out ("ans.txt");							//输出答案
	for (int i=0; i<test_case.test_info.row; i++)
		out << test_case.predict(i,w) << endl;
	out.close();
	return 0;
}
