#include <bits/stdc++.h>
using namespace std;

#define max_row 800
#define max_col 11

int train_set[max_row][max_col];
int val_set[max_row][max_col];
int train_row, train_col, val_row, val_col;

struct Node
{
	int root;           //根，为了打印好看点
	int attr;			//属性
	int value;			//标签
	vector <Node *> childs;
	Node()
	{
		root = -10086;//叶子结点root是-10086
		attr = -10086;//当前结点和父节点之间的attr
		value = -10086;//非叶子结点的value是-10086
	}
};
Node *root;

set<int> Attr_set;
set<int> s;

void read_file (string f, int &row, int &col, int set[max_row][max_col])
{
	fstream in(f.c_str());
	if (!in.is_open()) {cout << f << " open failed\n"; exit(1);}
	const char *p = ",?";
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
	// for (int i=0; i<row; i++) //预处理年龄
	// 	set[i][0] /= 10; 	
	in.close();
}

void initial (int row, int col, int set[max_row][max_col])
{
	for (int i=0; i<col-1; i++)
		Attr_set.insert(i);

	for (int i=0; i<row; i++)
		s.insert(i);
}

int most (set<int> s)//返回标签数最多的
{
	int t = 0, f = 0;
	for (set<int> :: iterator i=s.begin(); i!=s.end(); i++)
	{
		if (train_set[*i][train_col-1] == 1) t++;
		else f++;
	}
	return t >= f ? 1 : -1;
}

int same_label (set<int> s)//全部标签一样
{
	bool allt = 1, allf = 1;
	for (std::set<int>::iterator it=s.begin(); it!=s.end(); it++)
	{
		if (train_set[*it][train_col-1] == 1) allf = 0;
		else allt = 0;
		if (!allt && !allf) return 0; //标签其实还是有正有负
	}
	if (allt) return 1;
	return -1;
}

double cal_exp_etp(set<int> s, set<int> Attr_set)//计算经验熵
{
	double t = 0, f = 0, exp_etp = 0;
	// for (int i=0; i<row; i++)
	for (set<int>::iterator i=s.begin(); i!=s.end(); i++)
	{
		if (train_set[*i][train_col-1] == 1) t++;
		else f++;
	}
	exp_etp = -(t/s.size())*log(t/s.size())/log(2) - (f/s.size())*log(f/s.size())/log(2);
	return exp_etp;
}

int chose_root (set<int> s, set<int> Attr_set) //s<int>存attr不同取值的行，set<attr>还存在的attr
{
	double gain[10] = {0}; //信息增益
	double gain_ratio[10] = {0}; //信息增益率
	double gini[10] = {0}; //基尼系数
	double splitinfo[10] = {0}; 
	double conditial_etp[10] = {0}; //条件熵
	map <int, int> tmp;	//<特征下的某个值,该值出现次数>
	double exp_etp = cal_exp_etp(s, Attr_set); //经验熵 
	// cout  << "exp_etp = " << exp_etp << endl;
	map <int, int> :: iterator it;
	double mx = -1, mx2 = 0, gini_mn = 999, gini_pos = 0; 
	
	for (set <int>:: iterator set_it=Attr_set.begin(); set_it!=Attr_set.end(); set_it++)//遍历可用attr
	{
		//计算某特征下不同值分别出现的总次数
		for (set<int>::iterator i=s.begin(); i!=s.end(); i++)
		{
			it = tmp.find(train_set[*i][*set_it]);
			if (it == tmp.end()) tmp.insert(pair<int, int>(train_set[*i][*set_it], 1));
			else it->second ++;			
		}
		//计算每个特征下的条件熵，splitInfo，gini系数
		//map <int, int> tmp;	//<特征下的某个值,该值出现次数>
		double t, f;
		for (it = tmp.begin(); it != tmp.end(); it++) //遍历该属性的不同取值
		{
			t = f = 0;
			for (set<int>::iterator i=s.begin(); i!=s.end(); i++) //*i行数
				if (train_set[*i][*set_it] == it->first)          //*set_it：特征
				{
					if (train_set[*i][train_col-1] == 1)	t ++; 
					else f++;
				}
			double k = 0;
			if (t) k -= (t/it->second)*log(t/it->second)/log(2);
			if (f) k -= (f/it->second)*log(f/it->second)/log(2);
			conditial_etp[*set_it] += (double)it->second/s.size() * k; //计算条件熵
			splitinfo[*set_it] -= (double)it->second /s.size() * log ((double)it->second / s.size())/log(2);
			gini[*set_it] += (double)it->second / s.size() * (1 - pow((double)t/it->second, 2) - pow((double)f/it->second, 2));//选小的
		}
		tmp.clear();
		gain[*set_it] = exp_etp - conditial_etp[*set_it];   //信息增益
		if (splitinfo[*set_it] == 0) continue;	//该特征熵为0，只有一个特征
		gain_ratio[*set_it] = gain[*set_it] / splitinfo[*set_it]; //信息增益率
		if (mx < gain[*set_it]){mx = gain[*set_it]; mx2 = *set_it; } //*set_it列为决策点 ID3决策
		// if (gini_mn > gini[*set_it]){gini_mn = gini[*set_it]; gini_pos = *set_it;} //gini决策 
		// if (mx < gain_ratio[*set_it]){mx = gain_ratio[*set_it]; mx2 = *set_it; }//C4.5决策
	}			
	cout << "ID3: " << mx << endl;
	return mx2;
	// return gini_pos; //gini决策
}

map<int, set<int> > divide_data(int rt, set<int> s)//根据根节点分子数据集
{
	//比如说根据年龄（30，40，50）划分当前数据，rt则为年龄这个特征的列数
	//那么就以30为键值，把所有年龄为30的训练集行数放入s1中,s1放入m（subsets）中
	map<int, set<int> > m; //subsets
	map<int, set<int> >::iterator i;
	for (set<int>::iterator it=s.begin(); it!=s.end(); it++)//行
	{
		i = m.find(train_set[*it][rt]); //find属性rt下的某个值
		if (i == m.end())
		{
			set<int> ss;
			ss.insert(*it);
			// if (rt == 0) cout << train_set[*it][rt] << endl;
			m.insert(pair<int,set<int> > (train_set[*it][rt], ss));
		}
		else
			i->second.insert(*it);
	}
	return m;
}

void create_tree(Node *p, set<int> s, set<int> attr)
{
	// if (p == NULL) p = new Node();
	if (same_label(s) == 1 || same_label(s) == -1) 
	{
		p->value = same_label(s); 
		return ;
	}
	if (attr.size() == 0) 
	{
		p->value = most(s);
		return ;
	}

	int next_root = chose_root(s,attr); //选下一个决策特征
	p->root = next_root;				//把当前结点root改为选出的特征
	// cout << "next_root = " << next_root << endl;	//for debug
	map<int, set<int> >m = divide_data(next_root, s);	//根据选出的点划分数据集
	attr.erase(next_root);				//把选出的决策点从可用决策点中移走

	// 剩下的几个属性取值都一样，没办法再分	
	if (m.size() == 1)
	{
		// new_node->value = most(i->second);
		p->root = -10086;
		map<int, set<int> >::iterator it = m.begin();
		p->value = most(it->second);
	}	
	else
		for (map<int, set<int> >::iterator i = m.begin(); i!=m.end(); i++)//<某特征的值，该值对应的下一个dataset>
		{
			Node *new_node = new Node();
			new_node->attr = i->first;	//把新结点attr设为遍历到的该特征的值
			create_tree(new_node, i->second, attr); //往下建树
			p->childs.push_back(new_node); //把新结点加到当前结点的孩子中
		}
	attr.insert(next_root); //把刚刚移走的特征加回
}

void print_tree(Node *p, int depth)
{
	for (int i=0; i<depth; i++)	cout << "\t";
	if (p->attr != -10086 )
	{
		cout << "attr:" <<p->attr << endl;
		for (int i=0; i<depth+1; i++)	cout << '\t';
	} 
	if(p->root != -10086) cout <<"root:"<< p->root << endl;
	if(p->value != -10086)cout << "value:" << p->value << endl;
	for (vector<Node*> ::iterator it=p->childs.begin(); it!=p->childs.end(); it++){
		print_tree(*it, depth+1);
	}
}

int predict (Node *p, int test[], int father)
{
	if (p->value != -10086)
	{
		int ans = p->value;
		return ans; //叶子结点return
		 cout << p->value << "  return here\n";
	}
	int at;
	if (p->root != -10086)	at = p->root;//当前是根节点，找匹配的属性列it->root
	bool is_find = 0;
	for (vector<Node *>::iterator it=p->childs.begin(); it!=p->childs.end(); it++)
	{
		Node *tmp = *it;
		if (tmp->attr == test[at])//找得到匹配的属性
		{
			is_find = 1;
			return predict(*it, test, at);
		}
	}
	int t = 0, f = 0;	
	if (!is_find) //没有匹配的,找父节点
		for(int i=0; i<train_row; i++)
		{
			if (father != -1)
			{
				if (train_set[i][father] == p->attr) 
				{
					if (train_set[i][train_col-1] == 1) t++;
					else f++;
				}					
			}
			else
			{
				if (train_set[i][train_col-1] == 1)  t++;
				else f++;
			}
		}
				
	return t >= f? 1 : -1;		
}

int main () 
{
		train_row = train_col = 0;
		read_file("train.csv", train_row, train_col, train_set);	//读取训练集
		initial(train_row, train_col, train_set);
		root = new Node();
		create_tree(root, s, Attr_set);
		print_tree(root,0);
		read_file("val.csv", val_row, val_col, val_set);			//读取验证集
		int t = 0;
		for (int i=0; i<val_row; i++)
		{
			int test[val_col];
			for (int j=0; j<val_col-1; j++)
				test[j] = val_set[i][j];
			cout << predict(root,test,-1) << endl;
		}
		// cout << (double)t / val_row << endl;		
	return 0;
}