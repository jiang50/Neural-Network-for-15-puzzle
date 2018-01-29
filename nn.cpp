//csce 420 project 
//Yucheng Jiang 525006053
#include <iostream>
#include <fstream>
#include <cstdint>
#include <bitset>
#include <vector>
#include <cstdlib>
#include <math.h>
#include <unordered_map>
#include <sstream>
#include <numeric>
using namespace std;


struct Node{  //node in the neural net
    vector <double> weights;//weight for each node in previous layer
    double output, error;
    Node(){
        output = error= 0.0;
    }
};
typedef vector<Node> Layer;  

class Net{  
public:  
    Net(const vector<int>& shape);//shape of neural net, the number of nodes in each layer
    void front(const vector<double>& input);// calculate the output for each node in the neural net
    static double sig(const double& x);//sigmoid function
	static double hyper(const double& x);//hyperbolic function
    void backprop(const vector<double>& target);//back propagation
    Layer getResult();
	double alpha=0.1;//learning rate
private:  
    vector<Layer> net;
};

Net::Net(const vector<int>& shape){  
    int n = shape.size();
    for (int i = 0; i < n; i++){
        net.push_back(Layer());
        for (int j = 0; j <= shape[i]; j++){
            net.back().push_back(Node());
            if (i > 0){
                net[i][j].weights = vector <double> (shape[i - 1]+1);
                for (auto& val : net[i][j].weights){
                    val = rand()*(rand()%2?1:-1)*1.0/RAND_MAX;
                }
            }
        }
    }
    for (int i = 0; i < n; i++){
        net[i].back().output = 1;
    }
}

double Net::sig(const double& x){  
    return 1.0 / (1.0 + exp(-x));
}

double Net::hyper(const double& x){
	return (exp(x)-exp(-x))/(exp(x)+exp(-x));
}

void Net::front(const vector<double>& input){  
    for (int i = 0; i < net[0].size() - 1; i++){
        net[0][i].output = input[i];
    }
    int n = net.size();
    for (int i = 1; i < n; i++){
        for (int j = 0; j < net[i].size() - 1; j++){
           // double& res = net[i][j].output;
            double res = 0.0;
            for (int k = 0; k < net[i - 1].size(); k++){
                res += net[i - 1][k].output * net[i][j].weights[k];
            }
			if(i==n-1)res=sig(res);//last layer sigmoid
            else res = hyper(res);//tanh
			net[i][j].output=res;
        }
    }
}

void Net::backprop(const vector<double>& target){  
    for (int i = 0; i < target.size(); i++){
        Node& node = net.back()[i];
        node.error = target[i] - node.output; 
    }
    for (int i = net.size() - 2; i > 0; i--){
        Layer& cur_layer = net[i];
        Layer& nex_layer = net[i + 1];
        Layer& prv_layer = net[i - 1];
        for (int j = 0; j < cur_layer.size()-1; j++){
            Node& node = cur_layer[j];
            node.error = 0;
            for (int k = 0; k < nex_layer.size() - 1; k++){
                node.error += nex_layer[k].error * nex_layer[k].weights[j];
            }
        }
    }
    for (int i = 1; i < net.size(); i++){
        Layer& layer = net[i];
        Layer& prev = net[i - 1];
        for (int j = 0; j < layer.size(); j++){
            Node& node = layer[j];
			double d=0.0;
			if(i==net.size()-1)
				d = node.output* (1 - node.output);//tanh
			else d= 1- (node.output)*(node.output);//logistic sigmoid
            for (int k = 0; k < node.weights.size(); k++){
                node.weights[k] += (alpha * node.error) * d * prev[k].output;
            }
        }
    }
}

 
Layer Net::getResult() {  
    return net.back();
}

  

  
int get_manh(vector<int> state)  //calculate manhattan distance
{  
    int count = 0, place[16];  
	for(int i=0;i<16;i++){
		if(state[i]!=0)
		count+=(abs((state[i]-1)%4-i%4)+abs((state[i]-1)/4-i/4));
		
	}
    return count;  
}  

vector<double> get_input(vector<int> v){//get the format of input
	vector<double> res(240);
	int index=0;
	for(int i=1;i<v.size();i++){
		res[index+15-v[i]]=1.0;
		index+=16;
	}
	return res;
}

vector<double> get_output(int n){
	vector<double> res(29);
	res[28-n]=1.0;
	return res;
}

int parse_output(vector<Node> v){//get the number of moves left
	double ma=0.0;
	int index=0;
	for(int i=0;i<v.size()-1;i++){
		if(v[i].output>ma){
			index=i;
			ma=v[i].output;
		}
	}
	return 28-index;
}

int main(){
	vector<int> shape{240,120,60,29};
	Net net(shape);
	unordered_map<uint64_t,int> m;
	cout<<"read data\n";
	
	for(int i=28;i>=0;i--){//read data from file and save in an unordered_map
		fstream myFile;
		string dir="data/"+to_string(i)+".bin";
		myFile.open (dir, ios::in | ios::binary);
		if(!myFile)cout<<"cannot open file!\n";
		uint64_t n;
		int count=0;
		while(myFile.read((char*)&n,sizeof(n))&&count<1000){
			m[n]=i;
			count++;
		}
		myFile.close();
	}
	
	cout<<"training neural network\n";
	for(int iter=0;iter<10;iter++){//train the neural network
		for ( auto it = m.begin(); it != m.end(); ++it ){
			bitset<64> bs=it->first;
			vector<int> data(16);
			int sum=0;
			for(int i=59;i>=0;i-=4){
				int n=bs[i]*8+bs[i-1]*4+bs[i-2]*2+bs[i-3];
				data[15-i/4]=n;
				sum+=n;
				data[0]=120-sum;
			}
			vector<double> input=get_input(data);
			vector<double> output=get_output(it->second);
			net.front(input);
			net.backprop(output);
		}
		cout<<"iteration "+to_string(iter)<<endl;
		net.alpha=net.alpha/2;
    
	}

	for(int i=0;i<29;i++){//check the result
		cout<<"Test on "+to_string(i)+" state:"<<endl;
		fstream myFile;
		string dir="data/"+to_string(i)+".bin";
		myFile.open (dir, ios::in | ios::binary);
		if(!myFile)cout<<"cannot open file!\n";
		uint64_t n;
		int count=0;
		int right=0;
		int manh_right=0;
		int max_shortfall=0;
		int count_res=1;
		while(myFile.read((char*)&n,sizeof(n))&&count<1000){
			bitset<64> bs=n;
			vector<int> data(16);
			int sum=0;
			for(int i=59;i>=0;i-=4){
				int n=bs[i]*8+bs[i-1]*4+bs[i-2]*2+bs[i-3];
				data[15-i/4]=n;
				sum+=n;
				data[0]=120-sum;
			}
			vector<double> input=get_input(data);
			int man=get_manh(data);			
			net.front(input);			
			int result=parse_output(net.getResult());
			if(result==i)right++;
			if(result==man)manh_right++;
			if(man<result)max_shortfall=max(max_shortfall,result-man);
			count++;
		}
		cout<<"Accuracy by testing on training data: "<<(double)right/count<<endl;
		cout<<"Accuracy of manhattan distance: "<<(double)manh_right/count<<endl;
		cout<<"max shortfall = "<<max_shortfall<<endl;
	}
	
	return 0;
}
