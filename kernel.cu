#include<iostream>
#include<fstream>
#include <cuda_runtime.h>
#include "device_launch_parameters.h"  
#include<string>
#include<iomanip>
#include "device_functions.h"
#define Col 9067
#define Row 672

using namespace std;
bool isdigit(string a);
int toint(string a);
float tof(string a);
__global__ void cos_based(const float *pr, const int row, const int col, float *psim);//用来算相似度矩阵
void cal_p (float *R,const int row,const int col);
__global__ void cuda_p(float *pr,const int row,const int col, float *psim); //得到满的评分矩阵pr
bool InitCUDA();
void printDeviceProp(const cudaDeviceProp &prop);
void cpu_p(const float *R, const int row, int col, float *sim);


int main() {
	int idx[163950] = { 0 }; //从movie ID 映射到评分数组下标
	ifstream rate("ratings.csv");
	ifstream test("test.csv");
	ofstream result("result.csv");
	string value;
	int i, j;
	int count = 1;
	int flag = 0;
	int row, col;
	float rating;
	float *R = (float*)malloc(sizeof(float)*Col*Row);//储存用户评分矩阵
	int mid; //movie id
	float test_p;
	//读入数据，数据预处理
	getline(rate, value);
	while (rate.good())
	{
		getline(rate, value, ',');
		if (isdigit(value))
		{
			if (count == 1)
			{
				i = toint(value);//user id
				count++;
			}
			else if (count == 2)
			{
				mid = toint(value);//movie id
				if (idx[mid] == 0) //movie id 第一次出现 
				{
					j++;
					idx[mid] = j; //确定对应关系
				}
				//若是第二次出现则j不++
				count++;
			}
			else {
	  			R[i*Col + idx[mid]] = tof(value);
				count = 1;
				getline(rate, value);
			}
		}
	}

	bool state = InitCUDA();
	if (state)
		cout << "ok";
	


	cal_p(R, Row, Col);//运行结束后R为满的用户评分矩阵

	 /*读入测试数据输出结果*/
	result<< setw(8) << "userid" << setw(8) << "movieid" << setw(8) << "score" << endl;
	getline(test, value);
	while (test.good())
	{
		getline(test, value, ',');
		i = toint(value);
		getline(test, value);
		j = toint(value);
		result<<setw(8)<<i<<setw(8)<<j<<setw(8)<<R[i*Col+idx[j]]<<endl;
	
	}

		system("pause");
		return 0;
	
}


bool isdigit(string a)
{
	int len = a.length();
	if (a[0] >= '0'&&a[0] <= '9'&&len<7)
		return true;
	else
		return false;
}

int toint(string a)
{
	int result=0;
	int len = a.length();
	int i;
	for (i = 0; i < len; i++)
	{
		result += pow(10, len - i-1)*(a[i] - '0');
	}

	return result;
}

float tof(string a)
{
	float result;
	int len = a.length();
	int flag=0;
	int i;
	for (i = 0; i < len; i++)
	{
		if (a[i] == '.')
		{
			break;
		}
		result += pow(10, len - 1 - i)*(a[i] - '0');
	}
	result = result / 100;
	if(a[len-1]=='5')
	{
		result = result + 0.5;
	}
	return result;
}


__global__ void cos_based(const float *pr, const int row, const int col, float *psim)//用来算相似度矩阵
{
	
/*	float sim0;
	int i;
	const int tid = threadIdx.x; //第几个线程
	const int bid = blockIdx.x; //第几个块
	const int idx = bid * blockDim.x + tid;//总的第几个线程
//	const int idx = bid * THREAD_NUM + tid; //从 bid 和 tid 计算出这个 thread 应该计算的 row 和 column
	const int c1 = idx / col;
	const int c2 = idx % col;
	int t=0;
	float n1,n2,n3;

	if (c1<col && c2 < col)
	{

		n1 = 0;
		n2 = 0;
		n3 = 0;
		for (i = 1; i < row; i++)
		{
			n1 += pr[i*col + c1] * pr[i*col + c2];
			n2 += pr[i*col + c1] * pr[i*col + c1];
			n3+= pr[i*col + c2] * pr[i*col + c2];
		}
		sim0 = n1 / (sqrt(n2)*sqrt(n3));
		psim[c1*col + c2] = sim0;
		psim[c2*col + c1] = sim0;*/

	
    	int bx = blockIdx.x;
		int by = blockIdx.y;
		int tx = threadIdx.x;
		int ty = threadIdx.y;
		float n1, n2, n3;

		int i;
		int c1, c2;
		float sim0;

		
		// 由于矩阵B列的数目远大于GPU的grid size，所以需要以下的while循环。
	do {
		     n1 = 0; n2 = 0; n3 = 0;
			 c1 = by * blockDim.y + ty;
			 c2 = bx * blockDim.x + tx;

			if (c1<col&& c2 < col&&c1>0&&c2>0)
			{

				n1 = 0;
				n2 = 0;
				n3 = 0;
				for (i = 1; i < row; i++)
				{
					n1 += pr[i*col + c1] * pr[i*col + c2];
					n2 += pr[i*col + c1] * pr[i*col + c1];
					n3 += pr[i*col + c2] * pr[i*col + c2];
				}
				sim0 = n1 / (sqrt(n2)*sqrt(n3));
				psim[c1*col + c2] = sim0;
				psim[c2*col + c1] = sim0;

			}
			bx += gridDim.x;
			c2 = bx * blockDim.x + tx;
	} while (c2 < col);
	/*
	int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x
		+ blockIdx.x * blockDim.x + threadIdx.x;
	float n1, n2, n3, sim0;
	int i;
	int c1 = threadId / col;
	int c2 = threadId % col;
	if(threadId < col* col)
	{
			n1 = 0;
			n2 = 0;
			n3 = 0;
			for (i = 1; i < row; i++)
			{
				n1 += pr[i*col + c1] * pr[i*col + c2];
				n2 += pr[i*col + c1] * pr[i*col + c1];
				n3 += pr[i*col + c2] * pr[i*col + c2];
			}

			//sim0 = n1 / (sqrt(n2)*sqrt(n3));
			psim[c1*col + c2] = n1 / (sqrt(n2)*sqrt(n3));
			psim[c2*col + c1] = n1 / (sqrt(n2)*sqrt(n3));

	}
		*/
	
		
	
}
__global__ void cuda_p(float *pr, const int row, const int col, float *psim)
{
	/*float sim0;
	int i;
	float n1, n2, n3;
	const int tid = threadIdx.x; //第几个线程
	const int bid = blockIdx.x; //第几个块
	const int idx = bid * thread_num+ tid;//总的第几个线程
//	const int idx = bid * THREAD_NUM + tid; //从 bid 和 tid 计算出这个 thread 应该计算的 row 和 column
	const int r = idx / col;
	const int c = idx % col;
	if (r < row&&c < col) //r用户对c的评分
	{
    	n1 = 0; n2 = 0; n3 = 0;
		for (i = 1; i < col; i++)
		{
			if (pr[r*col + i] != 0)
			{
				n1 += pr[r*col + i] * psim[i*col + c]; //pr[r]行共col个元素与psim[c]列col个元素相乘累加
				n2 += psim[i*col+c];
			}
		}
		if (pr[r*col + c] == 0)
		{
			pr[r*col + c] = n1 / n2;
		}
		pr[r*col + c] = 1;
	}
	*/
	/*int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;
	float n1, n2, n3;
	n1 = 0; n2 = 0; n3 = 0;
	int i;
	int r,c;
	float sim0;
	// 由于矩阵B列的数目远大于GPU的grid size，所以需要以下的while循环。
	do {

		r = by * blockDim.y + ty;
		c = bx * blockDim.x + tx;

		if (r < row&&c < col) //r用户对c的评分
		{
			n1 = 0; n2 = 0; n3 = 0;
			for (i = 1; i < col; i++)
			{
				if (pr[r*col + i] != 0 && psim[i*col + c]>0.2)
				{
					n1 += pr[r*col + i] * psim[i*col + c]; //pr[r]行共col个元素与psim[c]列col个元素相乘累加
					n2 += psim[i*col + c];
				}
			}
			if (pr[r*col + c] == 0&&n2)
			{
				pr[r*col + c] = n1 / n2;
			}
		//	pr[r*col + c] = 1;
		}
		bx += gridDim.x;
		c = bx * blockDim.x + tx;
	} while (c < col);
		pr[5] = 2;*/
	int threadId = (blockIdx.y * blockDim.y + threadIdx.y) * gridDim.x * blockDim.x
		+ blockIdx.x * blockDim.x + threadIdx.x;
	float n1, n2, n3, sim0;
	int i;
	int r = threadId / col;
	int c = threadId % col;
	if (threadId < row* col&&r>0&&c>0)
	{

		n1 = 0;
		n2 = 0;
		n3 = 0;
		for (i = 1; i < col; i++)
		{
			if (pr[r*col + i] > 0 && psim[i*col + c]>0.1)
			{
				n1 += pr[r*col + i] * psim[i*col + c]; //pr[r]行共col个元素与psim[c]列col个元素相乘累加
				n2 += psim[i*col + c];
			}
		}
		if (pr[r*col + c] == 0 && n2)
		{
			pr[r*col + c] = n1 / n2;
		}

	}
}

void cal_p(float *R,const int row,const int col)//返回的R为满的用户评分矩阵
{
    cudaSetDevice(0);
	ofstream result("result.csv");
	//ofstream result3("result3.csv");

	//ofstream result2("result2.csv");
	float *psim; 
	float* sim = (float*)malloc(sizeof(float)*col*col);
	float* sim2= (float*)malloc(sizeof(float)*col*col);
	cudaMalloc((void**)&psim, sizeof(float)*col*col);
	float *pr;
	cudaMalloc((void**)&pr, sizeof(float)*col*row);
	cudaMemcpy(pr, R, sizeof(float)*col*row, cudaMemcpyHostToDevice);
	dim3 dimBlock(32, 32);
	dim3 dimGrid(255, 255);
	//dim3 dimGrid((col + dimBlock.x - 1) / dimBlock.x, (col + dimBlock.y - 1) / dimBlock.y,1);
	cos_based << <dimGrid,dimBlock>>>(pr, row, col, psim);
//	cos_based << <1000, 1024 >> >(pr, row, col, psim);
   // cuda_p << < 1000,1000 >> >(pr, row, col, psim);
    cudaMemcpy(sim, psim, sizeof(float)*col*col, cudaMemcpyDeviceToHost);
	//cudaMemcpy(R, pr, sizeof(float)*row*col, cudaMemcpyDeviceToHost);
	//cpu_p(R, Row, Col,sim2);

	//cout<<R[5];
	//getchar();
	for (int i = 1; i < col; i++)
		for (int j = 1; j < col; j++)	
			result<< sim[col*i + j]<< "  ";

	cout << endl << "***********";
/*	for (int i = 1; i < row; i++)
		for (int j = 1; j < col; j++)
		{
			if(R[col*i+j]>4)
			cout << R[col*i + j] << " ";
		}*/


	cudaFree(psim);
	cudaFree(pr);

}


bool InitCUDA()
{
	int count;
	cudaGetDeviceCount(&count);
	if (count == 0) {
		fprintf(stderr, "three is no device.\n");
		return false;
	}
	int i;
	for (i = 0; i<count; i++)
	{
		cudaDeviceProp prop;
		cudaGetDeviceProperties(&prop, i);
		printDeviceProp(prop);
		if (cudaGetDeviceProperties(&prop, i) == cudaSuccess) {
			if (prop.major >= 1) { break; }
		}
	}
	if (i == count) {
		fprintf(stderr, "There is no device supporting CUDA 1.x.\n");
		return false;
	}
	cudaSetDevice(i);
	return true;
}

void printDeviceProp(const cudaDeviceProp &prop)
{
	printf("Device Name : %s.\n", prop.name);
	printf("totalGlobalMem : %d.\n", prop.totalGlobalMem);
	printf("sharedMemPerBlock : %d.\n", prop.sharedMemPerBlock);
	printf("regsPerBlock : %d.\n", prop.regsPerBlock);
	printf("warpSize : %d.\n", prop.warpSize);
	printf("memPitch : %d.\n", prop.memPitch);
	printf("maxThreadsPerBlock : %d.\n", prop.maxThreadsPerBlock);
	printf("maxThreadsDim[0 - 2] : %d %d %d.\n", prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
	printf("maxGridSize[0 - 2] : %d %d %d.\n", prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
	printf("totalConstMem : %d.\n", prop.totalConstMem);
	printf("major.minor : %d.%d.\n", prop.major, prop.minor);
	printf("clockRate : %d.\n", prop.clockRate);
	printf("textureAlignment : %d.\n", prop.textureAlignment);
	printf("deviceOverlap : %d.\n", prop.deviceOverlap);
	printf("multiProcessorCount : %d.\n", prop.multiProcessorCount);
}
void cpu_p(const float *R, const int row, int col,float *sim)
{
	float *p = (float*)malloc(sizeof(float)*row*col);
	int c1, c2;
	float n1, n2, n3;
	int k;
	int i;
	int r;
	float sim0;
	memcpy(p, R, sizeof(float)*col*row);
	for (c1 = 1; c1 < col; c1++) {
		for (c2 = 1; c2 < col; c2++)
		{
			n1 = 0;
			n2 = 0;
			n3 = 0;
			for (i = 1; i < row; i++)
			{
				n1 += R[i*col + c1] * R[i*col + c2];
				n2 += R[i*col + c1] * R[i*col + c1];
				n3 += R[i*col + c2] * R[i*col + c2];
			}
			sim0 = n1 / (sqrt(n2)*sqrt(n3));
			sim[c1*col + c2] = sim0;
			sim[c2*col + c1] = sim0;

		}
	}
	/*for (int i = 1; i < col; i++)
		for (int j = 1; j < col; j++)
			cout << sim[col*i + j] << "  ";*/
/*	int c;
	for (r = 1; r < row; r++)
	
		for (c = 1; c < col; c++)
		{
			n1 = 0;
			n2 = 0;
			n3 = 0;
			for (i = 1; i <col; i++)
			{
				if (p[r*col + i] > 0 && sim[i*col + c]>0.1)
				{
					n1 += p[r*col + i] * sim[i*col + c]; //pr[r]行共col个元素与psim[c]列col个元素相乘累加
					n2 += sim[i*col + c];
				}
			}
			if (p[r*col + c] == 0 && n2)
			{
				p[r*col + c] = n1 / n2;
			}
	}
	/*	for (c1 = 1; c1 < col; c1++)
			for (c2 = 1; c2 < col; c2++)
				cout << sim[c1*col + c2] << " ";

*/
}