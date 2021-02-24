#include <iostream>
#include <cuda_runtime.h>
#include <chrono>
#include <stdlib.h>
#include <stdio.h>

#define BLOCK 16

/*

0 0 X X    0 0 0 0
0 0 X X    0 0 0 0 
0 0 0 0    X X 0 0 
0 0 0 0    X X 0 0

*/

struct Matrix{
	int width;
	int height;
	float* elements;

};

__global__ void blockedMatMulKernel(const Matrix A, const Matrix B, Matrix C){
	int row = threadIdx.y;
	int col = threadIdx.x;
	if (blockDim.y * blockIdx.y + row >= A.height || blockDim.x * blockIdx.x + col >= B.width )
		return;
	float C_rc = 0; 
	for (int e = 0; e < A.width/blockDim.x; ++e){
		__shared__ float A_block [BLOCK][BLOCK];
		__shared__ float B_block [BLOCK][BLOCK];

		A_block[row][col] = A.elements[A.width * (blockDim.y * blockIdx.y + row) + blockDim.x * e + col];
		B_block[row][col] = B.elements[B.width * (e * blockDim.x + row) + blockIdx.x * blockDim.x + col];
		__syncthreads();

		for (int i = 0; i < BLOCK; ++i)
			C_rc += A_block[row][i] * B_block[i][col];

		__syncthreads(); // make sure that C is calculated before you go and overwrite A & B in next iter

	}

	C.elements[B.width * (blockDim.y * blockIdx.y + row) + col + blockDim.x * blockIdx.x] = C_rc;
	__syncthreads();
}

void matMulBlocked(Matrix A, Matrix B, Matrix C){
	Matrix d_A, d_B, d_C;

	cudaMalloc(&(d_A.elements), sizeof(float) * A.width * A.height);
	cudaMalloc(&(d_B.elements), sizeof(float) * B.width * B.height);
	cudaMalloc(&(d_C.elements), sizeof(float) * C.width * C.height);

	cudaMemcpy(d_A.elements, A.elements, sizeof(float) * A.width* A.height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B.elements, B.elements, sizeof(float) * B.width* B.height, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C.elements, C.elements, sizeof(float) * C.width* C.height, cudaMemcpyHostToDevice);

	d_A.width = A.width;
	d_A.height = A.height;
	d_B.width = B.width;
	d_B.height = B.height;
	d_C.width = C.width;
	d_C.height = C.height;

	dim3 block (BLOCK,BLOCK);
	dim3 numBlocks ((A.height + BLOCK - 1)/BLOCK, (B.width + BLOCK - 1)/BLOCK);
	blockedMatMulKernel<<<numBlocks, block>>>(d_A, d_B, d_C);
	cudaDeviceSynchronize();

	cudaMemcpy(C.elements, d_C.elements, sizeof(float) * C.width* C.height, cudaMemcpyDeviceToHost);

	for (int i = 0; i < A.height; ++i){
		for (int j = 0; j < B.width; ++j){
			std::cout<<C.elements[i*B.width + j]<<" ";
		}
		std::cout<<std::endl;
	}

	cudaFree(d_A.elements);
	cudaFree(d_B.elements);
	cudaFree(d_C.elements);
}

void serialMatMul(int n, int p, int m, float* A, float* B, float* C){
	for (int i = 0; i < m; ++i){
		for (int j = 0; j < n; ++j){
			float sum = 0;
			for (int k = 0; k < p; ++k){
				sum += A[i*p + k] * B[k*n + j];
			}
			C[i * n + j] = sum;
		}
	}
}

int main(int argc, char* argv[]){
	if (argc < 4){
		std::cout<<"please include 3 integers for matrix dimensions"<<std::endl;
		exit(1);
	}

	int m = atoi(argv[1]);
	int p = atoi(argv[2]);
	int n = atoi(argv[3]);

	Matrix A, B, C;
	A.height = m;
	C.height = m;
	B.height = p;
	B.width = n;
	A.width = p;
	C.width = n;

	A.elements = new float[m*p];
	B.elements = new float[p*n];
	C.elements = new float[m*n];

	for (int i = 0; i < m; ++i)
		for (int j = 0; j < p; ++j)
			A.elements[i*p + j] = (float)rand()/(float)(RAND_MAX);

	for (int i = 0; i < p; ++i)
		for (int j = 0; j < n; ++j)		
			B.elements[i*n + j] = (float)rand()/(float)(RAND_MAX);

	auto s = std::chrono::high_resolution_clock::now();
    serialMatMul(n, p, m, A.elements, B.elements, C.elements);
    auto end = std::chrono::high_resolution_clock::now();
    auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - s);
    std::cout << "time: " << diff.count() << std::endl;
	for (int i = 0; i < m; ++i){
		for (int j = 0; j < n; ++j){
			std::cout<<C.elements[i*n + j]<<" ";
		}
		std::cout<<std::endl;
	}
	std::cout<<std::endl;

	cudaEvent_t start;
	cudaEvent_t stop;
	float time;

	cudaEventCreate(&start);
    cudaEventCreate(&stop);
	cudaEventRecord(start);

	matMulBlocked(A, B, C);

	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&time, start, stop);

    std::cout<<"cuda time="<< time << std::endl;



	free(A.elements);
	free(B.elements);
	free(C.elements);

	return 0;
}