# simple-cuda-gemm

compile with

nvcc gemm-study.cu -o gemm.o

run with

./gemm.o m n p

Where m, n and p are the dimensions of the matrices to be multiplied and are powers of 2 (values randomly generated, powers of two for simplicity).

Floats may overflow for very large dimensions.  No checks have been added to the code for this.  

std::cout will include the results of serial and CUDA matrix multiplication.

This implementation uses shared memory feature of cuda.  Each thread block fills its shared memory before 
performing dot products.  There is a little ASCI doodle showing how this works at the top of the file.  
