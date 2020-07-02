#include "stdio.h"

#define BLOCK_SIZE 4
#define N   32

__global__ void prod(int *A, int *B, int *C)
{
    int blockRow = blockIdx.y;
    int blockCol = blockIdx.x;

    int Cvalue = 0;

    int row = threadIdx.y;
    int col = threadIdx.x;

    for (int m = 0; m < (N / BLOCK_SIZE); ++m) 
    {
        __shared__ int slice_A[BLOCK_SIZE * BLOCK_SIZE];
        __shared__ int slice_B[BLOCK_SIZE * BLOCK_SIZE];
        
        slice_A[row * BLOCK_SIZE + col] = A[(BLOCK_SIZE * blockRow + row) * N + m * BLOCK_SIZE + col];
        slice_B[row * BLOCK_SIZE + col] = B[(BLOCK_SIZE * m + row) * N + blockCol * BLOCK_SIZE + col];

        __syncthreads();
        for (int e = 0; e < BLOCK_SIZE; ++e)
            Cvalue += slice_A[row * BLOCK_SIZE + e] * slice_B[e * BLOCK_SIZE + col];
        __syncthreads();
    }
    C[(BLOCK_SIZE * blockRow + row) * N + BLOCK_SIZE * blockCol + col] = Cvalue;
}

int main( void )
{

  int a[N * N], b[N * N], c[N * N];
    int *dev_a, *dev_b, *dev_c;

    for (int i = 0; i < N * N; ++i) 
    {
        a[i] = 1;
        b[i] = 1;
    }

    cudaMalloc((void**)&dev_a, N * N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * N * sizeof(int));

    cudaMemcpy(dev_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    dim3 dimBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 dimGrid(N / dimBlock.x, N / dimBlock.y);
    prod<<<dimGrid, dimBlock>>>(dev_a, dev_b, dev_c);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float worktime;

    cudaEventElapsedTime(&worktime, start, stop);

    printf("Time = %f ms \n", worktime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(c, dev_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("c[0][0] = %d\n", c[0]);
    printf("c[N-1][N-1] = %d\n", c[N * N - 1]);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}
