#include "stdio.h"

#define N   128

__global__ void add(int *A, int *B, int *C)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i < N && j < N)
    {
        C[i * N + j] = A[i * N + j] + B[i * N + j];
    }
}

int main( void )
{
    int a[N * N], b[N * N], c[N * N];
    int *dev_a, *dev_b, *dev_c;

    for (int i = 0; i < N * N; ++i) 
    {
        a[i] = -i;
        b[i] = i * i;
    }

    cudaMalloc((void**)&dev_a, N * N * sizeof(int));
    cudaMalloc((void**)&dev_b, N * N * sizeof(int));
    cudaMalloc((void**)&dev_c, N * N * sizeof(int));

    cudaMemcpy(dev_a, a, N * N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, N * N * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(N / 16, N / 16);
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, 0);

    add<<<numBlocks, threadsPerBlock>>>(dev_a, dev_b, dev_c);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float worktime;

    cudaEventElapsedTime(&worktime, start, stop);

    printf("Time = %f ms \n", worktime);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaMemcpy(c, dev_c, N * N * sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d + %d = %d\n", a[0], b[0], c[0]);
    printf("%d + %d = %d\n", a[N * N - 1], b[N * N - 1], c[N * N - 1]);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);
    return 0;
}
