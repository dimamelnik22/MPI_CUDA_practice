#include <stdio.h>

const int N = 2048;

__global__ void add_complex(int *a , int *b , int *c)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    while (tid < N)
    {          
        c[tid] = a[tid] + b[tid];
        tid += blockDim.x * gridDim.x;
    }
}

int main (void)
{
    int a[N], b[N], c[N];
    for (int i = 0; i < N; ++i)
    {
        a[i] = -i;
        b[i] = i*i; 
    }

    int *dev_a, *dev_b, *dev_c;

    cudaMalloc((void **)&dev_a, N*sizeof(int));
    cudaMalloc((void **)&dev_b, N*sizeof(int));
    cudaMalloc((void **)&dev_c, N*sizeof(int));

    cudaMemcpy (dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy (dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    printf("Results for <<<%d, %d>>>:\n", (N+127)/128, 128);

    cudaEventRecord(start, 0);

    add_complex<<<(N+127)/128, 128>>>(dev_a, dev_b, dev_c);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    float worktime;
    cudaEventElapsedTime(&worktime, start, stop);

    cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

    printf("%d + %d = %d\n", a[0], b[0], c[0]);
    printf("%d + %d = %d\n", a[N - 1], b[N - 1], c[N - 1]);

    printf("Time = %f ms \n", worktime);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(dev_a);
    cudaFree(dev_b);
    cudaFree(dev_c);

    system("pause");
    return 0 ;
}