#include "stdio.h"

#define N 10

__global__ void add_single(int *a, int *b, int *c)
{
  for (int tid = 0; tid < N; ++tid)
  {
    c[tid] = a[tid] + b[tid];
  }
}

__global__ void add_multiblock(int *a, int *b, int *c)
{
  int tid = blockIdx.x;
  if (tid < N)
  {
    c[tid] = a[tid] + b[tid];
  }
}

__global__ void add_multithread(int *a, int *b, int *c)
{
  int tid = threadIdx.x;
  if (tid < N)
  {
    c[tid] = a[tid] + b[tid];
  }
}

int main (void)
{
  int a[N], b[N], c[N];

  for (int i = 0; i <N; ++i)
  {
    a[i] = -i;
    b[i] = i*i;
  }

  int *dev_a, *dev_b, *dev_c;

  cudaMalloc((void **)&dev_a, N*sizeof(int));
  cudaMalloc((void **)&dev_b, N*sizeof(int));
  cudaMalloc((void **)&dev_c, N*sizeof(int));

  cudaMemcpy(dev_a, a, N*sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(dev_b, b, N*sizeof(int), cudaMemcpyHostToDevice);

  printf("Results for <<<%d, %d>>>:\n", 1, 1);

  add_single<<<1, 1>>>(dev_a, dev_b, dev_c);

  cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i<N; ++i)
    printf("%d + %d = %d\n", a[i], b[i], c[i]);

  printf("Results for <<<%d, %d>>>:\n", N, 1);

  add_multiblock<<<N, 1>>>(dev_a, dev_b, dev_c);

  cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i<N; ++i)
    printf("%d + %d = %d\n", a[i], b[i], c[i]);

  printf("Results for <<<%d, %d>>>:\n", 1, N);

  add_multithread<<<1, N>>>(dev_a, dev_b, dev_c);

  cudaMemcpy(c, dev_c, N*sizeof(int), cudaMemcpyDeviceToHost);

  for (int i = 0; i<N; ++i)
    printf("%d + %d = %d\n", a[i], b[i], c[i]);

  cudaFree(dev_a);
  cudaFree(dev_b);
  cudaFree(dev_c);

  system("pause");
  return 0;
}

