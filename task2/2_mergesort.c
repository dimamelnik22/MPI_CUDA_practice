#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int binarysearch(int a, int mass[], int n)
{
  int low, high, middle;
  low = 0;
  high = n - 1;
  while (low <= high)
  {
  middle = (low + high) / 2;
    if (a < mass[middle])
      high = middle - 1;
    else if (a > mass[middle])
      low = middle + 1;
    else
      return middle;
  }
  return low;
}

int* merge_sort(int *up, int *down, int left, int right)
{
  if (left == right)
  {
    down[left] = up[left];
    return down;
  }
  int middle = (left + right) / 2;

  int *l_buff = merge_sort(up, down, left, middle);
  int *r_buff = merge_sort(up, down, middle + 1, right);

  int *target = l_buff == up ? down : up;

  int l_cur = left, r_cur = middle + 1;
  for (int i = left; i <= right; ++i)
  {
    if (l_cur <= middle && r_cur <= right)
    {
      if (l_buff[l_cur] < r_buff[r_cur])
      {
        target[i] = l_buff[l_cur];
        l_cur++;
      }
      }
      else
      {
        target[i] = r_buff[r_cur];
        r_cur++;
      }
    }
    else if (l_cur <= middle)
    {
      target[i] = l_buff[l_cur];
      l_cur++;
    }
    else
    {
      target[i] = r_buff[r_cur];
      r_cur++;
    }
  }
  return target;
}

int main(int argc, char** argv) {

  MPI_Init(NULL, NULL);

  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

  MPI_Request request;

  double worktime = 0;
  for (int n = 1000; n <= 1000000; n *= 2)
  {
   for (int k = 0; k < 5; ++k)
   {
    int *arr = (int*)malloc(n*sizeof(int));
    int *buff = (int*)malloc(n*sizeof(int));
    double start = MPI_Wtime();
    int linopt = 0;
    if (world_rank == 0)
    {
      for (int i = 0; i < n; ++i)
      {
        arr[i] = rand() % 1000000;
      }
      if (world_size == 1 || n < world_size || n < 7)
      {
        linopt = 1;
      }
    }
    if (linopt == 1)
    {
      printf("Considered to use linear sort\n");
      arr = merge_sort(arr, buff, 0, n - 1);
      worktime = MPI_Wtime() - start;
      printf("Length is %d, time is %f\n", n, worktime);
      printf("\n");
    }
    else
    {
      if (world_rank == 0)
      {
        printf("Linear result:\n");
        int *arrlin = merge_sort(arr, buff, 0, n - 1);
        worktime = MPI_Wtime() - start;
        printf("Length is %d, time is %f\n", n, worktime);

        printf("Parallel result:\n");
        start = MPI_Wtime();
        int slice_size = n / world_size;
        for (int i = 1; i < world_size; ++i)
        {
          MPI_Send(&slice_size, 1, MPI_INT, i, n, MPI_COMM_WORLD);
          MPI_Send(&arr[slice_size * (i - 1)], slice_size, MPI_INT, i, n + 1, MPI_COMM_WORLD);
        }
        int last_size = n - slice_size * (world_size - 1);
        MPI_Isend(&last_size, 1, MPI_INT, 0, n, MPI_COMM_WORLD, &request);
        MPI_Isend(&arr[slice_size * (world_size - 1)], last_size, MPI_INT, 0, n + 1, MPI_COMM_WORLD, &request);
      }
      int slice_size;
      MPI_Recv(&slice_size, 1, MPI_INT, 0, n, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int *target = (int*)malloc(slice_size*sizeof(int));
      MPI_Recv(&target[0], slice_size, MPI_INT, 0, n + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      target = merge_sort(target, buff, 0, slice_size - 1);
      if (world_rank == 0)
      {
        int mids = slice_size / world_size;
        int *pivots = (int*)malloc((world_size - 1) * sizeof(int));
        for (int i = 1; i < world_size; ++i)
        {
          pivots[i - 1] = target[i * mids];
        }
        for (int i = 0; i < world_size; ++i)
        {
          MPI_Isend(&pivots[0], world_size - 1, MPI_INT, i, n + 2, MPI_COMM_WORLD, &request);
        }
      }
      int *pivots = (int*)malloc((world_size - 1) * sizeof(int));
      MPI_Recv(&pivots[0], world_size - 1, MPI_INT, 0, n + 2, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int left = 0;
      int size = 0;
      for (int i = 0; i < world_size - 1; ++i)
      {
        int right = binarysearch(pivots[i], target, slice_size);
        size = right - left;
        MPI_Isend(&size, 1, MPI_INT, i, n + 3, MPI_COMM_WORLD, &request);
        MPI_Isend(&target[left], size, MPI_INT, i, n + 4, MPI_COMM_WORLD, &request);
        left = right;
      }
      size = slice_size - left;
      MPI_Isend(&size, 1, MPI_INT, world_size - 1, n + 3, MPI_COMM_WORLD, &request);
      MPI_Isend(&target[left], size, MPI_INT, world_size - 1, n + 4, MPI_COMM_WORLD, &request);
      int arr_size = 0;
      for (int i = 0; i < world_size; ++i)
      {
        int getsize = 0;
        MPI_Status status;
        MPI_Recv(&getsize, 1, MPI_INT, MPI_ANY_SOURCE, n + 3, MPI_COMM_WORLD, &status);
        MPI_Recv(&arr[arr_size], getsize, MPI_INT, status.MPI_SOURCE, n + 4, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        arr_size += getsize;
      }
      int *res = merge_sort(arr, buff, 0, arr_size - 1);
      MPI_Isend(&arr_size, 1, MPI_INT, 0, n + 5, MPI_COMM_WORLD, &request);
      MPI_Isend(&res[0], arr_size, MPI_INT, 0, n + 6, MPI_COMM_WORLD, &request);
      if (world_rank == 0)
      {
        int res_size = 0;
        for (int i = 0; i < world_size; ++i)
        {
          int getsize = 0;
          MPI_Status status;
          MPI_Recv(&getsize, 1, MPI_INT, i, n + 5, MPI_COMM_WORLD, &status);
          MPI_Recv(&arr[res_size], getsize, MPI_INT, status.MPI_SOURCE, n + 6, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          res_size += getsize;
        }
        worktime = MPI_Wtime() - start;
        printf("Pivots are:\n");
        int mids = n / world_size;
        for (int i = 0; i < world_size; ++i)
        {
          printf("%d ",arr[i * mids]);
        }
        printf("\n");

        printf("Length is %d, time is %f\n", n, worktime);
        printf("\n");
      }
    }
   }
  }

  MPI_Finalize();
}

