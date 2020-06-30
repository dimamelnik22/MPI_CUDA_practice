#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>


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

  double worktime = 0;
  for (int n = 4; n <= 20; n *= 2)
  {

    if (world_rank == 0)
    {
      int *arr = (int*)malloc(n*sizeof(int));
      int *buff = (int*)malloc(n*sizeof(int));
      double start = MPI_Wtime();

      for (int i = 0; i < n; ++i)
      {
        arr[i] = rand() % 100;
      }
      if (world_size == 1)
      {
        arr = merge_sort(arr, buff, 0, n - 1);
      }
      else
      {
        int slice_size = n / world_size;
        for (int i = 1; i < world_size; ++i)
        {
          MPI_Send(&slice_size, 1, MPI_INT, i, n, MPI_COMM_WORLD);
          MPI_Send(&arr[slice_size * (i - 1)], slice_size, MPI_INT, i, n + 1, MPI_COMM_WORLD);
        }
        int *res = merge_sort(arr, buff, slice_size * (world_size - 1), n - 1);
        for (int i = 1; i < world_size; ++i)
        {
          int *tmp = (int*)malloc(slice_size * sizeof(int));
          MPI_Status status;
          MPI_Recv(&tmp[0], slice_size, MPI_INT, MPI_ANY_SOURCE, n + 1, MPI_COMM_WORLD, &status);
          for (int k = 0; k < slice_size; ++k)
          {
            arr[slice_size * (status.MPI_SOURCE - 1) + k] = tmp[k];
          }
        }
      }
      worktime = MPI_Wtime() - start;
      printf("Length is %d, time is %f\n", n, worktime);
        for (int i = 0; i < n; ++i)
        {
          printf("%d ", arr[i]);
        }
        printf("\n");
    }
    else
    {
      int slice_size;
      MPI_Recv(&slice_size, 1, MPI_INT, 0, n, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
      int *target = (int*)malloc(slice_size*sizeof(int));
      MPI_Recv(&target[0], slice_size, MPI_INT, 0, n + 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        int *buff = (int*)malloc(slice_size*sizeof(int));
      target = merge_sort(target, buff, 0, slice_size - 1);
      MPI_Send(&target[0], slice_size, MPI_INT, 0, n + 1, MPI_COMM_WORLD);
    }
  }

  MPI_Finalize();
}
