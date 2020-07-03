#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {

  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int msg;
  int forward_tag = 0;
  int backward_tag = 1;

  for(int i = 0; i < world_size; ++i)
  {
    if (i != world_rank)
    {
      msg = world_rank * 10 + i;
      MPI_Send(&msg, 1, MPI_INT, i, forward_tag, MPI_COMM_WORLD);
      printf("Process %d sent message \"%d\" to process %d\n", world_rank, msg, i);
    }
  }

  int recv_count = 0;

  while (recv_count < (world_size - 1) * 2)
  {
    MPI_Status status;
    MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, MPI_ANY_TAG, MPI_COMM_WORLD, &status);
    recv_count++;
    if (status.MPI_TAG == forward_tag)
    {
      printf("Process %d received message \"%d\" from process %d and sent back \"%d\"\n", world_rank, msg, status.MPI_SOURCE, -msg);
      msg = -msg;
      MPI_Send(&msg, 1, MPI_INT, status.MPI_SOURCE, backward_tag, MPI_COMM_WORLD);
    }
    else
    {
      printf("Process %d received back message \"%d\" from process %d\n", world_rank, msg, status.MPI_SOURCE);
    }
  }

  MPI_Finalize();
}

