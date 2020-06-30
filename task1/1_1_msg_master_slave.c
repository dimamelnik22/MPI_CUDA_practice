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

  if (world_rank != 0)
  {
    MPI_Recv(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Process %d received message \"%d\" from process 0 and sent  back \"%d\"\n", world_rank, msg, -msg);
  }
  else
  {
    for(int i = 1; i < world_size; ++i)
    {
      msg = -i;
      MPI_Send(&msg, 1, MPI_INT, i, 0, MPI_COMM_WORLD);
    }
  }


  if (world_rank != 0)
  {
    msg = -msg;
    MPI_Send(&msg, 1, MPI_INT, 0, 0, MPI_COMM_WORLD);
  }
  else
  {
    int recv_count = 0;
    while (recv_count < world_size - 1)
    {
      MPI_Status status;
      MPI_Recv(&msg, 1, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &status);
      printf("Process %d received message \"%d\" from process %d\n", world_rank, msg, status.MPI_SOURCE);
      recv_count++;
      printf("Recieved %d messages. %d messages left\n", recv_count, world_size - recv_count - 1);
    }
  }

  MPI_Finalize();
}
