#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {

  MPI_Init(NULL, NULL);

  int world_rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
  int world_size;
  MPI_Comm_size(MPI_COMM_WORLD, &world_size);

  int rmsg;
  int smsg = -world_rank;

  if (world_rank != 0)
  {
    MPI_Sendrecv(&smsg, 1, MPI_INT, (world_rank + 1) % world_size, 0, &rmsg, 1, MPI_INT, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Process %d received message \"%d\" from process %d and sent \"%d\" to process %d\n", world_rank, rmsg, world_rank - 1, smsg, (world_rank + 1) % world_size);
  }
  else
  {
    MPI_Sendrecv(&smsg, 1, MPI_INT, (world_rank + 1) % world_size, 0, &rmsg, 1, MPI_INT, world_size - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    printf("Process %d received message \"%d\" from process %d and sent \"%d\" to process %d\n", world_rank, rmsg, world_size - 1, smsg, (world_rank + 1) % world_size);
  }

  MPI_Finalize();
}
