/*
   Student name: Yuguang Meng
   Cannon's Algorithm implemented with MPI functions
*/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"

void CannonAlgorithm();
void MatrixMultiply();

int main( int argc, char *argv[] )
{
  int N, every_matrix, P, p, myID;
  double *A, *B, *C, *a, *b, *c;
  int i,j;
  double start, end;

  N = atoi(argv[1]);

  MPI_Init(&argc,&argv);
  MPI_Comm_size(MPI_COMM_WORLD, &P);
  MPI_Comm_rank(MPI_COMM_WORLD, &myID);

  if( myID == 0 )
  {
    A = (double *)malloc(N*N * sizeof(double));
    B = (double *)malloc(N*N * sizeof(double));
    C = (double *)malloc(N*N * sizeof(double));

    /* creat dense matrix of A and B */
    for (i=0; i<N; i++)
      for (j=0; j<N; j++)
        A[i*N+j]= i+j;
    for (i=0; i<N; i++)
      for (j=0; j<N; j++)
        B[i*N+j]= i*j;
  }

  every_matrix = N/sqrt(P);

  a = (double *)calloc(every_matrix*every_matrix, sizeof(double));
  b = (double *)calloc(every_matrix*every_matrix, sizeof(double));
  c = (double *)calloc(every_matrix*every_matrix, sizeof(double));

  if( myID == 0 )
    start = MPI_Wtime();

  MPI_Scatter(A, every_matrix*every_matrix, MPI_DOUBLE, a, every_matrix*every_matrix, MPI_DOUBLE, 0, MPI_COMM_WORLD);
  MPI_Scatter(B, every_matrix*every_matrix, MPI_DOUBLE, b, every_matrix*every_matrix, MPI_DOUBLE, 0, MPI_COMM_WORLD);

  CannonAlgorithm(N,a,b,c,MPI_COMM_WORLD);

  MPI_Gather(c, every_matrix*every_matrix, MPI_DOUBLE,C, every_matrix*every_matrix, MPI_DOUBLE,0, MPI_COMM_WORLD);

  if( myID == 0 )
  {
    end = MPI_Wtime();
    printf("%.1f s\n",end-start);
    free(A);
    free(B);
    free(C);
  }

  MPI_Finalize();
  return 0;
}

void CannonAlgorithm(int n, double *a, double *b, double *c, MPI_Comm comm)
{
  int i;
  int nlocal;
  int npes, dims[2], periods[2];
  int myrank, my2drank, mycoords[2];
  int uprank, downrank, leftrank, rightrank, coords[2];
  int shiftsource, shiftdest;
  MPI_Status status;
  MPI_Comm comm_2d;

  MPI_Comm_size(comm, &npes);
  MPI_Comm_rank(comm, &myrank);

  dims[0] = dims[1] = sqrt(npes);
  periods[0] = periods[1] = 1;

  MPI_Cart_create(comm, 2, dims, periods, 1, &comm_2d);

  MPI_Comm_rank(comm_2d, &my2drank);
  MPI_Cart_coords(comm_2d, my2drank, 2, mycoords);

  MPI_Cart_shift(comm_2d, 1, -1, &rightrank, &leftrank);
  MPI_Cart_shift(comm_2d, 0, -1, &downrank, &uprank);

  nlocal = n/dims[0];

  MPI_Cart_shift(comm_2d, 1, -mycoords[0], &shiftsource, &shiftdest);
  MPI_Sendrecv_replace(a, nlocal*nlocal, MPI_DOUBLE, shiftdest,
                       1, shiftsource, 1, comm_2d, &status);

  MPI_Cart_shift(comm_2d, 0, -mycoords[1], &shiftsource, &shiftdest);
  MPI_Sendrecv_replace(b, nlocal*nlocal, MPI_DOUBLE,
                       shiftdest, 1, shiftsource, 1, comm_2d, &status);

  for (i=0; i<dims[0]; i++)
  {
    MatrixMultiply(nlocal, a, b, c);
    MPI_Sendrecv_replace(a, nlocal*nlocal, MPI_DOUBLE,
                         leftrank, 1, rightrank, 1, comm_2d, &status);
    MPI_Sendrecv_replace(b, nlocal*nlocal, MPI_DOUBLE,
                         uprank, 1, downrank, 1, comm_2d, &status);
  }

  MPI_Comm_free(&comm_2d);
}

void MatrixMultiply(int n, double *a, double *b, double *c)
{
  int i, j, k;
  for (i=0; i<n; i++)
    for (j=0; j<n; j++)
      for (k=0; k<n; k++)
        c[i*n+j] += a[i*n+k]*b[k*n+j];
}