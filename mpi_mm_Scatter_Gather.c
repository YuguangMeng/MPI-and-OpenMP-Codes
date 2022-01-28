/******************************************************************************
* FILE: mpi_mm.c
* DESCRIPTION:
*   MPI Matrix Multiply - C Version
*   In this code, the master task distributes a matrix multiply
*   operation to numtasks worker tasks.
*   By Yuguang Meng
******************************************************************************/
#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MASTER 0               /* taskid of first task */

int main (int argc, char *argv[])
{
  int	numtasks,              /* number of tasks in partition */
      taskid,                /* a task identifier */
      numworkers,            /* number of worker tasks */
      source,                /* task id of message source */
      dest,                  /* task id of message destination */
      rows,                  /* rows of matrix A sent to each worker */
      averow, extra, offset, /* used to determine rows sent to each worker */
      i, j, k, rc;           /* misc */
  int NRA, NCA, NCB;
  double	*a,           /* matrix A to be multiplied a[NRA][NCA] */
          *b,           /* matrix B to be multiplied b[NCA][NCB] */
          *c;           /* result matrix C c[NRA][NCB] */

  double *a_worker, *c_worker; //a_worker[NRA][NCA], a_worker[rows][NCA]
  int   *displs_a, *displs_c, *rcounts_a, *rcounts_c;
  double start, end;

  MPI_Status status;

  MPI_Init(&argc,&argv);
  MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
  MPI_Comm_size(MPI_COMM_WORLD,&numtasks);

  if(argc != 4)
  {
    if (taskid == MASTER)
      printf("Parameters must be 4!\n" );
    MPI_Finalize();
    exit(1);
  }

  NRA=atoi(argv[1]);
  NCA=atoi(argv[2]);
  NCB=atoi(argv[3]);

  a = (double *)malloc(NRA*NCA*sizeof(double));
  b = (double *)malloc(NCA*NCB*sizeof(double));
  c = (double *)malloc(NRA*NCB*sizeof(double));
  a_worker = (double *)malloc(NRA*NCA*sizeof(double));

  numworkers = numtasks-1;

  /* Send matrix data to the worker tasks */
  averow = NRA/numtasks;
  extra = NRA%numtasks;
  offset = 0;

  displs_a = (int *)malloc(numtasks*sizeof(int));
  displs_c = (int *)malloc(numtasks*sizeof(int));
  rcounts_a = (int *)malloc(numtasks*sizeof(int));
  rcounts_c = (int *)malloc(numtasks*sizeof(int));
  for (dest=1; dest<=numtasks; dest++)
  {
    rows = (dest <= extra) ? averow+1 : averow;
    displs_a[dest-1] = offset*NCA;
    displs_c[dest-1] = offset*NCB;
    rcounts_a[dest-1] = rows*NCA;
    rcounts_c[dest-1] = rows*NCB;
    offset = offset + rows;
  }

  /**************************** master task ************************************/
  if (taskid == MASTER)
  {
    //printf("mpi_mm has started with %d tasks.\n",numtasks);
    //printf("Initializing arrays...\n");
    for (i=0; i<NRA; i++)
      for (j=0; j<NCA; j++)
        a[i*NCA+j]= i+j; //a[i][j]
  }
  for (i=0; i<NCA; i++)
    for (j=0; j<NCB; j++)
      b[i*NCB+j]= i*j; //b[i][j]

  MPI_Barrier(MPI_COMM_WORLD);

  start = MPI_Wtime();

  MPI_Scatterv(a, rcounts_a, displs_a, MPI_DOUBLE, a_worker, rcounts_a[taskid], MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

  rows = (int)(rcounts_a[taskid]/NCA);
  c_worker = (double *)malloc(NCB*rows*sizeof(double));
  for (k=0; k<NCB; k++)
    for (i=0; i<rows; i++)
    {
      c_worker[i*NCB+k] = 0.0;
      for (j=0; j<NCA; j++)
        c_worker[i*NCB+k] = c_worker[i*NCB+k] + a_worker[i*NCA+j] * b[j*NCB+k];  //a_worker[i][j], b[j][k]
    }

  /* Receive results from worker tasks */
  MPI_Gatherv(c_worker, rcounts_c[taskid], MPI_DOUBLE, c, rcounts_c, displs_c, MPI_DOUBLE, MASTER, MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  if (taskid == MASTER)
  {
    end = MPI_Wtime();
    printf("%.8f \n", (end-start)*1e3);
  }

  /*
    if (taskid == MASTER)
    {
      // Print results
      printf("******************************************************\n");
      printf("Result Matrix:\n");
      for (i=0; i<NRA; i++)
      {
        printf("\n");
        for (j=0; j<NCB; j++)
          printf("%6.2f   ", c[i*NCB+j]); //c[i][j]
      }
      printf("\n******************************************************\n");
      printf ("Done.\n");
    }
  */
  free(displs_a);
  free(displs_c);
  free(rcounts_a);
  free(rcounts_c);
  free(c_worker);
  free(a);
  free(b);
  free(c);
  free(a_worker);

  MPI_Finalize();

}
