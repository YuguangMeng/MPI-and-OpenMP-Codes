/* Broadcast operation using MPI_Send and MPI_Recv.
   Program for hand-coded one-to-all broadcast on Lab machines and cheetah;
   Submit timing measurements (times measured using MPI_Wtime() and plot) for 5 message sizes (1,1K, 4K, 512K,1M bytes)
   for 5 different machine sizes (2, 4, 16, 64, 128 processes).

   The implementation of the one-to-all broadcast operation is based on a hypercube, but it can apply to any network topology
   and can be easily extended to work for any number of processes
   (see reference in: Ananth Grama, et al. Introduction to parallel computing, second edition. 2003, p158).
   This program has been extended to be applicable to any number of tasks.

   In the program, the data type for 1, 1K, 4K data is ¡°unsigned char¡± (in C language, type unsigned char is 1 byte);
   the data type for 512K, 1M data is double (in C language, type double is 8 bytes).
   For broadcasting 1, 1K, 4K data, the number of the data is respectively 1, 1024, 4096 and the data type in MPI is MPI_BYTE.
   For broadcasting 512K, 1M data, the number of data is respectively 65536 and 131072 and the data type in MPI is MPI_DOUBLE.

   In each process n, ¡°start¡± is the time for starting broadcasting and denoted as Sn; ¡°end¡± is the time for completing broadcasting and denoted as En.
   For each process, Sn and En are the averaged results of the values of 10 running times.
   Thus, the time for one-to-call broadcast for all process is Emax - Smin,
   where  Emax = Max{En,  0 ¡Ü n < task numbers -1 } and Smin = Min{Sn,  0 ¡Ü n < task numbers -1}.
   The time is the unit of micro-second (ms).
   
   By Yuguang Meng
*/

#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

#define  ROOT	0

/* Define the longest length of 1M data as double data type; it was also used for unsigned char */
#define  M1  131072

int main (int argc, char *argv[])
{
  /*  numtasks is the actual number of task;
  numbertasks2 is the task number that needs to calculate the sending/receiving nodes in broadcast;
  numtask2 can be more than numtask.
  */
  int           i, numtasks, numtasks2;

  /*   For 1, 1K, 4K */
  unsigned char     SEND_MSG[M1], RECV_MSG[M1];
  /*   For 512K, 1M */
//double                 SEND_MSG[M1], RECV_MSG[M1];

  /* define number of data to be send.
      Here is just an example; num can be changed according to different transmissions */
  unsigned long int num = 1;

  double     start, end;     /* the starting and ending time got MPI */
  MPI_Status       status;
  unsigned int      mask, d;
  unsigned int      taskid, source_dest_id, i2;   /* i2 is 2^i, see reference p158*/

  for ( i = 0; i < M1; i++)
  /* For unsigned char */
     SEND_MSG[i] = '0';
  /* For double */
  // SEND_MSG[i] = 0.0;

  MPI_Init(&argc, &argv);
  MPI_Comm_size ( MPI_COMM_WORLD,   &numtasks);
  MPI_Comm_rank ( MPI_COMM_WORLD,   &taskid);

  /* We use the newly calculated d and numtasks; numtasks can be more than numtask */

  d = (unsigned int) ceil (log((double)numtasks)/log(2.0));
  numtasks2 = 1 << d;
  mask = numtasks2 - 1;

  start = MPI_Wtime();

  for ( i = d - 1; i >= 0 ; i--)
  {
    i2 = 1 << i;
    mask = mask ^ i2;
    if (source_dest_id  <  numtasks)    // broadcasting only within the actural number of task
      if ((taskid & mask)==0)
      {
        source_dest_id = taskid ^ i2;
        if ((taskid & i2)==0)
// datatype: MPI_BYTE or MPI_DOUBLE
          MPI_Send(SEND_MSG, num, MPI_BYTE, source_dest_id, 0, MPI_COMM_WORLD);
        else
//data type: MPI_BYTE or MPI_DOUBLE
        MPI_Recv(RECV_MSG, num, MPI_BYTE, source_dest_id, 0, MPI_COMM_WORLD, &status);
      }
  }

  end = MPI_Wtime();

  printf("%.8f %.8f\n",  start*1e3, end*1e3);
  MPI_Finalize();

}
