#include <pthread.h>
#include <stdlib.h>
#include <limits.h>
#include <unistd.h>
#define SIZE 100

typedef struct matrix_type_t {
  int row, col;
  double(*MA )[SIZE], (*MB)[SIZE], (*MC)[SIZE];
} matrix_type_t ;

void *thread_mult(void *arg) {
  matrix_type_t *work = ( matrix_type_t *) arg ;
  int i, r = work->row , c = work->col ;
  work ->MC[r][c] = 0.0;
  for (i = 0; i < SIZE ; i++)
    work->MC[r][c] += work ->MA[r][i] * work ->MB[i][c];
  return NULL;
}

double MA[SIZE][SIZE], MB[SIZE][SIZE], MC[SIZE][SIZE];
matrix_type_t *work ;
pthread_t thread[SIZE][SIZE];

int main (int argc, char *argv []) {
  int row, col;
  //long int maxThreads;
  //maxThreads = sysconf(_SC_THREAD_THREADS_MAX);
  //printf("%ld\n",maxThreads);
  for ( row = 0; row < SIZE ; row ++)
    for ( col = 0; col < SIZE ; col ++) 
    {
      work = (matrix_type_t *) malloc(sizeof ( matrix_type_t ));
      work -> row = row ;
      work -> col = col ;
      work ->MA = MA;
      work ->MB = MB;
      work ->MC = MC;
      pthread_create(&thread[row][col], NULL, thread_mult, work);
    }
  for ( row = 0; row < SIZE ; row++)
    for ( col = 0; col < SIZE ; col++)
      pthread_join(thread[row][col], NULL);

}