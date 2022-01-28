/*
* =====================================================================================
Student Name: Yuguang Meng
Cannon's Algorithm implemented with OpenMP
* =====================================================================================
*/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "omp.h"

void process_mult();
void shift_matrix_left();
void shift_matrix_up();
void matrix_product();

int main(int argc, char *argv[])
{
  double *A, *B, *C;
  double t1, t2;
  int i, j, N, nprocs, nprocs_sqrt, block_sz;

  N = atoi(argv[1]);
  nprocs = atoi(argv[2]);
  nprocs_sqrt = (int)sqrt((double)nprocs);
  block_sz = N / nprocs_sqrt;

  A = (double *)malloc(N*N * sizeof(double));
  B = (double *)malloc(N*N * sizeof(double));
  C = (double *)malloc(N*N * sizeof(double));

  /* creat dense matrix of A and B */
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      A[i*N+j]= 0.1;
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      B[i*N+j]= 0.1;

  shift_matrix_left(A, N, block_sz, 1);
  shift_matrix_up(B, N, block_sz, 1);

  t1 = omp_get_wtime();
  for(i = 0; i < nprocs_sqrt; i++) {
    process_mult(A, B, C, N, block_sz, nprocs_sqrt, nprocs);
    shift_matrix_left(A, N, block_sz, 0);
    shift_matrix_up(B, N, block_sz, 0);
  }
  t2 = omp_get_wtime();
  printf("%.1f s\n", (t2 - t1));

  free(A);
  free(B);
  free(C);

  return 0;
}

void process_mult(double *A, double *B, double *C, int N, int block_sz, int nprocs_sqrt, int nprocs)
{
  int r, c, id, k,
      rbegin, rend, cbegin, cend, // block delimiters
      l, m;
  double *sa, *sb, *sc;
  #pragma omp parallel default(none) \
  private(l, m, r, c, k, rbegin, rend, cbegin, cend, id, sa, sb, sc) \
  shared(A, B, C, block_sz, N, nprocs, nprocs_sqrt) num_threads(nprocs)
  {
    id = omp_get_thread_num();
    rbegin = (id / nprocs_sqrt) * block_sz;
    rend = rbegin + block_sz;
    cbegin = (id % nprocs_sqrt) * block_sz;
    cend = cbegin + block_sz;

    sa = (double *)malloc(block_sz*block_sz * sizeof(double));
    sb = (double *)malloc(block_sz*block_sz * sizeof(double));
    sc = (double *)malloc(block_sz*block_sz * sizeof(double));

    for(r = rbegin, l = 0; r < rend; r++, l++)
      for(c = cbegin, m = 0; c < cend; c++, m++)
      {
        sa[l*block_sz+m] = A[r*N+c];
        sb[l*block_sz+m] = B[r*N+c];
        sc[l*block_sz+m] = C[r*N+c];
      }

    matrix_product(sc, sa, sb, block_sz);

    for(r = rbegin, l = 0; r < rend; r++, l++) {
      for(c = cbegin, m = 0; c < cend; c++, m++) {
        C[r*N+c] = sc[l*block_sz+m];
      }
    }

    free(sa);
    free(sb);
    free(sc);
  }
}


void shift_matrix_left(double *m, int N, int block_sz, int initial)
{
  int i, j, k, s, step = block_sz;
  double *aux;
  aux = (double *)malloc(N * sizeof(double));
  for(k = 0, s = 0; k < N; k += block_sz, s++) {
    for(i = k; i < (k + block_sz); i++) {
      if(initial > 0) {
        step = s * block_sz;
      }
      for(j = 0; j < N; j++) {
        aux[j] = m[i*N+(j + step) % N];
      }
      for(j = 0; j < N; j++) {
        m[i*N+j] = aux[j];
      }
    }
  }
  free(aux);
}

void shift_matrix_up(double *m, int N, int block_sz, int initial)
{
  int i, j, k, s, step = block_sz;
  double *aux;
  aux = (double *)malloc(N * sizeof(double));
  for(k = 0, s = 0; k < N; k += block_sz, s++) {
    for(i = k; i < (k + block_sz); i++) {
      if(initial > 0) {
        step = s * block_sz;
      }
      for(j = 0; j < N; j++) {
        aux[j] = m[((j + step) % N)*N+i];
      }
      for(j = 0; j < N; j++) {
        m[j*N+i] = aux[j];
      }
    }
  }
  free(aux);
}

void matrix_product(double *c, double *a, double *b, int block_sz)
{
  int r, s, k;
  for(r = 0; r < block_sz; r++)
    for(s = 0; s < block_sz; s++)
      for(k = 0; k < block_sz; k++)
        c[r*block_sz+s] += a[r*block_sz+k] * b[k*block_sz+s];
}