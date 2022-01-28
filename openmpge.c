/* Student Name: Yuguang Meng
   -- Parallel Matrix Operations with OpenMP
------------------------------------------------------------------------ */

#include <stdio.h>
#include <stdlib.h>
#include "omp.h"

// This program was writting in the Centre of Advanced Calculations in
// Engineering Science (ACES).
#define TAG_MATRIX_PARTITION	44261 // Try to calculate hex value :-) 
// Type definitions
typedef struct
{ unsigned int   m, n;  // Rows, cols
  double        *data;  // Data, ordered by row, then by col
  double       **rows;  // Pointers to rows in data
} TMatrix;

// Function declaractions
TMatrix createMatrix (const unsigned int rows, const unsigned int cols);
void    freeMatrix   (TMatrix *matrix);
int     validMatrix  (TMatrix matrix);
TMatrix initMatrix   (void);
void    printMatrix  (TMatrix A);
int     readMatrix   (char *filename, TMatrix *A);
int     writeMatrix  (char *filename, TMatrix A);

/* ------------------------------------------------------------------------ */
int main (int argc, char *argv[])
/* ------------------------------------------------------------------------ */
{
  TMatrix      A,B;
  unsigned int nprocs, i, j, k;
  double       time0, time1;
  double       ratio;

  // Initiate all matrices to 0 x 0 matrices.
  A = initMatrix();
  B = initMatrix();

  // Read data, argument 0 = executable, 1 = matrix A, 2 = matrix B, 3 = output filename
  if (argc == 5)
    do
    {
      if (!readMatrix(argv[1], &A)) break;
      if (!readMatrix(argv[2], &B)) break;
      if (!validMatrix(A) || !validMatrix(B) || (A.m != B.m) || (A.m != A.n)) break;
      nprocs = atoi(argv[4]);
    } while (0);
  else
    printf ("Usage : %s <matrix filename> <matrix filename> <output filename>\n", argv[0]);

  time0 = omp_get_wtime();

  for (i = 0; i < A.m-1; i++)
  {
    #pragma omp parallel for schedule(static) num_threads(nprocs)
    for (j = i; j < A.m; j++)
    {
      ratio = A.rows[j][i]/A.rows[i][i];
      for (k = i; k < A.m; k++)
      {
        A.rows[j][k] -= (ratio*A.rows[i][k]);
        B.rows[j][0] -= (ratio*B.rows[i][0]);
      }
    }
  }

  // Record finish time
  time1 = omp_get_wtime();
  // Write result into file
  writeMatrix(argv[3],B);
  // Print time statistics
  printf ("Total time using [%2d] processors : [%f] seconds\n", nprocs, time1 - time0);

  // Free matrix data
  freeMatrix(&A);
  freeMatrix(&B);

  return 0;
}
/* ------------------------------------------------------------------------ */
TMatrix createMatrix(const unsigned int rows, const unsigned int cols)
/* ------------------------------------------------------------------------ */
{ TMatrix           matrix;
  unsigned long int m, n;
  unsigned int      i;
  m = rows;
  n = cols;
  matrix.m    = rows;
  matrix.n    = cols;
  matrix.data = (double *) malloc(sizeof(double) * m * n);
  matrix.rows = (double **) malloc(sizeof(double *) * m);
  if (validMatrix(matrix))
  { matrix.m = rows;
    matrix.n = cols;
    for (i = 0; i < rows; i++)
    { matrix.rows[i] = matrix.data + (i * cols);
    }
  }
  else
  { freeMatrix(&matrix);
  }
  return matrix;
}
/* ------------------------------------------------------------------------ */
void freeMatrix (TMatrix *matrix)
/* ------------------------------------------------------------------------ */
{ if (matrix == NULL) return;
  if (matrix -> data) {
    free(matrix -> data);
    matrix -> data = NULL;
  }
  if (matrix -> rows) {
    free(matrix -> rows);
    matrix -> rows = NULL;
  }
  matrix -> m = 0;
  matrix -> n = 0;
}
/* ------------------------------------------------------------------------ */
int validMatrix (TMatrix matrix)
/* ------------------------------------------------------------------------ */
{ if ((matrix.data == NULL) || (matrix.rows == NULL) ||
      (matrix.m == 0) || (matrix.n == 0))
    return 0;
  else return 1;
}
/* ------------------------------------------------------------------------ */
TMatrix initMatrix()
/* ------------------------------------------------------------------------ */
{ TMatrix matrix;
  matrix.m = 0;
  matrix.n = 0;
  matrix.data = NULL;
  matrix.rows = NULL;
  return matrix;
}

/* ------------------------------------------------------------------------ */
void printMatrix(TMatrix A)
/* ------------------------------------------------------------------------ */
{ unsigned int i, j;
  if (validMatrix(A))
  { for (i = 0; i < A.m; i++)
    { for (j = 0; j < A.n; j++) printf ("%7.3f ", A.rows[i][j]);
      printf ("\n");
    }
  }
}
/* ------------------------------------------------------------------------ */
int readMatrix(char *filename, TMatrix *A)
/* ------------------------------------------------------------------------ */
{ FILE *fp;
  unsigned int m, n, i, j;
  float d;
  int result = 0;
  if ((fp = fopen (filename, "r")) == NULL) return 0;
  do
  {
    if (fscanf (fp, "%d%d",  &m, &n) != 2) break;
    if ((m == 0) || (n == 0)) break;
    *A = createMatrix(m,n);
    if (!validMatrix(*A)) break;
    for (i = 0; i < m; i ++)
    { for (j = 0; j < n; j ++)
      { if (fscanf (fp, "%f", &d) != 1) break;
        A -> rows[i][j] = d;
      }
      if (j != n) break;
    }
    if (i != m) break;
    result = 1;
  } while (0);
  fclose (fp);
  return result;
}
/* ------------------------------------------------------------------------ */
int writeMatrix(char *filename, TMatrix A)
/* ------------------------------------------------------------------------ */
{ FILE *fp;
  unsigned int i, j;
  if ((fp = fopen (filename, "w")) == NULL) return 0;
  fprintf (fp, "%d %d\n", A.m, A.n);
  for (i = 0; i < A.m; i++)
  { for (j = 0; j < A.n; j++)
    { fprintf (fp, "%20.10f ", A.rows[i][j]);
    }
    fprintf (fp, "\n");
  }
  fclose(fp);
  return 1;
}
