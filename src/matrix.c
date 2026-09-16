#include "matrix.h"
#include <stdlib.h>

Matrix matrix_create(int rows, int cols) {
   /*Création de la matrice*/
   Matrix m;
   scalar_t* tab = (scalar_t*)malloc(rows*cols*sizeof(float));
   m.cols = cols;
   m.rows = rows;
   m.data = tab;
   return m;
}

Matrix