#ifndef MATRIX_H
#define MATRIX_H


typedef float scalar_t;
typedef struct {
    int rows;
    int cols;
    scalar_t *data;  // taille rows*cols, row-major : data[i*cols + j] = élément (i,j)
} Matrix;


Matrix matrix_create(int rows, int cols);

#endif