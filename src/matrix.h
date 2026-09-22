#ifndef MATRIX_H
#define MATRIX_H

typedef float scalar_t;

typedef struct {
    int rows;
    int cols;
    scalar_t *data;  // taille rows*cols, row-major : data[i*cols + j] = élément (i,j)
} Matrix;

typedef struct {
    int dim;
    scalar_t *data;  
} Vector;

Matrix matrix_create(int rows, int cols);
void matrix_free(Matrix* m);
Matrix matrix_mul(Matrix a, Matrix b);
Vector vector_create(int dims);
void vector_free(Vector* v);
Vector matrix_vec_mul(Matrix m, Vector v);
Matrix matrix_transpose(Matrix m);
Matrix matrix_add(Matrix a, Matrix b);
Vector vector_add(Vector a, Vector b);
Matrix matrix_hadamard(Matrix a, Matrix b);
Vector vector_hadamard(Vector a, Vector b);
Matrix  matrix_outer_product(Vector a, Vector b);
Matrix vector_to_matrix(Vector v);
Vector matrix_to_vector(Matrix m);


#endif