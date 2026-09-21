#include "matrix.h"
#include <stdlib.h>
#include <assert.h>

Matrix matrix_create(int rows, int cols) {
   Matrix m;
   scalar_t* tab = (scalar_t*)malloc(rows*cols*sizeof(scalar_t));
   for(int i = 0; i < rows*cols; i++){tab[i]=0;}
   m.cols = cols;
   m.rows = rows;
   m.data = tab;
   return m;
}

void matrix_free(Matrix* m){
   free(m->data);
}

Matrix matrix_mul(Matrix a, Matrix b){
   assert(a.cols == b.rows);
   Matrix c = matrix_create(a.rows, b.cols);
   for(int i = 0; i < a.rows; i++){
      for(int k = 0; k < a.cols; k++){
         scalar_t a_i_k = a.data[i*a.cols + k];
         for(int j = 0; j < b.cols; j++){
            c.data[i*c.cols + j] += a_i_k * b.data[k*b.cols + j];
         }
      }
   }
   return c;
}

Vector vector_create(int dims) {   
   Vector v;
   scalar_t* tab = (scalar_t*)malloc(dims*sizeof(scalar_t));
   for(int i = 0; i < dims; i++){tab[i]=0;}
   v.dim = dims;
   v.data = tab;
   return v;
}

void vector_free(Vector* v){
   free(v->data);
}

Vector matrix_vec_mul(Matrix m, Vector v){
   assert(m.cols == v.dim);
   Vector c = vector_create(m.rows);
   for(int i = 0; i < m.rows; i++){
      scalar_t sum = 0;
      for(int j = 0; j < m.cols; j++){
         sum += m.data[i*m.cols+j] * v.data[j];
      }
      c.data[i] = sum;
   }
   return c;
}

Matrix matrix_transpose(Matrix m){
   Matrix mt = matrix_create(m.cols, m.rows);
   for(int i = 0; i < m.rows; i++){
      for(int j = 0; j < m.cols; j++){
         mt.data[j*mt.cols+i] = m.data[i*m.cols+j];
      }
   }
   return mt;
}

Matrix matrix_add(Matrix a, Matrix b){
   assert(a.rows == b.rows && a.cols == b.cols);   
   Matrix c = matrix_create(a.rows, a.cols);
   for(int i = 0; i < a.rows; i++){
      for(int j = 0; j < a.cols; j++){
         c.data[i*a.cols+j] = a.data[i*a.cols+j] + b.data[i*a.cols+j];
      }
   }
   return c;
}

Vector vector_add(Vector a, Vector b){
   assert(a.dim = b.dim);   
   Vector c = vector_create(a.dim);
   for(int i = 0; i < a.dim; i++){
      c.data[i] = a.data[i] + b.data[i];
   }
   return c;
}

Matrix matrix_hadamard(Matrix a, Matrix b){
   assert(a.rows == b.rows && a.cols == b.cols);   
   Matrix c = matrix_create(a.rows, a.cols);
   for(int i = 0; i < a.rows; i++){
      for(int j = 0; j < a.cols; j++){
         c.data[i*a.cols+j] = a.data[i*a.cols+j] * b.data[i*a.cols+j];
      }
   }
   return c;
}

Matrix vector_to_matrix(Vector v){
   Matrix m = matrix_create(v.dim,1);
   for(int i = 0; i < v.dim; i++){
      m.data[i] = v.data[i];
   }
   return m;
}

Vector matrix_to_vector(Matrix m){
   assert(m.cols = 1);
   Vector v = vector_create(m.rows);
   for(int i = 0; i < m.rows; i++){
      v.data[i] = m.data[i];
   }
   return v;
}