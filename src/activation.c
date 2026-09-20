#include <math.h>
#include "matrix.h"

scalar_t sigmoid(scalar_t x){
    return 1/(1 + expf(-x));
}

scalar_t sigmoid_prime(scalar_t x){
    scalar_t sigmoidx = sigmoid(x);
    return sigmoidx * (1 - sigmoidx);
}

Vector sigmoid_vector(Vector v){
    Vector sv = vector_create(v.dim);
    for(int i = 0; i < v.dim; i++){
        sv.data[i] = sigmoid(v.data[i]);
    }
    return sv;
}

Vector sigmoid_prime_vector(Vector v){
    Vector sv = vector_create(v.dim);
    for(int i = 0; i < v.dim; i++){
        sv.data[i] = sigmoid_prime(v.data[i]);
    }
    return sv;
}
Matrix sigmoid_matrix(Matrix m){
    Matrix sm = matrix_create(m.rows, m.cols);
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            sm.data[i*m.cols+j] = sigmoid(m.data[i*m.cols+j]);
        }
    }
    return sm;
}

Matrix sigmoid_prime_matrix(Matrix m){
    Matrix sm = matrix_create(m.rows, m.cols);
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            sm.data[i*m.cols+j] = sigmoid_prime(m.data[i*m.cols+j]);
        }
    }
    return sm;
}

scalar_t tanh_activation(scalar_t x){
    return tanhf(x);
}

scalar_t tanh_activation_prime(scalar_t x){
    scalar_t tanhx = tanh_activation(x);
    return 1 - tanhx*tanhx;
}
Vector tanh_vector(Vector v){
    Vector sv = vector_create(v.dim);
    for(int i = 0; i < v.dim; i++){
        sv.data[i] = tanh_activation(v.data[i]);
    }
    return sv;
}

Vector tanh_prime_vector(Vector v){
    Vector sv = vector_create(v.dim);
    for(int i = 0; i < v.dim; i++){
        sv.data[i] = tanh_activation_prime(v.data[i]);
    }
    return sv;
}
Matrix tanh_matrix(Matrix m){
    Matrix sm = matrix_create(m.rows, m.cols);
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            sm.data[i*m.cols+j] = tanh_activation(m.data[i*m.cols+j]);
        }
    }
    return sm;
}

Matrix tanh_prime_matrix(Matrix m){
    Matrix sm = matrix_create(m.rows, m.cols);
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            sm.data[i*m.cols+j] = tanh_activation_prime(m.data[i*m.cols+j]);
        }
    }
    return sm;
}

scalar_t relu(scalar_t x){
    if(x >= 0){return x;}
    return 0;
}

scalar_t relu_prime(scalar_t x){
    if(x > 0){return 1;}
    return 0;
}

Vector relu_vector(Vector v){
    Vector sv = vector_create(v.dim);
    for(int i = 0; i < v.dim; i++){
        sv.data[i] = relu(v.data[i]);
    }
    return sv;
}

Vector relu_prime_vector(Vector v){
    Vector sv = vector_create(v.dim);
    for(int i = 0; i < v.dim; i++){
        sv.data[i] = relu_prime(v.data[i]);
    }
    return sv;
}
Matrix relu_matrix(Matrix m){
    Matrix sm = matrix_create(m.rows, m.cols);
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            sm.data[i*m.cols+j] = relu(m.data[i*m.cols+j]);
        }
    }
    return sm;
}

Matrix relu_prime_matrix(Matrix m){
    Matrix sm = matrix_create(m.rows, m.cols);
    for(int i = 0; i < m.rows; i++){
        for(int j = 0; j < m.cols; j++){
            sm.data[i*m.cols+j] = relu_prime(m.data[i*m.cols+j]);
        }
    }
    return sm;
}