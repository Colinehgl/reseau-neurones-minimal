#ifndef ACTIVATION_H
#define ACTIVATION_H
#include "matrix.h"

typedef enum {
    ACTIVATION_SIGMOID,
    ACTIVATION_TANH,
    ACTIVATION_RELU
} ActivationType;


scalar_t sigmoid(scalar_t x);
scalar_t sigmoid_prime(scalar_t x);
Vector sigmoid_vector(Vector v);
Vector sigmoid_prime_vector(Vector v);
Matrix sigmoid_matrix(Matrix m);
Matrix sigmoid_prime_matrix(Matrix m);

scalar_t tanh_activation(scalar_t x);
scalar_t tanh_activation_prime(scalar_t x);
Vector tanh_vector(Vector v);
Vector tanh_prime_vector(Vector v);
Matrix tanh_matrix(Matrix m);
Matrix tanh_prime_matrix(Matrix m);

scalar_t relu(scalar_t x);
scalar_t relu_prime(scalar_t x);
Vector relu_vector(Vector v);
Vector relu_prime_vector(Vector v);
Matrix relu_matrix(Matrix m);
Matrix relu_prime_matrix(Matrix m);


Vector activation_apply(ActivationType type, Vector z);
Vector activation_apply_prime(ActivationType type, Vector z);

#endif