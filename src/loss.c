#include "loss.h"
#include "matrix.h"
#include <math.h>
#include <assert.h>

scalar_t mse_loss(Vector prediction, Vector target){
    /*calcule la loss selon la formule*/
    scalar_t n = (scalar_t)prediction.dim;
    assert(n == target.dim);
    scalar_t nloss = 0;
    for(int i = 0; i < (int)n; i++){
        nloss = nloss + (prediction.data[i] - target.data[i])*(prediction.data[i] - target.data[i]);
    }
    return (1/n)*nloss;
}


Vector mse_loss_prime(Vector prediction, Vector target){
    /*calcule dL/da; le carré de la fonction au dessus donne le 2 en dessous*/
    int n = prediction.dim;
    assert(n == target.dim);
    Vector dloss = vector_create(n);
    for(int i = 0; i < (int)n; i++){
        dloss.data[i] = ( 2 / (scalar_t)n) * (prediction.data[i] - target.data[i]);
    }
    return dloss;
}

scalar_t cross_entropy_loss(Vector prediction, Vector target){
    scalar_t n = (scalar_t)prediction.dim;
    assert(n == target.dim);
    scalar_t ce = 0;
    for(int i = 0; i < (int)n; i++){
        ce = ce + target.data[i] * logf(prediction.data[i]);
    }
    return -ce;
}

Vector cross_entropy_loss_prime(Vector prediction, Vector target){
    int n = prediction.dim;
    assert(n == target.dim);
    Vector dce = vector_create(n);
    for(int i = 0; i < (int)n; i++){
        dce.data[i] = - (target.data[i] / prediction.data[i]) ;
    }
    return dce;
}