#include <stdlib.h>
#include "matrix.h"
#include "activation.h"
#include "layer.h"

void layer_init_weights(Layer* l){
    /*rempli les coefficient de la mtrixe W de l par des coefficient 
     * uniformément piochés dans [-borne; borne]*/
    float borne = sqrt(6.0f / (float)(l-> n_in + l->n_out));
    for(int i = 0; i < l->n_out; i++){
        for(int j = 0; j < l->n_in; j++){
            double r =
            l->W.data[i*l->n_out+j] = borne * (2*((float)rand() / (float)RAND_MAX) - 1);
        }
    }
}

Layer layer_create(int n_in, int n_out, ActivationType activation){
    Layer l;
    l.n_in = n_in;
    l.n_out = n_out;
    l.W = matrix_create(n_out,n_in);
    l.dW = matrix_create(n_out,n_in);
    l.b = vector_create(n_out);
    l.db = vector_create(n_out);
    l.activation = activation;
    layer_init_weights(&l);
    return l;
}

void layer_free(Layer* l){
    matrix_free(&l->W);
    matrix_free(&l->dW);
    vector_free(&l->b);
    vector_free(&l->db);
}

void layer_zero_grad(Layer* l){
    /*remet à 0 le dW et db de l*/
    matrix_free(&l->dW);
    matrix_free(&l->db);
    l->dW = matrix_create(l->n_out,l->n_in);
    l->db = vector_create(l->n_out);
}

void layer_forward(Layer* l, Vector input, LayerCache* cache){
    Vector z = vector_add(matrix_vec_mul(l->W, input), l->b);
    Vector a = activation_apply(l->activation, z);
    cache->z = z;
    cache->a = a;
}

Vector layer_backward_output(Layer* l, LayerCache* cache, Vector prev_a, Vector loss_grad){

}
