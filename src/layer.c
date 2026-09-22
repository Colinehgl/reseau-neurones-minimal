#include <stdlib.h>
#include <math.h>
#include "matrix.h"
#include "activation.h"
#include "layer.h"

void layer_init_weights(Layer* l){
    /* remplie les coefficient de la mtrixe W de l par des coefficient 
     * uniformément piochés dans [-borne; borne]*/
    float borne = sqrt(6.0f / (float)(l-> n_in + l->n_out));
    for(int i = 0; i < l->n_out; i++){
        for(int j = 0; j < l->n_in; j++){
            l->W.data[i*l->n_in+j] = borne * (2*((float)rand() / (float)RAND_MAX) - 1);
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

LayerCache layer_cache_create(){
    /* initialise un cache vide (data a NULL) pour un usage sur avec layer_forward */
    LayerCache c;
    c.z = (Vector){0};
    c.a = (Vector){0};
    c.delta = (Vector){0};
    return c;
}

void layer_free(Layer* l){
    /* libere tout ce qui a ete alloue pour la couche : poids et gradients */
    matrix_free(&l->W);
    matrix_free(&l->dW);
    vector_free(&l->b);
    vector_free(&l->db);
}

void layer_cache_free(LayerCache* c){
    vector_free(&c->z);
    vector_free(&c->a);
    vector_free(&c->delta);
}

void layer_zero_grad(Layer* l){
    /*remet à 0 le dW et db de l*/
    matrix_free(&l->dW);
    vector_free(&l->db);
    l->dW = matrix_create(l->n_out,l->n_in);
    l->db = vector_create(l->n_out);
}

void layer_forward(Layer* l, Vector input, LayerCache* cache){
    /* calcule z = W*input + b puis a = activation(z), stocke les deux dans cache
     * (cache doit etre initialise a {0} avant le tout premier appel) */
    Vector weighted_input = matrix_vec_mul(l->W, input);
    Vector z = vector_add(weighted_input, l->b);
    Vector a = activation_apply(l->activation, z);
    vector_free(&cache->z);
    vector_free(&cache->a);
    cache->z = z;
    cache->a = a;
    vector_free(&weighted_input);
}

Vector layer_backward_output(Layer* l, LayerCache* cache, Vector prev_a, Vector loss_grad){
    /* backprop pour la couche de sortie : delta = loss_grad * activation'(z),
     * puis accumule dW et db a partir de ce delta */
    Vector deriv = activation_apply_prime(l->activation, cache->z);
    Vector delta = vector_hadamard(loss_grad, deriv);
    cache->delta = delta;
    Matrix dW_contribution = matrix_outer_product(delta, prev_a);
    Matrix new_dW = matrix_add(l->dW, dW_contribution);
    Vector new_db = vector_add(l->db, delta);
    matrix_free(&l->dW);
    vector_free(&l->db);
    l->dW = new_dW;
    l->db = new_db;
    matrix_free(&dW_contribution);
    vector_free(&deriv);
    return delta;
}

Vector layer_backward_hidden(Layer* l, LayerCache* cache, Vector prev_a, Matrix next_W, Vector next_delta){
    /* backprop pour une couche cachee : propage le delta de la couche suivante
     * via next_W, le combine avec activation'(z), puis accumule dW et db */
    Matrix next_W_transposed = matrix_transpose(next_W);
    Vector propagated = matrix_vec_mul(next_W_transposed, next_delta);
    Vector deriv = activation_apply_prime(l->activation, cache->z);
    Vector delta = vector_hadamard(propagated, deriv);
    cache->delta = delta;
    Matrix dW_contribution = matrix_outer_product(delta, prev_a);
    Matrix new_dW = matrix_add(l->dW, dW_contribution);
    Vector new_db = vector_add(l->db, delta);
    matrix_free(&l->dW);
    vector_free(&l->db);
    l->dW = new_dW;
    l->db = new_db;
    matrix_free(&next_W_transposed);
    matrix_free(&dW_contribution);
    vector_free(&propagated);
    vector_free(&deriv);
    return delta;
}