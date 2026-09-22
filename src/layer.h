#ifndef LAYER_H
#define LAYER_H
#include "matrix.h"
#include "activation.h"


typedef struct {
    int n_in, n_out;
    Matrix W;              // n_out x n_in
    Vector b;              // dimension n_out
    Matrix dW;             // meme forme que W, gradient accumule
    Vector db;             // meme forme que b
    ActivationType activation;
} Layer;

typedef struct {
    Vector z;      // avant activation
    Vector a;      // apres activation
    Vector delta;  // erreur de cette couche (rempli pendant la backprop)
} LayerCache;

void layer_init_weights(Layer* l);
Layer layer_create(int n_in, int n_out, ActivationType activation);
LayerCache layer_cache_create();
void layer_free(Layer* l);
void layer_cache_free(LayerCache* c);
void layer_zero_grad(Layer* l);
void layer_forward(Layer* l, Vector input, LayerCache *cache);
Vector layer_backward_output(Layer* l, LayerCache* cache, Vector prev_a, Vector loss_grad);
Vector layer_backward_hidden(Layer* l, LayerCache* cache, Vector prev_a, Matrix next_W, Vector next_delta);

#endif  