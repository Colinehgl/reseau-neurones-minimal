#ifndef MLP_H
#define MLP_H
#include "layer.h"

typedef struct {
    Layer* layers;         // tableau de couches
    LayerCache* caches;    // un cache par couche, meme longueur
    int n_layers;
} MLP;

MLP mlp_create(int* sizes, ActivationType* activations, int n_layers);
void mlp_free(MLP* mlp);
Vector mlp_forward(MLP* mlp, Vector input);
void mlp_backward(MLP* mlp, Vector input, Vector loss_grad);
void mlp_zero_grad(MLP* mlp);

#endif 