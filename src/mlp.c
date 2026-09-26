#include <stdlib.h>
#include "mlp.h"
#include "layer.h"

MLP mlp_create(int* sizes, ActivationType* activations, int n_layers){
    MLP mlp;
    mlp.n_layers = n_layers;
    Layer* layers = (Layer*)malloc(n_layers*sizeof(Layer));
    LayerCache* layerCaches = (LayerCache*)malloc(n_layers*sizeof(LayerCache));
    for(int i = 0; i < n_layers; i++){
        layers[i] = layer_create(sizes[i],sizes[i+1], activations[i]);
        layerCaches[i] = layer_cache_create();
    }
    mlp.layers = layers;
    mlp.caches = layerCaches;
    return mlp;
}

void mlp_free(MLP* mlp){
    for(int i = 0; i < mlp->n_layers; i++){
        layer_free(&mlp->layers[i]);
        layer_cache_free(&mlp->caches[i]);
    }
    free(mlp->layers);
    free(mlp->caches);
}

Vector mlp_forward(MLP* mlp, Vector input){
    Vector in = input;
    for(int i = 0; i < mlp->n_layers; i++){
        layer_forward(&mlp->layers[i],in,&mlp->caches[i]);
        in = mlp->caches[i].a;
    }
    return in;
}

void mlp_backward(MLP* mlp, Vector input, Vector loss_grad){
    int derniere_couche = mlp->n_layers-1;
    Vector prev_a = input;
    if(derniere_couche != 0){prev_a = mlp->caches[derniere_couche-1].a;}
    Vector delta = layer_backward_output(&mlp->layers[derniere_couche],&mlp->caches[derniere_couche],prev_a,loss_grad);
    for(int i = derniere_couche-1; i >= 0; i--){
        Vector prev_a = input;
        if(i != 0){prev_a = mlp->caches[i-1].a;}
        delta = layer_backward_hidden(&mlp->layers[i],&mlp->caches[i],prev_a,mlp->layers[i+1].W,delta);
    }
}

void mlp_zero_grad(MLP* mlp){
    for(int i = 0; i < mlp->n_layers; i++){
        layer_zero_grad(&mlp->layers[i]);
    }
}