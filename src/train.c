#include "layer.h"
#include "loss.h"
#include "mlp.h"

void sgd_update(Layer* l, scalar_t lr){
    for(int i = 0; i < l->W.rows * l->W.cols; i++){
        l->W.data[i] -= lr * l->dW.data[i];
    }
    for(int i = 0; i < l->b.dim; i++){
        l->b.data[i] -= lr * l->db.data[i];
    }
}



void train_epoch(MLP* mlp, Vector* inputs, Vector* targets, int n_examples, scalar_t learning_rate){
    for(int i = 0; i < n_examples; i++){
        mlp_zero_grad(mlp);
        Vector prediction = mlp_forward(mlp, inputs[i]);
        Vector dL = mse_loss_prime(prediction,targets[i]);
        mlp_backward(mlp,inputs[i],dL);
        for(int j = 0; j < mlp->n_layers; j++){
            sgd_update(&mlp->layers[j],learning_rate);
        }
        vector_free(&dL);
    }
}


