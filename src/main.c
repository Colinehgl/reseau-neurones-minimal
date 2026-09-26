#include <stdio.h>
#include "matrix.h"
#include "activation.h"
#include "mlp.h"
#include "train.h"
#include "loss.h"
#include "dataset.h"

int main(void){
    Vector inputs[4], targets[4];
    get_xor_dataset(inputs, targets);

    int sizes[3] = {2, 4, 1};
    ActivationType acts[2] = {ACTIVATION_SIGMOID, ACTIVATION_SIGMOID};
    MLP net = mlp_create(sizes, acts, 2);

    int n_epochs = 10000;
    scalar_t lr = 0.5f;

    for(int epoch = 0; epoch < n_epochs; epoch++){
        train_epoch(&net, inputs, targets, 4, lr);
        if(epoch % 1000 == 0){
            scalar_t total_loss = 0;
            for(int i = 0; i < 4; i++){
                Vector pred = mlp_forward(&net, inputs[i]);
                total_loss += mse_loss(pred, targets[i]);
            }
            printf("epoch %d : loss = %f\n", epoch, total_loss / 4);
        }
    }

    printf("\nResultats finaux :\n");
    for(int i = 0; i < 4; i++){
        Vector pred = mlp_forward(&net, inputs[i]);
        printf("  %.0f XOR %.0f = %.4f (attendu %.0f)\n",
               inputs[i].data[0], inputs[i].data[1], pred.data[0], targets[i].data[0]);
    }

    for(int i = 0; i < 4; i++){
        vector_free(&inputs[i]);
        vector_free(&targets[i]);
    }
    mlp_free(&net);
    return 0;
}