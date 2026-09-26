#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "activation.h"
#include "layer.h"
#include "loss.h"
#include "mlp.h"

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; } \
    else      { printf("[FAIL] %s (ligne %d)\n", name, __LINE__); } \
} while (0)

/* epsilon pour la difference finie centree : ni trop grand (approximation
 * imprecise) ni trop petit (bruit d'arrondi en float) */
#define EPSILON 1e-3f

/* compare deux gradients avec une tolerance relative, sauf quand les deux
 * sont deja negligeables (le bruit de la difference finie devient alors
 * plus grand que la valeur elle-meme, une tolerance relative n'a plus de sens) */
static int grad_close(scalar_t analytic, scalar_t numeric) {
    if (fabsf(analytic) < 1e-4f && fabsf(numeric) < 1e-4f) {
        return fabsf(analytic - numeric) < 1e-4f;
    }
    scalar_t diff = fabsf(analytic - numeric);
    scalar_t denom = fabsf(analytic) + fabsf(numeric);
    return (diff / denom) < 0.05f; /* 5% d'ecart relatif tolere */
}

/* refait un forward complet et calcule la loss, sans toucher aux gradients */
static scalar_t forward_loss(MLP* net, Vector input, Vector target) {
    Vector pred = mlp_forward(net, input);
    return mse_loss(pred, target);
}

int main(void) {
    printf("== Gradient checking (mlp_backward vs difference finie) ==\n");

    int sizes[3] = {2, 3, 1};
    ActivationType acts[2] = {ACTIVATION_SIGMOID, ACTIVATION_SIGMOID};
    MLP net = mlp_create(sizes, acts, 2);

    Vector input = vector_create(2);
    input.data[0] = 0.0f; input.data[1] = 1.0f;
    Vector target = vector_create(1);
    target.data[0] = 1.0f;

    /* --- gradient analytique : un seul forward + backward --- */
    mlp_zero_grad(&net);
    Vector pred = mlp_forward(&net, input);
    Vector loss_grad = mse_loss_prime(pred, target);
    mlp_backward(&net, input, loss_grad);
    vector_free(&loss_grad);

    /* --- gradient numerique : on perturbe chaque poids un par un --- */
    char name[64];
    for (int l = 0; l < net.n_layers; l++) {
        Layer* layer = &net.layers[l];

        for (int k = 0; k < layer->W.rows * layer->W.cols; k++) {
            scalar_t original = layer->W.data[k];

            layer->W.data[k] = original + EPSILON;
            scalar_t loss_plus = forward_loss(&net, input, target);

            layer->W.data[k] = original - EPSILON;
            scalar_t loss_minus = forward_loss(&net, input, target);

            layer->W.data[k] = original; /* remettre la vraie valeur */

            scalar_t numeric  = (loss_plus - loss_minus) / (2 * EPSILON);
            scalar_t analytic = layer->dW.data[k];

            snprintf(name, sizeof(name), "dW couche %d, indice %d", l, k);
            CHECK(grad_close(analytic, numeric), name);
        }

        for (int k = 0; k < layer->b.dim; k++) {
            scalar_t original = layer->b.data[k];

            layer->b.data[k] = original + EPSILON;
            scalar_t loss_plus = forward_loss(&net, input, target);

            layer->b.data[k] = original - EPSILON;
            scalar_t loss_minus = forward_loss(&net, input, target);

            layer->b.data[k] = original;

            scalar_t numeric  = (loss_plus - loss_minus) / (2 * EPSILON);
            scalar_t analytic = layer->db.data[k];

            snprintf(name, sizeof(name), "db couche %d, indice %d", l, k);
            CHECK(grad_close(analytic, numeric), name);
        }
    }

    vector_free(&input);
    vector_free(&target);
    mlp_free(&net);

    printf("== %d/%d gradients coherents ==\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}