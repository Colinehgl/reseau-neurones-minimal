#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "activation.h"
#include "layer.h"
#include "loss.h"
#include "mlp.h"
#include "train.h"

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; printf("[OK]   %s\n", name); } \
    else      { printf("[FAIL] %s (ligne %d)\n", name, __LINE__); } \
} while (0)

static int float_eq(scalar_t a, scalar_t b) {
    return fabsf(a - b) < 1e-4f;
}

int main(void) {
    printf("== Tests train.c ==\n");

    /* --- sgd_update : valeurs choisies pour un calcul simple a la main ---
     * W = [0.1, 0.2, 0.3, 0.4], dW = [1, 2, 3, 4], lr = 0.1
     * W_new = W - lr*dW = [0, 0, 0, 0] (tout tombe pile a zero) */
    Layer l = layer_create(2, 2, ACTIVATION_SIGMOID);
    l.W.data[0] = 0.1f; l.W.data[1] = 0.2f;
    l.W.data[2] = 0.3f; l.W.data[3] = 0.4f;
    l.dW.data[0] = 1; l.dW.data[1] = 2;
    l.dW.data[2] = 3; l.dW.data[3] = 4;
    l.b.data[0] = 0.5f; l.b.data[1] = 0.6f;
    l.db.data[0] = 5; l.db.data[1] = 6;

    sgd_update(&l, 0.1f);
    CHECK(float_eq(l.W.data[0], 0.0f) && float_eq(l.W.data[1], 0.0f) &&
          float_eq(l.W.data[2], 0.0f) && float_eq(l.W.data[3], 0.0f),
          "sgd_update: W correctement mis a jour");
    CHECK(float_eq(l.b.data[0], 0.0f) && float_eq(l.b.data[1], 0.0f),
          "sgd_update: b correctement mis a jour");
    layer_free(&l);

    /* --- train_epoch : verifie que la loss diminue reellement ---
     * un seul exemple fixe, repete sur plusieurs epochs : si le reseau
     * apprend, la loss doit etre nettement plus basse a la fin qu'au debut */
    int sizes[3] = {2, 3, 1};
    ActivationType acts[2] = {ACTIVATION_SIGMOID, ACTIVATION_SIGMOID};
    MLP net = mlp_create(sizes, acts, 2);

    Vector input = vector_create(2);
    input.data[0] = 1.0f; input.data[1] = 0.0f;
    Vector target = vector_create(1);
    target.data[0] = 1.0f;

    Vector inputs[1]  = { input };
    Vector targets[1] = { target };

    Vector pred_before = mlp_forward(&net, input);
    scalar_t loss_before = mse_loss(pred_before, target);

    for (int epoch = 0; epoch < 500; epoch++) {
        train_epoch(&net, inputs, targets, 1, 0.5f);
    }

    Vector pred_after = mlp_forward(&net, input);
    scalar_t loss_after = mse_loss(pred_after, target);

    printf("   (loss avant : %f, loss apres : %f)\n", loss_before, loss_after);
    CHECK(loss_after < loss_before * 0.1f, "train_epoch: la loss a nettement diminue apres 500 epochs");
    /*le test est un peu approximatif car il n'y a pas encore de dataset à ce moment*/
    vector_free(&input);
    vector_free(&target);
    mlp_free(&net);

    printf("== %d/%d tests passes ==\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}