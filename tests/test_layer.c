#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "activation.h"
#include "layer.h"

static int tests_run = 0;
static int tests_passed = 0;

#define CHECK(cond, name) do { \
    tests_run++; \
    if (cond) { tests_passed++; printf("[OK]   %s\n", name); } \
    else      { printf("[FAIL] %s (ligne %d)\n", name, __LINE__); } \
} while (0)

static int float_eq(scalar_t a, scalar_t b) {
    return fabsf(a - b) < 1e-3f;
}

int main(void) {
    printf("== Tests layer.c ==\n");

    /* --- layer_create : verifie juste les tailles --- */
    Layer l = layer_create(2, 2, ACTIVATION_SIGMOID);
    CHECK(l.W.rows == 2 && l.W.cols == 2, "layer_create: W a la bonne taille");
    CHECK(l.b.dim == 2, "layer_create: b a la bonne taille");
    CHECK(l.dW.rows == 2 && l.dW.cols == 2, "layer_create: dW a la meme taille que W");
    CHECK(float_eq(l.dW.data[0], 0.0f), "layer_create: dW commence a zero");
    layer_free(&l);

    /* --- layer_init_weights : teste avec une couche PAS carree (2 in, 3 out)*/
    Layer l2 = layer_create(2, 3, ACTIVATION_RELU);
    int toutes_non_nulles = 1;
    for (int i = 0; i < l2.W.rows * l2.W.cols; i++) {
        if (fabsf(l2.W.data[i]) < 1e-6f) toutes_non_nulles = 0;
    }
    CHECK(toutes_non_nulles, "layer_init_weights: aucune case de W oubliee (2x3)");
    layer_free(&l2);

    /* --- layer_forward : test avec :
     * W = identite, b = 0, entree x = [1, 2]
     * z = W*x + b = [1, 2]
     * a = sigmoid(z) = [sigmoid(1), sigmoid(2)] */
    Layer L1 = layer_create(2, 2, ACTIVATION_SIGMOID);
    L1.W.data[0] = 1; L1.W.data[1] = 0;
    L1.W.data[2] = 0; L1.W.data[3] = 1;
    L1.b.data[0] = 0; L1.b.data[1] = 0;

    Vector x = vector_create(2);
    x.data[0] = 1; x.data[1] = 2;

    LayerCache cache1 = layer_cache_create();
    layer_forward(&L1, x, &cache1);
    CHECK(float_eq(cache1.z.data[0], 1.0f) && float_eq(cache1.z.data[1], 2.0f),
          "layer_forward: z = W*x+b correct");
    CHECK(float_eq(cache1.a.data[0], 0.7310586f) && float_eq(cache1.a.data[1], 0.8807971f),
          "layer_forward: a = sigmoid(z) correct");

    /* --- layer_zero_grad : on met dW/db a une valeur non nulle, puis on verifie le reset --- */
    L1.dW.data[0] = 42.0f;
    L1.db.data[0] = 7.0f;
    layer_zero_grad(&L1);
    CHECK(float_eq(L1.dW.data[0], 0.0f) && float_eq(L1.db.data[0], 0.0f),
          "layer_zero_grad: dW et db remis a zero");

    /* --- layer_backward_hidden : on simule une couche suivante avec next_W et next_delta
     * next_W = [[1, 1]], next_delta = [0.5]
     * propagated = transpose(next_W) * next_delta = [0.5, 0.5]
     * deriv = sigmoid_prime(z) = sigmoid_prime([1,2])
     * delta = propagated * deriv  */
    Matrix next_W = matrix_create(1, 2);
    next_W.data[0] = 1; next_W.data[1] = 1;
    Vector next_delta = vector_create(1);
    next_delta.data[0] = 0.5f;

    Vector delta1 = layer_backward_hidden(&L1, &cache1, x, next_W, next_delta);
    /* valeurs attendues calculees a la main :*/
    CHECK(float_eq(delta1.data[0], 0.098306f), "layer_backward_hidden: delta[0]");
    CHECK(float_eq(delta1.data[1], 0.052497f), "layer_backward_hidden: delta[1]");


    CHECK(float_eq(L1.dW.data[0], 0.098306f), "layer_backward_hidden: dW[0][0]");
    CHECK(float_eq(L1.dW.data[1], 0.196612f), "layer_backward_hidden: dW[0][1]");
    CHECK(float_eq(L1.dW.data[2], 0.052497f), "layer_backward_hidden: dW[1][0]");
    CHECK(float_eq(L1.dW.data[3], 0.104994f), "layer_backward_hidden: dW[1][1]");
    CHECK(float_eq(L1.db.data[0], 0.098306f) && float_eq(L1.db.data[1], 0.052497f),
          "layer_backward_hidden: db = delta");

    matrix_free(&next_W);
    vector_free(&next_delta);
    vector_free(&x);
    layer_cache_free(&cache1);
    layer_free(&L1);

    /* --- layer_backward_output : couche de sortie a un seul neurone
     * W = [1, 1], b = 0, prev_a = [1, 2]
     * z = 1*1 + 1*2 = 3, a = sigmoid(3)
     * loss_grad = [1] (valeur arbitraire simple pour le test)
     * deriv = sigmoid_prime(3) ~= 0.045177
     * delta = loss_grad * deriv = 0.045177 */
    Layer L2 = layer_create(2, 1, ACTIVATION_SIGMOID);
    L2.W.data[0] = 1; L2.W.data[1] = 1;
    L2.b.data[0] = 0;

    Vector prev_a = vector_create(2);
    prev_a.data[0] = 1; prev_a.data[1] = 2;

    LayerCache cache2 = layer_cache_create();
    layer_forward(&L2, prev_a, &cache2);

    Vector loss_grad = vector_create(1);
    loss_grad.data[0] = 1.0f;

    Vector delta2 = layer_backward_output(&L2, &cache2, prev_a, loss_grad);
    CHECK(float_eq(delta2.data[0], 0.045177f), "layer_backward_output: delta correct");

    /* dW = delta * prev_a : dW[0]=0.045177*1, dW[1]=0.045177*2 */
    CHECK(float_eq(L2.dW.data[0], 0.045177f), "layer_backward_output: dW[0]");
    CHECK(float_eq(L2.dW.data[1], 0.090353f), "layer_backward_output: dW[1]");
    CHECK(float_eq(L2.db.data[0], 0.045177f), "layer_backward_output: db = delta");

    vector_free(&prev_a);
    vector_free(&loss_grad);
    layer_cache_free(&cache2);
    layer_free(&L2);

    printf("== %d/%d tests passes ==\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}