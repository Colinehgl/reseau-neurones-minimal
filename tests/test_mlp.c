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
    if (cond) { tests_passed++; printf("[OK]   %s\n", name); } \
    else      { printf("[FAIL] %s (ligne %d)\n", name, __LINE__); } \
} while (0)

/* tolerance plus large ici : plusieurs etapes enchainees a la main,
 * les petits arrondis s'accumulent un peu plus que dans les tests isoles */
static int float_eq(scalar_t a, scalar_t b) {
    return fabsf(a - b) < 5e-3f;
}

int main(void) {
    printf("== Tests mlp.c ==\n");

    /* reseau 2 -> 2 -> 1, sigmoid partout */
    int sizes[3] = {2, 2, 1};
    ActivationType acts[2] = {ACTIVATION_SIGMOID, ACTIVATION_SIGMOID};
    MLP net = mlp_create(sizes, acts, 2);

    CHECK(net.n_layers == 2, "mlp_create: nombre de couches");
    CHECK(net.layers[0].W.rows == 2 && net.layers[0].W.cols == 2, "mlp_create: taille W couche 0");
    CHECK(net.layers[1].W.rows == 1 && net.layers[1].W.cols == 2, "mlp_create: taille W couche 1");

    /* on ecrase les poids initialises aleatoirement par des valeurs connues
     * couche 0 : W = identite, b = 0 (deja a zero par defaut)
     * couche 1 : W = [1, 1],   b = 0 */
    net.layers[0].W.data[0] = 1; net.layers[0].W.data[1] = 0;
    net.layers[0].W.data[2] = 0; net.layers[0].W.data[3] = 1;
    net.layers[1].W.data[0] = 1; net.layers[1].W.data[1] = 1;

    Vector x = vector_create(2);
    x.data[0] = 1; x.data[1] = 2;

    /* --- mlp_forward ---
     * couche 0 : z0 = [1,2], a0 = [sigmoid(1), sigmoid(2)] = [0.731059, 0.880797]
     * couche 1 : z1 = a0[0]+a0[1] = 1.611856, a1 = sigmoid(1.611856) ~= 0.833668 */
    Vector prediction = mlp_forward(&net, x);
    CHECK(prediction.dim == 1, "mlp_forward: dimension de sortie");
    CHECK(float_eq(prediction.data[0], 0.833668f), "mlp_forward: valeur de sortie");
    CHECK(float_eq(net.caches[0].a.data[0], 0.731059f) &&
          float_eq(net.caches[0].a.data[1], 0.880797f),
          "mlp_forward: activation intermediaire (couche 0) coherente");

    /* --- mlp_zero_grad : verifie juste que dW repasse a zero --- */
    net.layers[0].dW.data[0] = 42.0f;
    mlp_zero_grad(&net);
    CHECK(float_eq(net.layers[0].dW.data[0], 0.0f), "mlp_zero_grad: dW remis a zero");

    /* --- mlp_backward ---
     * target = 1.0, loss_grad = mse_loss_prime([0.833668], [1.0]) = 2*(0.833668-1) = -0.332664 */
    Vector target = vector_create(1);
    target.data[0] = 1.0f;
    Vector loss_grad = mse_loss_prime(prediction, target);

    mlp_backward(&net, x, loss_grad);

    /* valeurs calculees a la main (voir echange precedent pour le detail) :
     * delta1 (couche de sortie)  ~= -0.046130
     * dW1 = delta1 * a0          ~= [-0.033727, -0.040634]
     * delta0 (couche cachee)     ~= [-0.009070, -0.004843]
     * dW0 = delta0 (produit exterieur) x = [[-0.009070,-0.018140],[-0.004843,-0.009686]] */
    CHECK(float_eq(net.layers[1].db.data[0], -0.046130f), "mlp_backward: db couche 1 (delta sortie)");
    CHECK(float_eq(net.layers[1].dW.data[0], -0.033727f), "mlp_backward: dW couche 1 [0]");
    CHECK(float_eq(net.layers[1].dW.data[1], -0.040634f), "mlp_backward: dW couche 1 [1]");

    CHECK(float_eq(net.layers[0].db.data[0], -0.009070f) &&
          float_eq(net.layers[0].db.data[1], -0.004843f),
          "mlp_backward: db couche 0 (delta propage)");
    CHECK(float_eq(net.layers[0].dW.data[0], -0.009070f), "mlp_backward: dW couche 0 [0][0]");
    CHECK(float_eq(net.layers[0].dW.data[1], -0.018140f), "mlp_backward: dW couche 0 [0][1]");
    CHECK(float_eq(net.layers[0].dW.data[2], -0.004843f), "mlp_backward: dW couche 0 [1][0]");
    CHECK(float_eq(net.layers[0].dW.data[3], -0.009686f), "mlp_backward: dW couche 0 [1][1]");

    vector_free(&loss_grad);
    vector_free(&target);
    vector_free(&x);
    mlp_free(&net);

    printf("== %d/%d tests passes ==\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}