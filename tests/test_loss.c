#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "loss.h"

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
    printf("== Tests loss.c ==\n");

    Vector prediction = vector_create(3);
    Vector target = vector_create(3);
    prediction.data[0] = 0.5f; prediction.data[1] = 0.2f; prediction.data[2] = 0.8f;
    target.data[0]     = 1.0f; target.data[1]     = 0.0f; target.data[2]     = 1.0f;

    /* --- mse_loss ---
     * diffs : -0.5, 0.2, -0.2
     * carres : 0.25, 0.04, 0.04 -> somme = 0.33 -> / 3 = 0.11 */
    scalar_t mse = mse_loss(prediction, target);
    CHECK(float_eq(mse, 0.11f), "mse_loss: valeur correcte");

    /* --- mse_loss_prime ---
     * (2/3) * diff pour chaque element */
    Vector mse_grad = mse_loss_prime(prediction, target);
    CHECK(mse_grad.dim == 3, "mse_loss_prime: dimension correcte");
    CHECK(float_eq(mse_grad.data[0], -0.333333f), "mse_loss_prime: grad[0]");
    CHECK(float_eq(mse_grad.data[1], 0.133333f),  "mse_loss_prime: grad[1]");
    CHECK(float_eq(mse_grad.data[2], -0.133333f), "mse_loss_prime: grad[2]");
    vector_free(&mse_grad);

    /* --- cross_entropy_loss ---
     * -(1*ln(0.5) + 0*ln(0.2) + 1*ln(0.8)) = -(-0.693147 - 0.223144) = 0.916291 */
    scalar_t ce = cross_entropy_loss(prediction, target);
    CHECK(float_eq(ce, 0.916291f), "cross_entropy_loss: valeur correcte");

    /* --- cross_entropy_loss_prime ---
     * -(target[i]/prediction[i]) */
    Vector ce_grad = cross_entropy_loss_prime(prediction, target);
    CHECK(ce_grad.dim == 3, "cross_entropy_loss_prime: dimension correcte");
    CHECK(float_eq(ce_grad.data[0], -2.0f), "cross_entropy_loss_prime: grad[0]");
    CHECK(float_eq(ce_grad.data[1], 0.0f),  "cross_entropy_loss_prime: grad[1]");
    CHECK(float_eq(ce_grad.data[2], -1.25f),"cross_entropy_loss_prime: grad[2]");
    vector_free(&ce_grad);

    /* --- cas simple : prediction == target -> loss nulle (MSE uniquement,
     * la cross-entropy n'est nulle que si target est un one-hot parfait) --- */
    Vector perfect = vector_create(2);
    Vector goal = vector_create(2);
    perfect.data[0] = 0.3f; perfect.data[1] = 0.7f;
    goal.data[0]    = 0.3f; goal.data[1]    = 0.7f;
    CHECK(float_eq(mse_loss(perfect, goal), 0.0f), "mse_loss: nulle si prediction = target");
    vector_free(&perfect);
    vector_free(&goal);

    vector_free(&prediction);
    vector_free(&target);

    printf("== %d/%d tests passes ==\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}