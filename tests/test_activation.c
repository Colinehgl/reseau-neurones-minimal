#include <stdio.h>
#include <math.h>
#include "matrix.h"
#include "activation.h"

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
    printf("== Tests activation.c ==\n");

    /* --- sigmoid --- */
    CHECK(float_eq(sigmoid(0.0f), 0.5f), "sigmoid(0) = 0.5");
    CHECK(float_eq(sigmoid_prime(0.0f), 0.25f), "sigmoid_prime(0) = 0.25");
    CHECK(float_eq(sigmoid(2.0f), 0.8807970f), "sigmoid(2) valeur connue");

    /* --- tanh --- */
    CHECK(float_eq(tanh_activation(0.0f), 0.0f), "tanh(0) = 0");
    CHECK(float_eq(tanh_activation_prime(0.0f), 1.0f), "tanh_prime(0) = 1");
    CHECK(float_eq(tanh_activation(1.0f), 0.7615942f), "tanh(1) valeur connue");
    CHECK(float_eq(tanh_activation_prime(1.0f), 0.4199743f), "tanh_prime(1) valeur connue");

    /* --- relu --- */
    CHECK(float_eq(relu(-3.0f), 0.0f), "relu(-3) = 0");
    CHECK(float_eq(relu(5.0f), 5.0f), "relu(5) = 5");
    CHECK(float_eq(relu(0.0f), 0.0f), "relu(0) = 0");
    CHECK(float_eq(relu_prime(-1.0f), 0.0f), "relu_prime(-1) = 0");
    CHECK(float_eq(relu_prime(1.0f), 1.0f), "relu_prime(1) = 1");
    CHECK(float_eq(relu_prime(0.0f), 0.0f), "relu_prime(0) = 0 (convention)");

    /* --- versions Vector : sigmoid --- */
    Vector v = vector_create(3);
    v.data[0] = 0.0f; v.data[1] = 2.0f; v.data[2] = -2.0f;

    Vector sv = sigmoid_vector(v);
    CHECK(sv.dim == 3, "sigmoid_vector: dimension preservee");
    CHECK(float_eq(sv.data[0], 0.5f), "sigmoid_vector: element 0");
    CHECK(float_eq(sv.data[1], 0.8807970f), "sigmoid_vector: element 1");
    CHECK(float_eq(sv.data[2], 0.1192029f), "sigmoid_vector: element 2");
    vector_free(&sv);

    Vector spv = sigmoid_prime_vector(v);
    CHECK(float_eq(spv.data[0], 0.25f), "sigmoid_prime_vector: element 0");
    vector_free(&spv);

    /* --- versions Vector : tanh --- */
    Vector tv = tanh_vector(v);
    CHECK(float_eq(tv.data[0], 0.0f), "tanh_vector: element 0");
    vector_free(&tv);

    Vector tpv = tanh_prime_vector(v);
    CHECK(float_eq(tpv.data[0], 1.0f), "tanh_prime_vector: element 0");
    vector_free(&tpv);

    /* --- versions Vector : relu --- */
    Vector rv = relu_vector(v);
    CHECK(float_eq(rv.data[0], 0.0f) && float_eq(rv.data[1], 2.0f) && float_eq(rv.data[2], 0.0f),
          "relu_vector: valeurs correctes");
    vector_free(&rv);

    Vector rpv = relu_prime_vector(v);
    CHECK(float_eq(rpv.data[1], 1.0f) && float_eq(rpv.data[2], 0.0f),
          "relu_prime_vector: valeurs correctes");
    vector_free(&rpv);
    vector_free(&v);

    /* --- versions Matrix : sigmoid  --- */
    Matrix m = matrix_create(2, 2); 
    Matrix sm = sigmoid_matrix(m);
    CHECK(sm.rows == 2 && sm.cols == 2, "sigmoid_matrix: dimensions preservees");
    CHECK(float_eq(sm.data[0], 0.5f) && float_eq(sm.data[3], 0.5f), "sigmoid_matrix: valeurs = 0.5");
    matrix_free(&sm);

    Matrix spm = sigmoid_prime_matrix(m);
    CHECK(float_eq(spm.data[0], 0.25f), "sigmoid_prime_matrix: valeur = 0.25");
    matrix_free(&spm);

    /* --- versions Matrix : tanh --- */
    Matrix tm = tanh_matrix(m);
    CHECK(float_eq(tm.data[0], 0.0f), "tanh_matrix: valeur = 0");
    matrix_free(&tm);

    Matrix tpm = tanh_prime_matrix(m);
    CHECK(float_eq(tpm.data[0], 1.0f), "tanh_prime_matrix: valeur = 1");
    matrix_free(&tpm);
    matrix_free(&m);

    /* --- versions Matrix : relu s --- */
    Matrix m2 = matrix_create(2, 2);
    m2.data[0] = -1.0f; m2.data[1] = 3.0f; m2.data[2] = 0.0f; m2.data[3] = -5.0f;

    Matrix rm = relu_matrix(m2);
    CHECK(float_eq(rm.data[0], 0.0f) && float_eq(rm.data[1], 3.0f) &&
          float_eq(rm.data[2], 0.0f) && float_eq(rm.data[3], 0.0f),
          "relu_matrix: valeurs correctes");
    matrix_free(&rm);

    Matrix rpm = relu_prime_matrix(m2);
    CHECK(float_eq(rpm.data[0], 0.0f) && float_eq(rpm.data[1], 1.0f) &&
          float_eq(rpm.data[3], 0.0f),
          "relu_prime_matrix: valeurs correctes");
    matrix_free(&rpm);
    matrix_free(&m2);

    printf("== %d/%d tests passes ==\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}