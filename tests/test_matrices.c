#include <stdio.h>
#include <math.h>
#include "matrix.h"

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
    printf("== Tests matrix.c ==\n");

    /* --- matrix_create --- */
    Matrix m = matrix_create(2, 3);
    CHECK(m.rows == 2 && m.cols == 3, "matrix_create: dimensions");
    CHECK(m.data != NULL, "matrix_create: data alloue");
    CHECK(float_eq(m.data[0], 0.0f), "matrix_create: initialise a zero");

    m.data[1*m.cols + 2] = 6.0f;
    CHECK(float_eq(m.data[1*m.cols + 2], 6.0f), "indexation row-major OK");
    matrix_free(&m);

    /* --- matrix_mul --- */
    Matrix A = matrix_create(2, 3);
    Matrix B = matrix_create(3, 2);
    scalar_t a_vals[6] = {1,2,3,4,5,6};
    scalar_t b_vals[6] = {7,8,9,10,11,12};
    for (int i = 0; i < 6; i++) A.data[i] = a_vals[i];
    for (int i = 0; i < 6; i++) B.data[i] = b_vals[i];

    Matrix C = matrix_mul(A, B);
    CHECK(C.rows == 2 && C.cols == 2, "matrix_mul: dimensions du resultat");
    CHECK(float_eq(C.data[0], 58.0f),  "matrix_mul: C[0][0]");
    CHECK(float_eq(C.data[1], 64.0f),  "matrix_mul: C[0][1]");
    CHECK(float_eq(C.data[2], 139.0f), "matrix_mul: C[1][0]");
    CHECK(float_eq(C.data[3], 154.0f), "matrix_mul: C[1][1]");
    matrix_free(&C);

    /* --- matrix_transpose --- */
    Matrix At = matrix_transpose(A);
    CHECK(At.rows == 3 && At.cols == 2, "matrix_transpose: dimensions");
    CHECK(float_eq(At.data[0*At.cols+0], 1.0f), "matrix_transpose: At[0][0]");
    CHECK(float_eq(At.data[0*At.cols+1], 4.0f), "matrix_transpose: At[0][1]");
    CHECK(float_eq(At.data[1*At.cols+0], 2.0f), "matrix_transpose: At[1][0]");
    CHECK(float_eq(At.data[2*At.cols+1], 6.0f), "matrix_transpose: At[2][1]");
    matrix_free(&At);
    matrix_free(&A);
    matrix_free(&B);

    /* --- matrix_add / matrix_hadamard --- */
    Matrix X = matrix_create(2, 2);
    Matrix Y = matrix_create(2, 2);
    scalar_t x_vals[4] = {1, 2, 3, 4};
    scalar_t y_vals[4] = {10, 20, 30, 40};
    for (int i = 0; i < 4; i++) X.data[i] = x_vals[i];
    for (int i = 0; i < 4; i++) Y.data[i] = y_vals[i];

    Matrix S = matrix_add(X, Y);
    CHECK(float_eq(S.data[0], 11.0f) && float_eq(S.data[3], 44.0f), "matrix_add: valeurs correctes");
    matrix_free(&S);

    Matrix H = matrix_hadamard(X, Y);
    CHECK(float_eq(H.data[0], 10.0f) && float_eq(H.data[3], 160.0f), "matrix_hadamard: valeurs correctes");
    matrix_free(&H);
    matrix_free(&X);
    matrix_free(&Y);

    /* --- vector_create / vector_free --- */
    Vector v = vector_create(3);
    CHECK(v.dim == 3, "vector_create: dimension");
    CHECK(float_eq(v.data[0], 0.0f), "vector_create: initialise a zero");

    /* --- matrix_vec_mul : M (2x3) * v (dim 3), avec valeurs fractionnaires
     * pour detecter une accumulation en int au lieu de scalar_t */
    Matrix M = matrix_create(2, 3);
    scalar_t m_vals[6] = {0.5f, 1.0f, 2.0f, 1.5f, 0.0f, 3.0f};
    for (int i = 0; i < 6; i++) M.data[i] = m_vals[i];
    v.data[0] = 1.0f; v.data[1] = 2.0f; v.data[2] = 0.5f;

    Vector r = matrix_vec_mul(M, v);
    CHECK(r.dim == M.rows, "matrix_vec_mul: dimension du resultat = m.rows");
    CHECK(float_eq(r.data[0], 3.5f), "matrix_vec_mul: r[0] (valeur fractionnaire)");
    CHECK(float_eq(r.data[1], 3.0f), "matrix_vec_mul: r[1]");
    vector_free(&r);
    vector_free(&v);
    matrix_free(&M);

    /* --- vector_to_matrix / matrix_to_vector --- */
    Vector w = vector_create(3);
    w.data[0] = 7.0f; w.data[1] = 8.0f; w.data[2] = 9.0f;
    Matrix Wm = vector_to_matrix(w);
    CHECK(Wm.rows == 3 && Wm.cols == 1, "vector_to_matrix: dimensions (colonne)");
    CHECK(float_eq(Wm.data[1], 8.0f), "vector_to_matrix: valeurs preservees");

    Vector w2 = matrix_to_vector(Wm);
    CHECK(w2.dim == 3, "matrix_to_vector: dimension");
    CHECK(float_eq(w2.data[2], 9.0f), "matrix_to_vector: valeurs preservees");
    vector_free(&w);
    vector_free(&w2);
    matrix_free(&Wm);

    printf("== %d/%d tests passes ==\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}