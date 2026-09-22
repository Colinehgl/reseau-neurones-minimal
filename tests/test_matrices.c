#include <stdio.h>
#include <math.h>
#include "matrix.h"

/* Suppose les signatures corrigées :
 *   void matrix_free(Matrix *m);
 *   void vector_free(Vector *v);
 *   Vector matrix_vec_mul(Matrix m, Vector v);   (renommé depuis matrix_vex_mul)
 * Adapte les appels si tu gardes d'autres noms/signatures. */

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

    /* --- matrix_create / indexation --- */
    Matrix m = matrix_create(2, 3);
    CHECK(m.rows == 2 && m.cols == 3, "matrix_create: dimensions");
    CHECK(m.data != NULL, "matrix_create: data alloue");
    CHECK(float_eq(m.data[0], 0.0f), "matrix_create: initialise a zero");

    m.data[1*m.cols + 2] = 6.0f;
    CHECK(float_eq(m.data[1*m.cols + 2], 6.0f), "indexation row-major OK");
    matrix_free(&m);

    /* --- matrix_mul : A (2x3) x B (3x2) = C (2x2), calcule a la main --- */
    Matrix A = matrix_create(2, 3);
    Matrix B = matrix_create(3, 2);
    scalar_t a_vals[6] = {1,2,3,4,5,6};
    scalar_t b_vals[6] = {7,8,9,10,11,12};
    for (int i = 0; i < 6; i++) A.data[i] = a_vals[i];
    for (int i = 0; i < 6; i++) B.data[i] = b_vals[i];

    Matrix C = matrix_mul(A, B);
    /* C[0][0]=1*7+2*9+3*11=58   C[0][1]=1*8+2*10+3*12=64
     * C[1][0]=4*7+5*9+6*11=139  C[1][1]=4*8+5*10+6*12=154 */
    CHECK(C.rows == 2 && C.cols == 2, "matrix_mul: dimensions du resultat");
    CHECK(float_eq(C.data[0], 58.0f),  "matrix_mul: C[0][0]");
    CHECK(float_eq(C.data[1], 64.0f),  "matrix_mul: C[0][1]");
    CHECK(float_eq(C.data[2], 139.0f), "matrix_mul: C[1][0]");
    CHECK(float_eq(C.data[3], 154.0f), "matrix_mul: C[1][1]");
    matrix_free(&C);

    /* --- matrix_transpose : A (2x3) -> At (3x2) --- */
    Matrix At = matrix_transpose(A);
    /* A = [[1,2,3],[4,5,6]] => At = [[1,4],[2,5],[3,6]] */
    CHECK(At.rows == 3 && At.cols == 2, "matrix_transpose: dimensions");
    CHECK(float_eq(At.data[0*At.cols+0], 1.0f), "matrix_transpose: At[0][0]");
    CHECK(float_eq(At.data[0*At.cols+1], 4.0f), "matrix_transpose: At[0][1]");
    CHECK(float_eq(At.data[1*At.cols+0], 2.0f), "matrix_transpose: At[1][0]");
    CHECK(float_eq(At.data[2*At.cols+1], 6.0f), "matrix_transpose: At[2][1]");
    matrix_free(&At);
    matrix_free(&A);
    matrix_free(&B);

    /* --- matrix_add / matrix_hadamard sur 2x2 --- */
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

    /* --- vector_add / vector_hadamard / matrix_outer_product --- */
    Vector va = vector_create(3);
    Vector vb = vector_create(3);
    va.data[0]=2; va.data[1]=3; va.data[2]=4;
    vb.data[0]=5; vb.data[1]=1; vb.data[2]=2;

    Vector vsum = vector_add(va, vb);
    CHECK(float_eq(vsum.data[0],7) && float_eq(vsum.data[1],4) && float_eq(vsum.data[2],6),
          "vector_add: valeurs correctes");
    vector_free(&vsum);

    Vector vhad = vector_hadamard(va, vb);
    /* 2*5=10, 3*1=3, 4*2=8 */
    CHECK(float_eq(vhad.data[0],10) && float_eq(vhad.data[1],3) && float_eq(vhad.data[2],8),
          "vector_hadamard: valeurs correctes");
    vector_free(&vhad);
    vector_free(&va);
    vector_free(&vb);

    /* --- matrix_outer_product : deux vecteurs de dimensions differentes --- */
    Vector oa = vector_create(2);
    Vector ob = vector_create(3);
    oa.data[0]=2; oa.data[1]=3;
    ob.data[0]=5; ob.data[1]=7; ob.data[2]=1;
    Matrix O = matrix_outer_product(oa, ob);
    /* O = [[10,14,2],[15,21,3]] */
    CHECK(O.rows == 2 && O.cols == 3, "matrix_outer_product: dimensions");
    CHECK(float_eq(O.data[0*O.cols+0], 10.0f), "matrix_outer_product: O[0][0]");
    CHECK(float_eq(O.data[0*O.cols+1], 14.0f), "matrix_outer_product: O[0][1]");
    CHECK(float_eq(O.data[0*O.cols+2], 2.0f),  "matrix_outer_product: O[0][2]");
    CHECK(float_eq(O.data[1*O.cols+0], 15.0f), "matrix_outer_product: O[1][0]");
    CHECK(float_eq(O.data[1*O.cols+1], 21.0f), "matrix_outer_product: O[1][1]");
    CHECK(float_eq(O.data[1*O.cols+2], 3.0f),  "matrix_outer_product: O[1][2]");
    vector_free(&oa);
    vector_free(&ob);
    matrix_free(&O);

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
    /* r[0] = 0.5*1 + 1.0*2 + 2.0*0.5 = 3.5
     * r[1] = 1.5*1 + 0.0*2 + 3.0*0.5 = 3.0 */
    CHECK(r.dim == M.rows, "matrix_vec_mul: dimension du resultat = m.rows");
    CHECK(float_eq(r.data[0], 3.5f), "matrix_vec_mul: r[0] (valeur fractionnaire)");
    CHECK(float_eq(r.data[1], 3.0f), "matrix_vec_mul: r[1]");
    vector_free(&r);
    vector_free(&v);
    matrix_free(&M);

    /* --- vector_to_matrix / matrix_to_vector : aller-retour --- */
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