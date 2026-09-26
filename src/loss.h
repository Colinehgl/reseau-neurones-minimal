#ifndef LOSS_H
#define LOSS_H
#include "matrix.h"

scalar_t mse_loss(Vector prediction, Vector target);
Vector mse_loss_prime(Vector prediction, Vector target);

scalar_t cross_entropy_loss(Vector prediction, Vector target);
Vector cross_entropy_loss_prime(Vector prediction, Vector target);

#endif