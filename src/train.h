#ifndef TRAIN_H
#define TRAIN_H
#include "layer.h"
#include "mlp.h"

void sgd_update(Layer* l, scalar_t learning_rate);
void train_epoch(MLP* mlp, Vector* inputs, Vector* targets, int n_examples, scalar_t learning_rate);

#endif