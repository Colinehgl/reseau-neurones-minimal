#include "dataset.h"
#include "matrix.h"

void get_xor_dataset(Vector* inputs, Vector* targets){
    //paramètres adaptés pour le xor  
    scalar_t x_vals[4][2]   = {{0,0},{0,1},{1,0},{1,1}};
    scalar_t y_vals[4]      = {0,1,1,0};
    for(int i = 0; i < 4; i++){
        inputs[i]           = vector_create(2);
        inputs[i].data[0]   = x_vals[i][0];
        inputs[i].data[1]   = x_vals[i][1];
        targets[i]          = vector_create(1);
        targets[i].data[0]  = y_vals[i];
    }
}


