#include "fann_trained.h"

struct fann *fann_create_trained(void)
{
    const unsigned int num_layers = 3;
    const unsigned int layer_sizes[3*2] = {2, 2, 3, 3, 1, 3};
    const float steepness[] = {0x1p+0, 0x1p+0, 0x1p+0, 0x1p+0};
    const float weight[] = {
 0x1.005842p+2,
 -0x1.f14a14p+1,
 0x1.4c9bf4p+2,
 -0x1.53cacp+1,
 -0x1.0f5a5cp+2,
 0x1.75cf82p+1,
 -0x1.915f32p+1,
 -0x1.a0786ep+0,
 -0x1.1343ep+1,
 -0x1.696cf6p+0,
 0x1.6add2p+1,
 -0x1.7031b8p+1,
 -0x1.746c9cp+0,
    };
    struct fann_layer *layer_it, *prev_layer;
    struct fann_neuron *neuron_it;
    struct fann *ann;
    int i, idx;

    fann_const_init();
    ann = fann_allocate_structure(num_layers);
    if (ann == NULL) {
        return NULL;
    }
    idx = 0;
    for (layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++) {
        layer_it->neuron = NULL;
        layer_it->num_neurons = layer_sizes[idx++];
        layer_it->num_connections = layer_it->num_neurons + 1;
        layer_it->activation = layer_sizes[idx++];
    }
    ann->num_input = ann->first_layer->num_neurons;
    ann->num_output = ((ann->last_layer - 1)->num_neurons);
    if (fann_allocate_neurons(ann, NULL)) {
        fann_destroy(ann);
        return NULL;
    }
    // "neurons (num_inputs, activation_steepness)"
    idx = 0;
    prev_layer = NULL;
    for (layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++) {
        unsigned int num_n = layer_it->num_neurons;
        for (i = 0; i < num_n; i++) {
            neuron_it = layer_it->neuron + i;
            neuron_it->steepness = fann_float_to_ff(steepness[idx++]);
        }
        layer_it->value[i] = ff_p100;
        prev_layer = layer_it;

    }
    // "connections (layer, connected_to_neuron, weight)"
    idx = 0;
    prev_layer = ann->first_layer;
    for (layer_it = prev_layer + 1; layer_it != ann->last_layer; layer_it++) {
        unsigned int w, num_con;
        /* the neurons */
        num_con = prev_layer->num_connections;
        for (i = 0; i < layer_it->num_neurons; i++) {
            neuron_it = layer_it->neuron + i;
            for (w = 0; w < num_con; w++) {
                neuron_it->weight[w] = fann_float_to_ff(weight[idx++]);
            }
        }
        prev_layer = layer_it;

    } 
    return ann;
}

/*
#include <stdio.h>

int main(int argc, char *argv[])
{
    float f1 = 0x1.921fb6p+1;
    float f2 = 3.14159265359;
    //         3.14159274101257324218750000000000

    printf("%a\n%.32f\n%.32f\n", f2, f1, f2);
    return 0;
}
*/

