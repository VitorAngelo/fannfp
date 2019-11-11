/*

  Fast Artificial Neural Network Library - Floating Point Tests Version
  Copyright (C) 2017-2019 Vitor Angelo (vitorangelo@gmail.com)
  
  This library is free software; you can redistribute it and/or
  modify it under the terms of the GNU Lesser General Public
  License v2.1 as published by the Free Software Foundation.
  
  This library is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
  Lesser General Public License for more details.
  
  You should have received a copy of the GNU Lesser General Public
  License along with this library; if not, write to the Free Software
  Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA

  This library was based on the Fast Artificial Neural Network Library.
  See README.md for details.
 
*/

/* Easy way to allow for build of multiple binaries with inlined functions */

#include "doublefann.h"

#ifdef FANN_SQRT_EMULATION
#include "fann_sqrt.c"
#endif
#ifdef FANN_EXP_EMULATION
#include "fann_exp.c"
#endif
#include "fann.c"
#include "fann_io.c"
#include "fann_train.c"
#include "fann_train_data.c"
#include "fann_error.c"
#include "fann_mem.c"
#include "fann_activation.c"
#include "fann_const.c"

const char * fann_float_type = "DOUBLE";

#ifndef FANN_INFERENCE_ONLY
FANN_EXTERNAL void FANN_API fann_gradient_check(struct fann *ann, fann_type_ff * input, fann_type_ff * output)
{
    struct fann_neuron *neuron_it;
    unsigned int n, g, t, prev_neurons;
    struct fann_layer *layer_it, *last_layer, *prev_layer;
    unsigned int num_neurons;
    int softmax = 0, out = 0, max = 0;
    double loss = 0.0, *prev_values;
    double epsilon, restore_in;
    static double restore_out[1000];
    static double deriv[1000], deriv_loss;
    static double check[1000], check_loss;

    /* first set the input *
    //printf("input: ");
    layer_it = ann->first_layer;
    for (n = 0; n < ann->num_input; n++) {
        layer_it->value[n] = fann_float_to_ff(input[n]);
        //printf("%+le ", layer_it->value[n]);
    }*/
    //printf("\n");
    prev_layer = ann->first_layer;
    last_layer = ann->last_layer;
    prev_layer->value = input;
    epsilon = 0.0001;
    for (layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++) {
        if ((layer_it + 1) == last_layer) {
            out = 1;
            loss = 0.0;
            deriv_loss = 0.0;
            check_loss = 0.0;
            if (layer_it->activation == FANN_SOFTMAX) {
                // only in the last layer...
                softmax = 1;
            }
        }
        prev_values = prev_layer->value;
        prev_neurons = prev_layer->num_neurons;
        num_neurons = layer_it->num_neurons; // exclude BIAS
        for (g = 0; g < prev_neurons; g++) {
            //printf("g=%u\n", g);
            restore_in = prev_values[g];
            for (t = 0; t < 3; t++) {
                //printf("t=%u\n", t);
                if (t == 1) {
                    //printf("eps -> %+le + %+le\n", restore_in, epsilon);
                    prev_values[g] = restore_in + epsilon;
                } else if (t == 2) {
                    //printf("eps -> %+le + %+le\n", restore_in, epsilon);
                    prev_values[g] = restore_in - epsilon;
                }
                fann_run_layer(layer_it, prev_layer);
                loss = max = 0;
                if (out) {
                    fann_reset_loss(ann);
                    for (n = 0; n < num_neurons; n++) {
                        if (softmax) {
                            if (output[n] == 0.0)
                                continue;
                            max = n;
                            loss -= output[n] * log(layer_it->value[n]);
                            //printf("max = %u -> %+le * %+le = tot = %+le\n", max, output[n], log(layer_it->value[n]), tot);
#ifdef CALCULATE_ERROR
                        } else {
                            fann_update_er_loss(ann, (uint_fast8_t)(output[n] < 0.5), (layer_it->value[n] - output[n]));
#endif
                        }
                    }
                    if (!softmax) {
                        loss = fann_get_loss(ann);
                    }
                    if (t == 0) {
                        deriv_loss = 0.0;
                    } else if (t == 1) {
                        check_loss = loss;
                    } else if (t == 2) {
                        double div;
                        check_loss = (check_loss - loss) / (epsilon * 2.0);
                        deriv_loss /= num_neurons; // ann->loss_count
                        div = fann_bp_max(fabs(deriv_loss), fabs(check_loss));
                        printf("L=%d        LOSS: D=%+le C=%+le ER=%+lf\n",
                               (int)(layer_it - ann->first_layer),
                               deriv_loss, check_loss,
                               fabs(deriv_loss-check_loss)/div);
                    }
                }
                for (n = 0; n < num_neurons; n++) {
                    //if (softmax && (n != max))
                    //    continue;
                    neuron_it = layer_it->neuron + n;
                    if (t == 0) {
                        if (softmax) {
                            //deriv[n] = neuron_it->steepness * (layer_it->value[n] - output[n]);
                            unsigned int l;
                            deriv[n] = 0.0;
                            for (l = 0; l < num_neurons; l++) {
                                if (l == n) {
                                    deriv[n] += neuron_it->steepness * (layer_it->value[n] * (1.0 - layer_it->value[n]));
                                    printf("%u = %+le\n", l, neuron_it->steepness * (layer_it->value[n] * (1.0 - layer_it->value[n])));
                                } else {
                                    deriv[n] -= neuron_it->steepness * (layer_it->value[n] * layer_it->value[l]);
                                    printf("%u = %+le\n", l, neuron_it->steepness * (layer_it->value[n] * layer_it->value[l]));
                                }
                            }
                        } else {
                            deriv[n] = 0.0;//fann_activation_derived(layer_it->activation,
                                           //                    neuron_it->steepness,
                                           //                    layer_it->value[n]) * neuron_it->weight[g];
                        }
                        restore_out[n] = layer_it->value[n];
                        if (out) {
                            if (softmax) {
                                deriv_loss -= neuron_it->steepness * (layer_it->value[n] - output[n]);
                            } else {
                                deriv_loss += deriv[n] * (layer_it->value[n] - output[n]);
                            }
                        }
                    } else if (t == 1) {
                        check[n] = layer_it->value[n];
                    } else if (t == 2) {
                        double div;
                        check[n] = (check[n] - layer_it->value[n]) / (epsilon * 2.0);
                        div = fann_bp_max(fabs(deriv[n]), fabs(check[n]));
                        printf("L=%d ", (int)(layer_it - ann->first_layer));
                        if (fabs((deriv[n]-check[n])/div) > 0.01) {
                            printf("CHECK ERROR: D=%+le C=%+le V=%+le N=%u W=%u ER=%+le\n",
                                    deriv[n], check[n], restore_out[n], n, g,
                                    fabs((deriv[n]-check[n])/div));
                        } else {
                            printf("   CHECK OK: D=%+le C=%+le V=%+le\n", deriv[n], check[n], restore_out[n]);
                        }
                        layer_it->value[n] = restore_out[n];
                    }
                }
            }
            prev_values[g] = restore_in;
        }
        prev_layer = layer_it;
        softmax = 0;
    }
}
#endif // FANN_INFERENCE_ONLY

