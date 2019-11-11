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

#ifndef FANN_INFERENCE_ONLY
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <math.h>
#endif // FANN_INFERENCE_ONLY

/*
 * Comparison log between IEEE and AP:
 *
 * - added "clip" in derivative: no effect or small perf. decrease.
 * - init prev_step to 10% of max_init: mixed affect, 1 faster conv.  
 * - init prev_step to 1% of max_init: slower and worse
 * - init 10% weig., step 1% weig with min 0.0001
 */

#include "fann.h"

#define DEBUGTRAIN
#undef DEBUGTRAIN

#ifndef FANN_INFERENCE_ONLY
/* INTERNAL FUNCTION
   Helper function to update the loss value and return a diff which takes symmetric functions into account
fann_type_bp */
// TODO: check inverval [0.0,0.1] and [0.9-1.0] for simple and [-1.0,-0.8] and [0.8-1.0] for symmetric
#ifdef CALCULATE_ERROR
static inline void fann_update_er_loss(struct fann *ann, uint_fast8_t low, float diff)
{
#ifdef CALCULATE_LOSS
    double diff2;

    diff2 = diff;
    diff2 *= diff2;

    // thread UNSAFE:
    ann->loss_value += diff2;
    ann->loss_count++;
#endif // CALCULATE_LOSS

    if (fabsf(diff) > ann->bit_fail_limit) {
        if (low)
            ann->num_bit_fail[1]++; // false positive
        else
            ann->num_bit_fail[0]++; // false negative
        return;
    }
    if (low)
        ann->num_bit_ok[0]++; // true negative
    else
        ann->num_bit_ok[1]++; // true positive
}
#else // ! CALCULATE_ERROR
#define fann_update_er_loss(a, l, d)
#endif // ! CALCULATE_ERROR

/* Tests the network.
 */
#ifdef CALCULATE_ERROR
FANN_EXTERNAL fann_type_ff *FANN_API fann_test(struct fann *ann, fann_type_ff * input,
                                            fann_type_ff * desired_output)
{
    fann_type_ff *output_begin = fann_run(ann, input);
    unsigned int u, maxdesidx = 0, maxoutidx = 0;
    fann_type_ff *output_it;
    fann_type_ff maxdes, maxout;
    const fann_type_ff *output_end = output_begin + ann->num_output;
    
    ann->first_layer->value = NULL; // revert temporary pointer set by fann_run
    maxout = output_begin[0];
    maxdes = desired_output[0];
    fann_set_ff_bias();
    /* calculate the error */
    for (u = 0, output_it = output_begin; output_it < output_end; u++, output_it++) {
        fann_update_er_loss(ann, (uint_fast8_t)fann_ff_lt(*desired_output, ff_p050), 
                fann_ff_to_float(fann_ff_sub(*desired_output, *output_it)));
        if (ann->num_output > 1) {
            if (fann_ff_lt(maxdes, *desired_output)) {
                maxdes = *desired_output;
                maxdesidx = u;
            }
            if (fann_ff_lt(maxout, *output_it)) {
                maxout = *output_it;
                maxoutidx = u;
            }
        }
        desired_output++;
    }
    if ((ann->num_output > 1) && (maxoutidx == maxdesidx)) {
        ann->num_max_ok[maxoutidx]++;
    }

    return output_begin;
}
#endif // CALCULATE_ERROR

/* get the mean square error.
 */
FANN_EXTERNAL float FANN_API fann_get_loss(struct fann *ann)
{
#ifdef CALCULATE_LOSS
    if (ann->loss_count > 0) {
        return ann->loss_value / ((double) ann->loss_count * 2.0);
    }
#endif // CALCULATE_LOSS
    return 0.0;
}

/* squared error percentage (as proposed in PROBEN1) */
FANN_EXTERNAL float FANN_API fann_get_sep(struct fann *ann)
{
#ifdef CALCULATE_LOSS
    if (ann->loss_count > 0) {
        const double omax = 1.0;
        double omin;

        switch ((ann->last_layer - 1)->activation) {
        case FANN_SOFTMAX:
        case FANN_SIGMOID:
#ifdef STEPWISE_LUT
        case FANN_SIGMOID_STEPWISE:
#endif
            omin = 0.0;
            break;
        case FANN_SIGMOID_SYMMETRIC:
#ifdef STEPWISE_LUT
        case FANN_SIGMOID_SYMMETRIC_STEPWISE:
#endif
            omin = -1.0;
            break;
        default:
            return 0.0;
        }
        return 100.0 * (omax - omin) * ann->loss_value / ((double) ann->loss_count);
    }
#endif // CALCULATE_LOSS
    return 0.0;
}

FANN_EXTERNAL float FANN_API fann_get_erp(struct fann *ann)
{
#ifdef CALCULATE_LOSS
    if (ann->loss_count > 0) {
        const double omax = 1.0;
        double omin;

        switch ((ann->last_layer - 1)->activation) {
        case FANN_SOFTMAX:
        case FANN_SIGMOID:
#ifdef STEPWISE_LUT
        case FANN_SIGMOID_STEPWISE:
#endif
            omin = 0.0;
            break;
        case FANN_SIGMOID_SYMMETRIC:
#ifdef STEPWISE_LUT
        case FANN_SIGMOID_SYMMETRIC_STEPWISE:
#endif
            omin = -1.0;
            break;
        default:
            return 0.0;
        }
        return 100.0 * ann->loss_value / ((double) ann->loss_count * (double)(omax - omin) * (double)(omax - omin));
    }
#endif // CALCULATE_LOSS
    return 0.0;
}

/*
Precision P = TP / (TP + FP)
Recall R = TP/ (TP + FN)
F = 2 * ((Precision * Recall)/(Precision + Recall)) or F1-score
*/

#ifdef CALCULATE_ERROR
FANN_EXTERNAL float FANN_API fann_get_true_positive(struct fann *ann)
{
    return 100.0 * (float)(ann->num_bit_ok[1]) / (float)(ann->num_bit_ok[1] + ann->num_bit_fail[0]);
}

FANN_EXTERNAL float FANN_API fann_get_true_negative(struct fann *ann)
{
    return 100.0 * (float)(ann->num_bit_ok[0]) / (float)(ann->num_bit_ok[0] + ann->num_bit_fail[1]);
}
#endif // CALCULATE_ERROR

/* reset the mean square error. must access only thread safe buffers.
 */
FANN_EXTERNAL void FANN_API fann_reset_loss(struct fann *ann)
{
    //struct fann_layer * layer_it;
#ifdef CALCULATE_LOSS
    ann->loss_count = 0;
    ann->loss_value = 0.0;
#endif // CALCULATE_LOSS
#ifdef CALCULATE_ERROR
    ann->num_bit_ok[0] = 0;
    ann->num_bit_ok[1] = 0;
    ann->num_bit_fail[0] = 0;
    ann->num_bit_fail[1] = 0;
    if (ann->num_max_ok != NULL) {
        unsigned int u;
        for (u = 0; u < ann->num_output; u++) {
            ann->num_max_ok[u] = 0;
        }
    }
#endif // CALCULATE_ERROR
    /*for (layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++) {
        layer_it->min_abs_sum = ff_p1k5; // MOVE to print stats?
        layer_it->max_abs_sum = ff_0000; // MOVE to print stats?
    }*/
}

static int fann_initialize_errors(struct fann *ann)
{
    struct fann_layer * layer_it;

    /* if no room allocated for the error variables, allocate it now */
    for (layer_it = ann->first_layer + 1; layer_it != (ann->last_layer - 1); layer_it++) {
        /*if (layer_it->train_errors == NULL) {
            fann_calloc(layer_it->train_errors, layer_it->num_neurons);
            if (layer_it->train_errors == NULL) {//|| (layer_it->skip_errors == NULL)) {
                fann_error(FANN_E_CANT_ALLOCATE_MEM);
                return -1;
            }
        } else if (layer_it < (ann->last_layer - 1)) { */
            unsigned int u;
            // last layer errors are overwriten
            for (u = 0; u < layer_it->num_neurons; u++) {
                //layer_it->train_errors[u] = bp_0000;
                layer_it->neuron[u].train_error = bp_0000;//fann_int_to_bp(0, layer_it->neuron[u].bp_fp16_bias);
            }
        //}
    }
    return 0;
}

static void fann_initialize_prev_steps(struct fann *ann, struct fann_layer * layer_it, struct fann_neuron * neuron_it, unsigned int num_connections)
{
    if (neuron_it->prev_steps == NULL) {
        fann_calloc(neuron_it->prev_steps, num_connections);
        if (neuron_it->prev_steps == NULL) {
            fann_error(FANN_E_CANT_ALLOCATE_MEM);
            fann_exit();
        }
    }
    if (ann->training_algorithm == FANN_TRAIN_RPROP) {
        const fann_type_bp step = fann_float_to_bp(fann_nt_to_float(fann_nt_mul(layer_it->max_init, fann_float_to_nt(0.1))));//, neuron_it->bp_fp16_bias);
        do {
            num_connections--;
            // better thyroid, same soybean, slightly slower breast
            neuron_it->prev_steps[num_connections] = step;
        } while(num_connections);
    } else {
        do {
            neuron_it->prev_steps[--num_connections] = bp_0000;//fann_int_to_bp(0, neuron_it->bp_fp16_bias);
        } while(num_connections);
    }
}

static fann_type_bp fann_initialize_prev_slopes_ini; // work around ARM GCC limitation
static void fann_initialize_prev_slopes(/*struct fann *ann,*/ struct fann_neuron * neuron_it,
        unsigned int num_connections)
{
    unsigned int u;
    if (neuron_it->prev_slopes == NULL) {
        fann_calloc(neuron_it->prev_slopes, num_connections);
        if (neuron_it->prev_slopes == NULL) {
            fann_error(FANN_E_CANT_ALLOCATE_MEM);
            fann_exit();
        }
    }
    for (u = 0; u < num_connections; u++) {
        neuron_it->prev_slopes[u] = fann_initialize_prev_slopes_ini;
    }
}
 
/* INTERNAL FUNCTION
   Clears arrays used for training before a new training session.
   Also creates the arrays that do not exist yet.
 */
void fann_clear_train_arrays(struct fann *ann)
{
    struct fann_layer *layer_begin = ann->first_layer + 1;
    struct fann_layer *layer_end = ann->last_layer - 1;
    struct fann_neuron *neuron_it, *last_neuron;
    struct fann_layer *prev_layer;
    unsigned int i, num_connections;

#ifdef DEBUGTRAIN
    fprintf(stderr, "### %s @ %s : %d\n", __FUNCTION__, __FILE__, __LINE__);
#endif
    prev_layer = layer_begin - 1;
    for (; layer_begin <= layer_end; layer_begin++) {
        //layer_begin->tot_delta_delta = bp_0000;
#ifdef DEBUGTRAIN
        fprintf(stderr, "layer %02ld\n", layer_begin - ann->first_layer);
#endif
        // DO NOT update weights in BIAS 'NEURONS'...
        last_neuron = layer_begin->neuron + layer_begin->num_neurons;
        // but include weights to BIAS 'NEURONS'
        num_connections = prev_layer->num_connections;
        for (neuron_it = layer_begin->neuron; neuron_it != last_neuron; neuron_it++) {
            fann_set_bp_bias(neuron_it->bp_fp16_bias);
            if (neuron_it->weight_slopes != NULL) {
                for (i = 0; i < num_connections; i++) {
                    neuron_it->weight_slopes[i] = bp_0000;//fann_int_to_bp(0, neuron_it->bp_fp16_bias);
                }
            }
            if (neuron_it->prev_steps != NULL) {
                fann_initialize_prev_steps(ann, layer_begin, neuron_it, num_connections);
            }
            if (neuron_it->prev_slopes != NULL) {
                for (i = 0; i < num_connections; i++) {
                    neuron_it->prev_slopes[i] = bp_0000;//fann_int_to_bp(0, neuron_it->bp_fp16_bias);
                }
            }
        }
        prev_layer = layer_begin;
    }
}

/* INTERNAL FUNCTION
    compute the error at the network output
    (usually, after forward propagation of a certain input vector, fann_run)
    the error is a sum of squares for all the output units
    also increments a counter because loss is an average of such errors

    After this train_errors in the output layer will be set to:
    neuron_value_derived * (desired_output - neuron_value)
 */
int fann_compute_loss(struct fann *ann, fann_type_ff * desired_output)
{
    fann_type_ff max_desired_val;
    unsigned int max_desired_idx;
#ifdef CALCULATE_ERROR
    unsigned int max_neuron_idx;
    fann_type_ff * first_desired = desired_output;
    fann_type_ff max_neuron_val;
#endif // CALCULATE_ERROR
    //fann_type_bp *train_errors = NULL;
    fann_type_ff *values;
    const struct fann_layer * layer_out = ann->last_layer - 1;
    struct fann_neuron * neuron_it = layer_out->neuron;
    const struct fann_neuron *last_neuron_it = neuron_it + ann->num_output;
    int err;

#ifdef DEBUGTRAIN
    char *errfunc;
    fprintf(stderr, "### %s @ %s : %d\n", __FUNCTION__, __FILE__, __LINE__);
    fprintf(stderr, "layer %02ld\n", layer_out - ann->first_layer);
#endif

    if (fann_initialize_errors(ann))
        return 0;

    max_desired_idx = 0;
    max_desired_val = desired_output[0];
    for (err = 1; err < ann->num_output; err++) {
        if (fann_ff_gt(desired_output[err], max_desired_val)) {
            max_desired_val = desired_output[err];
            max_desired_idx = err;
        }
    }

    /* calculate the error and place it in the output layer */
    //train_errors = layer_out->train_errors;
    values = layer_out->value;
    fann_activation_derive_select(layer_out->activation);
#ifdef CALCULATE_ERROR
    max_neuron_idx = 0;
    max_neuron_val = *values;
#endif // CALCULATE_ERROR
    for (; neuron_it != last_neuron_it; neuron_it++, desired_output++, /*train_errors++,*/ values++)
    {
        fann_set_bp_bias(neuron_it->bp_fp16_bias);
        
        // already gets the negative sign from here
        neuron_it->train_error = fann_ff_to_bp(fann_ff_sub(*desired_output, *values));
#if 0
        if (fann_bp_lt(fann_bp_abs(*train_errors), bp_p01m)) {
            *train_errors = bp_0000;
            continue;
        }
#endif
        if (ann->unbal_er_adjust != NULL) {
            neuron_it->train_error = fann_bp_mul(neuron_it->train_error, fann_ff_to_bp(ann->unbal_er_adjust[max_desired_idx]));
        }
        //*train_errors = fann_bp_mul(*train_errors, fann_float_to_bp(256.0));
        fann_update_er_loss(ann, (uint_fast8_t)fann_ff_lt(*desired_output, ff_p050),
                            fann_bp_to_float(neuron_it->train_error));
#ifdef CALCULATE_ERROR
        if (ann->num_output > 1) {
            if (fann_ff_gt(*values, max_neuron_val)) {
                max_neuron_val = *values;
                max_neuron_idx = desired_output - first_desired;
            }
        }
#endif // CALCULATE_ERROR
#ifdef DEBUGTRAIN
        fann_set_ff_bias();
        fprintf(stderr, "neuron %04ld: target=%+e value=%+e ",
                neuron_it - layer_out->neuron,
                (float)fann_ff_to_float(*desired_output),
                (float)fann_ff_to_float(*values));
        fann_set_bp_bias(neuron_it->bp_fp16_bias);
        fprintf(stderr, "approx_diff=%+e\n",
                (float)fann_bp_to_float(neuron_it->train_error));
        errfunc = "LIN";
#endif
        /*if (ann->train_error_function == FANN_ERRORFUNC_INV_TANH) {
#ifdef DEBUGTRAIN
            errfunc = "TANH";
#endif
            // for x real < 1:
            // tanh-1(x) = 1/2 * ln((1+x)/(1-x))
            // derivative = 1 / (1 - x^2)
            if(fann_bp_lt(*train_errors, fann_float_to_bp(-.99999)))
                *train_errors = fann_float_to_bp(-12.0);
            else if(fann_bp_gt(*train_errors, fann_float_to_bp(.99999)))
                *train_errors = fann_float_to_bp(12.0);
            else
                *train_errors = fann_float_to_bp(log((1.0 + fann_bp_to_float(*train_errors)) /
                                                  (1.0 - fann_bp_to_float(*train_errors))));
        }*/

        fann_derive_info.steepness = fann_ff_to_bp(neuron_it->steepness);
        fann_derive_info.value = fann_ff_to_bp(*values);
        fann_derive_info.error = neuron_it->train_error;
        (*fann_derive_funcptr)();
        neuron_it->train_error = fann_derive_info.error;
#ifdef DEBUGTRAIN
        fann_set_ff_bias();
        fprintf(stderr, "             func=%s, steep=%+le, ",
                errfunc,
                fann_ff_to_float(neuron_it->steepness));
        fann_set_bp_bias(neuron_it->bp_fp16_bias);
        fprintf(stderr, "err=%+le\n",
                fann_bp_to_float(neuron_it->train_error));
#endif

#if (defined SWF16_AP) || (defined HWF16)
        neuron_it->bp_batch_overflows += fann_ap_overflow;
        neuron_it->bp_epoch_overflows += fann_ap_overflow;
#endif
    }
#ifdef CALCULATE_ERROR
    if ((ann->num_output > 1) && (max_desired_idx == max_neuron_idx)) {
        ann->num_max_ok[max_neuron_idx]++;
    }
#endif // CALCULATE_ERROR
    fann_set_ff_bias();
    return 1;
}

/* INTERNAL FUNCTION
   Propagate the error backwards from the output layer.

   After this the train_errors in the hidden layers will be:
   neuron_value_derived * sum(outgoing_weights * connected_neuron)
*/
void fann_backpropagate_loss(struct fann *ann)
{
    unsigned int n, skipped;
    struct fann_layer *layer_it, *prev_layer;
    struct fann_neuron *neuron_it, *last_neuron;
    //fann_type_bp *prev_train_errors, *this_train_errors;
    fann_type_ff *weights;
    const struct fann_layer *second_layer = ann->first_layer + 1;
    struct fann_layer *last_layer = ann->last_layer;

#ifdef DEBUGTRAIN
    fprintf(stderr, "### %s @ %s : %d\n", __FUNCTION__, __FILE__, __LINE__);
#endif
    /* go through all the layers, from last to first.
     * And propagate the error backwards */
    for (layer_it = last_layer - 1; layer_it > second_layer; --layer_it) {
#ifdef DEBUGTRAIN
        fprintf(stderr, "layer %02ld\n", layer_it - ann->first_layer);
#endif
        skipped = 0;
        // DO NOT backpropagate BIAS...
        last_neuron = layer_it->neuron + layer_it->num_neurons;
        prev_layer = layer_it - 1;
        /* for each connection in this layer, propagate the error backwards */
        //prev_train_errors = prev_layer->train_errors;
        //this_train_errors = layer_it->train_errors;
        for (neuron_it = layer_it->neuron; neuron_it != last_neuron; neuron_it++) {//, this_train_errors++) {
#ifdef DEBUGTRAIN
            fprintf(stderr, "neuron %03ld\n", neuron_it - layer_it->neuron);
#endif
            if (fann_bp_is_zero(neuron_it->train_error)) {
                // there is nothing to backpropagate
                skipped++;
                continue;
            }
            weights = neuron_it->weight;
            // no need to calculate BIAS error...
            for (n = prev_layer->num_neurons; n--;) {
                fann_type_bp train_error;
                fann_set_bp_bias(prev_layer->neuron[n].bp_fp16_bias);
#ifdef DEBUGTRAIN
                fann_set_ff_bias();
                fprintf(stderr, "weight = %+le ", fann_ff_to_float(weights[n]));
                fann_set_bp_bias(prev_layer->neuron[n].bp_fp16_bias);
                fprintf(stderr, "prev_train_errors = %+le ", fann_bp_to_float(prev_layer->neuron[n].train_error));
                fprintf(stderr, "prev_train_errors += %+le ", fann_bp_to_float(fann_bp_mul(neuron_it->train_error, fann_ff_to_bp(weights[n]))));
                fprintf(stderr, "[%03d]\n", n);
#endif
#if (defined SWF16_AP) || (defined HWF16)
                // converts this neuron's train_error to the previous neuron format
                train_error = fann_bp_to_bp(neuron_it->train_error, neuron_it->bp_fp16_bias);
#else
                train_error = neuron_it->train_error;
#endif
                prev_layer->neuron[n].train_error = fann_bp_mac(train_error, fann_ff_to_bp(weights[n]), prev_layer->neuron[n].train_error);
#if (defined SWF16_AP) || (defined HWF16)
                prev_layer->neuron[n].bp_batch_overflows += fann_ap_overflow;
                prev_layer->neuron[n].bp_epoch_overflows += fann_ap_overflow;
#endif
            }
        }
        if (skipped < layer_it->num_neurons) {
            /* then calculate the actual errors in the previous layer */
            //prev_train_errors = prev_layer->train_errors;
            // DO NOT backpropagate BIAS...
            last_neuron = prev_layer->neuron + prev_layer->num_neurons;
            neuron_it = prev_layer->neuron;
            fann_activation_derive_select(prev_layer->activation);
            for (n = 0; n < prev_layer->num_neurons; /*prev_train_errors++,*/ n++) {
                fann_set_bp_bias(neuron_it[n].bp_fp16_bias);
                fann_derive_info.steepness = fann_ff_to_bp(neuron_it[n].steepness);
                fann_derive_info.value = fann_ff_to_bp(prev_layer->value[n]);
                fann_derive_info.error = neuron_it[n].train_error;
                (*fann_derive_funcptr)();
                neuron_it[n].train_error = fann_derive_info.error;
#if (defined SWF16_AP) || (defined HWF16)
                neuron_it->bp_batch_overflows += fann_ap_overflow;
                neuron_it->bp_epoch_overflows += fann_ap_overflow;
#endif
#ifdef DEBUGTRAIN
                fprintf(stderr, "neuron %03d -> %+le\n", n, fann_bp_to_float(neuron_it[n].train_error));
#endif
            }
        } else {
            // backpropagation ends in this layer :-(
            break;
        }
    }
    fann_set_ff_bias();
}

/* INTERNAL FUNCTION
   Update weights for incremental training
*/
void fann_update_weights_incremental(struct fann *ann)
{
    struct fann_neuron *neuron_it, *last_neuron;
    fann_type_bp tmp_error, delta_w;//, *train_errors;
    fann_type_ff *weights;
    struct fann_layer *layer_it, *prev_layer;
    unsigned int w, prev_neurons;
//    uint_fast8_t * skip_errors;

    /* store some variables local for fast access */
    const fann_type_ff learning_rate = ann->learning_rate;
    const fann_type_ff learning_momentum = ann->learning_momentum;        
    const struct fann_layer *last_layer = ann->last_layer;
    fann_type_bp *weight_slopes;

#ifdef DEBUGTRAIN
    fprintf(stderr, "### %s @ %s : %d\n", __FUNCTION__, __FILE__, __LINE__);
#endif
    prev_layer = ann->first_layer;
    for (layer_it = (prev_layer + 1); layer_it != last_layer; layer_it++, prev_layer++) {
#ifdef DEBUGTRAIN
        fprintf(stderr, "layer %02ld\n", layer_it - ann->first_layer);
#endif
//        skip_errors = layer_it->skip_errors;
        //train_errors = layer_it->train_errors;
        // DO NOT update weights in BIAS 'NEURONS'...
        last_neuron = layer_it->neuron + layer_it->num_neurons;
        // but include weights to BIAS 'NEURONS'
        prev_neurons = prev_layer->num_neurons;
        for (neuron_it = layer_it->neuron; neuron_it != last_neuron; neuron_it++) {
            fann_set_bp_bias(neuron_it->bp_fp16_bias);
#ifdef DEBUGTRAIN
            fprintf(stderr, "neuron[%ld]\n", neuron_it - layer_it->neuron);
#endif
            tmp_error = fann_bp_mul((neuron_it->train_error), fann_ff_to_bp(learning_rate));
            //train_errors++;
            /*if (ann->postpone_bp && *skip_errors++) {
                continue;
            }*/
            weights = neuron_it->weight;
            weight_slopes = neuron_it->weight_slopes;
            w = prev_neurons;
            //delta_w = fann_bp_add(tmp_error, fann_bp_mul(learning_momentum, weight_slopes[w]));
            delta_w = fann_bp_mac(fann_ff_to_bp(learning_momentum), weight_slopes[w], tmp_error);
            weights[w] = fann_bp_to_ff(fann_bp_add(delta_w, fann_ff_to_bp(weights[w])));
            weight_slopes[w] = delta_w;
#ifdef DEBUGTRAIN
            fprintf(stderr, "bias_delta = %+le\n", fann_bp_to_float(weight_slopes[w]));
#endif
            while (w--) {
                delta_w = fann_bp_add(
                        fann_bp_mul(tmp_error, fann_ff_to_bp(prev_layer->value[w])),
                        fann_bp_mul(fann_ff_to_bp(learning_momentum), weight_slopes[w]));
                weights[w] = fann_bp_to_ff(fann_bp_add(delta_w, fann_ff_to_bp(weights[w])));
                weight_slopes[w] = delta_w;
#ifdef DEBUGTRAIN
                fprintf(stderr, "delta[%u] = %+le\n", w, fann_bp_to_float(weight_slopes[w]));
#endif
            }
#if (defined SWF16_AP) || (defined HWF16)
            neuron_it->bp_batch_overflows += fann_ap_overflow;
            neuron_it->bp_epoch_overflows += fann_ap_overflow;
#endif
        }
    }
    ann->first_layer->value = NULL; // revert temporary pointer set by fann_run
    fann_set_ff_bias();
}

/* INTERNAL FUNCTION
   Update slopes for batch training
   layer_begin = ann->first_layer+1 and layer_end = ann->last_layer-1
   will update all slopes.

*/
void fann_update_slopes_batch(struct fann *ann)
{
    struct fann_layer *layer_begin, *layer_end;
    struct fann_neuron *neuron_it, *last_neuron;
    //fann_type_bp *train_errors;
    unsigned int w, prev_neurons;
    struct fann_layer *prev_layer;
    /* store some variabels local for fast access */
    fann_type_bp *weight_slopes;
    fann_type_ff *values;

    layer_begin = ann->first_layer + 1;
    layer_end = ann->last_layer - 1;

#ifdef DEBUGTRAIN
    fprintf(stderr, "### %s @ %s : %d\n", __FUNCTION__, __FILE__, __LINE__);
#endif
    prev_layer = layer_begin - 1;
    for (; layer_begin <= layer_end; layer_begin++, prev_layer++) {
#ifdef DEBUGTRAIN
        fprintf(stderr, "layer %02ld\n", layer_begin - ann->first_layer);
#endif
        // DO NOT update weights in BIAS 'NEURONS'...
        last_neuron = layer_begin->neuron + layer_begin->num_neurons;
        //train_errors = layer_begin->train_errors;
        // but include weights to BIAS 'NEURONS'
        prev_neurons = prev_layer->num_neurons;
        for (neuron_it = layer_begin->neuron; neuron_it != last_neuron; neuron_it++/*, train_errors++*/) {
            fann_set_bp_bias(neuron_it->bp_fp16_bias);
            if (fann_bp_is_zero(neuron_it->train_error)) {
                continue;
            }
            weight_slopes = neuron_it->weight_slopes;
            w = prev_neurons;
            weight_slopes[w] = fann_bp_add((neuron_it->train_error), weight_slopes[w]);
#ifdef DEBUGTRAIN
            fprintf(stderr, "neuron %ld, error=%+le, wslope[%u]=%+le\n", neuron_it - layer_begin->neuron,
                   fann_bp_to_float(neuron_it->train_error), w, fann_bp_to_float(weight_slopes[w]));
#endif
            values = prev_layer->value;
            while (w--) {
                weight_slopes[w] = fann_bp_mac((neuron_it->train_error), fann_ff_to_bp(values[w]), weight_slopes[w]);
#ifdef DEBUGTRAIN
                fprintf(stderr, "neuron %ld, error=%+le, wslope[%u]=%+le\n", neuron_it - layer_begin->neuron,
                       fann_bp_to_float(neuron_it->train_error), w, fann_bp_to_float(weight_slopes[w]));
#endif
            }
#if (defined SWF16_AP) || (defined HWF16)
            neuron_it->bp_batch_overflows += fann_ap_overflow;
            neuron_it->bp_epoch_overflows += fann_ap_overflow;
#endif
        }
    }
    ann->first_layer->value = NULL; // revert temporary pointer set by fann_run
    fann_set_ff_bias();
}

/* INTERNAL FUNCTION
   Update weights for batch training
 */
void fann_update_weights_batch(struct fann *ann,// unsigned int num_data,
        struct fann_layer *layer_begin, struct fann_layer *layer_end)
{
    unsigned int i, speed, num_connections;
    struct fann_neuron *neuron_it, *last_neuron;
    struct fann_layer *prev_layer;
    fann_type_bp *weight_slopes;//, mac;
    fann_type_bp *prev_steps;
    fann_type_ff *weights;
    fann_type_bp epsilon;// = fann_bp_div(ann->learning_rate, fann_int_to_bp(num_data));
    fann_type_bp momentum;

    if (fann_ff_is_non_zero(ann->learning_momentum)) {
        speed = 1;
        //momentum = ann->learning_momentum;
    } else {
        speed = 0;
        //momentum = bp_0000;
    }

#ifdef DEBUGTRAIN
    int l = 0;
    fprintf(stderr, "### %s @ %s : %d\n", __FUNCTION__, __FILE__, __LINE__);
#endif
    if (layer_begin == NULL) {
        layer_begin = ann->first_layer + 1;
    }
    if (layer_end == NULL) {
        layer_end = ann->last_layer - 1;
    }

    prev_layer = layer_begin - 1;
    for (; layer_begin <= layer_end; prev_layer++, layer_begin++) {
#ifdef DEBUGTRAIN
        fprintf(stderr, "layer %d\n", ++l);
#endif
        // DO NOT update weights in BIAS 'NEURONS'...
        last_neuron = layer_begin->neuron + layer_begin->num_neurons;
        // but include weights to BIAS 'NEURONS'
        num_connections = prev_layer->num_connections;
        for (neuron_it = layer_begin->neuron; neuron_it != last_neuron; neuron_it++) {
            fann_set_bp_bias(neuron_it->bp_fp16_bias);
            //epsilon = fann_bp_div(fann_ff_to_bp(ann->learning_rate), fann_int_to_bp(num_data, neuron_it->bp_fp16_bias));
            epsilon = fann_ff_to_bp(ann->learning_rate);
#ifdef DEBUGTRAIN
            fprintf(stderr, "  neuron %d\n", (int)(neuron_it - layer_begin->neuron));
#endif
            weight_slopes = neuron_it->weight_slopes;
            if ((neuron_it->prev_steps == NULL) && (speed)) {
                fann_initialize_prev_steps(ann, layer_begin, neuron_it, num_connections);
            }
            prev_steps = neuron_it->prev_steps;
            /*if (ann->num_procs != 1) {
                if (pthread_spin_trylock(&(neuron_it->spin)))
                    continue;
                if (neuron_it->step_done) {
                    pthread_spin_unlock(&(neuron_it->spin));
                    continue;
                }
                neuron_it->step_done = 1;
                pthread_spin_unlock(&(neuron_it->spin));
                for (p = 0; p < (ann->num_procs - 1); p++) {
                    struct fann *ann_p = ann->ann[p];
                    fann_type_bp *weight_slopes_p = ann_p->first_layer[l].neuron[n].weight_slopes;
                    for (w = 0; w < num_connections; w++) {
                        weight_slopes[w] = fann_bp_add(weight_slopes_p[w], weight_slopes[w]);
                    }
                }
            }*/
            weights = neuron_it->weight;
            momentum = fann_ff_to_bp(ann->learning_momentum);
            for (i = 0; i < num_connections; i++) {
#ifdef DEBUGTRAIN
                fann_set_ff_bias();
                fprintf(stderr, "    %f + ", fann_ff_to_float(weights[i]));
                fann_set_bp_bias(neuron_it->bp_fp16_bias);
                fprintf(stderr, "(%f * %f) = w[%d]\n", fann_bp_to_float(weight_slopes[i]),
                        fann_bp_to_float(epsilon), i);
#endif
                if (speed) {
                    prev_steps[i] = fann_bp_add(fann_bp_mul(momentum, prev_steps[i]),
                                                    fann_bp_mul(weight_slopes[i], epsilon));
                    weights[i] = fann_bp_to_ff(fann_bp_add(prev_steps[i], fann_ff_to_bp(weights[i])));
                } else {
                    weights[i] = fann_bp_to_ff(fann_bp_mac(weight_slopes[i], epsilon, fann_ff_to_bp(weights[i])));
                }
            }
#if (defined SWF16_AP) || (defined HWF16)
            neuron_it->bp_batch_overflows += fann_ap_overflow;
            neuron_it->bp_epoch_overflows += fann_ap_overflow;
#endif
        }
    }
    fann_set_ff_bias();
}

/* INTERNAL FUNCTION
   Update weights for RMSProp training
 */
void fann_update_weights_rmsprop(struct fann *ann,// unsigned int num_data,
        struct fann_layer *layer_begin, struct fann_layer *layer_end)
{
    static unsigned int debug_fallback_last = 0;
    unsigned int debug_fallback = 0, debug_total = 0;
    fann_type_bp learning_momentum = bp_0000;// = ann->learning_momentum;        
    fann_type_bp *prev_steps = NULL; // momentum memory
    unsigned int i, num_connections;
    struct fann_neuron *neuron_it, *last_neuron;
    struct fann_layer *prev_layer;
    fann_type_bp delta_w;
    fann_type_bp *weight_slopes;//, mac;
    fann_type_bp *prev_slopes; /* average quadratic slope */
    fann_type_ff *weights, epsilon_ff;
    fann_type_bp epsilon;// = fann_bp_div(ann->learning_rate, fann_int_to_bp(num_data));
    fann_type_bp rmsprop_avg;// = ann->rmsprop_avg;
    fann_type_bp rmsprop_1mavg;// = ann->rmsprop_1mavg;

    fann_set_ff_bias();
    epsilon_ff = fann_bp_to_ff(fann_bp_b_rsqrt_a(fann_ff_to_bp(ann->learning_rate),
                                                 fann_int_to_bp(ann->train_epoch))); // saturates

#ifdef DEBUGTRAIN
    int l = 0;
    fprintf(stderr, "### %s @ %s : %d\n", __FUNCTION__, __FILE__, __LINE__);
#endif
    if (layer_begin == NULL) {
        layer_begin = ann->first_layer + 1;
    }
    if (layer_end == NULL) {
        layer_end = ann->last_layer - 1;
    }

    prev_layer = layer_begin - 1;
    for (; layer_begin <= layer_end; prev_layer++, layer_begin++) {
#ifdef DEBUGTRAIN
        fprintf(stderr, "layer %d\n", ++l);
#endif
        // DO NOT update weights in BIAS 'NEURONS'...
        last_neuron = layer_begin->neuron + layer_begin->num_neurons;
        // but include weights to BIAS 'NEURONS'
        num_connections = prev_layer->num_connections;
        for (neuron_it = layer_begin->neuron; neuron_it != last_neuron; neuron_it++) {
            fann_set_bp_bias(neuron_it->bp_fp16_bias);
#ifdef DEBUGTRAIN
            fprintf(stderr, "  neuron %d\n", (int)(neuron_it - layer_begin->neuron));
#endif
            rmsprop_avg = fann_ff_to_bp(ann->rmsprop_avg);
            rmsprop_1mavg = fann_ff_to_bp(ann->rmsprop_1mavg);
            if (neuron_it->prev_slopes == NULL) {
                //fann_initialize_prev_slopes(ann, neuron_it, bp_0000, num_connections);
                fann_initialize_prev_slopes_ini = fann_ff_to_bp(ff_p01m); 
                fann_initialize_prev_slopes(/*ann,*/ neuron_it, /*fann_ff_to_bp(ff_p01m),*/ num_connections);
            }
            if (fann_ff_is_non_zero(ann->learning_momentum)) {
                if (neuron_it->prev_steps == NULL) { // only with momentum
                    fann_initialize_prev_steps(ann, layer_begin, neuron_it, num_connections);
                }
                learning_momentum = fann_ff_to_bp(ann->learning_momentum);
                prev_steps = neuron_it->prev_steps;
            }
            prev_slopes = neuron_it->prev_slopes;
            weight_slopes = neuron_it->weight_slopes;
            /*if (ann->num_procs != 1) {
                if (pthread_spin_trylock(&(neuron_it->spin)))
                    continue;
                if (neuron_it->step_done) {
                    pthread_spin_unlock(&(neuron_it->spin));
                    continue;
                }
                neuron_it->step_done = 1;
                pthread_spin_unlock(&(neuron_it->spin));
                for (p = 0; p < (ann->num_procs - 1); p++) {
                    struct fann *ann_p = ann->ann[p];
                    fann_type_bp *weight_slopes_p = ann_p->first_layer[l].neuron[n].weight_slopes;
                    for (w = 0; w < num_connections; w++) {
                        weight_slopes[w] = fann_bp_add(weight_slopes_p[w], weight_slopes[w]);
                    }
                }
            }*/
            weights = neuron_it->weight;
            //epsilon = fann_bp_div(fann_ff_to_bp(ann->learning_rate), fann_int_to_bp(num_data, neuron_it->bp_fp16_bias));
            //epsilon = fann_ff_to_bp(ann->learning_rate);
            //epsilon = fann_bp_b_rsqrt_a(epsilon, fann_int_to_bp(ann->train_epoch, neuron_it->bp_fp16_bias));
            epsilon = fann_ff_to_bp(epsilon_ff);
            for (i = 0; i < num_connections; i++) {
#ifdef DEBUGTRAIN
                fann_set_ff_bias();
                fprintf(stderr, "    %f + ", fann_ff_to_float(weights[i]));
                fann_set_bp_bias(neuron_it->bp_fp16_bias);
                fprintf(stderr, "(%f * %f) = w[%d]\n", fann_bp_to_float(weight_slopes[i]),
                        fann_bp_to_float(epsilon), i);
#endif
                // the new quadratic slope running average:
                if (fann_bp_is_zero(learning_momentum) || fann_bp_is_non_zero(prev_slopes[i])) { // && (ann->train_epoch < 5)) {
                    delta_w = fann_bp_mul(weight_slopes[i], weight_slopes[i]);
#if 1
                    prev_slopes[i] = fann_bp_mac(rmsprop_1mavg, delta_w,
                                                 fann_bp_mul(rmsprop_avg, prev_slopes[i]));
#else // adagrad
                    prev_slopes[i] = fann_bp_add(delta_w, prev_slopes[i]);
#endif
                }
                if (prev_steps != NULL) { // only with momentum
                    debug_total++;
                    if (fann_bp_is_zero(prev_slopes[i])) {
                        //prev_slopes[i] = delta_w;
                        //delta_w = fann_bp_mul(weight_slopes[i], epsilon); // no momentum
                        delta_w = fann_bp_mul(prev_steps[i], learning_momentum);
                        delta_w = fann_bp_mac(weight_slopes[i], epsilon, delta_w);
                        prev_steps[i] = delta_w;
                        debug_fallback++;
                    } else {
                        delta_w = fann_bp_b_rsqrt_a(fann_bp_mul(weight_slopes[i], epsilon), prev_slopes[i]);
                    }
                } else {
                    if (fann_bp_is_non_zero(prev_slopes[i])) {
                        delta_w = fann_bp_b_rsqrt_a(fann_bp_mul(weight_slopes[i], epsilon), prev_slopes[i]);
                    } else {
                        delta_w = bp_0000;
                    }
                }
                weights[i] = fann_bp_to_ff(fann_bp_add(delta_w, fann_ff_to_bp(weights[i])));
            }
#if (defined SWF16_AP) || (defined HWF16)
            neuron_it->bp_batch_overflows += fann_ap_overflow;
            neuron_it->bp_epoch_overflows += fann_ap_overflow;
#endif
        } // neuron
    } // layer
    fann_set_ff_bias();
    if ((prev_steps != NULL) && (debug_fallback != debug_fallback_last)) {
        debug_fallback_last = debug_fallback;
        fprintf(stdout, "rmsprop: total=%u, fallback=%u\n", debug_total, debug_fallback);
    }
}

/* INTERNAL FUNCTION
   The quickprop training algorithm
 */
#if 0
void fann_update_weights_quickprop(struct fann *ann, unsigned int num_data,
        struct fann_layer *layer_begin, struct fann_layer *layer_end)
{
    fann_type_bp *weight_slopes, *prev_steps, *prev_slopes;
    fann_type_ff *weights;
    struct fann_neuron *neuron_it, *last_neuron;
    struct fann_layer *prev_layer;
    unsigned int i, num_connections;
    
    fann_type_bp w, prev_step;
    fann_type_bp slope, prev_slope;
    fann_type_bp next_step, sub;

    fann_type_bp epsilon;// = fann_bp_div(ann->learning_rate, fann_int_to_bp(num_data));
    fann_type_bp decay;// = ann->quickprop_decay;    /* -0.0001 */
    fann_type_bp mu;// = ann->quickprop_mu;    /* 1.75 */
    fann_type_bp shrink_factor;// = fann_bp_div(mu, fann_bp_add(fann_ff_to_bp(ff_p100), mu)); /* 0.63636 */

#ifdef DEBUGTRAIN
    fprintf(stderr, "### %s @ %s : %d\n", __FUNCTION__, __FILE__, __LINE__);
#endif

    if (layer_begin == NULL) {
        layer_begin = ann->first_layer + 1;
    }
    if (layer_end == NULL) {
        layer_end = ann->last_layer - 1;
    }

    prev_layer = layer_begin - 1;
    for (; layer_begin <= layer_end; layer_begin++, prev_layer++) {
#ifdef DEBUGTRAIN
        fprintf(stderr, "layer %02ld\n", layer_begin - ann->first_layer);
#endif
        // DO NOT update weights in BIAS 'NEURONS'...
        last_neuron = layer_begin->neuron + layer_begin->num_neurons;
        // but include weights to BIAS 'NEURONS'
        num_connections = prev_layer->num_connections;
        for (neuron_it = layer_begin->neuron; neuron_it != last_neuron; neuron_it++) {
#ifdef SWF16_AP
            fann_ap_overflow = 0;
            FP_BIAS = neuron_it->bp_fp16_bias;
#endif
            weight_slopes = neuron_it->weight_slopes;
            if (neuron_it->prev_steps == NULL) { 
                fann_initialize_prev_steps(ann, layer_begin, neuron_it, num_connections);
            }
            prev_steps = neuron_it->prev_steps;
            if (neuron_it->prev_slopes == NULL) {
                fann_initialize_prev_slopes(ann, neuron_it, fann_int_to_bp(0, neuron_it->bp_fp16_bias), num_connections);
            }
            prev_slopes = neuron_it->prev_slopes;
            weights = neuron_it->weight;
            //epsilon = fann_bp_div(fann_ff_to_bp(ann->learning_rate), fann_int_to_bp(num_data, neuron_it->bp_fp16_bias));
            epsilon = fann_ff_to_bp(ann->learning_rate);
            decay = fann_ff_to_bp(ann->quickprop_decay);
            mu = fann_ff_to_bp(ann->quickprop_mu);
            shrink_factor = fann_bp_div(mu, fann_bp_add(fann_ff_to_bp(ff_p100), mu)); /* 0.63636 */

            for (i = 0; i < num_connections; i++) {
                w = fann_ff_to_bp(weights[i]);
                prev_step = prev_steps[i];
                slope = fann_bp_mac(decay, w, weight_slopes[i]);
                prev_slope = prev_slopes[i];
                next_step = fann_int_to_bp(0, neuron_it->bp_fp16_bias);
                /* The step must always be in direction opposite to the slope. */
                if(fann_bp_gt(prev_step, fann_ff_to_bp(ff_p1ml))) {
                    /* If last step was positive...  */
                    if(fann_bp_is_pos(slope)) {/*  Add in linear term if current slope is still positive. */
                        next_step = fann_bp_mac(epsilon, slope, next_step);
                    }
                    /*If current slope is close to or larger than prev slope...  */
                    if(fann_bp_gt(slope, fann_bp_mul(shrink_factor, prev_slope))) {
                        next_step = fann_bp_mac(mu, prev_step, next_step);
                        /* Take maximum size negative step. */
                    } else {
                        sub = fann_bp_sub(prev_slope, slope);
                        if (fann_bp_is_non_zero(sub))
                            next_step = fann_bp_mac(prev_step, fann_bp_div(slope, sub), next_step);
                        /* Else, use quadratic estimate. */
                    }
                } else if(fann_bp_lt(prev_step, fann_ff_to_bp(ff_n1ml))) {
                    /* If last step was negative...  */
                    if (fann_bp_is_neg(slope)) { /*  Add in linear term if current slope is still negative. */
                        next_step = fann_bp_mac(epsilon, slope, next_step);
                    }
                    /* If current slope is close to or more neg than prev slope... */
                    if(fann_bp_lt(slope, fann_bp_mul(shrink_factor, prev_slope))) {
                        next_step = fann_bp_mac(mu, prev_step, next_step);
                        /* Take maximum size negative step. */
                    } else {
                        sub = fann_bp_sub(prev_slope, slope);
                        if (fann_bp_is_non_zero(sub))
                            next_step = fann_bp_mac(prev_step, fann_bp_div(slope, sub), next_step);
                        /* Else, use quadratic estimate. */
                    }
                } else {/* Last step was zero, so use only linear term. */
                    next_step = fann_bp_mac(epsilon, slope, next_step);
                }
                /*
                if(next_step > 1000 || next_step < -1000)
                {
                    fprintf(stderr, "quickprop[%d] weight=%f, slope=%f, prev_slope=%f, next_step=%f, prev_step=%f\n",
                        i, weights[i], slope, prev_slope, next_step, prev_step);
            
                    if(next_step > 1000)
                        next_step = 1000;
                    else
                        next_step = -1000;
                }
                */
                /* update global data arrays */
                prev_steps[i] = next_step;

                weights[i] = fann_bp_to_ff(fann_bp_add(next_step, w));
                        //fann_bp_clip(fann_bp_add(next_step, w),
                        //                      bp_n1k5, bp_p1k5));

                prev_slopes[i] = slope;
            }
#ifdef SWF16_AP
            neuron_it->bp_epoch_overflows += fann_ap_overflow; 
            neuron_it->bp_batch_overflows += fann_ap_overflow; 
#endif
        } // neuron
    } // layer
    fann_set_ff_bias();
}
#endif // 0

/* INTERNAL FUNCTION
   The iRprop- algorithm
*/
void fann_update_weights_irpropm(struct fann *ann)
{
    struct fann_layer *layer_begin, *layer_end;
    //unsigned int count[4] = {0, 0, 0, 0}; double tot;
    fann_type_bp *weight_slopes, *prev_steps, *prev_slopes;
    fann_type_ff *weights;
    fann_type_bp prev_step, slope, next_step, same_sign;

    fann_type_bp increase_factor;// = ann->rprop_increase_factor;    /*1.2; */
    fann_type_bp decrease_factor;// = ann->rprop_decrease_factor;    /*0.5; */

    struct fann_neuron *neuron_it, *last_neuron;
    struct fann_layer *prev_layer;
#ifdef FANN_THREADS
    unsigned int p;
#endif
    unsigned int l, n, w, num_connections;
    fann_type_bp delta_min;// = ann->rprop_delta_min;
    //fann_type_bp delta_max = ann->rprop_delta_max;    /*50.0; */

#ifdef DEBUGTRAIN
    fprintf(stderr, "### %s @ %s : %d\n", __FUNCTION__, __FILE__, __LINE__);
#endif

    prev_layer = ann->first_layer;
    layer_begin = prev_layer + 1;
    layer_end = ann->last_layer - 1;
    for (l = 1; layer_begin <= layer_end; layer_begin++, l++, prev_layer++) {
#ifdef DEBUGTRAIN
        fprintf(stderr, "layer[%d]\n", ++l);
#endif
        // DO NOT update weights in BIAS 'NEURONS'...
        last_neuron = layer_begin->neuron + layer_begin->num_neurons;
        // but include weights to BIAS 'NEURONS'
        num_connections = prev_layer->num_connections;
        for (n = 0, neuron_it = layer_begin->neuron; neuron_it != last_neuron; n++, neuron_it++) {
            fann_set_bp_bias(neuron_it->bp_fp16_bias);
#ifdef DEBUGTRAIN
            fprintf(stderr, "  neuron[%d]\n", (int)(neuron_it-layer_begin->neuron));
#endif
            weight_slopes = neuron_it->weight_slopes;
#ifdef FANN_THREADS
            if (ann->num_procs != 1) {
                if (pthread_spin_trylock(&(neuron_it->spin)))
                    continue;
                if (neuron_it->step_done) {
                    pthread_spin_unlock(&(neuron_it->spin));
                    continue;
                }
                neuron_it->step_done = 1;
                pthread_spin_unlock(&(neuron_it->spin));
                for (p = 0; p < (ann->num_procs - 1); p++) {
                    struct fann *ann_p = ann->ann[p];
                    fann_type_bp *weight_slopes_p = ann_p->first_layer[l].neuron[n].weight_slopes;
                    for (w = 0; w < num_connections; w++) {
                        weight_slopes[w] = fann_bp_add(weight_slopes_p[w], weight_slopes[w]);
                    }
                }
            }
#endif // FANN_THREADS
            if (neuron_it->prev_steps == NULL) {
                fann_initialize_prev_steps(ann, layer_begin, neuron_it, num_connections);
            }
            prev_steps = neuron_it->prev_steps;
            if (neuron_it->prev_slopes == NULL) {
                fann_initialize_prev_slopes_ini = bp_0000; 
                fann_initialize_prev_slopes(/*ann,*/ neuron_it /*, bp_0000 fann_int_to_bp(0, neuron_it->bp_fp16_bias)*/, num_connections);
            }
            prev_slopes = neuron_it->prev_slopes;
            weights = neuron_it->weight;
            increase_factor = fann_ff_to_bp(ann->rprop_increase_factor);
            decrease_factor = fann_ff_to_bp(ann->rprop_decrease_factor);
            delta_min = fann_ff_to_bp(ann->rprop_delta_min);
            for (w = 0; w < num_connections; w++) {
                /* this commit marks a more exact implementation of irprop- algorithm
                 * it improved convergence in the breast cancer, thyroid and soybean datasets (A LOT!)
                 * it also reduced fluctuations after maximum accuracy is reached */
                prev_step = prev_steps[w];
                slope = weight_slopes[w];
                same_sign = fann_bp_mul(prev_slopes[w], slope);

                if (fann_bp_is_pos(same_sign)) {
                    //count[0]++;
                    // No sign change: speed up movement in the correct direction.
                    // Increase speed and change weight according to slope sign.
                    next_step = fann_bp_mul(prev_step, increase_factor);
                    //next_step = fann_bp_min(fann_bp_mul(prev_step, increase_factor), delta_max);
                } else if (fann_bp_is_neg(same_sign)) {
                    //count[1]++;
                    // Sign change. The (-) algorithm does not revert the change. 
                    next_step = fann_bp_mul(prev_step, decrease_factor);
                    //next_step = fann_bp_max(fann_bp_mul(prev_step, decrease_factor), delta_min);
                    slope = bp_0000;//fann_int_to_bp(0, neuron_it->bp_fp16_bias); // save this step for the next iteraction
                } else {
                    //count[2]++;
                    // use the stored step with the current slope
                    next_step = prev_step;
                }
                
#if 1
                if (fann_bp_is_zero(next_step) && fann_bp_is_non_zero(delta_min)) {
                    //count[3]++;
                    // only the current slope matters
                    fann_type_bp wmin = fann_bp_abs(fann_ff_to_bp(weights[w]));
                    //next_step = fann_ff_max(fann_bp_mul(wmin, bp_p001), bp_p01m);//bp_p0000);//
                    next_step = fann_bp_mul(wmin, delta_min);
                }
#endif

                if (fann_bp_is_neg(slope)) {
                    weights[w] = fann_bp_to_ff(fann_bp_sub(fann_ff_to_bp(weights[w]), next_step));
                } else if (fann_bp_is_pos(slope)) {
                    weights[w] = fann_bp_to_ff(fann_bp_add(fann_ff_to_bp(weights[w]), next_step));
                }
                
#ifdef DEBUGTRAIN
                fann_set_ff_bias();
                fprintf(stderr, "    weight[%d]=%f, ", w, fann_ff_to_float(weights[w]));
                fann_set_bp_bias(neuron_it->bp_fp16_bias);
                fprintf(stderr, "slope=%f, next_step=%f, prev_step=%f\n",
                        fann_bp_to_float(slope),
                        fann_bp_to_float(next_step), fann_bp_to_float(prev_step));
#endif

                /* update global data arrays */
                prev_steps[w] = next_step;
                prev_slopes[w] = slope;
            }
#if (defined SWF16_AP) || (defined HWF16)
            neuron_it->bp_batch_overflows += fann_ap_overflow;
            neuron_it->bp_epoch_overflows += fann_ap_overflow;
#endif
        } // neuron
    } // layer
#if 0
    tot = (double)(count[0] + count[1] + count[2]);
    fprintf(stderr, "RProp=%04u,%9.4f,%9.4f,%9.4f,%9.4f\n", ann->train_epoch,
            100.0*(double)count[0]/tot, 100.0*(double)count[1]/tot,
            100.0*(double)count[2]/tot, 100.0*(double)count[3]/tot);
#endif
    fann_set_ff_bias();
}

/* INTERNAL FUNCTION
   The SARprop- algorithm
*
void fann_update_weights_sarprop(struct fann *ann, unsigned int epoch, unsigned int first_weight, unsigned int past_end)
{
#if 1 
    fprintf(stderr, "%s, %d, %s BROKEN!\n", __FILE__, __LINE__, __FUNCTION__);
#else
    fann_type_bp *weight_slopes = ann->weight_slopes;
    fann_type_bp *weights = ann->weights;
    fann_type_bp *prev_steps = ann->prev_steps;
    fann_type_bp *prev_slopes = ann->prev_slopes;

    fann_type_bp prev_step;
    fann_type_bp slope, prev_slope;
    fann_type_bp next_step = fann_int_to_bp(0); 
    fann_type_bp same_sign;

    // These should be set from variables
    fann_type_bp increase_factor = ann->rprop_increase_factor;    //1.2;
    fann_type_bp decrease_factor = ann->rprop_decrease_factor;    //0.5;

    // TODO: why is delta_min 0.0 in iRprop? SARPROP uses 1x10^-6 (Braun and Riedmiller, 1993)
    fann_type_bp delta_min = fann_float_to_bp(0.000001f);
    fann_type_bp delta_max = ann->rprop_delta_max;    //50.0;
    float weight_decay_shift = fann_bp_to_float(ann->sarprop_weight_decay_shift); // ld 0.01 = -6.644
    float step_error_threshold_factor = fann_bp_to_float(ann->sarprop_step_error_threshold_factor); // 0.1
    float step_error_shift = fann_bp_to_float(ann->sarprop_step_error_shift); // ld 3 = 1.585
    float T = fann_bp_to_float(ann->sarprop_temperature);
    float loss = fann_get_loss(ann);
    float Rloss = sqrtf(loss);

    unsigned int i = first_weight;

    // for all weights; TODO: are biases included?
    for(; i != past_end; i++)
    {
        // TODO: confirm whether 1x10^-6 == delta_min is really better
        prev_step = fann_bp_max(prev_steps[i], fann_float_to_bp(0.000001));    // prev_step may not be zero because then the training will stop
        // calculate SARPROP slope; TODO: better as new error function? (see SARPROP paper)
        slope = fann_bp_mul(fann_bp_neg(fann_bp_add(weight_slopes[i], weights[i])), fann_float_to_bp(fann_exp2(-T * epoch + weight_decay_shift)));

        // TODO: is prev_slopes[i] 0.0 in the beginning? 
        prev_slope = prev_slopes[i];

        same_sign = fann_bp_mul(prev_slope, slope);

        if (fann_bp_is_pos(same_sign)) {
            next_step = fann_bp_min(fann_bp_mul(prev_step, increase_factor), delta_max);
            // TODO: are the signs inverted? see differences between SARPROP paper and iRprop
            if (fann_bp_is_neg(slope)) {
                weights[i] = fann_bp_add(weights[i], next_step);
            } else {
                weights[i] = fann_bp_sub(weights[i], next_step);
            }
        } else if (fann_bp_is_neg(same_sign)) {
            if (fann_bp_lt(prev_step, fann_float_to_bp(step_error_threshold_factor * loss))) {
                fann_bp_mac(prev_step, decrease_factor,
                        fann_float_to_bp((float)rand() / RAND_MAX * Rloss * (fann_type_ex)fann_exp2(-T * epoch + step_error_shift)),
                        next_step);
            } else {
                next_step = fann_bp_max(fann_bp_mul(prev_step, decrease_factor), delta_min);
            }
            slope = fann_int_to_bp(0);
        } else {
            if (fann_bp_is_neg(slope)) {
                weights[i] = fann_bp_add(weights[i], prev_step);
            } else {
                weights[i] = fann_bp_sub(weights[i], prev_step);
            }
        }


        //if(i == 2){
         * fprintf(stderr, "weight=%f, slope=%f, next_step=%f, prev_step=%f\n", weights[i], slope, next_step, prev_step);
         * }

        // update global data arrays
        prev_steps[i] = next_step;
        prev_slopes[i] = slope;
        weight_slopes[i] = fann_int_to_bp(0);
    }
#endif
}
*/

#if (defined SWF16_AP) || (defined HWF16)
FANN_EXTERNAL void FANN_API fann_set_fixed_bp_bias(struct fann *ann)
{
    ann->change_bias = 0;
}

FANN_EXTERNAL void FANN_API fann_set_dynamic_bp_bias(struct fann *ann)
{
    ann->change_bias = 1;
}

FANN_EXTERNAL void FANN_API fann_initialize_bp_bias(struct fann *ann, int bias)
{
    struct fann_neuron *last_neuron, *neuron_it;
    struct fann_layer *layer_it;
    struct fann_layer *last_layer = ann->last_layer;

    for (layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++) {
        last_neuron = layer_it->neuron + layer_it->num_neurons; // NOT THE BIAS
        for (neuron_it = layer_it->neuron; neuron_it != last_neuron; neuron_it++) {
            neuron_it->bp_fp16_bias = (int_fast8_t)bias;
        }
    }
#ifdef FANN_THREADS
    if (ann->num_procs > 1) {
        unsigned int p = ann->num_procs - 1;
        while (p--) {
            fann_initialize_bp_bias(ann->ann[p], bias);
        }
    }
#endif
}
#endif
#endif // FANN_INFERENCE_ONLY

FANN_EXTERNAL void FANN_API fann_set_activation_function_hidden(struct fann *ann,
                                                                enum fann_activationfunc_enum activation_function)
{
    struct fann_layer *layer_it;
    struct fann_layer *last_layer = ann->last_layer - 1;    /* -1 to not update the output layer */

    if (activation_function == FANN_SOFTMAX)
        return;
    for(layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++) {
        layer_it->activation = activation_function;
    }
#ifdef FANN_THREADS
    if (ann->num_procs > 1) {
        unsigned int p = ann->num_procs - 1;
        while (p--) {
            fann_set_activation_function_hidden(ann->ann[p], activation_function);
        }
    }
#endif
}

static struct fann_layer* FANN_API fann_get_layer(struct fann *ann, int layer)
{
    if(layer <= 0 || layer >= (ann->last_layer - ann->first_layer))
    {
        fann_error(FANN_E_INDEX_OUT_OF_BOUND, layer);
        return NULL;
    }
    
    return ann->first_layer + layer;    
}

static struct fann_neuron* FANN_API fann_get_neuron_layer(struct fann_layer* layer, unsigned int neuron)
{
    if(neuron >= layer->num_neurons)
    {
        fann_error(FANN_E_INDEX_OUT_OF_BOUND, neuron);
        return NULL;    
    }
    
    return layer->neuron + neuron;
}

static struct fann_neuron* FANN_API fann_get_neuron(struct fann *ann, unsigned int layer, unsigned int neuron)
{
    struct fann_layer *layer_it = fann_get_layer(ann, layer);
    if(layer_it == NULL)
        return NULL;
    return fann_get_neuron_layer(layer_it, neuron);
}

FANN_EXTERNAL enum fann_activationfunc_enum FANN_API
    fann_get_activation_function(struct fann *ann, int layer)
{
    struct fann_layer *layer_it = fann_get_layer(ann, layer);
    if (layer_it == NULL)
    {
        return (enum fann_activationfunc_enum)-1; /* layer or neuron out of bounds */
    }
    else
    {
        return layer_it->activation;
    }
}

FANN_EXTERNAL void FANN_API fann_set_activation_function_layer(struct fann *ann,
                                                                enum fann_activationfunc_enum
                                                                activation_function,
                                                                int layer)
{
    struct fann_layer *layer_it = fann_get_layer(ann, layer);
    
    if(layer_it == NULL)
        return;
    if (layer == (ann->last_layer - ann->first_layer - 1))
        return;

    layer_it->activation = activation_function;
#ifdef FANN_THREADS
    if (ann->num_procs > 1) {
        unsigned int p = ann->num_procs - 1;
        while (p--) {
            fann_set_activation_function_layer(ann->ann[p], activation_function, layer);
        }
    }
#endif
}


FANN_EXTERNAL void FANN_API fann_set_activation_function_output(struct fann *ann,
                                                                enum fann_activationfunc_enum activation_function)
{
    struct fann_layer *last_layer = ann->last_layer - 1;

    last_layer->activation = activation_function;
#ifdef FANN_THREADS
    if (ann->num_procs > 1) {
        unsigned int p = ann->num_procs - 1;
        while (p--) {
            fann_set_activation_function_output(ann->ann[p], activation_function);
        }
    }
#endif
}

FANN_EXTERNAL void FANN_API fann_set_activation_steepness_hidden(struct fann *ann,
                                                                 float steepness_f)
{
    fann_type_ff steepness;
    struct fann_neuron *last_neuron, *neuron_it;
    struct fann_layer *layer_it;
    struct fann_layer *last_layer = ann->last_layer - 1;    /* -1 to not update the output layer */

    fann_set_ff_bias();
    steepness = fann_float_to_ff(steepness_f);

    for(layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++)
    {
        last_neuron = layer_it->neuron + layer_it->num_neurons; // NOT THE BIAS
        for(neuron_it = layer_it->neuron; neuron_it != last_neuron; neuron_it++)
        {
            neuron_it->steepness = steepness;
        }
    }
#ifdef FANN_THREADS
    if (ann->num_procs > 1) {
        unsigned int p = ann->num_procs - 1;
        while (p--) {
            fann_set_activation_steepness_hidden(ann->ann[p], steepness_f);
        }
    }
#endif
}

/*FANN_EXTERNAL float FANN_API
    fann_get_activation_steepness(struct fann *ann, int layer, int neuron)
{
    struct fann_neuron* neuron_it = fann_get_neuron(ann, layer, neuron);
    if(neuron_it == NULL)
    {
        return -1.0;
    }
    else
    {
        return fann_ff_to_float(neuron_it->steepness);
    }
}*/

FANN_EXTERNAL void FANN_API fann_set_activation_steepness(struct fann *ann,
                                                                float steepness_f,
                                                                int layer,
                                                                int neuron)
{
    fann_type_ff steepness;
    struct fann_neuron* neuron_it = fann_get_neuron(ann, layer, neuron);
    if(neuron_it == NULL)
        return;
    fann_set_ff_bias();
    steepness = fann_float_to_ff(steepness_f);

    neuron_it->steepness = steepness;
#ifdef FANN_THREADS
    if (ann->num_procs > 1) {
        unsigned int p = ann->num_procs - 1;
        while (p--) {
            fann_set_activation_steepness(ann->ann[p], steepness_f, layer, neuron);
        }
    }
#endif
}

FANN_EXTERNAL void FANN_API fann_set_activation_steepness_layer(struct fann *ann,
                                                                float steepness_f,
                                                                int layer)
{
    fann_type_ff steepness;
    struct fann_neuron *last_neuron, *neuron_it;
    struct fann_layer *layer_it = fann_get_layer(ann, layer);
    fann_set_ff_bias();
    steepness = fann_float_to_ff(steepness_f);
   
    if(layer_it == NULL)
        return;

    last_neuron = layer_it->neuron + layer_it->num_neurons; // NOT THE BIAS
    for(neuron_it = layer_it->neuron; neuron_it != last_neuron; neuron_it++)
    {
        neuron_it->steepness = steepness;
    }
#ifdef FANN_THREADS
    if (ann->num_procs > 1) {
        unsigned int p = ann->num_procs - 1;
        while (p--) {
            fann_set_activation_steepness_layer(ann->ann[p], steepness_f, layer);
        }
    }
#endif
}

FANN_EXTERNAL void FANN_API fann_set_activation_steepness_output(struct fann *ann,
                                                                 float steepness_f)
{
    fann_type_ff steepness;
    struct fann_neuron *last_neuron, *neuron_it;
    struct fann_layer *last_layer = ann->last_layer - 1;
    fann_set_ff_bias();
    steepness = fann_float_to_ff(steepness_f);

    last_neuron = last_layer->neuron + last_layer->num_neurons; // NOT THE BIAS
    for(neuron_it = last_layer->neuron; neuron_it != last_neuron; neuron_it++)
    {
        neuron_it->steepness = steepness;
    }
#ifdef FANN_THREADS
    if (ann->num_procs > 1) {
        unsigned int p = ann->num_procs - 1;
        while (p--) {
            fann_set_activation_steepness_output(ann->ann[p], steepness_f);
        }
    }
#endif
}

#ifdef CALCULATE_ERROR
FANN_EXTERNAL void FANN_API fann_set_bit_fail_limit(struct fann *ann, float bit_fail_limit)
{
    ann->bit_fail_limit = bit_fail_limit;
}

FANN_EXTERNAL float FANN_API fann_get_bit_fail_limit(struct fann *ann)
{
    return ann->bit_fail_limit;
}
#endif // CALCULATE_ERROR

#ifndef FANN_INFERENCE_ONLY
FANN_EXTERNAL void FANN_API fann_set_learning_momentum(struct fann *ann, float learning_momentum)
{
    fann_set_ff_bias();
    ann->learning_momentum = fann_float_to_ff(learning_momentum);
}

FANN_EXTERNAL void FANN_API fann_set_learning_rate(struct fann *ann, float learning_rate)
{
    fann_set_ff_bias();
    ann->learning_rate = fann_float_to_ff(learning_rate);
}

FANN_EXTERNAL float FANN_API fann_get_learning_rate(struct fann *ann)
{
    fann_set_ff_bias();
    return fann_ff_to_float(ann->learning_rate);
}

FANN_EXTERNAL void FANN_API fann_set_train_stop_function(struct fann *ann, enum fann_stopfunc_enum stopfunc)
{
    ann->train_stop_function = stopfunc;
}
#endif // FANN_INFERENCE_ONLY

