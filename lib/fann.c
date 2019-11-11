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
#include <time.h>
#include <math.h>
#endif // FANN_INFERENCE_ONLY

#include "fann.h"

#ifndef FANN_INFERENCE_ONLY

FANN_EXTERNAL struct fann *FANN_API fann_create_standard_args(unsigned int extra_threads, ...)
{
    struct fann *ann;
    va_list layer_sizes;
    int i;
    int status;
    int arg;
    unsigned int *layers, num_layers;

#ifdef FANN_THREADS
    if (extra_threads > FANN_THREADS) {
        return NULL;
    }
#endif

    va_start(layer_sizes, extra_threads);
    num_layers = va_arg(layer_sizes, unsigned int);
    if (num_layers > 50) {
        return NULL;
    }
    fann_calloc(layers, num_layers);
    if(layers == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }
    
    status = 1;
    for(i = 0; i < (int) num_layers; i++)
    {
        arg = va_arg(layer_sizes, unsigned int);
        if(arg < 0 || arg > 1000000)
            status = 0;
        layers[i] = arg;
    }
    va_end(layer_sizes);

    if(!status)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        free(layers);
        return NULL;
    }

    ann = fann_create_standard_vector(extra_threads, num_layers, layers);

    free(layers);

    return ann;
}

FANN_EXTERNAL struct fann *FANN_API fann_create_standard_vector(
                                                               unsigned int extra_threads,
                                                               unsigned int num_layers, 
                                                               const unsigned int *layers)
{
    return fann_create_sparse_vector(extra_threads, num_layers, layers);    
}

static unsigned int FANN_SEED_FIXED = 0;

FANN_EXTERNAL void FANN_API fann_enable_seed_rand(void)
{
    FANN_SEED_FIXED = 0;
}

FANN_EXTERNAL void FANN_API fann_enable_seed_fixed(const unsigned int seed)
{
    FANN_SEED_FIXED = seed;
}

/* INTERNAL FUNCTION
   Seed the random function.
 */
static void fann_seed_rand(void)
{
    // FIXME: embedded replacement
    FILE *fp = fopen("/dev/urandom", "r");
    unsigned int foo;
    struct timeval t;

    if(!fp)
    {
        gettimeofday(&t, NULL);
        foo = t.tv_usec;
#ifdef DEBUG
        printf("unable to open /dev/urandom\n");
#endif
    }
    else
    {
            if(fread(&foo, sizeof(foo), 1, fp) != 1) 
            {
                 gettimeofday(&t, NULL);
               foo = t.tv_usec;
#ifdef DEBUG
               printf("unable to read from /dev/urandom\n");
#endif              
        }
        fclose(fp);
    }
    srand(foo);
}

static void fann_seed(void)
{
    if (FANN_SEED_FIXED) {
        srand(FANN_SEED_FIXED);
    } else {
        fann_seed_rand();
    }
}

// fann_print_structure(ann, __FILE__, __FUNCTION__, __LINE__);
void fann_print_structure(struct fann *ann, const char *file, const char* function, const int line)
{
    struct fann_layer *layer_it;
    struct fann_neuron *neuron_it;
    unsigned int l, n, w, num_w = ann->first_layer->num_connections;

    fann_set_ff_bias();
    fprintf(stderr, "%s: %s, %s, %d\n", __FUNCTION__, file, function, line);
    layer_it = ann->first_layer;
    fprintf(stderr, "layer=000: neu=%05d con=%05d ", layer_it->num_neurons, layer_it->num_connections);
    //fprintf(stderr, "n=%p s=%p p=%p t=%p\n", layer_it->neuron, layer_it->sum_w, layer_it->value, layer_it->train_errors);
    fprintf(stderr, "n=%p s=%p p=%p\n", layer_it->neuron, layer_it->sum_w, layer_it->value);
    layer_it++;
    for (l = 1; layer_it < ann->last_layer; l++, layer_it++) {
        fprintf(stderr, "layer=%03u: neu=%05d con=%05d ", l, layer_it->num_neurons, layer_it->num_connections);
        //fprintf(stderr, "n=%p s=%p p=%p t=%p ", layer_it->neuron, layer_it->sum_w, layer_it->value, layer_it->train_errors);
        fprintf(stderr, "n=%p s=%p p=%p ", layer_it->neuron, layer_it->sum_w, layer_it->value);
        fprintf(stderr, "act=%s\n", FANN_ACTIVATIONFUNC_NAMES[layer_it->activation]);
        if (layer_it->neuron != NULL) {
            for (n = 0; n < layer_it->num_neurons; n++) {
                neuron_it = layer_it->neuron + n;
                fprintf(stderr, "           neu=%05d st=%+le\n", n,
                        fann_ff_to_float(neuron_it->steepness));
                for (w = 0; w < num_w; w++) {
                    fprintf(stderr, "           w[%u]=%f\n", w, fann_ff_to_float(neuron_it->weight[w]));
                }
            }
        }
        num_w = layer_it->num_connections;
    }
}

FANN_EXTERNAL struct fann *FANN_API fann_create_sparse_vector(
        unsigned int extra_threads,
        unsigned int num_layers,
        const unsigned int *layers)
{
    struct fann_layer *layer_it, *last_layer, *prev_layer;
    struct fann *ann;
    struct fann_neuron *neuron_it, *last_neuron;
    unsigned int i;//, num_neurons_out;
    unsigned int tmp_con = extra_threads;
    fann_type_ff winit;

    fann_const_init();
    winit = ff_0000;//fann_int_to_bp(0);

    fann_seed();

    /* allocate the general structure */
    ann = fann_allocate_structure(num_layers);
    if(ann == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }
    fann_reset_loss(ann);

    /* determine how many neurons there should be in each layer */
    layer_it = ann->first_layer;
    for (i = 0; layer_it != ann->last_layer; i++, layer_it++) {
        /* we do not allocate room here, but we make sure that
         * last_neuron - first_neuron is the number of neurons */
        layer_it->neuron = NULL;
        layer_it->num_neurons = layers[i]; // FIXME: should be 0 in the first layer
        /* +1 for bias */
        layer_it->num_connections = layer_it->num_neurons + 1;
        layer_it->activation = FANN_SIGMOID;
#ifndef FANN_INFERENCE_ONLY
        layer_it->max_init = NT_0000;
        layer_it->var_init = NT_0000;
        fann_reset_batch_stats(layer_it);
#endif // FANN_INFERENCE_ONLY
    }

    ann->num_output = (unsigned int)((ann->last_layer - 1)->num_neurons);
    ann->num_input = (unsigned int)(ann->first_layer->num_neurons);
#ifdef CALCULATE_ERROR
    fann_malloc(ann->num_max_ok, ann->num_output);
    if (ann->num_max_ok == NULL) {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }
#endif // CALCULATE_ERROR

    /* allocate room for the actual neurons */
    if (fann_allocate_neurons(ann, NULL)) {
        fann_destroy(ann);
        return NULL;
    }

    last_layer = ann->last_layer;
    prev_layer = ann->first_layer;
    for (layer_it = ann->first_layer + 1; layer_it != last_layer; layer_it++) {
        last_neuron = layer_it->neuron + layer_it->num_neurons;
        for (neuron_it = layer_it->neuron; neuron_it != last_neuron; neuron_it++) {
            //tmp_con = neuron_it->prev_layer->num_neurons;
            tmp_con = prev_layer->num_neurons;
            for (i = 0; i < tmp_con; i++) {
                neuron_it->weight[i] = winit; //fann_random_weight();
            }
            /* bias weight */
            neuron_it->weight[tmp_con] = winit; //fann_random_bias_weight();
        }
        prev_layer = layer_it;
#ifdef DEBUG
        printf("  layer       : %d neurons, 1 bias\n", prev_layer->num_neurons);
#endif
    }
#ifdef FANN_THREADS
    ann->num_procs = extra_threads + 1;//FANN_THREADS + 1;
    for (i = 0; i < extra_threads; i++) {
        ann->ann[i] = fann_copy(ann);
        if (ann->ann[i] == NULL) {
            fann_destroy(ann);
            return NULL;
        }
    }
#endif
    return ann;
}
#else
#define fann_seed()
#endif // FANN_INFERENCE_ONLY

void fann_run_layer(struct fann_layer *layer_it, struct fann_layer *prev_layer)
{
#define DEBUG_RUN
#undef DEBUG_RUN

    unsigned int n, w, prev_neurons;
    fann_type_ff *weights, *prev_values, steepness;
    unsigned int num_neurons;
    struct fann_neuron *neuron_it;
    fann_type_nt neuron_sum, max_sum;// = fann_int_to_bp(0);    
    int softmax = 0;

#ifdef DEBUG_RUN
    fprintf(stderr, "### %s @ %s : %d\n", __FUNCTION__, __FILE__, __LINE__);
#endif

    prev_values = prev_layer->value;
    prev_neurons = prev_layer->num_neurons;
    num_neurons = layer_it->num_neurons; // exclude BIAS
    max_sum = NT_0000;//ff_0000;
    if (layer_it->activation == FANN_SOFTMAX) {
        // only in the last layer...
        softmax = 1;
    }
        for (n = 0; n < num_neurons; n++) {
            neuron_it = layer_it->neuron + n;
            steepness = neuron_it->steepness;
            weights = neuron_it->weight;
            //neuron_sum = ff_0000;
            neuron_sum = fann_ff_to_nt(weights[prev_neurons]); // BIAS 

#ifdef DEBUG_RUN
            fprintf(stderr, "  neuron %4u:\n", n);
#endif
            for (w = 0; w < prev_neurons; w++) {
#ifdef DEBUG_RUN
                fprintf(stderr, "    w=%u : %f += %f*%f\n", w,
                       (float)fann_ff_to_float(neuron_sum),
                       (float)fann_ff_to_float(weights[w]),
                       (float)fann_ff_to_float(prev_values[w]));
#endif
                neuron_sum = fann_nt_mac(fann_ff_to_nt(weights[w]), fann_ff_to_nt(prev_values[w]), neuron_sum);
            }
            //neuron_sum = fann_ff_mac(weights[w], ff_p100, neuron_sum);

            neuron_sum = fann_nt_mul(fann_ff_to_nt(steepness), neuron_sum);
            if (softmax && fann_nt_gt(neuron_sum, max_sum)) {
                max_sum = neuron_sum;
            }
            layer_it->sum_w[n] = fann_nt_to_ff(neuron_sum);
            //layer_it->value[n] = fann_activation_switch(layer_it->activation, neuron_sum);
            fann_activation_switch(layer_it, n);
#ifdef DEBUG_RUN
            fprintf(stderr, "  val_out=%f -> %f\n",
                    (float)fann_ff_to_float(layer_it->sum_w[n]),
                    (float)fann_ff_to_float(layer_it->value[n]));
#endif
#ifndef FANN_INFERENCE_ONLY
#if 0
            if (fann_ff_lt(layer_it->max_abs_sum, fann_ff_abs(layer_it->sum_w[n]))) {
                layer_it->max_abs_sum = fann_ff_abs(layer_it->sum_w[n]);
            }
            if (fann_ff_gt(layer_it->min_abs_sum, fann_ff_abs(layer_it->sum_w[n]))) {
                layer_it->min_abs_sum = fann_ff_abs(layer_it->sum_w[n]);
            }
#endif
#endif
        }
        if (softmax) {
            fann_type_ff e, tot, ff_max_sum;

            ff_max_sum = fann_nt_to_ff(max_sum);
            tot = ff_0000;
            for (n = 0; n < num_neurons; n++) {
                e = fann_ff_exp(fann_ff_sub(layer_it->value[n], ff_max_sum));
                tot = fann_ff_add(tot, e);
                layer_it->value[n] = e;
            }
            if (fann_ff_is_pos(tot)) {
                for (n = 0; n < num_neurons; n++) {
                    layer_it->value[n] = fann_ff_div(layer_it->value[n], tot);
                }
            }
            softmax = 0;
        }
}

FANN_EXTERNAL fann_type_ff *FANN_API fann_run(struct fann * ann, fann_type_ff * input)
{
    struct fann_layer *layer_it, *last_layer, *prev_layer;

    fann_set_ff_bias();
    /* first set the input */
    layer_it = ann->first_layer;
    layer_it->value = input;
    prev_layer = layer_it;
    last_layer = ann->last_layer;
    for (layer_it++; layer_it != last_layer; layer_it++) {
        fann_run_layer(layer_it, prev_layer);
        prev_layer = layer_it;
    }
    return (ann->last_layer - 1)->value; // this is the output
}

FANN_EXTERNAL void FANN_API fann_destroy(struct fann *ann)
{
    struct fann_layer *layer_it;
    if(ann == NULL)
        return;
    ann->first_layer->value = NULL;
    for (layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++) {
        struct fann_neuron *neuron_it, *last_neuron;
        
        fann_free(layer_it->value);
        fann_free(layer_it->sum_w);
#ifndef FANN_INFERENCE_ONLY
        //fann_free(layer_it->train_errors);
#endif
        fann_free(layer_it->neuron);

        if (layer_it->neuron == NULL)
            continue;

        last_neuron = layer_it->neuron + layer_it->num_neurons;
        for (neuron_it = layer_it->neuron; neuron_it != last_neuron; neuron_it++) {
            //fann_free(neuron_it->prev_layer);
            fann_free(neuron_it->weight);
#ifndef FANN_INFERENCE_ONLY
            fann_free(neuron_it->weight_slopes);
            fann_free(neuron_it->prev_steps);
            fann_free(neuron_it->prev_slopes);
#endif
        }

    }
#ifdef CALCULATE_ERROR
    fann_free(ann->num_max_ok);
#endif // CALCULATE_ERROR
#ifndef FANN_INFERENCE_ONLY
    fann_free(ann->unbal_er_adjust);
#endif
    fann_free(ann->first_layer);
    
#ifdef FANN_DATA_SCALE
    fann_free( ann->scale_mean_in );
    fann_free( ann->scale_deviation_in );
    fann_free( ann->scale_new_min_in );
    fann_free( ann->scale_factor_in );
    fann_free( ann->scale_mean_out );
    fann_free( ann->scale_deviation_out );
    fann_free( ann->scale_new_min_out );
    fann_free( ann->scale_factor_out );
#endif // FANN_DATA_SCALE
    
#ifndef FANN_INFERENCE_ONLY
#ifdef FANN_PRINT_STATS
    fann_free(ann->stats_weigs);
    fann_free(ann->stats_errors);
    fann_free(ann->stats_deltas);
    fann_free(ann->stats_steps);
    fann_free(ann->stats_slopes);
#endif // FANN_PRINT_STATS
#endif // FANN_INFERENCE_ONLY

    fann_free(ann);
}

#ifndef FANN_INFERENCE_ONLY
FANN_EXTERNAL void FANN_API fann_randomize_weights(struct fann *ann,
                                                   fann_type_nt min_weight,
                                                   fann_type_nt max_weight)
{
    struct fann_neuron * neuron, * last_neuron;
    struct fann_layer * layer_it, * prev_layer;
    fann_type_ff *weights, *last_weight;

    prev_layer = ann->first_layer;
    for (layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++) {
        layer_it->max_init = max_weight;
        layer_it->var_init = fann_nt_div(fann_nt_sub(max_weight, min_weight), fann_float_to_nt(2.0 * 1.73205080756888));
        layer_it->var_init = fann_nt_mul(layer_it->var_init, layer_it->var_init);
        last_neuron = layer_it->neuron + layer_it->num_neurons;
        for (neuron = layer_it->neuron; neuron != last_neuron; neuron++) {
            weights = neuron->weight;
            last_weight = weights + prev_layer->num_connections;
            for(; weights != last_weight; weights++) {
                *weights = fann_ff_random_weights(min_weight, max_weight);
            }
        }
        prev_layer = layer_it;
    }
    fann_clear_train_arrays(ann);
    fann_set_ff_bias();
}
#endif // FANN_INFERENCE_ONLY

/* deep copy of the fann structure sharing the neurons and read-only buffers */
FANN_EXTERNAL struct fann* FANN_API fann_copy(struct fann* orig)
{
    struct fann* copy;
    unsigned int num_layers = (unsigned int)(orig->last_layer - orig->first_layer);
    struct fann_layer *orig_layer_it, *copy_layer_it;

    copy = fann_allocate_structure(num_layers);
    if (copy == NULL) {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }
#ifdef FANN_THREADS
    copy->num_procs = 0;
    copy->ann[0] = orig;
#endif

    copy->num_input = orig->num_input;
    copy->num_output = orig->num_output;
#ifdef CALCULATE_LOSS
    copy->loss_count = orig->loss_count;
    copy->loss_value = orig->loss_value;
#endif // CALCULATE_LOSS
#ifdef CALCULATE_ERROR
    copy->num_bit_fail[0] = orig->num_bit_fail[0];
    copy->num_bit_fail[1] = orig->num_bit_fail[1];
    copy->num_bit_ok[0] = orig->num_bit_ok[0];
    copy->num_bit_ok[1] = orig->num_bit_ok[1];
    copy->bit_fail_limit = orig->bit_fail_limit;
#endif // CALCULATE_ERROR
#ifndef FANN_INFERENCE_ONLY
    copy->unbal_er_adjust = orig->unbal_er_adjust;
    copy->learning_rate = orig->learning_rate;
    copy->learning_momentum = orig->learning_momentum;
    copy->training_algorithm = orig->training_algorithm;
    copy->mini_batch = orig->mini_batch;

    //copy->train_loss_function = orig->train_loss_function;
    //copy->train_error_function = orig->train_error_function;
    copy->train_stop_function = orig->train_stop_function;
    copy->callback = NULL;
    //copy->user_data = NULL;

    copy->rmsprop_avg = orig->rmsprop_avg;
    copy->rmsprop_1mavg = orig->rmsprop_1mavg;
    
    //copy->quickprop_decay = orig->quickprop_decay;
    //copy->quickprop_mu = orig->quickprop_mu;
    copy->rprop_increase_factor = orig->rprop_increase_factor;
    copy->rprop_decrease_factor = orig->rprop_decrease_factor;
    copy->rprop_delta_min = orig->rprop_delta_min;
    copy->rprop_delta_max = orig->rprop_delta_max;
    copy->rprop_delta_zero = orig->rprop_delta_zero;

    /*copy->sarprop_weight_decay_shift = orig->sarprop_weight_decay_shift;
    copy->sarprop_step_error_threshold_factor = orig->sarprop_step_error_threshold_factor;
    copy->sarprop_step_error_shift = orig->sarprop_step_error_shift;
    copy->sarprop_temperature = orig->sarprop_temperature;*/
    copy->train_epoch = 0;
#endif // FANN_INFERENCE_ONLY
 
#ifndef FANN_INFERENCE_ONLY
#ifdef FANN_PRINT_STATS
    copy->stats_weigs = NULL;
    copy->stats_errors = NULL;
    copy->stats_deltas = NULL;
    copy->stats_slopes = NULL;
    copy->stats_steps = NULL;
#endif // FANN_PRINT_STATS
#endif // FANN_INFERENCE_ONLY

    /* copy layer sizes, prepare for fann_allocate_neurons */
    for (orig_layer_it = orig->first_layer, copy_layer_it = copy->first_layer;
            orig_layer_it != orig->last_layer; orig_layer_it++, copy_layer_it++)
    {
        copy_layer_it->neuron = orig_layer_it->neuron;
        copy_layer_it->num_neurons = orig_layer_it->num_neurons;
        copy_layer_it->num_connections = orig_layer_it->num_connections;
        copy_layer_it->activation = orig_layer_it->activation;
#ifndef FANN_INFERENCE_ONLY
        copy_layer_it->max_init = NT_0000;
        copy_layer_it->var_init = NT_0000;
#endif
        //fann_reset_batch_stats(copy_layer_it);
    }

#ifdef FANN_DATA_SCALE
    /* copy scale parameters, when used */
    if (orig->scale_mean_in != NULL)
    {
        copy->scale_mean_in = orig->scale_mean_in;
        copy->scale_deviation_in = orig->scale_deviation_in;
        copy->scale_new_min_in = orig->scale_new_min_in;
        copy->scale_factor_in = orig->scale_factor_in;
        copy->scale_mean_out = orig->scale_mean_out;
        copy->scale_deviation_out = orig->scale_deviation_out;
        copy->scale_new_min_out = orig->scale_new_min_out;
        copy->scale_factor_out = orig->scale_factor_out;
    }
#endif // FANN_DATA_SCALE

#ifdef CALCULATE_ERROR
    fann_malloc(copy->num_max_ok, copy->num_output);
    if (copy->num_max_ok == NULL) {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }
#endif // CALCULATE_ERROR

    /* allocate layer buffers */
    if (fann_allocate_neurons(copy, orig)) {
        fann_destroy(copy);
        return NULL;
    }
    return copy;
}

#ifndef FANN_INFERENCE_ONLY
#ifdef FANN_PRINT_STATS
static int cmp_float(const void *p1, const void *p2)
{
    float *f1, *f2;

    f1 = (float *)p1;
    f2 = (float *)p2;
    
    if (*f1 < *f2)
        return -1;
    if (*f1 > *f2)
        return 1;
    return 0;
}

#include <stdlib.h>
static void fann_stats_quart(float * vec,
                             const unsigned int c,
                             unsigned int * q1,
                             unsigned int * q2,
                             unsigned int * q3)
{
    qsort(vec, c, sizeof(*vec), cmp_float);
    if (q1 == NULL) {
        return;
    }
    if (c > 5) {
        *q1 = c / 4;
        *q2 = c / 2;
        *q3 = (3 * c) / 4;
    } else if (c > 2) {
        *q1 = 0;
        *q2 = 1;
        *q3 = 2;
    } else {
        *q1 = 0;
        *q2 = 0;
        *q3 = 1;
    }
}

static int fann_batch_stats_disabled = 1;
#endif // FANN_PRINT_STATS

void fann_reset_batch_stats(struct fann_layer *layer_it)
{
#ifdef FANN_PRINT_STATS
    // output errors:
    layer_it->zero_error = layer_it->count_error = 0;
    layer_it->avg_abs_error = layer_it->max_abs_error = 0.0;
    layer_it->min_abs_error = 0.0;
    // weight deltas:
    layer_it->zero_delta = layer_it->count_delta = 0;
    layer_it->avg_abs_delta = layer_it->max_abs_delta = 0.0;
    layer_it->min_abs_delta = 0.0;
    // adaptive step:
    layer_it->zero_step = layer_it->count_step = 0;
    layer_it->avg_abs_step = layer_it->max_abs_step = 0.0;
    layer_it->min_abs_step = 0.0;
    // adaptive slope:
    layer_it->zero_slope = layer_it->count_slope = 0;
    layer_it->avg_abs_slope = layer_it->max_abs_slope = 0.0;
    layer_it->min_abs_slope = 0.0;
#else
    layer_it = layer_it;
#endif // FANN_PRINT_STATS
}

void FANN_API fann_enable_batch_stats(void)
{
#ifdef FANN_PRINT_STATS
    fann_batch_stats_disabled = 0;
#endif // FANN_PRINT_STATS
}

void FANN_API fann_batch_stats(struct fann *ann)
{
#ifdef FANN_PRINT_STATS
    unsigned int n, num_w, num_n, w;
    struct fann_layer *layer_it, *prev_layer;
    struct fann_neuron *neuron_it;
    double ab;

    if ((fann_batch_stats_disabled) || (ann->stats_weigs == NULL)) {
        return;
    }
    prev_layer = ann->first_layer;
    for (layer_it = prev_layer + 1; layer_it < ann->last_layer; layer_it++) {
        prev_layer = layer_it;
        num_w = prev_layer->num_connections;
        num_n = layer_it->num_neurons;
        for (n = 0; n < num_n; n++) {
            neuron_it = layer_it->neuron + n;
            fann_set_bp_bias(neuron_it->bp_fp16_bias);
                //if (layer_it->train_errors != NULL) {
                    ab = fabs(fann_bp_to_float(neuron_it->train_error));
                if (ab == 0.0) {
                    layer_it->zero_error++;
                } else {
                    if (ab > layer_it->max_abs_error)
                        layer_it->max_abs_error = ab;
                    if ((ab < layer_it->min_abs_error) || (layer_it->min_abs_error == 0.0))
                        layer_it->min_abs_error = ab;
                    layer_it->count_error++;
                    layer_it->avg_abs_error += ab;
                }
            //}
            for (w = 0; w < num_w ; w++) {
                if (neuron_it->prev_steps != NULL) {
                    ab = fabs(fann_bp_to_float(neuron_it->prev_steps[w]));
                    if (ab == 0.0) {
                        layer_it->zero_step++;
                    } else {
                        if (ab > layer_it->max_abs_step)
                            layer_it->max_abs_step = ab;
                        if ((ab < layer_it->min_abs_step) || (layer_it->min_abs_step == 0.0))
                            layer_it->min_abs_step = ab;
                        layer_it->count_step++;
                        layer_it->avg_abs_step += ab;
                    }
                }
                if (neuron_it->prev_slopes != NULL) {
                    ab = fabs(fann_bp_to_float(neuron_it->prev_slopes[w]));
                    if (ab == 0.0) {
                        layer_it->zero_slope++;
                    } else {
                        if (ab > layer_it->max_abs_slope)
                            layer_it->max_abs_slope = ab;
                        if ((ab < layer_it->min_abs_slope) || (layer_it->min_abs_slope == 0.0))
                            layer_it->min_abs_slope = ab;
                        layer_it->count_slope++;
                        layer_it->avg_abs_slope += ab;
                    }
                }
                if (neuron_it->weight_slopes != NULL) {
                    ab = fabs(fann_bp_to_float(neuron_it->weight_slopes[w]));
                    if (ab == 0.0) {
                        layer_it->zero_delta++;
                    } else {
                        if (ab > layer_it->max_abs_delta)
                            layer_it->max_abs_delta = ab;
                        if ((ab < layer_it->min_abs_delta) || (layer_it->min_abs_delta == 0.0))
                            layer_it->min_abs_delta = ab;
                        layer_it->count_delta++;
                        layer_it->avg_abs_delta += ab;
                    }
                }
            } 
        }
    } 
#else
    ann = ann;
#endif // FANN_PRINT_STATS
}

FANN_EXTERNAL void FANN_API fann_print_stats(struct fann *ann)
{
#ifdef FANN_PRINT_STATS
    int layer;
    struct fann_layer *layer_it, *prev_layer;
    struct fann_neuron *neuron_it;
    unsigned int weig_c, n, num_w, num_n, w;
    unsigned int weig_q1, weig_q2, weig_q3;
    unsigned int err_q1, err_q2, err_q3;
    float min_abs_weigs, min_abs_error, min_abs_delta, ab;
    float min_abs_slope, min_abs_step;
    unsigned int zero_weigs, zero_error, zero_delta;
    unsigned int zero_slope, zero_step, incw;
    //unsigned int frozen_l = 0, frozen_n = 0;
    float * weigs, * errors, * deltas, * slopes, * steps;
    float max_bias_w, max_bias_d, max_bias_sl, max_bias_st;
    double avg_weigs, avg_error, avg_delta;
    double avg_slope, avg_step, sd_weigs;

    if (ann->stats_weigs == NULL) {
        unsigned int max_weigs = 0;
        unsigned int max_neurs = 0;

        prev_layer = ann->first_layer;
        for (layer_it = ann->first_layer + 1; layer_it < ann->last_layer; layer_it++) {
            if (max_neurs < layer_it->num_neurons) {
                max_neurs = layer_it->num_neurons;
            }
            w = layer_it->num_neurons * prev_layer->num_connections;
            if (max_weigs < w) {
                max_weigs = w;
            }
            prev_layer = layer_it;
        }
        fann_calloc(ann->stats_weigs, max_weigs);
        fann_calloc(ann->stats_errors, max_neurs);
        fann_calloc(ann->stats_deltas, max_weigs);
        //if (prev_layer->first_neuron->prev_steps != NULL)
            fann_calloc(ann->stats_steps, max_weigs);
        //if (prev_layer->first_neuron->prev_slopes != NULL)
            fann_calloc(ann->stats_slopes, max_weigs);
        if ((ann->stats_weigs == NULL) || (ann->stats_errors == NULL) ||
            (ann->stats_deltas == NULL) || (ann->stats_steps == NULL) ||
            (ann->stats_slopes == NULL)) {
            fann_free(ann->stats_weigs);
            fann_free(ann->stats_errors);
            fann_free(ann->stats_deltas);
            fann_free(ann->stats_steps);
            fann_free(ann->stats_slopes);
            return;
        }
    }
    weigs = ann->stats_weigs;
    errors = ann->stats_errors;
    deltas = ann->stats_deltas;
    slopes = ann->stats_slopes;
    steps = ann->stats_steps;

    prev_layer = ann->first_layer;
    for (layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++) {
        layer = (int)(layer_it - ann->first_layer);
        incw = weig_c = 0;
        sd_weigs = 0.0;
        fann_set_ff_bias();
        avg_weigs = fann_ff_to_float(layer_it->neuron[0].weight[0]);
        zero_weigs = zero_error = zero_delta = zero_slope = zero_step = 0;
        min_abs_slope = min_abs_step = min_abs_error = min_abs_weigs = min_abs_delta = HUGE_VALF;
        avg_slope = avg_step = max_bias_st = max_bias_sl = 0.0;
        max_bias_d = max_bias_w = avg_error = avg_delta = 0.0;
        num_w = prev_layer->num_connections;
        num_n = layer_it->num_neurons;
        for (n = 0; n < num_n; n++) {
            neuron_it = layer_it->neuron + n;
            fann_set_bp_bias(neuron_it->bp_fp16_bias);
            if (ann->training_algorithm == FANN_TRAIN_INCREMENTAL) {
                //if (layer_it->train_errors != NULL) {
                    //errors[n] = fann_bp_to_float(layer_it->train_errors[n]);
                    errors[n] = fann_bp_to_float(neuron_it->train_error);
                    avg_error += (double)errors[n];
                    ab = fabsf(errors[n]);
                    if (ab == 0.0) {
                        zero_error++;
                    } else if (ab < min_abs_error) {
                        min_abs_error = ab;
                    }
                /*} else {
                    errors[n] = 0.0;
                }*/
            }
            for (w = 0; w < num_w ; w++) {
                fann_set_ff_bias();
                weigs[weig_c] = fann_ff_to_float(neuron_it->weight[w]);
                ++incw;
                if (incw > 1) {
                    double tmp = avg_weigs + ((weigs[weig_c] - avg_weigs) / (double)incw);
                    sd_weigs += (weigs[weig_c] - avg_weigs) * (weigs[weig_c] - tmp);
                    avg_weigs = tmp;
                }
                ab = fabsf(weigs[weig_c]);
                if (ab == 0.0) {
                    zero_weigs++;
                } else if (ab < min_abs_weigs) {
                    min_abs_weigs = ab;
                }
                fann_set_bp_bias(neuron_it->bp_fp16_bias);
                if (neuron_it->prev_steps != NULL) {
                    steps[weig_c] = fann_bp_to_float(neuron_it->prev_steps[w]);
                    avg_step += (double)steps[weig_c];
                    ab = fabs(steps[weig_c]);
                    if (ab == 0.0) {
                        zero_step++;
                    } else if (ab < min_abs_step) {
                        min_abs_step = ab;
                    }
                }
                if (neuron_it->prev_slopes != NULL) {
                    slopes[weig_c] = fann_bp_to_float(neuron_it->prev_slopes[w]);
                    avg_slope += (double)slopes[weig_c];
                    ab = fabs(slopes[weig_c]);
                    if (ab == 0.0) {
                        zero_slope++;
                    } else if (ab < min_abs_slope) {
                        min_abs_slope = ab;
                    }
                }
                if (neuron_it->weight_slopes != NULL) {
                    deltas[weig_c] = fann_bp_to_float(neuron_it->weight_slopes[w]);
                    avg_delta += (double)deltas[weig_c];
                    ab = fabs(deltas[weig_c]);
                    if (ab == 0.0) {
                        /*if ((layer > 1) && (w < (num_w - 1))) {
                            printf("zero delta: prev[%u] _sum=%+le _val=%+le | this[%u] _sum=%+le _val=%+le\n",
                                   w, fann_ff_to_float(prev_layer->sum_w[w]),
                                   fann_ff_to_float(prev_layer->value[w]),
                                   n, fann_ff_to_float(layer_it->sum_w[n]),
                                   fann_ff_to_float(layer_it->value[n]));
                        }*/
                        zero_delta++;
                    } else if (ab < min_abs_delta) {
                        min_abs_delta = ab;
                    }
                } else {
                    //printf("forward stats only\n");
                    deltas[weig_c] = 0.0;
                }
                weig_c++;
            }
            w = weig_c - 1;
            if (fabsf(weigs[w]) > max_bias_w)
                max_bias_w = fabsf(weigs[w]);
            if (fabsf(deltas[w]) > max_bias_d)
                max_bias_d = fabsf(deltas[w]);
            if ((steps != NULL) && (fabsf(steps[w]) > max_bias_st))
                max_bias_st = fabsf(steps[w]);
            if ((slopes != NULL) && (fabsf(slopes[w]) > max_bias_sl))
                max_bias_sl = fabsf(slopes[w]);
        }
        if (ann->training_algorithm == FANN_TRAIN_INCREMENTAL) {
            avg_error /= (float)num_n;
            fann_stats_quart(errors, num_n, &err_q1, &err_q2, &err_q3);
        } else {
            err_q1 = err_q2 = err_q3 = 0;
        }
        //avg_weigs /= (float)weig_c;
        sd_weigs = sqrt(sd_weigs/(double)(incw-1));
        fann_stats_quart(weigs, weig_c, &weig_q1, &weig_q2, &weig_q3);
        avg_delta /= (float)weig_c;
        fann_stats_quart(deltas, weig_c, NULL, NULL, NULL);
        if (slopes != NULL) {
            avg_slope /= (float)weig_c;
            fann_stats_quart(slopes, weig_c, NULL, NULL, NULL);
        }
        if (steps != NULL) {
            avg_step /= (float)weig_c;
            fann_stats_quart(steps, weig_c, NULL, NULL, NULL);
        }

        printf("layer=%02d: Weights: bias=%+e min=%e av=%+e %+e %+e %+e %+e %+e z=%u sd=%e\n",
                layer, max_bias_w, min_abs_weigs, avg_weigs,
                weigs[0], weigs[weig_q1], weigs[weig_q2],
                weigs[weig_q3], weigs[weig_c-1], zero_weigs, sd_weigs);
        if (ann->training_algorithm == FANN_TRAIN_INCREMENTAL) {
            if (min_abs_error != HUGE_VALF) {
                printf("layer=%02d:  Errors:                    min=%e av=%+e %+e %+e %+e %+e %+e %u\n",
                        layer, min_abs_error,
                        avg_error, errors[0], errors[err_q1], errors[err_q2],
                        errors[err_q3], errors[num_n-1], zero_error);
            } else {
                printf("layer=%02d:  Errors: bias=0.0 min=0.0 av=0.0 0.0 0.0 0.0 0.0 0.0 %u\n", layer, zero_error);
            }
        }
        if (min_abs_delta != HUGE_VALF) {
            printf("layer=%02d:  Deltas: bias=%+e min=%e av=%+e %+e %+e %+e %+e %+e %u\n",
                    layer, max_bias_d, min_abs_delta,
                    avg_delta, deltas[0], deltas[weig_q1], deltas[weig_q2],
                    deltas[weig_q3], deltas[weig_c-1], zero_delta);
        } else if (deltas != NULL) {
            printf("layer=%02d:  Deltas: bias=0.0 min=0.0 av=0.0 0.0 0.0 0.0 0.0 0.0 %u\n", layer, zero_delta);
        }
        if (min_abs_step != HUGE_VALF) {
            printf("layer=%02d:   Steps: bias=%+e min=%e av=%+e %+e %+e %+e %+e %+e %u\n",
                    layer, max_bias_st, min_abs_step,
                    avg_step, steps[0], steps[weig_q1], steps[weig_q2],
                    steps[weig_q3], steps[weig_c-1], zero_step);
        } else if (steps != NULL) {
            printf("layer=%02d:   Steps: bias=0.0 min=0.0 av=0.0 0.0 0.0 0.0 0.0 0.0 %u\n", layer, zero_step);
        }
        if (min_abs_slope != HUGE_VALF) {
            printf("layer=%02d:  Slopes: bias=%+e min=%e av=%+e %+e %+e %+e %+e %+e %u\n",
                    layer, max_bias_sl, min_abs_slope,
                    avg_slope, slopes[0], slopes[weig_q1], slopes[weig_q2],
                    slopes[weig_q3], slopes[weig_c-1], zero_slope);
        } else if (slopes != NULL) {
            printf("layer=%02d:  Slopes: bias=0.0 min=0.0 av=0.0 0.0 0.0 0.0 0.0 0.0 %u\n", layer, zero_slope);
        }
        //printf("layer=%02d: MIN/MAX abs sum: %+e %+e\n", layer, fann_ff_to_float(layer_it->min_abs_sum),
        //        fann_ff_to_float(layer_it->max_abs_sum));
        if ((layer_it->zero_error != 0) || (layer_it->count_error != 0)) {
            printf("layer=%02d:  ERROR(z nz): %u %u ", layer, layer_it->zero_error, layer_it->count_error);
            if (layer_it->count_error == 0)
                printf("0.0 ");
            else
                printf("%+e ", layer_it->avg_abs_error / (double)layer_it->count_error);
            printf("%+e %+e\n", layer_it->min_abs_error, layer_it->max_abs_error);
            printf("layer=%02d:  DELTA(z nz): %u %u ", layer, layer_it->zero_delta, layer_it->count_delta);
            if (layer_it->count_delta == 0)
                printf("0.0 ");
            else
                printf("%+e ", layer_it->avg_abs_delta / (double)layer_it->count_delta);
            printf("%+e %+e\n", layer_it->min_abs_delta, layer_it->max_abs_delta);
            printf("layer=%02d:  SLOPE(z nz): %u %u ", layer, layer_it->zero_slope, layer_it->count_slope);
            if (layer_it->count_slope == 0)
                printf("0.0 ");
            else
                printf("%+e ", layer_it->avg_abs_slope / (double)layer_it->count_slope);
            printf("%+e %+e\n", layer_it->min_abs_slope, layer_it->max_abs_slope);
            printf("layer=%02d:   STEP(z nz): %u %u ", layer, layer_it->zero_step, layer_it->count_step);
            if (layer_it->count_step == 0)
                printf("0.0 ");
            else
                printf("%+e ", layer_it->avg_abs_step / (double)layer_it->count_step);
            printf("%+e %+e\n", layer_it->min_abs_step, layer_it->max_abs_step);
            fann_reset_batch_stats(layer_it);
        }
        prev_layer = layer_it;
    }
#else
    ann = ann;
#endif // FANN_PRINT_STATS
}

FANN_EXTERNAL void FANN_API fann_init_weights(struct fann *ann)//, struct fann_data *train_data)
{
    struct fann_layer *layer_it, *last_layer, *prev_layer;//, *next_layer;
    unsigned int n, w, num_input, num_output;
    struct fann_neuron *neuron_it;
    fann_type_nt min;

    prev_layer = ann->first_layer;
    last_layer = ann->last_layer;// - 1; // only the hidden ones
    for (layer_it = ann->first_layer + 1; layer_it != last_layer; ) {
        num_output = layer_it->num_neurons;
        num_input = prev_layer->num_connections;
        if ((layer_it->activation == FANN_RELU) ||
            (layer_it->activation == FANN_LEAKY_RELU)) {
            min = fann_float_to_nt(-sqrt(2.0/num_output));
        } else { // Glorot and Bengio (2010)
            // FANN_SIGMOID_STEPWISE, FANN_SIGMOID
            min = fann_float_to_nt(-sqrt(6.0/(float)(num_input + num_output)));
#ifdef STEPWISE_LUT
            if (layer_it->activation == FANN_SIGMOID_SYMMETRIC_STEPWISE)
                min = fann_nt_mul(min, fann_float_to_nt(4.0));
#endif // STEPWISE_LUT
            if (layer_it->activation == FANN_SIGMOID_SYMMETRIC)
                min = fann_nt_mul(min, fann_float_to_nt(4.0));
        }
        layer_it->max_init = fann_nt_neg(min);
        // std. dev. sym. unif: a/sqrt(3)
        layer_it->var_init = fann_nt_div(layer_it->max_init, fann_float_to_nt(1.73205080756888));
        layer_it->var_init = fann_nt_mul(layer_it->var_init, layer_it->var_init);
        num_input--; // leave bias weights zeroed
        for (n = 0; n < num_output; n++) {
            neuron_it = layer_it->neuron + n;
            for (w = 0; w < num_input ; w++) {
                neuron_it->weight[w] = fann_ff_random_weights(min, layer_it->max_init);
            }
        }
        prev_layer = layer_it;
        layer_it++;// = next_layer;
    }
    fann_clear_train_arrays(ann);
    fann_set_ff_bias();
}

FANN_EXTERNAL void FANN_API fann_print_parameters(struct fann *ann)
{
    struct fann_layer *layer_it;

    printf("Floating Point Type                  : %s\n", fann_float_type);
    printf("fann_exp()                           : %s\n", fann_exp_name);
    printf("fann_rsqrt()                         : %s\n", fann_rsqrt_name);
#ifdef FANN_THREADS
    printf("num_procs                            : %02u\n", ann->num_procs);
#endif
    printf("Input layer                          : %4d neurons, 1 bias\n", ann->num_input);
    fann_set_ff_bias();
    for(layer_it = ann->first_layer + 1; layer_it != ann->last_layer - 1; layer_it++)
    {
        printf("  Hidden layer [%02d]                  : %4u neurons, 1 bias, %s, steep=%+le\n",
               (int)(layer_it - ann->first_layer), layer_it->num_neurons,
               FANN_ACTIVATIONFUNC_NAMES[layer_it->activation],
               fann_ff_to_float(layer_it->neuron->steepness));
    }
    printf("Output layer                         : %4d neurons, ", ann->num_output);
    printf("%s, ", FANN_ACTIVATIONFUNC_NAMES[layer_it->activation]);
    printf("steep=%+le\n", fann_ff_to_float(layer_it->neuron->steepness));

    printf("Training algorithm                   : %s\n", FANN_TRAIN_NAMES[ann->training_algorithm]);
    //printf("Training loss function               : %s\n", FANN_LOSSFUNC_NAMES[ann->train_loss_function]);
    //printf("Training error function              : %s\n", FANN_ERRORFUNC_NAMES[ann->train_error_function]);
    printf("Training stop function               : %s\n", FANN_STOPFUNC_NAMES[ann->train_stop_function]);
#ifdef CALCULATE_ERROR
    printf("Bit fail limit                       : %+.9e\n", ann->bit_fail_limit);
#endif // CALCULATE_ERROR
    printf("Learning rate                        : %+.9e\n", (float)(fann_ff_to_float(ann->learning_rate)));
    printf("Learning momentum                    : %+.9e\n", (float)(fann_ff_to_float(ann->learning_momentum)));
    printf("RMSProp average                      : %+.9e\n", (float)(fann_ff_to_float(ann->rmsprop_avg)));
    //printf("Quickprop decay                      : %+.9e\n", (float)(fann_ff_to_float(ann->quickprop_decay)));
    //printf("Quickprop mu                         : %+.9e\n", (float)(fann_ff_to_float(ann->quickprop_mu)));
    printf("RPROP increase factor                : %+.9e\n", (float)(fann_ff_to_float(ann->rprop_increase_factor)));
    printf("RPROP decrease factor                : %+.9e\n", (float)(fann_ff_to_float(ann->rprop_decrease_factor)));
    printf("RPROP delta min                      : %+.9e\n", (float)(fann_ff_to_float(ann->rprop_delta_min)));
    printf("RPROP delta max                      : %+.9e\n", (float)(fann_ff_to_float(ann->rprop_delta_max)));
    printf("RPROP delta zero                     : %+.9e\n", (float)(fann_ff_to_float(ann->rprop_delta_zero)));
    /* TODO: dump scale parameters */
}

FANN_EXTERNAL unsigned int FANN_API fann_get_num_input(struct fann *ann)
{
    return ann->num_input;
}

FANN_EXTERNAL unsigned int FANN_API fann_get_num_output(struct fann *ann)
{
    return ann->num_output;
}

FANN_EXTERNAL unsigned int FANN_API fann_get_num_layers(struct fann *ann)
{
    return (unsigned int)(ann->last_layer - ann->first_layer);
}

FANN_EXTERNAL void FANN_API fann_get_layer_array(struct fann *ann, unsigned int *layers)
{
    struct fann_layer *layer_it;

    for (layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++) {
        unsigned int count = layer_it->num_neurons;
        *layers++ = count;
    }
}

FANN_EXTERNAL void FANN_API fann_get_bias_array(struct fann *ann, unsigned int *bias)
{
    struct fann_layer *layer_it;

    for (layer_it = ann->first_layer; layer_it != ann->last_layer; ++layer_it, ++bias) {
                if (layer_it != ann->last_layer-1)
                    *bias = 1;
                else
                    *bias = 0;
    }
}
#endif // FANN_INFERENCE_ONLY

/* INTERNAL FUNCTION
   Allocates the main structure and sets some default values.
 */
struct fann *fann_allocate_structure(unsigned int num_layers)
{
#ifdef FANN_THREADS
    unsigned int i;
#endif
    struct fann *ann;

    if(num_layers < 2)
    {
#ifdef DEBUG
        printf("less than 2 layers - ABORTING.\n");
#endif
        return NULL;
    }

    /* allocate and initialize the main network structure */
    fann_malloc(ann, 1);
    if(ann == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }

#ifdef FANN_THREADS
    for (i = 0; i < FANN_THREADS; i++) {
        ann->ann[i] = NULL;
        ann->thread[i] = 0;
    }
    pthread_mutex_init(&(ann->mutex), NULL);
    pthread_cond_init(&(ann->cond), NULL);
    ann->wait_procs = 0;
    ann->num_procs = 0;
#endif
    ann->num_input = 0;
    ann->num_output = 0;
#ifdef CALCULATE_LOSS
    ann->loss_count = 0;
    ann->loss_value = 0.0;//bp_0000;//fann_int_to_bp(0);
#endif // CALCULATE_LOSS
#ifdef CALCULATE_ERROR
    ann->num_bit_fail[0] = 0;
    ann->num_bit_fail[1] = 0;
    ann->num_bit_ok[0] = 0;
    ann->num_bit_ok[1] = 0;
    ann->num_max_ok = NULL;
    ann->bit_fail_limit = 0.4;//0.35);
#endif // CALCULATE_ERROR
#ifdef FANN_DATA_SCALE
    ann->scale_mean_in = NULL;
    ann->scale_deviation_in = NULL;
    ann->scale_new_min_in = NULL;
    ann->scale_factor_in = NULL;
    ann->scale_mean_out = NULL;
    ann->scale_deviation_out = NULL;
    ann->scale_new_min_out = NULL;
    ann->scale_factor_out = NULL;
#endif // FANN_DATA_SCALE

#if (defined SWF16_AP) || (defined HWF16)
    FP_BIAS = FP_BIAS_DEFAULT;
#endif

#ifndef FANN_INFERENCE_ONLY
    ann->data_input = NULL;
    ann->data_output = NULL;
    ann->data_batch = 0;
    ann->unbal_er_adjust = NULL;
    ann->learning_rate = fann_float_to_ff(0.7f);
    ann->learning_momentum = fann_int_to_ff(0);
    ann->training_algorithm = FANN_TRAIN_RPROP;
    ann->mini_batch = 0;
    //ann->train_loss_function = FANN_LOSSFUNC_MSE;
    //ann->train_error_function = FANN_ERRORFUNC_INV_TANH;
    //ann->train_error_function = FANN_ERRORFUNC_LINEAR;
    ann->train_stop_function = FANN_STOPFUNC_LOSS;
    ann->callback = NULL;
    //ann->user_data = NULL; /* User is responsible for deallocation */

    ann->rmsprop_avg = fann_float_to_ff(0.9f);
    ann->rmsprop_1mavg = fann_float_to_ff(1.0 - 0.9);

    /* Variables for use with with Quickprop training (reasonable defaults) *
    ann->quickprop_decay = fann_float_to_ff(-0.0001f);
    ann->quickprop_mu = fann_float_to_ff(1.75);*/

    /* Variables for use with with RPROP training (reasonable defaults) */
    ann->rprop_increase_factor = fann_float_to_ff(1.2f);
    ann->rprop_decrease_factor = fann_float_to_ff(0.5);
    ann->rprop_delta_min = fann_float_to_ff(0.01);//0001);
    ann->rprop_delta_max = fann_float_to_ff(50.0);
    ann->rprop_delta_zero = fann_float_to_ff(0.0125);//0.1f);
    
    /* Variables for use with SARPROP training (reasonable defaults) *
    ann->sarprop_weight_decay_shift = fann_float_to_ff(-6.644f);
    ann->sarprop_step_error_threshold_factor = fann_float_to_ff(0.1f);
    ann->sarprop_step_error_shift = fann_float_to_ff(1.385f);
    ann->sarprop_temperature = fann_float_to_ff(0.015f);*/
    ann->train_epoch = 0;
#endif // FANN_INFERENCE_ONLY
 
    //fann_init_error_data((struct fann_error *) ann);

    /* allocated only if fann_print_stats is called */
#ifndef FANN_INFERENCE_ONLY
#ifdef FANN_PRINT_STATS
    ann->stats_weigs = NULL;
    ann->stats_errors = NULL;
    ann->stats_deltas = NULL;
    ann->stats_slopes = NULL;
    ann->stats_steps = NULL;
#endif // FANN_PRINT_STATS
#endif // FANN_INFERENCE_ONLY

    /* allocate room for the layers */
    fann_calloc(ann->first_layer, num_layers);
    //ann->first_layer = (struct fann_layer *) calloc(num_layers, sizeof(struct fann_layer));
    if(ann->first_layer == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_free(ann);
        return NULL;
    }
    ann->last_layer = ann->first_layer + num_layers;
#if (defined SWF16_AP) || (defined HWF16)
    ann->change_bias = 1;
#endif // (defined SWF16_AP) || (defined HWF16)
    return ann;
}

#ifdef FANN_DATA_SCALE
/* INTERNAL FUNCTION
   Allocates room for the scaling parameters.
 */
int fann_allocate_scale(struct fann *ann)
{
    unsigned int i = 0;
#define SCALE_ALLOCATE( what, where, default_value )                    \
        fann_calloc(ann->what##_##where, ann->num_##where##put);        \
        if( ann->what##_##where == NULL )                               \
        {                                                               \
            fann_error( FANN_E_CANT_ALLOCATE_MEM );                     \
            fann_destroy( ann );                                        \
            return 1;                                                   \
        }                                                               \
        for( i = 0; i < ann->num_##where##put; i++ )                    \
            ann->what##_##where[ i ] = ( default_value );

    SCALE_ALLOCATE( scale_mean,       in,        fann_float_to_nt(0.0) )
    SCALE_ALLOCATE( scale_deviation,  in,        fann_float_to_nt(1.0) )
    SCALE_ALLOCATE( scale_new_min,    in,        fann_float_to_nt(-1.0) )
    SCALE_ALLOCATE( scale_factor,     in,        fann_float_to_nt(1.0) )

    SCALE_ALLOCATE( scale_mean,       out,       fann_float_to_nt(0.0) )
    SCALE_ALLOCATE( scale_deviation,  out,       fann_float_to_nt(1.0) )
    SCALE_ALLOCATE( scale_new_min,    out,       fann_float_to_nt(-1.0) )
    SCALE_ALLOCATE( scale_factor,     out,       fann_float_to_nt(1.0) )
#undef SCALE_ALLOCATE
    return 0;
}
#endif // FANN_DATA_SCALE

/* INTERNAL FUNCTION
   Allocates room for the neurons.
 */
int fann_allocate_neurons(struct fann *ann, struct fann *orig)
{
    struct fann_neuron * neuron, * last_neuron;
    struct fann_layer * layer_it, * prev_layer;
    unsigned int num_neurons, n, l;

    num_neurons = 0;
    layer_it = ann->first_layer;
    //layer_it->frozen = 0;
    layer_it->neuron = NULL;
    layer_it->sum_w = NULL;
    layer_it->value = NULL;
#ifndef FANN_INFERENCE_ONLY
    //layer_it->train_errors = NULL;
#endif
    //printf("%p %p\n", layer_it, layer_it->value);
    prev_layer = layer_it;
    for (++layer_it, l = 1; layer_it != ann->last_layer; l++, layer_it++) {
        fann_calloc(layer_it->value, layer_it->num_connections); // FIXME num_neurons
        if (layer_it->value == NULL) {
            fann_error(FANN_E_CANT_ALLOCATE_MEM);
            return -1;
        }
        //printf("%p %p\n", layer_it, layer_it->value);
        num_neurons = layer_it->num_neurons;
        layer_it->value[num_neurons] = ff_0000;//ff_p1k5; // FIXME marker
        fann_calloc(layer_it->neuron, num_neurons);
        fann_calloc(layer_it->sum_w, num_neurons);
        if ((layer_it->sum_w == NULL) || (layer_it->neuron == NULL)) {
            fann_error(FANN_E_CANT_ALLOCATE_MEM);
            return -1;
        }
        last_neuron = layer_it->neuron + layer_it->num_neurons;
        for (neuron = layer_it->neuron, n = 0; neuron != last_neuron; n++, neuron++) {
#ifdef FANN_THREADS
            pthread_spin_init(&(neuron->spin), PTHREAD_PROCESS_SHARED);
            neuron->step_done = 0;
#endif
            /*neuron->prev_count = 1;
            fann_calloc(neuron->prev_layer, neuron->prev_count);
            if (neuron->prev_layer == NULL) {
                fann_error(FANN_E_CANT_ALLOCATE_MEM);
                return -1;
            } 
            neuron->prev_layer[0] = prev_layer;*/
            neuron->prev_layer = prev_layer;
            neuron->steepness = ff_p050;
            if (orig) {
                neuron->weight = orig->first_layer[l].neuron[n].weight;
            } else {
                fann_calloc(neuron->weight, prev_layer->num_connections);
                if (neuron->weight == NULL) {
                    fann_error(FANN_E_CANT_ALLOCATE_MEM);
                    return -1;
                }
            }
#ifndef FANN_INFERENCE_ONLY
#if (defined SWF16_AP) || (defined HWF16)
            neuron->bp_batch_overflows = 0;
            neuron->bp_epoch_overflows = 0;
            neuron->bp_fp16_bias = FP_BIAS_DEFAULT;
#endif
            neuron->weight_slopes = NULL;
            neuron->prev_steps = NULL;
            neuron->prev_slopes = NULL;
#endif
        }
#ifndef FANN_INFERENCE_ONLY
        //layer_it->train_errors = NULL;
#endif
        prev_layer = layer_it;
    }

    return 0;
}

#ifndef FANN_INFERENCE_ONLY

#include <time.h>

/*
 Usage example:

    double f = 0;
    uint32_t diff, u;
    void * ref;

    ref = fann_start_us_count();
    for (u = 0; u < 150000000; u++) {
        f = f / 2.0;
    }
    diff = fann_stop_us_count(ref);

    printf("%u\n", diff);
 */

//#define CLOCK_TIME CLOCK_PROCESS_CPUTIME_ID
//#define CLOCK_TIME CLOCK_REALTIME

struct count_time_t {
    struct timespec ts;
    enum count_time_type type;
};

#define CLOCK_COUNT_TYPES  (COUNT_WALL_TIME + 1)
static const clockid_t count_type[CLOCK_COUNT_TYPES] = {
    CLOCK_PROCESS_CPUTIME_ID, // COUNT_CPU_TIME
    CLOCK_REALTIME,           // COUNT_WALL_TIME
};

void * fann_start_count(void * ref, const enum count_time_type type)
{
    struct count_time_t * tm;

    if (ref == NULL) {
        fann_malloc(tm, 1);
    } else {
        tm = ref;
    }
    if (tm != NULL) {
        tm->type = count_type[type];
        if (clock_gettime(tm->type, &(tm->ts)) == 0) {
            return tm;
        }
        free(tm);
    }
    return NULL;
}

static int cmp_uint32(const void *p1, const void *p2)
{
    uint32_t *u1, *u2;

    u1 = (uint32_t *)p1;
    u2 = (uint32_t *)p2;
    
    if (*u1 < *u2)
        return -1;
    if (*u1 > *u2)
        return 1;
    return 0;
}

void fann_print_count(uint32_t * us, const uint32_t len)
{
    double avg;
    uint32_t u, a, q1, q2, q3;

    a = 1;
    avg = 0.0;
    for (u = 0; u < len; u++) {
        avg += ((double)us[u] - avg) / (double)a;
        a++;
    }
    q1 = len / 4;
    q2 = len / 2;
    q3 = (3 * len) / 4;
    qsort(us, len, sizeof(*us), cmp_uint32);
    printf("avg=%f -> %u %u [%u] %u %u\n",
            avg, us[0], us[q1], us[q2], us[q3], us[len-1]);
}

uint32_t fann_stop_count_us(void * ref)
{
    struct count_time_t * handle;
    struct timespec ini, end;
    uint32_t delta = 0;

    if (ref == NULL) {
        return 0;
    }
    handle = ref;
    if (clock_gettime(handle->type, &end) != 0) {
        goto error;
    }
    ini = handle->ts;
    if (end.tv_sec == ini.tv_sec) {
        delta = (uint32_t)((end.tv_nsec - ini.tv_nsec) / 1000UL);
    } else {
        delta = (uint32_t)((end.tv_sec - ini.tv_sec - 1) * 1000000UL);
        delta += (uint32_t)(end.tv_nsec / 1000UL);
        delta += (uint32_t)((1000000000UL - ini.tv_nsec) / 1000UL);
    }
error:
    //free(ref);
    return delta;
}

uint32_t fann_stop_count_ns(void * ref)
{
    struct count_time_t * handle;
    struct timespec ini, end;
    uint32_t delta = 0;

    if (ref == NULL) {
        return 0;
    }
    handle = ref;
    if (clock_gettime(handle->type, &end) != 0) {
        goto error;
    }
    ini = handle->ts;
    if (end.tv_sec == ini.tv_sec) {
        delta = (uint32_t)(end.tv_nsec - ini.tv_nsec);
    } else {
        delta = (uint32_t)((end.tv_sec - ini.tv_sec - 1) * 1000000000UL);
        delta += (uint32_t)(end.tv_nsec);
        delta += (uint32_t)(1000000000UL - ini.tv_nsec);
    }
error:
    //free(ref);
    return delta;
}

#endif // FANN_INFERENCE_ONLY

