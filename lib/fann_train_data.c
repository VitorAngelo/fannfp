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

#include "fann.h"

/*
 * Reads training data from a file. 
 */
FANN_EXTERNAL struct fann_data *FANN_API fann_read_data_from_file(const char *configuration_file)
{
    struct fann_data *data;
    FILE *file;
    
    if (configuration_file != NULL) {
        file = fopen(configuration_file, "r");
        if (!file) {
            fann_error(FANN_E_CANT_OPEN_CONFIG_R, configuration_file);
            return NULL;
        }
    } else {
        file = stdin;
    }

    data = fann_read_data_from_fd(file, configuration_file);
    if (configuration_file != NULL) {
        fclose(file);
    }
    return data;
}

/*
 * Save training data to a file 
 */
FANN_EXTERNAL int FANN_API fann_save_data(struct fann_data *data, const char *filename)
{
    return fann_save_data_internal(data, filename);
}

/*
 * deallocate the train data structure. 
 */
FANN_EXTERNAL void FANN_API fann_destroy_data(struct fann_data *data)
{
    if(data == NULL)
        return;
    if(data->input != NULL)
        fann_free(data->input[0]);
    if(data->output != NULL)
        fann_free(data->output[0]);
    fann_free(data->input);
    fann_free(data->output);
    fann_free(data);
}

#ifdef CALCULATE_ERROR
static void * fann_batch_test(void * ref)
{
    struct fann * ann = ref;
    unsigned int data;

    fann_reset_loss(ann);
    for (data = 0; data < ann->data_batch; data++) {
        fann_test(ann, ann->data_input[data], ann->data_output[data]);
    }

    return NULL;
}

/*
 * Test a set of training data and calculate the error and loss 
 */
FANN_EXTERNAL float FANN_API fann_test_data(struct fann *ann, struct fann_data *data)
{
    int t;
    unsigned int i, tot, mini_rem, mini_th, done, np;

    if (fann_check_input_output_sizes(ann, data) == -1)
        return 0;
    
    mini_rem = data->num_data;
#ifdef FANN_THREADS
    np = ann->num_procs;
#else
    np = 1;
#endif
    mini_th = mini_rem / np;
    for (done = 0, t = np - 2; t >= -1; t--) {
        //fprintf(stderr, "t = %d\n", t);
        if ((t >= 0) && (mini_th > 0)) {
#ifdef FANN_THREADS
            struct fann * th = ann->ann[t];

            th->data_input = data->input + done;
            th->data_output = data->output + done;
            th->data_batch = mini_th;
            mini_rem -= mini_th;
            done += mini_th;
            if (pthread_create(&(ann->thread[t]), NULL, fann_batch_test, th)) {
                fprintf(stderr, "pthread error \n");
                exit(1);
            }
#endif
        } else {
            ann->data_input = data->input + done;
            ann->data_output = data->output + done;
            ann->data_batch = mini_rem;
            done += mini_rem;
            fann_batch_test(ann);
#ifdef FANN_THREADS
            if (np > 1) {
                int p;
                for (p = ann->num_procs - 2; p >= 0; p--) {
                    pthread_join(ann->thread[p], NULL);
                }
            }
#endif
            break;
        }
    }
#ifdef FANN_THREADS
    if (np > 1) {
        int p;
        for (p = ann->num_procs - 2; p >= 0; p--) {
            unsigned int o;
            struct fann *ann_p = ann->ann[p];
#ifdef CALCULATE_LOSS
            ann->loss_value += ann_p->loss_value;
            ann->loss_count += ann_p->loss_count;
#endif // CALCULATE_LOSS
            ann->num_bit_fail[0] += ann_p->num_bit_fail[0];
            ann->num_bit_fail[1] += ann_p->num_bit_fail[1];
            ann->num_bit_ok[0] += ann_p->num_bit_ok[0];
            ann->num_bit_ok[1] += ann_p->num_bit_ok[1];
            for (o = 0; o < ann->num_output; o++) {
                ann->num_max_ok[o] += ann_p->num_max_ok[o];
            }
        }
    }
#endif

    if (ann->num_output > 1) {
        tot = 0;
        for (i = 0; i < ann->num_output; i++) {
            tot += ann->num_max_ok[i];
        }
        return 100.0 * (double)(tot) / (double)data->num_data;
    }
    return 100.0 * (double)(ann->num_bit_ok[0] + ann->num_bit_ok[1]) / (double)data->num_data;
}
#endif // CALCULATE_ERROR
#endif // FANN_INFERENCE_ONLY

#define TIME_MEAS
#undef TIME_MEAS

#ifdef TIME_MEAS
static uint32_t tms = 0;

static uint32_t us_tot;
static void * ref_tot = NULL;

static uint32_t * ns_fw = NULL;
static void * ref_fw = NULL;

static uint32_t * ns_er = NULL;
static void * ref_er = NULL;

static uint32_t * ns_bw = NULL;
static void * ref_bw = NULL;

static uint32_t * ns_up = NULL;
static void * ref_up = NULL;

#define START_FW() {ref_fw = fann_start_count(ref_fw);}
#define STOP_FW() {ns_fw[i] = fann_stop_count_ns(ref_fw);}
#define START_ER() {ref_er = fann_start_count(ref_er);}
#define STOP_ER() {ns_er[i] = fann_stop_count_ns(ref_er);}
#define START_BW() {ref_bw = fann_start_count(ref_bw);}
#define STOP_BW() {ns_bw[i] = fann_stop_count_ns(ref_bw);}
#define START_UP() {ref_up = fann_start_count(ref_up);}
#define STOP_UP() {ns_up[i] = fann_stop_count_ns(ref_up);}

#else
#define START_FW()
#define STOP_FW()
#define START_ER()
#define STOP_ER()
#define START_BW()
#define STOP_BW()
#define START_UP()
#define STOP_UP()
#endif

#if 0
static void fann_norm_neurons(struct fann *ann)
{
    static double maxvar = 0.0, minvar = 10000.0;
    fann_type_ff adj;
    struct fann_neuron *neuron_it, *last_neuron;
    struct fann_layer *layer_it, *prev_layer;
    unsigned int w, num_w, k, l, n;
    double x, var, tmp, avg;
    
    for (layer_it = ann->first_layer + 1, prev_layer = ann->first_layer, l = 0;
         layer_it < ann->last_layer; layer_it++, prev_layer++, l++) {
        num_w = prev_layer->num_neurons; // BIAS will be the first
        last_neuron = layer_it->neuron + layer_it->num_neurons;
        for (neuron_it = layer_it->neuron, n = 0; neuron_it < last_neuron; neuron_it++, n++) {
            k = 1;
            var = 0.0;
            avg = fann_ff_to_float(neuron_it->weight[num_w]);
            for (w = 0; w < num_w; w++) {
                x = fann_ff_to_float(neuron_it->weight[w]); 
                k++;
                tmp = avg + ((x - avg) / (double)k);
                var += (x - avg) * (x - tmp);
                avg = tmp;
            }
            var /= (double)k;
            // do not learn bias... adjust them here...
            if (var > (4.0 * layer_it->var_init)) {
                neuron_it->steepness = fann_ff_mul(neuron_it->steepness, ff_p200);
                adj = ff_p050;
            } else if (var < (0.25 * layer_it->var_init)) {
                neuron_it->steepness = fann_ff_mul(neuron_it->steepness, ff_p050);
                adj = ff_p200;
            } else {
                continue;
            }
            w = 0;
            if (maxvar < var) {
                maxvar = var;
                w = 1;
            }
            if (minvar > var) {
                minvar = var;
                w = 1;
            }
            if (w)
            fprintf(stderr, "@%u = %u:%u -> VARIANCE = %e, STEEP = %e, SUM = %e\n", ann->train_epoch,
                    l, n, var, fann_ff_to_float(neuron_it->steepness),
                    fann_ff_to_float(layer_it->max_abs_sum));
            for (w = 0; w <= num_w; w++) {
                neuron_it->weight[w] = fann_ff_mul(neuron_it->weight[w], adj);
                neuron_it->prev_steps[w] = fann_ff_mul(neuron_it->prev_steps[w], adj);
            }
        }
    }
}
#endif // 0

#ifndef FANN_INFERENCE_ONLY
// fann_clear_weight_slopes must be zero at the beginning of each batch
static void fann_clear_weight_slopes(struct fann *ann,
        struct fann_layer *layer_begin, struct fann_layer *layer_end)
{
    unsigned int i, num_connections;
    struct fann_neuron *neuron_it, *last_neuron;
    struct fann_layer *prev_layer;

    if (layer_begin == NULL) {
        layer_begin = ann->first_layer + 1;
    }
    if (layer_end == NULL) {
        layer_end = ann->last_layer - 1;
    }

    prev_layer = layer_begin - 1;
    for (; layer_begin <= layer_end; layer_begin++) {
        last_neuron = layer_begin->neuron + layer_begin->num_neurons;
        num_connections = prev_layer->num_connections;
        for (neuron_it = layer_begin->neuron; neuron_it != last_neuron; neuron_it++) {
            if (neuron_it->weight_slopes == NULL) {
                fann_malloc(neuron_it->weight_slopes, num_connections);
                if (neuron_it->weight_slopes == NULL) {
                    fann_error(FANN_E_CANT_ALLOCATE_MEM);
                    return;
                }
            }
#ifdef FANN_THREADS
            neuron_it->step_done = 0;
#endif
#if (defined SWF16_AP) || (defined HWF16)
            if ((neuron_it->bp_batch_overflows != 0) && (neuron_it->bp_fp16_bias > 15) &&
                (ann->change_bias)) {
                neuron_it->bp_fp16_bias--;
            }
            neuron_it->bp_batch_overflows = 0;
#endif
            for (i = 0; i < num_connections; i++) {
                neuron_it->weight_slopes[i] = bp_0000;//fann_int_to_bp(0, neuron_it->bp_fp16_bias);
            }
        }
        prev_layer = layer_begin;
    }
}

#if (defined SWF16_AP) || (defined HWF16)
unsigned int bias_histogram[32];

static void fann_check_no_overflows(struct fann *ann)
{
    struct fann_neuron *neuron_it, *last_neuron;
    struct fann_layer *layer_begin, *layer_end;
    unsigned int h;

    bias_histogram[0] = 0;
    for (h = 15; h < 32; h++) {
        bias_histogram[h] = 0;
    }

    layer_begin = ann->first_layer + 1;
    layer_end = ann->last_layer - 1;
    for (; layer_begin <= layer_end; layer_begin++) {
        last_neuron = layer_begin->neuron + layer_begin->num_neurons;
        for (neuron_it = layer_begin->neuron; neuron_it != last_neuron; neuron_it++) {
            if ((neuron_it->bp_epoch_overflows == 0) && (neuron_it->bp_fp16_bias < 31) &&
                (ann->change_bias)) {
                neuron_it->bp_fp16_bias++;
            }
            bias_histogram[neuron_it->bp_fp16_bias]++;
            bias_histogram[0] += neuron_it->bp_epoch_overflows;
            neuron_it->bp_epoch_overflows = 0;
        }
    }
}
#else // (defined SWF16_AP) || (defined HWF16)
#define fann_check_no_overflows(a)
#endif

/*
 * Internal train function 
 */
#if 0
static float fann_train_epoch_quickprop(struct fann *ann, struct fann_data *data)
{
    unsigned int i, done, mini, stop, tot_mse = 0;
#ifdef CALCULATE_ERROR
    unsigned int tot_max[50];
    unsigned int tot_bits_ok[2] = {0,0}, tot_bits_fail[2] = {0,0};
#endif // CALCULATE_ERROR
    double acc_mse = 0.0;

    if (ann->mini_batch == 0) {
        mini = data->num_data;
    } else {
        mini = ann->mini_batch;
    }

#ifdef CALCULATE_ERROR
    for (i = 0; (i < 50) && (i < ann->num_output); i++) {
        tot_max[i] = 0;
    }
#endif // CALCULATE_ERROR

#ifdef TIME_MEAS
    if (tms == 0) {
        fann_malloc(ref_tot, data->num_data);
        fann_malloc(ns_fw, data->num_data);
        fann_malloc(ref_fw, data->num_data);
        fann_malloc(ns_er, data->num_data);
        fann_malloc(ref_er, data->num_data);
        fann_malloc(ns_bw, data->num_data);
        fann_malloc(ref_bw, data->num_data);
        fann_malloc(ns_up, data->num_data);
        fann_malloc(ref_up, data->num_data);
        tms = data->num_data;
    }
    ref_tot = fann_start_count(ref_tot);
#endif

    for (done = 0; done < data->num_data; done += mini) {
        fann_reset_loss(ann);
        fann_clear_weight_slopes(ann, NULL, NULL);
        stop = done + mini;
        if (stop > data->num_data) {
            stop = data->num_data;
            mini = stop - done;
        }
        for (i = done; i < stop; i++) {
            START_FW()
            fann_run(ann, data->input[i]);
            STOP_FW()

            START_ER()
            if (fann_compute_loss(ann, data->output[i])) {
                STOP_ER()

                START_BW()
                fann_backpropagate_loss(ann);
                STOP_BW()
            } else {
                STOP_ER()
            }

            START_UP()
            fann_update_slopes_batch(ann);
            STOP_UP()
        }
        fann_update_weights_quickprop(ann, mini, NULL, NULL);
#ifndef FANN_INFERENCE_ONLY
        fann_batch_stats(ann);
#endif // FANN_INFERENCE_ONLY
#ifdef CALCULATE_LOSS
        tot_mse += ann->loss_count;
        acc_mse += ann->loss_value;
#endif // CALCULATE_LOSS
#ifdef CALCULATE_ERROR
        tot_bits_fail[0] += ann->num_bit_fail[0];
        tot_bits_fail[1] += ann->num_bit_fail[1];
        tot_bits_ok[0] += ann->num_bit_ok[0];
        tot_bits_ok[1] += ann->num_bit_ok[1];
        for (i = 0; (i < 50) && (i < ann->num_output); i++) {
            tot_max[i] += ann->num_max_ok[i];
        }
#endif // CALCULATE_ERROR
    }
    fann_check_no_overflows(ann);
#ifdef CALCULATE_ERROR
    for (i = 0; (i < 50) && (i < ann->num_output); i++) {
        ann->num_max_ok[i] = tot_max[i];
    }
    ann->num_bit_fail[0] = tot_bits_fail[0];
    ann->num_bit_fail[1] = tot_bits_fail[1];
    ann->num_bit_ok[0] = tot_bits_ok[0];
    ann->num_bit_ok[1] = tot_bits_ok[1];
#endif // CALCULATE_ERROR
#ifdef TIME_MEAS
    us_tot = fann_stop_count_us(ref_tot);
    printf("tot = %f\n", 1000.0 * (double)us_tot / (double)(data->num_data));
    fann_print_count(ns_fw, data->num_data);
    fann_print_count(ns_er, data->num_data);
    fann_print_count(ns_bw, data->num_data);
    fann_print_count(ns_up, data->num_data);
#endif
    return 0.5 * acc_mse / (float)tot_mse; //fann_get_loss(ann);
}
#endif // 0

static void * fann_batch_train(void * ref)
{
    struct fann * ann = ref;
    unsigned int data;

    fann_reset_loss(ann);
    fann_clear_weight_slopes(ann, NULL, NULL);
    for (data = 0; data < ann->data_batch; data++) {
        START_FW()
        fann_run(ann, ann->data_input[data]);
        STOP_FW()

        START_ER()
        if (fann_compute_loss(ann, ann->data_output[data])) {
            STOP_ER()

            START_BW()
            fann_backpropagate_loss(ann);
            STOP_BW()
        } else {
            STOP_ER()
        }

        START_UP()
        fann_update_slopes_batch(ann);
        STOP_UP()
    }
#ifdef FANN_THREADS
    if (ann->num_procs == 0) {
        ann = ann->ann[0];
    }
    if (ann->num_procs != 1) {
        pthread_mutex_lock(&(ann->mutex));
        ann->wait_procs--;
        while (ann->wait_procs > 0) {
            pthread_cond_wait(&(ann->cond), &(ann->mutex));
        }
        pthread_mutex_unlock(&(ann->mutex));
        pthread_cond_signal(&(ann->cond));
    }
#endif
    fann_update_weights_irpropm(ann);
    return NULL;
}

/*
 * Internal train function 
 */
static double fann_train_epoch_irpropm(struct fann *ann, struct fann_data *data)
{
    int t;
    unsigned int k, np;
#ifdef CALCULATE_LOSS
    double tmp;
#endif // CALCULATE_LOSS
    double x, var, avg;
    static double last_ratio = 1e3;
    double loss;
    unsigned int mini_rem, mini_th;
    unsigned int done, mini, stop, tot_mse = 0;
#ifdef CALCULATE_ERROR
    unsigned int i, tot_max[50];
    unsigned int tot_bits_ok[2] = {0,0}, tot_bits_fail[2] = {0,0};
#endif // CALCULATE_ERROR
    double acc_mse = 0.0;

    k = 0;
    var = avg = 0.0;
#ifdef FANN_THREADS
    np = ann->num_procs;
#else
    np = 1;
#endif
    if (ann->mini_batch < np) {
        mini = data->num_data;
    } else {
        mini = ann->mini_batch;
    }

#ifdef CALCULATE_ERROR
    for (i = 0; (i < 50) && (i < ann->num_output); i++) {
        tot_max[i] = 0;
    }
#endif // CALCULATE_ERROR

#ifdef TIME_MEAS
    if (tms == 0) {
        fann_malloc(ref_tot, data->num_data);
        fann_malloc(ns_fw, data->num_data);
        fann_malloc(ref_fw, data->num_data);
        fann_malloc(ns_er, data->num_data);
        fann_malloc(ref_er, data->num_data);
        fann_malloc(ns_bw, data->num_data);
        fann_malloc(ref_bw, data->num_data);
        fann_malloc(ns_up, data->num_data);
        fann_malloc(ref_up, data->num_data);
        tms = data->num_data;
    }
    ref_tot = fann_start_count(ref_tot);
#endif

    for (done = 0; done < data->num_data; ) {
        stop = done + mini;
        if (stop > data->num_data) {
            stop = data->num_data;
            mini = stop - done;
        }
        mini_rem = mini;
        mini_th = mini_rem / np;
#ifdef FANN_THREADS
        ann->wait_procs = ann->num_procs;
#endif
        for (t = np - 2; t >= -1; t--) {
            if ((t >= 0) && (mini_th > 0)) {
#ifdef FANN_THREADS
                struct fann * th = ann->ann[t];

                th->data_input = data->input + done;
                th->data_output = data->output + done;
                th->data_batch = mini_th;
                mini_rem -= mini_th;
                done += mini_th;
                if (pthread_create(&(ann->thread[t]), NULL, fann_batch_train, th)) {
                    fprintf(stderr, "pthread error \n");
                    exit(1);
                }
#endif
            } else {
                ann->data_input = data->input + done;
                ann->data_output = data->output + done;
                ann->data_batch = mini_rem;
                done += mini_rem;
                fann_batch_train(ann);
#ifdef FANN_THREADS
                if (ann->num_procs > 1) {
                    int p;
                    for (p = ann->num_procs - 2; p >= 0; p--) {
                        pthread_join(ann->thread[p], NULL);
                    }
                }
#endif
                break;
            }
        }
#ifdef FANN_THREADS
        if (ann->num_procs > 1) {
            int p;
            for (p = ann->num_procs - 2; p >= 0; p--) {
                unsigned int o;
                struct fann *ann_p = ann->ann[p];
#ifdef CALCULATE_LOSS
                ann->loss_value += ann_p->loss_value;
                ann->loss_count += ann_p->loss_count;
#endif // CALCULATE_LOSS
                ann->num_bit_fail[0] += ann_p->num_bit_fail[0];
                ann->num_bit_fail[1] += ann_p->num_bit_fail[1];
                ann->num_bit_ok[0] += ann_p->num_bit_ok[0];
                ann->num_bit_ok[1] += ann_p->num_bit_ok[1];
                for (o = 0; o < ann->num_output; o++) {
                    ann->num_max_ok[o] += ann_p->num_max_ok[o];
                }
            }
        }
#endif

#ifdef CALCULATE_LOSS
        x = 0.5 * (double)ann->loss_value / (double)ann->loss_count;
        if (k) {
            k++;
            tmp = avg + ((x - avg) / (double)k);
            var += (x - avg) * (x - tmp);
            avg = tmp;
        } else {
            k = 1;
            avg = x;
        }
        tot_mse += ann->loss_count;
        acc_mse += ann->loss_value;
#endif // CALCULATE_LOSS
#ifndef FANN_INFERENCE_ONLY
        fann_batch_stats(ann);
#endif // FANN_INFERENCE_ONLY
#ifdef CALCULATE_ERROR
        tot_bits_fail[0] += ann->num_bit_fail[0];
        tot_bits_fail[1] += ann->num_bit_fail[1];
        tot_bits_ok[0] += ann->num_bit_ok[0];
        tot_bits_ok[1] += ann->num_bit_ok[1];
        for (i = 0; (i < 50) && (i < ann->num_output); i++) {
            tot_max[i] += ann->num_max_ok[i];
        }
#endif // CALCULATE_ERROR
    }
    fann_check_no_overflows(ann);
#ifdef CALCULATE_ERROR
    for (i = 0; (i < 50) && (i < ann->num_output); i++) {
        ann->num_max_ok[i] = tot_max[i];
    }
    ann->num_bit_fail[0] = tot_bits_fail[0];
    ann->num_bit_fail[1] = tot_bits_fail[1];
    ann->num_bit_ok[0] = tot_bits_ok[0];
    ann->num_bit_ok[1] = tot_bits_ok[1];
#endif // CALCULATE_ERROR
#ifdef TIME_MEAS
    us_tot = fann_stop_count_us(ref_tot);
    printf("tot = %f\n", 1000.0 * (double)us_tot / (double)(data->num_data));
    fann_print_count(ns_fw, data->num_data);
    fann_print_count(ns_er, data->num_data);
    fann_print_count(ns_bw, data->num_data);
    fann_print_count(ns_up, data->num_data);
#endif
    //fann_norm_neurons(ann);
    loss = 0.5 * acc_mse / (float)tot_mse;
    if (ann->mini_batch != 0) {
        var /= (double)k;
        x = sqrt(var);
        if ((x / avg) > last_ratio) {
            ann->mini_batch = 0;
        }
        last_ratio = x / avg;
    }
    return loss;
}

/*
 * Internal train function 
 *
float fann_train_epoch_sarprop(struct fann *ann, struct fann_data *data)
{
    unsigned int i;

    //if(ann->prev_slopes == NULL)
    {
        fann_clear_train_arrays(ann);
    }
        //fann_zero_train_arrays(ann, NULL, NULL);

    fann_reset_loss(ann);

    for(i = 0; i < data->num_data; i++)
    {
        fann_run(ann, data->input[i]);
        if (fann_compute_loss(ann, data->output[i])) {
            fann_backpropagate_loss(ann);
        }
        fann_update_slopes_batch(ann, ann->first_layer + 1, ann->last_layer - 1);
    }
    fann_check_no_overflows(ann);
    if (ann->stats_weigs != NULL) {
        fann_batch_stats(ann);
    }

    //fann_update_weights_sarprop(ann, ann->sarprop_epoch, 0, ann->total_connections);

    ++(ann->sarprop_epoch);

    return fann_get_loss(ann);
}*/

/*
 * Internal train function 
 */
static float fann_train_epoch_batch(struct fann *ann, struct fann_data *data)
{
    unsigned int i, done, mini, stop, tot_mse = 0;
#ifdef CALCULATE_ERROR
    unsigned int tot_max[50];
    unsigned int tot_bits_ok[2] = {0,0}, tot_bits_fail[2] = {0,0};
#endif // CALCULATE_ERROR
    double acc_mse = 0.0;

    if (ann->mini_batch == 0) {
        mini = data->num_data;
    } else {
        mini = ann->mini_batch;
    }

#ifdef CALCULATE_ERROR
    for (i = 0; (i < 50) && (i < ann->num_output); i++) {
        tot_max[i] = 0;
    }
#endif // CALCULATE_ERROR

#ifdef TIME_MEAS
    if (tms == 0) {
        fann_malloc(ref_tot, data->num_data);
        fann_malloc(ns_fw, data->num_data);
        fann_malloc(ref_fw, data->num_data);
        fann_malloc(ns_er, data->num_data);
        fann_malloc(ref_er, data->num_data);
        fann_malloc(ns_bw, data->num_data);
        fann_malloc(ref_bw, data->num_data);
        fann_malloc(ns_up, data->num_data);
        fann_malloc(ref_up, data->num_data);
        tms = data->num_data;
    }
    ref_tot = fann_start_count(ref_tot);
#endif

    for (done = 0; done < data->num_data; done += mini) {
        fann_reset_loss(ann);
        fann_clear_weight_slopes(ann, NULL, NULL);
        stop = done + mini;
        if (stop > data->num_data) {
            stop = data->num_data;
            mini = stop - done;
        }
        for (i = done; i < stop; i++) {
            START_FW()
            fann_run(ann, data->input[i]);
            STOP_FW()

            START_ER()
            if (fann_compute_loss(ann, data->output[i])) {
                STOP_ER()

                START_BW()
                fann_backpropagate_loss(ann);
                STOP_BW()
            } else {
                STOP_ER()
            }

            START_UP()
            fann_update_slopes_batch(ann);
            STOP_UP()
        }
        fann_update_weights_batch(ann, /*mini,*/ NULL, NULL);
#ifndef FANN_INFERENCE_ONLY
        fann_batch_stats(ann);
#endif // FANN_INFERENCE_ONLY
#ifdef CALCULATE_LOSS
        tot_mse += ann->loss_count;
        acc_mse += ann->loss_value;
#endif // CALCULATE_LOSS
#ifdef CALCULATE_ERROR
        tot_bits_fail[0] += ann->num_bit_fail[0];
        tot_bits_fail[1] += ann->num_bit_fail[1];
        tot_bits_ok[0] += ann->num_bit_ok[0];
        tot_bits_ok[1] += ann->num_bit_ok[1];
        for (i = 0; (i < 50) && (i < ann->num_output); i++) {
            tot_max[i] += ann->num_max_ok[i];
        }
#endif // CALCULATE_ERROR
    }
    fann_check_no_overflows(ann);
#ifdef CALCULATE_ERROR
    for (i = 0; (i < 50) && (i < ann->num_output); i++) {
        ann->num_max_ok[i] = tot_max[i];
    }
    ann->num_bit_fail[0] = tot_bits_fail[0];
    ann->num_bit_fail[1] = tot_bits_fail[1];
    ann->num_bit_ok[0] = tot_bits_ok[0];
    ann->num_bit_ok[1] = tot_bits_ok[1];
#endif // CALCULATE_ERROR
#ifdef TIME_MEAS
    us_tot = fann_stop_count_us(ref_tot);
    printf("tot = %f\n", 1000.0 * (double)us_tot / (double)(data->num_data));
    fann_print_count(ns_fw, data->num_data);
    fann_print_count(ns_er, data->num_data);
    fann_print_count(ns_bw, data->num_data);
    fann_print_count(ns_up, data->num_data);
#endif
    return 0.5 * acc_mse / (float)tot_mse; //fann_get_loss(ann);
}

/*
 * Internal train function 
 */
static float fann_train_epoch_rmsprop(struct fann *ann, struct fann_data *data)
{
    unsigned int i, done, mini, stop, tot_mse = 0;
#ifdef CALCULATE_ERROR
    unsigned int tot_max[50];
    unsigned int tot_bits_ok[2] = {0,0}, tot_bits_fail[2] = {0,0};
#endif // CALCULATE_ERROR
    double acc_mse = 0.0;

    if (ann->mini_batch == 0) {
        mini = data->num_data;
    } else {
        mini = ann->mini_batch;
    }

#ifdef CALCULATE_ERROR
    for (i = 0; (i < 50) && (i < ann->num_output); i++) {
        tot_max[i] = 0;
    }
#endif // CALCULATE_ERROR

#ifdef TIME_MEAS
    if (tms == 0) {
        fann_malloc(ref_tot, data->num_data);
        fann_malloc(ns_fw, data->num_data);
        fann_malloc(ref_fw, data->num_data);
        fann_malloc(ns_er, data->num_data);
        fann_malloc(ref_er, data->num_data);
        fann_malloc(ns_bw, data->num_data);
        fann_malloc(ref_bw, data->num_data);
        fann_malloc(ns_up, data->num_data);
        fann_malloc(ref_up, data->num_data);
        tms = data->num_data;
    }
    ref_tot = fann_start_count(ref_tot);
#endif

    for (done = 0; done < data->num_data; done += mini) {
        fann_reset_loss(ann);
        fann_clear_weight_slopes(ann, NULL, NULL);
        stop = done + mini;
        if (stop > data->num_data) {
            stop = data->num_data;
            mini = stop - done;
        }
        for (i = done; i < stop; i++) {
            START_FW()
            fann_run(ann, data->input[i]);
            STOP_FW()

            START_ER()
            if (fann_compute_loss(ann, data->output[i])) {
                STOP_ER()

                START_BW()
                fann_backpropagate_loss(ann);
                STOP_BW()
            } else {
                STOP_ER()
            }

            START_UP()
            fann_update_slopes_batch(ann);
            STOP_UP()
        }
        fann_update_weights_rmsprop(ann, /*mini,*/ NULL, NULL);
#ifndef FANN_INFERENCE_ONLY
        fann_batch_stats(ann);
#endif // FANN_INFERENCE_ONLY
#ifdef CALCULATE_LOSS
        tot_mse += ann->loss_count;
        acc_mse += ann->loss_value;
#endif // CALCULATE_LOSS
#ifdef CALCULATE_ERROR
        tot_bits_fail[0] += ann->num_bit_fail[0];
        tot_bits_fail[1] += ann->num_bit_fail[1];
        tot_bits_ok[0] += ann->num_bit_ok[0];
        tot_bits_ok[1] += ann->num_bit_ok[1];
        for (i = 0; (i < 50) && (i < ann->num_output); i++) {
            tot_max[i] += ann->num_max_ok[i];
        }
#endif // CALCULATE_ERROR
    }
    fann_check_no_overflows(ann);
#ifdef CALCULATE_ERROR
    for (i = 0; (i < 50) && (i < ann->num_output); i++) {
        ann->num_max_ok[i] = tot_max[i];
    }
    ann->num_bit_fail[0] = tot_bits_fail[0];
    ann->num_bit_fail[1] = tot_bits_fail[1];
    ann->num_bit_ok[0] = tot_bits_ok[0];
    ann->num_bit_ok[1] = tot_bits_ok[1];
#endif // CALCULATE_ERROR
#ifdef TIME_MEAS
    us_tot = fann_stop_count_us(ref_tot);
    printf("tot = %f\n", 1000.0 * (double)us_tot / (double)(data->num_data));
    fann_print_count(ns_fw, data->num_data);
    fann_print_count(ns_er, data->num_data);
    fann_print_count(ns_bw, data->num_data);
    fann_print_count(ns_up, data->num_data);
#endif
    return 0.5 * acc_mse / (float)tot_mse; //fann_get_loss(ann);
}


/*
 * Internal train function 
 */
static float fann_train_epoch_incremental(struct fann *ann, struct fann_data *data)
{
    unsigned int i;

    fann_reset_loss(ann);
    if (ann->first_layer[1].neuron[0].weight_slopes == NULL)
        fann_clear_weight_slopes(ann, NULL, NULL);

#ifdef TIME_MEAS
    if (tms == 0) {
        fann_malloc(ref_tot, data->num_data);
        fann_malloc(ns_fw, data->num_data);
        fann_malloc(ref_fw, data->num_data);
        fann_malloc(ns_er, data->num_data);
        fann_malloc(ref_er, data->num_data);
        fann_malloc(ns_bw, data->num_data);
        fann_malloc(ref_bw, data->num_data);
        fann_malloc(ns_up, data->num_data);
        fann_malloc(ref_up, data->num_data);
        tms = data->num_data;
    }
    ref_tot = fann_start_count(ref_tot);
#endif

    for (i = 0; i != data->num_data; i++) {
        START_FW()
        fann_run(ann, data->input[i]);
        STOP_FW()

        START_ER()
        if (fann_compute_loss(ann, data->output[i])) {
            STOP_ER()

            START_BW()
            fann_backpropagate_loss(ann);
            STOP_BW()
        } else {
            STOP_ER()
        }

        START_UP()
        fann_update_weights_incremental(ann);
        STOP_UP()

#ifndef FANN_INFERENCE_ONLY
        fann_batch_stats(ann);
#endif // FANN_INFERENCE_ONLY
    }
    fann_check_no_overflows(ann);

#ifdef TIME_MEAS
    us_tot = fann_stop_count_us(ref_tot);
    printf("tot = %f\n", 1000.0 * (double)us_tot / (double)(data->num_data));
    fann_print_count(ns_fw, data->num_data);
    fann_print_count(ns_er, data->num_data);
    fann_print_count(ns_bw, data->num_data);
    fann_print_count(ns_up, data->num_data);
#endif

    //fann_norm_neurons(ann);
    return fann_get_loss(ann);
}

unsigned int fann_train_shuffle = 10000;

/*
 * Train for one epoch with the selected training algorithm 
 */
FANN_EXTERNAL float FANN_API fann_train_epoch(struct fann *ann, struct fann_data *data)
{
    if(fann_check_input_output_sizes(ann, data) == -1)
        return 0;
    
    if (fann_train_shuffle > 0) {
        fann_train_shuffle--;
        fann_shuffle_data(data);
    }
   
    ann->train_epoch++;
    switch (ann->training_algorithm)
    {
    //case FANN_TRAIN_QUICKPROP:
    //    return fann_train_epoch_quickprop(ann, data);
    case FANN_TRAIN_RPROP:
        return fann_train_epoch_irpropm(ann, data);
    case FANN_TRAIN_RMSPROP:
        return fann_train_epoch_rmsprop(ann, data);
    //case FANN_TRAIN_SARPROP:
    //    return fann_train_epoch_sarprop(ann, data);
    case FANN_TRAIN_BATCH:
        return fann_train_epoch_batch(ann, data);
    case FANN_TRAIN_INCREMENTAL:
        return fann_train_epoch_incremental(ann, data);
    }
    return 0;
}

FANN_EXTERNAL void FANN_API fann_train_on_data(struct fann *ann, struct fann_data *data,
                                               unsigned int max_epochs,
                                               unsigned int epochs_between_reports,
                                               float desired_error)
{
#ifdef CALCULATE_ERROR
    float error;
#endif // CALCULATE_ERROR
    int desired_error_reached = -1;

#ifdef DEBUG
    printf("Training with %s\n", FANN_TRAIN_NAMES[ann->training_algorithm]);
#endif

    if(epochs_between_reports && ann->callback == NULL)
    {
        printf("Max epochs %8d. Desired error: %.10f.\n", max_epochs, desired_error);
    }
    ann->train_epoch = 0;

    while (ann->train_epoch < max_epochs) {
        /*
         * train 
         */
#ifdef CALCULATE_ERROR
        error = fann_train_epoch(ann, data);
        desired_error_reached = fann_desired_error_reached(ann, desired_error);
#else
        fann_train_epoch(ann, data);
#endif // ! CALCULATE_ERROR

        /*
         * print current output 
         */
        if(epochs_between_reports &&
           (ann->train_epoch % epochs_between_reports == 0 || ann->train_epoch == max_epochs || ann->train_epoch  == 1 ||
            desired_error_reached == 0))
        {
            if(ann->callback == NULL) {
#ifdef CALCULATE_ERROR
                printf("Epochs     %8d. Current error: %.10f. Bit fail %u,%u.\n", ann->train_epoch, error,
                       ann->num_bit_fail[0], ann->num_bit_fail[1]);
#else
                printf("Epochs     %8d.\n", ann->train_epoch);
#endif // ! CALCULATE_ERROR
            } else {
                int ret = ((*ann->callback)(ann, data, max_epochs, epochs_between_reports, 
                                      desired_error));
                if (ret == -1) {
                    /* you can break the training by returning -1 */
                    break;
                } else if (ret == 1) {
                    epochs_between_reports = 1;
                }
            }
        }

        if(desired_error_reached == 0)
            break;
    }
}

FANN_EXTERNAL void FANN_API fann_train_on_file(struct fann *ann, const char *filename,
                                               unsigned int max_epochs,
                                               unsigned int epochs_between_reports,
                                               float desired_error)
{
    struct fann_data *data = fann_read_data_from_file(filename);

    if(data == NULL)
    {
        return;
    }
    fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);
    fann_destroy_data(data);
}

/*
 * shuffles training data, randomizing the order 
 */
FANN_EXTERNAL void FANN_API fann_shuffle_data(struct fann_data *train_data)
{
    unsigned int dat = 0, elem, swap;
    fann_type_ff temp;

    for(; dat < train_data->num_data; dat++)
    {
        swap = (unsigned int) (rand() % train_data->num_data);
        if(swap != dat)
        {
            for(elem = 0; elem < train_data->num_input; elem++)
            {
                temp = train_data->input[dat][elem];
                train_data->input[dat][elem] = train_data->input[swap][elem];
                train_data->input[swap][elem] = temp;
            }
            for(elem = 0; elem < train_data->num_output; elem++)
            {
                temp = train_data->output[dat][elem];
                train_data->output[dat][elem] = train_data->output[swap][elem];
                train_data->output[swap][elem] = temp;
            }
        }
    }
}

/*
 * INTERNAL FUNCTION calculates min and max of each feature in data
 */
#ifdef FANN_DATA_SCALE
static void fann_get_min_max_data(fann_type_ff ** data, unsigned int num_data, unsigned int num_elem,
        struct fann_ff_limits * lim)
{
    fann_type_ff temp;
    unsigned int dat, elem;

    for (elem = 0; elem < num_elem; elem++) {
        lim[elem].min = lim[elem].max = data[0][elem];
    }
    for(dat = 1; dat < num_data; dat++)
    {
        for(elem = 0; elem < num_elem; elem++)
        {
            temp = data[dat][elem];
            if (fann_ff_lt(temp, lim[elem].min))
                lim[elem].min = temp;
            else if (fann_ff_gt(temp, lim[elem].max))
                lim[elem].max = temp;
        }
    }
}

FANN_EXTERNAL float FANN_API fann_get_min_data_input(struct fann_data *train_data)
{
    struct fann_ff_limits lim;
    fann_get_min_max_data(train_data->input, train_data->num_data, train_data->num_input, &lim);
    fann_set_ff_bias();
    return fann_ff_to_float(lim.min);
}

FANN_EXTERNAL float FANN_API fann_get_max_data_input(struct fann_data *train_data)
{
    struct fann_ff_limits lim;
    fann_get_min_max_data(train_data->input, train_data->num_data, train_data->num_input, &lim);
    fann_set_ff_bias();
    return fann_ff_to_float(lim.max);
}

FANN_EXTERNAL float FANN_API fann_get_min_data_output(struct fann_data *train_data)
{
    struct fann_ff_limits lim;
    fann_get_min_max_data(train_data->output, train_data->num_data, train_data->num_output, &lim);
    fann_set_ff_bias();
    return fann_ff_to_float(lim.min);
}

FANN_EXTERNAL float FANN_API fann_get_max_data_output(struct fann_data *train_data)
{
    struct fann_ff_limits lim;
    fann_get_min_max_data(train_data->output, train_data->num_data, train_data->num_output, &lim);
    fann_set_ff_bias();
    return fann_ff_to_float(lim.max);
}
#endif // FANN_DATA_SCALE

FANN_EXTERNAL void fann_unbalance_adjust(struct fann *ann, unsigned int * class_count)
{
    unsigned int o, min;

    if (ann->unbal_er_adjust == NULL) {
        fann_malloc(ann->unbal_er_adjust, ann->num_output);
    }
    if (ann->unbal_er_adjust == NULL) {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        return;
    }
    min = UINT32_MAX;
    for (o = 0; o < ann->num_output; o++) {
        if (min > class_count[o])
            min = class_count[o];
    }
    fann_set_ff_bias();

    for (o = 0; o < ann->num_output; o++) {
        ann->unbal_er_adjust[o] = fann_float_to_ff((double)min/(double)class_count[o]);
        printf("%lf\n", fann_ff_to_float(ann->unbal_er_adjust[o]));
    }
}

FANN_EXTERNAL unsigned int fann_count_classes(struct fann_data * data, unsigned int ** class_count,
        float threshold_f, unsigned int * max_idx, unsigned int * min_idx)
{
    unsigned int dat, elem, max_class, min_class, noclass;
    fann_type_ff ** out = data->output, threshold;

    fann_set_ff_bias();
    threshold = fann_float_to_ff(threshold_f);

    if (*class_count == NULL) {
        fann_malloc(*class_count, data->num_output);
    }
    if (*class_count == NULL) {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        return 0;
    }
    for (elem = 0; elem < data->num_output; elem++) {
        (*class_count)[elem] = 0;
    }
    for (noclass = dat = 0; dat < data->num_data; dat++) {
        int match = 0;

        for (elem = 0; elem < data->num_output; elem++) {
            if (fann_ff_gt(out[dat][elem], threshold)) {
                (*class_count)[elem]++;
                if (match) {
                    fprintf(stderr, "ERROR: classes not mutually exclusive\n");
                    return 0;
                }
                match = 1;
            }
        }
        if (match == 0) {
            noclass++;
        }
    }
    *max_idx = data->num_output;
    if (min_idx != NULL) {
        *min_idx = data->num_output;
    }
    min_class = data->num_data; // make GCC happy
    for (max_class = elem = 0; elem < data->num_output; elem++) {
        if ((*class_count)[elem] > max_class) {
            max_class = (*class_count)[elem];
            *max_idx = elem;
        }
        if ((min_idx != NULL) && ((*class_count)[elem] < min_class)) {
            min_class = (*class_count)[elem];
            *min_idx = elem;
        }
    }
    return noclass;
}

#ifdef FANN_DATA_SCALE
/*
 * Scales data to a specific range 
 */
FANN_EXTERNAL void fann_scale_data_linear(fann_type_ff ** data, unsigned int num_data, unsigned int num_elem,
                     struct fann_ff_limits ** old_l, struct fann_ff_limits new_l)
{
    if (*old_l == NULL) {
        fann_malloc(*old_l, num_elem);
        if (*old_l != NULL) {
            fann_get_min_max_data(data, num_data, num_elem, *old_l);
            fann_scale_data_to_range(data, num_data, num_elem, *old_l, new_l);
        }
    } else {
        fann_scale_data_to_range(data, num_data, num_elem, *old_l, new_l);
    }
}

/*
 * Scales data to a specific range 
 */
FANN_EXTERNAL void FANN_API fann_scale_data_to_range(fann_type_ff ** data, unsigned int num_data, unsigned int num_elem,
                     struct fann_ff_limits * old_l, struct fann_ff_limits new_l)
{
    unsigned int dat, elem;
    fann_type_nt temp, old_span, new_span, factor;
    fann_type_nt new_minf, new_maxf;

    fann_set_ff_bias();
    new_minf = fann_ff_to_nt(new_l.min);
    new_maxf = fann_ff_to_nt(new_l.max);
    new_span = fann_nt_sub(new_maxf, new_minf);

    for(elem = 0; elem < num_elem; elem++)
    {
        old_span = fann_nt_sub(fann_ff_to_nt(old_l[elem].max), fann_ff_to_nt(old_l[elem].min));
        factor = fann_nt_div(new_span, old_span);
        //fprintf(stderr, "max %f, min %f, factor %f = %f / %f, #%u\n", fann_ff_to_float(old_max[elem]),
        //        fann_ff_to_float(old_min[elem]), factor, new_span, old_span, elem);
        if (fann_nt_is_zero(old_span)) {
            fann_type_ff nil = fann_nt_to_ff(fann_nt_div(new_span, fann_float_to_nt(2.0)));
            for (dat = 0; dat < num_data; dat++) {
                data[dat][elem] = nil;
            }
            continue;
        }
        for (dat = 0; dat < num_data; dat++) {
            temp = fann_nt_add(fann_nt_mul(fann_nt_sub(fann_ff_to_nt(data[dat][elem]), fann_ff_to_nt(old_l[elem].min)), factor), new_minf);
            if(fann_nt_lt(temp, new_minf)) {
                data[dat][elem] = new_l.min;
#ifndef FANN_INFERENCE_ONLY
                printf("%s|%s error %f < %f\n", __FILE__, __FUNCTION__, fann_nt_to_float(temp), fann_nt_to_float(new_minf));
#endif
            } else if(fann_nt_gt(temp, new_maxf)) {
                data[dat][elem] = new_l.max;
#ifndef FANN_INFERENCE_ONLY
                printf("%s|%s error %f > %f\n", __FILE__, __FUNCTION__, fann_nt_to_float(temp), fann_nt_to_float(new_maxf));
#endif
            } else {
                //printf(",%f", temp); 
                data[dat][elem] = fann_nt_to_ff(temp);
            }
        }
        //printf("\n");
    }
}

/*
 * Scales the inputs in the training data to the specified range 
 */
FANN_EXTERNAL void FANN_API fann_scale_input_data_linear(struct fann_data *train_data,
                     struct fann_ff_limits ** old_l, struct fann_ff_limits new_l)
{
    fann_scale_data_linear(train_data->input, train_data->num_data, train_data->num_input,
                    old_l, new_l);
}

/*
 * Scales the inputs in the training data to the specified range 
 */
FANN_EXTERNAL void FANN_API fann_scale_output_data_linear(struct fann_data *train_data,
                     struct fann_ff_limits ** old_l, struct fann_ff_limits new_l)
{
    fann_scale_data_linear(train_data->output, train_data->num_data, train_data->num_output,
                    old_l, new_l);
}
#endif // FANN_DATA_SCALE

/*
 * merges training data into a single struct. 
 */
FANN_EXTERNAL struct fann_data *FANN_API fann_merge_data(struct fann_data *data1,
                                                                     struct fann_data *data2)
{
    unsigned int i;
    fann_type_ff *data_input, *data_output;
    struct fann_data *dest;// =

    fann_malloc(dest, 1);
    if(dest == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }

    if((data1->num_input != data2->num_input) || (data1->num_output != data2->num_output))
    {
        fann_error(FANN_E_TRAIN_DATA_MISMATCH);
        return NULL;
    }

    dest->num_data = data1->num_data+data2->num_data;
    dest->num_input = data1->num_input;
    dest->num_output = data1->num_output;
    fann_calloc(dest->input, dest->num_data);
    if(dest->input == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_data(dest);
        return NULL;
    }

    fann_calloc(dest->output, dest->num_data);
    if(dest->output == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_data(dest);
        return NULL;
    }

    fann_calloc(data_input, (dest->num_input * dest->num_data));
    if(data_input == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_data(dest);
        return NULL;
    }
    fann_memcpy(data_input, data1->input[0], (dest->num_input * data1->num_data));
    fann_memcpy(data_input + (dest->num_input*data1->num_data), 
                data2->input[0], (dest->num_input * data2->num_data));

    fann_calloc(data_output, (dest->num_output * dest->num_data));
    if(data_output == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_data(dest);
        return NULL;
    }
    fann_memcpy(data_output, data1->output[0], (dest->num_output * data1->num_data));
    fann_memcpy((data_output + (dest->num_output*data1->num_data)),
                data2->output[0], (dest->num_output * data2->num_data));

    for(i = 0; i != dest->num_data; i++)
    {
        dest->input[i] = data_input;
        data_input += dest->num_input;
        dest->output[i] = data_output;
        data_output += dest->num_output;
    }
    return dest;
}

/*
 * return a copy of a fann_data struct 
 */
FANN_EXTERNAL struct fann_data *FANN_API fann_duplicate_data(struct fann_data
                                                                         *data)
{
    unsigned int i;
    fann_type_ff *data_input, *data_output;
    struct fann_data *dest;

    fann_malloc(dest, 1);
    if(dest == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }

    dest->num_data = data->num_data;
    dest->num_input = data->num_input;
    dest->num_output = data->num_output;
    fann_calloc(dest->input, dest->num_data);
    if(dest->input == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_data(dest);
        return NULL;
    }

    fann_calloc(dest->output, dest->num_data);
    if(dest->output == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_data(dest);
        return NULL;
    }

    fann_calloc(data_input, (dest->num_input * dest->num_data));
    if(data_input == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_data(dest);
        return NULL;
    }
    fann_memcpy(data_input, data->input[0], (dest->num_input * dest->num_data));

    fann_calloc(data_output, (dest->num_output * dest->num_data));
    if(data_output == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_data(dest);
        return NULL;
    }
    fann_memcpy(data_output, data->output[0], (dest->num_output * dest->num_data));

    for(i = 0; i != dest->num_data; i++)
    {
        dest->input[i] = data_input;
        data_input += dest->num_input;
        dest->output[i] = data_output;
        data_output += dest->num_output;
    }
    return dest;
}

/*FANN_EXTERNAL struct fann_data *FANN_API fann_getref_data(struct fann_data *data, unsigned int pos)
{
    static struct fann_data dataret = {0, 0, 0, NULL, NULL};

    dataret.num_data = data->num_data - pos;
    dataret.num_input = data->num_input;
    dataret.num_output = data->num_output;
    dataret.input = &(data->input[pos]);
    dataret.output = &(data->output[pos]);
    return &dataret;
}*/

FANN_EXTERNAL struct fann_data *FANN_API fann_subset_data(struct fann_data
                                                                         *data, unsigned int pos,
                                                                         unsigned int length)
{
    unsigned int i;
    fann_type_ff *data_input, *data_output;
    struct fann_data *dest;

    fann_malloc(dest, 1);
    if(dest == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }
    
    if(pos > data->num_data || pos+length > data->num_data)
    {
        fann_error(FANN_E_TRAIN_DATA_SUBSET, pos, length, data->num_data);
        return NULL;
    }

    dest->num_data = length;
    dest->num_input = data->num_input;
    dest->num_output = data->num_output;
    fann_calloc(dest->input, dest->num_data);
    if(dest->input == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_data(dest);
        return NULL;
    }

    fann_calloc(dest->output, dest->num_data);
    if(dest->output == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_data(dest);
        return NULL;
    }

    fann_calloc(data_input, (dest->num_input * dest->num_data));
    if(data_input == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_data(dest);
        return NULL;
    }
    fann_memcpy(data_input, data->input[pos], (dest->num_input * dest->num_data));

    fann_calloc(data_output, (dest->num_output * dest->num_data));
    if(data_output == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_data(dest);
        return NULL;
    }
    fann_memcpy(data_output, data->output[pos], (dest->num_output * dest->num_data));

    for(i = 0; i != dest->num_data; i++)
    {
        dest->input[i] = data_input;
        data_input += dest->num_input;
        dest->output[i] = data_output;
        data_output += dest->num_output;
    }
    return dest;
}

FANN_EXTERNAL unsigned int FANN_API fann_length_data(struct fann_data *data)
{
    return data->num_data;
}

FANN_EXTERNAL unsigned int FANN_API fann_num_input_data(struct fann_data *data)
{
    return data->num_input;
}

FANN_EXTERNAL unsigned int FANN_API fann_num_output_data(struct fann_data *data)
{
    return data->num_output;
}
#endif // FANN_INFERENCE_ONLY

#ifndef FANN_INFERENCE_ONLY
/* INTERNAL FUNCTION
   Save the train data structure.
 */
int fann_save_data_internal(struct fann_data *data, const char *filename)
{
    int retval = 0;
    FILE *file = fopen(filename, "w");

    if(!file)
    {
        fann_error(FANN_E_CANT_OPEN_TD_W, filename);
        return -1;
    }
    retval = fann_save_data_internal_fd(data, file);//, filename);
    fclose(file);
    
    return retval;
}
#endif // FANN_INFERENCE_ONLY

#define DATAPRINTF  "%.20e"
#define DATASCANF   "%e"
#define DATATYPE    float

#ifndef FANN_INFERENCE_ONLY
/* INTERNAL FUNCTION
   Save the train data structure.
 */
int fann_save_data_internal_fd(struct fann_data *data, FILE * file)//, const char *filename)
{
    unsigned int num_data = data->num_data;
    unsigned int num_input = data->num_input;
    unsigned int num_output = data->num_output;
    unsigned int i, j;
    int retval = 0;

    fprintf(file, "%u %u %u\n", data->num_data, data->num_input, data->num_output);
    fann_set_ff_bias();

    for(i = 0; i < num_data; i++)
    {
        for(j = 0; j < num_input; j++)
        {
            if(((int) floor(fann_ff_to_float(data->input[i][j]) + 0.5) * 1000000) ==
               ((int) floor(fann_ff_to_float(data->input[i][j]) * 1000000.0 + 0.5)))
            {
                fprintf(file, "%d ", (int) fann_ff_to_float(data->input[i][j]));
            }
            else
            {
                fprintf(file, "%.16f ", fann_ff_to_float(data->input[i][j]));
            }
        }
        fprintf(file, "\n");

        for(j = 0; j < num_output; j++)
        {
            if(((int) floor(fann_ff_to_float(data->output[i][j]) + 0.5) * 1000000) ==
               ((int) floor(fann_ff_to_float(data->output[i][j]) * 1000000.0 + 0.5)))
            {
                fprintf(file, "%d ", (int) fann_ff_to_float(data->output[i][j]));
            }
            else
            {
                fprintf(file, "%.16f ", fann_ff_to_float(data->output[i][j]));
            }
        }
        fprintf(file, "\n");
    }
    
    return retval;
}
#endif // FANN_INFERENCE_ONLY

/*
 * Creates an empty set of training data
 */
#ifndef FANN_INFERENCE_ONLY
FANN_EXTERNAL struct fann_data * FANN_API fann_create_data(unsigned int num_data, unsigned int num_input, unsigned int num_output)
{
    fann_type_ff *data_input, *data_output;
    unsigned int i;
    struct fann_data *data;

    fann_malloc(data, 1);
    if(data == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }
    
    data->num_data = num_data;
    data->num_input = num_input;
    data->num_output = num_output;

    fann_calloc(data->input, num_data);
    if(data->input == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_data(data);
        return NULL;
    }

    fann_calloc(data->output, num_data);
    if(data->output == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_data(data);
        return NULL;
    }

    fann_calloc(data_input, (num_input * num_data));
    if(data_input == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_data(data);
        return NULL;
    }

    fann_calloc(data_output, (num_output * num_data));
    if(data_output == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        fann_destroy_data(data);
        return NULL;
    }

    for(i = 0; i != num_data; i++)
    {
        data->input[i] = data_input;
        data_input += num_input;
        data->output[i] = data_output;
        data_output += num_output;
    }
    return data;
}

/*
FANN_EXTERNAL struct fann_data * FANN_API fann_create_data_pointer_array(unsigned int num_data, unsigned int num_input,
                                                                         fann_type_nt **input, unsigned int num_output,
                                                                         fann_type_nt **output)
{
    unsigned int i, j;
    struct fann_data *data;
    data = fann_create_data(num_data, num_input, num_output);

    if(data == NULL)
        return NULL;

    for (i = 0; i < num_data; ++i) {
        for (j = 0; j < num_input; j++) {
            data->input[i][j] = fann_float_to_ff(input[i][j]);
        }
        for (j = 0; j < num_output; j++) {
            data->output[i][j] = fann_float_to_ff(output[i][j]);
        }
        //fann_memcpy(data->input[i], input[i], num_input);
        //fann_memcpy(data->output[i], output[i], num_output);
    }
    
    return data;
}

FANN_EXTERNAL struct fann_data * FANN_API fann_create_data_array(unsigned int num_data, unsigned int num_input,
                                                                 fann_type_nt *input, unsigned int num_output,
                                                                 fann_type_nt *output)
{
    unsigned int i;
    struct fann_data *data;
    data = fann_create_data(num_data, num_input, num_output);

    if(data == NULL)
        return NULL;

    for (i = 0; i < num_data; ++i) {
        for (j = 0; j < num_input; j++) {
            data->input[i][j] = fann_float_to_ff(input[i][j]);
        }
        for (j = 0; j < num_output; j++) {
            data->output[i][j] = fann_float_to_ff(output[i][j]);
        }
        //fann_memcpy(data->input[i], &input[i*num_input], num_input);
        //fann_memcpy(data->output[i], &output[i*num_output], num_output);
    }
    
    return data;
}
*/

FANN_EXTERNAL fann_type_ff * FANN_API fann_get_data_input(struct fann_data * data, unsigned int position)
{
    if(position >= data->num_data)
        return NULL;
    return data->input[position];
}

FANN_EXTERNAL fann_type_ff * FANN_API fann_get_data_output(struct fann_data * data, unsigned int position)
{
    if(position >= data->num_data)
        return NULL;
    return data->output[position];
}


/*
 * INTERNAL FUNCTION Reads training data from a file descriptor. 
 */
struct fann_data *fann_read_data_from_fd(FILE * file, const char *filename)
{
    unsigned int num_input, num_output, num_data, i, j;
    unsigned int line = 1;
    struct fann_data *data;

    if(fscanf(file, "%u %u %u\n", &num_data, &num_input, &num_output) != 3)
    {
        fann_error(FANN_E_CANT_READ_TD, filename, line);
        return NULL;
    }
    line++;

    data = fann_create_data(num_data, num_input, num_output);
    if(data == NULL)
    {
        return NULL;
    }

#if (defined SWF16_AP) || (defined HWF16)
    FP_BIAS = FP_BIAS_DEFAULT;
#endif

    for(i = 0; i != num_data; i++)
    {
        DATATYPE tmpf;

        for(j = 0; j != num_input; j++)
        {
            if(fscanf(file, DATASCANF " ", &tmpf) != 1)
            {
                fann_error(FANN_E_CANT_READ_TD, filename, line);
                fann_destroy_data(data);
                return NULL;
            }
            //fprintf(stderr, "%u %u " DATAPRINTF " ", i, j, tmpf); 
            data->input[i][j] = fann_float_to_ff(tmpf);
        }
        //fprintf(stderr, "l=%u\n", line);
        line++;

        for(j = 0; j != num_output; j++)
        {
            if(fscanf(file, DATASCANF " ", &tmpf) != 1)
            {
                fann_error(FANN_E_CANT_READ_TD, filename, line);
                fann_destroy_data(data);
                return NULL;
            }
            data->output[i][j] = fann_float_to_ff(tmpf);
            //fprintf(stderr, "%u %u " DATAPRINTF " ", i, j, tmpf); 
        }
        //fprintf(stderr, "l=%u\n", line);
        line++;
    }
    return data;
}
#endif // FANN_INFERENCE_ONLY

#ifndef FANN_INFERENCE_ONLY
/*
 * INTERNAL FUNCTION returns 0 if the desired error is reached and -1 if it is not reached
 */
int fann_desired_error_reached(struct fann *ann, float desired_error)
{
    switch (ann->train_stop_function)
    {
    case FANN_STOPFUNC_LOSS:
#ifdef CALCULATE_LOSS
        if(fann_get_loss(ann) <= desired_error)
            return 0;
#endif // CALCULATE_LOSS
        break;
    case FANN_STOPFUNC_BIT:
#ifdef CALCULATE_ERROR
        if ((ann->num_bit_fail[0] + ann->num_bit_fail[1]) <= (unsigned int)desired_error)
            return 0;
#endif // CALCULATE_ERROR
        break;
    default:
        break;
    }
    return -1;
}

#endif // FANN_INFERENCE_ONLY

/*
 * Scale data in input vector before feed it to ann based on previously calculated parameters.
 */
#ifndef FANN_INFERENCE_ONLY
#ifdef FANN_DATA_SCALE
FANN_EXTERNAL void FANN_API fann_scale_input( struct fann *ann, fann_type_ff *input_vector )
{
    unsigned cur_neuron;
    if(ann->scale_mean_in == NULL)
    {
        fann_error( FANN_E_SCALE_NOT_PRESENT );
        return;
    }
    
    fann_set_ff_bias();
    for( cur_neuron = 0; cur_neuron < ann->num_input; cur_neuron++ )
        input_vector[ cur_neuron ] = fann_nt_to_ff(
            fann_nt_add(fann_nt_mul(
            (fann_nt_sub(
                fann_nt_div(( fann_nt_sub(fann_ff_to_nt(input_vector[ cur_neuron ]), ann->scale_mean_in[ cur_neuron ]) ),
                            ann->scale_deviation_in[ cur_neuron ]), 
                fann_float_to_nt(-1.0 ))) /* This is old_min */,
            ann->scale_factor_in[ cur_neuron ]), 
            ann->scale_new_min_in[ cur_neuron ])
            );
}

/*
 * Scale data in output vector before feed it to ann based on previously calculated parameters.
 */
FANN_EXTERNAL void FANN_API fann_scale_output( struct fann *ann, fann_type_ff *output_vector )
{
    unsigned cur_neuron;
    if(ann->scale_mean_in == NULL)
    {
        fann_error( FANN_E_SCALE_NOT_PRESENT );
        return;
    }

    fann_set_ff_bias();
    for( cur_neuron = 0; cur_neuron < ann->num_output; cur_neuron++ )
        output_vector[ cur_neuron ] = fann_nt_to_ff(
            fann_nt_add(fann_nt_mul(
            fann_nt_sub(
                fann_nt_div(( fann_nt_sub(fann_ff_to_nt(output_vector[ cur_neuron ]), ann->scale_mean_out[ cur_neuron ]) ),
                ann->scale_deviation_out[ cur_neuron ]),
                ( fann_float_to_nt(-1.0) ) /* This is old_min */),
            ann->scale_factor_out[ cur_neuron ]),
            ann->scale_new_min_out[ cur_neuron ]
            ));
}

/*
 * Descale data in input vector after based on previously calculated parameters.
 */
FANN_EXTERNAL void FANN_API fann_descale_input( struct fann *ann, fann_type_ff *input_vector )
{
    unsigned cur_neuron;
    if(ann->scale_mean_in == NULL)
    {
        fann_error( FANN_E_SCALE_NOT_PRESENT );
        return;
    }

    fann_set_ff_bias();
    for( cur_neuron = 0; cur_neuron < ann->num_input; cur_neuron++ )
        input_vector[ cur_neuron ] = fann_nt_to_ff(
            fann_nt_add(fann_nt_mul(
                fann_nt_add(fann_nt_div(
                    fann_nt_sub(fann_ff_to_nt(input_vector[ cur_neuron ]),
                    ann->scale_new_min_in[ cur_neuron ]),
                ann->scale_factor_in[ cur_neuron ]),
                fann_float_to_nt(-1.0)) /* This is old_min */,
            ann->scale_deviation_in[ cur_neuron ]),
            ann->scale_mean_in[ cur_neuron ])
            );
}

/*
 * Descale data in output vector after get it from ann based on previously calculated parameters.
 */
FANN_EXTERNAL void FANN_API fann_descale_output( struct fann *ann, fann_type_ff *output_vector )
{
    unsigned cur_neuron;
    if(ann->scale_mean_in == NULL)
    {
        fann_error( FANN_E_SCALE_NOT_PRESENT );
        return;
    }

    fann_set_ff_bias();
    for( cur_neuron = 0; cur_neuron < ann->num_output; cur_neuron++ )
        output_vector[ cur_neuron ] = fann_nt_to_ff(
            fann_nt_add(fann_nt_mul(
                fann_nt_add(fann_nt_div(
                    fann_nt_sub(fann_ff_to_nt(output_vector[ cur_neuron ]),
                    ann->scale_new_min_out[ cur_neuron ]),
                ann->scale_factor_out[ cur_neuron ]),
                fann_float_to_nt(-1.0) /* This is old_min */),
            ann->scale_deviation_out[ cur_neuron ]),
            ann->scale_mean_out[ cur_neuron ])
            );
}

/*
 * Scale input and output data based on previously calculated parameters.
 */
FANN_EXTERNAL void FANN_API fann_scale_train( struct fann *ann, struct fann_data *data )
{
    unsigned cur_sample;
    if(ann->scale_mean_in == NULL)
    {
        fann_error( FANN_E_SCALE_NOT_PRESENT );
        return;
    }
    /* Check that we have good training data. */
    if(fann_check_input_output_sizes(ann, data) == -1)
        return;

    fann_set_ff_bias();
    for( cur_sample = 0; cur_sample < data->num_data; cur_sample++ )
    {
        fann_scale_input( ann, data->input[ cur_sample ] );
        fann_scale_output( ann, data->output[ cur_sample ] );
    }
}

/*
 * Scale input and output data based on previously calculated parameters.
 */
FANN_EXTERNAL void FANN_API fann_descale_train( struct fann *ann, struct fann_data *data )
{
    unsigned cur_sample;
    if(ann->scale_mean_in == NULL)
    {
        fann_error( FANN_E_SCALE_NOT_PRESENT );
        return;
    }
    /* Check that we have good training data. */
    if(fann_check_input_output_sizes(ann, data) == -1)
        return;

    fann_set_ff_bias();
    for( cur_sample = 0; cur_sample < data->num_data; cur_sample++ )
    {
        fann_descale_input( ann, data->input[ cur_sample ] );
        fann_descale_output( ann, data->output[ cur_sample ] );
    }
}

#define SCALE_RESET( what, where, default_value )                            \
    for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )    \
        ann->what##_##where[ cur_neuron ] = ( fann_float_to_nt(default_value) );

#define SCALE_SET_PARAM( where )                                                                        \
    /* Calculate mean: sum(x)/length */                                                                    \
    for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )                                \
        ann->scale_mean_##where[ cur_neuron ] = fann_float_to_nt(0.0f);                                                    \
    for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )                                \
        for( cur_sample = 0; cur_sample < data->num_data; cur_sample++ )                                \
            ann->scale_mean_##where[ cur_neuron ] = fann_nt_add(ann->scale_mean_##where[ cur_neuron ], fann_ff_to_nt(data->where##put[ cur_sample ][ cur_neuron ]));\
    for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )                                \
        ann->scale_mean_##where[ cur_neuron ] = fann_nt_div(ann->scale_mean_##where[ cur_neuron ], fann_float_to_nt((float)data->num_data));                                    \
    /* Calculate deviation: sqrt(sum((x-mean)^2)/length) */                                                \
    for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )                                \
        ann->scale_deviation_##where[ cur_neuron ] = fann_float_to_nt(0.0f);                                                 \
    for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )                                \
        for( cur_sample = 0; cur_sample < data->num_data; cur_sample++ )                                \
            ann->scale_deviation_##where[ cur_neuron ] = fann_nt_add(ann->scale_deviation_##where[ cur_neuron ],        \
                /* Another local variable in macro? Oh no! */                                            \
                fann_nt_mul(                                                                                         \
                    fann_nt_sub(fann_ff_to_nt(data->where##put[ cur_sample ][ cur_neuron ]),             \
                    ann->scale_mean_##where[ cur_neuron ])                                             \
                ,                                                                                        \
                    fann_nt_sub(fann_ff_to_nt(data->where##put[ cur_sample ][ cur_neuron ]),             \
                    ann->scale_mean_##where[ cur_neuron ])                                             \
                ));                                                                                         \
    for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )                                \
        ann->scale_deviation_##where[ cur_neuron ] =                                                    \
            fann_float_to_nt(sqrtf( fann_nt_to_float(ann->scale_deviation_##where[ cur_neuron ]) / (float)data->num_data ));             \
    /* Calculate factor: (new_max-new_min)/(old_max(1)-old_min(-1)) */                                    \
    /* Looks like we dont need whole array of factors? */                                                \
    for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )                                \
        ann->scale_factor_##where[ cur_neuron ] =                                                        \
            fann_nt_div(( fann_nt_sub(new_##where##put_max, new_##where##put_min) )                                                \
            ,                                                                                            \
            fann_float_to_nt( 1.0f - ( -1.0f ) ));                                                                        \
    /* Copy new minimum. */                                                                                \
    /* Looks like we dont need whole array of new minimums? */                                            \
    for( cur_neuron = 0; cur_neuron < ann->num_##where##put; cur_neuron++ )                                \
        ann->scale_new_min_##where[ cur_neuron ] = new_##where##put_min;

FANN_EXTERNAL int FANN_API fann_set_input_scaling_params(
    struct fann *ann,
    const struct fann_data *data,
    fann_type_nt new_input_min,
    fann_type_nt new_input_max)
{
    unsigned cur_neuron, cur_sample;

    /* Check that we have good training data. */
    /* No need for if( !params || !ann ) */
    if(data->num_input != ann->num_input
       || data->num_output != ann->num_output)
    {
        fann_error( FANN_E_TRAIN_DATA_MISMATCH );
        return -1;
    }

    if(ann->scale_mean_in == NULL)
        fann_allocate_scale(ann);
    
    if(ann->scale_mean_in == NULL)
        return -1;
        
    fann_set_ff_bias();
    if( !data->num_data )
    {
        SCALE_RESET( scale_mean,        in,    0.0 )
        SCALE_RESET( scale_deviation,    in,    1.0 )
        SCALE_RESET( scale_new_min,        in,    -1.0 )
        SCALE_RESET( scale_factor,        in,    1.0 )
    }
    else
    {
        SCALE_SET_PARAM( in );
    }

    return 0;
}

FANN_EXTERNAL int FANN_API fann_set_output_scaling_params(
    struct fann *ann,
    const struct fann_data *data,
    fann_type_nt new_output_min,
    fann_type_nt new_output_max)
{
    unsigned cur_neuron, cur_sample;

    /* Check that we have good training data. */
    /* No need for if( !params || !ann ) */
    if(data->num_input != ann->num_input
       || data->num_output != ann->num_output)
    {
        fann_error( FANN_E_TRAIN_DATA_MISMATCH );
        return -1;
    }

    if(ann->scale_mean_out == NULL)
        fann_allocate_scale(ann);
    
    if(ann->scale_mean_out == NULL)
        return -1;
        
    fann_set_ff_bias();
    if( !data->num_data )
    {
        SCALE_RESET( scale_mean,        out,    0.0 )
        SCALE_RESET( scale_deviation,    out,    1.0 )
        SCALE_RESET( scale_new_min,        out,    -1.0 )
        SCALE_RESET( scale_factor,        out,    1.0 )
    }
    else
    {
        SCALE_SET_PARAM( out );
    }

    return 0;
}

/*
 * Calculate scaling parameters for future use based on training data.
 */
FANN_EXTERNAL int FANN_API fann_set_scaling_params(
    struct fann *ann,
    const struct fann_data *data,
    fann_type_nt new_input_min,
    fann_type_nt new_input_max,
    fann_type_nt new_output_min,
    fann_type_nt new_output_max)
{
    if(fann_set_input_scaling_params(ann, data, new_input_min, new_input_max) == 0)
        return fann_set_output_scaling_params(ann, data, new_output_min, new_output_max);
    else
        return -1;
}

/*
 * Clears scaling parameters.
 */
FANN_EXTERNAL int FANN_API fann_clear_scaling_params(struct fann *ann)
{
    unsigned cur_neuron;

    if(ann->scale_mean_out == NULL)
        fann_allocate_scale(ann);
    
    if(ann->scale_mean_out == NULL)
        return -1;
    
    fann_set_ff_bias();
    SCALE_RESET( scale_mean,        in,    0.0 )
    SCALE_RESET( scale_deviation,    in,    1.0 )
    SCALE_RESET( scale_new_min,        in,    -1.0 )
    SCALE_RESET( scale_factor,        in,    1.0 )

    SCALE_RESET( scale_mean,        out,    0.0 )
    SCALE_RESET( scale_deviation,    out,    1.0 )
    SCALE_RESET( scale_new_min,        out,    -1.0 )
    SCALE_RESET( scale_factor,        out,    1.0 )
    
    return 0;
}
#endif // FANN_DATA_SCALE

int fann_check_input_output_sizes(struct fann *ann, struct fann_data *data)
{
    if(ann->num_input != data->num_input)
    {
        fann_error(FANN_E_INPUT_NO_MATCH,
            ann->num_input, data->num_input);
        return -1;
    }
        
    if(ann->num_output != data->num_output)
    {
        fann_error(FANN_E_OUTPUT_NO_MATCH,
                    ann->num_output, data->num_output);
        return -1;
    }
    
    return 0;
}
#endif // FANN_INFERENCE_ONLY


