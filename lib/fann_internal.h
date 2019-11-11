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


#ifndef __fann_internal_h__
#define __fann_internal_h__
/* internal include file, not to be included directly
 */

#ifndef FANN_INFERENCE_ONLY
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#endif // FANN_INFERENCE_ONLY

#include "fann_data.h"

#ifndef FANN_INFERENCE_ONLY
struct fann_data;
#endif // FANN_INFERENCE_ONLY

struct fann *fann_allocate_structure(unsigned int num_layers);
int fann_allocate_neurons(struct fann *ann, struct fann *orig);

#ifndef FANN_INFERENCE_ONLY
int fann_save_internal(struct fann *ann, const char *configuration_file);
int fann_save_internal_fd(struct fann *ann, FILE * conf);//, const char *configuration_file);
int fann_save_data_internal(struct fann_data *data, const char *filename);
int fann_save_data_internal_fd(struct fann_data *data, FILE * file);//, const char *filename);
#endif // FANN_INFERENCE_ONLY

#ifdef FANN_INFERENCE_ONLY
#define fann_error(...)
#else
void fann_error(const enum fann_errno_enum errno_f, ...);
void FANN_API fann_enable_batch_stats(void);
void fann_reset_batch_stats(struct fann_layer *layer_it);
void FANN_API fann_batch_stats(struct fann *ann);
void fann_print_structure(struct fann *ann, const char *file, const char* function, const int line);
#endif // FANN_INFERENCE_ONLY

#ifndef FANN_INFERENCE_ONLY
struct fann *fann_create_from_fd(FILE * conf, const char *configuration_file);
struct fann_data *fann_read_data_from_fd(FILE * file, const char *filename);
int fann_check_input_output_sizes(struct fann *ann, struct fann_data *data);
#endif // FANN_INFERENCE_ONLY

void fann_run_layer(struct fann_layer *layer_it, struct fann_layer *prev_layer);

#ifndef FANN_INFERENCE_ONLY
int fann_compute_loss(struct fann *ann, fann_type_ff * desired_output);
void fann_update_output_weights(struct fann *ann);
void fann_backpropagate_loss(struct fann *ann);
void fann_update_weights_incremental(struct fann *ann);
void fann_update_slopes_batch(struct fann *ann);
void fann_update_weights_rmsprop(struct fann *ann,// unsigned int num_data,
        struct fann_layer *layer_begin, struct fann_layer *layer_end);
//void fann_update_weights_quickprop(struct fann *ann, unsigned int num_data,
//        struct fann_layer *layer_begin, struct fann_layer *layer_end);
void fann_update_weights_batch(struct fann *ann,// unsigned int num_data,
        struct fann_layer *layer_begin, struct fann_layer *layer_end);
void fann_update_weights_irpropm(struct fann *ann);
//void fann_update_weights_sarprop(struct fann *ann, unsigned int epoch, unsigned int first_weight,
//                                unsigned int past_end);

void fann_clear_train_arrays(struct fann *ann);
void fann_clear_batch_deltas(struct fann *ann,
                            struct fann_layer *layer_begin, struct fann_layer *layer_end);

int fann_desired_error_reached(struct fann *ann, float desired_error);
#endif // FANN_INFERENCE_ONLY

#ifdef FANN_DATA_SCALE
int fann_allocate_scale(struct fann *ann);

struct fann_ff_limits {
    fann_type_ff min;
    fann_type_ff max;
};

FANN_EXTERNAL void fann_scale_data_linear(fann_type_ff ** data, unsigned int num_data, unsigned int num_elem,
                     struct fann_ff_limits ** old_l, struct fann_ff_limits new_l);
FANN_EXTERNAL void FANN_API fann_scale_data_to_range(fann_type_ff ** data, unsigned int num_data, unsigned int num_elem,
                     struct fann_ff_limits * old_l, struct fann_ff_limits new_l);
#endif // FANN_DATA_SCALE

#ifdef SOFTFANN

#include "fann_internal_softfann.h"

#elif defined FIXEDFANN

#include "fann_internal_fixed.h"

#elif (defined FLOATFANN) && ((defined _GCC_ARM_F16_FF) || (defined _GCC_ARM_F16_BP) || (defined _FLOAT_UNION))

#include "fann_internal_union.h"

#elif (defined FLOATFANN) && (defined _BFLOAT16)

#include "fann_internal_bfloat16.h"

#else // FLOATFANN (native 32) or DOUBLEFANN

#include "fann_internal_float.h"

#endif // *FANN

#ifndef FANN_INFERENCE_ONLY
//#define fann_rand_bool() (rand() > (RAND_MAX/2))
#define fann_float_rand(min_value, max_value) (((float)(min_value))+(((float)(max_value)-((float)(min_value)))*rand()/(RAND_MAX+1.0f)))
#define fann_ff_random_weights(min,max) fann_float_to_ff(fann_float_rand(fann_nt_to_float(min),fann_nt_to_float(max)))
#endif // ! FANN_INFERENCE_ONLY

// FIXME: check uses
#define fann_abs(value) (((value) > 0) ? (value) : -(value))
#define fann_clip(x, lo, hi) (((x) < (lo)) ? (lo) : (((x) > (hi)) ? (hi) : (x)))

#if (defined SWF16_AP) || (defined HWF16)
#define fann_set_bp_bias(b)  {fann_ap_overflow = 0; FP_BIAS = b;}
#define fann_set_ff_bias()   {fann_ap_overflow = 0; FP_BIAS = FP_BIAS_DEFAULT;}
#else
#define fann_set_bp_bias(b)
#define fann_set_ff_bias()
#endif

#endif // __fann_internal_h__

