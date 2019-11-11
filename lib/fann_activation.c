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

#include "fann_activation.h"
#include "fann_const.h"

/* Implementation of the activation functions */

/*
FANN_ISRU f(x) = x / sqrt(1+x^2)   f'(x) = (1 / sqrt(1+x^2))^3
FANN_ISRLU f(x) = ISRU for x < 0, x otherwise
*/

/* stepwise linear functions used for some of the activation functions */

/* defines used for the stepwise linear functions */

#define fann_linear_func(v1, r1, v2, r2, sum) (((((r2)-(r1)) * ((sum)-(v1)))/((v2)-(v1))) + (r1))

#define fann_fixed_stepwise(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6, min, max, sum) (sum < v5 ? (sum < v3 ? (sum < v2 ? (sum < v1 ? min : fann_linear_func(v1, r1, v2, r2, sum)) : fann_linear_func(v2, r2, v3, r3, sum)) : (sum < v4 ? fann_linear_func(v3, r3, v4, r4, sum) : fann_linear_func(v4, r4, v5, r5, sum))) : (sum < v6 ? fann_linear_func(v5, r5, v6, r6, sum) : max))

#ifdef STEPWISE_LUT
#define fann_neuron_linear_func(v1, r1, v2, r2, sum) (fann_ff_add(fann_ff_div(fann_ff_mul(fann_ff_sub(r2,r1),fann_ff_sub(sum,v1)),fann_ff_sub(v2, v1)), r1)) 
#define fann_neuron_stepwise(v1, v2, v3, v4, v5, v6, r1, r2, r3, r4, r5, r6, min, max, sum) (fann_ff_lt(sum, v5) ? (fann_ff_lt(sum, v3) ? (fann_ff_lt(sum, v2) ? (fann_ff_lt(sum, v1) ? min : fann_neuron_linear_func(v1, r1, v2, r2, sum)) : fann_neuron_linear_func(v2, r2, v3, r3, sum)) : (fann_ff_lt(sum, v4) ? fann_neuron_linear_func(v3, r3, v4, r4, sum) : fann_neuron_linear_func(v4, r4, v5, r5, sum))) : (fann_ff_lt(sum, v6) ? fann_neuron_linear_func(v5, r5, v6, r6, sum) : max))
#endif // STEPWISE_LUT

/* FANN_LINEAR */
/* #define fann_linear(steepness, sum) fann_mult(steepness, sum) */
#define fann_linear_derive(steepness, value) (steepness)

/* FANN_SIGMOID */
/* #define fann_sigmoid(steepness, sum) (1.0f/(1.0f + fann_exp(-2.0f * steepness * sum))) */
//#define fann_sigmoid_real(sum) (1.0f/(1.0f + fann_exp(-2.0f * sum)))
#define fann_sigmoid_real(sum) (fann_ff_div(ff_p100, fann_ff_add(ff_p100, fann_ff_exp(fann_ff_mul(ff_n200, (sum))))))
//#define fann_sigmoid_derive(steepness, value) (2.0f * steepness * value * (1.0f - value))
#define fann_sigmoid_derive(steepness, value) (fann_bp_mul(fann_ff_to_bp(ff_p200), fann_bp_mul((steepness), fann_bp_mul((value), fann_bp_sub(fann_ff_to_bp(ff_p100), (value))))))

/* FANN_SIGMOID_SYMMETRIC (TanH) */
/* #define fann_sigmoid_symmetric(steepness, sum) (2.0f/(1.0f + fann_exp(-2.0f * steepness * sum)) - 1.0f) */
//#define fann_sigmoid_symmetric_real(sum) (2.0f/(1.0f + fann_exp(-2.0f * sum)) - 1.0f)
#define fann_sigmoid_symmetric_real(sum) (fann_ff_sub(fann_ff_div(ff_p200, fann_ff_add(ff_p100, fann_ff_exp(fann_ff_mul(ff_n200, (sum))))), ff_p100))
//#define fann_sigmoid_symmetric_derive(steepness, value) steepness * (1.0f - (value*value))
#define fann_sigmoid_symmetric_derive(steepness, value) fann_bp_mul((steepness), fann_bp_sub(fann_ff_to_bp(ff_p100), fann_bp_mul((value), (value))))

/* FANN_RELU */
//#define fann_relu_derive(steepness, value) ((value > 0.0f) ? steepness : 0.0f)
#define fann_relu_derive(steepness, value) (fann_bp_is_pos((value)) ? (steepness) : fann_ff_to_bp(ff_0000))

/* FANN_LEAKY_RELU */
//#define fann_leaky_relu_derive(steepness, value) ((value > 0.0f) ? steepness : (steepness * 0.01f))
#define fann_leaky_relu_derive(steepness, value) (fann_bp_is_pos((value)) ? (steepness) : fann_bp_mul((steepness), fann_ff_to_bp(ff_p001)))

//fann_type_ff fann_activation_switch(enum fann_activationfunc_enum activation_function, fann_type_ff neuron_value)
void fann_activation_switch(struct fann_layer * layer_it, unsigned int neuron)
{
    fann_type_ff neuron_value = layer_it->sum_w[neuron];

    switch(layer_it->activation) {
    case FANN_LINEAR:
    case FANN_SOFTMAX: // this is in 2 steps: first the maximum is found, than the exp()s are calculated
        layer_it->value[neuron] = neuron_value;
        return;
    case FANN_LINEAR_PIECE:
        if (fann_ff_is_neg(neuron_value)) {
            layer_it->value[neuron] = ff_0000;
            return;
        }
        layer_it->value[neuron] = fann_ff_min(neuron_value, ff_p100);
        return;
    case FANN_LINEAR_PIECE_SYMMETRIC:
        if (fann_ff_lt(neuron_value, ff_n100)) {
            layer_it->value[neuron] = ff_n100;
            return;
        }
        layer_it->value[neuron] = fann_ff_min(neuron_value, ff_p100);
        return;
    case FANN_RELU:
        if (fann_ff_is_neg(neuron_value)) {
            layer_it->value[neuron] = ff_0000;
            return;
        }
        layer_it->value[neuron] = neuron_value;
        return;
    case FANN_LEAKY_RELU:
        if (fann_ff_is_neg(neuron_value)) {
            layer_it->value[neuron] = fann_ff_mul(neuron_value, ff_p001);
            return;
        }
        layer_it->value[neuron] = neuron_value;
        return;
    case FANN_SIGMOID:
        layer_it->value[neuron] = fann_sigmoid_real(neuron_value);
        return;
    case FANN_SIGMOID_SYMMETRIC:
        layer_it->value[neuron] = fann_sigmoid_symmetric_real(neuron_value);
        return;
#ifdef STEPWISE_LUT
    case FANN_SIGMOID_SYMMETRIC_STEPWISE:
        layer_it->value[neuron] = fann_neuron_stepwise(
                ff_v1, ff_v2, ff_v3, ff_v4, ff_v5, ff_v6,
                ff_r1_sigsym, ff_r2_sigsym, ff_r3_sigsym,
                ff_r4_sigsym, ff_r5_sigsym, ff_r6_sigsym,
                ff_n100, ff_p100, neuron_value);
        return;
    case FANN_SIGMOID_STEPWISE:
        layer_it->value[neuron] = fann_neuron_stepwise(
                ff_v1, ff_v2, ff_v3, ff_v4, ff_v5, ff_v6,
                ff_r1_sig, ff_r2_sig, ff_r3_sig,
                ff_r4_sig, ff_r5_sig, ff_r6_sig,
                ff_0000, ff_p100, neuron_value);
        return;
#endif // STEPWISE_LUT
    default:
#ifndef FANN_INFERENCE_ONLY
        printf("%s @ %s %d -> ", __FUNCTION__, __FILE__, __LINE__);
        if (layer_it->activation >= FANN_ACTIV_FUNC_LIMIT)
            printf("ACTIVATION=%d\n", layer_it->activation);
        else
            printf("%s\n", FANN_ACTIVATIONFUNC_NAMES[layer_it->activation]);
#endif
        layer_it->value[neuron] = ff_0000;
        return;
    }
}

/*
void (*fann_activation_funcptr)(void) = NULL;
static fann_type_ff fann_neuron_value;

void fann_linear_activation_ptr(void)
{
}

void fann_relu_activation_ptr(void)
{
}

void fann_leaky_relu_activation_ptr(void)
{
}

void fann_sigmoid_activation_ptr(void)
{
}

void fann_sigmoid_symmetric_activation_ptr(void)
{
}

void fann_activation_select(enum fann_activationfunc_enum activation)
{
    switch (activation) {
    case FANN_LINEAR:
    case FANN_SOFTMAX:
    case FANN_LINEAR_PIECE:
    case FANN_LINEAR_PIECE_SYMMETRIC:
        fann_activation_funcptr = fann_linear_activation_ptr;
        break;
    case FANN_RELU:
        fann_activation_funcptr = fann_relu_activation_ptr;
        break;
    case FANN_LEAKY_RELU:
        fann_activation_funcptr = fann_leaky_relu_activation_ptr;
        break;
    case FANN_SIGMOID:
        fann_activation_funcptr = fann_sigmoid_activation_ptr;
        break;
    case FANN_SIGMOID_SYMMETRIC:
        fann_activation_funcptr = fann_sigmoid_symmetric_activation_ptr;
        break;
    default:
        fann_activation_funcptr = NULL;
        break;
    }
}
*/

#ifndef FANN_INFERENCE_ONLY
struct fann_derive fann_derive_info;
/* INTERNAL FUNCTION
  Calculates the derived of a value, given an activation function and a steepness
void fann_activation_derived(struct fann_derive * dev)
static void fann_activation_dev(void)
{
    fann_type_bp dval;

    switch (fann_derive_info.activation)
    {
        case FANN_LINEAR:
        case FANN_SOFTMAX:
        case FANN_LINEAR_PIECE:
        case FANN_LINEAR_PIECE_SYMMETRIC:
            dval = fann_linear_derive(fann_derive_info.steepness, fann_derive_info.value);
            break;
        
        case FANN_RELU:
            dval = fann_relu_derive(fann_derive_info.steepness, fann_derive_info.value);
            break;
        
        case FANN_LEAKY_RELU:
            dval = fann_leaky_relu_derive(fann_derive_info.steepness, fann_derive_info.value);
            break;
        
#ifdef STEPWISE_LUT
        case FANN_SIGMOID_STEPWISE:
            fann_derive_info.value = fann_bp_clip(fann_derive_info.value, bp_p001, bp_p099);
            // get the same derivative:
#endif
        case FANN_SIGMOID:
            dval = fann_sigmoid_derive(fann_derive_info.steepness, fann_derive_info.value);
            break;
        
#ifdef STEPWISE_LUT
        case FANN_SIGMOID_SYMMETRIC_STEPWISE:
            fann_derive_info.value = fann_bp_clip(fann_derive_info.value, bp_n098, bp_p098);
            // get the same derivative:
#endif
        case FANN_SIGMOID_SYMMETRIC:
            dval = fann_sigmoid_symmetric_derive(fann_derive_info.steepness, fann_derive_info.value);
            break;
        
        default:
            printf("%s @ %s %d -> ", __FUNCTION__, __FILE__, __LINE__);
            if (fann_derive_info.activation >= FANN_ACTIV_FUNC_LIMIT)
                printf("ACTIVATION=%d\n", fann_derive_info.activation);
            else
                printf("%s\n", FANN_ACTIVATIONFUNC_NAMES[fann_derive_info.activation]);
            return;
    }
    fann_derive_info.error = fann_bp_mul(fann_derive_info.error, dval);
}
*/

void (*fann_derive_funcptr)(void) = NULL;//fann_activation_dev;

static void fann_linear_derive_ptr(void)
{
    fann_derive_info.error = fann_bp_mul(fann_derive_info.error,
                                         fann_linear_derive(fann_derive_info.steepness, fann_derive_info.value));
}

static void fann_relu_derive_ptr(void)
{
    fann_derive_info.error = fann_bp_mul(fann_derive_info.error,
                                         fann_relu_derive(fann_derive_info.steepness, fann_derive_info.value));
}

static void fann_leaky_relu_derive_ptr(void)
{
    fann_derive_info.error = fann_bp_mul(fann_derive_info.error,
                                         fann_leaky_relu_derive(fann_derive_info.steepness, fann_derive_info.value));
}

static void fann_sigmoid_derive_ptr(void)
{
    fann_derive_info.error = fann_bp_mul(fann_derive_info.error,
                                         fann_sigmoid_derive(fann_derive_info.steepness, fann_derive_info.value));
}

static void fann_sigmoid_symmetric_derive_ptr(void)
{
    fann_derive_info.error = fann_bp_mul(fann_derive_info.error,
                                         fann_sigmoid_symmetric_derive(fann_derive_info.steepness, fann_derive_info.value));
}

void fann_activation_derive_select(enum fann_activationfunc_enum activation)
{
    switch (activation) {
    case FANN_LINEAR:
    case FANN_SOFTMAX:
    case FANN_LINEAR_PIECE:
    case FANN_LINEAR_PIECE_SYMMETRIC:
        fann_derive_funcptr = fann_linear_derive_ptr;
        break;
    case FANN_RELU:
        fann_derive_funcptr = fann_relu_derive_ptr;
        break;
    case FANN_LEAKY_RELU:
        fann_derive_funcptr = fann_leaky_relu_derive_ptr;
        break;
    case FANN_SIGMOID:
        fann_derive_funcptr = fann_sigmoid_derive_ptr;
        break;
    case FANN_SIGMOID_SYMMETRIC:
        fann_derive_funcptr = fann_sigmoid_symmetric_derive_ptr;
        break;
    default:
        fann_derive_funcptr = NULL;
        break;
    }
}

#endif // FANN_INFERENCE_ONLY

