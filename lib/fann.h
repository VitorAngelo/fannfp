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


/* This file defines the user interface to the fann library.
   It is included from softfann.h, floatfann.h and doublefann.h and should
   NOT be included directly. If included directly it will react as if
   floatfann.h was included.
*/ 

#ifdef FANN_EMBEDDED

#define fann_exit()
#define FANN_INFERENCE_ONLY
#undef CALCULATE_ERROR
#undef CALCULATE_LOSS
#undef FANN_DATA_SCALE 

#else // ! FANN_EMBEDDED

#include <stdlib.h>
#define fann_exit() exit(1)

#undef FANN_INFERENCE_ONLY
#define FANN_DATA_SCALE 

#if 0
#undef CALCULATE_ERROR
#undef CALCULATE_LOSS
#else
#define CALCULATE_ERROR
#define CALCULATE_LOSS
#endif

#endif // ! FANN_EMBEDDED

#define FANN_PRINT_STATS

/* Section: FANN Creation/Execution
   
   The FANN library is designed to be very easy to use. 
   A feedforward ann can be created by a simple <fann_create_standard> function, while
   other ANNs can be created just as easily. The ANNs can be trained by <fann_train_on_file>
   and executed by <fann_run>.
   
   All of this can be done without much knowledge of the internals of ANNs, although the ANNs created will
   still be powerful and effective. If you have more knowledge about ANNs, and desire more control, almost
   every part of the ANNs can be parametrized to create specialized and highly optimal ANNs.
 */
/* Group: Creation, Destruction & Execution */
    
#ifndef FANN_INCLUDE
/* just to allow for inclusion of fann.h in normal stuations where only floats are needed */ 
#ifdef FANN_SOFT
#include "softfann.h"
#elif defined FANN_FLOAT
#include "floatfann.h"
#elif defined FANN_DOUBLE
#include "doublefann.h"
#elif defined FANN_FIXED
#include "fixedfann.h"
#else
#error "Choose data type (FANN_DOUBLE, FANN_FLOAT, FANN_SOFT, FANN_FIXED)"
#endif
    
#else
    
#include <sys/time.h>
       
#ifndef __fann_h__
#define __fann_h__
    
#ifdef __cplusplus
extern "C"
{
    
#ifndef __cplusplus
} /* to fool automatic indention engines */ 
#endif
#endif    /* __cplusplus */
 
#ifndef NULL
#define NULL 0
#endif    /* NULL */
 
/* ----- Macros used to define DLL external entrypoints ----- */ 
#define FANN_EXTERNAL
#define FANN_API

extern const char * fann_float_type;

#include "fann_error.h"
#include "fann_internal.h"
#include "fann_exp.h"
#include "fann_sqrt.h"
#include "fann_data.h"
#include "fann_train.h"
#include "fann_io.h"
#include "fann_mem.h"
#include "fann_activation.h"
#include "fann_const.h"

#ifndef FANN_INFERENCE_ONLY
/* Function: fann_create_standard
    
    Creates a standard fully connected backpropagation neural network.

    There will be a bias neuron in each layer (except the output layer),
    and this bias neuron will be connected to all neurons in the next layer.
    When running the network, the bias nodes always emits 1.
    
    To destroy a <struct fann> use the <fann_destroy> function.

    Parameters:
        num_layers - The total number of layers including the input and the output layer.
        ... - Integer values determining the number of neurons in each layer starting with the 
            input layer and ending with the output layer.
            
    Returns:
        A pointer to the newly created <struct fann>.
            
    Example:
        > // Creating an ANN with 2 input neurons, 1 output neuron, 
        > // and two hidden layers with 8 and 9 neurons
        > struct fann *ann = fann_create_standard(4, 2, 8, 9, 1);
        
    See also:
        <fann_create_standard_array>, <fann_create_sparse>, <fann_create_shortcut>        
        
    This function appears in FANN >= 2.0.0.
*/ 
FANN_EXTERNAL struct fann *FANN_API fann_create_standard_args(unsigned int extra_threads, ...);

/* Function: fann_create_standard_array
   Just like <fann_create_standard>, but with an array of layer sizes
   instead of individual parameters.

    Example:
        > // Creating an ANN with 2 input neurons, 1 output neuron, 
        > // and two hidden layers with 8 and 9 neurons
        > unsigned int layers[4] = {2, 8, 9, 1};
        > struct fann *ann = fann_create_standard_array(4, layers);

    See also:
        <fann_create_standard>, <fann_create_sparse>, <fann_create_shortcut>

    This function appears in FANN >= 2.0.0.
*/ 
FANN_EXTERNAL struct fann *FANN_API fann_create_standard_vector(
                                                                unsigned int extra_threads,
                                                                unsigned int num_layers,
                                                                const unsigned int *layers);

/* Function: fann_create_sparse_array
   Just like <fann_create_sparse>, but with an array of layer sizes
   instead of individual parameters.

    See <fann_create_standard_array> for a description of the parameters.

    See also:
        <fann_create_sparse>, <fann_create_standard>, <fann_create_shortcut>

    This function appears in FANN >= 2.0.0.
*/
FANN_EXTERNAL struct fann *FANN_API fann_create_sparse_vector(
                                                            unsigned int extra_threads,
                                                             unsigned int num_layers, 
                                                             const unsigned int *layers);
#endif // FANN_INFERENCE_ONLY

/* Function: fann_destroy
   Destroys the entire network, properly freeing all the associated memory.

    This function appears in FANN >= 1.0.0.
*/ 
FANN_EXTERNAL void FANN_API fann_destroy(struct fann *ann);


/* Function: fann_copy
   Creates a copy of a fann structure. 
   
   Data in the user data <fann_set_user_data> is not copied, but the user data pointer is copied.

    This function appears in FANN >= 2.2.0.
*/ 
FANN_EXTERNAL struct fann * FANN_API fann_copy(struct fann *ann);


/* Function: fann_run
    Will run input through the neural network, returning an array of outputs, the number of which being 
    equal to the number of neurons in the output layer.

    See also:
        <fann_test>

    This function appears in FANN >= 1.0.0.
*/ 
FANN_EXTERNAL fann_type_ff * FANN_API fann_run(struct fann *ann, fann_type_ff * input);

/* Function: fann_randomize_weights
    Give each connection a random weight between *min_weight* and *max_weight*
   
    From the beginning the weights are random between -0.1 and 0.1.

    See also:
        <fann_init_weights>

    This function appears in FANN >= 1.0.0.
*/ 
#ifndef FANN_INFERENCE_ONLY
FANN_EXTERNAL void FANN_API fann_randomize_weights(struct fann *ann, fann_type_nt min_weight,
                                                   fann_type_nt max_weight);
#endif // FANN_INFERENCE_ONLY

/* Function: fann_init_weights
      Initialize the weights using Widrow + Nguyen's algorithm.
    
     This function behaves similarly to fann_randomize_weights. It will use the algorithm developed 
    by Derrick Nguyen and Bernard Widrow to set the weights in such a way 
    as to speed up training. This technique is not always successful, and in some cases can be less 
    efficient than a purely random initialization.

    The algorithm requires access to the range of the input data (ie, largest and smallest input), 
    and therefore accepts a second argument, data, which is the training data that will be used to 
    train the network.

    See also:
        <fann_randomize_weights>, <fann_read_data_from_file>

    This function appears in FANN >= 1.1.0.
*/ 
#ifndef FANN_INFERENCE_ONLY
FANN_EXTERNAL void FANN_API fann_init_weights(struct fann *ann);//, struct fann_data *train_data);

/* Function: fann_print_stats
*/ 
FANN_EXTERNAL void FANN_API fann_print_stats(struct fann *ann);

/* Group: Parameters */
/* Function: fann_print_parameters

      Prints all of the parameters and options of the ANN 

    This function appears in FANN >= 1.2.0.
*/ 
FANN_EXTERNAL void FANN_API fann_print_parameters(struct fann *ann);

/* Function: fann_get_num_input

   Get the number of input neurons.

    This function appears in FANN >= 1.0.0.
*/ 
FANN_EXTERNAL unsigned int FANN_API fann_get_num_input(struct fann *ann);


/* Function: fann_get_num_output

   Get the number of output neurons.

    This function appears in FANN >= 1.0.0.
*/ 
FANN_EXTERNAL unsigned int FANN_API fann_get_num_output(struct fann *ann);


/* Function: fann_get_total_connections

   Get the total number of connections in the entire network.

    This function appears in FANN >= 1.0.0.
*/ 
FANN_EXTERNAL unsigned int FANN_API fann_get_total_connections(struct fann *ann);

/* Function: fann_get_network_type

    Get the type of neural network it was created as.

    Parameters:
        ann - A previously created neural network structure of
            type <struct fann> pointer.

    Returns:
        The neural network type from enum <fann_network_type_enum>

    See Also:
        <fann_network_type_enum>

   This function appears in FANN >= 2.1.0
*/
FANN_EXTERNAL enum fann_nettype_enum FANN_API fann_get_network_type(struct fann *ann);

/* Function: fann_get_num_layers

    Get the number of layers in the network

    Parameters:
        ann - A previously created neural network structure of
            type <struct fann> pointer.
            
    Returns:
        The number of layers in the neural network
            
    Example:
        > // Obtain the number of layers in a neural network
        > struct fann *ann = fann_create_standard(4, 2, 8, 9, 1);
        > unsigned int num_layers = fann_get_num_layers(ann);

   This function appears in FANN >= 2.1.0
*/
FANN_EXTERNAL unsigned int FANN_API fann_get_num_layers(struct fann *ann);

/*Function: fann_get_layer_array

    Get the number of neurons in each layer in the network.

    Bias is not included so the layers match the fann_create functions.

    Parameters:
        ann - A previously created neural network structure of
            type <struct fann> pointer.

    The layers array must be preallocated to at least
    sizeof(unsigned int) * fann_num_layers() long.

   This function appears in FANN >= 2.1.0
*/
FANN_EXTERNAL void FANN_API fann_get_layer_array(struct fann *ann, unsigned int *layers);

/* Function: fann_get_bias_array

    Get the number of bias in each layer in the network.

    Parameters:
        ann - A previously created neural network structure of
            type <struct fann> pointer.

    The bias array must be preallocated to at least
    sizeof(unsigned int) * fann_num_layers() long.

   This function appears in FANN >= 2.1.0
*/
FANN_EXTERNAL void FANN_API fann_get_bias_array(struct fann *ann, unsigned int *bias);

/* Function: fann_get_weights

    Get all the network weights.

    Parameters:
        ann - A previously created neural network structure of
            type <struct fann> pointer.
        weights - A fann_type_bp pointer to user data. It is the responsibility
            of the user to allocate sufficient space to store all the weights.

   This function appears in FANN >= x.y.z
FANN_EXTERNAL void FANN_API fann_get_weights(struct fann *ann, fann_type_bp *weights);
*/


/* Function: fann_set_weights

    Set network weights.

    Parameters:
        ann - A previously created neural network structure of
            type <struct fann> pointer.
        weights - A fann_type_bp pointer to user data. It is the responsibility
            of the user to make the weights array sufficient long 
            to store all the weights.

   This function appears in FANN >= x.y.z
FANN_EXTERNAL void FANN_API fann_set_weights(struct fann *ann, fann_type_bp *weights);
*/

/* Function: fann_set_user_data

    Store a pointer to user defined data. The pointer can be
    retrieved with <fann_get_user_data> for example in a
    callback. It is the user's responsibility to allocate and
    deallocate any data that the pointer might point to.

    Parameters:
        ann - A previously created neural network structure of
            type <struct fann> pointer.
        user_data - A void pointer to user defined data.

   This function appears in FANN >= 2.1.0
FANN_EXTERNAL void FANN_API fann_set_user_data(struct fann *ann, void *user_data);
*/

/* Function: fann_get_user_data

    Get a pointer to user defined data that was previously set
    with <fann_set_user_data>. It is the user's responsibility to
    allocate and deallocate any data that the pointer might point to.

    Parameters:
        ann - A previously created neural network structure of
            type <struct fann> pointer.

    Returns:
        A void pointer to user defined data.

   This function appears in FANN >= 2.1.0
FANN_EXTERNAL void * FANN_API fann_get_user_data(struct fann *ann);
*/

/* Function: fann_disable_seed_rand

   Disables the automatic random generator seeding that happens in FANN.

   Per default FANN will always seed the random generator when creating a new network,
   unless FANN_NO_SEED is defined during compilation of the library. This method can
   disable this at runtime.

   This function appears in FANN >= 2.3.0
*/
FANN_EXTERNAL void FANN_API fann_enable_seed_fixed(const unsigned int non_zero);

/* Function: fann_enable_seed_rand

   Enables the automatic random generator seeding that happens in FANN.

   Per default FANN will always seed the random generator when creating a new network,
   unless FANN_NO_SEED is defined during compilation of the library. This method can
   disable this at runtime.

   This function appears in FANN >= 2.3.0
*/
FANN_EXTERNAL void FANN_API fann_enable_seed_rand(void);

#if (defined SWF16_AP) || (defined HWF16)
FANN_EXTERNAL void FANN_API fann_initialize_bp_bias(struct fann *ann, int bias);
FANN_EXTERNAL void FANN_API fann_set_fixed_bp_bias(struct fann *ann);
FANN_EXTERNAL void FANN_API fann_set_dynamic_bp_bias(struct fann *ann);
#else
#define fann_initialize_bp_bias(ann, b)
#define fann_set_fixed_bp_bias(ann)
#define fann_set_dynamic_bp_bias(ann)
#endif

#define fann_set_mini_batch(s, a) {s->mini_batch = a;}
#define fann_set_training_algorithm(s, a) {s->training_algorithm = a;}
//#define fann_set_train_error_function(s, a) {s->train_error_function = a;}
#define fann_set_callback(s, a) {s->callback = a;}

#include <stdint.h>
enum count_time_type {
    COUNT_CPU_TIME,
    COUNT_WALL_TIME,
};
void * fann_start_count(void * ref, const enum count_time_type type);
uint32_t fann_stop_count_us(void * ref);
uint32_t fann_stop_count_ns(void * ref);
void fann_print_count(uint32_t * us, const uint32_t len);
#endif // FANN_INFERENCE_ONLY

#ifdef __cplusplus
#ifndef __cplusplus
/* to fool automatic indention engines */ 
{
    
#endif
} 
#endif    /* __cplusplus */
    
#endif    /* __fann_h__ */
    
#endif /* NOT FANN_INCLUDE */
