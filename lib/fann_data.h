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

#ifndef __fann_data_h__
#define __fann_data_h__

//#include <stdio.h>

/* Section: FANN Datatypes

   The two main datatypes used in the fann library are <struct fann>, 
   which represents an artificial neural network, and <struct fann_data>,
   which represents training data.
 */

/* Enum: fann_train_enum
    The Training algorithms used when training on <struct fann_data> with functions like
    <fann_train_on_data> or <fann_train_on_file>. The incremental training alters the weights
    after each time it is presented an input pattern, while batch only alters the weights once after
    it has been presented to all the patterns.

    FANN_TRAIN_INCREMENTAL -  Standard backpropagation algorithm, where the weights are 
        updated after each training pattern. This means that the weights are updated many 
        times during a single epoch. For this reason some problems will train very fast with 
        this algorithm, while other more advanced problems will not train very well.
    FANN_TRAIN_BATCH -  Standard backpropagation algorithm, where the weights are updated after 
        calculating the mean square error for the whole training set. This means that the weights 
        are only updated once during an epoch. For this reason some problems will train slower with 
        this algorithm. But since the mean square error is calculated more correctly than in 
        incremental training, some problems will reach better solutions with this algorithm.
    FANN_TRAIN_RPROP - A more advanced batch training algorithm which achieves good results 
        for many problems. The RPROP training algorithm is adaptive, and does therefore not 
        use the learning_rate. Some other parameters can however be set to change the way the 
        RPROP algorithm works, but it is only recommended for users with insight in how the RPROP 
        training algorithm works. The RPROP training algorithm is described by 
        [Riedmiller and Braun, 1993], but the actual learning algorithm used here is the 
        iRPROP- training algorithm which is described by [Igel and Husken, 2000] which 
        is a variant of the standard RPROP training algorithm.
    FANN_TRAIN_RMSPROP - G. Hinton proposal for mini-batch RPROP like training
    FANN_TRAIN_QUICKPROP - A more advanced batch training algorithm which achieves good results 
        for many problems. The quickprop training algorithm uses the learning_rate parameter 
        along with other more advanced parameters, but it is only recommended to change these 
        advanced parameters, for users with insight in how the quickprop training algorithm works.
        The quickprop training algorithm is described by [Fahlman, 1988].
    FANN_TRAIN_SARPROP - THE SARPROP ALGORITHM: A SIMULATED ANNEALING ENHANCEMENT TO RESILIENT BACK PROPAGATION
    http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.47.8197&rep=rep1&type=pdf
    
    See also:
        <fann_set_training_algorithm>, <fann_get_training_algorithm>
*/
#ifndef FANN_INFERENCE_ONLY
enum fann_train_enum
{
    FANN_TRAIN_INCREMENTAL = 0,
    FANN_TRAIN_BATCH,
    FANN_TRAIN_RPROP,
    FANN_TRAIN_RMSPROP,
    //FANN_TRAIN_QUICKPROP,
    //FANN_TRAIN_SARPROP
};
#define FANN_TRAIN_LAST FANN_TRAIN_RMSPROP

/* Constant: FANN_TRAIN_NAMES
   
   Constant array consisting of the names for the training algorithms, so that the name of an
   training function can be received by:
   (code)
   char *name = FANN_TRAIN_NAMES[train_function];
   (end)

   See Also:
      <fann_train_enum>
*/
static char const *const FANN_TRAIN_NAMES[] = {
    "FANN_TRAIN_INCREMENTAL",
    "FANN_TRAIN_BATCH",
    "FANN_TRAIN_RPROP",
    "FANN_TRAIN_RMSPROP",
    //"FANN_TRAIN_QUICKPROP",
    //"FANN_TRAIN_SARPROP"
};
#endif // FANN_INFERENCE_ONLY

/* Enums: fann_activationfunc_enum
   
    The activation functions used for the neurons during training. The activation functions
    can either be defined for a group of neurons by <fann_set_activation_function_hidden> and
    <fann_set_activation_function_output> or it can be defined for a single neuron by <fann_set_activation_function>.

    The steepness of an activation function is defined in the same way by 
    <fann_set_activation_steepness_hidden>, <fann_set_activation_steepness_output> and <fann_set_activation_steepness>.
   
   The functions are described with functions where:
   * x is the input to the activation function,
   * y is the output,
   * s is the steepness and
   * d is the derivation.

   FANN_LINEAR - Linear activation function. 
     * span: -inf < y < inf
     * y = x*s, d = 1*s
     * Can NOT be used in fixed point.

   FANN_SIGMOID - Sigmoid activation function.
     * One of the most used activation functions.
     * span: 0 < y < 1
     * y = 1/(1 + exp(-2*s*x))
     * d = 2*s*y*(1 - y)

   FANN_SIGMOID_STEPWISE - Stepwise linear approximation to sigmoid.
     * Faster than sigmoid but a bit less precise.

   FANN_SIGMOID_SYMMETRIC - Symmetric sigmoid activation function, aka. tanh.
     * One of the most used activation functions.
     * span: -1 < y < 1
     * y = tanh(s*x) = 2/(1 + exp(-2*s*x)) - 1
     * d = s*(1-(y*y))

   FANN_SIGMOID_SYMMETRIC_STEPWISE - Stepwise linear approximation to symmetric sigmoid.
     * Faster than symmetric sigmoid but a bit less precise.

   FANN_RELU
    RECTIFIER LINEAR UNIT (ReLU)
    y = max(0, s*x)
    d = s (x > 0), 0 o.w.

    FANN_LINEAR_PIECE - Bounded linear activation function.
     * span: 0 <= y <= 1
     * y = x*s, d = 1*s
     
    FANN_LINEAR_PIECE_SYMMETRIC - Bounded linear activation function.
     * span: -1 <= y <= 1
     * y = x*s, d = 1*s
    
    SOFTPLUS
    y = ln(1 + exp(x))
    d = 1 / (1 + exp(-x))

    See also:
       <fann_set_activation_function_layer>, <fann_set_activation_function_hidden>,
       <fann_set_activation_function_output>, <fann_set_activation_steepness>,
    <fann_set_activation_function>
*/
enum fann_activationfunc_enum
{
    FANN_LINEAR = 0,
    FANN_SOFTMAX,
    FANN_SIGMOID,
    FANN_SIGMOID_SYMMETRIC,
    FANN_RELU,
    FANN_LEAKY_RELU,
    FANN_LINEAR_PIECE,
    FANN_LINEAR_PIECE_SYMMETRIC,
#ifdef STEPWISE_LUT
    FANN_SIGMOID_STEPWISE,
    FANN_SIGMOID_SYMMETRIC_STEPWISE,
#endif // STEPWISE_LUT
    FANN_ACTIV_FUNC_LIMIT,
};

/* Constant: FANN_ACTIVATIONFUNC_NAMES
   
   Constant array consisting of the names for the activation function, so that the name of an
   activation function can be received by:
   (code)
   char *name = FANN_ACTIVATIONFUNC_NAMES[activation_function];
   (end)

   See Also:
      <fann_activationfunc_enum>
*/
static char const *const FANN_ACTIVATIONFUNC_NAMES[] = {
    "FANN_LINEAR",
    "FANN_SOFTMAX",
    "FANN_SIGMOID",
    "FANN_SIGMOID_SYMMETRIC",
    "FANN_RELU",
    "FANN_LEAKY_RELU",
    "FANN_LINEAR_PIECE",
    "FANN_LINEAR_PIECE_SYMMETRIC",
#ifdef STEPWISE_LUT
    "FANN_SIGMOID_STEPWISE",
    "FANN_SIGMOID_SYMMETRIC_STEPWISE",
#endif // STEPWISE_LUT
};

#ifdef CALCULATE_LOSS 
/*
enum fann_lossfunc_enum
{
    FANN_LOSSFUNC_MSE = 0,  // Mean Squared Error
    FANN_LOSSFUNC_CROSS,    // Cross Entropy
    FANN_LOSSFUNC_MFE,      // Mean False Error
    FANN_LOSSFUNC_MSFE,     // Mean Squared False Error
};

static char const *const FANN_LOSSFUNC_NAMES[] = {
    "FANN_LOSSFUNC_MSE",
    "FANN_LOSSFUNC_CROSS",
    "FANN_LOSSFUNC_MFE"
    "FANN_LOSSFUNC_MSFE"
};
*/
#endif // CALCULATE_LOSS 

/* Enum: fann_errorfunc_enum
    Error function used during training.
    
    FANN_ERRORFUNC_LINEAR - Standard linear error function.
    FANN_ERRORFUNC_INV_TANH - Inverse Tanh error function, usually better 
        but can require a lower learning rate. This error function aggressively targets outputs that
        differ much from the desired, while not targeting outputs that only differ a little that much.
        This activation function is not recommended for cascade training and incremental training.

    See also:
        <fann_set_train_error_function>, <fann_get_train_error_function>
enum fann_errorfunc_enum
{
    FANN_ERRORFUNC_LINEAR = 0,
    FANN_ERRORFUNC_INV_TANH
};
#define TRAIN_ERRORFUNC_LAST FANN_ERRORFUNC_INV_TANH
*/

/* Constant: FANN_ERRORFUNC_NAMES
   
   Constant array consisting of the names for the training error functions, so that the name of an
   error function can be received by:
   (code)
   char *name = FANN_ERRORFUNC_NAMES[error_function];
   (end)

   See Also:
      <fann_errorfunc_enum>
static char const *const FANN_ERRORFUNC_NAMES[] = {
    "FANN_ERRORFUNC_LINEAR",
    "FANN_ERRORFUNC_INV_TANH"
};
*/

/* Enum: fann_stopfunc_enum
    Stop criteria used during training.

    FANN_STOPFUNC_LOSS - Stop criterion is Mean Square Error (loss) value.
    FANN_STOPFUNC_BIT - Stop criterion is number of bits that fail. The number of bits; means the
        number of output neurons which differ more than the bit fail limit 
        (see <fann_get_bit_fail_limit>, <fann_set_bit_fail_limit>). 
        The bits are counted in all of the training data, so this number can be higher than
        the number of training data.

    See also:
        <fann_set_train_stop_function>, <fann_get_train_stop_function>
*/
#ifndef FANN_INFERENCE_ONLY
enum fann_stopfunc_enum
{
    FANN_STOPFUNC_LOSS = 0,
    FANN_STOPFUNC_BIT,
    FANN_STOPFUNC_NONE
};
#define TRAIN_STOPFUNC_LAST FANN_STOPFUNC_NONE

/* Constant: FANN_STOPFUNC_NAMES
   
   Constant array consisting of the names for the training stop functions, so that the name of a
   stop function can be received by:
   (code)
   char *name = FANN_STOPFUNC_NAMES[stop_function];
   (end)

   See Also:
      <fann_stopfunc_enum>
*/
static char const *const FANN_STOPFUNC_NAMES[] = {
    "FANN_STOPFUNC_LOSS",
    "FANN_STOPFUNC_BIT",
    "FANN_STOPFUNC_NONE"
};
#endif // FANN_INFERENCE_ONLY

/* forward declarations for use with the callback */
struct fann;

#ifndef FANN_INFERENCE_ONLY
struct fann_data;
#endif // FANN_INFERENCE_ONLY

/* Type: fann_callback_type
   This callback function can be called during training when using <fann_train_on_data>, 
   <fann_train_on_file> or <fann_cascadetrain_on_data>.
    
    >typedef int (FANN_API * fann_callback_type) (struct fann *ann, struct fann_data *train, 
    >                                              unsigned int max_epochs, 
    >                                             unsigned int epochs_between_reports, 
    >                                             float desired_error, unsigned int epochs);
    
    The callback can be set by using <fann_set_callback> and is very useful for doing custom 
    things during training. It is recommended to use this function when implementing custom 
    training procedures, or when visualizing the training in a GUI etc. The parameters which the
    callback function takes are the parameters given to <fann_train_on_data>, plus an epochs
    parameter which tells how many epochs the training has taken so far.
    
    The callback function should return an integer, if the callback function returns -1, the training
    will terminate.
    
    Example of a callback function:
        >int FANN_API test_callback(struct fann *ann, struct fann_data *train,
        >                            unsigned int max_epochs, unsigned int epochs_between_reports, 
        >                            float desired_error, unsigned int epochs)
        >{
        >    printf("Epochs     %8d. loss: %.5f. Desired-loss: %.5f\n", epochs, fann_get_loss(ann), desired_error);
        >    return 0;
        >}
    
    See also:
        <fann_set_callback>, <fann_train_on_data>
 */ 
#ifndef FANN_INFERENCE_ONLY
FANN_EXTERNAL typedef int (FANN_API * fann_callback_type) (struct fann *ann, struct fann_data *train, 
                                                           unsigned int max_epochs, 
                                                           unsigned int epochs_between_reports, 
                                                           float desired_error);
#endif // FANN_INFERENCE_ONLY

/* ----- Data structures -----
 * No data within these structures should be altered directly by the user.
 */

#define FANN_THREADS 23
#undef  FANN_THREADS

#include <stdint.h>
#ifdef FANN_THREADS
#include <pthread.h>
#endif

struct fann_neuron
{
#ifdef FANN_THREADS
    // write access (during training)
    pthread_spinlock_t spin;
    uint_fast8_t step_done;
#endif

    /* Reference to the previous layer(s) */
    struct fann_layer * prev_layer;
    //unsigned int prev_count;
    
    /* The steepness of the activation function */
    fann_type_ff steepness; // SAVED

    /* The weight array */
    fann_type_ff * weight; // SAVED
    
#ifndef FANN_INFERENCE_ONLY
    fann_type_bp train_error; // SAVED

    /* The last delta applied to a connection weight.
     * This is used for the momentum term in the backpropagation algorithm.
     * Used only in incremental training. Not allocated if not used.     
     */
    fann_type_bp * weight_slopes;

    /* The previous step taken by the quickprop/rprop procedures.
     * Not allocated if not used.
     */
    fann_type_bp * prev_steps;
    /* The slope values used by the quickprop/rprop procedures.
     * Not allocated if not used.
     */
    fann_type_bp * prev_slopes;
    
#if (defined SWF16_AP) || (defined HWF16)
    /* The Back. Prop. FP bias */
    int_fast8_t bp_fp16_bias; // SAVED
    unsigned int bp_batch_overflows; // SAVED
    unsigned int bp_epoch_overflows; // SAVED
#endif
#endif // FANN_INFERENCE_ONLY
//#ifdef __GNUC__
//} __attribute__ ((packed));
//#else
};
//#endif

/* A single layer in the neural network. */
struct fann_layer
{
    /* The number of neurons in the layer */
    unsigned int num_neurons; // SAVED
    /* num_neurons + 1, for the BIAS (to avoid dozens of +1s) */
    unsigned int num_connections; // SAVED (except first_layer)
 
    /* Used to choose the activation function */
    enum fann_activationfunc_enum activation;
   
    /* A pointer to the first neuron in the layer */ 
    struct fann_neuron * neuron; // [num_neurons]
 
    /* The sum of the inputs multiplied with the weights */
    fann_type_ff * sum_w; // [num_neurons]

    /* The values of the activation functions applied to the sum */
    /* FIXME: NO NEED TO USE LAST POSITION (BIAS) */
    fann_type_ff * value; // [num_connections]
       
#ifndef FANN_INFERENCE_ONLY
    /* The maximum absolute dot product of weights and inputs *
    fann_type_ff min_abs_sum;
    fann_type_ff max_abs_sum;*/
 
    // random initialization limit (positive)
    fann_type_nt max_init;
    fann_type_nt var_init;

    // basic statistics updated only if fann_print_stats is called,
    // which also resets all the values. Zeros are not included in the
    // average calculation.
#ifdef FANN_PRINT_STATS
    double avg_abs_error; 
    double min_abs_error; 
    double max_abs_error; 
    unsigned int count_error;
    unsigned int zero_error;
    double avg_abs_delta;
    double min_abs_delta;
    double max_abs_delta;
    unsigned int count_delta;
    unsigned int zero_delta;
    double avg_abs_slope;
    double min_abs_slope;
    double max_abs_slope;
    unsigned int count_slope;
    unsigned int zero_slope;
    double avg_abs_step;
    double min_abs_step;
    double max_abs_step;
    unsigned int count_step;
    unsigned int zero_step;
#endif // FANN_PRINT_STATS
#endif // FANN_INFERENCE_ONLY
};

struct fann
{
#ifdef FANN_THREADS
    /* threads information */
    struct fann * ann[FANN_THREADS];
    pthread_t thread[FANN_THREADS];
    pthread_cond_t cond;
    pthread_mutex_t mutex;
    unsigned int wait_procs;
    unsigned int num_procs;
#endif // FANN_THREADS
#ifndef FANN_INFERENCE_ONLY
    fann_type_ff ** data_input;
    fann_type_ff ** data_output;
    unsigned int data_batch;

    /* the learning rate of the network */
    fann_type_ff learning_rate; // SAVED

    /* The learning momentum used for backpropagation algorithm. */
    fann_type_ff learning_momentum; // SAVED
    /* Non-zero nomentum is used as a fall-back mechanism for other methods */

#endif // FANN_INFERENCE_ONLY

    /* pointer to the first layer (input layer) in an array af all the layers,
     * including the input and outputlayers 
     */
    struct fann_layer *first_layer; // SAVED (count only)

    /* pointer to the layer past the last layer in an array af all the layers,
     * including the input and outputlayers 
     */
    struct fann_layer *last_layer; // SAVED (count only)

    /* Number of input neurons (not including bias) */
    unsigned int num_input;

    /* Number of output neurons (not including bias) */
    unsigned int num_output;

#ifndef FANN_INFERENCE_ONLY
    /* Unbalance error value adjust */
    fann_type_ff * unbal_er_adjust;

    /* Training algorithm used when calling fann_train_on_..
     */
    enum fann_train_enum training_algorithm; // SAVED

    /* if changed to non-zero, update weights more frequently in batch modes */
    unsigned int mini_batch; // SAVED
#endif // FANN_INFERENCE_ONLY

#ifdef CALCULATE_LOSS
    /* the number of data used to calculate the mean square error.
     */
    unsigned int loss_count;

    /* the total error value.
     * the real mean square error is loss_value/num_loss
     */
    double loss_value;
#endif // CALCULATE_LOSS

#ifdef CALCULATE_ERROR
    /* The number of outputs which would fail (only valid for classification problems)
     */
    unsigned int num_bit_fail[2];
    unsigned int num_bit_ok[2];
    unsigned int * num_max_ok; // for multi-class...

    /* The maximum difference between the actual output and the expected output 
     * which is accepted when counting the bit fails.
     * This difference is multiplied by two when dealing with symmetric activation functions,
     * so that symmetric and not symmetric activation functions can use the same limit.
     */
    float bit_fail_limit; // SAVED
#endif // CALCULATE_ERROR

#ifndef FANN_INFERENCE_ONLY
    //enum fann_lossfunc_enum train_loss_function;

    /* The error function used during training. (default FANN_ERRORFUNC_TANH)
    enum fann_errorfunc_enum train_error_function;
     */
    
    /* The stop function used during training. (default FANN_STOPFUNC_loss)
    */
    enum fann_stopfunc_enum train_stop_function; // SAVED

    /* The callback function used during training. (default NULL)
    */
    fann_callback_type callback;

    /* A pointer to user defined data. (default NULL)
    void *user_data;
    */

    /* Variable for use with RMSProp training */
    
    /* Running average memory and change factor (1 - rmsprop_avg) */
    fann_type_ff rmsprop_avg; // SAVED
    fann_type_ff rmsprop_1mavg;
    
    /* Variables for use with Quickprop training */

    /* Decay is used to make the weights not go so high *
    fann_type_ff quickprop_decay;*/

    /* Mu is a factor used to increase and decrease the stepsize *
    fann_type_ff quickprop_mu;*/

    /* Variables for use with with RPROP training */

    /* Tells how much the stepsize should increase during learning */
    fann_type_ff rprop_increase_factor; // SAVED

    /* Tells how much the stepsize should decrease during learning */
    fann_type_ff rprop_decrease_factor; // SAVED

    /* The minimum stepsize */
    fann_type_ff rprop_delta_min; // SAVED

    /* The maximum stepsize */
    fann_type_ff rprop_delta_max; // SAVED

    /* The initial stepsize */
    fann_type_ff rprop_delta_zero; // SAVED
        
    /* Defines how much the weights are constrained to smaller values at the beginning */
    //fann_type_ff sarprop_weight_decay_shift;

    /* Decides if the stepsize is too big with regard to the error */
    //fann_type_ff sarprop_step_error_threshold_factor;

    /* Defines how much the stepsize is influenced by the error */
    //fann_type_ff sarprop_step_error_shift;

    /* Defines how much the epoch influences weight decay and noise */
    //fann_type_ff sarprop_temperature;

    /* Current training epoch */
    unsigned int train_epoch;
#endif // FANN_INFERENCE_ONLY

#ifdef FANN_DATA_SCALE
    /* Arithmetic mean used to remove steady component in input data.  */
    fann_type_nt *scale_mean_in; // SAVED

    /* Standard deviation used to normalize input data (mostly to [-1;1]). */
    fann_type_nt *scale_deviation_in; // SAVED

    /* User-defined new minimum for input data.
     * Resulting data values may be less than user-defined minimum. 
     */
    fann_type_nt *scale_new_min_in; // SAVED

    /* Used to scale data to user-defined new maximum for input data.
     * Resulting data values may be greater than user-defined maximum. 
     */
    fann_type_nt *scale_factor_in; // SAVED
    
    /* Arithmetic mean used to remove steady component in output data.  */
    fann_type_nt *scale_mean_out; // SAVED

    /* Standard deviation used to normalize output data (mostly to [-1;1]). */
    fann_type_nt *scale_deviation_out; // SAVED

    /* User-defined new minimum for output data.
     * Resulting data values may be less than user-defined minimum. 
     */
    fann_type_nt *scale_new_min_out; // SAVED

    /* Used to scale data to user-defined new maximum for output data.
     * Resulting data values may be greater than user-defined maximum. 
     */
    fann_type_nt *scale_factor_out; // SAVED
#endif // FANN_DATA_SCALE

#ifndef FANN_INFERENCE_ONLY
#ifdef FANN_PRINT_STATS
    /* vectors allocated only if fann_print_stats() is called */
    float *stats_weigs;
    float *stats_errors;
    float *stats_deltas; /* batch_deltas or incre_deltas */
    float *stats_steps;
    float *stats_slopes;
#endif // FANN_PRINT_STATS
#endif // FANN_INFERENCE_ONLY
#if (defined SWF16_AP) || (defined HWF16)
    int change_bias;
#endif // (defined SWF16_AP) || (defined HWF16)
};

#endif // __fann_data_h__

