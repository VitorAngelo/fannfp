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


#ifndef __fann_error_h__
#define __fann_error_h__

#ifndef FANN_INFERENCE_ONLY

#include <stdio.h>

/* Enum: fann_errno_enum
    Used to define error events on <struct fann> and <struct fann_data>. 

    See also:
        <fann_get_errno>, <fann_reset_errno>, <fann_get_errstr>

    FANN_E_NO_ERROR - No error 
    FANN_E_CANT_OPEN_CONFIG_R - Unable to open configuration file for reading 
    FANN_E_CANT_OPEN_CONFIG_W - Unable to open configuration file for writing
    FANN_E_WRONG_CONFIG_VERSION - Wrong version of configuration file 
    FANN_E_CANT_READ_CONFIG - Error reading info from configuration file
    FANN_E_CANT_READ_NEURON - Error reading neuron info from configuration file
    FANN_E_CANT_READ_CONNECTIONS - Error reading connections from configuration file
    FANN_E_WRONG_NUM_CONNECTIONS - Number of connections not equal to the number expected
    FANN_E_CANT_OPEN_TD_W - Unable to open train data file for writing
    FANN_E_CANT_OPEN_TD_R - Unable to open train data file for reading
    FANN_E_CANT_READ_TD - Error reading training data from file
    FANN_E_CANT_ALLOCATE_MEM - Unable to allocate memory
    FANN_E_CANT_TRAIN_ACTIVATION - Unable to train with the selected activation function
    FANN_E_CANT_USE_ACTIVATION - Unable to use the selected activation function
    FANN_E_TRAIN_DATA_MISMATCH - Irreconcilable differences between two <struct fann_data> structures
    FANN_E_CANT_USE_TRAIN_ALG - Unable to use the selected training algorithm
    FANN_E_TRAIN_DATA_SUBSET - Trying to take subset which is not within the training set
    FANN_E_INDEX_OUT_OF_BOUND - Index is out of bound
    FANN_E_SCALE_NOT_PRESENT - Scaling parameters not present
    FANN_E_INPUT_NO_MATCH - The number of input neurons in the ann and data don't match
    FANN_E_OUTPUT_NO_MATCH - The number of output neurons in the ann and data don't match
    FANN_E_WRONG_PARAMETERS_FOR_CREATE - The parameters for create_standard are wrong, either too few parameters provided or a negative/very high value provided
*/
enum fann_errno_enum
{
    FANN_E_NO_ERROR = 0,
    FANN_E_CANT_OPEN_CONFIG_R,
    FANN_E_CANT_OPEN_CONFIG_W,
    FANN_E_WRONG_CONFIG_VERSION,
    FANN_E_CANT_READ_CONFIG,
    FANN_E_CANT_READ_NEURON,
    FANN_E_CANT_READ_CONNECTIONS,
    FANN_E_WRONG_NUM_CONNECTIONS,
    FANN_E_CANT_OPEN_TD_W,
    FANN_E_CANT_OPEN_TD_R,
    FANN_E_CANT_READ_TD,
    FANN_E_CANT_ALLOCATE_MEM,
    FANN_E_CANT_TRAIN_ACTIVATION,
    FANN_E_CANT_USE_ACTIVATION,
    FANN_E_TRAIN_DATA_MISMATCH,
    FANN_E_CANT_USE_TRAIN_ALG,
    FANN_E_TRAIN_DATA_SUBSET,
    FANN_E_INDEX_OUT_OF_BOUND,
    FANN_E_SCALE_NOT_PRESENT,
    FANN_E_INPUT_NO_MATCH,
    FANN_E_OUTPUT_NO_MATCH,
    FANN_E_WRONG_PARAMETERS_FOR_CREATE
};

#endif // FANN_INFERENCE_ONLY

#endif // __fann_error_h__

