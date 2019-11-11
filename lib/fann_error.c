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
#include <limits.h>

#include "fann.h"

/* INTERNAL FUNCTION
   Populate the error information
 */
void fann_error(const enum fann_errno_enum errno_f, ...)
{
    va_list ap;
    int i, i2, i3;
    char * s, *s2;

    va_start(ap, errno_f);
    switch (errno_f)
    {
    case FANN_E_NO_ERROR:
        return;
    case FANN_E_CANT_OPEN_CONFIG_R:
        s = va_arg(ap, char *);
        fprintf(stderr, "Unable to open configuration file \"%s\" for reading.\n", s);
        break;
    case FANN_E_CANT_OPEN_CONFIG_W:
        s = va_arg(ap, char *);
        fprintf(stderr, "Unable to open configuration file \"%s\" for writing.\n", s);
        break;
    case FANN_E_WRONG_CONFIG_VERSION:
        s = va_arg(ap, char *);
        fprintf(stderr, "Wrong configuration file version, aborting read \"%s\".\n", s);
        break;
    case FANN_E_CANT_READ_CONFIG:
        s = va_arg(ap, char *);
        s2 = va_arg(ap, char *);
        fprintf(stderr, "Error reading \"%s\" from configuration file \"%s\".\n", s, s2);
        break;
    case FANN_E_CANT_READ_NEURON:
        s = va_arg(ap, char *);
        fprintf(stderr, "Error reading neuron info from configuration file \"%s\".\n", s);
        break;
    case FANN_E_CANT_READ_CONNECTIONS:
        s = va_arg(ap, char *);
        fprintf(stderr, "Error reading connections from configuration file \"%s\".\n", s);
        break;
    case FANN_E_WRONG_NUM_CONNECTIONS:
        i = va_arg(ap, int);
        i2 = va_arg(ap, int);
        fprintf(stderr, "ERROR connections_so_far=%d, total_connections=%d\n", i, i2);
        break;
    case FANN_E_CANT_OPEN_TD_W:
        s = va_arg(ap, char *);
        fprintf(stderr, "Unable to open train data file \"%s\" for writing.\n", s);
        break;
    case FANN_E_CANT_OPEN_TD_R:
        s = va_arg(ap, char *);
        fprintf(stderr, "Unable to open train data file \"%s\" for writing.\n", s);
        break;
    case FANN_E_CANT_READ_TD:
        s = va_arg(ap, char *);
        i = va_arg(ap, int);
        fprintf(stderr, "Error reading info from train data file \"%s\", line: %d.\n", s, i);
        break;
    case FANN_E_CANT_ALLOCATE_MEM:
        fprintf(stderr, "Unable to allocate memory.\n");
        break;
    case FANN_E_CANT_TRAIN_ACTIVATION:
        fprintf(stderr, "Unable to train with the selected activation function.\n");
        break;
    case FANN_E_CANT_USE_ACTIVATION:
        fprintf(stderr, "Unable to use the selected activation function.\n");
        break;
    case FANN_E_TRAIN_DATA_MISMATCH:
        fprintf(stderr, "Training data must be of equivalent structure.\n");
        break;
    case FANN_E_CANT_USE_TRAIN_ALG:
        fprintf(stderr, "Unable to use the selected training algorithm.\n");
        break;
    case FANN_E_TRAIN_DATA_SUBSET:
        i = va_arg(ap, int);
        i2 = va_arg(ap, int);
        i3 = va_arg(ap, int);
        fprintf(stderr, "Subset from %d of length %d not valid in training set of length %d.\n", i, i2, i3);
        break;
    case FANN_E_INDEX_OUT_OF_BOUND:
        i = va_arg(ap, int);
        fprintf(stderr, "Index %d is out of bound.\n", i);
        break;
    case FANN_E_SCALE_NOT_PRESENT: 
        fprintf(stderr, "Scaling parameters not present.\n");
        break;
    case FANN_E_INPUT_NO_MATCH:
        i = va_arg(ap, int);
        i2 = va_arg(ap, int);
        fprintf(stderr, "The number of input neurons in the ann (%d) and data (%d) don't match\n", i, i2);
        break;
    case FANN_E_OUTPUT_NO_MATCH:
        i = va_arg(ap, int);
        i2 = va_arg(ap, int);
        fprintf(stderr, "The number of output neurons in the ann (%d) and data (%d) don't match\n", i, i2);
         break; 
    case FANN_E_WRONG_PARAMETERS_FOR_CREATE: 
        fprintf(stderr, "The parameters for create_standard are wrong, either too few parameters provided or a negative/very high value provided.\n");
        break;
    }
    va_end(ap);
}
#endif // FANN_INFERENCE_ONLY


