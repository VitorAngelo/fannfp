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


/* Easy way to allow for build of multiple binaries */

#include "floatfann.h"

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

#ifdef _GCC_ARM_F16_FF
const char * fann_float_type = "FF_ARMFP16 BP_FLOAT";
#elif (defined _GCC_ARM_F16_BP)
const char * fann_float_type = "FF AND BP ARMFP16";
#elif (defined _FLOAT_UNION)
const char * fann_float_type = "FLOAT_UNION";
#elif (defined _BFLOAT16)
const char * fann_float_type = "BFLOAT16";
#else
const char * fann_float_type = "FLOAT";
#endif

