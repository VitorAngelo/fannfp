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

#ifndef __floatfann_h__
#define __floatfann_h__

#ifndef FANN_INFERENCE_ONLY
typedef float fann_type_nt;
#endif

//#define _GCC_ARM_F16_BP
//#define _GCC_ARM_F16_FF
#if (defined _GCC_ARM_F16_FF) || (defined _GCC_ARM_F16_BP) || (defined _FLOAT_UNION) || (defined _BFLOAT16)
#include <stdint.h>
#endif

#ifndef FANN_INFERENCE_ONLY
#if (defined _GCC_ARM_F16_FF) || (defined _FLOAT_UNION)
typedef union { float f; uint32_t u; } fann_type_bp;
#elif defined _GCC_ARM_F16_BP
typedef union { __fp16 f; uint16_t u; } fann_type_bp;
#elif (defined _BFLOAT16)
typedef union { int16_t i; uint16_t u; } fann_type_bp;
#else // native FP32
typedef float fann_type_bp;
#endif
#endif // FANN_INFERENCE_ONLY

#if (defined _GCC_ARM_F16_FF) || (defined _GCC_ARM_F16_BP)
typedef union { __fp16 f; uint16_t u; } fann_type_ff;
#elif (defined _FLOAT_UNION)
typedef union { float f; uint32_t u; } fann_type_ff;
#elif (defined _BFLOAT16)
typedef union { int16_t i; uint16_t u; } fann_type_ff;
#else // both native FP32
typedef float fann_type_ff;
#endif

#undef FLOATFANN
#define FLOATFANN

#define FANN_INCLUDE
#include "fann.h"

#endif // __floatfann_h__

