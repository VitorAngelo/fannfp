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


#ifndef __softfann_h__
#define __softfann_h__

#ifndef FANN_INFERENCE_ONLY
void fann_reset_counters(void);
extern unsigned long int fann_mac_ops_count;
extern unsigned long int fann_add_ops_count;
extern unsigned long int fann_mult_ops_count;
extern unsigned long int fann_div_ops_count;
#endif // FANN_INFERENCE_ONLY

// One of these must be defined at compilation time:
//#define SWF32_IEEE
//#define ARMF16
//#define SWF16_IEEE
//#define SWF16_AP
//#define HWF16

#ifdef SWF16_IEEE
#include "fann_ieee_f16.h"
#define SWF16
#define EMUL_FLOAT

#elif defined SWF16_AP
#include "fann_ap_f16.h"
#define SWF16
#define EMUL_FLOAT

#elif defined HWF16
#include "fann_ap_f16.h" // only for conversions
#define SWF16
#define EMUL_FLOAT

#elif defined ARMF16
#include <stdint.h>
// use diff. types to force errors
#ifndef FANN_INFERENCE_ONLY
//typedef float32_t fann_type_bp;
typedef union { __fp16 f; uint16_t u; } fann_type_bp;
#endif // FANN_INFERENCE_ONLY
typedef union { __fp16 f; uint16_t u; } fann_type_ff;

#else

#error "Choose a specific SOFTFANN target"
/*
#define SWF32_IEEE
#define EMUL_FLOAT
#include "softfloat.h"
#ifndef FANN_INFERENCE_ONLY
typedef float32_t fann_type_bp;
#endif // FANN_INFERENCE_ONLY
typedef float32_t fann_type_ff;
*/

#endif

#ifdef SWF16 
#ifndef FANN_INFERENCE_ONLY
typedef float16_t fann_type_bp;
#endif // FANN_INFERENCE_ONLY
typedef float16_t fann_type_ff;
#endif // SWF16

#ifndef FANN_INFERENCE_ONLY
//typedef float fann_type_nt;
typedef fann_type_ff fann_type_nt;
#endif // FANN_INFERENCE_ONLY

#undef SOFTFANN
#define SOFTFANN

#define FANN_INCLUDE
#include "fann.h"

// Exported for unit test applications only:
struct float_bias_t {
    double min;
    double max;
};

extern const struct float_bias_t float_bias[32];

#define F16_BP_MAX (float_bias[FP_BIAS].max)
#define F16_BP_MIN (float_bias[FP_BIAS].min)

#define F16_FF_MAX (float_bias[FP_BIAS_DEFAULT].max)
#define F16_FF_MIN (float_bias[FP_BIAS_DEFAULT].min)

#endif
