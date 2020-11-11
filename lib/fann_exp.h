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

*/


#ifndef _FANN_EXP_H
#define _FANN_EXP_H

#undef FANN_EXP_EMULATION

#ifdef FLOATFANN

#if (defined _GCC_ARM_F16_FF)

#if 1
#define FANN_EXP_EMULATION
#else
#include <math.h>
#define fann_ff_exp(x) ({fann_type_ff ret; ret.f = expf((float)((x).f)); ret;})
#define fann_exp_name "expf((float)(__fp16))"
#endif

#elif (defined _GCC_ARM_F16_BP)

#if 0
#define FANN_EXP_EMULATION
#else
#include <math.h>
#define fann_ff_exp(x) ({fann_type_ff ret; ret.f = expf((float)((x).f)); ret;})
#define fann_exp_name "expf((float)(__fp16))"
#endif

#elif (defined _FLOAT_UNION)

#include <math.h>
#define fann_ff_exp(val) ({fann_type_ff ret; ret.f = expf((val).f); ret;})
#define fann_exp_name "expf(.f)"

#elif (defined _BFLOAT16)

#include <math.h>
#define fann_ff_exp(val) ({fann_type_ff ret; float e; e = fann_ff_to_float(val); e = expf(e); ret = fann_float_to_ff(e); ret;})
#define fann_exp_name "expf(float(bf16))"

#else // native FLOAT 32 

#include <math.h>
#define fann_ff_exp(x) expf(x)
#define fann_exp_name "expf"

#endif // native FLOAT 32

#elif defined DOUBLEFANN

#include <math.h>
#define fann_ff_exp(x) exp(x)
#define fann_exp_name "exp"

#elif defined SOFTFANN

#ifdef POSIT16

#include <math.h>
#define fann_ff_exp(val) ({fann_type_ff ret; float e; e = fann_ff_to_float(val); e = expf(e); ret = fann_float_to_ff(e); ret;})
#define fann_exp_name "expf(float(bf16))"

#else // ! POSIT16
#define FANN_EXP_EMULATION
#endif

#elif defined FIXEDFANN

#include <math.h>
#define fann_ff_exp(x) float_to_cpu(expf(cpu_to_float(x)))
#define fann_exp_name "expf(cpu_to_float)"

#else

#error "COMPILATION TYPE"

#endif // *FANN

#ifdef FANN_EXP_EMULATION
extern const char fann_exp_name[];
fann_type_ff fann_ff_exp(fann_type_ff x);
#endif // FANN_EXP_EMULATION

#endif // _FANN_EXP_H

