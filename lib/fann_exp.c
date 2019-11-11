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

#include "fann_exp.h"

#ifdef FANN_EXP_EMULATION

#if 0
const char fann_exp_name[] = "AP_exp_DOUBLE";
double fann_ap_exp(double x);
double fann_ap_exp(double x)
{
union
{
  double d;
  struct {
// LITTLE_ENDIAN
    int j,i;
// BIG_ENDIAN
//    int i,j;
  } n;
} _eco;

#define EXP_A (1048576/0.69314718055994530942)
#define EXP_C 60801
    _eco.n.i = EXP_A*(x) + (1072693248 - EXP_C);
#if 0
    {
        fann_type_ff a;
        a.f = _eco.d;
        if ((a.u & 0x7F800000) == 0x7F800000) {
            printf("NaN/Inf = %s -> %+le %08x\n", __FUNCTION__, x, a.u);
        }
    }
#endif
    return _eco.d;
}

#else  // 0 or 1

const char fann_exp_name[] = "AP_exp_FLOAT16";
static inline fann_type_ff fann_ap_exp(fann_type_ff x);
static inline fann_type_ff fann_ap_exp(fann_type_ff x)
{
#if (defined _GCC_ARM_F16_FF) || (defined _GCC_ARM_F16_BP)
    x.u = (int_fast32_t)((float)1477.31972187030 * (float)x.f) + 15320;
#else
    static const float16_t a16 = {.u = 0x65c5};
    int_fast32_t i;
    x = f16_mul(x, a16);
#if (defined SWF16_AP) || (defined HWF16)
    i = f16_to_i32(x);
#elif defined SWF16_IEEE
    i = f16_to_i32(x, softfloat_round_near_even, true);
#else
#error "SWF16"
#endif
    x.u = i + 15320;//(15360 - 40);
#endif
    return x;
}

#endif // 0 or 1

fann_type_ff fann_ff_exp(fann_type_ff x)
{
#if 1 // check limits
#if (defined SWF16_AP) || (defined HWF16)
    static const fann_type_ff maxx = { .u = 0x49E6 };
    static const fann_type_ff retxmax = { .u = 0x7FF8 };
    static const fann_type_ff minx = { .u = 0xC92F };
    static const fann_type_ff retxmin = { .u = 0x0008 }; // was 0x0008
#elif defined SWF16_IEEE || (defined _GCC_ARM_F16_FF) || (defined _GCC_ARM_F16_BP)
    static const fann_type_ff maxx = { .u = 0x498F };
    static const fann_type_ff retxmax = { .u = 0x7BFF }; // 0x7bf8
    static const fann_type_ff minx = { .u = 0xC92F };
    static const fann_type_ff retxmin = { .u = 0x0005 };
#else
#error "fann_exp definition"
#endif
#if 1
    // 0x0000 - 0x498E -> calc
    // 0x498F - 0x7FFF -> ret max
    // 0x8000 - 0xC8D9 -> calc
    // 0xC8DA - 0xFFFF -> ret min
    if (x.u > minx.u) {
        return retxmin;
    } else if ((x.u < 0x8000) && (x.u > maxx.u)) { // largest positive
        return retxmax;
    }
#else
    if (fann_ff_gt(x, maxx)) {
        return retxmax;
    }
    if (fann_ff_lt(x, minx)) {
        return retxmin;
    }
#endif
#endif // check limits
    return fann_ap_exp(x);
}

#endif // FANN_EXP_EMULATION

