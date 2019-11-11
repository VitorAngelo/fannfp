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

#include "fann_sqrt.h"

#ifdef FANN_SQRT_EMULATION

#undef USE_AP_rsqrt
//#define USE_AP_rsqrt

/*#ifdef HWF16

const char fann_rsqrt_name[] = "1.0/sqrt";
fann_type_bp fann_bp_rsqrt(fann_type_bp x)
{
    x.f = 1.0/sqrt(x.f);
    return x;
}

#else // ! HWF16 or ARMF16 */

/*
 * 6.1094760894775391e-05,1.2781250000000000e+02,1.2793754573914609e+02
 * 3.3952000000000000e+04,5.4244995117187500e-03,5.4270936881625929e-03
 max error 0.25 %
*/

#ifdef USE_AP_rsqrt
const uint16_t fp_sqrt_shift[32] = {
    0,0,0,0,0,
    0,0,0,0,0,
    0,0,0,0,0,
    0x59BB,
    0x5FBB,
    0x65BB,
    0x6BBB,
    0x71BB,
    0x77BB,
    0x7DBB,
    0x83BB,
    0x89BB,
    0x8FBB,
    0x95BB,
    0x9BBB,
    0xA1BB,
    0xA7BB,
    0xADBB,
    0xB3BB,
    0xB9BB,
};

// 0x3800 = 0.5
// 0x3E00 = 1.5
//float16_t inv_sqrt(float16_t x)
#define NEWTON_STEP
#undef NEWTON_STEP
#ifdef NEWTON_STEP
const char fann_rsqrt_name[] = "AP_rsqrt_NEWTON";
#else
const char fann_rsqrt_name[] = "AP_rsqrt_INT";
#endif
fann_type_bp fann_bp_rsqrt(fann_type_bp x)
{
    //static const uint16_t iq_shift = 0x59B8;
    //int_fast8 exp2 = expF16UI(x.u) - 1; 
    //x2.u = (x.u & 0x3FF) | (exp2 << 10);
#ifdef NEWTON_STEP
#error "FIXME: dynamic bias"
    float16_t x2;
    static const float16_t half = {.u=0x3800};
    static const float16_t threehalfs = {.u=0x3E00};
    x2 = f16_mul(x, half); // * 0.5
#endif
    x.u = fp_sqrt_shift[FP_BIAS] - (x.u >> 1);
#ifdef NEWTON_STEP
    return f16_mul(x, f16_sub(threehalfs, f16_mul(x2, f16_mul(x, x)))); // 1.5 -
#else
    return x;
#endif
}
#endif

//#endif // ! HWF16

#if (defined _GCC_ARM_F16_FF) || (defined _FLOAT_UNION) || (defined _BFLOAT16)

#define FP_BIAS_SHIFTED ((uint32_t)532676608) // 127 << 22
#define expF32UI( a ) ((int_fast16_t) ((a)>>23) & 0xFF)
#define expF16UI( a ) ((int_fast8_t) ((a)>>10) & 0x1F)

const char fann_rsqrt_name[] = "b_rsqrt_a_32";
fann_type_bp fann_bp_b_rsqrt_a(fann_type_bp b, fann_type_bp a)
{
    fann_type_bp ret;
    int_fast16_t ea, eb, er;
    
    ea = expF32UI(a.u);
    eb = expF32UI(b.u);
    er = eb - (ea >> 1);

    if (er < -62) {
        ret.u = 0; //(b.u & 0x80000000);
    } else if (er > 190) {
        ret.f = 1.7e+38;
        ret.u |= (b.u & 0x80000000);
    } else {
        ret.u = (b.u - (a.u >> 1)) + FP_BIAS_SHIFTED;
    }
    return ret;
}

#elif (defined _GCC_ARM_F16_BP) || (defined SWF16)

#ifdef _GCC_ARM_F16_BP
#define FP_BIAS 15
#define FP_BIAS_SHIFTED ((uint16_t)7680) // 15 << 9
#endif
#define expF16UI( a ) ((int_fast8_t) ((a)>>10) & 0x1F)
const char fann_rsqrt_name[] = "b_rsqrt_a_16";

fann_type_bp fann_bp_b_rsqrt_a(fann_type_bp b, fann_type_bp a)
{
#ifdef USE_AP_rsqrt
    return fann_bp_mul(b, fann_bp_rsqrt(a));
#else // ! USE_AP_rsqrt
    int_fast8_t eb, ea, er;
    fann_type_bp ret;

    if ((b.u & 0x7FFF) == 0) {
        ret.u = b.u;
    } else {
    eb = expF16UI(b.u);
    ea = expF16UI(a.u);
    er = FP_BIAS + (eb << 1) - ea;
    if (er < 1) {
        ret.u = 0;
    } else if (er > 62) {
        //fann_ap_overflow++;
        ret.u = (b.u & 0x8000) | 0x7FFF;
    } else {
        //ret.u = (uint16_t)((((uint32_t)b.u << 1) - (uint32_t)a.u) >> 1) + FP_BIAS_SHIFTED;
        ret.u = (uint16_t)( ((((uint32_t)b.u << 1) - (uint32_t)a.u) >> 1) + ((uint32_t)FP_BIAS << 9) );
    }
    }
    return ret;
#endif // ! USE_AP_rsqrt
}

#else
#error "sqrt: define SWF16, _GCC_ARM_F16_??, _FLOAT_UNION, _BFLOAT16"
#endif

#endif // FANN_SQRT_EMULATION

