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

/* Easy way to allow for build of multiple binaries */

#include "softfann.h"

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

#ifdef SWF32_IEEE
const char * fann_float_type = "SOFT-SWF32_IEEE";
#elif defined SWF16_IEEE
const char * fann_float_type = "SOFT-SWF16-IEEE";
#include "fann_ieee_f16.c"
#elif defined SWF16_AP
const char * fann_float_type = "SOFT-SWF16-AP";
#include "fann_ap_f16.c"
#elif defined HWF16
const char * fann_float_type = "SOFT-HWF16";
#include "fann_ap_f16.c" // not for arith. operations
#elif defined ARMF16
const char * fann_float_type = "SOFT-ARMF16";
#elif defined POSIT16
const char * fann_float_type = "SOFT-POSIT16";
#include "softposit.c"
#else
#error "Choose floating point type"
#endif

#include <math.h>

unsigned int fann_ff_fpu_limits(double *min_neg, double *max_neg, double *min_pos, double *max_pos)
{
#ifdef SWF32_IEEE
    *min_neg = *max_neg = *min_pos = *max_pos = 0.0;
#elif defined HWF16
    *min_neg = *max_neg = *min_pos = *max_pos = 0.0;
#elif defined ARMF16
    *min_neg = *max_neg = *min_pos = *max_pos = 0.0;
#elif defined SWF16
    fann_type_ff ff;
    ff.u = 0x0001;
    *min_pos = fann_ff_to_float(ff);
#ifdef SWF16_AP
    ff.u = 0x7FFF;
    *max_pos = fann_ff_to_float(ff);
    ff.u = 0xFFFF;
    *min_neg = fann_ff_to_float(ff);
#else // SWF16_IEEE
    ff.u = 0x7BFF;
    *max_pos = fann_ff_to_float(ff);
    ff.u = 0xFBFF;
    *min_neg = fann_ff_to_float(ff);
#endif
    ff.u = 0x8001;
    *max_neg = fann_ff_to_float(ff);
    return 16;
#elif defined POSIT16
    fann_type_ff ff;
    ff.v = 0x0001;
    *min_pos = fann_ff_to_float(ff);
    ff.v = 0x7FFF;
    *max_pos = fann_ff_to_float(ff);
    ff.v = 0x8001;
    *min_neg = fann_ff_to_float(ff);
    ff.v = 0xFFFF;
    *max_neg = fann_ff_to_float(ff);
    return 16;
#else
#error "Choose floating point type"
#endif
    return 0;
}

unsigned int fann_bp_fpu_limits(double *min_neg, double *max_neg, double *min_pos, double *max_pos)
{
#ifdef SWF32_IEEE
    *min_neg = *max_neg = *min_pos = *max_pos = 0.0;
#elif defined HWF16
    *min_neg = *max_neg = *min_pos = *max_pos = 0.0;
#elif defined ARMF16
    *min_neg = *max_neg = *min_pos = *max_pos = 0.0;
#elif defined SWF16
    fann_type_bp bp;
    bp.u = 0x0001;
    *min_pos = fann_bp_to_float(bp);
#ifdef SWF16_AP
    bp.u = 0x7FFF;
    *max_pos = fann_bp_to_float(bp);
    bp.u = 0xFFFF;
    *min_neg = fann_bp_to_float(bp);
#else // SWF16_IEEE
    bp.u = 0x7BFF;
    *max_pos = fann_bp_to_float(bp);
    bp.u = 0xFBFF;
    *min_neg = fann_bp_to_float(bp);
#endif
    bp.u = 0x8001;
    *max_neg = fann_bp_to_float(bp);
    return 16;
#elif defined POSIT16
    fann_type_bp bp;
    bp.v = 0x0001;
    *min_pos = fann_bp_to_float(bp);
    bp.v = 0x7FFF;
    *max_pos = fann_bp_to_float(bp);
    bp.v = 0x8001;
    *min_neg = fann_bp_to_float(bp);
    bp.v = 0xFFFF;
    *max_neg = fann_bp_to_float(bp);
    return 16;
#else
#error "Choose floating point type"
#endif
    return 0;
}
#ifndef FANN_INFERENCE_ONLY
unsigned long int fann_mac_ops_count;
unsigned long int fann_add_ops_count;
unsigned long int fann_mult_ops_count;
unsigned long int fann_div_ops_count;

void fann_reset_counters(void)
{
    fann_mac_ops_count = 0;
    fann_add_ops_count = 0;
    fann_mult_ops_count = 0;
    fann_div_ops_count = 0;
#ifdef SWF16_IEEE 
    printf("float exceptions: 0x%02X\n", (unsigned int)softfloat_exceptionFlags);
#endif
}

#define COUNT_MAC_OP()
#define COUNT_ADD_OP()
#define COUNT_MULT_OP()
#define COUNT_DIV_OP()

#else // FANN_INFERENCE_ONLY

#define COUNT_MAC_OP()
#define COUNT_ADD_OP()
#define COUNT_MULT_OP()
#define COUNT_DIV_OP()

#endif // FANN_INFERENCE_ONLY

#define DEBUG_NAN 400
#undef DEBUG_NAN

#ifdef DEBUG_NAN
static char debug_nan_args[DEBUG_NAN] = "";
#endif

fann_type_bp fann_ff_to_bp(fann_type_ff ff)
{
    // FF_exp = exp + FP_BIAS_DEFAULT (source bias)
    // BP_exp = exp + FP_BIAS (dest. bias)
    // BP_exp - FP_BIAS = FF_exp - FP_BIAS_DEFAULT
    // BP_exp = (FF_exp - FP_BIAS_DEFAULT) + FP_BIAS
#ifdef ARMF16 
    fann_type_bp ret;
    ret.u = ff.u;
    return ret;
#elif (defined SWF16_AP) || (defined HWF16)
#ifdef FANN_AP_INCLUDE_ZERO
    if ((FP_BIAS_DEFAULT != FP_BIAS) && (ff.u & 0x7FFF)) {
#endif
        fann_type_ff ret;
        int_fast8_t ff_exp = expF16UI(ff.u);
        ff_exp += FP_BIAS - FP_BIAS_DEFAULT;
        if (ff_exp < 0) {
            ret.u = 0;
        } else if (ff_exp > 31) {
            fann_ap_overflow++;
            ret.u = ff.u | 0x7FFF;
        } else {
            ret.u = ff.u & 0x83FF;
            ret.u |= (uint16_t)(ff_exp) << 10;
        }
        return ret;
#ifdef FANN_AP_INCLUDE_ZERO
    }
    return ff;
#endif
#else
    return ff;
#endif
}

fann_type_ff fann_bp_to_ff(fann_type_bp bp)
{
    // BP_exp = exp + FP_BIAS (source bias)
    // FF_exp = exp + FP_BIAS_DEFAULT (dest. bias)
    // BP_exp - FP_BIAS = FF_exp - FP_BIAS_DEFAULT
    // FF_exp = (BP_exp - FP_BIAS) + FP_BIAS_DEFAULT
#ifdef ARMF16 
    fann_type_ff ret;
    ret.u = bp.u;
    return ret;
#elif (defined SWF16_AP) || (defined HWF16)
#ifdef FANN_AP_INCLUDE_ZERO
    if ((FP_BIAS_DEFAULT != FP_BIAS) && (bp.u & 0x7FFF)) {
#endif
        fann_type_ff ret;
        int_fast8_t bp_exp = expF16UI(bp.u);
        bp_exp += FP_BIAS_DEFAULT - FP_BIAS;
        if (bp_exp < 0) {
            ret.u = 0;
        } else if (bp_exp > 31) {
            fann_ap_overflow++;
            ret.u = bp.u | 0x7FFF;
        } else {
            ret.u = bp.u & 0x83FF;
            ret.u |= (uint16_t)(bp_exp) << 10;
        }
        return ret;
#ifdef FANN_AP_INCLUDE_ZERO
    }
    return bp;
#endif
#else
    return bp;
#endif
}

#ifndef SWF16_IEEE
#ifndef POSIT16 
fann_type_bp fann_bp_to_bp(fann_type_bp src, int_fast8_t src_bias)
{
#if (defined SWF16_AP) || (defined HWF16)
    // src_exp = exp + src_bias (source bias)
    // ret_exp = exp + FP_BIAS (dest. bias)
    // src_exp - src_bias = ret_exp - FP_BIAS
    // ret_exp = (src_exp - src_bias) + FP_BIAS
#ifdef FANN_AP_INCLUDE_ZERO
    if ((src_bias != FP_BIAS) && (src.u & 0x7FFF)) {
#endif
        fann_type_bp ret;
        int_fast8_t src_exp = expF16UI(src.u);
        src_exp += FP_BIAS - src_bias;
        if (src_exp < 0) {
            ret.u = 0;
        } else if (src_exp > 31) {
            fann_ap_overflow++;
            ret.u = src.u | 0x7FFF;
        } else {
            ret.u = src.u & 0x83FF;
            ret.u |= (uint16_t)(src_exp) << 10;
        }
        return ret;
#ifdef FANN_AP_INCLUDE_ZERO
    }
    return src;
#endif
#elif defined ARMF16 
    fann_type_ff ret;
    ret.u = bp.u;
    return ret;
#else
    return src;
#endif
}
#endif // POSIT16 
#endif // SWF16_IEEE

// APPROX FUNCTIONS

fann_type_bp fann_int_to_bp(int i)//, int_fast8_t bias)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %d\n", __FUNCTION__, i);
#endif

#ifdef SWF32_IEEE 
    return i32_to_f32(i);
#elif (defined SWF16_AP) || (defined HWF16)
    //int_fast8_t save_bias = FP_BIAS;
    //fann_type_bp ret;
    //FP_BIAS = bias;
    return i32_to_f16(i);
    //FP_BIAS = save_bias;
    //return ret;
#elif defined SWF16_IEEE
    return i32_to_f16(i);
#elif defined POSIT16
    return i32_to_p16(i);
#else // ARMF16
    fann_type_bp ret;
    ret.f = (__fp16) i;
    return ret;
#endif
}

fann_type_ff fann_int_to_ff(int i)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %d\n", __FUNCTION__, i);
#endif

#ifdef SWF32_IEEE 
    return i32_to_f32(i);
#elif (defined SWF16_AP) || (defined HWF16)
#if 0
    int_fast8_t save_bias = FP_BIAS;
    fann_type_ff ret;
    FP_BIAS = FP_BIAS_DEFAULT;
    ret = i32_to_f16(i);
    FP_BIAS = save_bias;
    return ret;
#else
    return i32_to_f16(i);
#endif
#elif defined SWF16_IEEE
    return i32_to_f16(i);
#elif defined POSIT16
    return i32_to_p16(i);
#else // ARMF16
    fann_type_ff ret;
    ret.f = (__fp16) i;
    return ret;
#endif
}

fann_type_bp fann_float_to_bp(float f)//, int_fast8_t bias)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le\n", __FUNCTION__, f);
#endif

#ifdef SWF32_IEEE
    union { float32_t swf;
    float hwf; } un;
    un.hwf = (float)f;
    return un.swf;
#elif defined POSIT16
    return convertDoubleToP16(f);
#elif defined SWF16 
    float32_t swf;
    swf.f = (float)f;
#if (defined SWF16_AP) || (defined HWF16)
    //int_fast8_t save_bias = FP_BIAS;
    //fann_type_bp ret;
    //FP_BIAS = bias;
    return f32_to_f16(swf);
    //FP_BIAS = save_bias;
    //return ret;
#else // SWF16_IEEE
    return f32_to_f16(swf);
#endif
#else  // ARMF16
    fann_type_bp ret;
    ret.f = (__fp16) f;
    return ret;
#endif
}

inline fann_type_ff fann_float_to_ff(float f)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le\n", __FUNCTION__, f);
#endif

#ifdef SWF32_IEEE
    union { float32_t swf;
    float hwf; } un;
    un.hwf = (float)f;
    return un.swf;
#elif defined POSIT16
    return convertDoubleToP16(f);
#elif defined SWF16 
    float32_t swf;
    swf.f = (float)f;
#if (defined SWF16_AP) || (defined HWF16)
#if 0
    int_fast8_t save_bias = FP_BIAS;
    fann_type_ff ret;
    FP_BIAS = FP_BIAS_DEFAULT;
    ret = f32_to_f16(swf);
    FP_BIAS = save_bias;
    return ret;
#else
    return f32_to_f16(swf);
#endif
#else // SWF16_IEEE
    return f32_to_f16(swf);
#endif
#else  // ARMF16
    fann_type_ff ret;
    ret.f = (__fp16) f;
    return ret;
#endif
}

/*int fann_bp_to_int(fann_type_bp f, int_fast8_t bias)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le\n", __FUNCTION__, f.f);
#endif

#if (defined SWF16_AP) || (defined HWF16)
    int_fast8_t save_bias = FP_BIAS;
    int ret;
    FP_BIAS = bias;
    ret = f16_to_i32(f);
    FP_BIAS = save_bias;
    return ret;
#elif defined SWF32_IEEE
    return f32_to_i32(f, softfloat_round_near_even, false);
#elif defined SWF16_IEEE
    return f16_to_i32(f, softfloat_round_near_even, false);
#else  // ARMF16
    return ((int)f.f);
#endif
}*/

/*int fann_ff_to_int(fann_type_ff f)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le\n", __FUNCTION__, f.f);
#endif

#ifdef SWF32_IEEE
    return f32_to_i32(f, softfloat_round_near_even, false);
#elif defined SWF16_IEEE
    return f16_to_i32(f, softfloat_round_near_even, false);
#elif (defined SWF16_AP) || (defined HWF16)
    int_fast8_t save_bias = FP_BIAS;
    int ret;
    FP_BIAS = FP_BIAS_DEFAULT;
    ret = f16_to_i32(f);
    FP_BIAS = save_bias;
    return ret;
#else  // ARMF16
    return ((int)f.f);
#endif
}*/

inline float fann_bp_to_float(fann_type_bp f)//, int_fast8_t bias)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le\n", __FUNCTION__, f.f);
#endif

#ifdef SWF32_IEEE
    float hwf;
    memcpy(&hwf, &(f.v), 4);
    return (float)hwf;
#elif defined SWF16 
    float32_t swf;
    swf = f16_to_f32(f);
    return (float)swf.f;
#elif defined POSIT16
    return convertP16ToDouble(f);
#else  // ARMF16
    return (float)(f.f);
#endif
}

inline float fann_ff_to_float(fann_type_ff f)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le\n", __FUNCTION__, f.f);
#endif

#ifdef SWF32_IEEE
    float hwf;
    memcpy(&hwf, &(f.v), 4);
    return (float)hwf;
#elif defined SWF16 
    float32_t swf;
    swf = f16_to_f32(f);
    return (float)swf.f;
#elif defined POSIT16
    return convertP16ToDouble(f);
#else  // ARMF16
    return (float)(f.f);
#endif
}

fann_type_bp fann_bp_neg(fann_type_bp f)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le\n", __FUNCTION__, f.f);
#endif

#ifdef SWF32_IEEE
    f.u ^= 0x80000000;
#elif defined SWF16
    f.u ^= 0x8000;
#elif defined POSIT16
    if (f.v != 0x8000)
        f.v = (~f.v) + 1;
#else  // ARMF16 
    f.f = -(f.f);
#endif
    return f;
}

fann_type_ff fann_ff_neg(fann_type_ff f)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le\n", __FUNCTION__, f.f);
#endif

#ifdef SWF32_IEEE
    f.u ^= 0x80000000;
#elif defined SWF16
    f.u ^= 0x8000;
#elif defined POSIT16
    if (f.v != 0x8000)
        f.v = (~f.v) + 1;
#else  // ARMF16 
    f.f = -(f.f);
#endif
    return f;
}

fann_type_bp fann_bp_mac(fann_type_bp x, fann_type_bp y, fann_type_bp c)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le %+le %+le\n", __FUNCTION__, x.f, y.f, c.f);
#endif

    COUNT_MAC_OP();
#ifdef SWF32_IEEE
    return f32_mulAdd(x, y, c);
#elif (defined SWF16_AP) || (defined SWF16_IEEE)
    return f16_mulAdd(x, y, c);
#elif (defined HWF16)
    register float xf, yf, cf;
    xf = fann_bp_to_float(x);
    yf = fann_bp_to_float(y);
    cf = fann_bp_to_float(c);
    cf += xf * yf;
    return fann_float_to_bp(cf);
#elif defined POSIT16
    return p16_mulAdd(x, y, c);
#else  // ARMF16 
    c.f += x.f * y.f;
    return c;
#endif
}

fann_type_ff fann_ff_mac(fann_type_ff x, fann_type_ff y, fann_type_ff c)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le %+le %+le\n", __FUNCTION__, x.f, y.f, c.f);
#endif

    COUNT_MAC_OP();
#ifdef SWF32_IEEE
    return f32_mulAdd(x, y, c);
#elif (defined SWF16_AP) || (defined SWF16_IEEE)
    return f16_mulAdd(x, y, c);
#elif (defined HWF16)
    register float xf, yf, cf;
    xf = fann_ff_to_float(x);
    yf = fann_ff_to_float(y);
    cf = fann_ff_to_float(c);
    cf += xf * yf;
    return fann_float_to_ff(cf);
#elif defined POSIT16
    return p16_mulAdd(x, y, c);
#else  // ARMF16 
    c.f += x.f * y.f;
    return c;
#endif
}

fann_type_bp fann_bp_mul(fann_type_bp x, fann_type_bp y)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le %+le\n", __FUNCTION__, x.f, y.f);
#endif

    COUNT_MULT_OP();
#ifdef SWF32_IEEE
    return f32_mul(x, y);
#elif (defined SWF16_AP) || (defined SWF16_IEEE)
    return f16_mul(x, y);
#elif (defined HWF16)
    register float xf, yf;
    xf = fann_bp_to_float(x);
    yf = fann_bp_to_float(y);
    return fann_float_to_bp(xf * yf);
#elif defined POSIT16
    return p16_mul(x, y);
#else // ARMF16
    return x * y;
#endif
}

fann_type_ff fann_ff_mul(fann_type_ff x, fann_type_ff y)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le %+le\n", __FUNCTION__, x.f, y.f);
#endif

    COUNT_MULT_OP();
#ifdef SWF32_IEEE
    return f32_mul(x, y);
#elif (defined SWF16_AP) || (defined SWF16_IEEE)
    return f16_mul(x, y);
#elif (defined HWF16)
    register float xf, yf;
    xf = fann_ff_to_float(x);
    yf = fann_ff_to_float(y);
    return fann_float_to_ff(xf * yf);
#elif defined POSIT16
    return p16_mul(x, y);
#else // ARMF16
    return x * y;
#endif
}

/*
fann_type_bp fann_bp_div(fann_type_bp x, fann_type_bp y)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le %+le\n", __FUNCTION__, x.f, y.f);
#endif

    COUNT_DIV_OP();
#ifdef SWF32_IEEE
    return f32_div(x, y);
#elif defined SWF16 
    return f16_div(x, y);
#else  // HWF16
    x.f /= y.f;
#ifdef HWF16
    return fann_bp_clip_limits(x.f);
#else // ARMF16
    return x;
#endif
#endif
}
*/

fann_type_ff fann_ff_div(fann_type_ff x, fann_type_ff y)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le %+le\n", __FUNCTION__, x.f, y.f);
#endif

    COUNT_DIV_OP();
#ifdef SWF32_IEEE
    return f32_div(x, y);
#elif (defined SWF16_AP) || (defined SWF16_IEEE)
    return f16_div(x, y);
#elif (defined HWF16)
    register float xf, yf;
    xf = fann_ff_to_float(x);
    yf = fann_ff_to_float(y);
    return fann_float_to_ff(xf / yf);
#elif defined POSIT16
    return p16_div(x, y);
#else // ARMF16
    return x / y;
#endif
}

fann_type_bp fann_bp_add(fann_type_bp x, fann_type_bp y)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le %+le\n", __FUNCTION__, x.f, y.f);
#endif

    COUNT_ADD_OP();
#ifdef SWF32_IEEE
    return f32_add(x, y);
#elif (defined SWF16_AP) || (defined SWF16_IEEE)
    return f16_add(x, y);
#elif (defined HWF16)
    register float xf, yf;
    xf = fann_bp_to_float(x);
    yf = fann_bp_to_float(y);
    return fann_float_to_bp(xf + yf);
#elif defined POSIT16
    return p16_add(x, y);
#else // ARMF16
    return x + y;
#endif
}

fann_type_ff fann_ff_add(fann_type_ff x, fann_type_ff y)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le %+le\n", __FUNCTION__, x.f, y.f);
#endif

    COUNT_ADD_OP();
#ifdef SWF32_IEEE
    return f32_add(x, y);
#elif (defined SWF16_AP) || (defined SWF16_IEEE)
    return f16_add(x, y);
#elif (defined HWF16)
    register float xf, yf;
    xf = fann_ff_to_float(x);
    yf = fann_ff_to_float(y);
    return fann_float_to_ff(xf + yf);
#elif defined POSIT16
    return p16_add(x, y);
#else // ARMF16
    return x + y;
#endif
}

fann_type_bp fann_bp_sub(fann_type_bp x, fann_type_bp y)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le %+le\n", __FUNCTION__, x.f, y.f);
#endif

    COUNT_ADD_OP();
#ifdef SWF32_IEEE
    return f32_sub(x, y);
#elif (defined SWF16_AP) || (defined SWF16_IEEE)
    return f16_sub(x, y);
#elif (defined HWF16)
    register float xf, yf;
    xf = fann_bp_to_float(x);
    yf = fann_bp_to_float(y);
    return fann_float_to_bp(xf - yf);
#elif defined POSIT16
    return p16_sub(x, y);
#else // ARMF16
    return x - y;
#endif
}

fann_type_ff fann_ff_sub(fann_type_ff x, fann_type_ff y)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le %+le\n", __FUNCTION__, x.f, y.f);
#endif

    COUNT_ADD_OP();
#ifdef SWF32_IEEE
    return f32_sub(x, y);
#elif (defined SWF16_AP) || (defined SWF16_IEEE)
    return f16_sub(x, y);
#elif (defined HWF16)
    register float xf, yf;
    xf = fann_ff_to_float(x);
    yf = fann_ff_to_float(y);
    return fann_float_to_ff(xf - yf);
#elif defined POSIT16
    return p16_sub(x, y);
#else // ARMF16
    return x - y;
#endif
}

fann_type_bp fann_bp_abs(fann_type_bp x)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le\n", __FUNCTION__, x.f);
#endif

#ifdef SWF32_IEEE
    x.u &= 0x7FFFFFFF;
#elif defined SWF16 
    x.u &= 0x7FFF;
#elif defined ARMF16 
    x.u &= 0x7FFF;
#elif defined POSIT16
    if (x.v > 0x8000)
        x = fann_bp_neg(x);
#endif
    return x;
}

fann_type_ff fann_ff_abs(fann_type_ff x)
{
#ifdef DEBUG_NAN
    snprintf(debug_nan_args, DEBUG_NAN, "%s: %+le\n", __FUNCTION__, x.f);
#endif

#ifdef SWF32_IEEE
    x.u &= 0x7FFFFFFF;
#elif defined SWF16 
    x.u &= 0x7FFF;
#elif defined ARMF16 
    x.u &= 0x7FFF;
#elif defined POSIT16
    if (x.v > 0x8000)
        x = fann_bp_neg(x);
#endif
    return x;
}

#ifdef EMUL_FLOAT

int fann_bp_gt(fann_type_bp x, fann_type_bp y)
{
    COUNT_ADD_OP();
#ifdef SWF32_IEEE
    return ! f32_le(x, y);
#elif defined SWF16 
    return ! f16_le(x, y);
#elif defined POSIT16
    return ! p16_le(x, y);
#else
#error "EMUL_FLOAT"
#endif
}

int fann_ff_gt(fann_type_ff x, fann_type_ff y)
{
    COUNT_ADD_OP();
#ifdef SWF32_IEEE
    return ! f32_le(x, y);
#elif defined SWF16 
    return ! f16_le(x, y);
#elif defined POSIT16
    return ! p16_le(x, y);
#else
#error "EMUL_FLOAT"
#endif
}

int fann_bp_lt(fann_type_bp x, fann_type_bp y)
{
    COUNT_ADD_OP();
#ifdef SWF32_IEEE
    return f32_lt(x, y);
#elif defined SWF16 
    return f16_lt(x, y);
#elif defined POSIT16
    return p16_lt(x, y);
#endif
}

int fann_ff_lt(fann_type_ff x, fann_type_ff y)
{
    COUNT_ADD_OP();
#ifdef SWF32_IEEE
    return f32_lt(x, y);
#elif defined SWF16 
    return f16_lt(x, y);
#elif defined POSIT16
    return p16_lt(x, y);
#endif
}

int fann_bp_ne(fann_type_bp x, fann_type_bp y)
{
    COUNT_ADD_OP();
#ifdef SWF32_IEEE
    return ! f32_eq(x, y);
#elif defined SWF16 
    return ! f16_eq(x, y);
#elif defined POSIT16
    return ! p16_eq(x, y);
#endif
}

int fann_ff_ne(fann_type_ff x, fann_type_ff y)
{
    COUNT_ADD_OP();
#ifdef SWF32_IEEE
    return ! f32_eq(x, y);
#elif defined SWF16 
    return ! f16_eq(x, y);
#elif defined POSIT16
    return ! p16_eq(x, y);
#endif
}

fann_type_bp fann_bp_max(fann_type_bp x, fann_type_bp y)
{
    if (fann_bp_gt(x,y))
        return x;
    return y;
}

fann_type_ff fann_ff_max(fann_type_ff x, fann_type_ff y)
{
    if (fann_bp_gt(x,y)) // FIXME
        return x;
    return y;
}

fann_type_bp fann_bp_min(fann_type_bp x, fann_type_bp y)
{
    if (fann_bp_lt(x,y))
        return x;
    return y;
}

fann_type_ff fann_ff_min(fann_type_ff x, fann_type_ff y)
{
    if (fann_bp_lt(x,y)) // FIXME
        return x;
    return y;
}

fann_type_bp fann_bp_clip(fann_type_bp x, fann_type_bp lo, fann_type_bp hi)
{
    if (fann_bp_lt(x, lo))
        return lo;
    if (fann_bp_gt(x, hi))
        return hi;
    return x;
}

fann_type_ff fann_ff_clip(fann_type_ff x, fann_type_ff lo, fann_type_ff hi)
{
    if (fann_ff_lt(x, lo))
        return lo;
    if (fann_ff_gt(x, hi))
        return hi;
    return x;
}

int fann_bp_is_neg(fann_type_bp x)
{
#ifdef SWF32_IEEE
    return ((x.u & 0x80000000) == 0x80000000) && ((x.u & 0x7FFFFFFF) != 0);
#elif defined SWF16 
    return ((x.u & 0x8000) == 0x8000) && ((x.u & 0x7FFF) != 0);
#elif defined POSIT16
    return ((x.v & 0x8000) == 0x8000) && (x.v != 0x8000);
#endif
}

int fann_ff_is_neg(fann_type_ff x)
{
#ifdef SWF32_IEEE
    return ((x.u & 0x80000000) == 0x80000000) && ((x.u & 0x7FFFFFFF) != 0);
#elif defined SWF16 
    return ((x.u & 0x8000) == 0x8000) && ((x.u & 0x7FFF) != 0);
#elif defined POSIT16
    return ((x.v & 0x8000) == 0x8000) && (x.v != 0x8000);
#endif
}

int fann_bp_is_pos(fann_type_bp x)
{
#ifdef SWF32_IEEE
    return ((x.u & 0x80000000) == 0) && (x.u != 0);
#elif defined SWF16 
    return ((x.u & 0x8000) == 0) && (x.u != 0);
#elif defined POSIT16
    return ((x.v & 0x8000) == 0);
#endif
}

int fann_ff_is_pos(fann_type_ff x)
{
#ifdef SWF32_IEEE
    return ((x.u & 0x80000000) == 0) && (x.u != 0);
#elif defined SWF16 
    return ((x.u & 0x8000) == 0) && (x.u != 0);
#elif defined POSIT16
    return ((x.v & 0x8000) == 0);
#endif
}

int fann_bp_is_zero(fann_type_bp x)
{
#ifdef SWF32_IEEE
    return ((x.u & 0x7FFFFFFF) == 0);
#elif defined SWF16 
    return ((x.u & 0x7FFF) == 0);
#elif defined POSIT16
    return x.v == 0;
#endif
}

int fann_ff_is_zero(fann_type_ff x)
{
#ifdef SWF32_IEEE
    return ((x.u & 0x7FFFFFFF) == 0);
#elif defined SWF16 
    return ((x.u & 0x7FFF) == 0);
#elif defined POSIT16
    return x.v == 0;
#endif
}

#endif // EMUL_FLOAT

/*
bias = 0, min = +1.0009765625000000e+00, max = +4.2928701440000000e+09
bias = 1, min = +5.0048828125000000e-01, max = +2.1464350720000000e+09
bias = 2, min = +2.5024414062500000e-01, max = +1.0732175360000000e+09
bias = 3, min = +1.2512207031250000e-01, max = +5.3660876800000000e+08
bias = 4, min = +6.2561035156250000e-02, max = +2.6830438400000000e+08
bias = 5, min = +3.1280517578125000e-02, max = +1.3415219200000000e+08
bias = 6, min = +1.5640258789062500e-02, max = +6.7076096000000000e+07
bias = 7, min = +7.8201293945312500e-03, max = +3.3538048000000000e+07
bias = 8, min = +3.9100646972656250e-03, max = +1.6769024000000000e+07
bias = 9, min = +1.9550323486328125e-03, max = +8.3845120000000000e+06
bias = 10, min = +9.7751617431640625e-04, max = +4.1922560000000000e+06
bias = 11, min = +4.8875808715820312e-04, max = +2.0961280000000000e+06
bias = 12, min = +2.4437904357910156e-04, max = +1.0480640000000000e+06
bias = 13, min = +1.2218952178955078e-04, max = +5.2403200000000000e+05
bias = 14, min = +6.1094760894775391e-05, max = +2.6201600000000000e+05
*/

const struct float_bias_t float_bias[32] = {
    {.min=0,.max=0}, {.min=0,.max=0}, {.min=0,.max=0}, // 0 to 2
    {.min=0,.max=0}, {.min=0,.max=0}, {.min=0,.max=0}, // 3 to 5
    {.min=0,.max=0}, {.min=0,.max=0}, {.min=0,.max=0}, // 6 to 8
    {.min=0,.max=0}, {.min=0,.max=0}, {.min=0,.max=0}, // 9 to 11
    {.min=0,.max=0}, {.min=0,.max=0}, {.min=0,.max=0}, // 12 to 14
    { .min = +3.0547380447387695e-05, .max = +1.3100800000000000e+05 }, // 15 
    { .min = +1.5273690223693848e-05, .max = +6.5504000000000000e+04 },
    { .min = +7.6368451118469238e-06, .max = +3.2752000000000000e+04 },
    { .min = +3.8184225559234619e-06, .max = +1.6376000000000000e+04 },
    { .min = +1.9092112779617310e-06, .max = +8.1880000000000000e+03 },
    { .min = +9.5460563898086548e-07, .max = +4.0940000000000000e+03 }, // 20
    { .min = +4.7730281949043274e-07, .max = +2.0470000000000000e+03 },
    { .min = +2.3865140974521637e-07, .max = +1.0235000000000000e+03 },
    { .min = +1.1932570487260818e-07, .max = +5.1175000000000000e+02 },
    { .min = +5.9662852436304092e-08, .max = +2.5587500000000000e+02 },
    { .min = +2.9831426218152046e-08, .max = +1.2793750000000000e+02 }, // 25
    { .min = +1.4915713109076023e-08, .max = +6.3968750000000000e+01 },
    { .min = +7.4578565545380116e-09, .max = +3.1984375000000000e+01 },
    { .min = +3.7289282772690058e-09, .max = +1.5992187500000000e+01 },
    { .min = +1.8644641386345029e-09, .max = +7.9960937500000000e+00 },
    { .min = +9.3223206931725144e-10, .max = +3.9980468750000000e+00 }, // 30
    { .min = +4.6611603465862572e-10, .max = +1.9990234375000000e+00 } 
};

/*
#if 0 //def HWF16

static inline fann_type_bp fann_bp_clip_limits(float x)
{
    fann_type_bp a;

    a.f = (float)x;
    // float32 significand mask is 0x007FFFFF, for 23 stored bits
    // float16 has 10 stored significand bits:
    //      7   F    F    F    F    F
    // 32: 0xxx xxxx xxxx xxxx xxxx xxxx
    // 16: 0xxx xxxx xxx, which is:
    //      7   F    E    0    0    0
    a.u &= 0xFFFFE000;
    // exceptions:
    if (a.f > 0.0) {
        if (a.f > F16_BP_MAX)
            a.f = F16_BP_MAX;
        else if (a.f < F16_BP_MIN)
            a.f = 0.0;
    } else if (a.f < 0.0) {
        if (a.f < -F16_BP_MAX)
            a.f = -F16_BP_MAX;
        else if (a.f > -F16_BP_MIN)
            a.f = -0.0;
    }
    // NaN / infinity mask: 0x7F800000
#if 0 //def DEBUG_NAN
    if ((a.u & 0x7F800000) == 0x7F800000)
        fprintf(stderr, "NaN/Inf = %s -> %+le %08x\n", debug_nan_args, x, a.u);
    if ((((int_fast16_t) ((a.u)>>23) & 0xFF) == 0) &&
        (((a.u) & 0x007FFFFF) != 0)) {
        fprintf(stderr, "SubN -> %+le %08x\n", x, a.u);
    }
#endif
    return a;
}

static inline fann_type_ff fann_ff_clip_limits(float x)
{
    fann_type_ff a;

    a.f = (float)x;
    // float32 significand mask is 0x007FFFFF, for 23 stored bits
    // float16 has 10 stored significand bits:
    //      7   F    F    F    F    F
    // 32: 0xxx xxxx xxxx xxxx xxxx xxxx
    // 16: 0xxx xxxx xxx, which is:
    //      7   F    E    0    0    0
    a.u &= 0xFFFFE000;
    // exceptions:
    if (a.f > 0.0) {
        if (a.f > F16_FF_MAX)
            a.f = F16_FF_MAX;
        else if (a.f < F16_FF_MIN)
            a.f = 0.0;
    } else if (a.f < 0.0) {
        if (a.f < -F16_FF_MAX)
            a.f = -F16_FF_MAX;
        else if (a.f > -F16_FF_MIN)
            a.f = -0.0;
    }
    // NaN / infinity mask: 0x7F800000
#if 0 //def DEBUG_NAN
    if ((a.u & 0x7F800000) == 0x7F800000)
        fprintf(stderr, "NaN/Inf = %s -> %+le %08x\n", debug_nan_args, x, a.u);
#endif
    return a;
}

#else // !HWF16

#define fann_bp_clip_limits(x)
#define fann_ff_clip_limits(x)

#endif
*/

