
#ifndef _FANN_SQRT_H
#define _FANN_SQRT_H

#undef FANN_SQRT_EMULATION

#ifdef FLOATFANN

#if (defined _GCC_ARM_F16_FF)

#if 1
#define FANN_SQRT_EMULATION
#else
#include <math.h>
#define fann_bp_rsqrt(x) fann_float_to_bp(((float)(1.0/sqrtf(((float)(x).f)))))
#define fann_bp_b_rsqrt_a(b, a) fann_float_to_bp(((float)((float)(b).f/sqrtf(((float)(a).f)))))
#define fann_rsqrt_name "1/sqrtf((float)(__fp16))"
#endif

#elif (defined _GCC_ARM_F16_BP)

#if 0
#define FANN_SQRT_EMULATION
#else
#include <math.h>
#define fann_bp_rsqrt(x) fann_float_to_bp(((float)(1.0/sqrtf(((float)(x).f)))))
#define fann_bp_b_rsqrt_a(b, a) fann_float_to_bp(((float)((float)(b).f/sqrtf(((float)(a).f)))))
#define fann_rsqrt_name "1/sqrtf((float)(__fp16))"
#endif

#elif (defined _FLOAT_UNION)

#if 0
#define FANN_SQRT_EMULATION
#else
#include <math.h>
#define fann_bp_rsqrt(val) ({fann_type_bp ret; ret.f = 1.0/sqrtf((val).f); ret;})
#define fann_bp_b_rsqrt_a(b, a) ({fann_type_bp ret; ret.f = (b).f/sqrtf((a).f); ret;})
#define fann_rsqrt_name "1/sqrtf(.f)"
#endif

#elif (defined _BFLOAT16)

#if 0
#define FANN_SQRT_EMULATION
#else
#include <math.h>
#define fann_bp_rsqrt(val) ({fann_type_bp ret; float e; e = fann_bp_to_float(val); e = 1.0/sqrtf(e); ret = fann_float_to_bp(e); ret;})
#define fann_bp_b_rsqrt_a(b, a) ({fann_type_bp ret; float ea, eb; ea = fann_bp_to_float(a); eb = fann_bp_to_float(b); eb = (eb)/sqrtf((ea)); ret = fann_float_to_bp(eb); ret;})
#define fann_rsqrt_name "1/sqrtf(.f)"
#endif

#else // native float 

#include <math.h>
#define fann_bp_rsqrt(x) (1.0/sqrtf((x)))
#define fann_bp_b_rsqrt_a(b, a) ((b)/sqrtf((a)))
#define fann_rsqrt_name "1/sqrtf"

#endif

#elif defined DOUBLEFANN

#include <math.h>
#define fann_bp_rsqrt(x) (1.0/sqrt((x)))
#define fann_bp_b_rsqrt_a(b, a) ((b)/sqrt((a)))
#define fann_rsqrt_name "1/sqrt"

#elif defined SOFTFANN

#if 1
#define FANN_SQRT_EMULATION
#else
#include <math.h>
#define fann_bp_rsqrt(x) fann_float_to_bp(((float)(1.0/sqrtf(fann_bp_to_float(x)))))
#define fann_bp_b_rsqrt_a(b, a) fann_float_to_bp((float)(fann_bp_to_float(b)/sqrtf(fann_bp_to_float(a))))
#define fann_rsqrt_name "1/sqrtf((float)(__fp16))"
#endif

#elif defined FIXEDFANN

#include <math.h>
#define fann_bp_rsqrt(x) float_to_cpu(1.0/sqrtf((cpu_to_float(x))))
#define fann_bp_b_rsqrt_a(b, a) float_to_cpu(((cpu_to_float(b))/sqrtf(cpu_to_float(a))))
#define fann_rsqrt_name "1/sqrtf"

#else

#error "COMPILATION TYPE"

#endif // *FANN

#ifdef FANN_SQRT_EMULATION
fann_type_bp fann_bp_b_rsqrt_a(fann_type_bp b, fann_type_bp a);
extern const char fann_rsqrt_name[];
#endif

#endif // _FANN_SQRT_H

