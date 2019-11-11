// bfloat16: 8bit exponent (google / intel)

/*         1          2          3
01234567 89012345 67890123 45678901
01234567 01234567 01234567 01234567
SEEEEEEE EMMMMMMM MMMMMMMM MMMMMMMM
          1234567 
*/

typedef union { float f; uint32_t u; } fcpu;

#define bp_cpu(x) ({fcpu ret; ret.u = ((uint32_t)((x).u) << 16); ret;})
#define ff_cpu(x) ({fcpu ret; ret.u = ((uint32_t)((x).u) << 16); ret;})
#if 0 // truncate:
#define cpu_bp(x) ({fann_type_bp ret; ret.u = ((x).u >> 16); ret;})
#define cpu_ff(x) ({fann_type_ff ret; ret.u = ((x).u >> 16); ret;})
#else // round to nearest:
#define cpu_bp(x) ({fann_type_bp ret; uint32_t e, m, s; \
                    e = (x).u & 0x7F800000; s = (x).u & 0x80000000; \
                    if (e == 0) { ret.u = s; } else if (e == 0x7F800000) { ret.u = s | 0x7FFFFFFF; } else { \
                    m = ((x).u & 0x007FFFFF) | 0x00800000; \
                    m += 0x00007FFF; if (m & 0x01000000) {m >>= 1; e += 0x00800000;} m &= 0x007FFFFF; \
                    ret.u = ((s | e | m) >> 16);} ret;})
#define cpu_ff(x) ({fann_type_ff ret; uint32_t e, m, s; \
                    e = (x).u & 0x7F800000; s = (x).u & 0x80000000; \
                    if (e == 0) { ret.u = s; } else if (e == 0x7F800000) { ret.u = s | 0x7FFFFFFF; } else { \
                    m = ((x).u & 0x007FFFFF) | 0x00800000; \
                    m += 0x00007FFF; if (m & 0x01000000) {m >>= 1; e += 0x00800000;} m &= 0x007FFFFF; \
                    ret.u = ((s | e | m) >> 16);} ret;})
#endif
#define float_bp(fp) ({fcpu retf; fann_type_bp retbp; retf.f = (fp); retbp = cpu_bp(retf); retbp;})
#define float_ff(fp) ({fcpu retf; fann_type_ff retff; retf.f = (fp); retff = cpu_ff(retf); retff;})

#define fann_bp_max(a,b) ({fann_type_bp ret; fcpu x, y; x = bp_cpu(a); y = bp_cpu(b); ret = ((((x).f) > ((y).f)) ? (a) : (b)); ret;})
#define fann_ff_max(a,b) ({fann_type_ff ret; fcpu x, y; x = ff_cpu(a); y = ff_cpu(b); ret = ((((x).f) > ((y).f)) ? (a) : (b)); ret;})
#define fann_bp_min(a,b) ({fann_type_bp ret; fcpu x, y; x = bp_cpu(a); y = bp_cpu(b); ret = ((((x).f) < ((y).f)) ? (a) : (b)); ret;})
#define fann_ff_min(a,b) ({fann_type_ff ret; fcpu x, y; x = ff_cpu(a); y = ff_cpu(b); ret = ((((x).f) < ((y).f)) ? (a) : (b)); ret;})

#define fann_bp_mul(a,b) ({fann_type_bp ret; fcpu x, y; x = bp_cpu(a); y = bp_cpu(b); ret = float_bp((((x).f) * ((y).f))); ret;})
#define fann_ff_mul(a,b) ({fann_type_ff ret; fcpu x, y; x = bp_cpu(a); y = bp_cpu(b); ret = float_ff((((x).f) * ((y).f))); ret;})
//#define fann_bp_div(x,y) ({fann_type_bp ret; ret.f = ((x).f)/((y).f); ret;})
#define fann_ff_div(a,b) ({fann_type_ff ret; fcpu x, y; x = bp_cpu(a); y = bp_cpu(b); ret = float_ff((((x).f) / ((y).f))); ret;})

#define fann_bp_add(a,b) ({fann_type_bp ret; fcpu x, y; x = bp_cpu(a); y = bp_cpu(b); ret = float_bp((((x).f) + ((y).f))); ret;})
#define fann_ff_add(a,b) ({fann_type_ff ret; fcpu x, y; x = bp_cpu(a); y = bp_cpu(b); ret = float_ff((((x).f) + ((y).f))); ret;})
#define fann_bp_sub(a,b) ({fann_type_bp ret; fcpu x, y; x = bp_cpu(a); y = bp_cpu(b); ret = float_bp((((x).f) - ((y).f))); ret;})
#define fann_ff_sub(a,b) ({fann_type_ff ret; fcpu x, y; x = bp_cpu(a); y = bp_cpu(b); ret = float_ff((((x).f) - ((y).f))); ret;})

#define fann_bp_mac(a,b,c) ({fann_type_bp ret; fcpu x, y, z; x = bp_cpu(a); y = bp_cpu(b); z = bp_cpu(c); ret = float_bp(((((x).f) * ((y).f)) + ((z).f))); ret;})
#define fann_ff_mac(a,b,c) ({fann_type_ff ret; fcpu x, y, z; x = ff_cpu(a); y = ff_cpu(b); z = ff_cpu(c); ret = float_ff(((((x).f) * ((y).f)) + ((z).f))); ret;})

#define fann_int_to_bp(i) ({fann_type_bp ret; fcpu x; x.f = (float)i; ret = cpu_bp(x); ret;})
#define fann_int_to_ff(i) ({fann_type_ff ret; fcpu x; x.f = (float)i; ret = cpu_ff(x); ret;})
#define fann_float_to_bp(i) ({fann_type_bp ret; fcpu x; x.f = i; ret = cpu_bp(x); ret;})
#define fann_float_to_ff(i) ({fann_type_ff ret; fcpu x; x.f = i; ret = cpu_ff(x); ret;})

//#define fann_bp_to_int(w, b) ((int)((w).f))
//#define fann_ff_to_int(w) ((int)((w).f))

#define fann_bp_to_float(w) ({float ret; fcpu x; x = bp_cpu(w); ret = x.f; ret;})
#define fann_ff_to_float(w) ({float ret; fcpu x; x = ff_cpu(w); ret = x.f; ret;})

#define fann_ff_to_bp(ff) ({fann_type_bp ret; ret.u = (ff).u; ret;})
#define fann_bp_to_ff(bp) ({fann_type_ff ret; ret.u = (bp).u; ret;})
#define fann_bp_to_bp(bp, bias) ({fann_type_bp ret; ret.u = (bp).u; ret;})

#define fann_bp_abs(val) ({fann_type_bp ret; fcpu x; x = bp_cpu(val); x.f = fabsf(x.f); ret = cpu_bp(x); ret;})
#define fann_ff_abs(val) ({fann_type_ff ret; fcpu x; x = ff_cpu(val); x.f = fabsf(x.f); ret = cpu_ff(x); ret;})
#define fann_bp_neg(val) ({fann_type_bp ret; fcpu x; x = bp_cpu(val); x.f = -x.f; ret = cpu_bp(x); ret;})
#define fann_ff_neg(val) ({fann_type_ff ret; fcpu x; x = ff_cpu(val); x.f = -x.f; ret = cpu_ff(x); ret;})

#define fann_ff_clip(x, lo, hi) (fann_bp_min(fann_bp_max(x, lo), hi)) 
#define fann_bp_clip(x,y,z) fann_ff_clip((x),(y),(z))

#define fann_bp_is_non_zero(x) (((x).u&0x7FFF)!=0)
#define fann_ff_is_non_zero(x) (((x).u&0x7FFF)!=0)
#define fann_bp_is_zero(x) (((x).u&0x7FFF)==0)
#define fann_ff_is_zero(x) (((x).u&0x7FFF)==0)

#include <stdbool.h>
#define fann_bp_ne(a,b) ({bool ret; fcpu x, y; x = bp_cpu(a); y = bp_cpu(b); ret = ((x).f)!=((y).f); ret;})
#define fann_ff_ne(a,b) ({bool ret; fcpu x, y; x = ff_cpu(a); y = ff_cpu(b); ret = ((x).f)!=((y).f); ret;})
#define fann_bp_lt(a,b) ({bool ret; fcpu x, y; x = bp_cpu(a); y = bp_cpu(b); ret = ((x).f)<((y).f); ret;})
#define fann_ff_lt(a,b) ({bool ret; fcpu x, y; x = ff_cpu(a); y = ff_cpu(b); ret = ((x).f)<((y).f); ret;})
#define fann_bp_gt(a,b) ({bool ret; fcpu x, y; x = bp_cpu(a); y = bp_cpu(b); ret = ((x).f)>((y).f); ret;})
#define fann_ff_gt(a,b) ({bool ret; fcpu x, y; x = ff_cpu(a); y = ff_cpu(b); ret = ((x).f)>((y).f); ret;})
#define fann_bp_is_neg(a) ({bool ret; fcpu x; x = bp_cpu(a); ret = ((x).f)<0.0; ret;})
#define fann_ff_is_neg(a) ({bool ret; fcpu x; x = ff_cpu(a); ret = ((x).f)<0.0; ret;})
#define fann_bp_is_pos(a) ({bool ret; fcpu x; x = bp_cpu(a); ret = ((x).f)>0.0; ret;})
#define fann_ff_is_pos(a) ({bool ret; fcpu x; x = ff_cpu(a); ret = ((x).f)>0.0; ret;})

// native type for temporary calculations:
#define NT_0000                     0.0
#define fann_float_to_nt(x)         ((x))
#define fann_nt_to_float(x)         ((x))
#define fann_ff_to_nt(x)            fann_ff_to_float((x))
#define fann_nt_to_ff(x)            fann_float_to_ff((x))
#define fann_nt_is_zero(x)          ((x)==0.0)
#define fann_nt_neg(x)              (-(x))
#define fann_nt_gt(x, y)            ((x) > (y))
#define fann_nt_lt(x, y)            ((x) < (y))
#define fann_nt_add(x, y)           ((x) + (y))
#define fann_nt_sub(x, y)           ((x) - (y))
#define fann_nt_mul(x, y)           ((x) * (y))
#define fann_nt_div(x, y)           ((x) / (y))
#define fann_nt_mac(x, y, c)        ((x) * (y) + (c))


