//int fann_ff_to_int(fann_type_ff f);
float fann_ff_to_float(fann_type_ff f);
fann_type_ff fann_ff_mac(fann_type_ff x, fann_type_ff y, fann_type_ff c);
fann_type_ff fann_ff_mul(fann_type_ff x, fann_type_ff y);
fann_type_ff fann_ff_div(fann_type_ff x, fann_type_ff y);
fann_type_ff fann_ff_add(fann_type_ff x, fann_type_ff y);
fann_type_ff fann_ff_sub(fann_type_ff x, fann_type_ff y);
fann_type_ff fann_int_to_ff(int i);
fann_type_ff fann_float_to_ff(float f);
fann_type_ff fann_ff_abs(fann_type_ff x);
fann_type_ff fann_ff_neg(fann_type_ff x);
fann_type_ff fann_ff_neg(fann_type_ff f);
#ifdef EMUL_FLOAT
fann_type_ff fann_ff_min(fann_type_ff x, fann_type_ff y);
fann_type_ff fann_ff_max(fann_type_ff x, fann_type_ff y);
fann_type_ff fann_ff_clip(fann_type_ff x, fann_type_ff lo, fann_type_ff hi);
int fann_ff_gt(fann_type_ff x, fann_type_ff y);
int fann_ff_ne(fann_type_ff x, fann_type_ff y);
int fann_ff_lt(fann_type_ff x, fann_type_ff y);
int fann_ff_is_neg(fann_type_ff x);
int fann_ff_is_pos(fann_type_ff x);
int fann_ff_is_zero(fann_type_ff x);
#define fann_ff_is_non_zero(x) (!fann_ff_is_zero(x))
#else // ! EMUL_FLOAT
#define fann_ff_min(x,y) (((x.f) < (y.f)) ? (x) : (y)) 
#define fann_ff_max(x,y) (((x.f) > (y.f)) ? (x) : (y))
#define fann_ff_clip(x, lo, hi) (((x.f) < (lo.f)) ? (lo) : (((x.f) > (hi.f)) ? (hi) : (x)))
#define fann_ff_ne(x,y) (((x).f)!=((y).f))
#define fann_ff_lt(x,y) (((x).f)<((y).f))
#define fann_ff_gt(x,y) ((x).f)>((y).f)
#define fann_ff_is_neg(x) (((x).f)<0.0)
#define fann_ff_is_pos(x) (((x).f)>0.0)
#define fann_ff_is_zero(x) (((x).f)==0.0)
#define fann_ff_is_non_zero(x) (((x).f)!=0.0)
#endif // ! EMUL_FLOAT

#ifndef FANN_INFERENCE_ONLY

fann_type_bp fann_ff_to_bp(fann_type_ff n);
fann_type_ff fann_bp_to_ff(fann_type_bp a);
float fann_bp_to_float(fann_type_bp f);
fann_type_bp fann_float_to_bp(float f);
fann_type_bp fann_int_to_bp(int i);

#if (defined SWF16_AP) || (defined HWF16)
fann_type_bp fann_bp_to_bp(fann_type_bp n, int_fast8_t bp_bias);
#elif (defined SWF16_IEEE) || (defined SWF32_IEEE)
#define fann_bp_to_bp(n, b) (n)
#else
#error "SOFTFANN definition"
#endif

fann_type_bp fann_bp_mac(fann_type_bp x, fann_type_bp y, fann_type_bp c);
fann_type_bp fann_bp_mul(fann_type_bp x, fann_type_bp y);
//fann_type_bp fann_bp_div(fann_type_bp x, fann_type_bp y);
fann_type_bp fann_bp_add(fann_type_bp x, fann_type_bp y);
fann_type_bp fann_bp_sub(fann_type_bp x, fann_type_bp y);
fann_type_bp fann_bp_abs(fann_type_bp x);
fann_type_bp fann_bp_neg(fann_type_bp x);
fann_type_bp fann_bp_neg(fann_type_bp f);
#ifdef EMUL_FLOAT
fann_type_bp fann_bp_min(fann_type_bp x, fann_type_bp y);
fann_type_bp fann_bp_max(fann_type_bp x, fann_type_bp y);
fann_type_bp fann_bp_clip(fann_type_bp x, fann_type_bp lo, fann_type_bp hi);
int fann_bp_gt(fann_type_bp x, fann_type_bp y);
int fann_bp_ne(fann_type_bp x, fann_type_bp y);
int fann_bp_lt(fann_type_bp x, fann_type_bp y);
int fann_bp_is_neg(fann_type_bp x);
int fann_bp_is_pos(fann_type_bp x);
int fann_bp_is_zero(fann_type_bp x);
#define fann_bp_is_non_zero(x) (!fann_bp_is_zero(x))
#else // EMUL_FLOAT
#define fann_bp_min(x,y) (((x.f) < (y.f)) ? (x) : (y)) 
#define fann_bp_max(x,y) (((x.f) > (y.f)) ? (x) : (y))
#define fann_bp_clip(x, lo, hi) (((x.f) < (lo.f)) ? (lo) : (((x.f) > (hi.f)) ? (hi) : (x)))
#define fann_bp_ne(x,y) (((x).f)!=((y).f))
#define fann_bp_lt(x,y) (((x).f)<((y).f))
#define fann_bp_gt(x,y) ((x).f)>((y).f)
#define fann_bp_is_neg(x) (((x).f)<0.0)
#define fann_bp_is_pos(x) (((x).f)>0.0)
#define fann_bp_is_zero(x) (((x).f)==0.0)
#define fann_bp_is_non_zero(x) (((x).f)!=0.0)
#endif // EMUL_FLOAT
#endif // FANN_INFERENCE_ONLY

/* native type for temporary calculations: */
#if 1

#define NT_0000                     ff_0000
#define fann_float_to_nt(x)         fann_float_to_ff((x))
#define fann_nt_to_float(x)         fann_ff_to_float((x))
#define fann_ff_to_nt(x)            (x)
#define fann_nt_to_ff(x)            (x)
#define fann_nt_is_zero(x)          fann_ff_is_zero((x))
#define fann_nt_neg(x)              fann_ff_neg((x))
#define fann_nt_gt(x, y)            fann_ff_gt((x), (y))
#define fann_nt_lt(x, y)            fann_ff_lt((x), (y))
#define fann_nt_add(x, y)           fann_ff_add((x), (y))
#define fann_nt_sub(x, y)           fann_ff_sub((x), (y))
#define fann_nt_mul(x, y)           fann_ff_mul((x), (y))
#define fann_nt_div(x, y)           fann_ff_div((x), (y))
#define fann_nt_mac(x, y, c)        fann_ff_mac((x), (y), (c))

#else

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

#endif


