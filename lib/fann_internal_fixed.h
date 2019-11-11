#include <stdint.h>
typedef int32_t cpu_type;

#define FIX_FRAC 22 
//#define FIX_FRAC 16

#define cpu_to_float(cpu) ((float)(cpu) / (float)(1<<FIX_FRAC))
#define cpu_to_int(cpu)   ((int)cpu_to_float(cpu))
#define float_to_cpu(val) ((cpu_type)((val) * (float)(1<<FIX_FRAC)))
#define int_to_cpu(val)   (float_to_cpu((float)(val)))

#define cpu_abs(x) (((x) < 0) ? -x : x)
#define cpu_min(x, y) (((x) < (y)) ? (x) : (y))
#define cpu_max(x, y) (((x) > (y)) ? (x) : (y))
#define cpu_neg(x) (-(x))
#define cpu_add(x, y) ((x) + (y))
#define cpu_sub(x, y) ((x) - (y))

/*
The C11 standard, section 6.5, paragraph 4 [ISO/IEC 9899:2011], states:

Some operators (the unary operator ~ , and the binary operators <<, >>, &,
^, and |, collectively described as bitwise operators) shall have operands
that have integral type. These operators return values that depend on the
internal representations of integers, and thus have implementation-defined
and undefined aspects for signed types.

Code generated for all tested processors generate 'arithmetic right shift'
for signed values and 'logical right shift' for unsigned. Check the
behaviour for your compiler/processor. Even when supported by hardware,
'divisions' rounds towards zero while shift rounds towards -infinity.
*/

#define cpu_mul(x, y) ((cpu_type)(((int64_t)(x) * (y)) >> FIX_FRAC))
#define cpu_div(x, y) ((cpu_type)(((int64_t)(x) << FIX_FRAC) / (y)))

#define cpu_mul_2(x)      ((x) << 1)
#define cpu_mul_4(x)      ((x) << 2)
#define cpu_mul_8(x)      ((x) << 3)
#define cpu_mul_16(x)     ((x) << 4)
#define cpu_mul_32(x)     ((x) << 5)

#define cpu_div_2(x)      ((x) >> 1)
#define cpu_div_4(x)      ((x) >> 2)
#define cpu_div_8(x)      ((x) >> 3)
#define cpu_div_16(x)     ((x) >> 4)
#define cpu_div_32(x)     ((x) >> 5)

#define cpu_mul_05(x)     cpu_div_2(x)
#define cpu_mul_025(x)    cpu_div_4(x)
#define cpu_mul_0125(x)   cpu_div_8(x)
#define cpu_mul_0625(x)   cpu_mul((x), float_to_cpu(0.625))
#define cpu_mul_075(x)    cpu_mul((x), float_to_cpu(0.75))
#define cpu_mul_0875(x)   cpu_mul((x), float_to_cpu(0.875))

#define cpu_eq(x, y)  ((x) == (y))
#define cpu_neq(x, y) ((x) != (y))
#define cpu_gte(x, y) ((x) >= (y))
#define cpu_lte(x, y) ((x) <= (y))
#define cpu_gt(x, y)  ((x) > (y))
#define cpu_lt(x, y)  ((x) < (y))

#define cpu_is_zero(x) ((x) == 0)
#define cpu_non_zero(x) ((x) != 0)
#define cpu_is_pos(x) ((x) > 0)
#define cpu_is_neg(x) ((x) < 0)
#define cpu_not_neg(x) ((x) >= 0)
#define cpu_not_pos(x) ((x) <= 0)

#define fann_bp_max(x,y) ((cpu_gt((x), (y))) ? (x) : (y))
#define fann_ff_max(x,y) ((cpu_gt((x), (y))) ? (x) : (y))
#define fann_bp_min(x,y) ((cpu_lt((x), (y))) ? (x) : (y)) 
#define fann_ff_min(x,y) ((cpu_lt((x), (y))) ? (x) : (y))
#define fann_bp_mul(x,y) (cpu_mul((x), (y))) 
#define fann_ff_mul(x,y) (cpu_mul((x), (y)))
//#define fann_bp_div(x,y) (cpu_div((x), (y)))
#define fann_ff_div(x,y) (cpu_div((x), (y)))
#define fann_bp_add(x,y) (cpu_add((x), (y))) 
#define fann_ff_add(x,y) (cpu_add((x), (y))) 
#define fann_bp_sub(x,y) (cpu_sub((x), (y))) 
#define fann_ff_sub(x,y) (cpu_sub((x), (y)))
#define fann_bp_mac(x,y,c) (cpu_add((c), (cpu_mul((x), (y)))))
#define fann_ff_mac(x,y,c) (cpu_add((c), (cpu_mul((x), (y))))) 

#define fann_bp_abs(w) (fann_abs(w))
#define fann_ff_abs(w) (fann_abs(w))
#define fann_ff_neg(w) (cpu_neg(w))
#define fann_bp_neg(w) (cpu_neg(w))

#define fann_int_to_bp(i) (int_to_cpu(i))
#define fann_int_to_ff(i) (int_to_cpu(i))
#define fann_float_to_bp(f) (float_to_cpu(f))
#define fann_float_to_ff(f) (float_to_cpu(f))
//#define fann_bp_to_int(w, b)
//#define fann_ff_to_int(w)
#define fann_bp_to_float(w) (cpu_to_float(w))
#define fann_ff_to_float(w) (cpu_to_float(w))
#define fann_ff_to_bp(n) ((cpu_type)(n))
#define fann_bp_to_ff(a) ((cpu_type)(a))
#define fann_bp_to_bp(a, s) ((cpu_type)(a))

#define fann_ff_clip(x,y,z) fann_clip((x),(y),(z))
#define fann_bp_clip(x,y,z) fann_clip((x),(y),(z))
#define fann_bp_lt(x,y) (cpu_lt((x), (y))) 
#define fann_ff_lt(x,y) (cpu_lt((x), (y)))
#define fann_bp_ne(x,y) (cpu_ne((x), (y)))
#define fann_ff_ne(x,y) (cpu_ne((x), (y)))
#define fann_bp_gt(x,y) (cpu_gt((x), (y)))
#define fann_ff_gt(x,y) (cpu_gt((x), (y)))
#define fann_bp_is_neg(x) ((x)<0)
#define fann_ff_is_neg(x) ((x)<0)
#define fann_bp_is_pos(x) ((x)>0)
#define fann_ff_is_pos(x) ((x)>0)
#define fann_bp_is_non_zero(x) ((x)!=0)
#define fann_ff_is_non_zero(x) ((x)!=0)
#define fann_bp_is_zero(x) ((x)==0)
#define fann_ff_is_zero(x) ((x)==0)

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


