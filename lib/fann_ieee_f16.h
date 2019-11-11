
/*============================================================================

This C header file is part of the SoftFloat IEEE Floating-Point Arithmetic
Package, Release 3c, by John R. Hauser.

Copyright 2011, 2012, 2013, 2014, 2015, 2016 The Regents of the University of
California.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions, and the following disclaimer in the documentation
    and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors may
    be used to endorse or promote products derived from this software without
    specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

THIS IS A MERGE OF SEVERAL ORIGINAL HEADERS ONLY FOR FLOAT 16

=============================================================================*/

#define fann_ap_reset_stats()

#define fann_f16_debug()

#define LITTLEENDIAN 1

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
#ifdef __GNUC_STDC_INLINE__
#define INLINE inline
#else
#define INLINE extern inline
#endif

#define INLINE_LEVEL 3

#ifndef specialize_h
#define specialize_h 1

#include <stdbool.h>
#include <stdint.h>

#ifndef softfloat_types_h
#define softfloat_types_h 1

/*----------------------------------------------------------------------------
| Types used to pass 16-bit, 32-bit, 64-bit, and 128-bit floating-point
| arguments and results to/from functions.  These types must be exactly
| 16 bits, 32 bits, 64 bits, and 128 bits in size, respectively.  Where a
| platform has "native" support for IEEE-Standard floating-point formats,
| the types below may, if desired, be defined as aliases for the native types
| (typically 'float' and 'double', and possibly 'long double').
*----------------------------------------------------------------------------*/
//typedef uint16_t float16_t;
typedef union { uint16_t u; int16_t i; } float16_t;
typedef union { float f; uint32_t u; } float32_t;

#endif

/*----------------------------------------------------------------------------
| Default value for `softfloat_detectTininess'.
*----------------------------------------------------------------------------*/
#define init_detectTininess softfloat_tininess_afterRounding

/*----------------------------------------------------------------------------
| The values to return on conversions to 32-bit integer formats that raise an
| invalid exception.
*----------------------------------------------------------------------------*/
#define ui32_fromPosOverflow 0xFFFFFFFF
#define ui32_fromNegOverflow 0
#define ui32_fromNaN         0xFFFFFFFF
#define i32_fromPosOverflow  0x7FFFFFFF
#define i32_fromNegOverflow  (-0x7FFFFFFF - 1)
#define i32_fromNaN          0x7FFFFFFF

/*----------------------------------------------------------------------------
| "Common NaN" structure, used to transfer NaN representations from one format
| to another.
*----------------------------------------------------------------------------*/
struct commonNaN {
    bool sign;
#ifdef LITTLEENDIAN
    uint64_t v0, v64;
#else
    uint64_t v64, v0;
#endif
};

/*----------------------------------------------------------------------------
| The bit pattern for a default generated 16-bit floating-point NaN.
*----------------------------------------------------------------------------*/
#define defaultNaNF16UI 0xFE00

/*----------------------------------------------------------------------------
| Returns true when 16-bit unsigned integer `uiA' has the bit pattern of a
| 16-bit floating-point signaling NaN.
| Note:  This macro evaluates its argument more than once.
*----------------------------------------------------------------------------*/
#define softfloat_isSigNaNF16UI( uiA ) ((((uiA) & 0x7E00) == 0x7C00) && ((uiA) & 0x01FF))

/*----------------------------------------------------------------------------
| Assuming `uiA' has the bit pattern of a 16-bit floating-point NaN, converts
| this NaN to the common NaN form, and stores the resulting common NaN at the
| location pointed to by `zPtr'.  If the NaN is a signaling NaN, the invalid
| exception is raised.
*----------------------------------------------------------------------------*/
void softfloat_f16UIToCommonNaN( uint_fast16_t uiA, struct commonNaN *zPtr );

/*----------------------------------------------------------------------------
| Converts the common NaN pointed to by `aPtr' into a 16-bit floating-point
| NaN, and returns the bit pattern of this value as an unsigned integer.
*----------------------------------------------------------------------------*/
uint_fast16_t softfloat_commonNaNToF16UI( const struct commonNaN *aPtr );

/*----------------------------------------------------------------------------
| Interpreting `uiA' and `uiB' as the bit patterns of two 16-bit floating-
| point values, at least one of which is a NaN, returns the bit pattern of
| the combined NaN result.  If either `uiA' or `uiB' has the pattern of a
| signaling NaN, the invalid exception is raised.
*----------------------------------------------------------------------------*/
uint_fast16_t softfloat_propagateNaNF16UI( uint_fast16_t uiA, uint_fast16_t uiB );

/*----------------------------------------------------------------------------
| The bit pattern for a default generated 32-bit floating-point NaN.
*----------------------------------------------------------------------------*/
//#define defaultNaNF32UI 0xFFC00000 // only for F32 operations

/*----------------------------------------------------------------------------
| Returns true when 32-bit unsigned integer `uiA' has the bit pattern of a
| 32-bit floating-point signaling NaN.
| Note:  This macro evaluates its argument more than once.
*----------------------------------------------------------------------------*/
#define softfloat_isSigNaNF32UI( uiA ) ((((uiA) & 0x7FC00000) == 0x7F800000) && ((uiA) & 0x003FFFFF))

/*----------------------------------------------------------------------------
| Assuming `uiA' has the bit pattern of a 32-bit floating-point NaN, converts
| this NaN to the common NaN form, and stores the resulting common NaN at the
| location pointed to by `zPtr'.  If the NaN is a signaling NaN, the invalid
| exception is raised.
*----------------------------------------------------------------------------*/
void softfloat_f32UIToCommonNaN( uint_fast32_t uiA, struct commonNaN *zPtr );

/*----------------------------------------------------------------------------
| Converts the common NaN pointed to by `aPtr' into a 32-bit floating-point
| NaN, and returns the bit pattern of this value as an unsigned integer.
*----------------------------------------------------------------------------*/
uint_fast32_t softfloat_commonNaNToF32UI( const struct commonNaN *aPtr );

#endif


#ifndef primitiveTypes_h
#define primitiveTypes_h 1

/*----------------------------------------------------------------------------
| These macros are used to isolate the differences in word order between big-
| endian and little-endian platforms.
*----------------------------------------------------------------------------*/
#ifdef LITTLEENDIAN
#define wordIncr 1
#define indexWord( total, n ) (n)
#define indexWordHi( total ) ((total) - 1)
#define indexWordLo( total ) 0
#define indexMultiword( total, m, n ) (n)
#define indexMultiwordHi( total, n ) ((total) - (n))
#define indexMultiwordLo( total, n ) 0
#define indexMultiwordHiBut( total, n ) (n)
#define indexMultiwordLoBut( total, n ) 0
#define INIT_UINTM4( v3, v2, v1, v0 ) { v0, v1, v2, v3 }
#else
#define wordIncr -1
#define indexWord( total, n ) ((total) - 1 - (n))
#define indexWordHi( total ) 0
#define indexWordLo( total ) ((total) - 1)
#define indexMultiword( total, m, n ) ((total) - 1 - (m))
#define indexMultiwordHi( total, n ) 0
#define indexMultiwordLo( total, n ) ((total) - (n))
#define indexMultiwordHiBut( total, n ) 0
#define indexMultiwordLoBut( total, n ) (n)
#define INIT_UINTM4( v3, v2, v1, v0 ) { v3, v2, v1, v0 }
#endif

#endif

#ifndef internals_h
#define internals_h 1

union ui16_f16 { uint16_t ui; float16_t f; };
union ui32_f32 { uint32_t ui; float32_t f; };
//union ui64_f64 { uint64_t ui; float64_t f; };

#ifdef SOFTFLOAT_FAST_INT64
//union extF80M_extF80 { struct extFloat80M fM; extFloat80_t f; };
//union ui128_f128 { struct uint128 ui; float128_t f; };
#endif

enum {
    softfloat_mulAdd_subC    = 1,
    softfloat_mulAdd_subProd = 2
};

int_fast32_t softfloat_roundToI32( bool, uint_fast64_t, uint_fast8_t, bool );

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/
#define signF16UI( a ) ((bool) ((uint16_t) (a)>>15))
#define expF16UI( a ) ((int_fast8_t) ((a)>>10) & 0x1F)
#define fracF16UI( a ) ((a) & 0x03FF)
#define packToF16UI( sign, exp, sig ) (((uint16_t) (sign)<<15) + ((uint16_t) (exp)<<10) + (sig))

#define isNaNF16UI( a ) (((~(a) & 0x7C00) == 0) && ((a) & 0x03FF))

struct exp8_sig16 { int_fast8_t exp; uint_fast16_t sig; };
struct exp8_sig16 softfloat_normSubnormalF16Sig( uint_fast16_t );

float16_t softfloat_roundPackToF16( bool, int_fast16_t, uint_fast16_t );
float16_t softfloat_normRoundPackToF16( bool, int_fast16_t, uint_fast16_t );

float16_t softfloat_addMagsF16( uint_fast16_t, uint_fast16_t );
float16_t softfloat_subMagsF16( uint_fast16_t, uint_fast16_t );
float16_t
 softfloat_mulAddF16(
     uint_fast16_t, uint_fast16_t, uint_fast16_t, uint_fast8_t );

/*----------------------------------------------------------------------------
*----------------------------------------------------------------------------*/

#define signF32UI( a ) ((bool) ((uint32_t) (a)>>31))
#define expF32UI( a ) ((int_fast32_t) ((a)>>23) & 0xFF)
#define fracF32UI( a ) ((a) & 0x007FFFFF)
#define packToF32UI( sign, exp, sig ) (((uint32_t) (sign)<<31) + ((uint32_t) (exp)<<23) + (sig))

//#define isNaNF32UI( a ) (((~(a) & 0x7F800000) == 0) && ((a) & 0x007FFFFF))

#endif

#ifndef primitives_h
#define primitives_h 1

#ifndef primitiveTypes_h
#define primitiveTypes_h 1

#ifdef SOFTFLOAT_FAST_INT64

#endif

/*----------------------------------------------------------------------------
| These macros are used to isolate the differences in word order between big-
| endian and little-endian platforms.
*----------------------------------------------------------------------------*/
#ifdef LITTLEENDIAN
#define wordIncr 1
#define indexWord( total, n ) (n)
#define indexWordHi( total ) ((total) - 1)
#define indexWordLo( total ) 0
#define indexMultiword( total, m, n ) (n)
#define indexMultiwordHi( total, n ) ((total) - (n))
#define indexMultiwordLo( total, n ) 0
#define indexMultiwordHiBut( total, n ) (n)
#define indexMultiwordLoBut( total, n ) 0
#define INIT_UINTM4( v3, v2, v1, v0 ) { v0, v1, v2, v3 }
#else
#define wordIncr -1
#define indexWord( total, n ) ((total) - 1 - (n))
#define indexWordHi( total ) 0
#define indexWordLo( total ) ((total) - 1)
#define indexMultiword( total, m, n ) ((total) - 1 - (m))
#define indexMultiwordHi( total, n ) 0
#define indexMultiwordLo( total, n ) ((total) - (n))
#define indexMultiwordHiBut( total, n ) 0
#define indexMultiwordLoBut( total, n ) (n)
#define INIT_UINTM4( v3, v2, v1, v0 ) { v3, v2, v1, v0 }
#endif

#endif

#ifndef softfloat_shiftRightJam32
/*----------------------------------------------------------------------------
| Shifts 'a' right by the number of bits given in 'dist', which must not
| be zero.  If any nonzero bits are shifted off, they are "jammed" into the
| least-significant bit of the shifted value by setting the least-significant
| bit to 1.  This shifted-and-jammed value is returned.
|   The value of 'dist' can be arbitrarily large.  In particular, if 'dist' is
| greater than 32, the result will be either 0 or 1, depending on whether 'a'
| is zero or nonzero.
*----------------------------------------------------------------------------*/
/*#if defined INLINE_LEVEL && (2 <= INLINE_LEVEL)
uint32_t softfloat_shiftRightJam32( uint32_t a, uint_fast16_t dist )
{
    return
        (dist < 31) ? a>>dist | ((uint32_t) (a<<(-dist & 31)) != 0) : (a != 0);
}
#else*/
uint32_t softfloat_shiftRightJam32( uint32_t a, uint_fast16_t dist );
//#endif
#endif

#if defined INLINE_LEVEL && (3 <= INLINE_LEVEL)
INLINE uint64_t softfloat_shiftRightJam64( uint64_t a, uint_fast32_t dist )
{
    return
        (dist < 63) ? a>>dist | ((uint64_t) (a<<(-dist & 63)) != 0) : (a != 0);
}
#else
uint64_t softfloat_shiftRightJam64( uint64_t a, uint_fast32_t dist );
#endif

/*----------------------------------------------------------------------------
| A constant table that translates an 8-bit unsigned integer (the array index)
| into the number of leading 0 bits before the most-significant 1 of that
| integer.  For integer zero (index 0), the corresponding table element is 8.
*----------------------------------------------------------------------------*/
extern const uint_least8_t softfloat_countLeadingZeros8[256];

#ifndef softfloat_countLeadingZeros16
/*----------------------------------------------------------------------------
| Returns the number of leading 0 bits before the most-significant 1 bit of
| 'a'.  If 'a' is zero, 16 is returned.
*----------------------------------------------------------------------------*/
/*#if defined INLINE_LEVEL && (2 <= INLINE_LEVEL)
INLINE uint_fast8_t softfloat_countLeadingZeros16( uint16_t a )
{
    uint_fast8_t count = 8;
    if ( 0x100 <= a ) {
        count = 0;
        a >>= 8;
    }
    count += softfloat_countLeadingZeros8[a];
    return count;
}
#else*/
uint_fast8_t softfloat_countLeadingZeros16( uint16_t a );
//#endif
#endif

#ifndef softfloat_countLeadingZeros32
/*----------------------------------------------------------------------------
| Returns the number of leading 0 bits before the most-significant 1 bit of
| 'a'.  If 'a' is zero, 32 is returned.
*----------------------------------------------------------------------------*/
/*#if defined INLINE_LEVEL && (3 <= INLINE_LEVEL)
INLINE uint_fast8_t softfloat_countLeadingZeros32( uint32_t a )
{
    uint_fast8_t count = 0;
    if ( a < 0x10000 ) {
        count = 16;
        a <<= 16;
    }
    if ( a < 0x1000000 ) {
        count += 8;
        a <<= 8;
    }
    count += softfloat_countLeadingZeros8[a>>24];
    return count;
}
#else*/
uint_fast8_t softfloat_countLeadingZeros32( uint32_t a );
//#endif
#endif

#ifndef softfloat_countLeadingZeros64
/*----------------------------------------------------------------------------
| Returns the number of leading 0 bits before the most-significant 1 bit of
| 'a'.  If 'a' is zero, 64 is returned.
*----------------------------------------------------------------------------*/
//uint_fast8_t softfloat_countLeadingZeros64( uint64_t a );
#endif

extern const uint16_t softfloat_approxRecip_1k0s[16];
extern const uint16_t softfloat_approxRecip_1k1s[16];

extern const uint16_t softfloat_approxRecipSqrt_1k0s[16];
extern const uint16_t softfloat_approxRecipSqrt_1k1s[16];

#ifdef SOFTFLOAT_FAST_INT64

#endif

#endif

/*============================================================================
| Note:  If SoftFloat is made available as a general library for programs to
| use, it is strongly recommended that a platform-specific version of this
| header, "softfloat.h", be created that folds in "softfloat_types.h" and that
| eliminates all dependencies on compile-time macros.
*============================================================================*/


#ifndef softfloat_h
#define softfloat_h 1

#ifndef THREAD_LOCAL
#define THREAD_LOCAL
#endif

/*----------------------------------------------------------------------------
| Software floating-point underflow tininess-detection mode.
*----------------------------------------------------------------------------*/
extern THREAD_LOCAL uint_fast8_t softfloat_detectTininess;
enum {
    softfloat_tininess_beforeRounding = 0,
    softfloat_tininess_afterRounding  = 1
};

/*----------------------------------------------------------------------------
| Software floating-point rounding mode.  (Mode "odd" is supported only if
| SoftFloat is compiled with macro 'SOFTFLOAT_ROUND_ODD' defined.)
*----------------------------------------------------------------------------*/
extern THREAD_LOCAL uint_fast8_t softfloat_roundingMode;
enum {
    softfloat_round_near_even   = 0,
    softfloat_round_minMag      = 1,
    softfloat_round_min         = 2,
    softfloat_round_max         = 3,
    softfloat_round_near_maxMag = 4,
    softfloat_round_odd         = 5
};

/*----------------------------------------------------------------------------
| Software floating-point exception flags.
*----------------------------------------------------------------------------*/
extern THREAD_LOCAL uint_fast8_t softfloat_exceptionFlags;
enum {
    softfloat_flag_inexact   =  1,
    softfloat_flag_underflow =  2,
    softfloat_flag_overflow  =  4,
    softfloat_flag_infinite  =  8,
    softfloat_flag_invalid   = 16
};

/*----------------------------------------------------------------------------
| Routine to raise any or all of the software floating-point exception flags.
*----------------------------------------------------------------------------*/
void softfloat_raiseFlags( uint_fast8_t );

/*----------------------------------------------------------------------------
| Integer-to-floating-point conversion routines.
*----------------------------------------------------------------------------*/
float16_t ui32_to_f16( uint32_t );
float16_t i32_to_f16( int32_t );

/*----------------------------------------------------------------------------
| 16-bit (half-precision) floating-point operations.
*----------------------------------------------------------------------------*/
uint_fast32_t f16_to_ui32( float16_t, uint_fast8_t, bool );
int_fast32_t f16_to_i32( float16_t, uint_fast8_t, bool );
uint_fast32_t f16_to_ui32_r_minMag( float16_t, bool );
int_fast32_t f16_to_i32_r_minMag( float16_t, bool );
float32_t f16_to_f32( float16_t );
float16_t f16_roundToInt( float16_t, uint_fast8_t, bool );
float16_t f16_add( float16_t, float16_t );
float16_t f16_sub( float16_t, float16_t );
float16_t f16_mul( float16_t, float16_t );
float16_t f16_mulAdd( float16_t, float16_t, float16_t );
float16_t f16_div( float16_t, float16_t );
float16_t f16_rem( float16_t, float16_t );
float16_t f16_sqrt( float16_t );
bool f16_eq( float16_t, float16_t );
bool f16_le( float16_t, float16_t );
bool f16_lt( float16_t, float16_t );
bool f16_eq_signaling( float16_t, float16_t );
bool f16_le_quiet( float16_t, float16_t );
bool f16_lt_quiet( float16_t, float16_t );
//bool f16_isSignalingNaN( float16_t );

int_fast32_t f16_to_fix16( float16_t a ); // fixed point with decimal = mant

#endif

//float32_t i32_to_f32( int32_t a );
//int_fast32_t f32_to_i32( float32_t a, uint_fast8_t roundingMode, bool exact );
float16_t f32_to_f16( float32_t a );

#define FP_BIAS 15
#define FP_BIAS_DEFAULT 15

//#define F16_MAX 6.550400000000e+04
//#define F16_MIN 6.10351562500000e-05 // normal
//#define F16_MIN 5.9604644775390625000000e-08 

extern unsigned int fann_ap_cancel;
extern unsigned int fann_ap_underflow;
extern unsigned int fann_ap_overflow;



