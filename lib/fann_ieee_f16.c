/*

This file was merged from several parts of the Berkeley SoftFloat
library, by John R. Hauser. For this reason, his copyright notice is
copied below.

  Fast Artificial Neural Network Library - Floating Point Tests Version
  Copyright (C) 2017-2018 Vitor Angelo (vangelo.ft@gmail.com)
  
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

  This library was based on: Fast Artificial Neural Network Library (fann)

License for Berkeley SoftFloat Release 3c

John R. Hauser
2017 February 10

The following applies to the whole of SoftFloat Release 3c as well as to
each source file individually.

Copyright 2011, 2012, 2013, 2014, 2015, 2016, 2017 The Regents of the
University of California.  All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions, and the following disclaimer.

 2. Redistributions in binary form must reproduce the above copyright
    notice, this list of conditions, and the following disclaimer in the
    documentation and/or other materials provided with the distribution.

 3. Neither the name of the University nor the names of its contributors
    may be used to endorse or promote products derived from this software
    without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE REGENTS AND CONTRIBUTORS "AS IS", AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE, ARE
DISCLAIMED.  IN NO EVENT SHALL THE REGENTS OR CONTRIBUTORS BE LIABLE FOR ANY
DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF
THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
*/

//#include <stdio.h>

#include "fann_ieee_f16.h"

// FIXME
unsigned int fann_ap_cancel = 0;
unsigned int fann_ap_underflow = 0;
unsigned int fann_ap_overflow = 0;

THREAD_LOCAL uint_fast8_t softfloat_roundingMode = softfloat_round_near_even;
THREAD_LOCAL uint_fast8_t softfloat_detectTininess = init_detectTininess;
THREAD_LOCAL uint_fast8_t softfloat_exceptionFlags = 0;

#define SOFTFLOAT_FAST_DIV32TO16
#undef SOFTFLOAT_FAST_DIV32TO16

#ifndef SOFTFLOAT_FAST_DIV32TO16
const uint16_t softfloat_approxRecip_1k0s[16] = {
    0xFFC4, 0xF0BE, 0xE363, 0xD76F, 0xCCAD, 0xC2F0, 0xBA16, 0xB201,
    0xAA97, 0xA3C6, 0x9D7A, 0x97A6, 0x923C, 0x8D32, 0x887E, 0x8417
};
const uint16_t softfloat_approxRecip_1k1s[16] = {
    0xF0F1, 0xD62C, 0xBFA1, 0xAC77, 0x9C0A, 0x8DDB, 0x8185, 0x76BA,
    0x6D3B, 0x64D4, 0x5D5C, 0x56B1, 0x50B6, 0x4B55, 0x4679, 0x4211
};
#endif

void softfloat_raiseFlags( uint_fast8_t flags )
{
    softfloat_exceptionFlags |= flags;
}

struct exp8_sig16 softfloat_normSubnormalF16Sig( uint_fast16_t sig )
{
    int_fast8_t shiftDist;
    struct exp8_sig16 z;

    shiftDist = softfloat_countLeadingZeros16( sig ) - 5;
    z.exp = 1 - shiftDist;
    z.sig = sig<<shiftDist;
    return z;
}

/*----------------------------------------------------------------------------
| Interpreting `uiA' and `uiB' as the bit patterns of two 16-bit floating-
| point values, at least one of which is a NaN, returns the bit pattern of
| the combined NaN result.  If either `uiA' or `uiB' has the pattern of a
| signaling NaN, the invalid exception is raised.
*----------------------------------------------------------------------------*/
uint_fast16_t
 softfloat_propagateNaNF16UI( uint_fast16_t uiA, uint_fast16_t uiB )
{
    bool isSigNaNA;

    isSigNaNA = softfloat_isSigNaNF16UI( uiA );
    if ( isSigNaNA || softfloat_isSigNaNF16UI( uiB ) ) {
        softfloat_raiseFlags( softfloat_flag_invalid );
        if ( isSigNaNA ) return uiA | 0x0200;
    }
    return (isNaNF16UI( uiA ) ? uiA : uiB) | 0x0200;
}

/*----------------------------------------------------------------------------
| Converts the common NaN pointed to by `aPtr' into a 16-bit floating-point
| NaN, and returns the bit pattern of this value as an unsigned integer.
*----------------------------------------------------------------------------*/
uint_fast16_t softfloat_commonNaNToF16UI( const struct commonNaN *aPtr )
{
    return (uint_fast16_t) aPtr->sign<<15 | 0x7E00 | aPtr->v64>>54;
}

/*----------------------------------------------------------------------------
| Converts the common NaN pointed to by `aPtr' into a 32-bit floating-point
| NaN, and returns the bit pattern of this value as an unsigned integer.
*----------------------------------------------------------------------------*/
uint_fast32_t softfloat_commonNaNToF32UI( const struct commonNaN *aPtr )
{
    return (uint_fast32_t) aPtr->sign<<31 | 0x7FC00000 | aPtr->v64>>41;
}

/*----------------------------------------------------------------------------
| Assuming `uiA' has the bit pattern of a 16-bit floating-point NaN, converts
| this NaN to the common NaN form, and stores the resulting common NaN at the
| location pointed to by `zPtr'.  If the NaN is a signaling NaN, the invalid
| exception is raised.
*----------------------------------------------------------------------------*/
void softfloat_f16UIToCommonNaN( uint_fast16_t uiA, struct commonNaN *zPtr )
{
    if ( softfloat_isSigNaNF16UI( uiA ) ) {
        softfloat_raiseFlags( softfloat_flag_invalid );
    }
    zPtr->sign = uiA>>15;
    zPtr->v64  = (uint_fast64_t) uiA<<54;
    zPtr->v0   = 0;
}

/*----------------------------------------------------------------------------
| Assuming `uiA' has the bit pattern of a 32-bit floating-point NaN, converts
| this NaN to the common NaN form, and stores the resulting common NaN at the
| location pointed to by `zPtr'.  If the NaN is a signaling NaN, the invalid
| exception is raised.
*----------------------------------------------------------------------------*/
void softfloat_f32UIToCommonNaN( uint_fast32_t uiA, struct commonNaN *zPtr )
{
    if ( softfloat_isSigNaNF32UI( uiA ) ) {
        softfloat_raiseFlags( softfloat_flag_invalid );
    }
    zPtr->sign = uiA>>31;
    zPtr->v64  = (uint_fast64_t) uiA<<41;
    zPtr->v0   = 0;
}

float16_t
 softfloat_roundPackToF16( bool sign, int_fast16_t exp, uint_fast16_t sig )
{
    uint_fast8_t roundingMode;
    bool roundNearEven;
    uint_fast8_t roundIncrement, roundBits;
    bool isTiny;
    uint_fast16_t uiZ;
    union ui16_f16 uZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    roundingMode = softfloat_roundingMode;
    roundNearEven = (roundingMode == softfloat_round_near_even);
    roundIncrement = 0x8;
    if ( ! roundNearEven && (roundingMode != softfloat_round_near_maxMag) ) {
        roundIncrement =
            (roundingMode
                 == (sign ? softfloat_round_min : softfloat_round_max))
                ? 0xF
                : 0;
    }
    roundBits = sig & 0xF;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( 0x1D <= (unsigned int) exp ) {
        if ( exp < 0 ) {
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            isTiny =
                (softfloat_detectTininess == softfloat_tininess_beforeRounding)
                    || (exp < -1) || (sig + roundIncrement < 0x8000);
            sig = softfloat_shiftRightJam32( sig, -exp );
            exp = 0;
            roundBits = sig & 0xF;
            if ( isTiny && roundBits ) {
                softfloat_raiseFlags( softfloat_flag_underflow );
            }
        } else if ( (0x1D < exp) || (0x8000 <= sig + roundIncrement) ) {
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            softfloat_raiseFlags(
                softfloat_flag_overflow | softfloat_flag_inexact );
            uiZ = packToF16UI( sign, 0x1F, 0 ) - ! roundIncrement;
            goto uiZ;
        }
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sig = (sig + roundIncrement)>>4;
    if ( roundBits ) {
        softfloat_exceptionFlags |= softfloat_flag_inexact;
#ifdef SOFTFLOAT_ROUND_ODD
        if ( roundingMode == softfloat_round_odd ) {
            sig |= 1;
            goto packReturn;
        }
#endif
    }
    // roundBit == 0 -> mask = FFFF
    // discarded bits non-zero -> mask = FFFF
    // roundBit == 1 AND discarded bits zero -> mask = FFFE
    sig &= ~(uint_fast16_t) (! (roundBits ^ 8) & roundNearEven);
    if ( ! sig ) exp = 0;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
#ifdef SOFTFLOAT_ROUND_ODD
 packReturn:
#endif
    uiZ = packToF16UI( sign, exp, sig );
 uiZ:
    uZ.ui = uiZ;
    return uZ.f;

}

float16_t i32_to_f16( int32_t a )
{
    bool sign;
    uint_fast32_t absA;
    int_fast8_t shiftDist;
    union ui16_f16 u;
    uint_fast16_t sig;

    sign = (a < 0);
    absA = sign ? -(uint_fast32_t) a : (uint_fast32_t) a;
    shiftDist = softfloat_countLeadingZeros32( absA ) - 21;
    if ( 0 <= shiftDist ) {
        u.ui =
            a ? packToF16UI(
                    sign, 0x18 - shiftDist, (uint_fast16_t) absA<<shiftDist )
                : 0;
        return u.f;
    } else {
        shiftDist += 4;
        sig =
            (shiftDist < 0)
                ? absA>>(-shiftDist)
                      | ((uint32_t) (absA<<(shiftDist & 31)) != 0)
                : (uint_fast16_t) absA<<shiftDist;
        return softfloat_roundPackToF16( sign, 0x1C - shiftDist, sig );
    }
}

int_fast32_t f16_to_fix16( float16_t a ) // fixed point with decimal = mant
{
    bool sign;
    int_fast8_t exp;
    uint_fast16_t frac;
    int_fast32_t sig32;
    int_fast8_t shiftDist;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sign = signF16UI( a.u );
    exp  = expF16UI( a.u );
    frac = fracF16UI( a.u );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sig32 = frac;
    if ( !exp && !frac )
        return 0;

    sig32 |= 0x0400;
    shiftDist = exp - 15;
    if ( 0 <= shiftDist ) {
        sig32 <<= shiftDist;
        return sign ? -sig32 : sig32;
    }
    shiftDist = exp - (15 - 12);
    if ( 0 < shiftDist ) {
        sig32 <<= shiftDist;
    }
    return softfloat_roundToI32(sign, (uint_fast32_t) sig32, softfloat_round_near_even, false);
}

int_fast32_t f16_to_i32( float16_t a, uint_fast8_t roundingMode, bool exact )
{
    union ui16_f16 uA;
    uint_fast16_t uiA;
    bool sign;
    int_fast8_t exp;
    uint_fast16_t frac;
    int_fast32_t sig32;
    int_fast8_t shiftDist;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uA.f = a;
    uiA = uA.ui;
    sign = signF16UI( uiA );
    exp  = expF16UI( uiA );
    frac = fracF16UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( exp == 0x1F ) {
        softfloat_raiseFlags( softfloat_flag_invalid );
        return
            frac ? i32_fromNaN
                : sign ? i32_fromNegOverflow : i32_fromPosOverflow;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sig32 = frac;
    if ( exp ) {
        sig32 |= 0x0400;
        shiftDist = exp - 0x19;
        if ( 0 <= shiftDist ) {
            sig32 <<= shiftDist;
            return sign ? -sig32 : sig32;
        }
        shiftDist = exp - 0x0D;
        if ( 0 < shiftDist ) sig32 <<= shiftDist;
    }
    return
        softfloat_roundToI32(
            sign, (uint_fast32_t) sig32, roundingMode, exact );

}

#if 0
static float32_t
 softfloat_roundPackToF32( bool sign, int_fast16_t exp, uint_fast32_t sig )
{
    uint_fast8_t roundingMode;
    bool roundNearEven;
    uint_fast8_t roundIncrement, roundBits;
    bool isTiny;
    uint_fast32_t uiZ;
    union ui32_f32 uZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    roundingMode = softfloat_roundingMode;
    roundNearEven = (roundingMode == softfloat_round_near_even);
    roundIncrement = 0x40;
    if ( ! roundNearEven && (roundingMode != softfloat_round_near_maxMag) ) {
        roundIncrement =
            (roundingMode
                 == (sign ? softfloat_round_min : softfloat_round_max))
                ? 0x7F
                : 0;
    }
    roundBits = sig & 0x7F;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( 0xFD <= (unsigned int) exp ) {
        if ( exp < 0 ) {
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            isTiny =
                (softfloat_detectTininess == softfloat_tininess_beforeRounding)
                    || (exp < -1) || (sig + roundIncrement < 0x80000000);
            sig = softfloat_shiftRightJam32( sig, -exp );
            exp = 0;
            roundBits = sig & 0x7F;
            if ( isTiny && roundBits ) {
                softfloat_raiseFlags( softfloat_flag_underflow );
            }
        } else if ( (0xFD < exp) || (0x80000000 <= sig + roundIncrement) ) {
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            softfloat_raiseFlags(
                softfloat_flag_overflow | softfloat_flag_inexact );
            uiZ = packToF32UI( sign, 0xFF, 0 ) - ! roundIncrement;
            goto uiZ;
        }
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sig = (sig + roundIncrement)>>7;
    if ( roundBits ) {
        softfloat_exceptionFlags |= softfloat_flag_inexact;
#ifdef SOFTFLOAT_ROUND_ODD
        if ( roundingMode == softfloat_round_odd ) {
            sig |= 1;
            goto packReturn;
        }
#endif
    }
    sig &= ~(uint_fast32_t) (! (roundBits ^ 0x40) & roundNearEven);
    if ( ! sig ) exp = 0;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
#ifdef SOFTFLOAT_ROUND_ODD
 packReturn:
#endif
    uiZ = packToF32UI( sign, exp, sig );
 uiZ:
    uZ.ui = uiZ;
    return uZ.f;

}
#endif // if 0

/*
static float32_t
 softfloat_normRoundPackToF32( bool sign, int_fast16_t exp, uint_fast32_t sig )
{
    int_fast8_t shiftDist;
    union ui32_f32 uZ;

    shiftDist = softfloat_countLeadingZeros32( sig ) - 1;
    exp -= shiftDist;
    if ( (7 <= shiftDist) && ((unsigned int) exp < 0xFD) ) {
        uZ.ui = packToF32UI( sign, sig ? exp : 0, sig<<(shiftDist - 7) );
        return uZ.f;
    } else {
        return softfloat_roundPackToF32( sign, exp, sig<<shiftDist );
    }

}
*/

/*
float32_t i32_to_f32( int32_t a )
{
    bool sign;
    union ui32_f32 uZ;
    uint_fast32_t absA;

    sign = (a < 0);
    if ( ! (a & 0x7FFFFFFF) ) {
        uZ.ui = sign ? packToF32UI( 1, 0x9E, 0 ) : 0;
        return uZ.f;
    }
    absA = sign ? -(uint_fast32_t) a : (uint_fast32_t) a;
    return softfloat_normRoundPackToF32( sign, 0x9C, absA );

}

int_fast32_t f32_to_i32( float32_t a, uint_fast8_t roundingMode, bool exact )
{
    union ui32_f32 uA;
    uint_fast32_t uiA;
    bool sign;
    int_fast16_t exp;
    uint_fast32_t sig;
    uint_fast64_t sig64;
    int_fast16_t shiftDist;

    //------------------------------------------------------------------------
    uA.f = a;
    uiA = uA.ui;
    sign = signF32UI( uiA );
    exp  = expF32UI( uiA );
    sig  = fracF32UI( uiA );
    //------------------------------------------------------------------------
#if (i32_fromNaN != i32_fromPosOverflow) || (i32_fromNaN != i32_fromNegOverflow)
    if ( (exp == 0xFF) && sig ) {
#if (i32_fromNaN == i32_fromPosOverflow)
        sign = 0;
#elif (i32_fromNaN == i32_fromNegOverflow)
        sign = 1;
#else
        softfloat_raiseFlags( softfloat_flag_invalid );
        return i32_fromNaN;
#endif
    }
#endif
    //------------------------------------------------------------------------
    if ( exp ) sig |= 0x00800000;
    sig64 = (uint_fast64_t) sig<<32;
    shiftDist = 0xAA - exp;
    if ( 0 < shiftDist ) sig64 = softfloat_shiftRightJam64( sig64, shiftDist );
    return softfloat_roundToI32( sign, sig64, roundingMode, exact );
}
*/

int_fast32_t
 softfloat_roundToI32(
     bool sign, uint_fast64_t sig, uint_fast8_t roundingMode, bool exact )
{
    bool roundNearEven;
    uint_fast16_t roundIncrement, roundBits;
    uint_fast32_t sig32;
    union { uint32_t ui; int32_t i; } uZ;
    int_fast32_t z;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    roundNearEven = (roundingMode == softfloat_round_near_even);
    roundIncrement = 0x800;
    if ( ! roundNearEven && (roundingMode != softfloat_round_near_maxMag) ) {
        roundIncrement =
            (roundingMode
                 == (sign ? softfloat_round_min : softfloat_round_max))
                ? 0xFFF
                : 0;
    }
    roundBits = sig & 0xFFF;
    sig += roundIncrement;
    if ( sig & UINT64_C( 0xFFFFF00000000000 ) ) goto invalid;
    sig32 = sig>>12;
    sig32 &= ~(uint_fast32_t) (! (roundBits ^ 0x800) & roundNearEven);
    uZ.ui = sign ? -sig32 : sig32;
    z = uZ.i;
    if ( z && ((z < 0) ^ sign) ) goto invalid;
    if ( exact && roundBits ) {
        softfloat_exceptionFlags |= softfloat_flag_inexact;
    }
    return z;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 invalid:
    softfloat_raiseFlags( softfloat_flag_invalid );
    return sign ? i32_fromNegOverflow : i32_fromPosOverflow;

}

float16_t f32_to_f16( float32_t a )
{
    union ui32_f32 uA;
    uint_fast32_t uiA;
    bool sign;
    int_fast16_t exp;
    uint_fast32_t frac;
    struct commonNaN commonNaN;
    uint_fast16_t uiZ, frac16;
    union ui16_f16 uZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uA.f = a;
    uiA = uA.ui;
    sign = signF32UI( uiA );
    exp  = expF32UI( uiA );
    frac = fracF32UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( exp == 0xFF ) {
        if ( frac ) {
            softfloat_f32UIToCommonNaN( uiA, &commonNaN );
            uiZ = softfloat_commonNaNToF16UI( &commonNaN );
        } else {
            uiZ = packToF16UI( sign, 0x1F, 0 );
        }
        goto uiZ;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    frac16 = frac>>9 | ((frac & 0x1FF) != 0);
    if ( ! (exp | frac16) ) {
        uiZ = packToF16UI( sign, 0, 0 );
        goto uiZ;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    return softfloat_roundPackToF16( sign, exp - 0x71, frac16 | 0x4000 );
 uiZ:
    uZ.ui = uiZ;
    return uZ.f;

}

float32_t f16_to_f32( float16_t a )
{
    union ui16_f16 uA;
    uint_fast16_t uiA;
    bool sign;
    int_fast8_t exp;
    uint_fast16_t frac;
    struct commonNaN commonNaN;
    uint_fast32_t uiZ;
    struct exp8_sig16 normExpSig;
    union ui32_f32 uZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uA.f = a;
    uiA = uA.ui;
    sign = signF16UI( uiA );
    exp  = expF16UI( uiA );
    frac = fracF16UI( uiA );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( exp == 0x1F ) {
        if ( frac ) {
            softfloat_f16UIToCommonNaN( uiA, &commonNaN );
            uiZ = softfloat_commonNaNToF32UI( &commonNaN );
        } else {
            uiZ = packToF32UI( sign, 0xFF, 0 );
        }
        goto uiZ;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( ! exp ) {
        if ( ! frac ) {
            uiZ = packToF32UI( sign, 0, 0 );
            goto uiZ;
        }
        normExpSig = softfloat_normSubnormalF16Sig( frac );
        exp = normExpSig.exp - 1;
        frac = normExpSig.sig;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uiZ = packToF32UI( sign, exp + 0x70, (uint_fast32_t) frac<<13 );
 uiZ:
    uZ.ui = uiZ;
    return uZ.f;

}

float16_t f16_mulAdd( float16_t a, float16_t b, float16_t c )
{
    union ui16_f16 uA;
    uint_fast16_t uiA;
    union ui16_f16 uB;
    uint_fast16_t uiB;
    union ui16_f16 uC;
    uint_fast16_t uiC;

    uA.f = a;
    uiA = uA.ui;
    uB.f = b;
    uiB = uB.ui;
    uC.f = c;
    uiC = uC.ui;
    return softfloat_mulAddF16( uiA, uiB, uiC, 0 );
}

float16_t
 softfloat_mulAddF16(
     uint_fast16_t uiA, uint_fast16_t uiB, uint_fast16_t uiC, uint_fast8_t op )
{
    bool signA;
    int_fast8_t expA;
    uint_fast16_t sigA;
    bool signB;
    int_fast8_t expB;
    uint_fast16_t sigB;
    bool signC;
    int_fast8_t expC;
    uint_fast16_t sigC;
    bool signProd;
    uint_fast16_t magBits, uiZ;
    struct exp8_sig16 normExpSig;
    int_fast8_t expProd;
    uint_fast32_t sigProd;
    bool signZ;
    int_fast8_t expZ;
    uint_fast16_t sigZ;
    int_fast8_t expDiff;
    uint_fast32_t sig32Z, sig32C;
    int_fast8_t shiftDist;
    union ui16_f16 uZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    signA = signF16UI( uiA );
    expA  = expF16UI( uiA );
    sigA  = fracF16UI( uiA );
    signB = signF16UI( uiB );
    expB  = expF16UI( uiB );
    sigB  = fracF16UI( uiB );
    signC = signF16UI( uiC ) ^ (op == softfloat_mulAdd_subC);
    expC  = expF16UI( uiC );
    sigC  = fracF16UI( uiC );
    signProd = signA ^ signB ^ (op == softfloat_mulAdd_subProd);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( expA == 0x1F ) {
        if ( sigA || ((expB == 0x1F) && sigB) ) goto propagateNaN_ABC;
        magBits = expB | sigB;
        goto infProdArg;
    }
    if ( expB == 0x1F ) {
        if ( sigB ) goto propagateNaN_ABC;
        magBits = expA | sigA;
        goto infProdArg;
    }
    if ( expC == 0x1F ) {
        if ( sigC ) {
            uiZ = 0;
            goto propagateNaN_ZC;
        }
        uiZ = uiC;
        goto uiZ;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( ! expA ) {
        if ( ! sigA ) goto zeroProd;
        normExpSig = softfloat_normSubnormalF16Sig( sigA );
        expA = normExpSig.exp;
        sigA = normExpSig.sig;
    }
    if ( ! expB ) {
        if ( ! sigB ) goto zeroProd;
        normExpSig = softfloat_normSubnormalF16Sig( sigB );
        expB = normExpSig.exp;
        sigB = normExpSig.sig;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expProd = expA + expB - 0xE;
    sigA = (sigA | 0x0400)<<4;
    sigB = (sigB | 0x0400)<<4;
    sigProd = (uint_fast32_t) sigA * sigB;
    if ( sigProd < 0x20000000 ) {
        --expProd;
        sigProd <<= 1;
    }
    signZ = signProd;
    if ( ! expC ) {
        if ( ! sigC ) {
            expZ = expProd - 1;
            sigZ = sigProd>>15 | ((sigProd & 0x7FFF) != 0);
            goto roundPack;
        }
        normExpSig = softfloat_normSubnormalF16Sig( sigC );
        expC = normExpSig.exp;
        sigC = normExpSig.sig;
    }
    sigC = (sigC | 0x0400)<<3;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expDiff = expProd - expC;
    if ( signProd == signC ) {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        if ( expDiff <= 0 ) {
            expZ = expC;
            sigZ = sigC + softfloat_shiftRightJam32( sigProd, 16 - expDiff );
        } else {
            expZ = expProd;
            sig32Z =
                sigProd
                    + softfloat_shiftRightJam32(
                          (uint_fast32_t) sigC<<16, expDiff );
            sigZ = sig32Z>>16 | ((sig32Z & 0xFFFF) != 0 );
        }
        if ( sigZ < 0x4000 ) {
            --expZ;
            sigZ <<= 1;
        }
    } else {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        sig32C = (uint_fast32_t) sigC<<16;
        if ( expDiff < 0 ) {
            signZ = signC;
            expZ = expC;
            sig32Z = sig32C - softfloat_shiftRightJam32( sigProd, -expDiff );
        } else if ( ! expDiff ) {
            expZ = expProd;
            sig32Z = sigProd - sig32C;
            if ( ! sig32Z ) goto completeCancellation;
            if ( sig32Z & 0x80000000 ) {
                signZ = ! signZ;
                sig32Z = -sig32Z;
            }
        } else {
            expZ = expProd;
            sig32Z = sigProd - softfloat_shiftRightJam32( sig32C, expDiff );
        }
        shiftDist = softfloat_countLeadingZeros32( sig32Z ) - 1;
        expZ -= shiftDist;
        shiftDist -= 16;
        if ( shiftDist < 0 ) {
            sigZ =
                sig32Z>>(-shiftDist)
                    | ((uint32_t) (sig32Z<<(shiftDist & 31)) != 0);
        } else {
            sigZ = (uint_fast16_t) sig32Z<<shiftDist;
        }
    }
 roundPack:
    return softfloat_roundPackToF16( signZ, expZ, sigZ );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN_ABC:
    uiZ = softfloat_propagateNaNF16UI( uiA, uiB );
    goto propagateNaN_ZC;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 infProdArg:
    if ( magBits ) {
        uiZ = packToF16UI( signProd, 0x1F, 0 );
        if ( expC != 0x1F ) goto uiZ;
        if ( sigC ) goto propagateNaN_ZC;
        if ( signProd == signC ) goto uiZ;
    }
    softfloat_raiseFlags( softfloat_flag_invalid );
    uiZ = defaultNaNF16UI;
 propagateNaN_ZC:
    uiZ = softfloat_propagateNaNF16UI( uiZ, uiC );
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 zeroProd:
    uiZ = uiC;
    if ( ! (expC | sigC) && (signProd != signC) ) {
 completeCancellation:
        uiZ =
            packToF16UI(
                (softfloat_roundingMode == softfloat_round_min), 0, 0 );
    }
 uiZ:
    uZ.ui = uiZ;
    return uZ.f;

}

/*
struct exp8_sig16 softfloat_normSubnormalF16Sig( uint_fast16_t sig )
{
    int_fast8_t shiftDist;
    struct exp8_sig16 z;

    shiftDist = softfloat_countLeadingZeros16( sig ) - 5;
    z.exp = 1 - shiftDist;
    z.sig = sig<<shiftDist;
    return z;

}
*/

float16_t f16_mul( float16_t a, float16_t b )
{
    union ui16_f16 uA;
    uint_fast16_t uiA;
    bool signA;
    int_fast8_t expA;
    uint_fast16_t sigA;
    union ui16_f16 uB;
    uint_fast16_t uiB;
    bool signB;
    int_fast8_t expB;
    uint_fast16_t sigB;
    bool signZ;
    uint_fast16_t magBits;
    struct exp8_sig16 normExpSig;
    int_fast8_t expZ;
    uint_fast32_t sig32Z;
    uint_fast16_t sigZ, uiZ;
    union ui16_f16 uZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uA.f = a;
    uiA = uA.ui;
    signA = signF16UI( uiA );
    expA  = expF16UI( uiA );
    sigA  = fracF16UI( uiA );
    uB.f = b;
    uiB = uB.ui;
    signB = signF16UI( uiB );
    expB  = expF16UI( uiB );
    sigB  = fracF16UI( uiB );
    signZ = signA ^ signB;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( expA == 0x1F ) {
        if ( sigA || ((expB == 0x1F) && sigB) ) goto propagateNaN;
        magBits = expB | sigB;
        goto infArg;
    }
    if ( expB == 0x1F ) {
        if ( sigB ) goto propagateNaN;
        magBits = expA | sigA;
        goto infArg;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( ! expA ) {
        if ( ! sigA ) goto zero;
        normExpSig = softfloat_normSubnormalF16Sig( sigA );
        expA = normExpSig.exp;
        sigA = normExpSig.sig;
    }
    if ( ! expB ) {
        if ( ! sigB ) goto zero;
        normExpSig = softfloat_normSubnormalF16Sig( sigB );
        expB = normExpSig.exp;
        sigB = normExpSig.sig;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expZ = expA + expB - 0xF;
    sigA = (sigA | 0x0400)<<4;
    sigB = (sigB | 0x0400)<<5;
    sig32Z = (uint_fast32_t) sigA * sigB;
    sigZ = sig32Z>>16;
    if ( sig32Z & 0xFFFF ) sigZ |= 1;
    if ( sigZ < 0x4000 ) {
        --expZ;
        sigZ <<= 1;
    }
    return softfloat_roundPackToF16( signZ, expZ, sigZ );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN:
    uiZ = softfloat_propagateNaNF16UI( uiA, uiB );
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 infArg:
    if ( ! magBits ) {
        softfloat_raiseFlags( softfloat_flag_invalid );
        uiZ = defaultNaNF16UI;
    } else {
        uiZ = packToF16UI( signZ, 0x1F, 0 );
    }
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 zero:
    uiZ = packToF16UI( signZ, 0, 0 );
 uiZ:
    uZ.ui = uiZ;
    return uZ.f;
}

float16_t f16_div( float16_t a, float16_t b )
{
    union ui16_f16 uA;
    uint_fast16_t uiA;
    bool signA;
    int_fast8_t expA;
    uint_fast16_t sigA;
    union ui16_f16 uB;
    uint_fast16_t uiB;
    bool signB;
    int_fast8_t expB;
    uint_fast16_t sigB;
    bool signZ;
    struct exp8_sig16 normExpSig;
    int_fast8_t expZ;
#ifdef SOFTFLOAT_FAST_DIV32TO16
    uint_fast32_t sig32A;
    uint_fast16_t sigZ;
#else
    int index;
    uint16_t r0;
    uint_fast16_t sigZ, rem;
#endif
    uint_fast16_t uiZ;
    union ui16_f16 uZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    uA.f = a;
    uiA = uA.ui;
    signA = signF16UI( uiA );
    expA  = expF16UI( uiA );
    sigA  = fracF16UI( uiA );
    uB.f = b;
    uiB = uB.ui;
    signB = signF16UI( uiB );
    expB  = expF16UI( uiB );
    sigB  = fracF16UI( uiB );
    signZ = signA ^ signB;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( expA == 0x1F ) {
        if ( sigA ) goto propagateNaN;
        if ( expB == 0x1F ) {
            if ( sigB ) goto propagateNaN;
            goto invalid;
        }
        goto infinity;
    }
    if ( expB == 0x1F ) {
        if ( sigB ) goto propagateNaN;
        goto zero;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    if ( ! expB ) {
        if ( ! sigB ) {
            if ( ! (expA | sigA) ) goto invalid;
            softfloat_raiseFlags( softfloat_flag_infinite );
            goto infinity;
        }
        normExpSig = softfloat_normSubnormalF16Sig( sigB );
        expB = normExpSig.exp;
        sigB = normExpSig.sig;
    }
    if ( ! expA ) {
        if ( ! sigA ) goto zero;
        normExpSig = softfloat_normSubnormalF16Sig( sigA );
        expA = normExpSig.exp;
        sigA = normExpSig.sig;
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expZ = expA - expB + 0xE;
    sigA |= 0x0400;
    sigB |= 0x0400;
#ifdef SOFTFLOAT_FAST_DIV32TO16
    if ( sigA < sigB ) {
        --expZ;
        sig32A = (uint_fast32_t) sigA<<15;
    } else {
        sig32A = (uint_fast32_t) sigA<<14;
    }
    sigZ = sig32A / sigB;
    if ( ! (sigZ & 7) ) sigZ |= ((uint_fast32_t) sigB * sigZ != sig32A);
#else
    if ( sigA < sigB ) {
        --expZ;
        sigA <<= 5;
    } else {
        sigA <<= 4;
    }
    index = sigB>>6 & 0xF;
    r0 = softfloat_approxRecip_1k0s[index]
             - (((uint_fast32_t) softfloat_approxRecip_1k1s[index]
                     * (sigB & 0x3F))
                    >>10);
    sigZ = ((uint_fast32_t) sigA * r0)>>16;
    rem = (sigA<<10) - sigZ * sigB;
    sigZ += (rem * (uint_fast32_t) r0)>>26;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    ++sigZ;
    if ( ! (sigZ & 7) ) {
        sigZ &= ~1;
        rem = (sigA<<10) - sigZ * sigB;
        if ( rem & 0x8000 ) {
            sigZ -= 2;
        } else {
            if ( rem ) sigZ |= 1;
        }
    }
#endif
    return softfloat_roundPackToF16( signZ, expZ, sigZ );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN:
    uiZ = softfloat_propagateNaNF16UI( uiA, uiB );
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 invalid:
    softfloat_raiseFlags( softfloat_flag_invalid );
    uiZ = defaultNaNF16UI;
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 infinity:
    uiZ = packToF16UI( signZ, 0x1F, 0 );
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 zero:
    uiZ = packToF16UI( signZ, 0, 0 );
 uiZ:
    uZ.ui = uiZ;
    return uZ.f;

}

float16_t softfloat_addMagsF16( uint_fast16_t uiA, uint_fast16_t uiB )
{
    int_fast8_t expA;
    uint_fast16_t sigA;
    int_fast8_t expB;
    uint_fast16_t sigB;
    int_fast8_t expDiff;
    uint_fast16_t uiZ;
    bool signZ;
    int_fast8_t expZ;
    uint_fast16_t sigZ;
    uint_fast16_t sigX, sigY;
    int_fast8_t shiftDist;
    uint_fast32_t sig32Z;
    int_fast8_t roundingMode;
    union ui16_f16 uZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expA = expF16UI( uiA );
    sigA = fracF16UI( uiA );
    expB = expF16UI( uiB );
    sigB = fracF16UI( uiB );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expDiff = expA - expB;
    if ( ! expDiff ) {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        if ( ! expA ) { // both subnormals, if carry : exp = 1
            uiZ = uiA + sigB;
            goto uiZ;
        }
        if ( expA == 0x1F ) {
            if ( sigA | sigB ) goto propagateNaN;
            uiZ = uiA;
            goto uiZ;
        }
        signZ = signF16UI( uiA );
        expZ = expA;
        sigZ = 0x0800 + sigA + sigB;
        if ( ! (sigZ & 1) && (expZ < 0x1E) ) {
            sigZ >>= 1;
            goto pack;
        }
        sigZ <<= 3;
    } else {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        signZ = signF16UI( uiA );
        if ( expDiff < 0 ) {
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            if ( expB == 0x1F ) {
                if ( sigB ) goto propagateNaN;
                uiZ = packToF16UI( signZ, 0x1F, 0 );
                goto uiZ;
            }
            if ( expDiff <= -13 ) {
                uiZ = packToF16UI( signZ, expB, sigB );
                if ( expA | sigA ) goto addEpsilon;
                goto uiZ;
            }
            expZ = expB;
            sigX = sigB | 0x0400;
            sigY = sigA + (expA ? 0x0400 : sigA);
            shiftDist = 19 + expDiff;
        } else {
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            uiZ = uiA;
            if ( expA == 0x1F ) {
                if ( sigA ) goto propagateNaN;
                goto uiZ;
            }
            if ( 13 <= expDiff ) {
                if ( expB | sigB ) goto addEpsilon;
                goto uiZ;
            }
            expZ = expA;
            sigX = sigA | 0x0400;
            sigY = sigB + (expB ? 0x0400 : sigB);
            shiftDist = 19 - expDiff;
        }
        sig32Z =
            ((uint_fast32_t) sigX<<19) + ((uint_fast32_t) sigY<<shiftDist);
        if ( sig32Z < 0x40000000 ) {
            --expZ;
            sig32Z <<= 1;
        }
        sigZ = sig32Z>>16;
        if ( sig32Z & 0xFFFF ) {
            sigZ |= 1;
        } else {
            if ( ! (sigZ & 0xF) && (expZ < 0x1E) ) {
                sigZ >>= 4;
                goto pack;
            }
        }
    }
    return softfloat_roundPackToF16( signZ, expZ, sigZ );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN:
    uiZ = softfloat_propagateNaNF16UI( uiA, uiB );
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 addEpsilon:
    roundingMode = softfloat_roundingMode;
    if ( roundingMode != softfloat_round_near_even ) {
        if (
            roundingMode
                == (signF16UI( uiZ ) ? softfloat_round_min
                        : softfloat_round_max)
        ) {
            ++uiZ;
            if ( (uint16_t) (uiZ<<1) == 0xF800 ) {
                softfloat_raiseFlags(
                    softfloat_flag_overflow | softfloat_flag_inexact );
            }
        }
#ifdef SOFTFLOAT_ROUND_ODD
        else if ( roundingMode == softfloat_round_odd ) {
            uiZ |= 1;
        }
#endif
    }
    softfloat_exceptionFlags |= softfloat_flag_inexact;
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 pack:
    uiZ = packToF16UI( signZ, expZ, sigZ );
 uiZ:
    uZ.ui = uiZ;
    return uZ.f;

}

float16_t softfloat_subMagsF16( uint_fast16_t uiA, uint_fast16_t uiB )
{
    int_fast8_t expA;
    uint_fast16_t sigA;
    int_fast8_t expB;
    uint_fast16_t sigB;
    int_fast8_t expDiff;
    uint_fast16_t uiZ;
    int_fast16_t sigDiff;
    bool signZ;
    int_fast8_t shiftDist, expZ;
    uint_fast16_t sigZ, sigX, sigY;
    uint_fast32_t sig32Z;
    int_fast8_t roundingMode;
    union ui16_f16 uZ;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expA = expF16UI( uiA );
    sigA = fracF16UI( uiA );
    expB = expF16UI( uiB );
    sigB = fracF16UI( uiB );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expDiff = expA - expB;
    if ( ! expDiff ) {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        if ( expA == 0x1F ) {
            if ( sigA | sigB ) goto propagateNaN;
            softfloat_raiseFlags( softfloat_flag_invalid );
            uiZ = defaultNaNF16UI;
            goto uiZ;
        }
        sigDiff = sigA - sigB;
        if ( ! sigDiff ) {
            uiZ =
                packToF16UI(
                    (softfloat_roundingMode == softfloat_round_min), 0, 0 );
            goto uiZ;
        }
        if ( expA ) --expA;
        signZ = signF16UI( uiA );
        if ( sigDiff < 0 ) {
            signZ = ! signZ;
            sigDiff = -sigDiff;
        }
        shiftDist = softfloat_countLeadingZeros16( sigDiff ) - 5;
        expZ = expA - shiftDist;
        if ( expZ < 0 ) {
            shiftDist = expA;
            expZ = 0;
        }
        sigZ = sigDiff<<shiftDist;
        goto pack;
    } else {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        signZ = signF16UI( uiA );
        if ( expDiff < 0 ) {
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            signZ = ! signZ;
            if ( expB == 0x1F ) {
                if ( sigB ) goto propagateNaN;
                uiZ = packToF16UI( signZ, 0x1F, 0 );
                goto uiZ;
            }
            if ( expDiff <= -13 ) {
                uiZ = packToF16UI( signZ, expB, sigB );
                if ( expA | sigA ) goto subEpsilon;
                goto uiZ;
            }
            expZ = expA + 19;
            sigX = sigB | 0x0400;
            sigY = sigA + (expA ? 0x0400 : sigA);
            expDiff = -expDiff;
        } else {
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            uiZ = uiA;
            if ( expA == 0x1F ) {
                if ( sigA ) goto propagateNaN;
                goto uiZ;
            }
            if ( 13 <= expDiff ) {
                if ( expB | sigB ) goto subEpsilon;
                goto uiZ;
            }
            expZ = expB + 19;
            sigX = sigA | 0x0400;
            sigY = sigB + (expB ? 0x0400 : sigB);
        }
        sig32Z = ((uint_fast32_t) sigX<<expDiff) - sigY;
        shiftDist = softfloat_countLeadingZeros32( sig32Z ) - 1;
        sig32Z <<= shiftDist;
        expZ -= shiftDist;
        sigZ = sig32Z>>16;
        if ( sig32Z & 0xFFFF ) {
            sigZ |= 1;
        } else {
            if ( ! (sigZ & 0xF) && ((unsigned int) expZ < 0x1E) ) {
                sigZ >>= 4;
                goto pack;
            }
        }
        return softfloat_roundPackToF16( signZ, expZ, sigZ );
    }
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 propagateNaN:
    uiZ = softfloat_propagateNaNF16UI( uiA, uiB );
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 subEpsilon:
    roundingMode = softfloat_roundingMode;
    if ( roundingMode != softfloat_round_near_even ) {
        if (
            (roundingMode == softfloat_round_minMag)
                || (roundingMode
                        == (signF16UI( uiZ ) ? softfloat_round_max
                                : softfloat_round_min))
        ) {
            --uiZ;
        }
#ifdef SOFTFLOAT_ROUND_ODD
        else if ( roundingMode == softfloat_round_odd ) {
            uiZ = (uiZ - 1) | 1;
        }
#endif
    }
    softfloat_exceptionFlags |= softfloat_flag_inexact;
    goto uiZ;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 pack:
    uiZ = packToF16UI( signZ, expZ, sigZ );
 uiZ:
    uZ.ui = uiZ;
    return uZ.f;

}

float16_t f16_add( float16_t a, float16_t b )
{
    union ui16_f16 uA;
    uint_fast16_t uiA;
    union ui16_f16 uB;
    uint_fast16_t uiB;
#if ! defined INLINE_LEVEL || (INLINE_LEVEL < 1)
    float16_t (*magsFuncPtr)( uint_fast16_t, uint_fast16_t );
#endif

    uA.f = a;
    uiA = uA.ui;
    uB.f = b;
    uiB = uB.ui;
#if defined INLINE_LEVEL && (1 <= INLINE_LEVEL)
    if ( signF16UI( uiA ^ uiB ) ) {
        return softfloat_subMagsF16( uiA, uiB );
    } else {
        return softfloat_addMagsF16( uiA, uiB );
    }
#else
    magsFuncPtr =
        signF16UI( uiA ^ uiB ) ? softfloat_subMagsF16 : softfloat_addMagsF16;
    return (*magsFuncPtr)( uiA, uiB );
#endif

}

float16_t f16_sub( float16_t a, float16_t b )
{
    union ui16_f16 uA;
    uint_fast16_t uiA;
    union ui16_f16 uB;
    uint_fast16_t uiB;
#if ! defined INLINE_LEVEL || (INLINE_LEVEL < 1)
    float16_t (*magsFuncPtr)( uint_fast16_t, uint_fast16_t );
#endif

    uA.f = a;
    uiA = uA.ui;
    uB.f = b;
    uiB = uB.ui;
#if defined INLINE_LEVEL && (1 <= INLINE_LEVEL)
    if ( signF16UI( uiA ^ uiB ) ) {
        return softfloat_addMagsF16( uiA, uiB );
    } else {
        return softfloat_subMagsF16( uiA, uiB );
    }
#else
    magsFuncPtr =
        signF16UI( uiA ^ uiB ) ? softfloat_addMagsF16 : softfloat_subMagsF16;
    return (*magsFuncPtr)( uiA, uiB );
#endif

}

//float16_t f16_rem( float16_t, float16_t );
//float16_t f16_sqrt( float16_t );

bool f16_eq( float16_t a, float16_t b )
{
    union ui16_f16 uA;
    uint_fast16_t uiA;
    union ui16_f16 uB;
    uint_fast16_t uiB;

    uA.f = a;
    uiA = uA.ui;
    uB.f = b;
    uiB = uB.ui;
    if ( isNaNF16UI( uiA ) || isNaNF16UI( uiB ) ) {
        if (
            softfloat_isSigNaNF16UI( uiA ) || softfloat_isSigNaNF16UI( uiB )
        ) {
            softfloat_raiseFlags( softfloat_flag_invalid );
        }
        return false;
    }
    return (uiA == uiB) || ! (uint16_t) ((uiA | uiB)<<1);
}

bool f16_le( float16_t a, float16_t b )
{
    union ui16_f16 uA;
    uint_fast16_t uiA;
    union ui16_f16 uB;
    uint_fast16_t uiB;
    bool signA, signB;

    uA.f = a;
    uiA = uA.ui;
    uB.f = b;
    uiB = uB.ui;
    if ( isNaNF16UI( uiA ) || isNaNF16UI( uiB ) ) {
        softfloat_raiseFlags( softfloat_flag_invalid );
        return false;
    }
    signA = signF16UI( uiA );
    signB = signF16UI( uiB );
    return
        (signA != signB) ? signA || ! (uint16_t) ((uiA | uiB)<<1)
            : (uiA == uiB) || (signA ^ (uiA < uiB));

}

bool f16_lt( float16_t a, float16_t b )
{
    union ui16_f16 uA;
    uint_fast16_t uiA;
    union ui16_f16 uB;
    uint_fast16_t uiB;
    bool signA, signB;

    uA.f = a;
    uiA = uA.ui;
    uB.f = b;
    uiB = uB.ui;
    if ( isNaNF16UI( uiA ) || isNaNF16UI( uiB ) ) {
        softfloat_raiseFlags( softfloat_flag_invalid );
        return false;
    }
    signA = signF16UI( uiA );
    signB = signF16UI( uiB );
    return
        (signA != signB) ? signA && ((uint16_t) ((uiA | uiB)<<1) != 0)
            : (uiA != uiB) && (signA ^ (uiA < uiB));

}

const uint_least8_t softfloat_countLeadingZeros8[256] = {
    8, 7, 6, 6, 5, 5, 5, 5, 4, 4, 4, 4, 4, 4, 4, 4,
    3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
};

uint32_t softfloat_shiftRightJam32( uint32_t a, uint_fast16_t dist )
{

    if (dist < 31) {
        return a>>dist | ((uint32_t) (a<<(-dist & 31)) != 0);
    }
    return (a != 0);
}

uint_fast8_t softfloat_countLeadingZeros32( uint32_t a )
{
    uint_fast8_t count;

    count = 0;
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

uint_fast8_t softfloat_countLeadingZeros16( uint16_t a )
{
    uint_fast8_t count;

    count = 8;
    if ( 0x100 <= a ) {
        count = 0;
        a >>= 8;
    }
    count += softfloat_countLeadingZeros8[a];
    return count;

}


