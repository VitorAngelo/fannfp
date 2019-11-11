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

  This library was based on the Fast Artificial Neural Network Library.
  See README.md for details.

  Although heavily modified, the starting point for this file was the
  file fann_ieee_f16.c. Please see the header of that file for credits.

*/

#include "fann_ap_f16.h"

volatile int_fast8_t FP_BIAS = FP_BIAS_DEFAULT;

#ifdef FANN_AP_INCLUDE_ZERO
unsigned int fann_ap_cancel;
#endif
unsigned int fann_ap_underflow;
unsigned int fann_ap_overflow;

void fann_ap_reset_stats(void)
{
#ifdef FANN_AP_INCLUDE_ZERO
    fann_ap_cancel = 0;
#endif
    fann_ap_underflow = 0;
    fann_ap_overflow = 0;
}

#ifdef FANN_AP_F16_DEBUG
int fann_ap_debug = 0;
void fann_ap_f16_debug(void)
{
    fann_ap_debug = 1;
}
#include <stdio.h>
#define dprintf(format, ...) if (fann_ap_debug) fprintf (stderr, "%s %d: " format, __FUNCTION__, __LINE__, __VA_ARGS__)
#else
#define dprintf(format, ...)
#endif

#define SOFTFLOAT_FAST_DIV32TO16
//#undef SOFTFLOAT_FAST_DIV32TO16

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

const int_fast8_t softfloat_countLeadingZeros8[256] = {
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

// sig CXMM MMMM MMMM EEEE
// sig is shifted << 4 and the MSbit X, if present, is the one to be ommited
// MSbit C is handled as a carry: exp adjustment (--) is required
// exp is biased, but may be negative (demanding ajustments)
float16_t softfloat_roundPackToF16( bool sign, int_fast16_t exp, uint_fast16_t sig )
{
    float16_t ret;
    const uint_fast8_t roundIncrement = 0x8;
    //uint_fast8_t roundBits;
    int_fast8_t shiftBits;
    dprintf("(%d %d %04X)\n", (int)sign, (int)exp, (unsigned int)sig);
    
    // adjust smaller sigs to put 1 at X
    //shiftBits = softfloat_countLeadingZeros16(sig) - 1;
    if ( 0x100 > sig ) {
        shiftBits = 7 + softfloat_countLeadingZeros8[sig];
    } else {
        shiftBits = softfloat_countLeadingZeros8[sig >> 8] - 1;
    }
    dprintf("%d %d %04X\n", (int)shiftBits, (int)exp, (unsigned int)sig);

    if (shiftBits > 0) {
        if (exp >= shiftBits) {
            sig <<= shiftBits;
            exp -= shiftBits;
        } else {
            goto smallest;
        }
    } else if (shiftBits < 0) { //(sig & 0x8000) {
        if (exp < 0x1F) {
            sig >>= 1;
            exp++;
        } else {
            goto largest;
        }
    }
#if 1
    // round and recheck overflow
    //roundBits = sig & 0xF;
    sig = (sig + roundIncrement)>>4;
    dprintf("%04X\n", (unsigned int)sig);
    //sig &= ~(uint_fast16_t) (! (roundBits ^ 8) & 1);
    //dprintf("%04X\n", (unsigned int)sig);
    if (sig & 0x0800) {
        exp++; // may overflow, "infinity" below
        sig >>= 1;
    }
#else
    sig >>= 4;
#endif
    dprintf("(%d %d %04X)\n", (int)sign, (int)exp, (unsigned int)sig);
    if (exp > 0x1F) {
largest:
        fann_ap_overflow++;
        ret.u = packToF16UI( sign, 0x1F, 0x03FF );
        return ret;
    }
    if ((exp < 0) || ((sig & 0x0400) == 0)) {
smallest:
        fann_ap_underflow++;
        ret.u = packToF16UI( sign, 0, 0 );
        return ret;
    }
    ret.u = packToF16UI( sign, exp, (sig & 0x3FF) );
    return ret;
}

float16_t i32_to_f16( int32_t a )
{
    float16_t ret;
    bool sign;
    uint_fast32_t absA;
    int_fast8_t shiftDist;
    uint_fast16_t sig;

    if (a < 0) {
        sign = 1;
        absA = (uint_fast32_t) (-a);
    } else if (a > 0) {
        sign = 0;
        absA = (uint_fast32_t) (a);
    } else {
        ret.u = 0;
        return ret;
    }
    shiftDist = softfloat_countLeadingZeros32( absA ) - 21;
    dprintf("shiftDist=%d abs=%08X\n", shiftDist, (unsigned int)absA);
    if ( 0 <= shiftDist ) { // [0, 11]
        // 0x19 = 25
        uint_fast8_t exp = (FP_BIAS + 10) - shiftDist;
        if (exp < 32) {
            ret.u = packToF16UI(sign, exp,
                    (uint_fast16_t) ((absA<<shiftDist) & 0x3FF) );
        } else {
            fann_ap_overflow++;
            ret.u = packToF16UI(sign, 31, 0x3FF);
        }
        return ret;
    } // [-21, -1]
    shiftDist += 4;
    dprintf("shiftDist=%d\n", shiftDist);
    if (shiftDist < 0) {
        sig = (absA>>(-shiftDist)) | ((uint32_t) (absA<<(shiftDist & 31)) != 0);
    } else {
        sig = (uint_fast16_t) (absA<<shiftDist);
    } // 0x1D = 29
    return softfloat_roundPackToF16( sign, (FP_BIAS + 14) - shiftDist, sig );
}

int_fast32_t f16_to_i32( float16_t a )
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
#ifdef FANN_AP_INCLUDE_ZERO
    if ( !exp && !frac )
        return 0;
#endif

    sig32 |= 0x0400;
    if (FP_BIAS < 22) {
        shiftDist = exp - (FP_BIAS + 10);
        dprintf("shiftDist=%d\n", shiftDist);
        if ( 0 <= shiftDist ) {
            sig32 <<= shiftDist;
            return sign ? -sig32 : sig32;
        }
    }
    shiftDist = exp - (FP_BIAS - 2);
    dprintf("shiftDist=%d\n", shiftDist);
    if ( 0 < shiftDist ) {
        sig32 <<= shiftDist;
    }
    return softfloat_roundToI32(sign, (uint_fast32_t) sig32);
}

int_fast32_t
 softfloat_roundToI32(
     bool sign, uint_fast64_t sig )
{
    uint_fast16_t roundIncrement, roundBits;
    uint_fast32_t sig32;
    union { uint32_t ui; int32_t i; } uZ;
    int_fast32_t z;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    roundIncrement = 0x800;
    roundBits = sig & 0xFFF;
    sig += roundIncrement;
    sig32 = sig>>12;
    sig32 &= ~(uint_fast32_t) (! (roundBits ^ 0x800) & 1);
    uZ.ui = sign ? -sig32 : sig32;
    z = uZ.i;
    return z;
}

int_fast32_t f16_to_fix16( float16_t a )
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
#ifdef FANN_AP_INCLUDE_ZERO
    if ( !exp && !frac )
        return 0;
#endif

    sig32 |= 0x0400;
    shiftDist = exp - FP_BIAS;
    dprintf("shiftDist=%d\n", shiftDist);
    if ( 0 <= shiftDist ) {
        sig32 <<= shiftDist;
        return sign ? -sig32 : sig32;
    }
    shiftDist = exp - (FP_BIAS - 12);
    dprintf("shiftDist=%d\n", shiftDist);
    if ( 0 < shiftDist ) {
        sig32 <<= shiftDist;
    }
    return softfloat_roundToI32(sign, (uint_fast32_t) sig32);
}

float16_t f32_to_f16( float32_t a )
{
    bool sign;
    int_fast16_t exp;
    uint_fast32_t frac;
    uint_fast16_t frac16;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sign = signF32UI( a.u );
    exp  = expF32UI( a.u );
    frac = fracF32UI( a.u );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    frac16 = frac>>9 | ((frac & 0x1FF) != 0); // 13 - 4 = 9
#ifdef FANN_AP_INCLUDE_ZERO
    if ( ! (exp | frac16) ) {
        float16_t ret;
        ret.u = packToF16UI( sign, 0, 0 );
        return ret;
    }
#endif
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    // 0x70 = 127 - 15
    return softfloat_roundPackToF16( sign, exp - (127 - FP_BIAS), frac16 | 0x4000 );
}

float32_t f16_to_f32( float16_t a )
{
    uint_fast32_t sign, expn, frac;
    float32_t ret;

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    sign = signF16UI( a.u );
    expn = expF16UI( a.u );
    frac = fracF16UI( a.u );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
#ifdef FANN_AP_INCLUDE_ZERO
    if ( ! expn ) {
        if ( ! frac ) {
            ret.u = packToF32UI( sign, 0, 0 );
            return ret;
        }
    }
#endif
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    // 0x70 = 127 - 15
    ret.u = packToF32UI( sign, expn + (127 - FP_BIAS), (uint_fast32_t) frac<<13 );
    return ret;
}

float16_t f16_mulAdd( float16_t a, float16_t b, float16_t c )
{
#define bitFP 10
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
    int_fast8_t expProd;
    //uint_fast32_t sigProd;
    bool signZ;
    int_fast8_t expZ;
    uint_fast16_t sigZ;
    int_fast8_t expDiff;
    uint_fast32_t sig32Z, sig32C;
    int_fast8_t shiftDist;

    dprintf("(%04X %04X %04X)\n", (unsigned int)a.u, (unsigned int)b.u, (unsigned int)c.u);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    signA = signF16UI( a.u );
    expA  = expF16UI( a.u );
    sigA  = fracF16UI( a.u );
    signB = signF16UI( b.u );
    expB  = expF16UI( b.u );
    sigB  = fracF16UI( b.u );
    signC = signF16UI( c.u );// ^ (op == softfloat_mulAdd_subC);
    expC  = expF16UI( c.u );
    sigC  = fracF16UI( c.u );
    signProd = signA ^ signB;// ^ (op == softfloat_mulAdd_subProd);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
#ifdef FANN_AP_INCLUDE_ZERO
    if ( (expA == 0) && (sigA == 0) ) return c;
    if ( (expB == 0) && (sigB == 0) ) return c;
#endif
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expProd = expA + expB - FP_BIAS;
    sigA = sigA | 0x0400;
    sigB = sigB | 0x0400;
    sig32Z = (uint_fast32_t)sigA * (uint_fast32_t)sigB;
    signZ = signProd;
    expZ = expProd;
    if ( !(expC | sigC) ) {
        dprintf("%d\n", (int)expZ);
        goto roundPack;
    }
    dprintf("%08X\n", (unsigned int)sig32Z);
    sig32C = ((uint_fast32_t)sigC | 0x00000400) << bitFP;
    dprintf("%08X %08X\n", (unsigned int)sig32Z,  (unsigned int)sig32C);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expDiff = expProd - expC;
    dprintf("%d\n", (int)expDiff);
    if ( expDiff == 0 ) {
        if ( signProd == signC ) {
            dprintf("exp==,sign== %d %08X %08X\n", (int)expZ, (unsigned int)sig32Z,  (unsigned int)sig32C);
            sig32Z += sig32C;
        } else {
            dprintf("exp==,sign!=%d %08X %08X\n", (int)expZ, (unsigned int)sig32Z,  (unsigned int)sig32C);
            if (sig32Z < sig32C) {
                signZ = ! signZ;
                sig32Z = sig32C - sig32Z;
            } else {
                sig32Z -= sig32C;
            }
            if (!sig32Z) {
                float16_t ret;
                ret.u = 0;
                return ret;
            }
        }
    } else {
        dprintf("expDiff=%d\n", (int)expDiff);
        if (expDiff < -16) { // expC >> expProd
            dprintf("ProdCancel %d %08X\n", (int)expZ, (unsigned int)sig32Z);
#ifdef FANN_AP_INCLUDE_ZERO
            if (expZ | sig32Z) {
            	fann_ap_cancel++;
            }
#endif
            expZ = expC;
            sig32Z = sig32C;
            signZ = signC;
            goto roundPack;
        } else if (expDiff > 16) { // expProd >> expC
            dprintf("AddCancel %d %08X\n", (int)expC, (unsigned int)sig32C);
#ifdef FANN_AP_INCLUDE_ZERO
            if (expC | sigC) {
                fann_ap_cancel++;
            }
#endif
            goto roundPack;
        }
        if (sig32Z > sig32C) {
            shiftDist = softfloat_countLeadingZeros32(sig32Z) - 1;
        } else {
            shiftDist = softfloat_countLeadingZeros32(sig32C) - 1;
        }
        dprintf("shiftDist=%d %08X %08X\n", (int)shiftDist, (unsigned int)sig32Z,  (unsigned int)sig32C);
        sig32Z <<= shiftDist;
        sig32C <<= shiftDist;
        dprintf("%08X %08X\n", (unsigned int)sig32Z,  (unsigned int)sig32C);
        if ( expDiff < 0 ) { // expC > expProd
            expZ = expC;
            sig32Z >>= -expDiff;
            dprintf("AddDom %08X >> %d\n", (unsigned int)sig32Z, (int)-expDiff);
        } else { // expProd > expC
            sig32C >>= expDiff;
            dprintf("ProdDom %08X >> %d\n", (unsigned int)sig32C, (int)expDiff);
        }
        if ( signProd == signC ) { // addMags
            dprintf("sign== %08X %08X\n", (unsigned int)sig32Z, (unsigned int)sig32C);
            sig32Z += sig32C;
            dprintf("%08X\n", (unsigned int)sig32Z);
        } else { // subMags
            if (sig32Z > sig32C) {
                dprintf("sign!= %08X -= %08X\n", (unsigned int)sig32Z, (unsigned int)sig32C);
                sig32Z = sig32Z - sig32C;
            } else {
                signZ = signC;
                dprintf("sign!= Z = %08X - %08X\n", (unsigned int)sig32C, (unsigned int)sig32Z);
                sig32Z = sig32C - sig32Z;
            }
        }
        //dprintf("%d %08X\n", (int)expZ, (unsigned int)sig32Z);
        //sig32Z >>= shiftDist;
        expZ -= shiftDist;
        dprintf("%d %08X\n", (int)expZ, (unsigned int)sig32Z);
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
    }
 roundPack:
    shiftDist = 17 - softfloat_countLeadingZeros32(sig32Z);
    dprintf("roundPack %08X %d\n", (unsigned int)sig32Z, (int)shiftDist);
    if (shiftDist > 0) {
        sigZ = sig32Z >> shiftDist;
        expZ += shiftDist - (bitFP - 4);
        dprintf("%04X %d %d\n", (unsigned int)sigZ, (int)expZ, (int)shiftDist);
        if (sig32Z & ((1 << shiftDist) - 1)) {
            sigZ |= 1;
        }
    } else {
        expZ -= bitFP - 4;
        sigZ = sig32Z;
    }
    return softfloat_roundPackToF16( signZ, expZ, sigZ );
}

#ifndef F16_JUST_MUL_ADD
float16_t f16_mul( float16_t a, float16_t b )
{
    bool signA;
    int_fast8_t expA;
    uint_fast16_t sigA;
    bool signB;
    int_fast8_t expB;
    uint_fast16_t sigB;
    bool signZ;
    int_fast8_t expZ;
    uint_fast32_t sig32Z;
    uint_fast16_t sigZ;

    dprintf("(%04X %04X)\n", (unsigned int)a.u, (unsigned int)b.u);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    signA = signF16UI( a.u );
    expA  = expF16UI( a.u );
    sigA  = fracF16UI( a.u );
    signB = signF16UI( b.u );
    expB  = expF16UI( b.u );
    sigB  = fracF16UI( b.u );
    signZ = signA ^ signB;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
#ifdef FANN_AP_INCLUDE_ZERO
    if ( ((expA == 0) && (sigA == 0)) ||
         ((expB == 0) && (sigB == 0)) ) {
        float16_t ret;
        ret.u = packToF16UI( signZ, 0, 0 );
        return ret;
    }
#endif
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expZ = expA + expB - FP_BIAS;
    sigA = sigA | 0x0400;
    sigB = sigB | 0x0400;
    sig32Z = (uint_fast32_t)sigA * (uint_fast32_t)sigB;
#if 0
    if (sig32Z & 0x00200000) {
        sigZ = sig32Z >> 6; // 10 - 4
        if ( sig32Z & 0x3F ) {
            sigZ |= 1;
        }
    } else {
        expZ--;
        sigZ = sig32Z >> 5;
        if ( sig32Z & 0x1F ) {
            sigZ |= 1;
        }
    }
#else
    int_fast8_t shiftDist;
    dprintf("%08X\n", (unsigned int)sig32Z);
    shiftDist = 17 - softfloat_countLeadingZeros32(sig32Z);
    if (shiftDist > 0) {
        sigZ = sig32Z >> shiftDist;
        expZ += shiftDist - (10 - 4);
        dprintf("%04X %d %d\n", (unsigned int)sigZ, (int)expZ, (int)shiftDist);
        if (sig32Z & ((1 << shiftDist) - 1)) {
            sigZ |= 1;
        }
    } else {
        sigZ = sig32Z;
    }
#endif
    return softfloat_roundPackToF16( signZ, expZ, sigZ );
}
#endif

float16_t f16_div( float16_t a, float16_t b )
{
    bool signA;
    int_fast8_t expA;
    uint_fast16_t sigA;
    bool signB;
    int_fast8_t expB;
    uint_fast16_t sigB;
    bool signZ;
    int_fast8_t expZ;
#ifdef SOFTFLOAT_FAST_DIV32TO16
    uint_fast32_t sig32A;
    uint_fast16_t sigZ;
#else
    int index;
    uint16_t r0;
    uint_fast16_t sigZ, rem;
#endif

    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    signA = signF16UI( a.u );
    expA  = expF16UI( a.u );
    sigA  = fracF16UI( a.u );
    signB = signF16UI( b.u );
    expB  = expF16UI( b.u );
    sigB  = fracF16UI( b.u );
    signZ = signA ^ signB;
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
#ifdef FANN_AP_INCLUDE_ZERO
    if ( (expB == 0) && (sigB == 0) ) {
        float16_t ret;
        //fann_ap_overflow++;
        ret.u = packToF16UI( signZ, 0x1F, 0x03FF );
        return ret;
    }
    if ( (expA == 0) && (sigA == 0) ) {
        float16_t ret;
        ret.u = packToF16UI( signZ, 0, 0 );
        return ret;
    }
#endif
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expZ = expA - expB + FP_BIAS;
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
}

#ifndef F16_JUST_MUL_ADD
static float16_t softfloat_addMagsF16( uint_fast16_t uiA, uint_fast16_t uiB )
{
    int_fast8_t expA;
    uint_fast16_t sigA;
    int_fast8_t expB;
    uint_fast16_t sigB;
    int_fast8_t expDiff;
    bool signZ;
    int_fast8_t expZ;
    uint_fast16_t sigZ;
    uint_fast16_t sigX, sigY;
    int_fast8_t shiftDist;
    uint_fast32_t sig32Z;

    dprintf("(%04X %04X)\n", (unsigned int)uiA, (unsigned int)uiB);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expA = expF16UI( uiA );
    sigA = fracF16UI( uiA );
    expB = expF16UI( uiB );
    sigB = fracF16UI( uiB );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    signZ = signF16UI( uiA );
    expDiff = expA - expB;
    if ( ! expDiff ) {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        expZ = expA;
        if (expA) {
            sigA |= 0x0800;
        } else {
            if (sigA)
                sigA |= 0x0400;
            if (sigB)
                sigB |= 0x0400;
        }
        sigZ = sigA + sigB;
        dprintf("%d %04X\n", (int)expZ, (unsigned int)sigZ);
        sigZ <<= 4;
        dprintf("%04X\n", (unsigned int)sigZ);
    } else {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        dprintf("%d %04X %04X\n", (int)expDiff, (unsigned int)sigA, (unsigned int)sigB);
        if ( expDiff < 0 ) { // expA < expB
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            if ( (expDiff <= -13) || ((expA == 0) && (sigA == 0)) ) {
                float16_t ret;
#ifdef FANN_AP_INCLUDE_ZERO
                if (expA | sigA) {
                    fann_ap_cancel++;
                }
#endif
                ret.u = packToF16UI( signZ, expB, sigB );
                return ret;
            }
            expZ = expB;
            sigX = sigB | 0x0400;
            sigY = sigA | 0x0400;
            shiftDist = 19 + expDiff;
        } else { // expA > expB
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            if ( (expDiff >= 13) || ((expB == 0) && (sigB == 0)) ) {
                float16_t ret;
#ifdef FANN_AP_INCLUDE_ZERO
                if (expB | sigB) {
                    fann_ap_cancel++;
                }
#endif
                ret.u = uiA;
                return ret;
            }
            expZ = expA;
            sigX = sigA | 0x0400;
            sigY = sigB | 0x0400;
            shiftDist = 19 - expDiff;
        } // expZ / sigX are the larger ones:
        dprintf("%d %04X %04X\n", (int)shiftDist, (unsigned int)sigX, (unsigned int)sigY);
        sig32Z = ((uint_fast32_t) sigX<<19) + ((uint_fast32_t) sigY<<shiftDist);
        sigZ = sig32Z>>15; // equiv. << 4, ready to softfloat_roundPackToF16
        dprintf("%04X %04X\n", (unsigned int)sig32Z, (unsigned int)sigZ);
    }
    return softfloat_roundPackToF16( signZ, expZ, sigZ );
}

static float16_t softfloat_subMagsF16( uint_fast16_t uiA, uint_fast16_t uiB )
{
    float16_t ret;
    int_fast8_t expA;
    uint_fast16_t sigA;
    int_fast8_t expB;
    uint_fast16_t sigB;
    int_fast8_t expDiff;
    int_fast16_t sigDiff;
    bool signZ;
    int_fast8_t shiftDist, expZ;
    uint_fast16_t sigZ, sigX, sigY;
    uint_fast32_t sig32Z;

    dprintf("(%04X %04X)\n", (unsigned int)uiA, (unsigned int)uiB);
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    expA = expF16UI( uiA );
    sigA = fracF16UI( uiA );
    expB = expF16UI( uiB );
    sigB = fracF16UI( uiB );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
    signZ = signF16UI( uiA );
    expDiff = expA - expB;
    if ( ! expDiff ) {
        if (sigA || expA)
            sigA |= 0x0400;
        if (sigB || expB)
            sigB |= 0x0400;
        sigDiff = sigA - sigB;
        if ( ! sigDiff ) {
            ret.u = 0;
            return ret;
        }
        dprintf("%d\n", (int)sigDiff);
        if ( sigDiff < 0 ) {
            signZ = ! signZ;
            sigDiff = -sigDiff;
        }
        /*shiftDist = softfloat_countLeadingZeros16( sigDiff ) - 5;
        //if (shiftDist < 0) printf("%s BUG at %d\n", __FUNCTION__, __LINE__);
        dprintf("%04X %d %d\n", (unsigned int)sigDiff, (int)expA, (int)shiftDist);
        expZ = expA - shiftDist;
        dprintf("%d\n", (int)expZ);
        if (expZ < 0) {
            sigZ = 0;
            expZ = 0;
        } else if (shiftDist > 0) {
            sigZ = (sigDiff << shiftDist);
            dprintf("%04X\n", (unsigned int)sigZ);
        } else {
            sigZ = sigDiff;
        }
        sigZ &= 0x03FF;
        goto pack;*/
        expZ = expA; // this is slower but shares code
        sigZ = sigDiff << 4;
        dprintf("%d %04X\n", (int)expZ, (unsigned int)sigZ);
        //return softfloat_roundPackToF16( signZ, expZ, sigZ );
    } else {
        /*--------------------------------------------------------------------
        *--------------------------------------------------------------------*/
        dprintf("%d %04X %04X\n", (int)expDiff, (unsigned int)sigA, (unsigned int)sigB);
        if ( expDiff < 0 ) { // expB > expA
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            signZ = ! signZ;
            if ( (expDiff <= -13) || ((expA == 0) && (sigA == 0)) ) {
                float16_t ret;
#ifdef FANN_AP_INCLUDE_ZERO
                if (sigA | expA) {
                    fann_ap_cancel++;
                }
#endif
                ret.u = packToF16UI( signZ, expB, sigB );
                return ret;
            }
            expZ = expA + 19;
            sigX = sigB | 0x0400;
            sigY = sigA | 0x0400;
            expDiff = -expDiff;
        } else { // expB < expA
            /*----------------------------------------------------------------
            *----------------------------------------------------------------*/
            if ( (expDiff >= 13) || ((expB == 0) && (sigB == 0)) ) {
                float16_t ret;
#ifdef FANN_AP_INCLUDE_ZERO
                if (sigB | expB) {
                    fann_ap_cancel++;
                }
#endif
                ret.u = uiA;
                return ret;
            }
            expZ = expB + 19;
            sigX = sigA | 0x0400;
            sigY = sigB | 0x0400;
        } // expZ / sigX are the larger ones:
        dprintf("%d %d %04X\n", (int)expZ, (unsigned int)sigX, (unsigned int)sigY);
        sig32Z = ((uint_fast32_t) sigX<<expDiff) - sigY;
        shiftDist = softfloat_countLeadingZeros32( sig32Z ) - 1;
        dprintf("%08x %d\n", (unsigned int)sig32Z, (int)shiftDist); 
        sig32Z <<= shiftDist;
        expZ -= shiftDist - 1;
        sigZ = sig32Z>>16;
        if (sig32Z & 0xFFFF)
            sigZ |= 1;
        dprintf("%08x %04x %d\n", (unsigned int)sig32Z, (unsigned int)sigZ, (int)expZ); 
    }
        return softfloat_roundPackToF16( signZ, expZ, sigZ );
    /*------------------------------------------------------------------------
    *------------------------------------------------------------------------*/
 /*pack:
    dprintf("%d %d %04X\n", (int)signZ, (int)expZ, (unsigned int)sigZ);
    ret.u = packToF16UI( signZ, expZ, sigZ );
    dprintf("%04X\n", (unsigned int)ret.u);
    return ret;*/
}

float16_t f16_add( float16_t a, float16_t b )
{
#if ! defined INLINE_LEVEL || (INLINE_LEVEL < 1)
    float16_t (*magsFuncPtr)( uint_fast16_t, uint_fast16_t );
#endif

#if defined INLINE_LEVEL && (1 <= INLINE_LEVEL)
    if ( signF16UI( a.u ^ b.u ) ) {
        return softfloat_subMagsF16( a.u, b.u );
    } else {
        return softfloat_addMagsF16( a.u, b.u );
    }
#else
    magsFuncPtr =
        signF16UI( a ^ b ) ? softfloat_subMagsF16 : softfloat_addMagsF16;
    return (*magsFuncPtr)( a.u, b.u );
#endif
}

float16_t f16_sub( float16_t a, float16_t b )
{
#if ! defined INLINE_LEVEL || (INLINE_LEVEL < 1)
    float16_t (*magsFuncPtr)( uint_fast16_t, uint_fast16_t );
#endif

#if defined INLINE_LEVEL && (1 <= INLINE_LEVEL)
    if ( signF16UI( a.u ^ b.u ) ) {
        return softfloat_addMagsF16( a.u, b.u );
    } else {
        return softfloat_subMagsF16( a.u, b.u );
    }
#else
    magsFuncPtr =
        signF16UI( a ^ b ) ? softfloat_addMagsF16 : softfloat_subMagsF16;
    return (*magsFuncPtr)( a.u, b.u );
#endif
}
#endif // F16_JUST_MUL_ADD

//float16_t f16_rem( float16_t, float16_t );
//float16_t f16_sqrt( float16_t );

bool f16_eq( float16_t a, float16_t b )
{
    return (a.u == b.u) || ! (uint16_t) ((a.u | b.u)<<1);
}


bool f16_le( float16_t a, float16_t b )
{
    bool signA, signB;

    signA = signF16UI( a.u );
    signB = signF16UI( b.u );
    return
        (signA != signB) ? signA || ! (uint16_t) ((a.u | b.u)<<1)
            : (a.u == b.u) || (signA ^ (a.u < b.u));
}

bool f16_lt( float16_t a, float16_t b )
{
    bool signA, signB;

    signA = signF16UI( a.u );
    signB = signF16UI( b.u );
    return
        (signA != signB) ? signA && ((uint16_t) ((a.u | b.u)<<1) != 0)
            : (a.u != b.u) && (signA ^ (a.u < b.u));
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

/*
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
*/

