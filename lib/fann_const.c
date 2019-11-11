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
 
*/

// Flexible way do automatically adjust constants when softfloat changes

#include "fann_const.h"

#if (defined SOFTFANN) || (defined FIXEDFANN) || (defined _GCC_ARM_F16_FF) || (defined _GCC_ARM_F16_BP) || (defined _FLOAT_UNION) || (defined _BFLOAT16)
fann_type_ff ff_p200;
fann_type_ff ff_p100;
fann_type_ff ff_p099;
fann_type_ff ff_p098;
fann_type_ff ff_p050;
fann_type_ff ff_p001;
fann_type_ff ff_p1ml;
fann_type_ff ff_p01m;
fann_type_ff ff_0000;
fann_type_ff ff_n1ml;
fann_type_ff ff_n001;
fann_type_ff ff_n098;
fann_type_ff ff_n100;
fann_type_ff ff_n200;
#ifndef FANN_INFERENCE_ONLY
fann_type_bp bp_0000;
#endif
#endif // SOFTFANN

#ifdef STEPWISE_LUT
fann_type_ff ff_v1;
fann_type_ff ff_v2;
fann_type_ff ff_v3;
fann_type_ff ff_v4;
fann_type_ff ff_v5;
fann_type_ff ff_v6;
fann_type_ff ff_r1_sigsym;
fann_type_ff ff_r2_sigsym;
fann_type_ff ff_r3_sigsym;
fann_type_ff ff_r4_sigsym;
fann_type_ff ff_r5_sigsym;
fann_type_ff ff_r6_sigsym;
fann_type_ff ff_r1_sig;
fann_type_ff ff_r2_sig;
fann_type_ff ff_r3_sig;
fann_type_ff ff_r4_sig;
fann_type_ff ff_r5_sig;
fann_type_ff ff_r6_sig;
#endif // STEPWISE_LUT

void fann_const_init(void)
{
#if (defined SOFTFANN) || (defined FIXEDFANN) || (defined _GCC_ARM_F16_FF) || (defined _GCC_ARM_F16_BP) || (defined _FLOAT_UNION) || (defined _BFLOAT16)
    // FIXME: use limits for each data type
    // general useful and frequent numbers
#if (defined SWF16_AP) || (defined HWF16)
    FP_BIAS = FP_BIAS_DEFAULT;
#endif

    ff_p200 = fann_int_to_ff(CI_TWO);
    ff_p100 = fann_int_to_ff(CI_ONE);
    ff_p099 = fann_float_to_ff(CF_099);
    ff_p098 = fann_float_to_ff(CF_098);
    ff_p050 = fann_float_to_ff(CF_05);
    ff_p001 = fann_float_to_ff(CF_001);
    ff_p1ml = fann_float_to_ff(CF_0001);
    ff_p01m = fann_float_to_ff(CF_00001);
    ff_0000 = fann_int_to_ff(CI_ZERO);
    ff_n1ml = fann_float_to_ff(-CF_0001);
    ff_n001 = fann_float_to_ff(-CF_001);
    ff_n098 = fann_float_to_ff(-CF_098);
    ff_n100 = fann_int_to_ff(-CI_ONE);
    ff_n200 = fann_int_to_ff(-CI_TWO);

#ifndef FANN_INFERENCE_ONLY
    bp_0000 = fann_ff_to_bp(fann_int_to_ff(0));
#endif
#endif // SOFTFANN ....

#ifdef STEPWISE_LUT
    // default stepwise parameters
    ff_v1 = fann_float_to_ff(-2.64665293693542480469e+00);
    ff_v2 = fann_float_to_ff(-1.47221934795379638672e+00);
    ff_v3 = fann_float_to_ff(-5.49306154251098632812e-01);
    ff_v4 = fann_float_to_ff(5.49306154251098632812e-01);
    ff_v5 = fann_float_to_ff(1.47221934795379638672e+00);
    ff_v6 = fann_float_to_ff(2.64665293693542480469e+00);
    ff_r1_sigsym = fann_float_to_ff(-9.90000009536743164062e-01);
    ff_r2_sigsym = fann_float_to_ff(-8.99999976158142089844e-01);
    ff_r3_sigsym = fann_float_to_ff(-5.00000000000000000000e-01);
    ff_r4_sigsym = fann_float_to_ff(5.00000000000000000000e-01);
    ff_r5_sigsym = fann_float_to_ff(8.99999976158142089844e-01);
    ff_r6_sigsym = fann_float_to_ff(9.90000009536743164062e-01);
    ff_r1_sig = fann_float_to_ff(4.99999988824129104614e-03);
    ff_r2_sig = fann_float_to_ff(5.00000007450580596924e-02);
    ff_r3_sig = fann_float_to_ff(2.50000000000000000000e-01);
    ff_r4_sig = fann_float_to_ff(7.50000000000000000000e-01);
    ff_r5_sig = fann_float_to_ff(9.49999988079071044922e-01);
    ff_r6_sig = fann_float_to_ff(9.95000004768371582031e-01);
#endif // STEPWISE_LUT
}

