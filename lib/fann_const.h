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


#ifndef _fann_const_h
#define _fann_const_h

void fann_const_init(void);

#define CI_TWO        (2)
#define CI_ONE        (1)
#define CI_ZERO       (0)

#define CF_099     (0.99)
#define CF_098     (0.98)
#define CF_05       (0.5)
#define CF_01       (0.1)
#define CF_001     (0.01)
#define CF_0001   (0.001)
#define CF_00001 (0.0001)

#if (defined SOFTFANN) || (defined FIXEDFANN) || (defined _GCC_ARM_F16_FF) || (defined _GCC_ARM_F16_BP) || (defined _FLOAT_UNION) || (defined _BFLOAT16)

extern fann_type_ff ff_p200;
extern fann_type_ff ff_p100;
extern fann_type_ff ff_p099;
extern fann_type_ff ff_p098;
extern fann_type_ff ff_p050;
extern fann_type_ff ff_p001;
extern fann_type_ff ff_p1ml;
extern fann_type_ff ff_p01m;
extern fann_type_ff ff_0000;
extern fann_type_ff ff_n1ml;
extern fann_type_ff ff_n001;
extern fann_type_ff ff_n098;
extern fann_type_ff ff_n100;
extern fann_type_ff ff_n200;
#ifndef FANN_INFERENCE_ONLY
extern fann_type_bp bp_0000;
#endif

#else // ! SOFTFANN

// FIXME: use limits for each data type
// general useful and frequent numbers
#define ff_p200 (CI_TWO)
#define ff_p100 (CI_ONE)
#define ff_p099 (CF_099)
#define ff_p098 (CF_098)
#define ff_p050 (CF_05)
#define ff_p001 (CF_001)
#define ff_p1ml (CF_0001)
#define ff_p01m (CF_00001)
#define ff_0000 (CI_ZERO)
#define ff_n1ml (-CF_0001)
#define ff_n001 (-CF_001)
#define ff_n098 (-CF_098)
#define ff_n100 (-CI_ONE)
#define ff_n200 (-CI_TWO)

#define bp_0000 (0)

#endif // ! SOFTFANN

#ifdef STEPWISE_LUT
extern fann_type_ff ff_v1;
extern fann_type_ff ff_v2;
extern fann_type_ff ff_v3;
extern fann_type_ff ff_v4;
extern fann_type_ff ff_v5;
extern fann_type_ff ff_v6;
extern fann_type_ff ff_r1_sigsym;
extern fann_type_ff ff_r2_sigsym;
extern fann_type_ff ff_r3_sigsym;
extern fann_type_ff ff_r4_sigsym;
extern fann_type_ff ff_r5_sigsym;
extern fann_type_ff ff_r6_sigsym;
extern fann_type_ff ff_r1_sig;
extern fann_type_ff ff_r2_sig;
extern fann_type_ff ff_r3_sig;
extern fann_type_ff ff_r4_sig;
extern fann_type_ff ff_r5_sig;
extern fann_type_ff ff_r6_sig;
#endif // STEPWISE_LUT

#endif // _fann_const_h

