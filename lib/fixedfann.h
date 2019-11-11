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


#ifndef __fixedfann_h__
#define __fixedfann_h__

#include <stdint.h>

#ifndef FANN_INFERENCE_ONLY
typedef float fann_type_nt;
#endif

typedef int32_t fann_type_bp;
typedef int32_t fann_type_ff;

#undef FIXEDFANN
#define FIXEDFANN

#define FANN_INCLUDE
#include "fann.h"

#endif
