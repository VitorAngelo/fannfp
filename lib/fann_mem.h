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


#ifndef _fann_mem_h
#define _fann_mem_h

#ifdef FANN_INFERENCE_ONLY
#define STATIC_MEMORY_ALLOCS
#else
#undef STATIC_MEMORY_ALLOCS
#endif

extern unsigned int fann_mem_current;

#ifdef STATIC_MEMORY_ALLOCS

//void * fann_mem_memset(void * ptr, int c, size_t len);
//void * fann_mem_memmove(void * dest, void * src, size_t sz);
//void * fann_mem_realloc(void * ptr, size_t sz);

void * fann_mem_memcpy(void * dest, void * src, unsigned int sz);
void * fann_mem_calloc(unsigned int len, unsigned int sz);
void * fann_mem_malloc(unsigned int len);
//void fann_mem_free(void * ptr);

//#define fann_memset(ptr, c, len) { fann_mem_memset(ptr, (int)c, (len) * sizeof(*(ptr))); }
//#define fann_memmove(dest, src, len) { fann_mem_memmove(dest, src, (len) * sizeof(*(dest))); }
//#define fann_realloc(ptr, len) { ptr = (typeof(ptr)) fann_mem_realloc(ptr, (len) * sizeof(*(ptr))); }

#define fann_memcpy(dest, src, len) { fann_mem_memcpy(dest, src, (len) * sizeof(*(dest))); }
#define fann_calloc(ptr, len) { ptr = (typeof(ptr)) fann_mem_calloc((len), sizeof(*(ptr))); }
#define fann_malloc(ptr, len) { ptr = (typeof(ptr)) fann_mem_malloc((len) * sizeof(*(ptr))); }
#define fann_free(x) {if(x != NULL) { x = NULL; }}
//#define fann_free(x) {if(x != NULL) { fann_mem_free(x); x = NULL; }}

unsigned int fann_mem_debug(void);

#else // Macros for system APIs

#include <stdlib.h>
#include <string.h>

//#define fann_memset(ptr, c, len) { memset(ptr, (int)c, (len) * sizeof(*(ptr))); }
//#define fann_memmove(dest, src, len) { memmove(dest, src, (len) * sizeof(*(dest))); }
//#define fann_realloc(ptr, len) { ptr = (typeof(ptr)) realloc(ptr, (len) * sizeof(*(ptr))); }

#define fann_memcpy(dest, src, len) { memcpy(dest, src, (len) * sizeof(*(dest))); }
#define fann_calloc(ptr, len) { ptr = (typeof(ptr)) calloc((len), sizeof(*(ptr))); fann_mem_current += len * sizeof(*(ptr)); }
#define fann_malloc(ptr, len) { ptr = (typeof(ptr)) malloc((len) * sizeof(*(ptr))); fann_mem_current += len; }
#define fann_free(ptr) { if (ptr != NULL) { free(ptr); ptr = NULL; }}

#endif // DEBUG_MEMORY_ALLOCS

#endif // _fann_mem_h

