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


#include "fann_mem.h"

unsigned int fann_mem_current = 0;

#ifdef STATIC_MEMORY_ALLOCS

//#include <stdlib.h>
#include <string.h>

#define STATIC_MEM_SIZE (10*1024)
static uint8_t mem_alloc[STATIC_MEM_SIZE];
//static void * mem_ptr = &mem_alloc;
//static const void * last_byte = (&mem_alloc + STATIC_MEM_SIZE - 1);

/*
void * fann_mem_memset(void * ptr, int c, size_t len)
{
    return memset(ptr, c, len);
}

void * fann_mem_memmove(void * dest, void * src, size_t sz)
{
    return memmove(dest, src, sz);
}

void * fann_mem_realloc(void * ptr, size_t sz)
{
    return realloc(ptr, sz);
}
*/

void * fann_mem_memcpy(void * dest, void * src, unsigned int sz)
{
#if 0
    while (sz > 0) {
        sz--;
        *((uint8_t *)dest) = *((uint8_t *)src);
        dest++;
        src++;
    }
    return dest;
#else
    return memcpy(dest, src, sz);
#endif
}

void * fann_mem_calloc(unsigned int len, unsigned int sz)
{
    unsigned int bytes;
    uint8_t * ret;

    bytes = len * sz;
    ret = fann_mem_malloc(bytes);
    if (ret != NULL) {
#if 0
        while (bytes > 0) {
            bytes--;
            ret[bytes] = 0;
        }
#else
        memset(ret, 0, bytes);
#endif
    }
    return ret;
}

void * fann_mem_malloc(unsigned int sz)
{
    void * ret = &(mem_alloc[fann_mem_current]);

    fann_mem_current += sz;
    if (fann_mem_current > STATIC_MEM_SIZE) {
        fann_mem_current -= sz;
        return NULL;
    }
    return ret;
}

unsigned int fann_mem_debug(void)
{
    return fann_mem_current;
}

/*void fann_mem_free(void * ptr)
{
    free(ptr);
}*/

#endif // STATIC_MEMORY_ALLOCS

