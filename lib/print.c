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

#ifndef FANN_INFERENCE_ONLY

#include <stdio.h>

#include "print.h"

static void _print_u16(uint16_t u, char c, int space)
{
    static char str[16 + 3 + 1];
    int i;

    i = 15;
    if (space)
        i += 3;
    str[i + 1] = '\0';
    for (; i >= 0; i--) {
        if (space && (i > 5) && (i < 9)) {
            str[i] = ' ';
            continue;
        }
        if ((u & 1) == 1) {
            str[i] = '1';
        } else {
            str[i] = '0';
        }
        u >>= 1;
    }
    if (c != '\0')
        printf("%s%c", str, c);
    else
        printf("%s", str);
}

void print_u16(uint16_t u, char c)
{
    printf("seeeee   ffffffffff\n");
    _print_u16(u, c, 1);
}

void print_u32(uint32_t u, char c)
{
    printf("seeeeeeeefffffffffffffffffffffff\n");
    _print_u16((uint16_t)((u & 0xFFFF0000) >> 16), '\0', 0);
    _print_u16((uint16_t)(u & 0x0000FFFF), c, 0);
}

#endif // FANN_INFERENCE_ONLY
