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


#ifndef _fann_activation_h
#define _fann_activation_h

void fann_activation_switch(struct fann_layer * layer_it, unsigned int neuron);

#ifndef FANN_INFERENCE_ONLY
struct fann_derive {
    //enum fann_activationfunc_enum activation;
    fann_type_bp steepness;
    fann_type_bp value;
    fann_type_bp error;
};
// overcome GCC ARM limitation:
extern struct fann_derive fann_derive_info;
extern void (*fann_derive_funcptr)(void);
void fann_activation_derive_select(enum fann_activationfunc_enum activation);

//void fann_activation_derived(struct fann_derive * dev);
#endif // FANN_INFERENCE_ONLY

#endif // _fann_activation_h

