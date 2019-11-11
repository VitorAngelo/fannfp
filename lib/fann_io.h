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


#ifndef __fann_io_h__
#define __fann_io_h__
    
#ifndef FANN_INFERENCE_ONLY

/* Section: FANN File Input/Output 
   
   It is possible to save an entire ann to a file with <fann_save> for future loading with <fann_create_from_file>.
 */    

/* Group: File Input and Output */    

/* Function: fann_create_from_file
   
   Constructs a backpropagation neural network from a configuration file, which has been saved by <fann_save>.
   
   See also:
       <fann_save>, <fann_save_to_fixed>
       
   This function appears in FANN >= 1.0.0.
 */
FANN_EXTERNAL struct fann *FANN_API fann_create_from_file(const char *configuration_file);


/* Function: fann_save

   Save the entire network to a configuration file.
   
   The configuration file contains all information about the neural network and enables 
   <fann_create_from_file> to create an exact copy of the neural network and all of the
   parameters associated with the neural network.
   
   These three parameters (<fann_set_callback>, <fann_set_error_log>,
   <fann_set_user_data>) are *NOT* saved to the file because they cannot safely be
   ported to a different location. Also temporary parameters generated during training
   like <fann_get_loss> are not saved.
   
   Return:
   The function returns 0 on success and -1 on failure.
   
   See also:
    <fann_create_from_file>, <fann_save_to_fixed>

   This function appears in FANN >= 1.0.0.
 */
FANN_EXTERNAL int FANN_API fann_save(struct fann *ann, const char *configuration_file);

#endif
#endif

