/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2016 Steffen Nissen (steffen.fann@gmail.com)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
*/

#include <stdio.h>
#include "fann.h"

int main()
{
	fann_type_nt *calc_out;
	fann_type_nt input[2];

	struct fann *ann;

	printf("Creating network.\n");
#ifdef FIXEDFANN
	ann = fann_create_from_file("xor_fixed.net");
#else
	ann = fann_create_from_file("xor_float.net");
#endif
	if(!ann) {
		printf("Error creating ann --- ABORTING.\n");
		return -1;
	}

	printf("Testing network.\n");
	input[0] = -1;
	input[1] = 1;
	calc_out = fann_run(ann, input);

	printf("xor test (%f,%f) -> %f (should be %d)\n", input[0], input[1], calc_out[0], 1);

	fann_destroy(ann);
	return 0;
}
