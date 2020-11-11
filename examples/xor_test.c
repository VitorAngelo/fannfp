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

#if 1
#include <stdio.h>
#define dprintf(format, ...) fprintf(stdout, format, __VA_ARGS__)
#else
#define dprintf(format, ...)
#endif

#include "fann.h"
#include "fann_trained.h"

struct fann_data
{
    unsigned int num_data;
    unsigned int num_input;
    unsigned int num_output;
    fann_type_ff input[4][2];
    fann_type_ff output[4][1];
} data;/* = {
    .num_data = 4,
    .num_input = 2,
    .num_output = 1,
    .input = {
        {-1, -1},
        {-1, 1},
        {1, -1},
        {1, 1},
    },
    .output = {
        {-1},
        {1},
        {1},
        {-1},
    },
};*/

int main()
{
	fann_type_ff *calc_out, maxerr, err;
	unsigned int i;
	int ret = 0;
	struct fann *ann;

    data.num_data = 4;
    data.num_input = 2;
    data.num_output = 1;

    data.input[0][0] = fann_int_to_ff(-1);
    data.input[0][1] = fann_int_to_ff(-1);
    data.output[0][0] = fann_int_to_ff(-1);

    data.input[1][0] = fann_int_to_ff(-1);
    data.input[1][1] = fann_int_to_ff(1);
    data.output[1][0] = fann_int_to_ff(1);

    data.input[2][0] = fann_int_to_ff(1);
    data.input[2][1] = fann_int_to_ff(-1);
    data.output[2][0] = fann_int_to_ff(1);
    
    data.input[3][0] = fann_int_to_ff(1);
    data.input[3][1] = fann_int_to_ff(1);
    data.output[3][0] = fann_int_to_ff(-1);

    maxerr = fann_float_to_ff(0.01);
	dprintf("Creating network: %u\n", fann_mem_debug());
	ann = fann_create_trained();
	if (!ann) {
		dprintf("Error creating ann: %u\n", fann_mem_debug());
		return -1;
	}
	dprintf("Testing network: %u\n", fann_mem_debug());

	for(i = 0; i < data.num_data; i++)
	{
		calc_out = fann_run(ann, data.input[i]);
        err = fann_ff_abs(fann_ff_sub(calc_out[0], data.output[i][0]));
        if (fann_ff_gt(err, maxerr)) {
            ret = 1;
        }
		dprintf("XOR test (%f, %f) -> %f, should be %f, difference=%f\n",
			   fann_ff_to_float(data.input[i][0]),
               fann_ff_to_float(data.input[i][1]),
               fann_ff_to_float(calc_out[0]),
               fann_ff_to_float(data.output[i][0]),
			   fann_ff_to_float(err));
	}
	dprintf("Cleaning up: %u\n", fann_mem_debug());
	fann_destroy(ann);

	return ret;
}

