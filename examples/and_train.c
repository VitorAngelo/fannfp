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

struct fann_data *data;

int FANN_API test_callback(struct fann *ann, struct fann_data *train,
	unsigned int max_epochs, unsigned int epochs_between_reports, 
	float desired_error)
{
	printf("\nEpochs     %8d. loss: %.5f. Desired-loss: %.5f\n", ann->train_epoch, fann_get_loss(ann), desired_error);
#ifndef FANN_LIGHT
	for(unsigned int i = 0; i < fann_length_data(data); i++) {
        fann_gradient_check(ann, data->input[i], data->output[i]);
    }	
#endif
	return 0;
}

int main(int argc, char *argv[])
{
	fann_type_nt *calc_out;
	const unsigned int num_input = 2;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 1;
	const float desired_error = (const float) 0;
	const unsigned int max_epochs = 100;
	const unsigned int epochs_between_reports = 1;
	struct fann *ann;
	unsigned int i = 0;

    if (argc == 2) {
        fann_enable_seed_fixed(1);
    }

	printf("Creating network.\n");
	ann = fann_create_standard_args(0, num_layers, num_input, num_neurons_hidden, num_output);

	data = fann_read_data_from_file("and.data");

	fann_set_activation_steepness_hidden(ann, 1);
	fann_set_activation_steepness_output(ann, 1);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	fann_set_bit_fail_limit(ann, 0.0f);

	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
    if (argc == 2) {
        if (argv[1][0] == '1') {
            fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);
            printf("FANN_TRAIN_INCREMENTAL\n");
        } else if (argv[1][0] == '3') {
	        fann_set_training_algorithm(ann, FANN_TRAIN_BATCH);
            printf("FANN_TRAIN_BATCH\n");
		    fann_set_learning_rate(ann, 0.6*(float)(data->num_data)*fann_get_learning_rate(ann));
        }
    }

	fann_init_weights(ann);//, data);
#ifndef FANN_LIGHT
	fann_print_parameters(ann);
#endif
    fann_set_callback(ann, test_callback);
	
#ifndef FANN_LIGHT
	fann_print_stats(ann);
#endif
	printf("Training network.\n");
	fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);

	printf("Testing network. %f\n", fann_test_data(ann, data));

	for(i = 0; i < fann_length_data(data); i++)
	{
		calc_out = fann_run(ann, data->input[i]);
		printf("AND test (%f,%f) -> %f, should be %f, difference=%f\n",
			   data->input[i][0], data->input[i][1], calc_out[0], data->output[i][0],
			   fann_abs(calc_out[0] - data->output[i][0]));
	}

	printf("Saving network.\n");

	fann_save(ann, "and_float.net");

	printf("Cleaning up.\n");
	fann_destroy_data(data);
	fann_destroy(ann);

	return 0;
}
