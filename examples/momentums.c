/*
Fast Artificial Neural Network Library (fann)
Copyright (C) 2003-2016 Steffen Nissen (steffen.fann@gmail.com)

This library is free software; you can redistribute it and/or
modify it under the terms of the GNU Lesser General Public
License as published by the Free Software Foundation; either
version 2.1 of the License, or (at your option) any later version.

This library is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the GNU
Lesser General Public License for more details.

You should have received a copy of the GNU Lesser General Public
License along with this library; if not, write to the Free Software
Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA02111-1307USA
*/

#include <stdio.h>

#include "fann.h"

int main(int argc, char *argv[])
{
	const unsigned int num_layers = 3;
#ifndef FANN_LIGHT
    int printit = 1;
#endif
	const unsigned int num_neurons_hidden = 96;
	const float desired_error = (const float) 0.001;
	struct fann *ann;
	struct fann_data *train_data, *test_data;

	float momentum;

	train_data = fann_read_data_from_file("../datasets/robot.train");
	test_data = fann_read_data_from_file("../datasets/robot.test");
    if (argc == 2) {
        fann_enable_seed_fixed(1);
    }
	for ( momentum = 0.0f; momentum < 0.7f; momentum += 0.1f ) {
		printf("============= momentum = %f =============\n", momentum);

		ann = fann_create_standard_args(0, num_layers,
						train_data->num_input, num_neurons_hidden, train_data->num_output);
        fann_init_weights(ann);//, train_data);

		fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);
        if (argc == 2) {
            if (argv[1][0] == '2') {
	            fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
                printf("FANN_TRAIN_RPROP\n");
            }
        }

		fann_set_learning_momentum(ann, momentum);

		fann_train_on_data(ann, train_data, 200, 20, desired_error);

        fann_test_data(ann, train_data);
		printf("Loss on train data: %f\n", fann_get_loss(ann));
        fann_test_data(ann, test_data);
		printf("Loss on test data : %f\n", fann_get_loss(ann));

#ifndef FANN_LIGHT
        fann_print_stats(ann);
        if (printit) {
            fann_print_parameters(ann);
            printit = 0;
        }
#endif

		fann_destroy(ann);
	}

	fann_destroy_data(train_data);
	fann_destroy_data(test_data);
	return 0;
}
