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

struct fann_data *train_data, *test_data;

int train_callback(struct fann *ann, struct fann_data *train, 
                   unsigned int max_epochs, 
                   unsigned int epochs_between_reports, 
                   float desired_error)
{
    unsigned int tot_errors;
    float tpr, tnr;
    double bit_acc;

    fann_test_data(ann, train_data);
    tpr = fann_get_true_positive(ann);
    tnr = fann_get_true_negative(ann);
    bit_acc = 100.0*(double)(ann->num_bit_ok[0] + ann->num_bit_ok[1]) /
                    (double)(ann->num_bit_ok[0] + ann->num_bit_ok[1] +
                             ann->num_bit_fail[0] + ann->num_bit_fail[1]);
    tot_errors = ann->num_bit_fail[0] + ann->num_bit_fail[1];
    if ((ann->train_epoch == 1) || ((ann->train_epoch % 10) == 0)) {
        printf("Train Epoch %u: SEP=%.2f ERP=%.2f ", ann->train_epoch, fann_get_sep(ann), fann_get_erp(ann));
        printf("Loss=%f, bits=%u/%u/%.2lf ", fann_get_loss(ann),
                ann->num_bit_fail[0], ann->num_bit_fail[1], bit_acc);
        printf(" AVG=%.2f, TPR=%.2f, TNR=%.2f\n", (tpr+tnr)/2.0, tpr, tnr);
#ifndef FANN_LIGHT
    	fann_print_stats(ann);
        //fann_print_structure(ann, __FILE__, __FUNCTION__, __LINE__);
#endif
    }
    fann_test_data(ann, test_data);
    tot_errors += ann->num_bit_fail[0] + ann->num_bit_fail[1];

    if (tot_errors == 0) {
        printf("Train Epoch %u: 0 errors\n", ann->train_epoch);
        return -1;
    }
    return 0;
}

int main(int argc, char *argv[])
{
	unsigned int threads = 0; 
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 96;
	const float desired_error = (const float) 0.001;
	struct fann *ann;

	unsigned int i = 0;

	printf("Creating network.\n");

	train_data = fann_read_data_from_file("../datasets/robot.train");
	test_data = fann_read_data_from_file("../datasets/robot.test");
    if (argc > 1) {
        fann_enable_seed_fixed(1);
        if (argc == 3)
            threads = 3;
    }

	ann = fann_create_standard_args(threads, num_layers,
					  train_data->num_input, num_neurons_hidden, train_data->num_output);
    fann_init_weights(ann);//, train_data);
    fann_set_callback(ann, train_callback);

	printf("Training network.\n");

	fann_set_training_algorithm(ann, FANN_TRAIN_INCREMENTAL);
    if (argc > 1) {
        if (argv[1][0] == '2') {
	        fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
            printf("FANN_TRAIN_RPROP\n");
        } else if (argv[1][0] == '3') {
	        fann_set_training_algorithm(ann, FANN_TRAIN_BATCH);
            printf("FANN_TRAIN_BATCH\n");
		    fann_set_learning_rate(ann, 0.03*(float)(train_data->num_data)*fann_get_learning_rate(ann));
            fann_set_mini_batch(ann, 34);
        /*} else if (argv[1][0] == '4') {
	        fann_set_training_algorithm(ann, FANN_TRAIN_QUICKPROP);
            printf("FANN_TRAIN_QUICKPROP\n");
		    fann_set_learning_rate(ann, 0.02*(float)(train_data->num_data)*fann_get_learning_rate(ann));*/
        }
    }
	fann_set_learning_momentum(ann, 0.4f);
#ifndef FANN_LIGHT
    fann_print_parameters(ann);
    fann_print_stats(ann);
#endif
	fann_train_on_data(ann, train_data, 500, 20, desired_error);
#ifndef FANN_LIGHT
    fann_print_stats(ann);
#endif

	printf("Testing network.\n");

	fann_reset_loss(ann);
	for(i = 0; i < fann_length_data(test_data); i++)
	{
		fann_test(ann, test_data->input[i], test_data->output[i]);
	}
	printf("Loss on test data: %f\n", fann_get_loss(ann));

	printf("Saving network.\n");

	fann_save(ann, "robot_float.net");

	printf("Cleaning up.\n");
	fann_destroy_data(train_data);
	fann_destroy_data(test_data);
	fann_destroy(ann);

	return 0;
}
