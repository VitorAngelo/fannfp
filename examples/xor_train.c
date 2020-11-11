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

struct fann_data *train_data;

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

    if (tot_errors == 0) {
        printf("Train Epoch %u: 0 errors\n", ann->train_epoch);
        return -1;
    }
    return 0;
}

int main(int argc, char *argv[])
{
	unsigned int threads = 0;
	fann_type_nt *calc_out;
	const unsigned int num_input = 2;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 3;
	const unsigned int num_neurons_hidden = 3;
	const float desired_error = (const float) 0;
	const unsigned int max_epochs = 100;
	const unsigned int epochs_between_reports = 1;
	struct fann *ann;
	unsigned int i = 0, rand_seed;
    int train_algo = FANN_TRAIN_RPROP;

    if (argc > 1) {
        if (sscanf(argv[1], "%u", &rand_seed) == 1) {
            fann_enable_seed_fixed(rand_seed);
        }
        if (argc > 2) {
            if (strcmp(argv[2], "FANN_TRAIN_INCREMENTAL") == 0) {
                train_algo = FANN_TRAIN_INCREMENTAL;
            } else if (strcmp(argv[2], "FANN_TRAIN_BATCH") == 0) {
                train_algo = FANN_TRAIN_BATCH;
            }
        }
        if (argc == 3) {
            threads = 3;
        }
    }

	printf("Creating network.\n");
	ann = fann_create_standard_args(threads, num_layers, num_input, num_neurons_hidden, num_output);

	train_data = fann_read_data_from_file("xor.data");

	fann_set_activation_steepness_hidden(ann, 1);
	fann_set_activation_steepness_output(ann, 1);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_SIGMOID_SYMMETRIC);

	fann_set_train_stop_function(ann, FANN_STOPFUNC_BIT);
	fann_set_bit_fail_limit(ann, 0.01f);

	fann_set_training_algorithm(ann, train_algo);
    if (train_algo == FANN_TRAIN_BATCH) {
	    fann_set_learning_rate(ann, 0.6*(float)(train_data->num_data)*fann_get_learning_rate(ann));
    }

	fann_init_weights(ann);//, train_data);
    if (threads == 3) {
        fann_set_callback(ann, train_callback);
    }
#ifndef FANN_LIGHT
	fann_print_parameters(ann);
    if (threads == 3) {
	    fann_print_stats(ann);
    }
#endif

	printf("Training network.\n");
	fann_train_on_data(ann, train_data, max_epochs, epochs_between_reports, desired_error);

#ifndef FANN_LIGHT
    if (threads == 3)
	    fann_print_stats(ann);
#endif
    fann_test_data(ann, train_data);
	printf("Testing network. %f\n", fann_get_loss(ann));

	for(i = 0; i < fann_length_data(train_data); i++)
	{
		calc_out = fann_run(ann, train_data->input[i]);
		printf("XOR test (%f, %f) -> %f, should be %f, difference=%f\n",
			   train_data->input[i][0], train_data->input[i][1], calc_out[0], train_data->output[i][0],
			   fann_abs(calc_out[0] - train_data->output[i][0]));
	}

	printf("Saving network.\n");

	fann_save(ann, "xor_float.net");

	printf("Cleaning up.\n");
	fann_destroy_data(train_data);
	fann_destroy(ann);

	return 0;
}
