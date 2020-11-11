#include "fann.h"

int main( int argc, char *argv[] )
{
#ifdef FANN_DATA_SCALE
	const unsigned int num_input = 3;
	const unsigned int num_output = 1;
	const unsigned int num_layers = 4;
	const unsigned int num_neurons_hidden = 5;
	const float desired_error = (const float) 0.0001;
	const unsigned int max_epochs = 5000;
	const unsigned int epochs_between_reports = 1000;
	struct fann_data * data = NULL;
	struct fann *ann;

    if (argc == 2) {
        fann_enable_seed_fixed(1);
    }

	ann = fann_create_standard_args(0, num_layers, num_input, num_neurons_hidden, num_neurons_hidden, num_output);

	fann_set_activation_function_hidden(ann, FANN_SIGMOID_SYMMETRIC);
	fann_set_activation_function_output(ann, FANN_LINEAR);
	fann_set_training_algorithm(ann, FANN_TRAIN_RPROP);
	data = fann_read_data_from_file("../datasets/scaling.data");
	fann_set_scaling_params(
		    ann,
			data,
			-1,	/* New input minimum */
			1,	/* New input maximum */
			-1,	/* New output minimum */
			1);	/* New output maximum */

    fann_init_weights(ann);//, data);
#ifndef FANN_LIGHT
	fann_print_parameters(ann);
#endif
	fann_scale_train( ann, data );

	fann_train_on_data(ann, data, max_epochs, epochs_between_reports, desired_error);
#ifndef FANN_LIGHT
	fann_print_stats(ann);
#endif

	fann_save(ann, "scaling_float.net");
	fann_save_data(data, "scaling_float.data");

    fann_destroy_data( data );
	fann_destroy(ann);
#endif // FANN_DATA_SCALE
	return 0;
}
