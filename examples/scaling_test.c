#include <stdio.h>
#include "fann.h"

int main( int argc, char** argv )
{
	int ret = 0;
#ifdef FANN_DATA_SCALE
    float err, max_err = 0.0;
	fann_type_ex *calc_out;
	unsigned int i;
	struct fann *ann;
	struct fann_data *data;
	printf("Creating network.\n");
#ifdef FIXEDFANN
	ann = fann_create_from_file("scaling_fixed.net");
#else
	ann = fann_create_from_file("scaling_float.net");
#endif
	if(!ann) {
		printf("Error creating ann --- ABORTING.\n");
		return 0;
	}
#ifndef FANN_LIGHT
	fann_print_stats(ann);
	fann_print_parameters(ann);
#endif

	printf("Testing network.\n");
#ifdef FIXEDFANN
	data = fann_read_data_from_file("scaling_fixed.data");
#else
	//data = fann_read_data_from_file("scaling_float.data");
	data = fann_read_data_from_file("../datasets/scaling.data");
#endif
	for(i = 0; i < fann_length_data(data); i++)
	{
		fann_reset_loss(ann);
#ifdef FIXEDFANN
		calc_out = fann_run( ann, data->input[i] );
		printf("Scaling test (%d, %d) -> %d, should be %d, difference=%f\n",
			   data->input[i][0], data->input[i][1], calc_out[0], data->output[i][0],
			   (float) fann_abs(calc_out[0] - data->output[i][0]) / fann_get_multiplier(ann));
        err = (float) (fann_abs(calc_out[0] - data->output[i][0])) / (float)data->output[i][0];
        if (max_err < err) {
            max_err = err;
        }
#else
    	fann_scale_input( ann, data->input[i] );
		calc_out = fann_run( ann, data->input[i] );
		fann_descale_output( ann, calc_out );
        err = (float) (fann_abs(calc_out[0] - data->output[i][0])) / data->output[i][0];
        if (max_err < err) {
            max_err = err;
        }
		printf("Result %f original %f error %f\n",
			calc_out[0], data->output[i][0], err);
#endif
	}
    printf("Max. Error: %f\n", max_err);
	printf("Cleaning up.\n");
	fann_destroy_data(data);
	fann_destroy(ann);
#endif // FANN_DATA_SCALE
	return ret;
}
