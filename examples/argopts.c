#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <getopt.h>
#include <string.h>
#include <math.h>

#include "fann.h"

int print_param = 0;
int print_stats = 0;
int print_grads = 0;
int train_acc = 1;
int unbalanced = 0;
int unbalanced_skip = 0;
unsigned int fixed_bias = 0;
unsigned int bp_bias = 15;
unsigned int max_idx_train = 0, max_idx_test = 0;
unsigned int num_layers = 0;
unsigned int rand_seed = 0;
unsigned int threads = 0;
extern unsigned int fann_train_shuffle;
//unsigned int train_shuffle = 0;
#define MAX_LAYERS 12
unsigned int num_neurons_hidden[MAX_LAYERS] = {0, };
//unsigned int max_cascade_neurons = 0;
struct fann_data *train_data = NULL, *test_data = NULL, *validation_data = NULL;
enum fann_activationfunc_enum activation_function_hidden = FANN_SIGMOID_SYMMETRIC;
enum fann_activationfunc_enum activation_function_output = FANN_SIGMOID;
//enum fann_activationfunc_enum activation_function_cascade = FANN_SIGMOID_SYMMETRIC;
//enum fann_errorfunc_enum train_err_func = FANN_ERRORFUNC_LINEAR;
enum fann_stopfunc_enum train_stop_func = FANN_STOPFUNC_LOSS;
unsigned int mini_batch = 0;
unsigned int max_epochs = 0;
unsigned int epochs_between_reports = 0;
unsigned int train_algo;
float max_error = 0.0;
const char * save_file = NULL;
char * from_file = NULL;
float learn_momentum = 0.0;
float steepness_hidden;
float steepness_output;
float steepness_start;
float steepness_scale;
float steepness_end;
#ifdef FANN_DATA_SCALE
float scale_lin_min = 0.0;
float scale_lin_max = 0.0;
#endif
float learning_rate = 0.0;
float randw_min = 0.0;
float randw_max = 0.0;
float bit_fail_lim = 0.0;
float rprop_delta_min = -1.0;
int bin_class = 1; 
unsigned int * class_count_train = NULL;
unsigned int * class_count_test = NULL;
unsigned int no_class_train = 0;
unsigned int no_class_test = 0;

static struct fann * arg_parse(int argc, char *argv[]);

void train_on_steepness(struct fann *ann, struct fann_data *data,
                        unsigned int max_epochs,
                        unsigned int epochs_between_reports)
{
	float acc, loss;

	if (epochs_between_reports)	{
		printf("Max epochs %8d. Desired loss: %.10f\n", max_epochs, max_error);
	}

    fann_set_activation_steepness_output(ann, steepness_start);
    fann_set_activation_steepness_hidden(ann, steepness_start);
	for (ann->train_epoch = 0; ann->train_epoch < max_epochs; ) {
		/* train */
		loss = fann_train_epoch(ann, data);

		/* print current output */
		if(epochs_between_reports && (ann->callback != NULL) &&
		   (((ann->train_epoch % epochs_between_reports) == 0) || (ann->train_epoch == max_epochs) ||
            (ann->train_epoch == 1) || (loss < max_error))) {
            int ret = ((*ann->callback)(ann, data, max_epochs, epochs_between_reports, max_error));
            if (ret == -1) {
                break;
            }
			//printf("Epochs     %8d. Current loss: %.10f\n", i, loss);
		}

		while (loss < max_error) {
			steepness_start *= steepness_scale;
			if (steepness_start > steepness_end) {
                ann->train_epoch = max_epochs;
                break;
            }
			printf("Steepness: %f -> ", steepness_start);
            fann_set_activation_steepness_hidden(ann, steepness_start);
            fann_set_activation_steepness_output(ann, steepness_start);
            acc = fann_test_data(ann, train_data);
            loss = fann_get_loss(ann);
            printf("LOSS on train data [%f], ACC = [%f]\n", loss, acc);
		}
	}
}

static void print_accuracy(struct fann *ann, FILE * fd, unsigned int * class_count)
{
    double bit_acc = 100.0*(double)(ann->num_bit_ok[0] + ann->num_bit_ok[1]) /
                           (double)(ann->num_bit_ok[0] + ann->num_bit_ok[1] +
                                    ann->num_bit_fail[0] + ann->num_bit_fail[1]);
    if (fd == NULL)
        return;
    fprintf(fd, "Loss=%f, bits=%u/%u/%.2lf ", fann_get_loss(ann),
            ann->num_bit_fail[0], ann->num_bit_fail[1], bit_acc);
    if ((ann->num_output > 1) && (class_count != NULL)) {
        unsigned int u;
        double acc, avg = 0.0, gmean = 1.0;

        for (u = 0; u < ann->num_output; u++) {
            if (class_count[u] == 0) {
                if (ann->num_max_ok[u] == 0)
                    acc = 100.0;
                else
                    acc = 0.0;
            } else {
                acc = 100.0 * (double)ann->num_max_ok[u] / (double)class_count[u];
            }
            fprintf(fd, " %.2f", acc);
            if (unbalanced_skip && (max_idx_train == u)) {
                //exclude maj. class from avg
                continue;
            }
            avg += acc;
            gmean *= acc;
        }
        u = ann->num_output;
        if (unbalanced_skip) {
            u--;
        }
        fprintf(fd, " %.2f (avg)", avg / (double)u);
        fprintf(fd, " GMEAN=%.2f\n", (float)pow(gmean, 1.0/(double)u));
    } else {
        float tpr = fann_get_true_positive(ann);
        float tnr = fann_get_true_negative(ann);
        fprintf(fd, " AVG=%.2f, TPR=%.2f, TNR=%.2f, GMEAN=%.2f\n", (tpr+tnr)/2.0, tpr, tnr, sqrtf(tpr*tnr));
    }
}

#include <time.h>

FILE * status = NULL;

static void print_time_eta(uint32_t train_diff_cpu, uint32_t train_diff_wall, uint32_t test_diff, double left)
{
    static char hms1[80], hms2[80], ymd1[80], ymd2[80];
    static double tot = 0.0;
    double secs, cpu, wall;
    time_t now;
    struct tm ts;

    if (status == NULL)
        return;

    fprintf(status, "train=");
    if (train_diff_cpu < 1000000) {
        fprintf(status, "%.3lf_ms", (double)train_diff_cpu/1.0e3);
    } else {
        fprintf(status, "%.3lf_s", (double)train_diff_cpu/1.0e6);
    }
    fprintf(status, " test=");
    if (test_diff < 1000000) {
        fprintf(status, "%.3lf_ms", (double)test_diff/1.0e3);
    } else {
        fprintf(status, "%.3lf_s", (double)test_diff/1.0e6);
    }
    time(&now);
    ts = *localtime(&now);
    strftime(hms1, sizeof(hms1), "now=%H:%M:%S", &ts);
    strftime(ymd1, sizeof(ymd1), "@%y-%m-%d", &ts);
    if (train_diff_cpu == 0) {
        fprintf(status, "\n");
        return;
    }
    cpu = (double)train_diff_cpu;
    wall = (double)train_diff_wall;
    secs = (double)(train_diff_cpu+test_diff)/1.0e6;
    if (tot == 0.0) {
        tot = secs;
    }
    tot = (0.8*tot) + (0.2*secs);
    now += (time_t)(left * tot);
    ts = *localtime(&now);
    strftime(hms2, sizeof(hms2), "ETA=%H:%M:%S", &ts);
    strftime(ymd2, sizeof(ymd2), "@%y-%m-%d", &ts);
    if (strcmp(ymd1, ymd2) == 0) {
        ymd2[0] = '\0';
    }
    fprintf(status, " avg=%.3lf_s ratio=%.2f %s %s%s\n", tot, cpu / wall, hms1, hms2, ymd2);
}

#if (defined SWF16_AP) || (defined HWF16)
extern unsigned int bias_histogram[32];
#endif

//static double tot_train_time = 0.0;

int train_callback(struct fann *ann, struct fann_data *train, 
                   unsigned int max_epochs, 
                   unsigned int epochs_between_reports, 
                   float desired_error)
{
    static void * ref_cpu = NULL;
    static void * ref_wall = NULL;
    uint32_t train_time_cpu, train_time_wall;
    double call_left;

    if (status == NULL) {
        status = fopen("status", "r");
        if (status != NULL) {
            fclose(status);
            status = stderr;
        }
    }

    train_time_cpu = fann_stop_count_us(ref_cpu);
    train_time_wall = fann_stop_count_us(ref_wall);
    ref_cpu = fann_start_count(ref_cpu, COUNT_CPU_TIME); // account for all testing time
    
    //fprintf(stderr, "train_time = %u\n", train_time);
    //tot_train_time += (double)train_time / 1000.0;

    /*if (train_shuffle > 0) {
        train_shuffle--;
        fann_shuffle_data(train_data);
    }*/
    call_left = (double)(max_epochs-ann->train_epoch)/(double)epochs_between_reports;
    //fprintf(stderr, "%u,%u,%u\n", epochs, epochs_between_reports, max_epochs);
    //fprintf(stderr, "call_left=%lf\n", call_left);

    if (save_file != NULL) {
        char epochNfile[4000];

        snprintf(epochNfile, sizeof(epochNfile)-1, "%s-%04u", save_file, ann->train_epoch);
        fann_save(ann, epochNfile);
    }
    /*if (status != NULL) {
        fprintf(status, "Train Epoch %u: SEP=%.2f ERP=%.2f ", ann->train_epoch,
                fann_get_sep(ann), fann_get_erp(ann));
        print_accuracy(ann, status, class_count_train);
    }*/
    fprintf(stdout, "Train Epoch %u: SEP=%.2f ERP=%.2f ", ann->train_epoch,
            fann_get_sep(ann), fann_get_erp(ann));
    print_accuracy(ann, stdout, class_count_train);
#ifndef FANN_LIGHT
    if (print_stats) {
        fann_print_stats(ann);
    }
#endif // FANN_LIGHT
    /*if (train->num_output > 1) {
        if (train_acc) {
            printf("ACCURACY on train data [%.2lf]:", fann_test_max_data(ann, train_data));
            print_accuracy(ann, stdout, class_count_train);
        }
        if (test_data != NULL) {
            printf("ACCURACY on test data [%.2lf]:", fann_test_max_data(ann, test_data));
            print_accuracy(ann, stdout, class_count_test);
        }
        if (validation_data != NULL) {
            printf("ACCURACY on valid. data [%.2lf]:", fann_test_max_data(ann, validation_data));
            print_accuracy(ann, stdout, class_count_test);
        }
    } else {*/
        if (train_acc) {
            printf("ACCURACY on train data [%.2lf]:", fann_test_data(ann, train_data));
            print_accuracy(ann, stdout, class_count_train);
        }
        if (test_data != NULL) {
            printf("ACCURACY on test data [%.2lf]:", fann_test_data(ann, test_data));
            print_accuracy(ann, stdout, class_count_test);
        }
        if (validation_data != NULL) {
            printf("ACCURACY on valid. data [%.2lf]:", fann_test_data(ann, validation_data));
            print_accuracy(ann, stdout, class_count_test);
        }
    //}
#ifdef DOUBLEFANN
#ifndef FANN_LIGHT
    if (print_grads && (ann->train_epoch == 1)) {
        fann_gradient_check(ann, train_data->input[0], train_data->output[0]);
    }
#endif // FANN_LIGHT
#endif
    print_time_eta(train_time_cpu, train_time_wall, fann_stop_count_us(ref_cpu), call_left);
#if (defined SWF16_AP) || (defined HWF16)
#ifdef FANN_AP_INCLUDE_ZERO
    printf("canc=%u, under=%u\nfp_bias=", fann_ap_cancel, fann_ap_underflow);
    fann_ap_cancel = 0;
#else
    printf("under=%u\nfp_bias=", fann_ap_underflow);
#endif
    fann_ap_underflow = 0;
    fann_ap_overflow = 0;
    for (int i = 15; i < 32; i++) {
        printf("%u:", bias_histogram[i]);
    }
    printf("%u\n", bias_histogram[0]);
#endif
    ref_cpu = fann_start_count(ref_cpu, COUNT_CPU_TIME);
    ref_wall = fann_start_count(ref_wall, COUNT_WALL_TIME);
    return 0;//epochs > 510;
}


int main(int argc, char *argv[])
{
    struct fann *ann = arg_parse(argc, argv);
    uint32_t diff;
    static void * ref = NULL;

    if (ann == NULL) {
        return 1;
    }
#ifndef FANN_LIGHT
#ifdef SOFTFANN 
    fann_reset_counters();
#endif
    if (print_param) {
        fann_print_parameters(ann);
    }
    if (print_stats) {
        fann_print_stats(ann);
    }
#endif // FANN_LIGHT
    //fann_print_structure(ann, __FILE__, __FUNCTION__, __LINE__);
    ref = fann_start_count(ref, COUNT_CPU_TIME);
    if (steepness_end > steepness_start) {
        train_on_steepness(ann, train_data, max_epochs, epochs_between_reports);
#if 0
    } else if (max_cascade_neurons > 0) {
		fann_set_cascade_activation_steepnesses(ann, &steepness_start, 1);
        fann_set_cascade_activation_functions(ann, &activation_function_cascade, 1);
        fann_set_cascade_num_candidate_groups(ann, 8);
        fann_cascadetrain_on_data(ann, train_data, max_cascade_neurons,
                                  1, max_error);
#endif
    } else {
        if (save_file != NULL) {
            char epoch0file[4000];

            snprintf(epoch0file, sizeof(epoch0file)-1, "%s-0000", save_file);
            fann_save(ann, epoch0file);
        }
	    fann_train_on_data(ann, train_data, max_epochs,
                           epochs_between_reports, max_error);
    }
    diff = fann_stop_count_us(ref);
    //fprintf(stderr, "tot_train_time = %f\n", tot_train_time / 1e3);
    if (rand_seed == 0) {
        printf("Time diff. = %u\n", diff);
#ifdef SOFTFANN
#ifndef FANN_LIGHT
    printf("Weight MAC OPs: %lu\n", fann_mac_ops_count);
    printf("Weight ADD OPs: %lu\n", fann_add_ops_count);
    printf("Weight MUL OPs: %lu\n", fann_mult_ops_count);
    printf("Weight DIV OPs: %lu\n", fann_div_ops_count);
#endif
#endif
    }
    if (save_file != NULL) {
        printf("Saving FLOAT network.\n");
        fann_save(ann, save_file);
    }

    printf("Dynamic Memory: %u bytes (ff=%d, bp=%d)\n", fann_mem_current, (int)sizeof(fann_type_ff), (int)sizeof(fann_type_bp));
    printf("Cleaning up.\n");
    fann_destroy_data(train_data);
    fann_destroy_data(test_data);
    fann_destroy_data(validation_data);
    fann_destroy(ann);
    return 0;
}

/*static int arg_test_file(const char *filename)
{
    if (filename != NULL) {
        FILE * tmp = fopen(filename, "r");
        if (tmp == NULL) {
            fprintf(stderr, "error openning file %s\n", filename);
            return -1;
        }
        fclose(tmp);
    }
    return 0;
}*/

enum cmd_options {
    PRINT_STATUS = 1,
    PRINT_PARAM,
    PRINT_STATS,
    PRINT_GRADS,
    SKIP_TRAIN_ACC,
    UNBALANCED,
    UNBALANCED_SKIP,
    RAND_SEED,
    BP_BIAS,
    FIXED_BIAS,
    THREADS,
    FROM_FILE,
    NUM_LAYERS,
    NUM_NEURONS_HIDDEN,
    //MAX_CASCADE_NEURONS,
    FILE_TRAIN,
    TRAIN_SHUFFLE,
    FILE_TEST,
    FILE_VALIDATION,
    FILE_FOLDS,
    ACTIV_FUNC_HIDDEN,
    ACTIV_FUNC_OUTPUT,
    //ACTIV_FUNC_CASCADE,
    MINI_BATCH,
    MAX_EPOCHS,
    REPORT_EPOCHS,
    TRAIN_ALGO,
    MAX_ERROR,
    SAVE_FILE,
    LEARN_MOMENTUM,
    STEEPNESS_CHANGE,
    STEEPNESS_HIDDEN,
    STEEPNESS_OUTPUT,
#ifdef FANN_DATA_SCALE
    SCALE_LINEAR,
#endif
    TRAIN_ERR_FUNC,
    TRAIN_STOP_FUNC,
    LEARNING_RATE,
    RAND_WEIGHTS,
    BIT_FAIL_LIM,
    RPROP_DELTA_MIN,
};

static struct fann * arg_parse(int argc, char *argv[])
{
    static struct option long_options[] = {
        {"print_status",        required_argument, NULL, PRINT_STATUS},
        {"print_param",         no_argument,       NULL, PRINT_PARAM},
        {"print_stats",         no_argument,       NULL, PRINT_STATS},
        {"print_grads",         no_argument,       NULL, PRINT_GRADS},
        {"skip_train_acc",      no_argument,       NULL, SKIP_TRAIN_ACC},
        {"unbalanced",          no_argument,       NULL, UNBALANCED},
        {"unbalanced_skip",     no_argument,       NULL, UNBALANCED_SKIP},
        {"rand_seed",           required_argument, NULL, RAND_SEED},
        {"bp_bias",             required_argument, NULL, BP_BIAS},
        {"fixed_bias",          no_argument,       NULL, FIXED_BIAS},
        {"threads",             required_argument, NULL, THREADS},
        {"from_file",           required_argument, NULL, FROM_FILE},
        {"num_layers",          required_argument, NULL, NUM_LAYERS},
        {"num_neurons_hidden",  required_argument, NULL, NUM_NEURONS_HIDDEN,},
        //{"max_cascade_neurons", required_argument, NULL, MAX_CASCADE_NEURONS},
        {"file_train",          required_argument, NULL, FILE_TRAIN},
        {"train_shuffle",       required_argument, NULL, TRAIN_SHUFFLE},
        {"file_test",           required_argument, NULL, FILE_TEST},
        {"file_validation",     required_argument, NULL, FILE_VALIDATION},
        {"file_folds",          required_argument, NULL, FILE_FOLDS},
        {"active_func_hidden",  required_argument, NULL, ACTIV_FUNC_HIDDEN},
        {"active_func_output",  required_argument, NULL, ACTIV_FUNC_OUTPUT},
        //{"active_func_cascade", required_argument, NULL, ACTIV_FUNC_CASCADE},
        {"mini_batch",          required_argument, NULL, MINI_BATCH},
        {"max_epochs",          required_argument, NULL, MAX_EPOCHS},
        {"report_epochs",       required_argument, NULL, REPORT_EPOCHS},
        {"train_algo",          required_argument, NULL, TRAIN_ALGO},
        {"max_error",           required_argument, NULL, MAX_ERROR},
        {"save_file",           required_argument, NULL, SAVE_FILE},
        {"learn_momentum",      required_argument, NULL, LEARN_MOMENTUM},
        {"steepness_change",    required_argument, NULL, STEEPNESS_CHANGE},
        {"steepness_hidden",    required_argument, NULL, STEEPNESS_HIDDEN},
        {"steepness_output",    required_argument, NULL, STEEPNESS_OUTPUT},
#ifdef FANN_DATA_SCALE
        {"scale_linear",        required_argument, NULL, SCALE_LINEAR},
#endif
        //{"train_err_func",      required_argument, NULL, TRAIN_ERR_FUNC},
        {"train_stop_func",     required_argument, NULL, TRAIN_STOP_FUNC},
        {"learning_rate",       required_argument, NULL, LEARNING_RATE},
        {"rand_weights",        required_argument, NULL, RAND_WEIGHTS},
        {"bit_fail_lim",        required_argument, NULL, BIT_FAIL_LIM},
        {"rprop_delta_min",     required_argument, NULL, RPROP_DELTA_MIN},
        {0, 0, NULL,  0 }
    };
    const unsigned int last_opt = sizeof(long_options)/sizeof(long_options)[0] - 1;
    int ret, option_index;
    struct fann * ann = NULL;
    unsigned int en, uarg;
    float tmpf1, tmpf2, tmpf3;
    
    steepness_start = 0.5;
    steepness_scale = 0.0;
    steepness_end = 0.0;
    steepness_hidden = 0.0;
    steepness_output = 0.0;

    for (;;) {
        ret = getopt_long_only(argc, argv, "", long_options, &option_index);
        if (ret == -1) {
            break;
        }
        if (ret == '?') {
            goto parse_error;
        }
        //printf("ret=%d\n", ret);
        switch (ret) {
        case PRINT_STATUS:
            if (sscanf(optarg, "%u", &uarg) != 1) {
                goto parse_error;
            } else if (uarg == 2) {
                status = stderr;
            } else if (uarg == 1) {
                status = stdout;
            } else {
                goto parse_error;
            }
            break;
        case PRINT_PARAM:
            print_param = 1;
            break;
        case PRINT_STATS:
            print_stats = 1;
            break;
        case PRINT_GRADS:
            print_grads = 1;
            break;
        case SKIP_TRAIN_ACC:
            train_acc = 0;
            break;
        case UNBALANCED:
            unbalanced = 1;
            break;
        case UNBALANCED_SKIP:
            unbalanced_skip = 1;
            break;
        case RAND_SEED:
            if (sscanf(optarg, "%u", &rand_seed) != 1) {
                goto parse_error;
            }
            fann_enable_seed_fixed(rand_seed);
            break;
        case BP_BIAS:
            if ((sscanf(optarg, "%u", &bp_bias) != 1) || (bp_bias > 31)) {
                goto parse_error;
            }
            break;
        case FIXED_BIAS:
            fixed_bias = 1;
            break;
        case THREADS:
            if ((sscanf(optarg, "%u", &threads) != 1) || (threads > 23)) {
                goto parse_error;
            }
        case FROM_FILE:
            from_file = optarg;
            break;
        case NUM_LAYERS:
            if (sscanf(optarg, "%u", &num_layers) != 1) {
                goto parse_error;
            }
            if (num_layers > MAX_LAYERS) {
                printf("num_layers=%u\n", num_layers);
                goto parse_error;
            }
            break;
        case NUM_NEURONS_HIDDEN:
            if (sscanf(optarg, "%u:%u:%u:%u:%u:%u:%u:%u:%u:%u",
                       num_neurons_hidden+1, num_neurons_hidden+2,
                       num_neurons_hidden+3, num_neurons_hidden+4,
                       num_neurons_hidden+5, num_neurons_hidden+6,
                       num_neurons_hidden+7, num_neurons_hidden+8,
                       num_neurons_hidden+9, num_neurons_hidden+10
                       ) != (num_layers-2)) {
                goto parse_error;
            }
            for (en = 1; en < (MAX_LAYERS-1); en++) {
                if (num_neurons_hidden[en] > 1000) {
                    printf("num_neurons_hidden=%u\n", num_neurons_hidden[en]);
                    goto parse_error;
                }
            }
            break;
            /*
        case MAX_CASCADE_NEURONS:
            if (sscanf(optarg, "%u", &max_cascade_neurons) != 1) {
                goto parse_error;
            }
            if (max_cascade_neurons > 1000) {
                printf("max_cascade_neurons=%u\n", max_cascade_neurons);
                goto parse_error;
            }
            break;
            */
        case FILE_TRAIN:
            if (strcmp(optarg, "-") == 0) {
                train_data = fann_read_data_from_file(NULL);
            } else {
                train_data = fann_read_data_from_file(optarg);
            }
            if (train_data == NULL) {
                goto parse_error;
            }
            break;
        case TRAIN_SHUFFLE:
            if (sscanf(optarg, "%u", &fann_train_shuffle) != 1) {
                goto parse_error;
            }
            /*if (train_data != NULL) {
                fann_shuffle_data(train_data);
            }*/
            break;
        case FILE_TEST:
            if (strcmp(optarg, "-") == 0) {
                test_data = fann_read_data_from_file(NULL);
            } else {
                test_data = fann_read_data_from_file(optarg);
            }
            if (test_data == NULL) {
                goto parse_error;
            }
            break;
        case FILE_VALIDATION:
            validation_data = fann_read_data_from_file(optarg);
            if (validation_data == NULL) {
                goto parse_error;
            }
            break;
        case FILE_FOLDS:
            if (test_data != NULL) {
                goto parse_error;
            }
            break;
        case ACTIV_FUNC_HIDDEN:
        case ACTIV_FUNC_OUTPUT:
        //case ACTIV_FUNC_CASCADE:
            for (en = 0; en < FANN_ACTIV_FUNC_LIMIT; en++) {
                if (strcmp(optarg, FANN_ACTIVATIONFUNC_NAMES[en]) == 0) {
                    if (ret == ACTIV_FUNC_HIDDEN)
                        activation_function_hidden = en;
                    else if (ret == ACTIV_FUNC_OUTPUT)
                        activation_function_output = en;
                    else
                        goto parse_error;
                        //activation_function_cascade = en;
                    break;
                }
            }
            if (en >= FANN_ACTIV_FUNC_LIMIT) {
                printf("invalid activation function %s\n", optarg);
                goto parse_error;
            }
            break;
        case MINI_BATCH:
            if (sscanf(optarg, "%u", &mini_batch) != 1) {
                goto parse_error;
            }
            break;
        case MAX_EPOCHS:
            if (sscanf(optarg, "%u", &max_epochs) != 1) {
                goto parse_error;
            }
            if (max_epochs > 10000) {
                printf("max_epochs=%u\n", max_epochs);
                goto parse_error;
            } else if (epochs_between_reports == 0) {
                epochs_between_reports = max_epochs / 10;
            }
            break;
        case REPORT_EPOCHS:
            if (sscanf(optarg, "%u", &epochs_between_reports) != 1) {
                goto parse_error;
            }
            if (epochs_between_reports > max_epochs) {
                printf("epochs_between_reports=%u\n", epochs_between_reports);
                goto parse_error;
            }
            break;
        case TRAIN_ALGO:
            for (en = 0; en <= FANN_TRAIN_LAST; en++) {
                if (strcmp(optarg, FANN_TRAIN_NAMES[en]) == 0) {
                    train_algo = en;
                    break;
                }
            }
            if (en > FANN_TRAIN_LAST) {
                printf("invalid training algorithm %s\n", optarg);
                goto parse_error;
            }
            break;
        case MAX_ERROR:
            if (sscanf(optarg, "%f", &max_error) != 1) {
                goto parse_error;
            }
            if ((max_error > 1000.0) || (max_error < 0.0)) {
                printf("max_error=%f\n", max_error);
                goto parse_error;
            }
            break;
        case SAVE_FILE:
            save_file = optarg;
            break;
        case LEARN_MOMENTUM:
            if (sscanf(optarg, "%f", &learn_momentum) != 1) {
                goto parse_error;
            }
            if ((learn_momentum < 0.0) || (learn_momentum > 10.0)) {
                printf("learn_momentum=%f\n", learn_momentum);
                goto parse_error;
            }
            break;
        case STEEPNESS_CHANGE:
            if (sscanf(optarg, "%f:%f:%f", &tmpf1, &tmpf2, &tmpf3) != 3) {
                goto parse_error;
            }
            steepness_start = tmpf1;
            steepness_scale = tmpf2;
            steepness_end = tmpf3;
            break;
        case STEEPNESS_HIDDEN:
            if (sscanf(optarg, "%f", &tmpf1) != 1) {
                goto parse_error;
            }
            steepness_hidden = tmpf1;
            break;
        case STEEPNESS_OUTPUT:
            if (sscanf(optarg, "%f", &tmpf1) != 1) {
                goto parse_error;
            }
            steepness_output = tmpf1;
            break;
#ifdef FANN_DATA_SCALE
        case SCALE_LINEAR:
            if (sscanf(optarg, "%f:%f", &scale_lin_min, &scale_lin_max) != 2) {
                goto parse_error;
            }
            break;
#endif
        /*case TRAIN_ERR_FUNC:
            for (en = 0; en <= TRAIN_ERRORFUNC_LAST; en++) {
                if (strcmp(optarg, FANN_ERRORFUNC_NAMES[en]) == 0) {
                    train_err_func = en;
                    break;
                }
            }
            if (en > TRAIN_ERRORFUNC_LAST) {
                printf("invalid training error function %s\n", optarg);
                goto parse_error;
            }
            break;*/
        case TRAIN_STOP_FUNC:
            for (en = 0; en <= TRAIN_STOPFUNC_LAST; en++) {
                //printf("[%s]\n", FANN_STOPFUNC_NAMES[en]);
                if (strcmp(optarg, FANN_STOPFUNC_NAMES[en]) == 0) {
                    train_stop_func = en;
                    break;
                }
            }
            if (en > TRAIN_STOPFUNC_LAST) {
                printf("[%s] -> invalid training stop function\n", optarg);
                goto parse_error;
            }
            break;
        case LEARNING_RATE:
            if (sscanf(optarg, "%f", &learning_rate) != 1) {
                goto parse_error;
            }
            if (learning_rate == 0.0) {
                printf("learning_rate=%f\n", learning_rate);
                goto parse_error;
            }
            break;
        case RAND_WEIGHTS:
            if (sscanf(optarg, "%f:%f", &randw_min, &randw_max) != 2) {
                goto parse_error;
            }
            break;
        case BIT_FAIL_LIM:
            if (sscanf(optarg, "%f", &bit_fail_lim) != 1) {
                goto parse_error;
            }
            if (bit_fail_lim == 0.0) {
                printf("bit_fail_lim=%f\n", bit_fail_lim);
                goto parse_error;
            }
            break;
        case RPROP_DELTA_MIN:
            if (sscanf(optarg, "%f", &rprop_delta_min) != 1) {
                goto parse_error;
            }
            if (rprop_delta_min < 0.0) {
                printf("rprop_delta_min=%f\n", rprop_delta_min);
                goto parse_error;
            }
            break;
        }
        printf("option %s", long_options[option_index].name);
        if (optarg)
            printf(" with arg %s", optarg);
        printf("\n");
    }
    if (optind < argc) {
        unsigned int totlen;
        for (totlen = 0; optind < argc; optind++) {
            if (strlen(argv[optind]) > 0) {
                totlen += strlen(argv[optind]);
                fprintf(stderr, "ignored command line argument: %d, %d -> ", optind, argc);
                fprintf(stderr, "[%s] ", argv[optind]);
            }
        }
        if (totlen > 0) {
            fprintf(stderr, "\n");
            return NULL;
        }
    }
    if ((train_data != NULL) && (num_layers > 1)) {
        for (en = 1; en < (num_layers-1); en++) {
            if (num_neurons_hidden[en] < 1) {
                printf("num_neurons_hidden[%u]\n", en);
                goto parse_error;
            }
        }
        num_neurons_hidden[0] = fann_num_input_data(train_data);
        num_neurons_hidden[num_layers-1] = fann_num_output_data(train_data);
        /*for (en = 0; en < num_layers; en++) {
            fprintf(stderr, "num_neurons[%u] = %u\n", en, num_neurons_hidden[en]);
        }*/
        ann = fann_create_standard_vector(threads, num_layers, num_neurons_hidden);
    }
    if ((ann == NULL) && (from_file != NULL)) {
        return fann_create_from_file(from_file);
    }
    //fann_print_structure(ann, __FILE__, __FUNCTION__, __LINE__);
    if (ann != NULL) {
        if (fixed_bias)
            fann_set_fixed_bp_bias(ann);
        else
            fann_set_dynamic_bp_bias(ann);
        fann_initialize_bp_bias(ann, bp_bias);
        if (rprop_delta_min >= 0.0)
            ann->rprop_delta_min = fann_float_to_ff(rprop_delta_min);
        fann_set_mini_batch(ann, mini_batch);
        fann_set_activation_function_hidden(ann, activation_function_hidden);
        if (steepness_hidden != 0.0)
            fann_set_activation_steepness_hidden(ann, steepness_hidden);
        fann_set_activation_function_output(ann, activation_function_output);
        if (steepness_output != 0.0)
            fann_set_activation_steepness_output(ann, steepness_output);
        fann_set_training_algorithm(ann, train_algo);
        //fann_set_train_error_function(ann, train_err_func);
        fann_set_train_stop_function(ann, train_stop_func);
        fann_set_callback(ann, train_callback);
        if (learn_momentum != 0.0)
		    fann_set_learning_momentum(ann, learn_momentum);
        if (learning_rate != 0.0)
		    fann_set_learning_rate(ann, learning_rate);
        if (bit_fail_lim != 0.0)
            fann_set_bit_fail_limit(ann, bit_fail_lim);
        if ((randw_min != 0.0) || (randw_max != 0.0))
            fann_randomize_weights(ann, fann_float_to_nt(randw_min),
                                        fann_float_to_nt(randw_max));
        else if (train_data != NULL)
            fann_init_weights(ann);//, train_data);
    }
#ifdef FANN_DATA_SCALE
    if ((scale_lin_min != 0.0) || (scale_lin_max != 0.0)) {
        struct fann_ff_limits * old_l = NULL, new_l;
        new_l.min = fann_float_to_ff(scale_lin_min);
        new_l.max = fann_float_to_ff(scale_lin_max);
        //fprintf(stderr, "scale_min=%f, scale_max=%f\n", scale_lin_min, scale_lin_max);
        if (train_data != NULL) {
            fann_scale_input_data_linear(train_data, &old_l, new_l);
            //fann_set_scaling_params( ann, train_data, scale_lin_min, scale_lin_max, 0, 1);
        }
        if (test_data != NULL) {
            fann_scale_input_data_linear(test_data, &old_l, new_l);
            //fann_set_scaling_params( ann, data, scale_lin_min, scale_lin_max, 0, 1);
        }
        if (validation_data != NULL) {
            fann_scale_input_data_linear(validation_data, &old_l, new_l);
        }
        fann_free(old_l);
    }
#endif
    if (bin_class) {
        if (train_data != NULL) {
            no_class_train = fann_count_classes(train_data, &class_count_train, 0.5, &max_idx_train, NULL);
            fprintf(stdout, "train: count=%u, no_class=%u, max=%u ->", train_data->num_data, no_class_train, max_idx_train);
            for (en = 0; en < train_data->num_output; en++) {
                fprintf(stdout, " %u", class_count_train[en]);
            }
            fprintf(stdout, "\n");
        }
        if (test_data != NULL) {
            no_class_test = fann_count_classes(test_data, &class_count_test, 0.5, &max_idx_test, NULL);
            fprintf(stdout, "test: count=%u, no_class=%u, max=%u ->", test_data->num_data, no_class_test, max_idx_test);
            for (en = 0; en < test_data->num_output; en++) {
                fprintf(stdout, " %u", class_count_test[en]);
            }
            fprintf(stdout, "\n");
        } else {
            max_idx_test = max_idx_train;
        }
        if (unbalanced) {
            fann_unbalance_adjust(ann, class_count_train);
            if (max_idx_train != max_idx_test) {
                fprintf(stdout, "WARNING: Unbalanced classes do not match (train != test)!\n");
            } else if (unbalanced_skip) {
                fprintf(stdout, "WARNING: Class %u will be disregarded from average accuracy\n", max_idx_train);
            }
        }
    }
    if (!print_stats) {
        FILE * fil = fopen("stats", "r");
        if (fil != NULL) {
            fclose(fil);
            print_stats = 1;
        }
    }
    return ann;

parse_error:
    if (option_index < last_opt) {
        if (optarg == NULL) {
            fprintf(stderr, "error parsing: %s\n", long_options[option_index].name);
        } else {
            fprintf(stderr, "error parsing: %s [%s]\n", long_options[option_index].name, optarg);
        }
    } else if (optarg != NULL) {
        fprintf(stderr, "invalid option: [%s]\n", optarg);
    } else {
        fprintf(stderr, "unknown error: %d, %d\n", ret, option_index);
    }
    return NULL;
}
