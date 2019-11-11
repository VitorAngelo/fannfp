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

#ifndef FANN_INFERENCE_ONLY

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

#include "fann.h"
#include "fann_data.h"

#define FANN_CONF_VERSION "FANN_FLO_2.1"

/* Create a network from a configuration file.
 */
FANN_EXTERNAL struct fann *FANN_API fann_create_from_file(const char *configuration_file)
{
    struct fann *ann;
    FILE *conf = fopen(configuration_file, "r");

    if(!conf)
    {
        fann_error(FANN_E_CANT_OPEN_CONFIG_R, configuration_file);
        return NULL;
    }
    ann = fann_create_from_fd(conf, configuration_file);
    fclose(conf);
    return ann;
}

#define IOPRINTF  "%a"
#define IOSCANF   "%a"
#define IOTYPE    float

#ifndef FANN_INFERENCE_ONLY
/* Save the network.
 */
FANN_EXTERNAL int FANN_API fann_save(struct fann *ann, const char *configuration_file)
{
    return fann_save_internal(ann, configuration_file);
}

/* INTERNAL FUNCTION
   Used to save the network to a file.
 */
int fann_save_internal(struct fann *ann, const char *configuration_file)
{
    int retval;
    FILE *conf = fopen(configuration_file, "w+");

    if(!conf)
    {
        fann_error(FANN_E_CANT_OPEN_CONFIG_W, configuration_file);
        return -1;
    }
    retval = fann_save_internal_fd(ann, conf);//, configuration_file);
    fclose(conf);
    return retval;
}

/* INTERNAL FUNCTION
   Used to save the network to a file descriptor.
 */
int fann_save_internal_fd(struct fann *ann, FILE * conf)//, const char *configuration_file)
{
    struct fann_layer *layer_it, *prev_layer;
    int calculated_decimal_point = 0;
    struct fann_neuron *neuron_it;//, *first_neuron;
    //fann_type_bp *weights;
    //struct fann_neuron **connected_neurons;
    unsigned int n, w;
#ifdef FANN_DATA_SCALE
    unsigned int i = 0;
#endif

    /* save the version information */
    fprintf(conf, FANN_CONF_VERSION "\n");
    /* Save network parameters */
    fann_set_ff_bias();
    fprintf(conf, "num_layers=%d\n", (int)(ann->last_layer - ann->first_layer));
    fprintf(conf, "learning_rate="IOPRINTF"\n", (IOTYPE)(fann_ff_to_float(ann->learning_rate)));
    fprintf(conf, "mini_batch=%u\n", ann->mini_batch);
    fprintf(conf, "learning_momentum="IOPRINTF"\n", (IOTYPE)(fann_ff_to_float(ann->learning_momentum)));
    fprintf(conf, "training_algorithm=%u\n", ann->training_algorithm);
    //fprintf(conf, "train_loss_function=%u\n", ann->train_loss_function);
    //fprintf(conf, "train_error_function=%u\n", ann->train_error_function);
    fprintf(conf, "train_stop_function=%u\n", ann->train_stop_function);
    fprintf(conf, "rmsprop_avg="IOPRINTF"\n", (IOTYPE)fann_ff_to_float(ann->rmsprop_avg));
    //fprintf(conf, "quickprop_decay="IOPRINTF"\n", (IOTYPE)fann_ff_to_float(ann->quickprop_decay));
    //fprintf(conf, "quickprop_mu="IOPRINTF"\n", (IOTYPE)fann_ff_to_float(ann->quickprop_mu));
    fprintf(conf, "rprop_increase_factor="IOPRINTF"\n", (IOTYPE)fann_ff_to_float(ann->rprop_increase_factor));
    fprintf(conf, "rprop_decrease_factor="IOPRINTF"\n", (IOTYPE)fann_ff_to_float(ann->rprop_decrease_factor));
    fprintf(conf, "rprop_delta_min="IOPRINTF"\n", (IOTYPE)fann_ff_to_float(ann->rprop_delta_min));
    fprintf(conf, "rprop_delta_max="IOPRINTF"\n", (IOTYPE)fann_ff_to_float(ann->rprop_delta_max));
    fprintf(conf, "rprop_delta_zero="IOPRINTF"\n", (IOTYPE)fann_ff_to_float(ann->rprop_delta_zero));
#ifdef CALCULATE_ERROR
    fprintf(conf, "bit_fail_limit="IOPRINTF"\n", (IOTYPE)ann->bit_fail_limit);
#endif // CALCULATE_ERROR
    fprintf(conf, "layer_sizes=");
    for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++)
    {
        /* the number of neurons in the layers (in the last layer, there is always one too many neurons, because of an unused bias) */
        fprintf(conf, " %u %u", layer_it->num_neurons, layer_it->activation);
    }
    fprintf(conf, "\n");

#ifdef FANN_DATA_SCALE
    /* 2.1 */
    #define SCALE_SAVE( what, where )                                        \
        fprintf( conf, #what "_" #where "=" );                                \
        for( i = 0; i < ann->num_##where##put; i++ )                        \
            fprintf( conf, "%f ", fann_nt_to_float(ann->what##_##where[ i ]) );                \
        fprintf( conf, "\n" );

    {
        if(ann->scale_mean_in != NULL)
        {
            fprintf(conf, "scale_included=1\n");
            SCALE_SAVE( scale_mean,            in )
            SCALE_SAVE( scale_deviation,    in )
            SCALE_SAVE( scale_new_min,        in )
            SCALE_SAVE( scale_factor,        in )
        
            SCALE_SAVE( scale_mean,            out )
            SCALE_SAVE( scale_deviation,    out )
            SCALE_SAVE( scale_new_min,        out )
            SCALE_SAVE( scale_factor,        out )
        }
        else
            fprintf(conf, "scale_included=0\n");
    }
#undef SCALE_SAVE
#endif // FANN_DATA_SCALE

    /* 2.0 */
    fprintf(conf, "neurons (num_inputs, activation_steepness)=\n");
    prev_layer = NULL;
    for(layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++) {
        unsigned int num_con;
        if (prev_layer == NULL)
            num_con = 0;
        else
            num_con = prev_layer->num_connections;
        fprintf(conf, "%u\n", num_con);
        /* the neurons */
        for (n = 0; n < layer_it->num_neurons; n++) {
            neuron_it = layer_it->neuron + n;
            fprintf(conf, IOPRINTF "\n",
                    (IOTYPE) fann_ff_to_float(neuron_it->steepness));
        }
        prev_layer = layer_it;
    }
    fprintf(conf, "connections (layer, connected_to_neuron, weight)=\n");
    prev_layer = ann->first_layer;
    for (layer_it = prev_layer + 1; layer_it != ann->last_layer; layer_it++) {
        unsigned int num_con;
        /* the neurons */
        num_con = prev_layer->num_connections;
        for (n = 0; n < layer_it->num_neurons; n++) {
            neuron_it = layer_it->neuron + n;
            for (w = 0; w < num_con; w++) {
                /* save the connection "(source weight) " */
                fprintf(conf, "%u, %u, " IOPRINTF "\n", n, w,
                        (IOTYPE) fann_ff_to_float(neuron_it->weight[w]));
            }
        }
        prev_layer = layer_it;
    }
    fprintf(conf, "\n");
    return calculated_decimal_point;
}
#endif // FANN_INFERENCE_ONLY

#define fann_scanf(type, name, val) \
{ \
    if(fscanf(conf, name"="type"\n", val) != 1) \
    { \
        fann_error(FANN_E_CANT_READ_CONFIG, name, configuration_file); \
        fann_destroy(ann); \
        return NULL; \
    } \
}

//printf("# %d %c %c\n", c, ch, name[c]);

#define fann_skip(name) \
{ \
    size_t len; int c; char ch; \
    for (len = strlen(name), c = 0; c < len; c++) \
    { \
        if ((fread(&ch, sizeof(ch), 1, conf) != 1) || (ch != name[c])) { \
            fann_error(FANN_E_CANT_READ_CONFIG, name, configuration_file); \
            fann_destroy(ann); \
            return NULL; \
        } \
    } \
}

/* INTERNAL FUNCTION
   Create a network from a configuration file descriptor.
 */
struct fann *fann_create_from_fd(FILE * conf, const char *configuration_file)
{
    unsigned int num_layers, layer_size, /*input_neuron,*/ i, num_connections;
    unsigned int tmpu;
    IOTYPE tmpf;
#ifdef FANN_DATA_SCALE
    unsigned int scale_included;
#endif
    struct fann_neuron /* *first_neuron, */ *neuron_it;//, *last_neuron;//, **connected_neurons;
    //fann_type_bp *weights;
    struct fann_layer *layer_it, *prev_layer;
    struct fann *ann = NULL;

    char *read_version;

    fann_const_init();
    fann_calloc(read_version, strlen(FANN_CONF_VERSION "\n"));
    //read_version = (char *) calloc(strlen(FANN_CONF_VERSION "\n"), 1);
    if(read_version == NULL)
    {
        fann_error(FANN_E_CANT_ALLOCATE_MEM);
        return NULL;
    }

    if(fread(read_version, 1, strlen(FANN_CONF_VERSION "\n"), conf) == 1)
    {
            fann_error(FANN_E_CANT_READ_CONFIG, "FANN_VERSION", configuration_file);
        return NULL;
    }

    /* compares the version information */
    if(strncmp(read_version, FANN_CONF_VERSION "\n", strlen(FANN_CONF_VERSION "\n")) != 0)
    {
        /* Maintain compatibility with 2.0 version that doesnt have scale parameters. */
        if(strncmp(read_version, "FANN_FLO_2.0\n", strlen("FANN_FLO_2.0\n")) != 0 &&
           strncmp(read_version, "FANN_FLO_2.1\n", strlen("FANN_FLO_2.1\n")) != 0)
        {
            free(read_version);
            fann_error(FANN_E_WRONG_CONFIG_VERSION, configuration_file);

            return NULL;
        }
    }

    free(read_version);

    fann_scanf("%u", "num_layers", &num_layers);

    ann = fann_allocate_structure(num_layers);
    if(ann == NULL)
    {
        return NULL;
    }
    fann_reset_loss(ann);

#if (defined SWF16_AP) || (defined HWF16)
    FP_BIAS = FP_BIAS_DEFAULT;
#endif

#ifdef FANN_INFERENCE_ONLY
    fann_scanf(IOSCANF, "learning_rate", &tmpf);
    fann_scanf("%u", "mini_batch", &tmpu);
    fann_scanf(IOSCANF, "learning_momentum", &tmpf);
    fann_scanf("%u", "training_algorithm", &tmpu);
    fann_scanf("%u", "train_loss_function", &tmpu);
    //fann_scanf("%u", "train_error_function", &tmpu);
    //ann->train_error_function = (enum fann_errorfunc_enum)tmpu;
    fann_scanf("%u", "train_stop_function", &tmpu);
    fann_scanf(IOSCANF, "rmsprop_avg", &tmpf);
    fann_scanf(IOSCANF, "quickprop_decay", &tmpf);
    fann_scanf(IOSCANF, "quickprop_mu", &tmpf);
    fann_scanf(IOSCANF, "rprop_increase_factor", &tmpf);
    fann_scanf(IOSCANF, "rprop_decrease_factor", &tmpf);
    fann_scanf(IOSCANF, "rprop_delta_min", &tmpf);
    fann_scanf(IOSCANF, "rprop_delta_max", &tmpf);
    fann_scanf(IOSCANF, "rprop_delta_zero", &tmpf);
#else
    fann_scanf(IOSCANF, "learning_rate", &tmpf);
    ann->learning_rate = fann_float_to_ff(tmpf);
    fann_scanf("%u", "mini_batch", &(ann->mini_batch));
    fann_scanf(IOSCANF, "learning_momentum", &tmpf);
    ann->learning_momentum = fann_float_to_ff(tmpf);
    fann_scanf("%u", "training_algorithm", &tmpu);
    ann->training_algorithm = (enum fann_train_enum)tmpu;
    //fann_scanf("%u", "train_loss_function", &tmpu);
    //ann->train_loss_function = (enum fann_lossfunc_enum)tmpu;
    //fann_scanf("%u", "train_error_function", &tmpu);
    //ann->train_error_function = (enum fann_errorfunc_enum)tmpu;
    fann_scanf("%u", "train_stop_function", &tmpu);
    ann->train_stop_function = (enum fann_stopfunc_enum)tmpu;
    fann_scanf(IOSCANF, "rmsprop_avg", &tmpf);
    ann->rmsprop_avg = fann_float_to_ff(tmpf);
    /*fann_scanf(IOSCANF, "quickprop_decay", &tmpf);
    ann->quickprop_decay = fann_float_to_ff(tmpf);
    fann_scanf(IOSCANF, "quickprop_mu", &tmpf);
    ann->quickprop_mu = fann_float_to_ff(tmpf);*/
    fann_scanf(IOSCANF, "rprop_increase_factor", &tmpf);
    ann->rprop_increase_factor = fann_float_to_ff(tmpf);
    fann_scanf(IOSCANF, "rprop_decrease_factor", &tmpf);
    ann->rprop_decrease_factor = fann_float_to_ff(tmpf);
    fann_scanf(IOSCANF, "rprop_delta_min", &tmpf);
    ann->rprop_delta_min = fann_float_to_ff(tmpf);
    fann_scanf(IOSCANF, "rprop_delta_max", &tmpf);
    ann->rprop_delta_max = fann_float_to_ff(tmpf);
    fann_scanf(IOSCANF, "rprop_delta_zero", &tmpf);
    ann->rprop_delta_zero = fann_float_to_ff(tmpf);
#endif // FANN_INFERENCE_ONLY
#ifdef CALCULATE_ERROR
    fann_scanf(IOSCANF, "bit_fail_limit", &tmpf);
    ann->bit_fail_limit = tmpf;
#endif // CALCULATE_ERROR
#ifdef DEBUG
    printf("creating network with %d layers\n", num_layers);
    printf("input\n");
#endif

    fann_skip("layer_sizes=");
    /* determine how many neurons there should be in each layer */
    for(layer_it = ann->first_layer; layer_it != ann->last_layer; layer_it++)
    {
        if(fscanf(conf, " %u %u", &layer_size, &tmpu) != 2)
        {
            fann_error(FANN_E_CANT_READ_CONFIG, "layer_sizes", configuration_file);
            fann_destroy(ann);
            return NULL;
        }
        /* we do not allocate room here, but we make sure that
         * last_neuron - first_neuron is the number of neurons */
        layer_it->neuron = NULL;
        layer_it->num_neurons = layer_size;
        layer_it->num_connections = layer_it->num_neurons + 1;
        layer_it->activation = tmpu;
    }
    fann_skip("\n");

    // ignore the BIAS
    ann->num_input = ann->first_layer->num_neurons;
    ann->num_output = ((ann->last_layer - 1)->num_neurons);

#ifdef FANN_DATA_SCALE
#define SCALE_LOAD( what, where )                                            \
    fann_skip( #what "_" #where "=" );                                    \
    for(i = 0; i < ann->num_##where##put; i++)                                \
    {                                                                        \
        if(fscanf( conf, "%f ", (float *)&ann->what##_##where[ i ] ) != 1)  \
        {                                                                    \
            fann_error(FANN_E_CANT_READ_CONFIG, #what "_" #where, configuration_file); \
            fann_destroy(ann);                                                 \
            return NULL;                                                    \
        }                                                                    \
    }
    
    if(fscanf(conf, "scale_included=%u\n", &scale_included) == 1 && scale_included == 1)
    {
        fann_allocate_scale(ann);
        SCALE_LOAD( scale_mean,            in )
        SCALE_LOAD( scale_deviation,    in )
        SCALE_LOAD( scale_new_min,        in )
        SCALE_LOAD( scale_factor,        in )
    
        SCALE_LOAD( scale_mean,            out )
        SCALE_LOAD( scale_deviation,    out )
        SCALE_LOAD( scale_new_min,        out )
        SCALE_LOAD( scale_factor,        out )
    }
#undef SCALE_LOAD
#endif // FANN_DATA_SCALE
    
    /* allocate room for the actual neurons */
    if (fann_allocate_neurons(ann, NULL)) {
        fann_destroy(ann);
        return NULL;
    }

    fann_skip("neurons (num_inputs, activation_steepness)=\n");
    prev_layer = NULL;
    for(layer_it = ann->first_layer + 1; layer_it != ann->last_layer; layer_it++) {
        /* the neurons */
        unsigned int num_n = layer_it->num_neurons;
        if (fscanf(conf, "%u\n", &num_connections) != 1) {
            fann_error(FANN_E_CANT_READ_CONFIG, "num_connections", configuration_file);
            fann_destroy(ann);
            return NULL;
        }
        if ((prev_layer != NULL) && (prev_layer->num_connections != num_connections)) {
            fann_error(FANN_E_CANT_READ_CONFIG, "wrong_connections", configuration_file);
            fann_destroy(ann);
            return NULL;
        }
        for (i = 0; i < num_n; i++) {
            neuron_it = layer_it->neuron + i;
            if (fscanf(conf, IOSCANF "\n", &tmpf) != 1) {
                fann_error(FANN_E_CANT_READ_NEURON, configuration_file);
                fann_destroy(ann);
                return NULL;
            }
            neuron_it->steepness = fann_float_to_ff(tmpf);
        }
        // BIAS value:
        layer_it->value[i] = ff_p100;//fann_int_to_bp(1);
        prev_layer = layer_it;
    }

    fann_skip("connections (layer, connected_to_neuron, weight)=\n");
    prev_layer = ann->first_layer;
    for (layer_it = prev_layer + 1; layer_it != ann->last_layer; layer_it++) {
        unsigned int w, tmpl, num_con;
        /* the neurons */
        num_con = prev_layer->num_connections;
        for (i = 0; i < layer_it->num_neurons; i++) {
            neuron_it = layer_it->neuron + i;
            for (w = 0; w < num_con; w++) {
                if ((fscanf(conf, "%u, %u, " IOSCANF "\n", &tmpl, &tmpu, &tmpf) != 3) || (tmpu != w) || (tmpl != i)) {
                    fann_error(FANN_E_CANT_READ_CONNECTIONS, configuration_file);
                    fann_destroy(ann);
                    return NULL;
                }
                neuron_it->weight[w] = fann_float_to_ff(tmpf);
            }
        }
        prev_layer = layer_it;
    }
    return ann;
}

#endif // FANN_INFERENCE_ONLY

