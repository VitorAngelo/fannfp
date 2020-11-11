#include <stdio.h>
#include <math.h>

#include "fann.h"

union
{
  double d;
  struct {
// LITTLE_ENDIAN
    int j,i;
// BIG_ENDIAN
//    int i,j;
  } n;
} _eco;

#define EXP_A (1048576/0.69314718055994530942)
#define EXP_C 60801
#define EXP(y) (_eco.n.i = EXP_A*(y) + (1072693248 - EXP_C), _eco.d)

int main(int argc, char *argv[])
{
    double step;
    fann_type_ff x;
    double e, y1a, y1b, y1c, y2a, y2b, y2c;
    FILE *fdsig, *fdexp;

    fann_const_init();

    if (argc != 4) {
        goto usage;
    }
    fdsig = fopen(argv[1], "w");
    fdexp = fopen(argv[2], "w");
    if ((fdsig == NULL) || (fdexp == NULL) || (sscanf(argv[3], "%lf", &step) != 1)) {
        goto usage;
    }
    for (x = -16.0; x < 16.0; x += step) {
        y1a = (2.0 / (1.0 + exp(-2.0 * x))) - 1.0;
        y1b = 0.0;//fann_activation_switch(FANN_SIGMOID_SYMMETRIC_STEPWISE, x);
        y1c = (2.0 / (1.0 + EXP(-2.0 * x))) - 1.0;
        y2a = 1.0 / (1.0 + exp(2.0 * -x));
        y2b = 0.0;//fann_activation_switch(FANN_SIGMOID_STEPWISE, x);
        y2c = 1.0 / (1.0 + EXP(2.0 * -x));
        fprintf(fdsig, "%.5e,%.5e,%.5e,%.5e,%.5e,%.5e,%.5e,%.5e\n", x, y1a, y1b, y1c, 2.0 * x, y2a, y2b, y2c);
    }
    for (x = -16.0; x < 16.0; x += step) {
        e = EXP(x);
        fprintf(fdexp, "%.5e,%.5e,%.5e\n", x, exp(x), e);
    }
	return 0;
usage:
    printf("./stepwise sigmoid.csv exp.csv step\n");
	return 1;
}
