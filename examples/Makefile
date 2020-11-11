BINS = momentums mushroom robot steepness_train stepwise
BINS += scaling_test_double scaling_train simple_test simple_train
BINS += add_train and_train xor_train xor_test_float
#BINS += f16_fann
#BINS += testfixed scaling_test_fixed xor_test_fixed

XBINS = double_fann float_fann soft-ap_fann soft-ieee_fann soft-hwf16_fann soft-posit16_fann floatunion_fann bfloat16_fann

ABINS = floatfp16_fann fp16fp16_fann

BINS += $(XBINS)

DFLAGS = -g
DFLAGS += -pg

ARCH_X86 = -march=westmere -mtune=westmere

#ARCH_PI = -march=armv6 -mtune=cortex-a53 -mfp16-format=alternative -mfpu=neon-fp16
ARCH_PI = -mcpu=cortex-a53 -march=armv8-a+crc -mtune=cortex-a53 -mfpu=crypto-neon-fp-armv8 -mfloat-abi=hard
ARCH_PI += -mfp16-format=alternative

x86: ARCH = $(ARCH_X86)
x86: STRIP = strip
x86: $(BINS) sha1.x86

pi: ARCH = $(ARCH_PI)
pi: STRIP = strip
pi: $(ABINS) $(BINS) sha1.pi

dx86: ARCH = $(ARCH_X86)
dx86: STRIP = touch
dx86: CFLAGS += $(DFLAGS)
dx86: $(BINS)

dpi: ARCH = $(ARCH_PI)
dpi: STRIP = touch
dpi: CFLAGS += $(DFLAGS)
dpi: $(ABINS) $(BINS)

sha1.x86: $(XBINS)
	sha1sum $(XBINS) > sha1.x86

sha1.pi: $(ABINS) $(XBINS)
	sha1sum $(ABINS) $(XBINS) > sha1.pi

CFLAGS = -Wall -pipe -I ../lib
#CFLAGS += -Wmissing-prototypes -Wstrict-prototypes -Werror-implicit-function-declaration -Wundef -Wunused-parameter
#CFLAGS += -DFANN_LIGHT

floatfp16_fann: argopts.c ../lib/floatfp16.o
	gcc $(CFLAGS) $(ARCH_PI) -DFANN_FLOAT -D_GCC_ARM_F16_FF ../lib/floatfp16.o -o $@ argopts.c -lm -lpthread -static
	$(STRIP) floatfp16_fann

fp16fp16_fann: argopts.c ../lib/fp16fp16.o
	gcc $(CFLAGS) $(ARCH_PI) -DFANN_FLOAT -D_GCC_ARM_F16_BP ../lib/fp16fp16.o -o $@ argopts.c -lm -lpthread -static
	$(STRIP) fp16fp16_fann

double_fann: argopts.c ../lib/doublefann.o
	gcc $(CFLAGS) $(ARCH) -DFANN_DOUBLE ../lib/doublefann.o -o $@ argopts.c -lm -lpthread -static
	$(STRIP) double_fann

float_fann: argopts.c ../lib/floatfann.o
	gcc $(CFLAGS) $(ARCH) -DFANN_FLOAT ../lib/floatfann.o -o $@ argopts.c -lm -lpthread -static
	$(STRIP) float_fann

floatunion_fann: argopts.c ../lib/floatunion.o
	gcc $(CFLAGS) $(ARCH) -DFANN_FLOAT -D_FLOAT_UNION ../lib/floatunion.o -o $@ argopts.c -lm -lpthread -static
	$(STRIP) floatunion_fann

bfloat16_fann: argopts.c ../lib/bfloat16.o
	gcc $(CFLAGS) $(ARCH) -DFANN_FLOAT -D_BFLOAT16 ../lib/bfloat16.o -o $@ argopts.c -lm -lpthread -static
	$(STRIP) bfloat16_fann

f16_fann: argopts.c ../lib/f16fann.o
	gcc $(CFLAGS) $(ARCH) -DFANN_FLOAT ../lib/f16fann.o -D_GCC_ARM_F16 -mfp16-format=ieee -o $@ argopts.c -lm -lpthread
	$(STRIP) f16_fann

soft-ap_fann: argopts.c ../lib/softfann-ap.o
	gcc $(CFLAGS) $(ARCH) -DFANN_SOFT -DSWF16_AP ../lib/softfann-ap.o -o $@ argopts.c -lm -lpthread -static
	$(STRIP) soft-ap_fann

soft-ieee_fann: argopts.c ../lib/softfann-ieee.o
	gcc $(CFLAGS) $(ARCH) -DFANN_SOFT -DSWF16_IEEE ../lib/softfann-ieee.o -o $@ argopts.c -lm -lpthread -static
	$(STRIP) soft-ieee_fann

soft-hwf16_fann: argopts.c ../lib/softfann-hwf16.o
	gcc $(CFLAGS) $(ARCH) -DFANN_SOFT -DHWF16 ../lib/softfann-hwf16.o -o $@ argopts.c -lm -lpthread -static
	$(STRIP) soft-hwf16_fann
soft-posit16_fann: argopts.c ../lib/softfann-posit16.o
	gcc $(CFLAGS) $(ARCH) -DFANN_SOFT -DPOSIT16 ../lib/softfann-posit16.o -o $@ argopts.c -lm -lpthread -static
	$(STRIP) soft-posit16_fann

COMPILE_DOUBLE = gcc $(CFLAGS) $(ARCH) -DFANN_DOUBLE ../lib/doublefann.o -o $@ $@.c -lm -lpthread

BUILD_FLOAT = gcc $(CFLAGS) $(ARCH) -DFANN_EMBEDDED -DFANN_FLOAT

COMPILE_FLOAT = $(BUILD_FLOAT) ../lib/x86-embedded.o -o $@ $@.c -lm -lpthread

COMPILE_FIXED = gcc $(CFLAGS) $(ARCH) -DFANN_FIXED ../lib/fixedfann.o -o $@ $@.c -lm -lpthread

stepwise: stepwise.c ../lib/doublefann.o
	$(COMPILE_DOUBLE)

cascade_train: cascade_train.c ../lib/doublefann.o
	$(COMPILE_DOUBLE)

momentums: momentums.c ../lib/doublefann.o
	$(COMPILE_DOUBLE)

mushroom: mushroom.c ../lib/doublefann.o
	$(COMPILE_DOUBLE)

robot: robot.c ../lib/doublefann.o
	$(COMPILE_DOUBLE)

scaling_test_double: scaling_test.c ../lib/doublefann.o
	$(COMPILE_DOUBLE)

scaling_test_fixed: scaling_test.c ../lib/fixedfann.o
	$(COMPILE_FIXED)

scaling_train: scaling_train.c ../lib/doublefann.o
	$(COMPILE_DOUBLE)

simple_test: simple_test.c ../lib/doublefann.o
	$(COMPILE_DOUBLE)

simple_train: simple_train.c ../lib/doublefann.o
	$(COMPILE_DOUBLE)

steepness_train: steepness_train.c ../lib/doublefann.o
	$(COMPILE_DOUBLE)

xor_test_float: xor_test.c fann_trained.c ../lib/embedded-float.o
	$(BUILD_FLOAT) fann_trained.c -c -o fann_trained.o
	$(BUILD_FLOAT) ../lib/embedded-float.o fann_trained.o -o xor_test_float xor_test.c -lm -lpthread
	$(STRIP) xor_test_float

xor_test_fixed: xor_test.c ../lib/fixedfann.o
	$(COMPILE_FIXED)

xor_train: xor_train.c ../lib/doublefann.o
	$(COMPILE_DOUBLE)

and_train: and_train.c ../lib/doublefann.o
	$(COMPILE_DOUBLE)

add_train: add_train.c ../lib/doublefann.o
	$(COMPILE_DOUBLE)

.PHONY: clean
clean:
	rm -fv $(BINS) floatfp16_fann fp16fp16_fann
