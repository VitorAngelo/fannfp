## Objects handled by generic rule
GOBJS := fann_ap_f16.o fann_ieee_f16.o print.o
GOBJS += fixedfann.o
GOBJS += floatfann.o
GOBJS += doublefann.o

## Objects handled by specific rules
EOBJS := embedded-float.o
EOBJS += embedded-fixed.o
EOBJS += softfann-ap.o
EOBJS += softfann-ieee.o
EOBJS += softfann-hwf16.o
EOBJS += softfann-posit16.o
EOBJS += bfloat16.o
EOBJS += floatunion.o

## Objects handled only in ARM Cortex-A53 (natively)
POBJS = floatfp16.o fp16fp16.o

## Cross-compile targets
COBJS = embedded-cortex-m3.o

CFLAGS = -Wall -pipe
CFLAGS += -Wmissing-prototypes -Wstrict-prototypes -Werror-implicit-function-declaration -Wundef -Wunused-parameter
CFLAGS += -O3
#CFLAGS += -ftree-vectorize -funsafe-math-optimizations

DFLAGS = -g
DFLAGS += -pg

ARCH_X86 = -march=westmere -mtune=westmere

#ARCH_PI = -march=armv6 -mtune=cortex-a53 -mfp16-format=alternative -mfpu=neon-fp16
ARCH_PI = -mcpu=cortex-a53 -march=armv8-a+crc -mtune=cortex-a53 -mfpu=crypto-neon-fp-armv8 -mfloat-abi=hard
ARCH_PI += -mfp16-format=alternative

ARMCPUFLAGS = -mcpu=cortex-m3 -mthumb
ARMCFLAGS = -Wall -Wextra -Werror -O3 -MD $(ARMCPUFLAGS) -DSTM32F1 -I../../sw/3rd/libopencm3/include
ARMLDFLAGS = $(ARMCPUFLAGS) -nostartfiles -L../../sw/3rd/libopencm3/lib -Wl,-T,bluepill.ld
ARMLDLIBS = -lopencm3_stm32f1 -lc -lnosys

x86: ARCH = $(ARCH_X86)
x86: $(EOBJS) $(GOBJS)

pi: ARCH = $(ARCH_PI)
pi: $(POBJS) $(GOBJS) $(EOBJS)

dx86: ARCH = $(ARCH_X86)
dx86: CFLAGS += $(DFLAGS)
dx86: $(EOBJS) $(GOBJS)

dpi: ARCH = $(ARCH_PI)
dpi: CFLAGS += $(DFLAGS)
dpi: $(POBJS) $(GOBJS) $(EOBJS)

-include $(GOBJS:.o=.d)
-include $(EOBJS:.o=.d)
-include $(POBJS:.o=.d)

x86dep: $(EOBJS:.o=.d) $(GOBJS:.o=.d)

pidep: $(EOBJS:.o=.d) $(GOBJS:.o=.d) $(POBJS:.o=.d)

floatfp16.o: floatfp16.d
	gcc -c $(CFLAGS) $(ARCH_PI) -D_GCC_ARM_F16_FF floatfann.c -o floatfp16.o

floatfp16.d: floatfann.c
	gcc -MM $(CFLAGS) -D_GCC_ARM_F16_FF floatfann.c | sed 's,floatfann.o:,floatfp16.o:,' > floatfp16.d

fp16fp16.o: fp16fp16.d
	gcc -c $(CFLAGS) $(ARCH_PI) -D_GCC_ARM_F16_BP floatfann.c -o fp16fp16.o

fp16fp16.d: floatfann.c
	gcc -MM $(CFLAGS) -D_GCC_ARM_F16_BP floatfann.c | sed 's,floatfann.o:,fp16fp16.o:,' > fp16fp16.d

floatunion.o: floatunion.d
	gcc -c $(CFLAGS) $(ARCH) -D_FLOAT_UNION floatfann.c -o floatunion.o

floatunion.d: floatfann.c
	gcc -MM $(CFLAGS) -D_FLOAT_UNION floatfann.c | sed 's,floatfann.o:,floatunion.o:,' > floatunion.d

bfloat16.o: bfloat16.d
	gcc -c $(CFLAGS) $(ARCH) -D_BFLOAT16 floatfann.c -o bfloat16.o

bfloat16.d: floatfann.c
	gcc -MM $(CFLAGS) -D_BFLOAT16 floatfann.c | sed 's,floatfann.o:,bfloat16.o:,' > bfloat16.d

embedded-cortex-m3.o: embedded-cortex-m3.d
	arm-none-eabi-gcc -c $(ARMCFLAGS) -DFANN_EMBEDDED fixedfann.c -o embedded-cortex-m3.o

embedded-cortex-m3.d: fixedfann.c
	arm-none-eabi-gcc -MM $(ARMCFLAGS) -DFANN_EMBEDDED fixedfann.c | sed 's,fixedfann.o:,embedded-cortex-m3:.o,' > embedded-cortex-m3.d

embedded-float.o: embedded-float.d
	gcc -c $(CFLAGS) $(ARCH) -DFANN_EMBEDDED floatfann.c -o embedded-float.o

embedded-float.d: floatfann.c
	gcc -MM $(CFLAGS) -DFANN_EMBEDDED floatfann.c | sed 's,floatfann.o:,embedded-float.o:,' > embedded-float.d

embedded-fixed.o: embedded-fixed.d
	gcc -c $(CFLAGS) $(ARCH) -DFANN_EMBEDDED fixedfann.c -o embedded-fixed.o

embedded-fixed.d: fixedfann.c
	gcc -MM $(CFLAGS) -DFANN_EMBEDDED fixedfann.c | sed 's,fixedfann.o:,embedded-fixed.o:,' > embedded-fixed.d

softfann-ap.o: softfann-ap.d
	gcc -c $(CFLAGS) $(ARCH) -DSWF16_AP softfann.c -o softfann-ap.o

softfann-ap.d: softfann.c
	gcc -MM $(CFLAGS) -DSWF16_AP softfann.c | sed 's,softfann.o:,softfann-ap.o:,' > softfann-ap.d

softfann-ieee.o: softfann-ieee.d
	gcc -c $(CFLAGS) $(ARCH) -DSWF16_IEEE softfann.c -o softfann-ieee.o

softfann-ieee.d: softfann.c
	gcc -MM $(CFLAGS) -DSWF16_IEEE softfann.c | sed 's,softfann.o:,softfann-ieee.o:,' > softfann-ieee.d

softfann-hwf16.o: softfann-hwf16.d
	gcc -c $(CFLAGS) $(ARCH) -DHWF16 softfann.c -o softfann-hwf16.o

softfann-hwf16.d: softfann.c
	gcc -MM $(CFLAGS) -DHWF16 softfann.c | sed 's,softfann.o:,softfann-hwf16.o:,' > softfann-hwf16.d
softfann-posit16.o: softfann-posit16.d
	gcc -c $(CFLAGS) $(ARCH) -DPOSIT16 softfann.c -o softfann-posit16.o

softfann-posit16.d: softfann.c
	gcc -MM $(CFLAGS) -DPOSIT16 softfann.c | sed 's,softfann.o:,softfann-posit16.o:,' > softfann-posit16.d

softposit.o: softposit.c
	gcc -c $(CFLAGS) $(ARCH) softposit.c -o softposit.o 

%.o: %.c
	gcc -c $(CFLAGS) $(ARCH) $*.c -o $*.o

%.d: %.c
	gcc -MM $(CFLAGS) $*.c > $*.d

# remove compilation products
.PHONY: clean
clean:
	rm -fv *.a *.o *.s

.PHONY: depclean
depclean:
	rm -fv *.a *.o *.s *.d

#	gcc -S -fverbose-asm -g -c $(CFLAGS) $*.c -o $*.s
#	as -alhnd $*.s > $*.lst
# comments regard 1 flag at a time, comparing double x double
#CFLAGS += -fno-math-errno
#CFLAGS += -funsafe-math-optimizations # slower convergence in 1 best generalization in 2
#CFLAGS += -ffinite-math-only
#CFLAGS += -fno-rounding-math # slower convergence in 1 best generalization in 2
#CFLAGS += -fno-signaling-nans
#CFLAGS += -fno-signed-zeros

#-Wstrict-overflow=4

# gcc (Ubuntu 5.4.0-6ubuntu1~16.04.4) 5.4.0 20160609

# -O3 optimizations:
#CFLAGS += -finline-functions # include causes considerable diff
#CFLAGS += -funswitch-loops -fpredictive-commoning -fgcse-after-reload -fipa-cp-clone
#CFLAGS += -ftree-loop-distribute-patterns -ftree-slp-vectorize -fvect-cost-model -ftree-partial-pre

# -ftree-loop-vectorize

# -O2 optimizations:
#CFLAGS += -O2
#CFLAGS += -ftree-pre # remove causes small float diff
#CFLAGS += -fstrict-aliasing # remove causes considerable diff
#CFLAGS += -fstrict-overflow # remove causes considerable diff
#CFLAGS += -fgcse # remove causes considerable diff
#CFLAGS += -finline-small-functions # remove causes considerable diff

#CFLAGS += -fthread-jumps -falign-functions -falign-jumps -falign-loops -falign-labels -fcaller-saves
#CFLAGS += -fcrossjumping -fcse-follow-jumps -fcse-skip-blocks -fdelete-null-pointer-checks
#CFLAGS += -fdevirtualize -fexpensive-optimizations -ftree-vrp
#CFLAGS += -fgcse-lm -fhoist-adjacent-loads -findirect-inlining -fipa-cp
#CFLAGS += -fipa-sra -foptimize-sibling-calls
#CFLAGS += -foptimize-strlen -fpartial-inlining -fpeephole2 -freorder-blocks -freorder-blocks-and-partition
#CFLAGS += -freorder-functions -frerun-cse-after-loop -fsched-interblock -fsched-spec -fschedule-insns
#CFLAGS += -fschedule-insns2 -ftree-builtin-call-dce -ftree-switch-conversion -ftree-tail-merge

# -flra-remat -fipa-icf -fipa-cp-alignment -fipa-ra -fdevirtualize-speculatively -fisolate-erroneous-paths-dereference

#CFLAGS += -march=westmere -mtune=westmere
#CFLAGS += -march=core2
#CFLAGS += -march=native
#CFLAGS += -O1
#gcc -march=native -Q --help=target | grep march
# notebook = sandybridge
# xeon = westmere
# pi = armv6

#Thus in summary:
#1. -mfpu=neon implies the presence of the float16x(4/8) types and all the intrinsics that treat these values as bags of bits.
#2. -mfpu=neon-fp16 implies the presence of the vcvt* intrinsics that are needed for the float16 types.


