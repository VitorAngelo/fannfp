# fannfp

This library was based on the Fast Artificial Neural Network Library.
Please visit http://leenissen.dk/fann/ for the original software, which
is copyrighted 2003-2016 by Steffen Nissen (steffen.fann@gmail.com).

FP emulation code was cherry-picked and adapted from the SoftFloat library
(http://www.jhauser.us/arithmetic/SoftFloat.html) and the Posit 16 was based on the
the SoftPosit library (https://gitlab.com/cerlane/SoftPosit/), which in turn was
also based on an earlier version of SoftFloat.

This is a subset of the original code, which kept most of its strucure,
but was changed to allow tests with different numeric representations and
numerical approximations. The new features can be summarized as follows:

* A complete abstraction of all arithmetic operations and conversions
* Separate abstract types for inference and training
* Fast FP approximations for exp(x), 1/sqrt(x) and y/sqrt(x)
* Restructuring of the ANN internal representation (structures and methods)
* Added support for POSIX Multi-threading
* Removal of unwanted or broken features
* Implementation of detailed operation statistics
* Added compilation option optimized for embedded targets (without statistics and terminal IO)
* Trimmed compilation only with inference capabilities
* Creation of a flexible binary with all ANN definitions selectable at runtime
* Adaptation of RProp to conform to the original iRProp-
* Added support for RMSProp and normalized initialization
* Added support for ReLU activation and Softmax outputs

Native builds: Tested on several Intel CPUs and ARM Cortex A53 (Rasp. Pi)

Cross compilation: Cortex M3 (fixed point only)

Emulated types and operations:

* IEEE FP 32
* IEEE FP 16 
* Aprox. FP 16 with dynamic bias
* Posit 16

Native types:

* IEEE FP 64
* IEEE FP 32
* IEEE FP 16 (ARM Cortex A53 only)
* fixed point (integer ops, complex functions are interpolated)

