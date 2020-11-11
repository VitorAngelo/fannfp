# fannfp

This library was based on the Fast Artificial Neural Network Library.
Please visit http://leenissen.dk/fann/ for the original software, which
is copyrighted 2003-2016 by Steffen Nissen (steffen.fann@gmail.com).

This is a subset of the original code, which kept most of its strucure,
but was changed to allow tests with different numeric representations and
numerical approximations. The new features can be summarized as follows:

* A complete abstraction of all arithmetic operations and conversions
* Restructuring of the ANN internal representation (structures and methods)
* Added support for POSIX Multi-threading
* Removal of unwanted or broken features
* Implementation of detailed operation statistics
* Added compilation option optimized for embedded targets (without statistics and terminal IO)
* Creation of a flexible binary with all ANN definitions selectable at runtime
* Adaptation of RProp to conform to the original iRProp-
* Added support for RMSProp and normalized initialization
* Added support for ReLU activation and Softmax outputs

It currently builds cleanly on Debian based Linux distributions and is
tested on Intel PCs and ARM boards.

