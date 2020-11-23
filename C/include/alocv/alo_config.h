#ifndef WENDA_ALO_CONFIG_H_INCLUDED
#define WENDA_ALO_CONFIG_H_INCLUDED

#if defined(MATLAB_MEX_FILE) || defined(MKL_ILP64)
#include "stddef.h"
typedef ptrdiff_t blas_size;
#else
typedef int blas_size;
#endif

#endif
