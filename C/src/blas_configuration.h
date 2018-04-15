#ifndef WENDA_BLAS_CONFIGURATION_H_INCLUDED
#define WENDA_BLAS_CONFIGURATION_H_INCLUDED

#define USE_MKL

#ifdef USE_MKL
#include "mkl_blas.h"
#include "mkl_lapack.h"
#elif USE_R
#include "R_ext/blas.h"
#include "R_ext/lapack.h"
#else
#include "blas.h"
#include "lapack.h"
#endif

#endif