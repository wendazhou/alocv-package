#ifndef WENDA_BLAS_CONFIGURATION_H_INCLUDED
#define WENDA_BLAS_CONFIGURATION_H_INCLUDED

#include "stddef.h"

#ifdef USE_MKL

#define MKL_DIRECT_CALL

#include "mkl.h"
#include "mkl_blas.h"
#include "mkl_lapack.h"
#elif USE_R
#include "R_ext/blas.h"
#include "R_ext/lapack.h"
#else
#include "blas.h"
#include "lapack.h"
#endif

#ifdef USE_MKL
inline void* blas_malloc(size_t alignment, size_t size) {
    return mkl_malloc(size, alignment);
}

inline void blas_free(void* ptr) {
    mkl_free(ptr);
}
#elif MATLAB_MEX_FILE
#include "mex.h"

inline void* blas_malloc(size_t alignment, size_t size) {
    return mxMalloc(size);
}

inline void blas_free(void* ptr) {
    return mxFree(ptr);
}
#else
#include "stdlib.h"

inline void* blas_malloc(size_t alignment, size_t size) {
    return aligned_alloc(alignment, alignment * (size + alignment - 1) / alignment);
}

inline void blas_free(void* ptr) {
    free(ptr);
}
#endif

#endif