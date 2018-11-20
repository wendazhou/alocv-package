#ifndef WENDA_BLAS_CONFIGURATION_H_INCLUDED
#define WENDA_BLAS_CONFIGURATION_H_INCLUDED

#include <stddef.h>

#ifdef USE_MKL

#define MKL_DIRECT_CALL

#include <mkl.h>
#include <mkl_blas.h>
#include <mkl_lapack.h>

#elif USE_R
#include <R_ext/BLAS.h>
#include <R_ext/Lapack.h>

// R may use F77 convention with additionall underscore at the end.
// Redefine the functions so that we pick up the right names.
#define drot F77_CALL(drot)
#define drotg F77_CALL(drotg)
#define daxpy F77_CALL(daxpy)
#define dscal F77_CALL(dscal)
#define dlacpy F77_CALL(dlacpy)
#define dcopy F77_CALL(dcopy)
#define dgemm F77_CALL(dgemm)
#define dtrsm F77_CALL(dtrsm)
#define dtrmm F77_CALL(dtrmm)
#define dsymm F77_CALL(dsymm)
#define ddot F77_CALL(ddot)
#define dgemv F77_CALL(dgemv)
#define dsymv F77_CALL(dsymv)
#define dpotrf F77_CALL(dpotrf)
#define dgels F77_CALL(dgels)
#define dgeqrf F77_CALL(dgeqrf)
#define dormqr F77_CALL(dormqr)
#define dtrtri F77_CALL(dtrtri)
#define dsyrk F77_CALL(dsyrk)

#define ALOCV_LAPACK_NO_RFP

#else
#include <blas.h>
#include <lapack.h>
#endif

#ifdef USE_MKL
static inline void* blas_malloc(blas_size alignment, blas_size size) {
    return mkl_malloc(size, alignment);
}

static inline void blas_free(void* ptr) {
    mkl_free(ptr);
}
#elif MATLAB_MEX_FILE
#include "mex.h"

static inline void* blas_malloc(blas_size alignment, blas_size size) {
    return mxMalloc(size);
}

static inline void blas_free(void* ptr) {
    mxFree(ptr);
}
#else

#if defined(_WIN32) || defined(_WIN64)

// on windows use platform-specific _aligned_malloc
#include <malloc.h>
static inline void* blas_malloc(blas_size alignment, blas_size size) {
    return _aligned_malloc(size, alignment);
}

static inline void blas_free(void* ptr) {
    return _aligned_free(ptr);
}

#else // _WIN32 || _WIN64
#include <stdlib.h>

static inline void* blas_malloc(blas_size alignment, blas_size size) {
    return aligned_alloc(alignment, alignment * (size + alignment - 1) / alignment);
}

static inline void blas_free(void* ptr) {
    free(ptr);
}
#endif // _WIN32 || _WIN64

#endif // platform specific allocation

#ifdef WENDA_RFP_ENABLE_SHIMS
// If the platform requires RFP shims, we also include the file for compatibility.
#include "rfp_shims.h"
#endif

#ifdef __cplusplus
#include <memory>

template<typename T>
struct blas_deleter {
	void operator()(T* ptr) const {
		blas_free(ptr);
	}
};

template<typename T>
using unique_aligned_array = std::unique_ptr<T[], blas_deleter<T>>;

template<typename T>
unique_aligned_array<T> blas_unique_alloc(blas_size alignment, blas_size count) noexcept {
	return unique_aligned_array<T>(static_cast<T*>(blas_malloc(alignment, count * sizeof(T))));
}
#endif

#endif // WENDA_BLAS_CONFIGURATION_H_INCLUDED
