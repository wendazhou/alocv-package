#ifndef WENDA_RFP_SHIMS_H_INCLUDED
#define WENDA_RFP_SHIMS_H_INCLUDED

/*! Shims for BLAS/LAPACK operations in rectangular full packed format.
 *
 * Some platforms (e.g. R) do not provide these newer BLAS/LAPACK operations,
 * but they can be build from standard BLAS/LAPACK operations. We provide reference C
 * implementations here for compatibility.
 * 
 * These implementations are provided by default suffixed by "_shim", but when
 * WENDA_RFP_ENABLE_SHIMS is defined, they are defined under their original
 * name instead. This allows the use of such functions without code changes.
 * 
 */


#include "alocv/alo_config.h"
#include "blas_configuration.h"
#include <string.h>


#ifdef WENDA_RFP_ENABLE_SHIMS
#define SHIM_FUNCTION(X) X
#else
#define SHIM_FUNCTION(X) X ## _shim
#endif

inline const char* transpose_value(bool value, bool transpose) {
    return value ^ transpose ? "T" : "N";
}

/*! Reference implementation of dsfrk (symmetric rank-k update in rectangular full packed format).
 *
 * This function is a C implementation of the reference Fortran implementation found in LAPACK.
 *
 */
inline void SHIM_FUNCTION(dsfrk)(const char* transr, const char* uplo, const char* trans,
                  const blas_size* n, const blas_size* k, const double* alpha,
                  const double* A, const blas_size* lda, const double* beta,
                  double* C) {
    bool normaltransr = strcmp(transr, "N");
    bool lower = strcmp(uplo, "L");
    bool notrans = strcmp(trans, "N");

    blas_size n1, n2;

    if(lower) {
        n2 = *n / 2;
        n1 = *n - n2;
    } else {
        n1 = *n / 2;
        n2 = *n - n1;
    }

    // output offsets into C array.
    blas_size co1, co2, co3;

    if(*n % 2 != 0) {
        if(normaltransr) {
            co1 = lower ? 0 : n2;
            co2 = lower ? *n : n1;
            co3 = lower ? n1 : 0;
        } else {
            co1 = lower ? 0 : n2 * n2;
            co2 = lower ? 1 : n1 * n2;
            co3 = lower ? n1 * n1 : 0;
        }
    } else {
        if(normaltransr) {
            co1 = lower ? 1 : n1 + 1;
            co2 = lower ? 0 : n1;
            co3 = lower ? n1 : 0;
        } else {
            co1 = lower ? n1 : n1 * (n1 + 1);
            co2 = lower ? 0 : n1 * n1;
            co3 = lower ? n1 * (n1 + 1) : 0;
        }
    }

    const char* t1 = transpose_value(false, notrans ^ lower);
    const char* t2 = transpose_value(true, notrans ^ lower);
    blas_size stride_trans = notrans ? 1 : *lda;

    blas_size ldc = normaltransr ? *n : (*n + 1) / 2;

    dsyrk(normaltransr ? "L" : "U", t1, &n1, k, alpha, A, lda, beta, C + co1, &ldc);
    dsyrk(normaltransr ? "U" : "L", t1, &n2, k, alpha, A + n1 * stride_trans, lda, beta, C + co2, &ldc);
    dgemm(t1, t2, &n2, &n1, k, alpha, A + n1 * stride_trans, lda, A, lda, beta, C + co3, &ldc);
}

#endif // WENDA_RFP_SHIMS_H_INCLUDED