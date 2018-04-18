#include "mex.h"
#include "alocv/alo_lasso.h"

void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {
    if(nlhs > 2)  {
        mexErrMsgIdAndTxt("alocv:alo_lasso_mex:nrhs", "Only one or two output supported.");
    }

    double tolerance;

    if(nrhs == 3) {
        tolerance = 1e-5;
    }
    else if(nrhs == 4) {
        tolerance = mxGetScalar(prhs[3]);
    }
    else {
        mexErrMsgIdAndTxt("alocv:alo_lasso_mex:nlhs", "Must input three or four arguments.");
    }

    double* A = mxGetDoubles(prhs[0]);
    double* y = mxGetDoubles(prhs[1]);
    double* B = mxGetDoubles(prhs[2]);
    mwSize n = mxGetM(prhs[0]);
    mwSize p = mxGetN(prhs[0]);

    if(mxGetM(prhs[2]) != p) {
        mexErrMsgIdAndTxt("alocv:alo_lasso_mex:compatibility", "The regression matrix and fitted values dimension do not match.");
    }

    if(mxGetM(prhs[1]) * mxGetN(prhs[1]) != n) {
        mexErrMsgIdAndTxt("alocv:alo_lasso_mex:y_compatibility", "The regression matrix and observation vector dimension do not match.");
    }

    mwSize num_tuning = mxGetN(prhs[2]);

    plhs[0] = mxCreateDoubleMatrix(num_tuning, 1, mxREAL);
    double* alo = mxGetDoubles(plhs[0]);

    double* leverage = NULL;

    if (nlhs == 2) {
        plhs[1] = mxCreateDoubleMatrix(n, num_tuning, mxREAL);
        leverage = mxGetDoubles(plhs[1]);
    }

    lasso_compute_alo_d(n, p, num_tuning, A, n, B, p, y, 1, tolerance, alo, leverage);
}