% alo_lasso_mex.m Help file for alo_lasso_mex MEX file
%
% alo_lasso_mex Computes the approximate-leave-one cross-validation for a LASSO regression
%
%  Syntax: [alo, h] = alo_lasso_mex(X, y, B, tolerance)
%
%  Inputs:
%   A - Predictor data: numeric matrix containing the predictors.
%   y - Response data: numeric matrix containing the responses.
%   B - Fitted Coefficients: numeric matrix containing the fitted response.
%   tolerance - Active set tolerance: positive scalar representing the value below which
%               fitted coefficients are considered zero.
%