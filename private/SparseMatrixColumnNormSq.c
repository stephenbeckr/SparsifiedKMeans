/*
 * [normX2] = SparseMatrixColumnNormSq(X);
 *  for a p x n sparse matrix "X" 
 *  returns a 1 x n vector "normX2" such that
 *  if ind=find( X(:,i) ) then
 *      normX2(i)        =  X(ind,i)'*X(ind,i);
 *  i.e.,
 *      normX2  = sum( X.^2, 1 )
 *  
 *
 * Compile this as:
 *      mex -largeArrayDims SparseMatrixColumNormSq.c
 *
 * Code by Stephen Becker, stephen.becker@colorado.edu
 * May 25 2016
 *
 * */


#if defined(__GNUC__) && !(defined(__clang__)) && defined(NEEDS_UCHAR)
#include <uchar.h>
#endif
#include <math.h>
#include "mex.h"


#define XMATRIX 0
#define NORMX 0

void printUsage() {
    mexPrintf("[normX2] = SparseMatrixColumnNormSq(X)\n");
    mexPrintf("  returns normX2 = sum( X.^2, 1 ), X should be p x n sparse\n");
}

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
     
{ 
    double *normX, *x;
    double nrm;
    mwIndex *ir, *jc;
    mwSize n, i, j;
    
    /* Check for proper number of arguments */
    if (nrhs != 1) {
        printUsage();
        mexErrMsgIdAndTxt( "MATLAB:mexFile:invalidNumInputs",
                "One input arguments required.");
    } else if (nlhs != 1) {
        printUsage();
        mexErrMsgIdAndTxt( "MATLAB:mexFile:maxlhs",
                "Too many output arguments, needs 1 output.");
    }
    
    if (!(mxIsSparse(prhs[XMATRIX]))) {
        printUsage();
        mexErrMsgTxt("Requires first input to be a sparse matrix");
    }

    n  = mxGetN(prhs[XMATRIX]);
    ir = mxGetIr(prhs[XMATRIX]);      /* Row indexing      */
    jc = mxGetJc(prhs[XMATRIX]);      /* Column count      */
    x  = mxGetPr(prhs[XMATRIX]);      /* Non-zero elements */
    
    
    plhs[NORMX]     = mxCreateDoubleMatrix( (mwSize)1, (mwSize)n, mxREAL);
    normX           = mxGetPr( plhs[NORMX] );
    
    
    /* Loop through columns of X */
    for ( i=0; i < n; i++ ) {
        nrm = 0.;
        for ( j=jc[i]; j<jc[i+1]; j++ ){
            nrm     += x[j]*x[j];
        }
        normX[i] = nrm;
    }

}
