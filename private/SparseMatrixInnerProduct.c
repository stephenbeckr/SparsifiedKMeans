/*
 * [innerProd, normX2] = SparseMatrixInnerProduct(X,c);
 *  for a p x n sparse matrix "X" and a p x 1 (or 1xp) dense vector c
 *   (where "c" represents a cluster center)
 *  returns a 1 x n vector "innerProd" and "norm2" such that
 *  if ind=find( X(:,i) ) then
 *      innerProd(i)     =  X(ind,i)'*c(ind);
 *      normX2(i)        =  X(ind,i)'*X(ind,i);
 *  
 *
 * Compile this as:
 *      mex -largeArrayDims SparseMatrixInnerProduct.c
 *
 * Code by Stephen Becker, stephen.becker@colorado.edu
 * May 23 2016
 *
 * */


#if defined(__GNUC__) && !(defined(__clang__)) && defined(NEEDS_UCHAR)
#include <uchar.h>
#endif
#include <math.h>
#include "mex.h"


#define XMATRIX 0
#define CENTER 1
#define INNERPRODUCT 0
#define NORMX 1

void printUsage() {
    mexPrintf("[innerProd, normX2] = SparseMatrixInnerProduct(X,c)\n");
    mexPrintf("  returns (if  ind=find( X(:,i) ) ) \n");
    mexPrintf("      innerProd(i)     =  X(ind,i)'*c(ind);\n");
    mexPrintf(" and  normX2 = sum( X.^2, 1 ).\n");
    mexPrintf("  X should be a sparse p x n matrix\n");
}

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
     
{ 
    double *innerProd, *normX, *center, *x;
    double nrm, inrProd;
    mwIndex *ir, *jc;
    mwSize p, n, i, j;
    
    /* Check for proper number of arguments */
    if (nrhs != 2) {
        printUsage();
        mexErrMsgIdAndTxt( "MATLAB:mexFile:invalidNumInputs",
                "Two input arguments required.");
    } else if (nlhs > 2) {
        printUsage();
        mexErrMsgIdAndTxt( "MATLAB:mexFile:maxlhs",
                "Too many output arguments, needs 1 or 2 outputs.");
    }
    
    if (!(mxIsSparse(prhs[XMATRIX]))) {
        printUsage();
        mexErrMsgTxt("Requires first input to be a sparse matrix");
    }

    p  = mxGetM(prhs[XMATRIX]);
    n  = mxGetN(prhs[XMATRIX]);
    ir = mxGetIr(prhs[XMATRIX]);      /* Row indexing      */
    jc = mxGetJc(prhs[XMATRIX]);      /* Column count      */
    x  = mxGetPr(prhs[XMATRIX]);      /* Non-zero elements */
    
    if (mxGetM( prhs[CENTER] )==1){
        if (mxGetN( prhs[CENTER] ) != n)
            mexErrMsgTxt("Center vector muyst be 1xn or nx1");
    }else if (mxGetM( prhs[CENTER] )==n){
        if (mxGetN( prhs[CENTER] ) != 1)
            mexErrMsgTxt("Center vector muyst be 1xn or nx1");
    }
    center         = mxGetPr( prhs[CENTER] );
    
    plhs[INNERPRODUCT] = mxCreateDoubleMatrix( (mwSize)1, (mwSize)n, mxREAL);
    innerProd       = mxGetPr( plhs[INNERPRODUCT] );
    plhs[NORMX]     = mxCreateDoubleMatrix( (mwSize)1, (mwSize)n, mxREAL);
    normX           = mxGetPr( plhs[NORMX] );
    
    
    /* Loop through columns of X */
    for ( i=0; i < n; i++ ) {
        nrm = 0.;
        inrProd = 0.;
        for ( j=jc[i]; j<jc[i+1]; j++ ){
            /*innerProd[i] += x[j]*center[ir[j]];
            normX[i] += x[j]*x[j]; */
            /* For some reason, the above code is quite slow! Use it
             * with these temporary variables instead */
            inrProd += x[j]*center[ir[j]];
            nrm     += x[j]*x[j];
        }
        normX[i] = nrm;
        innerProd[i] = inrProd;
    }

}
