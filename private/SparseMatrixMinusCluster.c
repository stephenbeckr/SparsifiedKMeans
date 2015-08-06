/*
 * dist = SparseMatrixMinusCluster(X,c);
 *  for a p x n matrix "X" and a p x 1 (or 1xp) vector c
 *   (where "c" represents a cluster center)
 *  returns a 1 x n vector "dist" such that
 *  if ind=find( X(:,i) ) then
 *  dist(i)     =  norm( X(ind,j) - c(ind) );
 *
 * Compile this as:
 *      mex -largeArrayDims SparseMatrixMinusCluster.c
 *
 * Code by Stephen Becker, stephen.becker@colorado.edu
 * July 22 2015
 *
 * */


#if defined(__GNUC__) && !(defined(__clang__)) && defined(NEEDS_UCHAR)
#include <uchar.h>
#endif
#include <math.h>
#include "mex.h"


#define XMATRIX 0
#define CENTER 1
#define DISTANCE 0

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
     
{ 
    double *distance, dist, *center, *x;
    mwIndex *ir, *jc;
    mwSize p, n, i, j;
    
    /* Check for proper number of arguments */
    if (nrhs != 2) {
        mexErrMsgIdAndTxt( "MATLAB:mexFile:invalidNumInputs",
                "Two input arguments required.");
    } else if (nlhs > 1) {
        mexErrMsgIdAndTxt( "MATLAB:mexFile:maxlhs",
                "Too many output arguments.");
    }
    
    if (!(mxIsSparse(prhs[XMATRIX])))
        mexErrMsgTxt("Requires first input to be a sparse matrix");

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
    
    plhs[DISTANCE] = mxCreateDoubleMatrix( (mwSize)1, (mwSize)n, mxREAL);
    distance       = mxGetPr( plhs[DISTANCE] );
    
    
    /* Loop through columns of X */
    
    for ( i=0; i < n; i++ ) {
        dist = 0.;
        for ( j=jc[i]; j<jc[i+1]; j++ ){
            dist += ( x[j] - center[ ir[j] ] )*( x[j] - center[ ir[j] ] );
        }
        distance[i] = sqrt(dist);
    }

}
