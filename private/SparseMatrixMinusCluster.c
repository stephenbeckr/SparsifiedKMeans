/*
 * dist = SparseMatrixMinusCluster(X,c);
 *  for a p x n matrix "X" and a p x 1 (or 1xp) vector c
 *   (where "c" represents a cluster center)
 *  returns a 1 x n vector "dist" such that
 *  if ind=find( X(:,i) ) then
 *  dist(i)     =  norm( X(ind,j) - c(ind) );
 *  
 *  ... = SparseMatrixMinusCluster(X,c,beta)
 *  computes
 *  dist(i)     = sqrt( X(ind,j)^2 -2*beta*X(ind,j)c(ind) + c(ind)^2)
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

void printUsage() {
    mexPrintf(" dist = SparseMatrixMinusCluster(X,c) \n");
    mexPrintf("  returns dist(i) = norm( X(ind,j) - c(ind) )\n");
    mexPrintf("  where ind=find( X(:,i) ).\n");
    mexPrintf("    dist is a 1 x n vector\n");
    mexPrintf("  Note that this is NOT norm( X(:,j) - c ).\n");
    mexPrintf("  X should be a sparse p x n matrix,\n");
    mexPrintf("   and c should be a dense p x 1 vector\n");
}

void mexFunction( int nlhs, mxArray *plhs[], 
		  int nrhs, const mxArray*prhs[] )
     
{ 
    double *distance, dist, *center, *x;
    double beta =0.;
    int    use_beta = 0;
    mwIndex *ir, *jc;
    mwSize p, n, i, j;
    
    /* Check for proper number of arguments */
    if ( (nrhs != 2) & (nrhs!=3) ) {
        printUsage();
        mexErrMsgIdAndTxt( "MATLAB:mexFile:invalidNumInputs",
                "Two input arguments required.");
    } else if (nlhs > 1) {
        printUsage();
        mexErrMsgIdAndTxt( "MATLAB:mexFile:maxlhs",
                "Too many output arguments.");
    }
    if (nrhs==3) {
        use_beta = 1;
        beta     = mxGetScalar( prhs[2] );
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
        if (mxGetN( prhs[CENTER] ) != p) {
            printUsage();
            mexErrMsgTxt("Center vector must be 1xn or nx1");
        }
    }else{
        if (mxGetM( prhs[CENTER] )==p){
            if (mxGetN( prhs[CENTER] ) != 1) {
                printUsage();
                mexErrMsgTxt("Center vector must be 1xn or nx1 (error 2)");
            }
        } else {
            printUsage();
            mexErrMsgTxt("Center vector must be 1xn or nx1 (error 3)");
        }
    }
    center         = mxGetPr( prhs[CENTER] );
    
    plhs[DISTANCE] = mxCreateDoubleMatrix( (mwSize)1, (mwSize)n, mxREAL);
    distance       = mxGetPr( plhs[DISTANCE] );
    
    
    /* Loop through columns of X */
    if (use_beta==1) {
        beta *= -2.;
        for ( i=0; i < n; i++ ) {
            dist = 0.;
            for ( j=jc[i]; j<jc[i+1]; j++ ){
                dist += x[j]*x[j] + beta*x[j]*center[ ir[j] ] + center[ir[j]]*center[ir[j]];
            }
/*             distance[i] = dist;  */
            distance[i] = sqrt(dist); 
        }
    } else {
        for ( i=0; i < n; i++ ) {
            dist = 0.;
            for ( j=jc[i]; j<jc[i+1]; j++ ){
                dist += ( x[j] - center[ ir[j] ] )*( x[j] - center[ ir[j] ] ); 
            }
            distance[i] = sqrt(dist);
        }
    }

}
