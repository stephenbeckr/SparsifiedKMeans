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
#define BETA 2

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
    double *distance, *center, *x;
    register double dist; /* unlikely to have much effect */
    double *distArray; /* for K>1 case, helper variables */
    double distHelper[3]; /* for 1<K<=3 case, keep it on the stack*/
    double beta =0.;
    int    use_beta = 0;
    mwIndex *ir, *jc;
    mwSize p, n, i, j, k, K;
    
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
        beta     = mxGetScalar( prhs[BETA] );
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
    
    /* ** old code, required K=1 **
    if (mxGetM( prhs[CENTER] )==1){
        if (mxGetN( prhs[CENTER] ) != p) {
            printUsage();
            mexErrMsgTxt("Center vector must be 1xp or px1");
        }
    }else{
        if (mxGetM( prhs[CENTER] )==p){
            if (mxGetN( prhs[CENTER] ) != 1) {
                printUsage();
                mexErrMsgTxt("Center vector must be 1xp or px1 (error 2)");
            }
        } else {
            printUsage();
            mexErrMsgTxt("Center vector must be 1xp or px1 (error 3)");
        }
    }
    */
    center         = mxGetPr( prhs[CENTER] );

    /* July 2016, allowing center to be p x k instead of just p x 1 */
    if (mxGetM( prhs[CENTER] )!=p){
        printUsage();
        mexErrMsgTxt("Center vector must be or pxk, but this vector did not have p rows");
    }
    K = mxGetN( prhs[CENTER] ); /* often this is 1 */
    if (K>3) {
        distArray = (double*)mxMalloc( K * sizeof(double) );
    }
    
    plhs[DISTANCE] = mxCreateDoubleMatrix( (mwSize)K, (mwSize)n, mxREAL);
    distance       = mxGetPr( plhs[DISTANCE] );
    
    
    /* Loop through columns of X */
    if (use_beta==1) {
        if (K!=1)
            mexErrMsgTxt("Have not yet implemented case for using 'beta' with p x k (k!=`) centers");
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
        /* For speed, explicitly unroll a few cases */
        switch (K) {
            case 1 :
                for ( i=0; i < n; i++ ) {
                    dist = 0.;
                    for ( j=jc[i]; j<jc[i+1]; j++ ){
                        dist += ( x[j] - center[ ir[j] ] )*( x[j] - center[ ir[j] ] ); 
                    }
                    distance[i] = sqrt(dist);
                }
                break;
            case 2:
                for ( i=0; i < n; i++ ) {
                    distHelper[0] = 0.;
                    distHelper[1] = 0.;
                    for ( j=jc[i]; j<jc[i+1]; j++ ){
                        distHelper[0] += ( x[j] - center[ ir[j] ] )*( x[j] - center[ ir[j] ] ); 
                        distHelper[1] += ( x[j] - center[ p + ir[j] ] )*( x[j] - center[ p + ir[j] ] ); 
                    }
                    distance[i*K  ] = sqrt(distHelper[0]);
                    distance[i*K+1] = sqrt(distHelper[1]);
                }
                break;
            case 3:
                for ( i=0; i < n; i++ ) {
                    distHelper[0] = 0.;
                    distHelper[1] = 0.;
                    distHelper[2] = 0.;
                    for ( j=jc[i]; j<jc[i+1]; j++ ){
                        distHelper[0] += ( x[j] - center[       ir[j] ] )*( x[j] - center[       ir[j] ] ); 
                        distHelper[1] += ( x[j] - center[   p + ir[j] ] )*( x[j] - center[   p + ir[j] ] ); 
                        distHelper[2] += ( x[j] - center[ 2*p + ir[j] ] )*( x[j] - center[ 2*p + ir[j] ] ); 
                    }
                    distance[i*K  ] = sqrt(distHelper[0]);
                    distance[i*K+1] = sqrt(distHelper[1]);
                    distance[i*K+2] = sqrt(distHelper[2]);
                }
                break;
            default:
                for ( i=0; i < n; i++ ) {
                    for (k=0; k<K; k++ )
                        distArray[k] = 0.;
                    for ( j=jc[i]; j<jc[i+1]; j++ ){
                        for (k=0; k<K; k++ )
                            /*distance[i*K+k] += ( x[j] - center[ k*p + ir[j] ] )*( x[j] - center[ k*p + ir[j] ] ); */
                            distArray[k] += ( x[j] - center[ k*p + ir[j] ] )*( x[j] - center[ k*p + ir[j] ] ); 
                    }
                    for (k=0; k<K; k++ )
                        /*distance[i*K+k] = sqrt(distance[i*K+k]);*/
                        distance[i*K+k] = sqrt(distArray[k]);
                }
                break;
        } /* end of switch */
    }
    if (K>3)
        mxFree( distArray );

}
