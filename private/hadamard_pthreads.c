/* Hadamard Transform
   mex function to take hadamard transform

   Usage: w = hadamard_pthreads(x)
   x must be a REAL VALUED COLUMN VECTOR or MATRIX
   m = size(x,1) must be a POWER OF TWO

   Notes:
   1) This implementation uses exactly m*log2(m) additions/subtractions.
   2) This is symmetric and orthogonal. To invert, apply again and
      divide by vector length.

   Written by: Peter Stobbe, Caltech
   Email: stobbe@acm.caltech.edu
   Created: August 2008
   Edits by Stephen Becker, 2009--2014
 
   This "pthread" version by Stephen Becker, May 13 2014

      Note: in R2008b, Matlab added "fwht" and "ifwht" (the Fast Walsh-
          Hadamart Transform and the inverse) to its Signal Processing
          Toolbox.  With the default ordering and scaling, it's not
          equivalent to this, but you can change this with the following:
          y = length(x) * fwht( x, [], 'hadamard' );
          Then y should be the same as hadamard(x) up to roundoff.
          However, it appears that this code is faster than fwht.

 Update Stephen Becker, Feb 27 2014, fix compiling issue for Mac OS X
 Update Stephen Becker, Mar  3 2014, issue error if input data is sparse
 
*/

#include <stdlib.h>


/* SRB: Feb 27 2014, gcc-4.8 has problems with char16_t not being defined. 
 * This  seems to fix it
 * (and do this BEFORE including mex.h) */
/* See http://gcc.gnu.org/bugzilla/show_bug.cgi?id=56086#c4 */
#ifndef NO_UCHAR
#define UCHAR_OK
#endif
#if defined(__GNUC__) && !(defined(__clang__)) && defined(UCHAR_OK)
#include <uchar.h>
#endif



#include "mex.h"

#ifdef NTHREADS
#define NTHREADS_ NTHREADS
#else
#define NTHREADS_ 4
#endif

#include <pthread.h>
typedef struct info_struct {
    long    id;
    double      *y, *x;
    unsigned    length, n;
}Info_t ;

/* 
 y - output
 x - input
 m - length of vector
 */
void hadamard_apply_vector(double *y, double *x, unsigned m)
{
  unsigned bit, j, k;
  double temp;

  for (j = 0; j < m; j+=2) {
      k = j+1;
      y[j] = x[j] + x[k];
      y[k] = x[j] - x[k];
  }

  for (bit = 2; bit < m; bit <<= 1) {   
    for (j = 0; j < m; j++) {
        if( (bit & j) == 0 ) {
              k = j | bit;
              temp = y[j];
              y[j] = y[j] + y[k];
              y[k] = temp - y[k];
        }
    }
  }
}



/* 
 y - output
 x - input
 m - length of vectors (number of rows)
 n - number of vectors (number of columns)
 
 */
void hadamard_apply_matrix(double *y, double *x, unsigned m, unsigned n)
{
    unsigned j;
    for(j = 0; j < n; j++) {
        hadamard_apply_vector(y + j*m, x + j*m, m);
    }
}

/* Note this is of type (void *) and not (void)  */
void* worker( void* threadID){
    Info_t *info = (Info_t *) threadID;
    unsigned length     = info->length;
    unsigned n          = info->n;
    double  *x          = info->x;
    double  *y          = info->y;
    hadamard_apply_matrix(y, x, length, n);
    /* hadamard_apply_vector(y, x, length);*/
    pthread_exit(NULL);
}

void hadamard_apply_matrix_threads(double *y, double *x, unsigned m, unsigned n)
{
    unsigned j, jj, nn;
    unsigned nThreads;
    Info_t   *info;
    pthread_t *threads;
    
    
    if ( n==1 )
        hadamard_apply_vector(y, x, m);
    else {
        if (n <= NTHREADS_) {
            nThreads    = n;
            info        = (Info_t *)mxMalloc( nThreads*sizeof(Info_t) );
            threads     = (pthread_t *)mxMalloc( nThreads*sizeof(pthread_t) );
            for(j = 0; j < n; j++) {
                /* This launches each column to a new process
                 * so there's a bit of overhead since a lot of processes sometimes */
                info[j].id      = (long) j;
                info[j].length  = m;
                info[j].n       = (unsigned)1;
                info[j].x       = x + j*m;
                info[j].y       = y + j*m;
                pthread_create(&threads[j], NULL, worker, (void*)&info[j]);
            }
        } else {
            nThreads    = NTHREADS_;

            /* In this case, each worker does several columns */
            /*nn  = (unsigned)( 1 + (((int)n - 1) / nThreads)); */
            /* ceil */
            nn  = n/NTHREADS_;  /* floor */
            if ( n % NTHREADS_ != 0 )
                nThreads++;
            info        = (Info_t *)mxMalloc( nThreads*sizeof(Info_t) );
            threads     = (pthread_t *)mxMalloc( nThreads*sizeof(pthread_t) );
#ifdef DEBUG
            mexPrintf("n is %d, nn is %d,nThreads is %d\n", (int)n,(int)nn, (int)nThreads );
#endif
            
            /* Loop over the first nThreads-1 */
            jj = 0;
            for(j = 0; j < (unsigned)(NTHREADS_); j++) {
                info[j].id      = (long) j;
                info[j].length  = m;
                info[j].n       = nn;
                info[j].x       = x + jj*m;
                info[j].y       = y + jj*m;
                jj += nn;
#ifdef DEBUG
                mexPrintf("(j,jj,n) are (%d,%d,%d)\n", (int)j,(int)jj,(int)nn );
#else
                pthread_create(&threads[j], NULL, worker, (void*)&info[j]);
#endif
            }
                            
            /* And do the final one, which may have fewer columns than the rest */
            /* j++; */  /* DO NOT INCREMENT j. It has already been incremented at end of above loop */
            if ( n % NTHREADS_ != 0 ) {
                info[j].id      = (long) j;
                info[j].length  = m;
                info[j].x       = x + jj*m;
                info[j].y       = y + jj*m;
                info[j].n       = n - jj; /* may be less than nn; do NOT update jj */
#ifdef DEBUG
                mexPrintf("(j,jj,n) are (%d,%d,%d)\n", (int)j,(int)jj,(int)info[j].n );
#else
                pthread_create(&threads[j], NULL, worker, (void*)&info[j]);
#endif
            }
            
        }
#ifdef DEBUG
        mexPrintf("Launched %d threads\n", (int)nThreads );
#endif
        /* Wait for them all to finish */
#ifndef DEBUG
        for(j = 0; j < nThreads; j++)
            pthread_join( threads[j],NULL );
#endif
        mxFree(threads);
        mxFree(info);
    }    
}


/* check that the vector length is a power of 2,
   just using bitshifting instead of log */
void checkPowerTwo(unsigned m)
{
    /* check that it's not a degenerate 0 by 1 vector or singleton */
    if (m <= 1) {
        mexErrMsgTxt("Vector length must be greater than 1.");
    }
    /* keep dividing by two until result is odd */
    while( (m & 1) == 0 ){
        m >>= 1;
    }
    /* check that m is not a multiple of an odd number greater than 1 */
    if (m > 1) {
        mexErrMsgTxt("Vector length must be power of 2.");
    }
}


/* The gateway routine. */
void mexFunction(int nlhs, mxArray *plhs[],
                 int nrhs, const mxArray *prhs[])
{
  double *x, *y;
  unsigned m, n;
    
  /* Check for the proper number of arguments. */
  if (nrhs != 1) {
    mexErrMsgTxt("One and only one input required; must be a column vector or matrix, with # rows a power of 2.");
  }
  if (nlhs > 1) {
    mexErrMsgTxt("Too many output arguments.");
  }

  /* input size */
  m = mxGetM(prhs[0]);
  checkPowerTwo(m);
  n = mxGetN(prhs[0]);
  
  if (mxIsComplex(prhs[0])) {
    mexErrMsgTxt("Input must be real.");   
  } else if (mxIsSparse(prhs[0])) {
    mexErrMsgTxt("Input must be a full matrix, not sparse.");   
  } else if (!mxIsDouble(prhs[0])) {
    mexErrMsgTxt("Input must be of type double.");      
  }
  
  /* Create matrix for the return argument. */
  plhs[0] = mxCreateDoubleMatrix(m, n, mxREAL);
  
  /* Assign pointers to each input and output. */
  x = mxGetPr(prhs[0]);
  y = mxGetPr(plhs[0]);
  
  /* Call the C subroutine. */
  hadamard_apply_matrix_threads(y, x, m, n);
  return;
}
