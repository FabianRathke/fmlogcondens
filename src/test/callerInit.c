#include <mex.h>
#include <mat.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <omp.h>
#include <R.h>
#include <headers.h>

int main() {
#ifdef __AVX__
    printf("Perform AVX optimziation\n");
#else
    printf("No AVX optimization\n");
#endif

    const char *file = FILELOC;
    MATFile *pmat;

    /* Open matfile */
    pmat = matOpen(file, "r");
    if (pmat == NULL) {
        printf("Error opening file %s\n", file);
        return(1);
    }

   /* Matlab input variables */
    double *X = (double*)mxGetData(matGetVariable(pmat,"X"));
    double *XW = (double *) mxGetData(matGetVariable(pmat,"sW")); /* Weight vector for X */
    double *params = (double*)mxGetData(matGetVariable(pmat,"params"));
    double *box  = mxGetData(matGetVariable(pmat,"box"));
    double *ACVH  = mxGetData(matGetVariable(pmat,"ACVH"));
    double *bCVH  = mxGetData(matGetVariable(pmat,"bCVH"));
    double *logLike = malloc(sizeof(double));

    int n = mxGetM(matGetVariable(pmat,"X")); /* number of data points */
    int dim = mxGetN(matGetVariable(pmat,"X"));
    int lenP = mxGetNumberOfElements(matGetVariable(pmat,"params")); /* number of hyperplanes */
    int lenCVH = mxGetNumberOfElements(matGetVariable(pmat,"bCVH"));

    double intEps = 1e-3;
    double lambdaSqEps = 1e-5; // for the initialization

    printf("%d data points, %d\n",n,dim);

    newtonBFGSLInitC(X, XW, box, params, &dim, &lenP, &n, ACVH, bCVH, &lenCVH, &intEps, &lambdaSqEps, logLike);
    printf("%.4f\n",logLike[0]);
}

