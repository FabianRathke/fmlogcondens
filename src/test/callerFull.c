#include <mex.h>
#include <mat.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>
#include <time.h>
#include <headers.h>

extern void newtonBFGSLC(double *X_,  double *XW_, double *box, double *params_, double *paramsB, int *lenP, int *lenPB_, int *dim_, int *n_, double *ACVH, double *bCVH, int *lenCVH_, double *intEps_, double *lambdaSqEps_, double *cutoff_, int *verbose_, double *gamma_, int *maxIter_);

int main() {
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
    double *paramsA = (double*)mxGetData(matGetVariable(pmat,"paramsA"));
    double *paramsB = (double*)mxGetData(matGetVariable(pmat,"paramsB"));
    double *box  = mxGetData(matGetVariable(pmat,"box"));
    double *ACVH  = mxGetData(matGetVariable(pmat,"ACVH"));
    double *bCVH  = mxGetData(matGetVariable(pmat,"bCVH"));
    int verbose = 0;
    double intEps = 1e-3;
    double lambdaSqEps = 1e-7;
    double cutoff = 1e-1;
	double gamma = 1000;
	int maxIter = 50;

    int n = mxGetM(matGetVariable(pmat,"X")); /* number of data points */
    int dim = mxGetN(matGetVariable(pmat,"X"));
    int lenPA = mxGetNumberOfElements(matGetVariable(pmat,"paramsA")); /* number of hyperplanes */
    int lenPB = mxGetNumberOfElements(matGetVariable(pmat,"paramsB")); /* number of hyperplanes */
    int lenCVH = mxGetNumberOfElements(matGetVariable(pmat,"bCVH"));

#ifdef __AVX__  
	printf("AVX\n");
#else
 	printf("No AVX\n"); 
#endif


	printf("A: %d, B: %d\n",lenPA, lenPB);
	printf("n: %d, d: %d\n",n, dim);

    newtonBFGSLC(X, XW, box, paramsA, paramsB, &lenPA, &lenPB, &dim, &n, ACVH, bCVH, &lenCVH, &intEps, &lambdaSqEps, &cutoff, &verbose, &gamma, &maxIter);
}

