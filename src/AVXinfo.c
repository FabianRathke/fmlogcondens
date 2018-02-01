#include <R.h>

void printAVXInfo() {
#ifdef __AVX2__
	Rprintf("AVX2 and AXV vector extensions activated.");
#elseif __AVX__
	Rprintf("AVX vector extensions activated.");
#else
	Rprintf("No vector extensions activated.");
#endif
}
