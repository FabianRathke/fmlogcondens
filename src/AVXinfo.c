#include <R.h>

void printAVXInfo() {
#ifdef __AVX2__
	Rprintf("AVX2 and AXV vector extensions activated.\n");
#elif __AVX__
	Rprintf("AVX vector extensions activated.\n");
#else
	Rprintf("No vector extensions activated.\n");
#endif
}
