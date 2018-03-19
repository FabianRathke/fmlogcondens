## Test environments

#### Ubuntu 16.04 (R 3.4.3)

0 NOTEs, 0 WARNINGs, 0 ERRORs

#### Windows (R 3.4.3 and R-devel 2018-03-05 r74359)

via https://win-builder.r-project.org/

1 NOTEs (see below), 0 WARNING, 0 ERRORs

#### macOS 10.9 (R 3.3.3)

via https://builder.r-hub.io

0 NOTEs, 0 WARNINGs, 0 ERRORs



## R CMD check results

Windows NOTE (same for both checked versions)

```
* checking CRAN incoming feasibility ... NOTE
Maintainer: 'Fabian Rathke <frathke@gmail.com>'

Possibly mis-spelled words in DESCRIPTION:
  Christoph (17:68)
  LogConcDEAD (16:18)
  Rathke (17:60)
  Schnörr (18:3)
  fmlogcondens (16:31)
```

The words mentioned refer to persons [1, 3, 4] and package names [2,5].



## Downstream dependencies

No downstream dependencies.



## Changes for 1.0.1

As requested

```
On macOS:

/usr/local/clang6/bin/clang -I"/Users/ripley/R/R-devel/include" -DNDEBUG 
-I../src  -I/usr/local/include  -fopenmp -fPIC  -g -O2 -Wall -pedantic 
-Wconversion -Wno-sign-conversion  -c calcGradAVX.c -o calcGradAVX.o
calcGradAVX.c:3:10: fatal error: 'malloc.h' file not found
#include <malloc.h>

1 error generated.

'malloc.h' is long obsolete, as you were warned in §1.6 of the manual.

Please correct ASAP after all the check results are in, and before Mar 

24 to safely retain the package on CRAN.

-- 

Brian D. Ripley,                  ripley@stats.ox.ac.uk

Emeritus Professor of Applied Statistics, University of Oxford
```

I removed malloc.h from the header file. Also fixed some OMP related issues with macOS.