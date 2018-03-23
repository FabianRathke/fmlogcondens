## Test environments

#### Ubuntu 16.04 (R 3.4.4)

1 NOTEs (see below), 0 WARNINGs, 0 ERRORs

#### Windows (R 3.4.4 and R-devel 2018-03-21 r74436)

via https://win-builder.r-project.org/

1 NOTEs (see below), 0 WARNING, 0 ERRORs

#### macOS 10.9 (R 3.3.3)

via https://builder.r-hub.io

0 NOTEs, 0 WARNINGs, 0 ERRORs

#### Oracle SunOS 5.10 (R 3.4.1)

via https://builder.r-hub.io

0 NOTEs, 1 WARNINGs (related to the testing environment, see below), 0 ERRORs



## R CMD check results

All notes refer to the fact that I submitted version 1.0.1 three days ago. Due to the deadline by Prof. Ripley (see below) i am sending a second update, that fixes problems related to Solaris/SunOS.

```
* checking CRAN incoming feasibility ... NOTE
Maintainer: ‘Fabian Rathke <frathke@gmail.com>’

Days since last update: 3
```



Windows NOTE (R 3.4.4 only)

```
* checking CRAN incoming feasibility ... NOTE
Maintainer: 'Fabian Rathke <frathke@gmail.com>'

Days since last update: 3

Possibly mis-spelled words in DESCRIPTION:
  Christoph (17:68)
  LogConcDEAD (16:18)
  Rathke (17:60)
  Schnörr (18:3)
  fmlogcondens (16:31)
```

The words mentioned refer to persons [1, 3, 4] and package names [2,5].



SunOS 5.10 WARNING

```
* checking re-building of vignette outputs ... WARNING
Error in re-building vignettes:
  ...
Warning in engine$weave(file, quiet = quiet, encoding = enc) :
  Pandoc (>= 1.12.3) and/or pandoc-citeproc not available. Falling back to R Markdown v1.
Warning in (function (filename = "Rplot%03d.png", width = 480, height = 480,  :
  unable to open connection to X11 display ''
Quitting from lines 88-89 (documentation.Rmd) 
Error: processing vignette 'documentation.Rmd' failed with diagnostics:
unable to start device PNG
Execution halted
```

Not related to the package.



## Downstream dependencies

No downstream dependencies.



## Changes for 1.0.2

Due to the deadline of 24.03.2018:

```
And now version 1.0.1 has introduced another error: 
https://cran.r-project.org/web/checks/check_results_fmlogcondens.html . 
As the manual says, non-C99 functions need to be tested for and used 
conditionally:

'Writing portable C and C++ code is mainly a matter of observing the 
standards (C99, C++98 or where declared C++11/14/17) and testing that 
extensions (such as POSIX functions) are supported.'

The clue is in the name ....

It is easy enough to include a substitute: see R's R_allocLD for the 
idea (or Google).

The deadline still applies.

-- 
Brian D. Ripley,                  ripley@stats.ox.ac.uk
Emeritus Professor of Applied Statistics, University of Oxford
```

Using 'posix_memalign' caused an error for Solaris. 

* I added a C preprocessor directive ``__sun`` to fall back to memalign which is supported in Solaris/SunOS. 

Successfully tested on https://builder.r-hub.io for SunOS 5.10. 

I additionally addressed the sanitizer check https://www.stats.ox.ac.uk/pub/bdr/memtests/gcc-ASAN/fmlogcondens/00check.log 

```
AddressSanitizer: memcpy-param-overlap: memory ranges
```

* by replacing memcpy with memmove in bfgsFullC.c.