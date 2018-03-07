## Test environments

#### Ubuntu 16.04 (R 3.4.3)

1 NOTEs (see below), 0 WARNINGs, 0 ERRORs

#### Windows (R 3.4.3 and R-devel 2018-03-05 r74359)

via https://win-builder.r-project.org/

1 NOTEs (see below), 0 WARNING, 0 ERRORs



## R CMD check results

Linux NOTE

```
checking CRAN incoming feasibility ... NOTE

- Maintainer: ‘Fabian Rathke frathke@gmail.com’

New submission
```



Windows NOTE (same for both checked versions)

```
* checking CRAN incoming feasibility ... NOTE
Maintainer: 'Fabian Rathke <frathke@gmail.com>'

New submission

Possibly mis-spelled words in DESCRIPTION:
  Christoph (17:68)
  LogConcDEAD (16:18)
  Rathke (17:60)
  Schnörr (18:3)
  fmlogcondens (16:31)

Found the following (possibly) invalid URLs:
  URL: https://CRAN.R-project.org/package=fmlogcondens/index.html
    From: inst/doc/documentation.html
    Status: 404
    Message: Not Found
```

I already use the future URL of the package, when describing the installation process from source in the vignette `documentation.Rmd`. The words mentioned refer to persons [1, 3, 4] and package names [2,5].



## Downstream dependencies

No downstream dependencies as this is the initial release.



## Resubmission

As requested after the first submission, I modified the DESCRIPTION file and added

* the author Giovanni Garberoglio with the tag "cph" due to the use of avx_mathfun.h,
* a reference (paper) which contains background information about our method at the end of the description field.
