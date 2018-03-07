## Test environments

#### Ubuntu 16.04 (R 3.4.3)

1 NOTEs, 0 WARNINGs, 0 ERRORs

#### Windows (R 3.4.3 and R-devel 2018-03-05 r74359)

via https://win-builder.r-project.org/

1 NOTEs (see below), 0 WARNING, 0 ERRORs



## R CMD check results

```
* checking CRAN incoming feasibility ... NOTE
Maintainer: 'Fabian Rathke <frathke@gmail.com>'

New submission

Found the following (possibly) invalid URLs:
  URL: https://CRAN.R-project.org/package=fmlogcondens/index.html
    From: inst/doc/documentation.html
    Status: 404
    Message: Not Found
```

I already use the future URL of the package, when describing the installation process from source in the vignette `documentation.Rmd`.



## Downstream dependencies

No downstream dependencies as this is the initial release.



## Resubmission

As requested by the first submission, I added

* the author Giovanni Garberoglio with the tag "cph" due to the use of avx_mathfun.h,
* a reference which contains background about our method at the end of the description field.