/* File : example.i */
%module example

/* A few preprocessor macros */

#define    ICONST      42
#define    FCONST      2.1828
#define    CCONST      'x'
#define    CCONST2     '\n'
#define    SCONST      "Hello World"
#define    SCONST2     "\"Hello World\""

/* This should work just fine */
#define    EXPR        ICONST + 3*(FCONST)

/* This shouldn't do anything */
#define    EXTERN      extern

/* Neither should this (BAR isn't defined) */
#define    FOO         (ICONST + BAR)

/* The following directives also produce constants.  Remember that
   CHICKEN is normally case-insensitive, so don't rely on differing
   case to differentiate variable names */

%constant int iconstX = 37;
%constant double fconstX = 3.14;
