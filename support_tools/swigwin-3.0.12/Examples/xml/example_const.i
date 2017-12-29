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

/* The following statements also produce constants */
const int     iconst = 37;
const double  fconst = 3.14;


