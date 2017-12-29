/* File : example.i */
%module example

/* Force the generated D code to use the C constant values rather than
   retrieving them at runtime. You can also try disabling the feature and
   compare the generated code. */
%dmanifestconst;


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


/* The following directives also produce constants */

%constant int iconst = 37;
%constant double fconst = 3.14;
