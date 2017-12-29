%module xxx

/* Test line number reporting for multiple macro expansions */

#define MACRO2(a, b) 

#define MACRO1(NAME) MACRO2(NAME,2,3) 

MACRO1(abc)

