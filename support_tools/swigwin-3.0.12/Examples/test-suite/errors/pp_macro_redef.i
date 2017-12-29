%module xxx

#define foo(a,x)      a+x
#define foo           4

/* Should not generate an error */
#define foo           4

