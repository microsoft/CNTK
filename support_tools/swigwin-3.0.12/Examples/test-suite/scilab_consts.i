%module scilab_consts

/* Default mode: constants are wrapped as getter functions */
%scilabconst(0);

#define ICONST0   42
#define FCONST0   2.1828
#define CCONST0   'x'
#define CCONST0_2 '\n'
#define SCONST0   "Hello World"
#define SCONST0_2 "\"Hello World\""

/* Constants with type */
#define UNSIGNED0 0x5FFFU
#define LONG0 0x3FFF0000L
#define ULONG0 0x5FF0000UL

/* Expressions should work too */
#define EXPR0 ICONST0 + 3*FCONST0

/* This shouldn't do anything, bar is not defined */
#define BAR0 bar

/* SWIG directive %constant produces constants too */
%constant int iconst0 = 37;
%constant double fconst0 = 42.2;


/* Alternative mode: constants are wrapped as variables */
%scilabconst(1);

#define ICONST1   42
#define FCONST1   2.1828
#define CCONST1   'x'
#define CCONST1_2 '\n'
#define SCONST1   "Hello World"
#define SCONST1_2 "\"Hello World\""

/* Constants with type */
#define UNSIGNED1 0x5FFFU
#define LONG1 0x3FFF0000L
#define ULONG1 0x5FF0000UL

/* Expressions should work too */
#define EXPR1 ICONST1 + 3*FCONST1

/* This shouldn't do anything, bar is not defined */
#define BAR1 bar

/* SWIG directive %constant produces constants too */
%constant int iconst1 = 37;
%constant double fconst1 = 42.2;
