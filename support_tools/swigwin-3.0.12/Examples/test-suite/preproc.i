%module preproc

%warnfilter(SWIGWARN_RUBY_WRONG_NAME) one; /* Ruby, wrong constant name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) two; /* Ruby, wrong constant name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) three; /* Ruby, wrong constant name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) __GMP_HAVE_CONST; /* Ruby, wrong constant name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) __GMP_HAVE_PROTOTYPES; /* Ruby, wrong constant name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) __GMP_HAVE_TOKEN_PASTE; /* Ruby, wrong constant name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) __GMP_HAVE_TOKEN_PASTE; /* Ruby, wrong constant name */

#pragma SWIG nowarn=890                                      /* lots of Go name conflicts */
#pragma SWIG nowarn=206                                      /* Unexpected tokens after #endif directive. */

%{
#if defined(__clang__)
//Suppress: warning: use of logical '&&' with constant operand [-Wconstant-logical-operand]
#pragma clang diagnostic ignored "-Wconstant-logical-operand"
#endif
%}

/* check __cplusplus case */
%header
%{
#ifdef __cplusplus
extern "C"
{
#endif /* __cplusplus */
  /* C code */
#ifdef __cplusplus
}
#endif /* __cplusplus */

%}


/* This interface file tests whether SWIG's extended C
   preprocessor is working right. 

   In this example, SWIG 1.3.6 chokes on "//" in a #define with a
   syntax error.
*/

#define SLASHSLASH "//"

/* This SWIG -*- c -*- interface is to test for some strange
   preprocessor bug.

   I get syntax errors unless I remove the apostrophe in the comment
   or the sharp-sign substitution.  (The apostrophe seems to disable
   sharp-sign substitution.)
*/


%define TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(SCM_TYPE)

     /* Don't check for NULL pointers (override checks). */

     %typemap(argout, doc="($arg <vector of <" #SCM_TYPE ">>)") 
          int *VECTORLENOUTPUT
     {
     }

%enddef

TYPEMAP_LIST_VECTOR_INPUT_OUTPUT(boolean)

// preproc_3

#define Sum( A, B, \
             C)    \
        A + B + C 


// preproc_4
%{
  int hello0()
  {
    return 0;
  }

  int hello1()
  {
    return 1;
  }

  int hello2()
  {
    return 2;
  }  
  int f(int min) { return min; }
%}

#define ARITH_RTYPE(A1, A2) A2

#define HELLO_TYPE(A, B) ARITH_RTYPE(A, ARITH_RTYPE(A,B))

//
// These two work fine
//
int hello0();
ARITH_RTYPE(double,int) hello1();


//
// This doesn't work with 1.3.17+ ( but it was ok in 1.3.16 )
// it gets expanded as (using -E)
// 
//   ARITH_RTYPE(double,int) hello2();
//
HELLO_TYPE(double,int) hello2();

#define min(x,y) ((x) < (y)) ? (x) : (y) 
int f(int min);

// preproc_5

%warnfilter(SWIGWARN_PARSE_REDEFINED) A5;	// Ruby, wrong constant name
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) a5;	// Ruby, wrong constant name
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) b5;	// Ruby, wrong constant name
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) c5;	// Ruby, wrong constant name
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) d5;	// Ruby, wrong constant name

// Various preprocessor bits of nastiness.


/* Test argument name substitution */
#define foo(x,xx) #x #xx
#define bar(x,xx) x + xx

%constant char *a5 = foo(hello,world);
%constant int   b5 = bar(3,4);

// Wrap your brain around this one ;-)

%{
#define cat(x,y) x ## y
%}

#define cat(x,y) x ## y

/* This should expand to cat(1,2);  
   See K&R, p. 231 */

%constant int c5 = cat(cat(1,2),;)

#define xcat(x,y) cat(x,y)

/* This expands to 123.  See K&R, p. 231 */
%constant int d5 = xcat(xcat(1,2),3);


#define C1\
"hello"

#define C2
#define C3 C2

#define ALONG_\
NAME 42

#define C4"Hello"

// preproc_6

%warnfilter(SWIGWARN_PARSE_REDEFINED) A6; /* Ruby, wrong constant name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) a6; /* Ruby, wrong constant name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) b6; /* Ruby, wrong constant name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) c6; /* Ruby, wrong constant name */
%warnfilter(SWIGWARN_RUBY_WRONG_NAME) d6; /* Ruby, wrong constant name */

#define add(a, b) (a + b)
#define times(a, b) (a * b)
#define op(x) x(1, 5)
 
/* expand to (1 + 5) */
%constant int a6 = op(add);
/* expand to (1 * 5) */
%constant int b6 = op(times);
/* expand to ((1 + 5) * 5) */
%constant int c6 = times(add(1, 5), 5);
/* expand to ((1 + 5) * 5) */
%constant int d6 = times(op(add), 5);                 

/* This interface file tests whether SWIG's extended C
   preprocessor is working right. 

   In this example, SWIG 1.3a5 reports missing macro arguments, which
   is bogus.
*/

%define MACRO1(C_TYPE, GETLENGTH)
     /* nothing */
%enddef

%define MACRO2(XYZZY)
  MACRO1(XYZZY, 1)
%enddef

MACRO2(int)

// cpp_macro_noarg.  Tests to make sure macros with no arguments work right.
#define MACROWITHARG(x) something(x) 

typedef int MACROWITHARG; 

/* 
This testcase tests for embedded defines and embedded %constants
*/

%inline %{

typedef struct EmbeddedDefines {
  int dummy;
#define  EMBEDDED_DEFINE 44
#ifdef SWIG
%constant EMBEDDED_SWIG_CONSTANT = 55;
#endif
} EmbeddedDefines;

%}

/* 
This testcase tests operators for defines
*/

#define A1   1 + 2
#define A2   3 - 4
#define A3   5 * 6
#define A4   7 / 8
#define A5   9 >> 10
#define A6   11 << 12
#define A7   13 & 14
#define A8   15 | 16
#define A9   17 ^ 18
#define A10  1 && 0
#define A11  1 || 0
#define A12  ~22
#define A13  !23



#ifdef __cplusplus
		   
#define %mangle_macro(...) #@__VA_ARGS__
#define %mangle_macro_str(...) ##@__VA_ARGS__

%define my_func(...)
inline const char* mangle_macro ## #@__VA_ARGS__ () {
  return %mangle_macro_str(__VA_ARGS__);
}
%enddef

%inline {
  my_func(class Int) ;
  my_func(std::pair<double, std::complex< double > >*) ;
}

#endif


#if defined (__cplusplus) \
|| defined (_AIX) \
|| defined (__DECC) \
|| (defined (__mips) && defined (_SYSTYPE_SVR4)) \
|| defined (_MSC_VER) \
|| defined (_WIN32)
#define __GMP_HAVE_CONST 1
#define __GMP_HAVE_PROTOTYPES 1
#define __GMP_HAVE_TOKEN_PASTE 1
#else
#define __GMP_HAVE_CONST 0
#define __GMP_HAVE_PROTOTYPES 0
#define __GMP_HAVE_TOKEN_PASTE 0
#endif


/* empty TWO() macro is broken */
#define ONE 1
#define TWO() 2
#define THREE(FOO) 3

#define one ONE
#define two TWO()
#define three THREE(42)


#if defined(one)
/* hello */
#else
/* chiao */
#endif;

#ifdef SWIGCHICKEN
/* define is a scheme keyword (and thus an invalid variable name), so SWIG warns about it */
%warnfilter(SWIGWARN_PARSE_KEYWORD) define; 
#endif

#ifdef SWIGRUBY
%rename(ddefined) defined;
#endif
#ifdef SWIGPHP
%rename(endif_) endif;
#endif
%inline %{
const int endif = 1;
const int define = 1;
const int defined = 1; 
int test(int defined)
{
  return defined;
}
 
%}

#pragma SWIG nowarn=SWIGWARN_PP_CPP_WARNING
#warning "Some warning"

/* check that #error can be turned into a warning, but suppress the warning message for the test-suite! */
#pragma SWIG nowarn=SWIGWARN_PP_CPP_ERROR
#pragma SWIG cpperraswarn=1
#error "Some error"


#define MASK(shift, size) (((1 << (size)) - 1) <<(shift))
#define SOME_MASK_DEF (80*MASK(8, 10))

/* some constants */
#define BOLTZMANN    (1.380658e-23)
#define AVOGADRO     (6.0221367e23)
#define RGAS         (BOLTZMANN*AVOGADRO)
#define RGASX        (BOLTZMANN*AVOGADRO*BOLTZMANN)

%{
#define TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(TYPE) \
struct TypeNameTraits { \
  int val; \
} \

%}


#define TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(TYPE) \
struct TypeNameTraits { \
  int val; \
} \

%inline %{
TEUCHOS_TYPE_NAME_TRAITS_BUILTIN_TYPE_SPECIALIZATION(int);
%}

%inline %{
int method(struct TypeNameTraits tnt) {
  return tnt.val;
}
%}

/* Null directive */
# /* comment 1 */
# // comment 2
# /** comment 3 */
# /* comment 4 */ /*comment 5*/
# /** comment 6
#
# more comment 6 */
# 
#
#	    
int methodX(int x);
%{
int methodX(int x) { return x+100; }
%}

