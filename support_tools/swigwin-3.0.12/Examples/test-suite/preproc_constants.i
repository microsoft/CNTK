%module preproc_constants

%{
#if defined(__clang__)
//Suppress: warning: use of logical '&&' with constant operand [-Wconstant-logical-operand]
#pragma clang diagnostic ignored "-Wconstant-logical-operand"
#endif
%}

// Note: C types are slightly different to C++ types as (a && b) is int in C and bool in C++

// Simple constants
#define CONST_INT1      10
#define CONST_INT2      0xFF

#define CONST_UINT1     10u
#define CONST_UINT2     10U
#define CONST_UINT3     0xFFu
#define CONST_UINT4     0xFFU

#define CONST_LONG1     10l
#define CONST_LONG2     10L
#define CONST_LONG3     0xFFl
#define CONST_LONG4     0xFFL

#define CONST_LLONG1    10LL
#define CONST_LLONG2    10ll
#define CONST_LLONG3    0xFFll
#define CONST_LLONG4    0xFFLL

#define CONST_ULLONG1   10ull
#define CONST_ULLONG2   10ULL
#define CONST_ULLONG3   0xFFull
#define CONST_ULLONG4   0xFFULL

#define CONST_DOUBLE1   10e1
#define CONST_DOUBLE2   10E1
#define CONST_DOUBLE3   12.3
#define CONST_DOUBLE4   12.
#define CONST_DOUBLE5   12.3f
#define CONST_DOUBLE6   12.3F

#define CONST_BOOL1     true
#define CONST_BOOL2     false

#define CONST_CHAR      'x'
#define CONST_STRING1   "const string"
#define CONST_STRING2   "const" " string"
#define CONST_STRING3   "log-revprops"

// Expressions - runtime tests check the type for any necessary type promotions of the expressions

#define INT_AND_BOOL    0xFF & true
#define INT_AND_CHAR    0xFF & 'A'
#define INT_AND_INT     0xFF & 2
#define INT_AND_UINT    0xFF & 2u
#define INT_AND_LONG    0xFF & 2l
#define INT_AND_ULONG   0xFF & 2ul
#define INT_AND_LLONG   0xFF & 2ll
#define INT_AND_ULLONG  0xFF & 2ull

#define BOOL_AND_BOOL   true & true // Note integral promotion to type int
#define CHAR_AND_CHAR   'A' & 'B'   // Note integral promotion to type int 


#define EXPR_MULTIPLY    0xFF * 2
#define EXPR_DIVIDE      0xFF / 2
//FIXME #define EXPR_MOD         0xFF % 2

#define EXPR_PLUS        0xFF + 2
#define EXPR_MINUS       0xFF + 2

#define EXPR_LSHIFT      0xFF << 2
#define EXPR_RSHIFT      0xFF >> 2
/* FIXME
#define EXPR_LT          0xFF < 255
#define EXPR_GT          0xFF > 255
*/
#define EXPR_LTE         0xFF <= 255
#define EXPR_GTE         0xFF >= 255
#define EXPR_INEQUALITY  0xFF != 255
#define EXPR_EQUALITY    0xFF == 255
#define EXPR_AND         0xFF & 1
#define EXPR_XOR         0xFF ^ 1
#define EXPR_OR          0xFF | 1
#define EXPR_LAND        0xFF && 1
#define EXPR_LOR         0xFF || 1
#define EXPR_CONDITIONAL true ? 2 : 2.2

#define EXPR_CHAR_COMPOUND_ADD 'A' + 12
#define EXPR_CHAR_COMPOUND_LSHIFT 'B' << 6
#define H_SUPPRESS_SCALING_MAGIC (('s'<<24) | ('u'<<16) | ('p'<<8) | 'p')

/// constant assignment in enum
#if defined(SWIGCSHARP)
%csconstvalue("1<<2") kValue;
#elif defined(SWIGD)
%dconstvalue("1<<2") kValue;
#endif

%{
#define BIT(n) (1ULL << (n))

enum MyEnum {
  kValue = BIT(2)
};
%}

#define BIT(n) (1ULL << (n))

enum MyEnum {
  kValue = BIT(2)
};

