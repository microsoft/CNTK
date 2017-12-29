// Test "Unexpected tokens after ..." errors
%module xxx

#ifndef AAA
#endif rubbish

#ifdef AAA
#endif rubbish

#ifdef AAA
#else rubbish
#endif

#define BBB

#ifdef BBB
#else
#endif rubbish

#if !defined(BBB)
#else rubbish
#endif

