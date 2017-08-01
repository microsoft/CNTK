//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// This implements the elementwise tensor operations, including helper macros and some actual functions.
//

#pragma once

#include "Basics.h"
#include "CommonMatrix.h"

#pragma push_macro("TENSOR_OPS_DECL")
#ifndef TENSOR_OPS_DECL // to make these accessible to CUDA kernels, say '#define TENSOR_OPS_DECL __device__ __host__'
#define TENSOR_OPS_DECL
#endif

#pragma push_macro("DECL")
#define DECL static inline TENSOR_OPS_DECL

// This class is exported from the Math.dll.
namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// unified overloads for float/double math functions
//
// Declare float and double versions of the functions x we need as x_().
// This macro overloads x_() with float and double arguments, and inlines the correct library function,
// e.g. exp_ -> exp(double), expf(float). This simplifies templated kernel code.
// -----------------------------------------------------------------------

#pragma push_macro("OverloadUnaryMathFns")
#define OverloadUnaryMathFns(x) \
    DECL float x##_(float f)    \
    {                           \
        return x##f(f);         \
    }                           \
    DECL double x##_(double f)  \
    {                           \
        return x(f);            \
    }

OverloadUnaryMathFns(exp);
OverloadUnaryMathFns(log);
OverloadUnaryMathFns(tanh);
OverloadUnaryMathFns(sqrt);
OverloadUnaryMathFns(fabs);
OverloadUnaryMathFns(cos);
OverloadUnaryMathFns(sin);
OverloadUnaryMathFns(floor);
OverloadUnaryMathFns(log1p);
OverloadUnaryMathFns(asin);
OverloadUnaryMathFns(acos);
OverloadUnaryMathFns(sinh);
OverloadUnaryMathFns(cosh);

#pragma pop_macro("OverloadUnaryMathFns")

#pragma push_macro("OverloadBinaryMathFns")
#define OverloadBinaryMathFns(x)         \
    DECL float x##_(float f, float y)    \
    {                                    \
        return x##f(f, y);               \
    }                                    \
    DECL double x##_(double f, double y) \
    {                                    \
        return x(f, y);                  \
    }

// Because we compile with fast math the following produces nan for negative numbers raised to integer power.
// To avoid this we define safepow_ further below.
// Is there an nvcc pragma to disable fast math temporarily? Something like 
// #pragma fast-math push
// #pragma fast-math off
// OverloadBinaryMathFns(pow);
// #pragma fast-math pop
OverloadBinaryMathFns(pow);

template<typename T>
DECL T safepow_(T base, T exponent)        
{
    if (exponent == 0) 
        return T(1);
    if (base == 0)
        return T(0);
    else if (base > 0)
        return pow_(base, exponent);
    else 
    {
        int exp_as_int = static_cast<int>(exponent);
        if (exponent != exp_as_int)
            return T(NAN);
        else
            return pow_(fabs_(base), exponent) * (1 - 2 * (exp_as_int & 1));
    }
}                                    

#pragma pop_macro("OverloadBinaryMathFns")



// -----------------------------------------------------------------------
// additional functions that are standard in our context
// -----------------------------------------------------------------------

template <class ElemType>
DECL ElemType Sigmoid(ElemType z)
{
#if 1 // BUGBUG: Numerically bad. But if I don't use this, results change.
    ElemType negElem = -z;
    ElemType e = exp_(negElem);

    return 1 / (e + 1);
#else
#if 1 // Efficient implementation that avoids to divergent CUDA code paths that both compute exp() [jdroppo]. This version compiles to PTX without branches.
    ElemType q = exp_(-fabs_(z));
    ElemType numer;
    if (z > 0) // q = exp(-z)
        numer = 1;
    else // q = exp(z)
        numer = q;
    return numer / (1 + q);
#else // Reference code:
    if (z > 0)
        return 1 / (1 + exp_(-z));
    else
    {
        ElemType v = exp_(z);
        return v / (1 + v);
    }
#endif
#endif
}

// Numerically stable Sigmoid, we can't remove the old one due to Speech dependency.
template <class ElemType>
DECL ElemType StableSigmoid(ElemType z)
{
    ElemType q = exp_(-fabs_(z));
    ElemType numer;
    if (z > 0) // q = exp(-z)
        numer = 1;
    else // q = exp(z)
        numer = q;
    return numer / (1 + q);
}

template <class ElemType>
DECL ElemType SigmoidDerivative(ElemType z)
{
    ElemType v = Sigmoid(z);
    return v * (1 - v);
}

template <class ElemType>
DECL ElemType StableSigmoidDerivative(ElemType z)
{
    ElemType v = StableSigmoid(z);
    return v * (1 - v);
}

template <class ElemType>
DECL ElemType LinearRectifierDerivative(ElemType z)
{
    return z > 0 ? (ElemType) 1 : 0;
}

template <class ElemType>
DECL ElemType ExponentialLinearUnitDerivative(ElemType z)
{
    return z >= 0 ? (ElemType)1 : exp_(z);
}

template <class ElemType>
DECL ElemType Sgn(ElemType z)
{
    if (z > 0.0) return 1.0;
    if (z < 0.0) return -1.0;
    return z;
}

template <class ElemType>
DECL ElemType Sqr(ElemType z)
{
    return z * z;
}

template <class ElemType>
DECL ElemType Sqrt(ElemType z)
{
    // BUGBUG: Why clip to 0? An invalid sqrt() should show up as a NaN in the result, instead of hiding it.
    return sqrt_(z > 0 ? z : 0);
}

template <class ElemType>
DECL ElemType ClippedLog(ElemType z)
{
    return z < EPS_IN_LOG ? LOG_OF_EPS_IN_LOG : log_(z);
}

template <class ElemType>
DECL ElemType ClippedQuotient(ElemType a, ElemType b)
{
    if (fabs(b) < EPS_IN_INVERSE) // clip the denominator
    {
        if (b > 0)
            b = EPS_IN_INVERSE;
        else
            b = -EPS_IN_INVERSE;
    }
    return a / b;
}

template <typename ElemType>
DECL ElemType LogAdd(ElemType x, ElemType y)
{
    // The reason that we don't use std::swap, is because this code is used in Cuda and not just cpu.
    if (x < y)
    {
        ElemType temp = x;
        x = y;
        y = temp;
    }

    return x + log1p_(exp_(y - x));
}

// IndexElement reindexes a tensor along one dimension.
// For the indexed dimension, the tensor op is prepared by setting 'a' to be broadcasting along the indexed dimension.
// I.e. pa = &a points to the first element (as if index == 0).
// This function then must now adjust the address:
//  pa <- pa + stride * index
// The stride is passed in as third parameter.
//template<class ElemType> DECL ElemType IndexElement(const ElemType & a, ElemType b, int stride) { const ElemType * pa = &a; return pa[stride * (ptrdiff_t)b]; }

// -----------------------------------------------------------------------
// ElementWiseOperator implementations
//
// Define a static function for every ElementWiseOperator (CommonMatrix.h).
// -----------------------------------------------------------------------

#pragma push_macro("DefNullaryOp")
#define DefNullaryOp(op, expr) \
    template <class ElemType>  \
    DECL ElemType Op##op()     \
    {                          \
        return expr;           \
    }

DefNullaryOp(ConstOne, 1);
#pragma pop_macro("DefNullaryOp")

#pragma push_macro("DefUnaryOp")
#define DefUnaryOp(op, expr)         \
    template <class ElemType>        \
    DECL ElemType Op##op(ElemType a) \
    {                                \
        return expr;                 \
    }

DefUnaryOp(Copy, a);
DefUnaryOp(Negate, -a);
DefUnaryOp(Not, !a);
DefUnaryOp(Abs, fabs_(a));
DefUnaryOp(Floor, floor_(a));
DefUnaryOp(Sigmoid, Sigmoid(a));
DefUnaryOp(Tanh, tanh_(a));
DefUnaryOp(Sqr, Sqr(a));
DefUnaryOp(Sqrt, Sqrt(a));
DefUnaryOp(Exp, exp_(a));
DefUnaryOp(Log, ClippedLog(a));
DefUnaryOp(LinearRectifier, a > 0 ? a : 0);
DefUnaryOp(Cosine, cos_(a));
DefUnaryOp(Sin, sin_(a));
DefUnaryOp(Reciprocal, a == 0 ? 0 : 1 / a);
DefUnaryOp(ExponentialLinearUnit, a >= 0 ? a : (exp_(a)-1));
DefUnaryOp(StableSigmoid, StableSigmoid(a));
DefUnaryOp(Asin, asin_(a));
DefUnaryOp(Acos, acos_(a));
DefUnaryOp(Sinh, sinh_(a));
DefUnaryOp(Cosh, cosh_(a));
#pragma pop_macro("DefUnaryOp")

#pragma push_macro("DefBinaryOp")
#define DefBinaryOp(op, expr)                    \
    template <class ElemType>                    \
    DECL ElemType Op##op(ElemType a, ElemType b) \
    {                                            \
        return expr;                             \
    }
//#define DefBinaryOp(op, expr) template<class ElemType> DECL ElemType Op ## op(const ElemType & a, ElemType b, int i = 0) { UNUSED(i); return expr; }
DefBinaryOp(CopyIf, a != 0 ? b : 0);
DefBinaryOp(CopyIfNot, a == 0 ? b : 0);
DefBinaryOp(Sum, a + b);
DefBinaryOp(Difference, a - b);
DefBinaryOp(ElementwiseProduct, a* b);
DefBinaryOp(ElementwiseQuotient, ClippedQuotient(a, b));
DefBinaryOp(LogSum, LogAdd(a, b));
DefBinaryOp(Pow, safepow_(a, b));
DefBinaryOp(Max, a > b ? a : b);
DefBinaryOp(Min, a < b ? a : b);
DefBinaryOp(Equal, a == b);
DefBinaryOp(NotEqual, a != b);
DefBinaryOp(Greater, a > b);
DefBinaryOp(Less, a < b);
DefBinaryOp(GreaterEqual, a >= b);
DefBinaryOp(LessEqual, a <= b);
DefBinaryOp(And, (float)((!!a) && (!!b)));
DefBinaryOp(Or, (float)((!!a) || (!!b)));
DefBinaryOp(Xor, (float)((!!a) ^ (!!b)));
DefBinaryOp(MaskNegative, b >= 0 ? a : 0);
DefBinaryOp(ElementwiseProductWithSigmoidDerivativeFromOutput, a*(b*(1 - b))); // b = output
DefBinaryOp(ElementwiseProductWithTanhDerivativeFromOutput, a*(1 - b * b));
DefBinaryOp(ElementwiseProductWithLinearRectifierDerivativeFromOutput, b > 0 ? a : 0);
DefBinaryOp(ElementwiseProductWithLogDerivativeFromOutput, a* exp_(-b));
DefBinaryOp(ElementwiseProductWithCosDerivative, a * -sin_(b)); // note: b = input for cos()
DefBinaryOp(ElementwiseProductWithSinDerivative, a * cos_(b)); // note: b = input for sin()
DefBinaryOp(ElementwiseProductWithAbsDerivative, a * Sgn(b)); // note: b = input for abs()
DefBinaryOp(ElementwiseProductWithReciprocalDerivative, a * -Sqr(b)); // b = output
DefBinaryOp(ElementwiseProductWithSqrtDerivative, a / (2 * b)); // b = output; d/dx sqrt(x) = 1/(2 * sqrt(x)) --> note this is the same as ElementwiseQuotient w a constant; if more show up like this we should add more template params
DefBinaryOp(SqrOfDifference, Sqr(a - b));
DefBinaryOp(ElementwiseProductWithExponentialLinearUnitDerivativeFromOutput, b >= 0 ? a : a*(1+b)); // b = output;
DefBinaryOp(ElementwiseProductWithAsinDerivative, a / sqrt_(1 - b * b)); // note: b = input for asin()
DefBinaryOp(ElementwiseProductWithAcosDerivative, -a / sqrt_(1 - b * b)); // note: b = input for acos()
DefBinaryOp(ElementwiseProductWithSinhDerivative, a * cosh_(b)); // note: b = input for sinh()
DefBinaryOp(ElementwiseProductWithCoshDerivative, a * sinh_(b)); // note: b = input for cosh()
//DefBinaryOp(Index, IndexElement(a, b, i));  // note: this one uses the third argument

#pragma pop_macro("DefBinaryOp")

#pragma push_macro("DefTernaryOp")
#define DefTernaryOp(op, expr)                               \
    template <class ElemType>                                \
    DECL ElemType Op##op(ElemType a, ElemType b, ElemType c) \
    {                                                        \
        return expr;                                         \
    }

DefTernaryOp(Cond, a ? b : c);
DefTernaryOp(CopyIfEqual, a == b ? c : 0); // CopyIfEqual(a,b)(c) -- if a==b copy c, otherwise 0; used for gradient of clip, min, max, etc.
DefTernaryOp(Clip, c < a ? a : (c > b ? b : c)); // Clip(min,max)(data) => a=min, b=max, c=data
DefTernaryOp(ElementwiseProductWithLogSumDerivative, a * StableSigmoid(c - b));
DefTernaryOp(ElementwiseProductWithExpOfDiff, a * exp_(b - c));
DefTernaryOp(ElementwiseProductWithQuotient, a * b * OpReciprocal(c));
DefTernaryOp(ElementwiseProductWithPowExponentDerivative, c <= 0 ? 0 : a * b * log_(c)); // same behavior as other toolkits
DefTernaryOp(ElementwiseProductWithPowBaseDerivative, a * c * OpPow(b, c - 1)); // Using the output of pow would be faster but it requires a quaternary op and users will likely only use pow in forward mode

#pragma pop_macro("DefTernaryOp")

}}}
#pragma pop_macro("DECL")
#pragma pop_macro("TENSOR_OPS_DECL")
