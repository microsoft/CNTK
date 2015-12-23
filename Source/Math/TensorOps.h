//
// <copyright file="TensorView.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

// This implements the elementwise tensor operations, including helper macros and some actual functions.

#pragma once

#include "Basics.h"
#include "CommonMatrix.h"

#pragma push_macro("TENSOR_OPS_DECL")
#ifndef TENSOR_OPS_DECL     // to make these accessible to CUDA kernels, say '#define TENSOR_OPS_DECL __device__ __host__'
#define TENSOR_OPS_DECL
#endif

#pragma push_macro("DECL")
#define DECL static inline TENSOR_OPS_DECL

// This class is exported from the Math.dll.
namespace Microsoft { namespace MSR { namespace CNTK {

    // -----------------------------------------------------------------------
    // unified overloads for float/double math functions
    //
    // Declare float and double versions of the functions f we need as f_(),
    // e.g. exp_ -> exp(double), expf(float).
    // -----------------------------------------------------------------------

#pragma push_macro("OverloadUnaryMathFns")
    #define OverloadUnaryMathFns(func) \
        DECL float func ## _(float arg) { return func ## f(arg); } \
        DECL double func ## _(double arg) { return func(arg); }

    OverloadUnaryMathFns(fabs); OverloadUnaryMathFns(sqrt);
    OverloadUnaryMathFns(exp); OverloadUnaryMathFns(log);
    OverloadUnaryMathFns(tanh); OverloadUnaryMathFns(cos); OverloadUnaryMathFns(sin);
#pragma push_macro("OverloadUnaryMathFns")

    // -----------------------------------------------------------------------
    // additional functions that are standard in our context
    // -----------------------------------------------------------------------

    template<class ElemType>
    DECL ElemType Sigmoid(ElemType z)
    {
        if (z >= 0)
            return 1 / (1 + exp_(-z));
        else
        {
            ElemType v = exp_(z);
            return v / (1 + v);
        }
    }

    template<class ElemType>
    DECL ElemType SigmoidDerivative(ElemType z)
    {
        ElemType v = Sigmoid(z);
        return v * (1 - v);
    }

    template<class ElemType>
    DECL ElemType LinearRectifierDerivative(ElemType z)
    {
        return z > 0 ? (ElemType)1 : 0;
    }

    template<class ElemType>
    DECL ElemType Sqrt(ElemType z)
    {
        // BUGBUG: Why clip to 0? An invalid sqrt() should show up as a NaN in the result, instead of hiding it.
        return sqrt_(z > 0 ? z : 0);
    }

    // TODO: call this LogAdd() for consistency
    template<typename ElemType>
    DECL ElemType LogAdd(ElemType x, ElemType y)
    {
        if (x < y)
        {
            ElemType temp = x; x = y; y = temp;
        }
        ElemType diff = y - x;
        if (diff < (ElemType)MINLOGEXP)
        {
            return (x < (ElemType)LSMALL) ? (ElemType)LZERO : x;
        }
        else
        {
            ElemType z = exp_(diff);
            return x + log_((ElemType)1.0 + z);
        }
    }

    // -----------------------------------------------------------------------
    // ElementWiseOperator implementations
    //
    // Define a static function for every ElementWiseOperator (CommonMatrix.h).
    // -----------------------------------------------------------------------

#pragma push_macro("DefUnaryOp")
    #define DefUnaryOp(op, expr) template<class ElemType> DECL ElemType Op ## op(ElemType a) { return expr; }

    DefUnaryOp(Copy, a);
    DefUnaryOp(Negate, -a); DefUnaryOp(Not, !a);
    DefUnaryOp(Abs, fabs_(a));
    DefUnaryOp(Sigmoid, Sigmoid(a)); DefUnaryOp(SigmoidDerivative, SigmoidDerivative(a)); DefUnaryOp(Tanh, tanh_(a)); DefUnaryOp(Sqrt, Sqrt(a)); DefUnaryOp(Exp, exp_(a)); DefUnaryOp(Log, log_(a)); DefUnaryOp(LinearRectifierDerivative, LinearRectifierDerivative(a)); DefUnaryOp(Cosine, cos_(a)); DefUnaryOp(NegativeSine, -sin_(a));
#pragma pop_macro("DefUnaryOp")

    // parameterized unary ops
    //DefUnaryOp(SaturateBetaAlpha); DefUnaryOp(SumAlpha); DefUnaryOp(SubDifferenceToAlpha); DefUnaryOp(SubDifferenceFromAlpha);

#pragma push_macro("DefBinaryOp")
    #define DefBinaryOp(op, expr) template<class ElemType> DECL ElemType Op ## op(ElemType a, ElemType b) { return expr; }

    DefBinaryOp(Sum, a + b); DefBinaryOp(Difference, a - b); DefBinaryOp(ElementwiseProduct, a*b); DefBinaryOp(ElementwiseQuotient, a / b);
    DefBinaryOp(LogSum, LogAdd(a, b)); DefBinaryOp(Max, a > b ? a : b); DefBinaryOp(Min, a < b ? a : b);
    DefBinaryOp(EQ, a == b); DefBinaryOp(NE, a != b); DefBinaryOp(GT, a > b); DefBinaryOp(LT, a < b); DefBinaryOp(GE, a >= b); DefBinaryOp(LE, a <= b);
    DefBinaryOp(MaskNegative, b >= 0 ? a : 0);
#pragma pop_macro("DefBinaryOp")

#pragma push_macro("DefTernaryOp")
    #define DefTernaryOp(op, expr) template<class ElemType> DECL ElemType Op ## op(ElemType a, ElemType b, ElemType c) { return expr; }

    DefTernaryOp(Cond, a ? b : c);
#pragma pop_macro("DefTernaryOp")

}}}
#pragma pop_macro("DECL")
#pragma pop_macro("TENSOR_OPS_DECL")
