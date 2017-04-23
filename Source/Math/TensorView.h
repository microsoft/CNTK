//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// This implements the TensorView class, which is a layer around Matrix that reinterprets its content as a generic tensor. [fseide]
//

#pragma once

#include "Basics.h"
#include "Matrix.h"
#include "TensorShape.h"
#include "Quantizers.h"

#pragma warning(push)
#pragma warning(disable : 4251) // needs to have dll-interface to be used by clients of... caused by TensorView::m_shape which is only private. We use the same compiler everywhere.

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {
    template <class ElemType> struct TensorTest;
}}}}

// This class is exported from the Math.dll.
namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class MATH_API TensorView
{
public:
    // -------------------------------------------------------------------
    // construction
    // -------------------------------------------------------------------

    // reinterpret a matrix storage object (SOB) as a TensorView with a given TensorShape  --this is the main constructor
    TensorView(const MatrixBasePtr& sob, const TensorShape& shape);
#if 0
    // cast a Matrix as a 2D TensorView (without shape change)
    TensorView(const MatrixBasePtr& sob)
        : m_sob(sob), m_shape(TensorShape(array<size_t, 2>{sob->GetNumRows(), sob->GetNumCols()}))
    {
    }
#endif
    // reshape a TensorView
    TensorView(const TensorView<ElemType>& other, const TensorShape& shape)
        : m_sob(other.m_sob), m_shape(shape)
    {
    }
    // copy constructor
    TensorView(const TensorView<ElemType>& other)
        : m_sob(other.m_sob), m_shape(other.m_shape)
    {
    }
    // dummy constructor; this is an invalid object
    TensorView()
    {
    }

    // reshaped view
    TensorView<ElemType> Reshaped(const TensorShape& shape) const
    {
        return TensorView(*this, shape);
    }

    // -------------------------------------------------------------------
    // elementwise operations
    // Result goes into 'this', and can optionally be added to the existing value.
    // E.g. c.DoSumOf(beta,a,b,alpha) means c := beta * c + alpha * (a + b),
    //      c.AssignDiffOf(c,a) means c -= a,
    //  and c.AddElementwiseProductOf(a, b, 1) means c += a .* b.
    // All operators support elementwise in-place operations, i.e. a, b, and c
    // may all reference the same underlying SOB, with one exception:
    // The output cannot be in-place and inverse-broadcasting at the same time.
    // E.g. with c=[10] and a=[10 x 20], c.AssignDiffOf(c,a) will fail.
    // In that case, you can use c.AddCopyOf(a,-1).
    // Aliasing is not detected, so don't pass distinct TensorView objects that
    // reference overlapping but not identical slices.
    // If beta == 0, c is not read out, i.e. it can be uninitialized or contain NaNs.
    // -------------------------------------------------------------------

#pragma push_macro("DeclareUnaryTensorOp")
#define DeclareUnaryTensorOp(oper)                                                              \
    void Do##oper##Of(ElemType beta, const TensorView& a, ElemType alpha)                       \
    {                                                                                           \
        DoUnaryOpOf(beta, a, alpha, ElementWiseOperator::op##oper, ElementWiseOperator::opSum); \
    }                                                                                           \
    void Assign##oper##Of(const TensorView& a, ElemType alpha = 1.0f)                           \
    {                                                                                           \
        DoUnaryOpOf(0, a, alpha, ElementWiseOperator::op##oper, ElementWiseOperator::opSum);    \
    }                                                                                           \
    void Add##oper##Of(const TensorView& a, ElemType alpha = 1.0f)                              \
    {                                                                                           \
        DoUnaryOpOf(1.0f, a, alpha, ElementWiseOperator::op##oper, ElementWiseOperator::opSum); \
    }

    ForAllUnaryOps(DeclareUnaryTensorOp);
#pragma pop_macro("DeclareUnaryTensorOp")

#pragma push_macro("DeclareBinaryTensorOp")
#define DeclareBinaryTensorOp(oper)                                                                 \
    void Do##oper##Of(ElemType beta, const TensorView& a, const TensorView& b, ElemType alpha)      \
    {                                                                                               \
        DoBinaryOpOf(beta, a, b, alpha, ElementWiseOperator::op##oper, ElementWiseOperator::opSum); \
    }                                                                                               \
    void Assign##oper##Of(const TensorView& a, const TensorView& b, ElemType alpha = 1.0f)          \
    {                                                                                               \
        DoBinaryOpOf(0, a, b, alpha, ElementWiseOperator::op##oper, ElementWiseOperator::opSum);    \
    }                                                                                               \
    void Add##oper##Of(const TensorView& a, const TensorView& b, ElemType alpha = 1.0f)             \
    {                                                                                               \
        DoBinaryOpOf(1.0f, a, b, alpha, ElementWiseOperator::op##oper, ElementWiseOperator::opSum); \
    }

    ForAllBinaryOps(DeclareBinaryTensorOp);
#pragma pop_macro("DeclareBinaryTensorOp")

#pragma push_macro("DeclareTernaryTensorOp")
#define DeclareTernaryTensorOp(oper)                                                                                \
    void Do##oper##Of(ElemType beta, const TensorView& a, const TensorView& b, const TensorView& c, ElemType alpha) \
    {                                                                                                               \
        DoTernaryOpOf(beta, a, b, c, alpha, ElementWiseOperator::op##oper, ElementWiseOperator::opSum);             \
    }                                                                                                               \
    void Assign##oper##Of(const TensorView& a, const TensorView& b, const TensorView& c, ElemType alpha = 1.0f)     \
    {                                                                                                               \
        DoTernaryOpOf(0, a, b, c, alpha, ElementWiseOperator::op##oper, ElementWiseOperator::opSum);                \
    }                                                                                                               \
    void Add##oper##Of(const TensorView& a, const TensorView& b, const TensorView& c, ElemType alpha = 1.0f)        \
    {                                                                                                               \
        DoTernaryOpOf(1.0f, a, b, c, alpha, ElementWiseOperator::op##oper, ElementWiseOperator::opSum);             \
    }

    ForAllTernaryOps(DeclareTernaryTensorOp);
#pragma pop_macro("DeclareTernaryTensorOp")

    void DoUnaryOpOf  (ElemType beta, const TensorView& a,                                           ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp);
    void DoBinaryOpOf (ElemType beta, const TensorView& a, const TensorView& b,                      ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp);
    void DoTernaryOpOf(ElemType beta, const TensorView& a, const TensorView& b, const TensorView& c, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp);

    // -------------------------------------------------------------------
    // arg based operations
    // -------------------------------------------------------------------
    void DoArgReductionOpOf(const TensorView& a, ElementWiseOperator reductionOp);

    // -------------------------------------------------------------------
    // matrix product -- GEMM for flattened tensors
    // Result goes into 'this', and can optionally be added to the existing value.
    // [I x J x K x L] * [K x L x M x N] -> [I x J x M x N] reducing over (K,L)
    // Reduction range is inferred from tensor ranks.
    // [I x J], [K x L], and [M x N] must each be dense.
    // Being a matrix product, the output cannot be in-place.
    // If beta == 0, c is not read out, i.e. it can be uninitialized or contain NaNs.
    // -------------------------------------------------------------------

    void DoMatrixProductOf(ElemType beta, bool transC, const TensorView& a, bool transA, const TensorView& b, bool transB, ElemType alpha, shared_ptr<QuantizedMultiplier<ElemType>> pQuantizedMultiplier = nullptr);
    void AssignMatrixProductOf(           bool transC, const TensorView& a, bool transA, const TensorView& b, bool transB, ElemType alpha = 1.0f, shared_ptr<QuantizedMultiplier<ElemType>> pQuantizedMultiplier = nullptr) { DoMatrixProductOf(0, transC, a, transA, b, transB, alpha, pQuantizedMultiplier); }
    void AddMatrixProductOf   (           bool transC, const TensorView& a, bool transA, const TensorView& b, bool transB, ElemType alpha = 1.0f) { DoMatrixProductOf(1.0f, transC, a, transA, b, transB, alpha); }

    shared_ptr<Matrix<ElemType>> AsMatrix() const;
    const TensorShape& GetShape() const { return m_shape; }

    // -------------------------------------------------------------------
    // accessors
    // -------------------------------------------------------------------

    const Matrix<ElemType>& GetSOB() const { return *m_sob; }
    Matrix<ElemType>&       GetSOB()       { return *m_sob; }
    friend Test::TensorTest<ElemType>;

private:
    // -------------------------------------------------------------------
    // sob members
    // -------------------------------------------------------------------

    shared_ptr<Matrix<ElemType>> m_sob; // Storage OBject that holds the data that is being viewed with this TensorView. This is really a reference (not owing the buffer).
    TensorShape m_shape;                // the meta-data that describes the data's shape and/or access pattern
};

}}}

#pragma warning(pop)
