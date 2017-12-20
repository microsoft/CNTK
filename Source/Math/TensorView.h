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
    typedef typename Matrix<ElemType>::MatrixPtr MatrixPtr;

    // -------------------------------------------------------------------
    // construction
    // -------------------------------------------------------------------

    // main constructor
    TensorView(const MatrixPtr& sob, const TensorShape& shape);
    // reinterpret a matrix storage object (SOB) as a TensorView with a given TensorShape  --this is the main constructor
    __forceinline TensorView(const MatrixBasePtr& sob, const TensorShape& shape)
        : TensorView(dynamic_pointer_cast<Matrix<ElemType>>(sob), shape)
    {
        if (!m_sob)
            LogicError("TensorView: Attempted to create a TensorView<ElemType> on a storage object of a different ElemType.");
    }
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

    // change the view onto the storage object (returns a new view)
    // This updates the TensorShape including slice information and offset.
    TensorView<ElemType> Reviewed(const TensorShape& shape) const
    {
        return TensorView(*this, shape);
    }

    // change the shape (returns a new view)
    // This updates the dimensions while retaining the offset. (And in the future possibly the slice information if the new shape allows that.)
    TensorView<ElemType> Reshaped(const TensorShape& dims) const
    {
        TensorShape tensorShape = m_shape;
        tensorShape.ReshapeInPlace(dims.GetDims()); // this retains the offset
        return TensorView(*this, tensorShape);
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

    static ElementWiseOperator OpFromName(const wstring& opName);

#pragma push_macro("DeclareNullaryTensorOp")
#define DeclareNullaryTensorOp(oper) \
    void     Do##oper##Of(ElemType beta, ElemType alpha       ) { Do<1>(1-1, { ViewRef(*this) }, ElementWiseOperator::op##oper, ElementWiseOperator::opSum, alpha, beta); } \
    void Assign##oper##Of(               ElemType alpha = 1.0f) { Do<1>(1-1, { ViewRef(*this) }, ElementWiseOperator::op##oper, ElementWiseOperator::opSum, alpha, 0   ); } \
    void    Add##oper##Of(               ElemType alpha = 1.0f) { Do<1>(1-1, { ViewRef(*this) }, ElementWiseOperator::op##oper, ElementWiseOperator::opSum, alpha, 1.0f); }

    ForAllNullaryOps(DeclareNullaryTensorOp);
#pragma pop_macro("DeclareNullaryTensorOp")

#pragma push_macro("DeclareUnaryTensorOp")
#define DeclareUnaryTensorOp(oper) \
    void     Do##oper##Of(ElemType beta, const TensorView& a, ElemType alpha       ) { Do<2>(2-1, { ViewRef(a), ViewRef(*this) }, ElementWiseOperator::op##oper, ElementWiseOperator::opSum, alpha, beta); } \
    void Assign##oper##Of(               const TensorView& a, ElemType alpha = 1.0f) { Do<2>(2-1, { ViewRef(a), ViewRef(*this) }, ElementWiseOperator::op##oper, ElementWiseOperator::opSum, alpha, 0   ); } \
    void    Add##oper##Of(               const TensorView& a, ElemType alpha = 1.0f) { Do<2>(2-1, { ViewRef(a), ViewRef(*this) }, ElementWiseOperator::op##oper, ElementWiseOperator::opSum, alpha, 1.0f); }

    ForAllUnaryOps(DeclareUnaryTensorOp);
#pragma pop_macro("DeclareUnaryTensorOp")

#pragma push_macro("DeclareBinaryTensorOp")
#define DeclareBinaryTensorOp(oper) \
    void     Do##oper##Of(ElemType beta, const TensorView& a, const TensorView& b, ElemType alpha       ) { Do<3>(3-1, { ViewRef(a), ViewRef(b), ViewRef(*this) }, ElementWiseOperator::op##oper, ElementWiseOperator::opSum, alpha, beta); } \
    void Assign##oper##Of(               const TensorView& a, const TensorView& b, ElemType alpha = 1.0f) { Do<3>(3-1, { ViewRef(a), ViewRef(b), ViewRef(*this) }, ElementWiseOperator::op##oper, ElementWiseOperator::opSum, alpha, 0   ); } \
    void    Add##oper##Of(               const TensorView& a, const TensorView& b, ElemType alpha = 1.0f) { Do<3>(3-1, { ViewRef(a), ViewRef(b), ViewRef(*this) }, ElementWiseOperator::op##oper, ElementWiseOperator::opSum, alpha, 1.0f); }

    ForAllBinaryOps(DeclareBinaryTensorOp);
#pragma pop_macro("DeclareBinaryTensorOp")

#pragma push_macro("DeclareTernaryTensorOp")
#define DeclareTernaryTensorOp(oper) \
    void     Do##oper##Of(ElemType beta, const TensorView& a, const TensorView& b, const TensorView& c, ElemType alpha       ) { Do<4>(4-1, { ViewRef(a), ViewRef(b), ViewRef(c), ViewRef(*this) }, ElementWiseOperator::op##oper, ElementWiseOperator::opSum, alpha, beta); } \
    void Assign##oper##Of(               const TensorView& a, const TensorView& b, const TensorView& c, ElemType alpha = 1.0f) { Do<4>(4-1, { ViewRef(a), ViewRef(b), ViewRef(c), ViewRef(*this) }, ElementWiseOperator::op##oper, ElementWiseOperator::opSum, alpha, 0   ); } \
    void    Add##oper##Of(               const TensorView& a, const TensorView& b, const TensorView& c, ElemType alpha = 1.0f) { Do<4>(4-1, { ViewRef(a), ViewRef(b), ViewRef(c), ViewRef(*this) }, ElementWiseOperator::op##oper, ElementWiseOperator::opSum, alpha, 1.0f); }

    ForAllTernaryOps(DeclareTernaryTensorOp);
#pragma pop_macro("DeclareTernaryTensorOp")

#pragma push_macro("DeclareQuaternaryTensorOp")
#define DeclareQuaternaryTensorOp(oper) \
    void     Do##oper##Of(ElemType beta, const TensorView& a, const TensorView& b, const TensorView& c, const TensorView& d, ElemType alpha       ) { Do<5>(5-1, { ViewRef(a), ViewRef(b), ViewRef(c), ViewRef(d), ViewRef(*this) }, ElementWiseOperator::op##oper, ElementWiseOperator::opSum, alpha, beta); } \
    void Assign##oper##Of(               const TensorView& a, const TensorView& b, const TensorView& c, const TensorView& d, ElemType alpha = 1.0f) { Do<5>(5-1, { ViewRef(a), ViewRef(b), ViewRef(c), ViewRef(d), ViewRef(*this) }, ElementWiseOperator::op##oper, ElementWiseOperator::opSum, alpha, 0   ); } \
    void    Add##oper##Of(               const TensorView& a, const TensorView& b, const TensorView& c, const TensorView& d, ElemType alpha = 1.0f) { Do<5>(5-1, { ViewRef(a), ViewRef(b), ViewRef(c), ViewRef(d), ViewRef(*this) }, ElementWiseOperator::op##oper, ElementWiseOperator::opSum, alpha, 1.0f); }

    ForAllQuaternaryOps(DeclareQuaternaryTensorOp);
#pragma pop_macro("DeclareQuaternaryTensorOp")

    // all different arities are routed through a single function template
    template<size_t N> static void Do(size_t arity, const std::array<std::reference_wrapper<TensorView<ElemType>>, N>& args, ElementWiseOperator op, ElementWiseOperator reductionOp, ElemType alpha, ElemType beta);
    static std::reference_wrapper<TensorView> ViewRef(const TensorView& arg) { return std::ref(const_cast<TensorView&>(arg)); } // helper for calling Do()

    // some code may use this interface
    inline void DoNullaryOpOf   (ElemType beta,                                                                                     ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp) { Do<1>(0, {                                                 ViewRef(*this) }, op, reductionOp, alpha, beta); }
    inline void DoUnaryOpOf     (ElemType beta, const TensorView& a,                                                                ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp) { Do<2>(1, { ViewRef(a),                                     ViewRef(*this) }, op, reductionOp, alpha, beta); }
    inline void DoBinaryOpOf    (ElemType beta, const TensorView& a, const TensorView& b,                                           ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp) { Do<3>(2, { ViewRef(a), ViewRef(b),                         ViewRef(*this) }, op, reductionOp, alpha, beta); }
    inline void DoTernaryOpOf   (ElemType beta, const TensorView& a, const TensorView& b, const TensorView& c,                      ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp) { Do<4>(3, { ViewRef(a), ViewRef(b), ViewRef(c),             ViewRef(*this) }, op, reductionOp, alpha, beta); }
    inline void DoQuaternaryOpOf(ElemType beta, const TensorView& a, const TensorView& b, const TensorView& c, const TensorView& d, ElemType alpha, ElementWiseOperator op, ElementWiseOperator reductionOp) { Do<5>(4, { ViewRef(a), ViewRef(b), ViewRef(c), ViewRef(d), ViewRef(*this) }, op, reductionOp, alpha, beta); }

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

    // -------------------------------------------------------------------
    // gather/scatter batch -- splice multiple TensorViews into/back from a
    // batched tensor along the trailing (slowest-increasing) axis.
    // Gather: Result overwrites 'this'.
    // Scatter: 'this' is the input; target objects are 'outputs'.
    // The batched tensor must match all individual tensor's 1-paded shapes,
    // except for the trailing dimension, which must be the sum.
    // Instead of passing TensorView objects, a functor is passed, to avoid an unnecessary malloc().
    // -------------------------------------------------------------------

    template<typename T> // TODO: move this to a more generic place
    struct IArrayRef
    {
        virtual size_t size() const = 0;
        virtual T* data() const = 0;
        virtual T operator[](size_t i) const { return data()[i]; }
        // TODO: ^^ how to make this a T for simple types, and a ref for complex ones?
        virtual T& operator[](size_t i) { return const_cast<IArrayRef*>(this)->operator[](i); }
        virtual const T* begin() const { return data(); }; // TODO: get the const-ness thingy right
        virtual const T* end() const { return data() + size(); }
    };

    void DoGatherBatchOf(const IArrayRef<const TensorView*>& inputs, size_t axis);
    void DoScatterBatchOf(ElemType beta, const IArrayRef<TensorView*>& outputs, size_t axis) const;

    MatrixPtr AsMatrix() const;
    const TensorShape& GetShape() const { return m_shape; }

    // -------------------------------------------------------------------
    // accessors
    // -------------------------------------------------------------------

    const MatrixPtr&        GetSOBPtr() const { return m_sob; }
    const Matrix<ElemType>& GetSOB()    const { return *m_sob; }
    Matrix<ElemType>&       GetSOB()          { return *m_sob; }
    friend Test::TensorTest<ElemType>;

    MatrixPtr GetSOBViewPtr() const;

    // -------------------------------------------------------------------
    // others
    // -------------------------------------------------------------------

    std::string AsString(size_t maxItems = 6, bool columnMajor = true) const;

private:
    // -------------------------------------------------------------------
    // sob members
    // -------------------------------------------------------------------

    MatrixPtr m_sob;     // Storage OBject that holds the data that is being viewed with this TensorView. This is really a reference (not owing the buffer).
    TensorShape m_shape; // the meta-data that describes the data's shape and/or access pattern
};

}}}

#pragma warning(pop)
