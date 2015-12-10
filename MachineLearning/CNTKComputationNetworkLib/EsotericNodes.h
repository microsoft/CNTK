//
// <copyright file="EsotericNodes.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include "ComputationNode.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // This header collects special-purpose nodes.
    // It is likely that these are no longer functional.

    // -----------------------------------------------------------------------
    /// DummyCriterionNode (objectives, derivatives, prediction)
    // -----------------------------------------------------------------------

    // This training criterion node needs derivatives and objectives to be
    // computed out of the node. Derivatives and objectives will be fed to the
    // node as input features. It has 3 inputs:
    // 1. feature node that feeds objectives
    // 2. feature node that feeds derivatives
    // 3. neural network output
    //
    // This node is useful in sequence training for speech recognition, so that
    // we can separate lattice computation (which may rely other softwares, such
    // as Kaldi) with the neural network training.
    template<class ElemType>
    class DummyCriterionNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>, public NumInputs<3>
    {
        typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"DummyCriterion"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(DummyCriterionNode);
        DummyCriterionNode(DEVICEID_TYPE deviceId, const wstring & name) :
          Base(deviceId, name)
        { }

        virtual void BackpropToNonLooping(size_t inputIndex) override
        {
            FrameRange fr(Input(0)->GetMBLayout());
            if (inputIndex == 0)
                LogicError("DummyCriterionNode: derivatives with respect to objective features are not necessary, not implemented yet.\n");
            else if (inputIndex == 1)
                LogicError("DummyCriterionNode: derivatives with respect to derivative features are not necessary, not implemented yet.\n");
            else if (inputIndex == 2)
            {
                auto gradient = Input(2)->GradientFor(fr);
                //Matrix<ElemType>::ScaleAndAdd(Gradient().Get00Element(), Input(1)->ValueFor(fr), gradient);
                Matrix<ElemType>::Multiply1x1AndWeightedAdd(+1.0f, Gradient()/*1x1*/, Input(1)->ValueFor(fr), 1.0f, gradient);
            }
        }

        virtual void /*ComputationNodeNonLooping::*/ForwardPropNonLooping() override
        {
            Value().VerifySize(1, 1);
            Input(0)->Value().VerifySize(1, 1);
            Value().SetValue(Input(0)->Value());
#if NANCHECK
            Value().HasNan("DummyCriterionNode");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            if (Input(0)->OperationName() != L"InputValue")
                LogicError("DummyCriterionNode criterion requires the first input to be computed objectives.");
            if (Input(0)->OperationName() != L"InputValue")
                LogicError("DummyCriterionNode criterion requires the first input to be computed derivatives.");
            if (isFinalValidationPass)
            {
                if (Input(0)->GetNumRows() != 1)
                LogicError("DummyCriterionNode criterion requires the first input to have dimension 1.");
                if (Input(0)->GetNumRows() == 0 || Input(1)->GetNumRows() == 0 || Input(2)->GetNumRows() == 0)
                    LogicError("DummyCriterionNode operation: one of the operands has 0 elements.");
                if (Input(1)->GetNumRows() != Input(2)->GetNumRows())
                LogicError("The Matrix dimension in the DummyCriterionNode operation does not match.");
            }

            SetDims(1,1);
            m_pMBLayout = nullptr;    // this node does not hold mini-batch data
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            m_sampleLayout = TensorShape();
        }
    };

    template class DummyCriterionNode<float>; 
    template class DummyCriterionNode<double>;


    // -----------------------------------------------------------------------
    // SequenceDecoderNode (label, position_dependent_score, transition_score)
    // this node does sequence decoding only
    // it corresponds to a decoder
    //  - label : output label vector of [0:T-1]
    //  - position_dependent_score : score from position dependent node,
    //    in the R-CRF case, it is the RNN output score before softmax
    //  - transition score : score from the transition node, 
    //    in the R-CRF case, it is the transition probability between labels
    // -----------------------------------------------------------------------

    template<class ElemType>
    class SequenceDecoderNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>, public NumInputs<3>
    {
        typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"SequenceDecoderNode"; }
    private:
        // TODO: member variables go to the end
        Matrix<ElemType> mAlpha;
        Matrix<ElemType> mBacktrace;

        int mStartLab; // the starting output label
        int mEndLab;   // the ending output label, if avaliable
        ElemType  m_default_activity;
    public:
        DeclareConstructorFromConfigWithNumInputs(SequenceDecoderNode);
        SequenceDecoderNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name),
            mAlpha(deviceId), mBacktrace(deviceId),
            mStartLab(-1), mEndLab(-1)
        { }

        static void DecideStartEndingOutputLab(const Matrix<ElemType>& lbls, int & stt, int & stp)
        {
            if (stt != -1 && stp != -1)
                return; /// have computed before

            int iNumPos = lbls.GetNumCols();

            int firstLbl = -1;
            for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                if (lbls(ik, 0) != 0){
                    firstLbl = ik; break;
                }

            int lastLbl = -1;
            for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                if (lbls(ik, iNumPos - 1) != 0){
                    lastLbl = ik; break;
                }

            stt = firstLbl;
            stp = lastLbl;
        };

        virtual void BackpropToNonLooping(size_t /*inputIndex*/) override  //scaled by 2*number of elements in the Matrix<ElemType>
        {
            LogicError("SequenceDecoder is used for evaluation only.");
        }

        /// compute posterior probability of label y at position t
        virtual void /*ComputationNodeNonLooping::*/ForwardPropNonLooping() override
        {
            DecideStartEndingOutputLab(Input(0)->Value(), mStartLab, mEndLab);
            ForwardPropS(mAlpha, mBacktrace, Value(), Input(1)->Value(),
                              Input(2)->Value(), mStartLab, mEndLab);
        }

        // compute forward backward algorithm
        void ForwardPropS(Matrix<ElemType>& alpha, Matrix<ElemType>& backtrace, Matrix<ElemType>& functionValues, const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, const size_t stt, const size_t stp)
        {
            /// to-do, each slice is for one sentence
            /// to-do, number of slices correspond to number of frames 
            /// this implementation only supports one sentence per minibatch

            /// change to other values so can support multiple sentences in each minibatch
            ForwardCompute(alpha, backtrace, pos_scores, pair_scores, stt);
            BackwardCompute(functionValues, backtrace, stp);

        };

        /// compute forward backward algorithm
        static void ForwardCompute(Matrix<ElemType>& alpha,
            Matrix<ElemType>& backtrace,
            const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores,
            const size_t stt)
        {
            /// to-do, shift more than 1 to support muliple sentences per minibatch
            int iNumPos = pos_scores.GetNumCols();
            int iNumLab = pos_scores.GetNumRows();
            size_t iTmp = 0;

            /// need to have 
            alpha.Resize(iNumLab, iNumPos);
            backtrace.Resize(iNumLab, iNumPos);

            for (int t = 0; t < iNumPos; t++)
            {
                for (int k = 0; k < iNumLab; k++)
                {
                    ElemType fTmp = (ElemType)LZERO;
                    if (t > 1){
                        for (int j = 0; j < iNumLab; j++)
                        {
                            ElemType fAlpha = alpha(j, t - 1) + pair_scores(k, j);
                            if (fAlpha > fTmp){
                                fTmp = fAlpha;
                                iTmp = j;
                            }
                        }
                        fTmp += pos_scores(k, t);  /// include position dependent score
                    }
                    else
                    {
                        /// with constrain that the first word is labeled as a given symbol
                        iTmp = stt;
                        fTmp = 0;
                        if (t == 1){
                            fTmp = alpha(iTmp, t - 1);
                            fTmp += pair_scores(k, iTmp);
                            fTmp += pos_scores(k, t);
                        }
                        else {
                            fTmp = (k == stt) ? pos_scores(k, t) : (ElemType)LZERO;
                        }
                    }
                    alpha(k, t) = fTmp;
                    backtrace(k, t) = (ElemType)iTmp;
                }
            }

        };

        /// compute backward algorithm
        static void BackwardCompute(
            Matrix<ElemType>& decodedpath,
            const Matrix<ElemType>& backtrace, const size_t stp)
        {
            int iNumPos = backtrace.GetNumCols();
            int iNumLab = backtrace.GetNumRows();

            decodedpath.Resize(iNumLab, iNumPos);
            decodedpath.SetValue(0);

            size_t lastlbl = stp;
            decodedpath(lastlbl, iNumPos - 1) = 1;

            for (int t = iNumPos - 1; t > 0; t--)
            {
                lastlbl = (size_t)backtrace(lastlbl, t);
                decodedpath(lastlbl, t - 1) = 1;
            }
        };

        /// need to feed in pseudo label data, which tells the decoder what is the beginning
        /// and ending output symbol. these symbols will constrain the search space
        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            if (isFinalValidationPass)
                if (!(Input(1)->GetNumRows() == Input(2)->GetNumRows() &&  // position dependent and pair scores have same number of labels
                    Input(0)->GetNumRows() == Input(1)->GetNumRows() &&
                    Input(0)->GetNumCols() == Input(1)->GetNumCols() && // position dependent and pair scores have the same observation numbers
                    Input(2)->GetNumCols() == Input(2)->GetNumRows()))
                {
                    LogicError("The Matrix<ElemType>  dimension in the SequenceDecoderNode operation does not match.");
                }
            // BUGBUG: Not resizing FunctionValues?

            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            m_sampleLayout = TensorShape();
        }
    };

    template class SequenceDecoderNode<float>;
    template class SequenceDecoderNode<double>;

    // -----------------------------------------------------------------------
    // StrideTimesNode (left, right, stride/*0=row, 1=col*/)
    // TODO: why is 'stride' an Input and not just an initialization parameter?
    // -----------------------------------------------------------------------

    /**
    Has a stride in particular dimensions of left matrix when doing times operation. 
    Example 1: column stride s
    A in d x [s x T1] 
    B in T1 x s
    C = A x B  in d x s, and each element is computed as 
    c_{i,k} = \sum_j a_{i,j*s+k} b_{j,k}
    where s is the stride in column.

    Example 2:
    A in [s x T1] x d
    B in d x s
    C = A x B  in T1 x s, and each element is computed as
    c_{i,k} = \sum_j a_{i*s+k,j} b_{j,k}
    where s is the stride in rows.

    Notice that s is equal to k. 
    */
    template<class ElemType>
    class StrideTimesNode : public ComputationNode<ElemType>, public NumInputs<3>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"StrideTimes"; }

        size_t m_strideDim; // the dimension index on which stride works 
        size_t m_stride;    // the stride 
    private:
        void UpdateStride(const Matrix<ElemType>& input1) 
        {
            m_stride = input1.GetNumCols();
        }
    public:
        DeclareConstructorFromConfigWithNumInputs(StrideTimesNode);
        StrideTimesNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name),
            m_stride(1)
        { }
        // BUGBUG: This node needs to serialize and CopyTo m_stride

        virtual void /*ComputationNode::*/BackpropTo(const size_t inputIndex, const FrameRange & fr) override
        {
            if (fr.IsAllFrames()) { NOT_IMPLEMENTED; return; } // TODO: remove these one by one. And why is this not implemented?
            if (inputIndex > 2)
                InvalidArgument("StrideTimes operation only takes three inputs.");
            else if (inputIndex == 2)
                return;     // that's a constant

            Matrix<ElemType> sliceOutputGrad = GradientFor(fr);

            if (m_strideDim == 1) // column stride
            {
                if (inputIndex == 0)  //left derivative
                {
                    Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);

                    //BackpropToLeft1(sliceInput1Value, Input(0)->Gradient(), sliceOutputGrad);

                    size_t r = Input(0)->GetNumRows();
                    size_t T1 = Input(0)->GetNumCols() / GetNumParallelSequences();    // TODO: if T1 == GetNumTimeSteps() then we can simplify code below.
                    Matrix<ElemType> mTmp1(r, T1, sliceInput1Value.GetDeviceId());

                    // process sequence by sequence
                    for (size_t k = 0; k < GetNumParallelSequences(); k++)
                    {
                        mTmp1.SetValue(0);
                        auto mTmp2 = sliceInput1Value.ColumnSlice(k, 1);
                        auto mTmp3 = sliceOutputGrad.ColumnSlice(k, 1);

                        BackpropToLeft1(mTmp2, mTmp1, mTmp3);

                        for (size_t t = 0; t < T1; t++)
                        {
                            Input(0)->Gradient().ColumnSlice(t*GetNumParallelSequences() + k, 1) += mTmp1.ColumnSlice(t, 1);
                        }
                    }
                }
                else  //right derivative
                {
                    Matrix<ElemType> sliceInput1Grad = Input(1)->GradientFor(fr);

                    //BackpropToRight(Input(0)->Value(), sliceInput1Grad, sliceOutputGrad);

                    // process sequence by sequence
                    for (size_t k = 0; k < GetNumParallelSequences(); k++)
                    {
                        size_t r = Input(0)->GetNumRows();
                        size_t T1 = Input(0)->GetNumCols() / GetNumParallelSequences();    // TODO: if T1 == GetNumTimeSteps() then we can simplify code below.
                        Matrix<ElemType> mTmp1(r, T1, sliceOutputGrad.GetDeviceId());
                        for (size_t t = 0; t < T1; t++)
                        {
                            mTmp1.ColumnSlice(t, 1).SetValue(Input(0)->Value().ColumnSlice(t*GetNumParallelSequences() + k, 1));
                        }
                        auto mTmp2 = sliceInput1Grad.ColumnSlice(k, 1);
                        auto mTmp3 = sliceOutputGrad.ColumnSlice(k, 1);

                        BackpropToRight(mTmp1, mTmp2, mTmp3);
                    }
                }
            }
            else if (m_strideDim == 0) // row stride
            {
                if (inputIndex == 0)  //left derivative
                {
                    Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);

                    for (size_t k = 0; k < GetNumParallelSequences(); k++)
                    {
                        size_t d = Input(1)->GetNumRows();
                        size_t T1 = Input(0)->GetNumRows() / GetNumParallelSequences();
                        Matrix<ElemType> mTmp1(sliceInput1Value.GetDeviceId());
                        mTmp1.Resize(d, T1);
                        Matrix<ElemType> mTmp2 = sliceInput1Value.ColumnSlice(k, 1);
                        Matrix<ElemType> mTmp3 = sliceOutputGrad.ColumnSlice(k, 1);
                        BackpropToLeft(mTmp2, mTmp1, mTmp3);

                        Matrix<ElemType> mTmp4(sliceInput1Value.GetDeviceId());
                        for (size_t t = 0; t < T1; t++)
                        {
                            mTmp4 = mTmp1.ColumnSlice(t, 1);
                            mTmp4.Reshape(1, d);
                            Input(0)->Gradient().AddToRowSliceValuesOf(mTmp4, t*GetNumParallelSequences() + k, 1);
                        }
                    }
                }
                else  //right derivative
                {
                    Matrix<ElemType> sliceInput1Grad = Input(1)->GradientFor(fr);

                    for (size_t k = 0; k < GetNumParallelSequences(); k++)
                    {
                        size_t d = Input(1)->GetNumRows();
                        size_t T1 = Input(0)->GetNumRows() / GetNumParallelSequences();

                        Matrix<ElemType> mTmp0(sliceOutputGrad.GetDeviceId());
                        mTmp0.Resize(1, d);

                        Matrix<ElemType> mTmp1(sliceOutputGrad.GetDeviceId());
                        mTmp1.Resize(T1, d);
                        for (size_t t = 0; t < T1; t++)
                        {
                            mTmp0.SetValue(0);
                            mTmp0.AddWithRowSliceValuesOf(Input(0)->Value(), t * GetNumParallelSequences() + k, 1);
                            mTmp1.AssignToRowSliceValuesOf(mTmp0, t, 1);
                        }
                        Matrix<ElemType> mTmp2 = sliceInput1Grad.ColumnSlice(k, 1);
                        Matrix<ElemType> mTmp3 = sliceOutputGrad.ColumnSlice(k, 1);

                        BackpropToRight(mTmp1, mTmp2, mTmp3);
                    }
                }
            }
        }

        // TODO: the following two functions only differ in the order of argument use in the final MultiplyAndAdd()  --is that intended??
        static /*TODO: merge with call site*/void BackpropToLeft1(const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
        {
#if DUMPOUTPUT
            gradientValues.Print("Gradient-in");
            inputGradientValues.Print("child Gradient-in/out");
            inputFunctionValues.Print("child Function values");
#endif
            //currently we only support one combination when the input is sparse.
            if (inputFunctionValues.GetMatrixType() == SPARSE && inputGradientValues.GetMatrixType() == DENSE && gradientValues.GetMatrixType() == DENSE)
                inputGradientValues.SwitchToMatrixType(SPARSE, MatrixFormat::matrixFormatSparseBlockCol, false);

            Matrix<ElemType>::MultiplyAndAdd(gradientValues, false, inputFunctionValues, true, inputGradientValues);
#if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
#endif
        }

        static /*TODO: merge with call site*/void BackpropToLeft(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
        {
#if DUMPOUTPUT   
            gradientValues.Print("Gradient-in");   
            inputGradientValues.Print("child Gradient-in/out");   
            inputFunctionValues.Print("child Function values");   
#endif
            //currently we only support one combination when the input is sparse.   
            if (inputFunctionValues.GetMatrixType() == SPARSE && inputGradientValues.GetMatrixType() == DENSE && gradientValues.GetMatrixType() == DENSE)
                inputGradientValues.SwitchToMatrixType(SPARSE, MatrixFormat::matrixFormatSparseBlockCol, false);

            Matrix<ElemType>::MultiplyAndAdd(inputFunctionValues, false, gradientValues, true, inputGradientValues);

#if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
#endif
        }

        static /*TODO: merge with call site*/void BackpropToRight(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
        {
#if DUMPOUTPUT   
            gradientValues.Print("Gradient-in");   
            inputGradientValues.Print("child Gradient-in/out");
            inputFunctionValues.Print("child Function values");
#endif   
            Matrix<ElemType>::MultiplyAndAdd(inputFunctionValues, true, gradientValues, false, inputGradientValues);
#if DUMPOUTPUT
            inputGradientValues.Print("child Gradient-out");
#endif
        }

        virtual void /*ComputationNode::*/ForwardProp(const FrameRange & fr) override
        {
            size_t rows0 = Input(0)->GetNumRows(), cols1 = Input(1)->GetNumCols();
            Matrix<ElemType> sliceInput1Value = Input(1)->ValueFor(fr);
            UpdateStride(sliceInput1Value);

            if (m_strideDim == 0)
                SetDims(rows0 / GetNumParallelSequences(), cols1);
            if (m_strideDim == 1)       // TODO: no else??
                SetDims(rows0, cols1);

            Matrix<ElemType> sliceOutputValue = ValueFor(fr);

            // (TODO: these following assignments are leftovers of refactoring and can be short-circuited)
            Matrix<ElemType>& functionValues = sliceOutputValue;
            const Matrix<ElemType>& input0 = Input(0)->Value();
            const Matrix<ElemType>& input1 = sliceInput1Value;

            /**
            A in d x [s x T1]
            B in T1 x s
            C = A x B  in d x s, and each element is computed as 
            c_{i,k} = \sum_j a_{i,j*s+k} b_{j,k}
            C in d x s
            where s is the stride in column.
    
            Example 2:
            A in [s x T1] x d
            B in d x s
            C = A x B  in T1 x s, and each element is computed as
            c_{i,k} = \sum_j a_{i*s+k,j} b_{j,k}
            where s is the stride in rows.
            C in T1 x s
    
            strideDim : 0 or 1 (meaning to apply to row or column)
            */
#if DUMPOUTPUT
            input0.Print("StrideTimesNode - Input0");
#endif
            assert(m_strideDim == 0 || m_strideDim == 1);
            Matrix<ElemType> mTmp1(input0.GetDeviceId());
            Matrix<ElemType> mTmp2(input0.GetDeviceId());
            if (m_strideDim == 1) // 1 = col stride; the example 1 case at column
            {
                assert(m_stride == input1.GetNumCols());
                size_t T1 = input0.GetNumCols() / m_stride;
                assert(T1 == input1.GetNumRows());
                size_t d = input0.GetNumRows();
                functionValues.Resize(d, m_stride);
                for (size_t k = 0; k < m_stride; k++)
                {
                    mTmp1.Resize(d, T1);
                    for (size_t j = 0; j < T1; j++)
                    {
                        mTmp1.ColumnSlice(j, 1).SetValue(input0.ColumnSlice(j * m_stride + k, 1));
                    }

                    mTmp2 = input1.ColumnSlice(k, 1);
                    functionValues.ColumnSlice(k, 1).AssignProductOf(mTmp1, false, mTmp2, false);

                }
            }
            else if (m_strideDim == 0) // 0 = row stride; the example 2 case at row
            {
                assert(m_stride == input1.GetNumCols());
                size_t T1 = input0.GetNumRows() / m_stride;
                size_t d = input1.GetNumRows();
                assert(d == input0.GetNumCols());
                functionValues.Resize(T1, m_stride);
                mTmp1.Resize(d, T1);
                for (size_t k = 0; k < m_stride; k++)
                {
                    for (size_t j = 0; j < T1; j++)
                    {
                        mTmp1.ColumnSlice(j, 1).AssignRowSliceValuesOf(input0, k + j * m_stride, 1);
                    }

                    mTmp2 = input1.ColumnSlice(k, 1);
                    functionValues.ColumnSlice(k, 1).AssignProductOf(mTmp1, true, mTmp2, false);

                }
            }
#if NANCHECK
            functionValues.HasNan("StrideTimes");
#endif
#if DUMPOUTPUT
            functionValues.Print("StrideTimesNode");
#endif
        }

        /**
        three inputs
        input0: left matrix
        input1: right matrix
        stridedim: single element no gradient matrix, 0 row stride / 1 column stride
        */
        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            if (Input(2)->Value().GetNumElements() != 1)
                RuntimeError("%ls %ls operation: Input(2) should be a single element matrix and have the value 0 (row) or 1 (col).", NodeName().c_str(), OperationName().c_str());
            m_strideDim = (size_t) Input(2)->Value().Get00Element();
            if (m_strideDim != 0 && m_strideDim != 1)
                RuntimeError("%ls %ls operation: Input(2) should be a single element matrix and have the value 0 (row) or 1 (col).", NodeName().c_str(), OperationName().c_str());
            //if (Input(2)->m_needGradient)        // disabled because this is a flag that belongs to Network. Node should simply not propagate anything into it
            //    RuntimeError("StrideTimes: No gradient update should be on input(2).");

            size_t rows0 = Input(0)->GetNumRows(), cols0 = Input(0)->GetNumCols();
            size_t rows1 = Input(1)->GetNumRows(), cols1 = Input(1)->GetNumCols();

            if (m_strideDim == 0) // by row
            {
                if (isFinalValidationPass && rows1 != cols0)
                    RuntimeError("The Matrix dimension in the StrideTimes operation in dim %d does not match for cols %d in A and rows %d in B.", (int)m_strideDim, (int)cols0, (int)rows1);
                size_t T1 = rows0 / m_stride;
                SetDims(T1, cols1);
            }

            else // by col
            {
                if (isFinalValidationPass && cols0 != rows1 * m_stride)
                    RuntimeError("The Matrix dimension in the StrideTimes operation in dim %d does not match for cols %d in A and row number %d in B.", (int)m_strideDim, (int)cols0, (int)rows1);
                SetDims(rows0, cols1);
            }
            LinkToMBLayout(Input(1)->GetMBLayout());   // retains the layout of the right input

            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(1, false); //the second one is the input since it's column wize

            //after multiplication the structure is lost
            m_sampleLayout = ImageLayoutWHC(1, Input(0)->GetNumRows(), 1);
        }
    };

    template class StrideTimesNode<float>;
    template class StrideTimesNode<double>;

    // -----------------------------------------------------------------------
    // PairNetworkNode (input)
    // -----------------------------------------------------------------------

    /**
    pair this node to a node in another network
    this node provide an interface from this network. The next layer network then can use this interface to know which node to connect to.
    */
    template<class ElemType>
    class PairNetworkNode : public ComputationNode<ElemType>, public NumInputs<1>
    {
        typedef ComputationNode<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"PairNetwork"; }

        void Init(size_t row_size, size_t col_size)
        {
            CreateMatrixIfNull(m_value);
            SetDims(row_size, col_size);
            UpdateFunctionValuesSize();
        }
    public:
        DeclareConstructorFromConfigWithNumInputs(PairNetworkNode);
        PairNetworkNode(DEVICEID_TYPE deviceId, const wstring & name, size_t row_size = 1, size_t col_size = 1) :
            Base(deviceId, name)
        {
            Init(row_size, col_size);
            CreateMatrixIfNull(m_gradient);
            m_gradient->Resize(row_size, col_size);
            m_gradient->SetValue(0.0f);
        }

        virtual void Load(File& fstream, size_t modelVersion) override
        {
            Init(1, 1); // TODO: this looks wrong; should the dimension not come from the loaded model data?
            Base::Load(fstream, modelVersion);
        }

        /// to-do: need to change to the new way of resetting state
        void BackpropToMap(const size_t inputIndex)
        {
            if (inputIndex > 0)
                InvalidArgument("PairNetwork operation only takes one input.");

            Matrix<ElemType>::ScaleAndAdd(1.0, Gradient(), Input(inputIndex)->Gradient());
        }

        virtual void /*ComputationNode::*/BackpropTo(const size_t inputIndex, const FrameRange & fr) override
        {
            if (fr.IsAllFrames()) { BackpropToMap(inputIndex); return; } // TODO: remove these one by one
            assert(m_value->GetNumRows() == Gradient().GetNumRows()); // original used m_value->GetNumRows() for loop dimension
            assert(m_pMBLayout);

            Matrix<ElemType> mTmp = Input(inputIndex)->GradientFor(fr);
            Matrix<ElemType>::ScaleAndAdd(1.0, GradientFor(fr), mTmp);
        }

        virtual void /*ComputationNode::*/ForwardProp(const FrameRange & fr) override
        {
            Matrix<ElemType> mTmp = ValueFor(fr);
            mTmp.SetValue(Input(0)->ValueFor(fr));
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            size_t rows0 = Input(0)->GetNumRows(), cols0 = Input(0)->GetNumCols();
            if (rows0 > 0 && cols0 > 0) // TODO: is this check needed?
                SetDims(Input(0));

            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();
        }
    };

    template class PairNetworkNode<float>;
    template class PairNetworkNode<double>;

    // -----------------------------------------------------------------------
    // LSTMNode (obs, inputGate, forgetGate, outputGate, memoryCellWgt)
    // deprecated early implementation of LSTM operating on minibatches directly
    //  - input(0) : child with dimension [inputdim x T]
    //  - input(1) : input gate [outputdim x [inputdim + outputdim + 2]] bi, Wxi, Whi, Wci
    //  - input(2) : forget gate [outputdim x [inputdim + outputdim + 2]] for bf, Wxf, Whf, Wcf
    //  - input(3) : output gate [outputdim x [inputdim + outputdim + 2]] for bo, Wxo, Who, and Wco
    //  - input(4) : memory cell weight [outputdim x [inputdim + outputdim + 1]] for bc, Wxc, and Whc 
    //  - output : dimension [outputdim x T]
    // -----------------------------------------------------------------------

    /**
    LSTM specific node. This node uses matrix operations to have LSTM functionality. 
    It avoids using general recurrent loop operations in the network operations in ComputationNetwork.

    Developed by Kaisheng Yao
    Used in the following works:
    K. Yao, G. Zweig, "Sequence to sequence neural net models for graphone to phoneme conversion", in Interspeech 2015
    */
    template<class ElemType>
    class LSTMNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>, public NumInputs<5>
    {
        typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"LSTM"; }
    public:
        DeclareConstructorFromConfigWithNumInputs(LSTMNode);
        LSTMNode(DEVICEID_TYPE deviceId, const wstring & name) : Base(deviceId, name),
            m_State(deviceId), m_PastState(deviceId),
            m_PastOutput(deviceId), m_Gi(deviceId), m_Gf(deviceId), m_Go(deviceId), grdToObs(deviceId), grdToInputGate(deviceId),
            grdToForgetGate(deviceId), grdToOutputGate(deviceId), grdToCellWgt(deviceId), tanhObs(deviceId),
            tanhState(deviceId), m_tempMatrix(deviceId),
            mSlicePrevState(deviceId), mSlicePrevOutput(deviceId),
            grdBeforeInputGate(deviceId),
            grdBeforeForget(deviceId), grdBeforeGo(deviceId), grdToCell(deviceId),
            grdBeforeTanhInputGate(deviceId), m_obs_error_from_future_minibatch(deviceId),
            m_state_error_from_future_minibatch(deviceId), mLastState(deviceId), mLastOutput(deviceId),
            m_inputDim(0),
            m_outputDim(0),
            m_use_errors_from_future_minibatch(false),
            m_DefaultState((ElemType)DEFAULT_HIDDEN_ACTIVATION)
        {
        }

        virtual void Save(File& fstream) const override
        {
            Base::Save(fstream);
            fstream << m_inputDim << m_outputDim;
            fstream << m_DefaultState;
        }

        virtual void Load(File& fstream, size_t modelVersion) override
        {
            Base::Load(fstream, modelVersion);
            if (modelVersion == 2)
                fstream >> m_inputDim >> m_outputDim;
            fstream >> m_DefaultState;
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<LSTMNode<ElemType>>(nodeP);
                node->m_inputDim = m_inputDim;
                node->m_outputDim = m_outputDim;

                node->m_State = m_State;  // hidden state activity
                node->m_PastState = m_PastState; // state activity in the previous minibatch
                node->m_PastOutput = m_PastOutput; // output in the previou minibatch 

                node->m_Gi = m_Gi;     // input gate activity
                node->m_Gf = m_Gf;     // forget gate activity
                node->m_Go = m_Go;     // output gate activity

                node->mSlicePrevOutput = mSlicePrevOutput;
                node->mSlicePrevState = mSlicePrevState;

                node->m_use_errors_from_future_minibatch = m_use_errors_from_future_minibatch;

                node->m_DefaultState = m_DefaultState;
            }
        }

        virtual void BackpropToNonLooping(size_t inputIndex) override
        {
            if (inputIndex > 4)
                InvalidArgument("LSTM operation only takes five inputs.");

            size_t nT = Input(0)->GetNumCols();
            size_t inputDim = Input(0)->GetNumRows();
            size_t outputDim = Input(1)->GetNumRows();

            if (m_GradientComputed == false)
            {
                if (GetNumCols() != Gradient().GetNumCols() ||
                    GetNumRows() != Gradient().GetNumRows())
                {
                    RuntimeError("LSTMNode::GradientValue size doesn't match to the function value size");
                }

                // reset gradients
                grdToObs.Resize(inputDim, nT); grdToObs.SetValue(0);
                grdToInputGate.Resize(Input(1)->GetNumRows(), Input(1)->GetNumCols()); grdToInputGate.SetValue(0);
                grdToForgetGate.Resize(Input(2)->GetNumRows(), Input(2)->GetNumCols()); grdToForgetGate.SetValue(0);
                grdToOutputGate.Resize(Input(3)->GetNumRows(), Input(3)->GetNumCols()); grdToOutputGate.SetValue(0);
                grdToCellWgt.Resize(Input(4)->GetNumRows(), Input(4)->GetNumCols()); grdToCellWgt.SetValue(0);

                Matrix<ElemType> slicePrevOutput(m_deviceId), slicePrevState(m_deviceId);
                Matrix<ElemType> grdToPrevOutput(m_deviceId), grdToPrevState(m_deviceId);
                Matrix<ElemType> stateError(m_deviceId);
                slicePrevState.Resize(outputDim, GetNumParallelSequences());
                slicePrevOutput.Resize(outputDim, GetNumParallelSequences());
                slicePrevOutput.SetValue(0);

                stateError.Resize(slicePrevState.GetNumRows(), slicePrevState.GetNumCols());

                grdToPrevOutput.Resize(slicePrevOutput.GetNumRows(), slicePrevOutput.GetNumCols());
                grdToPrevState.Resize(slicePrevState.GetNumRows(), slicePrevState.GetNumCols());
                grdToPrevOutput.SetValue(0);
                grdToPrevState.SetValue(0);

                for (int timeIdxInSeq = nT - GetNumParallelSequences(); timeIdxInSeq >= 0; timeIdxInSeq -= GetNumParallelSequences())
                {
                    FrameRange fr(m_pMBLayout, timeIdxInSeq);
                    Matrix<ElemType> sliceObs = Input(0)->ValueFor(fr);
                    Matrix<ElemType> sliceOutput = ValueFor(fr);
                    Matrix<ElemType> sliceState = DataFor(m_State, fr);

                    Matrix<ElemType> sliceGi = DataFor(m_Gi, fr);
                    Matrix<ElemType> sliceGf = DataFor(m_Gf, fr);
                    Matrix<ElemType> sliceGo = DataFor(m_Go, fr);

                    Matrix<ElemType> sliceTanhState = DataFor(tanhState, fr);
                    Matrix<ElemType> sliceTanhObs = DataFor(tanhObs, fr);

                    Matrix<ElemType> error = GradientFor(fr);

                    Matrix<ElemType> grdToObsSlice(this->m_deviceId);


#ifdef DEBUG_DECODER
                    fprintf(stderr, "original output error [%ld] norm = %.8e\n", timeIdxInSeq, error.FrobeniusNorm());
#endif

                    PrepareThisErrorsBeforeBackProp(timeIdxInSeq, nT, error, stateError, grdToPrevOutput, grdToPrevState,
                                                    m_obs_error_from_future_minibatch, m_state_error_from_future_minibatch, GetNumParallelSequences(), &m_pMBLayout->GetM());

#ifdef DEBUG_DECODER
                    fprintf(stderr, "output error [%ld] norm = %.8e\n", timeIdxInSeq, error.FrobeniusNorm());
                    fprintf(stderr, "state error [%ld] norm = %.8e\n", timeIdxInSeq, stateError.FrobeniusNorm());
#endif

                    grdToPrevOutput.Resize(slicePrevOutput.GetNumRows(), slicePrevOutput.GetNumCols());
                    grdToPrevState.Resize(slicePrevState.GetNumRows(), slicePrevState.GetNumCols());
                    grdToPrevOutput.SetValue(0);
                    grdToPrevState.SetValue(0);

                    PrepareHistory(timeIdxInSeq, mSlicePrevOutput, mSlicePrevState, Value(), m_State, m_PastOutput, m_PastState, GetNumParallelSequences(), m_DefaultState, &m_pMBLayout->GetM());

                    ComputeInputGradientWrtGates(
                        error,
                        sliceObs,
                        grdToObsSlice,
                        Input(1)->Value(),
                        grdToInputGate,
                        Input(2)->Value(),
                        grdToForgetGate,
                        Input(3)->Value(),
                        grdToOutputGate,
                        Input(4)->Value(),
                        grdToCellWgt,
                        mSlicePrevOutput,
                        mSlicePrevState,
                        stateError,
                        sliceState,
                        sliceTanhState,
                        sliceTanhObs,
                        sliceGi,
                        sliceGf,
                        sliceGo,
                        grdToPrevOutput,
                        grdToPrevState,
                        m_tempMatrix
                    );
                    DataFor(grdToObs, fr).SetValue(grdToObsSlice);

                    PrepareErrors(timeIdxInSeq, grdToPrevOutput, grdToPrevState, GetNumParallelSequences(), &m_pMBLayout->GetM());
                }
#ifdef DEBUG_DECODER
                fprintf(stderr, "after error prop b_c norm = %.8e\n", Input(4)->Value().ColumnSlice(0, 1).FrobeniusNorm());
#endif
                m_obs_error_from_future_minibatch = grdToPrevOutput;
                m_state_error_from_future_minibatch = grdToPrevState;


#ifdef DEBUG_DECODER
                fprintf(stderr, "pass error to encoder error = %.4e state error = %.4e\n", m_obs_error_from_future_minibatch.FrobeniusNorm(), m_state_error_from_future_minibatch.FrobeniusNorm());
#endif
                m_GradientComputed = true;
            }

            if (inputIndex == 0)  //derivative with regard to the observation
            {
                if (Input(inputIndex)->Gradient().HasNoElements())
                    Input(inputIndex)->Gradient().SetValue(grdToObs);
                else
                    Input(inputIndex)->Gradient() += grdToObs;
            }

            if (inputIndex == 1)
            {
                if (Input(inputIndex)->Gradient().HasNoElements())
                    Input(inputIndex)->Gradient().SetValue(grdToInputGate);
                else
                    Input(inputIndex)->Gradient() += grdToInputGate;
            }

            if (inputIndex == 2)
            {
                if (Input(inputIndex)->Gradient().HasNoElements())
                    Input(inputIndex)->Gradient().SetValue(grdToForgetGate);
                else
                    Input(inputIndex)->Gradient() += grdToForgetGate;
            }

            if (inputIndex == 3)
            {
                if (Input(inputIndex)->Gradient().HasNoElements())
                    Input(inputIndex)->Gradient().SetValue(grdToOutputGate);
                else
                    Input(inputIndex)->Gradient() += grdToOutputGate;
            }

            if (inputIndex == 4)
            {
                if (Input(inputIndex)->Gradient().HasNoElements())
                    Input(inputIndex)->Gradient().SetValue(grdToCellWgt);
                else
                    Input(inputIndex)->Gradient() += grdToCellWgt;
            }
#ifdef DEBUG_DECODER
            fprintf(stderr, "LSTM gradient[%d] norm = %.8e\n", inputIndex, Input(inputIndex)->Gradient().FrobeniusNorm());
#endif

        }

        static void WINAPI GradientOfTanh(const Matrix<ElemType>& functionValues,
            const Matrix<ElemType>& gradientOut,
            Matrix<ElemType>& inputGradientValues,
            Matrix<ElemType>& extTmp)
        {
            Matrix<ElemType> mTmp(inputGradientValues.GetDeviceId());
            extTmp.AssignElementProductOf(functionValues, functionValues); // v .* v
            mTmp.AssignDifferenceOf(1, extTmp); // 1-v^2
            if (inputGradientValues.GetNumRows() != functionValues.GetNumRows() ||
                inputGradientValues.GetNumCols() != functionValues.GetNumCols())
                LogicError("LSTMNode::GradientOfTanh : inputGradientValues need to be pre-allocated!");
            inputGradientValues.AddElementProductOf(gradientOut, mTmp); //  d .* ((1-v) .* v))
        }

        static void WINAPI ComputeInputGradientWrtGates(
            const Matrix<ElemType>& outGrd,  // the error to h_t from upper layer
            const Matrix<ElemType> & obs,
            Matrix<ElemType> &grdToObs,
            const Matrix<ElemType>& mInputGate,
            Matrix<ElemType> &grdToInputGate,
            const Matrix<ElemType> &mForgetGate,
            Matrix<ElemType> &grdToForgetGate,
            const Matrix<ElemType> &mOutputGate,
            Matrix<ElemType>& grdToOutputGate,
            const Matrix<ElemType> &mCellWgt,
            Matrix<ElemType> &grdToCellWgt,
            const Matrix<ElemType>& prevOutput,
            const Matrix<ElemType>& prevState,
            const Matrix<ElemType>& stateError,  // the error propagated to cell from t+1
            const Matrix<ElemType> &state,
            const Matrix<ElemType> &tanhState,
            const Matrix<ElemType> & tanhBeforeApplyingInputGating,
            const Matrix<ElemType> &gi,
            const Matrix<ElemType> &gf,
            const Matrix<ElemType> &go,
            Matrix<ElemType> &grdToPrevOutput,
            Matrix<ElemType> &grdToPrevState,
            Matrix<ElemType> & tmpMat
            )
        {
            int inputDim = obs.GetNumRows();
            int outputDim = mOutputGate.GetNumRows();

            assert(grdToPrevOutput.FrobeniusNorm() == 0);
            assert(grdToPrevState.FrobeniusNorm() == 0);
            assert(state.FrobeniusNorm() > 0);
            Matrix<ElemType> Who = mOutputGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> Wco = mOutputGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> Wxo = mOutputGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdToWho = grdToOutputGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> grdToWco = grdToOutputGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> grdToWxo = grdToOutputGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdTobo = grdToOutputGate.ColumnSlice(0, 1);

            Matrix<ElemType> Whf = mForgetGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> Wcf = mForgetGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> Wxf = mForgetGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdToWhf = grdToForgetGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> grdToWcf = grdToForgetGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> grdToWxf = grdToForgetGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdTobf = grdToForgetGate.ColumnSlice(0, 1);

            Matrix<ElemType> Wxc = mCellWgt.ColumnSlice(1, inputDim);
            Matrix<ElemType> Whc = mCellWgt.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> grdToWxc = grdToCellWgt.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdToWhc = grdToCellWgt.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> grdTobc = grdToCellWgt.ColumnSlice(0, 1);

            Matrix<ElemType> Whi = mInputGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> Wci = mInputGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> Wxi = mInputGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdToWhi = grdToInputGate.ColumnSlice(1 + inputDim, outputDim);
            Matrix<ElemType> grdToWci = grdToInputGate.ColumnSlice(1 + inputDim + outputDim, 1);
            Matrix<ElemType> grdToWxi = grdToInputGate.ColumnSlice(1, inputDim);
            Matrix<ElemType> grdTobi = grdToInputGate.ColumnSlice(0, 1);

            // error backpropagate to output gate
            Matrix<ElemType> grdToGo(tmpMat.GetDeviceId()), gradientOfSigmoid(tmpMat.GetDeviceId());
            Matrix<ElemType> grdBeforeGo(tmpMat.GetDeviceId()), grdBeforeInputGate(tmpMat.GetDeviceId());
            Matrix<ElemType> grdToCell(tmpMat.GetDeviceId());

            tmpMat.AssignElementProductOf(outGrd, tanhState);  // error to o_t
            gradientOfSigmoid.AssignSigmoidDerivativeOf(go);
            grdBeforeGo.AssignElementProductOf(tmpMat, gradientOfSigmoid);  // error before softmax
#ifdef DEBUG_DECODER
            fprintf(stderr, "output gate error = %.4e\n", grdBeforeGo(0, 0));
#endif
            Matrix<ElemType>::MultiplyAndAdd(Who, true, grdBeforeGo, false, grdToPrevOutput);  // error to previous output
            Matrix<ElemType>::MultiplyAndAdd(Wxo, true, grdBeforeGo, false, grdToObs);      // error to observation 
            tmpMat = grdBeforeGo;
            tmpMat.ColumnElementMultiplyWith(Wco);
            grdToCell = tmpMat;                                                            // error to memory cell

            Matrix<ElemType>::MultiplyAndAdd(grdBeforeGo, false, prevOutput, true, grdToWho); // gradient to Who
            Matrix<ElemType>::MultiplyAndAdd(grdBeforeGo, false, obs, true, grdToWxo); // gradient to Wxo
            tmpMat.AssignInnerProductOf(grdBeforeGo, state, false);
            grdToWco += tmpMat;                    // to Wco
            for (size_t i = 0; i < grdBeforeGo.GetNumCols(); i++)
            {
                grdTobo += grdBeforeGo.ColumnSlice(i, 1);  // gradient to bo
            }

            grdToGo.AssignElementProductOf(outGrd, go);  // error to tanh
            GradientOfTanh(tanhState, grdToGo, grdToCell, tmpMat); // error to memory cell
            grdToCell += stateError; // add error to memory cell from t+1
#ifdef DEBUG_DECODER
            fprintf(stderr, "previous state[0] = %.4e norm = %.4e\n", prevState(0, 0), prevState.FrobeniusNorm());
            fprintf(stderr, "state error = %.4e\n", grdToCell(0, 0));
            fprintf(stderr, "state error norm = %.4e\n", grdToCell.FrobeniusNorm());
#endif
            // error backpropagate to memory cells
            grdToPrevState.AssignElementProductOf(gf, grdToCell);  // error to previous memory cell
            // be careful, need to double check if errors are missing

            Matrix<ElemType> grdBeforeForget(tmpMat.GetDeviceId());
            tmpMat.AssignElementProductOf(prevState, grdToCell);  // error to f_t
            gradientOfSigmoid.AssignSigmoidDerivativeOf(gf);
            grdBeforeForget.AssignElementProductOf(gradientOfSigmoid, tmpMat); // error before forget gate
#ifdef DEBUG_DECODER
            fprintf(stderr, "forget gate error = %.4e\n", grdBeforeForget(0, 0));
#endif

            Matrix<ElemType>::MultiplyAndAdd(Whf, true, grdBeforeForget, false, grdToPrevOutput);  // error to previous output
            tmpMat = grdBeforeForget;
            tmpMat.ColumnElementMultiplyWith(Wcf);
            grdToPrevState += tmpMat;                                                            // error to previous state

            Matrix<ElemType>::MultiplyAndAdd(Wxf, true, grdBeforeForget, false, grdToObs);  // error to observation

            Matrix<ElemType>::MultiplyAndAdd(grdBeforeForget, false, prevOutput, true, grdToWhf); // gradient to Whf
            tmpMat.AssignInnerProductOf(grdBeforeForget, prevState, false);
            grdToWcf += tmpMat;                                                             // gradient to Wcf

            Matrix<ElemType>::MultiplyAndAdd(grdBeforeForget, false, obs, true, grdToWxf); // gradient to Wxf
            for (size_t i = 0; i < grdBeforeForget.GetNumCols(); i++)
                grdTobf += grdBeforeForget.ColumnSlice(i, 1);                                                    // gradient to bf

            // error backpropagate to input gate
            tmpMat.AssignElementProductOf(tanhBeforeApplyingInputGating, grdToCell);
            gradientOfSigmoid.AssignSigmoidDerivativeOf(gi);
            grdBeforeInputGate.AssignElementProductOf(gradientOfSigmoid, tmpMat); // error before input gate
#ifdef DEBUG_DECODER
            fprintf(stderr, "input gate error = %.4e\n", grdBeforeInputGate(0, 0));
#endif

            Matrix<ElemType>::MultiplyAndAdd(Whi, true, grdBeforeInputGate, false, grdToPrevOutput);  // error to previous output
            tmpMat = grdBeforeInputGate;
            tmpMat.ColumnElementMultiplyWith(Wci);
            grdToPrevState += tmpMat;                                                            // error to previous state

#ifdef DEBUG_DECODER
            fprintf(stderr, "to previous state error = %.4e\n", grdToPrevState(0, 0));
            fprintf(stderr, "to previous state error norm = %.4e\n", grdToPrevState.FrobeniusNorm());
#endif
            Matrix<ElemType>::MultiplyAndAdd(Wxi, true, grdBeforeInputGate, false, grdToObs);  // error to observation

            Matrix<ElemType>::MultiplyAndAdd(grdBeforeInputGate, false, prevOutput, true, grdToWhi); // gradient to Whi
            tmpMat.AssignInnerProductOf(grdBeforeInputGate, prevState, false);
            grdToWci += tmpMat;                                                             // gradient to Wci
            Matrix<ElemType>::MultiplyAndAdd(grdBeforeInputGate, false, obs, true, grdToWxi); // gradient to Wxi
            for (size_t i = 0; i < grdBeforeInputGate.GetNumCols(); i++)
                grdTobi += grdBeforeInputGate.ColumnSlice(i, 1);                                                  // gradient to bi

            // error backpropagate to inputs
            Matrix<ElemType> grdTmp2(tmpMat.GetDeviceId());
            Matrix<ElemType> grdBeforeTanhInputGate(tmpMat.GetDeviceId());
            grdTmp2.AssignElementProductOf(gi, grdToCell);
            grdBeforeTanhInputGate.Resize(tanhBeforeApplyingInputGating.GetNumRows(), tanhBeforeApplyingInputGating.GetNumCols());
            GradientOfTanh(tanhBeforeApplyingInputGating, grdTmp2, grdBeforeTanhInputGate, tmpMat); // error to memory cell
            Matrix<ElemType>::MultiplyAndAdd(Wxc, true, grdBeforeTanhInputGate, false, grdToObs);  // error to observation
#ifdef DEBUG_DECODER
            fprintf(stderr, "to observation error = %.4e\n", grdToObs(0, 0));
#endif

            Matrix<ElemType>::MultiplyAndAdd(Whc, true, grdBeforeTanhInputGate, false, grdToPrevOutput);  // error to previous output
            Matrix<ElemType>::MultiplyAndAdd(grdBeforeTanhInputGate, false, obs, true, grdToWxc); // gradient to Wxc

            Matrix<ElemType>::MultiplyAndAdd(grdBeforeTanhInputGate, false, prevOutput, true, grdToWhc); // gradient to Whc
            for (size_t i = 0; i < grdBeforeTanhInputGate.GetNumCols(); i++)
                grdTobc += grdBeforeTanhInputGate.ColumnSlice(i, 1);                                                    // gradient to bc

        }

        /**
        get the segmentation information, SENTENECE_BEGIN, ((int) MinibatchPackingFlags::None), ((int) MinibatchPackingFlags::NoInput) 
        for time at t and stream of streamid
        */
        int GetSegInfo(size_t t, size_t streamid)
        {
            if (streamid >= GetNumParallelSequences())
                LogicError("GetSegInfo: stream id %d is larger than the number of streams %d", (int)streamid, (int)GetNumParallelSequences());

            size_t nT = Input(0)->GetNumCols();
            if (t >= nT)
                LogicError("GetSegInfo: time %d times is larger than the total number of observations %d", (int)t, (int)nT);

            int utt_t = (int)t / GetNumParallelSequences();
            auto thisCol = m_pMBLayout->GetFrame(utt_t).first;
            thisCol.Reshape(1, GetNumParallelSequences());
            return (int) thisCol.ColumnSlice(streamid, 1).Get00Element();
        }

        /**
        save the last hidden layer activity and output
        */
        void SaveLastStateActity()
        {
            size_t nT = Input(0)->GetNumCols();
            size_t outputDim = Input(1)->GetNumRows();
            
            // save the hidden activities and output for the next minibatch
            mLastOutput.Resize(outputDim, GetNumParallelSequences());
            mLastState.Resize(outputDim, GetNumParallelSequences());

            for (size_t i = 0; i < GetNumParallelSequences(); i++)
            {
                for (int t = nT - GetNumParallelSequences() + i; t >= 0; t -= GetNumParallelSequences())
                {
                    if (GetSegInfo(t, i) == ((int) MinibatchPackingFlags::None))
                    {
                        mLastOutput.ColumnSlice(i, 1).SetValue(Value().ColumnSlice(t, 1));
                        mLastState.ColumnSlice(i, 1).SetValue(m_State.ColumnSlice(t, 1));
                        break;
                    }
                }
            }
        }

        virtual void /*ComputationNodeNonLooping::*/ForwardPropNonLooping() override
        {
            size_t nT = Input(0)->GetNumCols();
            size_t outputDim = Input(1)->GetNumRows();

            {
                SetDims(outputDim, nT);
                Value().SetValue(NAN);  // set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                m_State.Resize(outputDim, nT);
                m_State.SetValue(NAN);  // set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                m_Gi.Resize(outputDim, nT);
                m_Gi.SetValue(NAN);  // set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                m_Gf.Resize(outputDim, nT);
                m_Gf.SetValue(NAN);  // set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                m_Go.Resize(outputDim, nT);
                m_Go.SetValue(NAN);  // set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                tanhState.Resize(outputDim, nT);
                tanhState.SetValue(NAN);  // set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
                tanhObs.Resize(outputDim, nT);
                tanhObs.SetValue(NAN);  // set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 

                if (m_PastState.IsEmpty() || m_PastState.GetNumCols() != GetNumParallelSequences())
                {
                    m_PastState.Resize(outputDim, GetNumParallelSequences());
                    m_PastState.SetValue(m_DefaultState);
                }
                if (m_PastOutput.IsEmpty() || m_PastOutput.GetNumCols() != GetNumParallelSequences())
                {
                    m_PastOutput.Resize(outputDim, GetNumParallelSequences());
                }

#ifdef DEBUG_DECODER
                if (m_PastOutput.IsEmpty() == false)
                    fprintf(stderr, "LSTM node %ls past output norm = %.8e\n", this->NodeName().c_str(), m_PastOutput.FrobeniusNorm());
                if (m_PastState.IsEmpty() == false)
                    fprintf(stderr, "LSTM node %ls past state norm = %.8e\n", this->NodeName().c_str(), m_PastState.FrobeniusNorm());
#endif

                for (size_t timeIdxInSeq = 0; timeIdxInSeq < nT; timeIdxInSeq += GetNumParallelSequences())
                {
                    FrameRange fr(m_pMBLayout, timeIdxInSeq);
                    Matrix<ElemType> sliceObs = Input(0)->ValueFor(fr);
                    Matrix<ElemType> sliceOutput = ValueFor(fr);
                    Matrix<ElemType> sliceState = DataFor(m_State, fr);

                    Matrix<ElemType> sliceGi = DataFor(m_Gi, fr);
                    Matrix<ElemType> sliceGf = DataFor(m_Gf, fr);
                    Matrix<ElemType> sliceGo = DataFor(m_Go, fr);

                    Matrix<ElemType> sliceTanhState = DataFor(tanhState, fr);
                    Matrix<ElemType> sliceTanhInput = DataFor(tanhObs, fr);

                    PrepareHistory(timeIdxInSeq, mSlicePrevOutput, mSlicePrevState, Value(), m_State, m_PastOutput, m_PastState, GetNumParallelSequences(), m_DefaultState, &m_pMBLayout->GetM());

                    ForwardPropS(Input(1)->Value(), Input(2)->Value(), Input(3)->Value(), Input(4)->Value(),
                            sliceObs, mSlicePrevOutput, mSlicePrevState, sliceOutput, sliceState, sliceGi, sliceGf, sliceGo, sliceTanhState, sliceTanhInput, m_tempMatrix);
                }

                // save the hidden activities and output for the next minibatch
                SaveLastStateActity();

#ifdef DEBUG_DECODER
                if (mLastOutput.IsEmpty() == false)
                    fprintf(stderr, "LSTM node %ls last output norm = %.8e\n", this->NodeName().c_str(), mLastOutput.FrobeniusNorm());
                if (mLastState.IsEmpty() == false)
                    fprintf(stderr, "LSTM node %ls last state norm = %.8e\n", this->NodeName().c_str(), mLastState.FrobeniusNorm());
#endif

#ifdef DEBUG_DECODER
                ElemType tmpnorm = Value().FrobeniusNorm();
                if (ISCLOSE(tmpnorm, 0.834251, 0.002))
                    fprintf(stderr, "check!");
                fprintf(stderr, "LSTM function norm = %.8e\n", tmpnorm);
                for (size_t i = 0; i < 5; i++)
                    fprintf(stderr, "LSTM input[%d] norm = %.8e ", i, Input(i)->Value().FrobeniusNorm());
                fprintf(stderr, "\n");
#endif

                m_GradientComputed = false;
            }
        }

        /**
        Prepare history for LSTMnode

        This function returns state and output from the previous time instance. For recurrent network, the initial state needs to be set in the case of sentence begining, which is carried over from sentenceBegin. In case of sentence begining, the state activity is set to an initial value. The sentenceBegin has element of ((int) MinibatchPackingFlags::SequenceStart), ((int) MinibatchPackingFlags::None) and ((int) MinibatchPackingFlags::NoInput), which are 0, 1, and -1, respectively. 
        To compute the initial value, we use
        prevState = sentenceBegin * delayedActivation + ~sentenceBegin * initialStateValue
        and ~sentenceBegin is computed as -1*(sentenceBegin - 1), assuming that sentenceBegin is either 0 or 1. For example, when sentenceBegin == 1, ~sentenceBegin == 0. 
        The previous-time output doesn't have initial value, so it is computed as 
        prevOutput = sentenceBegin * pastOutput

        */
        // prepare prevstate and prevoutput
        static void WINAPI PrepareHistory(
            size_t timeIdxInSeq,
            Matrix<ElemType> & slicePrevOutput,
            Matrix<ElemType> & slicePrevState,
            const Matrix<ElemType> & output,
            const Matrix<ElemType> & state,
            const Matrix<ElemType> & pastOutput,
            const Matrix<ElemType> & pastState,
            size_t nsamples, const ElemType & initStateValue, const Matrix<float>* sentenceBegin)
        {
            size_t nRow = pastOutput.GetNumRows();
            size_t nStream = sentenceBegin->GetNumRows();

            assert(nStream == nsamples);

            int utt_t = (int)floor(timeIdxInSeq / nsamples);
            if (slicePrevOutput.IsEmpty() || slicePrevOutput.GetNumRows() != nRow || slicePrevOutput.GetNumCols() != nsamples)
                slicePrevOutput.Resize(nRow, nsamples);
            if (slicePrevState.IsEmpty() || slicePrevState.GetNumRows() != nRow || slicePrevState.GetNumCols() != nsamples)
                slicePrevState.Resize(nRow, nsamples);

            if (sentenceBegin->GetNumRows() != nsamples)
                LogicError("Number of rows should be the same as the number of data streams");

            Matrix<float> colBegin(sentenceBegin->GetDeviceId());
            colBegin.SetValue(sentenceBegin->ColumnSlice(utt_t, 1));
            Matrix<ElemType> colSeg(colBegin.GetDeviceId());
            colSeg.Resize(nStream, nStream);
            // will reset to 0 if sentence begining at a position is 0
            // will keep the output if it is not the sentence begining
            colBegin.InplaceTruncateBottom(((int) MinibatchPackingFlags::SequenceStart));
            colBegin.InplaceTruncateTop(((int) MinibatchPackingFlags::None));
#if 1
            initStateValue; pastState; pastOutput; state; output;
            LogicError("PrepareHistory: finish this");
#else
            // BUGBUG: we need to upcast float to double here
            colSeg.SetDiagonalValue(colBegin);

            Matrix<ElemType> newPrevOutput(colBegin.GetDeviceId());
            Matrix<ElemType> newPrevState(colBegin.GetDeviceId());
            if (utt_t == 0)
            {
                // this is the begining of this minibatch
                Matrix<ElemType>::Multiply(pastOutput.ColumnSlice(0, nsamples), false, colSeg, false, newPrevOutput);
                Matrix<ElemType>::Multiply(pastState.ColumnSlice(0, nsamples), false, colSeg, false, newPrevState);
            }
            else
            {
                // this is in the minibatch
                FrameRange fr(timeIdxInSeq, nsamples);
                Matrix<ElemType>::Multiply(DataFor(output, fr/*TODO: delete the next two parameters*/, fr.t() - nsamples, nsamples), false, colSeg, false, newPrevOutput);
                Matrix<ElemType>::Multiply(DataFor(state, fr/*TODO: delete the next two parameters*/, fr.t() - nsamples, nsamples), false, colSeg, false, newPrevState);
            }

            Base::SetToInitStateValueForResetSeg(sentenceBegin->ColumnSlice(utt_t, 1), nStream, initStateValue, newPrevState);

            slicePrevOutput.ColumnSlice(0, nsamples).SetValue(newPrevOutput);
            slicePrevState.ColumnSlice(0, nsamples).SetValue(newPrevState);
#endif
        }

        // prepare prevstate and prevoutput
        void PrepareThisErrorsBeforeBackProp(
            size_t timeIdxInSeq,
            size_t nT, // number of columns
            Matrix<ElemType> & error,
            Matrix<ElemType> & stateError,
            const Matrix<ElemType>& grdToPrevOutput,
            const Matrix<ElemType>& grdToPrevState,
            const Matrix<ElemType>& obs_error_from_future_minibatch,
            const Matrix<ElemType>& state_error_from_future_minibatch,
            size_t nsamples, const Matrix<float>* sentenceBegin)
        {
            int utt_t = (int)floor(timeIdxInSeq / nsamples);
            int total_utt_t = (int)floor(nT / nsamples);

            error += grdToPrevOutput;
            stateError = grdToPrevState;

            if (m_use_errors_from_future_minibatch)
            {
                for (size_t utt_id = 0; utt_id < nsamples; utt_id++)
                {
                    // if uses errors from future minibatch
                    if ((GetSegInfo(timeIdxInSeq, utt_id) == ((int) MinibatchPackingFlags::None) && utt_t == total_utt_t - 1) // last time 
                        || (utt_t < total_utt_t - 1 && GetSegInfo(timeIdxInSeq, utt_id) == ((int) MinibatchPackingFlags::None) && GetSegInfo(timeIdxInSeq + nsamples, utt_id) == ((int) MinibatchPackingFlags::NoInput)) // future observation is no observation
                        )
                    {
                        error.ColumnSlice(utt_id, 1) += obs_error_from_future_minibatch.ColumnSlice(utt_id, 1);
                        stateError.ColumnSlice(utt_id, 1) += state_error_from_future_minibatch.ColumnSlice(utt_id, 1);
                    }
                }
            }


#if 1
            sentenceBegin;
            LogicError("PrepareThisErrorsBeforeBackProp: finish this");
#else
            Matrix<ElemType> colBegin(sentenceBegin->GetDeviceId());
            colBegin.SetValue(sentenceBegin->ColumnSlice(utt_t, 1));
            colBegin.InplaceTruncateBottom(((int) MinibatchPackingFlags::NoInput));
            colBegin.InplaceTruncateTop(((int) MinibatchPackingFlags::SequenceStart));
            colBegin += fabs((ElemType)((int) MinibatchPackingFlags::NoInput)); // raise this so that -1 -> 0 and therefore 
            Matrix<ElemType> colSeg(colBegin.GetDeviceId());
            colSeg.Resize(nsamples, nsamples);
            colSeg.SetDiagonalValue(colBegin);

            // times the errors with the mask
            Matrix<ElemType> newOutputError(colBegin.GetDeviceId());
            Matrix<ElemType> newStateError(colBegin.GetDeviceId());

            Matrix<ElemType>::Multiply(error, false, colSeg, false, newOutputError);
            Matrix<ElemType>::Multiply(stateError, false, colSeg, false, newStateError);
            
            error.ColumnSlice(0, nsamples).SetValue(newOutputError);
            stateError.ColumnSlice(0, nsamples).SetValue(newStateError);
#endif
        }

        // prepare prevstate and prevoutput
        static void WINAPI PrepareErrors(
            size_t timeIdxInSeq,
            Matrix<ElemType> & errors,
            Matrix<ElemType> & stateError,
            size_t nsamples, const Matrix<float>* sentenceBegin)
        {
            int utt_t = (int)floor(timeIdxInSeq / nsamples);
            Matrix<ElemType> colBegin(sentenceBegin->GetDeviceId());
#if 1
            errors; stateError; utt_t;
            LogicError("PrepareErrors: finish this");
#else
            colBegin.SetValue(sentenceBegin->ColumnSlice(utt_t, 1));
            // will reset to 0 if sentence begining at a posiiton is 0
            // will keep the output if it is not the sentence begining
            colBegin.InplaceTruncateBottom(((int) MinibatchPackingFlags::SequenceStart));
            colBegin.InplaceTruncateTop(((int) MinibatchPackingFlags::None));

            Matrix<ElemType> colSeg(colBegin.GetDeviceId());
            colSeg.Resize(nsamples, nsamples);
            colSeg.SetDiagonalValue(colBegin);

            // times the errors with the mask
            Matrix<ElemType> newOutputError(colBegin.GetDeviceId());
            Matrix<ElemType> newStateError(colBegin.GetDeviceId());

            Matrix<ElemType>::Multiply(errors, false, colSeg, false, newOutputError);
            Matrix<ElemType>::Multiply(stateError, false, colSeg, false, newStateError);

            errors.ColumnSlice(0, nsamples).SetValue(newOutputError);
            stateError.ColumnSlice(0, nsamples).SetValue(newStateError);
#endif
        }

        /*TODO: merge with call site*/void ForwardPropS(
            const Matrix<ElemType>& mInputGate,
            const Matrix<ElemType> &mForgetGate, const Matrix<ElemType> &mOutputGate,
            const Matrix<ElemType> &mCellWgt,
            const Matrix<ElemType> &obs,
            const Matrix<ElemType>& prevOutput,
            const Matrix<ElemType>& prevState,
            Matrix<ElemType> &output,
            Matrix<ElemType> &state,
            Matrix<ElemType> &gi,
            Matrix<ElemType> &gf,
            Matrix<ElemType> &go,
            Matrix<ElemType> &tanhState,
            Matrix<ElemType> &tanhObs,
            Matrix<ElemType> &tmp)
        {
            int inputDim = obs.GetNumRows();
            int outputDim = mOutputGate.GetNumRows();

            // for input gate
            Matrix<ElemType>::Multiply(mInputGate.ColumnSlice(1, inputDim), false, obs, false, gi);
            Matrix<ElemType>::MultiplyAndAdd(mInputGate.ColumnSlice(1 + inputDim, outputDim), false, prevOutput, false, gi);
            gi += mInputGate.ColumnSlice(0, 1);
            tmp = prevState;
            tmp.ColumnElementMultiplyWith(mInputGate.ColumnSlice(1 + inputDim + outputDim, 1));
            gi += tmp;
            gi.AssignSigmoidOf(gi);

            // for forget gate
            Matrix<ElemType>::Multiply(mForgetGate.ColumnSlice(1, inputDim), false, obs, false, gf);
            Matrix<ElemType>::MultiplyAndAdd(mForgetGate.ColumnSlice(1 + inputDim, outputDim), false, prevOutput, false, gf);
            gf += mForgetGate.ColumnSlice(0, 1);
            tmp = prevState;
            tmp.ColumnElementMultiplyWith(mForgetGate.ColumnSlice(1 + inputDim + outputDim, 1));
            gf += tmp;
            gf.AssignSigmoidOf(gf);

            // for cell state
            Matrix<ElemType>::Multiply(mCellWgt.ColumnSlice(1, inputDim), false, obs, false, state);
            Matrix<ElemType>::MultiplyAndAdd(mCellWgt.ColumnSlice(1 + inputDim, outputDim), false, prevOutput, false, state);
            state += mCellWgt.ColumnSlice(0, 1);
#ifdef DEBUG_DECODER
//            fprintf(stderr, "W_xc norm = %.8e\n", mCellWgt.ColumnSlice(1, inputDim).FrobeniusNorm());
//            fprintf(stderr, "W_hc norm = %.8e\n", mCellWgt.ColumnSlice(1 + inputDim, outputDim).FrobeniusNorm());
//            fprintf(stderr, "b_c norm = %.8e\n", mCellWgt.ColumnSlice(0, 1).FrobeniusNorm());
#endif
            tanhObs.AssignTanhOf(state);
            state.AssignElementProductOf(gi, tanhObs);
            state.AddElementProductOf(gf, prevState);

            // for output gate
            Matrix<ElemType>::Multiply(mOutputGate.ColumnSlice(1, inputDim), false, obs, false, go);
            Matrix<ElemType>::MultiplyAndAdd(mOutputGate.ColumnSlice(1 + inputDim, outputDim), false, prevOutput, false, go);
            go += mOutputGate.ColumnSlice(0, 1);
            tmp = state;
            tmp.ColumnElementMultiplyWith(mOutputGate.ColumnSlice(1 + inputDim + outputDim, 1));
            go += tmp;
            go.AssignSigmoidOf(go);

            // to return output
            tanhState.AssignTanhOf(state);
            output.AssignElementProductOf(go, tanhState);
        }


        // input(0) : child with dimension [inputdim x T]
        // input(1) : input gate [outputdim x [inputdim + outputdim + 2]] bi, Wxi, Whi, Wci
        // input(2) : forget gate [outputdim x [inputdim + outputdim + 2]] for bf, Wxf, Whf, Wcf
        // input(3) : output gate [outputdim x [inputdim + outputdim + 2]] for bo, Wxo, Who, and Wco
        // input(4) : memory cell weight [outputdim x [inputdim + outputdim + 1]] for bc, Wxc, and Whc 
        // output : dimension [outputdim x T]
        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            InferMBLayoutFromInputsForStandardCase();
            InferImageDimsFromInputs();

            if (Input(0)->Value().GetMatrixType() == SPARSE)
                LogicError("LSTMNode: input to LSTM has to be dense matrix. Consider adding a project layer using lookuptable before LSTM node. ");

#if 0
            // TODO: use dynamic_pointer_cast instead
            if (Input(1)->OperationName() != OperationNameOf(LearnableParameter) ||
                Input(2)->OperationName() != OperationNameOf(LearnableParameter) ||
                Input(3)->OperationName() != OperationNameOf(LearnableParameter) ||
                Input(4)->OperationName() != OperationNameOf(LearnableParameter))
                LogicError("LSTM validation: need to have learnable parameters ");
#endif

            //if (Input(0)->GetNumRows() == 0)
            //    LogicError("LSTM validation: input size is zero!");

            //if (Input(1)->GetNumRows() == 0 ||
            //    Input(2)->GetNumRows() == 0 ||
            //    Input(3)->GetNumRows() == 0 ||
            //    Input(4)->GetNumRows() == 0)
            //    LogicError("LSTM validation : parameter size is zero!");

            size_t nindim = Input(0)->GetNumRows();
            size_t noutdim = Input(1)->GetNumRows();
            size_t nT = Input(0)->GetNumCols();
            size_t nCol = nindim + noutdim + 2;
            if (isFinalValidationPass)
            {
                if (Input(1)->GetNumCols() != nCol)
                {
                    LogicError("LSTM validation : dimension mismatched between child and inputGate");
                }
                if (Input(2)->GetNumCols() != nCol)
                {
                    LogicError("LSTM validation : dimension mismatched between child and forgetGate");
                }
                if (Input(3)->GetNumCols() != nCol)
                {
                    LogicError("LSTM validation : dimension mismatched between child and outputGate");
                }

                if (noutdim != Input(2)->GetNumRows() ||
                    noutdim != Input(3)->GetNumRows() ||
                    noutdim != Input(4)->GetNumRows())
                {
                    LogicError("LSTM validation: output dimension mismatched!");
                }
            }

            SetDims(noutdim, nT);
            Value().SetValue(NAN);  // set to this extrem value so, if anything wrong in later procedure, problems can be easily spotted. 
        }

        bool UnitTest()
        {
            {
                size_t nT = 3;
                size_t nInput = 2;
                size_t nHidden = 3;
                size_t nOutput = 3;

                // backup 
                Matrix<ElemType> f0(m_deviceId), f1(m_deviceId), f2(m_deviceId), f3(m_deviceId), f4(m_deviceId), func(m_deviceId), f5(m_deviceId);
                Matrix<ElemType> target(m_deviceId);
                Matrix<ElemType> giWeight, ghWeight, goWeight;
                ElemType initStateValue = m_DefaultState;
                auto pMBLayout = make_shared<MBLayout>();
                pMBLayout->Init(1, nT);
                //Matrix<float> & boundary = pMBLayout->m_sentenceBoundaryFlags;
                //vector<MinibatchPackingFlags> & minibatchPackingFlags = pMBLayout->m_minibatchPackingFlags;
                //boundary.ColumnSlice(0, 1).SetValue(((int) MinibatchPackingFlags::SequenceStart));
                //minibatchPackingFlags[1] = MinibatchPackingFlags::SequenceStart;
                pMBLayout->AddSequence(MAKE_SEQUENCE_ID, 0, 0, nT);
                Base::LinkToMBLayout(pMBLayout);

                f0 = Input(0)->Value();
                f1 = Input(1)->Value();
                f2 = Input(2)->Value();
                f3 = Input(3)->Value();
                f4 = Input(4)->Value();
                func = Value();

                target.Resize(nOutput, nT);
                for (size_t i = 0; i < nT; i++)
                    target(0, i) = 1;

                Input(0)->SetDims(nInput, nT);
                Input(0)->Value().SetValue(ConstOnes(nInput, nT, m_deviceId));
                Input(0)->Value().SetValue((ElemType)0.1);
                Input(1)->SetDims(nHidden, nInput + nOutput + 2);
                Input(1)->Value().SetValue((ElemType)0.1);
                Input(2)->SetDims(nHidden, nInput + nHidden + 2);
                Input(2)->Value().SetValue((ElemType)0.1);
                Input(3)->SetDims(nOutput, nInput + nHidden + 2);
                Input(3)->Value().SetValue((ElemType)0.1);
                Input(4)->SetDims(nOutput, nHidden + nInput + 1);
                Input(4)->Value().SetValue((ElemType)0.1);
                SetDims(nOutput, nT);

                m_DefaultState = 0.0;
                ForwardProp(FrameRange(m_pMBLayout));

                // check with expected values
                if (!ISCLOSE(Value()(0, 0), 0.0335975, EPSILON) ||
                    !ISCLOSE(Value()(0, 1), 0.05485132, EPSILON) ||
                    !ISCLOSE(Value()(0, 2), 0.06838435, EPSILON) ||
                    !(Value()(0, 0) == Value()(1, 0)))
                    throw("LSTMNode forward computation error");

                
                    Value().TransferToDeviceIfNotThere( m_deviceId, true);

                Gradient().Resize(nOutput, nT);
                Gradient().SetValue(1.0);
                for (size_t i = 0; i < 5; i++)
                {
                    Input(i)->Gradient().Resize(Input(i)->GetNumRows(), Input(i)->GetNumCols());
                    Input(i)->Gradient().SetValue(0);
                }
                for (size_t i = 0; i < 5; i++)
                    BackpropTo(i, FrameRange(m_pMBLayout));

                // check with expected values
                if (!ISCLOSE(Input(1)->Gradient()(0, 0), 0.07843818, EPSILON) // bi
                    || !ISCLOSE(Input(1)->Gradient()(0, 1), 0.00784382, EPSILON)  // Wxi
                    || !ISCLOSE(Input(1)->Gradient()(0, 3), 0.00192997, EPSILON)  // Whi
                    || !ISCLOSE(Input(1)->Gradient()(0, 6), 0.00362767, EPSILON)  // Wci
                    )
                    throw("LSTMNode gradient error on input gates");
                if (!ISCLOSE(Input(2)->Gradient()(0, 0), 0.02738655, EPSILON)  // bf
                    || !ISCLOSE(Input(2)->Gradient()(0, 1), 0.00273866, EPSILON)  // Wxf
                    || !ISCLOSE(Input(2)->Gradient()(0, 3), 0.00120922, EPSILON)  // Whf
                    || !ISCLOSE(Input(2)->Gradient()(0, 6), 0.00227184, EPSILON)  // Wcf
                    )
                    throw("LSTMNode gradient error on forget gates");
                if (!ISCLOSE(Input(3)->Gradient()(0, 0), 0.07801557, EPSILON)  // bo
                    || !ISCLOSE(Input(3)->Gradient()(0, 1), 0.00780156, EPSILON)  // Wxo
                    || !ISCLOSE(Input(3)->Gradient()(0, 3), 0.00268089, EPSILON)  // Who
                    || !ISCLOSE(Input(3)->Gradient()(0, 6), 0.00809852, EPSILON)  // Wco
                    )
                    throw("LSTMNode gradient error on output gates");
                if (!ISCLOSE(Input(4)->Gradient()(0, 0), 1.3075038, EPSILON)  // bc
                    || !ISCLOSE(Input(4)->Gradient()(0, 1), 0.13075038, EPSILON)  // Wxc
                    || !ISCLOSE(Input(4)->Gradient()(0, 3), 0.03080355, EPSILON)  // Whc
                    )
                    throw("LSTMNode gradient error on memory cells");

                for (size_t i = 0; i < 5; i++)
                {
                    
                        Input(i)->Gradient().TransferToDeviceIfNotThere( m_deviceId, true);
                }
                m_DefaultState = initStateValue;
            }

            fprintf(stderr, "LSTMNode unit test passed!\n");
            return true;
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(1, false);
        }

        virtual void DumpNodeInfo(const bool printValues, File& fstream) const override
        {
            Base::DumpNodeInfo(printValues, fstream);
            fstream << L"Input[Width:" << m_inputDim << L"]  \n" ; 
            fstream << L"Hidden[Width:" << m_outputDim << L"]    Output[Width:" << m_outputDim << L"]  \n";
        }
    public:
        bool GetHistory(Matrix<ElemType>& hist, bool bLastTime)
        {
            size_t tRow = m_PastOutput.GetNumRows();
            size_t tCol = m_PastOutput.GetNumCols();
            size_t rCol = m_PastState.GetNumCols();

            DEVICEID_TYPE device = hist.GetDeviceId();
            hist.TransferFromDeviceToDevice(device, m_deviceId, true);
            hist.Resize(tRow, tCol + rCol);

            if (bLastTime)
            {
                hist.ColumnSlice(0, tCol).SetValue(mLastOutput);
                hist.ColumnSlice(tCol, rCol).SetValue(mLastState);
            }
            else{
                hist.ColumnSlice(0, tCol).SetValue(m_PastOutput);
                hist.ColumnSlice(tCol, rCol).SetValue(m_PastState);
            }

            hist.TransferFromDeviceToDevice(m_deviceId, device, true);
            return true;
        }

        void SetHistory(const Matrix<ElemType>& hist)
        {
            size_t tRow = hist.GetNumRows();
            size_t tCol = hist.GetNumCols();
            size_t eCols = tCol / 2;

            DEVICEID_TYPE device = hist.GetDeviceId();
            hist.TransferFromDeviceToDevice(device, m_deviceId, true);

            m_PastOutput.Resize(tRow, eCols);
            m_PastState.Resize(tRow, eCols);
            m_PastOutput.SetValue(hist.ColumnSlice(0, eCols));
            m_PastState.SetValue(hist.ColumnSlice(eCols, eCols));

            hist.TransferFromDeviceToDevice(m_deviceId, device, true);
        }

        virtual void GetErrorsToPreviousMinibatch(Matrix<ElemType>& hist)
        {
            size_t tRow = m_obs_error_from_future_minibatch.GetNumRows();
            size_t tCol = m_obs_error_from_future_minibatch.GetNumCols();
            size_t rCol = m_state_error_from_future_minibatch.GetNumCols();

            DEVICEID_TYPE device = hist.GetDeviceId();

            hist.TransferFromDeviceToDevice(device, m_deviceId, true);
            hist.Resize(tRow, tCol + rCol);

            hist.ColumnSlice(0, tCol).SetValue(m_obs_error_from_future_minibatch);
            hist.ColumnSlice(tCol, rCol).SetValue(m_state_error_from_future_minibatch);

            hist.TransferFromDeviceToDevice(m_deviceId, device, true);
        }

        virtual void SetErrorsFromFutureMinibatch(Matrix<ElemType>& hist)
        {
            size_t tCol = hist.GetNumCols();
            size_t rCol = tCol / 2;

            DEVICEID_TYPE device = hist.GetDeviceId();

            hist.TransferFromDeviceToDevice(device, m_deviceId, true);

            m_obs_error_from_future_minibatch.SetValue(hist.ColumnSlice(0, rCol));
            m_state_error_from_future_minibatch.SetValue(hist.ColumnSlice(rCol, rCol));

            m_use_errors_from_future_minibatch = true;

            hist.TransferFromDeviceToDevice(m_deviceId, device, true);
        }

    protected:
        size_t m_inputDim;
        size_t m_outputDim;

        Matrix<ElemType> m_State;  // hidden state activity
        Matrix<ElemType> m_PastState; // state activity in the previous minibatch
        Matrix<ElemType> m_PastOutput; // output in the previou minibatch 

        Matrix<ElemType> mLastState; // last state activity 
        Matrix<ElemType> mLastOutput; // last output 

        Matrix<ElemType> m_Gi;     // input gate activity
        Matrix<ElemType> m_Gf;     // forget gate activity
        Matrix<ElemType> m_Go;     // output gate activity

        Matrix<ElemType> grdToObs, grdToInputGate, grdToForgetGate, grdToOutputGate, grdToCellWgt;
        Matrix<ElemType> tanhState, tanhObs;

        Matrix<ElemType> m_tempMatrix; // temp matrix for speed-up

        bool     m_GradientComputed; // true if LSTM node has computed gradients, set to false if forward computation is just finished 

        Matrix<ElemType> mSlicePrevOutput, mSlicePrevState;

        Matrix<ElemType> grdBeforeInputGate, grdBeforeForget, grdBeforeGo, grdToCell, grdBeforeTanhInputGate;

    public:
        // errors from future minibatch
        Matrix<ElemType> m_obs_error_from_future_minibatch;
        Matrix<ElemType> m_state_error_from_future_minibatch;
        bool m_use_errors_from_future_minibatch;

        ElemType m_DefaultState;

    };

    template class LSTMNode<float>;
    template class LSTMNode<double>;

}}}
