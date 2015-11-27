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

        virtual void ComputeInputPartialNonLooping(size_t inputIndex) override
        {
            FrameRange frameRange(Inputs(0)->GetMBLayout());
            if (inputIndex == 0)
                LogicError("DummyCriterionNode: derivatives with respect to objective features are not necessary, not implemented yet.\n");
            else if (inputIndex == 1)
                LogicError("DummyCriterionNode: derivatives with respect to derivative features are not necessary, not implemented yet.\n");
            else if (inputIndex == 2)
            {
                auto gradient = Inputs(2)->GradientSlice(frameRange);
                //Matrix<ElemType>::ScaleAndAdd(GradientValues().Get00Element(), Inputs(1)->ValueSlice(frameRange), gradient);
                Matrix<ElemType>::Multiply1x1AndWeightedAdd(+1.0f, GradientValues()/*1x1*/, Inputs(1)->ValueSlice(frameRange), 1.0f, gradient);
            }
        }

        virtual void /*ComputationNodeNonLooping::*/EvaluateThisNodeNonLooping() override
        {
            FunctionValues().VerifySize(1, 1);
            Inputs(0)->FunctionValues().VerifySize(1, 1);
            FunctionValues().SetValue(Inputs(0)->FunctionValues());
#if NANCHECK
            FunctionValues().HasNan("DummyCriterionNode");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            Base::Validate(isFinalValidationPass);

            if (Inputs(0)->OperationName() != L"InputValue")
                LogicError("DummyCriterionNode criterion requires the first input to be computed objectives.");
            if (Inputs(0)->OperationName() != L"InputValue")
                LogicError("DummyCriterionNode criterion requires the first input to be computed derivatives.");
            if (isFinalValidationPass)
            {
                if (Inputs(0)->GetNumRows() != 1)
                LogicError("DummyCriterionNode criterion requires the first input to have dimension 1.");
                if (Inputs(0)->GetNumRows() == 0 || Inputs(1)->GetNumRows() == 0 || Inputs(2)->GetNumRows() == 0)
                    LogicError("DummyCriterionNode operation: one of the operands has 0 elements.");
                if (Inputs(1)->GetNumRows() != Inputs(2)->GetNumRows())
                LogicError("The Matrix dimension in the DummyCriterionNode operation does not match.");
            }
            // TODO: What is this about?
            //if (Inputs(1)->GetNumCols() != Inputs(2)->GetNumCols())
            //    ValidateInferChildDims(1, Inputs(1)->GetNumRows(), Inputs(2)->GetNumCols()); 

            SetDims(1,1);
            m_pMBLayout = nullptr;    // this node does not hold mini-batch data
            InferImageDimsFromInputs(); 
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            m_imageLayout = ImageLayout();
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

        virtual void ComputeInputPartialNonLooping(size_t /*inputIndex*/) override  //scaled by 2*number of elements in the Matrix<ElemType>
        {
            LogicError("SequenceDecoder is used for evaluation only.");
        }

        /// compute posterior probability of label y at position t
        virtual void /*ComputationNodeNonLooping::*/EvaluateThisNodeNonLooping() override
        {
            DecideStartEndingOutputLab(Inputs(0)->FunctionValues(), mStartLab, mEndLab);
            EvaluateThisNodeS(mAlpha, mBacktrace, FunctionValues(), Inputs(1)->FunctionValues(),
                              Inputs(2)->FunctionValues(), mStartLab, mEndLab);
        }

        // compute forward backward algorithm
        void EvaluateThisNodeS(Matrix<ElemType>& alpha, Matrix<ElemType>& backtrace, Matrix<ElemType>& functionValues, const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, const size_t stt, const size_t stp)
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
                if (!(Inputs(1)->GetNumRows() == Inputs(2)->GetNumRows() &&  // position dependent and pair scores have same number of labels
                    Inputs(0)->GetNumRows() == Inputs(1)->GetNumRows() &&
                    Inputs(0)->GetNumCols() == Inputs(1)->GetNumCols() && // position dependent and pair scores have the same observation numbers
                    Inputs(2)->GetNumCols() == Inputs(2)->GetNumRows()))
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

            m_imageLayout = ImageLayout();
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

        virtual void /*ComputationNode::*/ComputeInputPartial(const size_t inputIndex, const FrameRange & frameRange) override
        {
            if (frameRange.IsAllFrames()) { NOT_IMPLEMENTED; return; } // TODO: remove these one by one. And why is this not implemented?
            if (inputIndex > 2)
                InvalidArgument("StrideTimes operation only takes three inputs.");
            else if (inputIndex == 2)
                return;     // that's a constant

            Matrix<ElemType> sliceOutputGrad = GradientSlice(frameRange);

            if (m_strideDim == 1) // column stride
            {
                if (inputIndex == 0)  //left derivative
                {
                    Matrix<ElemType> sliceInput1Value = Inputs(1)->ValueSlice(frameRange);

                    //ComputeInputPartialLeft1(sliceInput1Value, Inputs(0)->GradientValues(), sliceOutputGrad);

                    size_t r = Inputs(0)->GetNumRows();
                    size_t T1 = Inputs(0)->GetNumCols() / GetNumParallelSequences();    // TODO: if T1 == GetNumTimeSteps() then we can simplify code below.
                    Matrix<ElemType> mTmp1(r, T1, sliceInput1Value.GetDeviceId());

                    // process sequence by sequence
                    for (size_t k = 0; k < GetNumParallelSequences(); k++)
                    {
                        mTmp1.SetValue(0);
                        auto mTmp2 = sliceInput1Value.ColumnSlice(k, 1);
                        auto mTmp3 = sliceOutputGrad.ColumnSlice(k, 1);

                        ComputeInputPartialLeft1(mTmp2, mTmp1, mTmp3);

                        for (size_t t = 0; t < T1; t++)
                        {
                            Inputs(0)->GradientValues().ColumnSlice(t*GetNumParallelSequences() + k, 1) += mTmp1.ColumnSlice(t, 1);
                        }
                    }
                }
                else  //right derivative
                {
                    Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientSlice(frameRange);

                    //ComputeInputPartialRight(Inputs(0)->FunctionValues(), sliceInput1Grad, sliceOutputGrad);

                    // process sequence by sequence
                    for (size_t k = 0; k < GetNumParallelSequences(); k++)
                    {
                        size_t r = Inputs(0)->GetNumRows();
                        size_t T1 = Inputs(0)->GetNumCols() / GetNumParallelSequences();    // TODO: if T1 == GetNumTimeSteps() then we can simplify code below.
                        Matrix<ElemType> mTmp1(r, T1, sliceOutputGrad.GetDeviceId());
                        for (size_t t = 0; t < T1; t++)
                        {
                            mTmp1.ColumnSlice(t, 1).SetValue(Inputs(0)->FunctionValues().ColumnSlice(t*GetNumParallelSequences() + k, 1));
                        }
                        auto mTmp2 = sliceInput1Grad.ColumnSlice(k, 1);
                        auto mTmp3 = sliceOutputGrad.ColumnSlice(k, 1);

                        ComputeInputPartialRight(mTmp1, mTmp2, mTmp3);
                    }
                }
            }
            else if (m_strideDim == 0) // row stride
            {
                if (inputIndex == 0)  //left derivative
                {
                    Matrix<ElemType> sliceInput1Value = Inputs(1)->ValueSlice(frameRange);

                    for (size_t k = 0; k < GetNumParallelSequences(); k++)
                    {
                        size_t d = Inputs(1)->GetNumRows();
                        size_t T1 = Inputs(0)->GetNumRows() / GetNumParallelSequences();
                        Matrix<ElemType> mTmp1(sliceInput1Value.GetDeviceId());
                        mTmp1.Resize(d, T1);
                        Matrix<ElemType> mTmp2 = sliceInput1Value.ColumnSlice(k, 1);
                        Matrix<ElemType> mTmp3 = sliceOutputGrad.ColumnSlice(k, 1);
                        ComputeInputPartialLeft(mTmp2, mTmp1, mTmp3);

                        Matrix<ElemType> mTmp4(sliceInput1Value.GetDeviceId());
                        for (size_t t = 0; t < T1; t++)
                        {
                            mTmp4 = mTmp1.ColumnSlice(t, 1);
                            mTmp4.Reshape(1, d);
                            Inputs(0)->GradientValues().AddToRowSliceValuesOf(mTmp4, t*GetNumParallelSequences() + k, 1);
                        }
                    }
                }
                else  //right derivative
                {
                    Matrix<ElemType> sliceInput1Grad = Inputs(1)->GradientSlice(frameRange);

                    for (size_t k = 0; k < GetNumParallelSequences(); k++)
                    {
                        size_t d = Inputs(1)->GetNumRows();
                        size_t T1 = Inputs(0)->GetNumRows() / GetNumParallelSequences();

                        Matrix<ElemType> mTmp0(sliceOutputGrad.GetDeviceId());
                        mTmp0.Resize(1, d);

                        Matrix<ElemType> mTmp1(sliceOutputGrad.GetDeviceId());
                        mTmp1.Resize(T1, d);
                        for (size_t t = 0; t < T1; t++)
                        {
                            mTmp0.SetValue(0);
                            mTmp0.AddWithRowSliceValuesOf(Inputs(0)->FunctionValues(), t * GetNumParallelSequences() + k, 1);
                            mTmp1.AssignToRowSliceValuesOf(mTmp0, t, 1);
                        }
                        Matrix<ElemType> mTmp2 = sliceInput1Grad.ColumnSlice(k, 1);
                        Matrix<ElemType> mTmp3 = sliceOutputGrad.ColumnSlice(k, 1);

                        ComputeInputPartialRight(mTmp1, mTmp2, mTmp3);
                    }
                }
            }
        }

        // TODO: the following two functions only differ in the order of argument use in the final MultiplyAndAdd()  --is that intended??
        static /*TODO: merge with call site*/void ComputeInputPartialLeft1(const Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
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

        static /*TODO: merge with call site*/void ComputeInputPartialLeft(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
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

        static /*TODO: merge with call site*/void ComputeInputPartialRight(Matrix<ElemType>& inputFunctionValues, Matrix<ElemType>& inputGradientValues, const Matrix<ElemType>& gradientValues)
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

        virtual void /*ComputationNode::*/EvaluateThisNode(const FrameRange & frameRange) override
        {
            size_t rows0 = Inputs(0)->GetNumRows(), cols1 = Inputs(1)->GetNumCols();
            Matrix<ElemType> sliceInput1Value = Inputs(1)->ValueSlice(frameRange);
            UpdateStride(sliceInput1Value);

            if (m_strideDim == 0)
                SetDims(rows0 / GetNumParallelSequences(), cols1);
            if (m_strideDim == 1)       // TODO: no else??
                SetDims(rows0, cols1);

            Matrix<ElemType> sliceOutputValue = ValueSlice(frameRange);

            // (TODO: these following assignments are leftovers of refactoring and can be short-circuited)
            Matrix<ElemType>& functionValues = sliceOutputValue;
            const Matrix<ElemType>& input0 = Inputs(0)->FunctionValues();
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

            if (Inputs(2)->FunctionValues().GetNumElements() != 1)
                RuntimeError("%ls %ls operation: Input(2) should be a single element matrix and have the value 0 (row) or 1 (col).", NodeName().c_str(), OperationName().c_str());
            m_strideDim = (size_t) Inputs(2)->FunctionValues().Get00Element();
            if (m_strideDim != 0 && m_strideDim != 1)
                RuntimeError("%ls %ls operation: Input(2) should be a single element matrix and have the value 0 (row) or 1 (col).", NodeName().c_str(), OperationName().c_str());
            //if (Inputs(2)->m_needGradient)        // disabled because this is a flag that belongs to Network. Node should simply not propagate anything into it
            //    RuntimeError("StrideTimes: No gradient update should be on input(2).");

            size_t rows0 = Inputs(0)->GetNumRows(), cols0 = Inputs(0)->GetNumCols();
            size_t rows1 = Inputs(1)->GetNumRows(), cols1 = Inputs(1)->GetNumCols();

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
            LinkToMBLayout(Inputs(1)->GetMBLayout());   // retains the layout of the right input

            InferImageDimsFromInputs();
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(1, false); //the second one is the input since it's column wize

            //after multiplication the structure is lost
            m_imageLayout = ImageLayoutWHC(1, Inputs(0)->GetNumRows(), 1);
        }
    };

    template class StrideTimesNode<float>;
    template class StrideTimesNode<double>;

}}}
