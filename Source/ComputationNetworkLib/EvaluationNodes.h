//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "gammacalculation.h"

#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// ErrorPredictionNode (label, prediction)   or ErrorPredictionNode (prediction, label)
// Performs classification and error counting.
// Result is an error rate, lower = better.
// -----------------------------------------------------------------------

template <class ElemType>
class ErrorPredictionNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"ErrorPrediction";
    }

public:
    DeclareConstructorFromConfig(ErrorPredictionNode);
    ErrorPredictionNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void BackpropToNonLooping(size_t /*inputIndex*/) override
    {
        LogicError("%ls operation is used for evaluation only.", OperationName().c_str());
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override
    {
        return false;
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(Input(0)->GetMBLayout());
        Input(0)->ValueFor(fr).VectorMax(*m_maxIndexes0, *m_maxValues, true);
        Input(1)->ValueFor(fr).VectorMax(*m_maxIndexes1, *m_maxValues, true, m_topK);
        MaskMissingColumnsToZero(*m_maxIndexes0, Input(0)->GetMBLayout(), fr);
        MaskMissingColumnsToZero(*m_maxIndexes1, Input(1)->GetMBLayout(), fr);
        Value().AssignNumOfDiff(*m_maxIndexes0, *m_maxIndexes1, m_topK > 1);
#if NANCHECK
        Value().HasNan("ErrorPrediction");
#endif
#if DUMPOUTPUT
        Value().Print("ErrorPredictionNode");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateBinaryReduce(isFinalValidationPass);

        m_topK = 1;
        // TODO: Make topK a constructor parameter
        if (m_inputs.size() == 3)
        {
            if (Input(2)->GetSampleLayout().GetNumElements() != 1)
                InvalidArgument("%ls %ls operation requires TopK to be a scalar value.", NodeName().c_str(), OperationName().c_str());
            m_topK = static_cast<int>(Input(2)->Get00Element());
        }
    }

    virtual void UpdateFunctionMBSize() override
    {
        Base::UpdateFunctionMBSize();

        // resize the temporaries to their proper size
        size_t cols = Input(0)->Value().GetNumCols();
        m_maxIndexes0->Resize(m_topK, cols);
        m_maxIndexes1->Resize(m_topK, cols);
        m_maxValues->Resize(m_topK, cols);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<ErrorPredictionNode<ElemType>>(nodeP);
            node->m_maxIndexes0->SetValue(*m_maxIndexes0);
            node->m_maxIndexes1->SetValue(*m_maxIndexes1);
            node->m_maxValues->SetValue(*m_maxValues);
        }
    }
    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_maxIndexes0, matrixPool);
        RequestMatrixFromPool(m_maxIndexes1, matrixPool);
        RequestMatrixFromPool(m_maxValues, matrixPool);
    }

    // release temp matrices that are only used by forward computation
    // don't release matrices that need to be used in the gradient computation
    virtual void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterForwardProp(matrixPool);
        ReleaseMatrixToPool(m_maxIndexes0, matrixPool);
        ReleaseMatrixToPool(m_maxIndexes1, matrixPool);
        ReleaseMatrixToPool(m_maxValues, matrixPool);
    }

private:
    shared_ptr<Matrix<ElemType>> m_maxIndexes0, m_maxIndexes1;
    shared_ptr<Matrix<ElemType>> m_maxValues;
    int m_topK;
};

template class ErrorPredictionNode<float>;
template class ErrorPredictionNode<double>;

#ifdef COMING_SOON

// -----------------------------------------------------------------------
// SequenceDecoderNode (label, position_dependent_score, transition_score)
// Decoder that matches CRF training.
//  - label : output label vector of [0:T-1]
//  - position_dependent_score : score from position dependent node,
//    in the R-CRF case, it is the RNN output score before softmax
//  - transition score : score from the transition node,
//    in the R-CRF case, it is the transition probability between labels
// -----------------------------------------------------------------------

template <class ElemType>
class SequenceDecoderNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<3>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"SequenceDecoderNode";
    }

private:
    // TODO: member variables go to the end
    Matrix<ElemType> mAlpha;
    Matrix<ElemType> mBacktrace;

    int mStartLab; // the starting output label
    int mEndLab;   // the ending output label, if avaliable
    ElemType m_default_activity;

public:
    DeclareConstructorFromConfigWithNumInputs(SequenceDecoderNode);
    SequenceDecoderNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name),
          mAlpha(deviceId),
          mBacktrace(deviceId),
          mStartLab(-1),
          mEndLab(-1)
    {
    }

    static void DecideStartEndingOutputLab(const Matrix<ElemType>& lbls, int& stt, int& stp)
    {
        if (stt != -1 && stp != -1)
            return; // have computed before

        int iNumPos = lbls.GetNumCols();

        int firstLbl = -1;
        for (int ik = 0; ik < lbls.GetNumRows(); ik++)
            if (lbls(ik, 0) != 0)
            {
                firstLbl = ik;
                break;
            }

        int lastLbl = -1;
        for (int ik = 0; ik < lbls.GetNumRows(); ik++)
            if (lbls(ik, iNumPos - 1) != 0)
            {
                lastLbl = ik;
                break;
            }

        stt = firstLbl;
        stp = lastLbl;
    };

    virtual void BackpropToNonLooping(size_t /*inputIndex*/) override // scaled by 2*number of elements in the Matrix<ElemType>
    {
        LogicError("SequenceDecoder is used for evaluation only.");
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override
    {
        return false;
    }

    // compute posterior probability of label y at position t
    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        DecideStartEndingOutputLab(Input(0)->Value(), mStartLab, mEndLab);
        ForwardPropS(mAlpha, mBacktrace, Value(), Input(1)->Value(),
                     Input(2)->Value(), mStartLab, mEndLab);
    }

    // compute forward backward algorithm
    void ForwardPropS(Matrix<ElemType>& alpha, Matrix<ElemType>& backtrace, Matrix<ElemType>& functionValues, const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, const size_t stt, const size_t stp)
    {
        // to-do, each slice is for one sentence
        // to-do, number of slices correspond to number of frames
        // this implementation only supports one sentence per minibatch

        // change to other values so can support multiple sentences in each minibatch
        ForwardCompute(alpha, backtrace, pos_scores, pair_scores, stt);
        BackwardCompute(functionValues, backtrace, stp);
    };

    // compute forward backward algorithm
    static void ForwardCompute(Matrix<ElemType>& alpha,
                               Matrix<ElemType>& backtrace,
                               const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores,
                               const size_t stt)
    {
        // to-do, shift more than 1 to support muliple sentences per minibatch
        int iNumPos = pos_scores.GetNumCols();
        int iNumLab = pos_scores.GetNumRows();
        size_t iTmp = 0;

        // need to have
        alpha.Resize(iNumLab, iNumPos);
        backtrace.Resize(iNumLab, iNumPos);

        for (int t = 0; t < iNumPos; t++)
        {
            for (int k = 0; k < iNumLab; k++)
            {
                ElemType fTmp = (ElemType) LZERO;
                if (t > 1)
                {
                    for (int j = 0; j < iNumLab; j++)
                    {
                        ElemType fAlpha = alpha(j, t - 1) + pair_scores(k, j);
                        if (fAlpha > fTmp)
                        {
                            fTmp = fAlpha;
                            iTmp = j;
                        }
                    }
                    fTmp += pos_scores(k, t); // include position dependent score
                }
                else
                {
                    // with constrain that the first word is labeled as a given symbol
                    iTmp = stt;
                    fTmp = 0;
                    if (t == 1)
                    {
                        fTmp = alpha(iTmp, t - 1);
                        fTmp += pair_scores(k, iTmp);
                        fTmp += pos_scores(k, t);
                    }
                    else
                    {
                        fTmp = (k == stt) ? pos_scores(k, t) : (ElemType) LZERO;
                    }
                }
                alpha(k, t) = fTmp;
                backtrace(k, t) = (ElemType) iTmp;
            }
        }
    };

    // compute backward algorithm
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
            lastlbl = (size_t) backtrace(lastlbl, t);
            decodedpath(lastlbl, t - 1) = 1;
        }
    };

    // need to feed in pseudo label data, which tells the decoder what is the beginning
    // and ending output symbol. these symbols will constrain the search space
    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        if (isFinalValidationPass)
            if (!(Input(1)->GetSampleMatrixNumRows() == Input(2)->GetSampleMatrixNumRows() && // position dependent and pair scores have same number of labels
                  Input(0)->GetSampleMatrixNumRows() == Input(1)->GetSampleMatrixNumRows() &&
                  Input(0)->GetSampleMatrixNumCols() == Input(1)->GetSampleMatrixNumCols() && // position dependent and pair scores have the same observation numbers
                  Input(2)->GetSampleMatrixNumCols() == Input(2)->GetSampleMatrixNumRows()))
            {
                LogicError("The Matrix<ElemType>  dimension in the SequenceDecoderNode operation does not match.");
            }
        // BUGBUG: No SetDims()?
        m_sampleLayout = TensorShape();
    }
};

template class SequenceDecoderNode<float>;
template class SequenceDecoderNode<double>;

#endif

} } }
