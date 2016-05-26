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

// -----------------------------------------------------------------------
// IRMetricEvalNode (label, prediction, pair index)
// Performs IRMetric calculation
// Result is an IRMetric in the range of [0,100], the higher the better
// -----------------------------------------------------------------------

template <class ElemType>
class IRMetricEvalNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<3>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"IRMetricEval";
    }

public:
    DeclareConstructorFromConfig(IRMetricEvalNode);
    IRMetricEvalNode(DEVICEID_TYPE deviceId, const wstring& name)
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
        // Input(0) is l (label), Input(1) is s (scores), Input(2) is k (pair index)
        FrameRange fr(Input(0)->GetMBLayout());
        // construct matrices for further computation
        const Matrix<ElemType>& singleLabels = Input(0)->ValueFor(fr);
        const Matrix<ElemType>& scores = Input(1)->ValueFor(fr);
        const Matrix<ElemType>& pairIndeces = Input(2)->ValueFor(fr);
        size_t nCols = singleLabels.GetNumCols();
        // iterate through all samples
        size_t i = 0, totalUrls = 0, nUrls = 0;
        QueryUrls aqu;

        // Populate m_queryUrls
        while (i < nCols)
        {
            nUrls = (size_t)pairIndeces(0, i) + 1;
            totalUrls += nUrls;

            m_queryUrls.push_back(aqu);
            QueryUrls& qub = m_queryUrls.back();
            qub.urls.resize(nUrls);

            std::vector<Url>& urls = qub.urls;
            typename std::vector<Url>::iterator its = m_urlSorter.begin(), it = urls.begin();
            typename std::vector<Url>::iterator its0 = its;
            int rk0 = 0; // rk0 is original rank, rk is the sorted rank 
            for (; it != urls.end(); it++)
            {
                it->id = i;
                it->rk0 = rk0++;
                it->sc = scores(0, i);
                it->K = (int)pairIndeces(0, i);
                it->gn = singleLabels(0, i++);

                *its++ = *it;
            }

            std::sort(its0, its);

            // set the sorted rk order to each url
            // the urls are still in the original order
            int rk = 0;
            for (it = its0; it != its; it++)
                urls[it->rk0].rk = rk++;
        }

        // the total number of samples should be the same as totalK and nCols
        if (totalUrls != nCols)
        {
            InvalidArgument("In %ls %ls totalK != nCols.", NodeName().c_str(), OperationName().c_str());
        }

        // calculate IRMetrics
        size_t sampleCount = 0;
        for (typename std::list<QueryUrls>::iterator itqu = m_queryUrls.begin(); itqu != m_queryUrls.end(); itqu++)
        {
            for (typename std::vector<Url>::iterator iturl = itqu->urls.begin(); iturl != itqu->urls.end(); iturl++, sampleCount++)
            {
                Url& aUrl = *iturl;
                (*m_perUrlGainsOrig)(0, sampleCount) = aUrl.gn;
                (*m_perUrlGainsSort)(0, sampleCount) = aUrl.gn;
                (*m_perUrlWeightsOrig)(0, sampleCount) = (ElemType)aUrl.rk0;
                (*m_perUrlWeightsSort)(0, sampleCount) = (ElemType)aUrl.rk;
            }
        }

        // log(2+rank)
        *m_perUrlWeightsOrig += 2.0;
        m_perUrlWeightsOrig->InplaceLog();
        *m_perUrlWeightsSort += 2.0;
        m_perUrlWeightsSort->InplaceLog();
        // gain/log(2+rank)
        m_perUrlGainsOrig->AssignElementDivisionOf(*m_perUrlGainsOrig, *m_perUrlWeightsOrig);
        m_perUrlGainsSort->AssignElementDivisionOf(*m_perUrlGainsSort, *m_perUrlWeightsSort);

        // per query aggregation
        const Matrix<ElemType>& perUrlGainOrig = *m_perUrlGainsOrig;
        const Matrix<ElemType>& perUrlGainSort = *m_perUrlGainsSort;
        ElemType IRMetricValue = 0.0;
        size_t nValidQueries = 0;
        ElemType irm0 = 0.0, irm1 = 0.0;
        // IRMetric @ 1
        for (typename std::list<QueryUrls>::iterator itqu = m_queryUrls.begin(); itqu != m_queryUrls.end(); itqu++)
        {
            QueryUrls& qu = *itqu;
            irm0 = perUrlGainOrig(0, qu.urls.begin()->id);
            if (irm0 == 0.0) continue;

            for (typename std::vector<Url>::iterator iturl = itqu->urls.begin(); iturl != itqu->urls.end(); iturl++)
            {
                Url& url = *iturl;
                if (url.rk == 0)
                {
                    irm1 = perUrlGainSort(0, url.id);
                    break;
                }
            }

            IRMetricValue += (irm1 / irm0);
            nValidQueries++;
        }

        if (nValidQueries == 0)
            LogicError("In %ls %ls nValidQueries==0, check your data.", NodeName().c_str(), OperationName().c_str());

        IRMetricValue = IRMetricValue / nValidQueries * 100 * nCols;
        Value().SetValue(IRMetricValue);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        if (m_inputs.size() != 3)
            InvalidArgument("%ls %ls operation requires three inputs.", NodeName().c_str(), OperationName().c_str());

        if (Input(0)->NeedsGradient() == true || Input(2)->NeedsGradient() == true)
            InvalidArgument("%ls %ls operation needs input type (no gradient) for the 1st and 3rd inputs.", NodeName().c_str(), OperationName().c_str());

        ValidateBinaryReduce(isFinalValidationPass);
    }

    virtual void UpdateFunctionMBSize() override
    {
        FrameRange fr(Input(0)->GetMBLayout());

        // clean up first
        if (!m_queryUrls.empty()) m_queryUrls.clear();
        if (!m_urlSorter.empty()) m_urlSorter.clear();
        if (!m_logWeights.empty()) m_logWeights.clear();

        const Matrix<ElemType>& pairIndeces = Input(2)->ValueFor(fr);
        m_samples = pairIndeces.GetNumCols();
        m_pairCounts = (size_t)pairIndeces.SumOfElements();

        // m_pairwiseDifferences->Resize(1, m_pairCounts);
        // m_sigmaPairwiseDiff->Resize(1, m_pairCounts);
        // m_logexpterm->Resize(1, m_pairCounts);

        m_perUrlGainsOrig->Resize(1, m_samples);
        m_perUrlGainsSort->Resize(1, m_samples);
        m_perUrlWeightsOrig->Resize(1, m_samples);
        m_perUrlWeightsSort->Resize(1, m_samples);

        // prepare sorting buffer
        m_maxPairIndexIndex->Resize(1, 1);
        m_maxPairValues->Resize(1, 1);
        pairIndeces.VectorMax(*m_maxPairIndexIndex, *m_maxPairValues, false);

        // TODO: Double check this part
        size_t maxNumofUrlsPerQuery = (size_t)(*m_maxPairValues)(0, 0) + 1;
        // keep one additional space to avoid pointer moving out
        m_urlSorter.resize(maxNumofUrlsPerQuery + 1);

        // prepared lookup table
        m_logWeights.resize(maxNumofUrlsPerQuery);
        size_t i = 0;
        for (typename std::vector<ElemType>::iterator it = m_logWeights.begin(); it != m_logWeights.end(); it++, i++)
        {
            *it = (ElemType)log(2.0 + i);
        }
    }


    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<IRMetricEvalNode<ElemType>>(nodeP);
            // node->m_pairwiseDifferences->SetValue(*m_pairwiseDifferences);
            // node->m_sigmaPairwiseDiff->SetValue(*m_sigmaPairwiseDiff);
            // node->m_logexpterm->SetValue(*m_logexpterm);
            node->m_maxPairIndexIndex->SetValue(*m_maxPairIndexIndex);
            node->m_maxPairValues->SetValue(*m_maxPairValues);
            node->m_perUrlGainsOrig->SetValue(*m_perUrlGainsOrig);
            node->m_perUrlGainsSort->SetValue(*m_perUrlGainsSort);
            node->m_perUrlWeightsOrig->SetValue(*m_perUrlWeightsOrig);
            node->m_perUrlWeightsSort->SetValue(*m_perUrlWeightsSort);

            node->m_queryUrls = m_queryUrls;
            node->m_urlSorter = m_urlSorter;
            node->m_logWeights = m_logWeights;
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        // RequestMatrixFromPool(m_pairwiseDifferences, matrixPool);
        // RequestMatrixFromPool(m_sigmaPairwiseDiff, matrixPool);
        // RequestMatrixFromPool(m_logexpterm, matrixPool);
        RequestMatrixFromPool(m_maxPairIndexIndex, matrixPool);
        RequestMatrixFromPool(m_maxPairValues, matrixPool);
        RequestMatrixFromPool(m_perUrlGainsOrig, matrixPool);
        RequestMatrixFromPool(m_perUrlGainsSort, matrixPool);
        RequestMatrixFromPool(m_perUrlWeightsOrig, matrixPool);
        RequestMatrixFromPool(m_perUrlWeightsSort, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        // ReleaseMatrixToPool(m_pairwiseDifferences, matrixPool);
        // ReleaseMatrixToPool(m_sigmaPairwiseDiff, matrixPool);
        // ReleaseMatrixToPool(m_logexpterm, matrixPool);
        ReleaseMatrixToPool(m_maxPairIndexIndex, matrixPool);
        ReleaseMatrixToPool(m_maxPairValues, matrixPool);
        ReleaseMatrixToPool(m_perUrlGainsOrig, matrixPool);
        ReleaseMatrixToPool(m_perUrlGainsSort, matrixPool);
        ReleaseMatrixToPool(m_perUrlWeightsOrig, matrixPool);
        ReleaseMatrixToPool(m_perUrlWeightsSort, matrixPool);

        // is this the right place?  it was not called after bp.
        m_queryUrls.clear();
        m_urlSorter.clear();
        m_logWeights.clear();
    }

protected:

    struct Url
    {
        int id; // sample id
        int rk0; // original rank based on label
        int rk; // rank based on s in the associated query
        ElemType sc; // score
        ElemType gn; // gain
        int K; // the pair index
        bool operator < (const Url &url) const{
            // tie breaking
            if (sc == url.sc || isnan(sc) || isnan(url.sc))
            {
                return gn < url.gn;
            }
            return sc > url.sc;
        }
    };

    struct QueryUrls
    {
        ElemType irm0; // the ideal IR Metric
        ElemType irm; // IR metric based on the current scores

        std::vector<Url> urls;
    };

    // master data structure
    std::list<QueryUrls> m_queryUrls;
    // buffer for sorting
    std::vector<Url> m_urlSorter;
    // lookup table for position based weights
    std::vector<ElemType> m_logWeights;

    size_t m_samples;
    size_t m_pairCounts;
    // TODO: to make it a config?
    // ElemType m_sigma;
    // shared_ptr<Matrix<ElemType>> m_pairwiseDifferences;
    // sigma*(si - sj)
    // shared_ptr<Matrix<ElemType>> m_sigmaPairwiseDiff;
    // 1/(1+exp(sigma*(si - sj)))
    // shared_ptr<Matrix<ElemType>> m_logexpterm;
    // used to calculate the max number of urls per query
    shared_ptr<Matrix<ElemType>> m_maxPairIndexIndex;
    shared_ptr<Matrix<ElemType>> m_maxPairValues;
    // store the gains and weights
    shared_ptr<Matrix<ElemType>> m_perUrlGainsOrig;
    shared_ptr<Matrix<ElemType>> m_perUrlGainsSort;
    shared_ptr<Matrix<ElemType>> m_perUrlWeightsOrig;
    shared_ptr<Matrix<ElemType>> m_perUrlWeightsSort;
};

template class IRMetricEvalNode<float>;
template class IRMetricEvalNode<double>;

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
