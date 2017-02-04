//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "BatchNormalizationEngine.h"
#include "RNGHandle.h"
#include "InputAndParamNodes.h"
#include "CPURNGHandle.h"

#define __STDC_FORMAT_MACROS
#include <inttypes.h>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <random>

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// SquareErrorNode (left, right)
// = SumElements ((left - right) .* (left - right))
// -----------------------------------------------------------------------

template <class ElemType>
class SquareErrorNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<2>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"SquareError"; }

public:
    DeclareConstructorFromConfigWithNumInputs(SquareErrorNode);
    SquareErrorNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void UpdateFunctionMBSize() override
    {
        m_leftMinusRight->Resize(Input(0)->Value());
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        m_leftMinusRight->AssignDifferenceOf(InputRef(0).ValueFor(fr), InputRef(1).ValueFor(fr));
        MaskMissingColumnsToZero(*m_leftMinusRight, InputRef(0).GetMBLayout(), fr); // we are fine since it will only be called with full minibatch.
        ElemType v = m_leftMinusRight->FrobeniusNorm(); // v = sqrt( sum{ (I0[i] - I1[i])^2 } )
        Value().VerifySize(1, 1);
        Value().SetValue(v * v);  // Value = sum{ (I0[i] - I1[i])^2 }
#if NANCHECK
        Value().HasNan("SquareError");
#endif
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        auto gradient = InputRef(inputIndex).GradientFor(fr);
        Matrix<ElemType>::Multiply1x1AndWeightedAdd(inputIndex == 0 ? 2.0f : -2.0f, Gradient() /*1x1*/, *m_leftMinusRight, 1.0f, gradient); // O = (I0-I1)^2; dO/dI0 = 2*(I0-I1); dO/dI1 = -2*(I0-I1)
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateBinaryReduce(isFinalValidationPass);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<SquareErrorNode<ElemType>>(nodeP);
            node->m_leftMinusRight->SetValue(*m_leftMinusRight);
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_leftMinusRight, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_leftMinusRight, matrixPool);
    }

private:
    shared_ptr<Matrix<ElemType>> m_leftMinusRight;
};

template class SquareErrorNode<float>;
template class SquareErrorNode<double>;

// -----------------------------------------------------------------------
// CrossEntropyWithSoftmaxNode (labels, prediction)
// calculates: -sum(left_i * log(softmax_i(right)))
// -----------------------------------------------------------------------

template <class ElemType>
class CrossEntropyWithSoftmaxNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<2>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"CrossEntropyWithSoftmax"; }

public:
    DeclareConstructorFromConfigWithNumInputs(CrossEntropyWithSoftmaxNode);
    CrossEntropyWithSoftmaxNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        // left input is scalar
        if (inputIndex == 0) // left derivative
        {
#if DUMPOUTPUT
            m_logSoftmaxOfRight->Print("CrossEntropyWithSoftmax Partial-logSoftmaxOfRight");
            Gradient().Print("CrossEntropyWithSoftmax Partial-gradientValues");
            InputRef(0).GradientFor(fr).Print("CrossEntropyWithSoftmaxNode Partial-Left-in");
#endif

            auto gradient = InputRef(0).GradientFor(fr);
            Matrix<ElemType>::Multiply1x1AndWeightedAdd(-1.0f, Gradient() /*1x1*/, *m_logSoftmaxOfRight, 1.0f, gradient);
#if DUMPOUTPUT
            InputRef(0).GradientFor(fr).Print("CrossEntropyWithSoftmaxNode Partial-Left-out");
#endif
        }

        else if (inputIndex == 1) // right derivative
        {
#if DUMPOUTPUT
            m_softmaxOfRight->Print("CrossEntropyWithSoftmax Partial-softmaxOfRight");
            InputRef(0).ValueFor(fr).Print("CrossEntropyWithSoftmax Partial-inputFunctionValues");
            Gradient().Print("CrossEntropyWithSoftmax Partial-gradientValues");
            InputRef(1).GradientFor(fr).Print("CrossEntropyWithSoftmaxNode Partial-Right-in");
#endif

            auto gradient = InputRef(1).GradientFor(fr);
            Matrix<ElemType>::AddScaledDifference(Gradient(), *m_softmaxOfRight, InputRef(0).ValueFor(fr), gradient);
#if DUMPOUTPUT
            InputRef(1).GradientFor(fr).Print("CrossEntropyWithSoftmaxNode Partial-Right");
#endif
#ifdef _DEBUG
            InputRef(1).InvalidateMissingGradientColumns(fr); // TODO: This should not be necessary.
#endif
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

    virtual void UpdateFunctionMBSize() override
    {
        m_logSoftmaxOfRight->Resize(Input(1)->Value());
        m_softmaxOfRight->Resize(*m_logSoftmaxOfRight);
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override // -sum(left_i * log(softmax_i(right)))
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        // first compute the softmax (column-wise)
        // Note that we need both log and non-log for gradient computation.
        m_logSoftmaxOfRight->AssignLogSoftmaxOf(InputRef(1).ValueFor(fr), true);
        // BUGBUG: No need to compute m_softmaxOfRight in ForwardProp, should be moved to BackpropTo().
        m_softmaxOfRight->SetValue(*m_logSoftmaxOfRight);
        m_softmaxOfRight->InplaceExp();
        // flatten all gaps to zero, such that gaps will contribute zero to the sum
        MaskMissingColumnsToZero(*m_logSoftmaxOfRight, InputRef(1).GetMBLayout(), fr);
        // reduce over all frames
        Value().AssignInnerProductOfMatrices(InputRef(0).MaskedValueFor(fr), *m_logSoftmaxOfRight);
        Value() *= -1;
#if NANCHECK
        Value().HasNan("CrossEntropyWithSoftmax");
#endif
#if DUMPOUTPUT
        Value().Print("CrossEntropyWithSoftmaxNode");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateBinaryReduce(isFinalValidationPass);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<CrossEntropyWithSoftmaxNode<ElemType>>(nodeP);
            node->m_logSoftmaxOfRight->SetValue(*m_logSoftmaxOfRight);
            node->m_softmaxOfRight->SetValue(*m_softmaxOfRight);
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_logSoftmaxOfRight, matrixPool);
        RequestMatrixFromPool(m_softmaxOfRight, matrixPool);
    }

protected:
    shared_ptr<Matrix<ElemType>> m_logSoftmaxOfRight;
    shared_ptr<Matrix<ElemType>> m_softmaxOfRight;
};

template class CrossEntropyWithSoftmaxNode<float>;
template class CrossEntropyWithSoftmaxNode<double>;

// -----------------------------------------------------------------------
/// CrossEntropyNode (labels, prediction)
// -----------------------------------------------------------------------

// calculates: -sum(left_i * log(right_i))
// assume softmax is already done
// You probably want to use CrossEntropyWithSoftMaxNode instead, it is more efficient in most cases.
template <class ElemType>
class CrossEntropyNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<2>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"CrossEntropy";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(CrossEntropyNode);
    CrossEntropyNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNodeNonLooping::*/ BackpropToNonLooping(size_t inputIndex) override
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        // left Node must be a scalar
        if (inputIndex == 0) // left derivative
        {
            BackpropToLeft(*m_logOfRight, InputRef(0).GradientFor(fr), Gradient());
        }
        else
        {
            // Resize m_lefDivRight as it has not been done before.
            m_leftDivRight->Resize(InputRef(1).Value());
            BackpropToRight(*m_leftDivRight, InputRef(0).ValueFor(fr), InputRef(1).ValueFor(fr), InputRef(1).GradientFor(fr), Gradient());
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

    /*TODO: merge with call site*/ void BackpropToLeft(const Matrix<ElemType>& logOfRight, Matrix<ElemType> inputGradientValues,
                                                       const Matrix<ElemType>& gradientValues)
    {
        Matrix<ElemType>::Multiply1x1AndWeightedAdd(-1.0f, gradientValues /*1x1*/, logOfRight, 1.0f, inputGradientValues);
    }

    /*TODO: merge with call site*/ void BackpropToRight(Matrix<ElemType>& leftDivRight,
                                                        const Matrix<ElemType> inputFunctionValues0, const Matrix<ElemType> inputFunctionValues1,
                                                        Matrix<ElemType> inputGradientValues, const Matrix<ElemType>& gradientValues)
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        leftDivRight.AssignElementDivisionOf(inputFunctionValues0, inputFunctionValues1);
        MaskMissingColumnsToZero(leftDivRight, InputRef(0).GetMBLayout(), fr);
        Matrix<ElemType>::Multiply1x1AndWeightedAdd(-1.0f, gradientValues /*1x1*/, leftDivRight, 1.0f, inputGradientValues);
    }

    virtual void UpdateFunctionMBSize() override
    {
        // Delay resize of m_leftDivRight to backprop, as it is not allocated for forwardprop.
        m_logOfRight->Resize(InputRef(1).Value());
    }

    // -sum(left_i * log(right_i))
    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        m_logOfRight->SetValue(InputRef(1).ValueFor(fr));
        m_logOfRight->InplaceLog();
        MaskMissingColumnsToZero(*m_logOfRight, InputRef(1).GetMBLayout(), fr);
        Value().AssignInnerProductOfMatrices(InputRef(0).MaskedValueFor(fr), *m_logOfRight);
        Value() *= -1;
#if NANCHECK
        Value().HasNan("CrossEntropy");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateBinaryReduce(isFinalValidationPass);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<CrossEntropyNode<ElemType>>(nodeP);
            node->m_logOfRight->SetValue(*m_logOfRight);
            if (m_leftDivRight != nullptr)
            {
                node->m_leftDivRight->SetValue(*m_leftDivRight);
            }
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_logOfRight, matrixPool);
    }

    // request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_leftDivRight, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_logOfRight, matrixPool);
        ReleaseMatrixToPool(m_leftDivRight, matrixPool);
    }

private:
    // matrix value passed from evaluate to computePartial
    shared_ptr<Matrix<ElemType>> m_logOfRight;
    // temporary
    shared_ptr<Matrix<ElemType>> m_leftDivRight;
};

template class CrossEntropyNode<float>;
template class CrossEntropyNode<double>;

// -----------------------------------------------------------------------
// MatrixL1RegNode (input)
// TODO: share most code with MatrixL2RegNode
// -----------------------------------------------------------------------

template <class ElemType>
class MatrixL1RegNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<1>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"MatrixL1Reg"; }

public:
    DeclareConstructorFromConfigWithNumInputs(MatrixL1RegNode);
    MatrixL1RegNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNodeNonLooping::*/ BackpropToNonLooping(size_t inputIndex) override // scale by number of cols (or samples)
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        assert(inputIndex == 0);
        inputIndex;
        BackpropToS(*m_gradientOfL1Norm, InputRef(0).GradientFor(fr), Gradient(), InputRef(0).ValueFor(fr));
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

    /*TODO: merge with call site*/ void BackpropToS(Matrix<ElemType>& gradientOfL1Norm,
                                                    Matrix<ElemType> inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& inputFunctionValues)
    {
        gradientOfL1Norm.AssignSignOf(inputFunctionValues);
        Matrix<ElemType>::Multiply1x1AndWeightedAdd(+1.0f, gradientValues /*1x1*/, gradientOfL1Norm, 1.0f, inputGradientValues);
    }

    virtual void UpdateFunctionMBSize() override
    {
        m_gradientOfL1Norm->Resize(InputRef(0).Value());
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        Value().VerifySize(1, 1);
        Value().SetValue(InputRef(0).MaskedValueFor(fr).MatrixNorm1());
#if NANCHECK
        Value().HasNan("MatrixL1Reg");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateUnaryReduce(isFinalValidationPass);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<MatrixL1RegNode<ElemType>>(nodeP);
            node->m_gradientOfL1Norm->SetValue(*m_gradientOfL1Norm);
        }
    }

    // request matrices that are needed for gradient computation
    virtual void RequestMatricesBeforeBackprop(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_gradientOfL1Norm, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_gradientOfL1Norm, matrixPool);
    }

private:
    shared_ptr<Matrix<ElemType>> m_gradientOfL1Norm; // temporary
};

template class MatrixL1RegNode<float>;
template class MatrixL1RegNode<double>;

// -----------------------------------------------------------------------
// LambdaRankNode (gain, prediction, queryId)
// Check "From RankNet to LambdaRank to LambdaMART: An Overview" for details.
// -----------------------------------------------------------------------

template <class ElemType>
class LambdaRankNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<3>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"LambdaRank";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(LambdaRankNode);
    LambdaRankNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name), m_sigma(1.0)
    {
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        FrameRange fr(Input(0)->GetMBLayout());

        if (inputIndex == 1 &&  // right derivative
            m_numberOfUrlPairs > 0)   // 
        {
            auto gradient = Input(1)->GradientFor(fr);

            // sigma * (si - sj)
            *m_pairwiseDifferences *= m_sigma;
            // exp(sigma * (si - sj))
            m_lambdas->AssignExpOf(*m_pairwiseDifferences);
            // 1 + exp(sigma * (si - sj))
            *m_lambdas += 1;
            // 1 / (1 + exp(sigma * (si - sj)))
            m_lambdas->AssignElementInverseOf(*m_lambdas);
            // -sigma/(1+exp(sigma*(si - sj)))
            m_lambdas->AssignProductOf(-m_sigma, *m_lambdas);

            // Get the update of weight.
            const Matrix<ElemType>& lambdas = *m_lambdas;
            m_weightUpdate->SetValue(gradient);
            size_t pairsCount = 0;
            ElemType discountI, discountJ;
            ElemType gainI;
            ElemType lambdaIJ;
            for (auto qu : m_queryUrls)
            {
                ElemType idealMetric = qu.m_idealMetric;
                for (typename std::vector<Url>::iterator itUrlI = qu.m_urls.begin(); itUrlI != qu.m_urls.end(); itUrlI++)
                {
                    Url& UrlI = *itUrlI;
                    size_t k = UrlI.m_K;
                    discountI = m_logWeights[UrlI.m_rank];
                    gainI = UrlI.m_gain;
                    if (k == 0) continue;
                    for (typename std::vector<Url>::iterator itUrlJ = itUrlI + 1; itUrlJ <= itUrlI + k; itUrlJ++)
                    {
                        Url& UrlJ = *itUrlJ;
                        discountJ = m_logWeights[UrlJ.m_rank];
                        if (abs(gainI - UrlJ.m_gain) < 0.0000001)
                        {
                            continue;
                        }

                        // delta DCG
                        lambdaIJ = (gainI - UrlJ.m_gain) * (discountI - discountJ) / (discountI * discountJ);

                        // |delta NDCG|
                        lambdaIJ = (idealMetric == 0.0 ? (ElemType) 0.0 : abs(lambdaIJ / idealMetric));

                        // Combine lambda
                        lambdaIJ = lambdas(0, pairsCount++) * lambdaIJ;

                        // Assign to gradient
                        (*m_weightUpdate)(0, UrlI.m_id) += lambdaIJ;
                        (*m_weightUpdate)(0, UrlJ.m_id) -= lambdaIJ;
                        
                    }
                }
            }

            gradient.SetValue(*m_weightUpdate);
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

    virtual void UpdateFunctionMBSize() override
    {
        UpdateCounts();

        // clean up first
        if (!m_queryUrls.empty()) m_queryUrls.clear();
        if (!m_urlSorter.empty()) m_urlSorter.clear();
        if (!m_logWeights.empty()) m_logWeights.clear();

        m_pairwiseDifferences->Resize(1, m_numberOfUrlPairs);
        m_lambdas->Resize(1, m_numberOfUrlPairs);

        m_urlGain0->Resize(1, m_numberOfQueryUrls);
        m_urlGain1->Resize(1, m_numberOfQueryUrls);
        m_urlDiscount0->Resize(1, m_numberOfQueryUrls);
        m_urlDiscount1->Resize(1, m_numberOfQueryUrls);

        // keep one additional space to avoid pointer moving out
        m_urlSorter.resize(m_maxNumberOfUrlsPerQuery + 1);
        
        // prepared lookup table
        m_logWeights.resize(m_maxNumberOfUrlsPerQuery);
        size_t i = 0;
        for (typename std::vector<ElemType>::iterator it = m_logWeights.begin(); it != m_logWeights.end(); it++, i++)
        {
            *it = (ElemType) log(2.0 + i);
        }
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        // Inputs:
        //      0. gain
        //      1. predicted score
        //      2. query id (used to separate urls belonging to different queries)
        // 
        // Following is an example: two queries (0 and 1) and 6 urls (three for each).
        // 31,  0.9,    0
        // 7,   0.3,    0
        // 0,   0.0,    0
        // 3,   0.4,    1
        // 0,   0.5,    1
        // 0,   0.3,    1
        FrameRange fr(Input(0)->GetMBLayout());

        // Construct matrices for further computation.
        const Matrix<ElemType>& gains = Input(0)->ValueFor(fr);
        const Matrix<ElemType>& preds = Input(1)->ValueFor(fr);
        const Matrix<ElemType>& queryIds = Input(2)->ValueFor(fr);

        // Iterate through all samples
        size_t numberOfSamples = gains.GetNumCols();
        QueryUrls aqu;
        int previousQueryId = -1;
        int numberOfQueries = 0;

        // Iterate all samples and populate m_queryUrls table. 
        for (size_t i = 0; i < numberOfSamples; i++)
        {
            int queryId = (int)queryIds(0, i);
            // Samples are grouped by queries. Find all the urls 
            // belonging to each individual query.
            if (queryId != previousQueryId)
            {
                m_queryUrls.push_back(aqu);
                numberOfQueries++;
                previousQueryId = queryId;
            }

            // Get the last QueryUrls and add the Url.
            QueryUrls& qu = m_queryUrls.back();
            Url u(i, qu.m_urls.size(), preds(0, i), gains(0, i));
            qu.m_urls.push_back(u);
        }

        // Update K (number of url pairs that have smaller or equal gain), rk (rank of 
        // score in descending order) and and m_pairwiseDifferences (save for gradient 
        // computation).
        size_t pairCount = 0;
        for (auto &qu : m_queryUrls)
        {
            std::vector<Url>& urls = qu.m_urls;
            size_t numberOfUrls = urls.size();
            // Urls are pre-sorted in descending order of gains.
            ElemType minGain = urls[numberOfUrls - 1].m_gain;
            for (size_t j = 0; j < urls.size(); j++)
            {
                if (urls[j].m_gain > minGain)
                {
                    size_t numberOfPairs = numberOfUrls - j - 1;
                    urls[j].m_K = numberOfPairs;
                    if (numberOfPairs > 0)
                    {
                        for (size_t k = 0; k < numberOfPairs; k++)
                        {
                            (*m_pairwiseDifferences)(0, pairCount++) = urls[j].m_score - urls[j + 1 + k].m_score;
                        }
                    }
                }
                // Skip urls with gain equal to min (0).
                else 
                {
                    urls[j].m_K = 0;
                }
            }

            typename std::vector<Url>::iterator its = m_urlSorter.begin(), it = urls.begin();
            typename std::vector<Url>::iterator its0 = its;
            its = std::copy(it, urls.end(), its);
            std::sort(its0, its);
            // Set the sorted rk order to each url and 
            // the urls are still in the original order
            int rk = 0;
            for (it = its0; it != its; it++)
            {
                urls[it->m_rank0].m_rank = rk++;
            }
        }

        // Compute ir metric.
        size_t sampleCount = 0;
        for (const auto &qu: m_queryUrls)
        {
            for (const auto &url : qu.m_urls)
            {
                (*m_urlGain0)(0, sampleCount) = url.m_gain;
                (*m_urlGain1)(0, sampleCount) = url.m_gain;
                (*m_urlDiscount0)(0, sampleCount) = (ElemType)url.m_rank0;
                (*m_urlDiscount1)(0, sampleCount) = (ElemType)url.m_rank;
                sampleCount++;
            }
        }
        
        // log(2+rank)
        *m_urlDiscount0 += 2.0;
        m_urlDiscount0->InplaceLog();
        *m_urlDiscount1 += 2.0;
        m_urlDiscount1->InplaceLog();
        // gain/log(2+rank)
        m_urlGain0->AssignElementDivisionOf(*m_urlGain0, *m_urlDiscount0);
        m_urlGain1->AssignElementDivisionOf(*m_urlGain1, *m_urlDiscount1);

        // Aggregate at query level.
        const Matrix<ElemType>& urlDiscountedGain0 = *m_urlGain0;
        const Matrix<ElemType>& urlDiscountedGain1 = *m_urlGain1;
        ElemType irMetricValue = 0.0;
        for (auto &qu : m_queryUrls)
        {
            qu.m_idealMetric = 0.0;
            qu.m_metric = 0.0;

            for (const auto &url : qu.m_urls)
            {
                qu.m_idealMetric += urlDiscountedGain0(0, url.m_id);
                qu.m_metric += urlDiscountedGain1(0, url.m_id);
            }

            if (qu.m_idealMetric != 0.0)
            {
                irMetricValue += (qu.m_metric / qu.m_idealMetric);
            }
        }

        if (numberOfQueries == 0)
        {
            LogicError("In %ls %ls numberOfQueries==0, check your data.", NodeName().c_str(), OperationName().c_str());
        }

        // to make up the reporting
        irMetricValue = (1.0f - irMetricValue / numberOfQueries) * 100 * numberOfSamples;
        Value().SetValue(irMetricValue);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        if (m_inputs.size() != 3)
            InvalidArgument("%ls operation requires three inputs instead of %d.", NodeDescription().c_str(), (int)m_inputs.size());

        if (Input(0)->NeedsGradient() == true)
            InvalidArgument("%ls %ls operation needs input type (no gradient) for gain input.", NodeName().c_str(), OperationName().c_str());

        if (Input(2)->NeedsGradient() == true)
            InvalidArgument("%ls %ls operation needs input type (no gradient) for group input.", NodeName().c_str(), OperationName().c_str());

        ValidateBinaryReduce(isFinalValidationPass);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<LambdaRankNode<ElemType>>(nodeP);
            node->m_pairwiseDifferences->SetValue(*m_pairwiseDifferences);
            node->m_lambdas->SetValue(*m_lambdas);
            node->m_weightUpdate->SetValue(*m_weightUpdate);
            node->m_urlGain0->SetValue(*m_urlGain0);
            node->m_urlGain1->SetValue(*m_urlGain1);
            node->m_urlDiscount0->SetValue(*m_urlDiscount0);
            node->m_urlDiscount1->SetValue(*m_urlDiscount1);

            node->m_queryUrls = m_queryUrls;
            node->m_urlSorter = m_urlSorter;
            node->m_logWeights = m_logWeights;
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_pairwiseDifferences, matrixPool);
        RequestMatrixFromPool(m_lambdas, matrixPool);
        RequestMatrixFromPool(m_weightUpdate, matrixPool);
        RequestMatrixFromPool(m_urlGain0, matrixPool);
        RequestMatrixFromPool(m_urlGain1, matrixPool);
        RequestMatrixFromPool(m_urlDiscount0, matrixPool);
        RequestMatrixFromPool(m_urlDiscount1, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_pairwiseDifferences, matrixPool);
        ReleaseMatrixToPool(m_lambdas, matrixPool);
        ReleaseMatrixToPool(m_weightUpdate, matrixPool);
        ReleaseMatrixToPool(m_urlGain0, matrixPool);
        ReleaseMatrixToPool(m_urlGain1, matrixPool);
        ReleaseMatrixToPool(m_urlDiscount0, matrixPool);
        ReleaseMatrixToPool(m_urlDiscount1, matrixPool);
        
        // is this the right place?  it was not called after bp.
        m_queryUrls.clear();
        m_urlSorter.clear();
        m_logWeights.clear();
    }

protected:

    void UpdateCounts()
    {
        FrameRange fr(Input(0)->GetMBLayout());
        const Matrix<ElemType>& gains = Input(0)->ValueFor(fr);
        const Matrix<ElemType>& queryIds = Input(2)->ValueFor(fr);
        const size_t numberOfQueryUrls = gains.GetNumCols();
        int previousQueryId = -1;

        // Number of urls we have seen for the current query
        size_t numberOfUrls = 0;

        // Number of urls that have gains greater than 0 for the current query 
        size_t numberOfUrlsWithNonZeroGain = 0;
        size_t numberOfUrlPairs = 0;
        size_t maxNumberOfUrlsPerQuery = 0;
        for (size_t i = 0; i < numberOfQueryUrls; i++)
        {
            int queryId = (int)queryIds(0, i);
            ElemType gain = gains(0, i);
            if (queryId != previousQueryId)
            {
                // Ignore pairs between urls with zero gains.
                numberOfUrlPairs += (2 * numberOfUrls - 1 - numberOfUrlsWithNonZeroGain) * numberOfUrlsWithNonZeroGain / 2;
                if (numberOfUrls > maxNumberOfUrlsPerQuery)
                {
                    maxNumberOfUrlsPerQuery = numberOfUrls;
                }

                // New query
                numberOfUrls = 0;
                numberOfUrlsWithNonZeroGain = 0;
                previousQueryId = queryId;
            }
            
            numberOfUrls++;
            if (gain > 0)
            {
                numberOfUrlsWithNonZeroGain++;
            }
        }

        // Add last query.
        {
            numberOfUrlPairs += (2 * numberOfUrls - 1 - numberOfUrlsWithNonZeroGain) * numberOfUrlsWithNonZeroGain / 2;
            if (numberOfUrls > maxNumberOfUrlsPerQuery)
            {
                maxNumberOfUrlsPerQuery = numberOfUrls;
            }
        }

        m_numberOfQueryUrls = numberOfQueryUrls;
        m_numberOfUrlPairs = numberOfUrlPairs;
        m_maxNumberOfUrlsPerQuery = maxNumberOfUrlsPerQuery;
    }

    struct Url
    {
        Url()
        {
            m_id = 0;
            m_rank0 = 0;
            m_rank = 0;
            m_score = (ElemType)0;
            m_gain = (ElemType)0;
            m_K = 0;
        }

        Url(int id, int rk0, ElemType sc, ElemType gn) : m_id(id), m_rank0(rk0), m_rank(0), m_score(sc), m_gain(gn), m_K(0) {}

        int m_id;           // sample id
        int m_rank0;        // original rank based on label
        int m_rank;         // rank based on s in the associated query
        ElemType m_score;   // score
        ElemType m_gain;    // gain
        int m_K;            // the pair index
        bool operator < (const Url &url) const {
            // tie breaking
            if (m_score == url.m_score || std::isnan(m_score) || std::isnan(url.m_score))
            {
                return m_gain < url.m_gain;
            }

            return m_score > url.m_score;
        }
    };

    struct QueryUrls
    {
        ElemType m_idealMetric;  // the ideal NDCG
        ElemType m_metric;   // NDCG based on the current scores

        std::vector<Url> m_urls;
    };

    // master data structure
    std::list<QueryUrls> m_queryUrls;
    // buffer for sorting
    std::vector<Url> m_urlSorter;
    // lookup table for position based weights
    std::vector<ElemType> m_logWeights;

    size_t m_numberOfQueryUrls;
    size_t m_numberOfUrlPairs;
    size_t m_maxNumberOfUrlsPerQuery;

    ElemType m_sigma;
    // Score differences between url pairs
    shared_ptr<Matrix<ElemType>> m_pairwiseDifferences;
    // 1/(1+exp(sigma*(si - sj)))
    shared_ptr<Matrix<ElemType>> m_lambdas;
    
    // update of weight matrix
    shared_ptr<Matrix<ElemType>> m_weightUpdate;
    
    // store the gains and position discounts
    shared_ptr<Matrix<ElemType>> m_urlGain0;
    shared_ptr<Matrix<ElemType>> m_urlGain1;
    shared_ptr<Matrix<ElemType>> m_urlDiscount0;
    shared_ptr<Matrix<ElemType>> m_urlDiscount1;
};

template class LambdaRankNode<float>;
template class LambdaRankNode<double>;

// -----------------------------------------------------------------------
// MatrixL2RegNode (input)
// TODO: share most code with MatrixL1RegNode
// -----------------------------------------------------------------------

template <class ElemType>
class MatrixL2RegNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<1>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"MatrixL2Reg"; }

public:
    DeclareConstructorFromConfigWithNumInputs(MatrixL2RegNode);
    MatrixL2RegNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void /*ComputationNodeNonLooping::*/ BackpropToNonLooping(size_t inputIndex) override // scale by number of cols (or samples)
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        assert(inputIndex == 0);
        inputIndex;
        BackpropToS(InputRef(0).GradientFor(fr), Gradient(), InputRef(0).ValueFor(fr), Value());
    }

    /*TODO: merge with call site*/ void BackpropToS(Matrix<ElemType> inputGradientValues, const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& inputFunctionValues, const Matrix<ElemType>& functionValues)
    {
        ElemType v = gradientValues.Get00Element() / (functionValues.Get00Element() + EPS_IN_INVERSE); // TODO: GPU inefficiency
        inputGradientValues.AddWithScaleOf(v, inputFunctionValues);
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        Value().VerifySize(1, 1);
        Value().SetValue(InputRef(0).MaskedValueFor(fr).FrobeniusNorm());
#if NANCHECK
        Value().HasNan("MatrixL2Reg");
#endif
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateUnaryReduce(isFinalValidationPass);
    }
};

template class MatrixL2RegNode<float>;
template class MatrixL2RegNode<double>;

// -----------------------------------------------------------------------
// NoiseContrastiveEstimationNode (labels, input, inputWeights, biasWeights)
//  -labels: label in dense matrix in [4 x T]
//           the first row is the word index, the second row is the class index, the third row is the first word index of the class
//           the last row is the first word index of the next class
//  - input: hidden layer activity to the node in [hdsize x T]. for a simple rnn, this is the hidden layer activty
//  - inputWeights: weight matrix in [hdsize x vocab_size], for speed-up, as per word matrix can be simply obtained as column slice
//  - biasWeights: clsprob in dense matrix in [nbr_cls x T]. this is the output from logsoftmax node for the log-posterior probabilty of class given observations
// */
// BUGBUG: This node has not been converted to memshare conventions.
// -----------------------------------------------------------------------

enum NCEEvalMode
{
    Softmax = 0,
    Unnormalized = 1,
    None = 2
};
template <class ElemType>
class NoiseContrastiveEstimationNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<4>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"NCEBasedCrossEntropyWithSoftmax";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(NoiseContrastiveEstimationNode);
    NoiseContrastiveEstimationNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name),
          m_logSoftmax(deviceId),
          m_softMax(deviceId),
          m_grdToSoftMaxInput(deviceId),
          m_ncePrediction(deviceId),
          m_evalMode(NCEEvalMode::None)
    {
    }
    NoiseContrastiveEstimationNode(DEVICEID_TYPE deviceId, const wstring& name, NCEEvalMode xm_evalMode)
        : Base(deviceId, name),
          m_logSoftmax(deviceId),
          m_softMax(deviceId),
          m_grdToSoftMaxInput(deviceId),
          m_ncePrediction(deviceId),
          m_evalMode(xm_evalMode)
    {
    }
    // ^^ TODO: we can merge these two

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_evalMode;
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        fstream >> m_evalMode;
        if (m_evalMode > NCEEvalMode::None)
        {
            m_evalMode = NCEEvalMode::None;
            fstream.SetPosition(fstream.GetPosition() - sizeof(m_evalMode));
        }
    }

    void SetEvalMode(NCEEvalMode& xevMode)
    {
        m_evalMode = xevMode;
    }
    NCEEvalMode& EvalMode()
    {
        return m_evalMode;
    } // TODO: really? Return a reference to a local? TODO: change to const? and call it GetEvalMode()

    /**
        compute gradients to input observations, the weights to the observations, and the class log posterior probabilities
        */
    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        m_needRecomputeGradientToSoftmaxInput = false;
        // gradient computation@yinggongzhao
        // inputIndex should be 2 this time
        if (m_evalMode != NCEEvalMode::None)
            LogicError("BackpropTo should only be called in training mode");
        if (inputIndex == 0)
            InvalidArgument("ComputeInput partial should not be called for label");
        //                                                                              samples+probs                   hidden                  embedding
        // InputRef(inputIndex).GradientFor(fr).AssignNCEDerivative(m_ncePrediction, InputRef(0).ValueFor(fr), InputRef(1).ValueFor(fr), InputRef(2).Value(), inputIndex);
        if (inputIndex >= 2)
            InputRef(inputIndex).Gradient().AssignNCEDerivative(m_ncePrediction, InputRef(0).ValueFor(fr), InputRef(1).ValueFor(fr), InputRef(2).ValueAsMatrix(), inputIndex);
        else
            InputRef(inputIndex).GradientFor(fr).AssignNCEDerivative(m_ncePrediction, InputRef(0).ValueFor(fr), InputRef(1).ValueFor(fr), InputRef(2).ValueAsMatrix(), inputIndex);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

    virtual void UpdateFunctionMBSize() override
    {
        // TODO (this does not really break it since for full matrices, class Matrix will resize by itself)
    }

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override // -sum(left_i * log(softmax_i(right)))
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        if (InputRef(0).HasMBLayout() && InputRef(0).GetMBLayout()->HasGaps())
            LogicError("%ls %ls operation does not handle multiple parallel sequences with gaps correctly. Contact fseide@microsoft.com if you have a need and a test case.", NodeName().c_str(), OperationName().c_str());

        int positive = 0, negative = 0;
        if (InputRef(0).GetSampleLayout().GetNumElements() == 1)
        {
            for (int i = 0; i < InputRef(0).Value().GetNumCols(); i++) // BUGBUG: Loops must be over frames, not columns. Columns may contain gaps.
            {
                if (InputRef(0).Value()(0, i) > 0)
                    positive++;
                else if (InputRef(0).Value()(0, i) < 0)
                    negative++;
            }
            assert(positive * negative == 0);
        }
        if (m_evalMode == NCEEvalMode::Softmax || (InputRef(0).GetSampleLayout().GetNumElements() == 1 && positive > 0))
        {
            // evaluation uses softmax
            m_logSoftmax.AssignProductOf(InputRef(1).Value(), true, InputRef(2).ValueAsMatrix(), false);
            m_logSoftmax += InputRef(3).Value();
            m_logSoftmax.InplaceLogSoftmax(false);
            MaskMissingColumnsToZero(m_logSoftmax, InputRef(1).GetMBLayout(), fr); // TODO: is this the right way to neutralize gaps?
            Value().AssignSoftmaxSum(InputRef(0).Value(), m_logSoftmax);
        }
        else if (m_evalMode == NCEEvalMode::Unnormalized || (InputRef(0).GetSampleLayout().GetNumElements() == 1 && negative > 0))
        {
            // TODO: are we treating gaps correctly here?
            Value().AssignNceUnnormalizedEval(InputRef(0).Value(), InputRef(1).Value(), InputRef(2).ValueAsMatrix(), InputRef(3).Value());
        }
        else
        {
            // TODO: are we treating gaps correctly here?
            // training criterion uses NCE
            // likelihood                                         samples+probs                        hidden                       embedding            bias
            Value().AssignNoiseContrastiveEstimation(InputRef(0).Value(), InputRef(1).Value(), InputRef(2).ValueAsMatrix(), InputRef(3).Value(), m_ncePrediction);
        }
        m_needRecomputeGradientToSoftmaxInput = true;
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        m_pMBLayout = nullptr; // this node does not hold mini-batch data

        if (isFinalValidationPass)
        {
            if (Input(1)->GetSampleMatrixNumRows() != Input(2)->GetAsMatrixNumRows())
                LogicError("The Matrix dimension for observation and weight in the NoiseContrastiveEstimationNode operation does not match.");
            if (!Input(0)->HasMBLayout() || !Input(1)->HasMBLayout() || Input(2)->HasMBLayout() || !Input(3)->HasMBLayout())
                LogicError("%ls %ls operation requires inputs 0, 1, and 3 to be a minibatch, and input 2 to be a matrix.", NodeName().c_str(), OperationName().c_str());
        }

        SetDims(TensorShape(1), false);
    }

protected:
    Matrix<ElemType> m_logSoftmax;
    Matrix<ElemType> m_softMax;
    Matrix<ElemType> m_ncePrediction;

    // gradient of cross entropy with respect to the input of softmax
    // a 1 row by \sum_t m_nbrWordsInEachTime[t] vector
    // one slice of size m_nbrWordsInEachTime[t] saves the input to softmax for word y_t
    Matrix<ElemType> m_grdToSoftMaxInput;
    bool m_needRecomputeGradientToSoftmaxInput;

    size_t m_nbrNoise;
    size_t m_totalNbrWords;

private:
    NCEEvalMode m_evalMode;
};
template class NoiseContrastiveEstimationNode<float>;
template class NoiseContrastiveEstimationNode<double>;



// Nodes using a random number generators should derive from this interface.
// One purpose of this interface is to have a common interface for setting the seeds when setting up a network.
class IRngUser
{
public:
    virtual RNGHandle& GetRNGHandle(DEVICEID_TYPE deviceId) = 0;
    virtual void SetRngState(uint64_t seed, uint64_t offset = 0) = 0;
};

// This implements IRngUser using RNGHandle.
class RngUser : public IRngUser
{
public:
    RNGHandle& GetRNGHandle(DEVICEID_TYPE deviceId) override
    {
        if (!m_RNGHandle)
            m_RNGHandle = RNGHandle::Create(deviceId, m_rngSeed, m_rngOffset);

        return *m_RNGHandle;
    }

    // E.g. called from ComputationNetwork to make sure that CNTK running on different nodes will have different seed.
    void SetRngState(uint64_t seed, uint64_t offset = 0) override
    {
        m_rngSeed = seed;
        m_rngOffset = offset;
        m_RNGHandle.reset(); // Reset handle. New handle will be generated with next call of GetRNGHandle(...).
    }

    uint64_t GetRngSeed() const
    {
        return m_rngSeed;
    }

    uint64_t GetRngOffset() const
    {
        return m_rngOffset;
    }

    void UpdateRngOffset(uint64_t val)
    {
        m_rngOffset = val;
    }

protected:

    void Load(File& fstream, size_t modelVersion)
    {
        if (modelVersion < CNTK_MODEL_VERSION_16)
            return;

        uint64_t seed;
        uint64_t offset;

        if (modelVersion == CNTK_MODEL_VERSION_16)
        {
            unsigned long seed_16;
            fstream >> seed_16;
            seed = seed_16;
        }
        else 
        {
            fstream >> seed;
        }

        fstream >> offset;
        SetRngState(seed, offset);  
    }

    void Save(File& fstream) const
    {
        fstream << GetRngSeed();
        fstream << GetRngOffset();
    }

    uint64_t m_rngSeed = 0;
    uint64_t m_rngOffset = 0;
    std::shared_ptr<RNGHandle> m_RNGHandle;
};

// ------------------------------------------------------------------------------------------------------------------------------------------------
// RandomSampleNodeBase(samplingWeights, sizeOfSampledSet, allowDuplicates): 
// Base class for RandomSampleNode and RandomSampleInclusionFrequencyNode.
// Provides random sampling functionality.
//
// Parameters:
// * Input(0) Sampling weight vector: Matrix of shape [numClasses x 1] providing sampling weights >= 0.
// * sizeOfSampledSet: Size of the sampled set.
// * allowDuplicates: controls if sampled set is allowed to contain duplicates.
// --------------------------------------------------------------------------------------------------------------------------------------------------

template <class ElemType>
class RandomSampleNodeBase : public ComputationNodeNonLooping<ElemType>, public NumInputs<1>, public RngUser
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName(){return L"RandomSampleNodeBase";}

public:
    RandomSampleNodeBase(DEVICEID_TYPE deviceId, const wstring& name, size_t sizeOfSampledSet = 0, bool allowDuplicates = false)
        : Base(deviceId, name), m_sizeOfSampledSet(sizeOfSampledSet), m_allowDuplicates(allowDuplicates)
    {
        SetRngState(CreateUniqId());
    }

    RandomSampleNodeBase(const ScriptableObjects::IConfigRecordPtr configp)
        : RandomSampleNodeBase(CPUDEVICE, L"<placeholder>", configp->Get(L"sizeOfSampledSet"), configp->Get(L"allowDuplicates"))
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override;

    virtual void Save(File& fstream) const override;
    virtual void Load(File& fstream, size_t modelVersion) override;

protected:

    void UpdateWeightsPrefixSum();

    // Runs the sampling returning a vector with the id's of the samples. The parameter nTries is used to return the number of draws that was needed
    // to get the expected number of samples.
    const std::vector<size_t> RunSampling(size_t& nTries);

public:
    virtual void /*ComputationNode::*/ BackpropToNonLooping(size_t inputIndex) override {} // This node does not propagate gradients.
    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override;
    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false;}
    virtual void /*ComputationNode::*/ ForwardPropNonLooping() override{}
    virtual bool GetAllowDuplicates() const { return m_allowDuplicates; }
    virtual size_t GetNumSamples() const { return m_sizeOfSampledSet; }

protected:
    bool m_allowDuplicates; // The node can create samples allowing for duplicates (sampling with replacement) or not (sampling without replacement).
    size_t m_sizeOfSampledSet; // Requested size of sample in case of run-mode = CREATE_SAMPLES.
    std::vector<double> m_samplingWeightsPrefixSum;
};

// ------------------------------------------------------------------------------------------------------------------------------------------------
// RandomSampleNode(samplingWeights, sizeOfSampledSet, allowDuplicates):
// The node's value is a set of sizeOfSampledSet random samples represented as a (sparse) matrix 
// of shape [numClasses x sizeOfSampledSet] where numClasses is the number of classes (categories) to choose from.
// The output has no dynamic axis.
// The samples are drawn with a probability proportional to the weights w of the vector 'samplingWeights' : p(w_i) = w_i / sum_k(w_k)
// We get one set of samples for per minibatch.
// Multiply a 'numClasses' - dimensional vector with this matrix to randomly sample 'sizeOfSampledSet' values from it.
// The resulting vector has a dimension of 'sizeOfSampledSet'.Currently, only rank - 1 tensors are supported.
// Intended uses are e.g. sampled softmax, noise contrastive estimation etc.
//
// Parameters:
// * Input(0): Sampling weight vector. Matrix of shape [numClasses x 1] providing sampling weights >= 0.
// * sizeOfSampledSet: Size of the sampled set.
// * allowDuplicates: controls if sampled set is allowed to contain duplicates.
// --------------------------------------------------------------------------------------------------------------------------------------------------
template<class ElemType> 
class RandomSampleNode : public RandomSampleNodeBase<ElemType>
{
    typedef RandomSampleNodeBase<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName(){ return L"RandomSample"; }

public:
    RandomSampleNode(DEVICEID_TYPE deviceId, const wstring& name, size_t sizeOfSampledSet = 0, bool allowDuplicates = false)
        : Base(deviceId, name, sizeOfSampledSet, allowDuplicates)
    {}

    RandomSampleNode(const ScriptableObjects::IConfigRecordPtr configp)
        : RandomSampleNode(CPUDEVICE, L"<placeholder>", configp->Get(L"sizeOfSampledSet"), configp->Get(L"allowDuplicates"))
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }

    virtual void /*ComputationNode::*/ ForwardPropNonLooping() override;
    const std::vector<size_t> GetWeightedSamples();
    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override;
    virtual bool IsOutOfDateWrtInputs() const override;
};

// ------------------------------------------------------------------------------------------------------------------------------------------------
// RandomSampleInclusionFrequencyNode(samplingWeights, sizeOfSampledSet, allowDuplicates): 
// Intended uses are e.g. sampled softmax, noise contrastive estimation etc. where it is used together with RandomSampleNode.
// This node estimates how often each class will occur in a set sampled with RandomSampleNode(...) on the average. 
// If the sampling mode 'allowDuplicates = true' is choosen this is trivial and exact. 
// For allowDuplicates = false we get some estimate. The value is updated only when the input weights change.
//
// Parameters:
// * Input(0): Sampling weight vector. Matrix of shape (numClasses x 1) providing sampling weights >= 0.
// * sizeOfSampledSet: Size of the sampled set.
// * allowDuplicates: controls if sampled set is allowed to contain duplicates.
// --------------------------------------------------------------------------------------------------------------------------------------------------
template<class ElemType>
class RandomSampleInclusionFrequencyNode : public RandomSampleNodeBase<ElemType>
{
    typedef RandomSampleNodeBase<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName(){ return L"RandomSampleInclusionFrequency"; }
public:
    RandomSampleInclusionFrequencyNode(DEVICEID_TYPE deviceId, const wstring& name, size_t sizeOfSampledSet = 0, bool allowDuplicates = false)
        : Base(deviceId, name, sizeOfSampledSet, allowDuplicates)
    {}

    RandomSampleInclusionFrequencyNode(const ScriptableObjects::IConfigRecordPtr configp)
        : RandomSampleInclusionFrequencyNode(CPUDEVICE, L"<placeholder>", configp->Get(L"sizeOfSampledSet"), configp->Get(L"allowDuplicates"))
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }
    virtual void /*ComputationNode::*/ ForwardPropNonLooping() override;
    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override;
private:
    // Approximates the expected number of occurences of a class in the sampled set.
    // Assuming (falsely) that the number of tries to get a sampled set with the requested number of distinct values is always estimatedNumTries
    // the probability that a specific class in in the sampled set is (1 - (1-p)^estimatedNumTries), where p is the probablity to pick the clas in one draw.
    // The estimate can be quite a bit off but should be better than nothing. Better alternatives?
    double EstimateInSampleFrequency(double p, double estimatedNumTries) const;

    double EstimateNumberOfTries();
};

// -----------------------------------------------------------------------
// ClassBasedCrossEntropyWithSoftmaxNode (labeldata(.,t), inputdata(.,t), embeddingMatrix, clsProbBeforeSoftmaxData(.,t))
//  - Input(0) [4 x T] label in dense matrix in
//              (0,t) the first row is the word index
//              (1,t) the second row is the class index
//              (2,t) the third row is the first word index of the class
//              (3,t) the last row is the first word index of the next class
//  - Input(1) [hdsize x T] hidden layer activation to the node in. for a simple rnn, this is the hidden layer activty
//  - Input(2) [hdsize x vocab_size] weight matrix in, for speed-up, as per word matrix can be simply obtained as column slice
//  - Input(3) [nbr_cls x T] clsprob in dense matrix in. This input, if applied softmax on, is the posterior probabilty of class given observations
// -----------------------------------------------------------------------
// calculates: -sum(left_i * log(softmax_i(right))) for class given history and for word given history
// need to provide class probabilty from external node
template <class ElemType>
class ClassBasedCrossEntropyWithSoftmaxNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<4>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"ClassBasedCrossEntropyWithSoftmax"; }

    // our inputs
    static const size_t LABELDATA = 0;
    static const size_t INPUTDATA = 1;
    static const size_t EMBEDDINGMATRIX = 2;
    static const size_t CLASSPROBINDATA = 3;

public:
    DeclareConstructorFromConfigWithNumInputs(ClassBasedCrossEntropyWithSoftmaxNode);
    ClassBasedCrossEntropyWithSoftmaxNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name),
          m_logSoftmax(deviceId),
          m_softMax(deviceId),
          m_grdToSoftMaxInput(deviceId),
          m_clsLogSoftmax(deviceId),
          m_clsSoftmax(deviceId)
    {
    }

private:
    // iterate over a large workspace that contains all class-conditioned probs concatenated
    // 'sz' is the offset into that vector. We will iterate over these vectors at a few places. Always use this same boilerplate code.
    template<class F>
    size_t ForColumnsWithClass(const F& op)
    {
        const size_t nT = Input(LABELDATA)->GetNumTimeSteps();
        const size_t nS = Input(LABELDATA)->GetNumParallelSequences();
        size_t sz = 0; // iterate over the packed concatenated class-conditioned prob vectors
        for (size_t s = 0; s < nS; s++)
            for (size_t t = 0; t < nT; t++)
            {
                FrameRange fr = FrameRange(Input(LABELDATA)->GetMBLayout(), t).Sequence(s);
                if (Input(LABELDATA)->GetMBLayout()->IsGap(fr)) // skip gaps
                    continue;

                const Matrix<ElemType>& lbl_t = Input(LABELDATA)->ValueFor(fr);
                size_t y_t = (size_t)lbl_t(0, 0);     // current word token index
                size_t c_t = (size_t)lbl_t(1, 0);     // current word token's class index
                size_t lft_bnd = (size_t)lbl_t(2, 0); // index of first word belonging to current word token's class
                size_t rgt_bnd = (size_t)lbl_t(3, 0); // and end of that range
                size_t nbr_wrd = (rgt_bnd - lft_bnd); // number of words in the class

                // perform the operation
                op(s, t, fr, y_t, c_t, sz, lft_bnd, nbr_wrd);

                sz += nbr_wrd;
            }
        return sz;
    }

    // compute gradients to input observations, the weights to the observations, and the class log posterior probabilites
    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        // this should never be called for input[0], which is controlled through learningRateMultiplier == 0
        if (inputIndex != 1 && inputIndex != 2 && inputIndex != 3)
            InvalidArgument("ClassCrossEntropyWithSoftmaxNode criterion only takes with respect to input, weight to the input and class log posterior probability.");

        ComputeSoftMaxPartial(); // Note: Flag m_needRecomputeGradientToSoftmaxInput guards so that this computes only once.

        ForColumnsWithClass([&](size_t /*s*/, size_t /*t*/, const FrameRange& fr, size_t /*y_t*/, size_t c_t, size_t sz, size_t lft_bnd, size_t nbr_wrd)
        {
            // compute prb - 1 and prb
            Matrix<ElemType> weightForClass = InputRef(EMBEDDINGMATRIX).ValueAsMatrix().ColumnSlice(lft_bnd, nbr_wrd);
            Matrix<ElemType> obs = InputRef(INPUTDATA).ValueFor(fr); // hidden activation vector for current word token
            Matrix<ElemType> grd_to_soft_max_input = m_grdToSoftMaxInput.ColumnSlice(sz, nbr_wrd);

            switch (inputIndex)
            {
                case 1:
                {
                    // gradient to input
                    Matrix<ElemType> grd_t = InputRef(INPUTDATA).GradientFor(fr);
                    Matrix<ElemType>::MultiplyAndAdd(weightForClass, false, grd_to_soft_max_input, true, grd_t);
                    break;
                }
                case 2:
                {
                    // gradient to input weight
                    Matrix<ElemType> grd_to_wgt_t = InputRef(EMBEDDINGMATRIX).GradientAsMatrix().ColumnSlice(lft_bnd, nbr_wrd);
                    Matrix<ElemType>::MultiplyAndAdd(obs, false, grd_to_soft_max_input, false, grd_to_wgt_t);
                    break;
                }
                case 3:
                {
                    Matrix<ElemType> grd_t = InputRef(CLASSPROBINDATA).GradientFor(fr);
                    grd_t.AssignValuesOf(InputRef(CLASSPROBINDATA).DataFor(m_clsSoftmax, fr));
                    ComputeCEPartialToSoftmaxInputs(grd_t, Gradient(), c_t);
                    break;
                }
            }
        });
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }

private:
    void ComputeCEPartialToSoftmaxInputs(Matrix<ElemType>& inputGradientValues, Matrix<ElemType>& gradientValues, size_t y_t)
    {
        Matrix<ElemType>::MinusOneAt(inputGradientValues, y_t);
        Matrix<ElemType>::Scale(gradientValues, inputGradientValues);
    }

    // gradient of cross entropy w.r.t. to input to softmax
    void ComputeSoftMaxPartial()
    {
        if (m_needRecomputeGradientToSoftmaxInput)
        {
            m_grdToSoftMaxInput.Resize(1, m_totalNbrWords); // buffer that contains a concatenation of class-conditional values

            ForColumnsWithClass([&](size_t /*s*/, size_t /*t*/, const FrameRange& /*fr*/, size_t y_t, size_t /*c_t*/, size_t sz, size_t lft_bnd, size_t nbr_wrd)
            {
                Matrix<ElemType> softMax = m_softMax.ColumnSlice(sz, nbr_wrd);

                size_t idx_in_class = y_t - lft_bnd;
                ComputeCEPartialToSoftmaxInputs(softMax, Gradient(), idx_in_class);

                m_grdToSoftMaxInput.ColumnSlice(sz, nbr_wrd).AssignValuesOf(softMax);
            });

            m_needRecomputeGradientToSoftmaxInput = false;
        }
    }

public:
    virtual void UpdateFunctionMBSize() override
    {
        // TODO: Resize temp matrices here (not doing so does not really fail since for full matrices, class Matrix will resize by itself)
    }

    // -sum(left_i * log(softmax_i(right)))
    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        // get the label matrix to CPU, ideally in location=BOTH state
        InputRef(LABELDATA).Value().TransferToDeviceIfNotThere(CPUDEVICE, /*ismoved =*/ false/*means: BOTH state OK*/, /*emptyTransfer =*/ false, /*updatePreferredDevice =*/ false);

        auto& functionValues = Value();

        const size_t hdSize = InputRef(INPUTDATA).GetSampleMatrixNumRows();
        assert(m_nbrCls == InputRef(CLASSPROBINDATA).GetSampleMatrixNumRows());

        // compute the class posteriors
        m_clsLogSoftmax.SetValue(InputRef(CLASSPROBINDATA).Value());
        m_clsLogSoftmax.InplaceLogSoftmax(true);   // log
        m_clsSoftmax.AssignExpOf(m_clsLogSoftmax); // non-log

        // create a large workspace to contain all class-conditioned probs concatenated
        m_totalNbrWords = ForColumnsWithClass([](size_t /*s*/, size_t /*t*/, const FrameRange& /*fr*/, size_t y_t, size_t /*c_t*/, size_t /*sz*/, size_t lft_bnd, size_t nbr_wrd)
        {
            if (nbr_wrd == 0)
                LogicError("ClassBasedCrossEntropyWithSoftmax: Encountered a class of size 0.");
            if (y_t < lft_bnd || y_t >= lft_bnd + nbr_wrd)
                LogicError("ClassBasedCrossEntropyWithSoftmax: Word index out of bounds of class-member index range (word not a class member).");
        });
        // now m_totalNbrWords = total size of concatenated vector

        // buffer to hold the concatenated class-conditioned prob vectors
        m_softMax.Resize(1, m_totalNbrWords);
        m_logSoftmax.Resize(1, m_totalNbrWords);

        // accumulate objective
        functionValues.SetValue(0);
        ForColumnsWithClass([&](size_t s, size_t t, const FrameRange& fr, size_t y_t, size_t c_t, size_t sz, size_t lft_bnd, size_t nbr_wrd)
        {
            // now get views of various arrays that correspond to the index range of words belonging to this class

            // get hidden vectors for the words in this class
            Matrix<ElemType> weightForClass = InputRef(EMBEDDINGMATRIX).ValueAsMatrix().ColumnSlice(lft_bnd, nbr_wrd); // [hdSize x nbr_wrd]

            // buffer to hold the class-conditional distribution
            Matrix<ElemType> softMax_t = m_softMax.ColumnSlice(sz, nbr_wrd); // TODO: declare these outside of the loop to avoid the malloc
            Matrix<ElemType> logSoftMax_t = m_logSoftmax.ColumnSlice(sz, nbr_wrd);

            Matrix<ElemType> obs = InputRef(INPUTDATA).ValueFor(fr); // hidden activation vector for current word token

            // multiply hidden activation with weight matrix (the slice of the weight matrix for the range of class members)
            // TODO: can we use 'true' here instead? Above transposition hack won't work with row slices. 'obs' not used elsewhere
            obs.Reshape(1, hdSize);                                                                                // transpose it (make it a column vector)
            logSoftMax_t.AssignProductOf(obs /*(1 x hdSize)*/, false, weightForClass /*hdSize x nbr_wrd*/, false); // -> 1 x nbr_word

            // log softmax(W x_t)
            logSoftMax_t.InplaceLogSoftmax(false);

            // and non-log version
            softMax_t.SetValue(logSoftMax_t);
            softMax_t.InplaceExp();
            // we now have a column vector of class-conditional probabilities over the class members

            // add  the word's class-conditional log posterior
            size_t idx_in_class = y_t - lft_bnd;
            Matrix<ElemType>::AddElementToElement(logSoftMax_t, 0, idx_in_class, functionValues, 0, 0); // (1x1)

            // add the class log posterior probability (for backprop)
            auto clsLogSoftmax_t = InputRef(CLASSPROBINDATA).DataFor(m_clsLogSoftmax, fr);
            Matrix<ElemType>::AddElementToElement(clsLogSoftmax_t, c_t, 0, functionValues, 0, 0); // (1x1)
        });

        functionValues *= (-1);

#if NANCHECK
        functionValues.HasNan("ClassBasedCrossEntropyWithSoftmax");
#endif
        m_needRecomputeGradientToSoftmaxInput = true;
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        m_pMBLayout = nullptr; // this node does not hold mini-batch data

        if (isFinalValidationPass)
        {
            if (Input(LABELDATA)->GetSampleMatrixNumRows() != 4) // label data needs to have 4 rows
                LogicError("The label data in the ClassBasedCrossEntropyWithSoftmax operation must have 4 rows.");
            if (Input(INPUTDATA)->GetSampleMatrixNumRows() != Input(EMBEDDINGMATRIX)->GetAsMatrixNumRows()) // input and matrix can be timed
                LogicError("The matrix dimension for observation and weight in the ClassBasedCrossEntropyWithSoftmax operation does not match.");
            if (Input(LABELDATA)->GetMBLayout() != Input(INPUTDATA)->GetMBLayout() || Input(LABELDATA)->GetMBLayout() != Input(CLASSPROBINDATA)->GetMBLayout())
                InvalidArgument("%ls %ls operation requires that the layouts of inputs 0 (label), 1 (hidden activation), and 3 (log softmax) match.", NodeName().c_str(), OperationName().c_str());
        }

        SetDims(TensorShape(1), false);

        m_nbrCls = Input(CLASSPROBINDATA)->GetSampleMatrixNumRows();
    }

protected:
    Matrix<ElemType> m_logSoftmax;
    Matrix<ElemType> m_softMax;

    Matrix<ElemType> m_clsLogSoftmax;
    Matrix<ElemType> m_clsSoftmax;

    // gradient of cross entropy with respect to the input of softmax
    // a 1 row by \sum_t m_nbrWordsInEachTime[t] vector
    // one slice of size m_nbrWordsInEachTime[t] saves the input to softmax for word y_t
    Matrix<ElemType> m_grdToSoftMaxInput;
    bool m_needRecomputeGradientToSoftmaxInput;

    size_t m_nbrCls;
    size_t m_totalNbrWords;
};

template class ClassBasedCrossEntropyWithSoftmaxNode<float>;
template class ClassBasedCrossEntropyWithSoftmaxNode<double>;

#ifdef COMING_SOON

// -----------------------------------------------------------------------
// CRFNode (labels, position_dependent_scores, transition_scores)
//  - labels: output label vector of [0:T-1]
//  - position_dependent_scores [0:T-1]: score from position dependent node,
//    in the R-CRF case, it is the RNN output score before softmax
//  - transition scores: square transition matrix,  --TODO: log?
//    in the R-CRF case, it is the transition probability between labels
// BUGBUG: This node cannot operate with truncated BPTT, but does not detect it. It also does not handle gaps or test boundary flags.
// -----------------------------------------------------------------------

/**
        CRF training criterion
        It uses forward-backward algorithm within a minibatch to compute statistics for sequence level optimization
        This node can serve a base class for other sequence level optimization

        Developed by Kaisheng Yao
        This node is for replicating results of the following work
        K. Yao, B. Peng, G. Zweig, D. Yu, X. Li and F. Gao, "Recurrent Conditional Random Fields", NIPS Deep Learning Workshop 2014
        K. Yao, B. Peng, G. Zweig, D. Yu, X. Li and F. Gao, "Recurrent Conditional Random Fields for Language Understanding", ICASSP 2014
        http://research.microsoft.com/pubs/210167/rcrf_v9.pdf

        The forward-backward algorithm follows the derivation in
        http://jmlr.org/papers/volume12/collobert11a/collobert11a.pdf

    */
template <class ElemType>
class CRFNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<3>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"CRF";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(CRFNode);
    CRFNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name),
          mAlpha(deviceId),
          mBeta(deviceId),
          mPostProb(deviceId)
    {
    }

    // compute posterior probability of label y at position t
    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        size_t nrow = InputRef(0).Value().GetNumRows();
        size_t ncol = InputRef(0).Value().GetNumCols();

        mAlpha.Resize(nrow, ncol);
        mBeta.Resize(nrow, ncol);
        mPostProb.Resize(nrow, ncol);

        Value().SetValue(0.0);
        Matrix<ElemType> funcVal = Value(); // TODO: This just creates a 1x1 matrix set to 0.

        size_t nS = InputRef(0).GetNumParallelSequences();
        if (nS != 1)
            LogicError("CRFNode: >1 parallel sequences are curently not implemented correctly.");
        for (size_t i = 0; i < nS; i++) // process parallel sequences one by one  --BUGBUG: We should loop over individual sequences.
        {
            FrameRange sequenceRange = fr.Sequence(i); // FrameRange to select one sequence
            // BUGBUG: This ^^ is neither supported nor correct, since this code does not handle gaps or start/end flags.
            ForwardPropS(
                DataWithMBLayoutFor(mPostProb, sequenceRange, InputRef(0).GetMBLayout()),
                DataWithMBLayoutFor(mAlpha, sequenceRange, InputRef(0).GetMBLayout()),
                DataWithMBLayoutFor(mBeta, sequenceRange, InputRef(0).GetMBLayout()),
                funcVal,
                InputRef(0).ValueFor(sequenceRange),
                InputRef(1).ValueFor(sequenceRange),
                InputRef(2).ValueAsMatrix(), mStartLbl,
                mEndLbl);

            Value() += funcVal; // aggregate over sequences
        }
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override // scaled by 2*number of colmns (samples) in the Matrix<ElemType>
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        // this should never be called for input[0], which is controlled through learningRateMultiplier == 0
        if (inputIndex != 1 && inputIndex != 2)
            InvalidArgument("CRFNode only takes with respect to input and weight.");

        if (inputIndex == 1)
        {
            auto gradient = InputRef(1).GradientFor(fr);
            Matrix<ElemType>::AddScaledDifference(Gradient(), mPostProb, InputRef(0).ValueFor(fr), gradient);
        }
        else if (inputIndex == 2)
        {
            assert(InputRef(inputIndex).GradientFor(fr).GetNumElements() > 0);
            size_t nS = InputRef(0).GetNumParallelSequences();
            for (size_t i = 0; i < nS; i++) // process all sequences one by one
            {
                FrameRange sequenceRange = fr.Sequence(i); // FrameRange to select one sequence
                auto& gradient = InputRef(2).GradientAsMatrix();
                TransGrdCompute(InputRef(0).ValueFor(sequenceRange),
                                DataWithMBLayoutFor(mAlpha, sequenceRange, InputRef(0).GetMBLayout()),
                                DataWithMBLayoutFor(mBeta, sequenceRange, InputRef(0).GetMBLayout()),
                                InputRef(2).ValueAsMatrix(),
                                gradient,
                                mStartLbl, 1);
            }
        }
        else
            return;
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

    // compute forward backward algorithm
    /*TODO: merge with call site*/ void ForwardPropS(Matrix<ElemType> postprob, Matrix<ElemType> alpha, Matrix<ElemType> beta, Matrix<ElemType>& functionValues, const Matrix<ElemType>& lbls, const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, int& firstLbl, int& lastLbl, const int iStep = 1)
    {
        // to-do, each slice is for one sentence
        // to-do, number of slices correspond to number of frames
        // this implementation only supports one sentence per minibatch

        int nObs = lbls.GetNumCols();

        // change to other values so can support multiple sentences in each minibatch
        assert(iStep == 1);
        ForwardCompute(alpha, lbls, pos_scores, pair_scores);
        BackwardCompute(alpha, beta, functionValues, lbls, pos_scores, pair_scores, iStep);
        PostProbCompute(postprob, alpha, beta);

        firstLbl = -1;
        for (int ik = 0; ik < lbls.GetNumRows(); ik++)
            if (lbls(ik, 0) != 0)
            {
                firstLbl = ik;
                break;
            }

        lastLbl = -1;
        for (int ik = 0; ik < lbls.GetNumRows(); ik++)
            if (lbls(ik, nObs - 1) != 0)
            {
                lastLbl = ik;
                break;
            }

        functionValues.AssignInnerProductOfMatrices(lbls, pos_scores);

        Matrix<ElemType> a = alpha.ColumnSlice(nObs - 1, 1);
        ElemType fAlpha;
        fAlpha = a.LogSumOfElements();

        // transition score
        ElemType tscore = 0;
        for (int t = 0; t < nObs - 1; t++)
        {
            int i = -1;
            for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                if (lbls(ik, t) != 0)
                {
                    i = ik;
                    break;
                }
            int j = -1;
            for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                if (lbls(ik, t + 1) != 0)
                {
                    j = ik;
                    break;
                }
            tscore += pair_scores(j, i);
        }
        tscore += functionValues.Get00Element(); // correct path score
        tscore -= fAlpha;                        // reduced by the scores from all paths
        functionValues.SetValue(tscore);

        functionValues *= (-1);
    }

    // compute forward backward algorithm
    static void ForwardCompute(Matrix<ElemType>& alpha,
                               const Matrix<ElemType>& lbls,
                               const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores)
    {
        // to-do, shift more than 1 to support muliple sentences per minibatch
        int iNumPos = lbls.GetNumCols();
        int iNumLab = lbls.GetNumRows();

        int firstLbl = -1;
        for (int ik = 0; ik < lbls.GetNumRows(); ik++)
            if (lbls(ik, 0) != 0)
            {
                firstLbl = ik;
                break;
            }

        // need to have
        alpha.Resize(iNumLab, iNumPos);

        for (int t = 0; t < iNumPos; t++)
        {
            for (int k = 0; k < iNumLab; k++)
            {
                ElemType fTmp = (ElemType) LZERO;
                for (int j = 0; j < iNumLab; j++)
                {
                    ElemType fAlpha = (j == firstLbl) ? (ElemType) 0.0 : (ElemType) LZERO;
                    if (t > 0)
                        fAlpha = alpha(j, t - 1);
                    fTmp = alpha.LogAdd(fTmp, fAlpha + pair_scores(k, j));
                }
                fTmp += pos_scores(k, t); // include position dependent score
                alpha(k, t) = fTmp;
            }
        }
    }

    // compute backward algorithm
    static void BackwardCompute(const Matrix<ElemType>& alpha, Matrix<ElemType>& beta,
                                Matrix<ElemType>& functionValues, const Matrix<ElemType>& lbls,
                                const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, const int shift = 1)
    {
        assert(shift == 1);

        alpha.RCRFBackwardCompute(alpha, beta, functionValues, lbls, pos_scores, pair_scores, shift);
    }

    static void TransGrdCompute(const Matrix<ElemType>& lbls,
                                const Matrix<ElemType>& alpha,
                                const Matrix<ElemType>& beta,
                                const Matrix<ElemType>& pair_scores,
                                Matrix<ElemType>& grd,
                                const int startLbl,
                                const int shift = 1)
    {
        assert(shift == 1);

        alpha.RCRFTransGrdCompute(lbls,
                                  alpha,
                                  beta,
                                  pair_scores,
                                  grd,
                                  startLbl, shift);
    }

    // compute forward backward algorithm
    static void PostProbCompute(Matrix<ElemType>& postprob, const Matrix<ElemType>& alpha, const Matrix<ElemType>& beta)
    {
        int iNumPos = alpha.GetNumCols();
        int iNumLab = alpha.GetNumRows();

        postprob.Resize(iNumLab, iNumPos);
        postprob.SetValue(beta);
        postprob.InplaceExp();
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);
        m_pMBLayout = nullptr; // this node does not hold mini-batch data

        if (isFinalValidationPass)
            if (!(InputRef(1).GetSampleMatrixNumRows() == InputRef(2).GetAsMatrixNumRows() && // position dependent and pair scores have same number of labels
                  InputRef(0).GetSampleMatrixNumRows() == InputRef(1).GetSampleMatrixNumRows() &&
                  InputRef(0).HasMBLayout() && InputRef(0).GetMBLayout() == InputRef(1).GetMBLayout() &&
                  // InputRef(0).GetNumCols() == InputRef(1).GetNumCols() && // position dependent and pair scores have the same observation numbers
                  InputRef(2).GetAsMatrixNumCols() == InputRef(2).GetAsMatrixNumRows()))
            {
                LogicError("The Matrix dimension in the CRFNode operation does not match.");
            }

        SetDims(TensorShape(1), false);
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<CRFNode<ElemType>>(nodeP);
            node->mAlpha = mAlpha;
            node->mBeta = mBeta;
            node->mPostProb = mPostProb;

            node->mStartLbl = mStartLbl;
            node->mEndLbl = mEndLbl;
        }
    }

private:
    Matrix<ElemType> mAlpha; // TODO: m_Alpha etc.
    Matrix<ElemType> mBeta;
    Matrix<ElemType> mPostProb;
    int mStartLbl;
    int mEndLbl;
};

#endif

// -----------------------------------------------------------------------
// Logistic (labels, prediction, weight)
// calculates: -sum(left * log(right) + (1-left)*log(1-right)) (optionally * weight)
// -----------------------------------------------------------------------

template <class ElemType>
class LogisticNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"Logistic";
    }

public:
    DeclareConstructorFromConfig(LogisticNode);
    LogisticNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name)
    {
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        FrameRange fr(InputRef(0).GetMBLayout());
        if (inputIndex != 1)
            InvalidArgument("%ls %ls operation cannot compute the gradient for its first inpute.", NodeName().c_str(), OperationName().c_str());

        // BackpropToRight(m_temp, InputRef(0).Value(), InputRef(2).Value(), InputRef(inputIndex).Gradient(), Gradient(), m_classZeroLabels, m_result);
        // Create vector with 1 for class 1, and -1 for class 0
        m_temp->AssignDifferenceOf(InputRef(0).ValueFor(fr), *m_classZeroLabels); // TODO: need a slice for m_classZeroLabels?

        // Multiply the vector by the InputRef(2).Value()
        if (m_inputs.size() == 3)                                            // with weight
            m_temp->AssignElementProductOf(*m_temp, InputRef(2).ValueFor(fr)); // TODO: is Input(2) minibatch data? Confirm

        // divide class by p (class 1) or (1-p) (class 0)
        m_temp->AssignElementDivisionOf(*m_temp, *m_result); // TODO: this is in-place--does this function allow that?

        auto gradient = InputRef(inputIndex).GradientFor(fr);
        Matrix<ElemType>::Multiply1x1AndWeightedAdd(-1.0f, Gradient() /*1x1*/, *m_temp, 1.0f, gradient);
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

    virtual void UpdateFunctionMBSize() override
    {
        m_classZeroLabels->Resize(InputRef(0).Value());
        m_result->Resize(InputRef(0).Value());
        m_temp->Resize(InputRef(0).Value());
        m_sumOfWeights->Resize(Value());
    }

    // -sum(left * log(right) + (1-left)*log(1-right)) (optionally * weight)
    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        FrameRange fr(InputRef(0).GetMBLayout());

        const Matrix<ElemType>& classOneLabels = InputRef(0).ValueFor(fr);
        const Matrix<ElemType>& classOneProbabilities = InputRef(1).ValueFor(fr);
        Matrix<ElemType>& classZeroLabels = *m_classZeroLabels;

        const Matrix<ElemType>& ones = ConstOnes(classOneLabels.GetNumRows(), classOneLabels.GetNumCols(), classOneLabels.GetDeviceId());

        // compute the indices for the class 0 indices
        classZeroLabels.AssignDifferenceOf(ones, classOneLabels);

        /* We're computing result = weight*(y*p + (1-y)*(1-p) = 2*y*p + (1-y) - p) */

        /* First compute result = y*p */
        m_result->AssignElementProductOf(classOneLabels, classOneProbabilities);

        // TODO: verify that all these operations on m_result really can do in-place (or use different methods instead)
        /* Now compute result = 2*y*p */
        m_result->AssignProductOf((ElemType) 2.0, *m_result);

        /* Now compute result = 2*y*p + (1-y) */
        m_result->AssignSumOf(*m_result, classZeroLabels);

        /* Finally compute result = 2*y*p + (1-y) - p */
        m_result->AssignDifferenceOf(*m_result, classOneProbabilities);

        // compute the log, resulting in y*log(p) + (1-y)*log(1-p)
        m_temp->AssignLogOf(*m_result);

        // The error is the negative of the sum of the result
        if (m_inputs.size() == 2)
        {
            Value().AssignSumOfElements(*m_temp);
            Value() *= (-1);
        }
        else
        {
            // sum of weights
            m_sumOfWeights->AssignSumOfElements(InputRef(2).ValueFor(fr));
            // number of elements
            ElemType numOfElements = (ElemType)ones.GetNumCols();
            // sum of weighted log loss
            Value().AssignInnerProductOf(InputRef(2).ValueFor(fr), *m_temp, false).ElementDivideBy(*m_sumOfWeights);
            Value() *= -numOfElements;
        }
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        if (m_inputs.size() != 2 && m_inputs.size() != 3)
            InvalidArgument("%ls %ls operation requires two or three inputs.", NodeName().c_str(), OperationName().c_str());

        ValidateBinaryReduce(isFinalValidationPass);

        /* Note that this is the same as ValidateInferBinaryInputDims, but done for the 3rd child if it exists */
        if (m_inputs.size() == 3)
        {
            auto weights = Input(2);
            auto other = Input(1);
            // borrow any unset dimension on one input from the other input
            weights->ValidateInferInputDimsFrom(other->GetSampleLayout());

            if (isFinalValidationPass &&
                !(Input(0)->GetSampleMatrixNumRows() == Input(2)->GetSampleMatrixNumRows() &&
                  (Input(0)->GetMBLayout() == Input(2)->GetMBLayout() || !Input(0)->HasMBLayout() || !Input(2)->HasMBLayout())))
            {
                LogicError("The Matrix dimensions of the second argument weights the %ls %ls operation do not match.", NodeName().c_str(), OperationName().c_str());
            }
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_classZeroLabels, matrixPool);
        RequestMatrixFromPool(m_result, matrixPool);
        RequestMatrixFromPool(m_temp, matrixPool);
        RequestMatrixFromPool(m_sumOfWeights, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_classZeroLabels, matrixPool);
        ReleaseMatrixToPool(m_result, matrixPool);
        ReleaseMatrixToPool(m_temp, matrixPool);
        ReleaseMatrixToPool(m_sumOfWeights, matrixPool);
    }

    virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<LogisticNode<ElemType>>(nodeP);
            node->m_classZeroLabels->SetValue(*m_classZeroLabels);
            node->m_result->SetValue(*m_result);
            node->m_temp->SetValue(*m_temp);
            node->m_sumOfWeights->SetValue(*m_sumOfWeights);
        }
    }

private:
    shared_ptr<Matrix<ElemType>> m_classZeroLabels;
    shared_ptr<Matrix<ElemType>> m_result;
    shared_ptr<Matrix<ElemType>> m_temp;

    // for weighted log-loss
    shared_ptr<Matrix<ElemType>> m_sumOfWeights;
};

template class LogisticNode<float>;
template class LogisticNode<double>;

// -----------------------------------------------------------------------
// DropoutNode (input) -- perform drop-out
// Output is scaled such that no post-scaling is necessary.
// -----------------------------------------------------------------------

template <class ElemType>
class DropoutNode : public ComputationNode<ElemType>, public NumInputs<1>, public RngUser
{
    typedef ComputationNode<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"Dropout";
    }

public:
    DeclareConstructorFromConfigWithNumInputs(DropoutNode);
    DropoutNode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name),
        m_dropoutRate(0)
    {
        SetRngState(CreateUniqId());
    }

    virtual void Save(File& fstream) const override;
    virtual void Load(File& fstream, size_t modelVersion) override;

    virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Grad = InputRef(0).GradientFor(fr);
        Matrix<ElemType> sliceOutputGrad = GradientFor(fr);

        if (m_dropoutRate > 0)
            sliceInput0Grad.AddElementProductOf(sliceOutputGrad, DataFor(*m_maskOfDropout, fr));
        else
            sliceInput0Grad += sliceOutputGrad;
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }
    virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override { return false; }

    virtual void UpdateFunctionMBSize() override
    {
        Base::UpdateFunctionMBSize();
        // resize temporaries to their proper size
        if (m_dropoutRate > 0)
            m_maskOfDropout->Resize(Input(0)->Value());
    }

    virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
    {
        Matrix<ElemType> sliceInput0Value = Input(0)->ValueFor(fr);
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        if (Environment().IsInferring() || m_dropoutRate <= 0)
        {
            sliceOutputValue.SetValue(sliceInput0Value);
        }
        else
        {
            // determine drop-out mask for this minibatch
            auto sliceMask = DataFor(*m_maskOfDropout, fr);
            sliceMask.SetUniformRandomMask((ElemType)m_dropoutRate, (ElemType)(1.0 / (1.0 - m_dropoutRate)) /*pre-scaled*/, GetRNGHandle());
            // apply dropout mask
            sliceOutputValue.AssignElementProductOf(sliceMask, sliceInput0Value);
            UpdateRngOffset(GetRngOffset() + sliceMask.GetNumElements());
        }
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateUnaryMap(isFinalValidationPass);
    }

    // special methods for this node type which ComputationNetwork knows about and calls to pass parameters
    void SetDropoutRate(const double val)
    {
        if (val < 0 || val >= 1)
            LogicError("DropoutRate must be >= 0 and < 1.");
        m_dropoutRate = val;
    }

    RNGHandle& GetRNGHandle()
    {
        return RngUser::GetRNGHandle(ValuePtr()->GetDeviceId());
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<DropoutNode<ElemType>>(nodeP);
            node->m_dropoutRate = m_dropoutRate;
            node->SetRngState(GetRngSeed(), GetRngOffset());
            node->m_maskOfDropout = m_maskOfDropout;
        }
    }
    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_maskOfDropout, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_maskOfDropout, matrixPool);
    }

    double GetDropoutRate() const { return m_dropoutRate; }

private:
    double m_dropoutRate;
    shared_ptr<Matrix<ElemType>> m_maskOfDropout;
};

// -----------------------------------------------------------------------
// BatchNormalizationNode (input, scale, bias, runMean, runVariance, runCount,
//                         spatial, normalizationTimeConstant = 0, blendTimeConstant = 0,
//                         epsilon = 0.00001,
//                         useCntkEngine = true, imageLayout = 'cudnn')
//
// Implements batch normalization technique as described in:
// Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift [S. Ioffe, C. Szegedy]
// http://arxiv.org/abs/1502.03167
// In short, it normalizes layer outputs for every minibatch for each output(feature) independently and applies affine transformation to preserve representation of the layer.
// That is, for layer input:
// 
// m = mean(input)
// var = variance(input)
// input_norm = (input - mean) / sqrt(epsilon + var)
// output = gamma * input_norm + beta
// 
// where gamma and beta are trainable parameters(represented as LearnableParameter).
// 
// * input is the input of the batch normalization node
// * scale is a LearnableParameter that stores scale vector (gamma term in the equation above).
// * bias is a LearnableParameter that stores bias vector (beta term). scale and bias must have the same dimensions which must be equal 
//      to the input dimensions in case of spatial = false or number of output convolution feature maps in case of spatial = true.
//      BUGBUG: Number of convolution feature maps are considered the last axis of the input.
//              More correct would be to infer that from broadcasting dimensions (spatial mode is broadcasting).
// * runMean is the running mean which is used during evaluation phase and might be used during training as well.
//      It is represented as a LearnableParameter with the same dimensions as scale and bias.
// * runVariance is the running variance which is used during evaluation phase and might be used during training as well.
//      It is represented as a LearnableParameter with the same dimensions as scale and bias.
// * runCount is the sample count needed for estimating runMean and runVariance. Pass a [1] tensor here.
// * spatial is a flag that specifies whether to compute mean / var for each feature in a minibatch independently or, in case of convolutional layers, per feature map.
//      TODO: This must be configured in a generic fashion where tensor axes are chosen along which parameters are tied.
// * normalizationTimeConstant is the time constant which is used to compute running average of mean and variance.
//      Value 0 (default) means there will be no exponential smoothing and running mean/variance will always have values computed for the last seen minibatch.
//      Value 1#INF (infinity) means running values are "frozen" (i.e., they will not be updated).
// * blendTimeConstant is the time constant which allows to specify how much of running mean / var should be "blended" into mean / var of the current minibatch.
//      Value 0 (default) means no blending will happen and only the current minibatch statistics will be used.
//      Value 1#INF (infinity) means only running mean / var will be used(this is used, for example, in evaluation phase).
// * epsilon is a conditioner constant used in computing inverse standard deviation
// * useCntkEngine is a Boolean flag that specifies which batch normalization implementation to use: CNTK or cuDNN-based.
// * imageLayout is the image layout. Only cudnn is supported at present.
// -----------------------------------------------------------------------
template <class ElemType>
class BatchNormalizationNode : public ComputationNodeNonLooping<ElemType>, public NumInputs<6>, public IFreezable,
    public IdentityTransformerNodeOnOneInput<0>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"BatchNormalization"; }

    // inputs
    // TODO: Change all of these throughout the codebase to 'class enum'. Also change all places where we still use integer constants.
    static const size_t DATA      = 0;
    static const size_t SCALE     = 1;
    static const size_t BIAS      = 2;
    static const size_t RUN_MEAN  = 3;
    static const size_t RUN_VAR   = 4;
    static const size_t RUN_COUNT = 5;
public:
    BatchNormalizationNode(DEVICEID_TYPE deviceId, const wstring& name, bool spatial = false,
                           double normalizationTimeConstant=0, double blendTimeConstant=0,
                           double epsilon = 0, bool useCntkEngine = true, ImageLayoutKind imageLayoutKind = ImageLayoutKind::CHW) :
        Base(deviceId, name), m_spatial(spatial), m_normTimeConst(normalizationTimeConstant), m_blendTimeConst(blendTimeConstant),
        m_epsilon(epsilon), m_useCntkEngine(useCntkEngine), m_imageLayoutKind(imageLayoutKind),
        m_runCountIsZero(true),
        m_pre19SampleCount(0),
        m_convertRunningVariancePending(false),
        m_one(1, 1, deviceId)
    {
        m_one.SetValue((ElemType)1); // (constant value used for GPU-side update of runCount)
    }
    BatchNormalizationNode(const ScriptableObjects::IConfigRecordPtr configp) :
        BatchNormalizationNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"spatial"),
                               configp->Get(L"normalizationTimeConstant"), configp->Get(L"blendTimeConstant"), 
                               configp->Get(L"epsilon"), configp->Get(L"useCntkEngine"),
                               ImageLayoutKindFrom(configp->Get(L"imageLayout")))
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }

    void Save(File& fstream) const override
    {
        Base::Save(fstream);

        fstream << m_spatial;
        fstream << m_normTimeConst;
        fstream << m_blendTimeConst;
        fstream << (int32_t)m_imageLayoutKind;
        fstream << m_runCountIsZero;  // this is only saved for optimizing 0-checks;
        fstream << m_epsilon;
        fstream << m_useCntkEngine;
    }

    void Load(File& fstream, size_t modelVersion) override
    {
        size_t mbCount = 0, runSampleCount = 0;
        Base::Load(fstream, modelVersion);

        if (modelVersion >= CNTK_MODEL_VERSION_6)
        {
            fstream >> m_spatial;
            fstream >> m_normTimeConst;
            fstream >> m_blendTimeConst;
            fstream >> m_imageLayoutKind;

            if (modelVersion >= CNTK_MODEL_VERSION_19)
            {
                fstream >> m_runCountIsZero; // the correct count will be loaded into the tied run count paramter.
            }
            else if (modelVersion >= CNTK_MODEL_VERSION_13) 
            {
                fstream >> runSampleCount;
            }    
            else 
            {
                fstream >> mbCount; // converted below
            }

            fstream >> m_epsilon;
            fstream >> m_useCntkEngine;
        }
        else
        {
            // Use old versioning scheme for older models.

            // Read and check version.
            // REVIEW alexeyk: extract version checking so it can be re-used in other places.
            int32_t verWritten;
            int32_t verReadable;
            fstream >> verWritten >> verReadable;
    
            if (verReadable > verWritten)
                RuntimeError("Corrupt model file.");
            if (verWritten < m_version.VerWeCanReadBack())
                RuntimeError("Model is too old.");
            if (verReadable > m_version.VerWrittenCur())
                RuntimeError("Model is too new.");

            bool eval;
            fstream >> eval;
            UNUSED(eval);
            fstream >> m_spatial;
            if (verWritten >= 0x00010004)
                fstream >> m_normTimeConst;
            else
            {
                double expAvgFactor;
                fstream >> expAvgFactor;
                UNUSED(expAvgFactor); // Used in previous versions, replaced by m_normTimeConst.
            }
            if (verWritten >= 0x00010002)
            {
                fstream >> m_imageLayoutKind;
                fstream >> mbCount; // converted below
            }
            if (verWritten >= 0x00010003)
            {
                fstream >> m_epsilon;
                fstream >> m_useCntkEngine;
            }
        }

        if (modelVersion < CNTK_MODEL_VERSION_13)
        {
            // Prior to version 12, and prior to storing counts in a shared Parameter, minibatch count was stored instead of samples seen.
            // Approximate by assuming minibatch size 16, inform about that.
            runSampleCount = 16 * mbCount;
            fprintf(stderr,
                    "INFO: %ls: loading pre-CuDNNv5 model: approximated mini-batch count of %" PRIu64 " as %" PRIu64 " trained samples.\n"
                    "      Statistics in further training may be biased; consider re-training instead.\n",
                    NodeName().c_str(), mbCount, runSampleCount);

            // Prior to version 12, running inverse standard deviation was
            // stored in Input 4. Now variance is used. We (approximately)
            // convert it during validation later (and then clear the flag).
            m_convertRunningVariancePending = true;
        }


        if (modelVersion < CNTK_MODEL_VERSION_19)
        {
            m_runCountIsZero = (runSampleCount == 0);
            // Cannot call "Input(RUN_COUNT)->Value().SetValue(ElemType(runSampleCount))" right here,
            // inputs are not yet attached at this point.
            m_pre19SampleCount = runSampleCount;
        }
    }

    void AttachInputs(const std::vector<ComputationNodeBasePtr>& inputs) override;

    void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<BatchNormalizationNode<ElemType>>(nodeP);
            assert(node != nullptr);

            node->m_spatial         = m_spatial;
            node->m_normTimeConst   = m_normTimeConst;
            node->m_blendTimeConst  = m_blendTimeConst;
            node->m_imageLayoutKind = m_imageLayoutKind;
            node->m_runCountIsZero = m_runCountIsZero;
            node->m_epsilon         = m_epsilon;
            node->m_useCntkEngine   = m_useCntkEngine;
        }
    }

private: // time-constant conversions
    
    void ResetRunCount()
    {
        Input(RUN_COUNT)->Value().SetValue(0);
        m_runCountIsZero = true;
    }

    void AggregateRunCount(size_t countToAdd)
    {
        Input(RUN_COUNT)->Value().AddWithScaleOf(/*alpha=*/(ElemType)countToAdd, m_one); // this += countToAdd * (1)
        // this is only needed for 0 checks.
        m_runCountIsZero = m_runCountIsZero && (countToAdd == 0);
    }

    size_t RunCount() const // const version of above; keep identical
    {
        auto count = (size_t)Input(RUN_COUNT)->Value().Get00Element();
        m_runCountIsZero = (count == 0);
        return count;
    }
    
    bool IsRunCount0() const 
    { 
        // check the flag first to avoid GPU sync 
        // (when it's false, we know for a fact that the running count is non-zero)
        return m_runCountIsZero && RunCount() == 0;
    } 

    // map time constants to exp avg factor
    // This is the factor for the current MB's estimate (1-factor is used for the previous value of the running stats).
    double ComputeExpAvgFactor() const
    {
        // in inference mode, only use long-term mean and do not update running estimates
        if (!Environment().IsTraining())
        {
            return 0; // (m_normTimeConst == infinity) no new contribution from current minibatch
        }

        double numSamples = (double)GetMBLayout()->GetActualNumSamples();

        // m_normTimeConst < 0 is used to denote corpus-level statistics (without forgetting factor).
        // BUGBUG: Corpus aggregation in float is numerically instable. E.g. over a corpus of 36000 samples,
        //         the first few MBs are the same when sharing the same node twice at a split point vs.
        //         applying it right before the split. However, at the end, averages differ notably.
        //         Errs increase from 2.9% to 3.2%. For 36000 samples (~500 MBs), the last summation has 13 bits.
        //         In contrast, after 1000 samples, accuracy is identical; and marginally different after 5000.
        if (m_normTimeConst < 0)
            return numSamples / (numSamples + RunCount());

        // Initialization case: only use current minibatch.
        // BUGBUG: This gives the first MB an unduely high contribution.
        //         This now implements a low-pass filter over a sequence
        //         ...AAAAAAAAAABCDEFG... where A is the first MB, which is infinitely repeated.
        //         The error will taper off; the first MB will have reduced to 37% of the average after
        //         #samples = batchNormTimeConstant. So for our default of 5000, it likely matter little.
        if (IsRunCount0())
            return 1.0;

        // Convert to per-minibatch factor. The limit, positive infinity, means that running mean/var parameters are "frozen"
        // that is, do not require updates.
        // The code below special-cases two boundary cases, but those are just the limit cases of the main formula.
        if (!isfinite(m_normTimeConst))                      // infinite
            return 0;                                        // no new contribution from current minibatch (infinitely long memory)
        else if (m_normTimeConst > 0)                        // not zero
            return -expm1(-numSamples / m_normTimeConst);    // interpolate expAvgFactor * MB stats + (1-expAvgFactor) * prev running stats
        else                                                 // zero
            return 1.0;                                      // don't use running stats at all
    }

    // map sample count to blend factor
    // This is the interpolation weight for the running statistics (the current MB statistics are weighted with 1-this).
    double ComputeBlendFactor() const
    {
        // in inference mode, only use long-term mean and do not update running estimates
        if (!Environment().IsTraining())
        {
            return 1.0; // (m_blendTimeConst == infinity) estimate is taken 100% from the long-term running estimate
        }

        // Initialization case: only use current minibatch.
        if (IsRunCount0())
            return 0;

        // convert to blend factor (= weight for running stats)
        // The code below special-cases two boundary cases, but those are just the limit cases of the main formula.
        double numSamples = (double)GetMBLayout()->GetActualNumSamples();
        if (!isfinite(m_blendTimeConst))                               // infinite weight for prior stats
            return 1.0;                                                // only use running statistics
        else if (m_blendTimeConst > 0)                                 // not zero
            return m_blendTimeConst / (m_blendTimeConst + numSamples); // interpolate blendFactor * running stats + (1-blendFactor) * MB stats
        else                                                           // zero
            return 0;                                                  // no weight for prior stats, only use MB stats
    }
public:

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        if (m_convertRunningVariancePending)
            LogicError("%ls: Failed to convert running variance until forward prop", NodeName().c_str());
        FrameRange fr(Input(DATA)->GetMBLayout());

        Matrix<ElemType> sliceInputValue  = Input(DATA)->MaskedValueFor(fr);
        const Matrix<ElemType>& scale     = Input(SCALE)->Value();
        const Matrix<ElemType>& bias      = Input(BIAS)->Value();
        Matrix<ElemType>& runMean         = Input(RUN_MEAN)->Value();
        Matrix<ElemType>& runVariance     = Input(RUN_VAR)->Value();
        Matrix<ElemType> sliceOutputValue = ValueFor(fr);

        assert(scale.GetNumRows() == bias.GetNumRows());
        assert(scale.GetNumCols() == bias.GetNumCols());
        assert(runMean.GetNumRows() == scale.GetNumRows());
        assert(runMean.GetNumCols() == scale.GetNumCols());
        assert(runMean.GetNumRows() == runVariance.GetNumRows());
        assert(runMean.GetNumCols() == runVariance.GetNumCols());

        // determine the factors from the time constants
        double expAvgFactor = ComputeExpAvgFactor(); // weight for the new MB statistics in the running estimate. The previous value of the running statistics is kept with weight (1-this)
        double blendFactor  = ComputeBlendFactor();  // interpolation weight for the running statistics (the current MB statistics are weighted with 1-this)

        // In inference-only mode, m_savedMean and m_saveInvStdDev will not be
        // produced and BackpropToNonLooping() may not be called. In
        // non-inference (training) mode, saved statistics must be produced.
        bool inferenceOnly = !Environment().IsTraining();
        m_bnEng->Forward(/*in=*/ sliceInputValue, scale, bias,   // (in)
                         inferenceOnly, expAvgFactor, blendFactor,
                         runMean, runVariance,                   // (in/out) running estimates, updated from the current MB mean/variance
                         /*out=*/ sliceOutputValue,              // (out) batch-normalized output value
                         m_epsilon,
                         *m_savedMean, *m_savedInvStdDev);       // (out) actual interpolated mean/stddev values. Note: unused/empty for blendFactor==1 for CNTK engine

        // and update the denominator
        if (expAvgFactor != 0 || blendFactor != 1)
            AggregateRunCount(GetMBLayout()->GetActualNumSamples());

        // gradient is as of now invalid
        m_gradientValid = false;
    }

    // Note: This function assumes that inputIndex=0 is called before the others, unless the DATA input takes no gradient.
    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        // Must be in training mode.
        if (!Environment().IsTraining())
            LogicError("%ls: BackpropToNonLooping() cannot be called in inference mode", NodeName().c_str());
        // In non-inference mode, the batch normalization engine must provide
        // saved statistics, m_savedMean and m_savedInvStdDev
        if (m_savedMean->IsEmpty())
            LogicError("%ls: m_savedMean cannot be empty", NodeName().c_str());
        if (m_savedInvStdDev->IsEmpty())
            LogicError("%ls: m_savedInvStdDev cannot be empty", NodeName().c_str());

        FrameRange fr(Input(DATA)->GetMBLayout());

        if (inputIndex == DATA || !m_gradientValid) // derivative with respect to the input.
        {
            auto sliceOutputGrad          = MaskedGradientFor(fr);
            auto sliceInputValue          = Input(DATA)->ValueFor(fr);
            const Matrix<ElemType>& scale = Input(SCALE)->Value();
            const Matrix<ElemType>& bias  = Input(BIAS)->Value();

            // If inputIndex is not DATA and we get here, then it means that DATA receives no gradient.
            // However, the underlying engine does not foresee this case, and thus always needs a place
            // to store the gradient. Hence, in that case, we create a dummy object and use that instead.
            bool needsInputGradient = (inputIndex == DATA);
            if (needsInputGradient && m_gradientValid) // because otherwise we already computed it into the dummy location
                LogicError("BackpropTo: Batch-normalization data gradient must be requested before all others.");
            if (!needsInputGradient)
                m_dDataDummy->Resize(sliceInputValue);
            auto sliceInputGrad = needsInputGradient ? Input(DATA)->GradientFor(fr) : m_dDataDummy->AsReference();

            m_dScale->Resize(scale); // gradients for scale and bias get stored here
            m_dBias->Resize(bias);

            double blendFactor = ComputeBlendFactor();  // interpolation weight for the running statistics (the current MB statistics are weighted with 1-this)

            // Compute all derivatives in one step. Save derivatives with respect to scale and bias in temp matrices.
            m_bnEng->Backward(sliceInputValue, sliceOutputGrad, // (in)  input from below, gradient from above
                              sliceInputGrad,                   // (out) gradient for data input goes here  --TODO: Check if cudnn engine adds the gradient, or just overwrites (BUGBUG). CNTK engine is OK.
                              scale,                            // (in)  out of scale and bias, only scale is needed in gradient propagation
                              blendFactor,                      // (in)  smoothing weight for running stats (1=use only running stats)
                              *m_savedMean, *m_savedInvStdDev,  // (in)  saved mean/invstddev values used in ForwardProp()
                              *m_dScale, *m_dBias);             // (out) gradients for scale and bias

            m_gradientValid = true;
        }
        if (inputIndex == SCALE) // derivative with respect to the scale, precomputed during input derivative computation
        {
            // Derivative with respect to the scale was precomputed during input derivative computation.
            assert(m_gradientValid);
            Input(SCALE)->Gradient() += *m_dScale;
        }
        else if (inputIndex == BIAS) // derivative with respect to the bias, precomputed during input derivative computation
        {
            // Derivative with respect to the bias was precomputed during input derivative computation.
            assert(m_gradientValid);
            Input(BIAS)->Gradient() += *m_dBias;
        }
        // No derivatives with respect to running mean and variance.
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override { return false; }

    void Validate(bool isFinalValidationPass) override
    {
        Base::Validate(isFinalValidationPass);

        if (GetNumInputs() != RUN_COUNT && GetNumInputs() != RUN_COUNT + 1)
            InvalidArgument("%ls %ls operation accepts %d inputs.", NodeName().c_str(), OperationName().c_str(), (int)RUN_COUNT + 1);
        // (we won't report that it also accepts RUN_COUNT inputs, as this is the deprecated legacy case)

        InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);

        SetDims(Input(DATA));

        const auto& inputLayout = Input(DATA)->GetSampleLayout();

        // running statistics inputs must be learnable parameters, since we update them directly here
        for (size_t i = RUN_MEAN; i < GetNumInputs(); i++)
            //if (!Input(i)->Is<LearnableParameter<ElemType>>()) // somehow this does not compile on gcc (works on VS)
            if (!dynamic_cast<LearnableParameter<ElemType>*>(Input(i).get()))
                InvalidArgument("%ls: Inputs [%d..%d] must be learnable parameters.", NodeDescription().c_str(), (int)RUN_MEAN, (int)GetNumInputs());

        // infer dimensions of learnable parameters
        // BUGBUG: Parameter dimensions are totally wrong. E.g. a valid spatial bias for [15 x 15 x 32] is currently [32 x 1].
        //         The correct bias shape should be [1 x 1 x 32]. That can be specified but leads to different results for unknown reasons.
        //         Until this has been corrected, we need a workaround that infers the wrong dimensions.
#if 1   // Workaround for today's definition: Trigger on [0 x 1] and infer that 0 as the total # elements needed.
        for (size_t i = SCALE; i < RUN_COUNT; i++) // scale, bias, run_mean, and run_variance
        {
            auto paramLayout = Input(i)->GetSampleLayout();
            if (paramLayout.GetRank() == 2 && paramLayout[0] == 0 && paramLayout[1] == 1 && inputLayout.GetNumElements() > 0) // [0 x 1]
            {
                size_t total = m_spatial ? inputLayout.GetDims().back() : inputLayout.GetNumElements();
                Input(i)->ValidateInferInputDimsFrom(TensorShape(total, 1));
            }
        }
#else
        // These are here only inferred like for elementwise operations. We must check more.
        ValidateNaryZip(isFinalValidationPass, /*allowBroadcast=*/ true, GetNumInputs());
#endif

        if (isFinalValidationPass)
        {
            // The current implementation requires that the gradient of the first operand/input be computed
            // in order to compute gradients for the bias and scale parameters (2nd and 3rd inputs)
            if (Environment().IsTraining() && ((Input(1)->NeedsGradient() || Input(2)->NeedsGradient()) && !Input(0)->NeedsGradient()))
                InvalidArgument("%ls %ls currently supports learnable scale and bias parameters only if the first input also needs gradient (i.e. is dependent on at-least one learnable parameter).", NodeName().c_str(), OperationName().c_str());

            if (m_convertRunningVariancePending)
            {
                // Prior to CNTK CuDNN v5 support (and the CNTK engine of the same time), mean and inverse standard deviation
                // statistics were computed and stored. With CuDNN v5 (and the corresponding CNTK engine update), this was changed
                // to mean and variance.
                // To load an old model for further training or inference, Input(RUN_VAR) (which is inverse standard deviation) needs to
                // be converted to variance, via v = 1/(isd^2) + epsilon, where 'v' is variance and 'isd' is inverse standard
                // Since this is an approximation, we output a warning.
                fprintf(stderr, "WARNING: %ls: loading pre-CuDNNv5 model: approximately converting variance statistics format\n",
                        NodeName().c_str());
                Matrix<ElemType>& runInvStdDev = Input(RUN_VAR)->Value();
                runInvStdDev.AssignElementPowerOf(runInvStdDev, 2);
                runInvStdDev.ElementInverse();
                runInvStdDev += (float) m_epsilon;
                m_convertRunningVariancePending = false;
            }

            // check inputs
            for (size_t i = SCALE; i < RUN_COUNT; i++) // scale, bias, run_mean, and run_variance
            {
                if (Input(i)->HasMBLayout())
                    InvalidArgument("%ls: Input[%d] has a dynamic axis. BatchNormalization parameters cannot have that.", NodeDescription().c_str(), (int)i);
                auto paramLayout = Input(i)->GetSampleLayout();
                if (paramLayout != Input(SCALE)->GetSampleLayout())
                    InvalidArgument("%ls: Input[%d] has a layout different from Input[1]. All must be identical.", NodeDescription().c_str(), (int)i);
#if 0           // BUGBUG: For this to work, parameter shapes must be correct (cf. comment above on inference).
                if (paramLayout.GetRank() > inputLayout.GetRank())
                    InvalidArgument("%ls: Input[%d] has a tensor rank greated than the data input.", NodeDescription().c_str(), (int)i);
                for (size_t k = 0; k < paramLayout.size(); k++)
                    if (paramLayout[k] > inputLayout[k])
                        InvalidArgument("%ls: Data input cannot broadcast.", NodeDescription().c_str());
#endif
            }
            
            // This must always be a [1] tensor. No inference allowed.
            size_t i = RUN_COUNT;
            if (Input(i)->HasMBLayout() || Input(i)->GetSampleLayout() != TensorShape(1))
                InvalidArgument("%ls: Input[%d] must be a vector of 1 element without dynamic axis.", NodeDescription().c_str(), (int)i);
            RunCount(); // refresh the zero count flag value, for 0 checks

            if (m_spatial && m_imageLayoutKind != CHW)
            {
                InvalidArgument(
                    "%ls %ls currently supports only cuDNN (CHW) data layout. " 
                    "Please specify imageLayout=\"cudnn\" in BatchNormalization node in your NDL/BrainScript "
                    "and make sure your input data layout is CHW", NodeName().c_str(), OperationName().c_str());
            }

            if (!m_useCntkEngine)
            {
                // Fallback to cntk engine on CPU device if cuDnn is not available,
                bool cpuDevice = (m_deviceId == CPUDEVICE);

                // or if parameters cannot be handled by cuDnn (which is needed for compatibility when changing the default to cudnn)
                // In Source/Math/CuDnnBatchNormalization.cu :
                //   if (blendFactor != 0 && (blendFactor != 1 || expAvgFactor > 0))
                //      InvalidArgument("cuDNN batch normalization engine currently supports blendTimeConstant of 0 or 1 only.");
                // Check ComputeBlendFactor()/ComputeExpAvgFactor() for inferring blendFactor/expAvgFactor from m_blendTimeConst/m_normTimeConst
                bool cuDnnUnsupportedParams = (m_blendTimeConst != 0 && (isfinite(m_blendTimeConst) || isfinite(m_normTimeConst)));

                if (cpuDevice || cuDnnUnsupportedParams)
                {
                    m_useCntkEngine = true;
                    if (cuDnnUnsupportedParams)
                    {
                        fprintf(stderr, "\nWARNING: batch normalization falls back to cntk engine for parameters not supported by cuDnn.\n");
                    }
                }
            }

            double cudnnMinEps = 1e-5; // CUDNN_BN_MIN_EPSILON
            if (!m_useCntkEngine && m_epsilon < cudnnMinEps) 
                fprintf(stderr, "\nWARNING: cuDNN batch normalization requires epsilon >= %e. Epsilon will be reset to that value.\n", cudnnMinEps);

            if (m_blendTimeConst < 0)
                InvalidArgument("%ls %ls requires blend time constant to be >= 0.", NodeName().c_str(), OperationName().c_str());

            if (m_bnEng == nullptr)
            {
                auto shape = GetSampleLayout();
                m_bnEng = BatchNormEngine<ElemType>::Create(m_deviceId, shape, m_spatial, m_imageLayoutKind,
                                                            m_useCntkEngine ? BatchNormEngineKind::Cntk : BatchNormEngineKind::CuDnn);
            }
        }
    }

    void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_savedMean, matrixPool);
        RequestMatrixFromPool(m_savedInvStdDev, matrixPool);
    }

    void RequestMatricesBeforeBackprop(MatrixPool& matrixPool) override
    {
        Base::RequestMatricesBeforeBackprop(matrixPool);
        RequestMatrixFromPool(m_dDataDummy, matrixPool);
        RequestMatrixFromPool(m_dScale, matrixPool);
        RequestMatrixFromPool(m_dBias, matrixPool);
    }

    void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool) override
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_savedMean, matrixPool);
        ReleaseMatrixToPool(m_savedInvStdDev, matrixPool);
        ReleaseMatrixToPool(m_dDataDummy, matrixPool);
        ReleaseMatrixToPool(m_dScale, matrixPool);
        ReleaseMatrixToPool(m_dBias, matrixPool);
    }

    void SetNormalizationTimeConstants(double normalizationTimeConstant, double prevNormalizationTimeConstant,
                                       double blendTimeConstant, double prevBlendTimeConstant)
    {
        // As this function is called from SGD solver (global), make sure we don't
        // override settings set in NDL when it's not necessary.
        if (normalizationTimeConstant != prevNormalizationTimeConstant)
            m_normTimeConst = normalizationTimeConstant;
        if (blendTimeConstant != prevBlendTimeConstant)
            m_blendTimeConst = blendTimeConstant;
    }

    // called from CloneFunction(..., parameters="constant")
    // Once called, this node is put into inference mode.
    virtual void FreezeParameters() override // from IFreezable
    {
        m_normTimeConst  = std::numeric_limits<double>::infinity();
        m_blendTimeConst = std::numeric_limits<double>::infinity();
    }

    // ResetStatisticsState() will set the batch normal statistics into initial state
    // used for re-statistics the mean and variance of BN.
    // In case of multiple BN nodes sharing statistics, this must be called on all.
    // Any other use is undefined behavior.
    void ResetStatisticsState()
    {
        ResetRunCount();
        m_normTimeConst = 0;
        m_blendTimeConst = 0;
    }
    // Turn off the L1 and L2 regularization
    void DisableRegInBatchNormalization()
    {
        let scaleNode = dynamic_pointer_cast<LearnableParameter<ElemType>>(Input(SCALE));
        let biasNode  = dynamic_pointer_cast<LearnableParameter<ElemType>>(Input(BIAS));
        scaleNode->SetRegMultiplier(0.f);
        biasNode->SetRegMultiplier(0.f);
    }
    double NormalizationTimeConstant() const { return m_normTimeConst; }
    double BlendTimeConstant() const { return m_blendTimeConst; }
    bool Spatial() const { return m_spatial; }
    double Epsilon() const { return m_epsilon; }
    bool UseCNTKEngine() const { return m_useCntkEngine; }

private:
    // Old versioning - do not use. Do not remove until we're sure there are no old models around.
    struct VersionInfo
    {
        //int32_t VerWrittenCur() const      { return 0x00010001; } // Initial
        //int32_t VerWrittenCur() const      { return 0x00010002; } // Added m_imageLayoutKind and m_mbCount
        //int32_t VerWrittenCur() const      { return 0x00010003; } // Added m_epsilon and m_useCntkEngine
        int32_t VerWrittenCur() const        { return 0x00010004; } // Added m_normTimeConst
        int32_t VerReadableCur() const       { return 0x00010004; }
        int32_t VerWeCanReadBack() const     { return 0x00010001; }
    };
    VersionInfo m_version;

private:
    // --- configuration parameters

    // Determines whether to use per-activation (used after non-convolutional layers like fully connected)
    // or spatial (used after convolutional layers).
    // TODO: This should not be a config option, but rather inferred from dimensions of the Parameters.
    bool m_spatial;

    // Time constant for estimating the running mean and variance.
    // This is the time constant of a low-pass filter.
    // If 0, running mean and variance just remember the last minibatch.
    // If infinity, running mean and variance are not updated, like in inference mode.
    double m_normTimeConst;

    // Equivalent sample count for blending running mean/var and current minibatch mean/var.
    // Roughly, this specifies how many samples "worth" is the running statistics,
    // relative to the current minibatch statistics.
    // If 0, only use the current MB statistics. If infinity, use only the running mean, like in inference mode.
    // The main idea is to estimate the mean/variance as a MAP estimate using the running mean/var as a prior.
    // This should make the method more robust to the case of very small minibatches,
    // and also provides a meaningful interpretation of inference mode, where only the prior is used.
    // Effectively, this ends up in a linear interpolation of running and minibatch statistics.
    // The idea is due to Frank Seide et al.
    // It should also work well in data parallelism scenario, as opposed to plain vanilla BN implementation
    // which would require aggregation of statistics from all nodes.
    // REVIEW alexeyk: if this works, document it properly in Wiki.
    double m_blendTimeConst;

    // Epsilon used to compute inverse standard deviation (m_savedInvStdDev).
    double m_epsilon;
    // Whether to use CNTK or cuDNN BN implementation.
    bool m_useCntkEngine;
    // Layout (e.g. CHW).
    ImageLayoutKind m_imageLayoutKind;

    // --- working variables

    //  - 0 checks test this first, to avoid unnecessary GPU syncs
    mutable bool m_runCountIsZero;
    
    // we use this count to temporarily store the number of samples seen, loaded from a model file
    // with a version < 19, before the running mean sample count was added as an addition input for
    // BatchNorm node. The value is copied over into the corresponding input parameter inside the 
    // AttachInputs() call.
    size_t m_pre19SampleCount;
    Matrix<ElemType> m_one;  // constant [1x1] matrix that contains a 1 (used for updating the shared count)

    // Interpolated actual mean/inverse stddev values. Pre-computed on forward pass, also used in gradient computation.
    shared_ptr<Matrix<ElemType>> m_savedMean;
    shared_ptr<Matrix<ElemType>> m_savedInvStdDev;
    // Temp buffer for scale and bias derivatives. Only used in BackpropTo(), carrying info from first call to subsequent calls.
    // Not used for blendFactor=1 in CNTK engine.
    shared_ptr<Matrix<ElemType>> m_dDataDummy;
    shared_ptr<Matrix<ElemType>> m_dScale;
    shared_ptr<Matrix<ElemType>> m_dBias;

    bool m_gradientValid = false;

    std::unique_ptr<BatchNormEngine<ElemType>> m_bnEng;

    bool m_convertRunningVariancePending;
};

}}}
