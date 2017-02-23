//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "gammacalculation.h"
#include "InputAndParamNodes.h"
#include "Sequences.h"
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>


namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// ClassificationErrorNode (label, prediction)   or ClassificationErrorNode (prediction, label)
// Performs classification and error counting.
// Result is an error rate, lower = better.
// -----------------------------------------------------------------------

template <class ElemType>
class ClassificationErrorNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"ClassificationError"; }

public:
    DeclareConstructorFromConfig(ClassificationErrorNode);
    ClassificationErrorNode(DEVICEID_TYPE deviceId, const wstring& name)
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
        FrameRange fr(InputRef(0).GetMBLayout());
        InputRef(0).ValueFor(fr).VectorMax(*m_maxIndexes0, *m_maxValues, true);
        InputRef(1).ValueFor(fr).VectorMax(*m_maxIndexes1, *m_maxValues, true, m_topK);
        MaskMissingColumnsToZero(*m_maxIndexes0, InputRef(0).GetMBLayout(), fr);
        MaskMissingColumnsToZero(*m_maxIndexes1, InputRef(1).GetMBLayout(), fr);
        Value().AssignNumOfDiff(*m_maxIndexes0, *m_maxIndexes1, m_topK > 1);
#if NANCHECK
        Value().HasNan("ClassificationError");
#endif
#if DUMPOUTPUT
        Value().Print("ClassificationErrorNode");
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
            auto node = dynamic_pointer_cast<ClassificationErrorNode<ElemType>>(nodeP);
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

template class ClassificationErrorNode<float>;
template class ClassificationErrorNode<double>;

// -----------------------------------------------------------------------
// NDCG1EvalNode (gain, prediction, queryId)
// NDCG @ 1 
// -----------------------------------------------------------------------

template <class ElemType>
class NDCG1EvalNode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<3>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"NDCG1Eval";
    }

public:
    DeclareConstructorFromConfig(NDCG1EvalNode);
    NDCG1EvalNode(DEVICEID_TYPE deviceId, const wstring& name)
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
        // Inputs:
        //      0. gain
        //      1. pred
        //      2. query id
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
            QueryUrls& qub = m_queryUrls.back();
            Url u(i, qub.m_urls.size(), preds(0, i), gains(0, i));
            qub.m_urls.push_back(u);
        }

        for (auto &qu : m_queryUrls)
        {
            std::vector<Url>& urls = qu.m_urls;
            // Urls are pre-sorted in descending order of gains.
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

        // calculate IRMetrics
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

        //Aggregate at query level.
        const Matrix<ElemType>& urlDiscountedGain0 = *m_urlGain0;
        const Matrix<ElemType>& urlDiscountedGain1 = *m_urlGain1;
        ElemType irMetricValue = 0.0;
        ElemType idealMetric = 0.0, metric = 0.0;

        // IRMetric @ 1
        for (auto &qu : m_queryUrls)
        {
            idealMetric = urlDiscountedGain0(0, qu.m_urls.begin()->m_id);
            if (idealMetric == 0.0) continue;

            for (auto &url : qu.m_urls)
            {
                if (url.m_rank == 0)
                {
                    metric = urlDiscountedGain1(0, url.m_id);
                    break;
                }
            }

            irMetricValue += (metric / idealMetric);
        }

        if (numberOfQueries == 0)
        {
            LogicError("In %ls %ls numberOfQueries==0, check your data.", NodeName().c_str(), OperationName().c_str());
        }

        irMetricValue = irMetricValue / numberOfQueries * 100 * numberOfSamples;
        Value().SetValue(irMetricValue);
    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        if (m_inputs.size() != 3)
            InvalidArgument("%ls operation requires three inputs instead of %d.", NodeDescription().c_str(), (int)m_inputs.size());

        if (Input(0)->NeedsGradient() == true)
            InvalidArgument("%ls %ls operation needs input type (no gradient) for the 1st input.", NodeName().c_str(), OperationName().c_str());

        if (Input(2)->NeedsGradient() == true)
            InvalidArgument("%ls %ls operation needs input type (no gradient) for the 3rd input.", NodeName().c_str(), OperationName().c_str());

        ValidateBinaryReduce(isFinalValidationPass);
    }

    virtual void UpdateFunctionMBSize() override
    {
        UpdateCounts();

        // clean up first
        if (!m_queryUrls.empty()) m_queryUrls.clear();
        if (!m_urlSorter.empty()) m_urlSorter.clear();
        if (!m_logWeights.empty()) m_logWeights.clear();

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
            *it = (ElemType)log(2.0 + i);
        }
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<NDCG1EvalNode<ElemType>>(nodeP);
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
        RequestMatrixFromPool(m_urlGain0, matrixPool);
        RequestMatrixFromPool(m_urlGain1, matrixPool);
        RequestMatrixFromPool(m_urlDiscount0, matrixPool);
        RequestMatrixFromPool(m_urlDiscount1, matrixPool);
    }

    // release gradient and temp matrices that no longer needed after all the children's gradients are computed.
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
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
        size_t maxNumberOfUrlsPerQuery = 0;
        for (size_t i = 0; i < numberOfQueryUrls; i++)
        {
            int queryId = (int)queryIds(0, i);
            if (queryId != previousQueryId)
            {
                if (numberOfUrls > maxNumberOfUrlsPerQuery)
                {
                    maxNumberOfUrlsPerQuery = numberOfUrls;
                }

                // New query
                numberOfUrls = 0;
                previousQueryId = queryId;
            }
            
            numberOfUrls++;
        }

        // Add last query.
        {
            if (numberOfUrls > maxNumberOfUrlsPerQuery)
            {
                maxNumberOfUrlsPerQuery = numberOfUrls;
            }
        }

        m_numberOfQueryUrls = numberOfQueryUrls;
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
        }

        Url(int _id, int _rk0, ElemType _sc, ElemType _gn) : m_id(_id), m_rank0(_rk0), m_rank(0), m_score(_sc), m_gain(_gn) {}

        int m_id;         // sample id
        int m_rank0;        // original rank based on label
        int m_rank;         // rank based on s in the associated query
        ElemType m_score;    // score
        ElemType m_gain;    // gain
        bool operator < (const Url &url) const{
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
        std::vector<Url> m_urls;
    };

    // master data structure
    std::list<QueryUrls> m_queryUrls;
    // buffer for sorting
    std::vector<Url> m_urlSorter;
    // lookup table for position based weights
    std::vector<ElemType> m_logWeights;

    size_t m_numberOfQueryUrls;
    size_t m_maxNumberOfUrlsPerQuery;
    // store the gains and weights
    shared_ptr<Matrix<ElemType>> m_urlGain0;
    shared_ptr<Matrix<ElemType>> m_urlGain1;
    shared_ptr<Matrix<ElemType>> m_urlDiscount0;
    shared_ptr<Matrix<ElemType>> m_urlDiscount1;
};

template class NDCG1EvalNode<float>;
template class NDCG1EvalNode<double>;

// Edit distance error evaluation node with the option of specifying penalty of substitution, deletion and insertion, as well as squashing the input sequences and ignoring certain samples.
// Using the classic DP algorithm as described in https://en.wikipedia.org/wiki/Edit_distance, adjusted to take into account the penalties.
// 
// The node allows to squash sequences of repeating labels and ignore certain labels. For example, if squashInputs is true and tokensToIgnore contains label '-' then
// given first input sequence as s1="a-ab-" and second as s2="-aa--abb" the edit distance will be computed against s1' = "aab" and s2' = "aab".
//
// The returned error is computed as: EditDistance(s1,s2) * length(s1') / length(s1)
//
// Just like ClassificationError and other evaluation nodes, when used as an evaluation criterion, the SGD process will aggregate all values over an epoch and report the average, i.e. the error rate.
// Primary objective of this node is for error evaluation of CTC training, see formula (1) in "Connectionist Temporal Classification: Labelling Unsegmented
// Sequence Data with Recurrent Neural Networks", http://machinelearning.wustl.edu/mlpapers/paper_files/icml2006_GravesFGS06.pdf
template<class ElemType>
class EditDistanceErrorNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>, public NumInputs<2>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"EditDistanceError"; }

public:
    // subPen - substitution penalty
    // delPen - deletion penalty
    // insPen - insertion penalty
    // squashInputs - whether to merge sequences of identical samples.
    // tokensToIgnore - list of samples to ignore during edit distance evaluation
    EditDistanceErrorNode(DEVICEID_TYPE deviceId, const wstring & name, float subPen = 0.0f, float delPen = 0.0f, float insPen = 0.0f, bool squashInputs = false, vector<size_t> tokensToIgnore = {})
        : Base(deviceId, name), m_SubPen(subPen), m_DelPen(delPen), m_InsPen(insPen), m_SquashInputs(squashInputs), m_tokensToIgnore(tokensToIgnore)
    {
    }

    EditDistanceErrorNode(const ScriptableObjects::IConfigRecordPtr configp)
        : EditDistanceErrorNode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"subPen"), configp->Get(L"delPen"), configp->Get(L"insPen"), configp->Get(L"squashInputs"), {})
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
        m_tokensToIgnore = ScriptableObjects::ConfigArray::FlattenedVectorFrom<size_t>(configp->Get(L"tokensToIgnore"));
    }

    virtual void BackpropToNonLooping(size_t /*inputIndex*/) override
    {
        LogicError("%ls operation is used for evaluation only.", OperationName().c_str());
    }

    virtual void ForwardPropNonLooping() override
    {
        bool isInput0Sparse = Input(0)->template Is<SparseInputValue<ElemType>>();
        bool isInput1Sparse = Input(1)->template Is<SparseInputValue<ElemType>>();
        if (isInput0Sparse || isInput1Sparse)
            LogicError("EditDistanceError node was not tested for sparse inputs.");

        FrameRange frameRange(Input(0)->GetMBLayout());
        Input(0)->ValueFor(frameRange).VectorMax(*m_maxIndexes0, *m_maxValues, true);
        Input(1)->ValueFor(frameRange).VectorMax(*m_maxIndexes1, *m_maxValues, true);

        MaskMissingColumnsToZero(*m_maxIndexes0, Input(0)->GetMBLayout(), frameRange);
        MaskMissingColumnsToZero(*m_maxIndexes1, Input(1)->GetMBLayout(), frameRange);
        Value()(0, 0) = ComputeEditDistanceError(*m_maxIndexes0, *m_maxIndexes1, Input(0)->GetMBLayout(), m_SubPen, m_DelPen, m_InsPen, m_SquashInputs, m_tokensToIgnore);
    }

    virtual void Validate(bool isFinalValidationPass) override
    {
        ValidateBinaryReduce(isFinalValidationPass);
    }

    virtual void UpdateFunctionMBSize() override
    {
        Base::UpdateFunctionMBSize();

        // resize the temporaries to their proper size
        size_t cols = Input(0)->Value().GetNumCols();
        m_maxIndexes0->Resize(1, cols);
        m_maxIndexes1->Resize(1, cols);
        m_maxValues->Resize(1, cols);
    }

    virtual void CopyTo(ComputationNodeBasePtr  nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);

        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<EditDistanceErrorNode<ElemType>>(nodeP);
            node->m_maxIndexes0 = m_maxIndexes0;
            node->m_maxIndexes1 = m_maxIndexes1;
            node->m_maxValues = m_maxValues;
            node->m_SquashInputs = m_SquashInputs;
            node->m_SubPen = m_SubPen;
            node->m_DelPen = m_DelPen;
            node->m_InsPen = m_InsPen;
            node->m_tokensToIgnore = m_tokensToIgnore;
        }
    }

    //request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_maxIndexes0, matrixPool);
        RequestMatrixFromPool(m_maxIndexes1, matrixPool);
        RequestMatrixFromPool(m_maxValues, matrixPool);
    }

    //release temp matrices that are only used by forward computation
    //don't release matrices that need to be used in the gradient computation
    virtual void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterForwardProp(matrixPool);
        ReleaseMatrixToPool(m_maxIndexes0, matrixPool);
        ReleaseMatrixToPool(m_maxIndexes1, matrixPool);
        ReleaseMatrixToPool(m_maxValues, matrixPool);
    }

    // firstSeq - first sequence of samples
    // secondSeq - second sequence of samples
    // numParallelSequences - number of parallel sequences in the minibatch
    // subPen - substitution penalty
    // delPen - deletion penalty
    // insPen - insertion penalty
    // squashInputs - whether to merge sequences of identical samples.
    // tokensToIgnore - list of samples to ignore during edit distance evaluation
    static ElemType ComputeEditDistanceError(Matrix<ElemType>& firstSeq, const Matrix<ElemType> & secondSeq, MBLayoutPtr pMBLayout, 
        float subPen, float delPen, float insPen, bool squashInputs, const vector<size_t>& tokensToIgnore)
    {
        std::vector<int> firstSeqVec, secondSeqVec;

        // Edit distance between subsequences
        Matrix<float> grid(CPUDEVICE);
        
        // Number of insertions between subsequences
        Matrix<float> insMatrix(CPUDEVICE);
        
        //Number of deletions between subsequences
        Matrix<float> delMatrix(CPUDEVICE);

        // Number of substitutions between subsequences
        Matrix<float> subMatrix(CPUDEVICE);

        float del, ins, sub;
        ElemType wrongSampleNum = 0.0;
        size_t totalSampleNum = 0, totalframeNum = 0;
        size_t sequenceStartFrame = 0;

        for (const auto& sequence : pMBLayout->GetAllSequences())
        {
            if (sequence.seqId == GAP_SEQUENCE_ID)
                continue;

            auto numFrames = pMBLayout->GetNumSequenceFramesInCurrentMB(sequence);

            if (numFrames > 0)
            {
                totalframeNum += numFrames;

                auto columnIndices = pMBLayout->GetColumnIndices(sequence);

                ExtractSampleSequence(firstSeq, columnIndices, squashInputs, tokensToIgnore, firstSeqVec);
                ExtractSampleSequence(secondSeq, columnIndices, squashInputs, tokensToIgnore, secondSeqVec);

                //calculate edit distance
                size_t firstSize = firstSeqVec.size();
                totalSampleNum += firstSize;
                size_t secondSize = secondSeqVec.size();
                grid.Resize(firstSize + 1, secondSize + 1);
                insMatrix.Resize(firstSize + 1, secondSize + 1);
                delMatrix.Resize(firstSize + 1, secondSize + 1);
                subMatrix.Resize(firstSize + 1, secondSize + 1);
                insMatrix.SetValue(0.0f);
                delMatrix.SetValue(0.0f);
                subMatrix.SetValue(0.0f);

                for (size_t i = 0; i < firstSize + 1; i++)
                {
                    grid(i, 0) = (float)(i * delPen);
                    delMatrix(i, 0) = (float)i;
                }

                for (size_t j = 0; j < secondSize + 1; j++)
                {
                    grid(0, j) = (float)(j * insPen);
                    insMatrix(0, j) = (float)j;
                }
                for (size_t i = 1; i < firstSize + 1; i++)
                {
                    for (size_t j = 1; j < secondSize + 1; j++)
                    {
                        if (firstSeqVec[i - 1] == secondSeqVec[j - 1])
                        {
                            grid(i, j) = grid(i - 1, j - 1);
                            insMatrix(i, j) = insMatrix(i - 1, j - 1);
                            delMatrix(i, j) = delMatrix(i - 1, j - 1);
                            subMatrix(i, j) = subMatrix(i - 1, j - 1);
                        }
                        else
                        {
                            del = grid(i - 1, j) + delPen; //deletion 
                            ins = grid(i, j - 1) + insPen;  //insertion
                            sub = grid(i - 1, j - 1) + subPen; //substitution 
                            if (sub <= del && sub <= ins)
                            {
                                insMatrix(i, j) = insMatrix(i - 1, j - 1);
                                delMatrix(i, j) = delMatrix(i - 1, j - 1);
                                subMatrix(i, j) = subMatrix(i - 1, j - 1) + 1.0f;
                                grid(i, j) = sub;
                            }
                            else if (del < ins)
                            {
                                insMatrix(i, j) = insMatrix(i - 1, j);
                                subMatrix(i, j) = subMatrix(i - 1, j);
                                delMatrix(i, j) = delMatrix(i - 1, j) + 1.0f;
                                grid(i, j) = del;
                            }
                            else
                            {
                                delMatrix(i, j) = delMatrix(i, j - 1);
                                subMatrix(i, j) = subMatrix(i, j - 1);
                                insMatrix(i, j) = insMatrix(i, j - 1) + 1.0f;
                                grid(i, j) = ins;
                            }
                        }
                    }
                }

                wrongSampleNum += insMatrix(firstSize, secondSize) + delMatrix(firstSize, secondSize) + subMatrix(firstSize, secondSize);
            }

            sequenceStartFrame += numFrames;
        }

        return (ElemType)(wrongSampleNum * totalframeNum / totalSampleNum);
    }

    float SubstitutionPenalty() const { return m_SubPen; }
    float DeletionPenalty() const { return m_DelPen; }
    float InsertionPenalty() const { return m_InsPen; }
    bool SquashInputs() const { return m_SquashInputs; }
    std::vector<size_t> TokensToIgnore() const { return m_tokensToIgnore; }

private:
    shared_ptr<Matrix<ElemType>> m_maxIndexes0, m_maxIndexes1;
    shared_ptr<Matrix<ElemType>> m_maxValues;
    bool m_SquashInputs;
    float m_SubPen;
    float m_DelPen;
    float m_InsPen;
    std::vector<size_t> m_tokensToIgnore;

    // Clear out_SampleSeqVec and extract a vector of samples from the matrix into out_SampleSeqVec.
    static void ExtractSampleSequence(const Matrix<ElemType>& firstSeq, vector<size_t>& columnIndices, bool squashInputs, const vector<size_t>& tokensToIgnore, std::vector<int>& out_SampleSeqVec)
    {
        out_SampleSeqVec.clear();

        // Get the first element in the sequence
        size_t lastId = (int)firstSeq(0, columnIndices[0]);
        if (std::find(tokensToIgnore.begin(), tokensToIgnore.end(), lastId) == tokensToIgnore.end())
            out_SampleSeqVec.push_back(lastId);

        // Remaining elements
        if (squashInputs)
        {
            //squash sequences of identical samples
            for (size_t i = 1; i < columnIndices.size(); i++)
            {
                size_t refId = (int)firstSeq(0, columnIndices[i]);
                if (lastId != refId)
                {
                    lastId = refId;
                    if (std::find(tokensToIgnore.begin(), tokensToIgnore.end(), refId) == tokensToIgnore.end())
                        out_SampleSeqVec.push_back(refId);
                }
            }
        }
        else
        {
            for (size_t i = 1; i < columnIndices.size(); i++)
            {
                auto refId = (int)firstSeq(0, columnIndices[i]);
                if (std::find(tokensToIgnore.begin(), tokensToIgnore.end(), refId) == tokensToIgnore.end())
                    out_SampleSeqVec.push_back(refId);
            }
        }
    }
};

template class EditDistanceErrorNode<float>;
template class EditDistanceErrorNode<double>;

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
        DecideStartEndingOutputLab(InputRef(0).Value(), mStartLab, mEndLab);
        ForwardPropS(mAlpha, mBacktrace, Value(), InputRef(1).Value(),
                     InputRef(2).Value(), mStartLab, mEndLab);
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
