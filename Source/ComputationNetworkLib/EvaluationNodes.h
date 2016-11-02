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

template<class ElemType>
class PhoneErrorNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>
{
    typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName() { return L"PhoneError"; }
public:
    DeclareConstructorFromConfig(PhoneErrorNode);
    PhoneErrorNode(DEVICEID_TYPE deviceId, const wstring & name)
        : Base(deviceId, name)
    {
    }

    virtual void BackpropToNonLooping(size_t /*inputIndex*/) override
    {
        LogicError("%ls operation is used for evaluation only.", OperationName().c_str());
    }


    virtual void ForwardPropNonLooping() override
    {
        size_t sequenceNum = Input(1)->GetNumParallelSequences();
        FrameRange frameRange(Input(0)->GetMBLayout());
        Input(0)->ValueFor(frameRange).VectorMax(*m_maxIndexes0, *m_maxValues, true);
        Input(1)->ValueFor(frameRange).VectorMax(*m_maxIndexes1, *m_maxValues, true);

        MaskMissingColumnsToZero(*m_maxIndexes0, Input(0)->GetMBLayout(), frameRange);
        MaskMissingColumnsToZero(*m_maxIndexes1, Input(1)->GetMBLayout(), frameRange);
        CalErrorphoneWER(Value(), *m_maxIndexes0, *m_maxIndexes1, sequenceNum, Input(0)->GetMBLayout(), Input(0)->Value().GetNumRows(), m_blanknum);
#if NANCHECK
        FunctionValues().HasNan("ErrorPrediction");
#endif
#if DUMPOUTPUT
        FunctionValues().Print("ErrorPredictionNode");
#endif


        /*EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_maxIndexes0, m_maxIndexes1, m_maxValues, shared_from_this(),
        Inputs(0)->GetMBLayout(), sequenceNum, m_blanknum);*/
    }





    virtual void Validate(bool isFinalValidationPass) override
    {
        ValidateBinaryReduce(isFinalValidationPass);

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

    


    virtual void CopyTo(ComputationNodeBasePtr  nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);


        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<PhoneErrorNode<ElemType>>(nodeP);
            node->m_maxIndexes0 = m_maxIndexes0;
            node->m_maxIndexes1 = m_maxIndexes1;
            node->m_maxValues = m_maxValues;
            node->m_blanknum = m_blanknum;
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


    void SetBlankNum(const size_t blanknum)
    {
        m_blanknum = blanknum;
    }
protected:
    virtual bool NodeDoesItsOwnCustomizedMissingColumnsMasking() { return true; }

private:
    static void CalErrorphoneWER(Matrix<ElemType>& functionvalue, Matrix<ElemType>& refframeseq, Matrix<ElemType> & outputframeseq, size_t samplesInRecurrentStep, MBLayoutPtr pMBLayout,
        size_t allphonenum, size_t blanknum)
    {
        std::vector<size_t> labelseq, outputseq;
        static const int subPen = 10;     /* error penalties */
        static const int delPen = 7;
        static const int insPen = 7;

        Matrix<float> grid(CPUDEVICE);
        Matrix<float> insmatrix(CPUDEVICE);
        Matrix<float> delmatrix(CPUDEVICE);
        Matrix<float> submatrix(CPUDEVICE);

        float d, h, v;
        //size_t fullcolsize = refframeseq.GetNumCols() / samplesInRecurrentStep;
        ElemType wrongPhoneNum = 0.0;
        size_t totalphonenum = 0, totalframenum = 0;
        size_t mbsize = refframeseq.GetNumCols() / samplesInRecurrentStep;
        size_t lastsentend = 0;
        size_t FrameNum = 0;
        size_t j = 0;
        for (size_t nchannel = 0; nchannel < samplesInRecurrentStep; nchannel++)
        {
            //zhaorui for CTC merge
            lastsentend = 0;
            j = 0;
            while (j < mbsize)
            {
                FrameNum = 0;
                for (j = lastsentend; j < mbsize; j++)
                {
                    if (pMBLayout->IsEnd(nchannel, j))
                    {
                        FrameNum = j - lastsentend + 1;
                        break;
                        //validframes.push_back(j + 1);
                    }
                }
                if (FrameNum > 0)
                {
                    totalframenum += FrameNum;
                    size_t lastid = 65535;
                    labelseq.clear();
                    //merge same phone id
                    for (size_t i = lastsentend; i < FrameNum + lastsentend; i++)
                    {
                        if (lastid != refframeseq(0, i * samplesInRecurrentStep + nchannel))
                        {
                            lastid = (size_t)refframeseq(0, i * samplesInRecurrentStep + nchannel);
                            if (lastid < allphonenum - blanknum)  //hard code
                                labelseq.push_back(lastid);
                        }
                        /*lastid = (size_t)refframeseq(0, i * samplesInRecurrentStep + nchannel);

                        if (lastid != 65535)
                        labelseq.push_back(lastid);*/
                    }
                    outputseq.clear();
                    lastid = 65535;
                    for (size_t i = lastsentend; i < FrameNum + lastsentend; i++)
                    {
                        if (lastid != outputframeseq(0, i * samplesInRecurrentStep + nchannel))
                        {
                            lastid = (size_t)outputframeseq(0, i * samplesInRecurrentStep + nchannel);
                            if (lastid < allphonenum - blanknum)  //hard code
                                outputseq.push_back(lastid);
                        }
                    }

                    //calcualate phone error rate
                    size_t labelsize = labelseq.size();
                    totalphonenum += labelsize;
                    size_t outputsize = outputseq.size();
                    grid.Resize(labelsize + 1, outputsize + 1);
                    insmatrix.Resize(labelsize + 1, outputsize + 1);
                    delmatrix.Resize(labelsize + 1, outputsize + 1);
                    submatrix.Resize(labelsize + 1, outputsize + 1);
                    insmatrix.SetValue(0.0f);
                    delmatrix.SetValue(0.0f);
                    submatrix.SetValue(0.0f);


                    for (size_t i = 0; i < labelsize + 1; i++){
                        grid(i, 0) = (float)(i * delPen);
                        delmatrix(i, 0) = (float)i;
                    }
                    fprintf(stderr, "label: ");
                    for (size_t i = 0; i < labelsize; i++){
                        fprintf(stderr, "%d\t", (int)labelseq[i]);
                    }
                    fprintf(stderr, "\n");

                    fprintf(stderr, "output: ");
                    for (size_t i = 0; i < outputsize; i++){
                        fprintf(stderr, "%d\t", (int)outputseq[i]);
                    }
                    fprintf(stderr, "\n");

                    for (size_t j = 0; j < outputsize + 1; j++) {
                        grid(0, j) = (float)(j * insPen);
                        insmatrix(0, j) = (float)j;
                    }
                    for (size_t i = 1; i < labelsize + 1; i++){
                        for (size_t j = 1; j < outputsize + 1; j++) {

                            if (labelseq[i - 1] == outputseq[j - 1])
                            {
                                grid(i, j) = grid(i - 1, j - 1);
                                insmatrix(i, j) = insmatrix(i - 1, j - 1);
                                delmatrix(i, j) = delmatrix(i - 1, j - 1);
                                submatrix(i, j) = submatrix(i - 1, j - 1);

                            }
                            else
                            {
                                d = grid(i - 1, j) + (float)delPen; //deletion 
                                h = grid(i, j - 1) + (float)insPen;  //insertion
                                v = grid(i - 1, j - 1) + (float)subPen; //substitution 
                                if (v <= d && v <= h)
                                {
                                    insmatrix(i, j) = insmatrix(i - 1, j - 1);
                                    delmatrix(i, j) = delmatrix(i - 1, j - 1);

                                    submatrix(i, j) = submatrix(i - 1, j - 1) + 1.0f;
                                    grid(i, j) = v;

                                }
                                else if (d < h)
                                {
                                    insmatrix(i, j) = insmatrix(i - 1, j);
                                    submatrix(i, j) = submatrix(i - 1, j);
                                    delmatrix(i, j) = delmatrix(i - 1, j) + 1.0f;
                                    grid(i, j) = d;
                                }
                                else
                                {
                                    delmatrix(i, j) = delmatrix(i, j - 1);
                                    submatrix(i, j) = submatrix(i, j - 1);
                                    insmatrix(i, j) = insmatrix(i, j - 1) + 1.0f;
                                    grid(i, j) = h;
                                }
                            }
                        }
                    }
                    wrongPhoneNum += insmatrix(labelsize, outputsize) + delmatrix(labelsize, outputsize) + submatrix(labelsize, outputsize);
                    }
                lastsentend += FrameNum;
                if (lastsentend < mbsize)
                {
                    FrameRange fr(pMBLayout, lastsentend);
                    if (pMBLayout->IsGap(fr.Sequence(nchannel)))
                        break;
                }
                }
            }

        functionvalue(0, 0) = (ElemType)(wrongPhoneNum * totalframenum / totalphonenum);

        }

    shared_ptr<Matrix<ElemType>> m_maxIndexes0, m_maxIndexes1;
    shared_ptr<Matrix<ElemType>> m_maxValues;
    size_t m_blanknum=1;
    int m_topK;
    };

template class PhoneErrorNode<float>;
template class PhoneErrorNode<double>;


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
