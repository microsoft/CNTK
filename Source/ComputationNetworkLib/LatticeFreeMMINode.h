//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "ComputationNode.h"
#include "gammacalculation.h"
#include <float.h>

#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>

#include <iostream>
#include <cuda_runtime.h>
#include <cusparse.h>
#include <cublas_v2.h>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

// -----------------------------------------------------------------------
// LatticeFreeMMINode
// -----------------------------------------------------------------------

template <class ElemType>
class LatticeFreeMMINode : public ComputationNodeNonLooping /*ComputationNode*/<ElemType>, public NumInputs<3>
{
    typedef ComputationNodeNonLooping<ElemType> Base;
    UsingComputationNodeMembersBoilerplate;
    static const std::wstring TypeName()
    {
        return L"LatticeFreeMMI";
    }

    void InitTransitionMatrixes()
    {
        CreateMatrixIfNull(m_tmap);
        CreateMatrixIfNull(m_smap);
    }

    void InitializeFromTfstFiles(const wstring& fstFilePath, const wstring& smapFilePath, bool useSenoneLM, const wstring& transFilePath);

    inline double Logadd(double a, double b) {
        if (b > a) {
            const double tmp = a;
            a = b;
            b = tmp;
        }
        if (b - a >= DBL_MIN_EXP)
            a += log1p(exp(b - a));
        return a;
    }
    
public:
    LatticeFreeMMINode(DEVICEID_TYPE deviceId, const wstring& name)
        : Base(deviceId, name), m_acweight(1.0), m_usePrior(true), m_alignmentWindow(0), m_ceweight(0), m_l2NormFactor(0)
    {
        InitTransitionMatrixes();
    }

    LatticeFreeMMINode(DEVICEID_TYPE deviceId, const wstring& name, const wstring& fstFilePath, const wstring& smapFilePath, const ElemType acweight, const bool usePrior, const int alignmentWindow, const ElemType ceweight, const ElemType l2NormFactor, const bool useSenoneLM, const wstring& transFilePath)
        : Base(deviceId, name), m_acweight(acweight), m_usePrior(usePrior), m_alignmentWindow(alignmentWindow), m_ceweight(ceweight), m_l2NormFactor(l2NormFactor)
    {
        InitTransitionMatrixes();
        InitializeFromTfstFiles(fstFilePath, smapFilePath, useSenoneLM, transFilePath);
    }

    LatticeFreeMMINode(const ScriptableObjects::IConfigRecordPtr configp)
        : LatticeFreeMMINode(configp->Get(L"deviceId"), L"<placeholder>", configp->Get(L"fstFilePath"), configp->Get(L"smapFilePath"), configp->Get(L"acweight"), configp->Get(L"usePrior"), configp->Get(L"alignmentWindow"), configp->Get(L"ceweight"), configp->Get(L"l2NormFactor"), configp->Get(L"useSenoneLM"), configp->Get(L"transFilePath"))
    {
        AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
    }

    virtual void BackpropToNonLooping(size_t inputIndex) override
    {
        if (inputIndex == 1)
        {
            FrameRange fr(Input(0)->GetMBLayout());
            auto gradient = Input(1)->GradientFor(fr);
            // k * (1-alpha) * r_DEN + alpha * P_net - (k * (1-alpha) + alpha) * r_NUM + c * y
            if (m_ceweight != 0)
            {
                m_softmax->InplaceExp();
                Matrix<ElemType>::ScaleAndAdd(m_ceweight, *m_softmax, m_acweight * (1 - m_ceweight), *m_posteriorsDen);
                Matrix<ElemType>::Scale(m_acweight * (1 - m_ceweight) + m_ceweight, *m_posteriorsNum);
            }
            if (m_l2NormFactor != 0)
            {
                Matrix<ElemType>::ScaleAndAdd(m_l2NormFactor, Input(1)->ValueFor(fr), *m_posteriorsDen);
            }

            Matrix<ElemType>::AddScaledDifference(Gradient(), *m_posteriorsDen, *m_posteriorsNum, gradient);
        }
    }

    virtual bool OutputUsedInComputingInputNodesGradients() const override
    {
        return false;
    }

#ifdef _DEBUG
    void SaveMatrix(wchar_t *fileName, const Matrix<ElemType>& m) const
    {
        FILE *fin = _wfopen(fileName, L"w");
        fprintf(fin, "%d %d\n", m.GetNumRows(), m.GetNumCols());
        for (int i = 0; i < m.GetNumRows(); i++){
            for (int j = 0; j < m.GetNumCols(); j++){
                fprintf(fin, "%e\n", m.GetValue(i, j));
            }
        }
        fclose(fin);
    }
#endif

    void GetLabelSequence(const Matrix<ElemType>& labelMatrix)
    {
        labelMatrix.VectorMax(*m_maxLabelIndexes, *m_maxLabelValues, true);
        assert(m_maxLabelIndexes->GetNumRows() == 1);

        size_t size = m_maxLabelIndexes->GetNumCols();
        m_labelVector.resize(size);

        ElemType* resultPointer = &m_labelVector[0];
        m_maxLabelIndexes->CopyToArray(resultPointer, size);
    }

    double CalculateNumeratorsWithCE(const Matrix<ElemType>& labelMatrix, const size_t nf);

    double ForwardBackwardProcessForDenorminator(const size_t nf, Matrix<ElemType>& posteriors,
        const Matrix<ElemType>& tmap, const Matrix<ElemType>& tmapTranspose, const Matrix<ElemType>& smap, const Matrix<ElemType>& smapTranspose);

    virtual void /*ComputationNodeNonLooping::*/ ForwardPropNonLooping() override
    {
        if (!m_tmapTranspose)
            m_tmapTranspose = make_shared<Matrix<ElemType>>(m_tmap->Transpose(), m_deviceId);
        if (!m_smapTranspose)
            m_smapTranspose = make_shared<Matrix<ElemType>>(m_smap->Transpose(), m_deviceId);

        FrameRange fr(Input(0)->GetMBLayout());

        // first compute the softmax (column-wise)
        // Note that we need both log and non-log for gradient computation.
        m_likelihoods->AssignLogSoftmaxOf(Input(1)->ValueFor(fr), true);
        if (m_ceweight != 0)
            m_softmax->SetValue(*m_likelihoods);

        if (m_usePrior)
            (*m_likelihoods) -= Input(2)->ValueAsMatrix();

        if (m_acweight != (ElemType)1.0)    // acoustic squashing factor
            (*m_likelihoods) *= m_acweight;

        m_likelihoods->InplaceExp(); // likelihood
        (*m_likelihoods) += (ElemType)1e-15;

        size_t nf = m_likelihoods->GetNumCols();
        double logNumeratorWithCE = CalculateNumeratorsWithCE(Input(0)->MaskedValueFor(fr), nf);
        double logDenominator = ForwardBackwardProcessForDenorminator(nf, *m_posteriorsDen, *m_tmap, *m_tmapTranspose, *m_smap, *m_smapTranspose);

        double l2NormScore = 0;
        if (m_l2NormFactor != 0)
        {
            l2NormScore = Matrix<ElemType>::InnerProductOfMatrices(Input(1)->ValueFor(fr), Input(1)->ValueFor(fr)) * 0.5 * m_l2NormFactor;
        }

        // Got the final numbers
        ElemType finalValue = (ElemType)((1 - m_ceweight) * logDenominator - logNumeratorWithCE + l2NormScore);
        Value().Resize(1, 1);
        Value().SetValue(finalValue);

#ifdef _DEBUG
        //SaveMatrix(L"D:\\temp\\LFMMI\\testoutput\\p.txt", *m_posteriorsDen);
        cout << "value: " << Value().GetValue(0, 0) << endl;
#endif

#if NANCHECK
        Value().HasNan("LatticeFreeMMI");
#endif
#if DUMPOUTPUT
        Value().Print("LatticeFreeMMINode");
#endif

    }

    virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
    {
        ValidateBinaryReduce(isFinalValidationPass);
        //Base::Validate(isFinalValidationPass);
        //InferMBLayoutFromInputsForStandardCase(isFinalValidationPass);
        //let shape0 = GetInputSampleLayout(1);
        //SmallVector<size_t> dims = shape0.GetDims();
        //SetDims(TensorShape(dims), HasMBLayout());
        if (isFinalValidationPass)
        {
            //auto r0 = Input(0)->GetSampleMatrixNumRows();
            //auto r3 = Input(3)->ValueAsMatrix().GetNumRows();
            //auto r4 = Input(4)->ValueAsMatrix().GetNumRows();
            //auto c3 = Input(3)->ValueAsMatrix().GetNumCols();
            //auto c4 = Input(4)->ValueAsMatrix().GetNumCols();
            //if (r0 != r4 || c3 != r3 || c3 != c4)
            //    LogicError("The Matrix dimension in the LatticeFreeMMINode operation does not match.");
        }
    }

    virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
    {
        Base::CopyTo(nodeP, newName, flags);
        if (flags & CopyNodeFlags::copyNodeValue)
        {
            auto node = dynamic_pointer_cast<LatticeFreeMMINode<ElemType>>(nodeP);
            node->m_acweight = m_acweight;
            node->m_usePrior = m_usePrior;
            node->m_alignmentWindow = m_alignmentWindow;
            node->m_ceweight = m_ceweight;
            node->m_l2NormFactor = m_l2NormFactor;
            node->m_fsa = m_fsa;
            node->m_tmap->SetValue(*m_tmap);
            node->m_smap->SetValue(*m_smap);
            node->m_tmapTranspose->SetValue(*m_tmapTranspose);
            node->m_smapTranspose->SetValue(*m_smapTranspose);
        }
    }

    // request matrices needed to do node function value evaluation
    virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
    {
        Base::RequestMatricesBeforeForwardProp(matrixPool);
        RequestMatrixFromPool(m_currAlpha, matrixPool);
        RequestMatrixFromPool(m_nextAlpha, matrixPool);
        RequestMatrixFromPool(m_alphas, matrixPool);
        RequestMatrixFromPool(m_obsp, matrixPool);
        RequestMatrixFromPool(m_likelihoods, matrixPool);
        RequestMatrixFromPool(m_posteriorsDen, matrixPool);

        RequestMatrixFromPool(m_maxLabelIndexes, matrixPool);
        RequestMatrixFromPool(m_maxLabelValues, matrixPool);
        RequestMatrixFromPool(m_posteriorsNum, matrixPool);
        if (m_ceweight != 0)
            RequestMatrixFromPool(m_softmax, matrixPool);
    }

    virtual void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterForwardProp(matrixPool);
        ReleaseMatrixToPool(m_currAlpha, matrixPool);
        ReleaseMatrixToPool(m_nextAlpha, matrixPool);
        ReleaseMatrixToPool(m_alphas, matrixPool);
        ReleaseMatrixToPool(m_obsp, matrixPool);
        ReleaseMatrixToPool(m_likelihoods, matrixPool);
        ReleaseMatrixToPool(m_maxLabelIndexes, matrixPool);
        ReleaseMatrixToPool(m_maxLabelValues, matrixPool);
    }

    // request matrices needed to do node function value evaluation
    virtual void ReleaseMatricesAfterBackprop(MatrixPool& matrixPool)
    {
        Base::ReleaseMatricesAfterBackprop(matrixPool);
        ReleaseMatrixToPool(m_posteriorsDen, matrixPool);
        ReleaseMatrixToPool(m_posteriorsNum, matrixPool);
        if (m_ceweight != 0)
            ReleaseMatrixToPool(m_softmax, matrixPool);
    }

    void SaveFsa(File& fstream) const
    {
        fstream.PutMarker(fileMarkerBeginSection, std::wstring(L"BFSA"));
        fstream << m_fsa.size();
        for (int i = 0; i < m_fsa.size(); i++)
        {
            map<int, pair<int, ElemType>> map = m_fsa[i];
            fstream << map.size();
            for (auto const &it : map)
            {
                fstream << it.first;
                fstream << it.second.first;
                fstream << it.second.second;
            }
        }
        fstream.PutMarker(fileMarkerEndSection, std::wstring(L"EFSA"));
    }

    void LoadFsa(File& fstream)
    {
        fstream.GetMarker(fileMarkerBeginSection, std::wstring(L"BFSA"));
        m_fsa.clear();
        size_t size;
        fstream >> size;
        for (int i = 0; i < size; i++)
        {
            map<int, pair<int, ElemType>> map;
            size_t mapSize;
            fstream >> mapSize;
            for (int j = 0; j < mapSize; j++)
            {
                int a1, b1;
                ElemType b2;
                fstream >> a1;
                fstream >> b1;
                fstream >> b2;
                map[a1] = make_pair(b1, b2);
            }

            m_fsa.push_back(map);
        }
        fstream.GetMarker(fileMarkerEndSection, std::wstring(L"EFSA"));
    }

    virtual void Save(File& fstream) const override
    {
        Base::Save(fstream);
        fstream << m_acweight;
        fstream << m_usePrior;
        fstream << m_alignmentWindow;
        fstream << m_ceweight;
        fstream << m_l2NormFactor;
        fstream << *m_tmap;
        fstream << *m_smap;
        SaveFsa(fstream);
    }

    void LoadMatrix(File& fstream, shared_ptr<Matrix<ElemType>>& matrixPtr)
    {
        CreateMatrixIfNull(matrixPtr);
        fstream >> *matrixPtr;
        // above reads dimensions, so we must update our own dimensions
        SetDims(TensorShape(matrixPtr->GetNumRows(), matrixPtr->GetNumCols()), false);
    }

    virtual void Load(File& fstream, size_t modelVersion) override
    {
        Base::Load(fstream, modelVersion);
        fstream >> m_acweight;
        fstream >> m_usePrior;
        fstream >> m_alignmentWindow;
        fstream >> m_ceweight;
        fstream >> m_l2NormFactor;
        LoadMatrix(fstream, m_tmap);
        LoadMatrix(fstream, m_smap);
        m_tmapTranspose = make_shared<Matrix<ElemType>>(m_tmap->Transpose(), m_deviceId);
        m_smapTranspose = make_shared<Matrix<ElemType>>(m_smap->Transpose(), m_deviceId);
        LoadFsa(fstream);
    }

    virtual void DumpNodeInfo(const bool printValues, const bool printMetadata, File& fstream) const override
    {
        if (printMetadata)
        {
            Base::DumpNodeInfo(printValues, printMetadata, fstream);

            char str[4096];
            sprintf(str, "acweight=%f usePrior=%d alignmentWindow=%d ceweight=%f l2NormFactor=%f", this->m_acweight, this->m_usePrior, this->m_alignmentWindow, this->m_ceweight, this->m_l2NormFactor);
            fstream << string(str);
        }

        PrintNodeValuesToFile(printValues, printMetadata, fstream);
    }

private: 
    struct DataArc{
        int From;
        int To;
        int Senone;
        ElemType Cost;
    };

    struct SenoneLabel{
        int Senone;
        int Begin;
        int End;
    };

    struct arc {
        int source;
        int destination;    // destination state
        int label;            // 0..N for acoustic leaf labels
        int statenum;  // the id of the arc
        ElemType lm_cost;  // from the graph
        ElemType logsp, logfp; // log of self and forward loop probabilities
    };

private:
    void Graph2matrixWithSelfLoop(vector<DataArc> input, size_t maxstate, vector<ElemType>& transVal, vector<CPUSPARSE_INDEX_TYPE>& transRow, vector<CPUSPARSE_INDEX_TYPE>& transCol, size_t &nstates, size_t &transCount, vector<ElemType>& smapVal, vector<CPUSPARSE_INDEX_TYPE>& smapRow, vector<CPUSPARSE_INDEX_TYPE>& smapCol, size_t &smapCount, size_t numSenone, vector<map<int, pair<int, ElemType>>>& fsa);
    void Graph2matrix(vector<DataArc> input, vector<ElemType>& transVal, vector<CPUSPARSE_INDEX_TYPE>& transRow, vector<CPUSPARSE_INDEX_TYPE>& transCol, size_t &nstates, size_t &transCount, vector<ElemType>& smapVal, vector<CPUSPARSE_INDEX_TYPE>& smapRow, vector<CPUSPARSE_INDEX_TYPE>& smapCol, size_t &smapCount, size_t numSenone, vector<map<int, pair<int, ElemType>>>& fsa, const wstring &transFilePath);
    
    void Read_senone_map(const wchar_t *infile, map<string, int> &idx4senone) {
        FILE *fin = fopenOrDie(infile, L"r");
        const int slen = 1000;
        char buff[slen];
        int snum = 0;
        while (fscanf(fin, "%s", buff) == 1) {
            char *p = strchr(buff, '.');
            if (p)
                *p = '_';  // convert Jasha's "." to an "_" for consistency with the graph
            string sn(buff);
            sn = "[" + sn + "]";
            assert(!idx4senone.count(sn)); // each should only be listed once
            idx4senone[sn] = snum++;
        }
        fclose(fin);
    }

    vector<DataArc> LoadTfstFile(const wchar_t *infile, map<string, int> &idx4senone, int &maxstate) const
    {
        FILE *fin = fopenOrDie(infile, L"r");
        vector<DataArc> input;
        const int llen = 1000;
        char line[llen];
        maxstate = 0;
        while (fgets(line, llen, fin))
        {
            if (line[0] == '#')
                continue;
            char f1[100], f2[100], f3[100], f4[100];
            DataArc arc;
            int num_cols = sscanf(line, "%s %s %s %s", f1, f2, f3, f4);
            arc.From = stoi(f1);
            if (num_cols <= 2)
            {
                arc.Senone = -1;
                arc.Cost = pow(10.0f, (num_cols == 1) ? (0.0f) : ((ElemType)-stof(f2)));
            }
            else
            {
                assert(f3[0] == '[');  // in this program, reading a specialized graph with no epsilons
                arc.To = stoi(f2);
                arc.Cost = pow(10.0f, (num_cols == 3) ? (0.0f) : ((ElemType)-stof(f4)));
                assert(idx4senone.count(f3));  // should be on the statelist or there is a AM/graph mismatch
                arc.Senone = idx4senone[f3];
            }
            input.push_back(arc);
            if (arc.From > maxstate) maxstate = arc.From;
        }

        fclose(fin);
        return input;
    }

protected:
    ElemType m_acweight;
    bool m_usePrior;
    int m_alignmentWindow;
    ElemType m_ceweight;
    ElemType m_l2NormFactor;
    vector<map<int, pair<int, ElemType>>> m_fsa;
    shared_ptr<Matrix<ElemType>> m_tmap;
    shared_ptr<Matrix<ElemType>> m_smap;
    shared_ptr<Matrix<ElemType>> m_tmapTranspose;
    shared_ptr<Matrix<ElemType>> m_smapTranspose;
    shared_ptr<Matrix<ElemType>> m_currAlpha;
    shared_ptr<Matrix<ElemType>> m_nextAlpha;
    shared_ptr<Matrix<ElemType>> m_alphas;
    shared_ptr<Matrix<ElemType>> m_obsp;
    shared_ptr<Matrix<ElemType>> m_maxLabelIndexes;
    shared_ptr<Matrix<ElemType>> m_maxLabelValues;
    shared_ptr<Matrix<ElemType>> m_posteriorsNum;
    shared_ptr<Matrix<ElemType>> m_posteriorsDen;
    shared_ptr<Matrix<ElemType>> m_likelihoods;

    // For CE
    shared_ptr<Matrix<ElemType>> m_softmax;

    vector<ElemType> m_labelVector;
    vector<ElemType> m_likelihoodBuffer;
    vector<SenoneLabel> m_senoneSequence;
    vector<int> m_stateSequence;
    vector<double> m_alphaNums;
    vector<double> m_betas;
    vector<double> m_betasTemp;
    vector<ElemType> m_posteriorsAtHost;
    vector<ElemType> m_obspAtHost;
    vector<ElemType> m_initialAlpha;
};

} } }
