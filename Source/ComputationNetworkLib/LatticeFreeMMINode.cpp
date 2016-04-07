//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "Basics.h"
#include "ComputationNode.h"
#include "gammacalculation.h"

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

#include "LatticeFreeMMINode.h"

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {
    
template <class ElemType>
void LatticeFreeMMINode<ElemType>::Graph2matrixWithSelfLoop(const vector<DataArc> input, size_t maxstate, vector<ElemType>& transVal, vector<CPUSPARSE_INDEX_TYPE>& transRow, vector<CPUSPARSE_INDEX_TYPE>& transCol, size_t &nstates, size_t &transCount, vector<ElemType>& smapVal, vector<CPUSPARSE_INDEX_TYPE>& smapRow, vector<CPUSPARSE_INDEX_TYPE>& smapCol, size_t &smapCount, size_t numSenone, vector<map<int, pair<int, ElemType>>>& fsa)
{
    cout << "Loading with simple format" << endl;
    // decoding graph and turns it into a transition matrix
    // all costs on input graph are expected to be negative log base 10
    smapCount = 0;
    maxstate++;

    fsa.clear();

    size_t finalstate = maxstate;
    // add a notional start state
    size_t count = 0;
    transRow.push_back(0);

    map<int, vector<int> > states4senone;
    set<int> countedState;
    int currentState = 0;
    map<int, pair<int, ElemType>> currentMap;
    for (auto dataArc : input)
    {
        if (dataArc.From != currentState)
        {
            transRow.push_back(count);
            fsa.push_back(currentMap);
            currentMap.clear();
            currentState = dataArc.From;
        }

        transVal.push_back(dataArc.Cost);
        if (dataArc.Senone < 0)
        {
            transCol.push_back(finalstate);
            currentMap[-1] = make_pair((int)finalstate, log(dataArc.Cost));
        }
        else
        {
            transCol.push_back(dataArc.To);
            currentMap[dataArc.Senone] = make_pair(dataArc.To, log(dataArc.Cost));
            if (!countedState.count(dataArc.To))
            {
                states4senone[dataArc.Senone].push_back(dataArc.To);
                countedState.insert(dataArc.To);
                smapCount++;
            }
        }
        count++;
    }
    transRow.push_back(count);
    transRow.push_back(count);

    fsa.push_back(currentMap);

    assert(transRow.back() == transVal.size());
    assert(transVal.size() == transCol.size());
    assert(transRow.size() == finalstate + 2);
    cout << "Transition matrix has " << (finalstate + 1) << " states and " << transVal.size() << " nozeros " << endl;

    nstates = finalstate + 1;
    transCount = (int)transVal.size();

    // map the matrix that maps from states to senones
    bool seen_all = true;
    smapRow.push_back(0);
    for (size_t s = 0, smp = 1; s < numSenone; s++, smp++) {
        if (states4senone.find((int)s) == states4senone.end()) {  // graph build w/ small vocab - not all senones present
            seen_all = false;
            smapRow.push_back(smapRow[smp - 1]);
            continue;
        }
        const vector<int> &states = states4senone[(int)s];
        for (size_t j = 0; j < states.size(); j++) {
            //assert(j == 0 || states[j]>states[j - 1]);
            assert(states[j] > 0 && states[j] < finalstate);
            smapVal.push_back(1.0);
            smapCol.push_back(states[j]);
        }
        smapRow.push_back(smapRow[smp - 1] + (int)states.size());
    }
    if (!seen_all) {
        cout << "Warning: not all senones present in graph" << endl;
    }
}


template <class ElemType>
void LatticeFreeMMINode<ElemType>::Graph2matrix(vector<DataArc> input, vector<ElemType>& transVal, vector<CPUSPARSE_INDEX_TYPE>& transRow, vector<CPUSPARSE_INDEX_TYPE>& transCol, size_t &nstates, size_t &transCount, vector<ElemType>& smapVal, vector<CPUSPARSE_INDEX_TYPE>& smapRow, vector<CPUSPARSE_INDEX_TYPE>& smapCol, size_t &smapCount, size_t numSenone, vector<map<int, pair<int, ElemType>>>& fsa, const wstring &transFilePath)
{
    // decoding graph and turns it into a transition matrix
    // all costs on input graph are expected to be negative log base 10

    map<int, pair<ElemType, ElemType>> translp4;
    bool hasTransProbFromFile = false;
    if (transFilePath != L"" && transFilePath != L"''") {
        hasTransProbFromFile = true;
        FILE *fin = fopenOrDie(transFilePath.c_str(), L"r");
        const int slen = 1000;
        char buff[slen];
        float slp, flp;
        int count = 0;
        while (fscanf(fin, "%s%f%f", buff, &slp, &flp) == 3) {
            assert(slp <= 0 && flp <= 0); // log-probs are negative
            translp4[count] = pair<ElemType, ElemType>((ElemType)exp(-slp), (ElemType)exp(-flp));
            count++;
        }
        fclose(fin);
    }

    map<int, vector<int> > states4senone;

    int arcstate = 1;  // state 0 will be the start state
    int start_state = -1;
    vector<vector<arc>> arcs;
    map<int, ElemType> cost4final_state;
    int curr_state = 0, fs = 0;
    arc ca;
    vector<arc> carcs;
    smapCount = 0;
    for (auto dataArc : input)
    {
        if (dataArc.Senone < 0)
        {
            int state = dataArc.From;
            fs = state;
            assert(cost4final_state.count(state) == 0);
            cost4final_state[state] = dataArc.Cost;
        }
        else
        {
            fs = dataArc.From;
            ca.source = dataArc.From;
            ca.destination = dataArc.To;
            ca.lm_cost = dataArc.Cost;
            ca.logsp = ca.logfp = 1.0;
            ca.label = dataArc.Senone;
            if (hasTransProbFromFile)
            {
                assert(translp4.count(dataArc.Senone) > 0);
                ca.logsp = translp4[dataArc.Senone].first;
                ca.logfp = translp4[dataArc.Senone].second;
            }
            ca.statenum = arcstate++;
            states4senone[dataArc.Senone].push_back(ca.statenum);
            smapCount++;
        }
        if (start_state < 0) start_state = dataArc.From;
        if (fs != curr_state) {
            assert(fs == curr_state + 1);  // att format ordered this way
            arcs.push_back(carcs);  // store the arcs for the previous state
            carcs.clear();
            curr_state = fs;
        }
        if (dataArc.Senone >= 0)
            carcs.push_back(ca);
    }

    // don't forget to store the arcs associated with the last state!
    arcs.push_back(carcs);  // store the arcs for the previous state

    assert(cost4final_state.size() != 0);
    assert(start_state == 0);

    const int finalstate = arcstate;  // make a fresh state to be the final state in the graph

    // now write the matrix

    // each arc becomes a state
    // the LM prob and forward transition probs are applied on transition out of the state
    // self-loop prob is applied for the on-diagonal transition

    fsa.clear();
    map<int, pair<int, ElemType>> currentMap;
    // add a notional start state
    assert(arcs[0].size());
    assert(arcs[0].front().source == 0);  // openfst should number the first state 0
    int counter = 0;
    for (size_t a = 0; a < arcs[0].size(); a++) {
        transVal.push_back(arcs[0][a].lm_cost);
        transCol.push_back(arcs[0][a].statenum);
        counter++;
        currentMap[arcs[0][a].label] = make_pair(arcs[0][a].statenum, log(arcs[0][a].lm_cost));
    }
    transRow.push_back(0);
    fsa.push_back(currentMap);

    // now add the rest
    for (size_t c = 0; c < arcs.size(); c++) {  // for each "from state"
        assert(arcs[c].size());
        for (size_t a = 0; a < arcs[c].size(); a++) {
            assert(arcs[c][a].statenum == fsa.size());
            transRow.push_back(counter);
            currentMap.clear();

            const arc &curr = arcs[c][a];
            assert(curr.statenum == transRow.size() - 1);
            const int dest = curr.destination;
            const vector<arc> &succs = arcs[dest];
            assert(succs.size());
            assert(succs[0].source == dest);  // should be 1-1 mapping of states to sets of successor arcs
            bool added_selfloop = false;
            for (size_t s = 0; s < succs.size(); s++) {
                if (s > 0)
                    assert(succs[s].statenum > succs[s - 1].statenum);
                if (succs[s].statenum > curr.statenum && !added_selfloop) {
                    transVal.push_back(curr.logsp);  // transition to curr.statenum
                    transCol.push_back(curr.statenum);
                    added_selfloop = true;
                    counter++;
                    currentMap[curr.label] = make_pair(curr.statenum, log(curr.logsp));
                }
                transVal.push_back(succs[s].lm_cost*curr.logfp);  // transition to succs[s].statenum
                transCol.push_back(succs[s].statenum);
                counter++;
                currentMap[succs[s].label] = make_pair(succs[s].statenum, log(succs[s].lm_cost*curr.logfp));
            }
            if (!added_selfloop) {
                transVal.push_back(curr.logsp);  // transition to curr.statenum
                transCol.push_back(curr.statenum);
                added_selfloop = true;
                counter++;
                currentMap[curr.label] = make_pair(curr.statenum, log(curr.logsp));
            }
            if (cost4final_state.count(dest)) {  // add transition to finalstate
                transVal.push_back(cost4final_state[dest] * curr.logfp);
                transCol.push_back(finalstate);
                counter++;
                currentMap[-1] = make_pair(finalstate, log(cost4final_state[dest] * curr.logfp));
            }
            fsa.push_back(currentMap);
        }
    }

    transRow.push_back(counter);
    transRow.push_back(counter);  // the final state has no transitions out, and no self-transition
    assert(transRow.back() == transVal.size());
    assert(transVal.size() == transCol.size());
    assert(transRow.size() == finalstate + 2);
    cout << "Transition matrix has " << (finalstate + 1) << " states and " << transVal.size() << " nozeros " << endl;

    nstates = finalstate + 1;
    transCount = transVal.size();

    assert(smapCount == arcstate - 1);  // -1 because arcstate 0 is a dummy start state

    bool seen_all = true;
    smapRow.push_back(0);
    for (size_t s = 0, smp = 1; s<numSenone; s++, smp++) {
        if (states4senone.find((int)s) == states4senone.end()) {  // graph build w/ small vocab - not all senones present
            seen_all = false;
            smapRow.push_back(smapRow[smp - 1]);
            continue;
        }
        const vector<int> &states = states4senone[(int)s];
        for (size_t j = 0; j < states.size(); j++) {
            assert(j == 0 || states[j]>states[j - 1]);
            assert(states[j] > 0 && states[j] < finalstate);
            smapVal.push_back(1.0);
            smapCol.push_back(states[j]);
        }
        smapRow.push_back(smapRow[smp - 1] + (int)states.size());
    }
    if (!seen_all) {
        cout << "Warning: not all senones present in graph" << endl;
    }
}

template <class ElemType>
void LatticeFreeMMINode<ElemType>::InitializeFromTfstFiles(const wstring& fstFilePath, const wstring& smapFilePath, bool useSenoneLM, const wstring& transFilePath)
{
    map<string, int> idx4senone;
    Read_senone_map(smapFilePath.c_str(), idx4senone);

    size_t nstates, transCount;
    vector<CPUSPARSE_INDEX_TYPE> transRow, transCol;
    vector<ElemType> transVal;

    // sparse matrix for the mapping from state to senone
    size_t nsenones, smapCount;
    vector<CPUSPARSE_INDEX_TYPE> smapRow, smapCol;
    vector<ElemType> smapVal;

    nsenones = (int)idx4senone.size();

    int maxstate;
    auto input = LoadTfstFile(fstFilePath.c_str(), idx4senone, maxstate);
    if (useSenoneLM)
        Graph2matrixWithSelfLoop(input, maxstate, transVal, transRow, transCol, nstates, transCount, smapVal, smapRow, smapCol, smapCount, nsenones, m_fsa);
    else
        Graph2matrix(input, transVal, transRow, transCol, nstates, transCount, smapVal, smapRow, smapCol, smapCount, nsenones, m_fsa, transFilePath);

    m_tmap->SwitchToMatrixType(SPARSE, matrixFormatSparseCSR, false);
    m_tmap->SetMatrixFromCSRFormat(&transRow[0], &transCol[0], &transVal[0], transCount, nstates, nstates);

    m_smap->SwitchToMatrixType(SPARSE, matrixFormatSparseCSR, false);
    m_smap->SetMatrixFromCSRFormat(&smapRow[0], &smapCol[0], &smapVal[0], smapCount, nsenones, nstates);
}

// If m_ceweight == 0, return the log numerator score of MMI
// Else, return (1-m_ceweight) * logNum - m_ceweight * logCE
template <class ElemType>
double LatticeFreeMMINode<ElemType>::CalculateNumeratorsWithCE(const Matrix<ElemType>& labelMatrix, const size_t nf)
{
    if (nf == 0) return 0;

    // Temp, hardalignment
    if (m_alignmentWindow == 0)
    {
        m_posteriorsNum->SetValue(labelMatrix);
        return Matrix<ElemType>::InnerProductOfMatrices(*m_likelihoods, labelMatrix);
    }

    size_t nsenones = labelMatrix.GetNumRows();
    GetLabelSequence(labelMatrix);
    assert(m_labelVector.size() == nf);

    // get labeled senone sequence
    m_senoneSequence.clear();
    m_stateSequence.clear();
    int lastState = 0;

    int index = 0;
    while (index < nf)
    {
        int currentSenone = (int)m_labelVector[index];
        int startIndex = index;
        index++;
        while (index < nf)
        {
            if ((int)m_labelVector[index] != currentSenone) break;
            index++;
        }

        int beginWithWindow = m_alignmentWindow < 0 ? 0 : std::max(0, startIndex - m_alignmentWindow);
        int endWithWindow = (m_alignmentWindow < 0 ? nf : std::min((int)nf, index + m_alignmentWindow)) - 1;
        m_senoneSequence.push_back({ currentSenone, beginWithWindow, endWithWindow });
        lastState = m_fsa[lastState][currentSenone].first;
        m_stateSequence.push_back(lastState);
    }

    // copy likelihoods to CPU
    size_t nstates = m_senoneSequence.size();
    size_t bufferSize = nsenones * nf;
    m_likelihoodBuffer.resize(bufferSize);
    ElemType* refArr = &m_likelihoodBuffer[0];
    m_likelihoods->CopyToArray(refArr, bufferSize);
    for (int i = 0; i < bufferSize; i++)
    {
        m_likelihoodBuffer[i] = log(m_likelihoodBuffer[i]);
    }

    m_alphaNums.clear();
    m_alphaNums.resize(nstates * nf, DBL_MIN_EXP);
    
    for (int i = 0; i < nf; i++)
    {
        if (i == 0)
        {
            int currentSenone = m_senoneSequence[0].Senone;
            m_alphaNums[0] = m_fsa[0][currentSenone].second + m_likelihoodBuffer[currentSenone];
        }
        else
        {
            for (int j = 0; j <= i, j < nstates; j++)
            {
                if (i < m_senoneSequence[j].Begin || i > m_senoneSequence[j].End) continue;
                if (j == 0)
                    m_alphaNums[i*nstates] = m_alphaNums[(i - 1) * nstates] + m_fsa[m_stateSequence[0]][m_senoneSequence[0].Senone].second + m_likelihoodBuffer[i * nsenones + m_senoneSequence[0].Senone];
                else
                {
                    int currentSenone = m_senoneSequence[j].Senone;
                    int baseIndex = (i - 1)*nstates + j;
                    assert(m_fsa[m_stateSequence[j]][currentSenone].second != 0);
                    assert(m_fsa[m_stateSequence[j - 1]][currentSenone].second != 0);
                    m_alphaNums[i * nstates + j] = Logadd(m_alphaNums[baseIndex] + m_fsa[m_stateSequence[j]][currentSenone].second, m_alphaNums[baseIndex - 1] + m_fsa[m_stateSequence[j - 1]][currentSenone].second)
                        + m_likelihoodBuffer[i * nsenones + currentSenone];
                }
            }
        }
    }

    double logForwardScore = m_alphaNums[nstates * nf - 1] + m_fsa[m_stateSequence[nstates - 1]][-1].second;
    if (std::isnan(logForwardScore))
        RuntimeError("logForwardScore for numerator should not be nan.");

    m_betas.clear();
    m_betas.resize(nstates, DBL_MIN_EXP);
    m_betasTemp.clear();
    m_betasTemp.resize(nstates, DBL_MIN_EXP);
    m_betas[nstates - 1] = m_fsa[m_stateSequence[nstates - 1]][-1].second;
    for (int i = nf - 1; i >= 0; i--)
    {
        double absum = DBL_MIN_EXP;
        for (int j = 0; j < nstates; j++)
        {
            double abTime = m_alphaNums[i * nstates + j] + m_betas[j];
            m_alphaNums[i * nstates + j] = abTime;
            absum = Logadd(absum, abTime);

            m_betasTemp[j] = m_betas[j] + m_likelihoodBuffer[i * nsenones + m_senoneSequence[j].Senone];
        }

        //assert(absum != -FLT_MAX);
        //cout << i << " : " << log(absum) << endl;
        for (int j = 0; j < nstates; j++)
        {
            m_alphaNums[i * nstates + j] -= absum;
        }

        if (i > 0)
        {
            for (int j = 0; j < nstates; j++)
            {
                if (i - 1 < m_senoneSequence[j].Begin || i - 1 > m_senoneSequence[j].End) m_betas[j] = DBL_MIN_EXP;
                else
                {
                    if (j < nstates - 1)
                    {
                        assert(m_fsa[m_stateSequence[j]][m_senoneSequence[j].Senone].second != 0);
                        assert(m_fsa[m_stateSequence[j]][m_senoneSequence[j + 1].Senone].second != 0);
                        m_betas[j] = Logadd(m_betasTemp[j] + m_fsa[m_stateSequence[j]][m_senoneSequence[j].Senone].second, m_betasTemp[j + 1] + m_fsa[m_stateSequence[j]][m_senoneSequence[j + 1].Senone].second);
                    }
                    else
                    {
                        assert(m_fsa[m_stateSequence[nstates - 1]][m_senoneSequence[nstates - 1].Senone].second != 0);
                        m_betas[nstates - 1] = m_betasTemp[nstates - 1] + m_fsa[m_stateSequence[nstates - 1]][m_senoneSequence[nstates - 1].Senone].second;
                    }
                }
            }
        }
    }

#ifdef _DEBUG
    cout << "log forward score: " << logForwardScore << endl;
    double logBackwardScore = m_betas[0] + m_likelihoodBuffer[m_senoneSequence[0].Senone];
    cout << "log backward score: " << logBackwardScore << endl;
#endif
    
    // asign posteriors to m_posteriorsNum
    m_posteriorsAtHost.clear();
    m_posteriorsAtHost.resize(nf * nsenones, 0);
    for (int i = 0; i < nf; i++)
    {
        for (int j = 0; j < nstates; j++)
        {
            m_posteriorsAtHost[i * nsenones + m_senoneSequence[j].Senone] += exp((ElemType)m_alphaNums[i * nstates + j]);
        }
    }
    m_posteriorsNum->Resize(nsenones, nf);
    m_posteriorsNum->SetValue(nsenones, nf, m_deviceId, &m_posteriorsAtHost[0]);

    // return the forward path score
    if (m_ceweight == 0)
        return logForwardScore;
    else
    {
        double logSum = 0;
        for (int i = 0; i < nf * nsenones; i++)
        {
            double curr = m_posteriorsAtHost[i];
            if (curr != 0){
                logSum += curr * log(curr);
            }
        }

        ElemType ce = Matrix<ElemType>::InnerProductOfMatrices(*m_posteriorsNum, *m_softmax);   // m_softmax is with logSoftmax of the NN output
        return (1 - m_ceweight) * logForwardScore - m_ceweight * (logSum - ce);
    }
}

template <class ElemType>
double LatticeFreeMMINode<ElemType>::ForwardBackwardProcessForDenorminator(const size_t nf, Matrix<ElemType>& posteriors,
    const Matrix<ElemType>& tmap, const Matrix<ElemType>& tmapTranspose, const Matrix<ElemType>& smap, const Matrix<ElemType>& smapTranspose)
{
    size_t nstates = tmap.GetNumCols();

    m_initialAlpha.clear();
    m_initialAlpha.resize(nstates, 0);
    m_initialAlpha[0] = 1.0;
    m_currAlpha->SetValue(nstates, 1, m_deviceId, &m_initialAlpha[0]);
    m_nextAlpha->Resize(nstates, 1);
    m_alphas->Resize(nstates, nf + 1);

    m_obspAtHost.clear();
    m_obspAtHost.resize(nstates * (nf + 1), 0);
    m_obspAtHost[0] = 1.0;
    m_obspAtHost[(nf + 1)*nstates - 1] = 1.0;
    m_obsp->SetValue(nstates, nf + 1, m_deviceId, &m_obspAtHost[0]);

    auto probpart = m_obsp->ColumnSlice(0, nf);
    Matrix<ElemType>::MultiplyAndWeightedAdd((ElemType)1.0, smapTranspose, false, *m_likelihoods, false, (ElemType)1.0, probpart);
    
    const int rescale_interval = 1; // rescale every this many frames
    ElemType scale = 1.0;
    double fwlogscale = 0.0;
#ifdef _DEBUG
    vector<double> sumfwscale;
    double bwlogscale = 0.0;
#endif
    for (int f = 0; f < nf + 1; f++)
    {
        scale = (ElemType)1.0 / scale;
        fwlogscale -= log(scale);

#ifdef _DEBUG
        sumfwscale.push_back(fwlogscale);
#endif
        Matrix<ElemType>::MultiplyAndWeightedAdd(scale, tmapTranspose, false, *m_currAlpha, false, (ElemType)0.0, *m_nextAlpha);

        m_currAlpha->AssignElementProductOf(*m_nextAlpha, m_obsp->ColumnSlice(f, 1));
        scale = (f % rescale_interval) == 0 ? m_currAlpha->MatrixNormInf() : (ElemType)1.0;
        
        m_alphas->SetColumnSlice(*m_currAlpha, f, 1);
    }

    double fwscore = m_currAlpha->GetValue(nstates - 1, 0);
    double logForwardPath = log(fwscore) + fwlogscale;

    m_initialAlpha[0] = 0.0;
    m_initialAlpha[nstates - 1] = 1.0;
    m_currAlpha->SetValue(nstates, 1, m_deviceId, &m_initialAlpha[0]);
    scale = 1.0;
    ElemType absum;

    for (int f = nf; f >= 0; f--) {  // not nf-1 because of transitions to final state at the end of the observation sequence
        // combine forward, backward probabilities
        auto column = m_alphas->ColumnSlice(f, 1);

        column.ElementMultiplyWith(*m_currAlpha);
        absum = (ElemType)1.0 / column.SumOfElements();

#ifdef _DEBUG
        double lfp = -log(absum) + bwlogscale + sumfwscale[f];
        assert((lfp / logForwardPath < 1.01 && lfp / logForwardPath > 0.99) || (lfp < 1e-3 && lfp > -1e-3 && logForwardPath < 1e-3 && logForwardPath > -1e-3));  // total path scores should remain constant
        bwlogscale -= log(scale);
#endif

        Matrix<ElemType>::Scale(absum, column);
        m_nextAlpha->AssignElementProductOf(*m_currAlpha, m_obsp->ColumnSlice(f, 1));

        // apply the transition matrix and scale by the maximum of the previous frame
        Matrix<ElemType>::MultiplyAndWeightedAdd(scale, tmap, false, *m_nextAlpha, false, (ElemType)0.0, *m_currAlpha);
        scale = (f % rescale_interval) == 0 ? (ElemType)1.0 / m_currAlpha->MatrixNormInf() : (ElemType)1.0;
    }

    posteriors.Resize(m_smap->GetNumRows(), nf);
    posteriors.AssignProductOf(smap, false, m_alphas->ColumnSlice(0, nf), false);

#ifdef _DEBUG

    cout << "log forward score: " << logForwardPath << endl;

    // get the total backward probability
    // verify it matches total forward probability
    ElemType bwscore = m_currAlpha->GetValue(0, 0);
    double logbwscore = log(bwscore) + bwlogscale;
    cout << "log backward score: " << logbwscore << endl;

    // verify the posterior sum
    ElemType tp = posteriors.SumOfElements();
    assert(tp / nf > 0.99 && tp / nf < 1.01);
#endif
    if (std::isnan(logForwardPath))
        RuntimeError("logForwardPath in denorminator should not be nan.");
    return logForwardPath;
}

template class LatticeFreeMMINode<float>;
template class LatticeFreeMMINode<double>;

} } }