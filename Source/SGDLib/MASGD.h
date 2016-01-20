//
// <copyright file="MASGD.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
#include "ComputationNetwork.h"
#include "Config.h"
#include "SGD.h"
#include "Matrix.h"
#include <vector>
#include <string>
#include <stdexcept>
#include <chrono> 
#include <random>


namespace Microsoft{ namespace MSR { namespace CNTK{

template<typename ElemType>
class MASGD
{
public:
    //========================================
    // Constructor and Initializer 
    //========================================
    MASGD(bool useNestrovBlockMomentum, double blockMomentum, bool resetSGDMomentum)
    {
        m_useNesterovBlockMomentum = useNestrovBlockMomentum; 
        m_blockMomentum = blockMomentum;
        if (blockMomentum > 0 && blockMomentum <= 1.0)
        {
            m_useBMUF = true; 
        }
        m_resetSGDMomentum = resetSGDMomentum; 
        fprintf(stderr, "Parallel training using MA-SGD: useBMUF=%s, useNesterovMomentum=%s, blockMomentum=%6.4f, resetSGDMomentum=%s\n",
            m_blockMomentum > 1e-6 ? "true" : "false", 
            m_useNesterovBlockMomentum ? "true" : "false", 
            m_blockMomentum, 
            m_resetSGDMomentum ? "true" : "false"
            );
    }
    //========================================
    // Interface 
    //========================================
    void Initialize(const std::list<ComputationNodeBasePtr>& LearnableNodes)
    {
        if (!m_useBMUF)
            return; 

        for (auto& pNode : LearnableNodes)
        {
            if (!pNode->IsParameterUpdateRequired())
            {
                continue;
            }
            wstring name = pNode->NodeName();
            auto pnode = ComputationNode<ElemType>::UpCast(pNode); 
            auto pvalue = make_shared<Matrix<ElemType>>(pnode->Value().GetDeviceId());
            auto pSmoothedGrad = make_shared<Matrix<ElemType>>(pnode->Value().GetDeviceId());
            pvalue->SetValue(pnode->Value());
            pSmoothedGrad->Resize(pvalue->GetNumRows(), pvalue->GetNumCols());
            pSmoothedGrad->SetValue((ElemType)0.0);
            m_prevParameters[name] = pvalue; 
            m_blockLevelSmoothedGradient[name] = pSmoothedGrad; 
        }
    }
    void PerformModelAveragingUpdate(const std::list<ComputationNodeBasePtr>& LearnableNodes,     /* input/output: */
        std::list<Matrix<ElemType>>& smoothedGradient,   /* input/output: under some setup, it will reset to zero*/
        size_t  samplesSinceLastSync,                          /* input:  samples processed since last sync on this worker only */
        size_t& samplesProcessed                               /* output: total samples processed on all the workers */
        )
    {
        if (m_resetSGDMomentum)
        {
            for (Matrix<ElemType>& x : smoothedGradient)
            {
                x.SetValue((ElemType)0);
            }
        }
        if (g_mpi->NumNodesInUse() <= 1) // we should not arrive here 
        {
            samplesProcessed = samplesSinceLastSync; 
            return;  // do nothing here 
        }

        //========================================
        // 1. communicate with other nodes to negoticate contribution weights
        //========================================
        float factor = 0;
        int   nTotalSamples = samplesSinceLastSync; 
        g_mpi->AllReduce(&nTotalSamples, 1);
        if (nTotalSamples <= 0)
        {
            // prepare for overflow 
            factor = 1.0f / g_mpi->NumNodesInUse();
            samplesProcessed = samplesSinceLastSync * g_mpi->NumNodesInUse(); 
            // give an estimated one 
        }
        else
        {
            factor = (samplesSinceLastSync + 0.0f) / nTotalSamples;
            samplesProcessed = nTotalSamples; 
        }


        //========================================
        // 2. process for each individual node
        //========================================
        for (auto& pBaseNode : LearnableNodes)
        {
            if (!pBaseNode->IsParameterUpdateRequired())
            {
                continue;
            }
            wstring name = pBaseNode->NodeName(); 
            if (m_useBMUF && (m_prevParameters.find(name) == m_prevParameters.end() || m_blockLevelSmoothedGradient.find(name) == m_blockLevelSmoothedGradient.end()))
            {
                LogicError("Cannot find block information for node %ls. Contact erw@microsoft.com\n", name.c_str()); 
            }

            // 2.1 model aggregation 
            auto pNode = ComputationNode<ElemType>::UpCast(pBaseNode); 
            // 2.1.1. aggregate model from individual models 
            Matrix<ElemType> mat (pNode->Value());
            // 2.1.2. normalize the weight matrix 
            Matrix<ElemType>::Scale(factor, mat);
            // 2.1.3. send weight matrix over MPI nodes; 
            ElemType* px = mat.CopyToArray();
            size_t    nx = mat.GetNumElements();
            // 2.1.4. inplace sum 
            g_mpi->AllReduce(px, nx);
            mat.SetValue(mat.GetNumRows(), mat.GetNumCols(), mat.GetDeviceId(), px);
            // 2.1.5. clean up 
            delete[]px;

            // 2.2 deal with momentum 
            if (m_useBMUF && m_blockMomentum > 0)
            {
                // some alias for better readability 
                Matrix<ElemType>& V = *m_blockLevelSmoothedGradient[name];       // smoothed graident 
                Matrix<ElemType>& W = pNode->Value();                           // model value 
                Matrix<ElemType>& preW = *m_prevParameters[name];               // prev model value 
                Matrix<ElemType>& negG = mat;                                   // negative gradient
                negG -= preW;
                // 2.2.1 update block level smoothed gradient 
                Matrix<ElemType>::Scale((ElemType)m_blockMomentum, V); 
                V -= negG;                                                  // V=\eta*V+G
                // 2.2.2 update global model 
                W.SetValue(*m_prevParameters[name]); 
                if (m_useNesterovBlockMomentum)
                {
                    Matrix<ElemType>::ScaleAndAdd((ElemType)-m_blockMomentum, V, W); 
                    W += negG;      // W=W-\eta*V-G;
                }
                else
                {
                    W += negG;      // W=W-G;
                }
                // 2.2.3 update prev model parameter
                preW = W;

            }
            else
            {
                // plain MA
                pNode->Value().SetValue(mat);
            }
        }


    }
    void OnOneEpochFinished(const std::list<ComputationNodeBasePtr>& LearnableNodes)
    {
        if (m_useBMUF && m_useNesterovBlockMomentum)
        {
            for (auto& pNode : LearnableNodes)
            {
                auto pnode = ComputationNode<ElemType>::UpCast(pNode);
                wstring name = pNode->NodeName();
                if (m_blockLevelSmoothedGradient.find(name) == m_blockLevelSmoothedGradient.end())
                {
                    LogicError("Cannot find block information for node %ls. Contact erw@microsoft.com\n", name.c_str());
                }
                Matrix<ElemType>& W = pnode->Value();
                Matrix<ElemType> V(*m_blockLevelSmoothedGradient[name]);
                Matrix<ElemType>::Scale((ElemType)m_blockMomentum, V);
                W += V;
            }
        }
    }
    void OnOneEpochStarted(const std::list<ComputationNodeBasePtr>& LearnableNodes)
    {
        if (m_useBMUF && m_useNesterovBlockMomentum)
        {
            for (auto& pNode : LearnableNodes)
            {
                auto pnode = ComputationNode<ElemType>::UpCast(pNode);
                wstring name = pNode->NodeName();
                if (m_blockLevelSmoothedGradient.find(name) == m_blockLevelSmoothedGradient.end())
                {
                    LogicError("Cannot find block information for node %ls. Contact erw@microsoft.com\n", name.c_str());
                }
                Matrix<ElemType>& W = pnode->Value();
                Matrix<ElemType> V(*m_blockLevelSmoothedGradient[name]);
                Matrix<ElemType>::Scale((ElemType)m_blockMomentum, V);
                W -= V;
            }
        }
    }


private:
    bool    m_useBMUF;
    bool    m_useNesterovBlockMomentum;
    bool    m_resetSGDMomentum; 
    double  m_blockMomentum;
    map<wstring, shared_ptr<Matrix<ElemType>> >     m_prevParameters; 
    map<wstring, shared_ptr<Matrix<ElemType>> >     m_blockLevelSmoothedGradient;
    // m_prevParameters holds model parameters before the next sync and after the last sync , required by BMUF


};

}}}


