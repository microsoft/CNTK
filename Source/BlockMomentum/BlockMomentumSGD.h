//
// <copyright file="BlockMomentumSGD.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma  once 

#include "../SGDLib/MASGD.h"



namespace Microsoft { namespace MSR { namespace CNTK {

    // Implementation of Blockwise Model Update and Filtering (BMUF, a.k.a. block momentum) 
    // For detail, see the following paper
    // Kai Chen and Qiang Huo, “Scalable training of deep learning machines by incremental block training 
    // with intra-block parallel optimization and blockwise model-update filtering”, 
    // in Internal Conference on Acoustics, Speech and Signal Processing , March 2016, Shanghai, China. 

    template<typename ElemType>
    class BlockMomentumSGD : public IMASGD<ElemType>
    {
        typedef IMASGD<ElemType> Base;
        using Base::m_pMPI;
        using Base::m_preferredDeviceID;
        using Base::DownCast;
    
     protected:
        bool m_resetSGDMomentumAfterAggregation; 
        double m_blockMomentum; 
        map < wstring, shared_ptr<Matrix<ElemType>>>     m_prevParameters;       // parameters at the last model aggregation point
        map < wstring, shared_ptr<Matrix<ElemType>>>    m_blockLevelSmoothedGradient; 

    public:
        BlockMomentumSGD(MPIWrapper* pMPI, size_t reportFreq, DEVICEID_TYPE devID, bool resetSGDM, double blockMomentum)
            :IMASGD<ElemType>(pMPI, reportFreq, devID)
        {
            m_blockMomentum = blockMomentum; 
            m_resetSGDMomentumAfterAggregation = resetSGDM; 
            fprintf(stderr, "Parallel training (%d workers) using BlockMomentum: useNesterovMomentum=true, blockMomentum=%6.4f, resetSGDMomentum=%s, preferredDeviceID=%d\n",
                (int)m_pMPI->NumNodesInUse(),
                m_blockMomentum,
                m_resetSGDMomentumAfterAggregation ? "true" : "false", 
                m_preferredDeviceID
                );
        }

        /*virtual*/ void OnEpochStart(const std::list<ComputationNodeBasePtr>& LearnableNodes) override
        {
            Base::OnEpochStart(LearnableNodes); 
            for (auto& pNode : LearnableNodes)
            {
                auto pnode = DownCast(pNode);
                wstring name = pNode->NodeName();

                Matrix<ElemType>& NodeValue = pnode->Value();
                if (m_blockLevelSmoothedGradient.find(name) == m_blockLevelSmoothedGradient.end())
                {
                    // has not been initialized yet
                    auto pSmoothedGrad = make_shared<Matrix<ElemType>> (NodeValue.GetDeviceId());
                    pSmoothedGrad->Resize(NodeValue.GetNumRows(), NodeValue.GetNumCols());
                    pSmoothedGrad->SetValue((ElemType)0); 
                    m_blockLevelSmoothedGradient[name] = pSmoothedGrad; 
                }
                if (m_prevParameters.find(name) == m_prevParameters.end())
                {
                    auto pValue = make_shared<Matrix<ElemType>>  (NodeValue.GetDeviceId());
                    pValue->SetValue(NodeValue);
                    m_prevParameters[name] = pValue;
                }
                else
                {
                    m_prevParameters[name]->SetValue(NodeValue);
                }
                // related with nestrov momentum, adjust it before training start
                Matrix<ElemType> V(m_blockLevelSmoothedGradient[name]->DeepClone());
                Matrix<ElemType>::Scale((ElemType)m_blockMomentum, V);
                NodeValue -= V;
            }
        }
        /*virtual*/ void OnEpochEnd(const std::list<ComputationNodeBasePtr>& LearnableNodes, 
            std::list<Matrix<ElemType>>&                smoothedGradient,
            size_t                                      samplesSinceLastSync) override
        {
            Base::OnEpochEnd(LearnableNodes, smoothedGradient, samplesSinceLastSync);
            for (auto& pNode : LearnableNodes)
            {
                auto pnode = DownCast(pNode);
                wstring name = pNode->NodeName();
                if (m_blockLevelSmoothedGradient.find(name) == m_blockLevelSmoothedGradient.end())
                {
                    LogicError("Cannot find block information for node %ls. Contact erw@microsoft.com\n", name.c_str());
                    // TODO: remove this part
                }
                Matrix<ElemType>& W = pnode->Value();
                Matrix<ElemType> V(m_blockLevelSmoothedGradient[name]->DeepClone());
                Matrix<ElemType>::Scale((ElemType)m_blockMomentum, V);
                W += V;
            }
        }
        /*virtual*/ void ModelAggregationProcessing(
            size_t samplesSinceLastSync,
            const std::list<ComputationNodeBasePtr>& learnableNodes,
            std::list<Matrix<ElemType>>& smoothedGradient,
            size_t& totalSamplesProcessed,
            float& secondsOnCommunication
            ) override
        {
            //----------------------------------------
            // 1. communicate with other nodes to negotiate  contribution weights
            //----------------------------------------
            float factor = 0;
            int   nTotalSamples = samplesSinceLastSync;
            Timer commTimer;
            secondsOnCommunication = 0.0f;
            commTimer.Start();
            m_pMPI->AllReduce(&nTotalSamples, 1);
            commTimer.Stop();
            secondsOnCommunication += (float)commTimer.ElapsedSeconds();

            if (nTotalSamples <= 0)
            {
                // prepare for overflow 
                factor = 1.0f / g_mpi->NumNodesInUse();
                totalSamplesProcessed = samplesSinceLastSync * m_pMPI->NumNodesInUse();
                // give an estimated one 
            }
            else
            {
                factor = (samplesSinceLastSync + 0.0f) / nTotalSamples;
                totalSamplesProcessed = nTotalSamples;
            }
            //----------------------------------------
            // 2. process for each individual node
            //----------------------------------------
            for (auto& pBaseNode : learnableNodes)
            {
                if (!pBaseNode->IsParameterUpdateRequired())
                {
                    continue;
                }
                wstring name = pBaseNode->NodeName();
                // 2.1 model averaging
                auto pNode = DownCast(pBaseNode);
                // 2.1.1. average model from individual models 
                Matrix<ElemType> mat(pNode->Value().DeepClone()); // pNode->Value returns lvalue, so a deep copy is invoked here
                // 2.1.2. normalize the weight matrix 
                Matrix<ElemType>::Scale(factor, mat);
                // 2.1.3. send weight matrix over MPI nodes; 
                unique_ptr<ElemType[]> px(mat.CopyToArray());
                //ElemType* px = mat.CopyToArray();
                size_t    nx = mat.GetNumElements();
                // 2.1.4. inplace sum 
                commTimer.Restart();
                m_pMPI->AllReduce(px.get(), nx);
                commTimer.Stop();
                secondsOnCommunication += (float)commTimer.ElapsedSeconds();
                // 2.1.5. averaged model
                mat.SetValue(mat.GetNumRows(), mat.GetNumCols(), mat.GetDeviceId(), px.get());
                // 2.2. blockwise model update and filtering 
                {
                    // some alias for better readability 
                    Matrix<ElemType>& V = *m_blockLevelSmoothedGradient[name];       // smoothed gradient 
                    Matrix<ElemType>& W = pNode->Value();                           // model value 
                    Matrix<ElemType>& preW = *m_prevParameters[name];               // prev model value 
                    Matrix<ElemType>& negG = mat;                                   // negative gradient
                    negG -= preW;                                                   // (W^-W_{t-1})

                    // 2.2.1 update block level smoothed gradient;
                    Matrix<ElemType>::Scale((ElemType)m_blockMomentum, V);
                    V -= negG;                                                  // V_t=\eta*V_{t-1}-(W^-W_{t-1})
                    // 2.2.2 update parameters; 
                    W.SetValue(*m_prevParameters[name]);
                    Matrix<ElemType>::ScaleAndAdd((ElemType)-m_blockMomentum, V, W); // W_t=W_{t-1}-\eta*V_t + (W^-W_{t-1})
                    W += negG;
                    // 2.2.3 update previous model parameters;
                    preW.SetValue(W);
                }
            }
            //----------------------------------------
            // 3. reset SGD momentum if necessary 
            //----------------------------------------
            if (m_resetSGDMomentumAfterAggregation)
            {
                for (Matrix<ElemType>& x : smoothedGradient)
                {
                    x.SetValue((ElemType)0);
                }
            }
        }

        /*virtual*/ bool requireCheckPointSaving() override
        {
            return true;
        }
        /*virtual*/ void SaveToCheckPoint(File& fstream) override
        {
            if (m_pMPI->IsMainNode())
            {
                fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BMACKP");
                fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BOptions");
                fstream << m_resetSGDMomentumAfterAggregation;
                fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EOptions");

                fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BMomentum");
                fstream << m_blockMomentum;
                fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EMomentum");

                fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BParam");
                SaveParameters(fstream, m_prevParameters);
                SaveParameters(fstream, m_blockLevelSmoothedGradient);
                fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"EParam");

                fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"EMACKP");
            }
        }
        /*virtual*/ void LoadFromCheckPoint(File& fstream) override
        {
            if (fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BMACKP"))
            {
                fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BOptions");
                fstream >> m_resetSGDMomentumAfterAggregation;
                fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EOptions");

                fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BMomentum");
                fstream >> m_blockMomentum;
                fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EMomentum");
                // logical change 
                if (m_blockMomentum < 0 || m_blockMomentum > 1.0)
                {
                    LogicError("Error in loading MASGD checkpoint: loading a momentum %f, but expect a block-momentum in [0,1].\n",
                        m_blockMomentum);
                }

                fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BParam");
                LoadParameters(fstream, m_prevParameters, m_preferredDeviceID);
                LoadParameters(fstream, m_blockLevelSmoothedGradient, m_preferredDeviceID);
                fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"EParam");

                fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EMACKP");
            }
            /* else 
            {
                // no MA checkpoint info, don't need to do anything here. 
                // It will still be initialized in OnEpochStart() function
            }
            */
        }
    private:
       // helper function to save/load map<wstring, shared_ptr<Matrix<ElemType>> structure 
       void SaveParameters(File& f, const map<wstring, shared_ptr<Matrix<ElemType>>>& parameters) const
            {
                // save sizeof(ElemType)
                unsigned int size = sizeof(ElemType);
                f << size;
                // save number of pairs 
                unsigned int numPairs = parameters.size();
                f << numPairs;
                for (auto& x : parameters)
                {
                    f << x.first;
                    f << *x.second;
                }
                f.Flush();
                return;
            }
       void LoadParameters(File& f, map<wstring, shared_ptr<Matrix<ElemType>>>& parameters, DEVICEID_TYPE deviceID)
            {
                unsigned int size = 0;
                unsigned int pair = 0;
                f >> size;
                f >> pair;
                if (size != sizeof(ElemType))
                {
                    LogicError("Mismatched ElemType in loading MASGD checkpoint. Expecting %s, while loading element size=%d\n",
                        sizeof(ElemType) == 4 ? "float" : "double",
                        size
                        );
                }
                parameters.clear();
                for (size_t i = 0; i < pair; i++)
                {
                    wstring name;
                    f >> name;
                    shared_ptr<Matrix<ElemType>> mat = make_shared<Matrix<ElemType>>(deviceID);
                    f >> *mat;
                    parameters[name] = mat;
                }

            }

    };




} } }
