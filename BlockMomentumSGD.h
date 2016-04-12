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
    // in International Conference on Acoustics, Speech and Signal Processing , March 2016, Shanghai, China. 

    template<typename ElemType>
    class BlockMomentumSGD : public IMASGD<ElemType>
    {
        typedef IMASGD<ElemType> Base;
        using Base::m_pMPI;
        using Base::m_deviceId;
        using Base::DownCast;
    
     protected:
        bool m_resetSGDMomentumAfterAggregation; 
        bool m_useNestrovMomentum;
        double m_blockMomentum; 
        double m_blockLearningRate; 
        map < wstring, shared_ptr<Matrix<ElemType>>>     m_prevParameters;       // parameters at the last model aggregation point
        map < wstring, shared_ptr<Matrix<ElemType>>>    m_blockLevelSmoothedGradient; 

    public:
        BlockMomentumSGD(const MPIWrapperPtr& pMPI, size_t reportFreq, DEVICEID_TYPE devID, 
                        bool useNestrovMomentum, bool resetSGDM, 
                        double blockMomentum, double blockLearningRate, double blockMomentumAsTimeConstant)
            :IMASGD<ElemType>(pMPI, reportFreq, devID)
        {
            m_blockMomentum = blockMomentum; 
            m_useNestrovMomentum = useNestrovMomentum;
            m_resetSGDMomentumAfterAggregation = resetSGDM; 
            m_blockLearningRate = blockLearningRate;
            fprintf(stderr, "Parallel training (%d workers) using BlockMomentum: "
                            "useNesterovMomentum=%s, resetSGDMomentum=%s, "
                            "blockMomentum=%6.4f, blockMomentumAsTimeConstant=%6.4f, "
                            "blockLearningRate=%6.4f\n",
                (int)m_pMPI->NumNodesInUse(),      
                m_useNestrovMomentum ? "true" : "false" , 
                m_resetSGDMomentumAfterAggregation ? "true" : "false", 
                m_blockMomentum, blockMomentumAsTimeConstant,
                m_blockLearningRate
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
            }
        }
        /*virtual*/ void OnEpochEnd(const std::list<ComputationNodeBasePtr>& LearnableNodes, 
            std::list<Matrix<ElemType>>&                smoothedGradient,
            size_t                                      samplesSinceLastSync) override
        {
            Base::OnEpochEnd(LearnableNodes, smoothedGradient, samplesSinceLastSync);
#if 0
            for (auto& pNode : LearnableNodes)
            {
                auto pnode = DownCast(pNode);
                wstring name = pNode->NodeName();
                if (m_blockLevelSmoothedGradient.find(name) == m_blockLevelSmoothedGradient.end())
                {
                    LogicError("Cannot find block information for node %ls. Contact erw@microsoft.com\n", name.c_str());
                    // TODO: remote this 
                }
            }
#endif 
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
            // 1. communicate with other nodes to negotiate contribution weights
            //----------------------------------------
            float factor = 0;
            int   nTotalSamples = samplesSinceLastSync;
            Timer commTimer;
            secondsOnCommunication = 0.0f;
            commTimer.Start();
            m_pMPI->AllReduce(&nTotalSamples, 1);
            commTimer.Stop();
            secondsOnCommunication += (float)commTimer.ElapsedSeconds();
            {
                factor = (samplesSinceLastSync + 0.0f) / nTotalSamples;
                totalSamplesProcessed = nTotalSamples;
                factor *= m_pMPI->NumNodesInUse();
            }
            // TODO: currently each worker's contribution to block gradient is proportional to the processed samples. 
            //       However, if some worker just diverge during his local SGD update, 
            //       we should disable its contribution by setting its contribution 
            //       factor to 0 
            


            for (auto& pBaseNode : learnableNodes)
            {
                if (!pBaseNode->IsParameterUpdateRequired())
                {
                    continue;
                }
                wstring name = pBaseNode->NodeName();
                // 2 block gradient aggregation 
                auto pNode = DownCast(pBaseNode);
                // 2.1. get current model  
                Matrix<ElemType>& preW = *m_prevParameters[name];               // prev model value 
                Matrix<ElemType>& curW = pNode->Value();                        // current model 
                // 2.1.2. subtract it from the previous model                   
                Matrix<ElemType>  blockGrad(preW.DeepClone());            
                blockGrad -= curW;                                              // matW becomes local block gradient (of one worker)
                Matrix<ElemType>::Scale(factor, blockGrad);
                // 2.1.3. send block gradient over MPI nodes; 
                unique_ptr<ElemType[]> px(blockGrad.CopyToArray());
                size_t    nx = blockGrad.GetNumElements();
                // 2.1.4. inplace sum 
                commTimer.Restart();
                m_pMPI->AllReduce(px.get(), nx);
                commTimer.Stop();
                secondsOnCommunication += (float)commTimer.ElapsedSeconds();
                // 2.1.5. global block gradient
                blockGrad.SetValue( blockGrad.GetNumRows(), 
                                    blockGrad.GetNumCols(), 
                                    blockGrad.GetDeviceId(), 
                                    px.get()
                                    ); // blockGrad becomes global block gradient (sum of all local block gradient)
                // 2.2. model update 
                {
                    // alias for better readability 
                    Matrix<ElemType>& V = *m_blockLevelSmoothedGradient[name];       // smoothed gradient                   
                    // 2.2.1 update block level smoothed gradient;
                    Matrix<ElemType>::ScaleAndAdd((ElemType)((1 - m_blockMomentum)*m_blockLearningRate), blockGrad, (ElemType)m_blockMomentum, V); 
                    // 2.2.2 update parameters; 
                    curW.SetValue(preW);
                    curW -= V;
                    preW.SetValue(curW);
                    // 2.2.3 Nestrov Momentum 
                    if (m_useNestrovMomentum)
                    {
                        Matrix<ElemType>::ScaleAndAdd((ElemType)-m_blockMomentum, V, curW);
                    }
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

        /*virtual*/ bool RequiresToSaveToCheckPoint() override
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
                LoadParameters(fstream, m_prevParameters, m_deviceId);
                LoadParameters(fstream, m_blockLevelSmoothedGradient, m_deviceId);
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
