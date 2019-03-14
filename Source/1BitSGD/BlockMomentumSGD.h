//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma  once 

#include "../SGDLib/MASGD.h"



namespace Microsoft { namespace MSR { namespace CNTK {

    // Implementation of Blockwise Model Update and Filtering (BMUF, a.k.a. block momentum) 
    // For detail, see the following paper
    // Kai Chen and Qiang Huo, "Scalable training of deep learning machines by incremental block training 
    // with intra-block parallel optimization and blockwise model-update filtering", 
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
        bool m_useNesterovMomentum;
        double m_blockLearningRate; 
        double m_blockMomentumAsTimeConstantPerWorker; 
        size_t m_syncPeriodPerWorker; 
        map < wstring, shared_ptr<Matrix<ElemType>>>     m_prevParameters;       // parameters at the last model aggregation point
        map < wstring, shared_ptr<Matrix<ElemType>>>    m_blockLevelSmoothedGradient; 

    public:
        BlockMomentumSGD(const MPIWrapperPtr& pMPI, size_t reportFreq, DEVICEID_TYPE devID, 
                        bool useNestrovMomentum, bool resetSGDM, 
                        double blockLearningRate, 
                        double blockMomentumAsTimeConstant, size_t syncPeriod)
            :IMASGD<ElemType>(pMPI, reportFreq, devID)
        {
            m_syncPeriodPerWorker = syncPeriod / pMPI->NumNodesInUse();
            m_blockMomentumAsTimeConstantPerWorker = blockMomentumAsTimeConstant / pMPI->NumNodesInUse(); 
            m_useNesterovMomentum = useNestrovMomentum;
            m_resetSGDMomentumAfterAggregation = resetSGDM; 
            m_blockLearningRate = blockLearningRate;
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
            fprintf(stderr, "Parallel training (%d workers) using BlockMomentumSGD with "
                            "block momentum = %6.4f, "
                            "block momentum time constant (per worker) = %6.4f, "
                            "block learning rate = %6.4f, "
                            "block size per worker = %d samples, "
                            "%s"
                            "%s"
                            "\n",
                            (int)m_pMPI->NumNodesInUse(),      
                            BlockMomentumSGD<double>::TimeConstant2Momentum(m_blockMomentumAsTimeConstantPerWorker, m_syncPeriodPerWorker), 
                            m_blockMomentumAsTimeConstantPerWorker,
                            m_blockLearningRate, 
                            (int)m_syncPeriodPerWorker, 
                            m_useNesterovMomentum ? "using Nesterov-style block momentum, " : "" , 
                            m_resetSGDMomentumAfterAggregation ? "resetting SGD momentum after sync." : "."
                );
        }
        /*virtual*/ void OnEpochEnd(const std::list<ComputationNodeBasePtr>& LearnableNodes, 
            std::list<MatrixBasePtr>&                   smoothedGradients,
            size_t                                      samplesSinceLastSync) override
        {
            Base::OnEpochEnd(LearnableNodes, smoothedGradients, samplesSinceLastSync);
        }
        /*virtual*/ void ModelAggregationProcessing(
            size_t samplesSinceLastSync,
            const std::list<ComputationNodeBasePtr>& learnableNodes,
            std::list<MatrixBasePtr>& smoothedGradients,
            size_t& totalSamplesProcessed,
            float& secondsOnCommunication
            ) override
        {
            //----------------------------------------
            // 1. communicate with other nodes to negotiate contribution weights
            //----------------------------------------
            int   nTotalSamples = samplesSinceLastSync;
            ElemType blockMomentum = (ElemType)BlockMomentumSGD<double>::TimeConstant2Momentum(m_blockMomentumAsTimeConstantPerWorker, m_syncPeriodPerWorker);
            Timer commTimer;
            secondsOnCommunication = 0.0f;
            commTimer.Start();
            m_pMPI->AllReduce(&nTotalSamples, 1);
            commTimer.Stop();
            secondsOnCommunication += (float)commTimer.ElapsedSeconds();
            totalSamplesProcessed = nTotalSamples;

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
                Matrix<ElemType>& prevWeight = *m_prevParameters[name];               // prev model value 
                Matrix<ElemType>& currentWeight = pNode->Value();                        // current model 
                // 2.1.2. subtract it from the previous model                   
                Matrix<ElemType>  blockGrad(prevWeight.DeepClone());            
                blockGrad -= currentWeight;                                              // matW becomes local block gradient (of one worker)
                // 2.1.3. send block gradient over MPI nodes; 
                unique_ptr<ElemType[]> px(blockGrad.CopyToArray());
                size_t    nx = blockGrad.GetNumElements();
                // 2.1.4. inplace sum 
                commTimer.Restart();
                m_pMPI->AllReduce(px.get(), nx);
                commTimer.Stop();
                secondsOnCommunication += (float)commTimer.ElapsedSeconds();
                // 2.1.5. global block gradient
                blockGrad.SetValue(blockGrad.GetNumRows(),
                                   blockGrad.GetNumCols(),
                                   blockGrad.GetDeviceId(),
                                   px.get()
                                   ); 
                // 2.2. model update 
                {
                    // alias for better readability 
                    Matrix<ElemType>& smoothedGradientUpdate = *m_blockLevelSmoothedGradient[name];       // smoothed gradient                   
                    // 2.2.1 update block level smoothed gradient; 
                    // This is essentially a first-order infinite impulse response (IIR) filter with the gain (1 - blockMomentum)*m_blockLearningRate:
                    // smoothedGradientUpdate(t)=blockMomentum * smoothedGradients(t-1) + (1 - blockMomentum)*m_blockLearningRate*blockGrad(t)
                    Matrix<ElemType>::ScaleAndAdd((ElemType)((1 - blockMomentum)*m_blockLearningRate), blockGrad, (ElemType)blockMomentum, smoothedGradientUpdate); 
                    // 2.2.2 update parameters; 
                    currentWeight.SetValue(prevWeight);
                    currentWeight -= smoothedGradientUpdate;
                    // 2.2.3 Nesterov Momentum 
                    // A Nesterov momentum here is to do a partial weight update before calculating the gradient, i.e., 
                    // (step 1) w(t) <-- w(t) - \eta* v(t) 
                    // (step 2) g(t+1) <-- forwardbackward on minibatches with initial model as w(t)
                    // (step 3) v(t+1) <-- \eta*v(t) + (1-\eta)*learningRate*g(t+1)
                    // (step 4) w(t+1) <-- w(t)-v(t)
                    // (step 5) t      <-- t+1
                    // without step 1, this becomes stanard momentum
                    if (m_useNesterovMomentum)
                    {
                        Matrix<ElemType>::ScaleAndAdd((ElemType)-blockMomentum, smoothedGradientUpdate, currentWeight);
                    }
                    // 2.2.4 update bookkeeping
                    prevWeight.SetValue(currentWeight);
                }
            }
            //----------------------------------------
            // 3. reset SGD momentum if necessary 
            //----------------------------------------
            if (m_resetSGDMomentumAfterAggregation)
            {
                for (auto sg : smoothedGradients)
                {
                    auto x = dynamic_pointer_cast<Matrix<ElemType>>(sg);
                    if (x != nullptr)
                        x->SetValue((ElemType)0);
                }
            }
        }

        /*virtual*/ void SaveToCheckPoint(File& fstream) override
        {
            if (m_pMPI->IsMainNode())
            {
                fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BMACKP");
                fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BOptions");
                fstream << m_resetSGDMomentumAfterAggregation;
                fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EOptions");

                fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BMomentumAsTimeConstant");
                fstream << m_blockMomentumAsTimeConstantPerWorker; 
                fstream.PutMarker(FileMarker::fileMarkerEndSection, L"EMomentumAsTimeConstant");

                fstream.PutMarker(FileMarker::fileMarkerBeginSection, L"BSyncPeriodInSamples"); 
                fstream << m_syncPeriodPerWorker; 
                fstream.PutMarker(FileMarker::fileMarkerEndSection, L"ESyncPeriodInSamples");

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

                fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BMomentumAsTimeConstant");
                fstream >> m_blockMomentumAsTimeConstantPerWorker;
                fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EMomentumAsTimeConstant");

                fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BSyncPeriodInSamples");
                fstream >> m_syncPeriodPerWorker;
                fstream.GetMarker(FileMarker::fileMarkerEndSection, L"ESyncPeriodInSamples");

                fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"BParam");
                LoadParameters(fstream, m_prevParameters, m_deviceId);
                LoadParameters(fstream, m_blockLevelSmoothedGradient, m_deviceId);
                fstream.GetMarker(FileMarker::fileMarkerBeginSection, L"EParam");

                fstream.GetMarker(FileMarker::fileMarkerEndSection, L"EMACKP");
            }
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
               LogicError("Mismatched ElemType in loading BlockMomentumSGD checkpoint. Expecting %s, while loading element size=%d\n",
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


    public:

       static double TimeConstant2Momentum(double timeConstant, size_t syncPeroid)
       {
           return exp(-((double)syncPeroid) / timeConstant);
       }
       static double Momentum2TimeConstant(double bm, size_t syncPeroid)
       {
           if (bm >= 1.0 || bm < 0.0)
           {
               InvalidArgument("Unexpected block momentum (%.2f). Block momentum should be in the range of [0,1)\n", bm);
           }
           return -(double)syncPeroid / log(bm); 
       }
    };
} } }
