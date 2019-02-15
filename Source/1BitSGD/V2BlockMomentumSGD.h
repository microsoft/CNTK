//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma  once 

#include "../SGDLib/MASGD.h"
#include <map>
#include <string>
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

    // Implementation of Blockwise Model Update and Filtering (BMUF, a.k.a. block momentum)
    // For detail, see the following paper
    // Kai Chen and Qiang Huo, "Scalable training of deep learning machines by incremental block training
    // with intra-block parallel optimization and blockwise model-update filtering",
    // in International Conference on Acoustics, Speech and Signal Processing , March 2016, Shanghai, China.
    template<typename ElemType>
    class V2BlockMomentumSGD : public IMASGD<ElemType>
    {
        typedef IMASGD<ElemType> Base;
        using Base::m_deviceId;
        using Base::DownCast;

        bool m_resetSGDMomentumAfterAggregation;
        bool m_useNesterovMomentum;
        double m_blockLearningRate;
        double m_blockMomentumAsTimeConstantPerWorker;
        size_t m_syncPeriodPerWorker;
        ::CNTK::DistributedCommunicatorPtr m_communicator;
        bool m_someWorkerHasFinished;

        // parameters at the last model aggregation point
        std::map<std::wstring, std::shared_ptr<Matrix<ElemType>>> m_prevParameters;
        std::map<std::wstring, std::shared_ptr<Matrix<ElemType>>> m_blockLevelSmoothedGradient;

    public:
        V2BlockMomentumSGD(const MPIWrapperPtr& pMPI,
            ::CNTK::DistributedCommunicatorPtr communicator,
            size_t reportFrequency,
            DEVICEID_TYPE deviceId,
            bool useNestrovMomentum,
            bool resetSGDM,
            double blockLearningRate,
            double blockMomentumAsTimeConstant,
            size_t syncPeriod)
            : IMASGD<ElemType>(pMPI, reportFrequency, deviceId),
            m_communicator(communicator),
            m_useNesterovMomentum(useNestrovMomentum),
            m_resetSGDMomentumAfterAggregation(resetSGDM),
            m_blockLearningRate(blockLearningRate),
            m_blockMomentumAsTimeConstantPerWorker(blockMomentumAsTimeConstant / communicator->Workers().size())
        {
            m_syncPeriodPerWorker = syncPeriod / communicator->Workers().size();
            if (m_syncPeriodPerWorker == 0)
                InvalidArgument("Sync period is too small.");
        }

        void OnEpochStart(const std::list<ComputationNodeBasePtr>& learnableNodes) override
        {
            m_someWorkerHasFinished = false;

            for (auto& n : learnableNodes)
            {
                auto node = DownCast(n);
                std::wstring name = node->NodeName();

                Matrix<ElemType>& value = node->Value();
                if (m_blockLevelSmoothedGradient.find(name) == m_blockLevelSmoothedGradient.end())
                {
                    // has not been initialized yet
                    auto pSmoothedGrad = make_shared<Matrix<ElemType>> (value.GetDeviceId());
                    pSmoothedGrad->Resize(value.GetNumRows(), value.GetNumCols());
                    pSmoothedGrad->SetValue((ElemType)0); 
                    m_blockLevelSmoothedGradient[name] = pSmoothedGrad; 
                }

                if (m_prevParameters.find(name) == m_prevParameters.end())
                {
                    auto newValue = make_shared<Matrix<ElemType>>(value.GetDeviceId());
                    newValue->SetValue(value);
                    m_prevParameters[name] = newValue;
                }
                else
                {
                    m_prevParameters[name]->SetValue(value);
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
                            (int)m_communicator->Workers().size(),
                            BlockMomentumSGD<double>::TimeConstant2Momentum(m_blockMomentumAsTimeConstantPerWorker, m_syncPeriodPerWorker),
                            m_blockMomentumAsTimeConstantPerWorker,
                            m_blockLearningRate, 
                            (int)m_syncPeriodPerWorker, 
                            m_useNesterovMomentum ? "using Nesterov-style block momentum, " : "" , 
                            m_resetSGDMomentumAfterAggregation ? "resetting SGD momentum after sync." : ".");
        }

        bool OnArrivingAtSyncPoint(
            const std::list<ComputationNodeBasePtr>& learnableNodes,        /* input/output: */
            std::list<MatrixBasePtr>& smoothedGradients,                    /* input/output: under some setup, it will reset to zero*/
            size_t  samplesSinceLastSync                                    /* input:  samples processed since last sync on this worker only */
            ) override
        {
            if (m_someWorkerHasFinished)
                return false;

            // Let's check the status.
            double statusValue = 0;
            auto status = ::CNTK::MakeSharedObject<::CNTK::NDArrayView>(::CNTK::DataType::Double, ::CNTK::NDShape{ 1 }, &statusValue, sizeof(double), ::CNTK::DeviceDescriptor::CPUDevice());
            std::vector<::CNTK::NDArrayViewPtr> aggregatedStatus { status };
            m_communicator->AggregateInPlace(aggregatedStatus, m_communicator->Workers());

            if (statusValue > 0)
            {
                m_someWorkerHasFinished = true;
                return false;
            }

            // Otherwise let update the weights.
            float secondsOnCommunication = 0.0f;
            size_t totalSamples = 0;
            ModelAggregationProcessing(samplesSinceLastSync, learnableNodes, smoothedGradients, totalSamples, secondsOnCommunication);
            return true;
        }

        /*virtual*/ void OnEpochEnd(const std::list<ComputationNodeBasePtr>& learnableNodes,
            std::list<MatrixBasePtr>& smoothedGradients,
            size_t samplesSinceLastSync) override
        {
            if (!m_someWorkerHasFinished)
            {
                // Let's update the other guys that we have finished.
                m_someWorkerHasFinished = true;

                double statusValue = 1;
                auto status = ::CNTK::MakeSharedObject<::CNTK::NDArrayView>(::CNTK::DataType::Double, ::CNTK::NDShape{ 1 }, &statusValue, sizeof(double), ::CNTK::DeviceDescriptor::CPUDevice());
                std::vector<::CNTK::NDArrayViewPtr> aggregatedStatus{ status };
                m_communicator->AggregateInPlace(aggregatedStatus, m_communicator->Workers());
            }

            // Let's update our weights no matter what.
            float secondsOnCommunication = 0.0f;
            size_t totalSamples = 0;
            ModelAggregationProcessing(samplesSinceLastSync, learnableNodes, smoothedGradients, totalSamples, secondsOnCommunication);
        }

        /*virtual*/ void ModelAggregationProcessing(
            size_t /*samplesSinceLastSync*/,
            const std::list<ComputationNodeBasePtr>& learnableNodes,
            std::list<MatrixBasePtr>& smoothedGradients,
            size_t&                                   /*totalSamplesProcessed*/,   /* out */
            float&                                    secondsOnCommunication   /* out */
            ) override
        {
            ElemType blockMomentum = (ElemType)BlockMomentumSGD<double>::TimeConstant2Momentum(m_blockMomentumAsTimeConstantPerWorker, m_syncPeriodPerWorker);
            Timer commTimer;
            secondsOnCommunication = 0.0f;

            // 1. Let's aggregate weights
            std::map<std::wstring, std::shared_ptr<Matrix<ElemType>>> aggregatedWeights;
            std::vector<::CNTK::NDArrayViewPtr> aggregatedWeightsPrepared;
            for (auto& pBaseNode : learnableNodes)
            {
                if (!pBaseNode->IsParameterUpdateRequired())
                    continue;

                wstring name = pBaseNode->NodeName();
                auto pNode = DownCast(pBaseNode);

                // Get current model
                Matrix<ElemType>& prevWeight = *m_prevParameters[name];                  // prev model value
                Matrix<ElemType>& currentWeight = pNode->Value();                        // current model

                // Subtract it from the previous model
                auto blockGrad = std::make_shared<Matrix<ElemType>>(prevWeight, CPUDEVICE);
                *blockGrad -= currentWeight;                                              // matW becomes local block gradient (of one worker)

                aggregatedWeights[name] = blockGrad;
                ::CNTK::NDShape shape{ blockGrad->GetNumElements() };
                auto data = ::CNTK::MakeSharedObject<::CNTK::NDArrayView>(::CNTK::AsDataType<ElemType>(), shape, blockGrad->Data(), blockGrad->GetNumElements() * sizeof(ElemType), ::CNTK::AsDeviceDescriptor(blockGrad->GetDeviceId()));
                aggregatedWeightsPrepared.push_back(data);
            }

            // Send block gradient over MPI nodes.
            m_communicator->AggregateInPlace(aggregatedWeightsPrepared, m_communicator->Workers());

            // 2. Let's update the model
            auto smoothedGradientIter = smoothedGradients.begin();
            for (auto nodeIter = learnableNodes.begin(); nodeIter != learnableNodes.end(); nodeIter++, smoothedGradientIter++)
            {
                ComputationNodeBasePtr pBaseNode = *nodeIter;
                if (!pBaseNode->IsParameterUpdateRequired())
                    continue;

                wstring name = pBaseNode->NodeName();
                auto pNode = DownCast(pBaseNode);

                // 2 block gradient aggregation
                // 2.1. get current model
                Matrix<ElemType>& prevWeight = *m_prevParameters[name];                  // prev model value
                Matrix<ElemType>& currentWeight = pNode->Value();                        // current model
                auto blockGrad = aggregatedWeights[name];
                // 2.2. model update 
                {
                    Matrix<ElemType>& sg = *m_blockLevelSmoothedGradient[name];       // smoothed gradient
                    blockGrad->TransferToDeviceIfNotThere(sg.GetDeviceId());
                    // 2.2.1 update block level smoothed gradient; 
                    // This is essentially a first-order infinite impulse response (IIR) filter with the gain (1 - blockMomentum)*m_blockLearningRate:
                    // smoothedGradient(t)=blockMomentum * smoothedGradients(t-1) + (1 - blockMomentum)*m_blockLearningRate*blockGrad(t)
                    Matrix<ElemType>::ScaleAndAdd((ElemType)((1 - blockMomentum)*m_blockLearningRate), *blockGrad, (ElemType)blockMomentum, sg);
                    // 2.2.2 update parameters; 
                    currentWeight.SetValue(prevWeight);
                    currentWeight -= sg;
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
                        Matrix<ElemType>::ScaleAndAdd((ElemType)-blockMomentum, sg, currentWeight);
                    }
                    // 2.2.4 update bookkeeping
                    prevWeight.SetValue(currentWeight);
                }

                //----------------------------------------
                // 3. reset SGD momentum if necessary 
                //----------------------------------------
                {
                    // For half, we keep a copy of float weights, update that too
                    if (std::is_same<ElemType, half>())
                    {
                        auto compoundMatrixPtr = dynamic_pointer_cast<Matrix<float>> (*smoothedGradientIter);
                        size_t numCols = currentWeight.GetNumCols();

                        auto parameterMatrix = compoundMatrixPtr->ColumnSlice(2 * numCols, numCols);
                        parameterMatrix.CastAssignValuesOf(currentWeight);

                        if (m_resetSGDMomentumAfterAggregation)
                        {
                            // Only reset smoothed gradients
                            auto smoothedGradientMatrix = compoundMatrixPtr->ColumnSlice(0, numCols);
                            smoothedGradientMatrix.SetValue(0.0f);
                        }
                    }
                    else
                    {
                        if (m_resetSGDMomentumAfterAggregation)
                        {
                            auto x = dynamic_pointer_cast<Matrix<ElemType>> (*smoothedGradientIter);
                            x->SetValue((ElemType)0);
                        }
                    }
                }
            }
        }

        void SaveToCheckPoint(File& fstream) override
        {
            if (!m_communicator->CurrentWorker().IsMain())
                return;

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

        void LoadFromCheckPoint(File& fstream) override
        {
            if (!fstream.TryGetMarker(FileMarker::fileMarkerBeginSection, L"BMACKP"))
                return;

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
