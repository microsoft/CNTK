//
// Copyright (c) Microsoft. All rights reserved.
//
// Licensed under custom Microsoft Research License Terms for
// 1-bit Stochastic Gradient Descent.
// See LICENSE.md file in the project root for full license information.
//

#pragma  once

#include <vector>
#include "CNTKLibrary.h"

namespace CNTK
{
    ///
    /// Block Momentum Trainer.
    ///
    class BlockMomentumDistributedTrainer : public DistributedTrainer
    {
        template<class T> using Matrix = Microsoft::MSR::CNTK::Matrix<T>;

    public:
        BlockMomentumDistributedTrainer(
            DistributedCommunicatorPtr communicator,
            size_t globalModelAggregationBlockSize,
            bool useNesterovMomentum,
            bool resetSGDMomentumAfterAggregation,
            double blockLearningRate,
            size_t parallelizationStartAfterSampleCount)
            : BlockMomentumDistributedTrainer(
                  communicator, 
                  globalModelAggregationBlockSize,
                  useNesterovMomentum,
                  resetSGDMomentumAfterAggregation,
                  blockLearningRate,
                  Momentum2TimeConstant(1.0 - 1.0 / (double)communicator->Workers().size(), globalModelAggregationBlockSize),
                  parallelizationStartAfterSampleCount)
        {}

        BlockMomentumDistributedTrainer(
            DistributedCommunicatorPtr communicator,
            size_t globalModelAggregationBlockSize,
            bool useNesterovMomentum,
            bool resetSGDMomentumAfterAggregation,
            double blockLearningRate,
            double blockMomentumAsTimeConstant,
            size_t parallelizationStartAfterSampleCount)
            : DistributedTrainer(parallelizationStartAfterSampleCount),
            m_communicator(communicator),
            m_useNesterovMomentum(useNesterovMomentum),
            m_resetSGDMomentumAfterAggregation(resetSGDMomentumAfterAggregation),
            m_blockLearningRate(blockLearningRate),
            m_blockMomentumAsTimeConstantPerWorker(blockMomentumAsTimeConstant / communicator->Workers().size()),
            m_globalModelAggregationBlockSize(globalModelAggregationBlockSize),
            m_numSamplesSeenInCurrentBlock(0)
        {
            m_syncPeriodPerWorker = globalModelAggregationBlockSize / communicator->Workers().size();
            if (m_syncPeriodPerWorker == 0)
                InvalidArgument("Sync period is too small.");
        }

        // Optional override that gets called before each minbatch during training
        void PreMinibatchCallback(const Trainer& /*trainer*/) override
        {
        }

        // Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
        void PreParameterUpdateCallback(const Trainer& trainer, std::vector<std::pair<Parameter, NDArrayViewPtr>>& parameters, MinibatchInfo& info) override
        {
            m_numSamplesSeenInCurrentBlock += info.numberOfSamples;
            if (m_numSamplesSeenInCurrentBlock < m_syncPeriodPerWorker)
                return;

            if (IsResetRequired(parameters))
            {
                Reset(parameters);
            }

            m_numSamplesSeenInCurrentBlock = 0;

            // Let update the weights.
            if (parameters.front().first.Value()->GetDataType() == DataType::Double)
                ModelAggregationProcessing<double>(parameters);
            else if (parameters.front().first.Value()->GetDataType() == DataType::Float)
                ModelAggregationProcessing<float>(parameters);
            else
                RuntimeError("Unsupported type.");

            if (m_resetSGDMomentumAfterAggregation)
                for(auto learner : trainer.ParameterLearners())
                    learner->ResetSmoothedGradients();
        }

        // Optionally overridable method to get checkpoint state associated with this Distributed train method
        Dictionary GetCheckpointState() const override
        {
            NOT_IMPLEMENTED;
        }

        // Optionally overridable method to restore state pertaining this distributed training method from a previous checkpoint
        void RestoreFromCheckpoint(const Dictionary& /*checkpoint*/) override
        {
            NOT_IMPLEMENTED;
        }

        DistributedCommunicatorPtr GetCommunicator() override
        {
            return m_communicator;
        }

    private:
        bool IsResetRequired(std::vector<std::pair<Parameter, NDArrayViewPtr>>& parameters) const
        {
            if (m_prevParameters.size() != parameters.size() ||
                m_blockLevelSmoothedGradient.size() != parameters.size())
                return true;

            for (size_t i = 0; i < parameters.size(); ++i)
            {
                if (m_prevParameters[i]->Shape() != parameters[i].first.Shape() ||
                    m_prevParameters[i]->Device() != parameters[i].first.Value()->Device() ||
                    m_blockLevelSmoothedGradient[i]->Shape() != parameters[i].first.Shape() ||
                    m_blockLevelSmoothedGradient[i]->Device() != parameters[i].first.Value()->Device())
                {
                    return true;
                }
            }
            return false;
        }

        void Reset(std::vector<std::pair<Parameter, NDArrayViewPtr>>& parameters)
        {
            m_blockLevelSmoothedGradient.resize(parameters.size());
            m_prevParameters.resize(parameters.size());

            for (size_t i = 0; i < parameters.size(); ++i)
            {
                auto& p = parameters[i];

                if (p.first.Value()->GetDataType() == DataType::Double)
                    ResetBuffer<double>(i, p.first);
                else if (p.first.Value()->GetDataType() == DataType::Float)
                    ResetBuffer<float>(i, p.first);
                else
                    RuntimeError("Unsupported type.");
            }
        }

        template<class ElemType>
        void ResetBuffer(size_t index, const Parameter& p)
        {
            auto data = p.Value()->GetMatrix<ElemType>();
            if (!m_blockLevelSmoothedGradient[index])
            {
                // has not been initialized yet
                ::CNTK::NDShape shape{ data->GetNumRows(), data->GetNumCols() };
                auto pSmoothedGrad = std::make_shared<NDArrayView>(AsDataType<ElemType>(), shape, AsDeviceDescriptor(data->GetDeviceId()));
                pSmoothedGrad->SetValue(static_cast<ElemType>(0));
                m_blockLevelSmoothedGradient[index] = pSmoothedGrad;
            }

            if (!m_prevParameters[index])
            {
                NDShape shape{ data->GetNumRows(), data->GetNumCols() };
                NDArrayViewPtr newValue = std::make_shared<NDArrayView>(AsDataType<ElemType>(), shape, AsDeviceDescriptor(data->GetDeviceId()));
                std::shared_ptr<Matrix<ElemType>> newData = newValue->GetWritableMatrix<ElemType>();
                newData->SetValue(*data);
                m_prevParameters[index] = newValue;
            }
            else
            {
                m_prevParameters[index]->GetWritableMatrix<ElemType>()->SetValue(*data);
            }
        }

        template<class ElemType>
        void ModelAggregationProcessing(const std::vector<std::pair<Parameter, NDArrayViewPtr>>& gradientValues)
        {
            ElemType blockMomentum = (ElemType)TimeConstant2Momentum(m_blockMomentumAsTimeConstantPerWorker, m_syncPeriodPerWorker);

            // 1. Let's aggregate weights
            std::vector<std::shared_ptr<Matrix<ElemType>>> aggregatedWeights;
            std::vector<::CNTK::NDArrayViewPtr> aggregatedWeightsPrepared;
            for (size_t i = 0; i < gradientValues.size(); ++i)
            {
                // Get current model
                Matrix<ElemType>& previousWeight = *m_prevParameters[i]->GetWritableMatrix<ElemType>();                  // prev model value
                Matrix<ElemType>& currentWeight = *gradientValues[i].first.Value()->GetWritableMatrix<ElemType>();

                // Subtract it from the previous model
                auto blockGrad = std::make_shared<Matrix<ElemType>>(previousWeight, CPUDEVICE);
                *blockGrad -= currentWeight;                                              // matW becomes local block gradient (of one worker)

                aggregatedWeights.push_back(blockGrad);
                ::CNTK::NDShape shape{ blockGrad->GetNumElements() };
                auto data = ::CNTK::MakeSharedObject<::CNTK::NDArrayView>(::CNTK::AsDataType<ElemType>(), shape, blockGrad->Data(), blockGrad->GetNumElements() * sizeof(ElemType), ::CNTK::AsDeviceDescriptor(blockGrad->GetDeviceId()));
                aggregatedWeightsPrepared.push_back(data);
            }

            // Send block gradient over MPI nodes.
            m_communicator->AggregateInPlace(aggregatedWeightsPrepared, m_communicator->Workers());

            // 2. Let's update the model
            for (size_t i = 0; i < gradientValues.size(); ++i)
            {
                // 2 block gradient aggregation
                // 2.1. get current model
                Matrix<ElemType>& previousWeight = *m_prevParameters[i]->GetWritableMatrix<ElemType>();                  // prev model value
                Matrix<ElemType>& currentWeight = *gradientValues[i].first.Value()->GetWritableMatrix<ElemType>();
                auto blockGrad = aggregatedWeights[i];
                // 2.2. model update 
                {
                    Matrix<ElemType>& sg = *m_blockLevelSmoothedGradient[i]->GetWritableMatrix<ElemType>();       // smoothed gradient
                    blockGrad->TransferToDeviceIfNotThere(sg.GetDeviceId());
                    // 2.2.1 update block level smoothed gradient; 
                    // This is essentially a first-order infinite impulse response (IIR) filter with the gain (1 - blockMomentum)*m_blockLearningRate:
                    // smoothedGradient(t)=blockMomentum * smoothedGradients(t-1) + (1 - blockMomentum)*m_blockLearningRate*blockGrad(t)
                    Matrix<ElemType>::ScaleAndAdd((ElemType)((1 - blockMomentum)*m_blockLearningRate), *blockGrad, (ElemType)blockMomentum, sg);
                    // 2.2.2 update parameters; 
                    currentWeight.SetValue(previousWeight);
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
                    previousWeight.SetValue(currentWeight);
                }
            }
        }

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

        DistributedCommunicatorPtr m_communicator;
        bool m_resetSGDMomentumAfterAggregation;
        bool m_useNesterovMomentum;
        double m_blockLearningRate;
        double m_blockMomentumAsTimeConstantPerWorker;
        size_t m_syncPeriodPerWorker;
        size_t m_globalModelAggregationBlockSize;
        size_t m_numSamplesSeenInCurrentBlock;

        // parameters at the last model aggregation point
        std::vector<NDArrayViewPtr> m_prevParameters;
        std::vector<NDArrayViewPtr> m_blockLevelSmoothedGradient;
    };
}