//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma  once

#include <vector>
#include "CNTKLibrary.h"
#include "DistributedLearnerBase.h"
#include <numeric>
#include <iostream>
#include <sstream>

namespace CNTK
{
    ///
    /// Block Momentum Trainer.
    ///
    class BlockMomentumDistributedLearner : public DistributedLearnerBase
    {
    private:
        enum class Action;
        friend std::ostream& operator<<(std::ostream& out, const Action action)
        {
            static std::map<Action, std::string> actionStr;
            if (actionStr.size() == 0)
            {
                actionStr[Action::Aggregate] = "Aggregate";
                actionStr[Action::AggregateMetrics] = "AggregateMetrics";
                actionStr[Action::Checkpoint] = "Checkpoint";
                actionStr[Action::Shutdown] = "Shutdown";
                actionStr[Action::Wait] = "Wait";
            }
            return out << actionStr[action];
        }

        // Print debug info about synchronization action requested and granted
        void DebugPrintSynchronizeInfo(Action requestedAction, Action grantedAction)
        {
            if (GetTraceLevel() >= TraceLevel::Info)
            {
                std::ostringstream outString;
                outString << "BMUF Rank " << m_communicator->CurrentWorker().m_globalRank << " Action requested " << requestedAction << " Action returned " << grantedAction << std::endl;
                std::cerr << outString.str(); //stderr output
            }
        }

        template<class T> using Matrix = Microsoft::MSR::CNTK::Matrix<T>;

    public:
        BlockMomentumDistributedLearner(
            DistributedCommunicatorPtr communicator,
            LearnerPtr learner,
            size_t distributedAfterSamples,
            size_t globalModelAggregationBlockSize,
            bool useNesterovMomentum,
            bool resetSGDMomentumAfterAggregation,
            double blockLearningRate)
            : BlockMomentumDistributedLearner(
                  communicator,
                  learner,
                  distributedAfterSamples,
                  globalModelAggregationBlockSize,
                  useNesterovMomentum,
                  resetSGDMomentumAfterAggregation,
                  blockLearningRate,
                  Momentum2TimeConstant(1.0 - 1.0 / (double)communicator->Workers().size(), globalModelAggregationBlockSize))
        {}

        BlockMomentumDistributedLearner(
            DistributedCommunicatorPtr communicator,
            LearnerPtr learner,
            size_t distributedAfterSamples,
            size_t globalModelAggregationBlockSize,
            bool useNesterovMomentum,
            bool resetSGDMomentumAfterAggregation,
            double blockLearningRate,
            double blockMomentumAsTimeConstant)
            : DistributedLearnerBase(communicator, learner, distributedAfterSamples),
            m_useNesterovMomentum(useNesterovMomentum),
            m_resetSGDMomentumAfterAggregation(resetSGDMomentumAfterAggregation),
            m_blockLearningRate(blockLearningRate),
            m_blockMomentumAsTimeConstantPerWorker(blockMomentumAsTimeConstant / communicator->Workers().size()),
            m_globalModelAggregationBlockSize(globalModelAggregationBlockSize),
            m_numSamplesSeenInCurrentBlock(0),
            m_endOfDataReached(false),
            m_localTotalNumSamplesSeen(0),
            m_syncPeriodPerWorker(globalModelAggregationBlockSize / communicator->Workers().size())
        {
            if (m_syncPeriodPerWorker == 0)
                InvalidArgument("Sync period is too small.");

            // Need to allocate memory here to make sure not hitting OOM
            std::vector<NDArrayViewPtr> parameterValues;
            GetParameterValues(learner->Parameters(), parameterValues);

            m_blockLevelSmoothedGradient.resize(parameterValues.size());
            m_prevParameters.resize(parameterValues.size());
            m_tempBlockGradient.resize(parameterValues.size());
            Reset(parameterValues);
        }

        size_t MinibatchSizeScaleFactor() override
        {
            return m_communicator->Workers().size();
        }

        bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, MinibatchInfo& info) override
        {
            // mark start of block before local update
            std::vector<NDArrayViewPtr> values;
            GetParameterValues(m_learner->Parameters(), values);

            // note this is only for the first update, after that SyncBlock handles the bookkeeping
            if (!m_prevParamInitialized)
            {
                Reset(values);
                m_prevParamInitialized = true;
            }

            // do local update first, then block update. Local update would have different gradient for each worker,
            // and this order is to make sure all workers got the same model after block update
            if (!info.IsEmpty())
            {
                // For block momentum the number of aggreagate/checkpoints should match, so for now we ignore the return value of local learners.
                auto profWeights = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainWeights);
                m_learner->Update(gradientValues, info.numberOfSamples, info.atEndOfSweep);

                // after local update, use the latest model for block update
                values.clear();
                GetParameterValues(m_learner->Parameters(), values);
            }

            auto profGradientAgg = Microsoft::MSR::CNTK::ProfilerTimeBegin();
            bool updated = PerformDistributedUpdateIfNeeded(values, info);
            Microsoft::MSR::CNTK::ProfilerTimeEnd(profGradientAgg, Microsoft::MSR::CNTK::profilerEvtMainGradient);

            return updated;
        }

        // Optionally overridable method to get checkpoint state associated with this Distributed train method
        Dictionary CreateCheckpoint() override
        {
            std::vector<NDArrayViewPtr> values;
            GetParameterValues(m_learner->Parameters(), values);

            // During checkpoint, other workers could be in aggregation state. Let's allow them to finish aggregation.
            Action action;
            while ((action = SynchronizeAction(Action::Checkpoint)) != Action::Checkpoint)
            {
                DebugPrintSynchronizeInfo(Action::Checkpoint, action);

                if (action == Action::Wait)
                    continue;
                if (action == Action::Aggregate)
                    AggregateImpl(values);
                else
                    RuntimeError("Unexpected action received.");
            }

            DebugPrintSynchronizeInfo(Action::Checkpoint, action);

            // Always aggregate before the checkpoint, so prevParameter and m_numSamplesSeenInCurrentBlock don't need to be saved
            SynchronizeAction(Action::Aggregate);
            AggregateImpl(values);
            
            std::vector<DictionaryValue> serializedSmoothedGradients;
            for (auto sg : m_blockLevelSmoothedGradient)
            {
                serializedSmoothedGradients.push_back(*sg);
            }

            Dictionary result;
            result[L"base"] = DistributedLearnerBase::CreateCheckpoint();
            result[L"localTotalNumSamplesSeen"] = m_localTotalNumSamplesSeen;
            result[L"blockLevelSmoothedGradient"] = serializedSmoothedGradients;
            return result;
        }

        void RestoreFromCheckpoint(const Dictionary& checkpoint) override
        {
            DistributedLearnerBase::RestoreFromCheckpoint(checkpoint[L"base"].Value<Dictionary>());
            m_localTotalNumSamplesSeen = checkpoint[L"localTotalNumSamplesSeen"].Value<size_t>();
            const auto& smoothedGradients = checkpoint[L"blockLevelSmoothedGradient"].Value<std::vector<DictionaryValue>>();

            if (m_blockLevelSmoothedGradient.size() != smoothedGradients.size())
                RuntimeError("Inconsistent parameter size between learner and checkpoint");

            for (size_t i = 0; i < m_blockLevelSmoothedGradient.size(); i++)
            {
                m_blockLevelSmoothedGradient[i]->CopyFrom(smoothedGradients[i].Value<NDArrayView>());
            }

            m_prevParamInitialized = false;
        }

    private:
        // Block momentum needs to do aggregation of loss and eval across workers.
        virtual void DoAggregateMetricsIfNeeded(NDArrayViewPtr& localTrainingLoss, NDArrayViewPtr& localEvalCriterion) override
        {
            m_shutDownSeenBefore = false;
            // If shutdown has been agreed upon before, then return from metrics aggregation. Other shutdown workers won't be able to sync now.
            if (m_communicator->Workers().size() == 1 || m_shutDownSeenBefore)
            {
                return;
            }

            Action action;
            while ((action = SynchronizeAction(Action::AggregateMetrics)) != Action::AggregateMetrics)
            {
                DebugPrintSynchronizeInfo(Action::AggregateMetrics, action);

                std::vector<NDArrayViewPtr> paramValues;
                GetParameterValues(m_learner->Parameters(), paramValues);

                switch (action)
                {
                    // Aggregate params first and try for aggregate metrics again
                    case Action::Aggregate:                        
                        AggregateImpl(paramValues);
                        break;
                    // Can't do checkpointing here since not called from checkpointing code, so return. Checkpointing will be called again eventually.
                    case Action::Checkpoint:
                        return;
                    // Can't aggregate metrics since others are going in shutdown. 
                    case Action::Shutdown:
                        m_shutDownSeenBefore = true;
                        return; // Can't aggregate if another worker is in shutdown mode
                }
            }

            DebugPrintSynchronizeInfo(Action::AggregateMetrics, action);

            // Synchronization complete - Start the loss and eval aggregation
            float averageTrainingLoss = 0;
            if (localTrainingLoss)
            {
                averageTrainingLoss = localTrainingLoss->AsScalar<float>();
            }

            float averageEvalCriterion = 0;
            if (localEvalCriterion)
            {
                averageEvalCriterion = localEvalCriterion->AsScalar<float>();
            }

            NDArrayViewPtr inPlaceAggregateTrainingLoss = std::make_shared<NDArrayView>(averageTrainingLoss, NDShape{}, DeviceDescriptor::CPUDevice());
            NDArrayViewPtr inPlaceAggregateEvalCriterion = std::make_shared<NDArrayView>(averageEvalCriterion, NDShape{}, DeviceDescriptor::CPUDevice());
            vector<NDArrayViewPtr> inPlaceAggregateVector = { inPlaceAggregateTrainingLoss, inPlaceAggregateEvalCriterion };

            m_communicator->AggregateInPlace(inPlaceAggregateVector, m_communicator->Workers());
            
            if (localTrainingLoss)
            {
                inPlaceAggregateTrainingLoss->SetValue(inPlaceAggregateTrainingLoss->AsScalar<float>() / m_communicator->Workers().size());
                localTrainingLoss->CopyFrom(*inPlaceAggregateTrainingLoss);
            }

            if (localEvalCriterion)
            {
                inPlaceAggregateEvalCriterion->SetValue(inPlaceAggregateEvalCriterion->AsScalar<float>() / m_communicator->Workers().size());
                localEvalCriterion->CopyFrom(*inPlaceAggregateEvalCriterion);
            }
        }

        // Optional override that gets called per minibatch after finishing gradient computation but before updating model parameters
        bool PerformDistributedUpdateIfNeeded(std::vector<NDArrayViewPtr>& parameterValues, MinibatchInfo& info)
        {
            // If the last minibatch, set the end of data state.
            if (info.atEndOfData)
                m_endOfDataReached = true;

            m_localTotalNumSamplesSeen += info.numberOfSamples;
            m_sampleCount += info.numberOfSamples;

            if (m_distributeAfterSamples > m_sampleCount)
            {
                if (m_endOfDataReached)
                {
                    // We have not even reached distributed state,
                    // simply stop processing by returning false.
                    return false;
                }
                return true;
            }

            if (!m_endOfDataReached)
            {
                m_numSamplesSeenInCurrentBlock += info.numberOfSamples;
                if (m_numSamplesSeenInCurrentBlock < m_syncPeriodPerWorker)
                    return true;

                Aggregate(parameterValues);
                return true;
            }

            return Shutdown(parameterValues);
        }

        // Before doing any work, the distributed learner synchronizes with other learners to
        // decide what to do next.
        // The priority of actons are:
        // 1) If any worker wants to aggregate - aggregation is done.
        // 2) If any worker wants to checkpoint and nobody wants to aggregate - checkpointing is done. If anyone wants to aggregate metrics, wait to allow it to come in checkpoint state.
        // 3) If all want to shutdown - it means we reached the end of the data and shutdown can be done.If anyone wants to aggregate metrics, wait to allow it to come in shutdown state.
        // 4) If all worker wants to aggregate metrics - metrics aggregation is done. Otherwise return aggregate, checkpoint or shutdown if anyone else wants it
        // The priority above eliminate resolves situations when some of the workers run out of data
        // and other workers require checkpointing or aggregation.
        enum class Action
        {
            Wait, // Waits in the current state without doing anything.
            Aggregate,
            AggregateMetrics, // Used to allow aggregation of loss and eval metrics.
            Checkpoint,
            Shutdown
        };

        void GetParameterValues(const std::vector<Parameter>& parameters, std::vector<NDArrayViewPtr>& result)
        {
            for (auto p : parameters)
                result.push_back(p.Value());
        }

        void Aggregate(std::vector<NDArrayViewPtr>& parameters)
        {
            // Synchronization action. Aggregate has the highest priority, so the expected result is aggregate.
            Action action = SynchronizeAction(Action::Aggregate);
            if (action != Action::Aggregate)
                LogicError("Unexpected action during aggregation.");

            AggregateImpl(parameters);
        }

        bool Shutdown(std::vector<NDArrayViewPtr>& parameters)
        {
            // During shutdown, other workers could be in checkpointing or aggregation state.
            // Finished workers should properly behave in this case.
            Action action;
            while ((action = SynchronizeAction(Action::Shutdown)) != Action::Shutdown)
            {
                DebugPrintSynchronizeInfo(Action::Shutdown, action);

                switch (action)
                {
                case Action::Aggregate:
                    AggregateImpl(parameters);
                    break;
                case Action::Checkpoint:
                    // Somebody still has to call the checkpoint from the outside.
                    return true;
                case Action::Wait:
                    // Someone is in aggregate metrics. Wait for it to come to shutdown.
                    continue;
                default:
                    RuntimeError("Unexpected action received.");
                }
            }

            DebugPrintSynchronizeInfo(Action::Shutdown, action);

            // Last synchronization
            AggregateImpl(parameters);
            return false; // Make compiler happy.
        }

        // Synchronize(Agree) on action before doing it. This is needed to prevent deadlock in MPI. 
        // Aggregate is highest priority. So AggregateImpl can be called after calling SynchronizeAction(Action::Aggreagte). 
        // Others need to ask for permission in a loop
        Action SynchronizeAction(Action self)
        {
            assert(self == Action::Checkpoint || self == Action::Aggregate || self == Action::Shutdown || self == Action::AggregateMetrics);

            double data[2] = { static_cast<double>(self), static_cast<double>(m_localTotalNumSamplesSeen) };
            auto a = std::make_shared<NDArrayView>(DataType::Double, NDShape{ 2 }, &data, sizeof(double) * 2, DeviceDescriptor::CPUDevice());
            m_communicator->Concatenate(std::vector<NDArrayViewPtr> { a }, m_actionBuffer, m_communicator->Workers());
            assert(m_actionBuffer.size() == 1);

            auto buffer = m_actionBuffer.front()->DataBuffer<double>();
            auto bufferSize = m_actionBuffer.front()->Shape().TotalSize();
            auto bufferEnd = buffer + bufferSize;

            std::vector<Action> actions;
            actions.reserve(m_communicator->Workers().size());

            std::vector<size_t> localNumberOfSamples;
            localNumberOfSamples.reserve(m_communicator->Workers().size());

            for (const double* start = buffer; start != bufferEnd; start +=2)
            {
                actions.push_back(static_cast<Action>((int)*start));
                localNumberOfSamples.push_back(static_cast<size_t>(*(start + 1)));
            }
            m_sampleCount = std::accumulate(localNumberOfSamples.begin(), localNumberOfSamples.end(), (size_t)0);

            // If all want to aggregate metrics, only then we aggregate metrics.
            if (std::all_of(actions.begin(), actions.end(), [](Action c) { return c == Action::AggregateMetrics; }))
                return Action::AggregateMetrics;

            // If all want to shutdown - we shutdown.
            if (std::all_of(actions.begin(), actions.end(), [](Action c) { return c == Action::Shutdown; }))
                return Action::Shutdown;

            // If all want to checkpoint - we checkpoint.
            if (std::all_of(actions.begin(), actions.end(), [](Action c) { return c == Action::Checkpoint; }))
                return Action::Checkpoint;

            // If all are either in Checkpoint, Shutdown or AggregateMetrics, 
            //      Then AggregateMetrics state has lowest priority. Workers in it return without doing anything. Other workers wait for Aggregate Metrics to come in their state.
            //      Between Checkpoint and Shutdown, Shutdown has lower priority. Shutdown worker will return and checkpoint worker will wait for others to come in checkpoint state.
            if (std::all_of(actions.begin(), actions.end(), [](Action c) { return c == Action::Checkpoint || c == Action::Shutdown || c == Action::AggregateMetrics; }))
            {
                bool isAnyCheckpoint = std::any_of(actions.begin(), actions.end(), [](Action c) { return c == Action::Checkpoint; });
                bool isAnyShutdown = std::any_of(actions.begin(), actions.end(), [](Action c) { return c == Action::Shutdown; });
                bool isAnyAggregateMetrics = std::any_of(actions.begin(), actions.end(), [](Action c) { return c == Action::AggregateMetrics; });
                if (self == Action::Shutdown)
                {
                    // Do checkpoint first if any other requests checkpoint. Then come back to shutdown.
                    if (isAnyCheckpoint)
                    {
                        return Action::Checkpoint;
                    }

                    // Allow the aggregate metrics to come in shutdown state and request again.
                    if (isAnyAggregateMetrics)
                    {
                        return Action::Wait;
                    }

                    return Action::Shutdown;
                }
                else if (self == Action::Checkpoint)
                {
                    // Wait for other in shutdown or aggregate metrics state to come to checkpoint state
                    if (isAnyShutdown || isAnyAggregateMetrics)
                    {
                        return Action::Wait;
                    }

                    return Action::Checkpoint;
                }
                else if (self == Action::AggregateMetrics)
                {
                    // AggregateMetrics can't do aggregate metrics if anyone is in shutdown
                    if (isAnyShutdown)
                    {
                        return Action::Shutdown;
                    }

                    // If all others are either metrics aggregate or checkpoint then state returned is checkpoint and we don't do metrics aggregation
                    return Action::Checkpoint;
                }
            }

            // Otherwise we aggregate. This is given priority by all other workers in checkpoint, shutdown or aggregate metrics states.
            return Action::Aggregate;
        }

        void AggregateImpl(std::vector<NDArrayViewPtr>& parameters)
        {
            // Let update the weights.
            if (parameters.front()->GetDataType() == DataType::Double)
                SynchronizeModel<double>(parameters);
            else if (parameters.front()->GetDataType() == DataType::Float)
                SynchronizeModel<float>(parameters);
            else if (parameters.front()->GetDataType() == DataType::Float16)
            {
                SynchronizeModel<half>(parameters);

                // For half, SynchronizeModel will update the parameters (half) in network.
                // However, the local learner also have a copy of full precision parameters (float), which needs to be updated too.
                // Set the flag so that the float copy will be updated / copied from half parameters.
                m_learner->SetNeedToUpdateMasterParameter();
            }
            else
                RuntimeError("Unsupported type.");

            m_numSamplesSeenInCurrentBlock = 0;

            if (m_resetSGDMomentumAfterAggregation)
                m_learner->ResetSmoothedGradients();
        }

        Dictionary CreateCheckpointImpl(std::vector<NDArrayViewPtr>& parameters)
        {
            // During checkpoint, other workers could be in aggregation state. Let's allow them to finish aggregation.
            Action action;
            while ((action = SynchronizeAction(Action::Checkpoint)) != Action::Checkpoint)
            {
                DebugPrintSynchronizeInfo(Action::Checkpoint, action);

                if (action == Action::Wait)
                    continue;
                if (action == Action::Aggregate)
                    AggregateImpl(parameters);
                else
                    RuntimeError("Unexpected action received.");
            }

            DebugPrintSynchronizeInfo(Action::Checkpoint, action);

            return DistributedLearnerBase::CreateCheckpoint();
        }

        bool IsResetRequired(std::vector<NDArrayViewPtr>& parameters) const
        {
            if (m_prevParameters.size() != parameters.size() ||
                m_blockLevelSmoothedGradient.size() != parameters.size())
                return true;

            for (size_t i = 0; i < parameters.size(); ++i)
            {
                if (m_prevParameters[i]->Shape() != parameters[i]->Shape() ||
                    m_prevParameters[i]->Device() != parameters[i]->Device() ||
                    m_blockLevelSmoothedGradient[i]->Shape() != parameters[i]->Shape() ||
                    m_blockLevelSmoothedGradient[i]->Device() != parameters[i]->Device())
                {
                    return true;
                }
            }
            return false;
        }

        void Reset(const std::vector<NDArrayViewPtr>& parameters)
        {
            for (size_t i = 0; i < parameters.size(); ++i)
            {
                auto& p = parameters[i];

                if (p->GetDataType() == DataType::Double)
                    ResetBuffer<double>(i, p);
                else if (p->GetDataType() == DataType::Float)
                    ResetBuffer<float>(i, p);
                else if (p->GetDataType() == DataType::Float16)
                    ResetBuffer<half, float16>(i, p);
                else
                    RuntimeError("Unsupported type.");
            }
        }

        template<class ElemTypeV1, class ElemTypeV2=ElemTypeV1>
        void ResetBuffer(size_t index, const NDArrayViewPtr& p)
        {
            auto data = p->GetMatrix<ElemTypeV1>();
            if (!m_blockLevelSmoothedGradient[index])
            {
                // has not been initialized yet
                auto pSmoothedGrad = std::make_shared<NDArrayView>(AsDataType<ElemTypeV2>(), p->Shape(), AsDeviceDescriptor(data->GetDeviceId()));
                pSmoothedGrad->SetValue(static_cast<ElemTypeV2>(0));
                m_blockLevelSmoothedGradient[index] = pSmoothedGrad;
            }

            if (!m_prevParameters[index])
            {
                NDArrayViewPtr newValue = std::make_shared<NDArrayView>(AsDataType<ElemTypeV2>(), p->Shape(), AsDeviceDescriptor(data->GetDeviceId()));
                std::shared_ptr<Matrix<ElemTypeV1>> newData = newValue->GetWritableMatrix<ElemTypeV1>();
                newData->SetValue(*data);
                m_prevParameters[index] = newValue;
            }
            else
            {
                m_prevParameters[index]->GetWritableMatrix<ElemTypeV1>()->SetValue(*data);
            }

            if (!m_tempBlockGradient[index])
            {
                m_tempBlockGradient[index] = std::make_shared<NDArrayView>(AsDataType<ElemTypeV2>(), p->Shape(), AsDeviceDescriptor(data->GetDeviceId()));
            }
        }

        template<class ElemType>
        void SynchronizeModel(const std::vector<NDArrayViewPtr>& parameterValues)
        {
            ElemType blockMomentum = (ElemType)TimeConstant2Momentum(m_blockMomentumAsTimeConstantPerWorker, m_numSamplesSeenInCurrentBlock);

            // 1. Let's aggregate weights
            for (size_t i = 0; i < parameterValues.size(); ++i)
            {
                // Get current model
                Matrix<ElemType>& previousWeight = *m_prevParameters[i]->GetWritableMatrix<ElemType>();                  // prev model value
                Matrix<ElemType>& currentWeight = *parameterValues[i]->GetWritableMatrix<ElemType>();
                Matrix<ElemType>& blockGrad = *m_tempBlockGradient[i]->GetWritableMatrix<ElemType>();

                // Subtract it from the previous model
                blockGrad = previousWeight - currentWeight; // matW becomes local block gradient (of one worker)
            }

            // Send block gradient over MPI nodes.
            m_communicator->AggregateInPlace(m_tempBlockGradient, m_communicator->Workers());

            // 2. Let's update the model
            for (size_t i = 0; i < parameterValues.size(); ++i)
            {
                // 2 block gradient aggregation
                // 2.1. get current model
                Matrix<ElemType>& previousWeight = *m_prevParameters[i]->GetWritableMatrix<ElemType>();                  // prev model value
                Matrix<ElemType>& currentWeight = *parameterValues[i]->GetWritableMatrix<ElemType>();
                Matrix<ElemType>& blockGrad = *m_tempBlockGradient[i]->GetWritableMatrix<ElemType>();
                // 2.2. model update 
                {
                    Matrix<ElemType>& sg = *m_blockLevelSmoothedGradient[i]->GetWritableMatrix<ElemType>();       // smoothed gradient
                    // 2.2.1 update block level smoothed gradient; 
                    // This is essentially a first-order infinite impulse response (IIR) filter with the gain (1 - blockMomentum)*m_blockLearningRate:
                    // smoothedGradient(t)=blockMomentum * smoothedGradients(t-1) + (1 - blockMomentum)*m_blockLearningRate*blockGrad(t)
                    Matrix<ElemType>::ScaleAndAdd((ElemType)((1 - blockMomentum)*m_blockLearningRate), blockGrad, (ElemType)blockMomentum, sg);
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
            if (timeConstant == 0)
                return 0;
            else
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

        const bool m_resetSGDMomentumAfterAggregation;
        const bool m_useNesterovMomentum;
        const double m_blockLearningRate;
        const double m_blockMomentumAsTimeConstantPerWorker;

        const size_t m_syncPeriodPerWorker;
        const size_t m_globalModelAggregationBlockSize;
        size_t m_numSamplesSeenInCurrentBlock;
        size_t m_localTotalNumSamplesSeen;

        // parameters at the last model aggregation point
        std::vector<NDArrayViewPtr> m_prevParameters;
        std::vector<NDArrayViewPtr> m_blockLevelSmoothedGradient;
        std::vector<NDArrayViewPtr> m_tempBlockGradient;

        // temp storage for MPI
        std::vector<NDArrayViewPtr> m_actionBuffer;

        bool m_prevParamInitialized = false;

        bool m_endOfDataReached;
        bool m_shutDownSeenBefore = false;

        DISABLE_COPY_AND_MOVE(BlockMomentumDistributedLearner);
     };
}
