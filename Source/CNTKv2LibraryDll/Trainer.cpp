//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "Learner.h"
#include "PerformanceProfiler.h"
#include "CompositeFunction.h"
#include "Serialization.h"

namespace
{
    const std::wstring versionPropertyName = L"Version";
    const std::wstring learnersPropertyName = L"Learners";
    const std::wstring externalStatePropertyName = L"ExternalState";
    const std::wstring distributedStatePropertyName = L"DistributedState";

    // Version history:
    // 0 -- a version number before the versioning was introduced for the trainer's checkpoints.
    // 1 -- initial version: added a key-value pair for the checkpoint version info, added
    //      distributed state key to save all local state collected from distributed workers.
    static const size_t trainerCheckpointVersion = 1;
}

namespace CNTK
{
    Trainer::Trainer(const FunctionPtr& model, const FunctionPtr& lossFunction,
                     const std::vector<LearnerPtr>& parameterLearners,
                     const std::vector<ProgressWriterPtr>& progressWriters)
        : Trainer(model, lossFunction, nullptr, parameterLearners, progressWriters)
    {}

    Trainer::Trainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const FunctionPtr& evaluationFunction,
                     const std::vector<LearnerPtr>& parameterLearners,
                    const std::vector<ProgressWriterPtr>& progressWriters) 
        : Evaluator(evaluationFunction, progressWriters, false),
          m_model(model),
          m_lossFunction(lossFunction),
          m_parameterLearners(std::make_shared<Learners>(parameterLearners)),
          m_prevMinibatchNumSamples(0),
          m_distributed(false),
          m_aggregatedTrainingLossValue(std::make_shared<Accumulator>()),
          m_aggregatedTrainingEvalCriterionValue(),
          m_prevDistributedTotalNumSamples(0)
    {
        std::vector<Variable> combinedFunctionArgs;
        if (m_model) // model is optional, since it may not be adding any information on top of lossFunction
            combinedFunctionArgs = m_model->Outputs();

        combinedFunctionArgs.push_back(m_lossFunction);

        if (m_lossFunction->Output().GetDataType() == DataType::Float16)
            fprintf(stderr, "WARNING: using Float16 for loss function may cause overflow, please cast to float");

        if (!m_lossFunction->Output().DynamicAxes().empty())
        {
            m_aggregatedLossFunction = ReduceSum(lossFunction, Axis::AllAxes(), L"aggregateLoss");
            combinedFunctionArgs.push_back(m_aggregatedLossFunction);
            m_trainingSampleCountVar = m_lossFunction;
        }
        else
        {
            m_aggregatedLossFunction = m_lossFunction;

            std::function<std::pair<Variable, bool>(const FunctionPtr& root)> FindTrainingSampleCountVar;
            FindTrainingSampleCountVar = [&FindTrainingSampleCountVar](const FunctionPtr& root) -> std::pair<Variable, bool> {
                const auto& outputs = root->Outputs();
                auto firstOutputWithDynamicAxes = std::find_if(outputs.begin(), outputs.end(), [](const Variable& var) { return !var.DynamicAxes().empty(); });
                if (firstOutputWithDynamicAxes != outputs.end())
                    return std::make_pair(*firstOutputWithDynamicAxes, true);

                const auto& inputs = root->Inputs();
                for (const auto& input : inputs)
                {
                    if (!input.DynamicAxes().empty())
                        return std::make_pair(input, true);

                    if (input.IsOutput())
                    {
                        auto retVal = FindTrainingSampleCountVar(input.Owner());
                        if (retVal.second)
                            return retVal;
                    }
                }
                return std::make_pair(Variable(), false);
            };

            auto findTrainingSampleCountVarRetVal = FindTrainingSampleCountVar(m_lossFunction->RootFunction());
            if (!findTrainingSampleCountVarRetVal.second)
                InvalidArgument("Trainer: Failed to find a variable underlying the graph rooted at specified loss function '%S', from which the training sample count can be determined.", m_lossFunction->RootFunction()->AsString().c_str());

            m_trainingSampleCountVar = findTrainingSampleCountVarRetVal.first;
            if (GetTraceLevel() >= TraceLevel::Info)
                fprintf(stderr, "Info: Trainer loss Function '%S' output does not have a batch axis; the first Variable '%S' with a batch axis found in the graph underlying the scalar "
                                "loss Function will be used to determine minibatch training sample count.\n", m_lossFunction->AsString().c_str(), m_trainingSampleCountVar.AsString().c_str());

            if (std::find(combinedFunctionArgs.begin(), combinedFunctionArgs.end(), m_trainingSampleCountVar) == combinedFunctionArgs.end())
                combinedFunctionArgs.push_back(m_trainingSampleCountVar);
        }

        if (evaluationFunction)
        {
            auto evalArgs = GetCombinedEvalFunctionArgs();
            combinedFunctionArgs.insert(combinedFunctionArgs.end(), evalArgs.begin(), evalArgs.end());

            m_aggregatedTrainingEvalCriterionValue = std::make_shared<Accumulator>();
        }

        // create a default eval value in case there's no criterion
        m_prevMinibatchAggregateEvalCriterionValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(0, m_aggregatedLossFunction->Output().GetDataType(), NDShape{}, DeviceDescriptor::CPUDevice()));

        m_combinedTrainingFunction = Combine(combinedFunctionArgs);
        SetCombinedEvalFunction(m_combinedTrainingFunction);

        auto modelParameters = m_combinedTrainingFunction->Parameters();
        m_learnerParameters = m_parameterLearners->GetParameters();
        std::unordered_set<Parameter> modelParametersSet(modelParameters.begin(), modelParameters.end());
        std::unordered_set<Parameter> learnerParametersNotPartOfModel;
        for (const auto& learnerParameter : m_learnerParameters)
        {
            if (modelParametersSet.find(learnerParameter) == modelParametersSet.end())
                learnerParametersNotPartOfModel.insert(learnerParameter);
        }

        for (const auto& modelParameter : modelParametersSet)
        {
            if (m_learnerParameters.find(modelParameter) == m_learnerParameters.end())
                m_modelParametersNotCoveredByLearners.insert(modelParameter);
        }

        if (!learnerParametersNotPartOfModel.empty())
            InvalidArgument("Trainer ctor: %d of the learner parameters '%S' are not part of the model specified", 
                            (int)learnerParametersNotPartOfModel.size(), NamedListString(learnerParametersNotPartOfModel).c_str());

        if (!m_modelParametersNotCoveredByLearners.empty())
            fprintf(stderr, "[Note:] Trainer ctor: %d of the model parameters are not covered by any of the specified Learners; these parameters will not be learned\n", (int)m_modelParametersNotCoveredByLearners.size());

        m_distributed = m_parameterLearners->IsDistributed();

        if (m_distributed)
            Evaluator::SetCommunicator(dynamic_cast<DistributedLearner*>(m_parameterLearners->ParameterLearners()[0].get())->GetCommunicator());

        for (auto& learner : m_parameterLearners->ParameterLearners())
        {
            learner->AddProgressWriters(progressWriters);
        }
    }

    bool Trainer::TrainMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        std::unordered_map<Variable, ValuePtr> outputsToFetch = {};
        return TrainMinibatch(arguments, outputsToFetch, computeDevice);
    }

    bool Trainer::TrainMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
#ifndef  CNTK_UWP
        auto profMinibatch = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainMinibatch);
#endif

        bool result = (!m_distributed) ?
            TrainLocalMinibatch(GetInputs(arguments), outputsToFetch, IsAtSweepEnd(arguments), computeDevice) :
            TrainDistributedMinibatch(GetInputs(arguments), outputsToFetch, IsAtSweepEnd(arguments), computeDevice);

        // TODO: exclude updating progress writers from profiling?
        UpdateTrainingProgress(m_prevMinibatchNumSamples, m_prevMinibatchAggregateTrainingLossValue,
                               m_prevMinibatchAggregateEvalCriterionValue, computeDevice);
        return result;
    }

    bool Trainer::TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, bool isSweepEndInArguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        std::unordered_map<Variable, ValuePtr> outputsToFetch = {};
        return TrainMinibatch(arguments, isSweepEndInArguments, outputsToFetch, computeDevice);
    }

    bool Trainer::TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, bool isSweepEndInArguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
#ifndef  CNTK_UWP
        auto profMinibatch = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainMinibatch);
#endif

        bool result = (!m_distributed) ?
            TrainLocalMinibatch(arguments, outputsToFetch, isSweepEndInArguments, computeDevice) :
            TrainDistributedMinibatch(arguments, outputsToFetch, isSweepEndInArguments, computeDevice);

        // TODO: exclude updating progress writers from profiling?
        UpdateTrainingProgress(m_prevMinibatchNumSamples, m_prevMinibatchAggregateTrainingLossValue,
                               m_prevMinibatchAggregateEvalCriterionValue, computeDevice);
        return result;
    }

    bool Trainer::TrainLocalMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, bool sweepEnd, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        bool emptyMinibatch = arguments.empty() || (arguments.begin()->second == nullptr);
        if (emptyMinibatch) // Nothing to train with.
        {
            m_prevMinibatchNumSamples = 0;
            return false;
        }

        std::unordered_map<Variable, ValuePtr> parameterGradients;
        ExecuteForwardBackward(arguments, outputsToFetch, computeDevice, parameterGradients);

#ifndef  CNTK_UWP
        auto profWeights = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainWeights);
#endif

        std::unordered_map<Parameter, NDArrayViewPtr> gradients;
        for (const auto& parameter : m_learnerParameters)
            gradients[parameter] = parameterGradients[parameter]->Data();
        return m_parameterLearners->Update(gradients, m_prevMinibatchNumSamples, sweepEnd);
    }

    bool Trainer::TrainDistributedMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, bool sweepEnd, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        std::unordered_map<Parameter, NDArrayViewPtr> gradients;
        gradients.reserve(m_learnerParameters.size());

        bool emptyMinibatch = arguments.empty() || (arguments.begin()->second == nullptr);
        NDArrayViewPtr trainingLoss = nullptr;
        NDArrayViewPtr evalCriterion = nullptr;
        if (emptyMinibatch)
        {
            m_prevMinibatchNumSamples = 0;
            // Gradients are not existing.
            for (const auto& parameter : m_learnerParameters)
                gradients[parameter] = nullptr;

            trainingLoss = MakeSharedObject<NDArrayView>(0, (m_aggregatedLossFunction ? m_aggregatedLossFunction->Output().GetDataType() : DataType::Float), NDShape{}, computeDevice);
            evalCriterion = MakeSharedObject<NDArrayView>(0, (m_aggregatedEvaluationFunction ? m_aggregatedEvaluationFunction->Output().GetDataType() : DataType::Float), NDShape{}, computeDevice);
        }
        else
        {
            // Get gradients after forward/backward pass.
            std::unordered_map<Variable, ValuePtr> parameterGradients;

            // ExecuteForwardBackward updates m_prevMinibatchNumSamples to the local value.
            ExecuteForwardBackward(arguments, outputsToFetch, computeDevice, parameterGradients);
            for (const auto& parameter : m_learnerParameters)
                gradients[parameter] = parameterGradients[parameter]->Data();
            trainingLoss = m_prevMinibatchAggregateTrainingLossValue->Data();
            evalCriterion = m_prevMinibatchAggregateEvalCriterionValue->Data();
        }

        auto currentWorkerNumSamples = m_prevMinibatchNumSamples;
        auto prevTotalNumSamples = TotalNumberOfSamplesSeen();

        MinibatchInfo info{ arguments.empty(), sweepEnd, m_prevMinibatchNumSamples, trainingLoss, evalCriterion };
        bool updated = m_parameterLearners->Update(gradients, info);

        // Here we update m_prevMinibatchNumSamples with aggregated value in the
        // case of distributed learner.
        m_prevMinibatchNumSamples = info.numberOfSamples;
    
        // Update internal state.
        if (emptyMinibatch)
        {
            // Have to reassign loss and criterion.
            m_prevMinibatchAggregateEvalCriterionValue = std::make_shared<Value>(info.evalCriterionValue);
            m_prevMinibatchAggregateTrainingLossValue = std::make_shared<Value>(info.trainingLossValue);
        }

        // Did we do a distributed sync?
        // We determine this by checking if the increase in total #samples is > #samples processed by local worker
        auto currentTotalNumSamples = TotalNumberOfSamplesSeen();
        if ((currentTotalNumSamples - prevTotalNumSamples) > currentWorkerNumSamples)
        {
            for (auto& progressWriter : m_progressWriters)
                progressWriter->UpdateDistributedSync(currentTotalNumSamples - m_prevDistributedTotalNumSamples, nullptr);

            m_prevDistributedTotalNumSamples = currentTotalNumSamples;
        }
        return updated;
    }

    void Trainer::UpdateTrainingProgress(size_t numSamples, const ValuePtr& loss, const ValuePtr& evalCriterion,
                                         const DeviceDescriptor& computeDevice)
    {
        if (numSamples == 0)
        {
            return;
        }

        m_aggregatedTrainingLossValue->Update(loss, computeDevice);
     
        if (m_aggregatedTrainingEvalCriterionValue)
        {
            m_aggregatedTrainingEvalCriterionValue->Update(evalCriterion, computeDevice);
        }

        for (auto& progressWriter : m_progressWriters)
        {
            progressWriter->UpdateTraining(numSamples, m_aggregatedTrainingLossValue, m_aggregatedTrainingEvalCriterionValue);
        }
    }

    void Trainer::SummarizeTrainingProgress()
    {
        // Aggregate across workers training loss and eval criteria. Needed for BMUF like learner which don't aggregate after every minibatch.
        if (m_parameterLearners->DoAggregateMetricsIfNeededLambda)
        {
            NDArrayViewPtr localLossValue = nullptr;
            if (m_aggregatedTrainingLossValue && m_aggregatedTrainingLossValue->IsInitialized())
            {
                localLossValue = m_aggregatedTrainingLossValue->Data();
            }

            NDArrayViewPtr localEvalCriterion = nullptr;
            if (m_aggregatedTrainingEvalCriterionValue && m_aggregatedTrainingEvalCriterionValue->IsInitialized())
            {
                localEvalCriterion = m_aggregatedTrainingEvalCriterionValue->Data();
            }

            m_parameterLearners->DoAggregateMetricsIfNeededLambda(localLossValue, localEvalCriterion);
        }

        for (auto& progressWriter : m_progressWriters)
        {
            progressWriter->WriteTrainingSummary(m_aggregatedTrainingLossValue, m_aggregatedTrainingEvalCriterionValue);
        }

        m_aggregatedTrainingLossValue->Reset();

        if (m_aggregatedTrainingEvalCriterionValue)
        {
            m_aggregatedTrainingEvalCriterionValue->Reset();
        }
    }

    void Trainer::AddProgressWriters(const std::vector<ProgressWriterPtr>& progressWriters)
    {
        for (auto& learner : m_parameterLearners->ParameterLearners()) 
        {
            learner->AddProgressWriters(progressWriters);
        }
        m_progressWriters.insert(progressWriters.begin(), progressWriters.end());
    }

    void Trainer::PrintNodeTiming()
    {
        if (m_combinedTrainingFunction)
        {
            m_combinedTrainingFunction->PrintNodeTiming();
        }
    }


    void Trainer::ExecuteForwardBackward(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice, std::unordered_map<Variable, ValuePtr>& parameterGradients)
    {
#ifndef  CNTK_UWP
        auto profForwardBackward = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainFB);
#endif
        std::unordered_map<Variable, ValuePtr> outputs = { { m_aggregatedLossFunction, nullptr }, { m_trainingSampleCountVar, nullptr } };
        if (m_aggregatedEvaluationFunction)
            outputs.insert({ m_aggregatedEvaluationFunction, nullptr });

        outputs.insert(outputsToFetch.begin(), outputsToFetch.end());

        auto backPropSate = m_combinedTrainingFunction->Forward(arguments, outputs, computeDevice, { m_aggregatedLossFunction }, m_modelParametersNotCoveredByLearners);
        m_prevMinibatchAggregateTrainingLossValue = outputs[m_aggregatedLossFunction];
        if (m_aggregatedEvaluationFunction)
            m_prevMinibatchAggregateEvalCriterionValue = outputs[m_aggregatedEvaluationFunction];

        for (auto outputToFetch : outputsToFetch)
        {
            if (outputToFetch.second == nullptr)
                outputsToFetch[outputToFetch.first] = outputs[outputToFetch.first];
        }

        if(!m_rootGradientValue ||
            m_aggregatedLossFunction->Output().GetDataType() != m_rootGradientValue->GetDataType() ||
            m_prevMinibatchAggregateTrainingLossValue->Shape() != m_rootGradientValue->Shape() ||
            computeDevice != m_rootGradientValue->Device() ||
            outputs.at(m_aggregatedLossFunction)->Mask() != m_rootGradientValue->Mask())
        {
            m_rootGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(m_aggregatedLossFunction->Output().GetDataType(), m_prevMinibatchAggregateTrainingLossValue->Shape(), computeDevice), outputs.at(m_aggregatedLossFunction)->Mask());
        }

        DataType aggregateDataType = m_aggregatedLossFunction->Output().GetDataType();

        if (aggregateDataType == DataType::Float)
            m_rootGradientValue->Data()->SetValue(1.0f);
        else if (aggregateDataType == DataType::Double)
            m_rootGradientValue->Data()->SetValue(1.0);
        else if (aggregateDataType == DataType::Float16)
            m_rootGradientValue->Data()->SetValue(float16(1.0));
        else
            RuntimeError("DataType %s is not supported for root gradients", DataTypeName(aggregateDataType));

        for (const auto& parameter : m_learnerParameters)
            parameterGradients[parameter] = nullptr;

        // TODO: Why Backward signature does not take Parameter instead of Variable for gradients?
        m_combinedTrainingFunction->Backward(backPropSate, { { m_aggregatedLossFunction, m_rootGradientValue } }, parameterGradients);
        m_prevMinibatchNumSamples = GetSampleCount(m_trainingSampleCountVar, outputs[m_trainingSampleCountVar]);
    }

    static std::wstring GetTrainerStateCheckpointFilePath(const std::wstring& modelFilePath)
    {
        const wchar_t* checkpointExt = L".ckp";
        return modelFilePath + checkpointExt;
    }

    void Trainer::SaveCheckpoint(const std::wstring& modelFilePath, Dictionary externalState)
    {
        auto learnersState = m_parameterLearners->CreateCheckpoint();

        if (!m_distributed)
            return Save(modelFilePath, learnersState, externalState);

        auto compositeFunction = dynamic_cast<CompositeFunction*>(m_combinedTrainingFunction.get());

        Dictionary state;
        state[internalWorkerStateKey] = compositeFunction->GetInternalState(); // this is the local worker's state.
        state[externalWorkerStateKey] = externalState;

        // Collect distributed external state.
        DistributedCommunicatorPtr communicator = MPICommunicator();
        communicator->Barrier();

        std::vector<DictionaryPtr> remoteState;
        communicator->Gather(state, remoteState, communicator->Workers());

        Dictionary aggregatedState;
        for (const auto& w : communicator->Workers())
        {
            aggregatedState[std::to_wstring(w.m_globalRank)] = *remoteState[w.m_globalRank];
        }

        if (communicator->CurrentWorker().IsMain())
            Save(modelFilePath, learnersState, externalState, aggregatedState);

        // all workers need to sync up after saving model to avoid read-after-write hazard
        // i.e. one worker is in the middle of write while another tries to read
        communicator->Barrier();
    }

    void Trainer::Save(const std::wstring& modelFilePath, const std::vector<DictionaryValue>& learnerState, const Dictionary& externalState, const Dictionary& distributedState)
    {
        std::wstring tempModelFile = modelFilePath + L".tmp";
        Dictionary state;
        state[versionPropertyName] = trainerCheckpointVersion;
        state[learnersPropertyName] = learnerState;
        state[externalStatePropertyName] = externalState;
        state[distributedStatePropertyName] = distributedState;

        m_combinedTrainingFunction->Save(tempModelFile);
        std::wstring trainerStateCheckpointFilePath = GetTrainerStateCheckpointFilePath(modelFilePath);
        std::wstring tempCheckpointFile = trainerStateCheckpointFilePath + L".tmp";

        state.Save(tempCheckpointFile);

        // The return value is ignored here.
        _wunlink(modelFilePath.c_str());
        _wunlink(trainerStateCheckpointFilePath.c_str());

        renameOrDie(tempModelFile, modelFilePath);
        renameOrDie(tempCheckpointFile, trainerStateCheckpointFilePath);
    }

    Dictionary Trainer::RestoreFromCheckpoint(const std::wstring& modelFilePath)
    {
        // Restore the model's parameters
        m_combinedTrainingFunction->Restore(modelFilePath);

        Dictionary checkpoint = Dictionary::Load(GetTrainerStateCheckpointFilePath(modelFilePath));

        size_t version = 0;

        if (checkpoint.Contains(versionPropertyName))
            version = checkpoint[versionPropertyName].Value<size_t>();
        
        auto learnerState = checkpoint[learnersPropertyName].Value<std::vector<DictionaryValue>>();
        auto externalState = checkpoint[externalStatePropertyName].Value<Dictionary>();

        m_parameterLearners->RestoreFromCheckpoint(learnerState);

        if (!m_distributed)
        {
            return externalState;
        }

        // this ensures that nobody will start writing to the model/checkpoint files, until
        // everybody is done reading them.
        DistributedCommunicatorPtr communicator = MPICommunicator();
        communicator->Barrier();

        auto mainWorkerId = std::to_wstring(0);
        auto localWorkerId = std::to_wstring(communicator->CurrentWorker().m_globalRank);

        // before version 1, there was no distributed state per se. Instead, the external state
        // contained a dictionary of worker-specific external states.
        if (version == 0)
        {
            auto key = externalState.Contains(localWorkerId) ? localWorkerId : mainWorkerId;
            return externalState[key].Value<Dictionary>();
        }

        Dictionary distributedState = checkpoint[distributedStatePropertyName].Value<Dictionary>();

        if (communicator->CurrentWorker().IsMain() || !distributedState.Contains(localWorkerId))
        {
            return externalState;
        }
        
        // the checkpoint contains internal state for this worker.
        Dictionary localState = distributedState[localWorkerId].Value<Dictionary>();

        auto internalState = localState[internalWorkerStateKey].Value<Dictionary>();
        auto compositeFunction = std::dynamic_pointer_cast<CompositeFunction>(m_combinedTrainingFunction);
        if (compositeFunction == nullptr)
            RuntimeError("Combined training function is not a CompositeFunction.");
            
        // this assumes the compositeFunction (restored form a checkpoint made by the main node) and 
        // the internal worker state both have identical UIDs.
        compositeFunction->SetInternalState(internalState);
        
        return localState[externalWorkerStateKey].Value<Dictionary>();
    }

    double Trainer::PreviousMinibatchLossAverage() const
    {
        // TODO: better return 0; it is then still valid to compute lossAverage * numSamples
        if (m_prevMinibatchNumSamples == 0)
            RuntimeError("There was no preceding call to TrainMinibatch or the minibatch was empty.");

        return m_prevMinibatchAggregateTrainingLossValue->AsScalar<double>() / m_prevMinibatchNumSamples;
    }

    double Trainer::PreviousMinibatchEvaluationAverage() const
    {
        if (!m_evaluationFunction)
            InvalidArgument("Trainer::PreviousMinibatchEvaluationAverage: Cannot get evaluation criterion value when no evaluation function was specified during 'this' trainer's construction");

        if (m_prevMinibatchNumSamples == 0)
            RuntimeError("There was no preceding call to TrainMinibatch or the minibatch was empty.");

        return m_prevMinibatchAggregateEvalCriterionValue->AsScalar<double>() / m_prevMinibatchNumSamples;
    }

    const std::vector<LearnerPtr>& Trainer::ParameterLearners() const
    {
        return m_parameterLearners->ParameterLearners();
    }

    size_t Trainer::TotalNumberOfSamplesSeen() const
    {
        return m_parameterLearners->GetMetricAggregatingLearner()->TotalNumberOfSamplesSeen();
    }

    size_t Trainer::TotalNumberOfUnitsSeen(DataUnit unit) const
    {
        switch (unit)
        {
        case DataUnit::Minibatch:
            return m_parameterLearners->GetMetricAggregatingLearner()->TotalNumberOfMinibatchesSeen();
            break;
        case DataUnit::Sweep:
            return m_parameterLearners->GetMetricAggregatingLearner()->TotalNumberOfSweepsSeen();
            break;
        case DataUnit::Sample:
            return m_parameterLearners->GetMetricAggregatingLearner()->TotalNumberOfSamplesSeen();
        default:
            //should not be here; whenever a new data unit is defined, there should be a new case in this function.
            LogicError("Unsupported data unit: %d", (int)unit);
        }
    }

    TrainerPtr CreateTrainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const std::vector<LearnerPtr>& parameterLearners,
                             const std::vector<ProgressWriterPtr>& progressWriters)
    {
        return MakeSharedObject<Trainer>(model, lossFunction, parameterLearners, progressWriters);
    }

    TrainerPtr CreateTrainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const FunctionPtr& evaluationFunction, const std::vector<LearnerPtr>& parameterLearners,
                             const std::vector<ProgressWriterPtr>& progressWriters)
    {
        return MakeSharedObject<Trainer>(model, lossFunction, evaluationFunction, parameterLearners, progressWriters);
    }
}
