//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "Learner.h"
#include "PerformanceProfiler.h"

namespace
{
    const std::wstring learnersPropertyName = L"Learners";
    const std::wstring externalStatePropertyName = L"ExternalState";
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
        : m_model(model),
          m_lossFunction(lossFunction),
          m_evaluationFunction(evaluationFunction),
          m_parameterLearners(std::make_shared<Learners>(parameterLearners)),
          m_prevMinibatchNumSamples(1),
          m_distributed(false),
          m_aggregatedTrainingLossValue(std::make_shared<Accumulator>()),
          m_aggregatedTrainingEvalCriterionValue(),
          m_aggregatedTestEvalCriterionValue(),
          m_progressWriters(progressWriters.begin(), progressWriters.end())
    {
        // By default we set the number of threads to hardware concurrency.
        if (!Internal::MaxNumCPUThreadsSet())
            SetMaxNumCPUThreads(std::thread::hardware_concurrency());

        std::vector<Variable> combinedFunctionArgs;
        if (m_model) // model is optional, since it may not be adding any information on top of lossFunction
            combinedFunctionArgs = m_model->Outputs();
        combinedFunctionArgs.push_back(m_lossFunction);
        if (!m_lossFunction->Output().DynamicAxes().empty())
        {
            m_aggregatedLossFunction = ReduceSum(lossFunction);
            combinedFunctionArgs.push_back(m_aggregatedLossFunction);
            m_trainingSampleCountVar = m_lossFunction;
        }
        else
        {
            m_aggregatedLossFunction = m_lossFunction;
            m_trainingSampleCountVar = m_lossFunction->RootFunction()->Inputs()[0];
            if (model->Output() != m_trainingSampleCountVar)
                combinedFunctionArgs.push_back(m_trainingSampleCountVar);
        }

        if (m_evaluationFunction)
        {
            combinedFunctionArgs.push_back(m_evaluationFunction);

            if (!m_evaluationFunction->Output().DynamicAxes().empty())
            {
                m_aggregatedEvaluationFunction = ReduceSum(m_evaluationFunction);
                combinedFunctionArgs.push_back(m_aggregatedEvaluationFunction);
                m_testSampleCountVar = m_evaluationFunction;
            }
            else
            {
                m_aggregatedEvaluationFunction = m_evaluationFunction;
                m_testSampleCountVar = m_evaluationFunction->RootFunction()->Inputs()[0];
                if ((m_testSampleCountVar != m_trainingSampleCountVar) && (model->Output() != m_testSampleCountVar))
                    combinedFunctionArgs.push_back(m_testSampleCountVar);
            }
            
            m_aggregatedTrainingEvalCriterionValue = std::make_shared<Accumulator>();
            m_aggregatedTestEvalCriterionValue = std::make_shared<Accumulator>();
        }

        m_combinedTrainingFunction = Combine(combinedFunctionArgs);

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
    }

    static size_t GetSampleCount(const Variable& var, const ValuePtr& value)
    {
        auto valueDataShape = value->Shape();
        size_t numMaskedSamples = value->MaskedCount();
        size_t numSamplesInDataArrayView = valueDataShape.SubShape(var.Shape().Rank()).TotalSize();
        if (numMaskedSamples > numSamplesInDataArrayView)
            LogicError("Number (%d) of masked values cannot exceed the number (%d) of samples that the Value object's Data NDArrayView can hold.",
                       (int)numMaskedSamples, (int)numSamplesInDataArrayView);

        return (numSamplesInDataArrayView - numMaskedSamples);
    }

    static std::unordered_map<Variable, ValuePtr> GetInputs(const std::unordered_map<Variable, MinibatchData>& arguments)
    {
        std::unordered_map<Variable, ValuePtr> inputs(arguments.size());
        for (const auto& kv : arguments)
        {
            inputs[kv.first] = kv.second.data;
        }
        return inputs;
    }

    static bool IsAtSweepEnd(const std::unordered_map<Variable, MinibatchData>& arguments)
    {
        return std::any_of(arguments.begin(), arguments.end(), [](const std::pair<const Variable, MinibatchData>& kv)
        {
            return kv.second.sweepEnd;
        });
    }

    double Trainer::TestMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        return TestMinibatch(GetInputs(arguments), computeDevice);
    }

    double Trainer::TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        size_t sampleCount = 0;
        return TestMinibatch(arguments, computeDevice, sampleCount);
    }

    double Trainer::TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice, size_t& sampleCount)
    {
        if (!m_aggregatedEvaluationFunction)
            InvalidArgument("Trainer::TestMinibatch: Cannot test when no evaluation function was specified during 'this' trainer's construction.");

        // TODO: Should we refactor this code that is somewhat similar to the prologue of the TrainMinibatch function
        std::unordered_map<Variable, ValuePtr> outputs = { { m_aggregatedEvaluationFunction, nullptr }, { m_testSampleCountVar, nullptr } };

        m_combinedTrainingFunction->Forward(arguments, outputs, computeDevice);
        const ValuePtr& aggregateEvalCriterionValue = outputs[m_aggregatedEvaluationFunction];
        sampleCount = GetSampleCount(m_testSampleCountVar, outputs[m_testSampleCountVar]);

        UpdateTestProgress(sampleCount, aggregateEvalCriterionValue, computeDevice);

        // TODO: it is not optimal to return average evaluation after each minibatch, since it potentially requires a
        // roundtrip to GPU. A better approach would be to have a separate method to return the average evaluation on
        // demand, as done for training. However, removing the below return is an API breaking change.
        return aggregateEvalCriterionValue->AsScalar<double>() / sampleCount;
    }

    bool Trainer::TrainMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        std::unordered_map<Variable, ValuePtr> outputsToFetch = {};
        return TrainMinibatch(arguments, outputsToFetch, computeDevice);
    }

    bool Trainer::TrainMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        auto profMinibatch = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainMinibatch);

        bool result = (!m_distributed) ?
            TrainLocalMinibatch(GetInputs(arguments), outputsToFetch, IsAtSweepEnd(arguments), computeDevice) :
            TrainDistributedMinibatch(GetInputs(arguments), outputsToFetch, IsAtSweepEnd(arguments), computeDevice);

        // TODO: exclude updating progress writers from profiling?
        UpdateTrainingProgress(m_prevMinibatchNumSamples, m_prevMinibatchAggregateTrainingLossValue,
                               m_prevMinibatchAggregateEvalCriterionValue, computeDevice);
        return result;
    }

    bool Trainer::TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        std::unordered_map<Variable, ValuePtr> outputsToFetch = {};
        return TrainMinibatch(arguments, outputsToFetch, computeDevice);
    }

    bool Trainer::TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        auto profMinibatch = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainMinibatch);

        bool result = (!m_distributed) ?
            TrainLocalMinibatch(arguments, outputsToFetch, false, computeDevice) :
            TrainDistributedMinibatch(arguments, outputsToFetch, false, computeDevice);

        // TODO: exclude updating progress writers from profiling?
        UpdateTrainingProgress(m_prevMinibatchNumSamples, m_prevMinibatchAggregateTrainingLossValue,
                               m_prevMinibatchAggregateEvalCriterionValue, computeDevice);
        return result;
    }

    bool Trainer::TrainLocalMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, bool sweepEnd, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        bool emptyMinibatch = arguments.empty() || (arguments.begin()->second == nullptr);
        if (emptyMinibatch) // Nothing to train with.
            return false;

        std::unordered_map<Variable, ValuePtr> parameterGradients;
        ExecuteForwardBackward(arguments, outputsToFetch, computeDevice, parameterGradients);

        auto profWeights = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainWeights);

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
        }
        else
        {
            // Get gradients after forward/backward pass.
            std::unordered_map<Variable, ValuePtr> parameterGradients;
            ExecuteForwardBackward(arguments, outputsToFetch, computeDevice, parameterGradients);
            for (const auto& parameter : m_learnerParameters)
                gradients[parameter] = parameterGradients[parameter]->Data();
            trainingLoss = m_prevMinibatchAggregateTrainingLossValue->Data();
            evalCriterion = m_prevMinibatchAggregateEvalCriterionValue->Data();
        }

        MinibatchInfo info{ arguments.empty(), sweepEnd, m_prevMinibatchNumSamples, trainingLoss, evalCriterion };
        bool updated = m_parameterLearners->Update(gradients, info);
        m_prevMinibatchNumSamples = info.numberOfSamples;

        // Update internal state.
        if (emptyMinibatch)
        {
            // Have to reassign loss and criterion.
            m_prevMinibatchAggregateEvalCriterionValue = std::make_shared<Value>(info.evalCriterionValue);
            m_prevMinibatchAggregateTrainingLossValue = std::make_shared<Value>(info.trainingLossValue);
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

    void Trainer::UpdateTestProgress(size_t numSamples, const ValuePtr& evalCriterion, const DeviceDescriptor& computeDevice)
    {
        if (numSamples == 0)
        {
            return;
        }

        if (m_aggregatedTestEvalCriterionValue)
        {
            m_aggregatedTestEvalCriterionValue->Update(evalCriterion, computeDevice);
        }

        for (auto& progressWriter : m_progressWriters)
        {
            progressWriter->UpdateTest(numSamples, m_aggregatedTestEvalCriterionValue);
        }
    }

    void Trainer::SummarizeTestProgress()
    {
        for (auto& progressWriter : m_progressWriters)
        {
            progressWriter->WriteTestSummary(m_aggregatedTestEvalCriterionValue);
        }

        if (m_aggregatedTestEvalCriterionValue)
        {
            m_aggregatedTestEvalCriterionValue->Reset();
        }
    }

    void Trainer::AddProgressWriters(const std::vector<ProgressWriterPtr>& progressWriters)
    {
        m_progressWriters.insert(progressWriters.begin(), progressWriters.end());
    }

    void Trainer::ExecuteForwardBackward(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice, std::unordered_map<Variable, ValuePtr>& parameterGradients)
    {
        auto profForwardBackward = Microsoft::MSR::CNTK::ScopeProfile(Microsoft::MSR::CNTK::profilerEvtMainFB);
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

        if (m_aggregatedLossFunction->Output().GetDataType() == DataType::Float)
            m_rootGradientValue->Data()->SetValue(1.0f);
        else
            m_rootGradientValue->Data()->SetValue(1.0);

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

        // Collect distrbuted external state.
        DistributedCommunicatorPtr communicator = MPICommunicator();
        communicator->Barrier();

        std::vector<DictionaryPtr> remoteState;
        communicator->Gather(externalState, remoteState, communicator->Workers());

        Dictionary aggregatedState;
        for (const auto& w : communicator->Workers())
        {
            aggregatedState[std::to_wstring(w.m_globalRank)] = *remoteState[w.m_globalRank];
        }

        if (communicator->CurrentWorker().IsMain())
            Save(modelFilePath, learnersState, aggregatedState);

        // all workers need to sync up after saving model to avoid read-after-write hazard
        // i.e. one worker is in the middle of write while another tries to read
        communicator->Barrier();
    }

    void Trainer::Save(const std::wstring& modelFilePath, const std::vector<DictionaryValue>& learnerState, const Dictionary& externalState)
    {
        std::wstring tempModelFile = modelFilePath + L".tmp";
        Dictionary state;
        state[learnersPropertyName] = learnerState;
        state[externalStatePropertyName] = externalState;

        m_combinedTrainingFunction->SaveModel(tempModelFile);
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
        m_combinedTrainingFunction->RestoreModel(modelFilePath);

        Dictionary checkpoint = Dictionary::Load(GetTrainerStateCheckpointFilePath(modelFilePath));

        auto learnerState = checkpoint[learnersPropertyName].Value<std::vector<DictionaryValue>>();
        auto externalState = checkpoint[externalStatePropertyName].Value<Dictionary>();

        if (!m_distributed)
        {
            m_parameterLearners->RestoreFromCheckpoint(learnerState);
            return externalState;
        }

        m_parameterLearners->RestoreFromCheckpoint(learnerState);
        DistributedCommunicatorPtr communicator = MPICommunicator();
        communicator->Barrier();

        auto key = std::to_wstring(communicator->CurrentWorker().m_globalRank);

        if (externalState.Contains(key))
            return externalState[key].Value<Dictionary>();
        else
            return externalState[std::to_wstring(0)].Value<Dictionary>();
    }

    double Trainer::PreviousMinibatchLossAverage() const
    {
        // TODO: better return 0; it is then still valid to compute lossAverage * numSamples
        if (m_prevMinibatchNumSamples == 0)
            RuntimeError("There was no preceeding call to TrainMinibatch or the minibatch was empty.");

        return m_prevMinibatchAggregateTrainingLossValue->AsScalar<double>() / m_prevMinibatchNumSamples;
    }

    double Trainer::PreviousMinibatchEvaluationAverage() const
    {
        if (!m_evaluationFunction)
            InvalidArgument("Trainer::PreviousMinibatchEvaluationAverage: Cannot get evaluation criterion value when no evaluation function was specified during 'this' trainer's construction");

        if (m_prevMinibatchNumSamples == 0)
            RuntimeError("There was no preceeding call to TrainMinibatch or the minibatch was empty.");

        return m_prevMinibatchAggregateEvalCriterionValue->AsScalar<double>() / m_prevMinibatchNumSamples;
    }

    const std::vector<LearnerPtr>& Trainer::ParameterLearners() const
    {
        return m_parameterLearners->ParameterLearners();
    }

    size_t Trainer::TotalNumberOfSamplesSeen() const
    {
        return m_parameterLearners->ParameterLearners().front()->TotalNumberOfSamplesSeen();
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
