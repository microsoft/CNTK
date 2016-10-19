//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "Function.h"

namespace CNTK
{
    Trainer::Trainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const FunctionPtr& evaluationFunction, const std::vector<LearnerPtr>& parameterLearners, const DistributedTrainerPtr& distributedTrainer)
        : m_model(model), m_lossFunction(lossFunction), m_evaluationFunction(evaluationFunction), m_parameterLearners(parameterLearners), m_prevMinibatchNumSamples(1), m_distributedTrainer(distributedTrainer)
    {
        std::vector<Variable> combinedFunctionArgs = { m_model, m_lossFunction };
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
        }

        m_combinedTrainingFunction = Combine(combinedFunctionArgs);

        auto modelParameters = m_combinedTrainingFunction->Parameters();
        std::unordered_set<Parameter> learnerParameters;
        for (const auto& learner : parameterLearners)
        {
            const auto& currentLearnerParameters = learner->Parameters();
            for (const auto& parameter : currentLearnerParameters)
            {
                auto insertRetVal = learnerParameters.insert(parameter);
                if (!insertRetVal.second)
                    InvalidArgument("Trainer ctor: Parameter named %S is covered by 2 different learners", parameter.Name().c_str());
            }
        }

        std::unordered_set<Parameter> modelParametersSet(modelParameters.begin(), modelParameters.end());
        if (modelParametersSet != learnerParameters)
        {
            InvalidArgument("Trainer ctor: Union of the parameters covered by the specified parameterLearners should match the specified model's parameters");
        }
            
    }

    Trainer::Trainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const FunctionPtr& evaluationFunction, const std::vector<LearnerPtr>& parameterLearners)
        : Trainer(model, lossFunction, evaluationFunction, parameterLearners, nullptr)
    {}

    Trainer::Trainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const std::vector<LearnerPtr>& parameterLearners)
        : Trainer(model, lossFunction, nullptr, parameterLearners)
    {}

    static double GetScalarValue(const ValuePtr& value)
    {
        if (value->Mask())
            LogicError("Scalar Value object cannot have an associated mask");

        auto scalarData = value->Data();
        if (scalarData->Shape().TotalSize() != 1)
            LogicError("Scalar Value object's has a size > 1");

        double scalar = std::numeric_limits<double>::quiet_NaN();
        NDArrayViewPtr cpuData;
        if (scalarData->Device() == DeviceDescriptor::CPUDevice())
            cpuData = scalarData;
        else
        {
            cpuData = std::make_shared<NDArrayView>(scalarData->GetDataType(), scalarData->Shape(), CNTK::DeviceDescriptor::CPUDevice());
            cpuData->CopyFrom(*scalarData);
        }

        if (scalarData->GetDataType() == DataType::Float)
            scalar = *(cpuData->DataBuffer<float>());
        else if (scalarData->GetDataType() == DataType::Double)
            scalar = *(cpuData->DataBuffer<double>());
        else
            LogicError("Unsupported DataType of training loss value");

        return scalar;
    }

    static size_t GetSampleCount(const Variable& var, const ValuePtr& value)
    {
        auto valueDataShape = value->Shape();
        size_t numMaskedSamples = value->MaskedCount();
        size_t numSamplesInDataArrayView = valueDataShape.SubShape(var.Shape().Rank()).TotalSize();
        if (numMaskedSamples > numSamplesInDataArrayView)
            LogicError("Number of masked values cannot exceed the number of samples that the Value object's Data NDArrayView can hold");

        return (numSamplesInDataArrayView - numMaskedSamples);
    }

    double Trainer::TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        if (!m_aggregatedEvaluationFunction)
            InvalidArgument("Trainer::TestMinibatch: Cannot test when no evaluation function was specified during 'this' trainer's construction");

        // TODO: Should we refactor this code that is somewhat similar to the prologue of the TrainMinibatch function
        std::unordered_map<Variable, ValuePtr> outputs = { { m_aggregatedEvaluationFunction, nullptr }, { m_testSampleCountVar, nullptr } };
        m_combinedTrainingFunction->Forward(arguments, outputs, computeDevice);

        auto sampleCount = GetSampleCount(m_testSampleCountVar, outputs[m_testSampleCountVar]);
        return (GetScalarValue(outputs[m_aggregatedEvaluationFunction]) / sampleCount);
    }

    bool Trainer::TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        std::unordered_map<Variable, ValuePtr> outputsToFetch = {};
        return TrainMinibatch(arguments, outputsToFetch, computeDevice);
    }

    bool Trainer::TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        std::unordered_map<Variable, ValuePtr> outputs = { { m_aggregatedLossFunction, nullptr }, { m_trainingSampleCountVar, nullptr } };
        if (m_aggregatedEvaluationFunction)
            outputs.insert({ m_aggregatedEvaluationFunction, nullptr });

        outputs.insert(outputsToFetch.begin(), outputsToFetch.end());

        if (m_distributedTrainer)
            m_distributedTrainer->PreMinibatchCallback(*this);

        auto backPropSate = m_combinedTrainingFunction->Forward(arguments, outputs, computeDevice, { m_aggregatedLossFunction });
        m_prevMinibatchAggregateTrainingLossValue = outputs[m_aggregatedLossFunction];
        if (m_aggregatedEvaluationFunction)
            m_prevMinibatchAggregateEvalCriterionValue = outputs[m_aggregatedEvaluationFunction];

        for (auto outputToFetch : outputsToFetch)
        {
            if (outputToFetch.second == nullptr)
                outputsToFetch[outputToFetch.first] = outputs[outputToFetch.first];
        }

        ValuePtr rootGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(m_aggregatedLossFunction->Output().GetDataType(), m_prevMinibatchAggregateTrainingLossValue->Shape(), computeDevice), outputs.at(m_aggregatedLossFunction)->Mask());
        if (m_aggregatedLossFunction->Output().GetDataType() == DataType::Float)
            rootGradientValue->Data()->SetValue(1.0f);
        else
            rootGradientValue->Data()->SetValue(1.0);

        auto modelParameters = m_combinedTrainingFunction->Parameters();
        std::unordered_map<Variable, ValuePtr> parameterGradients;
        for (const auto& parameter : modelParameters)
        {
            parameterGradients[parameter] = nullptr;
        }

        m_combinedTrainingFunction->Backward(backPropSate, { { m_aggregatedLossFunction, rootGradientValue } }, parameterGradients);

        m_prevMinibatchNumSamples = GetSampleCount(m_trainingSampleCountVar, outputs[m_trainingSampleCountVar]);

        if (m_distributedTrainer)
        {
            // Aggregation should happen in the same order, the order of parmaters is guaranteed to be the same.
            std::vector<std::pair<Variable, NDArrayViewPtr>> gradients;
            gradients.reserve(modelParameters.size());
            for (const auto& parameter : modelParameters)
                gradients.push_back(std::make_pair(parameter, parameterGradients[parameter]->Data()));

            MinibatchInfo info
            {
                m_prevMinibatchNumSamples,
                m_prevMinibatchAggregateTrainingLossValue->Data(),
                m_prevMinibatchAggregateEvalCriterionValue->Data()
            };

            m_distributedTrainer->PreParameterUpdateCallback(*this, gradients, info);
            m_prevMinibatchNumSamples = info.numberOfSamples;
        }

        bool anyUpdatesPerformed = false;
        for (auto learner : m_parameterLearners)
        {
            std::unordered_map<Parameter, NDArrayViewPtr> learnerParameterGradients;
            const auto& learnerParameters = learner->Parameters();
            for (const auto& parameter : learnerParameters)
            {
                learnerParameterGradients[parameter] = parameterGradients[parameter]->Data();

                if (parameterGradients[parameter]->Mask())
                    LogicError("The gradient value for a Parameter cannot have an associated mask!");
            }

            anyUpdatesPerformed |= learner->Update(learnerParameterGradients, m_prevMinibatchNumSamples);
        }

        return anyUpdatesPerformed;
    }

    static std::wstring GetTrainerStateCheckpointFilePath(const std::wstring& modelFilePath)
    {
        const wchar_t* checkpointExt = L".ckp";
        return modelFilePath + checkpointExt;
    }

    std::shared_ptr<std::fstream> GetFstream(const std::wstring& filePath, bool readOnly)
    {
        std::ios_base::openmode mode = std::ios_base::binary | (readOnly ? std::ios_base::in : std::ios_base::out);
#ifdef _MSC_VER
        return std::make_shared<std::fstream>(filePath, mode);
#else
        return std::make_shared<std::fstream>(wtocharpath(filePath.c_str()).c_str(), mode);
#endif
    }

    void Trainer::SaveCheckpoint(const std::wstring& modelFilePath, bool usinglegacyModelFormat)
    {
        if (usinglegacyModelFormat)
        {
            SaveAsLegacyModel(m_combinedTrainingFunction, modelFilePath);
        }
        else
        {
             Dictionary model = m_combinedTrainingFunction->Serialize();
             auto stream = GetFstream(modelFilePath, false);
            *stream << model;
             stream->flush();
        }

        vector<DictionaryValue> learnerStates;

        for (const auto& learner : m_parameterLearners)
        {
            // TODO: add DictionaryValue(T&&)
            learnerStates.push_back(DictionaryValue(learner->Serialize()));
        }
        
        std::wstring trainerStateCheckpointFilePath = GetTrainerStateCheckpointFilePath(modelFilePath);
        auto ckpStream = GetFstream(trainerStateCheckpointFilePath, false);
        // TODO: this will create an extra copy of all leaner states, 
        // add DictionaryValue ctor that takes an rvalue!
        *ckpStream << DictionaryValue(learnerStates);
        ckpStream->flush();
    }

    void Trainer::RestoreFromCheckpoint(const std::wstring& modelFilePath, bool usinglegacyModelFormat)
    {
        // Restore the model's parameters
        if (usinglegacyModelFormat)
        {
            m_combinedTrainingFunction->RestoreFromLegacyModel(modelFilePath);
        }
        else
        { 
            auto stream = GetFstream(modelFilePath, true);
            Dictionary model;
            *stream >> model;
            m_combinedTrainingFunction->RestoreFromCheckpoint(model);
        }

        std::wstring trainerStateCheckpointFilePath = GetTrainerStateCheckpointFilePath(modelFilePath);
        auto ckpStream = GetFstream(trainerStateCheckpointFilePath, true);
        DictionaryValue checkpoint;
        *ckpStream >> checkpoint;

        const vector<DictionaryValue>& learnerStates = checkpoint.Value<vector<DictionaryValue>>();

        if (learnerStates.size() != m_parameterLearners.size())
        {
            LogicError("Trainer::RestoreFromCheckpoint: "
                       "Number of learners in the checkpoint (%zu) does not match the expected number (%zu)",
                       learnerStates.size(), m_parameterLearners.size());
        }

        for (int i = 0; i < m_parameterLearners.size(); ++i)
        {
            m_parameterLearners[i]->RestoreFromCheckpoint(learnerStates[i].Value<Dictionary>());
        }
    }

    double Trainer::PreviousMinibatchLossAverage() const
    {
        return (GetScalarValue(m_prevMinibatchAggregateTrainingLossValue) / m_prevMinibatchNumSamples);
    }

    double Trainer::PreviousMinibatchEvaluationAverage() const
    {
        if (!m_evaluationFunction)
            InvalidArgument("Trainer::PreviousMinibatchEvaluationAverage: Cannot get evaluation criterion value when no evaluation function was specified during 'this' trainer's construction");

        return (GetScalarValue(m_prevMinibatchAggregateEvalCriterionValue) / m_prevMinibatchNumSamples);
    }
}
