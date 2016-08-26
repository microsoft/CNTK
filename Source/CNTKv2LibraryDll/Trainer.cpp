//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"

namespace CNTK
{
    Trainer::Trainer(const FunctionPtr& model, const Variable& trainingLoss, const std::unordered_set<LearnerPtr>& parameterLearners)
        : m_model(model), m_trainingLossVar(trainingLoss), m_parameterLearners(parameterLearners), m_prevMinibatchNumSamples(1)
    {
        auto modelParameters = model->Parameters();
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

        if (modelParameters != learnerParameters)
            InvalidArgument("Trainer ctor: Union of the parameters covered by the specified parameterLearners should match the specified model's parameters");
    }

    bool Trainer::TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        std::unordered_map<Variable, ValuePtr> outputs = { { m_trainingLossVar, nullptr } };
        auto backPropSate = m_model->Forward(arguments, outputs, computeDevice, { m_trainingLossVar });
        m_prevMinibatchTrainingLossValue = outputs.begin()->second;

        ValuePtr rootGradientValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(m_trainingLossVar.GetDataType(), outputs.at(m_trainingLossVar)->Data()->Shape(), computeDevice), outputs.at(m_trainingLossVar)->Mask());
        if (m_trainingLossVar.GetDataType() == DataType::Float)
            rootGradientValue->Data()->SetValue(1.0f);
        else
            rootGradientValue->Data()->SetValue(1.0);

        auto modelParameters = m_model->Parameters();
        std::unordered_map<Variable, ValuePtr> parameterGradients;
        for (const auto& parameter : modelParameters)
            parameterGradients[parameter] = nullptr;

        m_model->Backward(backPropSate, { { m_trainingLossVar, rootGradientValue } }, parameterGradients);

        auto trainingLossArgument = *(m_trainingLossVar.Owner()->Arguments().begin());

        // Find the argument whose dynamic axes match the criterion operation's dynamic axes (i.e. label dynamic axes)
        // Then we determine the actual number of samples contributing to the training loss from the argument's Value object
        auto argumentValue = std::find_if(arguments.begin(), arguments.end(), [trainingLossArgument](const std::pair<Variable, ValuePtr>& currentPair) {
            return (currentPair.first.DynamicAxes() == trainingLossArgument.DynamicAxes());
        })->second;
        auto argumentData = argumentValue->Data();
        auto argumentDataShape = argumentData->Shape();
        auto mask = argumentValue->Mask();
        m_prevMinibatchNumSamples = argumentDataShape[argumentDataShape.NumAxes() - 1] - ((mask != nullptr) ? mask->MaskedCount() : 0);

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

    double Trainer::PreviousMinibatchAverageTrainingLoss() const
    {
        double trainLossValue = std::numeric_limits<double>::quiet_NaN();
        auto prevMBTrainingLossValue = PreviousMinibatchTrainingLossValue()->Data();

        NDArrayViewPtr cpuTrainLossValue;
        if (prevMBTrainingLossValue->Device() == DeviceDescriptor::CPUDevice())
            cpuTrainLossValue = prevMBTrainingLossValue;
        else
        {
            cpuTrainLossValue = std::make_shared<NDArrayView>(prevMBTrainingLossValue->GetDataType(), prevMBTrainingLossValue->Shape(), CNTK::DeviceDescriptor::CPUDevice());
            cpuTrainLossValue->CopyFrom(*prevMBTrainingLossValue);
        }

        if (prevMBTrainingLossValue->GetDataType() == DataType::Float)
            trainLossValue = *(cpuTrainLossValue->DataBuffer<float>());
        else if (prevMBTrainingLossValue->GetDataType() == DataType::Double)
            trainLossValue = *(cpuTrainLossValue->DataBuffer<double>());
        else
            LogicError("Unsupported DataType of training loss value");

        return (trainLossValue / m_prevMinibatchNumSamples);
    }
}
