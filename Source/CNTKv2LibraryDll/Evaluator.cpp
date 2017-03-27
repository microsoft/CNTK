//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "PerformanceProfiler.h"

namespace CNTK
{
    EvaluatorPtr CreateEvaluator(const FunctionPtr& evaluationFunction, const std::vector<ProgressWriterPtr>& progressWriters)
    {
        return MakeSharedObject<Evaluator>(evaluationFunction, progressWriters, true);
    }

    Evaluator::Evaluator(
        const FunctionPtr& evaluationFunction,
        const std::vector<ProgressWriterPtr>& progressWriters,
        bool initializeCombined)
        :  m_evaluationFunction(evaluationFunction),
           m_aggregatedTestEvalCriterionValue(std::make_shared<Accumulator>()),
           m_progressWriters(progressWriters.begin(), progressWriters.end())
    {
        // By default we set the number of threads to hardware concurrency.
        if (!Internal::MaxNumCPUThreadsSet())
            SetMaxNumCPUThreads(std::thread::hardware_concurrency());

        // Nullptr evaluation is only allowed by the derived classes.
        if (!m_evaluationFunction)
        {
            if(initializeCombined)
                InvalidArgument("Eval function is not allowed to be null.");
            return;
        }

        if (!m_evaluationFunction->Output().DynamicAxes().empty())
        {
            m_aggregatedEvaluationFunction = ReduceSum(m_evaluationFunction);
            m_testSampleCountVar = m_evaluationFunction;
        }
        else
        {
            m_aggregatedEvaluationFunction = m_evaluationFunction;
            m_testSampleCountVar = m_evaluationFunction->RootFunction()->Inputs()[0];
        }

        if(initializeCombined)
            m_combinedEvalFunction = Combine(GetCombinedEvalFunctionArgs());
    }

    std::vector<Variable> Evaluator::GetCombinedEvalFunctionArgs() const
    {
        if (!m_evaluationFunction)
            return std::vector<Variable>();

        std::vector<Variable> result{ m_evaluationFunction };
        if (m_evaluationFunction != m_aggregatedEvaluationFunction)
            result.push_back(m_aggregatedEvaluationFunction);

        if (m_evaluationFunction != m_testSampleCountVar && m_aggregatedEvaluationFunction != m_testSampleCountVar)
            result.push_back(m_testSampleCountVar);

        return result;
    }

    size_t Evaluator::GetSampleCount(const Variable& var, const ValuePtr& value)
    {
        auto valueDataShape = value->Shape();
        size_t numMaskedSamples = value->MaskedCount();
        size_t numSamplesInDataArrayView = valueDataShape.SubShape(var.Shape().Rank()).TotalSize();
        if (numMaskedSamples > numSamplesInDataArrayView)
            LogicError("Number (%d) of masked values cannot exceed the number (%d) of samples that the Value object's Data NDArrayView can hold.",
            (int)numMaskedSamples, (int)numSamplesInDataArrayView);

        return (numSamplesInDataArrayView - numMaskedSamples);
    }

    std::unordered_map<Variable, ValuePtr> Evaluator::GetInputs(const std::unordered_map<Variable, MinibatchData>& arguments)
    {
        std::unordered_map<Variable, ValuePtr> inputs(arguments.size());
        for (const auto& kv : arguments)
        {
            inputs[kv.first] = kv.second.data;
        }
        return inputs;
    }

    double Evaluator::TestMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        std::unordered_map<Variable, ValuePtr> outputsToFetch = {};
        return TestMinibatch(GetInputs(arguments), outputsToFetch, computeDevice);
    }

    double Evaluator::TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice /*= DeviceDescriptor::UseDefaultDevice()*/)
    {
        std::unordered_map<Variable, ValuePtr> outputsToFetch = {};
        return TestMinibatch(arguments, outputsToFetch, computeDevice);
    }

    double Evaluator::TestMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice)
    {
        return TestMinibatch(GetInputs(arguments), outputsToFetch, computeDevice);
    }

    double Evaluator::TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice)
    {
        auto evalMinibatchValue = TestMinibatch(arguments, outputsToFetch, computeDevice, false);
        return evalMinibatchValue.first->AsScalar<double>() / evalMinibatchValue.second;
    }

    std::pair<ValuePtr, size_t> Evaluator::TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice, bool distributed)
    {
        std::unordered_map<Variable, ValuePtr> outputsToFetch = {};
        return TestMinibatch(arguments, outputsToFetch, computeDevice, distributed);
    }

    std::pair<ValuePtr, size_t> Evaluator::TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice, bool distributed)
    {
        if (distributed)
            RuntimeError("Currently distributed testing is not supported.");
        return TestLocalMinibatch(arguments, outputsToFetch, computeDevice);
    }

    std::pair<ValuePtr, size_t> Evaluator::TestLocalMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice)
    {
        if (!m_aggregatedEvaluationFunction)
            InvalidArgument("Evaluator::TestMinibatch: Cannot test when no evaluation function was specified during construction.");

        std::unordered_map<Variable, ValuePtr> outputs = { { m_aggregatedEvaluationFunction, nullptr }, { m_testSampleCountVar, nullptr } };
        outputs.insert(outputsToFetch.begin(), outputsToFetch.end());

        m_combinedEvalFunction->Forward(arguments, outputs, computeDevice);

        const ValuePtr& aggregateEvalCriterionValue = outputs[m_aggregatedEvaluationFunction];
        auto sampleCount = GetSampleCount(m_testSampleCountVar, outputs[m_testSampleCountVar]);
        UpdateTestProgress(sampleCount, aggregateEvalCriterionValue, computeDevice);

        // Copy back output values for requested variables only.
        for (auto& o : outputsToFetch)
            o.second = outputs[o.first];

        return make_pair(aggregateEvalCriterionValue, sampleCount);
    }

    void Evaluator::UpdateTestProgress(size_t numSamples, const ValuePtr& evalCriterion, const DeviceDescriptor& computeDevice)
    {
        if (numSamples == 0)
            return;

        if (m_aggregatedTestEvalCriterionValue)
            m_aggregatedTestEvalCriterionValue->Update(evalCriterion, computeDevice);

        for (auto& progressWriter : m_progressWriters)
            progressWriter->UpdateTest(numSamples, m_aggregatedTestEvalCriterionValue);
    }

    void Evaluator::SummarizeTestProgress()
    {
        for (auto& progressWriter : m_progressWriters)
            progressWriter->WriteTestSummary(m_aggregatedTestEvalCriterionValue);

        if (m_aggregatedTestEvalCriterionValue)
            m_aggregatedTestEvalCriterionValue->Reset();
    }
}
