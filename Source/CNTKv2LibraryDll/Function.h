//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include <iterator>
#include "ComputationNetwork.h"
#include "Utils.h"

namespace CNTK
{
    enum class PrimitiveOpType
    {
        Plus,
        Times,
        Sigmoid,
        Tanh,
        Combine,
        CrossEntropyWithSoftmax,
        PredictionError,
        Exp,
        PastValue,
        FutureValue,
        ElementTimes,
        ReduceSum
    };

    inline const char* PrimitiveOpTypeName(PrimitiveOpType opType)
    {
        if (opType == PrimitiveOpType::Plus)
            return "Plus";
        else if (opType == PrimitiveOpType::Times)
            return "Times";
        else if (opType == PrimitiveOpType::Sigmoid)
            return "Sigmoid";
        else if (opType == PrimitiveOpType::Tanh)
            return "Tanh";
        else if (opType == PrimitiveOpType::Combine)
            return "Combine";
        else if (opType == PrimitiveOpType::CrossEntropyWithSoftmax)
            return "CrossEntropyWithSoftmax";
        else if (opType == PrimitiveOpType::PredictionError)
            return "PredictionError";
        else if (opType == PrimitiveOpType::Exp)
            return "Exp";
        else if (opType == PrimitiveOpType::PastValue)
            return "PastValue";
        else if (opType == PrimitiveOpType::FutureValue)
            return "FutureValue";
        else if (opType == PrimitiveOpType::ElementTimes)
            return "ElementTimes";
        else if (opType == PrimitiveOpType::ReduceSum)
            return "ReduceSum";
        else
            LogicError("Unknown PrimitiveOpType");
    }

    class PrimitiveFunction final : public Function
    {
    public:
        PrimitiveFunction(PrimitiveOpType op, const std::vector<Variable>& inputs, Dictionary&& functionConfig, const std::wstring& functionName = L"")
            : Function(inputs, GetOutputVariables(op, inputs, this), nullptr, functionName), m_op(op), m_functionConfig(std::move(functionConfig))
        {
        }

        virtual BackPropStatePtr Forward(const _Internal::_SimpleMap<Variable, const ValuePtr>& /*arguments*/,
                                         _Internal::_SimpleMap<Variable, ValuePtr>& /*outputs*/,
                                         const _Internal::_SimpleSet<Variable>& /*outputsToRetainBackwardStateFor*/,
                                         const DeviceDescriptor& /*computeDevice*/) override
        {
            NOT_IMPLEMENTED;
        }

        virtual void Backward(const BackPropStatePtr& /*state*/,
                              const _Internal::_SimpleMap<Variable, const ValuePtr>& /*rootGradientValues*/,
                              _Internal::_SimpleMap<Variable, ValuePtr>& /*backPropagatedGradientValuesForInputs*/) override
        {
            NOT_IMPLEMENTED;
        }

        PrimitiveOpType OpType() const
        {
            return m_op;
        }

        const Dictionary& FunctionConfig() const
        {
            return m_functionConfig;
        }

    private:
        static NDShape UnaryElementwiseOpOutputShape(const NDShape& operandShape)
        {
            return operandShape;
        }

        static NDShape BinaryElementwiseOpOutputShape(PrimitiveOpType op, const NDShape& leftOperandShape, const NDShape& rightOperandShape, bool broadcastAllowed = true)
        {
            auto& shapeWithSmallerNumAxes = (leftOperandShape.NumAxes() > rightOperandShape.NumAxes()) ? rightOperandShape : leftOperandShape;
            auto& shapeWithLargerNumAxes = (leftOperandShape.NumAxes() > rightOperandShape.NumAxes()) ? leftOperandShape : rightOperandShape;
            size_t numOutputAxes = shapeWithLargerNumAxes.NumAxes();
            std::vector<size_t> outputDims(numOutputAxes);
            for (size_t i = 0; i < shapeWithSmallerNumAxes.NumAxes(); ++i)
            {
                if ((leftOperandShape[i] == NDShape::InferredDimension) && (rightOperandShape[i] == NDShape::InferredDimension))
                    outputDims[i] = NDShape::InferredDimension;
                else if ((leftOperandShape[i] == NDShape::InferredDimension) && (rightOperandShape[i] != NDShape::InferredDimension))
                    outputDims[i] = rightOperandShape[i];
                else if ((leftOperandShape[i] != NDShape::InferredDimension) && (rightOperandShape[i] == NDShape::InferredDimension))
                    outputDims[i] = leftOperandShape[i];
                else
                {
                    if (leftOperandShape[i] != rightOperandShape[i])
                        RuntimeError("Left operand's shape %s is not compatible with right operand's shape %s for the binary elementwise operation %s", AsString(leftOperandShape).c_str(), AsString(rightOperandShape).c_str(), PrimitiveOpTypeName(op));

                    outputDims[i] = leftOperandShape[i];
                }
            }

            // Broadcast in remaining axes
            for (size_t i = shapeWithSmallerNumAxes.NumAxes(); i < numOutputAxes; ++i)
                outputDims[i] = shapeWithLargerNumAxes[i];

            return NDShape(std::move(outputDims));
        }

        static NDShape TimesOpOutputShape(const NDShape& leftOperandShape, const NDShape& rightOperandShape, bool broadcastAllowed = true)
        {
            if (rightOperandShape.NumAxes() > 2)
                RuntimeError("The right operand of a times operation can have at most 2 axes");

            size_t numOutputAxes = rightOperandShape.NumAxes();

            if (leftOperandShape.NumAxes() != 2)
                RuntimeError("The left operand of a times operation must have 2 axes");

            std::vector<size_t> outputDims(numOutputAxes);
            outputDims[0] = leftOperandShape[0];
            if (numOutputAxes > 1)
                outputDims[1] = rightOperandShape[1];

            if (leftOperandShape[1] != rightOperandShape[0])
                RuntimeError("Left operand's shape %s is not compatible with right operand's shape %s for the times operation", AsString(leftOperandShape).c_str(), AsString(rightOperandShape).c_str());

            return NDShape(std::move(outputDims));
        }

        static NDShape ReductionOpOutputShape(PrimitiveOpType op, const NDShape& operandShape, const std::vector<size_t>& reductionAxes)
        {
            if (reductionAxes.size() > operandShape.NumAxes())
                RuntimeError("The number of reduction axes %d exceeds the number of axes in the operand shape %s of the reduction operation %s", (int)reductionAxes.size(), AsString(operandShape).c_str(), PrimitiveOpTypeName(op));

            size_t numOutputAxes = operandShape.NumAxes() - reductionAxes.size();
            std::vector<size_t> outputDims(numOutputAxes);
            for (size_t i = 0, j = 0; i < operandShape.NumAxes(); ++i)
            {
                // Skip axes being reduced over
                if (std::find(reductionAxes.begin(), reductionAxes.end(), i) != reductionAxes.end())
                    continue;

                outputDims[j++] = operandShape[i];
            }

            return NDShape(std::move(outputDims));
        }

        static std::vector<Variable> GetOutputVariables(PrimitiveOpType op, const std::vector<Variable>& inputs, Function* owner)
        {
            std::vector<Variable> outputs;

            // TODO: We are just using the input[0]'s DataType as output node's DataType. This is not always correct
            DataType outputDataType = inputs[0].GetDataType();

            // We currently require that the inputs' dynamic axes if any match
            std::vector<Axis> outputDynamicAxes = inputs[0].DynamicAxes();
            for (size_t i = 1; i < inputs.size(); ++i)
            {
                auto currentInputDynamicAxes = inputs[i].DynamicAxes();
                if (outputDynamicAxes.empty())
                    outputDynamicAxes = currentInputDynamicAxes;
                else
                {
                    if (!currentInputDynamicAxes.empty() && (currentInputDynamicAxes != outputDynamicAxes))
                        LogicError("Currently if an operand of a binary elementwise operation has any dynamic axes, those must match the dynamic axes of the other operand");
                }
            }

            switch (op)
            {
            case PrimitiveOpType::Sigmoid:
            case PrimitiveOpType::Tanh:
            case PrimitiveOpType::Exp:
                assert(inputs.size() == 1);
                outputs.push_back(Variable(UnaryElementwiseOpOutputShape(inputs[0].Shape()), outputDataType, owner, outputDynamicAxes));
                break;
            case PrimitiveOpType::PastValue:
            case PrimitiveOpType::FutureValue:
                assert(inputs.size() == 2);
                outputs.push_back(Variable(UnaryElementwiseOpOutputShape(inputs[1].Shape()), outputDataType, owner, outputDynamicAxes));
                break;
            case PrimitiveOpType::Plus:
            case PrimitiveOpType::ElementTimes:
                assert(inputs.size() == 2);
                outputs.push_back(Variable(BinaryElementwiseOpOutputShape(op, inputs[0].Shape(), inputs[1].Shape()), outputDataType, owner, outputDynamicAxes));
                break;
            case PrimitiveOpType::Times:
                assert(inputs.size() == 2);
                outputs.push_back(Variable(TimesOpOutputShape(inputs[0].Shape(), inputs[1].Shape()), outputDataType, owner, outputDynamicAxes));
                break;
            case PrimitiveOpType::CrossEntropyWithSoftmax:
            case PrimitiveOpType::PredictionError:
            {
                assert(inputs.size() == 2);

                if (inputs[0].Shape().NumAxes() > 1)
                    InvalidArgument("The shape of input operands for the %s operation should have at most one axis", PrimitiveOpTypeName(op));

                auto predictionShape = inputs[0].Shape();
                auto labelsShape = inputs[1].Shape();
                if (predictionShape != labelsShape)
                    RuntimeError("Prediction output operand's shape %s is incompatible with label operand's shape %s for the %s operation", AsString(predictionShape).c_str(), AsString(labelsShape).c_str(), PrimitiveOpTypeName(op));

                std::vector<size_t> reductionAxes;
                for (size_t i = 0; i < inputs[0].Shape().NumAxes(); ++i)
                    reductionAxes.push_back(i);

                outputs.push_back(Variable(ReductionOpOutputShape(op, predictionShape, reductionAxes), outputDataType, owner, {}));
                break;
            }
            case PrimitiveOpType::ReduceSum:
            {
                assert(inputs.size() == 1);

                // TODO: For reductions, we should remove any of the dynamic axes from 'outputDynamicAxes' that are being reduced over. 
                // Currently we only support reductions that reduce over all axes
                std::vector<Axis> reductionOutputDynamicAxes = {};
                std::vector<size_t> reductionAxes;
                for (size_t i = 0; i < inputs[0].Shape().NumAxes(); ++i)
                    reductionAxes.push_back(i);

                outputs.push_back(Variable(ReductionOpOutputShape(op, inputs[0].Shape(), reductionAxes), outputDataType, owner, reductionOutputDynamicAxes));
                break;
            }
            case PrimitiveOpType::Combine:
                outputs = inputs;
                break;
            default:
                LogicError("Specified op %s not yet supported", PrimitiveOpTypeName(op));
                break;
            }

            return outputs;
        }

    private:
        PrimitiveOpType m_op;
        Dictionary m_functionConfig;
    };

    class CNTKBackPropState final : public BackPropState
    {
    public:
        CNTKBackPropState(const FunctionPtr& function, const std::pair<Variable, int64_t>& evalTimeStamp)
            : BackPropState(function), m_evalTimeStamp(evalTimeStamp)
        {}

        std::pair<Variable, int64_t> EvalTimeStamp() const
        {
            return m_evalTimeStamp;
        }

    private:
        std::pair<Variable, int64_t> m_evalTimeStamp;
    };
    typedef _Internal::_ReferenceCounterSharedPtr<CNTKBackPropState> CNTKBackPropStatePtr;

    class CompositeFunction;
    typedef _Internal::_ReferenceCounterSharedPtr<CompositeFunction> CompositeFunctionPtr;

    class CompositeFunction final : public Function
    {
        friend class Function;

    public:
        static CompositeFunctionPtr Create(const FunctionPtr& rootFunction, const std::wstring& name = L"")
        {
            _Internal::_SimpleSet<FunctionPtr> visitedFunctions;

            // Call _DetermineInputs to get the set of all functions in the graph
            _DetermineInputs(rootFunction, visitedFunctions);

            auto func = new CompositeFunction(rootFunction, std::move(visitedFunctions), name);
            return CompositeFunctionPtr(func, [](_ReferenceCounter* ptr) { delete ptr; });
        }

        virtual BackPropStatePtr Forward(const _Internal::_SimpleMap<Variable, const ValuePtr>& arguments,
                                         _Internal::_SimpleMap<Variable, ValuePtr>& outputs,
                                         const _Internal::_SimpleSet<Variable>& outputsToRetainBackwardStateFor,
                                         const DeviceDescriptor& computeDevice) override;

        virtual void Backward(const BackPropStatePtr& state,
                              const _Internal::_SimpleMap<Variable, const ValuePtr>& rootGradientValues,
                              _Internal::_SimpleMap<Variable, ValuePtr>& backPropagatedGradientValuesForInputs) override;

    private:
        virtual void _ReplacePlaceholders(const _Internal::_SimpleMap<Placeholder, Variable>& placeholderReplacements, _Internal::_SimpleSet<const Function*>& visitedFunctions, _Internal::_SimpleSet<Placeholder>& replacedPlaceholders) override;

        CompositeFunction(const FunctionPtr& rootFunction, _Internal::_SimpleSet<FunctionPtr>&& allPrimitiveFunctions, const std::wstring& name)
            : Function({}, rootFunction->Outputs(), rootFunction, name), m_allPrimitiveFunctions(std::move(allPrimitiveFunctions))
        {
        }

        std::vector<Variable> DetermineInputs() const
        {
            _Internal::_SimpleSet<FunctionPtr> visitedFunctions;
            return _DetermineInputs(RootFunction(), visitedFunctions);
        }

        static std::vector<Variable> _DetermineInputs(const FunctionPtr& rootFunction, _Internal::_SimpleSet<FunctionPtr>& visitedFunctions)
        {
            visitedFunctions.Insert(rootFunction);

            std::vector<Variable> inputs;
            std::vector<Variable> rootFunctionInputs = rootFunction->Inputs();
            for (size_t i = 0; i < rootFunctionInputs.size(); ++i)
            {
                Variable currentInput = rootFunctionInputs[i];
                if (currentInput.Kind() != VariableKind::Output)
                    inputs.push_back(currentInput);
                else if (!visitedFunctions.Contains(currentInput.Owner()))
                {
                    FunctionPtr function = currentInput.Owner();
                    std::vector<Variable> functionInputs = _DetermineInputs(function, visitedFunctions);
                    std::copy(functionInputs.begin(), functionInputs.end(), std::back_inserter(inputs));
                }
            }

            return inputs;
        }

        template <typename ElementType>
        Microsoft::MSR::CNTK::ComputationNetworkPtr GetComputationNetwork(const DeviceDescriptor& device, const _Internal::_SimpleSet<Variable>& backpropRoots);

        template <typename ElementType>
        static Microsoft::MSR::CNTK::ComputationNodeBasePtr GetOutputVariableNode(const Variable& variable, Microsoft::MSR::CNTK::ComputationNetworkPtr& network, Microsoft::MSR::CNTK::ComputationNetworkBuilder<ElementType>& builder, std::unordered_map<Variable, Microsoft::MSR::CNTK::ComputationNodeBasePtr>& variableToNodeMap, std::unordered_map<Variable, bool>& isVariableRootMap);

        template <typename ElementType>
        static Microsoft::MSR::CNTK::ComputationNodeBasePtr GetNode(const Variable& variable, Microsoft::MSR::CNTK::ComputationNetworkPtr& network, Microsoft::MSR::CNTK::ComputationNetworkBuilder<ElementType>& builder, std::unordered_map<Variable, Microsoft::MSR::CNTK::ComputationNodeBasePtr>& variableToNodeMap, std::unordered_map<Variable, bool>& isVariableRootMap);

        void PopulateNetworkInputs(const _Internal::_SimpleMap<Variable, const ValuePtr>& arguments);
        void PopulateNetworkGradients(const _Internal::_SimpleMap<Variable, const ValuePtr>& gradients);

        void GetNetworkOutputs(std::unordered_map<Variable, ValuePtr>& outputs);
        void GetNetworkGradients(std::unordered_map<Variable, ValuePtr>& gradients);

        template <typename ElementType>
        static std::pair<std::shared_ptr<const Microsoft::MSR::CNTK::Matrix<ElementType>>, Microsoft::MSR::CNTK::MBLayoutPtr> GetCNTKImplMatrixAndMBLayoutFromValueObject(Variable var, const ValuePtr& value);

        template <typename ElementType>
        static ValuePtr GetValueObjectFromCNTKImplMatrixAndMBLayout(Variable var, const Microsoft::MSR::CNTK::Matrix<ElementType>& matrix, const Microsoft::MSR::CNTK::MBLayoutPtr& layout);

    private:
        _Internal::_SimpleSet<FunctionPtr> m_allPrimitiveFunctions;
        std::unordered_map<Variable, Microsoft::MSR::CNTK::ComputationNodeBasePtr> m_variableToNodeMap;
        std::unordered_map<Variable, bool> m_isVariableRootMap;
        Microsoft::MSR::CNTK::ComputationNetworkPtr m_computationNetwork;
        std::unordered_set<Variable> m_currentBackpropRoots;
    };
}
