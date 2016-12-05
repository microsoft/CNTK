//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include <iterator>
#include "ComputationNetwork.h"
#include "Utils.h"
#include "ConvolveGeometry.h"

namespace CNTK
{
    enum class PrimitiveOpType : unsigned int
    {
        Negate,
        Sigmoid,
        Tanh,
        ReLU,
        Exp,
        Log,
        Sqrt,
        Floor,
        Abs,
        Reciprocal,
        Softmax,
        Hardmax,
        TransposeAxes,
        Where,
        Slice,
        Pooling,
        SumAll,
        Plus,
        Minus,
        ElementTimes,
        Equal,
        NotEqual,
        Less,
        LessEqual,
        Greater,
        GreaterEqual,
        PackedIndex,
        GatherPacked,
        Times,
        TransposeTimes,
        Convolution,
        SquaredError,
        CrossEntropyWithSoftmax,
        ClassificationError,
        PastValue,
        FutureValue,
        ReduceElements,
        BatchNormalization,
        Combine,
    };
}

namespace std
{
    template <> struct hash<CNTK::PrimitiveOpType>
    {
        size_t operator()(const CNTK::PrimitiveOpType& x) const
        {
            return std::hash<unsigned int>()((unsigned int)x);
        }
    };
}

namespace CNTK
{
    inline const char* PrimitiveOpTypeName(PrimitiveOpType opType)
    {
        static std::unordered_map<PrimitiveOpType, const char*> primitiveOpNames = {
            { PrimitiveOpType::Negate, "Negate" },
            { PrimitiveOpType::Sigmoid, "Sigmoid" },
            { PrimitiveOpType::Tanh, "Tanh" },
            { PrimitiveOpType::ReLU, "ReLU" },
            { PrimitiveOpType::Exp, "Exp" },
            { PrimitiveOpType::Log, "Log" },
            { PrimitiveOpType::Sqrt, "Sqrt" },
            { PrimitiveOpType::Floor, "Floor" },
            { PrimitiveOpType::Abs, "Abs" },
            { PrimitiveOpType::Reciprocal, "Reciprocal" },
            { PrimitiveOpType::Softmax, "Softmax" },
            { PrimitiveOpType::Hardmax, "Hardmax" },
            { PrimitiveOpType::TransposeAxes, "TransposeAxes" },
            { PrimitiveOpType::Where, "Where" },
            { PrimitiveOpType::Slice, "Slice" },
            { PrimitiveOpType::Pooling, "Pooling" },
            { PrimitiveOpType::SumAll, "SumAll" },
            { PrimitiveOpType::Plus, "Plus" },
            { PrimitiveOpType::Minus, "Minus" },
            { PrimitiveOpType::ElementTimes, "ElementTimes" },
            { PrimitiveOpType::Equal, "Equal" },
            { PrimitiveOpType::NotEqual, "NotEqual" },
            { PrimitiveOpType::Less, "Less" },
            { PrimitiveOpType::LessEqual, "LessEqual" },
            { PrimitiveOpType::Greater, "Greater" },
            { PrimitiveOpType::GreaterEqual, "GreaterEqual" },
            { PrimitiveOpType::PackedIndex, "PackedIndex" },
            { PrimitiveOpType::GatherPacked, "GatherPacked" },
            { PrimitiveOpType::Times, "Times" },
            { PrimitiveOpType::TransposeTimes, "TransposeTimes" },
            { PrimitiveOpType::Convolution, "Convolution" },
            { PrimitiveOpType::SquaredError, "SquaredError" },
            { PrimitiveOpType::CrossEntropyWithSoftmax, "CrossEntropyWithSoftmax" },
            { PrimitiveOpType::ClassificationError, "ClassificationError" },
            { PrimitiveOpType::PastValue, "PastValue" },
            { PrimitiveOpType::FutureValue, "FutureValue" },
            { PrimitiveOpType::ReduceElements, "ReduceElements" },
            { PrimitiveOpType::BatchNormalization, "BatchNormalization" },
            { PrimitiveOpType::Combine, "Combine" }
        };

        if (primitiveOpNames.find(opType) == primitiveOpNames.end())
            LogicError("Unknown PrimitiveOpType");

        return primitiveOpNames.find(opType)->second;
    }

    class PrimitiveFunction final : public Function
    {
    public:
        static const std::wstring InternalSumReductionOpName;
        static const std::wstring InternalLogSumReductionOpName;
        static const std::wstring InternalMeanReductionOpName;
        static const std::wstring InternalMaxReductionOpName;
        static const std::wstring InternalMinReductionOpName;
        static const std::wstring InternalAllReductionOpName;
        static const std::wstring InternalAnyReductionOpName;

    public:
        PrimitiveFunction(PrimitiveOpType op, const std::vector<Variable>& inputs, Dictionary&& functionConfig, const std::wstring& functionName = L"")
            : Function(inputs, GetOutputVariables(op, inputs, this, functionConfig), nullptr, functionName), m_op(op), m_functionConfig(std::move(functionConfig))
        {
        }

        virtual BackPropStatePtr Forward(const std::unordered_map<Variable, ValuePtr>& /*arguments*/,
                                         std::unordered_map<Variable, ValuePtr>& /*outputs*/,
                                         const DeviceDescriptor& /*computeDevice*/,
                                         const std::unordered_set<Variable>& /*outputsToRetainBackwardStateFor*/) override
        {
            NOT_IMPLEMENTED;
        }

        virtual void Backward(const BackPropStatePtr& /*state*/,
                              const std::unordered_map<Variable, ValuePtr>& /*rootGradientValues*/,
                              std::unordered_map<Variable, ValuePtr>& /*backPropagatedGradientValuesForInputs*/) override
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
        // The following helper functions are used to determine the output shape for different 
        // types of primitive operations accounting for broadcasting and reductions where applicable.
        static NDShape UnaryElementwiseOpOutputShape(const NDShape& operandShape)
        {
            return operandShape;
        }

        static NDShape BinaryElementwiseOpOutputShape(PrimitiveOpType op, const NDShape& leftOperandShape, const NDShape& rightOperandShape, bool broadcastAllowed = true)
        {
            const auto& shapeWithSmallerNumAxes = (leftOperandShape.NumAxes() > rightOperandShape.NumAxes()) ? rightOperandShape : leftOperandShape;
            const auto& shapeWithLargerNumAxes = (leftOperandShape.NumAxes() > rightOperandShape.NumAxes()) ? leftOperandShape : rightOperandShape;
            size_t numOutputAxes = shapeWithLargerNumAxes.NumAxes();
            std::vector<size_t> outputDims(numOutputAxes);
            for (size_t i = 0; i < shapeWithSmallerNumAxes.NumAxes(); ++i)
            {
                if ((leftOperandShape[i] == NDShape::InferredDimension) && (rightOperandShape[i] == NDShape::InferredDimension))
                    outputDims[i] = NDShape::InferredDimension;
                else if (leftOperandShape[i] == NDShape::InferredDimension)
                    outputDims[i] = rightOperandShape[i];
                else if (rightOperandShape[i] == NDShape::InferredDimension)
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

        static NDShape TimesOpOutputShape(const NDShape& leftOperandShape, const NDShape& rightOperandShape, size_t numOutputAxes)
        {
            if (numOutputAxes == 0)
                InvalidArgument("Output #axes of times operation should be at least one");

            if (numOutputAxes > leftOperandShape.NumAxes())
                InvalidArgument("Output #axes of times operation can at most be the #axes of the left operand");

            size_t numReductionAxes = leftOperandShape.NumAxes() - numOutputAxes;

            // The 'numReductionAxes' trailing dimensions of the left operand's shape must match the corresponding leading
            // dimensions of the right operand

            if (rightOperandShape.NumAxes() != numReductionAxes)
                RuntimeError("The right operand's #axes in a times operation should equal #axes being reduced over!");

            if (leftOperandShape.SubShape(numOutputAxes) != rightOperandShape)
                InvalidArgument("The trailing dimensions of the left operand (%s) do not match the right operand's dimensions (%s)",
                                AsString(leftOperandShape.SubShape(numOutputAxes)).c_str(),
                                AsString(rightOperandShape).c_str());

            return leftOperandShape.SubShape(0, numOutputAxes);
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

        static NDShape ConvolutionOpOutputShape(const NDShape& operandShape, const NDShape& kernelShape, const NDShape& outputMapCount, const NDShape& strides,
                                                const std::vector<bool>& sharing,
                                                std::vector<bool>& autoPad, const NDShape& lowerPad, const NDShape& upperPad,
                                                bool transpose)
        {
            decltype(&Microsoft::MSR::CNTK::ConvolveGeometry::ComputeOutputShape) computeOutputShapeFunc;
            if (!transpose)
                computeOutputShapeFunc = &Microsoft::MSR::CNTK::ConvolveGeometry::ComputeOutputShape;
            else
                computeOutputShapeFunc = &Microsoft::MSR::CNTK::ConvolveGeometry::ComputeInputShape;

            return AsNDShape(computeOutputShapeFunc(AsTensorShape(operandShape, true), AsTensorShape(kernelShape, true), AsTensorShape(outputMapCount, true), AsTensorShape(strides, true), sharing, autoPad, AsTensorShape(lowerPad, true), AsTensorShape(upperPad, true)));
        }

        // TODO: Reconcile this with the ComputationNode::Validate functionality in core CNTK to avoid duplication of inference logic
        static std::vector<Variable> GetOutputVariables(PrimitiveOpType op, const std::vector<Variable>& inputs, Function* owner, const Dictionary& functionConfig)
        {
            std::vector<Variable> outputs;

            // TODO: We are just using the input[0]'s DataType as output node's DataType. This is not always correct
            DataType outputDataType = inputs[0].GetDataType();

            // We currently require that the inputs' dynamic axes if any match
            std::vector<Axis> outputDynamicAxes;
            if (op == PrimitiveOpType::Where)
                ;
            else if ((op == PrimitiveOpType::PackedIndex) || (op == PrimitiveOpType::GatherPacked))
            {
                outputDynamicAxes = inputs[1].DynamicAxes();
            }
            else
            {
                outputDynamicAxes = inputs[0].DynamicAxes();
                for (auto inputVar : inputs)
                {
                    auto currentInputDynamicAxes = inputVar.DynamicAxes();
                    if (outputDynamicAxes.empty())
                        outputDynamicAxes = currentInputDynamicAxes;
                    else
                    {
                        if (!currentInputDynamicAxes.empty() && (currentInputDynamicAxes != outputDynamicAxes))
                            LogicError("Currently if an operand of a binary elementwise operation has any dynamic axes, those must match the dynamic axes of the other operand");
                    }
                }
            }

            switch (op)
            {
            case PrimitiveOpType::Negate:
            case PrimitiveOpType::Sigmoid:
            case PrimitiveOpType::Tanh:
            case PrimitiveOpType::ReLU:
            case PrimitiveOpType::Exp:
            case PrimitiveOpType::Log:
            case PrimitiveOpType::Sqrt:
            case PrimitiveOpType::Floor:
            case PrimitiveOpType::Abs:
            case PrimitiveOpType::Reciprocal:
            case PrimitiveOpType::Softmax:
            case PrimitiveOpType::Hardmax:
                assert(inputs.size() == 1);
                if (((op == PrimitiveOpType::Softmax) || (op == PrimitiveOpType::Hardmax)) && (inputs[0].Shape().NumAxes() > 1))
                    InvalidArgument("Softmax/Hardmax operation can only be applied to a 1D input");

                outputs.push_back(Variable(UnaryElementwiseOpOutputShape(inputs[0].Shape()), outputDataType, owner, outputDynamicAxes));
                break;
            case PrimitiveOpType::TransposeAxes:
            {
                assert(inputs.size() == 1);
                auto axis1 = Axis(functionConfig[L"axis1"].GetValue<std::wstring>());
                auto axis2 = Axis(functionConfig[L"axis2"].GetValue<std::wstring>());

                if (!axis1.IsStaticAxis() || !axis2.IsStaticAxis())
                    LogicError("TransposeAxes operation currently does not support transposing dynamic axes");

                auto transposedTensorShape = AsTensorShape(inputs[0].Shape(), true);
                transposedTensorShape.SwapDimsInPlace(axis1.StaticAxisIndex(), axis2.StaticAxisIndex());
                outputs.push_back(Variable(AsNDShape(transposedTensorShape), outputDataType, owner, outputDynamicAxes));
                break;
            }
            case PrimitiveOpType::Where:
            {
                assert(inputs.size() == 1);
                std::vector<Axis> newDynamicAxes;
                auto newDynamicAxesNames = AsBasicElementTypeVector<std::wstring>(functionConfig[L"newDynamicAxes"].GetValue<std::vector<DictionaryValue>>());
                for (auto axisName : newDynamicAxesNames)
                    newDynamicAxes.push_back(Axis(axisName));

                outputs.push_back(Variable(UnaryElementwiseOpOutputShape(inputs[0].Shape()), outputDataType, owner, newDynamicAxes));
                break;
            }
            case PrimitiveOpType::Pooling:
            {
                assert(inputs.size() == 1);
                auto poolingWindowsShape = functionConfig[L"poolingWindowShape"].GetValue<NDShape>();
                auto strides = functionConfig[L"strides"].GetValue<NDShape>();
                auto lowerPad = functionConfig[L"lowerPad"].GetValue<NDShape>();
                auto upperPad = functionConfig[L"upperPad"].GetValue<NDShape>();
                auto autoPadding = AsBasicElementTypeVector<bool>(functionConfig[L"autoPadding"].GetValue<std::vector<DictionaryValue>>());
                outputs.push_back(Variable(ConvolutionOpOutputShape(inputs[0].Shape(), poolingWindowsShape, { 1 }, strides, { true }, autoPadding, lowerPad, upperPad, false), outputDataType, owner, outputDynamicAxes));
                break;
            }
            case PrimitiveOpType::SumAll:
                assert(inputs.size() == 1);
                outputs.push_back(Variable({}, outputDataType, owner, std::vector<Axis>({})));
                break;
            case PrimitiveOpType::Plus:
            case PrimitiveOpType::Minus:
            case PrimitiveOpType::ElementTimes:
            case PrimitiveOpType::Equal:
            case PrimitiveOpType::NotEqual:
            case PrimitiveOpType::Less:
            case PrimitiveOpType::LessEqual:
            case PrimitiveOpType::Greater:
            case PrimitiveOpType::GreaterEqual:
                assert(inputs.size() == 2);
                outputs.push_back(Variable(BinaryElementwiseOpOutputShape(op, inputs[0].Shape(), inputs[1].Shape()), outputDataType, owner, outputDynamicAxes));
                break;
            case PrimitiveOpType::Times:
            {
                assert(inputs.size() == 2);
                size_t numOutputAxes = functionConfig[L"numOutputAxes"].GetValue<size_t>();
                outputs.push_back(Variable(TimesOpOutputShape(inputs[0].Shape(), inputs[1].Shape(), numOutputAxes), outputDataType, owner, outputDynamicAxes));
                break;
            }
            case PrimitiveOpType::TransposeTimes:
            {
                assert(inputs.size() == 2);

                auto numLeftOperandAxes = inputs[0].Shape().NumAxes();
                if (numLeftOperandAxes > 2)
                    InvalidArgument("TransposeTimes operation only supports left operands of rank 1 or 2");

                NDShape transposedLeftOperandShape(2, 1);
                for (size_t i = 0; i < numLeftOperandAxes; ++i)
                    transposedLeftOperandShape[transposedLeftOperandShape.NumAxes() - i - 1] = inputs[0].Shape()[i];

                size_t numOutputAxes = functionConfig[L"numOutputAxes"].GetValue<size_t>();
                outputs.push_back(Variable(TimesOpOutputShape(transposedLeftOperandShape, inputs[1].Shape(), numOutputAxes), outputDataType, owner, outputDynamicAxes));
                break;
            }
            case PrimitiveOpType::Convolution:
            {
                assert(inputs.size() == 2);
                auto strides = functionConfig[L"strides"].GetValue<NDShape>();
                auto lowerPad = functionConfig[L"lowerPad"].GetValue<NDShape>();
                auto upperPad = functionConfig[L"upperPad"].GetValue<NDShape>();
                auto sharing = AsBasicElementTypeVector<bool>(functionConfig[L"sharing"].GetValue<std::vector<DictionaryValue>>());
                auto autoPadding = AsBasicElementTypeVector<bool>(functionConfig[L"autoPadding"].GetValue<std::vector<DictionaryValue>>());
                bool transpose = functionConfig[L"transpose"].GetValue<bool>();
                if (inputs[0].Shape().NumAxes() < inputs[1].Shape().NumAxes())
                    InvalidArgument("The convolution map should have at least as many axes as the shape of the input it operates on!");

                NDShape outputMapCount, kernelShape;
                std::tie(outputMapCount, kernelShape) = GetConvolutionOutputMapCountAndKernelShape(inputs[0].Shape(), inputs[1].Shape());
                outputs.push_back(Variable(ConvolutionOpOutputShape(inputs[1].Shape(), kernelShape, outputMapCount, strides, sharing, autoPadding, lowerPad, upperPad, transpose), outputDataType, owner, outputDynamicAxes));
                break;
            }
            case PrimitiveOpType::SquaredError:
            case PrimitiveOpType::CrossEntropyWithSoftmax:
            case PrimitiveOpType::ClassificationError:
            {
                assert(inputs.size() == 2);

                if ((inputs[0].Shape().NumAxes() > 2) || ((inputs[0].Shape().NumAxes() > 1) && (inputs[0].Shape()[1] != 1)))
                    InvalidArgument("The shape of input operands for the %s operation should have at most one axis", PrimitiveOpTypeName(op));

                auto predictionShape = inputs[0].Shape();
                auto labelsShape = inputs[1].Shape();
                if (predictionShape != labelsShape)
                    RuntimeError("Prediction output operand's shape %s is incompatible with label operand's shape %s for the %s operation", AsString(predictionShape).c_str(), AsString(labelsShape).c_str(), PrimitiveOpTypeName(op));

                std::vector<size_t> reductionAxes;
                for (size_t i = 0; i < inputs[0].Shape().NumAxes(); ++i)
                    reductionAxes.push_back(i);

                outputs.push_back(Variable(ReductionOpOutputShape(op, predictionShape, reductionAxes), outputDataType, owner, std::vector<Axis>({})));
                break;
            }
            case PrimitiveOpType::PastValue:
            case PrimitiveOpType::FutureValue:
            {
                assert(inputs.size() == 2);
                Variable initialStateVar = inputs[0];
                Variable inputOperandVar = inputs[1];
                // TODO: Current we only support a scalar initial state
                if (!initialStateVar.IsConstant() || (initialStateVar.Shape().NumAxes() > 0))
                    LogicError("Currently PastValue/FutureValue Function only supports scalar initial state");

                // TODO: We currently only support input operand with 1 static axis for PastValue/FutureValue
                if (inputOperandVar.Shape().NumAxes() > 1)
                    LogicError("Currently PastValue/FutureValue Function only supports input operand with <= 1 static axis");

                // TODO: We currently only support input operand with 1 dynamic axis for PastValue/FutureValue
                if (inputOperandVar.DynamicAxes().size() != 2)
                    LogicError("Currently PastValue/FutureValue Function only supports input operand with with 2 dynamic axis (1 sequence-axis and 1 batch-axis)");

                outputs.push_back(Variable(UnaryElementwiseOpOutputShape(inputs[1].Shape()), outputDataType, owner, outputDynamicAxes));
                break;
            }
            case PrimitiveOpType::ReduceElements:
            {
                assert(inputs.size() == 1);
                auto CNTKInternalReductionAxisIndex = functionConfig[L"CNTKInternalReductionAxisIndex"].GetValue<size_t>();
                std::vector<size_t> reductionAxes;
                // TODO: Do not use a integer literal for the special value of axis id that indicates all static axes
                if (CNTKInternalReductionAxisIndex == 0)
                {
                    for (size_t i = 0; i < inputs[0].Shape().NumAxes(); ++i)
                        reductionAxes.push_back(i);
                }
                else
                    reductionAxes.push_back(CNTKInternalReductionAxisIndex - 1);

                outputs.push_back(Variable(ReductionOpOutputShape(op, inputs[0].Shape(), reductionAxes), outputDataType, owner, inputs[0].DynamicAxes()));
                break;
            }
            case PrimitiveOpType::BatchNormalization:
                outputs.push_back(Variable(UnaryElementwiseOpOutputShape(inputs[0].Shape()), outputDataType, owner, outputDynamicAxes));
                break;
            case PrimitiveOpType::Combine:
                outputs = inputs;
                break;
            case PrimitiveOpType::PackedIndex:
                outputs.push_back(Variable(UnaryElementwiseOpOutputShape(inputs[1].Shape()), outputDataType, owner, outputDynamicAxes));
                break;
            case PrimitiveOpType::GatherPacked:
            {
                bool sourceHasDynamicAxis = !inputs[0].DynamicAxes().empty();
                NDShape outputShape;

                // inherit tensor dimension from sourceData, minus the last (column or time) dimension. TODO this needs to become simpler...
                if (sourceHasDynamicAxis)
                    outputShape = inputs[0].Shape();
                else
                {
                    if (inputs[0].Shape().NumAxes() > 1)
                        outputShape = outputShape.SubShape(0, outputShape.NumAxes() - 1);
                    else
                        outputShape = {};
                }

                outputs.push_back(Variable(outputShape, outputDataType, owner, outputDynamicAxes));
                break;
            }
            case PrimitiveOpType::Slice:
            {
                auto axis = Axis(functionConfig[L"axis"].GetValue<std::wstring>());
                int beginIndex = functionConfig[L"beginIndex"].GetValue<size_t>();
                int endIndex = functionConfig[L"endIndex"].GetValue<size_t>();
                if (!axis.IsStaticAxis())
                    LogicError("Built-in Slice operation currently does not support slicing along dynamic axis");

                if (axis.StaticAxisIndex() >= inputs[0].Shape().NumAxes())
                    InvalidArgument("The specified axis index (%d) for the Slice operation is outside the bounds of the available axes of the input", (int)axis.StaticAxisIndex());

                size_t sliceAxisDim = inputs[0].Shape()[axis.StaticAxisIndex()];
                int realBeginIndex = (beginIndex >= 0) ? beginIndex : beginIndex + sliceAxisDim;
                int realEndIndex = (endIndex > 0) ? endIndex : endIndex + sliceAxisDim;
                if ((sliceAxisDim < realEndIndex) || (realEndIndex < realBeginIndex) || (realBeginIndex < 0))
                    RuntimeError("Slice operation: Index range [%d,%d), interpreted as [%d,%d), is invalid for input ([%S]).",
                                 beginIndex,
                                 endIndex,
                                 realBeginIndex,
                                 realEndIndex,
                                 inputs[0].Shape().AsString().c_str());

                auto outputTensorShape = AsTensorShape(inputs[0].Shape(), true);

                // propagate as much as we can
                if ((axis.StaticAxisIndex() < outputTensorShape.GetRank()) && (0 <= realBeginIndex) && (realBeginIndex <= realEndIndex) && (realEndIndex <= sliceAxisDim))
                    outputTensorShape.NarrowTo(axis.StaticAxisIndex(), realBeginIndex, realEndIndex);


                outputs.push_back(Variable(AsNDShape(outputTensorShape), outputDataType, owner, outputDynamicAxes));
                break;
            }
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
    typedef std::shared_ptr<CNTKBackPropState> CNTKBackPropStatePtr;

    class CompositeFunction;
    typedef std::shared_ptr<CompositeFunction> CompositeFunctionPtr;

    class CompositeFunction final : public Function
    {
        friend class Function;
        friend class CompositeMinibatchSource;

        template <typename T, typename ...CtorArgTypes>
        friend inline std::shared_ptr<T> MakeSharedObject(CtorArgTypes&& ...ctorArgs);

        template <typename ElementType>
        friend void SaveAsLegacyModel(const FunctionPtr& rootFunction, const std::wstring& modelFile);

        friend void ComputeInputPerDimMeansAndInvStdDevs(const MinibatchSourcePtr& minibatchSource,
                                                         std::unordered_map<StreamInfo, std::pair<NDArrayViewPtr, NDArrayViewPtr>>& computedMeanAndInvStdDevs,
                                                         const DeviceDescriptor& device /*= DeviceDescriptor::CPUDevice()*/);

    public:
        static std::wstring s_internalDefaultDynamicAxisName;
        static std::wstring s_internalNoSequenceAxisName;

        static Axis NextAutoGeneratedDynamicAxis()
        {
            static std::atomic<unsigned int> nextAutoGeneratedDynamicAxis(0);
            static const std::wstring autoGeneratedDynamicAxisNamePrefix = L"autoGeneratedDynamicAxis_";
            return Axis(autoGeneratedDynamicAxisNamePrefix + std::to_wstring(nextAutoGeneratedDynamicAxis++));
        }

    public:
        static CompositeFunctionPtr Create(const FunctionPtr& rootFunction, const std::wstring& name = L"")
        {
            std::unordered_set<FunctionPtr> visitedFunctions;

            // Call DetermineInputs to get the set of all functions in the graph
            DetermineInputs(rootFunction, visitedFunctions);

            return MakeSharedObject<CompositeFunction>(rootFunction, std::move(visitedFunctions), name);
        }

        virtual BackPropStatePtr Forward(const std::unordered_map<Variable, ValuePtr>& arguments,
                                         std::unordered_map<Variable, ValuePtr>& outputs,
                                         const DeviceDescriptor& computeDevice,
                                         const std::unordered_set<Variable>& outputsToRetainBackwardStateFor) override;

        virtual void Backward(const BackPropStatePtr& state,
                              const std::unordered_map<Variable, ValuePtr>& rootGradientValues,
                              std::unordered_map<Variable, ValuePtr>& backPropagatedGradientValuesForInputs) override;

    private:
        virtual void ReplacePlaceholders(const std::unordered_map<Placeholder, Variable>& placeholderReplacements,
                                         std::unordered_set<const Function*>& visitedFunctions,
                                         std::unordered_set<Placeholder>& replacedPlaceholders) override;

        CompositeFunction(const FunctionPtr& rootFunction, std::unordered_set<FunctionPtr>&& allPrimitiveFunctions, const std::wstring& name)
            : Function({}, rootFunction->Outputs(), rootFunction, name), m_allPrimitiveFunctions(std::move(allPrimitiveFunctions))
        {
        }

        std::vector<Variable> DetermineInputs() const
        {
            std::unordered_set<FunctionPtr> visitedFunctions;
            return DetermineInputs(RootFunction(), visitedFunctions);
        }

        // Recursively traverses the Function graph underlying the 'rootFunction' to determine all the leaves (aka inputs) of the graph
        static std::vector<Variable> DetermineInputs(const FunctionPtr& rootFunction, std::unordered_set<FunctionPtr>& visitedFunctions)
        {
            visitedFunctions.insert(rootFunction);

            std::vector<Variable> inputs;
            std::vector<Variable> rootFunctionInputs = rootFunction->Inputs();
            for (auto rootInput : rootFunctionInputs)
            {
                if (!rootInput.IsOutput())
                    inputs.push_back(rootInput);
                else if (visitedFunctions.find(rootInput.Owner()) == visitedFunctions.end())
                {
                    FunctionPtr function = rootInput.Owner();
                    std::vector<Variable> functionInputs = DetermineInputs(function, visitedFunctions);
                    std::copy(functionInputs.begin(), functionInputs.end(), std::back_inserter(inputs));
                }
            }

            return inputs;
        }

        template <typename ElementType>
        Microsoft::MSR::CNTK::ComputationNetworkPtr GetComputationNetwork(const DeviceDescriptor& device, const std::unordered_set<Variable>& backpropRoots);

        template <typename ElementType>
        static Microsoft::MSR::CNTK::ComputationNodeBasePtr GetOutputVariableNode(const Variable& variable, Microsoft::MSR::CNTK::ComputationNetworkPtr& network, Microsoft::MSR::CNTK::ComputationNetworkBuilder<ElementType>& builder, std::unordered_map<Variable, Microsoft::MSR::CNTK::ComputationNodeBasePtr>& variableToNodeMap, std::unordered_map<Variable, bool>& isVariableRootMap);

        template <typename ElementType>
        static Microsoft::MSR::CNTK::ComputationNodeBasePtr GetNode(const Variable& variable, Microsoft::MSR::CNTK::ComputationNetworkPtr& network, Microsoft::MSR::CNTK::ComputationNetworkBuilder<ElementType>& builder, std::unordered_map<Variable, Microsoft::MSR::CNTK::ComputationNodeBasePtr>& variableToNodeMap, std::unordered_map<Variable, bool>& isVariableRootMap);

        template <typename ElementType>
        static void PopulateComputationNodeValue(const std::pair<Variable, ValuePtr>& variableValue, Microsoft::MSR::CNTK::ComputationNodeBasePtr& computationNode);
        void PopulateNetworkInputs(const std::unordered_map<Variable, ValuePtr>& arguments);

        template <typename ElementType>
        static void PopulateComputationNodeGradient(const std::pair<Variable, ValuePtr>& variableGradient, Microsoft::MSR::CNTK::ComputationNodeBasePtr& computationNode);
        void PopulateNetworkGradients(const std::unordered_map<Variable, ValuePtr>& gradients);

        static void GetNodeOutputOrGradient(Variable var, ValuePtr& varValue, Microsoft::MSR::CNTK::ComputationNodeBasePtr& computationNode, bool getGradient);
        void GetNetworkOutputs(std::unordered_map<Variable, ValuePtr>& outputs);
        void GetNetworkGradients(std::unordered_map<Variable, ValuePtr>& gradients);

        template <typename ElementType>
        static std::pair<std::shared_ptr<const Microsoft::MSR::CNTK::Matrix<ElementType>>, Microsoft::MSR::CNTK::MBLayoutPtr> GetCNTKImplMatrixAndMBLayoutFromValueObject(Variable var, const ValuePtr& value);

        template <typename ElementType>
        static ValuePtr GetValueObjectFromCNTKImplMatrixAndMBLayout(const NDShape& sampleShape, const Microsoft::MSR::CNTK::Matrix<ElementType>& matrix, const Microsoft::MSR::CNTK::MBLayoutPtr& layout, bool readOnly = true);
        template <typename ElementType>
        static ValuePtr GetValueObjectFromCNTKImplMatrixAndMBLayout(Variable var, const Microsoft::MSR::CNTK::Matrix<ElementType>& matrix, const Microsoft::MSR::CNTK::MBLayoutPtr& layout, bool readOnly = true);

    private:

        // Set of all primitive functions in the graph underlying 'this' Function. Also keeps the primitive Function objects alive 
        // by holding strong references to them
        std::unordered_set<FunctionPtr> m_allPrimitiveFunctions;

        // A map from Variable objects to ComputationNode objects in the ComputationNetwork instance that implements 'this' Composite Function
        std::unordered_map<Variable, Microsoft::MSR::CNTK::ComputationNodeBasePtr> m_variableToNodeMap;

        // A map that tells whether a Variable in the graph underlying 'this' Function is a root of the graph
        std::unordered_map<Variable, bool> m_isVariableRootMap;

        Microsoft::MSR::CNTK::ComputationNetworkPtr m_computationNetwork;

        // The backpropRoots sepecified in the most recent 'Forward' call on 'this' Function.
        // This indicates for which of it's roots has 'this' Function retained required intermediate 
        // states from the previos Forward call to be able to backpropagate gradients backwards from in
        // the next 'Backward' call.
        std::unordered_set<Variable> m_currentBackpropRoots;
    };

    inline std::vector<CNTK::Axis> DynamicAxesFromInternalDynamicAxisName(const std::wstring& internalDynamicAxisName)
    {
        std::vector<CNTK::Axis> inputVarDynamicAxes;
        if (internalDynamicAxisName == CNTK::CompositeFunction::s_internalDefaultDynamicAxisName)
            inputVarDynamicAxes = { CNTK::Axis::DefaultDynamicAxis(), CNTK::Axis::DefaultBatchAxis() };
        else if (internalDynamicAxisName == CNTK::CompositeFunction::s_internalNoSequenceAxisName)
            inputVarDynamicAxes = { CNTK::Axis::DefaultBatchAxis() };
        else
            inputVarDynamicAxes = { CNTK::Axis(internalDynamicAxisName), CNTK::Axis::DefaultBatchAxis() };

        return inputVarDynamicAxes;
    }
}
