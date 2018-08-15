//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "PrimitiveFunction.h"
#include "PrimitiveFunctionAttribute.h"
#include "ComputationNode.h"
#include "ReshapingNodes.h"
#include "EvaluationNodes.h"
#include "TrainingNodes.h"
#include "LinearAlgebraNodes.h"
#include "InputAndParamNodes.h"
#include "NonlinearityNodes.h"
#include "RecurrentNodes.h"
#include "Serialization.h"
#include "RNNNodes.h"
#include "BlockFunction.h"
#include "CompositeFunction.h"
#include "SpecialPurposeNodes.h"
#include "ConvolveGeometry.h"
#include "ConvolutionalNodes.h"
#include "Variable.h"
#include "UserFunctionFactory.h"

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    /*static*/ DataType PrimitiveFunction::GetOutputDataType(PrimitiveOpType op, std::vector<Variable>& inputs, bool inferDimensions)
    {

        // We use the first non-constant input operand's DataType as the output DataType
        // In case there are no non-constant known DataTypes, we just pick the first known operand DataType
        // Also, all the known DataTypes of operands should match except for constants where coercion is allowed
        DataType firstKnownInputDataType = DataType::Unknown;
        DataType outputDataType = DataType::Unknown;
        size_t i = 0;
        while (i < inputs.size())
        {
            auto input = inputs[i++];
            auto inputDataType = input.GetDataType();
            if (inputDataType != DataType::Unknown)
            {
                if (firstKnownInputDataType == DataType::Unknown)
                    firstKnownInputDataType = inputDataType;

                if (outputDataType == DataType::Unknown)
                {
                    if (!input.IsConstant())
                        outputDataType = inputDataType;
                }
                else
                {
                    // batch normalization on FP16 requires 32-bit scale/bias/mean/variance, so specialize that case
                    bool batchNormSpecialCase =
                        (op == PrimitiveOpType::BatchNormalization) &&
                        (outputDataType == DataType::Float16) &&
                        (inputDataType == DataType::Float);

                    // The DataType of all operands should match except for Constants where we allow coercion
                    if ((inputDataType != DataType::Unknown) && (inputDataType != outputDataType) && !input.IsConstant() && !batchNormSpecialCase)
                        InvalidArgument("Primitive op '%S' passed operands '%S' with different DataTypes '%s' and '%s'.",
                                        PrimitiveOpTypeName(op).c_str(), NamedListString(inputs).c_str(), DataTypeName(outputDataType), DataTypeName(inputDataType));
                }
            }
        }

        if (outputDataType == DataType::Unknown)
            outputDataType = firstKnownInputDataType;

        // Propagate the data type to any input Parameters/Constants with unknown data type
        if (inferDimensions && (outputDataType != DataType::Unknown))
        {
            for (auto& input : inputs)
            {
                if ((input.GetDataType() == DataType::Unknown) && (input.IsConstant() || input.IsParameter()))
                {
                    // batch normalization on FP16 requires 32-bit scale/bias/mean/variance, so specialize that case
                    if ((op == PrimitiveOpType::BatchNormalization) &&
                        (outputDataType == DataType::Float16))
                    {
                        input.m_dataFields->m_dataType = DataType::Float;
                    }
                    else
                    {
                        input.m_dataFields->m_dataType = outputDataType;
                    }
                }
            }
        }

        return outputDataType;
    }

    /*static*/ std::vector<Axis> PrimitiveFunction::GetOutputDynamicAxes(PrimitiveOpType op, std::vector<Variable>& inputs, PrimitiveFunction* owner, Dictionary& functionConfig)
    {
        auto reduceAxis = [](Axis reductionAxis, Variable input, std::vector<Axis>& outputDynamicAxes)
        {
            if (!reductionAxis.IsDynamicAxis()) return;
            reductionAxis = NormalizeAxis(reductionAxis, input);
            for (auto inputDynamicAxis : input.DynamicAxes())
            {
                if (inputDynamicAxis != reductionAxis)
                    outputDynamicAxes.push_back(inputDynamicAxis);
            }
        };
        auto anyOfAxesInReduction = [&functionConfig](std::function<bool (const Axis&)> axis_predicate)
        {
            if (functionConfig.Contains(PrimitiveFunctionAttribute::AttributeNameAxis))
                return axis_predicate(functionConfig[PrimitiveFunctionAttribute::AttributeNameAxis].Value<Axis>());
            else
            {
                auto &reductionAxes = functionConfig[PrimitiveFunctionAttribute::AttributeNameAxisVec].Value<std::vector<DictionaryValue>>();
                return std::any_of(reductionAxes.begin(),
                    reductionAxes.end(),
                    [&axis_predicate](const CNTK::DictionaryValue& axis) {return axis_predicate(axis.Value<Axis>()); });

            }
            return false;
        };

        // We currently require that the inputs' dynamic axes, if any, match
        std::vector<Axis> outputDynamicAxes;
        if ((op == PrimitiveOpType::SumAll) ||
            (op == PrimitiveOpType::ReduceElements &&  anyOfAxesInReduction([](const Axis& axis) { return axis == Axis::AllAxes(); })) ||
            (op == PrimitiveOpType::SquaredError) ||
            (op == PrimitiveOpType::CrossEntropyWithSoftmax) ||
            (op == PrimitiveOpType::LatticeSequenceWithSoftmax) ||
            (op == PrimitiveOpType::EditDistanceError) ||
            (op == PrimitiveOpType::ClassificationError) ||
            (op == PrimitiveOpType::ForwardBackward) ||
            (op == PrimitiveOpType::Logistic) ||
            (op == PrimitiveOpType::LambdaRank) ||
            (op == PrimitiveOpType::NDCG) ||
            (op == PrimitiveOpType::RandomDistribution && inputs.empty()) ||
            (op == PrimitiveOpType::UnpackBatch))
        {
            outputDynamicAxes = std::vector<Axis>({});
        }
        else if (op == PrimitiveOpType::ToBatch)
        {
            outputDynamicAxes = std::vector<Axis>({Axis::DefaultBatchAxis()});
        }
        else if (op == PrimitiveOpType::ReduceElements
            && anyOfAxesInReduction([](const Axis& axis) { return axis.IsDynamicAxis(); }) //still need this condition to go to the default dynamic axes setting branch!
            && (inputs[0].DynamicAxes() != Axis::UnknownDynamicAxes())
            )
        {
            if (functionConfig.Contains(PrimitiveFunctionAttribute::AttributeNameAxis))
            {
                auto reductionAxis = functionConfig[PrimitiveFunctionAttribute::AttributeNameAxis].Value<Axis>();
                reduceAxis(reductionAxis, inputs[0], outputDynamicAxes);
            }
            else if (functionConfig.Contains(PrimitiveFunctionAttribute::AttributeNameAxisVec))
            {
                auto &reductionAxes = functionConfig[PrimitiveFunctionAttribute::AttributeNameAxisVec].Value<std::vector<DictionaryValue>>();
                for (const auto& axis : reductionAxes)
                {
                    auto reductionAxis = axis.Value<Axis>();
                    reduceAxis(reductionAxis, inputs[0], outputDynamicAxes);
                }
            }
        }
        else if ((op == PrimitiveOpType::Times) && (functionConfig[PrimitiveFunctionAttribute::AttributeNameInferInputRankToMap].Value<int>() == TimesReduceSequenceAxisWithoutInferredInputRank))
        {
            reduceAxis(Axis::OperandSequenceAxis(), inputs[0], outputDynamicAxes);
        }
        else if (op == PrimitiveOpType::UnpackSequence)
        {
            reduceAxis(Axis::OperandSequenceAxis(), inputs[0], outputDynamicAxes);
        }
        else if (op == PrimitiveOpType::ConvolutionSequenceShape)
        {
            // There is no sequence axis for ConvolutionSequenceShape output, since the output is length of every sequence. 
            reduceAxis(Axis::OperandSequenceAxis(), inputs[1], outputDynamicAxes);
        }
        else if ((op == PrimitiveOpType::Where) || (op == PrimitiveOpType::ToSequence))
        {
            if (functionConfig.Contains(PrimitiveFunctionAttribute::AttributeNameNewDynamicAxes))
                outputDynamicAxes = AsVector<Axis>(functionConfig[PrimitiveFunctionAttribute::AttributeNameNewDynamicAxes].Value<std::vector<DictionaryValue>>());
            else
            {
                if (inputs[0].DynamicAxes() == Axis::UnknownDynamicAxes())
                    outputDynamicAxes = Axis::UnknownDynamicAxes();
                else
                {
                    if ((op == PrimitiveOpType::Where) &&
                        (functionConfig.Contains(PrimitiveFunctionAttribute::AttributeNameNewSequenceAxisLengthScalingFactor) &&
                        functionConfig.Contains(PrimitiveFunctionAttribute::AttributeNameNewSequenceAxisLengthAdditiveFactor)))
                    {
                        size_t newSequenceAxisLengthScalingFactor = functionConfig[PrimitiveFunctionAttribute::AttributeNameNewSequenceAxisLengthScalingFactor].Value<size_t>();
                        int newSequenceAxisLengthAdditiveFactor = functionConfig[PrimitiveFunctionAttribute::AttributeNameNewSequenceAxisLengthAdditiveFactor].Value<int>();

                        auto derivedDynamicAxes = GetDerivedDynamicAxes(inputs[0].DynamicAxes()[0], newSequenceAxisLengthScalingFactor, newSequenceAxisLengthAdditiveFactor);
                        std::copy(derivedDynamicAxes.begin(), derivedDynamicAxes.end(), std::back_inserter(outputDynamicAxes));
                    }
                    else
                    {
                        std::wstring axisNamePrefix = (op == PrimitiveOpType::Where) ? L"whereNodeDynamicAxis_conditionVar_" : functionConfig[PrimitiveFunctionAttribute::AttributeNameSequenceAxisNamePrefix].Value<std::wstring>();
                        auto sequenceAxis = Utils::NewDynamicAxisDerivedFromOperand(axisNamePrefix, inputs[0]);
                        outputDynamicAxes.push_back(sequenceAxis);
                    }

                    auto inputDynamicAxes = inputs[0].DynamicAxes();
                    if (op == PrimitiveOpType::Where)
                        outputDynamicAxes.insert(outputDynamicAxes.end(), ++inputDynamicAxes.begin(), inputDynamicAxes.end());
                    else
                        outputDynamicAxes.insert(outputDynamicAxes.end(), inputDynamicAxes.begin(), inputDynamicAxes.end());

                    functionConfig[PrimitiveFunctionAttribute::AttributeNameNewDynamicAxes] = AsDictionaryValueVector(outputDynamicAxes);
                }
            }
        }
        else if (op == PrimitiveOpType::ScatterPacked)
            outputDynamicAxes = inputs[2].DynamicAxes();
        else if ((op == PrimitiveOpType::PackedIndex) || (op == PrimitiveOpType::GatherPacked))
            outputDynamicAxes = inputs[1].DynamicAxes();
        else if ((op == PrimitiveOpType::ReconcileDynamicAxis) || (op == PrimitiveOpType::ToSequenceLike))
            outputDynamicAxes = inputs[1].DynamicAxes();
        else if (op == PrimitiveOpType::PastValue || op == PrimitiveOpType::FutureValue)
            outputDynamicAxes = inputs[0].DynamicAxes(); // second arg (initial state) may have different dynamic axis
        else
        {
            auto allInputDynamicAxesEmpty = std::find_if(inputs.begin(), inputs.end(), [](const Variable& input) { return !input.DynamicAxes().empty(); }) == inputs.end();
            if (!allInputDynamicAxesEmpty)
            {
                outputDynamicAxes = Axis::UnknownDynamicAxes();
                for (auto inputVar : inputs)
                {
                    auto currentInputDynamicAxes = inputVar.DynamicAxes();
                    if (!currentInputDynamicAxes.empty() && (currentInputDynamicAxes != Axis::UnknownDynamicAxes()))
                    {
                        if (outputDynamicAxes == Axis::UnknownDynamicAxes())
                            outputDynamicAxes = currentInputDynamicAxes;
                        else
                        {
                            if (currentInputDynamicAxes != outputDynamicAxes)
                                LogicError("Operation '%S': Operand '%S' has dynamic axes, that do not match the dynamic axes '%S' of the other operands.",
                                            PrimitiveOpTypeName(op).c_str(), inputVar.AsString().c_str(), DynamicAxesAsString(outputDynamicAxes, Internal::IsReversingTensorShapesInErrorMessagesEnabled()).c_str());
                        }
                    }
                }
            }
        }

        return outputDynamicAxes;
    }

    void PrimitiveFunction::InferOutputs(std::vector<Variable>& outputs)
    {
        if (m_op == PrimitiveOpType::Combine)
            outputs.assign(m_inputs.begin(), m_inputs.end());
        else if (m_op == PrimitiveOpType::NoOp)
            outputs.push_back(OutputVariable(m_inputs[0].Shape(), m_inputs[0].GetDataType(), m_inputs[0].DynamicAxes(), m_inputs[0].NeedsGradient(), Name()));
        else if (m_op == PrimitiveOpType::CustomProxyOp)
        {
            // Set the output data type and shape using attributes.
            DataType outputDataType = DataType::Unknown;
            if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameNewDataType))
            {
                outputDataType = static_cast<DataType>(m_attributes[PrimitiveFunctionAttribute::AttributeNameNewDataType].Value<int>());
            }
            else
            {
                InvalidArgument("Output type must be specified for CustomProxyOp.");
            }
            NDShape outputShape = NDShape::Unknown();
            if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameOutputShape))
            {
                outputShape = m_attributes[PrimitiveFunctionAttribute::AttributeNameOutputShape].Value<NDShape>();
            }
            else
            {
                InvalidArgument("Output shape must be specified for CustomProxyOp.");
            }

            std::vector<Axis> outputDynamicAxes = GetOutputDynamicAxes(m_op, m_inputs, this, m_attributes);
            outputs.push_back(OutputVariable(outputShape, outputDataType, outputDynamicAxes, false, Name().empty() ? L"" : Name()));
        }
        else
        {
            DataType outputDataType = GetOutputDataType(m_op, m_inputs, true);

            if (m_op == PrimitiveOpType::Cast)
                outputDataType = static_cast<DataType>(m_attributes[PrimitiveFunctionAttribute::AttributeNameNewDataType].Value<int>());

            std::vector<Axis> outputDynamicAxes = GetOutputDynamicAxes(m_op, m_inputs, this, m_attributes);
            bool needsGradient = std::any_of(m_inputs.begin(), m_inputs.end(), [](const Variable& input) { return input.NeedsGradient(); });

            NDShape outputShape = NDShape::Unknown();
            bool allInputShapesUnknown = (std::find_if(m_inputs.begin(), m_inputs.end(), [](const Variable& input) { return !input.Shape().IsUnknown(); }) == m_inputs.end());
            bool anyInputShapesUnknown = (std::find_if(m_inputs.begin(), m_inputs.end(), [](const Variable& input) { return input.Shape().IsUnknown(); }) != m_inputs.end());
            if (!anyInputShapesUnknown || (!allInputShapesUnknown && (outputDynamicAxes != Axis::UnknownDynamicAxes())))
            {
                switch (m_op)
                {
                    // Elementwise operators' shapes are a zip of inputs and can be determined even if some of the input shapes are unknown
                case PrimitiveOpType::Plus:
                case PrimitiveOpType::LogPlus:
                case PrimitiveOpType::Pow:
                case PrimitiveOpType::Minus:
                case PrimitiveOpType::ElementTimes:
                case PrimitiveOpType::Equal:
                case PrimitiveOpType::NotEqual:
                case PrimitiveOpType::Less:
                case PrimitiveOpType::LessEqual:
                case PrimitiveOpType::Greater:
                case PrimitiveOpType::GreaterEqual:
                case PrimitiveOpType::PastValue:
                case PrimitiveOpType::FutureValue:
                {
                    assert(m_inputs.size() == 2);
                    if ((m_op == PrimitiveOpType::PastValue) || (m_op == PrimitiveOpType::FutureValue))
                    {
                        Variable inputOperandVar = m_inputs[0];
                        Variable initialStateVar = m_inputs[1];

                        // TODO: We currently only support input operand with 1 dynamic axis for PastValue/FutureValue
                        if ((inputOperandVar.DynamicAxes() != Axis::UnknownDynamicAxes()) && (inputOperandVar.DynamicAxes().size() != 2))
                            LogicError("PastValue/FutureValue Function '%S': Input operand '%S' with #dynamic axes != 2 (1 sequence axis and 1 batch axis) is not supported.", AsString().c_str(), inputOperandVar.AsString().c_str());
                    }
                    // PastValue and FutureValue are used in RNN loops
                    // scalar broadcasting is disabled to make sure input/output shape matches exactly
                    bool isDelayOp = (m_op == PrimitiveOpType::PastValue || m_op == PrimitiveOpType::FutureValue);
                    outputShape = BinaryElementwiseOpOutputShape(m_op, m_inputs[0], m_inputs[1], /*inferInputDimensions =*/ true, /*allowScalarBroadcast*/ !isDelayOp);
                    break;
                }
                case PrimitiveOpType::Clip:
                    assert(m_inputs.size() == 3);
                    outputShape = NaryElementwiseOpOutputShape(m_op, m_inputs, /*inferInputDimensions =*/ true);
                    break;
                case PrimitiveOpType::Select:
                    assert(m_inputs.size() == 3);
                    outputShape = NaryElementwiseOpOutputShape(m_op, m_inputs, /*inferInputDimensions =*/ true);
                    break;
                default:
                    // For all other operations, shapes of all inputs must be known to determine the output shape
                    if (!anyInputShapesUnknown)
                    {
                        switch (m_op)
                        {
                        case PrimitiveOpType::RandomDistribution:
                        {
                            assert(m_inputs.size() == 0 || m_inputs.size() == 1);
                            if (m_inputs.size() == 1)
                                outputShape = UnaryElementwiseOpOutputShape(m_inputs[0].Shape());
                            else
                            {
                                outputShape = m_attributes[PrimitiveFunctionAttribute::AttributeNameNewShape].Value<NDShape>();
                                if (outputShape.HasUnboundDimension()) //review: is unbound right or should this be Free or Inferred?
                                    InvalidArgument("RandomDistribution: Output shape '%ls' must not have an unbound dimension.", outputShape.AsString().c_str());
                                auto dataType = static_cast<DataType>(m_attributes[PrimitiveFunctionAttribute::AttributeNameNewDataType].Value<int>());
                                if (dataType != DataType::Float && dataType != DataType::Double)
                                    InvalidArgument("RandomDistribution: data type must be one of float, double.");
                                outputDataType = dataType;
                            }
                            break;
                        }
                        case PrimitiveOpType::Negate:
                        case PrimitiveOpType::Sigmoid:
                        case PrimitiveOpType::Tanh:
                        case PrimitiveOpType::Atanh:
                        case PrimitiveOpType::ReLU:
                        case PrimitiveOpType::Exp:
                        case PrimitiveOpType::Log:
                        case PrimitiveOpType::Sqrt:
                        case PrimitiveOpType::Floor:
                        case PrimitiveOpType::Abs:
                        case PrimitiveOpType::Reciprocal:
                        case PrimitiveOpType::Softmax:
                        case PrimitiveOpType::Hardmax:
                        case PrimitiveOpType::Dropout:
                        case PrimitiveOpType::LogSoftmax:
                        case PrimitiveOpType::Asin:
                        case PrimitiveOpType::Acos:
                        case PrimitiveOpType::Atan:
                        case PrimitiveOpType::Sin:
                        case PrimitiveOpType::Cos:
                        case PrimitiveOpType::Tan:
                        case PrimitiveOpType::Cosh:
                        case PrimitiveOpType::Asinh:
                        case PrimitiveOpType::Sinh:
                        case PrimitiveOpType::Pass:
                        case PrimitiveOpType::LabelsToGraph:
                        case PrimitiveOpType::StopGradient:
                        case PrimitiveOpType::ELU:
                        case PrimitiveOpType::StableSigmoid:
                        case PrimitiveOpType::ConstantOp:
                        case PrimitiveOpType::Cast:
                        case PrimitiveOpType::StraightThrough:
                            assert(m_inputs.size() == 1);
                            outputShape = UnaryElementwiseOpOutputShape(m_inputs[0].Shape());
                            break;
                        case PrimitiveOpType::EyeLikeOp:
                        {
                            assert(m_inputs.size() == 1);
                            const auto& dynAxes = m_inputs[0].DynamicAxes();
                            if (dynAxes.size() + m_inputs[0].Shape().Rank() != 2)
                                InvalidArgument("EyeLike: Operand '%S' must have exactly 2 axes including dynamic and static axes.",
                                    m_inputs[0].AsString().c_str());
                            if (any_of(dynAxes.begin(), dynAxes.end(), [](const Axis& axis) {return axis.IsSequenceAxis(); }))
                                InvalidArgument("EyeLike: Operand '%S' can not have sequence axis.",
                                    m_inputs[0].AsString().c_str());

                            outputShape = UnaryElementwiseOpOutputShape(m_inputs[0].Shape());
                            break;
                        }
                        case PrimitiveOpType::Where:
                            assert(m_inputs.size() == 1);
                            outputShape = NDShape{}; // scalar
                            break;
                        case PrimitiveOpType::UnpackSequence:
                        {
                            assert(m_inputs.size() == 1);
                            if ((m_inputs[0].DynamicAxes() != Axis::UnknownDynamicAxes()) && (m_inputs[0].DynamicAxes().size() < 2))
                                InvalidArgument("UnpackSequence: Operand '%S' must have at least 2 dynamic axes.", m_inputs[0].AsString().c_str());

                            outputShape = m_inputs[0].Shape().AppendShape({ NDShape::FreeDimension });
                            break;
                        }
                        case PrimitiveOpType::ToSequence:
                        case PrimitiveOpType::ToSequenceLike:
                        {
                            assert(((m_op == PrimitiveOpType::ToSequence) && (m_inputs.size() == 1)) || (m_inputs.size() == 2));
                            if (m_inputs[0].DynamicAxes().empty())
                                InvalidArgument("Function '%S': Operand '%S' must have dynamic axes.", AsString().c_str(), m_inputs[0].AsString().c_str());

                            if ((m_inputs[0].DynamicAxes() != Axis::UnknownDynamicAxes()) && ((m_inputs[0].DynamicAxes().size() != 1) || (m_inputs[0].DynamicAxes()[0] != Axis::DefaultBatchAxis())))
                                InvalidArgument("Function '%S': Input operand '%S' with #dynamic axes != 1 (batch axis) is not supported.", AsString().c_str(), m_inputs[0].AsString().c_str());

                            if (m_inputs[0].Shape().Rank() < 1)
                                InvalidArgument("Function '%S': First input operand '%S' must be of rank >= 1.", AsString().c_str(), m_inputs[0].AsString().c_str());

                            if (m_op == PrimitiveOpType::ToSequence)
                            {
                                if ((m_inputs.size() == 2) &&
                                    (m_inputs[0].DynamicAxes() != Axis::UnknownDynamicAxes()) &&
                                    (m_inputs[1].DynamicAxes() != Axis::UnknownDynamicAxes()) &&
                                    (m_inputs[0].DynamicAxes() != m_inputs[1].DynamicAxes()))
                                    InvalidArgument("Function '%S': First input operand '%S' dynamic axes '%S' do not match second input operand '%S' dynamic axes '%S' .",
                                                    AsString().c_str(), m_inputs[0].AsString().c_str(), NamedListString(m_inputs[0].DynamicAxes()).c_str(), m_inputs[1].AsString().c_str(), NamedListString(m_inputs[1].DynamicAxes()).c_str());

                                if ((m_inputs.size() == 2) && (m_inputs[1].Shape().TotalSize() != 1))
                                    InvalidArgument("Function '%S': Second input operand '%S' must be a scalar.", AsString().c_str(), m_inputs[1].AsString().c_str());
                            }
                            else
                            {
                                if ((m_inputs[1].DynamicAxes() != Axis::UnknownDynamicAxes()) && (m_inputs[1].DynamicAxes().size() < 2))
                                    InvalidArgument("Function '%S': Operand(1) '%S' must be a sequence (have at least 2 dynamic axes).", AsString().c_str(), m_inputs[1].AsString().c_str());
                            }

                            auto operandShape = m_inputs[0].Shape();
                            outputShape = operandShape.SubShape(0, operandShape.Rank() - 1);
                            break;
                        }
                        case PrimitiveOpType::PackedIndex:
                            assert(m_inputs.size() == 2);
                            outputShape = UnaryElementwiseOpOutputShape(m_inputs[1].Shape());
                            break;
                        case PrimitiveOpType::Assign:
                            assert(m_inputs.size() == 2);
                            if (!m_inputs[0].DynamicAxes().empty() || !m_inputs[1].DynamicAxes().empty())
                                InvalidArgument("AssignNode: None of the operands '%S' can have dynamic axes.", NamedListString(m_inputs).c_str());
                            if (!(m_inputs[0].IsConstant() || m_inputs[0].IsParameter()))
                                InvalidArgument("AssignNode: Ref operand must be constant or parameter only.");
                            //delay the check for free dimension
                            if (m_inputs[0].Shape() != m_inputs[1].Shape() &&
                                !m_inputs[0].Shape().HasUnboundDimension() &&
                                !m_inputs[1].Shape().HasUnboundDimension())
                            {
                                InvalidArgument("AssignNode: All inputs should have same sample layout.");
                            }

                            outputShape = UnaryElementwiseOpOutputShape(m_inputs[1].Shape());
                            break;
                        case PrimitiveOpType::Pad:
                        {
                            assert(m_inputs.size() == 1);
                            auto head = AsVector<size_t>(m_attributes[PrimitiveFunctionAttribute::AttributeNamePaddingHead].Value<std::vector<DictionaryValue>>());
                            auto foot = AsVector<size_t>(m_attributes[PrimitiveFunctionAttribute::AttributeNamePaddingFoot].Value<std::vector<DictionaryValue>>());
                            PaddingMode mode = (PaddingMode)m_attributes[PrimitiveFunctionAttribute::AttributeNamePaddingMode].Value<size_t>();
                            auto inputDims = m_inputs[0].Shape().Dimensions();
                            if (head.size() != inputDims.size() || head.size() != foot.size())
                                LogicError("Pad: the length of head and foot does not match input operand's dimension.");

                            if (!m_inputs[0].Shape().HasUnboundDimension())
                            {
                                if (mode == PaddingMode::REFLECTPAD)
                                {
                                    for (int i = 0; i < inputDims.size(); i++)
                                        if (head[i] > inputDims[i] - 1 || foot[i] > inputDims[i] - 1)
                                            LogicError("Pad: with REFLECTPAD mode, the head and foot length must be no greater than input dimension - 1.");
                                }
                                else if (mode == PaddingMode::SYMMETRICPAD)
                                {
                                    for (int i = 0; i < inputDims.size(); i++)
                                        if (head[i] > inputDims[i] || foot[i] > inputDims[i])
                                            LogicError("Pad: with SYMMETRICPAD mode, the head and foot length must be no greater than input dimension.");
                                }
                            }

                            for (int i = 0; i < inputDims.size(); i++)
                                if (inputDims[i] != NDShape::FreeDimension && inputDims[i] != NDShape::InferredDimension)
                                    inputDims[i] += head[i] + foot[i];
                            outputShape = NDShape(inputDims);
                            break;
                        }
                            
                        case PrimitiveOpType::ScatterPacked:
                        {
                            assert(m_inputs.size() == 3);
                            if (m_inputs[0].DynamicAxes().empty() || m_inputs[1].DynamicAxes().empty() || m_inputs[2].DynamicAxes().empty())
                                InvalidArgument("ScatterPacked: All operands '%S' must have dynamic axes.", NamedListString(m_inputs).c_str());

                            outputShape = UnaryElementwiseOpOutputShape(m_inputs[0].Shape());
                            break;
                        }
                        case PrimitiveOpType::Squeeze:
                        {
                            assert(m_inputs.size() == 1);
                            outputShape = GetSqueezedShape(m_inputs[0].Shape(), m_attributes);
                            break;
                        }
                        case PrimitiveOpType::TransposeAxes:
                        {
                            assert(m_inputs.size() == 1);

                            if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameAxisVec))
                            {
                                auto perm = AsVector<Axis>(m_attributes[PrimitiveFunctionAttribute::AttributeNameAxisVec].Value<std::vector<DictionaryValue>>());
                                auto shape = m_inputs[0].Shape();
                                for (auto& p : perm)
                                    p = NormalizeStaticAxis(p, shape);
                                outputShape = shape;
                                for (size_t i = 0; i < perm.size(); ++i)
                                    outputShape[i] = shape[perm[i].StaticAxisIndex()];
                            }
                            else
                            {
                                auto axis1 = NormalizeStaticAxis(m_attributes[PrimitiveFunctionAttribute::AttributeNameAxis1].Value<Axis>(), m_inputs[0].Shape());
                                auto axis2 = NormalizeStaticAxis(m_attributes[PrimitiveFunctionAttribute::AttributeNameAxis2].Value<Axis>(), m_inputs[0].Shape());

                                if (!axis1.IsStaticAxis() || !axis2.IsStaticAxis())
                                    LogicError("Function '%S': TransposeAxes operation currently does not support transposing dynamic axes.", AsString().c_str());

                                // We allow to transpose with an axes that exceeds the rank of the input.
                                // The output rank is the max of the input rank, and either of the axes being transposed.
                                auto outputRank = std::max(m_inputs[0].Shape().Rank(), (size_t)(std::max(axis1.StaticAxisIndex(), axis2.StaticAxisIndex()) + 1));
                                outputShape = m_inputs[0].Shape().AppendShape(NDShape(outputRank - m_inputs[0].Shape().Rank(), 1));
                                std::swap(outputShape[axis1.StaticAxisIndex()], outputShape[axis2.StaticAxisIndex()]);
                            }
                            break;
                        }
                        case PrimitiveOpType::Slice:
                        {
                            assert(m_inputs.size() == 1);

                            std::vector<Axis> axis;
                            std::vector<int> beginIndex, endIndex, strides;
                            if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameAxisVec) &&
                                m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameBeginIndexVec) &&
                                m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameEndIndexVec))
                            {
                                auto &axisDictionary = m_attributes[PrimitiveFunctionAttribute::AttributeNameAxisVec].Value<std::vector<DictionaryValue>>();
                                for (auto& value : axisDictionary)
                                    axis.push_back(NormalizeStaticAxis(value.Value<Axis>(), m_inputs[0].Shape()));

                                beginIndex = AsVector<int>(m_attributes[PrimitiveFunctionAttribute::AttributeNameBeginIndexVec].Value<std::vector<DictionaryValue>>());
                                endIndex = AsVector<int>(m_attributes[PrimitiveFunctionAttribute::AttributeNameEndIndexVec].Value<std::vector<DictionaryValue>>());
                                if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameSliceStridesVec))
                                    strides = AsVector<int>(m_attributes[PrimitiveFunctionAttribute::AttributeNameSliceStridesVec].Value<std::vector<DictionaryValue>>());
                                else
                                    strides.resize(axis.size(), 1);
                            }
                            else if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameAxis) &&
                                m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameBeginIndex) &&
                                m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameEndIndex))
                            {
                                axis.push_back(NormalizeStaticAxis(m_attributes[PrimitiveFunctionAttribute::AttributeNameAxis].Value<Axis>(), m_inputs[0].Shape()));
                                beginIndex.push_back(m_attributes[PrimitiveFunctionAttribute::AttributeNameBeginIndex].Value<int>());
                                endIndex.push_back(m_attributes[PrimitiveFunctionAttribute::AttributeNameEndIndex].Value<int>());
                                if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameSliceStrides))
                                    strides.push_back(m_attributes[PrimitiveFunctionAttribute::AttributeNameSliceStrides].Value<int>());
                                else
                                    strides.push_back(1);
                            }
                            else
                            {
                                RuntimeError("Function '%S': Slice operation with inconsistent attributes", AsString().c_str());
                            }

                            auto outputTensorShape = AsTensorShape(m_inputs[0].Shape());
                            for (auto i = 0; i < axis.size(); i++)
                            {
                                auto& ax = axis[i];
                                if (!ax.IsStaticAxis())
                                    LogicError("Function '%S': Built-in Slice operation currently does not support slicing along dynamic axis.", AsString().c_str());
                                VerifyStaticAxis(ax, m_inputs[0].Shape());

                                size_t sliceAxisDim = m_inputs[0].Shape()[ax.StaticAxisIndex()];
                                if (sliceAxisDim == NDShape::FreeDimension && (beginIndex[i] < 0 || endIndex[i] <= 0))
                                {
                                    // not able to calculate real indices. do not narrow either.
                                    // note that endIndex[i] = 0 means to (and include) the last.
                                    // One case for this condition is to export and import, in ONNX format, a CNTK Sequence.Slice op.
                                    // In this case, if batch size is larger than 1 and input data are a zigged array (i.e. sequences of various lengths),
                                    // model evaludation will not march the original CNTK model.
                                }
                                else
                                {
                                    int realBeginIndex = (beginIndex[i] >= 0) ? beginIndex[i] : beginIndex[i] + sliceAxisDim;
                                    int realEndIndex = (endIndex[i] > 0) ? endIndex[i] : endIndex[i] + sliceAxisDim;
                                    if ((sliceAxisDim < realEndIndex) || (realEndIndex < realBeginIndex) || (realBeginIndex < 0))
                                        RuntimeError("Function '%S': Slice operation index range [%d,%d), interpreted as [%d,%d), is invalid for input '%S' shape '%S'.",
                                            AsString().c_str(),
                                            beginIndex[i],
                                            endIndex[i],
                                            realBeginIndex,
                                            realEndIndex,
                                            m_inputs[0].AsString().c_str(),
                                            m_inputs[0].Shape().AsString().c_str());
                                    // propagate as much as we can
                                    // Note: If the sliceAxisDim is a free dimension and the slice size is relative to the sliceAxisDim then the
                                    // corresponding outputDim is also a free dimension
                                    if ((((sliceAxisDim != NDShape::FreeDimension) && (sliceAxisDim != NDShape::InferredDimension)) || (((beginIndex[i] >= 0) && (endIndex[i] > 0)) || ((beginIndex[i] < 0) && (endIndex[i] <= 0)))) &&
                                        ((ax.StaticAxisIndex() < (int)outputTensorShape.GetRank()) && (0 <= realBeginIndex) && (realBeginIndex <= realEndIndex) && (realEndIndex <= sliceAxisDim)))
                                    {
                                        outputTensorShape.NarrowTo(ax.StaticAxisIndex(), realBeginIndex, realEndIndex, strides[i]);
                                    }
                                }
                            }
                            outputShape = AsNDShape(outputTensorShape, /*allowNonFlattenableTensorShapes = */ true);
                            break;
                        }
                        case PrimitiveOpType::Reshape:
                        {
                            auto replacementShape = m_attributes[PrimitiveFunctionAttribute::AttributeNameNewShape].Value<NDShape>();

                            auto beginAxis = Axis(0);
                            auto endAxis = Axis((int)m_inputs[0].Shape().Rank());
                            if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameBeginAxis))
                                beginAxis = NormalizeStaticAxis(m_attributes[PrimitiveFunctionAttribute::AttributeNameBeginAxis].Value<Axis>(), m_inputs[0].Shape());

                            if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameEndAxis))
                                endAxis = NormalizeStaticAxis(m_attributes[PrimitiveFunctionAttribute::AttributeNameEndAxis].Value<Axis>(), m_inputs[0].Shape());

                            outputShape = ReshapeOutputShape(m_inputs[0].Shape(), replacementShape, beginAxis, endAxis, /*inferDimensions =*/ false);
                            break;
                        }
                        case PrimitiveOpType::ROIPooling:
                        {
                            assert(m_inputs.size() == 2);
                            auto convMapShape = m_inputs[0].Shape();
                            auto roisShape = m_inputs[1].Shape();
                            auto roiOutputShape = m_attributes[PrimitiveFunctionAttribute::AttributeNameROIOutputShape].Value<NDShape>();

                            auto outW = roiOutputShape[0];
                            auto outH = roiOutputShape[1];
                            auto numChannels = convMapShape[2];
                            auto roisPerImage = roisShape[1];

                            if (roiOutputShape.Rank() != 2)
                                InvalidArgument("ROIPoolingNode: ROI shape '%S' must have rank 2 ([W x H]).", roiOutputShape.AsString().c_str());

                            if (!convMapShape.HasUnboundDimension())
                            {
                                if (convMapShape[0] < outW || convMapShape[1] < outH)
                                    InvalidArgument("ROIPoolingNode: input Width (%d) must be >= ROI window Width (%d) and input Height (%d) must be >= ROI window Height (%d).",
                                    (int)convMapShape[0], (int)outW, (int)convMapShape[1], (int)outH);

                                if (convMapShape[2] < 1)
                                    InvalidArgument("ROIPoolingNode: input '%S' must have at least one channel ([W x H x C]).", m_inputs[0].AsString().c_str());
                            }

                            if (roisShape[0] != 4)
                                InvalidArgument("ROIPoolingNode: ROI shape '%S' must be of the form: [4 x roisPerImage].", roisShape.AsString().c_str());

                            if (roisPerImage < 1)
                                InvalidArgument("ROIPoolingNode: ROI shape '%S' must contain at least one ROI ([4 x roisPerImage]).", roisShape.AsString().c_str());

                            outputShape = { outW, outH, numChannels, roisPerImage };
                            break;
                        }
                        case PrimitiveOpType::Pooling:
                        {
                            assert(m_inputs.size() == 1);
                            auto poolingWindowsShape = m_attributes[PrimitiveFunctionAttribute::AttributeNamePoolingWindowShape].Value<NDShape>();
                            auto strides = m_attributes[PrimitiveFunctionAttribute::AttributeNameStrides].Value<NDShape>();
                            auto lowerPad = m_attributes[PrimitiveFunctionAttribute::AttributeNameLowerPad].Value<NDShape>();
                            auto upperPad = m_attributes[PrimitiveFunctionAttribute::AttributeNameUpperPad].Value<NDShape>();
                            auto autoPadding = AsVector<bool>(m_attributes[PrimitiveFunctionAttribute::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
                            bool ceilOutDim = false;
                            if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameCeilOutDim))
                                ceilOutDim = m_attributes[PrimitiveFunctionAttribute::AttributeNameCeilOutDim].Value<bool>();
                            NDShape outputMapCount = { 1 };
                            std::vector<bool> sharing = { true };
                            auto inputShape = m_inputs[0].Shape();

                            // In case of pooling if the kernel shape is unknown, then treat it as global pooling.
                            if (poolingWindowsShape.IsUnknown() && !inputShape.SubShape(0, inputShape.Rank() - 1).HasUnboundDimension())
                            {
                                if ((std::find(autoPadding.begin(), autoPadding.end(), true) != autoPadding.end()) || (lowerPad.TotalSize() > 0) || (upperPad.TotalSize() > 0))
                                    RuntimeError("Padding isn't allowed for Unknown pooling window shape!");

                                poolingWindowsShape = inputShape.SubShape(0, inputShape.Rank() - 1);
                                m_attributes[PrimitiveFunctionAttribute::AttributeNamePoolingWindowShape] = poolingWindowsShape;
                            }

                            NDShape dilation = NDShape({ 1 });
                            outputShape = ConvolutionOpOutputShape(m_op, inputShape, poolingWindowsShape, outputMapCount, strides, sharing, autoPadding, lowerPad, upperPad, false, true, dilation, convolutionOpDefaultValueForGroups, ceilOutDim);
                            break;
                        }
                        case PrimitiveOpType::Unpooling:
                        {
                            assert(m_inputs.size() == 2);

                            auto inputShape = m_inputs[0].Shape();

                            outputShape = m_inputs[1].Shape();
                            PoolingType unpoolingType = (PoolingType)(m_attributes[PrimitiveFunctionAttribute::AttributeNamePoolingType].Value<size_t>());
                            if (unpoolingType != PoolingType::Max)
                                LogicError("Function '%S': Currently only max unpooling is supported.", AsString().c_str());

                            // Finding the shape of an unpooling operation from the input to be unpooled alone is ambiguous
                            // For example a 4x4 input with a 5x5 kernel a stride of 2x2
                            // and padding could have resulted from pooling a 7x7 or 8x8 image
                            // Therefore what needs to happen here is to check whether the
                            // outputShape can be pooled into the inputShape using the specified attributes
                            auto unpoolingWindowShape = m_attributes[PrimitiveFunctionAttribute::AttributeNameUnpoolingWindowShape].Value<NDShape>();
                            auto strides = m_attributes[PrimitiveFunctionAttribute::AttributeNameStrides].Value<NDShape>();
                            auto lowerPad = m_attributes[PrimitiveFunctionAttribute::AttributeNameLowerPad].Value<NDShape>();
                            auto upperPad = m_attributes[PrimitiveFunctionAttribute::AttributeNameUpperPad].Value<NDShape>();
                            auto autoPadding = AsVector<bool>(m_attributes[PrimitiveFunctionAttribute::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
                            NDShape inputMapCount = { 1 };
                            std::vector<bool> sharing = { true };
                            NDShape dilation = { 1 };

                            NDShape inferredInputShape = ConvolutionOpOutputShape(PrimitiveOpType::Pooling, outputShape, unpoolingWindowShape, inputMapCount, strides, sharing, autoPadding, lowerPad, upperPad, false, true, dilation, convolutionOpDefaultValueForGroups);
                            if (inferredInputShape != inputShape)
                                RuntimeError("Unpooling: The shape '%S' of the unpooling operand '%S' is different than the shape '%S from pooling the input argument '%S' using the provided options.",
                                             inputShape.AsString().c_str(), m_inputs[0].AsString().c_str(), inferredInputShape.AsString().c_str(), m_inputs[1].AsString().c_str());

                            break;
                        }
                        case PrimitiveOpType::SumAll:
                            assert(m_inputs.size() == 1);
                            outputShape = {};
                            break;
                        case PrimitiveOpType::OneHot:
                        {
                            assert(m_inputs.size() == 1);
                            auto num_class = m_attributes[PrimitiveFunctionAttribute::AttributeNameNumClass].Value<size_t>();

                            auto inputShape = m_inputs[0].Shape();
                            auto fakeShape = inputShape.AppendShape({num_class});
                            auto axis = NormalizeStaticAxis(m_attributes[PrimitiveFunctionAttribute::AttributeNameOneHotAxis].Value<Axis>(), fakeShape);
                            if (!axis.IsStaticAxis())
                                LogicError("Function '%S': one hot operation currently does not support on dynamic axis", AsString().c_str());

                            size_t len = inputShape.Rank();
                            int axisIndex = axis.StaticAxisIndex();

                            outputShape = {};
                            if (axisIndex > 0)
                                outputShape = outputShape.AppendShape(inputShape.SubShape(0, axisIndex));
                            outputShape = outputShape.AppendShape({num_class});
                            if (axisIndex < len)
                                outputShape = outputShape.AppendShape(inputShape.SubShape(axisIndex, len));
                            break;
                        }
                        case PrimitiveOpType::Gather:
                        {
                            assert(m_inputs.size() == 2);
                            auto inputShape1 = m_inputs[0].Shape();
                            auto inputShape2 = m_inputs[1].Shape();
                            auto inputDim2 = inputShape2.Dimensions();
                            inputDim2.pop_back();
                            outputShape = NDShape(inputDim2);
                            outputShape = outputShape.AppendShape(inputShape1);
                            break;
                        }
                        case PrimitiveOpType::ToBatch:
                        {
                            assert(m_inputs.size() == 1);
                            auto inputShape = m_inputs[0].Shape();
                            auto inputDims = inputShape.Dimensions();
                            if (inputDims.size() == 0)
                                LogicError("Function '%S': Input can't be scalar", AsString().c_str());
                            inputDims.pop_back();
                            outputShape = NDShape(inputDims);
                            break;
                        }
                        case PrimitiveOpType::UnpackBatch:
                        {
                            assert(m_inputs.size() == 1);
                            auto inputShape = m_inputs[0].Shape();
                            outputShape = NDShape(inputShape.Dimensions());
                            outputShape = outputShape.AppendShape({NDShape::FreeDimension});
                            break;
                        }
                        case PrimitiveOpType::Times:
                        {
                            assert(m_inputs.size() == 2);
                            auto outputRank = m_attributes[PrimitiveFunctionAttribute::AttributeNameOutputRank].Value<size_t>();
                            auto inferInputRankToMap = m_attributes[PrimitiveFunctionAttribute::AttributeNameInferInputRankToMap].Value<int>();
                            outputShape = TimesOpOutputShape(m_inputs[0], m_inputs[1], outputRank, inferInputRankToMap, true);
                            break;
                        }
                        case PrimitiveOpType::TransposeTimes:
                        {
                            assert(m_inputs.size() == 2);

                            auto transposeShapeFunc = [](const NDShape& shape) {
                                NDShape transposedShape(std::max<size_t>(2, shape.Rank()), 1);
                                for (size_t i = 0; i < shape.Rank(); ++i)
                                    transposedShape[transposedShape.Rank() - i - 1] = shape[i];

                                return transposedShape;
                            };

                            if (m_inputs[0].Shape().Rank() > 2)
                                LogicError("Function '%S': TransposeTimes operation currently requires the %s operand '%S' to be of rank 1 or 2", AsString().c_str(), Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "right" : "left", m_inputs[0].AsString().c_str());

                            NDShape transposedLeftOperandShape = transposeShapeFunc(m_inputs[0].Shape());
                            Variable dummyLeftOperand = PlaceholderVariable(transposedLeftOperandShape);
                            size_t outputRank = m_attributes[PrimitiveFunctionAttribute::AttributeNameOutputRank].Value<size_t>();
                            outputShape = TimesOpOutputShape(dummyLeftOperand, m_inputs[1], outputRank, -1, true);
                            if (dummyLeftOperand.Shape() != transposedLeftOperandShape)
                                m_inputs[0].m_dataFields->m_shape = transposeShapeFunc(dummyLeftOperand.Shape());

                            break;
                        }
                        case PrimitiveOpType::Convolution:
                        {
                            assert(m_inputs.size() == 2);
                            auto& strides = m_attributes[PrimitiveFunctionAttribute::AttributeNameStrides].Value<NDShape>();
                            NDShape dilation = { 1 };
                            if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameDilation))
                                dilation = m_attributes[PrimitiveFunctionAttribute::AttributeNameDilation].Value<NDShape>();
                            auto& lowerPad = m_attributes[PrimitiveFunctionAttribute::AttributeNameLowerPad].Value<NDShape>();
                            auto& upperPad = m_attributes[PrimitiveFunctionAttribute::AttributeNameUpperPad].Value<NDShape>();
                            NDShape tmpShape = NDShape::Unknown();
                            if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameOutputShape))
                                tmpShape = m_attributes[PrimitiveFunctionAttribute::AttributeNameOutputShape].Value<NDShape>();
                            auto sharing = AsVector<bool>(m_attributes[PrimitiveFunctionAttribute::AttributeNameSharing].Value<std::vector<DictionaryValue>>());
                            auto autoPadding = AsVector<bool>(m_attributes[PrimitiveFunctionAttribute::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
                            bool transpose = m_attributes[PrimitiveFunctionAttribute::AttributeNameTranspose].Value<bool>();                            
                            if (m_inputs[0].Shape().Rank() < m_inputs[1].Shape().Rank())
                                InvalidArgument("The convolution map operand '%S' rank (%d) should be >= rank (%d) of the shape of the input operand '%S'.",
                                                m_inputs[0].AsString().c_str(), (int)m_inputs[0].Shape().Rank(), (int)m_inputs[1].Shape().Rank(), m_inputs[1].AsString().c_str());

                            NDShape outputMapCount, kernelShape;
                            std::tie(outputMapCount, kernelShape) = GetConvolutionOutputMapCountAndKernelShape(m_inputs[0].Shape(), m_inputs[1].Shape(), transpose);
                            auto originalKernelShape = kernelShape;
                            auto groups = PrimitiveFunction::convolutionOpDefaultValueForGroups;
                            if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameGroups))
                                groups = m_attributes[PrimitiveFunctionAttribute::AttributeNameGroups].Value<size_t>();
                            auto inputShape = m_inputs[1].Shape();
                            if (!transpose || tmpShape.IsUnknown() || tmpShape[0] == 0)
                                outputShape = ConvolutionOpOutputShape(m_op, inputShape, kernelShape, outputMapCount, strides, sharing, autoPadding, lowerPad, upperPad, transpose, true, dilation, groups);
                            else
                            {
                                NDShape inferredInputShape = ConvolutionOpOutputShape(m_op, tmpShape, kernelShape, outputMapCount, strides, sharing, autoPadding, lowerPad, upperPad, false, true, dilation, groups);
                                if (inferredInputShape != inputShape)
                                {
                                    RuntimeError("Convolution transpose: The shape '%S' of the convolution transpose operand '%S' is different than the resulting shape '%S' from convolving the "
                                                 "specified output shape '%S' using the provided options.",
                                                 inputShape.AsString().c_str(), m_inputs[1].AsString().c_str(), inferredInputShape.AsString().c_str(), tmpShape.AsString().c_str());
                                }
                                outputShape = tmpShape;
                            }

                            auto kernelRank = kernelShape.Rank();
                            if (originalKernelShape != kernelShape)
                            {
                                for (size_t i2 = 0; i2 < kernelRank; ++i2)
                                    m_inputs[0].m_dataFields->m_shape[i2] = kernelShape[i2];
                            }
                            if (transpose && (m_inputs[0].Shape().Rank() > kernelRank) && (m_inputs[0].Shape()[kernelRank] == NDShape::InferredDimension))
                                m_inputs[0].m_dataFields->m_shape[kernelRank] = outputMapCount[outputMapCount.Rank()-1];

                            m_attributes[PrimitiveFunctionAttribute::AttributeNameSharing] = AsDictionaryValueVector(sharing);
                            m_attributes[PrimitiveFunctionAttribute::AttributeNameAutoPadding] = AsDictionaryValueVector(autoPadding);
                            m_attributes[PrimitiveFunctionAttribute::AttributeNameDilation] = dilation;
                            m_attributes[PrimitiveFunctionAttribute::AttributeNameKernelShape] = kernelShape;
                            break;
                        }
                        case PrimitiveOpType::ConvolutionSequenceShape:
                        {
                            assert(m_inputs.size() == 2);
                            auto& strides = m_attributes[PrimitiveFunctionAttribute::AttributeNameStrides].Value<NDShape>();
                            NDShape dilation = { 1 };
                            if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameDilation))
                                dilation = m_attributes[PrimitiveFunctionAttribute::AttributeNameDilation].Value<NDShape>();
                            auto& lowerPad = m_attributes[PrimitiveFunctionAttribute::AttributeNameLowerPad].Value<NDShape>();
                            auto& upperPad = m_attributes[PrimitiveFunctionAttribute::AttributeNameUpperPad].Value<NDShape>();
                            NDShape tmpShape = NDShape::Unknown();
                            if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameOutputShape))
                                tmpShape = m_attributes[PrimitiveFunctionAttribute::AttributeNameOutputShape].Value<NDShape>();
                            auto sharing = AsVector<bool>(m_attributes[PrimitiveFunctionAttribute::AttributeNameSharing].Value<std::vector<DictionaryValue>>());
                            auto autoPadding = AsVector<bool>(m_attributes[PrimitiveFunctionAttribute::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
                            bool transpose = m_attributes[PrimitiveFunctionAttribute::AttributeNameTranspose].Value<bool>();
                            if (transpose)
                                InvalidArgument("Transpose is currently not supported for sequential convolution. ");
                            // +1 for operand rank, as the real operand should have an additional unpacked sequence axis. 
                            if (m_inputs[0].Shape().Rank() < m_inputs[1].Shape().Rank() + 1)
                                InvalidArgument("The convolution map operand '%S' rank (%d) should be > rank (%d) of the shape of the input operand '%S'.",
                                    m_inputs[0].AsString().c_str(), (int)m_inputs[0].Shape().Rank(), (int)m_inputs[1].Shape().Rank(), m_inputs[1].AsString().c_str());
                            NDShape outputMapCount, kernelShape;
                            auto inputShape = m_inputs[1].Shape();
                            // insert NDShape::FreeDimension to index Rank() - 2.(on the left of channel axis)
                            inputShape = inputShape.SubShape(0, inputShape.Rank()-1).AppendShape({NDShape::FreeDimension}).AppendShape({inputShape[inputShape.Rank() - 1]});
                            std::tie(outputMapCount, kernelShape) = GetConvolutionOutputMapCountAndKernelShape(m_inputs[0].Shape(), inputShape, transpose);
                            auto originalKernelShape = kernelShape;
                            auto groups = PrimitiveFunction::convolutionOpDefaultValueForGroups;
                            if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameGroups))
                                groups = m_attributes[PrimitiveFunctionAttribute::AttributeNameGroups].Value<size_t>();
                            if (tmpShape.IsUnknown() || tmpShape[0] == 0)
                                outputShape = ConvolutionOpOutputShape(m_op, inputShape, kernelShape, outputMapCount, strides, sharing, autoPadding, lowerPad, upperPad, transpose, true, dilation, groups);
                            auto kernelRank = kernelShape.Rank();
                            if (originalKernelShape != kernelShape)
                            {
                                for (size_t i2 = 0; i2 < kernelRank; ++i2)
                                    m_inputs[0].m_dataFields->m_shape[i2] = kernelShape[i2];
                            }
                            if (transpose && (m_inputs[0].Shape().Rank() > kernelRank) && (m_inputs[0].Shape()[kernelRank] == NDShape::InferredDimension))
                                m_inputs[0].m_dataFields->m_shape[kernelRank] = outputMapCount[outputMapCount.Rank() - 1];

                            m_attributes[PrimitiveFunctionAttribute::AttributeNameSharing] = AsDictionaryValueVector(sharing);
                            m_attributes[PrimitiveFunctionAttribute::AttributeNameAutoPadding] = AsDictionaryValueVector(autoPadding);
                            m_attributes[PrimitiveFunctionAttribute::AttributeNameDilation] = dilation;
                            m_attributes[PrimitiveFunctionAttribute::AttributeNameKernelShape] = kernelShape;
                            // output shape is simply {1}: 1D denoting sequence length for each sequence. 
                            outputShape = NDShape({1});
                            break;
                        }
                        case PrimitiveOpType::CrossEntropyWithSoftmax:
                        case PrimitiveOpType::Logistic:
                        case PrimitiveOpType::LambdaRank:
                        case PrimitiveOpType::CosDistance:
                        case PrimitiveOpType::SquaredError:
                        case PrimitiveOpType::EditDistanceError:
                        case PrimitiveOpType::LatticeSequenceWithSoftmax:
                        case PrimitiveOpType::ClassificationError:
                        case PrimitiveOpType::NDCG:
                        {
                            if ((m_op == PrimitiveOpType::ClassificationError) || (m_op == PrimitiveOpType::Logistic))
                                assert(m_inputs.size() >= 2);
                            else if ((m_op == PrimitiveOpType::LambdaRank) || (m_op == PrimitiveOpType::NDCG))
                                assert(m_inputs.size() == 3);
                            else
                                assert(m_inputs.size() == 2);

                            // Validate that the first 2 operands are elementwise compatible and also infer operand shapes as needed
                            BinaryElementwiseOpOutputShape(m_op, m_inputs[0], m_inputs[1], /*inferInputDimensions =*/ true);

                            if (m_op == PrimitiveOpType::ClassificationError)
                            {
                                if ((m_inputs.size() == 3) && !IsConstantScalar(m_inputs[2]))
                                    InvalidArgument("ClassificationError: Input(2) '%S' correponds to topK input and must be a scalar constant.", m_inputs[2].AsString().c_str());
                            }
                            else if (m_op == PrimitiveOpType::Logistic)
                            {
                                if (m_inputs.size() == 3)
                                    BinaryElementwiseOpOutputShape(m_op, m_inputs[0], m_inputs[2], /*inferInputDimensions =*/ true);
                            }

                            outputShape = {};
                            break;
                        }
                        case PrimitiveOpType::ForwardBackward:
                        {
                            assert(m_inputs.size() == 2);
                            if (m_inputs[0].Shape().TotalSize() != m_inputs[1].Shape().TotalSize())
                                InvalidArgument("ForwardBackward: The shapes of operands '%S' and '%S' must have the same total size.", m_inputs[0].AsString().c_str(), m_inputs[1].AsString().c_str());

                            outputShape = {};
                            break;
                        }
                        case PrimitiveOpType::ReduceElements:
                        {
                            assert(m_inputs.size() == 1);
                            bool keepDimensions = true;
                            if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameReductionKeepDimensions))
                                keepDimensions = m_attributes[PrimitiveFunctionAttribute::AttributeNameReductionKeepDimensions].Value<bool>();
                            //Note that we need to normalize the axes inside the attributes here/in InferOutputs
                            if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameAxisVec))
                            {
                                auto &axisDictionary = m_attributes[PrimitiveFunctionAttribute::AttributeNameAxisVec].Value<std::vector<DictionaryValue>>();
                                for (auto& value : axisDictionary) {
                                    auto reductionAxis = NormalizeAxis(value.Value<Axis>(), m_inputs[0]);
                                }
                            }
                            else if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameAxis))
                            {
                                auto reductionAxis = NormalizeAxis(m_attributes[PrimitiveFunctionAttribute::AttributeNameAxis].Value<Axis>(), m_inputs[0]);
                            }
                            else
                            {
                                RuntimeError("Function '%ls': Reduce operation with no '%ls' or  '%ls' attributes",
                                    AsString().c_str(),
                                    PrimitiveFunctionAttribute::AttributeNameAxis.c_str(),
                                    PrimitiveFunctionAttribute::AttributeNameAxisVec.c_str()
                                );
                            }

                            std::vector<Axis> staticAxesToReduce;
                            std::vector<Axis> batchAxesToReduce;
                            std::vector<Axis> dynamicAxesToReduce;
                            bool  isAllAxesReduced;

                            CollectReduceOutputAxesForOutputShape(staticAxesToReduce, batchAxesToReduce, dynamicAxesToReduce, isAllAxesReduced);
                            if (isAllAxesReduced) {
                                outputShape = keepDimensions ? NDShape(m_inputs[0].Shape().Rank(), 1) : NDShape({});
                            }
                            else {
                                //TODO for very far future: Handle reduction on (multiple) batches all in once: batchAxesToReduce
                                //TODO for very far future: Handle reduction on (multiple) sequences all in once: sequenceAxesToReduce
                                if (!staticAxesToReduce.empty())
                                {
                                    std::vector<int> reductionAxesInIndcies(staticAxesToReduce.size());
                                    for (auto i = 0; i < staticAxesToReduce.size(); ++i)
                                    {
                                        reductionAxesInIndcies[i] = staticAxesToReduce[i].StaticAxisIndex();
                                    }
                                    outputShape = ReductionOpOutputShape(m_op, m_inputs[0].Shape(), reductionAxesInIndcies, /*preserveReductionAxes =*/ keepDimensions);
                                }
                                else
                                    outputShape = m_inputs[0].Shape();
                            }
                            break;
                        }
                        case PrimitiveOpType::BatchNormalization:
                        {
                            assert(m_inputs.size() == 6);
                            auto spatial = m_attributes[PrimitiveFunctionAttribute::AttributeNameSpatial].Value<bool>();
                            outputShape = BatchNormalizationOutputShape(m_inputs, spatial, true);
                            break;
                        }
                        case PrimitiveOpType::GatherPacked:
                        {
                            bool sourceHasDynamicAxis = !m_inputs[0].DynamicAxes().empty();

                            // inherit tensor dimension from sourceData, minus the last (column or time) dimension. TODO this needs to become simpler...
                            if (sourceHasDynamicAxis)
                                outputShape = m_inputs[0].Shape();
                            else
                            {
                                if (m_inputs[0].Shape().Rank() > 1)
                                    outputShape = outputShape.SubShape(0, outputShape.Rank() - 1);
                                else
                                    outputShape = {};
                            }

                            break;
                        }
                        case PrimitiveOpType::Splice:
                        {
                            assert(m_inputs.size() >= 2);
                            auto maxInputRank = MaxInputRank(m_inputs);
                            auto spliceAxis = NormalizeStaticAxis(m_attributes[PrimitiveFunctionAttribute::AttributeNameAxis].Value<Axis>(), NDShape(maxInputRank));

                            if (!spliceAxis.IsStaticAxis())
                                LogicError("Function '%S': Splice operation currently does not support splicing along dynamic axis", AsString().c_str());

                            if (spliceAxis.StaticAxisIndex() < 0)
                                InvalidArgument("Function '%S': Splice operation's axis index (%d) must be >= 0.", AsString().c_str(), spliceAxis.StaticAxisIndex());

                            outputShape = SpliceOutputShape(m_inputs, spliceAxis.StaticAxisIndex());
                            break;
                        }
                        case PrimitiveOpType::RandomSample:
                        case PrimitiveOpType::RandomSampleInclusionFrequency:
                        {
                            auto numSamples = m_attributes[PrimitiveFunctionAttribute::AttributeNameNumSamples].Value<size_t>();
                            auto allowDuplicates = m_attributes[PrimitiveFunctionAttribute::AttributeNameAllowDuplicates].Value<bool>();

                            if (numSamples == 0)
                                InvalidArgument("RandomSample/RandomSampleInclusionFrequency: Number of requested samples must be > 0.");

                            let& shape = m_inputs[0].Shape();
                            size_t numClasses = shape.Dimensions()[0];

                            if (numClasses != NDShape::InferredDimension && !allowDuplicates && numClasses <= numSamples)
                                InvalidArgument("RandomSample/RandomSampleInclusionFrequency: For sampling without duplicates the number of requested samples "
                                                "(%lu) must be less than the number of classes (%lu).", numSamples, numClasses);

                            // within this block we handle RandomSample and RandomSampleInclusionFrequency
                            if (m_op == PrimitiveOpType::RandomSampleInclusionFrequency)
                                outputShape = shape;
                            else
                            {
                                vector<size_t> dimensions{ numClasses, numSamples };
                                outputShape = NDShape(dimensions);
                            }

                            break;
                        }
                        case PrimitiveOpType::OptimizedRNNStack:
                        {
                            assert(m_inputs.size() == 2);
                            auto operand = m_inputs[0];
                            auto parameter = m_inputs[1];
                            if (operand.Shape().Rank() != 1)
                                InvalidArgument("OptimizedRNNStack: input '%S' must have rank 1; actual input rank is %lu.", operand.AsString().c_str(), operand.Shape().Rank());
                            if (operand.DynamicAxes().empty())
                                InvalidArgument("OptimizedRNNStack: input '%S' must have at least one dynamic axis.", operand.AsString().c_str());
                            auto numLayers = m_attributes[PrimitiveFunctionAttribute::AttributeNameNumLayers].Value<size_t>();
                            if (numLayers == 0)
                                InvalidArgument("Number of layers (%d) in OptimizedRNNStack operation must be > 0.", (int)numLayers);
                            auto bidirectional = m_attributes[PrimitiveFunctionAttribute::AttributeNameBidirectional].Value<bool>();
                            auto hiddenSize = m_attributes[PrimitiveFunctionAttribute::AttributeNameHiddenSize].Value<size_t>();

                            if (operand.Shape().HasFreeDimension())
                                InvalidArgument("OptimizedRNNStack: Operand '%S' with free dimension is unsupported.", operand.AsString().c_str());

                            // output dims
                            outputShape = operand.Shape();
                            outputShape[0] = (bidirectional ? 2 : 1) * hiddenSize;
                            // infer input size
                            // Note: Output dim is second axis, so say initOutputRank=-1.
                            if (!operand.Shape().HasUnboundDimension() && (parameter.Shape().Rank() == 2))
                            {
                                const auto recurrentOp = m_attributes[PrimitiveFunctionAttribute::AttributeNameRecurrentOp].Value<std::wstring>();
                                const auto attributes = RnnAttributes(bidirectional, numLayers, hiddenSize, recurrentOp, -1);
                                const auto numParameters = attributes.GetNumParameters(operand.Shape().TotalSize());
                                std::vector<std::pair<Variable, NDShape>> newOperandShapes = { { parameter, std::move(NDShape({ numParameters.first, numParameters.second })) } };
                                UpdateOperandShapes(newOperandShapes);
                            }
                            break;
                        }
                        case PrimitiveOpType::ReconcileDynamicAxis:
                        {
                            assert(m_inputs.size() == 2);
                            auto operand = m_inputs[0];
                            auto layout = m_inputs[1];
                            // data operand can be a constant or a param matrix
                            if (layout.DynamicAxes().empty())
                                InvalidArgument("ReconcileDynamicAxis: layout operand '%S' must have at least one dynamic axis.", layout.AsString().c_str());
                            outputShape = operand.Shape();
                            break;
                        }
                        case PrimitiveOpType::CosDistanceWithNegativeSamples:
                        {
                            assert(m_inputs.size() == 4);

                            auto shiftInput = m_inputs[2];
                            auto numNegativeSamplesInput = m_inputs[3];
                            if (!IsConstantScalar(shiftInput) || !IsConstantScalar(numNegativeSamplesInput))
                                InvalidArgument("CosDistanceWithNegativeSamples: Input(2) '%S' and Input(3) '%S' correpond to shift and numNegativeSamples inputs and must be scalar constants.",
                                                shiftInput.AsString().c_str(), numNegativeSamplesInput.AsString().c_str());

                            auto numNegativeSamples = (size_t)Constant(numNegativeSamplesInput).Value()->AsScalar<float>();
                            outputShape = NDShape({ numNegativeSamples + 1 });
                            break;
                        }
                        case PrimitiveOpType::Crop:
                        {
                            // Width and height are cropped, while remaining dimensions are unchanged.
                            assert(m_inputs.size() == 2 || m_inputs.size() == 4);
                            outputShape = m_inputs[1].Shape();
                            const NDShape& input0Shape = m_inputs[0].Shape();
                            if (input0Shape.Rank() != outputShape.Rank())
                            {
                                RuntimeError("Function '%S': cropped input '%S' and reference input '%S' have different ranks.",
                                    AsString().c_str(),
                                    m_inputs[0].AsString().c_str(),
                                    m_inputs[1].AsString().c_str()
                                );
                            }
                            if (input0Shape.Rank() < 2)
                            {
                                RuntimeError("Function '%S': cropped input '%S' must have rank at least 2.",
                                    AsString().c_str(),
                                    m_inputs[0].AsString().c_str()
                                );
                            }
                            for (int i = 2; i < input0Shape.Rank(); ++i)
                            {
                                outputShape[i] = input0Shape[i];
                            }
                            break;
                        }
                        case PrimitiveOpType::TopK:
                        {
                            assert(m_inputs.size() == 1);
                            auto k = m_attributes[PrimitiveFunctionAttribute::AttributeNameNumItems].Value<size_t>();
                            outputShape = m_inputs[0].Shape();
                            if (outputShape.Rank() > 0)
                                outputShape[0] = k;
                            else if (k != 1)
                                RuntimeError("Function '%S': cannot get k>1 items from a scalar.", AsString().c_str());
                            break;
                        }
                        default:
                            LogicError("Specified Primitive Function op %S is not supported", PrimitiveOpTypeName(m_op).c_str());
                            break;
                        }
                    }
                }
            }

            auto primaryOutput = OutputVariable(outputShape, outputDataType, outputDynamicAxes, needsGradient, Name().empty() ? L"" : Name());
            outputs.push_back(primaryOutput);
            if (m_op == PrimitiveOpType::UnpackSequence)
            {
                auto suppressMaskOutput = m_attributes[PrimitiveFunctionAttribute::AttributeNameSequenceUnpackSuppressMaskOutput].Value<bool>();
                if (!suppressMaskOutput)
                {
                    auto maskOutput = OutputVariable({ NDShape::FreeDimension }, outputDataType, outputDynamicAxes, /*needsGradient =*/ false, Name().empty() ? L"" : Name() + L"_UnpackSequenceMask");
                    outputs.push_back(maskOutput);
                }
            }
            else if (m_op == PrimitiveOpType::TopK)
            {
                auto IndexOutput = OutputVariable(outputShape, outputDataType, outputDynamicAxes, /*needsGradient =*/ false, Name().empty() ? L"" : Name() + L"_TopKIndexMask");
                outputs.push_back(IndexOutput);
            }
        }
    }

    /**
    * Collect output axes for reduce operation.
    * @param staticAxesToReduce a list of static axes to reduce
    * @param batchAxesToReduce a list of batch axes to reduce
    * @param sequenceAxesToReduce a list of sequence axes to reduce
    * @param isAllAxesReduced a flag which indicates whether all axes need to be reduced
    */
    void PrimitiveFunction::CollectReduceOutputAxesForOutputShape(
        std::vector<Axis>& staticAxesToReduce,
        std::vector<Axis>& batchAxesToReduce,
        std::vector<Axis>& sequenceAxesToReduce,
        bool & isAllAxesReduced)
    {
        isAllAxesReduced = false;

        auto collect_axis = [&](const Axis& reductionAxis) {
            if (reductionAxis == Axis::AllStaticAxes() || reductionAxis == Axis::AllAxes())
                isAllAxesReduced = true;
            else if (reductionAxis == Axis::DefaultBatchAxis())
            {
                auto dynamicAxes = m_inputs[0].DynamicAxes();
                if (dynamicAxes != Axis::UnknownDynamicAxes())
                {
                    if (std::find(dynamicAxes.begin(), dynamicAxes.end(), Axis::DefaultBatchAxis()) == dynamicAxes.end())
                        LogicError("ReduceElements: operand %S; No batch axis found during reduction along the batch axis.", m_inputs[0].AsString().c_str());

                    if (dynamicAxes.size() > 1)
                        LogicError("ReduceElements: operand %S; Reduction along the batch axis on input sequence is currently unsupported.", m_inputs[0].AsString().c_str());
                }
                batchAxesToReduce.push_back(reductionAxis);
            }
            else if (reductionAxis.IsSequenceAxis())
            {
                sequenceAxesToReduce.push_back(reductionAxis);
            }
            else
            {
                staticAxesToReduce.push_back(reductionAxis);
            }
        };
        if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameAxisVec))
        {
            const auto &axisDictionary = m_attributes[PrimitiveFunctionAttribute::AttributeNameAxisVec].Value<std::vector<DictionaryValue>>();
            for (const auto& value : axisDictionary) {
                collect_axis(value.Value<Axis>());
            }

        }
        else if (m_attributes.Contains(PrimitiveFunctionAttribute::AttributeNameAxis))
        {
            collect_axis(m_attributes[PrimitiveFunctionAttribute::AttributeNameAxis].Value<Axis>());
        }
        else
        {
            RuntimeError("Function '%ls': Reduce operation with no '%ls' or  '%ls' attributes",
                AsString().c_str(),
                PrimitiveFunctionAttribute::AttributeNameAxis.c_str(),
                PrimitiveFunctionAttribute::AttributeNameAxisVec.c_str()
            );
        }
    }

    vector<DictionaryValue> GetInputUids(const Function& f)
    {
        auto inputs = f.Inputs();
        vector<DictionaryValue> inputUids;
        inputUids.reserve(inputs.size());
        for (auto& input : inputs)
        {
            inputUids.push_back(input.Uid());
        }
        return inputUids;
    }

    Dictionary SerializeCommonFunctionAttributes(const Function& f, size_t version, const wstring& functionType)
    {
        Dictionary dict;
        dict[versionKey] = version;
        dict[typeKey] = functionType;
        dict[uidKey] = f.Uid();
        if (!f.Name().empty())
            dict[nameKey] = f.Name();
        dict[inputsKey] = std::move(GetInputUids(f));
        return dict;
    }

    static const std::wstring s_primitiveFunctionTypeValue = L"PrimitiveFunction";

    /*virtual*/ Dictionary PrimitiveFunction::Serialize() const
    {
        Dictionary dict = SerializeCommonFunctionAttributes(*this, CurrentVersion(), s_primitiveFunctionTypeValue);
        dict[opKey] = static_cast<size_t>(m_op);
        dict[attributesKey] = Attributes();

        if (m_op == PrimitiveOpType::Block)
        {
            auto blockFunction = dynamic_cast<const BlockFunction*>(this);
            auto blockCompositeFunc = dynamic_cast<const CompositeFunction*>(blockFunction->Composite().get());
            dict[blockFunctionCompositeKey] = blockCompositeFunc->SerializeBlockComposite();
            dict[blockFunctionOpNameKey] = OpName();

            const auto& blockArgumentsMap = BlockArgumentsMapping();
            std::vector<std::wstring> serializedArgumentsMapKeys;
            std::vector<std::wstring> serializedArgumentsMapValues;
            for (auto argumentMapping : blockArgumentsMap)
            {
                serializedArgumentsMapKeys.push_back(argumentMapping.first.Uid());
                serializedArgumentsMapValues.push_back(argumentMapping.second.Uid());
            }

            dict[blockFunctionCompositeArgumentsMapKeysKey] = AsDictionaryValueVector(serializedArgumentsMapKeys);
            dict[blockFunctionCompositeArgumentsMapValuesKey] = AsDictionaryValueVector(serializedArgumentsMapValues);
        }

        return dict;
    }

    std::vector<Variable> GetInputVariables(const Dictionary& dict, const std::unordered_map<std::wstring, Variable>& uidToVariableMap, size_t currentSerializationVersion)
    {
        const auto& inputUids = dict[inputsKey].Value<vector<DictionaryValue>>();

        vector<Variable> inputs;
        inputs.reserve(inputUids.size());

        for (const auto& dictionaryValue : inputUids)
        {
            const auto& inputUid = dictionaryValue.Value<std::wstring>();
            if (uidToVariableMap.find(inputUid) == uidToVariableMap.end())
            {
                auto version = dict[versionKey].Value<size_t>();
                CNTK::LogicError("There are no inputs corresponding to input uid = '%ls' (%s).",
                    inputUid.c_str(), GetVersionsString<PrimitiveFunction>(currentSerializationVersion, version).c_str());
            }
            inputs.push_back(uidToVariableMap.at(inputUid));
        }

        return inputs;
    }

    /*static*/ FunctionPtr PrimitiveFunction::Deserialize(const Dictionary& dict,
                                                          const std::unordered_map<std::wstring, Variable>& uidToVariableMap,
                                                          const std::unordered_set<FunctionPtr>& allPrimitiveFunctions,
                                                          const std::unordered_map<Variable, Variable>& placeholderReplacements,
                                                          const CNTK::DeviceDescriptor& device)
    {
        static const vector<std::wstring> s_requiredDictionaryKeys = { typeKey, opKey, uidKey, attributesKey, inputsKey };
        size_t version = ValidateDictionary<PrimitiveFunction>(dict, s_requiredDictionaryKeys, s_primitiveFunctionTypeValue, s_serializationVersion);

        PrimitiveOpType op = PrimitiveOpType(dict[opKey].Value<std::size_t>());

        // The hard requirement that the serialization depends on is that
        // new op type values are only added to the end of the list.
        // This also applies to other enums (DataType, VariableKind, etc.)
        if (op >= PrimitiveOpType::UnknownOP)
        {
            CNTK::LogicError("Unexpected op '%ls':'%u' (%s).",
                             opKey.c_str(),
                             static_cast<std::underlying_type<CNTK::PrimitiveOpType>::type>(op),
                             GetVersionsString<PrimitiveFunction>(s_serializationVersion, version).c_str());
        }

        const auto& uid = dict[uidKey].Value<std::wstring>();
        std::wstring name = L"";
        if (dict.Contains(nameKey))
            name = dict[nameKey].Value<std::wstring>();
        auto attributes = dict[attributesKey].Value<Dictionary>();


        if (op == PrimitiveOpType::Block)
        {
            static const vector<std::wstring> s_requiredBlockFunctionDictionaryKeys = { blockFunctionCompositeKey, blockFunctionOpNameKey, blockFunctionCompositeArgumentsMapKeysKey, blockFunctionCompositeArgumentsMapValuesKey };
            ValidateDictionary<PrimitiveFunction>(dict, s_requiredBlockFunctionDictionaryKeys, s_primitiveFunctionTypeValue, s_serializationVersion);

            auto composite = CompositeFunction::DeserializeBlockComposite(dict[blockFunctionCompositeKey].Value<Dictionary>(), allPrimitiveFunctions, placeholderReplacements, device);

            auto compositeArguments = composite->Arguments();
            auto findCompositeArgumentByUid = [&compositeArguments](const std::wstring& uid) {
                return *std::find_if(compositeArguments.begin(), compositeArguments.end(), [&uid](const Variable& argument) {
                    return (argument.Uid() == uid);
                });
            };

            const auto& blockOpName = dict[blockFunctionOpNameKey].Value<std::wstring>();

            auto blockArgumentsMapKeys = AsVector<std::wstring>(dict[blockFunctionCompositeArgumentsMapKeysKey].Value<std::vector<DictionaryValue>>());
            auto blockArgumentsMapValues = AsVector<std::wstring>(dict[blockFunctionCompositeArgumentsMapValuesKey].Value<std::vector<DictionaryValue>>());
            if (blockArgumentsMapKeys.size() != blockArgumentsMapValues.size())
                RuntimeError("Invalid block function dictionary found during deserialization; Number (%d) of block argument map keys does not match "
                             "the number (%d) of map values.", (int)blockArgumentsMapKeys.size(), (int)blockArgumentsMapValues.size());

            std::vector<std::pair<Variable, Variable>> argumentsMap;
            for (size_t i = 0; i < blockArgumentsMapKeys.size(); ++i)
                argumentsMap.push_back({ findCompositeArgumentByUid(blockArgumentsMapKeys[i]), uidToVariableMap.at(blockArgumentsMapValues[i]) });

            return std::shared_ptr<BlockFunction>(new BlockFunction(std::move(composite), argumentsMap, blockOpName, std::move(attributes), name, uid),
                                                  [](BlockFunction* ptr) { delete ptr; });
        }

        auto inputs = GetInputVariables(dict, uidToVariableMap, s_serializationVersion);

        if (version < 4 && op == PrimitiveOpType::BatchNormalization)
        {
            if (GetTraceLevel() >= TraceLevel::Warning)
            {
                // TODO: all logging functionality should be refactored to live in a logging utility class.
                fprintf(stderr, "WARNING: the dictionary (version=%zu) does not contain a required "
                    "BatchNormalization parameter for the running mean sample count. "
                    "Injected a new parameter with a value of '0'.", version);
            }

            // patch up old the model by adding an extra input
            auto runCount = Constant::Scalar(0.0f, device);
            // HACK: uid has to be changed (by adding some unique prefix to the auto-generated "Constant"+ID_counter)
            // to avoid conflicts with uids recorded in the function graph, which we are deserializing.
            runCount.m_dataFields->m_uid = L"BatchNormSampleCount" + runCount.m_dataFields->m_uid;
            inputs.push_back(runCount);
        }

        return std::shared_ptr<PrimitiveFunction>(new PrimitiveFunction(op, inputs, std::move(attributes), name, uid),
                                                  [](PrimitiveFunction* ptr) { delete ptr; });
    }

    Dictionary PrimitiveFunction::GetState() const
    {
        if (!IsStateful())
            LogicError("Function '%S' is not stateful.", AsString().c_str());

        Dictionary state;
        for (auto& key : PrimitiveFunctionAttribute::s_rngStateAttributes)
        {
            state[key] = m_attributes[key];
        }

        return state;
    }

    void PrimitiveFunction::SetState(const Dictionary& state)
    {
        if (!IsStateful())
            LogicError("Function '%S' is not stateful.", AsString().c_str());

        for (auto& key : PrimitiveFunctionAttribute::s_rngStateAttributes)
        {
            m_attributes[key] = state[key];
        }
    }

    /*static*/ void PrimitiveFunction::FixNDShape(size_t filterRank, size_t inputRank, NDShape& shape, size_t deflt, const NDShape& from/* = NDShape()*/)
    {
        auto dims = shape.Dimensions();
        Microsoft::MSR::CNTK::ConvolutionNodeBase<float>::FixVectorShape(filterRank, inputRank, dims, deflt, from.Dimensions());
        shape = NDShape(dims);
        if (shape.HasFreeDimension())
            InvalidArgument("Convolution kernel or stride/padding attribute shape has an illegal FreeDimension after shape inference.");
    }

    /*static*/ NDShape PrimitiveFunction::ConvolutionOpOutputShape(PrimitiveOpType op, const NDShape& operandShape, NDShape& kernelShape, NDShape& outputMapCount, NDShape& strides,
                                                                   std::vector<bool>& sharing, std::vector<bool>& autoPad, NDShape& lowerPad, NDShape& upperPad,
                                                                   bool transpose, bool inferDimensions, NDShape& dilation, size_t groups/*=1*/, bool ceilOutputDim/* = false*/)
    {
        if (inferDimensions)
        {
            size_t inputRank = operandShape.Rank();

            // Unknown kernel shape valid only for pooling, however, the shape should have expanded before
            // this call.
            if (kernelShape.IsUnknown())
                RuntimeError("Convolution: Kernel shape can't be Unknown.");

            // infer reduction dimensions if not given
            // If kernel has a lower rank than the input then the remaining dimensions are to be reduced over.
            size_t filterRank = kernelShape.Rank();

            // If the trailing axis dimensionality of the kernel shape is NDShape::InferredDimension, we reduce over it by
            // picking the corresponding operand shape dimensionality
            // This is done by shrinking the filter rank and let the dimensions be inferred from the operand's shape
            // TODO: Should we do this for all of the axes in kernelShape that have a dimensionailty of NDShape::InferredDimension?
            if ((filterRank > 0) && (kernelShape[filterRank - 1] == NDShape::InferredDimension))
            {
                filterRank--;
                kernelShape = kernelShape.SubShape(0, filterRank);
            }

            NDShape fromShape;
            if (op == PrimitiveOpType::Convolution)
                fromShape = operandShape;

            size_t fillRank = (!transpose) ? filterRank : filterRank - 1;
            FixNDShape(fillRank, inputRank, kernelShape, 1, fromShape); // convolve over red dim; pool over 1
            FixNDShape(fillRank, inputRank, strides, 1, fromShape); // stride for reduction dims is red dim or 1
            FixNDShape(fillRank, inputRank, dilation, 1);
            FixNDShape(fillRank, inputRank, lowerPad, 0);
            FixNDShape(fillRank, inputRank, upperPad, 0);
            Microsoft::MSR::CNTK::ConvolutionNodeBase<float>::FixVectorShape(fillRank, inputRank, sharing, true);
            Microsoft::MSR::CNTK::ConvolutionNodeBase<float>::FixVectorShape(fillRank, inputRank, autoPad, false); // no padding for reduction dims
        }

        decltype(&Microsoft::MSR::CNTK::ConvolveGeometry::ComputeOutputShape) computeOutputShapeFunc;
        if (!transpose)
            computeOutputShapeFunc = &Microsoft::MSR::CNTK::ConvolveGeometry::ComputeOutputShape;
        else
            computeOutputShapeFunc = &Microsoft::MSR::CNTK::ConvolveGeometry::ComputeInputShape;

        auto outputShape = AsNDShape(computeOutputShapeFunc(AsTensorShape(operandShape), AsTensorShape(kernelShape), AsTensorShape(outputMapCount),
            AsTensorShape(strides), sharing, autoPad, AsTensorShape(lowerPad), AsTensorShape(upperPad), AsTensorShape(dilation), groups, ceilOutputDim,
            operandShape.HasFreeDimension(), false));

        // Any input dimensions that are free/inferred pass through as free/inferred
        for (size_t i = 0; i < operandShape.Rank(); ++i)
        {
            if ((operandShape[i] == NDShape::FreeDimension) || (operandShape[i] == NDShape::InferredDimension))
                outputShape[i] = operandShape[i];
        }
        return outputShape;
    }

    /*static*/ bool PrimitiveFunction::UpdateOperandShapes(std::vector<std::pair<Variable, NDShape>>& newOperandShapes)
    {
        bool anyParameterOperandDimsInferred = false;
        auto updateOperandShapeFunc = [](Variable& operand, const NDShape& newOperandShape) {
            if ((operand.IsParameter() || operand.IsConstant()) && (operand.Shape() != newOperandShape))
            {
                operand.m_dataFields->m_shape = newOperandShape;
                return true;
            }

            return false;
        };

        for (auto& newOperandShapePair : newOperandShapes)
            anyParameterOperandDimsInferred = updateOperandShapeFunc(newOperandShapePair.first, newOperandShapePair.second) || anyParameterOperandDimsInferred;

        return anyParameterOperandDimsInferred;
    }

    /*static*/ NDShape PrimitiveFunction::NaryElementwiseOpOutputShape(PrimitiveOpType op, std::vector<Variable>& operands, bool inferInputDimensions)
    {
        assert(operands.size() > 1);

        // TODO: Is this logic of transitively constructing the output shape from the operands correct?
        Variable dummyOutputVariable = PlaceholderVariable(NDShape());
        for (auto& operand : operands)
            dummyOutputVariable.m_dataFields->m_shape = BinaryElementwiseOpOutputShape(op, dummyOutputVariable, operand, inferInputDimensions);

        return dummyOutputVariable.Shape();
    }

    void PrimitiveFunction::SetDropoutRate(double dropoutRate)
    {
        if (OpType() != PrimitiveOpType::Dropout)
            LogicError("Cannot set dropout rate on '%S' function.", OpName().c_str());

        m_attributes[PrimitiveFunctionAttribute::AttributeNameDropoutRate] = dropoutRate;
        m_dirtyAttributes.insert(PrimitiveFunctionAttribute::AttributeNameDropoutRate);
    }

    void PrimitiveFunction::SetRandomSeed(size_t seed)
    {
        if (!IsStateful())
            LogicError("Cannot set random seed on '%S' function.", OpName().c_str());

        m_attributes[PrimitiveFunctionAttribute::AttributeNameRngSeed] = seed;
        m_dirtyAttributes.insert(PrimitiveFunctionAttribute::AttributeNameRngSeed);
    }
}
