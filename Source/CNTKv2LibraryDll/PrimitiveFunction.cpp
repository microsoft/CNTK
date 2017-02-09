//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "PrimitiveFunction.h"
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

using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    // Names for the reduction operations as used by the CNTK ReduceElementsNode
    /*static*/ const std::wstring PrimitiveFunction::InternalSumReductionOpName = L"Sum";
    /*static*/ const std::wstring PrimitiveFunction::InternalLogSumReductionOpName = L"LogSum";
    /*static*/ const std::wstring PrimitiveFunction::InternalMeanReductionOpName = L"Mean";
    /*static*/ const std::wstring PrimitiveFunction::InternalMaxReductionOpName = L"Max";
    /*static*/ const std::wstring PrimitiveFunction::InternalMinReductionOpName = L"Min";
    /*static*/ const std::wstring PrimitiveFunction::InternalProdReductionOpName = L"Prod";
    /*static*/ const std::wstring PrimitiveFunction::InternalAllReductionOpName = L"All";
    /*static*/ const std::wstring PrimitiveFunction::InternalAnyReductionOpName = L"Any";

    // Names of the various attributes of CNTK primitive Functions
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAxis = L"axis";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAxis1 = L"axis1";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAxis2 = L"axis2";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAllowDuplicates = L"allowDuplicates";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNumSamples = L"numSamples";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameDropoutRate = L"dropoutRate";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNewShape = L"newShape";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBeginAxis = L"beginAxis";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameEndAxis = L"endAxis";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameOutputRank = L"outputRank";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameInferInputRankToMap = L"inferInputRankToMap";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameOffset = L"offset";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameStrides = L"strides";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSharing = L"sharing";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameAutoPadding = L"autoPadding";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameLowerPad = L"lowerPad";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameUpperPad = L"upperPad";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameTranspose = L"transpose";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameMaxTempMemSizeInSamples = L"maxTempMemSizeInSamples";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameROIOutputShape = L"roiOutputShape";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNamePoolingType = L"poolingType";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNamePoolingWindowShape = L"poolingWindowShape";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSpatial = L"spatial";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNormalizationTimeConstant = L"normalizationTimeConstant";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBlendTimeConstant = L"blendTimeConstant";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameEpsilon = L"epsilon";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameUseCuDNNEngine = L"useCuDNNEngine";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNewDynamicAxes = L"newDynamicAxes";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNewSequenceAxisLengthScalingFactor = L"newSequenceAxisLengthScalingFactor";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNewSequenceAxisLengthAdditiveFactor = L"newSequenceAxisLengthAdditiveFactor";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBeginIndex = L"beginIndex";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameEndIndex = L"endIndex";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameReductionOpName = L"reductionOpName";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameBidirectional = L"bidirectional";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameNumLayers = L"numLayers";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameHiddenSize = L"hiddenSize";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameRecurrentOp = L"recurrentOp";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameRngSeed = L"rngSeed";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameRngOffset = L"rngOffset";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameUnpoolingWindowShape = L"unpoolingWindowShape";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSubstitutionPenalty = L"SubstitutionPenalty";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameDeletionPenalty = L"DeletionPenalty";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameInsertionPenalty = L"InsertionPenalty";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSquashInputs = L"SquashInputs";
    /*static*/ const std::wstring PrimitiveFunction::AttributeNameSamplesToIgnore = L"SamplesToIgnore";

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
                    // The DataType of all operands should match except for Constants where we allow coercion
                    if ((inputDataType != DataType::Unknown) && (inputDataType != outputDataType) && !input.IsConstant())
                        InvalidArgument("Primitive function with op type %S has operands with different DataTypes %s and %s", PrimitiveOpTypeName(op).c_str(), DataTypeName(outputDataType), DataTypeName(inputDataType));
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
                    input.m_dataFields->m_dataType = outputDataType;
            }
        }

        return outputDataType;
    }

    /*static*/ std::vector<Axis> PrimitiveFunction::GetOutputDynamicAxes(PrimitiveOpType op, std::vector<Variable>& inputs, Dictionary& functionConfig)
    {
        // We currently require that the inputs' dynamic axes, if any, match
        std::vector<Axis> outputDynamicAxes;
        if ((op == PrimitiveOpType::SumAll) ||
            (op == PrimitiveOpType::ReduceElements && functionConfig[PrimitiveFunction::AttributeNameAxis].Value<Axis>() == Axis::AllAxes()) ||
            (op == PrimitiveOpType::SquaredError) ||
            (op == PrimitiveOpType::CrossEntropyWithSoftmax) ||
            (op == PrimitiveOpType::ClassificationError) ||
            (op == PrimitiveOpType::Logistic) ||
            (op == PrimitiveOpType::LambdaRank) ||
            (op == PrimitiveOpType::NDCG))
        {
            outputDynamicAxes = std::vector<Axis>({});
        }
        else if (op == PrimitiveOpType::Where)
        {
            if (functionConfig.Contains(PrimitiveFunction::AttributeNameNewDynamicAxes))
                outputDynamicAxes = AsVector<Axis>(functionConfig[PrimitiveFunction::AttributeNameNewDynamicAxes].Value<std::vector<DictionaryValue>>());
            else
            {
                if (inputs[0].DynamicAxes() == Axis::UnknownDynamicAxes())
                    outputDynamicAxes = Axis::UnknownDynamicAxes();
                else
                {
                    if (functionConfig.Contains(PrimitiveFunction::AttributeNameNewSequenceAxisLengthScalingFactor) &&
                        functionConfig.Contains(PrimitiveFunction::AttributeNameNewSequenceAxisLengthAdditiveFactor))
                    {
                        size_t newSequenceAxisLengthScalingFactor = functionConfig[PrimitiveFunction::AttributeNameNewSequenceAxisLengthScalingFactor].Value<size_t>();
                        int newSequenceAxisLengthAdditiveFactor = functionConfig[PrimitiveFunction::AttributeNameNewSequenceAxisLengthAdditiveFactor].Value<int>();

                        auto derivedDynamicAxes = GetDerivedDynamicAxes(inputs[0].DynamicAxes()[0], newSequenceAxisLengthScalingFactor, newSequenceAxisLengthAdditiveFactor);
                        std::copy(derivedDynamicAxes.begin(), derivedDynamicAxes.end(), std::back_inserter(outputDynamicAxes));
                    }
                    else
                    {
                        std::function<Variable(const Variable&)> GetActualSourceVariable;
                        GetActualSourceVariable = [&GetActualSourceVariable](const Variable& var) -> Variable {
                            if (var.BlockFunctionVariableMapping() == Variable())
                                return var;
                            else
                                return GetActualSourceVariable(var.BlockFunctionVariableMapping());
                        };

                        auto whereNodeConditionSourceVar = GetActualSourceVariable(inputs[0]);
                        auto whereNodeSequenceAxis = Axis(std::wstring(L"whereNodeDynamicAxis_conditionVar_") + whereNodeConditionSourceVar.Uid());
                        outputDynamicAxes.push_back(whereNodeSequenceAxis);
                    }

                    for (size_t i2 = 1; i2 < inputs[0].DynamicAxes().size(); ++i2)
                        outputDynamicAxes.push_back(inputs[0].DynamicAxes()[i2]);

                    functionConfig[PrimitiveFunction::AttributeNameNewDynamicAxes] = AsDictionaryValueVector(outputDynamicAxes);
                }
            }
        }
        else if (op == PrimitiveOpType::ScatterPacked)
            outputDynamicAxes = inputs[2].DynamicAxes();
        else if ((op == PrimitiveOpType::PackedIndex) || (op == PrimitiveOpType::GatherPacked))
            outputDynamicAxes = inputs[1].DynamicAxes();
        else if (op == PrimitiveOpType::ReconcileDynamicAxis)
            outputDynamicAxes = inputs[1].DynamicAxes();
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
                                LogicError("Currently if an operand of a elementwise operation has any dynamic axes, those must match the dynamic axes of the other operands");
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
            outputs.push_back(OutputVariable(m_inputs[0].Shape(), m_inputs[0].GetDataType(), m_inputs[0].DynamicAxes(), Name()));
        else
        {
            DataType outputDataType = GetOutputDataType(m_op, m_inputs, true);
            std::vector<Axis> outputDynamicAxes = GetOutputDynamicAxes(m_op, m_inputs, m_attributes);

            NDShape outputShape = NDShape::Unknown;
            bool allInputShapesUnknown = (std::find_if(m_inputs.begin(), m_inputs.end(), [](const Variable& input) { return !input.Shape().IsUnknown(); }) == m_inputs.end());
            bool anyInputShapesUnknown = (std::find_if(m_inputs.begin(), m_inputs.end(), [](const Variable& input) { return input.Shape().IsUnknown(); }) != m_inputs.end());
            if (!anyInputShapesUnknown || (!allInputShapesUnknown && (outputDynamicAxes != Axis::UnknownDynamicAxes())))
            {
                switch (m_op)
                {
                    // Elementwise operators' shapes are a zip of inputs and can be determined even if some of the input shapes are unknown
                case PrimitiveOpType::Plus:
                case PrimitiveOpType::LogPlus:
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
                            LogicError("Currently PastValue/FutureValue Function only supports input operand with 2 dynamic axis (1 sequence-axis and 1 batch-axis)");

                        if (!initialStateVar.DynamicAxes().empty())
                            LogicError("Currently PastValue/FutureValue Function does not support initial state operand with dynamic axes!");
                    }

                    outputShape = BinaryElementwiseOpOutputShape(m_op, m_inputs[0], m_inputs[1], true, true);
                    break;
                }
                case PrimitiveOpType::Clip:
                    assert(m_inputs.size() == 3);
                    outputShape = NaryElementwiseOpOutputShape(m_op, m_inputs, true, true);
                    break;
                case PrimitiveOpType::Select:
                    assert(m_inputs.size() == 3);
                    outputShape = NaryElementwiseOpOutputShape(m_op, m_inputs, true, true);
                    break;
                default:
                    // For all other operations, shapes of all inputs must be known to determine the output shape
                    if (!anyInputShapesUnknown)
                    {
                        switch (m_op)
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
                        case PrimitiveOpType::Dropout:
                        case PrimitiveOpType::Where:
                        case PrimitiveOpType::LogSoftmax:
                        case PrimitiveOpType::Sin:
                        case PrimitiveOpType::Cos:
                        case PrimitiveOpType::Pass:
                            assert(m_inputs.size() == 1);
                            outputShape = UnaryElementwiseOpOutputShape(m_inputs[0].Shape());
                            break;
                        case PrimitiveOpType::PackedIndex:
                            assert(m_inputs.size() == 2);
                            outputShape = UnaryElementwiseOpOutputShape(m_inputs[1].Shape());
                            break;
                        case PrimitiveOpType::ScatterPacked:
                        {
                            assert(m_inputs.size() == 3);
                            if (m_inputs[0].DynamicAxes().empty() || m_inputs[1].DynamicAxes().empty() || m_inputs[2].DynamicAxes().empty())
                                InvalidArgument("ScatterPacked requires all its operands to have dynamic axes");

                            outputShape = UnaryElementwiseOpOutputShape(m_inputs[0].Shape());
                            break;
                        }
                        case PrimitiveOpType::TransposeAxes:
                        {
                            assert(m_inputs.size() == 1);

                            auto axis1 = NormalizeStaticAxis(m_attributes[PrimitiveFunction::AttributeNameAxis1].Value<Axis>(), m_inputs[0].Shape());
                            auto axis2 = NormalizeStaticAxis(m_attributes[PrimitiveFunction::AttributeNameAxis2].Value<Axis>(), m_inputs[0].Shape());

                            if (!axis1.IsStaticAxis() || !axis2.IsStaticAxis())
                                LogicError("TransposeAxes operation currently does not support transposing dynamic axes");

                            // We allow to transpose with an axes that exceeds the rank of the input.
                            // The output rank is the max of the input rank, and either of the axes being transposed.
                            auto outputRank = std::max(m_inputs[0].Shape().Rank(), (size_t)(std::max(axis1.StaticAxisIndex(), axis2.StaticAxisIndex()) + 1));
                            outputShape = m_inputs[0].Shape().AppendShape(NDShape(outputRank - m_inputs[0].Shape().Rank(), 1));
                            std::swap(outputShape[axis1.StaticAxisIndex()], outputShape[axis2.StaticAxisIndex()]);
                            break;
                        }
                        case PrimitiveOpType::Slice:
                        {
                            assert(m_inputs.size() == 1);
                            auto axis = NormalizeStaticAxis(m_attributes[PrimitiveFunction::AttributeNameAxis].Value<Axis>(), m_inputs[0].Shape());

                            auto beginIndex = m_attributes[PrimitiveFunction::AttributeNameBeginIndex].Value<int>();
                            auto endIndex = m_attributes[PrimitiveFunction::AttributeNameEndIndex].Value<int>();
                            if (!axis.IsStaticAxis())
                                LogicError("Built-in Slice operation currently does not support slicing along dynamic axis");

                            VerifyStaticAxis(axis, m_inputs[0].Shape());

                            size_t sliceAxisDim = m_inputs[0].Shape()[axis.StaticAxisIndex()];
                            int realBeginIndex = (beginIndex >= 0) ? beginIndex : beginIndex + sliceAxisDim;
                            int realEndIndex = (endIndex > 0) ? endIndex : endIndex + sliceAxisDim;
                            if ((sliceAxisDim < realEndIndex) || (realEndIndex < realBeginIndex) || (realBeginIndex < 0))
                                RuntimeError("Slice operation: Index range [%d,%d), interpreted as [%d,%d), is invalid for input's shape ([%S]).",
                                    beginIndex,
                                    endIndex,
                                    realBeginIndex,
                                    realEndIndex,
                                    AsStringForErrorReporting(m_inputs[0].Shape()).c_str());

                            auto outputTensorShape = AsTensorShape(m_inputs[0].Shape());

                            // propagate as much as we can
                            if ((axis.StaticAxisIndex() < (int)outputTensorShape.GetRank()) && (0 <= realBeginIndex) && (realBeginIndex <= realEndIndex) && (realEndIndex <= sliceAxisDim))
                                outputTensorShape.NarrowTo(axis.StaticAxisIndex(), realBeginIndex, realEndIndex);

                            outputShape = AsNDShape(outputTensorShape, /*allowNonFlattenableTensorShapes = */ true);
                            break;
                        }
                        case PrimitiveOpType::Reshape:
                        {
                            auto& replacementShape = m_attributes[PrimitiveFunction::AttributeNameNewShape].Value<NDShape>();

                            auto beginAxis = Axis(0);
                            auto endAxis = Axis((int)m_inputs[0].Shape().Rank());
                            if (m_attributes.Contains(PrimitiveFunction::AttributeNameBeginAxis))
                                beginAxis = NormalizeStaticAxis(m_attributes[PrimitiveFunction::AttributeNameBeginAxis].Value<Axis>(), m_inputs[0].Shape());

                            if (m_attributes.Contains(PrimitiveFunction::AttributeNameEndAxis))
                                endAxis = NormalizeStaticAxis(m_attributes[PrimitiveFunction::AttributeNameEndAxis].Value<Axis>(), m_inputs[0].Shape());

                            outputShape = ReshapeOutputShape(m_inputs[0].Shape(), replacementShape, beginAxis, endAxis, true);
                            break;
                        }
                        case PrimitiveOpType::ROIPooling:
                        {
                            assert(m_inputs.size() == 2);
                            auto convMapShape = m_inputs[0].Shape();
                            auto roisShape = m_inputs[1].Shape();
                            auto roiOutputShape = m_attributes[PrimitiveFunction::AttributeNameROIOutputShape].Value<NDShape>();

                            auto outW = roiOutputShape[0];
                            auto outH = roiOutputShape[1];
                            auto numChannels = convMapShape[2];
                            auto roisPerImage = roisShape[1];

                            if (roiOutputShape.Rank() != 2)
                                InvalidArgument("ROIPoolingNode: roi output shape must have two dimensions ([W x H]).");

                            if (convMapShape[0] < outW || convMapShape[1] < outH)
                                InvalidArgument("ROIPoolingNode: inputWidth must >= windowWidth and inputHeight must >= windowHeight.");

                            if (convMapShape[2] < 1)
                                InvalidArgument("ROIPoolingNode: input must have at least one channel ([W x H x C]).");

                            if (roisShape[0] != 4)
                                InvalidArgument("ROIPoolingNode: ROI input must have the following shape: [4 x roisPerImage].");

                            if (roisPerImage < 1)
                                InvalidArgument("ROIPoolingNode: ROI input must contain at least one ROI ([4 x roisPerImage]).");

                            outputShape = { outW, outH, numChannels, roisPerImage };
                            break;
                        }
                        case PrimitiveOpType::Pooling:
                        {
                            assert(m_inputs.size() == 1);
                            auto poolingWindowsShape = m_attributes[PrimitiveFunction::AttributeNamePoolingWindowShape].Value<NDShape>();
                            auto strides = m_attributes[PrimitiveFunction::AttributeNameStrides].Value<NDShape>();
                            auto lowerPad = m_attributes[PrimitiveFunction::AttributeNameLowerPad].Value<NDShape>();
                            auto upperPad = m_attributes[PrimitiveFunction::AttributeNameUpperPad].Value<NDShape>();
                            auto autoPadding = AsVector<bool>(m_attributes[PrimitiveFunction::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
                            NDShape outputMapCount = { 1 };
                            std::vector<bool> sharing = { true };
                            auto inputShape = m_inputs[0].Shape();

                            // In case of pooling if the kernel shape is unknown, then treat it as global pooling.
                            if (poolingWindowsShape == NDShape::Unknown)
                            {
                                if ((std::find(autoPadding.begin(), autoPadding.end(), true) != autoPadding.end()) ||
                                    (lowerPad.TotalSize() > 0) || (upperPad.TotalSize() > 0))
                                    RuntimeError("Padding isn't allowed for Unknown shape!");

                                poolingWindowsShape = inputShape.SubShape(0, inputShape.Rank() - 1);
                                m_attributes[PrimitiveFunction::AttributeNamePoolingWindowShape] = poolingWindowsShape;
                            }

                            outputShape = ConvolutionOpOutputShape(m_op, inputShape, poolingWindowsShape, outputMapCount, strides, sharing, autoPadding, lowerPad, upperPad, false, true);
                            break;
                        }
                        case PrimitiveOpType::Unpooling:
                        {
                            assert(m_inputs.size() == 2);

                            auto inputShape = m_inputs[0].Shape();
                            outputShape = m_inputs[1].Shape();
                            PoolingType unpoolingType = (PoolingType)(m_attributes[PrimitiveFunction::AttributeNamePoolingType].Value<size_t>());
                            if (unpoolingType != PoolingType::Max)
                                LogicError("Only max unpooling is currently supported");

                            // Finding the shape of an unpooling operation from the input to be unpooled alone is ambiguous
                            // For example a 4x4 input with a 5x5 kernel a stride of 2x2
                            // and padding could have resulted from pooling a 7x7 or 8x8 image
                            // Therefore what needs to happen here is to check whether the
                            // outputShape can be pooled into the inputShape using the specified attributes
                            auto unpoolingWindowShape = m_attributes[PrimitiveFunction::AttributeNameUnpoolingWindowShape].Value<NDShape>();
                            auto strides = m_attributes[PrimitiveFunction::AttributeNameStrides].Value<NDShape>();
                            auto lowerPad = m_attributes[PrimitiveFunction::AttributeNameLowerPad].Value<NDShape>();
                            auto upperPad = m_attributes[PrimitiveFunction::AttributeNameUpperPad].Value<NDShape>();
                            auto autoPadding = AsVector<bool>(m_attributes[PrimitiveFunction::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
                            NDShape inputMapCount = { 1 };
                            std::vector<bool> sharing = { true };

                            NDShape inferredInputShape = ConvolutionOpOutputShape(PrimitiveOpType::Pooling, outputShape, unpoolingWindowShape, inputMapCount, strides, sharing, autoPadding, lowerPad, upperPad, false, true);
                            if (inferredInputShape != inputShape)
                                RuntimeError("The shape of the unpooling operand %ls is different from the result of pooling the poolingInput argument using the provided options %ls", inputShape.AsString().c_str(), inferredInputShape.AsString().c_str());

                            break;
                        }
                        case PrimitiveOpType::SumAll:
                            assert(m_inputs.size() == 1);
                            outputShape = { 1 };
                            break;
                        case PrimitiveOpType::Times:
                        {
                            assert(m_inputs.size() == 2);
                            auto outputRank = m_attributes[PrimitiveFunction::AttributeNameOutputRank].Value<size_t>();
                            auto inferInputRankToMap = m_attributes[PrimitiveFunction::AttributeNameInferInputRankToMap].Value<int>();
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
                                LogicError("TransposeTimes operation currently only supports %s operands of rank 1 or 2", Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "right" : "left");

                            NDShape transposedLeftOperandShape = transposeShapeFunc(m_inputs[0].Shape());
                            Variable dummyLeftOperand = PlaceholderVariable(transposedLeftOperandShape);
                            size_t outputRank = m_attributes[PrimitiveFunction::AttributeNameOutputRank].Value<size_t>();
                            outputShape = TimesOpOutputShape(dummyLeftOperand, m_inputs[1], outputRank, -1, true);
                            if (dummyLeftOperand.Shape() != transposedLeftOperandShape)
                                m_inputs[0].m_dataFields->m_shape = transposeShapeFunc(dummyLeftOperand.Shape());

                            break;
                        }
                        case PrimitiveOpType::Convolution:
                        {
                            assert(m_inputs.size() == 2);
                            auto& strides = m_attributes[PrimitiveFunction::AttributeNameStrides].Value<NDShape>();
                            auto& lowerPad = m_attributes[PrimitiveFunction::AttributeNameLowerPad].Value<NDShape>();
                            auto& upperPad = m_attributes[PrimitiveFunction::AttributeNameUpperPad].Value<NDShape>();
                            auto sharing = AsVector<bool>(m_attributes[PrimitiveFunction::AttributeNameSharing].Value<std::vector<DictionaryValue>>());
                            auto autoPadding = AsVector<bool>(m_attributes[PrimitiveFunction::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
                            bool transpose = m_attributes[PrimitiveFunction::AttributeNameTranspose].Value<bool>();
                            if (m_inputs[0].Shape().Rank() < m_inputs[1].Shape().Rank())
                                InvalidArgument("The convolution map should have at least as many axes as the shape of the input it operates on!");

                            NDShape outputMapCount, kernelShape;
                            std::tie(outputMapCount, kernelShape) = GetConvolutionOutputMapCountAndKernelShape(m_inputs[0].Shape(), m_inputs[1].Shape());
                            auto originalKernelShape = kernelShape;
                            outputShape = ConvolutionOpOutputShape(m_op, m_inputs[1].Shape(), kernelShape, outputMapCount, strides, sharing, autoPadding, lowerPad, upperPad, transpose, true);
                            if (originalKernelShape != kernelShape)
                            {
                                for (size_t i2 = 0; i2 < kernelShape.Rank(); ++i2)
                                    m_inputs[0].m_dataFields->m_shape[i2] = kernelShape[i2];
                            }

                            m_attributes[PrimitiveFunction::AttributeNameSharing] = AsDictionaryValueVector(sharing);
                            m_attributes[PrimitiveFunction::AttributeNameAutoPadding] = AsDictionaryValueVector(autoPadding);
                            break;
                        }
                        case PrimitiveOpType::CosDistance:
                        case PrimitiveOpType::EditDistanceError:
                        case PrimitiveOpType::Logistic:
                        case PrimitiveOpType::SquaredError:
                        case PrimitiveOpType::CrossEntropyWithSoftmax:
                        case PrimitiveOpType::ClassificationError:
                        case PrimitiveOpType::LambdaRank:
                        case PrimitiveOpType::NDCG:
                        {
                            if ((m_op == PrimitiveOpType::ClassificationError) || (m_op == PrimitiveOpType::Logistic))
                                assert(m_inputs.size() >= 2);
                            else if ((m_op == PrimitiveOpType::LambdaRank) || (m_op == PrimitiveOpType::NDCG))
                                assert(m_inputs.size() == 3);
                            else
                                assert(m_inputs.size() == 2);

                            if (((m_inputs[0].Shape().Rank() > 2) || ((m_inputs[0].Shape().Rank() > 1) && (m_inputs[0].Shape()[1] != 1))) ||
                                ((m_inputs[1].Shape().Rank() > 2) || ((m_inputs[1].Shape().Rank() > 1) && (m_inputs[1].Shape()[1] != 1))))
                                InvalidArgument("The shape of input operands for the %S operation should have at most one axis", PrimitiveOpTypeName(m_op).c_str());

                            auto predictionShape = m_inputs[0].Shape();
                            auto labelsShape = m_inputs[1].Shape();
                            auto numLeadingAxesToCompare = std::min(predictionShape.Rank(), labelsShape.Rank());
                            if (predictionShape.SubShape(0, numLeadingAxesToCompare) != labelsShape.SubShape(0, numLeadingAxesToCompare))
                                RuntimeError("Prediction output operand's shape %S is incompatible with label operand's shape %S for the %S operation", AsStringForErrorReporting(predictionShape).c_str(), AsStringForErrorReporting(labelsShape).c_str(), PrimitiveOpTypeName(m_op).c_str());

                            std::vector<int> reductionAxes;
                            for (int i3 = 0; i3 < (int)m_inputs[0].Shape().Rank(); ++i3)
                                reductionAxes.push_back(i3);

                            outputShape = ReductionOpOutputShape(m_op, predictionShape, reductionAxes, /*preserveReductionAxes =*/ false);
                            break;
                        }
                        case PrimitiveOpType::ReduceElements:
                        {
                            assert(m_inputs.size() == 1);
                            auto reductionAxis = NormalizeStaticAxis(m_attributes[PrimitiveFunction::AttributeNameAxis].Value<Axis>(), m_inputs[0].Shape());
                            if (reductionAxis == Axis::AllStaticAxes() || reductionAxis == Axis::AllAxes())
                                outputShape = {};
                            else
                            {
                                std::vector<int> reductionAxes = { reductionAxis.StaticAxisIndex() };
                                outputShape = ReductionOpOutputShape(m_op, m_inputs[0].Shape(), reductionAxes, /*preserveReductionAxes =*/ true);
                            }
                            break;
                        }
                        case PrimitiveOpType::BatchNormalization:
                        {
                            assert(m_inputs.size() == 6);
                            auto spatial = m_attributes[PrimitiveFunction::AttributeNameSpatial].Value<bool>();
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
                            auto spliceAxis = NormalizeStaticAxis(m_attributes[PrimitiveFunction::AttributeNameAxis].Value<Axis>(), NDShape(maxInputRank));

                            if (!spliceAxis.IsStaticAxis())
                                LogicError("Splice operation currently does not support splicing along dynamic axis");

                            if (spliceAxis.StaticAxisIndex() < 0)
                                InvalidArgument("Splice: The axis argument's static axis index must be >= 0!");

                            outputShape = SpliceOutputShape(m_inputs, spliceAxis.StaticAxisIndex());
                            break;
                        }
                        case PrimitiveOpType::RandomSample:
                        case PrimitiveOpType::RandomSampleInclusionFrequency:
                        {
                            auto numSamples = m_attributes[PrimitiveFunction::AttributeNameNumSamples].Value<size_t>();
                            auto allowDuplicates = m_attributes[PrimitiveFunction::AttributeNameAllowDuplicates].Value<bool>();

                            if (numSamples == 0)
                                InvalidArgument("Number of requested samples is zero.");

                            let& shape = m_inputs[0].Shape();
                            size_t numClasses = shape.Dimensions()[0];

                            if (numClasses != NDShape::InferredDimension && !allowDuplicates && numClasses <= numSamples)
                                InvalidArgument("For sampling without duplicates the number of requested samples (%lu) needs to be less than the number of classes (%lu).", numSamples, numClasses);

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
                                InvalidArgument("OptimizedRNNStack: input must have rank 1; actual input rank is %lu", operand.Shape().Rank());
                            if (operand.DynamicAxes().empty())
                                InvalidArgument("OptimizedRNNStack: input must have at least one dynamic axis");
                            auto numLayers = m_attributes[PrimitiveFunction::AttributeNameNumLayers].Value<size_t>();
                            if (numLayers == 0)
                                InvalidArgument("Number of layers in OptimizedRNNStack operation should be positive");
                            auto bidirectional = m_attributes[PrimitiveFunction::AttributeNameBidirectional].Value<bool>();
                            auto hiddenSize = m_attributes[PrimitiveFunction::AttributeNameHiddenSize].Value<size_t>();

                            // output dims
                            outputShape = operand.Shape();
                            outputShape[0] = (bidirectional ? 2 : 1) * hiddenSize;
                            // infer input size
                            // Note: Output dim is second axis, so say initOutputRank=-1.
                            if (parameter.Shape().Rank() == 2)
                            {
                                const auto recurrentOp = m_attributes[PrimitiveFunction::AttributeNameRecurrentOp].Value<std::wstring>();
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
                            if (layout.DynamicAxes().empty())
                                InvalidArgument("ReconcileDynamicAxis: layout must have at least one dynamic axis");
                            outputShape = operand.Shape();
                            break;
                        }
                        default:
                            LogicError("Specified m_op %S not yet supported", PrimitiveOpTypeName(m_op).c_str());
                            break;
                        }
                    }
                }
            }

            outputs.push_back({ OutputVariable(outputShape, outputDataType, outputDynamicAxes, Name().empty() ? L"" : Name()) });
        }
    }

    static const std::wstring s_primitiveFunctionTypeValue = L"PrimitiveFunction";

    /*virtual*/ Dictionary PrimitiveFunction::Serialize() const 
    {
        Dictionary dict;

        dict[versionKey] = CurrentVersion();
        dict[typeKey] = s_primitiveFunctionTypeValue;
        dict[opKey] = static_cast<size_t>(m_op);
        dict[attributesKey] = Attributes();
        dict[uidKey] = Uid();
        if (!Name().empty())
            dict[nameKey] = Name();
        
        auto inputs = Inputs();
        vector<DictionaryValue> inputUids;
        inputUids.reserve(inputs.size());
        for (auto& input : inputs)
        {
            inputUids.push_back(input.Uid());
        }

        dict[inputsKey] = std::move(inputUids);

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
        if (op > PrimitiveOpType::NoOp)
        {
            LogicError("Unexpected op '%ls':'%u' (%s).", 
                        opKey.c_str(), 
                        static_cast<std::underlying_type<CNTK::PrimitiveOpType>::type>(op),
                        GetVersionsString<PrimitiveFunction>(s_serializationVersion, version).c_str());
        }

        const auto& uid = dict[uidKey].Value<std::wstring>();
        std::wstring name = L"";
        if (dict.Contains(nameKey))
            name = dict[nameKey].Value<std::wstring>();
        auto attributes = dict[attributesKey].Value<Dictionary>();
        const auto& inputUids = dict[inputsKey].Value<vector<DictionaryValue>>();

        std::vector<Variable> inputs;
        inputs.reserve(inputUids.size());

        for (const auto& dictionaryValue : inputUids)
        {
            const auto& inputUid = dictionaryValue.Value<std::wstring>();
            if (uidToVariableMap.find(inputUid) == uidToVariableMap.end())
            {
                LogicError("There are no inputs corresponding to input uid = '%ls' "
                        "(%s).", inputUid.c_str(), GetVersionsString<PrimitiveFunction>(s_serializationVersion, version).c_str());
            }
            inputs.push_back(uidToVariableMap.at(inputUid));
        }

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
                RuntimeError("Invalid block function dictionary found during deserialization; Number of block argument map keys does not match the number of map values");

            std::vector<std::pair<Variable, Variable>> argumentsMap;
            for (size_t i = 0; i < blockArgumentsMapKeys.size(); ++i)
                argumentsMap.push_back({ findCompositeArgumentByUid(blockArgumentsMapKeys[i]), uidToVariableMap.at(blockArgumentsMapValues[i]) });

            return std::shared_ptr<BlockFunction>(new BlockFunction(std::move(composite), argumentsMap, blockOpName, std::move(attributes), name, uid),
                                                  [](BlockFunction* ptr) { delete ptr; });
        }


        if (version < 4 && op == PrimitiveOpType::BatchNormalization)
        {
            if (Internal::GetComputationNetworkTraceLevel() > 0)
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
}
