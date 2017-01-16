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
            (op == PrimitiveOpType::SquaredError) ||
            (op == PrimitiveOpType::CrossEntropyWithSoftmax) ||
            (op == PrimitiveOpType::ClassificationError) ||
            (op == PrimitiveOpType::Logistic) ||
            (op == PrimitiveOpType::CosDistance) || 
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

    /*static*/ std::vector<Variable> PrimitiveFunction::GetOutputVariables(PrimitiveOpType op,
                                                                           std::vector<Variable>& inputs,
                                                                           Dictionary& functionConfig,
                                                                           bool inferDimensions,
                                                                           const std::wstring& functionName)
    {
        if (op == PrimitiveOpType::Combine)
            return inputs;

        DataType outputDataType = GetOutputDataType(op, inputs, inferDimensions);
        std::vector<Axis> outputDynamicAxes = GetOutputDynamicAxes(op, inputs, functionConfig);

        NDShape outputShape = NDShape::Unknown;
        bool allInputShapesUnknown = (std::find_if(inputs.begin(), inputs.end(), [](const Variable& input) { return !input.Shape().IsUnknown(); }) == inputs.end());
        bool anyInputShapesUnknown = (std::find_if(inputs.begin(), inputs.end(), [](const Variable& input) { return input.Shape().IsUnknown(); }) != inputs.end());
        if (!anyInputShapesUnknown || (!allInputShapesUnknown && (outputDynamicAxes != Axis::UnknownDynamicAxes())))
        {
            switch (op)
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
                assert(inputs.size() == 2);
                if ((op == PrimitiveOpType::PastValue) || (op == PrimitiveOpType::FutureValue))
                {
                    Variable inputOperandVar = inputs[0];
                    Variable initialStateVar = inputs[1];

                    // TODO: We currently only support input operand with 1 dynamic axis for PastValue/FutureValue
                    if ((inputOperandVar.DynamicAxes() != Axis::UnknownDynamicAxes()) && (inputOperandVar.DynamicAxes().size() != 2))
                        LogicError("Currently PastValue/FutureValue Function only supports input operand with 2 dynamic axis (1 sequence-axis and 1 batch-axis)");

                    if (!initialStateVar.DynamicAxes().empty())
                        LogicError("Currently PastValue/FutureValue Function does not support initial state operand with dynamic axes!");
                }

                outputShape = BinaryElementwiseOpOutputShape(op, inputs[0], inputs[1], true, inferDimensions);
                break;
            }
            case PrimitiveOpType::Clip:
                assert(inputs.size() == 3);
                outputShape = NaryElementwiseOpOutputShape(op, inputs, true, inferDimensions);
                break;
            case PrimitiveOpType::Select:
                assert(inputs.size() == 3);
                outputShape = NaryElementwiseOpOutputShape(op, inputs, true, inferDimensions);
                break;
            default:
                // For all other operations, shapes of all inputs must be known to determine the output shape
                if (!anyInputShapesUnknown)
                {
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
                    case PrimitiveOpType::Dropout:
                    case PrimitiveOpType::Where:
                    case PrimitiveOpType::LogSoftmax:
                    case PrimitiveOpType::Sin:
                    case PrimitiveOpType::Cos:
                    case PrimitiveOpType::Pass:
                        assert(inputs.size() == 1);
                        outputShape = UnaryElementwiseOpOutputShape(inputs[0].Shape());
                        break;
                    case PrimitiveOpType::PackedIndex:
                        assert(inputs.size() == 2);
                        outputShape = UnaryElementwiseOpOutputShape(inputs[1].Shape());
                        break;
                    case PrimitiveOpType::ScatterPacked:
                    {
                        assert(inputs.size() == 3);
                        if (inputs[0].DynamicAxes().empty() || inputs[1].DynamicAxes().empty() || inputs[2].DynamicAxes().empty())
                            InvalidArgument("ScatterPacked requires all its operands to have dynamic axes");

                        outputShape = UnaryElementwiseOpOutputShape(inputs[0].Shape());
                        break;
                    }
                    case PrimitiveOpType::TransposeAxes:
                    {
                        assert(inputs.size() == 1);

                        auto axis1 = NormalizeStaticAxis(functionConfig[PrimitiveFunction::AttributeNameAxis1].Value<Axis>(), inputs[0].Shape());
                        auto axis2 = NormalizeStaticAxis(functionConfig[PrimitiveFunction::AttributeNameAxis2].Value<Axis>(), inputs[0].Shape());

                        if (!axis1.IsStaticAxis() || !axis2.IsStaticAxis())
                            LogicError("TransposeAxes operation currently does not support transposing dynamic axes");

                        // We allow to transpose with an axes that exceeds the rank of the input.
                        // The output rank is the max of the input rank, and either of the axes being transposed.
                        auto outputRank = std::max(inputs[0].Shape().Rank(), (size_t)(std::max(axis1.StaticAxisIndex(), axis2.StaticAxisIndex()) + 1));
                        outputShape = inputs[0].Shape().AppendShape(NDShape(outputRank - inputs[0].Shape().Rank(), 1));
                        std::swap(outputShape[axis1.StaticAxisIndex()], outputShape[axis2.StaticAxisIndex()]);
                        break;
                    }
                    case PrimitiveOpType::Slice:
                    {
                        assert(inputs.size() == 1);
                        auto axis = NormalizeStaticAxis(functionConfig[PrimitiveFunction::AttributeNameAxis].Value<Axis>(), inputs[0].Shape());

                        auto beginIndex = functionConfig[PrimitiveFunction::AttributeNameBeginIndex].Value<int>();
                        auto endIndex = functionConfig[PrimitiveFunction::AttributeNameEndIndex].Value<int>();
                        if (!axis.IsStaticAxis())
                            LogicError("Built-in Slice operation currently does not support slicing along dynamic axis");

                        VerifyStaticAxis(axis, inputs[0].Shape());

                        size_t sliceAxisDim = inputs[0].Shape()[axis.StaticAxisIndex()];
                        int realBeginIndex = (beginIndex >= 0) ? beginIndex : beginIndex + sliceAxisDim;
                        int realEndIndex = (endIndex > 0) ? endIndex : endIndex + sliceAxisDim;
                        if ((sliceAxisDim < realEndIndex) || (realEndIndex < realBeginIndex) || (realBeginIndex < 0))
                            RuntimeError("Slice operation: Index range [%d,%d), interpreted as [%d,%d), is invalid for input's shape ([%S]).",
                                beginIndex,
                                endIndex,
                                realBeginIndex,
                                realEndIndex,
                                AsStringForErrorReporting(inputs[0].Shape()).c_str());

                        auto outputTensorShape = AsTensorShape(inputs[0].Shape());

                        // propagate as much as we can
                        if ((axis.StaticAxisIndex() < (int)outputTensorShape.GetRank()) && (0 <= realBeginIndex) && (realBeginIndex <= realEndIndex) && (realEndIndex <= sliceAxisDim))
                            outputTensorShape.NarrowTo(axis.StaticAxisIndex(), realBeginIndex, realEndIndex);

                        outputShape = AsNDShape(outputTensorShape, /*allowNonFlattenableTensorShapes = */ true);
                        break;
                    }
                    case PrimitiveOpType::Reshape:
                    {
                        auto& replacementShape = functionConfig[PrimitiveFunction::AttributeNameNewShape].Value<NDShape>();

                        auto beginAxis = Axis(0);
                        auto endAxis = Axis((int)inputs[0].Shape().Rank());
                        if (functionConfig.Contains(PrimitiveFunction::AttributeNameBeginAxis))
                            beginAxis = NormalizeStaticAxis(functionConfig[PrimitiveFunction::AttributeNameBeginAxis].Value<Axis>(), inputs[0].Shape());

                        if (functionConfig.Contains(PrimitiveFunction::AttributeNameEndAxis))
                            endAxis = NormalizeStaticAxis(functionConfig[PrimitiveFunction::AttributeNameEndAxis].Value<Axis>(), inputs[0].Shape());

                        outputShape = ReshapeOutputShape(inputs[0].Shape(), replacementShape, beginAxis, endAxis, inferDimensions);
                        break;
                    }
                    case PrimitiveOpType::ROIPooling:
                    {
                        assert(inputs.size() == 2);
                        auto convMapShape = inputs[0].Shape();
                        auto roisShape = inputs[1].Shape();
                        auto roiOutputShape = functionConfig[PrimitiveFunction::AttributeNameROIOutputShape].Value<NDShape>();

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
                        assert(inputs.size() == 1);
                        auto poolingWindowsShape = functionConfig[PrimitiveFunction::AttributeNamePoolingWindowShape].Value<NDShape>();
                        auto strides = functionConfig[PrimitiveFunction::AttributeNameStrides].Value<NDShape>();
                        auto lowerPad = functionConfig[PrimitiveFunction::AttributeNameLowerPad].Value<NDShape>();
                        auto upperPad = functionConfig[PrimitiveFunction::AttributeNameUpperPad].Value<NDShape>();
                        auto autoPadding = AsVector<bool>(functionConfig[PrimitiveFunction::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
                        NDShape outputMapCount = { 1 };
                        std::vector<bool> sharing = { true };
                        auto inputShape = inputs[0].Shape();

                        // In case of pooling if the kernel shape is unknown, then treat it as global pooling.
                        if (poolingWindowsShape == NDShape::Unknown)
                        {
                            if ((std::find(autoPadding.begin(), autoPadding.end(), true) != autoPadding.end()) ||
                                (lowerPad.TotalSize() > 0) || (upperPad.TotalSize() > 0))
                                RuntimeError("Padding isn't allowed for Unknown shape!");

                            poolingWindowsShape = inputShape.SubShape(0, inputShape.Rank() - 1);
                            functionConfig[PrimitiveFunction::AttributeNamePoolingWindowShape] = poolingWindowsShape;
                        }

                        outputShape = ConvolutionOpOutputShape(op, inputShape, poolingWindowsShape, outputMapCount, strides, sharing, autoPadding, lowerPad, upperPad, false, inferDimensions);
                        break;
                    }
                    case PrimitiveOpType::Unpooling:
                    {
                        assert(inputs.size() == 2);

                        auto inputShape = inputs[0].Shape();
                        outputShape = inputs[1].Shape();
                        PoolingType unpoolingType = (PoolingType)(functionConfig[PrimitiveFunction::AttributeNamePoolingType].Value<size_t>());
                        if (unpoolingType != PoolingType::Max)
                            LogicError("Only max unpooling is currently supported");

                        // Finding the shape of an unpooling operation from the input to be unpooled alone is ambiguous
                        // For example a 4x4 input with a 5x5 kernel a stride of 2x2
                        // and padding could have resulted from pooling a 7x7 or 8x8 image
                        // Therefore what needs to happen here is to check whether the
                        // outputShape can be pooled into the inputShape using the specified attributes
                        auto unpoolingWindowShape = functionConfig[PrimitiveFunction::AttributeNameUnpoolingWindowShape].Value<NDShape>();
                        auto strides = functionConfig[PrimitiveFunction::AttributeNameStrides].Value<NDShape>();
                        auto lowerPad = functionConfig[PrimitiveFunction::AttributeNameLowerPad].Value<NDShape>();
                        auto upperPad = functionConfig[PrimitiveFunction::AttributeNameUpperPad].Value<NDShape>();
                        auto autoPadding = AsVector<bool>(functionConfig[PrimitiveFunction::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
                        NDShape inputMapCount = { 1 };
                        std::vector<bool> sharing = { true };

                        NDShape inferredInputShape = ConvolutionOpOutputShape(PrimitiveOpType::Pooling, outputShape, unpoolingWindowShape, inputMapCount, strides, sharing, autoPadding, lowerPad, upperPad, false, inferDimensions);
                        if (inferredInputShape != inputShape)
                            RuntimeError("The shape of the unpooling operand %ls is different from the result of pooling the poolingInput argument using the provided options %ls", inputShape.AsString().c_str(), inferredInputShape.AsString().c_str());

                        break;
                    }
                    case PrimitiveOpType::SumAll:
                        assert(inputs.size() == 1);
                        outputShape = { 1 };
                        break;
                    case PrimitiveOpType::Times:
                    {
                        assert(inputs.size() == 2);
                        auto outputRank = functionConfig[PrimitiveFunction::AttributeNameOutputRank].Value<size_t>();
                        auto inferInputRankToMap = functionConfig[PrimitiveFunction::AttributeNameInferInputRankToMap].Value<int>();
                        outputShape = TimesOpOutputShape(inputs[0], inputs[1], outputRank, inferInputRankToMap, inferDimensions);
                        break;
                    }
                    case PrimitiveOpType::TransposeTimes:
                    {
                        assert(inputs.size() == 2);

                        auto transposeShapeFunc = [](const NDShape& shape) {
                            NDShape transposedShape(std::max<size_t>(2, shape.Rank()), 1);
                            for (size_t i = 0; i < shape.Rank(); ++i)
                                transposedShape[transposedShape.Rank() - i - 1] = shape[i];

                            return transposedShape;
                        };

                        if (inputs[0].Shape().Rank() > 2)
                            LogicError("TransposeTimes operation currently only supports %s operands of rank 1 or 2", Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "right" : "left");

                        NDShape transposedLeftOperandShape = transposeShapeFunc(inputs[0].Shape());
                        Variable dummyLeftOperand = PlaceholderVariable(transposedLeftOperandShape);
                        size_t outputRank = functionConfig[PrimitiveFunction::AttributeNameOutputRank].Value<size_t>();
                        outputShape = TimesOpOutputShape(dummyLeftOperand, inputs[1], outputRank, -1, inferDimensions);
                        if (dummyLeftOperand.Shape() != transposedLeftOperandShape)
                            inputs[0].m_dataFields->m_shape = transposeShapeFunc(dummyLeftOperand.Shape());

                        break;
                    }
                    case PrimitiveOpType::Convolution:
                    {
                        assert(inputs.size() == 2);
                        auto& strides = functionConfig[PrimitiveFunction::AttributeNameStrides].Value<NDShape>();
                        auto& lowerPad = functionConfig[PrimitiveFunction::AttributeNameLowerPad].Value<NDShape>();
                        auto& upperPad = functionConfig[PrimitiveFunction::AttributeNameUpperPad].Value<NDShape>();
                        auto sharing = AsVector<bool>(functionConfig[PrimitiveFunction::AttributeNameSharing].Value<std::vector<DictionaryValue>>());
                        auto autoPadding = AsVector<bool>(functionConfig[PrimitiveFunction::AttributeNameAutoPadding].Value<std::vector<DictionaryValue>>());
                        bool transpose = functionConfig[PrimitiveFunction::AttributeNameTranspose].Value<bool>();
                        if (inputs[0].Shape().Rank() < inputs[1].Shape().Rank())
                            InvalidArgument("The convolution map should have at least as many axes as the shape of the input it operates on!");

                        NDShape outputMapCount, kernelShape;
                        std::tie(outputMapCount, kernelShape) = GetConvolutionOutputMapCountAndKernelShape(inputs[0].Shape(), inputs[1].Shape(), transpose);
                        auto originalKernelShape = kernelShape;
                        outputShape = ConvolutionOpOutputShape(op, inputs[1].Shape(), kernelShape, outputMapCount, strides, sharing, autoPadding, lowerPad, upperPad, transpose, inferDimensions);
                        if (originalKernelShape != kernelShape)
                        {
                            for (size_t i2 = 0; i2 < kernelShape.Rank(); ++i2)
                                inputs[0].m_dataFields->m_shape[i2] = kernelShape[i2];
                        }

                        functionConfig[PrimitiveFunction::AttributeNameSharing] = AsDictionaryValueVector(sharing);
                        functionConfig[PrimitiveFunction::AttributeNameAutoPadding] = AsDictionaryValueVector(autoPadding);
                        break;
                    }
                    case PrimitiveOpType::CosDistance:
                    case PrimitiveOpType::Logistic:
                    case PrimitiveOpType::SquaredError:
                    case PrimitiveOpType::CrossEntropyWithSoftmax:
                    case PrimitiveOpType::ClassificationError:
                    case PrimitiveOpType::LambdaRank:
                    case PrimitiveOpType::NDCG:
                    {
                        if ((op == PrimitiveOpType::ClassificationError) || (op == PrimitiveOpType::Logistic))
                            assert(inputs.size() >= 2);
                        else if ((op == PrimitiveOpType::LambdaRank) || (op == PrimitiveOpType::NDCG))
                            assert(inputs.size() == 3);
                        else
                            assert(inputs.size() == 2);

                        if ((inputs[0].Shape().Rank() > 2) || ((inputs[0].Shape().Rank() > 1) && (inputs[0].Shape()[1] != 1)))
                            InvalidArgument("The shape of input operands for the %S operation should have at most one axis", PrimitiveOpTypeName(op).c_str());

                        auto predictionShape = inputs[0].Shape();
                        auto labelsShape = inputs[1].Shape();
                        if (predictionShape != labelsShape)
                            RuntimeError("Prediction output operand's shape %S is incompatible with label operand's shape %S for the %S operation", AsStringForErrorReporting(predictionShape).c_str(), AsStringForErrorReporting(labelsShape).c_str(), PrimitiveOpTypeName(op).c_str());

                        std::vector<int> reductionAxes;
                        for (int i3 = 0; i3 < (int)inputs[0].Shape().Rank(); ++i3)
                            reductionAxes.push_back(i3);

                        outputShape = ReductionOpOutputShape(op, predictionShape, reductionAxes, /*preserveReductionAxes =*/ false);
                        break;
                    }
                    case PrimitiveOpType::ReduceElements:
                    {
                        assert(inputs.size() == 1);
                        auto reductionAxis = NormalizeStaticAxis(functionConfig[PrimitiveFunction::AttributeNameAxis].Value<Axis>(), inputs[0].Shape());
                        if (reductionAxis == Axis::AllStaticAxes())
                            outputShape = {};
                        else
                        {
                            std::vector<int> reductionAxes = { reductionAxis.StaticAxisIndex() };
                            outputShape = ReductionOpOutputShape(op, inputs[0].Shape(), reductionAxes, /*preserveReductionAxes =*/ true);
                        }
                        break;
                    }
                    case PrimitiveOpType::BatchNormalization:
                    {
                        assert(inputs.size() == 5);
                        auto spatial = functionConfig[PrimitiveFunction::AttributeNameSpatial].Value<bool>();
                        outputShape = BatchNormalizationOutputShape(inputs, spatial, inferDimensions);
                        break;
                    }
                    case PrimitiveOpType::GatherPacked:
                    {
                        bool sourceHasDynamicAxis = !inputs[0].DynamicAxes().empty();

                        // inherit tensor dimension from sourceData, minus the last (column or time) dimension. TODO this needs to become simpler...
                        if (sourceHasDynamicAxis)
                            outputShape = inputs[0].Shape();
                        else
                        {
                            if (inputs[0].Shape().Rank() > 1)
                                outputShape = outputShape.SubShape(0, outputShape.Rank() - 1);
                            else
                                outputShape = {};
                        }

                        break;
                    }
                    case PrimitiveOpType::Splice:
                    {
                        assert(inputs.size() >= 2);
                        auto maxInputRank = MaxInputRank(inputs);
                        auto spliceAxis = NormalizeStaticAxis(functionConfig[PrimitiveFunction::AttributeNameAxis].Value<Axis>(), NDShape(maxInputRank));

                        if (!spliceAxis.IsStaticAxis())
                            LogicError("Splice operation currently does not support splicing along dynamic axis");

                        if (spliceAxis.StaticAxisIndex() < 0)
                            InvalidArgument("Splice: The axis argument's static axis index must be >= 0!");

                        outputShape = SpliceOutputShape(inputs, spliceAxis.StaticAxisIndex());
                        break;
                    }
                    case PrimitiveOpType::RandomSample:
                    case PrimitiveOpType::RandomSampleInclusionFrequency:
                    {
                        auto numSamples = functionConfig[PrimitiveFunction::AttributeNameNumSamples].Value<size_t>();
                        auto allowDuplicates = functionConfig[PrimitiveFunction::AttributeNameAllowDuplicates].Value<bool>();

                        if (numSamples == 0)
                            InvalidArgument("Number of requested samples is zero.");

                        let& shape = inputs[0].Shape();
                        size_t numClasses = shape.Dimensions()[0];

                        if (numClasses != NDShape::InferredDimension && !allowDuplicates && numClasses <= numSamples)
                            InvalidArgument("For sampling without duplicates the number of requested samples (%lu) needs to be less than the number of classes (%lu).", numSamples, numClasses);

                        // within this block we handle RandomSample and RandomSampleInclusionFrequency
                        if (op == PrimitiveOpType::RandomSampleInclusionFrequency)
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
                        assert(inputs.size() == 2);
                        auto operand = inputs[0];
                        auto parameter = inputs[1];
                        if (operand.Shape().Rank() != 1)
                            InvalidArgument("OptimizedRNNStack: input must have rank 1; actual input rank is %lu", operand.Shape().Rank());
                        if (operand.DynamicAxes().empty())
                            InvalidArgument("OptimizedRNNStack: input must have at least one dynamic axis");
                        auto numLayers = functionConfig[PrimitiveFunction::AttributeNameNumLayers].Value<size_t>();
                        if (numLayers == 0)
                            InvalidArgument("Number of layers in OptimizedRNNStack operation should be positive");
                        auto bidirectional = functionConfig[PrimitiveFunction::AttributeNameBidirectional].Value<bool>();
                        auto hiddenSize = functionConfig[PrimitiveFunction::AttributeNameHiddenSize].Value<size_t>();

                        // output dims
                        outputShape = operand.Shape();
                        outputShape[0] = (bidirectional ? 2 : 1) * hiddenSize;
                        // infer input size
                        // Note: Output dim is second axis, so say initOutputRank=-1.
                        if (parameter.Shape().Rank() == 2)
                        {
                            const auto recurrentOp = functionConfig[PrimitiveFunction::AttributeNameRecurrentOp].Value<std::wstring>();
                            const auto attributes = RnnAttributes(bidirectional, numLayers, hiddenSize, recurrentOp, -1);
                            const auto numParameters = attributes.GetNumParameters(operand.Shape().TotalSize());
                            std::vector<std::pair<Variable, NDShape>> newOperandShapes = { { parameter, std::move(NDShape({ numParameters.first, numParameters.second })) } };
                            UpdateOperandShapes(newOperandShapes);
                        }
                        break;
                    }
                    case PrimitiveOpType::ReconcileDynamicAxis:
                    {
                        assert(inputs.size() == 2);
                        auto operand = inputs[0];
                        auto layout = inputs[1];
                        if (operand.DynamicAxes().empty())
                            InvalidArgument("ReconcileDynamicAxis: input must have at least one dynamic axis");
                        if (layout.DynamicAxes().empty())
                            InvalidArgument("ReconcileDynamicAxis: layout must have at least one dynamic axis");
                        outputShape = operand.Shape();
                        break;
                    }
                    default:
                        LogicError("Specified op %S not yet supported", PrimitiveOpTypeName(op).c_str());
                        break;
                    }
                }
            }
        }

        return{ OutputVariable(outputShape, outputDataType, outputDynamicAxes, functionName.empty() ? L"" : functionName) };
    }

    /*virtual*/ std::vector<Variable> PrimitiveFunction::GetOutputVariables(bool inferDimensions)
    {
        return GetOutputVariables(m_op, m_inputs, m_attributes, inferDimensions, Name());
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
        dict[nameKey] = Name();
        
        auto inputs = Inputs();
        vector<DictionaryValue> inputUids;
        inputUids.reserve(inputs.size());
        for (auto& input : inputs)
        {
            inputUids.push_back(input.Uid());
        }

        dict[inputsKey] = std::move(inputUids);

        if (m_op == PrimitiveOpType::Unpooling)
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
        static const vector<std::wstring> s_requiredDictionaryKeys = { typeKey, opKey, uidKey, attributesKey, inputsKey, nameKey };
        size_t version = ValidateDictionary<PrimitiveFunction>(dict, s_requiredDictionaryKeys, s_primitiveFunctionTypeValue, s_serializationVersion);

        PrimitiveOpType op = PrimitiveOpType(dict[opKey].Value<std::size_t>());

        // The hard requirement that the serialization depends on is that
        // new op type values are only added to the end of the list, after Combine.
        // This also applies to other enums (DataType, VariableKind, etc.)
        if (op > PrimitiveOpType::NDCG)
        {
            LogicError("Unexpected op '%ls':'%u' (%s).", 
                        opKey.c_str(), 
                        static_cast<std::underlying_type<CNTK::PrimitiveOpType>::type>(op),
                        GetVersionsString<PrimitiveFunction>(s_serializationVersion, version).c_str());
        }

        const auto& uid = dict[uidKey].Value<std::wstring>();
        const auto& name = dict[nameKey].Value<std::wstring>();
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
        else
            return std::shared_ptr<PrimitiveFunction>(new PrimitiveFunction(op, inputs, std::move(attributes), name, uid), 
                                                      [](PrimitiveFunction* ptr) { delete ptr; });
    }
}
