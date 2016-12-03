//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "PrimitiveOpType.h"
#include "Utils.h"
#include "ConvolveGeometry.h"
#include "ConvolutionalNodes.h"

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
    // Move primitiveOpNames out from PrimitiveOpTypeName(), as local static variables are not thread-safe under VS2013.
    // Todo: Move it into PrimitiveOpTypeName() as local static after upgraded to VS2015.
    static const std::unordered_map<PrimitiveOpType, std::wstring> primitiveOpNames = {
        {PrimitiveOpType::Negate, L"Negate"},
        {PrimitiveOpType::Sigmoid, L"Sigmoid"},
        {PrimitiveOpType::Tanh, L"Tanh"},
        {PrimitiveOpType::ReLU, L"ReLU"},
        {PrimitiveOpType::Exp, L"Exp"},
        {PrimitiveOpType::Log, L"Log"},
        {PrimitiveOpType::Sqrt, L"Sqrt"},
        {PrimitiveOpType::Floor, L"Floor"},
        {PrimitiveOpType::Abs, L"Abs"},
        {PrimitiveOpType::Reciprocal, L"Reciprocal"},
        {PrimitiveOpType::Softmax, L"Softmax"},
        {PrimitiveOpType::Hardmax, L"Hardmax"},
        {PrimitiveOpType::TransposeAxes, L"TransposeAxes"},
        {PrimitiveOpType::Where, L"Where"},
        {PrimitiveOpType::Slice, L"Slice"},
        {PrimitiveOpType::Dropout, L"Dropout"},
        {PrimitiveOpType::Reshape, L"Reshape"},
        {PrimitiveOpType::Pooling, L"Pooling"},
        {PrimitiveOpType::SumAll, L"SumAll"},
        {PrimitiveOpType::Plus, L"Plus"},
        {PrimitiveOpType::LogPlus, L"LogPlus"},
        {PrimitiveOpType::Minus, L"Minus"},
        {PrimitiveOpType::ElementTimes, L"ElementTimes"},
        {PrimitiveOpType::Equal, L"Equal"},
        {PrimitiveOpType::NotEqual, L"NotEqual"},
        {PrimitiveOpType::Less, L"Less"},
        {PrimitiveOpType::LessEqual, L"LessEqual"},
        {PrimitiveOpType::Greater, L"Greater"},
        {PrimitiveOpType::GreaterEqual, L"GreaterEqual"},
        {PrimitiveOpType::PackedIndex, L"PackedIndex"},
        {PrimitiveOpType::GatherPacked, L"GatherPacked"},
        {PrimitiveOpType::ScatterPacked, L"ScatterPacked"},
        {PrimitiveOpType::Times, L"Times"},
        {PrimitiveOpType::TransposeTimes, L"TransposeTimes"},
        {PrimitiveOpType::Convolution, L"Convolution"},
        {PrimitiveOpType::SquaredError, L"SquaredError"},
        {PrimitiveOpType::CrossEntropyWithSoftmax, L"CrossEntropyWithSoftmax"},
        {PrimitiveOpType::ClassificationError, L"ClassificationError"},
        {PrimitiveOpType::PastValue, L"PastValue"},
        {PrimitiveOpType::FutureValue, L"FutureValue"},
        {PrimitiveOpType::ReduceElements, L"ReduceElements"},
        {PrimitiveOpType::BatchNormalization, L"BatchNormalization"},
        {PrimitiveOpType::Clip, L"Clip"},
        {PrimitiveOpType::Select, L"Select"},
        {PrimitiveOpType::Splice, L"Splice"},
        {PrimitiveOpType::Combine, L"Combine"},
        {PrimitiveOpType::RandomSample, L"RandomSample"},
        {PrimitiveOpType::RandomSampleInclusionFrequency, L"RandomSampleInclusionFrequency"},
        {PrimitiveOpType::ROIPooling, L"ROIPooling"},
        {PrimitiveOpType::Logistic, L"Logistic"},
        {PrimitiveOpType::OptimizedRNNStack, L"OptimizedRNNStack"},
        {PrimitiveOpType::ReconcileDynamicAxis, L"ReconcileDynamicAxis"},
        {PrimitiveOpType::LogSoftmax, L"LogSoftmax"},
        {PrimitiveOpType::CosDistance, L"CosDistance"},
        {PrimitiveOpType::Sin, L"Sin"},
        {PrimitiveOpType::Cos, L"Cos"},
        {PrimitiveOpType::Pass, L"Pass"},
    };

    inline const std::wstring& PrimitiveOpTypeName(PrimitiveOpType opType)
    {
        if (primitiveOpNames.find(opType) == primitiveOpNames.end())
            LogicError("Unknown PrimitiveOpType");

        return primitiveOpNames.find(opType)->second;
    }

    inline std::wstring GenerateUid(PrimitiveOpType opType)
    {
        return Internal::GenerateUid(PrimitiveOpTypeName(opType));
    }

    inline std::unordered_map<size_t, size_t> GetPrimitiveFunctionInputsToCNTKNodeInputsIndexMap(PrimitiveOpType op, size_t numFunctionInputs)
    {
        std::unordered_map<size_t, size_t> indexMap;
        if (op == PrimitiveOpType::ClassificationError)
        {
            indexMap = std::unordered_map<size_t, size_t>({ { 0, 1 }, { 1, 0 } });
            if (numFunctionInputs > 2)
                indexMap.insert({2, 2});
        }
        else if (op == PrimitiveOpType::Logistic)
        {
            indexMap = std::unordered_map<size_t, size_t>({ { 0, 1 }, { 1, 0 } });
            if (numFunctionInputs > 2)
                indexMap.insert({ 2, 2 });
        }
        else if (op == PrimitiveOpType::CrossEntropyWithSoftmax)
            indexMap = std::unordered_map<size_t, size_t>({ { 0, 1 }, { 1, 0 } });
        else if (op == PrimitiveOpType::GatherPacked)
            indexMap = std::unordered_map<size_t, size_t>({ { 0, 1 }, { 1, 0 } });
        else if (op == PrimitiveOpType::ScatterPacked)
            indexMap = std::unordered_map<size_t, size_t>({ { 0, 2 }, { 1, 1 }, { 2, 0 } });
        else if (op == PrimitiveOpType::Clip)
            indexMap = std::unordered_map<size_t, size_t>({ { 0, 2 }, { 1, 0 }, { 2, 1 } });
        else if (op == PrimitiveOpType::OptimizedRNNStack)
            indexMap = std::unordered_map<size_t, size_t>({ { 0, 1 }, { 1, 0 } });
        else
        {
            for (size_t i = 0; i < numFunctionInputs; ++i)
                indexMap.insert(std::make_pair(i, i));
        }

        if (indexMap.size() != numFunctionInputs)
            LogicError("Size of the PrimitiveFunctionInputsToCNTKNodeInputsIndexMap does not match the actual number of Inputs of the PrimitiveFunction");

        for (auto indexPair : indexMap)
        {
            if ((indexPair.first >= numFunctionInputs) || (indexPair.second >= numFunctionInputs))
                LogicError("The index values in the PrimitiveFunctionInputsToCNTKNodeInputsIndexMap cannot be >= the number of Inputs of the PrimitiveFunction");
        }

        return indexMap;
    }

    template <typename T>
    inline void ReorderAsCNTKComputationNodeInputs(PrimitiveOpType op, std::vector<T>& vec)
    {
        auto indexMap = GetPrimitiveFunctionInputsToCNTKNodeInputsIndexMap(op, vec.size());
        auto vecCopy = vec;

        for (auto indexPair : indexMap)
            vec[indexPair.second] = vecCopy[indexPair.first];
    }

    inline void ReorderAsPrimitiveFunctionInputs(PrimitiveOpType op, std::vector<Variable>& vec)
    {
        auto indexMap = GetPrimitiveFunctionInputsToCNTKNodeInputsIndexMap(op, vec.size());
        auto vecCopy = vec;

        for (auto indexPair : indexMap)
            vec[indexPair.first] = vecCopy[indexPair.second];
    }

    class PrimitiveFunction final : public Function
    {
        friend class Function;
        template <typename T, typename ...CtorArgTypes>
        friend inline std::shared_ptr<T> MakeSharedObject(CtorArgTypes&& ...ctorArgs);

    public:
        static const std::wstring InternalSumReductionOpName;
        static const std::wstring InternalLogSumReductionOpName;
        static const std::wstring InternalMeanReductionOpName;
        static const std::wstring InternalMaxReductionOpName;
        static const std::wstring InternalMinReductionOpName;
        static const std::wstring InternalAllReductionOpName;
        static const std::wstring InternalAnyReductionOpName;

        static const std::wstring AttributeNameAxis;
        static const std::wstring AttributeNameAxis1;
        static const std::wstring AttributeNameAxis2;
        static const std::wstring AttributeNameAllowDuplicates;
        static const std::wstring AttributeNameNumSamples;
        static const std::wstring AttributeNameDropoutRate;
        static const std::wstring AttributeNameNewShape;
        static const std::wstring AttributeNameOutputRank;
        static const std::wstring AttributeNameInferInputRankToMap;
        static const std::wstring AttributeNameOffset;
        static const std::wstring AttributeNameStrides;
        static const std::wstring AttributeNameSharing;
        static const std::wstring AttributeNameAutoPadding;
        static const std::wstring AttributeNameLowerPad;
        static const std::wstring AttributeNameUpperPad;
        static const std::wstring AttributeNameTranspose;
        static const std::wstring AttributeNameMaxTempMemSizeInSamples;
        static const std::wstring AttributeNameROIOutputShape;
        static const std::wstring AttributeNamePoolingType;
        static const std::wstring AttributeNamePoolingWindowShape;
        static const std::wstring AttributeNameSpatial;
        static const std::wstring AttributeNameNormalizationTimeConstant;
        static const std::wstring AttributeNameBlendTimeConstant;
        static const std::wstring AttributeNameEpsilon;
        static const std::wstring AttributeNameUseCuDNNEngine;
        static const std::wstring AttributeNameNewDynamicAxes;
        static const std::wstring AttributeNameNewSequenceAxisLengthScalingFactor;
        static const std::wstring AttributeNameNewSequenceAxisLengthAdditiveFactor;
        static const std::wstring AttributeNameBeginIndex;
        static const std::wstring AttributeNameEndIndex;
        static const std::wstring AttributeNameReductionOpName;
        static const std::wstring AttributeNameRngSeed;
        static const std::wstring AttributeNameRngOffset;
        static const std::wstring AttributeNameBidirectional;
        static const std::wstring AttributeNameNumLayers;
        static const std::wstring AttributeNameHiddenSize;
        static const std::wstring AttributeNameRecurrentOp;

    public:
        PrimitiveFunction(PrimitiveOpType op, std::vector<Variable>& inputs, Dictionary&& functionConfig, const std::wstring& functionName = L"")
            : PrimitiveFunction(op, inputs, std::move(functionConfig), functionName, GenerateUid(op))
        {}

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

        virtual Dictionary Serialize() const override;

        virtual size_t CurrentVersion() const override { return s_serializationVersion; }

        static FunctionPtr Deserialize(const Dictionary& dictionary, 
                                       const std::unordered_map<std::wstring, Variable>& uidToVariableMap, 
                                       const CNTK::DeviceDescriptor& device);

        virtual const std::wstring& OpName() override
        {
            return PrimitiveOpTypeName(OpType());
        }

    public:
        PrimitiveOpType OpType() const
        {
            return m_op;
        }

        bool IsStateful() const
        {
            return (OpType() == PrimitiveOpType::Dropout) ||
                   (OpType() == PrimitiveOpType::RandomSample) ||
                   (OpType() == PrimitiveOpType::RandomSampleInclusionFrequency);
        }

    private:

        PrimitiveFunction(PrimitiveOpType op, std::vector<Variable>& inputs, Dictionary&& functionConfig, const std::wstring& functionName, const std::wstring& uid)
            : Function(inputs, GetOutputVariables(op, inputs, this, functionConfig, true, (functionName != L"" ? functionName : uid)), std::move(functionConfig), functionName, uid), m_op(op)
        {}

        // The following helper functions are used to determine the output shape for different 
        // types of primitive operations accounting for broadcasting and reductions where applicable.
        static NDShape UnaryElementwiseOpOutputShape(const NDShape& operandShape)
        {
            return operandShape;
        }

        static NDShape ReshapeOutputShape(const NDShape& operandShape, const NDShape& newShape)
        {
            size_t inputElementsCount = 1;
            for (size_t k = 0; k < operandShape.Rank(); k++)
                inputElementsCount *= operandShape[k];

            auto outputShape = newShape;
            size_t targetElementsCount = 1;
            size_t inferredAxisIndex = SIZE_MAX;
            for (size_t k = 0; k < outputShape.Rank(); k++)
            {
                if (outputShape[k] != NDShape::InferredDimension)
                    targetElementsCount *= outputShape[k];
                else if (inferredAxisIndex == SIZE_MAX)
                    inferredAxisIndex = k;
                else
                    InvalidArgument("CNTK::Reshape: More than one axis's dimension was specified as Inferred in the replacement shape %S", AsStringForErrorReporting(outputShape).c_str());
            }
            if (inferredAxisIndex != SIZE_MAX)
                outputShape[inferredAxisIndex] = inputElementsCount / targetElementsCount;

            return outputShape;
        }

        static size_t MaxInputRank(const std::vector<Variable>& inputs)
        {
            size_t maxRank = 0;
            for (int i = 0; i < inputs.size(); i++)
            {
                auto inputRank = inputs[i].Shape().Rank();
                if (maxRank < inputRank)
                    maxRank = inputRank;
            }

            return maxRank;
        }

        static NDShape SpliceOutputShape(const std::vector<Variable>& inputs, size_t axis)
        {
            // We must fuse all tensor shapes

            // Determine maximum rank (we can stack tensors with lower rank, which will have their dimensions paded to max automatically)
            auto maxInputRank = MaxInputRank(inputs);
            size_t maxRank = std::max<size_t>(axis + 1, maxInputRank); // spliceDim may exceed all of them, which will create a new dimension, e.g. stacking column vectors into a matrix

            // The following loop does multiple things:
            //  - Count total dimension along index
            //  - Verify all other dimension's compatibility (we allow broadcasting)

            // dimensions padded to max rank; start with dims of first input
            auto outputDims = inputs[0].Shape().AppendShape(NDShape(maxRank - inputs[0].Shape().Rank(), 1));

            // This dimension is created, while all others are verified for consistency
            size_t index = axis;
            outputDims[index] = 0;
            for (int i = 0; i < inputs.size(); i++)
            {
                // check/fuse dims and accumulate the spliced dimension
                auto& shape = inputs[i].Shape();
                for (size_t k = 0; k < maxRank; k++)
                {
                    size_t dim = (k >= shape.Rank()) ? 1 : shape[k];
                    // accumulate the spliced dimension
                    if (k == index)
                    {
                        if (dim == NDShape::InferredDimension)
                            outputDims[index] = NDShape::InferredDimension;
                        else
                            outputDims[index] += dim;
                    }
                    else
                    {
                        // check dimensions
                        if ((outputDims[k] == NDShape::InferredDimension) || (outputDims[k] == 1))
                            outputDims[k] = dim; // Broadcast
                        else if ((dim != outputDims[k]) && (dim != 1) && (dim != NDShape::InferredDimension))
                            InvalidArgument("CNTK::Splice: Conflicting dimension of axis %d between operand #%d (%d) and other(s) (%d)", (int)k, i, (int)dim, (int)outputDims[k]);
                    }
                }
            }

            return outputDims;
        }

        // Returns a boolean indicating if any operand shape was updated
        static bool UpdateOperandShapes(std::vector<std::pair<Variable, NDShape>>& newOperandShapes)
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

        // Returns a pair comprising of the output shape and boolean indicating if any input operand shape was modified
        static NDShape BinaryElementwiseOpOutputShape(PrimitiveOpType op, Variable& leftOperand, Variable& rightOperand, bool broadcastAllowed, bool inferInputDimensions)
        {
            auto leftOperandShape = leftOperand.Shape();
            auto rightOperandShape = rightOperand.Shape();

            // All operand shapes should be known
            assert((leftOperandShape != NDShape::Unknown) && (rightOperandShape != NDShape::Unknown));

            const auto& shapeWithSmallerNumAxes = (leftOperandShape.Rank() > rightOperandShape.Rank()) ? rightOperandShape : leftOperandShape;
            const auto& shapeWithLargerNumAxes = (leftOperandShape.Rank() > rightOperandShape.Rank()) ? leftOperandShape : rightOperandShape;
            size_t numOutputAxes = shapeWithLargerNumAxes.Rank();
            std::vector<size_t> outputDims(numOutputAxes);
            for (size_t i = 0; i < shapeWithSmallerNumAxes.Rank(); ++i)
            {
                if ((leftOperandShape[i] == NDShape::InferredDimension) && (rightOperandShape[i] == NDShape::InferredDimension))
                    outputDims[i] = NDShape::InferredDimension;
                else if ((leftOperandShape[i] == NDShape::InferredDimension) || (leftOperandShape[i] == 1))
                {
                    outputDims[i] = rightOperandShape[i];
                    if (leftOperandShape[i] == NDShape::InferredDimension)
                        leftOperandShape[i] = rightOperandShape[i];
                }
                else if ((rightOperandShape[i] == NDShape::InferredDimension) || (rightOperandShape[i] == 1))
                {
                    outputDims[i] = leftOperandShape[i];
                    if (rightOperandShape[i] == NDShape::InferredDimension)
                        rightOperandShape[i] = leftOperandShape[i];
                }
                else
                {
                    if (leftOperandShape[i] != rightOperandShape[i])
                        RuntimeError("Left operand's shape %S is not compatible with right operand's shape %S for the binary elementwise operation %S",
                                     AsStringForErrorReporting(leftOperandShape).c_str(),
                                     AsStringForErrorReporting(rightOperandShape).c_str(),
                                     PrimitiveOpTypeName(op).c_str());

                    outputDims[i] = leftOperandShape[i];
                }
            }

            // Broadcast in remaining axes
            for (size_t i = shapeWithSmallerNumAxes.Rank(); i < numOutputAxes; ++i)
                outputDims[i] = shapeWithLargerNumAxes[i];

            // See if we need to infer and propagate dimensions of any of the parameter operands
            if (inferInputDimensions)
            {
                std::vector<std::pair<Variable, NDShape>> newOperandShapes = { { leftOperand, leftOperandShape }, { rightOperand, rightOperandShape } };
                UpdateOperandShapes(newOperandShapes);
            }

            return NDShape(std::move(outputDims));
        }

        static NDShape NaryElementwiseOpOutputShape(PrimitiveOpType op, std::vector<Variable>& operands, bool broadcastAllowed)
        {
            assert(operands.size() > 1);

            // TODO: Is this logic of transitively constructing the output shape from the operands correct?
            Variable dummyOutputVariable = PlaceholderVariable(NDShape());
            for (auto& operand : operands)
                dummyOutputVariable.m_dataFields->m_shape = BinaryElementwiseOpOutputShape(op, dummyOutputVariable, operand, broadcastAllowed, false);

            return dummyOutputVariable.m_dataFields->m_shape;
        }

        // Returns a pair comprising of the output shape and boolean indicating if any input operand shape was modified
        static NDShape TimesOpOutputShape(Variable& leftOperand, Variable& rightOperand, size_t outputRank, int inferInputRankToMap, bool inferInputDimensions)
        {
            auto leftOperandShape = leftOperand.Shape();
            auto rightOperandShape = rightOperand.Shape();

            if (outputRank == 0)
                InvalidArgument("Output rank of times operation should be at least one");

            if (outputRank > leftOperandShape.Rank())
                InvalidArgument("Output rank of times operation can at most be the rank of the %s operand", Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "right" : "left");

            if (inferInputRankToMap >= (int)(rightOperandShape.Rank()))
                InvalidArgument("Input map rank of times operation must be less than the rank of the %s operand", Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "left" : "right");

            size_t numReductionAxes = leftOperandShape.Rank() - outputRank;

            // The 'numReductionAxes' trailing dimensions of the left operand's shape must match the corresponding leading
            // dimensions of the right operand

            if (rightOperandShape.Rank() < numReductionAxes)
                RuntimeError("The %s operand's rank in a times operation should not be less than #axes being reduced over!", Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "left" : "right");

            // outputRank dimensions cannot be inferred
            for (size_t k = 0; k < outputRank; k++)
            {
                if (leftOperandShape[k] == NDShape::InferredDimension)
                    InvalidArgument("The outputRank (%d) dimensions in times operation's %s operand's shape [%S] cannot be Inferred.",
                                    (int)outputRank,
                                    Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "right" : "left",
                                    AsStringForErrorReporting(leftOperandShape).c_str());
            }

            // infer rank of leftOperand
            // For purpose of dimension inference, Times() accepts an optional parameter inferInputRankToMap (default -1=unspecified).
            // The last 'inferInputRankToMap' axes are considered those that the matrix product should keep (Times()
            // is applied one by one, like a "map" operation) rather than reducing over.
            // Specifically, inferInputRankToMap=0 means to reduce over all input axes, e.g. for an image input that
            // should be flattened.
            // Examples:
            //  [I x Inferred] * [J x K],                    inferInputRankToMap=n/a --> Inferred  := J, result is [I x K]
            //  [I x Inferred] * [W x H x C],                inferInputRankToMap=n/a --> Inferred  := W, result is [I x H x C] (not desired)
            //  [I x Inferred x Inferred] * [W x H x C],     inferInputRankToMap=n/a --> Inf x Inf := [W x H], result is [I x C]
            //  [I x Inferred] * [W x H x C],                inferInputRankToMap=0   --> Inferred  := W x H x C, result is [I] (desired)
            //  [I x Inferred] * [W x H x C x R],            inferInputRankToMap=1   --> Inferred  := W x H x C, result is [I x R] (desired)
            // If W's shape is too short, it will be padded with 0 (i.e. inferred in a subsequent step).
            // (the second check below (dimsA.back() == 0) is required to infer dimensions correctly for fixed input tensors where a new dimension is added,
            // e.g. when adding an ROI dimension to a pretrained weights tensor of a dense layer after ROI pooling)
            if ((inferInputRankToMap >= 0) && (leftOperandShape[leftOperandShape.Rank() - 1] == NDShape::InferredDimension)) // if given, we pad if needed
            {
                while ((numReductionAxes + (size_t)inferInputRankToMap) < rightOperand.Shape().Rank())
                {
                    leftOperandShape = leftOperandShape.AppendShape({ NDShape::InferredDimension });
                    numReductionAxes++;
                }
            }

            for (size_t i = 0; i < numReductionAxes; ++i)
            {
                if ((leftOperandShape[outputRank + i] != NDShape::InferredDimension) && (rightOperandShape[i] != NDShape::InferredDimension))
                {
                    if (leftOperandShape[outputRank + i] != rightOperandShape[i])
                        InvalidArgument("The %d %s dimensions of the %s operand with shape %S do not match the %s operand's %s dimensions with shape %S",
                                        (int)numReductionAxes,
                                        Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "leading" : "trailing",
                                        Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "right" : "left",
                                        AsStringForErrorReporting(leftOperandShape.SubShape(outputRank)).c_str(),
                                        Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "left" : "right",
                                        Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "trailing" : "leading",
                                        AsStringForErrorReporting(rightOperandShape).c_str());
                }
                else if (leftOperandShape[outputRank + i] == NDShape::InferredDimension)
                    leftOperandShape[outputRank + i] = rightOperandShape[i];
                else if (rightOperandShape[i] == NDShape::InferredDimension)
                    rightOperandShape[i] = leftOperandShape[outputRank + i];

            }

            // See if we need to infer and propagate dimensions of any of the parameter operands
            if (inferInputDimensions)
            {
                std::vector<std::pair<Variable, NDShape>> newOperandShapes = { { leftOperand, leftOperandShape }, { rightOperand, rightOperandShape } };
                UpdateOperandShapes(newOperandShapes);
            }

            return leftOperandShape.SubShape(0, outputRank).AppendShape(rightOperandShape.SubShape(numReductionAxes));
        }

        static NDShape ReductionOpOutputShape(PrimitiveOpType op, const NDShape& operandShape, const std::vector<int>& reductionAxes, bool preserveReductionAxes)
        {
            if (reductionAxes.size() > operandShape.Rank())
                RuntimeError("The number of reduction axes %d exceeds the rank in the operand shape %S of the reduction operation %S",
                             (int)reductionAxes.size(),
                             AsStringForErrorReporting(operandShape).c_str(),
                             PrimitiveOpTypeName(op).c_str());

            size_t numOutputAxes = operandShape.Rank() - (preserveReductionAxes ? 0 : reductionAxes.size());
            std::vector<size_t> outputDims(numOutputAxes);
            for (int i = 0, j = 0; i < (int)operandShape.Rank(); ++i)
            {
                // Skip axes being reduced over
                if (std::find(reductionAxes.begin(), reductionAxes.end(), i) != reductionAxes.end())
                {
                    if (preserveReductionAxes)
                        outputDims[j++] = 1;
                }
                else
                    outputDims[j++] = operandShape[i];
            }

            return NDShape(std::move(outputDims));
        }

        static void FixNDShape(size_t filterRank, size_t inputRank, NDShape& shape, size_t deflt, const NDShape& from = NDShape())
        {
            auto dims = shape.Dimensions();
            Microsoft::MSR::CNTK::ConvolutionNodeBase<float>::FixVectorShape(filterRank, inputRank, dims, deflt, from.Dimensions());
            shape = NDShape(dims);
        }

        static NDShape ConvolutionOpOutputShape(PrimitiveOpType op, const NDShape& operandShape, NDShape& kernelShape, NDShape& outputMapCount, NDShape& strides,
                                                std::vector<bool>& sharing, std::vector<bool>& autoPad, NDShape& lowerPad, NDShape& upperPad,
                                                bool transpose, bool inferDimensions)
        {
            if (inferDimensions)
            {
                size_t inputRank = operandShape.Rank();

                // Unknown kernel shape valid only for pooling, however, the shape should have expanded before
                // this call.
                if (kernelShape == NDShape::Unknown)
                {
                    RuntimeError("Kernel shape can't be Unknown!");
                }

                // infer reduction dimensions if not given
                // If kernel has a lower rank than the input then the remaining dimensions are to be reduced over.
                size_t filterRank = kernelShape.Rank();

                // If the trailing axis dimensionality of the kernel shape is NDShape::InferredDimension, we reduce over it by 
                // picking the corresponding operand shape dimensionality
                // This is done by shrinking the filter rank and let the dimensions be inferred from the operand's shape
                // TODO: Should we do this for all of the axes in kernelShape that have a dimensionailty of NDShape::InferredDimension?
                if (kernelShape[filterRank - 1] == NDShape::InferredDimension)
                {
                    filterRank--;
                    kernelShape = kernelShape.SubShape(0, filterRank);
                }

                NDShape fromShape;
                if (op == PrimitiveOpType::Convolution)
                    fromShape = operandShape;

                FixNDShape(filterRank, inputRank, kernelShape, 1, fromShape); // convolve over red dim; pool over 1
                FixNDShape(filterRank, inputRank, strides, 1, fromShape); // stride for reduction dims is red dim or 1
                FixNDShape(filterRank, inputRank, lowerPad, 0);
                FixNDShape(filterRank, inputRank, upperPad, 0);
                Microsoft::MSR::CNTK::ConvolutionNodeBase<float>::FixVectorShape(filterRank, inputRank, sharing, true);
                Microsoft::MSR::CNTK::ConvolutionNodeBase<float>::FixVectorShape(filterRank, inputRank, autoPad, false); // no padding for reduction dims
            }

            decltype(&Microsoft::MSR::CNTK::ConvolveGeometry::ComputeOutputShape) computeOutputShapeFunc;
            if (!transpose)
                computeOutputShapeFunc = &Microsoft::MSR::CNTK::ConvolveGeometry::ComputeOutputShape;
            else
                computeOutputShapeFunc = &Microsoft::MSR::CNTK::ConvolveGeometry::ComputeInputShape;

            return AsNDShape(computeOutputShapeFunc(AsTensorShape(operandShape), AsTensorShape(kernelShape), AsTensorShape(outputMapCount), AsTensorShape(strides), sharing, autoPad, AsTensorShape(lowerPad), AsTensorShape(upperPad)));
        }

        static NDShape BatchNormalizationOutputShape(std::vector<Variable>& operands, bool spatial, bool inferDimensions)
        {
            NDShape mainOperandShape = operands[0].Shape();
            for (size_t i = 1; i < operands.size(); i++)
            {
                if (!operands[i].DynamicAxes().empty())
                    InvalidArgument("BatchNormalization: Input[%d] has a dynamic axis that is not allowed!", (int)i);

                // Infer dimensions of learnable parameters
                auto paramShape = operands[i].Shape();
                if (inferDimensions && ((paramShape.Rank() == 1) && paramShape.HasInferredDimension()) && !mainOperandShape.HasInferredDimension())
                {
                    size_t total = spatial ? mainOperandShape[mainOperandShape.Rank() - 1] : mainOperandShape.TotalSize();
                    paramShape[0] = total;
                    std::vector<std::pair<Variable, NDShape>> newParamShape = { { operands[i], paramShape } };
                    UpdateOperandShapes(newParamShape);
                }

                if (!paramShape.HasInferredDimension() && !operands[1].Shape().HasInferredDimension() && (paramShape != operands[1].Shape()))
                    InvalidArgument("BatchNormalization: Input[%d] has a shape (%S) different from Input[1] (%S), but they must be identical.", 
                                    (int)i,
                                    AsStringForErrorReporting(paramShape).c_str(),
                                    AsStringForErrorReporting(operands[1].Shape()).c_str());
            }

            return UnaryElementwiseOpOutputShape(mainOperandShape);
        }

        // TODO: Reconcile this with the ComputationNode::Validate functionality in core CNTK to avoid duplication of inference logic
        // Returns a pair of determined output variables and a bool indicating if any input operand shape was modified
        static std::vector<Variable> GetOutputVariables(PrimitiveOpType op,
                                                        std::vector<Variable>& inputs,
                                                        Function* owner,
                                                        Dictionary& functionConfig,
                                                        bool inferDimensions,
                                                        const std::wstring& functionName);

    private:
        PrimitiveOpType m_op;
        // Increasing s_serializationVersion every time we add more ops allows us to print 
        // a more meaningful message when trying to load a new model with a stale binary. 
        static const size_t s_serializationVersion = 2;
    };
}
