//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "CNTKLibrary.h"
#include "PrimitiveOpType.h"

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
        {PrimitiveOpType::Atanh, L"Atanh"},
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
        {PrimitiveOpType::LatticeSequenceWithSoftmax, L"LatticeSequenceWithSoftmax" },
        {PrimitiveOpType::ClassificationError, L"ClassificationError"},
        {PrimitiveOpType::EditDistanceError, L"EditDistanceError" },
        {PrimitiveOpType::ForwardBackward, L"ForwardBackward" },
        {PrimitiveOpType::LabelsToGraph, L"LabelsToGraph" },
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
        {PrimitiveOpType::Asin, L"Asin"},
        {PrimitiveOpType::Acos, L"Acos"},
        {PrimitiveOpType::Sin, L"Sin"},
        {PrimitiveOpType::Cos, L"Cos"},
        {PrimitiveOpType::Cosh, L"Cosh"},
        {PrimitiveOpType::Asinh, L"Asinh"},
        {PrimitiveOpType::Sinh, L"Sinh"},
        {PrimitiveOpType::Pass, L"Pass"},
        {PrimitiveOpType::Block, L"Block"},
        {PrimitiveOpType::Unpooling, L"Unpooling"},
        {PrimitiveOpType::LambdaRank, L"LambdaRank"},
        {PrimitiveOpType::NDCG, L"NDCG"},
        {PrimitiveOpType::NoOp, L"NoOp"},
        {PrimitiveOpType::StopGradient, L"StopGradient"},
        {PrimitiveOpType::ELU, L"ELU"},
        {PrimitiveOpType::CosDistanceWithNegativeSamples, L"CosDistanceWithNegativeSamples"},
        {PrimitiveOpType::OneHot, L"OneHotOp" },
        {PrimitiveOpType::Pow, L"Pow"},
        {PrimitiveOpType::ToSequence, L"ToSequenceOp"},
        {PrimitiveOpType::ToSequenceLike, L"ToSequenceLikeOp"},
        {PrimitiveOpType::UnpackSequence, L"UnpackSequenceOp"},
        {PrimitiveOpType::Assign, L"Assign" },
        {PrimitiveOpType::Gather, L"Gather"},
        {PrimitiveOpType::StableSigmoid, L"StableSigmoid"},
        {PrimitiveOpType::RandomDistribution, L"RandomDistribution"},
        {PrimitiveOpType::UnpackBatch, L"UnpackBatchAxis"},
        {PrimitiveOpType::ToBatch, L"ToBatchAxis"},
        {PrimitiveOpType::Pad, L"Pad"},
        {PrimitiveOpType::Crop, L"Crop"},
        {PrimitiveOpType::TopK, L"TopK"},
        {PrimitiveOpType::ConstantOp, L"ConstantOp"},
        {PrimitiveOpType::Squeeze, L"Squeeze"},
        {PrimitiveOpType::Cast, L"Cast" },
        { PrimitiveOpType::EyeLikeOp, L"EyeLikeOp" },
        { PrimitiveOpType::CustomProxyOp, L"CustomProxyOp" },
        {PrimitiveOpType::StraightThrough, L"StraightThrough"},
        { PrimitiveOpType::Tan, L"Tan" },
        { PrimitiveOpType::Atan, L"Atan" },
        { PrimitiveOpType::ConvolutionSequenceShape, L"ConvolutionSequenceShape" },
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
        else if (op == PrimitiveOpType::LambdaRank)
            indexMap = std::unordered_map<size_t, size_t>({ { 0, 1 }, { 1, 0 }, { 2, 2 } });
        else if (op == PrimitiveOpType::NDCG)
            indexMap = std::unordered_map<size_t, size_t>({ { 0, 1 },{ 1, 0 },{ 2, 2 } });
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
            LogicError("Size (%d) of the Primitive Function Inputs to CNTK Node Inputs Map does not match the actual number (%d) of Inputs of the PrimitiveFunction", (int)indexMap.size(), (int)numFunctionInputs);

        for (auto indexPair : indexMap)
        {
            if ((indexPair.first >= numFunctionInputs) || (indexPair.second >= numFunctionInputs))
                LogicError("The index values in the PrimitiveFunctionInputsToCNTKNodeInputsIndexMap must be < the number of Inputs of the PrimitiveFunction");
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

    class PrimitiveFunction : public Function
    {
        friend class Function;
        friend class Utils;
        template <typename T, typename ...CtorArgTypes>
        friend inline std::shared_ptr<T> MakeSharedObject(CtorArgTypes&& ...ctorArgs);

    public:
        static const size_t convolutionOpDefaultValueForGroups = 1;

    protected:
        PrimitiveFunction(PrimitiveOpType op, const std::vector<Variable>& inputs, Dictionary&& functionConfig, const std::wstring& functionName, const std::wstring& uid)
            : Function(inputs, std::move(functionConfig), nullptr, functionName, uid), m_op(op)
        {}

    public:
        PrimitiveFunction(PrimitiveOpType op, const std::vector<Variable>& inputs, Dictionary&& functionConfig, const std::wstring& functionName = L"")
            : PrimitiveFunction(op, inputs, std::move(functionConfig), functionName, GenerateUid(op))
        {}

        // Primitive functions are currently implemented using the core CNTK engine ComputationNode types
        virtual BackPropStatePtr Forward(const std::vector<ValuePtr>& /*inputValues*/,
                                         std::unordered_map<Variable, ValuePtr>& /*outputs*/,
                                         const DeviceDescriptor& /*computeDevice*/,
                                         const std::unordered_set<Variable>& /*outputsToRetainBackwardStateFor*/)
        {
            NOT_IMPLEMENTED;
        }

        virtual Dictionary Serialize() const override;

        virtual size_t CurrentVersion() const override { return s_serializationVersion; }

        static FunctionPtr Deserialize(const Dictionary& dictionary,
                                       const std::unordered_map<std::wstring, Variable>& uidToVariableMap,
                                       const std::unordered_set<FunctionPtr>& allPrimitiveFunctions,
                                       const std::unordered_map<Variable, Variable>& placeholderReplacements,
                                       const CNTK::DeviceDescriptor& device);

        virtual const std::wstring& OpName() const override
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
                   (OpType() == PrimitiveOpType::RandomSampleInclusionFrequency) ||
                   (OpType() == PrimitiveOpType::RandomDistribution);
        }

        Dictionary GetState() const;

        void SetState(const Dictionary& state);

    private:

        // The following helper functions are used to determine the output shape for different
        // types of primitive operations accounting for broadcasting and reductions where applicable.
        static NDShape UnaryElementwiseOpOutputShape(const NDShape& operandShape)
        {
            return operandShape;
        }

        static NDShape ReshapeOutputShape(const NDShape& operandShape, NDShape& replacementShape, const Axis& beginAxis, const Axis& endAxis, bool inferDimensions)
        {
            int beginAxisIdx = beginAxis.StaticAxisIndex();
            int endAxisIdx = endAxis.StaticAxisIndex();

            if (beginAxisIdx > endAxisIdx)
                InvalidArgument("Reshape: begin axis index (%d) must be <= the end axis index (%d)", beginAxisIdx, endAxisIdx);

            if ((beginAxisIdx < 0) || (beginAxisIdx > operandShape.Rank()))
                InvalidArgument("Reshape: begin axis index (%d) is invalid for operand shape '%S'", beginAxisIdx, operandShape.AsString().c_str());

            if ((endAxisIdx < 0) || (endAxisIdx > operandShape.Rank()))
                InvalidArgument("Reshape: end axis index (%d) is invalid for operand shape '%S'.", endAxisIdx, operandShape.AsString().c_str());

            auto operandSubshapeToReshape = operandShape.SubShape(beginAxisIdx, endAxisIdx);
            auto inferredReplacementShape = replacementShape;
            size_t inferredAxisIndex = SIZE_MAX;
            size_t targetElementsCount = 1;
            for (size_t k = 0; k < inferredReplacementShape.Rank(); k++)
            {
                if (inferredReplacementShape[k] != NDShape::InferredDimension)
                    targetElementsCount *= inferredReplacementShape[k];
                else if (inferredAxisIndex == SIZE_MAX)
                    inferredAxisIndex = k;
                else
                    InvalidArgument("Reshape: More than one axis's dimension was unspecified in the replacement shape '%S'", replacementShape.AsString().c_str());
            }

            if (inferredAxisIndex != SIZE_MAX)
            {
                if (!operandSubshapeToReshape.HasUnboundDimension())
                {
                    size_t inputElementsCount = operandSubshapeToReshape.TotalSize();
                    inferredReplacementShape[inferredAxisIndex] = inputElementsCount / targetElementsCount;
                }
                else
                    inferredReplacementShape[inferredAxisIndex] = operandSubshapeToReshape.HasInferredDimension() ? NDShape::InferredDimension : NDShape::FreeDimension;
            }

            auto outputShape = operandShape.SubShape(0, beginAxisIdx);
            outputShape = outputShape.AppendShape(inferredReplacementShape);
            outputShape = outputShape.AppendShape(operandShape.SubShape(endAxisIdx));

            if (!operandSubshapeToReshape.HasUnboundDimension() && (operandSubshapeToReshape.TotalSize() != inferredReplacementShape.TotalSize()))
            {
                auto replacedSubShape = operandShape.SubShape(beginAxisIdx, endAxisIdx);
                InvalidArgument("Reshape: Operand (sub-)dimensions '%S' incompatible with desired replacement (sub-)dimensions '%S'. Number of elements %s.",
                                replacedSubShape.AsString().c_str(), replacementShape.AsString().c_str(),
                                inferredAxisIndex == SIZE_MAX ? "must be the same." : "is not an integer multiple of the non-inferred dimensions.");
            }

            if (inferDimensions)
                replacementShape = inferredReplacementShape;

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

            // spliceDim may exceed all of them, which will create a new dimension, e.g. stacking column vectors into a matrix
            size_t maxRank = std::max<size_t>(axis + 1, maxInputRank);

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
                        if ((dim == NDShape::InferredDimension) || (outputDims[index] == NDShape::InferredDimension))
                            outputDims[index] = NDShape::InferredDimension;
                        else if (dim == NDShape::FreeDimension || (outputDims[index] == NDShape::FreeDimension))
                            //InvalidArgument("Splice: Illegal to splice along an axis (%d) for which any of the inputs has a free dimension.", (int)index);
                            outputDims[index] = NDShape::FreeDimension;
                        else
                            outputDims[index] += dim;
                    }
                    else
                    {
                        // check dimensions
                        if ((outputDims[k] == NDShape::InferredDimension) || (outputDims[k] == 1))
                            outputDims[k] = dim; // Broadcast
                        else if ((dim != outputDims[k]) && (dim != 1) && (dim != NDShape::InferredDimension))
                            InvalidArgument("Splice: Conflicting dimensionality of axis %d between operand #%d (%d) and other(s) (%d).", (int)k, i, (int)dim, (int)outputDims[k]);
                    }
                }
            }

            return outputDims;
        }

        // Returns a boolean indicating if any operand shape was updated
        static bool UpdateOperandShapes(std::vector<std::pair<Variable, NDShape>>& newOperandShapes);

        // Returns a pair comprising of the output shape and boolean indicating if any input operand shape was modified
        static NDShape BinaryElementwiseOpOutputShape(PrimitiveOpType op, Variable& leftOperand, Variable& rightOperand, bool inferInputDimensions, bool allowScalarBroadcast = true)
        {
            auto leftOperandShape = leftOperand.Shape();
            auto rightOperandShape = rightOperand.Shape();

            // when scalar allows broadcasting, keep unknown shape
            if (leftOperandShape.IsUnknown() && !(allowScalarBroadcast && rightOperandShape.IsScalar()))
                leftOperandShape = rightOperandShape;

            if (rightOperandShape.IsUnknown() && !(allowScalarBroadcast && leftOperandShape.IsScalar()))
                rightOperandShape = leftOperandShape;

            // All operand shapes should be known
            assert(!leftOperandShape.IsUnknown()&& !rightOperandShape.IsUnknown());

            const auto& shapeWithSmallerNumAxes = (leftOperandShape.Rank() > rightOperandShape.Rank()) ? rightOperandShape : leftOperandShape;
            const auto& shapeWithLargerNumAxes = (leftOperandShape.Rank() > rightOperandShape.Rank()) ? leftOperandShape : rightOperandShape;
            size_t numOutputAxes = shapeWithLargerNumAxes.Rank();
            std::vector<size_t> outputDims(numOutputAxes);
            for (size_t i = 0; i < shapeWithSmallerNumAxes.Rank(); ++i)
            {
                if ((leftOperandShape[i] == NDShape::InferredDimension) && (rightOperandShape[i] == NDShape::InferredDimension))
                    outputDims[i] = NDShape::InferredDimension;
                else if (leftOperandShape[i] == NDShape::FreeDimension)
                {
                    if (rightOperandShape[i] == NDShape::InferredDimension)
                        InvalidArgument("Binary elementwise operation %S: Right operand '%S' shape '%S' dimension cannot be inferred from a left operand '%S' shape '%S' free dimension.",
                            PrimitiveOpTypeName(op).c_str(),
                            rightOperand.AsString().c_str(),
                            rightOperandShape.AsString().c_str(),
                            leftOperand.AsString().c_str(),
                            leftOperandShape.AsString().c_str());

                    // Broadcast to a free-dimension, if the right operand axis's dimensionality is 1; otherwise the output axis dimensionality
                    // is the known right operands axis's dimensionality
                    outputDims[i] = (rightOperandShape[i] == 1) ? NDShape::FreeDimension : rightOperandShape[i];
                }
                else if (rightOperandShape[i] == NDShape::FreeDimension)
                {
                    if (leftOperandShape[i] == NDShape::InferredDimension)
                        InvalidArgument("Binary elementwise operation %S: Left operand '%S' shape '%S' dimension cannot be inferred from a right operand '%S' shape '%S' free dimension.",
                            PrimitiveOpTypeName(op).c_str(),
                            leftOperand.AsString().c_str(),
                            leftOperandShape.AsString().c_str(),
                            rightOperand.AsString().c_str(),
                            rightOperandShape.AsString().c_str());

                    // Broadcast to a free-dimension, if the left operand axis's dimensionality is 1; otherwise the output axis dimensionality
                    // is the known left operands axis's dimensionality
                    outputDims[i] = (leftOperandShape[i] == 1) ? NDShape::FreeDimension : leftOperandShape[i];
                }
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
                        RuntimeError("Binary elementwise operation %S: Left operand '%S' shape '%S' is not compatible with right operand '%S' shape '%S'.",
                                     PrimitiveOpTypeName(op).c_str(),
                                     leftOperand.AsString().c_str(),
                                     leftOperandShape.AsString().c_str(),
                                     rightOperand.AsString().c_str(),
                                     rightOperandShape.AsString().c_str());

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

        static NDShape NaryElementwiseOpOutputShape(PrimitiveOpType op, std::vector<Variable>& operands, bool inferInputDimensions);

        // Returns a pair comprising of the output shape and boolean indicating if any input operand shape was modified
        static NDShape TimesOpOutputShape(Variable& leftOperand, Variable& rightOperand, size_t outputRank, int inferInputRankToMap, bool inferInputDimensions)
        {
            auto leftOperandShape = leftOperand.Shape();
            auto rightOperandShape = rightOperand.Shape();

            if (outputRank == 0)
                InvalidArgument("Times: Output rank (%d) must be > 0.", (int)outputRank);

            if (outputRank > leftOperandShape.Rank())
                InvalidArgument("Times: Output rank (%d) must be <= rank (%d) of the %s operand '%S'.",
                                (int)outputRank, (int)leftOperandShape.Rank(), Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "right" : "left", leftOperand.AsString().c_str());

            if (inferInputRankToMap >= (int)(rightOperandShape.Rank()))
                InvalidArgument("Times: Input map rank (%d) must be < rank (%d) of the %s operand '%S'.",
                                inferInputRankToMap, (int)(rightOperandShape.Rank()), Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "left" : "right", rightOperand.AsString().c_str());

            size_t numReductionAxes = leftOperandShape.Rank() - outputRank;

            // The 'numReductionAxes' trailing dimensions of the left operand's shape must match the corresponding leading
            // dimensions of the right operand

            if (rightOperandShape.Rank() < numReductionAxes)
                RuntimeError("Times: The %s operand '%S' rank (%d) must be >= #axes (%d) being reduced over.",
                             Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "left" : "right", rightOperand.AsString().c_str(), (int)rightOperandShape.Rank(), (int)numReductionAxes);

            if (rightOperand.IsSparse() && (numReductionAxes > 1))
                LogicError("Times: For a sparse %s operand '%S', cannot reduce multiple (%zu) axes; currently only the %s axis can be reduced for the sparse operand.",
                            Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "left" : "right",
                            rightOperand.AsString().c_str(),
                            numReductionAxes,
                            Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "trailing" : "leading");

            // outputRank dimensions cannot be inferred
            for (size_t k = 0; k < outputRank; k++)
            {
                if (leftOperandShape[k] == NDShape::InferredDimension)
                    InvalidArgument("Times: The outputRank (%d) dimensions of %s operand's shape '%S' cannot be Inferred.",
                                    (int)outputRank,
                                    Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "right" : "left",
                                    leftOperandShape.AsString().c_str());
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
                if ((leftOperandShape[outputRank + i] != NDShape::InferredDimension
                     && leftOperandShape[outputRank + i] != NDShape::FreeDimension) && 
                     (rightOperandShape[i] != NDShape::InferredDimension
                      && rightOperandShape[i] != NDShape::FreeDimension))
                {
                    if (leftOperandShape[outputRank + i] != rightOperandShape[i])
                        InvalidArgument("Times: The %d %s dimensions of the %s operand with shape '%S' do not match the %s operand's %s dimensions with shape '%S'",
                                        (int)numReductionAxes,
                                        Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "leading" : "trailing",
                                        Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "right" : "left",
                                        leftOperandShape.SubShape(outputRank).AsString().c_str(),
                                        Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "left" : "right",
                                        Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "trailing" : "leading",
                                        rightOperandShape.AsString().c_str());
                }
                else if (leftOperandShape[outputRank + i] == NDShape::InferredDimension || leftOperandShape[outputRank + i] == NDShape::FreeDimension)
                {
                    if (rightOperandShape[i] == NDShape::FreeDimension && leftOperandShape[outputRank + i] == NDShape::InferredDimension)
                        InvalidArgument("Times: %s operand '%S' shape '%S' dimension cannot be inferred from a %s operand '%S' shape '%S' free dimension.",
                            Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "right" : "left",
                            leftOperand.AsString().c_str(),
                            leftOperandShape.AsString().c_str(),
                            Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "left" : "right",
                            rightOperand.AsString().c_str(),
                            rightOperandShape.AsString().c_str());

                    leftOperandShape[outputRank + i] = rightOperandShape[i];
                }
                else if (rightOperandShape[i] == NDShape::InferredDimension || rightOperandShape[i] == NDShape::FreeDimension)
                {
                    if (leftOperandShape[outputRank + i] == NDShape::FreeDimension && leftOperandShape[outputRank + i] == NDShape::InferredDimension)
                        InvalidArgument("Times: %s operand '%S' shape '%S' dimension cannot be inferred from a %s operand '%S' shape '%S' free dimension.",
                            Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "left" : "right",
                            rightOperand.AsString().c_str(),
                            rightOperandShape.AsString().c_str(),
                            Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "right" : "left",
                            leftOperand.AsString().c_str(),
                            leftOperandShape.AsString().c_str());

                    rightOperandShape[i] = leftOperandShape[outputRank + i];
                }
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
                RuntimeError("Reduction operation %S: number (%d) of reduction axes exceeds the rank (%d) of the operand shape '%S'.",
                             PrimitiveOpTypeName(op).c_str(),
                             (int)reductionAxes.size(),
                             (int)operandShape.Rank(),
                             operandShape.AsString().c_str());

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

        static void FixNDShape(size_t filterRank, size_t inputRank, NDShape& shape, size_t deflt, const NDShape& from = NDShape());

        static NDShape ConvolutionOpOutputShape(PrimitiveOpType op, const NDShape& operandShape, NDShape& kernelShape, NDShape& outputMapCount, NDShape& strides,
                                                std::vector<bool>& sharing, std::vector<bool>& autoPad, NDShape& lowerPad, NDShape& upperPad,
                                                bool transpose, bool inferDimensions, NDShape& dilation, size_t groups=1, bool ceilOutputDim = false);

        static NDShape BatchNormalizationOutputShape(std::vector<Variable>& operands, bool spatial, bool inferDimensions)
        {
            NDShape mainOperandShape = operands[0].Shape();
            for (size_t i = 1; i < operands.size(); i++) // all but first and last arguments must match the first; last one must be a scalar
            {
                if (!operands[i].DynamicAxes().empty())
                    InvalidArgument("BatchNormalization: Input[%d] '%S' must not have a dynamic axis.", (int)i, operands[i].AsString().c_str());

                // Infer dimensions of learnable parameters
                auto paramShape = operands[i].Shape();

                if (i < operands.size() - 1)
                {
                    if (inferDimensions && ((paramShape.Rank() == 1) && paramShape.HasInferredDimension()) && (!mainOperandShape.HasUnboundDimension() || (spatial && mainOperandShape[mainOperandShape.Rank() - 1] != NDShape::FreeDimension)))
                    {
                        size_t total = spatial ? mainOperandShape[mainOperandShape.Rank() - 1] : mainOperandShape.TotalSize();
                        paramShape[0] = total;
                        std::vector<std::pair<Variable, NDShape>> newParamShape = { { operands[i], paramShape } };
                        UpdateOperandShapes(newParamShape);
                    }

                    if (!paramShape.HasInferredDimension() && !operands[1].Shape().HasInferredDimension() && (paramShape != operands[1].Shape()))
                        InvalidArgument("BatchNormalization: Input[%d] shape '%S' must be identical to Input[1] shape '%S'.",
                                        (int)i,
                                        paramShape.AsString().c_str(),
                                        operands[1].Shape().AsString().c_str());
                }
            }

            const auto& runCount = operands[operands.size() - 1];
            auto runCountRank = runCount.Shape().Rank();
            if (runCountRank > 1 || (runCountRank == 1 && runCount.Shape()[0] != 1)) // last arguments is count, must be a scalar
                InvalidArgument("BatchNormalization: Input[%d] (running mean sample count) '%S' must be a scalar.", (int)(operands.size() - 1), runCount.AsString().c_str());

            return UnaryElementwiseOpOutputShape(mainOperandShape);
        }

        // TODO: Reconcile this with the ComputationNode::Validate functionality in core CNTK to avoid duplication of inference logic
        // Returns a pair of determined output variables and a bool indicating if any input operand shape was modified
        static DataType GetOutputDataType(PrimitiveOpType op, std::vector<Variable>& inputs, bool inferDimensions);
        static std::vector<Axis> GetOutputDynamicAxes(PrimitiveOpType op, std::vector<Variable>& inputs, PrimitiveFunction* owner, Dictionary& functionConfig);

        void InferOutputs(std::vector<Variable>& outputs) override;

        FunctionPtr Clone(const std::vector<Variable>& clonedInputs) override
        {
            return MakeSharedObject<PrimitiveFunction>(OpType(), clonedInputs, Dictionary(Attributes()), Name());
        }

        void SetDropoutRate(double dropoutRate);

        void SetRandomSeed(size_t seed);
    private:
        //aux functions
        void CollectReduceOutputAxesForOutputShape(std::vector<Axis>& staticAxesToReduce,
            std::vector<Axis>& batchAxesToReduce,
            std::vector<Axis>& dynamicAxesToReduce,
            bool & isAllAxesReduced);

    private:
        PrimitiveOpType m_op;
        // Increasing s_serializationVersion every time we add more ops allows us to print
        // a more meaningful message when trying to load a new model with a stale binary.
        // version 1: initial version.
        // version 2: Add maxUnpooling.
        // version 3: Add deconvolution.
        // version 4: added extra parameter (#6) for the running mean sample count in BatchNormalization.
        // Version 6: Add argmax and argmin to ReduceElement.
        // Version 8: Add ELU node.
        // Version 9: Add OneHot node.
        // Version 10: Add Pow operator.
        // Version 11: Add ToSequence, ToSequenceLike and UnpackSequence operators.
        // Version 12: Add Assign node.
        // Version 13: Add Gather op.
        // Version 14: Add StableSigmoid
        // Version 15: Add RandomDistribution
        // Version 16: Add to_batch/unpack_batch.
        // Version 17: Add Pad.
        // Version 18: Add Crop node.
        // Version 19: Add TopK
        // Version 20: Add squeeze, expand dims, zeros like, ones like
        // Version 21: Add EyeLikeOp
        // Version 22: Add StraightThrough
        // Version 23: Add Tan and Atan.
        // Version 24: Add ConvolutionSequenceShape.
        static const size_t s_serializationVersion = 24;
    };

    std::vector<DictionaryValue> GetInputUids(const Function& f);
    Dictionary SerializeCommonFunctionAttributes(const Function& f, size_t version, const std::wstring& functionType);
    std::vector<Variable> GetInputVariables(const Dictionary& dict, const std::unordered_map<std::wstring, Variable>& uidToVariableMap, size_t currentSerializationVersion);
}
