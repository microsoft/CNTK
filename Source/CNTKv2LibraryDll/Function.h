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
        Dropout,
        Reshape,
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
        ScatterPacked,
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
        Clip,
        Select,
        Splice,
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
        static const std::unordered_map<PrimitiveOpType, const char*> primitiveOpNames = {
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
            { PrimitiveOpType::Dropout, "Dropout" },
            { PrimitiveOpType::Reshape, "Reshape" },
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
            { PrimitiveOpType::ScatterPacked, "ScatterPacked" },
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
            { PrimitiveOpType::Clip, "Clip" },
            { PrimitiveOpType::Select, "Select" },
            { PrimitiveOpType::Splice, "Splice" },
            { PrimitiveOpType::Combine, "Combine" }
        };

        if (primitiveOpNames.find(opType) == primitiveOpNames.end())
            LogicError("Unknown PrimitiveOpType");

        return primitiveOpNames.find(opType)->second;
    }

    inline std::unordered_map<size_t, size_t> GetPrimitiveFunctionInputsToCNTKNodeInputsIndexMap(PrimitiveOpType op, size_t numFunctionInputs)
    {
        std::unordered_map<size_t, size_t> indexMap;
        if ((op == PrimitiveOpType::CrossEntropyWithSoftmax) || (op == PrimitiveOpType::ClassificationError) || (op == PrimitiveOpType::GatherPacked))
            indexMap = std::unordered_map<size_t, size_t>({ { 0, 1 }, { 1, 0 } });
        else if (op == PrimitiveOpType::ScatterPacked)
            indexMap = std::unordered_map<size_t, size_t>({ { 0, 2 }, { 1, 1 }, { 2, 0 } });
        else if (op == PrimitiveOpType::Clip)
            indexMap = std::unordered_map<size_t, size_t>({ { 0, 2 }, { 1, 0 }, { 2, 1 } });
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
        static const std::wstring AttributeNameDropoutRate;
        static const std::wstring AttributeNameNewShape;
        static const std::wstring AttributeNameOutputRank;
        static const std::wstring AttributeNameOffset;
        static const std::wstring AttributeNameStrides;
        static const std::wstring AttributeNameSharing;
        static const std::wstring AttributeNameAutoPadding;
        static const std::wstring AttributeNameLowerPad;
        static const std::wstring AttributeNameUpperPad;
        static const std::wstring AttributeNameTranspose;
        static const std::wstring AttributeNameMaxTempMemSizeInSamples;
        static const std::wstring AttributeNamePoolingType;
        static const std::wstring AttributeNamePoolingWindowShape;
        static const std::wstring AttributeNameSpatial;
        static const std::wstring AttributeNameNormalizationTimeConstant;
        static const std::wstring AttributeNameBlendTimeConstant;
        static const std::wstring AttributeNameEpsilon;
        static const std::wstring AttributeNameUseCuDNNEngine;
        static const std::wstring AttributeNameNewDynamicAxes;
        static const std::wstring AttributeNameBeginIndex;
        static const std::wstring AttributeNameEndIndex;
        static const std::wstring AttributeNameReductionOpName;

    public:
        PrimitiveFunction(PrimitiveOpType op, const std::vector<Variable>& inputs, Dictionary&& functionConfig, const std::wstring& functionName = L"")
            : Function(inputs, GetOutputVariables(op, inputs, this, functionConfig), std::move(functionConfig), nullptr, functionName), m_op(op)
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

    public:
        PrimitiveOpType OpType() const
        {
            return m_op;
        }

    private:
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

        static NDShape SpliceOutputShape(const std::vector<Variable>& inputs, size_t axis)
        {
            // We must fuse all tensor shapes

            // Determine maximum rank (we can stack tensors with lower rank, which will have their dimensions paded to max automatically)
            size_t index = axis;
            size_t maxRank = axis + 1; // spliceDim may exceed all of them, which will create a new dimension, e.g. stacking column vectors into a matrix
            for (int i = 0; i < inputs.size(); i++)
            {
                auto inputAxesCount = inputs[i].Shape().Rank();
                if (maxRank < inputAxesCount)
                    maxRank = inputAxesCount;
            }

            // The following loop does multiple things:
            //  - Count total dimension along index
            //  - Verify all other dimension's compatibility (we allow broadcasting)

            // dimensions padded to max rank; start with dims of first input
            auto outputDims = inputs[0].Shape().AppendShape(NDShape(maxRank - inputs[0].Shape().Rank(), 1));

            // This dimension is created, while all others are verified for consistency
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
                        outputDims[index] += dim;
                    else
                    {
                        // check/fuse dimensions
                        if (dim != outputDims[k] && dim != 1 && outputDims[k] != 1)
                            InvalidArgument("CNTK::Splice: Conflicting dimension of axis %d between operand #%d (%d) and other(s) (%d)", (int)k, i, (int)dim, (int)outputDims[k]);

                        // Broadcast
                        if (outputDims[k] == 1)
                            outputDims[k] = dim;
                    }
                }
            }

            return outputDims;
        }

        static NDShape BinaryElementwiseOpOutputShape(PrimitiveOpType op, const NDShape& leftOperandShape, const NDShape& rightOperandShape, bool broadcastAllowed = true)
        {
            const auto& shapeWithSmallerNumAxes = (leftOperandShape.Rank() > rightOperandShape.Rank()) ? rightOperandShape : leftOperandShape;
            const auto& shapeWithLargerNumAxes = (leftOperandShape.Rank() > rightOperandShape.Rank()) ? leftOperandShape : rightOperandShape;
            size_t numOutputAxes = shapeWithLargerNumAxes.Rank();
            std::vector<size_t> outputDims(numOutputAxes);
            for (size_t i = 0; i < shapeWithSmallerNumAxes.Rank(); ++i)
            {
                if ((leftOperandShape[i] == NDShape::InferredDimension) && (rightOperandShape[i] == NDShape::InferredDimension))
                    outputDims[i] = NDShape::InferredDimension;
                else if ((leftOperandShape[i] == NDShape::InferredDimension) || (leftOperandShape[i] == 1))
                    outputDims[i] = rightOperandShape[i];
                else if ((rightOperandShape[i] == NDShape::InferredDimension) || (rightOperandShape[i] == 1))
                    outputDims[i] = leftOperandShape[i];
                else
                {
                    if (leftOperandShape[i] != rightOperandShape[i])
                        RuntimeError("Left operand's shape %S is not compatible with right operand's shape %S for the binary elementwise operation %s", AsStringForErrorReporting(leftOperandShape).c_str(), AsStringForErrorReporting(rightOperandShape).c_str(), PrimitiveOpTypeName(op));

                    outputDims[i] = leftOperandShape[i];
                }
            }

            // Broadcast in remaining axes
            for (size_t i = shapeWithSmallerNumAxes.Rank(); i < numOutputAxes; ++i)
                outputDims[i] = shapeWithLargerNumAxes[i];

            return NDShape(std::move(outputDims));
        }

        static NDShape NaryElementwiseOpOutputShape(PrimitiveOpType op, const std::vector<NDShape>& operandShapes, bool broadcastAllowed = true)
        {
            assert(!operandShapes.empty());

            // TODO: Is this logic of transitively constructing the output shape from the operands correct?
            NDShape outputShape = {};
            for (auto& operandShape : operandShapes)
                outputShape = BinaryElementwiseOpOutputShape(op, outputShape, operandShape, broadcastAllowed);

            return outputShape;
        }

        static NDShape TimesOpOutputShape(const NDShape& leftOperandShape, const NDShape& rightOperandShape, size_t outputRank)
        {
            if (outputRank == 0)
                InvalidArgument("Output rank of times operation should be at least one");

            if (outputRank > leftOperandShape.Rank())
                InvalidArgument("Output rank of times operation can at most be the rank of the %s operand", Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "right" : "left");

            size_t numReductionAxes = leftOperandShape.Rank() - outputRank;

            // The 'numReductionAxes' trailing dimensions of the left operand's shape must match the corresponding leading
            // dimensions of the right operand

            if (rightOperandShape.Rank() < numReductionAxes)
                RuntimeError("The %s operand's rank in a times operation should not be less than #axes being reduced over!", Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "left" : "right");

            if (leftOperandShape.SubShape(outputRank) != rightOperandShape.SubShape(0, numReductionAxes))
            {
                InvalidArgument("The %d %s dimensions of the %s operand with shape %S do not match the %s operand's %s dimensions with shape %S",
                                (int)numReductionAxes,
                                Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "leading" : "trailing",
                                Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "right" : "left",
                                AsStringForErrorReporting(leftOperandShape.SubShape(outputRank)).c_str(),
                                Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "left" : "right",
                                Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "trailing" : "leading",
                                AsStringForErrorReporting(rightOperandShape).c_str());
            }

            return leftOperandShape.SubShape(0, outputRank).AppendShape(rightOperandShape.SubShape(numReductionAxes));
        }

        static NDShape ReductionOpOutputShape(PrimitiveOpType op, const NDShape& operandShape, const std::vector<size_t>& reductionAxes)
        {
            if (reductionAxes.size() > operandShape.Rank())
                RuntimeError("The number of reduction axes %d exceeds the number of axes in the operand shape %S of the reduction operation %s", (int)reductionAxes.size(), AsStringForErrorReporting(operandShape).c_str(), PrimitiveOpTypeName(op));

            size_t numOutputAxes = operandShape.Rank() - reductionAxes.size();
            std::vector<size_t> outputDims(numOutputAxes);
            for (size_t i = 0, j = 0; i < operandShape.Rank(); ++i)
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

            return AsNDShape(computeOutputShapeFunc(AsTensorShape(operandShape), AsTensorShape(kernelShape), AsTensorShape(outputMapCount), AsTensorShape(strides), sharing, autoPad, AsTensorShape(lowerPad), AsTensorShape(upperPad)));
        }

        // TODO: Reconcile this with the ComputationNode::Validate functionality in core CNTK to avoid duplication of inference logic
        static std::vector<Variable> GetOutputVariables(PrimitiveOpType op, const std::vector<Variable>& inputs, Function* owner, const Dictionary& functionConfig);

    private:
        PrimitiveOpType m_op;
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
        friend class Trainer;
        friend class CompositeMinibatchSource;

        template <typename T, typename ...CtorArgTypes>
        friend inline std::shared_ptr<T> MakeSharedObject(CtorArgTypes&& ...ctorArgs);

        friend void SaveAsLegacyModel(const FunctionPtr& rootFunction, const std::wstring& modelFile);

        friend void ComputeInputPerDimMeansAndInvStdDevs(const MinibatchSourcePtr& minibatchSource,
                                                         std::unordered_map<StreamInformation, std::pair<NDArrayViewPtr, NDArrayViewPtr>>& computedMeanAndInvStdDevs,
                                                         const DeviceDescriptor& device /*= DeviceDescriptor::CPUDevice()*/);

        static std::atomic<unsigned int> s_nextAutoGeneratedDynamicAxis;

    public:
        static const std::wstring InternalDefaultDynamicAxisName;
        static const std::wstring InternalNoSequenceAxisName;

        static Axis NextAutoGeneratedDynamicAxis()
        {
            static const std::wstring s_autoGeneratedDynamicAxisNamePrefix = L"autoGeneratedDynamicAxis_";
            return Axis(s_autoGeneratedDynamicAxisNamePrefix + std::to_wstring(s_nextAutoGeneratedDynamicAxis++));
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

    public:
        bool NetworkMatricesAllocated() const 
        {
            return (m_computationNetwork != nullptr) && m_networkMatricesAllocated; 
        }

        void PurgeComputationNetwork()
        {
            m_computationNetwork = nullptr;
        }

    private:
        virtual void ReplacePlaceholdersInPlace(const std::unordered_map<Variable, Variable>& placeholderReplacements,
                                                std::unordered_set<const Function*>& visitedFunctions,
                                                std::unordered_set<Variable>& replacedPlaceholders) override;

        CompositeFunction(const FunctionPtr& rootFunction, std::unordered_set<FunctionPtr>&& allPrimitiveFunctions, const std::wstring& name)
            : Function({}, rootFunction->Outputs(), Dictionary(), rootFunction, name), m_allPrimitiveFunctions(std::move(allPrimitiveFunctions))
        {}

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
        Microsoft::MSR::CNTK::ComputationNetworkPtr GetComputationNetwork(const DeviceDescriptor& device, const std::unordered_set<Variable>& backpropRoots, bool allocateNetworkMatrices);

        template <typename ElementType>
        static Microsoft::MSR::CNTK::ComputationNodeBasePtr CreateComputationNode(const Variable& variable,
                                                                                  PrimitiveFunction* primitiveFunction,
                                                                                  const std::vector<std::shared_ptr<Microsoft::MSR::CNTK::ComputationNode<ElementType>>>& inputNodes,
                                                                                  Microsoft::MSR::CNTK::ComputationNetworkPtr& network,
                                                                                  std::unordered_map<Variable, Microsoft::MSR::CNTK::ComputationNodeBasePtr>& variableToNodeMap);

        template <typename ElementType>
        static Microsoft::MSR::CNTK::ComputationNodeBasePtr GetOutputVariableNode(const Variable& variable,
                                                                                  Microsoft::MSR::CNTK::ComputationNetworkPtr& network,
                                                                                  Microsoft::MSR::CNTK::ComputationNetworkBuilder<ElementType>& builder,
                                                                                  std::unordered_map<Variable, Microsoft::MSR::CNTK::ComputationNodeBasePtr>& variableToNodeMap,
                                                                                  std::unordered_map<Variable, bool>& isVariableRootMap);

        template <typename ElementType>
        static Microsoft::MSR::CNTK::ComputationNodeBasePtr GetNode(const Variable& variable, Microsoft::MSR::CNTK::ComputationNetworkPtr& network,
                                                                    Microsoft::MSR::CNTK::ComputationNetworkBuilder<ElementType>& builder,
                                                                    std::unordered_map<Variable, Microsoft::MSR::CNTK::ComputationNodeBasePtr>& variableToNodeMap,
                                                                    std::unordered_map<Variable, bool>& isVariableRootMap);

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

        bool m_networkMatricesAllocated;
    };

    inline std::vector<CNTK::Axis> DynamicAxesFromInternalDynamicAxisName(const std::wstring& internalDynamicAxisName)
    {
        std::vector<CNTK::Axis> inputVarDynamicAxes;
        if (internalDynamicAxisName.substr(0, CNTK::CompositeFunction::InternalDefaultDynamicAxisName.length()) == CNTK::CompositeFunction::InternalDefaultDynamicAxisName)
            inputVarDynamicAxes = { CNTK::Axis::DefaultDynamicAxis(), CNTK::Axis::DefaultBatchAxis() };
        else if (internalDynamicAxisName.substr(0, CNTK::CompositeFunction::InternalNoSequenceAxisName.length()) == CNTK::CompositeFunction::InternalNoSequenceAxisName)
            inputVarDynamicAxes = { CNTK::Axis::DefaultBatchAxis() };
        else
            inputVarDynamicAxes = { CNTK::Axis(internalDynamicAxisName), CNTK::Axis::DefaultBatchAxis() };

        return inputVarDynamicAxes;
    }

    // Construct the dynamic axis name to be used internally for the CNTK InputNodes
    inline std::wstring InternalDynamicAxisNameFromDynamicAxes(const std::vector<Axis>& dynamicAxes)
    {
        if (dynamicAxes.empty())
            LogicError("Empty dynamic axes set");

        if (dynamicAxes == std::vector<Axis>({ Axis::DefaultBatchAxis() }))
            return CompositeFunction::InternalNoSequenceAxisName;
        else if (dynamicAxes == std::vector<Axis>({ Axis::DefaultDynamicAxis(), Axis::DefaultBatchAxis() }))
            return CompositeFunction::InternalDefaultDynamicAxisName;
        else
            return dynamicAxes[0].Name();
    }
}
