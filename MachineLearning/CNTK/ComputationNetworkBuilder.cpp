// ComputationNetworkBuilder -- helper class for constructing ComputationNetworks and ComputationNodes from C++ (internal and external)
//
// <copyright file="ComputationNode.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ComputationNetworkBuilder.h"

#include "ComputationNode.h"
#include "InputAndParamNodes.h"
#include "LinearAlgebraNodes.h"
#include "NonlinearityNodes.h"
#include "ConvolutionalNodes.h"
#include "RecurrentNodes.h"
#include "DecoderNode.h"
#include "TrainingCriterionNodes.h"
#include "CompositeComputationNodes.h"
#include "EvaluationCriterionNodes.h"

#include <string>

namespace Microsoft { namespace MSR { namespace CNTK {

    using namespace std;

    // create a new node of a type given as a string, with var args so that this can be used at multiple places
    // This function only creates nodes that accept (m_deviceId, nodeName).
    template<typename ElemType>
    /*static*/ shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::NewStandardNode(const std::wstring & nodeType, DEVICEID_TYPE deviceId, const wstring & name)
    {
        // please keep this table sorted
        if (nodeType == CRFNode<ElemType>::TypeName())	return New<CRFNode<ElemType>>(deviceId, name);
        else if (nodeType == ClassBasedCrossEntropyWithSoftmaxNode<ElemType>::TypeName()) return New<ClassBasedCrossEntropyWithSoftmaxNode<ElemType>>(deviceId, name);
        else if (nodeType == ColumnElementTimesNode<ElemType>::TypeName())  return New<ColumnElementTimesNode<ElemType>>(deviceId, name);
        else if (nodeType == CosDistanceNode<ElemType>::TypeName())	    return New<CosDistanceNode<ElemType>>(deviceId, name);
        else if (nodeType == CosDistanceWithNegativeSamplesNode<ElemType>::TypeName()) return New<CosDistanceWithNegativeSamplesNode<ElemType>>(deviceId, name);
        else if (nodeType == CosineNode<ElemType>::TypeName())	            return New<CosineNode<ElemType>>(deviceId, name);
        else if (nodeType == CrossEntropyNode<ElemType>::TypeName())	    return New<CrossEntropyNode<ElemType>>(deviceId, name);
        else if (nodeType == CrossEntropyWithSoftmaxNode<ElemType>::TypeName())	return New<CrossEntropyWithSoftmaxNode<ElemType>>(deviceId, name);
        else if (nodeType == DiagTimesNode<ElemType>::TypeName())	    return New<DiagTimesNode<ElemType>>(deviceId, name);
        else if (nodeType == DropoutNode<ElemType>::TypeName())	            return New<DropoutNode<ElemType>>(deviceId, name);
        else if (nodeType == DummyCriterionNode<ElemType>::TypeName())	    return New<DummyCriterionNode<ElemType>>(deviceId, name);
        else if (nodeType == ElementTimesNode<ElemType>::TypeName())	    return New<ElementTimesNode<ElemType>>(deviceId, name);
        else if (nodeType == ErrorPredictionNode<ElemType>::TypeName())	    return New<ErrorPredictionNode<ElemType>>(deviceId, name);
        else if (nodeType == ExpNode<ElemType>::TypeName())	            return New<ExpNode<ElemType>>(deviceId, name);
        else if (nodeType == FutureValueNode<ElemType>::TypeName())	    return New<FutureValueNode<ElemType>>(deviceId, name);
        else if (nodeType == GMMLogLikelihoodNode<ElemType>::TypeName())    return New<GMMLogLikelihoodNode<ElemType>>(deviceId, name);
        else if (nodeType == InvStdDevNode<ElemType>::TypeName())	    return New<InvStdDevNode<ElemType>>(deviceId, name);
        else if (nodeType == KhatriRaoProductNode<ElemType>::TypeName())    return New<KhatriRaoProductNode<ElemType>>(deviceId, name);
        else if (nodeType == LSTMNode<ElemType>::TypeName())	            return New<LSTMNode<ElemType>>(deviceId, name);
        else if (nodeType == LogNode<ElemType>::TypeName())	            return New<LogNode<ElemType>>(deviceId, name);
        else if (nodeType == LogSoftmaxNode<ElemType>::TypeName())	    return New<LogSoftmaxNode<ElemType>>(deviceId, name);
        else if (nodeType == LookupTableNode<ElemType>::TypeName())	    return New<LookupTableNode<ElemType>>(deviceId, name);
        else if (nodeType == MatrixL1RegNode<ElemType>::TypeName())	    return New<MatrixL1RegNode<ElemType>>(deviceId, name);
        else if (nodeType == MatrixL2RegNode<ElemType>::TypeName())	    return New<MatrixL2RegNode<ElemType>>(deviceId, name);
        else if (nodeType == MeanNode<ElemType>::TypeName())	            return New<MeanNode<ElemType>>(deviceId, name);
        else if (nodeType == MinusNode<ElemType>::TypeName())	            return New<MinusNode<ElemType>>(deviceId, name);
        else if (nodeType == NegateNode<ElemType>::TypeName())	            return New<NegateNode<ElemType>>(deviceId, name);
        else if (nodeType == NoiseContrastiveEstimationNode<ElemType>::TypeName()) return New<NoiseContrastiveEstimationNode<ElemType>>(deviceId, name);
        else if (nodeType == PairNetworkNode<ElemType>::TypeName())	    return New<PairNetworkNode<ElemType>>(deviceId, name);
        else if (nodeType == ParallelNode<ElemType>::TypeName())	    return New<ParallelNode<ElemType>>(deviceId, name);
        else if (nodeType == PastValueNode<ElemType>::TypeName() || nodeType == L"Delay") return New<PastValueNode<ElemType>>(deviceId, name);
        else if (nodeType == PerDimMeanVarDeNormalizationNode<ElemType>::TypeName() || nodeType == L"PerDimMeanVarDeNormalizationNode")	return New<PerDimMeanVarDeNormalizationNode<ElemType>>(deviceId, name);
        else if (nodeType == PerDimMeanVarNormalizationNode<ElemType>::TypeName() || nodeType == L"PerDimMeanVarNormalizationNode")	return New<PerDimMeanVarNormalizationNode<ElemType>>(deviceId, name);
        else if (nodeType == PlusNode<ElemType>::TypeName())	            return New<PlusNode<ElemType>>(deviceId, name);
        else if (nodeType == RectifiedLinearNode<ElemType>::TypeName())	    return New<RectifiedLinearNode<ElemType>>(deviceId, name);
        else if (nodeType == ReshapeNode<ElemType>::TypeName())	            return New<ReshapeNode<ElemType>>(deviceId, name);
        else if (nodeType == RowElementTimesNode<ElemType>::TypeName())	    return New<RowElementTimesNode<ElemType>>(deviceId, name);
        else if (nodeType == RowRepeatNode<ElemType>::TypeName())	    return New<RowRepeatNode<ElemType>>(deviceId, name);
        else if (nodeType == RowSliceNode<ElemType>::TypeName())	    return New<RowSliceNode<ElemType>>(deviceId, name);
        else if (nodeType == RowStackNode<ElemType>::TypeName())	    return New<RowStackNode<ElemType>>(deviceId, name);
        else if (nodeType == ScaleNode<ElemType>::TypeName())	            return New<ScaleNode<ElemType>>(deviceId, name);
        else if (nodeType == SequenceDecoderNode<ElemType>::TypeName())	    return New<SequenceDecoderNode<ElemType>>(deviceId, name);
        else if (nodeType == SigmoidNode<ElemType>::TypeName())	            return New<SigmoidNode<ElemType>>(deviceId, name);
        else if (nodeType == SoftmaxNode<ElemType>::TypeName())	            return New<SoftmaxNode<ElemType>>(deviceId, name);
        else if (nodeType == SquareErrorNode<ElemType>::TypeName())	    return New<SquareErrorNode<ElemType>>(deviceId, name);
        else if (nodeType == StrideTimesNode<ElemType>::TypeName())	    return New<StrideTimesNode<ElemType>>(deviceId, name);
        else if (nodeType == SumColumnElementsNode<ElemType>::TypeName())   return New<SumColumnElementsNode<ElemType>>(deviceId, name);
        else if (nodeType == SumElementsNode<ElemType>::TypeName())	    return New<SumElementsNode<ElemType>>(deviceId, name);
        else if (nodeType == TanhNode<ElemType>::TypeName())	            return New<TanhNode<ElemType>>(deviceId, name);
        else if (nodeType == TimeReverseNode<ElemType>::TypeName())	    return New<TimeReverseNode<ElemType>>(deviceId, name);
        else if (nodeType == TimesNode<ElemType>::TypeName())	            return New<TimesNode<ElemType>>(deviceId, name);
        else if (nodeType == TransposeNode<ElemType>::TypeName())	    return New<TransposeNode<ElemType>>(deviceId, name);
        else if (nodeType == TransposeTimesNode<ElemType>::TypeName())	    return New<TransposeTimesNode<ElemType>>(deviceId, name);
        else return nullptr;
    }

    // create a new node of a type given as a string, with var args so that this can be used at multiple places
    // This function is used for loading, while the above is used for creating standard-type networks.
    template<typename ElemType>
    /*static*/ shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::NewNode(const std::wstring & nodeType, DEVICEID_TYPE deviceId, const wstring & name)
    {
        // TODO: Is this ever called with additional _Args? If not, simplify
        // try first those that accept the standard two constructor arguments
        auto newNode = NewStandardNode(nodeType, deviceId, name);
        if (newNode) return newNode;
        // check more types
        else if (nodeType == AveragePoolingNode<ElemType>::TypeName())	     return New<AveragePoolingNode<ElemType>>(deviceId, name);
        else if (nodeType == ConvolutionNode<ElemType>::TypeName())	     return New<ConvolutionNode<ElemType>>(deviceId, name);
        else if (nodeType == InputValue<ElemType>::SparseTypeName())	     return New<InputValue<ElemType>>(deviceId, name, true);
        else if (nodeType == InputValue<ElemType>::TypeName())	             return New<InputValue<ElemType>>(deviceId, name);
        else if (nodeType == LearnableParameter<ElemType>::TypeName())	     return New<LearnableParameter<ElemType>>(deviceId, name);
        else if (nodeType == MaxPoolingNode<ElemType>::TypeName())	     return New<MaxPoolingNode<ElemType>>(deviceId, name);
        else if (nodeType == SparseLearnableParameter<ElemType>::TypeName()) return New<SparseLearnableParameter<ElemType>>(deviceId, name);
        else return nullptr;
    }

    // -----------------------------------------------------------------------
    // node creation
    // -----------------------------------------------------------------------

    // The following functions create nodes and add them to the net, but don't attach inputs (some don't have inputs).
    // There are special versions for nodes with custom constructors, and a catch-all, CreateComputationNode(), for all others.
    // TODO: Do we really need these? Folks who want to use C++ can instead say net->AddNodeToNet(New<>(...)), which is not that different.
    // TODO: separate into nodes that have inputs and those that duplicate functions with input adding except just not adding inputs. Clear?

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateLearnableParameter(const std::wstring & paramName, const size_t rows, const size_t cols)
    {
        // TODO: in SimpleNetworkBuilder, this is very often followed by InitLearnableParameter()--we should have an overload that just does it right away
        return net.AddNodeToNetWithElemType(New<LearnableParameter<ElemType>>(net.GetDeviceID(), paramName, rows, cols));
    }

    //sparse matrix size is optionally specified
    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateSparseLearnableParameter(const std::wstring & paramName, const size_t rows, const size_t cols, const size_t size)
    {
        return net.AddNodeToNetWithElemType(New<SparseLearnableParameter<ElemType>>(net.GetDeviceID(), paramName, rows, cols, size));
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateInputNode(const std::wstring & inputName, const size_t rows, const size_t cols)
    {
        return net.AddNodeToNetWithElemType(New<InputValue<ElemType>>(net.GetDeviceID(), inputName, rows, cols));
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateSparseInputNode(const std::wstring & inputName, const size_t rows, const size_t cols)
    {
        return net.AddNodeToNetWithElemType(New<InputValue<ElemType>>(net.GetDeviceID(), inputName, rows, cols, true));
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateInputNode(const std::wstring & inputName,
        const size_t imageWidth,
        const size_t imageHeight,
        const size_t imageChannels,
        const size_t numImages)
    {
        return net.AddNodeToNetWithElemType(New<InputValue<ElemType>>(net.GetDeviceID(), inputName, imageWidth, imageHeight, imageChannels, numImages));
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateSparseInputNode(const std::wstring & inputName,
        const size_t imageWidth,
        const size_t imageHeight,
        const size_t imageChannels,
        const size_t numImages)
    {
        return net.AddNodeToNetWithElemType(New<InputValue<ElemType>>(net.GetDeviceID(), inputName, imageWidth, imageHeight, imageChannels, numImages, true));
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreatePairNetworkNode(const std::wstring & inputName, const size_t rows, const size_t cols)
    {
        return net.AddNodeToNetWithElemType(New<PairNetworkNode<ElemType>>(net.GetDeviceID(), inputName, rows, cols));
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateConvolutionNode(const std::wstring & nodeName,
                                                                            const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels,
                                                                            const size_t horizontalSubsample, const size_t verticalSubsample,
                                                                            const bool zeroPadding,
                                                                            const size_t maxTempMemSizeInSamples)
    {
        return net.AddNodeToNetWithElemType(New<ConvolutionNode<ElemType>>(net.GetDeviceID(), nodeName,
            kernelWidth, kernelHeight,
            outputChannels,
            horizontalSubsample,
            verticalSubsample, zeroPadding,
            maxTempMemSizeInSamples));
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateMaxPoolingNode(const std::wstring & nodeName,
        const size_t windowWidth,
        const size_t windowHeight,
        const size_t horizontalSubsample,
        const size_t verticalSubsample)
    {
        return net.AddNodeToNetWithElemType(New<MaxPoolingNode<ElemType>>(net.GetDeviceID(), nodeName,
            windowWidth, windowHeight,
            horizontalSubsample,
            verticalSubsample));
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateAveragePoolingNode(const std::wstring & nodeName, const size_t windowWidth,
        const size_t windowHeight, const size_t horizontalSubsample,
        const size_t verticalSubsample)
    {
        return net.AddNodeToNetWithElemType(New<AveragePoolingNode<ElemType>>(net.GetDeviceID(), nodeName,
            windowWidth, windowHeight,
            horizontalSubsample,
            verticalSubsample));
    }

    // this is the catch-all for all cases not covered as special cases above
    // Unlike the specialized ones above, this one creates nodes by type given as a string.
    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateComputationNode(const std::wstring & nodeType, const std::wstring & nodeName)
    {
        return net.AddNodeToNetWithElemType(NewStandardNode(nodeType, net.GetDeviceID(), nodeName));
    }

    // -----------------------------------------------------------------------
    // node creation
    // -----------------------------------------------------------------------

    // The following functions create nodes and link them to the network and their inputs.
    // TODO: Do we need both this set and the one above that does not add inputs? Can they share more code?

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::PairNetwork(const ComputationNodePtr & a, const std::wstring nodeName)
    {
        if (net.GetNodeFromName(a->NodeName(), nullptr, false) != nullptr)
        {
            fprintf(stderr, "PairNetwork: asked to pair a node with name %ls in another network. However, this network has already a node with the same name. Should avoid this case.\n", a->NodeName().c_str());
            RuntimeError("PairNetwork: asked to pair a node with name in another network. However, this network has already a node with the same name. Should avoid this case.\n");
        }
        return net.AddNodeToNetAndAttachInputs(New<PairNetworkNode<ElemType>>(net.GetDeviceID(), nodeName), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Convolution(const ComputationNodePtr weight,
        const ComputationNodePtr inputValues,
        const size_t kernelWidth,
        const size_t kernelHeight,
        const size_t outputChannels,
        const size_t horizontalSubsample,
        const size_t verticalSubsample,
        const bool zeroPadding,
        const std::wstring nodeName,
        const size_t maxTempMemSizeInSamples)
    {
        return net.AddNodeToNetAndAttachInputs(New<ConvolutionNode<ElemType>>(net.GetDeviceID(), nodeName,
            kernelWidth, kernelHeight,
            outputChannels,
            horizontalSubsample,
            verticalSubsample, zeroPadding,
            maxTempMemSizeInSamples),
            weight, inputValues);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::MaxPooling(const ComputationNodePtr inputValues,
        const size_t windowWidth,
        const size_t windowHeight,
        const size_t horizontalSubsample,
        const size_t verticalSubsample,
        const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<MaxPoolingNode<ElemType>>(net.GetDeviceID(), nodeName,
            windowWidth, windowHeight,
            horizontalSubsample,
            verticalSubsample),
            inputValues);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::AveragePooling(const ComputationNodePtr inputValues,
        const size_t windowWidth,
        const size_t windowHeight,
        const size_t horizontalSubsample,
        const size_t verticalSubsample,
        const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<AveragePoolingNode<ElemType>>(net.GetDeviceID(), nodeName,
            windowWidth, windowHeight,
            horizontalSubsample,
            verticalSubsample),
            inputValues);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::ErrorPrediction(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<ErrorPredictionNode<ElemType>>(net.GetDeviceID(), nodeName), a, b);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::PerDimMeanVarNormalization(const ComputationNodePtr feature, const ComputationNodePtr mean,
        const ComputationNodePtr InvStdDev, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<PerDimMeanVarNormalizationNode<ElemType>>(net.GetDeviceID(), nodeName), feature, mean, InvStdDev);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::PerDimMeanVarDeNormalization(const ComputationNodePtr feature, const ComputationNodePtr mean,
        const ComputationNodePtr InvStdDev, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<PerDimMeanVarDeNormalizationNode<ElemType>>(net.GetDeviceID(), nodeName), feature, mean, InvStdDev);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::SquareError(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<SquareErrorNode<ElemType>>(net.GetDeviceID(), nodeName), a, b);
    }


    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::SequenceDecoder(const ComputationNodePtr label, const ComputationNodePtr prediction, const ComputationNodePtr pairscore, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<SequenceDecoderNode<ElemType>>(net.GetDeviceID(), nodeName), label, prediction, pairscore);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CrossEntropyWithSoftmax(const ComputationNodePtr label, const ComputationNodePtr prediction, const std::wstring nodeName)

    {
        return net.AddNodeToNetAndAttachInputs(New<CrossEntropyWithSoftmaxNode<ElemType>>(net.GetDeviceID(), nodeName), label, prediction);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::NoiseContrastiveEstimation(const ComputationNodePtr label, const ComputationNodePtr prediction,
        const ComputationNodePtr input_weight,
        const ComputationNodePtr input_bias, const std::wstring nodeName,
        NCEEvalMode mode)
    {
        return net.AddNodeToNetAndAttachInputs(New<NoiseContrastiveEstimationNode<ElemType>>(net.GetDeviceID(), nodeName, mode), label, prediction, input_weight, input_bias);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::ClassCrossEntropyWithSoftmax(const ComputationNodePtr label, const ComputationNodePtr prediction,
        const ComputationNodePtr input_weight,
        const ComputationNodePtr cls_log_post_prob,
        const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<ClassBasedCrossEntropyWithSoftmaxNode<ElemType>>(net.GetDeviceID(), nodeName), label, prediction, input_weight, cls_log_post_prob);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CRF(const ComputationNodePtr label,
        const ComputationNodePtr postDepScore,
        const ComputationNodePtr transition_score,
        const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<CRFNode<ElemType>>(net.GetDeviceID(), nodeName), label, postDepScore, transition_score);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::DummyCriterion(const ComputationNodePtr objectives, const ComputationNodePtr derivatives, const ComputationNodePtr prediction, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<DummyCriterionNode<ElemType>>(net.GetDeviceID(), nodeName), objectives, derivatives, prediction);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::LSTM(const ComputationNodePtr obs,
        const ComputationNodePtr inputGate,
        const ComputationNodePtr forgetGate,
        const ComputationNodePtr outputGate,
        const ComputationNodePtr memoryCellWgt,
        const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<LSTMNode<ElemType>>(net.GetDeviceID(), nodeName), obs, inputGate, forgetGate, outputGate, memoryCellWgt);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CrossEntropy(const ComputationNodePtr label, const ComputationNodePtr prediction, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<CrossEntropyNode<ElemType>>(net.GetDeviceID(), nodeName), label, prediction);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::MatrixL1Reg(const ComputationNodePtr a, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<MatrixL1RegNode<ElemType>>(net.GetDeviceID(), nodeName), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::MatrixL2Reg(const ComputationNodePtr a, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<MatrixL2RegNode<ElemType>>(net.GetDeviceID(), nodeName), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Mean(const ComputationNodePtr a, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<MeanNode<ElemType>>(net.GetDeviceID(), nodeName), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::InvStdDev(const ComputationNodePtr a, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<InvStdDevNode<ElemType>>(net.GetDeviceID(), nodeName), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Negate(const ComputationNodePtr a, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<NegateNode<ElemType>>(net.GetDeviceID(), nodeName), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::RectifiedLinear(const ComputationNodePtr a, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<RectifiedLinearNode<ElemType>>(net.GetDeviceID(), nodeName), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Sigmoid(const ComputationNodePtr a, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<SigmoidNode<ElemType>>(net.GetDeviceID(), nodeName), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Tanh(const ComputationNodePtr a, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<TanhNode<ElemType>>(net.GetDeviceID(), nodeName), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Exp(const ComputationNodePtr a, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<ExpNode<ElemType>>(net.GetDeviceID(), nodeName), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Log(const ComputationNodePtr a, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<LogNode<ElemType>>(net.GetDeviceID(), nodeName), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Cos(const ComputationNodePtr a, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<CosineNode<ElemType>>(net.GetDeviceID(), nodeName), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Softmax(const ComputationNodePtr a, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<SoftmaxNode<ElemType>>(net.GetDeviceID(), nodeName), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::LogSoftmax(const ComputationNodePtr a, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<LogSoftmaxNode<ElemType>>(net.GetDeviceID(), nodeName), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Sum(const ComputationNodePtr a, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<SumElementsNode<ElemType>>(net.GetDeviceID(), nodeName), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Scale(const ComputationNodePtr scalar, const ComputationNodePtr matrix, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<ScaleNode<ElemType>>(net.GetDeviceID(), nodeName), scalar, matrix);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Transpose(const ComputationNodePtr matrix, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<TransposeNode<ElemType>>(net.GetDeviceID(), nodeName), matrix);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Times(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<TimesNode<ElemType>>(net.GetDeviceID(), nodeName), a, b);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::TransposeTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<TransposeTimesNode<ElemType>>(net.GetDeviceID(), nodeName), a, b);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::ElementTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<ElementTimesNode<ElemType>>(net.GetDeviceID(), nodeName), a, b);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::RowElementTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<RowElementTimesNode<ElemType>>(net.GetDeviceID(), nodeName), a, b);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::ColumnElementTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<ColumnElementTimesNode<ElemType>>(net.GetDeviceID(), nodeName), a, b);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::StrideTimes(const ComputationNodePtr a, const ComputationNodePtr b, const ComputationNodePtr c, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<StrideTimesNode<ElemType>>(net.GetDeviceID(), nodeName), a, b, c);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::DiagTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<DiagTimesNode<ElemType>>(net.GetDeviceID(), nodeName), a, b);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CosDistance(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<CosDistanceNode<ElemType>>(net.GetDeviceID(), nodeName), a, b);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::KhatriRaoProduct(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<KhatriRaoProductNode<ElemType>>(net.GetDeviceID(), nodeName), a, b);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Plus(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<PlusNode<ElemType>>(net.GetDeviceID(), nodeName), a, b);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Minus(const ComputationNodePtr a,
        const ComputationNodePtr b,
        const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<MinusNode<ElemType>>(net.GetDeviceID(), nodeName), a, b);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Dropout(const ComputationNodePtr a, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<DropoutNode<ElemType>>(net.GetDeviceID(), nodeName), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Reshape(const ComputationNodePtr a,
        const size_t num_rows,
        const size_t img_width,
        const size_t img_height,
        const size_t img_channels,
        const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<ReshapeNode<ElemType>>(net.GetDeviceID(), nodeName, num_rows, img_width, img_height, img_channels), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::RowRepeat(const ComputationNodePtr a, const size_t num_repeat, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<RowRepeatNode<ElemType>>(net.GetDeviceID(), nodeName, num_repeat), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::PastValue(const ComputationNodePtr a, const float initHiddenActivity, const size_t row_size, const size_t col_size, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<PastValueNode<ElemType>>(net.GetDeviceID(), nodeName, initHiddenActivity, row_size, col_size), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::FutureValue(const ComputationNodePtr a, const float initHiddenActivity, const size_t row_size, const size_t col_size, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<FutureValueNode<ElemType>>(net.GetDeviceID(), nodeName, initHiddenActivity, row_size, col_size), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Parallel(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<ParallelNode<ElemType>>(net.GetDeviceID(), nodeName), a, b);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::RowSlice(const ComputationNodePtr a, const size_t start_index, const size_t num_rows, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<RowSliceNode<ElemType>>(net.GetDeviceID(), nodeName, start_index, num_rows), a);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::RowStack(const std::vector<ComputationNodePtr> pinputs, const std::wstring nodeName)
    {
        vector<ComputationNodeBasePtr> inputs(pinputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
            inputs[i] = pinputs[i]; // convert to ComputationNodeBasePtr
        return net.AddNodeToNetAndAttachInputs(New<RowStackNode<ElemType>>(net.GetDeviceID(), nodeName), inputs);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::GMMLogLikelihood(const ComputationNodePtr unnormedPrior,
        const ComputationNodePtr mean,
        const ComputationNodePtr logStddev,
        const ComputationNodePtr feature,
        const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<GMMLogLikelihoodNode<ElemType>>(net.GetDeviceID(), nodeName), unnormedPrior, mean, logStddev, feature);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::TimeReverse(const ComputationNodePtr input, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<TimeReverseNode<ElemType>>(net.GetDeviceID(), nodeName), input);
    }

    template<typename ElemType> shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::LookupTable(const ComputationNodePtr dictionary, const ComputationNodePtr input, const std::wstring nodeName)
    {
        return net.AddNodeToNetAndAttachInputs(New<LookupTableNode<ElemType>>(net.GetDeviceID(), nodeName), dictionary, input);
    }

    template class ComputationNetworkBuilder<float>;
    template class ComputationNetworkBuilder<double>;

}}}
