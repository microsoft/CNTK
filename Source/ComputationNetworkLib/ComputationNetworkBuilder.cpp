//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ComputationNetworkBuilder -- helper class for constructing ComputationNetworks and ComputationNodes from C++ (internal and external)
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "Basics.h"
#include "ComputationNetworkBuilder.h"
#include "ComputationNode.h"

#include "ConvolutionalNodes.h"
#include "RNNNodes.h"
#include "DeprecatedNodes.h"
#include "EvaluationNodes.h"
#include "InputAndParamNodes.h"
#include "LinearAlgebraNodes.h"
#include "NonlinearityNodes.h"
#include "PreComputeNodes.h"
#include "ReshapingNodes.h"
#include "RecurrentNodes.h"
#include "SpecialPurposeNodes.h"
#include "TrainingNodes.h"

#include <string>

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

// create a new node of a type given as a string, with var args so that this can be used at multiple places
template <class ElemType, class... _Types>
static shared_ptr<ComputationNode<ElemType>> CreateStandardNode(const std::wstring& nodeType, _Types&&... _Args)
{
    // please keep this table sorted
#ifdef COMING_SOON
         if (nodeType == OperationNameOf(CRFNode))                              return New<CRFNode<ElemType>>(forward<_Types>(_Args)...);
    else
#endif
         if (nodeType == OperationNameOf(AbsNode))                              return New<AbsNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(ClassBasedCrossEntropyWithSoftmaxNode))return New<ClassBasedCrossEntropyWithSoftmaxNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(ClassificationErrorNode))              return New<ClassificationErrorNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(ClipNode))                             return New<ClipNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(CosDistanceNode))                      return New<CosDistanceNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(CosDistanceWithNegativeSamplesNode))   return New<CosDistanceWithNegativeSamplesNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(CosineNode))                           return New<CosineNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(CropNode))                             return New<CropNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(CrossEntropyNode))                     return New<CrossEntropyNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(CrossEntropyWithSoftmaxNode))          return New<CrossEntropyWithSoftmaxNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(DiagonalNode))                         return New<DiagonalNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(DiagTimesNode))                        return New<DiagTimesNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(DropoutNode))                          return New<DropoutNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(DummyCriterionNode))                   return New<DummyCriterionNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(DynamicAxisNode))                      return New<DynamicAxisNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(EditDistanceErrorNode))                return New<EditDistanceErrorNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(ElementTimesNode))                     return New<ElementTimesNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(EnvironmentInputNode))                 return New<EnvironmentInputNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(EpochAccumulatorNode))                 return New<EpochAccumulatorNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(EqualNode))                            return New<EqualNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(ExpNode))                              return New<ExpNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(FloorNode))                            return New<FloorNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(FutureValueNode))                      return New<FutureValueNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(GatherPackedNode))                     return New<GatherPackedNode<ElemType>>(forward<_Types>(_Args)...);
#ifdef COMING_SOON
    else if (nodeType == OperationNameOf(GMMLogLikelihoodNode))                 return New<GMMLogLikelihoodNode<ElemType>>(forward<_Types>(_Args)...);
#endif
    else if (nodeType == OperationNameOf(GreaterEqualNode))                     return New<GreaterEqualNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(GreaterNode))                          return New<GreaterNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(HardmaxNode))                          return New<HardmaxNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(IfNode))                               return New<IfNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(InvStdDevNode))                        return New<InvStdDevNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(LambdaRankNode))                       return New<LambdaRankNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(NDCG1EvalNode))                        return New<NDCG1EvalNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(KhatriRaoProductNode))                 return New<KhatriRaoProductNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(LessEqualNode))                        return New<LessEqualNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(LessNode))                             return New<LessNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(LogNode))                              return New<LogNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(LogPlusNode))                          return New<LogPlusNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(LogSoftmaxNode))                       return New<LogSoftmaxNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(LookupTableNode))                      return New<LookupTableNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(MatrixL1RegNode))                      return New<MatrixL1RegNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(MatrixL2RegNode))                      return New<MatrixL2RegNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(MeanNode))                             return New<MeanNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(MinusNode))                            return New<MinusNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(NegateNode))                           return New<NegateNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(NotEqualNode))                         return New<NotEqualNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(NoiseContrastiveEstimationNode))       return New<NoiseContrastiveEstimationNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(OptimizedRNNStackNode))                return New<OptimizedRNNStackNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(PackedIndexNode))                      return New<PackedIndexNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(PastValueNode))                        return New<PastValueNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(PerDimMeanVarNormalizationNode))       return New<PerDimMeanVarNormalizationNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(PerDimMeanVarDeNormalizationNode))     return New<PerDimMeanVarDeNormalizationNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(PassNode))                             return New<PassNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(PlusNode))                             return New<PlusNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(RandomSampleNode))                     return New<RandomSampleNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(RandomSampleInclusionFrequencyNode))   return New<RandomSampleInclusionFrequencyNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(ReconcileDynamicAxisNode))             return New<ReconcileDynamicAxisNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(ReciprocalNode))                       return New<ReciprocalNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(RectifiedLinearNode))                  return New<RectifiedLinearNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(ReduceElementsNode))                   return New<ReduceElementsNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(ReshapeNode))                          return New<ReshapeNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(RowRepeatNode))                        return New<RowRepeatNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(RowStackNode))                         return New<RowStackNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(ScatterPackedNode))                    return New<ScatterPackedNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(SequenceWithSoftmaxNode))              return New<SequenceWithSoftmaxNode<ElemType>>(forward<_Types>(_Args)...);
#ifdef COMING_SOON
    else if (nodeType == OperationNameOf(SequenceDecoderNode))                  return New<SequenceDecoderNode<ElemType>>(forward<_Types>(_Args)...);
#endif
#ifdef COMING_SOON
    else if (nodeType == OperationNameOf(ShiftNode))                            return New<ShiftNode<ElemType>>(forward<_Types>(_Args)...);
#endif
    else if (nodeType == OperationNameOf(SigmoidNode))                          return New<SigmoidNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(SinNode))                              return New<SinNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(SliceNode))                            return New<SliceNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(SoftmaxNode))                          return New<SoftmaxNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(SqrtNode))                             return New<SqrtNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(SquareErrorNode))                      return New<SquareErrorNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(LogisticNode))                         return New<LogisticNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(SumColumnElementsNode))                return New<SumColumnElementsNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(SumElementsNode))                      return New<SumElementsNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(TanhNode))                             return New<TanhNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(TraceNode))                            return New<TraceNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(TimesNode))                            return New<TimesNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(TransposeDimensionsNode))              return New<TransposeDimensionsNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(TransposeTimesNode))                   return New<TransposeTimesNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(QuantizedTimesNode))                   return New<QuantizedTimesNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(WhereNode))                            return New<WhereNode<ElemType>>(forward<_Types>(_Args)...);
    // legacy names we also support for back compat of model-files
    else if (nodeType == L"ColumnElementTimes")                                 return New<ElementTimesNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == L"ErrorPrediction")                                    return New<ClassificationErrorNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == L"Delay")                                              return New<PastValueNode<ElemType>>(forward<_Types>(_Args)...);
    // TODO: DiagTimes is also an alias of ElementTimes; current separate implementation is unnecessary.
    else if (nodeType == L"PerDimMeanVarNormalizationNode")                     return New<PerDimMeanVarNormalizationNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == L"PerDimMeanVarDeNormalizationNode")                   return New<PerDimMeanVarDeNormalizationNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == L"ReconcileMBLayout")                                  return New<ReconcileDynamicAxisNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == L"RNN")                                                return New<OptimizedRNNStackNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == L"RowElementTimes")                                    return New<ElementTimesNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == L"RowSlice")                                           return New<SliceNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == L"Scale")                                              return New<ElementTimesNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == L"Transpose")                                          return New<TransposeDimensionsNode<ElemType>>(forward<_Types>(_Args)...);
#if 1
    else if (nodeType == OperationNameOf(LegacyReshapeNode))                    return New<LegacyReshapeNode<ElemType>>(forward<_Types>(_Args)...);
#endif
    else if (nodeType == OperationNameOf(MaxUnpoolingNode))                     return New<MaxUnpoolingNode<ElemType>>(forward<_Types>(_Args)...);
    else InvalidArgument("Attempted to instantiate undefined operation %ls.", nodeType.c_str());
}

// create a new node of a type given as a string, with var args so that this can be used at multiple places
// This function is used for loading, while the above is used for creating standard-type networks.
template <class ElemType, class... _Types>
static shared_ptr<ComputationNode<ElemType>> CreateNode(const std::wstring& nodeType, _Types&&... _Args)
{
    // check more types
    if      (nodeType == OperationNameOf(AveragePoolingNode))       return New<AveragePoolingNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(BatchNormalizationNode))   return New<BatchNormalizationNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(ConvolutionNode))          return New<ConvolutionNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(PoolingNode))              return New<PoolingNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(SparseInputValue))         return New<SparseInputValue<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(InputValue))               return New<InputValue<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(LearnableParameter))       return New<LearnableParameter<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(MaxPoolingNode))           return New<MaxPoolingNode<ElemType>>(forward<_Types>(_Args)...);
    else if (nodeType == OperationNameOf(ROIPoolingNode))           return New<ROIPoolingNode<ElemType>>(forward<_Types>(_Args)...);
    else return CreateStandardNode<ElemType>(nodeType, forward<_Types>(_Args)...);
}

// this function is called from SimpleNetworkBuilder and old NDL
template <class ElemType>
/*static*/ shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::NewStandardNode(const std::wstring& nodeType, DEVICEID_TYPE deviceId, const wstring& name)
{
    return CreateStandardNode<ElemType>(nodeType, deviceId, name);
}

// this function is used when loading from file
template <class ElemType>
/*static*/ shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::NewNode(const std::wstring& nodeType, DEVICEID_TYPE deviceId, const wstring& name)
{
    return CreateNode<ElemType>(nodeType, deviceId, name);
}

shared_ptr<ComputationNodeBase> NewComputationNodeFromConfig(const Microsoft::MSR::ScriptableObjects::IConfigRecordPtr configp)
{
    wstring precision = configp->Get(L"precision"); // dispatch on ElemType
    wstring operationName = configp->Get(L"operation");
    ComputationNodeBasePtr node;
    if (precision == L"float")
        node = CreateNode<float>(operationName, configp);
    else if (precision == L"double")
        node = CreateNode<double>(operationName, configp);
    else
        RuntimeError("NewStandardNode: Invalid value '%ls' for 'precision' parameter. Must be 'float' or 'double'.", precision.c_str());
    // add a tag
    // Tags are used to declare special node types to ComputationNetwork.
    // For now we support only a single tag, but we could in the future easily extend this to an array of tags.
    wstring tag = configp->Get(L"tag");
    if (!tag.empty())
        node->SetTag(tag);
    return node;
}

// -----------------------------------------------------------------------
// node creation
// -----------------------------------------------------------------------

// The following functions create nodes and add them to the net, but don't attach inputs (some don't have inputs).
// There are special versions for nodes with custom constructors, and a catch-all, CreateComputationNode(), for all others.
// TODO: Do we really need these? Folks who want to use C++ can instead say net->AddNodeToNet(New<>(...)), which is not that different.
// TODO: separate into nodes that have inputs and those that duplicate functions with input adding except just not adding inputs. Clear?

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateLearnableParameter(const std::wstring& paramName, const size_t rows, const size_t cols)
{
    // TODO: in SimpleNetworkBuilder, this is very often followed by InitLearnableParameter()--we should have an overload that just does it right away
    return net.AddNodeToNetWithElemType(New<LearnableParameter<ElemType>>(net.GetDeviceId(), paramName, rows, cols));
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateLearnableParameter(const std::wstring& paramName, const TensorShape& tensorShape)
{
    return net.AddNodeToNetWithElemType(New<LearnableParameter<ElemType>>(net.GetDeviceId(), paramName, tensorShape));
}

// TODO: change these to take an actual object instead of a name for dynamicAxis
template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateInputNode(const std::wstring& inputName, const size_t rows, const wstring& dynamicAxisName)
{
    return net.AddNodeToNetWithElemType(New<InputValue<ElemType>>(net.GetDeviceId(), inputName, rows, dynamicAxisName));
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateSparseInputNode(const std::wstring& inputName, const size_t rows, const wstring& dynamicAxisName)
{
    return net.AddNodeToNetWithElemType(New<SparseInputValue<ElemType>>(net.GetDeviceId(), inputName, rows, dynamicAxisName));
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateInputNode(const std::wstring& inputName, const TensorShape& sampleLayout, const wstring& dynamicAxisName)
{
    return net.AddNodeToNetWithElemType(New<InputValue<ElemType>>(net.GetDeviceId(), inputName, sampleLayout, dynamicAxisName));
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateSparseInputNode(const std::wstring& inputName, const TensorShape& imageLayout, const wstring& dynamicAxisName)
{
    return net.AddNodeToNetWithElemType(New<SparseInputValue<ElemType>>(net.GetDeviceId(), inputName, imageLayout, dynamicAxisName));
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateConvolutionNode(const std::wstring& nodeName,
                                                                                                 const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels,
                                                                                                 const size_t horizontalSubsample, const size_t verticalSubsample,
                                                                                                 ImageLayoutKind imageLayoutKind, const bool zeroPadding,
                                                                                                 const size_t maxTempMemSizeInSamples)
{
    return net.AddNodeToNetWithElemType(New<ConvolutionNode<ElemType>>(net.GetDeviceId(), nodeName,
                                                                       kernelWidth, kernelHeight, outputChannels,
                                                                       horizontalSubsample, verticalSubsample, imageLayoutKind,
                                                                       zeroPadding,
                                                                       maxTempMemSizeInSamples));
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateConvolutionNode(const std::wstring& nodeName, const TensorShape& kernelShape, const TensorShape& mapCount,
                                                                                                 const TensorShape& strideShape, const std::vector<bool>& sharing,
                                                                                                 const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                                                                                                 bool transpose, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples)
{
    return net.AddNodeToNetWithElemType(New<ConvolutionNode<ElemType>>(net.GetDeviceId(), nodeName,
                                                                       kernelShape, mapCount, strideShape,
                                                                       sharing, autoPadding, lowerPad, upperPad,
                                                                       transpose, imageLayout, maxTempMemSizeInSamples));
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreatePoolingNode(const std::wstring& nodeName, PoolKind poolKind, const TensorShape& kernelShape, const TensorShape& strideShape,
                                                                                             const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                                                                                             ImageLayoutKind imageLayout)
{
    return net.AddNodeToNetWithElemType(New<PoolingNode<ElemType>>(net.GetDeviceId(), nodeName,
                                                                   poolKind, kernelShape, strideShape, autoPadding, lowerPad, upperPad, imageLayout));
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateMaxPoolingNode(const std::wstring& nodeName,
                                                                                                const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind)
{
    return net.AddNodeToNetWithElemType(New<MaxPoolingNode<ElemType>>(net.GetDeviceId(), nodeName, windowWidth, windowHeight, horizontalSubsample, verticalSubsample, imageLayoutKind));
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateAveragePoolingNode(const std::wstring& nodeName,
                                                                                                    const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind)
{
    return net.AddNodeToNetWithElemType(New<AveragePoolingNode<ElemType>>(net.GetDeviceId(), nodeName, windowWidth, windowHeight, horizontalSubsample, verticalSubsample, imageLayoutKind));
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateROIPoolingNode(const std::wstring& nodeName, const TensorShape& roiOutputShape)
{
    return net.AddNodeToNetWithElemType(New<ROIPoolingNode<ElemType>>(net.GetDeviceId(), nodeName, roiOutputShape));
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateReconcileDynamicAxisNode(const std::wstring& nodeName)
{
    return net.AddNodeToNetWithElemType(New<ReconcileDynamicAxisNode<ElemType>>(net.GetDeviceId(), nodeName));
}

// this is the catch-all for all cases not covered as special cases above
// Unlike the specialized ones above, this one creates nodes by type given as a string.
template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CreateComputationNode(const std::wstring& nodeType, const std::wstring& nodeName)
{
    return net.AddNodeToNetWithElemType(NewStandardNode(nodeType, net.GetDeviceId(), nodeName));
}

// -----------------------------------------------------------------------
// node creation
// -----------------------------------------------------------------------

// The following functions create nodes and link them to the network and their inputs.
// TODO: Do we need both this set and the one above that does not add inputs? Can they share more code?

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Convolution(const ComputationNodePtr weight,
                                                                                       const ComputationNodePtr inputValues,
                                                                                       const size_t kernelWidth, const size_t kernelHeight, const size_t outputChannels, 
                                                                                       const size_t horizontalSubsample, const size_t verticalSubsample, 
                                                                                       ImageLayoutKind imageLayoutKind, const bool zeroPadding, const size_t maxTempMemSizeInSamples,
                                                                                       const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<ConvolutionNode<ElemType>>(net.GetDeviceId(), nodeName,
                                                                          kernelWidth, kernelHeight, outputChannels, horizontalSubsample, verticalSubsample, imageLayoutKind, zeroPadding,
                                                                          maxTempMemSizeInSamples), { weight, inputValues });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Convolution(const ComputationNodePtr weight,
                                                                                       const ComputationNodePtr inputValues, 
                                                                                       const TensorShape& kernelShape, const TensorShape& mapCount, 
                                                                                       const TensorShape& strideShape, const std::vector<bool>& sharing,
                                                                                       const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                                                                                       bool transpose, ImageLayoutKind imageLayout, size_t maxTempMemSizeInSamples,
                                                                                       const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<ConvolutionNode<ElemType>>(net.GetDeviceId(), nodeName,
                                                                          kernelShape, mapCount, strideShape,
                                                                          sharing, autoPadding, lowerPad, upperPad,
                                                                          transpose, imageLayout, maxTempMemSizeInSamples),
                                                                          { weight, inputValues });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Pooling(const ComputationNodePtr inputValues,
                                                                                   PoolKind poolKind, const TensorShape& kernelShape, const TensorShape& strideShape,
                                                                                   const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                                                                                   ImageLayoutKind imageLayout,
                                                                                   const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<PoolingNode<ElemType>>(net.GetDeviceId(), nodeName,
                                                                      poolKind, kernelShape, strideShape, autoPadding, lowerPad, upperPad, imageLayout),
                                                                      { inputValues });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::MaxUnpooling(const ComputationNodePtr unpoolInputValues,
                                                                                        const ComputationNodePtr poolInputValues,
                                                                                        const TensorShape& kernelShape, const TensorShape& strideShape,
                                                                                        const std::vector<bool>& autoPadding, const TensorShape& lowerPad, const TensorShape& upperPad,
                                                                                        ImageLayoutKind imageLayout,
                                                                                        const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<MaxUnpoolingNode<ElemType>>(net.GetDeviceId(), nodeName,
                                                                           kernelShape, strideShape, autoPadding, lowerPad, upperPad, imageLayout),
                                                                           { unpoolInputValues, poolInputValues });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::MaxPooling(const ComputationNodePtr inputValues,
                                                                                      const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind,
                                                                                      const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<MaxPoolingNode<ElemType>>(net.GetDeviceId(), nodeName, windowWidth, windowHeight, horizontalSubsample, verticalSubsample, imageLayoutKind), { inputValues });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::AveragePooling(const ComputationNodePtr inputValues,
                                                                                          const size_t windowWidth, const size_t windowHeight, const size_t horizontalSubsample, const size_t verticalSubsample, ImageLayoutKind imageLayoutKind,
                                                                                          const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<AveragePoolingNode<ElemType>>(net.GetDeviceId(), nodeName, windowWidth, windowHeight, horizontalSubsample, verticalSubsample, imageLayoutKind), { inputValues });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::ROIPooling(const ComputationNodePtr inputValues, const ComputationNodePtr inputROIs, const TensorShape& roiOutputShape, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<ROIPoolingNode<ElemType>>(net.GetDeviceId(), nodeName, roiOutputShape), { inputValues, inputROIs });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::ReconcileDynamicAxis(const ComputationNodePtr dataInput, const ComputationNodePtr layoutInput, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<ReconcileDynamicAxisNode<ElemType>>(net.GetDeviceId(), nodeName), { dataInput, layoutInput });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Crop(const ComputationNodePtr input1, const ComputationNodePtr input2, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<CropNode<ElemType>>(net.GetDeviceId(), nodeName), { input1, input2 });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Crop(const ComputationNodePtr input1, const ComputationNodePtr input2, size_t offsetX, size_t offsetY, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<CropNode<ElemType>>(offsetX, offsetY, net.GetDeviceId(), nodeName), { input1, input2 });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Crop(const ComputationNodePtr input1, const ComputationNodePtr input2, const ComputationNodePtr eqNode1, const ComputationNodePtr eqNode2, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<CropNode<ElemType>>(net.GetDeviceId(), nodeName), { input1, input2, eqNode1, eqNode2 });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::ClassificationError(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<ClassificationErrorNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::EditDistanceError(const ComputationNodePtr a, const ComputationNodePtr b, float subPen, float delPen, float insPen, bool squashInputs, vector<size_t> samplesToIgnore, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<EditDistanceErrorNode<ElemType>>(net.GetDeviceId(), subPen, delPen, insPen, squashInputs, samplesToIgnore, nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::PerDimMeanVarNormalization(const ComputationNodePtr feature, const ComputationNodePtr mean,
                                                                                                      const ComputationNodePtr InvStdDev, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<PerDimMeanVarNormalizationNode<ElemType>>(net.GetDeviceId(), nodeName), { feature, mean, InvStdDev });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::PerDimMeanVarDeNormalization(const ComputationNodePtr feature, const ComputationNodePtr mean,
                                                                                                        const ComputationNodePtr InvStdDev, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<PerDimMeanVarDeNormalizationNode<ElemType>>(net.GetDeviceId(), nodeName), { feature, mean, InvStdDev });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::SquareError(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<SquareErrorNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Logistic(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<LogisticNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Logistic(const ComputationNodePtr a, const ComputationNodePtr b, const ComputationNodePtr c, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<LogisticNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b, c });
}

#ifdef COMING_SOON
template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::SequenceDecoder(const ComputationNodePtr label, const ComputationNodePtr prediction, const ComputationNodePtr pairscore, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<SequenceDecoderNode<ElemType>>(net.GetDeviceId(), nodeName), { label, prediction, pairscore });
}
#endif

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CrossEntropyWithSoftmax(const ComputationNodePtr label, const ComputationNodePtr prediction, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<CrossEntropyWithSoftmaxNode<ElemType>>(net.GetDeviceId(), nodeName), { label, prediction });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::LambdaRank(const ComputationNodePtr gain, const ComputationNodePtr prediction, const ComputationNodePtr queryId, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<LambdaRankNode<ElemType>>(net.GetDeviceId(), nodeName), { gain, prediction, queryId });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::NDCG1Eval(const ComputationNodePtr gain, const ComputationNodePtr prediction, const ComputationNodePtr queryId, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<NDCG1EvalNode<ElemType>>(net.GetDeviceId(), nodeName), { gain, prediction, queryId });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::SequenceWithSoftmax(const ComputationNodePtr label, const ComputationNodePtr prediction, const ComputationNodePtr loglikelihood, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<SequenceWithSoftmaxNode<ElemType>>(net.GetDeviceId(), nodeName), { label, prediction, loglikelihood });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::NoiseContrastiveEstimation(const ComputationNodePtr label, const ComputationNodePtr prediction,
                                                                                                      const ComputationNodePtr input_weight,
                                                                                                      const ComputationNodePtr input_bias, const std::wstring nodeName,
                                                                                                      NCEEvalMode mode)
{
    return net.AddNodeToNetAndAttachInputs(New<NoiseContrastiveEstimationNode<ElemType>>(net.GetDeviceId(), nodeName, mode), { label, prediction, input_weight, input_bias });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::ClassCrossEntropyWithSoftmax(const ComputationNodePtr label, const ComputationNodePtr prediction,
                                                                                                        const ComputationNodePtr input_weight,
                                                                                                        const ComputationNodePtr cls_log_post_prob,
                                                                                                        const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<ClassBasedCrossEntropyWithSoftmaxNode<ElemType>>(net.GetDeviceId(), nodeName), { label, prediction, input_weight, cls_log_post_prob });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Clip(const ComputationNodePtr a, const ComputationNodePtr b, const ComputationNodePtr c, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<ClipNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b, c });
}

#ifdef COMING_SOON
template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CRF(const ComputationNodePtr label,
                                                                               const ComputationNodePtr postDepScore,
                                                                               const ComputationNodePtr transition_score,
                                                                               const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<CRFNode<ElemType>>(net.GetDeviceId(), nodeName), { label, postDepScore, transition_score });
}
#endif

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::DummyCriterion(const ComputationNodePtr objectives, const ComputationNodePtr derivatives, const ComputationNodePtr prediction, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<DummyCriterionNode<ElemType>>(net.GetDeviceId(), nodeName), { objectives, derivatives, prediction });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CrossEntropy(const ComputationNodePtr label, const ComputationNodePtr prediction, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<CrossEntropyNode<ElemType>>(net.GetDeviceId(), nodeName), { label, prediction });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::MatrixL1Reg(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<MatrixL1RegNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::MatrixL2Reg(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<MatrixL2RegNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Mean(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<MeanNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Pass(const ComputationNodePtr a, const std::wstring& nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<PassNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::DynamicAxis(const ComputationNodePtr a, const std::wstring& nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<DynamicAxisNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::InvStdDev(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<InvStdDevNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Negate(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<NegateNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::RectifiedLinear(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<RectifiedLinearNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Sigmoid(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<SigmoidNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Tanh(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<TanhNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Exp(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<ExpNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Log(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<LogNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Cos(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<CosineNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Sin(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<SinNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Abs(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<AbsNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Floor(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<FloorNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Hardmax(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<HardmaxNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::If(const ComputationNodePtr a, const ComputationNodePtr b, const ComputationNodePtr c, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<IfNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b, c });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Softmax(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<SoftmaxNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::LogSoftmax(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<LogSoftmaxNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Reciprocal(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<ReciprocalNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Sqrt(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<SqrtNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Sum(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<SumElementsNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::TransposeDimensions(const ComputationNodePtr matrix, int dim1, int dim2, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<TransposeDimensionsNode<ElemType>>(net.GetDeviceId(), nodeName, dim1, dim2), { matrix });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Times(const ComputationNodePtr a, const ComputationNodePtr b, size_t outputRank, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<TimesNode<ElemType>>(net.GetDeviceId(), nodeName, outputRank), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::TransposeTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<TransposeTimesNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::QuantizedTimes(const ComputationNodePtr a, const ComputationNodePtr b, size_t bitSmoothingA, size_t bitSmoothingB, size_t outputRank, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<QuantizedTimesNode<ElemType>>(net.GetDeviceId(), nodeName, bitSmoothingA, bitSmoothingB, outputRank), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::ElementTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<ElementTimesNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::DiagTimes(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<DiagTimesNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::CosDistance(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<CosDistanceNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::KhatriRaoProduct(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<KhatriRaoProductNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Plus(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<PlusNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::LogPlus(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<LogPlusNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Less(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<LessNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Equal(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<EqualNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Greater(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<GreaterNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::GreaterEqual(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<LessNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::NotEqual(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<EqualNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::LessEqual(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<GreaterNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Minus(const ComputationNodePtr a, const ComputationNodePtr b, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<MinusNode<ElemType>>(net.GetDeviceId(), nodeName), { a, b });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Dropout(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<DropoutNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Reshape(const ComputationNodePtr a,
                                                                                   const TensorShape& imageLayout,
                                                                                   const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<ReshapeNode<ElemType>>(net.GetDeviceId(), nodeName, imageLayout), { a });
}
#if 1
template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::LegacyReshape(const ComputationNodePtr a,
                                                                                         const size_t numRows,
                                                                                         const TensorShape& imageLayout,
                                                                                         const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<LegacyReshapeNode<ElemType>>(net.GetDeviceId(), nodeName, numRows, imageLayout), { a });
}
#endif

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::RowRepeat(const ComputationNodePtr a, const size_t num_repeat, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<RowRepeatNode<ElemType>>(net.GetDeviceId(), nodeName, num_repeat), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::Diagonal(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<DiagonalNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::PastValue(const ComputationNodePtr a, const float initHiddenActivity, const size_t row_size, size_t timeStep, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<PastValueNode<ElemType>>(net.GetDeviceId(), nodeName, initHiddenActivity, row_size, timeStep), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::FutureValue(const ComputationNodePtr a, const float initHiddenActivity, const size_t row_size, size_t timeStep, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<FutureValueNode<ElemType>>(net.GetDeviceId(), nodeName, initHiddenActivity, row_size, timeStep), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::RowSlice(const ComputationNodePtr a, const size_t start_index, const size_t num_rows, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<SliceNode<ElemType>>(net.GetDeviceId(), nodeName, (int)start_index, (int)(start_index + num_rows)), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::RowStack(const std::vector<ComputationNodePtr> pinputs, const std::wstring nodeName)
{
    vector<ComputationNodeBasePtr> inputs(pinputs.size());
    for (size_t i = 0; i < inputs.size(); i++)
        inputs[i] = pinputs[i]; // convert to ComputationNodeBasePtr
    return net.AddNodeToNetAndAttachInputs(New<RowStackNode<ElemType>>(net.GetDeviceId(), nodeName), { inputs });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::RandomSample(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<RandomSampleNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::RandomSampleInclusionFrequency(const ComputationNodePtr a, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<RandomSampleInclusionFrequencyNode<ElemType>>(net.GetDeviceId(), nodeName), { a });
}

#ifdef COMING_SOON
template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::GMMLogLikelihood(const ComputationNodePtr unnormedPrior,
                                                                                            const ComputationNodePtr mean,
                                                                                            const ComputationNodePtr logStddev,
                                                                                            const ComputationNodePtr feature,
                                                                                            const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<GMMLogLikelihoodNode<ElemType>>(net.GetDeviceId(), nodeName), { unnormedPrior, mean, logStddev, feature });
}
#endif

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::LookupTable(const ComputationNodePtr dictionary, const ComputationNodePtr input, const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<LookupTableNode<ElemType>>(net.GetDeviceId(), nodeName), { dictionary, input });
}

template <class ElemType>
shared_ptr<ComputationNode<ElemType>> ComputationNetworkBuilder<ElemType>::BatchNormalization(const ComputationNodePtr input,
                                                                                              const ComputationNodePtr scale, const ComputationNodePtr bias, const ComputationNodePtr runMean, const ComputationNodePtr runVariance,
                                                                                              bool spatial, double normalizationTimeConstant, double blendTimeConstant, double epsilon, bool useCntkEngine,
                                                                                              ImageLayoutKind imageLayoutKind,
                                                                                              const std::wstring nodeName)
{
    return net.AddNodeToNetAndAttachInputs(New<BatchNormalizationNode<ElemType>>(net.GetDeviceId(), nodeName, spatial, normalizationTimeConstant, blendTimeConstant, epsilon, useCntkEngine, imageLayoutKind), { input, scale, bias, runMean, runVariance });
}

template class ComputationNetworkBuilder<float>;
template class ComputationNetworkBuilder<double>;

}}}
