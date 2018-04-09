//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Basics.h"
#include "Globals.h"
#include "Matrix.h"
#include "ComputationNode.h"
#include "NonlinearityNodes.h"
#include "EltWiseEngine.h"
namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
class RectifiedLinearNodeV2 : public ComputationNode<ElemType>, public NumInputs<1>
{
protected:
  virtual void /*ComputationNode::*/ ForwardProp(const FrameRange& fr) override
  {
    size_t rank = DetermineElementwiseTensorRank();
    auto result = ValueTensorFor(rank, fr);
    auto input = InputRef(0).ValueTensorFor(rank, fr);
    if (m_reluEng != nullptr) {
      bool inferenceOnly = !Environment().IsTraining();
      m_reluEng->Forward(input.GetSOB(), result.GetSOB(), inferenceOnly);
    } else {
      result.DoUnaryOpOf(0, input, 1, opLinearRectifier, opSum);
    }
  }

  virtual void /*ComputationNode::*/ BackpropTo(const size_t inputIndex, const FrameRange& fr) override
  {
    assert(inputIndex == 0), inputIndex;
    // get the args
    size_t rank = DetermineElementwiseTensorRank();
    auto sliceOutputGrad = GradientTensorFor(rank, fr); // propagate from this one...
    auto sliceInputGrad = InputRef(0).GradientTensorFor(rank, fr); // ...to this one

    // If gradient can be compute from output rather than input, then that's better for mem sharing (and faster in most cases).
    // Not possible for Cos().
    auto sliceValue = ValueTensorFor(rank, fr);// using input or output value
    if (m_reluEng != nullptr) {
      m_reluEng->Backward(sliceValue.GetSOB(), sliceOutputGrad.GetSOB(), sliceInputGrad.GetSOB());
    } else {
      sliceInputGrad.DoBinaryOpOf(Input(inputIndex)->IsGradientInitializedBy(this) ? 0.0f : 1.0f, sliceOutputGrad, sliceValue, 1, opElementwiseProductWithLinearRectifierDerivativeFromOutput, opSum);
    }
  }
  virtual void /*ComputationNodeBase::*/ Validate(bool isFinalValidationPass) override
  {
    ValidateUnaryMap(isFinalValidationPass);
    auto shape = GetSampleLayout();
    if (m_reluEng == nullptr) {
        if (GetMathLibTraceLevel() > 0)
          fprintf(stderr, "use Cntk Rectified Linear engine.");
      m_reluEng = UnaryEltWiseEngine<ElemType>::Create(m_deviceId, shape,
        ImageLayoutKind::CHW,  //Todo : need to double check
        UnaryEltWiseKind::RELU, EltWiseEngineKind::All);
    }
  }
  virtual bool OutputUsedInComputingInputNodesGradients() const override
  {
    return true;
  }

  virtual bool InputUsedInComputingInputNodesGradients(size_t /*childIndex*/) const override
  {
    return false;
  }
  virtual ParentGradientOptimization ImplementsGradientOptimization(const ComputationNodeBase*) const override { return (binaryWithOutputGradient != noGradient) ? ParentGradientOptimization::Overwrite : ParentGradientOptimization::None; }

  virtual const std::wstring OperationName() const override
  {
    return   TypeName();
  }
  virtual ComputationNodeBase* NewThis(int deviceId, const wstring& name) const override
  {
    const ComputationNodeBase* p = new typename std::remove_reference<decltype(*this)>::type(deviceId, name);
    return const_cast<ComputationNodeBase*>(p);
  };
protected:
  typedef ComputationNode<ElemType> Base; UsingComputationNodeMembers;

  static const std::wstring TypeName()
  {
    return L"RectifiedLinear";
  }
public:
  RectifiedLinearNodeV2(const Microsoft::MSR::ScriptableObjects::IConfigRecordPtr configp) 
    : RectifiedLinearNodeV2(configp->Get(L"deviceId"), L"<placeholder>")
  {
    AttachInputsFromConfig(configp, this->GetExpectedNumInputs());
  };
  RectifiedLinearNodeV2(int deviceId, const wstring& RectifiedLinear) : Base(deviceId, RectifiedLinear)
    , m_reluEng(nullptr)
  {
  }
private:
  std::unique_ptr<UnaryEltWiseEngine<ElemType>> m_reluEng;
};


} } }
