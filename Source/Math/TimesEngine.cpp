//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "TimesEngine.h"
#pragma warning(disable: 4661)  
#include "./mkldnn/mkldnn_fully_connected-inl.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
void TimesEngine<ElemType>::Forward(const Mat& in, const Mat& weight, Mat& out)
{
    EnsureCompatible();
    ForwardCore(in, weight, out);
}

template <class ElemType>
void TimesEngine<ElemType>::BackwardWeight(const Mat& in, const Mat& grad, Mat& gradWeight, bool accumulateGradient, Mat& workspace)
{
    EnsureCompatible();
    BackwardWeightCore(in, grad, gradWeight, accumulateGradient, workspace);
}

template <class ElemType>
void TimesEngine<ElemType>::BackwardData(const Mat& grad, const Mat& weight, Mat& gradData, bool accumulateGradient, Mat& workspace)
{
  EnsureCompatible();
  BackwardDataCore(grad, weight, gradData, accumulateGradient, workspace);
}

template <typename T> bool HasFlag(T src, T testFlag)
{
  return ((int)src & (int)testFlag) != 0;
}

#ifdef USE_MKLDNN
template <class ElemType>
class MklDnnTimesEngine : public TimesEngine<ElemType>
{
public:
  size_t m_prevBatchSize = 0;
  using Base = TimesEngine<ElemType>;
  //using typename Base::Mat;
  using Mat = Matrix<ElemType>;
  MKLDNNFullyConnectedOp<ElemType> * m_fc;
  SmallVector<size_t> m_dimA;
  SmallVector<size_t> m_dimB;
public:
  MklDnnTimesEngine(DEVICEID_TYPE deviceId, SmallVector<size_t>& dimA, SmallVector<size_t>& dimB)
    : Base(deviceId, dimA, dimB), m_fc(NULL)
  {
    m_dimA = dimA;
    m_dimB = dimB;
  }

  ~MklDnnTimesEngine() {
    if (m_fc != NULL)
      delete m_fc;
  }
protected:
  using Base::m_deviceId;

  void EnsureCompatible() override
  {
  }

  void ForwardCore(const Mat& in, const Mat& weight, Mat& out) override
  {
    size_t batchSize = in.GetNumCols();
    if (m_prevBatchSize == 0)
      m_prevBatchSize = batchSize;
    bool samBatchSize = batchSize == m_prevBatchSize;
    if (!samBatchSize && m_fc != NULL) {
      delete m_fc;
      m_fc = NULL;
      m_prevBatchSize = batchSize;
    }
    if (m_fc == NULL) {
      m_fc = new MKLDNNFullyConnectedOp<ElemType>();
      m_fc->validate(m_dimA, m_dimB);
    }
    m_fc->Forward(in, weight, out);
  }

  virtual void BackwardWeightCore(const Mat& in, const Mat& grad, Mat& gradWeight, bool accumulateGradient, Mat& workspace)
  {
    if (accumulateGradient)
        workspace.AssignValuesOf(gradWeight);

    m_fc->BackwardWeight(in, grad, gradWeight);

    if (accumulateGradient)
      gradWeight.AssignSumOf(gradWeight, workspace);
  }

  virtual void BackwardDataCore(const Mat& grad, const Mat& weight, Mat& gradData, bool accumulateGradient, Mat& workspace)
  {
    if (accumulateGradient)
      workspace.AssignValuesOf(gradData);

    m_fc->BackwardData(grad, weight, gradData);

    if (accumulateGradient)
      gradData.AssignSumOf(gradData, workspace);
  }

public:
  static bool IsSupported(DEVICEID_TYPE deviceId) {
    // MKL DNN do not support double
    const std::type_info& ti1 = typeid(ElemType);
    const std::type_info& ti2 = typeid(float);
    if (ti1.hash_code() != ti2.hash_code()) {
      return false;
    }
    return deviceId < 0;
  }
};

template class MklDnnTimesEngine<float>;
template class MklDnnTimesEngine<double>;
#endif

template <class ElemType>
std::unique_ptr<TimesEngine<ElemType>> TimesEngine<ElemType>::Create(DEVICEID_TYPE deviceId,
  SmallVector<size_t>& dimA, SmallVector<size_t>& dimB,
  TimesKind enabledEngines)
{
#ifdef USE_MKLDNN
  if (HasFlag(enabledEngines, TimesKind::MKLDNN) &&
    MklDnnTimesEngine<ElemType>::IsSupported(deviceId))
  {
    if (GetMathLibTraceLevel() > 0)
      fprintf(stderr, "Using CNTK MKL DNN Inner Product engine.\n");

    return std::make_unique<MklDnnTimesEngine<ElemType>>(deviceId, dimA, dimB);
  }
#endif
  if (GetMathLibTraceLevel() > 0)
    fprintf(stderr, "Could not use MKLDNN Inner Product engine.");
  return nullptr;
}
template <>
std::unique_ptr<TimesEngine<half>> TimesEngine<half>::Create(DEVICEID_TYPE deviceId,
    SmallVector<size_t>& dimA, SmallVector<size_t>& dimB,
    TimesKind enabledEngines)
{
    UNUSED(deviceId);
    UNUSED(dimA);
    UNUSED(dimB);
    UNUSED(enabledEngines);
    return nullptr;
}

template class TimesEngine<float>;
template class TimesEngine<double>;
template class TimesEngine<half>;

}}}
