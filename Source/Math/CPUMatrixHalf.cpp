//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include "CPUMatrixImpl.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// General conversion function with no performance optimization
// this should only be used in CPU half precision
// For performance on inference on CPU, user should convert fp16 model to fp32 first, unless MKL supports half precision
template<typename SrcT, typename DstT>
static void ConvertBuffer(DstT* dst, const SrcT* src, size_t count)
{
    for (size_t i = 0; i < count; i++)
    {
        dst[i] = (DstT)src[i];
    }
}

// specialization to convert from half to float for computation, and then store in half
template <>
void CPUMatrix<half>::MultiplyAndWeightedAdd(half alpha, const CPUMatrix<half>& a, const bool transposeA, const CPUMatrix<half>& b, const bool transposeB,
    half beta, CPUMatrix<half>& c, shared_ptr<QuantizedMultiplier<half>> pQuantizedMultiplier)
{
    CPUMatrix<float> af(a.GetNumRows(), a.GetNumCols());
    CPUMatrix<float> bf(b.GetNumRows(), b.GetNumCols());
    CPUMatrix<float> cf(c.GetNumRows(), c.GetNumCols());

    if (alpha != 0)
    {
        ConvertBuffer<half, float>(af.Data(), a.Data(), a.GetNumElements());
        ConvertBuffer<half, float>(bf.Data(), b.Data(), b.GetNumElements());
    }

    if (beta != 0)
    {
        ConvertBuffer<half, float>(cf.Data(), c.Data(), c.GetNumElements());
    }

    if (pQuantizedMultiplier)
        RuntimeError("Quantized matrix multiply not supported for Half");

    CPUMatrix<float>::MultiplyAndWeightedAdd((float)alpha, af, transposeA, bf, transposeB, (float)beta, cf, nullptr);

    ConvertBuffer<float, half>(c.Data(), cf.Data(), c.GetNumElements());
}

// specialization to RunTimeError for now due to omp implementation only support build-in type
template <>
void CPUMatrix<half>::AssignSoftmaxSum(const CPUMatrix<half>& softmax, CPUMatrix<half>& c)
{
    RuntimeError("half AssignSoftmaxSum not supported.");
}

template <>
void CPUMatrix<half>::AssignNCEUnnormalizedEval(const CPUMatrix<half>& a,
                                                const CPUMatrix<half>& b, const CPUMatrix<half>& bias, CPUMatrix<half>& c)
{
    RuntimeError("half AssignNCEUnnormalizedEval not supported.");
}

template <>
void CPUMatrix<half>::VectorSum(const CPUMatrix<half>& a, CPUMatrix<half>& c, const bool isColWise)
{
    RuntimeError("half VectorSum not supported.");
}

template <>
void CPUMatrix<half>::VectorNorm1(CPUMatrix<half>& c, const bool isColWise) const
{
    RuntimeError("half VectorNorm1 not supported.");
}

template <>
half CPUMatrix<half>::SumOfElements() const
{
    RuntimeError("half SumOfElements not supported.");
}

template <>
half CPUMatrix<half>::MatrixNorm1() const
{
    RuntimeError("half MatrixNorm1 not supported.");
}

template <>
    half CPUMatrix<half>::FrobeniusNorm() const
{
    RuntimeError("half FrobeniusNorm not supported.");
}

template <>
void CPUMatrix<half>::MaxPoolingBackward(const CPUMatrix<half>& out, const CPUMatrix<half>& in,
                                         const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices,
                                         CPUMatrix<half>& grad, bool accumulateGradient) const
{
    RuntimeError("half MaxPoolingBackward not supported.");
}

template <>
void CPUMatrix<half>::MaxROIPoolingBackward(const size_t numRois, const size_t numImg, const size_t channels, const size_t width, const size_t height,
                                            const size_t pooledWidth, const size_t pooledHeight, const CPUMatrix<half>& roiData, CPUMatrix<half>& grad,
                                            CPUMatrix<half>& argmax, double spatialScale) const
{
    RuntimeError("half MaxROIPoolingBackward not supported.");
}

template <>
void CPUMatrix<half>::AveragePoolingBackward(const CPUMatrix<int>& mpRowCol, const CPUMatrix<int>& mpRowIndices, const CPUMatrix<int>& indices, CPUMatrix<half>& grad, const bool poolIncludePad, bool accumulateGradient) const
{
    RuntimeError("half AveragePoolingBackward not supported.");
}

// explicit instantiations, due to CPUMatrix being too big and causing VS2015 cl crash.
template class MATH_API CPUMatrix<half>;

// instantiate templated methods
template void CPUMatrix<float>::AdaDelta(CPUMatrix<float>& gradients, CPUMatrix<float>& functionValues, float learningRate, float rho, float epsilon);
template void CPUMatrix<double>::AdaDelta(CPUMatrix<double>& gradients, CPUMatrix<double>& functionValues, double learningRate, double rho, double epsilon);
template void CPUMatrix<float>::AdaDelta(CPUMatrix<half>& gradients, CPUMatrix<float>& functionValues, float learningRate, float rho, float epsilon);

template void CPUMatrix<float>::BatchNormalizationForward(const CPUMatrix<float>& scale, const CPUMatrix<float>& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, CPUMatrix<float>& runMean, CPUMatrix<float>& runVariance, CPUMatrix<float>& out, double epsilon, CPUMatrix<float>& saveMean, CPUMatrix<float>& saveInvStdDev) const;
template void CPUMatrix<double>::BatchNormalizationForward(const CPUMatrix<double>& scale, const CPUMatrix<double>& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, CPUMatrix<double>& runMean, CPUMatrix<double>& runVariance, CPUMatrix<double>& out, double epsilon, CPUMatrix<double>& saveMean, CPUMatrix<double>& saveInvStdDev) const;
template void CPUMatrix<half>::BatchNormalizationForward(const CPUMatrix<float>& scale, const CPUMatrix<float>& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, CPUMatrix<float>& runMean, CPUMatrix<float>& runVariance, CPUMatrix<half>& out, double epsilon, CPUMatrix<float>& saveMean, CPUMatrix<float>& saveInvStdDev) const;

template void CPUMatrix<float>::BatchNormalizationBackward(const CPUMatrix<float>& in, CPUMatrix<float>& grad, const CPUMatrix<float>& scale, double blendFactor, const CPUMatrix<float>& saveMean, const CPUMatrix<float>& saveInvStdDev, CPUMatrix<float>& scaleGrad, CPUMatrix<float>& biasGrad) const;
template void CPUMatrix<double>::BatchNormalizationBackward(const CPUMatrix<double>& in, CPUMatrix<double>& grad, const CPUMatrix<double>& scale, double blendFactor, const CPUMatrix<double>& saveMean, const CPUMatrix<double>& saveInvStdDev, CPUMatrix<double>& scaleGrad, CPUMatrix<double>& biasGrad) const;
template void CPUMatrix<half>::BatchNormalizationBackward(const CPUMatrix<half>& in, CPUMatrix<half>& grad, const CPUMatrix<float>& scale, double blendFactor, const CPUMatrix<float>& saveMean, const CPUMatrix<float>& saveInvStdDev, CPUMatrix<float>& scaleGrad, CPUMatrix<float>& biasGrad) const;

}}}
