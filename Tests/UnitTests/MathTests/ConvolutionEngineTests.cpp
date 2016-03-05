//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include <algorithm>
#include <array>
#include <random>
#include <numeric>
#include "../../../Source/Math/Matrix.h"
#include "../../../Source/Math/CPUMatrix.h"
#include "../../../Source/Math/GPUMatrix.h"
#include "../../../Source/Math/ConvolutionEngine.h"
#include "../../../Source/Math/CuDnnConvolutionEngine.h"

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

//using ConvFact = ConvolutionEngineFactory<float>;
//using ConvFactPtr = std::unique_ptr<ConvolutionEngineFactory<float>>;
//using ConvFactSPtr = std::shared_ptr<ConvolutionEngineFactory<float>>;
//using vec = std::vector<float>;
//using Tensor4DPtr = ConvFact::Tensor4DPtr;

using ConvEng = ConvolutionEngine<float>;

template <typename T>
struct Err
{
    static const T Rel;
    static const T Abs;
};
template <>
const float Err<float>::Rel = 1e-5f;
template <>
const double Err<double>::Rel = 1e-5f;
template <>
const float Err<float>::Abs = 1.192092896e-07f;
template <>
const double Err<double>::Abs = 2.2204460492503131e-016;

//static bool AreEqual(float a, float b, float maxRelError, float maxAbsError)
//{
//    float diff = std::abs(a - b);
//    if (diff <= maxAbsError)
//        return true;
//    float largest = std::max(std::abs(a), std::abs(b));
//    return diff < largest * maxRelError;
//}
//static bool AreEqual(double a, double b, double maxRelError, double maxAbsError)
//{
//    double diff = std::abs(a - b);
//    if (diff <= maxAbsError)
//        return true;
//    double largest = std::max(std::abs(a), std::abs(b));
//    return diff < largest * maxRelError;
//}

template <typename T>
static bool CheckEqual(const Matrix<T>& result, const Matrix<T>& reference, std::string& msg, T maxRelError, T maxAbsError)
{
    std::unique_ptr<T[]> res(result.CopyToArray());
    std::unique_ptr<T[]> ref(reference.CopyToArray());
    int count = 0;
    int badIndex = -1;
    for (int i = 0; i < result.GetNumElements(); ++i)
    {
        if (!AreEqual(res[i], ref[i], maxRelError, maxAbsError) && count++ == 0)
            badIndex = i;
    }
    if (count > 0)
    {
        float a = res[badIndex];
        float b = ref[badIndex];
        std::stringstream ss;
        ss << count << " mismatch" << (count > 1 ? "es" : "") << ", first mismatch at " << badIndex << ", " << a << " != " << b
            << ", rel = " << (std::abs(a - b) / std::max(std::abs(a), std::abs(b))) << ", abs = " << std::abs(a - b);
        msg = ss.str();
    }
    return count == 0;
}

std::vector<ConvolveGeometryPtr> GenerateConvTestConfigs()
{
    std::vector<ConvolveGeometryPtr> res;
    //for (size_t kW : {1, 2, 3})
    //{
    //    for (size_t kH : {1, 2, 3})
    //    {
    //        for (size_t inW : {kW, 2 * kW, 2 * kW - 1})
    //        {
    //            for (size_t inC : {1, 3})
    //            {
    //                for (size_t mapCount : {1, 5})
    //                {
    //                    for (size_t stride : {1, min((int)kW, min((int)kH, 2))})
    //                    {
    //                        res.push_back(std::make_shared<ConvolveGeometry>(TensorShape(inW, max(kH, inW) + 1, inC),
    //                            TensorShape(kW, kH, inC), TensorShape(mapCount), TensorShape(stride, stride, 1),
    //                            ConvolveGeometry::BoolVec(1, true), ConvolveGeometry::BoolVec(1, true),
    //                            TensorShape(0), TensorShape(0)));
    //                    }
    //                }
    //            }
    //        }
    //    }
    //}
    res.push_back(std::make_shared<ConvolveGeometry>(TensorShape(5, 5, 1),
        TensorShape(3, 3, 1), TensorShape(3), TensorShape(1, 1, 1),
        ConvolveGeometry::BoolVec(1, true), ConvolveGeometry::BoolVec(1, true),
        TensorShape(0), TensorShape(0)));
    return res;
}

BOOST_AUTO_TEST_SUITE(ConvolutionSuite)

BOOST_AUTO_TEST_CASE(ConvolutionForward)
{
    int baseDeviceId = 0;
    for (auto engKind : {ConvolutionEngineKind::Default})
    {
        for (int deviceId : {0})
        {
            for (const auto& g : GenerateConvTestConfigs())
            {
                auto baseEng = ConvEng::Create(g, baseDeviceId, ImageLayoutKind::CHW, 0, ConvolutionEngineKind::CuDnn);
                auto testEng = ConvEng::Create(g, deviceId, ImageLayoutKind::CHW, 0, engKind);
            }
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()

//static int GetNumOut(int i, int k, int s, bool pad)
//{
//    return (i - (pad ? 1 : k)) / s + 1;
//}
//
//static bool IsCuDnnSupported()
//{
//    fprintf(stderr, "ConvolutionEngineTests.cpp %d\n", __LINE__);
//    try
//    {
//        // TODO: Will this ever return nullptr?
//        return ConvFact::Create(0, ConvFact::EngineType::CuDnn, ImageLayoutKind::CHW) != nullptr;
//    }
//    catch (std::runtime_error)
//    {
//        fprintf(stderr, "ConvolutionEngineTests.cpp %d\n", __LINE__);
//        return false;
//    }
//    fprintf(stderr, "ConvolutionEngineTests.cpp %d\n", __LINE__);
//}
//
//BOOST_AUTO_TEST_SUITE(ConvolutionSuite)
//
//BOOST_AUTO_TEST_CASE(ConvolutionForward)
//{
//    if (!IsCuDnnSupported())
//        return;
//
//    int n = 2;
//    int cmapIn = 3;
//    int inW = 5;
//    int inH = 5;
//    int kW = 3;
//    int kH = 3;
//    int sW = 2;
//    int sH = 2;
//    int cmapOut = 2;
//    int outW = GetNumOut(inW, kW, sW, false);
//    int outH = GetNumOut(inH, kH, sH, false);
//
//    for (int deviceId : {0})
//    {
//        // BUGBUG: These will fail depending on whether we built with cuDNN or not. Without cuDNN we should use HWC
//        auto fact = ConvFact::Create(deviceId, ConvFact::EngineType::Auto, ImageLayoutKind::CHW);
//        auto tt = typeid(fact).name();
//        UNUSED(tt);
//        auto eng = fact->CreateConvEngine(deviceId, ImageLayoutKind::CHW, 0, BatchNormImpl::Cntk);
//        auto inT = fact->CreateTensor(inW, inH, cmapIn, n);
//        auto filtT = fact->CreateFilter(kW, kH, cmapIn, cmapOut);
//        auto outT = fact->CreateTensor(outW, outH, cmapOut, n);
//        auto convT = fact->CreateConvDescriptor(*inT, *filtT, sW, sH, false);
//        auto biasT = fact->CreateTensor(1, 1, cmapOut, 1);
//
//        vec buf(inW * inH * cmapIn * n);
//        int seed = 0;
//        // Create input, cmapIn feature maps, inW x inH each (NCHW format).
//        std::generate(buf.begin(), buf.end(), [=, &seed]
//                      {
//                          return seed++ % (inW * inH * cmapIn);
//                      });
//        SingleMatrix in(inW * inH * cmapIn, n, buf.data(), deviceId, matrixFlagNormal);
//
//        seed = 0;
//        buf.resize(kW * kH * cmapIn * cmapOut);
//        // Create cmapOut filters, each kW x kH x cmapIn (NCHW format).
//        std::generate(buf.begin(), buf.end(), [=, &seed]
//                      {
//                          return seed++ % (kW * kH * cmapIn);
//                      });
//        SingleMatrix filt(cmapOut, kW * kH * cmapIn, buf.data(), deviceId, matrixFlagNormal);
//
//        SingleMatrix out(outW * outH * cmapOut, n, deviceId);
//        SingleMatrix temp(deviceId);
//
//        eng->Forward(*inT, in, *filtT, filt, *convT, *outT, out, temp);
//
//        // Output is in NCHW format.
//        std::array<float, 4 * 2 * 2> expBuf = {
//            15219.0f, 15921.0f, 18729.0f, 19431.0f,
//            15219.0f, 15921.0f, 18729.0f, 19431.0f,
//            15219.0f, 15921.0f, 18729.0f, 19431.0f,
//            15219.0f, 15921.0f, 18729.0f, 19431.0f};
//        SingleMatrix exp(outW * outH * cmapOut, n, expBuf.data(), deviceId, matrixFlagNormal);
//        BOOST_CHECK_MESSAGE(out.IsEqualTo(exp), "Unexpected convolution output.");
//    }
//}
//
//// REVIEW alexeyk: this really should be rolled into ConvolutionForward, make it data-driven.
//BOOST_AUTO_TEST_CASE(ConvolutionForwardPad)
//{
//    if (!IsCuDnnSupported())
//        return;
//
//    int n = 2;
//    int cmapIn = 1;
//    int inW = 4;
//    int inH = 4;
//    int kW = 3;
//    int kH = 3;
//    int sW = 2;
//    int sH = 2;
//    int cmapOut = 1;
//    bool pad = true;
//    int outW = GetNumOut(inW, kW, sW, pad);
//    int outH = GetNumOut(inH, kH, sH, pad);
//
//    for (int deviceId : {0})
//    {
//        auto fact = ConvFact::Create(deviceId, ConvFact::EngineType::Auto, deviceId >= 0 ? ImageLayoutKind::CHW : ImageLayoutKind::HWC);
//        auto eng = fact->CreateConvEngine(deviceId, ImageLayoutKind::CHW, 0, BatchNormImpl::Cntk);
//        auto inT = fact->CreateTensor(inW, inH, cmapIn, n);
//        auto filtT = fact->CreateFilter(kW, kH, cmapIn, cmapOut);
//        auto outT = fact->CreateTensor(outW, outH, cmapOut, n);
//        auto convT = fact->CreateConvDescriptor(*inT, *filtT, sW, sH, pad);
//
//        // Input in NCHW format.
//        SingleMatrix in(inW * inH * cmapIn, n, vec(inW * inH * cmapIn * n, 1.0f).data(), deviceId, matrixFlagNormal);
//        // Create cmapOut filters, each kW x kH x cmapIn (NCHW format).
//        SingleMatrix filt(cmapOut, kW * kH * cmapIn, vec(kW * kH * cmapIn * cmapOut, 1.0f).data(), deviceId, matrixFlagNormal);
//
//        SingleMatrix out(outW * outH * cmapOut, n, deviceId);
//        SingleMatrix temp(deviceId);
//
//        eng->Forward(*inT, in, *filtT, filt, *convT, *outT, out, temp);
//
//        // Output is in NCHW format.
//        float expBuf[] = {
//            4.0f, 6.0f, 6.0f, 9.0f,
//            4.0f, 6.0f, 6.0f, 9.0f,
//        };
//        SingleMatrix exp(outW * outH * cmapOut, n, expBuf, deviceId, matrixFlagNormal);
//        BOOST_CHECK(out.IsEqualTo(exp));
//    }
//}
//
//BOOST_AUTO_TEST_CASE(ConvolutionBackwardData)
//{
//    if (!IsCuDnnSupported())
//        return;
//
//    // REVIEW alexeyk: very simple test, improve.
//    int n = 2;
//    int cmapIn = 3;
//    int inW = 3;
//    int inH = 3;
//    int kW = 3;
//    int kH = 3;
//    int sW = 1;
//    int sH = 1;
//    int cmapOut = 2;
//    int outW = GetNumOut(inW, kW, sW, false);
//    int outH = GetNumOut(inH, kH, sH, false);
//
//    for (int deviceId : {0})
//    {
//        auto fact = ConvFact::Create(deviceId, ConvFact::EngineType::Auto, ImageLayoutKind::CHW);
//        auto eng = fact->CreateConvEngine(deviceId, ImageLayoutKind::CHW, 0, BatchNormImpl::Cntk);
//        auto srcGradT = fact->CreateTensor(outW, outH, cmapOut, n);
//        auto filtT = fact->CreateFilter(kW, kH, cmapIn, cmapOut);
//        auto gradT = fact->CreateTensor(inW, inH, cmapIn, n);
//        auto convT = fact->CreateConvDescriptor(*gradT, *filtT, sW, sH, false);
//
//        // Source grads is in NCHW format.
//        float srcGradBuf[] = {
//            1.0f, 1.0f,
//            1.0f, 1.0f};
//        SingleMatrix srcGrad(outW * outH * cmapOut, n, srcGradBuf, deviceId, matrixFlagNormal);
//
//        vec filtB(kW * kH * cmapIn * cmapOut);
//        // Create cmapOut filters, each kW x kH x cmapIn (NCHW format).
//        int seed = 0;
//        std::generate(filtB.begin(), filtB.end(), [=, &seed]
//                      {
//                          return seed++ % (kW * kH * cmapIn);
//                      });
//        SingleMatrix filt(cmapOut, kW * kH * cmapIn, filtB.data(), deviceId, matrixFlagNormal);
//
//        SingleMatrix grad(inW * inH * cmapIn, n, deviceId);
//        grad.SetValue(1);
//        SingleMatrix temp(deviceId);
//
//        eng->BackwardData(*srcGradT, srcGrad, *filtT, filt, *convT, *gradT, grad, temp);
//
//        // Target grads is in NCHW format.
//        vec gradB(inW * inH * cmapIn * n);
//        seed = 0;
//        std::generate(gradB.begin(), gradB.end(), [=, &seed]
//                      {
//                          return 2 * (seed++ % (kW * kH * cmapIn)) + 1;
//                      });
//
//        SingleMatrix exp(inW * inH * cmapIn, n, gradB.data(), deviceId, matrixFlagNormal);
//        BOOST_CHECK(grad.IsEqualTo(exp));
//    }
//}
//
//BOOST_AUTO_TEST_CASE(ConvolutionBackwardFilter)
//{
//    if (!IsCuDnnSupported())
//        return;
//
//    // REVIEW alexeyk: very simple test, improve.
//    int n = 2;
//    int cmapIn = 3;
//    int inW = 3;
//    int inH = 3;
//    int kW = 3;
//    int kH = 3;
//    int sW = 1;
//    int sH = 1;
//    int cmapOut = 2;
//    int outW = GetNumOut(inW, kW, sW, false);
//    int outH = GetNumOut(inH, kH, sH, false);
//
//    for (int deviceId : {0})
//    {
//        auto fact = ConvFact::Create(deviceId, ConvFact::EngineType::Auto, ImageLayoutKind::CHW);
//        auto eng = fact->CreateConvEngine(deviceId, ImageLayoutKind::CHW, 0, BatchNormImpl::Cntk);
//        auto srcGradT = fact->CreateTensor(outW, outH, cmapOut, n);
//        auto filtT = fact->CreateFilter(kW, kH, cmapIn, cmapOut);
//        auto inT = fact->CreateTensor(inW, inH, cmapIn, n);
//        auto convT = fact->CreateConvDescriptor(*inT, *filtT, sW, sH, false);
//        auto biasT = fact->CreateTensor(1, 1, cmapOut, 1);
//
//        // Source grads is in NCHW format.
//        float srcGradBuf[] = {
//            1.0f, 1.0f,
//            1.0f, 1.0f,
//        };
//        SingleMatrix srcGrad(outW * outH * cmapOut, n, srcGradBuf, deviceId, matrixFlagNormal);
//
//        vec buf(inW * inH * cmapIn * n);
//        int seed = 0;
//        // Create input, cmapIn feature maps, inW x inH each, NCHW format.
//        std::generate(buf.begin(), buf.end(), [=, &seed]
//                      {
//                          return seed++ % (inW * inH * cmapIn);
//                      });
//        SingleMatrix in(inW * inH * cmapIn, n, buf.data(), deviceId, matrixFlagNormal);
//
//        SingleMatrix filt(cmapOut, kW * kH * cmapIn, deviceId);
//        filt.SetValue(1);
//        SingleMatrix temp(deviceId);
//
//        eng->BackwardFilter(*srcGradT, srcGrad, *inT, in, *convT, *filtT, filt, false, temp);
//
//        // Expected filter values in NCHW format.
//        vec expFiltB(cmapOut * kW * kH * cmapIn);
//        seed = 0;
//        std::generate(expFiltB.begin(), expFiltB.end(), [=, &seed]
//                      {
//                          return 2 * (seed++ % (kW * kH * cmapIn)) + 1;
//                      });
//        SingleMatrix exp(cmapOut, kW * kH * cmapIn, expFiltB.data(), deviceId, matrixFlagNormal);
//        BOOST_CHECK_MESSAGE(filt.IsEqualTo(exp), "Unexpected convolution gradient.");
//    }
//}
//
//BOOST_AUTO_TEST_CASE(MaxPoolForward)
//{
//    if (!IsCuDnnSupported())
//        return;
//
//    int n = 2;
//    int cmap = 2;
//    int inW = 4;
//    int inH = 4;
//    int kW = 2;
//    int kH = 2;
//    int sW = 2;
//    int sH = 2;
//    int outW = GetNumOut(inW, kW, sW, false);
//    int outH = GetNumOut(inH, kH, sH, false);
//
//    for (int deviceId : {0})
//    {
//        auto fact = ConvFact::Create(deviceId, ConvFact::EngineType::Auto, ImageLayoutKind::CHW);
//        auto eng = fact->CreatePoolEngine(deviceId, ImageLayoutKind::CHW);
//        auto inT = fact->CreateTensor(inW, inH, cmap, n);
//        auto outT = fact->CreateTensor(outW, outH, cmap, n);
//        auto poolT = fact->CreatePoolDescriptor(PoolingDescriptor::PoolKind::Max, kW, kH, sW, sH, 0, 0);
//
//        vec buf(inW * inH * cmap * n);
//        int seed = 0;
//        // Create input, cmapIn feature maps, inW x inH each (NCHW format).
//        std::generate(buf.begin(), buf.end(), [=, &seed]
//                      {
//                          return seed++ % (inW * inH * cmap);
//                      });
//        SingleMatrix in(inW * inH * cmap, n, buf.data(), deviceId, matrixFlagNormal);
//
//        SingleMatrix out(outW * outH * cmap, n, deviceId);
//
//        eng->Forward(*inT, in, *poolT, *outT, out);
//
//        // Output is in NCHW format.
//        float expBuf[] = {
//            5.0f, 7.0f,
//            13.0f, 15.0f,
//            21.0f, 23.0f,
//            29.0f, 31.0f,
//            5.0f, 7.0f,
//            13.0f, 15.0f,
//            21.0f, 23.0f,
//            29.0f, 31.0f,
//        };
//        SingleMatrix exp(outW * outH * cmap, n, expBuf, deviceId, matrixFlagNormal);
//        BOOST_CHECK(out.IsEqualTo(exp));
//    }
//}
//
//BOOST_AUTO_TEST_CASE(MaxPoolBackward)
//{
//    if (!IsCuDnnSupported())
//        return;
//
//    int n = 2;
//    int cmap = 2;
//    int inW = 4;
//    int inH = 4;
//    int kW = 2;
//    int kH = 2;
//    int sW = 2;
//    int sH = 2;
//    int outW = GetNumOut(inW, kW, sW, false);
//    int outH = GetNumOut(inH, kH, sH, false);
//
//    for (int deviceId : {0})
//    {
//        auto fact = ConvFact::Create(deviceId, ConvFact::EngineType::Auto, ImageLayoutKind::CHW);
//        auto eng = fact->CreatePoolEngine(deviceId, ImageLayoutKind::CHW);
//        auto inT = fact->CreateTensor(inW, inH, cmap, n);
//        auto outT = fact->CreateTensor(outW, outH, cmap, n);
//        auto poolT = fact->CreatePoolDescriptor(PoolingDescriptor::PoolKind::Max, kW, kH, sW, sH, 0, 0);
//
//        vec buf(inW * inH * cmap * n);
//        int seed = 0;
//        // Create input, cmapIn feature maps, inW x inH each (NCHW format).
//        std::generate(buf.begin(), buf.end(), [=, &seed]
//                      {
//                          return seed++ % (inW * inH * cmap);
//                      });
//        SingleMatrix in(inW * inH * cmap, n, buf.data(), deviceId, matrixFlagNormal);
//        SingleMatrix out(outW * outH * cmap, n, deviceId);
//        // Do forward pass first.
//        eng->Forward(*inT, in, *poolT, *outT, out);
//
//        // For gradients, use the same values as outputs.
//        SingleMatrix srcGrad(out);
//        SingleMatrix grad(inW * inH * cmap, n, deviceId);
//        grad.SetValue(1);
//
//        eng->Backward(*outT, out, srcGrad, *poolT, *inT, in, grad);
//
//        // Output is in NCHW format.
//        std::fill(buf.begin(), buf.end(), 1.0f);
//        vec expMap = {
//            5.0f, 7.0f,
//            13.0f, 15.0f,
//            21.0f, 23.0f,
//            29.0f, 31.0f,
//            5.0f, 7.0f,
//            13.0f, 15.0f,
//            21.0f, 23.0f,
//            29.0f, 31.0f,
//        };
//        for (size_t i = 0; i < expMap.size(); i++)
//            buf[(int) expMap[i] + inW * inH * cmap * (i / (expMap.size() / n))] += expMap[i];
//        SingleMatrix exp(inW * inH * cmap, n, buf.data(), deviceId, matrixFlagNormal);
//
//        BOOST_CHECK(grad.IsEqualTo(exp));
//    }
//}
//
//BOOST_AUTO_TEST_CASE(AvgPoolForward)
//{
//    if (!IsCuDnnSupported())
//        return;
//
//    int n = 2;
//    int cmap = 2;
//    int inW = 4;
//    int inH = 4;
//    int kW = 2;
//    int kH = 2;
//    int sW = 2;
//    int sH = 2;
//    int outW = GetNumOut(inW, kW, sW, false);
//    int outH = GetNumOut(inH, kH, sH, false);
//
//    for (int deviceId : {0})
//    {
//        auto fact = ConvFact::Create(deviceId, ConvFact::EngineType::Auto, ImageLayoutKind::CHW);
//        auto eng = fact->CreatePoolEngine(deviceId, ImageLayoutKind::CHW);
//        auto inT = fact->CreateTensor(inW, inH, cmap, n);
//        auto outT = fact->CreateTensor(outW, outH, cmap, n);
//        auto poolT = fact->CreatePoolDescriptor(PoolingDescriptor::PoolKind::Average, kW, kH, sW, sH, 0, 0);
//
//        vec buf(inW * inH * cmap * n);
//        int seed = 0;
//        // Create input, cmapIn feature maps, inW x inH each (NCHW format).
//        std::generate(buf.begin(), buf.end(), [=, &seed]
//                      {
//                          return seed++ % (inW * inH * cmap);
//                      });
//        SingleMatrix in(inW * inH * cmap, n, buf.data(), deviceId, matrixFlagNormal);
//
//        SingleMatrix out(outW * outH * cmap, n, deviceId);
//
//        eng->Forward(*inT, in, *poolT, *outT, out);
//
//        // Output is in NCHW format.
//        float expBuf[] = {
//            2.5f, 4.5f,
//            10.5f, 12.5f,
//            18.5f, 20.5f,
//            26.5f, 28.5f,
//            2.5f, 4.5f,
//            10.5f, 12.5f,
//            18.5f, 20.5f,
//            26.5f, 28.5f,
//        };
//        SingleMatrix exp(outW * outH * cmap, n, expBuf, deviceId, matrixFlagNormal);
//        BOOST_CHECK(out.IsEqualTo(exp));
//    }
//}
//
//BOOST_AUTO_TEST_CASE(AvgPoolBackward)
//{
//    if (!IsCuDnnSupported())
//        return;
//
//    int n = 1;
//    int cmap = 1;
//    int inW = 4;
//    int inH = 4;
//    int kW = 2;
//    int kH = 2;
//    int sW = 2;
//    int sH = 2;
//    int outW = GetNumOut(inW, kW, sW, false);
//    int outH = GetNumOut(inH, kH, sH, false);
//
//    for (int deviceId : {0})
//    {
//        auto fact = ConvFact::Create(deviceId, ConvFact::EngineType::Auto, ImageLayoutKind::CHW);
//        auto eng = fact->CreatePoolEngine(deviceId, ImageLayoutKind::CHW);
//        auto inT = fact->CreateTensor(inW, inH, cmap, n);
//        auto outT = fact->CreateTensor(outW, outH, cmap, n);
//        auto poolT = fact->CreatePoolDescriptor(PoolingDescriptor::PoolKind::Average, kW, kH, sW, sH, 0, 0);
//
//        vec buf(inW * inH * cmap * n);
//        int seed = 0;
//        // Create input, cmapIn feature maps, inW x inH each (NHWC format).
//        std::generate(buf.begin(), buf.end(), [=, &seed]
//                      {
//                          return seed++ % (inW * inH * cmap);
//                      });
//        SingleMatrix in(inW * inH * cmap, n, buf.data(), deviceId, matrixFlagNormal);
//        SingleMatrix out(outW * outH * cmap, n, deviceId);
//        // Do forward pass first.
//        eng->Forward(*inT, in, *poolT, *outT, out);
//
//        // For gradients, use the same values as outputs.
//        SingleMatrix srcGrad(out);
//        SingleMatrix grad(inW * inH * cmap, n, deviceId);
//        grad.SetValue(1);
//
//        eng->Backward(*outT, out, srcGrad, *poolT, *inT, in, grad);
//
//        // Output is in NHWC format.
//        float expBuf[] = {
//            1.625f, 1.625f, 2.125f, 2.125f,
//            1.625f, 1.625f, 2.125f, 2.125f,
//            3.625f, 3.625f, 4.125f, 4.125f,
//            3.625f, 3.625f, 4.125f, 4.125f,
//        };
//        SingleMatrix exp(inW * inH * cmap, n, expBuf, deviceId, matrixFlagNormal);
//        BOOST_CHECK(grad.IsEqualTo(exp));
//    }
//}
//
//BOOST_AUTO_TEST_SUITE_END()
//
//// Batch normalization unit tests.
//// REVIEW alexeyk: is this a right place?
//
//std::vector<std::tuple<Tensor4DPtr, bool, double>> GenerateBNTestConfigs(ConvFact& fact)
//{
//    std::vector<std::tuple<Tensor4DPtr, bool, double>> res;
//    // REVIEW alexeyk: how to test batches > 512? cuDNN does not support that so there is no baseline.
//    double expAvgFactor = 1;
//    // Per activation (non-spatial)
//    for (size_t n : {6, 13, 62, 512})
//    {
//        for (size_t c : {1})
//        {
//            for (size_t h : {1})
//            {
//                for (size_t w : {6, 17, 126, 2048})
//                {
//                    res.push_back(std::make_tuple(std::move(fact.CreateTensor(w, h, c, n)), false, expAvgFactor));
//                }
//            }
//        }
//    }
//    // Spatial
//    for (size_t n : {2, 11, 64})
//    {
//        for (size_t c : {2, 13, 32})
//        {
//            for (size_t h : {2, 11, 16})
//            {
//                for (size_t w : {2, 11, 16})
//                {
//                    res.push_back(std::make_tuple(std::move(fact.CreateTensor(w, h, c, n)), true, expAvgFactor));
//                }
//            }
//        }
//    }
//    // For perf testing (similar to first layers of ResNet).
//    res.push_back(std::make_tuple(std::move(fact.CreateTensor(56, 56, 64, 64)), true, expAvgFactor));
//    // Next test will fail in cuDNN due to bug we discovered (and reported to NVIDIA).
//    //res.push_back(std::make_tuple(std::move(fact.CreateTensor(2, 2, 2048, 2)), true));
//    res.push_back(std::make_tuple(std::move(fact.CreateTensor(2, 2, 2048, 64)), true, expAvgFactor));
//
//    // Test running mean/isd.
//    expAvgFactor = 0.1;
//    res.push_back(std::make_tuple(std::move(fact.CreateTensor(2, 2, 2, 8)), false, expAvgFactor));
//    res.push_back(std::make_tuple(std::move(fact.CreateTensor(2, 2, 2, 8)), true, expAvgFactor));
//
//    return res;
//}
//
//size_t CountNans(const SingleMatrix& src)
//{
//    size_t n = 0;
//    foreach_coord (i, j, src)
//    {
//        n += std::isnan(src(i, j)) ? 1 : 0;
//    }
//    return n;
//}
//
//BOOST_AUTO_TEST_SUITE(BatchNormalizationSuite)
//
//BOOST_AUTO_TEST_CASE(BatchNormalizationForwardTrain)
//{
//    if (!IsCuDnnSupported())
//        return;
//
//    std::mt19937 rng(0);
//    std::normal_distribution<float> nd;
//
//    auto initMat = [&](SingleMatrix& buf, size_t r, size_t c, vec& data) -> SingleMatrix
//    {
//        data.resize(r * 3 * c);
//        std::fill(begin(data), end(data), std::numeric_limits<float>::quiet_NaN());
//        std::generate(begin(data) + r * c, begin(data) + 2 * r * c, [&] { return nd(rng); });
//        buf.SetValue(r, 3 * c, buf.GetDeviceId(), data.data());
//        // Get center slice.
//        return buf.ColumnSlice(c, c);
//    };
//
//    for (int deviceId : {0})
//    {
//        auto fact = ConvFact::Create(deviceId, ConvFact::EngineType::Auto, ImageLayoutKind::CHW);
//        auto engCudnn = fact->CreateConvEngine(deviceId, ImageLayoutKind::CHW, 0, BatchNormImpl::CuDnn);
//        auto engCntk = fact->CreateConvEngine(deviceId, ImageLayoutKind::CHW, 0, BatchNormImpl::Cntk);
//        for (auto& cfg : GenerateBNTestConfigs(*fact))
//        {
//            auto& t = *std::move(std::get<0>(cfg));
//            bool spatial = std::get<1>(cfg);
//            double expAvg = std::get<2>(cfg);
//            double eps = 1e-5; // CUDNN_BN_MIN_EPSILON
//
//            size_t crow = t.w() * t.h() * t.c();
//            size_t ccol = t.n();
//
//            vec buf(crow * t.n());
//            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//            SingleMatrix in(crow, ccol, buf.data(), deviceId, matrixFlagNormal);
//
//            Tensor4DPtr scaleBiasT = spatial ? fact->CreateTensor(1, 1, t.c(), 1) : fact->CreateTensor(t.w(), t.h(), t.c(), 1);
//            size_t crowScaleBias = scaleBiasT->w() * scaleBiasT->h() * scaleBiasT->c();
//            buf.resize(crowScaleBias);
//
//            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//            SingleMatrix scale(crowScaleBias, 1, buf.data(), deviceId, matrixFlagNormal);
//            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//            SingleMatrix bias(crowScaleBias, 1, buf.data(), deviceId, matrixFlagNormal);
//
//            SingleMatrix runMeanBuf(deviceId);
//            SingleMatrix runMean = initMat(runMeanBuf, crowScaleBias, 1, buf);
//            SingleMatrix runMeanExp(runMean);
//            SingleMatrix runInvStdDevBuf(deviceId);
//            SingleMatrix runInvStdDev = initMat(runInvStdDevBuf, crowScaleBias, 1, buf);
//            SingleMatrix runInvStdDevExp(runInvStdDev);
//
//            SingleMatrix saveMeanBuf(deviceId);
//            SingleMatrix saveMean = initMat(saveMeanBuf, crowScaleBias, 1, buf);
//            SingleMatrix saveMeanExp(saveMean);
//            SingleMatrix saveInvStdDevBuf(deviceId);
//            SingleMatrix saveInvStdDev = initMat(saveInvStdDevBuf, crowScaleBias, 1, buf);
//            SingleMatrix saveInvStdDevExp(saveInvStdDev);
//
//            SingleMatrix outBuf(deviceId);
//            SingleMatrix out = initMat(outBuf, crow, ccol, buf);
//            SingleMatrix outExp(out);
//
//            CudaTimer time1;
//            time1.Start();
//            engCntk->NormalizeBatch(t, in, *scaleBiasT, scale, bias, spatial, expAvg, runMean, runInvStdDev,
//                                    out, eps, saveMean, saveInvStdDev);
//            time1.Stop();
//
//            CudaTimer time2;
//            time2.Start();
//            engCudnn->NormalizeBatch(t, in, *scaleBiasT, scale, bias, spatial, expAvg, runMeanExp, runInvStdDevExp,
//                                     outExp, eps, saveMeanExp, saveInvStdDevExp);
//            time2.Stop();
//
//            std::stringstream tmsg;
//            tmsg << "tensor: (w = " << t.w() << ", h = " << t.h() << ", c = " << t.c() << ", n = " << t.n() 
//                 << ", spatial = " << (spatial ? "true" : "false")
//                 << ", expAvg = " << expAvg << ")";
//            std::string msg = " are not equal, " + tmsg.str();
//            std::string msgNan = " has NaNs, " + tmsg.str();
//            std::string msgNotNan = " has buffer overflow/underflow, " + tmsg.str();
//
//            float relErr = Err<float>::Rel;
//            float absErr = Err<float>::Abs;
//            std::string emsg;
//
//            BOOST_REQUIRE_MESSAGE(!out.HasNan("out"), "out" << msgNan);
//            BOOST_REQUIRE_MESSAGE(CheckEqual(out, outExp, emsg, relErr, absErr * 20), "out" << msg << ". " << emsg);
//            BOOST_REQUIRE_MESSAGE(CountNans(outBuf) == crow * 2 * ccol, "out" << msgNotNan);
//            // REVIEW alexeyk: add cases for testing numerical stability.
//
//            BOOST_REQUIRE_MESSAGE(!runMean.HasNan("runMean"), "runMean" << msgNan);
//            BOOST_REQUIRE_MESSAGE(CheckEqual(runMean, runMeanExp, emsg, relErr, absErr), "runMean" << msg << ". " << emsg);
//            BOOST_REQUIRE_MESSAGE(CountNans(runMeanBuf) == crowScaleBias * 2, "runMean" << msgNotNan);
//
//            BOOST_REQUIRE_MESSAGE(!runInvStdDev.HasNan("runInvStdDev"), "runInvStdDev" << msgNan);
//            BOOST_REQUIRE_MESSAGE(CheckEqual(runInvStdDev, runInvStdDevExp, emsg, relErr, absErr), "runInvStdDev" << msg << ". " << emsg);
//            BOOST_REQUIRE_MESSAGE(CountNans(runInvStdDevBuf) == crowScaleBias * 2, "runInvStdDev" << msgNotNan);
//
//            BOOST_REQUIRE_MESSAGE(!saveMean.HasNan("saveMean"), "saveMean" << msgNan);
//            BOOST_REQUIRE_MESSAGE(CheckEqual(saveMean, saveMeanExp, emsg, relErr, absErr), "saveMean" << msg << ". " << emsg);
//            BOOST_REQUIRE_MESSAGE(CountNans(saveMeanBuf) == crowScaleBias * 2, "saveMean" << msgNotNan);
//
//            BOOST_REQUIRE_MESSAGE(!saveInvStdDev.HasNan("saveInvStdDev"), "saveInvStdDev" << msgNan);
//            BOOST_REQUIRE_MESSAGE(CheckEqual(saveInvStdDev, saveInvStdDevExp, emsg, relErr, absErr), "saveInvStdDev" << msg << ". " << emsg);
//            BOOST_REQUIRE_MESSAGE(CountNans(saveInvStdDevBuf) == crowScaleBias * 2, "saveInvStdDev" << msgNotNan);
//
//#ifndef _DEBUG
//            float elapsedCntk = time1.Elapsed();
//            float elapsedCudnn = time2.Elapsed();
//            // Check performance. Current version of cuDNN (v4 RC) is significanlty slower than CNTK implementation.
//            // For optimal cases (vectorSize % 32 == 0 and batchSize % 32 == 0), CNTK implementation can be >5x faster than cuDNN.
//            if (crow >= 32 && ccol >= 32)
//            {
//                // Use conservative estimates.
//                int speedup = 2;
//                BOOST_REQUIRE_MESSAGE(speedup * elapsedCntk < elapsedCudnn,
//                                      "CNTK implementation (" << elapsedCntk << "ms) must be faster than cuDNN (" << elapsedCudnn << "ms) by at least " << speedup << "x, what's changed? " << tmsg.str());
//            }
//#endif
//        }
//    }
//}
//
//BOOST_AUTO_TEST_CASE(BatchNormalizationForwardInference)
//{
//    if (!IsCuDnnSupported())
//        return;
//
//    std::mt19937 rng(0);
//    std::normal_distribution<float> nd;
//
//    auto initMat = [&](SingleMatrix& buf, size_t r, size_t c, vec& data) -> SingleMatrix
//    {
//        data.resize(r * 3 * c);
//        std::fill(begin(data), end(data), std::numeric_limits<float>::quiet_NaN());
//        std::generate(begin(data) + r * c, begin(data) + 2 * r * c, [&] { return nd(rng); });
//        buf.SetValue(r, 3 * c, buf.GetDeviceId(), data.data());
//        // Get center slice.
//        return buf.ColumnSlice(c, c);
//    };
//
//    for (int deviceId : {0})
//    {
//        auto fact = ConvFact::Create(deviceId, ConvFact::EngineType::Auto, ImageLayoutKind::CHW);
//        auto engCudnn = fact->CreateConvEngine(deviceId, ImageLayoutKind::CHW, 0, BatchNormImpl::CuDnn);
//        auto engCntk = fact->CreateConvEngine(deviceId, ImageLayoutKind::CHW, 0, BatchNormImpl::Cntk);
//        for (auto& cfg : GenerateBNTestConfigs(*fact))
//        {
//            auto& t = *std::move(std::get<0>(cfg));
//            bool spatial = std::get<1>(cfg);
//
//            size_t crow = t.w() * t.h() * t.c();
//            size_t ccol = t.n();
//
//            vec buf(crow * t.n());
//            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//            SingleMatrix in(crow, ccol, buf.data(), deviceId, matrixFlagNormal);
//
//            Tensor4DPtr scaleBiasT = spatial ? fact->CreateTensor(1, 1, t.c(), 1) : fact->CreateTensor(t.w(), t.h(), t.c(), 1);
//            size_t crowScaleBias = scaleBiasT->w() * scaleBiasT->h() * scaleBiasT->c();
//            buf.resize(crowScaleBias);
//
//            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//            SingleMatrix scale(crowScaleBias, 1, buf.data(), deviceId, matrixFlagNormal);
//            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//            SingleMatrix bias(crowScaleBias, 1, buf.data(), deviceId, matrixFlagNormal);
//
//            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//            SingleMatrix runMean(crowScaleBias, 1, buf.data(), deviceId, matrixFlagNormal);
//            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//            SingleMatrix runInvStdDev(crowScaleBias, 1, buf.data(), deviceId, matrixFlagNormal);
//
//            SingleMatrix outBuf(deviceId);
//            SingleMatrix out = initMat(outBuf, crow, ccol, buf);
//            SingleMatrix outExp(out);
//
//            CudaTimer time1;
//            time1.Start();
//            engCntk->NormalizeBatchInference(t, in, *scaleBiasT, scale, bias, spatial, runMean, runInvStdDev, out);
//            time1.Stop();
//
//            CudaTimer time2;
//            time2.Start();
//            engCudnn->NormalizeBatchInference(t, in, *scaleBiasT, scale, bias, spatial, runMean, runInvStdDev, outExp);
//            time2.Stop();
//
//            std::stringstream tmsg;
//            tmsg << "tensor: (w = " << t.w() << ", h = " << t.h() << ", c = " << t.c() << ", n = " << t.n() << ", spatial = " << (spatial ? "true" : "false") << ")";
//            std::string msg = " are not equal, " + tmsg.str();
//            std::string msgNan = " has NaNs, " + tmsg.str();
//            std::string msgNotNan = " has buffer overflow/underflow, " + tmsg.str();
//
//            float relErr = Err<float>::Rel;
//            float absErr = Err<float>::Abs;
//            std::string emsg;
//
//            BOOST_REQUIRE_MESSAGE(!out.HasNan("out"), "out" << msgNan);
//            BOOST_REQUIRE_MESSAGE(CheckEqual(out, outExp, emsg, relErr, absErr * 20), "out" << msg << ". " << emsg);
//            BOOST_REQUIRE_MESSAGE(CountNans(outBuf) == crow * 2 * ccol, "out" << msgNotNan);
//            // REVIEW alexeyk: add cases for testing numerical stability.
//
//#ifndef _DEBUG
//            float elapsedCntk = time1.Elapsed();
//            float elapsedCudnn = time2.Elapsed();
//            // Check performance. Current version of cuDNN (v4 RC) is significanlty slower than CNTK implementation.
//            // For optimal cases (vectorSize % 32 == 0 and batchSize % 32 == 0), CNTK implementation can be >5x faster than cuDNN.
//            if (crow >= 32 && ccol >= 32)
//            {
//                // Use conservative estimates.
//                int speedup = 2;
//                BOOST_REQUIRE_MESSAGE(speedup * elapsedCntk < elapsedCudnn,
//                                      "CNTK implementation (" << elapsedCntk << "ms) must be faster than cuDNN (" << elapsedCudnn << "ms) by at least " << speedup << "x, what's changed? " << tmsg.str());
//            }
//#endif
//        }
//    }
//}
//
//BOOST_AUTO_TEST_CASE(BatchNormalizationForwardInferenceCpu)
//{
//    if (!IsCuDnnSupported())
//        return;
//
//    std::mt19937 rng(0);
//    std::normal_distribution<float> nd;
//
//    auto initMat = [&](SingleMatrix& buf, size_t r, size_t c, vec& data) -> SingleMatrix
//    {
//        data.resize(r * 3 * c);
//        std::fill(begin(data), end(data), std::numeric_limits<float>::quiet_NaN());
//        std::generate(begin(data) + r * c, begin(data) + 2 * r * c, [&] { return nd(rng); });
//        buf.SetValue(r, 3 * c, buf.GetDeviceId(), data.data());
//        // Get center slice.
//        return buf.ColumnSlice(c, c);
//    };
//
//    int deviceId = -1;
//    int cudnnDeviceId = deviceId < 0 ? 0 : deviceId;
//    auto fact = ConvFact::Create(cudnnDeviceId, ConvFact::EngineType::CuDnn, ImageLayoutKind::CHW);
//    auto engCudnn = fact->CreateConvEngine(cudnnDeviceId, ImageLayoutKind::CHW, 0, BatchNormImpl::CuDnn);
//    auto testFact = ConvFact::Create(deviceId, ConvFact::EngineType::Auto, ImageLayoutKind::CHW);
//    auto engCntk = testFact->CreateConvEngine(deviceId, ImageLayoutKind::CHW, 0, BatchNormImpl::Cntk);
//    for (auto& cfg : GenerateBNTestConfigs(*fact))
//    {
//        auto& t = *std::move(std::get<0>(cfg));
//        bool spatial = std::get<1>(cfg);
//
//        size_t crow = t.w() * t.h() * t.c();
//        size_t ccol = t.n();
//
//        vec buf(crow * t.n());
//        std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//        SingleMatrix in(crow, ccol, buf.data(), deviceId, matrixFlagNormal);
//        SingleMatrix inExp(crow, ccol, buf.data(), cudnnDeviceId, matrixFlagNormal);
//
//        Tensor4DPtr scaleBiasT = spatial ? fact->CreateTensor(1, 1, t.c(), 1) : fact->CreateTensor(t.w(), t.h(), t.c(), 1);
//        size_t crowScaleBias = scaleBiasT->w() * scaleBiasT->h() * scaleBiasT->c();
//        buf.resize(crowScaleBias);
//
//        std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//        SingleMatrix scale(crowScaleBias, 1, buf.data(), deviceId, matrixFlagNormal);
//        SingleMatrix scaleExp(crowScaleBias, 1, buf.data(), cudnnDeviceId, matrixFlagNormal);
//        std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//        SingleMatrix bias(crowScaleBias, 1, buf.data(), deviceId, matrixFlagNormal);
//        SingleMatrix biasExp(crowScaleBias, 1, buf.data(), cudnnDeviceId, matrixFlagNormal);
//
//        std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//        SingleMatrix runMean(crowScaleBias, 1, buf.data(), deviceId, matrixFlagNormal);
//        SingleMatrix runMeanExp(crowScaleBias, 1, buf.data(), cudnnDeviceId, matrixFlagNormal);
//        std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//        SingleMatrix runInvStdDev(crowScaleBias, 1, buf.data(), deviceId, matrixFlagNormal);
//        SingleMatrix runInvStdDevExp(crowScaleBias, 1, buf.data(), cudnnDeviceId, matrixFlagNormal);
//
//        SingleMatrix outBuf(deviceId);
//        SingleMatrix out = initMat(outBuf, crow, ccol, buf);
//        SingleMatrix outExp(crow, ccol, out.CopyToArray(), cudnnDeviceId, matrixFlagNormal);
//
//        CudaTimer time1;
//        time1.Start();
//        engCntk->NormalizeBatchInference(t, in, *scaleBiasT, scale, bias, spatial, runMean, runInvStdDev, out);
//        time1.Stop();
//
//        CudaTimer time2;
//        time2.Start();
//        engCudnn->NormalizeBatchInference(t, inExp, *scaleBiasT, scaleExp, biasExp, spatial, runMeanExp, runInvStdDevExp, outExp);
//        time2.Stop();
//
//        std::stringstream tmsg;
//        tmsg << "tensor: (w = " << t.w() << ", h = " << t.h() << ", c = " << t.c() << ", n = " << t.n() << ", spatial = " << (spatial ? "true" : "false") << ")";
//        std::string msg = " are not equal, " + tmsg.str();
//        std::string msgNan = " has NaNs, " + tmsg.str();
//        std::string msgNotNan = " has buffer overflow/underflow, " + tmsg.str();
//
//        float relErr = Err<float>::Rel;
//        float absErr = Err<float>::Abs;
//        std::string emsg;
//
//        BOOST_REQUIRE_MESSAGE(!out.HasNan("out"), "out" << msgNan);
//        BOOST_REQUIRE_MESSAGE(CheckEqual(out, outExp, emsg, relErr, absErr * 20), "out" << msg << ". " << emsg);
//        BOOST_REQUIRE_MESSAGE(CountNans(outBuf) == crow * 2 * ccol, "out" << msgNotNan);
//    }
//}
//
//BOOST_AUTO_TEST_CASE(BatchNormalizationBackward)
//{
//    if (!IsCuDnnSupported())
//        return;
//
//    std::mt19937 rng(0);
//    std::normal_distribution<float> nd;
//
//    auto initMat = [&](SingleMatrix& buf, size_t r, size_t c, vec& data) -> SingleMatrix
//    {
//        data.resize(r * 3 * c);
//        std::fill(begin(data), end(data), std::numeric_limits<float>::quiet_NaN());
//        std::generate(begin(data) + r * c, begin(data) + 2 * r * c, [&] { return nd(rng); });
//        buf.SetValue(r, 3 * c, buf.GetDeviceId(), data.data());
//        // Get center slice.
//        return buf.ColumnSlice(c, c);
//    };
//
//    for (int deviceId : {0})
//    {
//        auto fact = ConvFact::Create(deviceId, ConvFact::EngineType::Auto, ImageLayoutKind::CHW);
//        auto engCudnn = fact->CreateConvEngine(deviceId, ImageLayoutKind::CHW, 0, BatchNormImpl::CuDnn);
//        auto engCntk = fact->CreateConvEngine(deviceId, ImageLayoutKind::CHW, 0, BatchNormImpl::Cntk);
//        for (auto& cfg : GenerateBNTestConfigs(*fact))
//        {
//            auto& t = *std::move(std::get<0>(cfg));
//            bool spatial = std::get<1>(cfg);
//
//            size_t crow = t.w() * t.h() * t.c();
//            size_t ccol = t.n();
//
//            vec buf(crow * t.n());
//            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//            SingleMatrix x(crow, ccol, buf.data(), deviceId, matrixFlagNormal);
//
//            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//            SingleMatrix dy(crow, ccol, buf.data(), deviceId, matrixFlagNormal);
//
//            Tensor4DPtr scaleBiasT = spatial ? fact->CreateTensor(1, 1, t.c(), 1) : fact->CreateTensor(t.w(), t.h(), t.c(), 1);
//            size_t crowScaleBias = scaleBiasT->w() * scaleBiasT->h() * scaleBiasT->c();
//            buf.resize(crowScaleBias);
//
//            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//            SingleMatrix scale(crowScaleBias, 1, buf.data(), deviceId, matrixFlagNormal);
//
//            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//            SingleMatrix saveMean(crowScaleBias, 1, buf.data(), deviceId, matrixFlagNormal);
//
//            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
//            SingleMatrix saveInvStdDev(crowScaleBias, 1, buf.data(), deviceId, matrixFlagNormal);
//
//            SingleMatrix dScaleBuf(deviceId);
//            SingleMatrix dScale = initMat(dScaleBuf, crowScaleBias, 1, buf);
//            SingleMatrix dScaleExp(dScale);
//            SingleMatrix dBiasBuf(deviceId);
//            SingleMatrix dBias = initMat(dBiasBuf, crowScaleBias, 1, buf);
//            SingleMatrix dBiasExp(dBias);
//
//            SingleMatrix dxBuf(deviceId);
//            SingleMatrix dx = initMat(dxBuf, crow, ccol, buf);
//            SingleMatrix dxExp(dx);
//
//            CudaTimer time1;
//            time1.Start();
//            engCntk->BackwardNormalizeBatch(t, x, dy, dx, *scaleBiasT, scale, spatial, saveMean, saveInvStdDev, dScale, dBias);
//            time1.Stop();
//
//            CudaTimer time2;
//            time2.Start();
//            engCudnn->BackwardNormalizeBatch(t, x, dy, dxExp, *scaleBiasT, scale, spatial, saveMean, saveInvStdDev, dScaleExp, dBiasExp);
//            time2.Stop();
//
//            std::stringstream tmsg;
//            tmsg << "tensor: (w = " << t.w() << ", h = " << t.h() << ", c = " << t.c() << ", n = " << t.n() << ", spatial = " << (spatial ? "true" : "false") << ")";
//            std::string msg = " are not equal, " + tmsg.str();
//            std::string msgNan = " has NaNs, " + tmsg.str();
//            std::string msgNotNan = " has buffer overflow/underflow, " + tmsg.str();
//
//            float relErr = Err<float>::Rel;
//            float absErr = Err<float>::Abs;
//            std::string emsg;
//
//            BOOST_REQUIRE_MESSAGE(!dx.HasNan("dx"), "dx" << msgNan);
//            BOOST_REQUIRE_MESSAGE(CheckEqual(dx, dxExp, emsg, relErr * 16, absErr * 8), "dx" << msg << ". " << emsg);
//            BOOST_REQUIRE_MESSAGE(CountNans(dxBuf) == crow * 2 * ccol, "out" << msgNotNan);
//            // REVIEW alexeyk: add cases for testing numerical stability.
//
//            BOOST_REQUIRE_MESSAGE(!dScale.HasNan("dScale"), "dScale" << msgNan);
//            BOOST_REQUIRE_MESSAGE(CheckEqual(dScale, dScaleExp, emsg, relErr * 32, absErr * 8), "dScale" << msg << ". " << emsg);
//            BOOST_REQUIRE_MESSAGE(CountNans(dScaleBuf) == crowScaleBias * 2, "dScale" << msgNotNan);
//
//            BOOST_REQUIRE_MESSAGE(!dBias.HasNan("dBias"), "dBias" << msgNan);
//            BOOST_REQUIRE_MESSAGE(CheckEqual(dBias, dBiasExp, emsg, relErr * 32, absErr * 8), "dBias" << msg << ". " << emsg);
//            BOOST_REQUIRE_MESSAGE(CountNans(dBiasBuf) == crowScaleBias * 2, "dBias" << msgNotNan);
//
//#ifndef _DEBUG
//            float elapsedCntk = time1.Elapsed();
//            float elapsedCudnn = time2.Elapsed();
//            // Check performance. Current version of cuDNN (v4 RC) is significanlty slower than CNTK implementation.
//            // For optimal cases (vectorSize % 32 == 0 and batchSize % 32 == 0), CNTK implementation can be >5x faster than cuDNN.
//            if (crow >= 32 && ccol >= 32)
//            {
//                // Use conservative estimates.
//                float speedup = 1.3f;
//                BOOST_REQUIRE_MESSAGE(speedup * elapsedCntk < elapsedCudnn,
//                                      "CNTK implementation (" << elapsedCntk << "ms) must be faster than cuDNN (" << elapsedCudnn << "ms) by at least " << speedup << "x, what's changed? " << tmsg.str());
//            }
//#endif
//        }
//    }
//}
//
//BOOST_AUTO_TEST_SUITE_END()

}
} } }
