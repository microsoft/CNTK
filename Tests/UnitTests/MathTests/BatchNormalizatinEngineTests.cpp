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
#include "../../../Source/Math/CuDnnConvolutionEngine.h"
#include "../../../Source/Math/BatchNormalizationEngine.h"
#include "common.h"

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

using vec = std::vector<float>;
using BNEng = BatchNormEngine<float>;

std::vector<std::tuple<TensorShape, TensorShape, bool, double>> GenerateBNTestConfigs()
{
    std::vector<std::tuple<TensorShape, TensorShape, bool, double>> res;
    // REVIEW alexeyk: how to test batches > 512? cuDNN does not support that so there is no baseline.
    double expAvgFactor = 1;
    // Per activation (non-spatial)
    for (size_t n : {6, 13, 62, 512})
    {
        for (size_t c : {1})
        {
            for (size_t h : {1})
            {
                for (size_t w : {6, 17, 126, 2048})
                {
                    res.push_back(std::make_tuple(TensorShape(w, h, c, n), TensorShape(w, h, c, 1), false, expAvgFactor));
                }
            }
        }
    }
    // Spatial
    for (size_t n : {2, 11, 64})
    {
        for (size_t c : {2, 13, 32})
        {
            for (size_t h : {2, 11, 16})
            {
                for (size_t w : {2, 11, 16})
                {
                    res.push_back(std::make_tuple(TensorShape(w, h, c, n), TensorShape(1, 1, c, 1), true, expAvgFactor));
                }
            }
        }
    }
    // For perf testing (similar to first layers of ResNet).
    res.push_back(std::make_tuple(TensorShape(56, 56, 64, 64), TensorShape(1, 1, 64, 1), true, expAvgFactor));
    // Next test will fail in cuDNN due to bug we discovered (and reported to NVIDIA).
    //res.push_back(std::make_tuple(std::move(fact.CreateTensor(2, 2, 2048, 2)), true));
    res.push_back(std::make_tuple(TensorShape(2, 2, 2048, 64), TensorShape(1, 1, 2048, 1), true, expAvgFactor));

    // Test running mean/isd.
    expAvgFactor = 0.1;
    res.push_back(std::make_tuple(TensorShape(2, 2, 2, 8), TensorShape(2, 2, 2, 1), false, expAvgFactor));
    res.push_back(std::make_tuple(TensorShape(2, 2, 2, 8), TensorShape(1, 1, 2, 1), true, expAvgFactor));

    return res;
}

BOOST_AUTO_TEST_SUITE(BatchNormalizationSuite)

BOOST_AUTO_TEST_CASE(BatchNormalizationForwardTrain)
{
    std::mt19937 rng(0);
    std::normal_distribution<float> nd;

    auto initMat = [&](SingleMatrix& buf, size_t r, size_t c, vec& data) -> SingleMatrix
    {
        data.resize(r * 3 * c);
        std::fill(begin(data), end(data), std::numeric_limits<float>::quiet_NaN());
        std::generate(begin(data) + r * c, begin(data) + 2 * r * c, [&] { return nd(rng); });
        buf.SetValue(r, 3 * c, buf.GetDeviceId(), data.data());
        // Get center slice.
        return buf.ColumnSlice(c, c);
    };

    for (int deviceId : {0})
    {
        for (const auto& cfg : GenerateBNTestConfigs())
        {
            const auto& inOutT = std::get<0>(cfg);
            const auto& scaleT = std::get<1>(cfg);
            bool spatial = std::get<2>(cfg);
            double expAvg = std::get<3>(cfg);
            double eps = 1e-5; // CUDNN_BN_MIN_EPSILON

            auto engCudnn = BNEng::Create(deviceId, inOutT, scaleT, spatial, ImageLayoutKind::CHW, BatchNormEngineKind::CuDnn);
            auto engCntk = BNEng::Create(deviceId, inOutT, scaleT, spatial, ImageLayoutKind::CHW, BatchNormEngineKind::Cntk);

            size_t crow = inOutT[0] * inOutT[1] * inOutT[2];
            size_t ccol = inOutT[3];

            vec buf(crow * ccol);
            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix in(crow, ccol, buf.data(), deviceId, matrixFlagNormal);

            size_t crowScaleBias = scaleT.GetNumElements();
            buf.resize(crowScaleBias);

            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix scale(crowScaleBias, 1, buf.data(), deviceId, matrixFlagNormal);
            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix bias(crowScaleBias, 1, buf.data(), deviceId, matrixFlagNormal);

            SingleMatrix runMeanBuf(deviceId);
            SingleMatrix runMean = initMat(runMeanBuf, crowScaleBias, 1, buf);
            SingleMatrix runMeanExp(runMean.DeepClone(), deviceId);
            SingleMatrix runInvStdDevBuf(deviceId);
            SingleMatrix runInvStdDev = initMat(runInvStdDevBuf, crowScaleBias, 1, buf);
            SingleMatrix runInvStdDevExp(runInvStdDev.DeepClone(), deviceId);

            SingleMatrix saveMeanBuf(deviceId);
            SingleMatrix saveMean = initMat(saveMeanBuf, crowScaleBias, 1, buf);
            SingleMatrix saveMeanExp(saveMean.DeepClone(), deviceId);
            SingleMatrix saveInvStdDevBuf(deviceId);
            SingleMatrix saveInvStdDev = initMat(saveInvStdDevBuf, crowScaleBias, 1, buf);
            SingleMatrix saveInvStdDevExp(saveInvStdDev.DeepClone(), deviceId);

            SingleMatrix outBuf(deviceId);
            SingleMatrix out = initMat(outBuf, crow, ccol, buf);
            SingleMatrix outExp(out.DeepClone(), deviceId);

            CudaTimer time1;
            time1.Start();
            engCntk->Forward(in, scale, bias, expAvg, runMean, runInvStdDev, out, eps, saveMean, saveInvStdDev);
            time1.Stop();

            CudaTimer time2;
            time2.Start();
            engCudnn->Forward(in, scale, bias, expAvg, runMean, runInvStdDev, out, eps, saveMean, saveInvStdDev);
            time2.Stop();

            std::stringstream tmsg;
            tmsg << "inOut tensor: " << (std::string)inOutT << ", scaleBias tensor: " << (std::string)scaleT
                 << ", spatial = " << (spatial ? "true" : "false")
                 << ", expAvg = " << expAvg << ")";
            std::string msg = " are not equal, " + tmsg.str();
            std::string msgNan = " has NaNs, " + tmsg.str();
            std::string msgNotNan = " has buffer overflow/underflow, " + tmsg.str();

            float relErr = Err<float>::Rel;
            float absErr = Err<float>::Abs;
            std::string emsg;

            BOOST_REQUIRE_MESSAGE(!out.HasNan("out"), "out" << msgNan);
            BOOST_REQUIRE_MESSAGE(CheckEqual(out, outExp, emsg, relErr, absErr * 20), "out" << msg << ". " << emsg);
            BOOST_REQUIRE_MESSAGE(CountNans(outBuf) == crow * 2 * ccol, "out" << msgNotNan);
            // REVIEW alexeyk: add cases for testing numerical stability.

            BOOST_REQUIRE_MESSAGE(!runMean.HasNan("runMean"), "runMean" << msgNan);
            BOOST_REQUIRE_MESSAGE(CheckEqual(runMean, runMeanExp, emsg, relErr, absErr), "runMean" << msg << ". " << emsg);
            BOOST_REQUIRE_MESSAGE(CountNans(runMeanBuf) == crowScaleBias * 2, "runMean" << msgNotNan);

            BOOST_REQUIRE_MESSAGE(!runInvStdDev.HasNan("runInvStdDev"), "runInvStdDev" << msgNan);
            BOOST_REQUIRE_MESSAGE(CheckEqual(runInvStdDev, runInvStdDevExp, emsg, relErr, absErr), "runInvStdDev" << msg << ". " << emsg);
            BOOST_REQUIRE_MESSAGE(CountNans(runInvStdDevBuf) == crowScaleBias * 2, "runInvStdDev" << msgNotNan);

            BOOST_REQUIRE_MESSAGE(!saveMean.HasNan("saveMean"), "saveMean" << msgNan);
            BOOST_REQUIRE_MESSAGE(CheckEqual(saveMean, saveMeanExp, emsg, relErr, absErr), "saveMean" << msg << ". " << emsg);
            BOOST_REQUIRE_MESSAGE(CountNans(saveMeanBuf) == crowScaleBias * 2, "saveMean" << msgNotNan);

            BOOST_REQUIRE_MESSAGE(!saveInvStdDev.HasNan("saveInvStdDev"), "saveInvStdDev" << msgNan);
            BOOST_REQUIRE_MESSAGE(CheckEqual(saveInvStdDev, saveInvStdDevExp, emsg, relErr, absErr), "saveInvStdDev" << msg << ". " << emsg);
            BOOST_REQUIRE_MESSAGE(CountNans(saveInvStdDevBuf) == crowScaleBias * 2, "saveInvStdDev" << msgNotNan);

#ifndef _DEBUG
            float elapsedCntk = time1.Elapsed();
            float elapsedCudnn = time2.Elapsed();
            // Check performance. Current version of cuDNN (v4 RC) is significanlty slower than CNTK implementation.
            // For optimal cases (vectorSize % 32 == 0 and batchSize % 32 == 0), CNTK implementation can be >5x faster than cuDNN.
            if (crow >= 32 && ccol >= 32)
            {
                // Use conservative estimates.
                int speedup = 2;
                BOOST_REQUIRE_MESSAGE(speedup * elapsedCntk < elapsedCudnn,
                                      "CNTK implementation (" << elapsedCntk << "ms) must be faster than cuDNN (" << elapsedCudnn << "ms) by at least " << speedup << "x, what's changed? " << tmsg.str());
            }
#endif
        }
    }
}

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

BOOST_AUTO_TEST_SUITE_END()

} } } }
