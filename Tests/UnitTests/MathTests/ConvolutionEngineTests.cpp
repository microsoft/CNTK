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
#include "../../../Source/Math/CuDnnFactories.h"
#include "common.h"

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

using vec = std::vector<float>;

using ConvEng = ConvolutionEngine<float>;

bool AreEqual(float a, float b, float maxRelError, float maxAbsError)
{
    float diff = std::abs(a - b);
    if (diff <= maxAbsError)
        return true;
    float largest = std::max(std::abs(a), std::abs(b));
    return diff < largest * maxRelError;
}
bool AreEqual(double a, double b, double maxRelError, double maxAbsError)
{
    double diff = std::abs(a - b);
    if (diff <= maxAbsError)
        return true;
    double largest = std::max(std::abs(a), std::abs(b));
    return diff < largest * maxRelError;
}

size_t CountNans(const SingleMatrix& src)
{
    size_t n = 0;
    foreach_coord (i, j, src)
    {
        n += std::isnan(src(i, j)) ? 1 : 0;
    }
    return n;
}

// Returns vector of engine config parameters: <kind, device, maxTempMemSizeInSamples>
std::vector<std::tuple<ConvolutionEngineKind, DEVICEID_TYPE, size_t>> GetTestEngineConfigs()
{
    std::vector<std::tuple<ConvolutionEngineKind, DEVICEID_TYPE, size_t>> res;
    // Reference engine. The engine does not use temp memory so safe to set it to 0.
    res.push_back(std::make_tuple(ConvolutionEngineKind::Reference, -1, 0));
    res.push_back(std::make_tuple(ConvolutionEngineKind::Reference, 0, 0));

    // Gemm engine. Implemented only for CPU for now. Uses temp memory.
    res.push_back(std::make_tuple(ConvolutionEngineKind::Gemm, -1, 0));
    res.push_back(std::make_tuple(ConvolutionEngineKind::Gemm, -1, 1));
    res.push_back(std::make_tuple(ConvolutionEngineKind::Gemm, -1, 3));
    return res;
}

std::vector<ConvolveGeometryPtr> GenerateConvTestConfigs()
{
    std::vector<ConvolveGeometryPtr> res;
    // REVIEW alexeyk: add test cases with even dimensions of a kernel. There are some corner cases which cuDNN does not support (which essentially require negative padding).
    for (size_t kW : {1, 3})
    {
        for (size_t kH : {1, 3})
        {
            for (size_t inW : {kW, 2 * kW, 2 * kW - 1})
            {
                for (size_t inC : {1, 3})
                {
                    for (size_t mapCount : {1, 5})
                    {
                        for (size_t stride : {1, min((int)kW, min((int)kH, 2))})
                        {
                            // Note: must use sharing=false in channel dimension otherwise geometry will not be cuDNN compatible but cuDNN won't fail.
                            res.push_back(std::make_shared<ConvolveGeometry>(TensorShape(inW, max(kH, inW) + 1, inC),
                                TensorShape(kW, kH, inC), TensorShape(mapCount), TensorShape(stride, stride, inC),
                                ConvolveGeometry::BoolVec{true},
                                ConvolveGeometry::BoolVec{(kW & 1) != 0, (kH & 1) != 0, false},
                                TensorShape(0), TensorShape(0)));
                        }
                    }
                }
            }
        }
    }
    // For debugging.
    res.push_back(std::make_shared<ConvolveGeometry>(TensorShape(3, 3, 1),
        TensorShape(3, 3, 1), TensorShape(2), TensorShape(1, 1, 1),
        ConvolveGeometry::BoolVec{true}, ConvolveGeometry::BoolVec{true, true, false},
        TensorShape(0), TensorShape(0)));

    // Simple 3D convolution.
    res.push_back(std::make_shared<ConvolveGeometry>(TensorShape(5, 5, 5, 2),
        TensorShape(3, 3, 3, 2), TensorShape(2), TensorShape(1),
        ConvolveGeometry::BoolVec{true}, ConvolveGeometry::BoolVec{false},
        TensorShape(0), TensorShape(0)));
    // Example of 3D convolution that can be represented with 3D tensors in reference engine
    // but requires 4D tensors in other engines.
    res.push_back(std::make_shared<ConvolveGeometry>(TensorShape(5, 5, 3, 1),
        TensorShape(3, 3, 2, 1), TensorShape(2), TensorShape(1),
        ConvolveGeometry::BoolVec{true}, ConvolveGeometry::BoolVec{false},
        TensorShape(0), TensorShape(0)));

    res.push_back(std::make_shared<ConvolveGeometry>(TensorShape(16, 16, 1),
        TensorShape(3, 3, 1), TensorShape(8), TensorShape(1, 2, 1),
        ConvolveGeometry::BoolVec{true}, ConvolveGeometry::BoolVec{true, true, false},
        TensorShape(0), TensorShape(0)));

    // 1x1 convolution (shortcuts in ResNet).
    res.push_back(std::make_shared<ConvolveGeometry>(TensorShape(16, 16, 2),
        TensorShape(1, 1, 2), TensorShape(1), TensorShape(2, 2, 1),
        ConvolveGeometry::BoolVec{true}, ConvolveGeometry::BoolVec{false},
        TensorShape(0, 0, 0), TensorShape(0)));
    return res;
}

std::vector<ConvolveGeometryPtr> GeneratePoolTestConfigs()
{
    std::vector<ConvolveGeometryPtr> res;
    for (size_t kW : {1, 2, 3})
    {
        for (size_t kH : {1, 2, 3})
        {
            for (size_t inW : {kW, 2 * kW, 2 * kW - 1})
            {
                for (size_t inC : {1, 3})
                {
                    for (size_t stride : {1, min((int)kW, min((int)kH, 2))})
                    {
                        // Note: must always use autopadding otherwise there might be configurations that 
                        // require negative padding that cuDNN does not support.
                        res.push_back(std::make_shared<ConvolveGeometry>(TensorShape(inW, max(kH, inW) + 1, inC),
                            TensorShape(kW, kH, 1), TensorShape(1), TensorShape(stride, stride, 1),
                            ConvolveGeometry::BoolVec{true},
                            ConvolveGeometry::BoolVec{true, true, false},
                            TensorShape(0), TensorShape(0)));
                    }
                }
            }
        }
    }
    // For debugging.
    // Ordinary pooling.
    res.push_back(std::make_shared<ConvolveGeometry>(TensorShape(4, 4, 1),
        TensorShape(2, 2, 1), TensorShape(1), TensorShape(2, 2, 1),
        ConvolveGeometry::BoolVec{true}, ConvolveGeometry::BoolVec{true, true, false},
        TensorShape(0), TensorShape(0)));
    // Overlapped with padding.
    res.push_back(std::make_shared<ConvolveGeometry>(TensorShape(4, 4, 1),
        TensorShape(3, 3, 1), TensorShape(1), TensorShape(2, 2, 1),
        ConvolveGeometry::BoolVec{true}, ConvolveGeometry::BoolVec{true, true, false},
        TensorShape(0), TensorShape(0)));
    return res;
}

BOOST_AUTO_TEST_SUITE(ConvolutionSuite)

BOOST_AUTO_TEST_CASE(ConvolutionForward)
{
    std::mt19937 rng(0);
    std::uniform_int_distribution<> batchSizeG(1, 8);
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

    int baseDeviceId = 0;
    for (const auto& engCfg : GetTestEngineConfigs())
    {
        auto engKind = std::get<0>(engCfg);
        auto deviceId = std::get<1>(engCfg);
        auto maxTempMem = std::get<2>(engCfg);
        for (const auto& g : GenerateConvTestConfigs())
        {
            auto baseEng = ConvEng::Create(g, baseDeviceId, ImageLayoutKind::CHW, 0, PoolKind::None, ConvolutionEngineKind::CuDnn);
            auto testEng = ConvEng::Create(g, deviceId, ImageLayoutKind::CHW, maxTempMem, PoolKind::None, engKind);

            size_t n = batchSizeG(rng);
            vec buf;
            buf.resize(g->InputShape().GetNumElements() * n);
            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix in(g->InputShape().GetNumElements(), n, buf.data(), deviceId, matrixFlagNormal);
            SingleMatrix inB(g->InputShape().GetNumElements(), n, buf.data(), baseDeviceId, matrixFlagNormal);

            size_t mapCount = g->GetMapCount(g->InputShape().GetRank() - 1);
            buf.resize(g->KernelShape().GetNumElements() * mapCount);
            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix kernel(mapCount, g->KernelShape().GetNumElements(), buf.data(), deviceId, matrixFlagNormal);
            SingleMatrix kernelB(mapCount, g->KernelShape().GetNumElements(), buf.data(), baseDeviceId, matrixFlagNormal);

            size_t crowOut = g->OutputShape().GetNumElements();
            SingleMatrix outBuf(deviceId);
            SingleMatrix out = initMat(outBuf, crowOut, n, buf);
            SingleMatrix outB(out.DeepClone(), baseDeviceId);

            SingleMatrix workspace(deviceId);
            SingleMatrix workspaceB(baseDeviceId);

            testEng->Forward(in, kernel, out, workspace);
            baseEng->Forward(inB, kernelB, outB, workspaceB);

            std::stringstream tmsg;
            tmsg << "Geometry: " << (std::string)(*g) << ", Batch: " << n << ", Device: " << deviceId << ", MaxTempMem: " << maxTempMem;
            std::string msg = " are not equal, " + tmsg.str();
            std::string msgNan = " has NaNs, " + tmsg.str();
            std::string msgNotNan = " has buffer overflow/underflow, " + tmsg.str();

            float relErr = Err<float>::Rel;
            float absErr = Err<float>::Abs;
            std::string emsg;

            BOOST_REQUIRE_MESSAGE(!out.HasNan("out"), "out" << msgNan);
            BOOST_REQUIRE_MESSAGE(CheckEqual(out, outB, emsg, relErr * 4, absErr * 9), "out" << msg << ". " << emsg);
            BOOST_REQUIRE_MESSAGE(CountNans(outBuf) == crowOut * 2 * n, "out" << msgNotNan);
        }
    }
}

BOOST_AUTO_TEST_CASE(ConvolutionBackwardData)
{
    std::mt19937 rng(0);
    std::uniform_int_distribution<> batchSizeG(1, 8);
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

    int baseDeviceId = 0;
    for (const auto& engCfg : GetTestEngineConfigs())
    {
        auto engKind = std::get<0>(engCfg);
        auto deviceId = std::get<1>(engCfg);
        auto maxTempMem = std::get<2>(engCfg);
        for (const auto& g : GenerateConvTestConfigs())
        {
            auto baseEng = ConvEng::Create(g, baseDeviceId, ImageLayoutKind::CHW, 0, PoolKind::None, ConvolutionEngineKind::CuDnn);
            auto testEng = ConvEng::Create(g, deviceId, ImageLayoutKind::CHW, maxTempMem, PoolKind::None, engKind);

            size_t n = batchSizeG(rng);
            vec buf;
            buf.resize(g->OutputShape().GetNumElements() * n);
            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix srcGrad(g->OutputShape().GetNumElements(), n, buf.data(), deviceId, matrixFlagNormal);
            SingleMatrix srcGradB(g->OutputShape().GetNumElements(), n, buf.data(), baseDeviceId, matrixFlagNormal);

            size_t mapCount = g->GetMapCount(g->InputShape().GetRank() - 1);
            buf.resize(g->KernelShape().GetNumElements() * mapCount);
            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix kernel(mapCount, g->KernelShape().GetNumElements(), buf.data(), deviceId, matrixFlagNormal);
            SingleMatrix kernelB(mapCount, g->KernelShape().GetNumElements(), buf.data(), baseDeviceId, matrixFlagNormal);

            size_t crowGrad = g->InputShape().GetNumElements();
            SingleMatrix gradBuf(deviceId);
            SingleMatrix grad = initMat(gradBuf, crowGrad, n, buf);
            SingleMatrix gradB(grad.DeepClone(), baseDeviceId);

            SingleMatrix workspace(deviceId);
            SingleMatrix workspaceB(baseDeviceId);

            testEng->BackwardData(srcGrad, kernel, grad, workspace);
            baseEng->BackwardData(srcGradB, kernelB, gradB, workspaceB);

            std::stringstream tmsg;
            tmsg << "Geometry: " << (std::string)(*g) << ", Batch: " << n << ", Device: " << deviceId;
            std::string msg = " are not equal, " + tmsg.str();
            std::string msgNan = " has NaNs, " + tmsg.str();
            std::string msgNotNan = " has buffer overflow/underflow, " + tmsg.str();

            float relErr = Err<float>::Rel;
            float absErr = Err<float>::Abs;
            std::string emsg;

            BOOST_REQUIRE_MESSAGE(!grad.HasNan("grad"), "grad" << msgNan);
            BOOST_REQUIRE_MESSAGE(CheckEqual(grad, gradB, emsg, relErr * 16, absErr * 16), "grad" << msg << ". " << emsg);
            BOOST_REQUIRE_MESSAGE(CountNans(gradBuf) == crowGrad * 2 * n, "grad" << msgNotNan);
        }
    }
}

BOOST_AUTO_TEST_CASE(ConvolutionBackwardKernel)
{
    std::mt19937 rng(0);
    std::uniform_int_distribution<> batchSizeG(1, 8);
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

    int baseDeviceId = 0;
    for (const auto& engCfg : GetTestEngineConfigs())
    {
        auto engKind = std::get<0>(engCfg);
        auto deviceId = std::get<1>(engCfg);
        auto maxTempMem = std::get<2>(engCfg);
        for (const auto& g : GenerateConvTestConfigs())
        {
            auto baseEng = ConvEng::Create(g, baseDeviceId, ImageLayoutKind::CHW, 0, PoolKind::None, ConvolutionEngineKind::CuDnn);
            auto testEng = ConvEng::Create(g, deviceId, ImageLayoutKind::CHW, maxTempMem, PoolKind::None, engKind);

            size_t n = batchSizeG(rng);
            vec buf;
            buf.resize(g->InputShape().GetNumElements() * n);
            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix in(g->InputShape().GetNumElements(), n, buf.data(), deviceId, matrixFlagNormal);
            SingleMatrix inB(g->InputShape().GetNumElements(), n, buf.data(), baseDeviceId, matrixFlagNormal);

            buf.resize(g->OutputShape().GetNumElements() * n);
            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix grad(g->OutputShape().GetNumElements(), n, buf.data(), deviceId, matrixFlagNormal);
            SingleMatrix gradB(g->OutputShape().GetNumElements(), n, buf.data(), baseDeviceId, matrixFlagNormal);

            size_t mapCount = g->GetMapCount(g->InputShape().GetRank() - 1);
            buf.resize(g->KernelShape().GetNumElements() * mapCount);
            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix kernelBuf(deviceId);
            SingleMatrix kernel = initMat(kernelBuf, mapCount, g->KernelShape().GetNumElements(), buf);
            SingleMatrix kernelB(kernel.DeepClone(), baseDeviceId);

            SingleMatrix workspace(deviceId);
            SingleMatrix workspaceB(baseDeviceId);
            
            testEng->BackwardKernel(grad, in, kernel, false, workspace);
            baseEng->BackwardKernel(gradB, inB, kernelB, false, workspaceB);
            
            std::stringstream tmsg;
            tmsg << "Geometry: " << (std::string)(*g) << ", Batch: " << n << ", Device: " << deviceId;
            std::string msg = " are not equal, " + tmsg.str();
            std::string msgNan = " has NaNs, " + tmsg.str();
            std::string msgNotNan = " has buffer overflow/underflow, " + tmsg.str();

            float relErr = Err<float>::Rel;
            float absErr = Err<float>::Abs;
            std::string emsg;

            BOOST_REQUIRE_MESSAGE(!kernel.HasNan("kernel"), "kernel" << msgNan);
            BOOST_REQUIRE_MESSAGE(CheckEqual(kernel, kernelB, emsg, relErr * 32, absErr * 32), "kernel" << msg << ". " << emsg);
            BOOST_REQUIRE_MESSAGE(CountNans(kernelBuf) == kernel.GetNumElements() * 2, "kernel" << msgNotNan);
        }
    }
}

BOOST_AUTO_TEST_CASE(PoolingForward)
{
    std::mt19937 rng(0);
    std::uniform_int_distribution<> batchSizeG(1, 8);
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

    int baseDeviceId = 0;
    auto engKind = ConvolutionEngineKind::Reference;
    for (auto kind : {PoolKind::Max, PoolKind::Average})
    {
        for (int deviceId : {-1, 0})
        {
            for (const auto& g : GeneratePoolTestConfigs())
            {
                auto baseEng = ConvEng::Create(g, baseDeviceId, ImageLayoutKind::CHW, 0, kind, ConvolutionEngineKind::CuDnn);
                auto testEng = ConvEng::Create(g, deviceId, ImageLayoutKind::CHW, 0, kind, engKind);

                size_t n = batchSizeG(rng);
                vec buf;
                buf.resize(g->InputShape().GetNumElements() * n);
                std::generate(begin(buf), end(buf), [&] { return nd(rng); });
                SingleMatrix in(g->InputShape().GetNumElements(), n, buf.data(), deviceId, matrixFlagNormal);
                SingleMatrix inB(g->InputShape().GetNumElements(), n, buf.data(), baseDeviceId, matrixFlagNormal);

                size_t crowOut = g->OutputShape().GetNumElements();
                SingleMatrix outBuf(deviceId);
                SingleMatrix out = initMat(outBuf, crowOut, n, buf);
                SingleMatrix outB(out.DeepClone(), baseDeviceId);

                testEng->ForwardPooling(in, out);
                baseEng->ForwardPooling(inB, outB);

                std::stringstream tmsg;
                tmsg << "Geometry: " << (std::string)(*g) << ", Pool: " << (int)kind << ", Batch: " << n << ", Device: " << deviceId;
                std::string msg = " are not equal, " + tmsg.str();
                std::string msgNan = " has NaNs, " + tmsg.str();
                std::string msgNotNan = " has buffer overflow/underflow, " + tmsg.str();

                float relErr = Err<float>::Rel;
                float absErr = Err<float>::Abs;
                std::string emsg;

                BOOST_REQUIRE_MESSAGE(!out.HasNan("out"), "out" << msgNan);
                BOOST_REQUIRE_MESSAGE(CheckEqual(out, outB, emsg, relErr, absErr * 8), "out" << msg << ". " << emsg);
                BOOST_REQUIRE_MESSAGE(CountNans(outBuf) == crowOut * 2 * n, "out" << msgNotNan);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(PoolingBackward)
{
    std::mt19937 rng(0);
    std::uniform_int_distribution<> batchSizeG(1, 8);
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

    int baseDeviceId = 0;
    auto engKind = ConvolutionEngineKind::Reference;
    for (auto kind : {PoolKind::Max, PoolKind::Average})
    {
        for (int deviceId : {-1, 0})
        {
            for (const auto& g : GeneratePoolTestConfigs())
            {
                auto baseEng = ConvEng::Create(g, baseDeviceId, ImageLayoutKind::CHW, 0, kind, ConvolutionEngineKind::CuDnn);
                auto testEng = ConvEng::Create(g, deviceId, ImageLayoutKind::CHW, 0, kind, engKind);

                size_t n = batchSizeG(rng);
                vec buf;
                size_t crowIn = g->InputShape().GetNumElements();
                buf.resize(crowIn * n);
                std::generate(begin(buf), end(buf), [&] { return nd(rng); });
                SingleMatrix inB(crowIn, n, buf.data(), baseDeviceId, matrixFlagNormal);
                SingleMatrix in(crowIn, n, buf.data(), deviceId, matrixFlagNormal);

                size_t crowOut = g->OutputShape().GetNumElements();
                buf.resize(crowOut * n);
                std::generate(begin(buf), end(buf), [&] { return nd(rng); });
                SingleMatrix srcGradB(crowOut, n, buf.data(), baseDeviceId, matrixFlagNormal);
                SingleMatrix srcGrad(crowOut, n, buf.data(), deviceId, matrixFlagNormal);
                // Do not generate for out as it will be replaced anyway.
                SingleMatrix outB(crowOut, n, buf.data(), baseDeviceId, matrixFlagNormal);
                SingleMatrix out(crowOut, n, buf.data(), deviceId, matrixFlagNormal);

                testEng->ForwardPooling(in, out);
                baseEng->ForwardPooling(inB, outB);

                SingleMatrix gradBuf(deviceId);
                SingleMatrix grad = initMat(gradBuf, crowIn, n, buf);
                SingleMatrix gradB(grad.DeepClone(), baseDeviceId);

                testEng->BackwardPooling(out, srcGrad, in, grad);
                baseEng->BackwardPooling(outB, srcGradB, inB, gradB);

                std::stringstream tmsg;
                tmsg << "Geometry: " << (std::string)(*g) << ", Pool: " << (int)kind << ", Batch: " << n << ", Device: " << deviceId;
                std::string msg = " are not equal, " + tmsg.str();
                std::string msgNan = " has NaNs, " + tmsg.str();
                std::string msgNotNan = " has buffer overflow/underflow, " + tmsg.str();

                float relErr = Err<float>::Rel;
                float absErr = Err<float>::Abs;
                std::string emsg;

                BOOST_REQUIRE_MESSAGE(!grad.HasNan("grad"), "grad" << msgNan);
                BOOST_REQUIRE_MESSAGE(CheckEqual(grad, gradB, emsg, relErr, absErr * 8), "grad" << msg << ". " << emsg);
                BOOST_REQUIRE_MESSAGE(CountNans(gradBuf) == crowIn * 2 * n, "grad" << msgNotNan);
            }
        }
    }
}

BOOST_AUTO_TEST_CASE(MaxUnpooling)
{
    using IntMatrix = Matrix<int>;

    std::mt19937 rng(0);
    std::uniform_int_distribution<> batchSizeG(1, 8);
    // Using uniform distribution with positive values to avoid issues with
    // unpooling negative values.
    std::uniform_real_distribution<float> nd(0, 1);

    auto initMat = [&](SingleMatrix& buf, size_t r, size_t c, vec& data) -> SingleMatrix
    {
        data.resize(r * 3 * c);
        std::fill(begin(data), end(data), std::numeric_limits<float>::quiet_NaN());
        std::generate(begin(data) + r * c, begin(data) + 2 * r * c, [&] { return nd(rng); });
        buf.SetValue(r, 3 * c, buf.GetDeviceId(), data.data());
        // Get center slice.
        return buf.ColumnSlice(c, c);
    };

    int cpuDeviceId = -1;
    int gpuDeviceId = 0;

    for (const auto& g : GeneratePoolTestConfigs())
    {
        // cpuEng and gpuEng are used to compare results against each other.
        auto cpuEng = ConvEng::Create(g, cpuDeviceId, ImageLayoutKind::CHW, 0, PoolKind::Max, ConvolutionEngineKind::Reference);
        auto gpuEng = ConvEng::Create(g, gpuDeviceId, ImageLayoutKind::CHW, 0, PoolKind::Max, ConvolutionEngineKind::Reference);

        size_t n = batchSizeG(rng);
        vec buf;
        buf.resize(g->InputShape().GetNumElements() * n);
        std::generate(begin(buf), end(buf), [&] { return nd(rng); });
        SingleMatrix inC(g->InputShape().GetNumElements(), n, buf.data(), cpuDeviceId, matrixFlagNormal);
        SingleMatrix inG(g->InputShape().GetNumElements(), n, buf.data(), gpuDeviceId, matrixFlagNormal);

        // First, compute max pooling output and corresponding mask.
        SingleMatrix outC(g->OutputShape().GetNumElements(), n, cpuDeviceId);
        SingleMatrix outG(g->OutputShape().GetNumElements(), n, gpuDeviceId);

        cpuEng->ForwardPooling(inC, outC);
        gpuEng->ForwardPooling(inG, outG);
        
        // Second, do the unpooling.
        size_t crowIn = g->InputShape().GetNumElements();
        SingleMatrix inUBufC(cpuDeviceId);
        SingleMatrix inUC = initMat(inUBufC, crowIn, n, buf);
        SingleMatrix inUBufG(inUBufC.DeepClone(), gpuDeviceId);
        SingleMatrix inUG = initMat(inUBufG, crowIn, n, buf);

        cpuEng->MaxUnpooling(outC, inC, inUC);
        gpuEng->MaxUnpooling(outG, inG, inUG);

        // Check that CPU/GPU results are the same.
        std::stringstream tmsg;
        tmsg << "Geometry: " << (std::string)(*g) << ", Batch: " << n;
        std::string msg = " are not equal, " + tmsg.str();
        std::string msgNan = " has NaNs, " + tmsg.str();
        std::string msgNotNan = " has buffer overflow/underflow, " + tmsg.str();

        float relErr = 0;
        float absErr = 0;
        std::string emsg;

        BOOST_REQUIRE_MESSAGE(!inUC.HasNan("inUC"), "inUC" << msgNan);
        BOOST_REQUIRE_MESSAGE(!inUG.HasNan("inUG"), "inUG" << msgNan);
        BOOST_REQUIRE_MESSAGE(CheckEqual(inUC, inUG, emsg, relErr, absErr), "inU" << msg << ". " << emsg);
        BOOST_REQUIRE_MESSAGE(CountNans(inUBufC) == crowIn * 2 * n, "inUBufC" << msgNotNan);
        BOOST_REQUIRE_MESSAGE(CountNans(inUBufG) == crowIn * 2 * n, "inUBufG" << msgNotNan);

        // Now do the pooling from unpooled source and compare with original pooling.
        SingleMatrix outC_2(g->OutputShape().GetNumElements(), n, cpuDeviceId);
        SingleMatrix outG_2(g->OutputShape().GetNumElements(), n, gpuDeviceId);
        cpuEng->ForwardPooling(inUC, outC_2);
        gpuEng->ForwardPooling(inUG, outG_2);

        BOOST_REQUIRE_MESSAGE(CheckEqual(outC_2, outC, emsg, relErr, absErr), "outC_2" << msg << ". " << emsg);
        BOOST_REQUIRE_MESSAGE(CheckEqual(outG_2, outG, emsg, relErr, absErr), "outG_2" << msg << ". " << emsg);
    }
}

BOOST_AUTO_TEST_SUITE_END()

} } } }
