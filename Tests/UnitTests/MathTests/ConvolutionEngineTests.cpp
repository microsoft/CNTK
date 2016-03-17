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

    res.push_back(std::make_shared<ConvolveGeometry>(TensorShape(16, 16, 1),
        TensorShape(3, 3, 1), TensorShape(8), TensorShape(1, 2, 1),
        ConvolveGeometry::BoolVec{true}, ConvolveGeometry::BoolVec{true, true, false},
        TensorShape(0), TensorShape(0)));
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
    auto engKind = ConvolutionEngineKind::Reference;
    for (int deviceId : {-1, 0})
    {
        for (const auto& g : GenerateConvTestConfigs())
        {
            auto baseEng = ConvEng::Create(g, baseDeviceId, ImageLayoutKind::CHW, 0, PoolKind::None, ConvolutionEngineKind::CuDnn);
            auto testEng = ConvEng::Create(g, deviceId, ImageLayoutKind::CHW, 0, PoolKind::None, engKind);

            size_t n = batchSizeG(rng);
            vec buf;
            buf.resize(g->InputShape().GetNumElements() * n);
            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix in(g->InputShape().GetNumElements(), n, buf.data(), deviceId, matrixFlagNormal);
            SingleMatrix inB(g->InputShape().GetNumElements(), n, buf.data(), baseDeviceId, matrixFlagNormal);
            
            size_t mapCount = g->GetMapCount(g->InputShape().GetRank() - 1);
            buf.resize(g->KernelShape().GetNumElements() * mapCount);
            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix filter(mapCount, g->KernelShape().GetNumElements(), buf.data(), deviceId, matrixFlagNormal);
            SingleMatrix filterB(mapCount, g->KernelShape().GetNumElements(), buf.data(), baseDeviceId, matrixFlagNormal);

            size_t crowOut = g->OutputShape().GetNumElements();
            SingleMatrix outBuf(deviceId);
            SingleMatrix out = initMat(outBuf, crowOut, n, buf);
            SingleMatrix outB(out.DeepClone(), baseDeviceId);

            SingleMatrix workspace(deviceId);
            SingleMatrix workspaceB(baseDeviceId);
            
            testEng->Forward(in, filter, out, workspace);
            baseEng->Forward(inB, filterB, outB, workspaceB);
            
            std::stringstream tmsg;
            tmsg << "Geometry: " << (std::string)(*g) << ", Batch: " << n;
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
    auto engKind = ConvolutionEngineKind::Reference;
    for (int deviceId : {-1})
    {
        for (const auto& g : GenerateConvTestConfigs())
        {
            auto baseEng = ConvEng::Create(g, baseDeviceId, ImageLayoutKind::CHW, 0, PoolKind::None, ConvolutionEngineKind::CuDnn);
            auto testEng = ConvEng::Create(g, deviceId, ImageLayoutKind::CHW, 0, PoolKind::None, engKind);

            size_t n = batchSizeG(rng);
            vec buf;
            buf.resize(g->OutputShape().GetNumElements() * n);
            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix srcGradB(g->OutputShape().GetNumElements(), n, buf.data(), baseDeviceId, matrixFlagNormal);
            SingleMatrix srcGradT(g->OutputShape().GetNumElements(), n, buf.data(), deviceId, matrixFlagNormal);
            
            size_t mapCount = g->GetMapCount(g->InputShape().GetRank() - 1);
            buf.resize(g->KernelShape().GetNumElements() * mapCount);
            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix filterB(mapCount, g->KernelShape().GetNumElements(), buf.data(), baseDeviceId, matrixFlagNormal);
            SingleMatrix filterT(mapCount, g->KernelShape().GetNumElements(), buf.data(), deviceId, matrixFlagNormal);
            
            size_t crowGrad = g->InputShape().GetNumElements();
            SingleMatrix gradBuf(deviceId);
            SingleMatrix gradT = initMat(gradBuf, crowGrad, n, buf);
            SingleMatrix gradB(gradT.DeepClone(), baseDeviceId);

            SingleMatrix workspaceT(deviceId);
            SingleMatrix workspaceB(baseDeviceId);
            
            testEng->BackwardData(srcGradT, filterT, gradT, workspaceT);
            baseEng->BackwardData(srcGradB, filterB, gradB, workspaceB);
            
            std::stringstream tmsg;
            tmsg << "Geometry: " << (std::string)(*g) << ", Batch: " << n;
            std::string msg = " are not equal, " + tmsg.str();
            std::string msgNan = " has NaNs, " + tmsg.str();
            std::string msgNotNan = " has buffer overflow/underflow, " + tmsg.str();

            float relErr = Err<float>::Rel;
            float absErr = Err<float>::Abs;
            std::string emsg;

            BOOST_REQUIRE_MESSAGE(!gradT.HasNan("grad"), "grad" << msgNan);
            BOOST_REQUIRE_MESSAGE(CheckEqual(gradT, gradB, emsg, relErr * 4, absErr * 8), "grad" << msg << ". " << emsg);
            BOOST_REQUIRE_MESSAGE(CountNans(gradBuf) == crowGrad * 2 * n, "grad" << msgNotNan);
        }
    }
}

BOOST_AUTO_TEST_CASE(ConvolutionBackwardFilter)
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
    for (int deviceId : {-1})
    {
        for (const auto& g : GenerateConvTestConfigs())
        {
            auto baseEng = ConvEng::Create(g, baseDeviceId, ImageLayoutKind::CHW, 0, PoolKind::None, ConvolutionEngineKind::CuDnn);
            auto testEng = ConvEng::Create(g, deviceId, ImageLayoutKind::CHW, 0, PoolKind::None, engKind);

            size_t n = batchSizeG(rng);
            vec buf;
            buf.resize(g->InputShape().GetNumElements() * n);
            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix inB(g->InputShape().GetNumElements(), n, buf.data(), baseDeviceId, matrixFlagNormal);
            SingleMatrix inT(g->InputShape().GetNumElements(), n, buf.data(), deviceId, matrixFlagNormal);

            buf.resize(g->OutputShape().GetNumElements() * n);
            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix gradB(g->OutputShape().GetNumElements(), n, buf.data(), baseDeviceId, matrixFlagNormal);
            SingleMatrix gradT(g->OutputShape().GetNumElements(), n, buf.data(), deviceId, matrixFlagNormal);

            size_t mapCount = g->GetMapCount(g->InputShape().GetRank() - 1);
            buf.resize(g->KernelShape().GetNumElements() * mapCount);
            std::generate(begin(buf), end(buf), [&] { return nd(rng); });
            SingleMatrix filterBuf(deviceId);
            SingleMatrix filterT = initMat(filterBuf, mapCount, g->KernelShape().GetNumElements(), buf);
            SingleMatrix filterB(filterT.DeepClone(), baseDeviceId);

            SingleMatrix workspaceT(deviceId);
            SingleMatrix workspaceB(baseDeviceId);
            
            testEng->BackwardFilter(gradT, inT, filterT, false, workspaceT);
            baseEng->BackwardFilter(gradB, inB, filterB, false, workspaceB);
            
            std::stringstream tmsg;
            tmsg << "Geometry: " << (std::string)(*g) << ", Batch: " << n;
            std::string msg = " are not equal, " + tmsg.str();
            std::string msgNan = " has NaNs, " + tmsg.str();
            std::string msgNotNan = " has buffer overflow/underflow, " + tmsg.str();

            float relErr = Err<float>::Rel;
            float absErr = Err<float>::Abs;
            std::string emsg;

            BOOST_REQUIRE_MESSAGE(!filterT.HasNan("filter"), "filter" << msgNan);
            BOOST_REQUIRE_MESSAGE(CheckEqual(filterT, filterB, emsg, relErr * 2, absErr * 8), "filter" << msg << ". " << emsg);
            BOOST_REQUIRE_MESSAGE(CountNans(filterBuf) == filterT.GetNumElements() * 2, "filter" << msgNotNan);
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
                tmsg << "Geometry: " << (std::string)(*g) << ", Pool: " << (int)kind << ", Batch: " << n;
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
                tmsg << "Geometry: " << (std::string)(*g) << ", Pool: " << (int)kind << ", Batch: " << n;
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

BOOST_AUTO_TEST_SUITE_END()

} } } }
