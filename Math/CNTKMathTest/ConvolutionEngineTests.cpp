//
// <copyright file="ConvolutionEngineTests.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include "stdafx.h"
#include "CppUnitTest.h"
#include <algorithm>
#include "..\Math\Matrix.h"
#include "..\Math\CPUMatrix.h"
#include "..\Math\GPUMatrix.h"
#include "..\Math\ConvolutionEngine.h"
#include "..\Math\CuDnnConvolutionEngine.h"

namespace CNTKMathTest
{
    using namespace Microsoft::MSR::CNTK;
    using namespace Microsoft::VisualStudio::CppUnitTestFramework;

    TEST_CLASS(MatrixUnitTest)
    {
        using ConvEng = ConvolutionEngine<float>;

    public:
        BEGIN_TEST_METHOD_ATTRIBUTE(SimpleConvolution)
            TEST_METHOD_ATTRIBUTE(L"Category", L"Convolution")
        END_TEST_METHOD_ATTRIBUTE()
        TEST_METHOD(SimpleConvolution)
        {
            int n = 1;
            int cmapIn = 3;
            int inW = 5;
            int inH = 5;
            int kW = 3;
            int kH = 3;
            int sW = 2;
            int sH = 2;
            int cmapOut = 2;
            int outW = GetNumOut(inW, kW, sW, false);
            int outH = GetNumOut(inH, kH, sH, false);

            int deviceId = 0;
            auto eng = ConvEng::Create(deviceId, 1000);
            auto inT = eng->CreateTensor(inW, inH, cmapIn, n);
            auto filtT = eng->CreateFilter(kW, kH, cmapIn, cmapOut);
            auto outT = eng->CreateTensor(outW, outH, cmapOut, n);
            auto convT = eng->CreateConvDescriptor(*inT, *filtT, sW, sH, false);

            std::vector<float> buf(inW * inH * cmapIn * n);
            int seed = 0;
            // Create input, cmapIn feature maps, inW x inH each.
            std::generate(buf.begin(), buf.end(), [&seed]{ return seed++; });
            SingleMatrix in(inW * inH * cmapIn, n, buf.data(), matrixFlagNormal, deviceId);

            seed = 0;
            buf.resize(kW * kH * cmapIn * cmapOut);
            // Create cmapOut filters, each kW x kH x cmapIn (NHWC format).
            std::generate(buf.begin(), buf.end(), [&seed]{ return seed++ / 2; });
            SingleMatrix filt(cmapOut, kW * kH * cmapIn, buf.data(), matrixFlagNormal, deviceId);

            SingleMatrix out(outW * outH * cmapOut, n, deviceId);

            eng->Forward(*inT, in, *filtT, filt, *convT, *outT, out);

            // Output is in NHWC format.
            float expBuf[] = {
                7695.0f, 7695.0f,
                9801.0f, 9801.0f,
                18225.0f, 18225.0f,
                20331.0f, 20331.0f
            };
            SingleMatrix exp(outW * outH * cmapOut, n, expBuf, matrixFlagNormal, deviceId);
            Assert::IsTrue(out.IsEqualTo(exp));
        }

    private:
        int GetNumOut(int i, int k, int s, bool pad)
        {
            return (i - (pad ? 1 : k)) / s + 1;
        }
    };
}
