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
        BEGIN_TEST_METHOD_ATTRIBUTE(ConvolutionForward)
            TEST_METHOD_ATTRIBUTE(L"Category", L"Convolution")
        END_TEST_METHOD_ATTRIBUTE()
        TEST_METHOD(ConvolutionForward)
        {
            int n = 2;
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

            for (int deviceId : { -1, 0 })
            {
                auto eng = ConvEng::Create(deviceId, 1000);
                auto inT = eng->CreateTensor(inW, inH, cmapIn, n);
                auto filtT = eng->CreateFilter(kW, kH, cmapIn, cmapOut);
                auto outT = eng->CreateTensor(outW, outH, cmapOut, n);
                auto convT = eng->CreateConvDescriptor(*inT, *filtT, sW, sH, false);

                std::vector<float> buf(inW * inH * cmapIn * n);
                int seed = 0;
                for (int i = 0; i < n; i++)
                {
                    seed = 0;
                    // Create input, cmapIn feature maps, inW x inH each.
                    std::generate(buf.begin() + i * buf.size() / n, buf.begin() + (i + 1) * buf.size() / n, [&seed]{ return seed++; });
                }
                SingleMatrix in(inW * inH * cmapIn, n, buf.data(), matrixFlagNormal, deviceId);

                seed = 0;
                buf.resize(kW * kH * cmapIn * cmapOut);
                // Create cmapOut filters, each kW x kH x cmapIn (CHWN format).
                std::generate(buf.begin(), buf.end(), [&seed]{ return seed++ / 2; });
                SingleMatrix filt(cmapOut, kW * kH * cmapIn, buf.data(), matrixFlagNormal, deviceId);

                SingleMatrix out(outW * outH * cmapOut, n, deviceId);

                eng->Forward(*inT, in, *filtT, filt, *convT, *outT, out);

                // Output is in NHWC format.
                float expBuf[] = {
                    7695.0f, 7695.0f,
                    9801.0f, 9801.0f,
                    18225.0f, 18225.0f,
                    20331.0f, 20331.0f,
                    7695.0f, 7695.0f,
                    9801.0f, 9801.0f,
                    18225.0f, 18225.0f,
                    20331.0f, 20331.0f
                };
                SingleMatrix exp(outW * outH * cmapOut, n, expBuf, matrixFlagNormal, deviceId);
                Assert::IsTrue(out.IsEqualTo(exp));
            }
        }

        BEGIN_TEST_METHOD_ATTRIBUTE(ConvolutionBackwardData)
            TEST_METHOD_ATTRIBUTE(L"Category", L"Convolution")
        END_TEST_METHOD_ATTRIBUTE()
        TEST_METHOD(ConvolutionBackwardData)
        {
            int n = 2;
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

            for (int deviceId : { -1 })
            {
                auto eng = ConvEng::Create(deviceId, 1000);
                auto srcGradT = eng->CreateTensor(outW, outH, cmapOut, n);
                auto filtT = eng->CreateFilter(kW, kH, cmapIn, cmapOut);
                auto gradT = eng->CreateTensor(inW, inH, cmapIn, n);
                auto convT = eng->CreateConvDescriptor(*gradT, *filtT, sW, sH, false);

                // Source grads is in NHWC format.
                float srcGradBuf[] = {
                    7695.0f, 7695.0f,
                    9801.0f, 9801.0f,
                    18225.0f, 18225.0f,
                    20331.0f, 20331.0f,
                    7695.0f, 7695.0f,
                    9801.0f, 9801.0f,
                    18225.0f, 18225.0f,
                    20331.0f, 20331.0f
                };
                SingleMatrix srcGrad(outW * outH * cmapOut, n, srcGradBuf, matrixFlagNormal, deviceId);

                std::vector<float> buf(inW * inH * cmapIn * n);
                buf.resize(kW * kH * cmapIn * cmapOut);
                // Create cmapOut filters, each kW x kH x cmapIn (CHWN format).
                int seed = 0;
                std::generate(buf.begin(), buf.end(), [&seed]{ return seed++ / 2; });
                SingleMatrix filt(cmapOut, kW * kH * cmapIn, buf.data(), matrixFlagNormal, deviceId);

                SingleMatrix grad(inW * inH * cmapIn, n, deviceId);
                eng->BackwardData(*srcGradT, srcGrad, *filtT, filt, *convT, *gradT, grad);

                // Target grads is in NHWC format.
                for (int i = 0; i < n; i++)
                {
                    seed = 0;
                    // Create input, cmapIn feature maps, inW x inH each.
                    std::generate(buf.begin() + i * buf.size() / n, buf.begin() + (i + 1) * buf.size() / n, [&seed]{ return seed++; });
                }
                SingleMatrix exp(inW * inH * cmapIn, n, buf.data(), matrixFlagNormal, deviceId);
                Assert::IsTrue(grad.IsEqualTo(exp));
            }
        }

    private:
        int GetNumOut(int i, int k, int s, bool pad)
        {
            return (i - (pad ? 1 : k)) / s + 1;
        }
    };
}
