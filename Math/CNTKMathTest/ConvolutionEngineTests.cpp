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
        using ConvFact = ConvolutionEngineFactory<float>;
        using vec = std::vector<float>;

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
                auto fact = ConvFact::Create(deviceId);
                auto eng = fact->CreateConvEngine(0);
                auto inT = fact->CreateTensor(inW, inH, cmapIn, n);
                auto filtT = fact->CreateFilter(kW, kH, cmapIn, cmapOut);
                auto outT = fact->CreateTensor(outW, outH, cmapOut, n);
                auto convT = fact->CreateConvDescriptor(*inT, *filtT, sW, sH, false);

                vec buf(inW * inH * cmapIn * n);
                int seed = 0;
                for (int i = 0; i < n; i++)
                {
                    seed = 0;
                    // Create input, cmapIn feature maps, inW x inH each (CHWN format).
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

        // REVIEW alexeyk: this really should be rolled into ConvolutionForward, make it data-driven.
        BEGIN_TEST_METHOD_ATTRIBUTE(ConvolutionForwardPad)
            TEST_METHOD_ATTRIBUTE(L"Category", L"Convolution")
        END_TEST_METHOD_ATTRIBUTE()
        TEST_METHOD(ConvolutionForwardPad)
        {
            int n = 2;
            int cmapIn = 1;
            int inW = 4;
            int inH = 4;
            int kW = 3;
            int kH = 3;
            int sW = 2;
            int sH = 2;
            int cmapOut = 1;
            bool pad = true;
            int outW = GetNumOut(inW, kW, sW, pad);
            int outH = GetNumOut(inH, kH, sH, pad);

            for (int deviceId : { -1, 0 })
            {
                auto fact = ConvFact::Create(deviceId);
                auto eng = fact->CreateConvEngine(0);
                auto inT = fact->CreateTensor(inW, inH, cmapIn, n);
                auto filtT = fact->CreateFilter(kW, kH, cmapIn, cmapOut);
                auto outT = fact->CreateTensor(outW, outH, cmapOut, n);
                auto convT = fact->CreateConvDescriptor(*inT, *filtT, sW, sH, pad);

                // Input in NHWC format.
                SingleMatrix in(inW * inH * cmapIn, n, vec(inW * inH * cmapIn * n, 1.0f).data(), matrixFlagNormal, deviceId);
                // Create cmapOut filters, each kW x kH x cmapIn (CHWN format).
                SingleMatrix filt(cmapOut, kW * kH * cmapIn, vec(kW * kH * cmapIn * cmapOut, 1.0f).data(), matrixFlagNormal, deviceId);

                SingleMatrix out(outW * outH * cmapOut, n, deviceId);

                eng->Forward(*inT, in, *filtT, filt, *convT, *outT, out);

                // Output is in NHWC format.
                float expBuf[] = {
                    4.0f, 6.0f,
                    6.0f, 9.0f,
                    4.0f, 6.0f,
                    6.0f, 9.0f,
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
            // REVIEW alexeyk: very simple test, improve.
            int n = 2;
            int cmapIn = 3;
            int inW = 3;
            int inH = 3;
            int kW = 3;
            int kH = 3;
            int sW = 2;
            int sH = 2;
            int cmapOut = 2;
            int outW = GetNumOut(inW, kW, sW, false);
            int outH = GetNumOut(inH, kH, sH, false);

            for (int deviceId : { -1, 0 })
            {
                auto fact = ConvFact::Create(deviceId);
                auto eng = fact->CreateConvEngine(0);
                auto srcGradT = fact->CreateTensor(outW, outH, cmapOut, n);
                auto filtT = fact->CreateFilter(kW, kH, cmapIn, cmapOut);
                auto gradT = fact->CreateTensor(inW, inH, cmapIn, n);
                auto convT = fact->CreateConvDescriptor(*gradT, *filtT, sW, sH, false);

                // Source grads is in NHWC format.
                float srcGradBuf[] = {
                    1.0f, 1.0f,
                    1.0f, 1.0f
                };
                SingleMatrix srcGrad(outW * outH * cmapOut, n, srcGradBuf, matrixFlagNormal, deviceId);

                vec filtB(kW * kH * cmapIn * cmapOut);
                // Create cmapOut filters, each kW x kH x cmapIn (CHWN format).
                int seed = 0;
                std::generate(filtB.begin(), filtB.end(), [&seed, cmapOut]{ return seed++ / cmapOut; });
                SingleMatrix filt(cmapOut, kW * kH * cmapIn, filtB.data(), matrixFlagNormal, deviceId);

                SingleMatrix grad(inW * inH * cmapIn, n, deviceId);
                eng->BackwardData(*srcGradT, srcGrad, *filtT, filt, *convT, *gradT, grad);

                // Target grads is in NHWC format.
                vec gradB(inW * inH * cmapIn * n);
                for (int i = 0; i < n; i++)
                {
                    for (int icur = 0; icur < inW * inH * cmapIn; icur++)
                    {
                        gradB[i * inW * inH * cmapIn + icur] = (float)(icur / cmapIn + (icur % cmapIn) * kW * kH) * cmapOut;
                    }
                }
                SingleMatrix exp(inW * inH * cmapIn, n, gradB.data(), matrixFlagNormal, deviceId);
                Assert::IsTrue(grad.IsEqualTo(exp));
            }
        }

        BEGIN_TEST_METHOD_ATTRIBUTE(ConvolutionBackwardFilter)
            TEST_METHOD_ATTRIBUTE(L"Category", L"Convolution")
        END_TEST_METHOD_ATTRIBUTE()
        TEST_METHOD(ConvolutionBackwardFilter)
        {
            // REVIEW alexeyk: very simple test, improve.
            int n = 2;
            int cmapIn = 3;
            int inW = 3;
            int inH = 3;
            int kW = 3;
            int kH = 3;
            int sW = 2;
            int sH = 2;
            int cmapOut = 2;
            int outW = GetNumOut(inW, kW, sW, false);
            int outH = GetNumOut(inH, kH, sH, false);

            for (int deviceId : { -1, 0 })
            {
                auto fact = ConvFact::Create(deviceId);
                auto eng = fact->CreateConvEngine(0);
                auto srcGradT = fact->CreateTensor(outW, outH, cmapOut, n);
                auto filtT = fact->CreateFilter(kW, kH, cmapIn, cmapOut);
                auto inT = fact->CreateTensor(inW, inH, cmapIn, n);
                auto convT = fact->CreateConvDescriptor(*inT, *filtT, sW, sH, false);

                // Source grads is in NHWC format.
                float srcGradBuf[] = {
                    1.0f, 1.0f,
                    1.0f, 1.0f,
                };
                SingleMatrix srcGrad(outW * outH * cmapOut, n, srcGradBuf, matrixFlagNormal, deviceId);

                vec buf(inW * inH * cmapIn * n);
                int seed = 0;
                for (int i = 0; i < n; i++)
                {
                    seed = 0;
                    // Create input, cmapIn feature maps, inW x inH each, NHWC format.
                    std::generate(buf.begin() + i * buf.size() / n, buf.begin() + (i + 1) * buf.size() / n, [&seed]{ return seed++; });
                }
                SingleMatrix in(inW * inH * cmapIn, n, buf.data(), matrixFlagNormal, deviceId);

                SingleMatrix filt(cmapOut, kW * kH * cmapIn, deviceId);
                filt.SetValue(0);
                
                eng->BackwardFilter(*srcGradT, srcGrad, *inT, in, *convT, *filtT, filt, false);

                vec expFiltB(cmapOut * kW * kH * cmapIn);
                for (int icur = 0; icur < inW * inH * cmapIn; icur++)
                {
                    float val = (float)(n * ((icur % (kW * kH)) * cmapIn + icur / (kW * kH)));
                    for (int i = icur * cmapOut; i < (icur + 1) * cmapOut; i++)
                        expFiltB[i] = val;
                }
                SingleMatrix exp(cmapOut, kW * kH * cmapIn, expFiltB.data(), matrixFlagNormal, deviceId);
                Assert::IsTrue(filt.IsEqualTo(exp));
            }
        }

        BEGIN_TEST_METHOD_ATTRIBUTE(MaxPoolForward)
            TEST_METHOD_ATTRIBUTE(L"Category", L"Convolution")
        END_TEST_METHOD_ATTRIBUTE()
        TEST_METHOD(MaxPoolForward)
        {
            int n = 2;
            int cmap = 2;
            int inW = 4;
            int inH = 4;
            int kW = 2;
            int kH = 2;
            int sW = 2;
            int sH = 2;
            int outW = GetNumOut(inW, kW, sW, false);
            int outH = GetNumOut(inH, kH, sH, false);

            for (int deviceId : { -1, 0 })
            {
                auto fact = ConvFact::Create(deviceId);
                auto eng = fact->CreatePoolEngine();
                auto inT = fact->CreateTensor(inW, inH, cmap, n);
                auto outT = fact->CreateTensor(outW, outH, cmap, n);
                auto poolT = fact->CreatePoolDescriptor(PoolingDescriptor::PoolKind::Max, kW, kH, sW, sH, 0, 0);

                vec buf(inW * inH * cmap * n);
                int seed = 0;
                // Create input, cmapIn feature maps, inW x inH each (NHWC format).
                std::generate(buf.begin(), buf.end(), [=, &seed]{ return seed++ % (inW * inH * cmap); });
                SingleMatrix in(inW * inH * cmap, n, buf.data(), matrixFlagNormal, deviceId);

                SingleMatrix out(outW * outH * cmap, n, deviceId);

                eng->Forward(*inT, in, *poolT, *outT, out);

                // Output is in NHWC format.
                float expBuf[] = {
                    10.0f, 11.0f,
                    14.0f, 15.0f,
                    26.0f, 27.0f,
                    30.0f, 31.0f,
                    10.0f, 11.0f,
                    14.0f, 15.0f,
                    26.0f, 27.0f,
                    30.0f, 31.0f,
                };
                SingleMatrix exp(outW * outH * cmap, n, expBuf, matrixFlagNormal, deviceId);
                Assert::IsTrue(out.IsEqualTo(exp));
            }
        }

        BEGIN_TEST_METHOD_ATTRIBUTE(MaxPoolBackward)
            TEST_METHOD_ATTRIBUTE(L"Category", L"Convolution")
        END_TEST_METHOD_ATTRIBUTE()
        TEST_METHOD(MaxPoolBackward)
        {
            int n = 2;
            int cmap = 2;
            int inW = 4;
            int inH = 4;
            int kW = 2;
            int kH = 2;
            int sW = 2;
            int sH = 2;
            int outW = GetNumOut(inW, kW, sW, false);
            int outH = GetNumOut(inH, kH, sH, false);

            for (int deviceId : { -1, 0 })
            {
                auto fact = ConvFact::Create(deviceId);
                auto eng = fact->CreatePoolEngine();
                auto inT = fact->CreateTensor(inW, inH, cmap, n);
                auto outT = fact->CreateTensor(outW, outH, cmap, n);
                auto poolT = fact->CreatePoolDescriptor(PoolingDescriptor::PoolKind::Max, kW, kH, sW, sH, 0, 0);

                vec buf(inW * inH * cmap * n);
                int seed = 0;
                // Create input, cmapIn feature maps, inW x inH each (NHWC format).
                std::generate(buf.begin(), buf.end(), [=, &seed]{ return seed++ % (inW * inH * cmap); });
                SingleMatrix in(inW * inH * cmap, n, buf.data(), matrixFlagNormal, deviceId);
                SingleMatrix out(outW * outH * cmap, n, deviceId);
                // Do forward pass first.
                eng->Forward(*inT, in, *poolT, *outT, out);

                // For gradients, use the same values as outputs.
                SingleMatrix srcGrad(out);
                SingleMatrix grad(inW * inH * cmap, n, deviceId);
                grad.SetValue(0);

                eng->Backward(*outT, out, srcGrad, *poolT, *inT, in, grad);

                // Output is in NHWC format.
                std::fill(buf.begin(), buf.end(), 0.0f);
                vec expMap = {
                    10.0f, 11.0f,
                    14.0f, 15.0f,
                    26.0f, 27.0f,
                    30.0f, 31.0f,
                    10.0f, 11.0f,
                    14.0f, 15.0f,
                    26.0f, 27.0f,
                    30.0f, 31.0f,
                };
                for (size_t i = 0; i < expMap.size(); i++)
                    buf[(int)expMap[i] + inW * inH * cmap * (i / (expMap.size() / n)) ] = expMap[i];
                SingleMatrix exp(inW * inH * cmap, n, buf.data(), matrixFlagNormal, deviceId);

                Assert::IsTrue(grad.IsEqualTo(exp));
            }
        }

        BEGIN_TEST_METHOD_ATTRIBUTE(AvgPoolForward)
            TEST_METHOD_ATTRIBUTE(L"Category", L"Convolution")
        END_TEST_METHOD_ATTRIBUTE()
        TEST_METHOD(AvgPoolForward)
        {
            int n = 2;
            int cmap = 2;
            int inW = 4;
            int inH = 4;
            int kW = 2;
            int kH = 2;
            int sW = 2;
            int sH = 2;
            int outW = GetNumOut(inW, kW, sW, false);
            int outH = GetNumOut(inH, kH, sH, false);

            for (int deviceId : { -1, 0 })
            {
                auto fact = ConvFact::Create(deviceId);
                auto eng = fact->CreatePoolEngine();
                auto inT = fact->CreateTensor(inW, inH, cmap, n);
                auto outT = fact->CreateTensor(outW, outH, cmap, n);
                auto poolT = fact->CreatePoolDescriptor(PoolingDescriptor::PoolKind::Average, kW, kH, sW, sH, 0, 0);

                vec buf(inW * inH * cmap * n);
                int seed = 0;
                // Create input, cmapIn feature maps, inW x inH each (NHWC format).
                std::generate(buf.begin(), buf.end(), [=, &seed]{ return seed++ % (inW * inH * cmap); });
                SingleMatrix in(inW * inH * cmap, n, buf.data(), matrixFlagNormal, deviceId);

                SingleMatrix out(outW * outH * cmap, n, deviceId);

                eng->Forward(*inT, in, *poolT, *outT, out);

                // Output is in NHWC format.
                float expBuf[] = {
                    5.0f,  6.0f,
                    9.0f,  10.0f,
                    21.0f, 22.0f,
                    25.0f, 26.0f,
                    5.0f,  6.0f,
                    9.0f,  10.0f,
                    21.0f, 22.0f,
                    25.0f, 26.0f,
                };
                SingleMatrix exp(outW * outH * cmap, n, expBuf, matrixFlagNormal, deviceId);
                Assert::IsTrue(out.IsEqualTo(exp));
            }
        }

        BEGIN_TEST_METHOD_ATTRIBUTE(AvgPoolBackward)
            TEST_METHOD_ATTRIBUTE(L"Category", L"Convolution")
        END_TEST_METHOD_ATTRIBUTE()
        TEST_METHOD(AvgPoolBackward)
        {
            int n = 1;
            int cmap = 1;
            int inW = 4;
            int inH = 4;
            int kW = 2;
            int kH = 2;
            int sW = 2;
            int sH = 2;
            int outW = GetNumOut(inW, kW, sW, false);
            int outH = GetNumOut(inH, kH, sH, false);

            for (int deviceId : { 0 })
            {
                auto fact = ConvFact::Create(deviceId);
                auto eng = fact->CreatePoolEngine();
                auto inT = fact->CreateTensor(inW, inH, cmap, n);
                auto outT = fact->CreateTensor(outW, outH, cmap, n);
                auto poolT = fact->CreatePoolDescriptor(PoolingDescriptor::PoolKind::Average, kW, kH, sW, sH, 0, 0);

                vec buf(inW * inH * cmap * n);
                int seed = 0;
                // Create input, cmapIn feature maps, inW x inH each (NHWC format).
                std::generate(buf.begin(), buf.end(), [=, &seed]{ return seed++ % (inW * inH * cmap); });
                SingleMatrix in(inW * inH * cmap, n, buf.data(), matrixFlagNormal, deviceId);
                SingleMatrix out(outW * outH * cmap, n, deviceId);
                // Do forward pass first.
                eng->Forward(*inT, in, *poolT, *outT, out);

                // For gradients, use the same values as outputs.
                SingleMatrix srcGrad(out);
                SingleMatrix grad(inW * inH * cmap, n, deviceId);
                grad.SetValue(0);

                eng->Backward(*outT, out, srcGrad, *poolT, *inT, in, grad);
                auto aa = grad.CopyToArray();
                UNUSED(aa);
                //// Output is in NHWC format.
                //float expBuf[] = {
                //    5.0f,  6.0f,
                //    9.0f,  10.0f,
                //    21.0f, 22.0f,
                //    25.0f, 26.0f,
                //    5.0f,  6.0f,
                //    9.0f,  10.0f,
                //    21.0f, 22.0f,
                //    25.0f, 26.0f,
                //};
                //SingleMatrix exp(outW * outH * cmap, n, expBuf, matrixFlagNormal, deviceId);
                //Assert::IsTrue(out.IsEqualTo(exp));
            }
        }

    private:
        int GetNumOut(int i, int k, int s, bool pad)
        {
            return (i - (pad ? 1 : k)) / s + 1;
        }
    };
}
