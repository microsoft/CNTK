//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "BatchNormalizationEngine.h"
#include "CuDnnFactories.h"
#include "Mkl2017DnnCommon.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class ElemType>
void BatchNormEngine<ElemType>::Forward(const Mat& in, const Mat& scale, const Mat& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, Mat& runMean, Mat& runVariance,
                                        Mat& out, double epsilon, Mat& savedMean, Mat& savedInvStdDev)
{
    assert(in.GetNumRows() == m_inOutT.GetNumElements());
    assert(out.GetNumRows() == m_inOutT.GetNumElements());
    assert(in.GetNumCols() == out.GetNumCols());
    assert(std::isfinite(expAvgFactor) && (0 <= expAvgFactor && expAvgFactor <= 1));
    assert(std::isfinite(blendFactor) && (0 <= blendFactor && blendFactor <= 1));
    // In inference mode, must only use running statistics
    assert(!inferenceOnly || ((expAvgFactor == 0.0) && (blendFactor == 1.0)));
    assert(std::isfinite(epsilon) && epsilon > 0);
    if (!m_spatial)
    {
        assert(m_inOutT.GetNumElements() == scale.GetNumRows());
        assert(m_inOutT.GetNumElements() == bias.GetNumRows());
        assert(m_inOutT.GetNumElements() == runMean.GetNumRows());
        assert(m_inOutT.GetNumElements() == runVariance.GetNumRows());
    }
    else
    {
        assert((m_inOutT.GetNumElements() % scale.GetNumRows()) == 0);
        assert((m_inOutT.GetNumElements() % bias.GetNumRows()) == 0);
        assert((m_inOutT.GetNumElements() % runMean.GetNumRows()) == 0);
        assert((m_inOutT.GetNumElements() % runVariance.GetNumRows()) == 0);
    }
    assert(scale.GetNumCols() == 1);
    assert(bias.GetNumCols() == 1);
    assert(runMean.GetNumCols() == 1);
    assert(runVariance.GetNumCols() == 1);

    EnsureCompatible();
    ForwardCore(in, scale, bias, inferenceOnly, expAvgFactor, blendFactor, runMean, runVariance, out, epsilon, savedMean, savedInvStdDev);

    if (!inferenceOnly)
    {
        assert(!savedMean.IsEmpty());
        assert(!savedInvStdDev.IsEmpty());
        if (!m_spatial)
        {
            assert(m_inOutT.GetNumElements() == savedMean.GetNumRows());
            assert(m_inOutT.GetNumElements() == savedInvStdDev.GetNumRows());
        }
        else
        {
            assert((m_inOutT.GetNumElements() % savedMean.GetNumRows()) == 0);
            assert((m_inOutT.GetNumElements() % savedInvStdDev.GetNumRows()) == 0);
        }
        assert(savedMean.GetNumCols() == 1);
        assert(savedInvStdDev.GetNumCols() == 1);
    }
}

template <class ElemType>
void BatchNormEngine<ElemType>::Backward(const Mat& in, const Mat& srcGrad, Mat& grad, const Mat& scale, double blendFactor,
                                         const Mat& savedMean, const Mat& savedInvStdDev, Mat& scaleGrad, Mat& biasGrad, bool accumulateDataGrad)
{
    assert(!savedMean.IsEmpty());
    assert(!savedInvStdDev.IsEmpty());
    EnsureCompatible();
    BackwardCore(in, srcGrad, grad, scale, blendFactor, savedMean, savedInvStdDev, scaleGrad, biasGrad, accumulateDataGrad);
}

template <class ElemType>
class CntkBatchNormEngine : public BatchNormEngine<ElemType>
{
public:
    using Base = BatchNormEngine<ElemType>;
    using typename Base::Mat;

public:
    CntkBatchNormEngine(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
        bool spatial, ImageLayoutKind imageLayout)
        : Base(deviceId, inOutT, spatial, imageLayout)
    {
    }

protected:
    using Base::m_deviceId;
    using Base::m_imageLayout;
    using Base::m_inOutT;
    using Base::m_spatial;

    void EnsureCompatible() override
    {
        if (m_spatial && m_imageLayout == ImageLayoutKind::HWC)
            InvalidArgument("CNTK batch normalization supports only cudnn(CHW) layout.");
    }

    void ForwardCore(const Mat& in, const Mat& scale, const Mat& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, Mat& runMean, Mat& runVariance,
        Mat& out, double epsilon, Mat& savedMean, Mat& savedInvStdDev) override
    {
#ifdef USE_MKL2017DNN
        if (in.GetCurrentMatrixLocation() == CPU &&
            ForwardCoreMKL(in, scale, bias, inferenceOnly, expAvgFactor, runMean, runVariance, out, epsilon, savedMean, savedInvStdDev))
            return;
#endif

        in.BatchNormalizationForward(scale, bias, inferenceOnly, expAvgFactor, blendFactor, runMean, runVariance, out, epsilon, savedMean, savedInvStdDev);
    }

    void BackwardCore(const Mat& in, const Mat& srcGrad, Mat& grad, const Mat& scale, double blendFactor, const Mat& savedMean, const Mat& savedInvStdDev,
        Mat& scaleGrad, Mat& biasGrad, bool accumulateDataGrad) override
    {
#ifdef USE_MKL2017DNN
        if (srcGrad.GetCurrentMatrixLocation() == CPU &&
            BackwardCoreMKL(in, srcGrad, grad, scale, savedMean, savedInvStdDev, scaleGrad, biasGrad, accumulateDataGrad))
            return;
#endif
        if (!accumulateDataGrad)
            grad.SetValue((ElemType)0);

        srcGrad.BatchNormalizationBackward(in, grad, scale, blendFactor, savedMean, savedInvStdDev, scaleGrad, biasGrad);
    }

private:
#ifdef USE_MKL2017DNN
    // default epsilon that matches cuDNN when no forward executed
    #define DEFAULT_EPSILON 1e-5

    enum ContextIndex
    {
        ContextIndex_ForwardInfer = 0,
        ContextIndex_ForwardTrain,
        ContextIndex_Backward,
        ContextIndex_Total
    };

    class MKLBatchNormalizationContext
    {
    private:
        int m_contextFlags = 0;

        // MKL uses a single buffer for both scale and shift, so allocate a buffer and convert
        struct MKLScaleShiftAdapter
        {
            bool isInput;
            std::shared_ptr<Matrix<ElemType>> mat;
            dnnResourceType_t resourceType;
            size_t numChannels;

            void Create(dnnResourceType_t rt, bool userToPrim, size_t n)
            {
                Clear();
                numChannels = n;
                mat = std::make_shared<Matrix<ElemType>>(numChannels, 2, CPUDEVICE);
                isInput = userToPrim;
                resourceType = rt;
            }

            void PrepareForExecution(void* scale, void* bias, void* resources[dnnResourceNumber])
            {
                ElemType* buffer = mat->Data();
                resources[resourceType] = buffer;
                if (isInput)
                {
                    memcpy(buffer, scale, sizeof(ElemType) * numChannels);
                    memcpy(buffer + numChannels, bias, sizeof(ElemType) * numChannels);
                }
            }

            void ConvertOutput(void* scale, void* bias)
            {
                if (isInput)
                    RuntimeError("Cannot execute output ResourceAdapter for input");

                ElemType* buffer = mat->Data();
                memcpy(scale, buffer, sizeof(ElemType) * numChannels);
                memcpy(bias, buffer + numChannels, sizeof(ElemType) * numChannels);
            }

            void Clear()
            {
                if (mat) mat.reset();
            }

            ~MKLScaleShiftAdapter()
            {
                Clear();
            }
        };

        struct PrimitiveContext
        {
            MKLDnnResourceAdapter<ElemType> input;
            MKLDnnResourceAdapter<ElemType> output;
            MKLScaleShiftAdapter scaleShift;
            std::shared_ptr<Mat> varianceMat; // variance matrix used for converting InvStdDev

            dnnPrimitive_t primitive = nullptr;
            dnnPrimitiveAttributes_t attributes = nullptr;

            void Clear()
            {
                if (primitive) { dnnDelete<ElemType>(primitive); primitive = nullptr; }
                input.Clear();
                scaleShift.Clear();
                output.Clear();
                if (attributes) { dnnPrimitiveAttributesDestroy<ElemType>(attributes); attributes = nullptr; }
            }

            ~PrimitiveContext()
            {
                Clear();
            }
        } m_context[ContextIndex_Total];

        TensorShape m_shape;
        size_t m_numSamples;
        ElemType m_epsilon;

    public:
        MKLBatchNormalizationContext() :
            m_numSamples(0),
            m_epsilon(0)
        {
        }

        bool HasPreparedFor(ContextIndex contextIndex) const
        {
            return !!(m_contextFlags & (1 << contextIndex));
        }

        void Prepare(const TensorShape& shape, bool spatial, size_t numSamples, ContextIndex contextIndex, ElemType epsilon = 0)
        {
            int flag = (1 << contextIndex);
            if (contextIndex == ContextIndex_Backward)
            {
                epsilon = HasPreparedFor(ContextIndex_ForwardTrain) ? m_epsilon : (ElemType)DEFAULT_EPSILON;
            }

            bool same = (shape == m_shape) && (numSamples == m_numSamples) && (epsilon == m_epsilon);
            if (same && !!(m_contextFlags & flag)) return;

            if (!same)
                m_contextFlags = 0;

            if (m_contextFlags)
            {
                if ((m_numSamples != numSamples) || (m_epsilon != epsilon) || (m_shape != shape))
                    RuntimeError("MKLBatchNormalizationContext: Inconsistent num samples between forward and backward");
            }
            else
            {
                m_shape = shape;
                m_numSamples = numSamples;
                m_epsilon = epsilon;
            }
            m_contextFlags |= flag;

            const size_t inoutDim = 4;
            size_t rank = m_shape.GetRank();
            size_t numElements = m_shape.GetNumElements();
            size_t numChannels =
                spatial ?
                ((rank > 0) ? m_shape.GetDim(rank - 1) : 1) :
                numElements; // flatten all dims of a sample when non-spatial
            size_t numPixels = numElements / numChannels;
            size_t dimFirst = (rank > 1 && spatial) ? m_shape.GetDim(0) : 1;
            size_t dimSecond = numPixels / dimFirst;
            size_t inoutSizes[4] = { dimFirst, dimSecond, numChannels, m_numSamples };
            size_t inoutStrides[4] = { 1, dimFirst, numPixels, numElements };

            auto& ctx = m_context[contextIndex];
            ctx.Clear();

            dnnLayout_t ltUserInput, ltPrimInput;
            dnnLayout_t ltUserOutput, ltPrimOutput;
            dnnResourceType_t inputType;
            dnnResourceType_t outputType;
            dnnResourceType_t scaleShiftType;
            switch (contextIndex)
            {
            case ContextIndex_ForwardInfer:
            case ContextIndex_ForwardTrain:
                CHECK_MKL(dnnLayoutCreate<ElemType>(&ltUserInput, inoutDim, inoutSizes, inoutStrides));
                CHECK_MKL(dnnLayoutCreate<ElemType>(&ltUserOutput, inoutDim, inoutSizes, inoutStrides));
                CHECK_MKL(dnnPrimitiveAttributesCreate<ElemType>(&ctx.attributes));
                CHECK_MKL(dnnBatchNormalizationCreateForward_v2<ElemType>(
                    &ctx.primitive,
                    ctx.attributes,
                    ltUserInput,
                    m_epsilon,
                    dnnUseScaleShift | ((contextIndex == ContextIndex_ForwardInfer) ? dnnUseInputMeanVariance : 0)));
                inputType = dnnResourceSrc;
                outputType = dnnResourceDst;
                scaleShiftType = dnnResourceScaleShift;
                break;
            case ContextIndex_Backward:
                CHECK_MKL(dnnLayoutCreate<ElemType>(&ltUserInput, inoutDim, inoutSizes, inoutStrides));
                CHECK_MKL(dnnLayoutCreate<ElemType>(&ltUserOutput, inoutDim, inoutSizes, inoutStrides));
                CHECK_MKL(dnnPrimitiveAttributesCreate<ElemType>(&ctx.attributes));
                CHECK_MKL(dnnBatchNormalizationCreateBackward_v2<ElemType>(
                    &ctx.primitive,
                    ctx.attributes,
                    ltUserInput,
                    m_epsilon,
                    dnnUseScaleShift));
                inputType = dnnResourceDiffDst;
                outputType = dnnResourceDiffSrc;
                scaleShiftType = dnnResourceDiffScaleShift;
                ctx.varianceMat = std::make_shared<Mat>(numChannels, 1, CPUDEVICE);
                break;
            default:
                RuntimeError("Unexpected context type %d", (int)contextIndex);
            }

            CHECK_MKL(dnnLayoutCreateFromPrimitive<ElemType>(&ltPrimInput, ctx.primitive, inputType));
            ctx.input.Create(ltUserInput, ltPrimInput, inputType, true);

            CHECK_MKL(dnnLayoutCreateFromPrimitive<ElemType>(&ltPrimOutput, ctx.primitive, outputType));
            ctx.output.Create(ltUserOutput, ltPrimOutput, outputType, false);

            ctx.scaleShift.Create(scaleShiftType, contextIndex != ContextIndex_Backward, numChannels);
        }

        void Forward(void* input, void* output, void* scale, void* bias, void* runMean, void* runVariance, ContextIndex contextIndex)
        {
            auto& ctx = m_context[contextIndex];
            void* resources[dnnResourceNumber] = { 0 };

            ctx.input.PrepareForExecution(input, resources);
            ctx.output.PrepareForExecution(output, resources);
            ctx.scaleShift.PrepareForExecution(scale, bias, resources);

            resources[dnnResourceMean] = runMean;
            resources[dnnResourceVariance] = runVariance;

            CHECK_MKL(dnnExecute<ElemType>(ctx.primitive, resources));

            ctx.output.ConvertOutput(output);
        }

        void Backward(void* in, void* srcGrad, void* grad, void* scale, void* savedMean, void* savedInvStdDev, void* scaleGrad, void* biasGrad)
        {
            auto& ctx = m_context[ContextIndex_Backward];
            void* resources[dnnResourceNumber] = { 0 };

            ctx.input.PrepareForExecution(srcGrad, resources);
            ctx.output.PrepareForExecution(grad, resources);
            ctx.scaleShift.PrepareForExecution(scaleGrad, biasGrad, resources);

            std::shared_ptr<Mat> scaleShiftMat;
            scaleShiftMat = std::make_shared<Mat>(ctx.scaleShift.numChannels, 2, CPUDEVICE);
            memcpy(scaleShiftMat->Data(), scale, ctx.scaleShift.numChannels * sizeof(ElemType));
            resources[dnnResourceScaleShift] = scaleShiftMat->Data();

            // convert from InvStdDev to variance
            for (size_t i = 0; i < ctx.scaleShift.numChannels; i++)
            {
                ElemType& v = ctx.varianceMat->Data()[i];
                ElemType& s = ((ElemType*)savedInvStdDev)[i];
                v = (1 / (s * s) - m_epsilon);
            }

            resources[dnnResourceSrc] = in;
            resources[dnnResourceMean] = savedMean;
            resources[dnnResourceVariance] = ctx.varianceMat->Data();

            CHECK_MKL(dnnExecute<ElemType>(ctx.primitive, resources));

            ctx.output.ConvertOutput(grad);
            ctx.scaleShift.ConvertOutput(scaleGrad, biasGrad);
        }
    };

    MKLBatchNormalizationContext m_mklContext;
    std::shared_ptr<Mat> m_dataGradWorkspace;

    bool ForwardCoreMKL(const Mat& in, const Mat& scale, const Mat& bias, bool inferenceOnly, double expAvgFactor, Mat& runMean, Mat& runVariance,
        Mat& out, double epsilon, Mat& savedMean, Mat& savedInvStdDev)
    {
        ContextIndex contextIndex = inferenceOnly ?
            ContextIndex_ForwardInfer :
            ContextIndex_ForwardTrain;
        m_mklContext.Prepare(m_inOutT, m_spatial, in.GetNumCols(), contextIndex, (ElemType)epsilon);

        if (inferenceOnly)
        {
            m_mklContext.Forward(in.Data(), out.Data(), scale.Data(), bias.Data(), runMean.Data(), runVariance.Data(), contextIndex);
        }
        else
        {
            savedMean.Resize(runMean);
            savedInvStdDev.Resize(runVariance);
            m_mklContext.Forward(in.Data(), out.Data(), scale.Data(), bias.Data(), savedMean.Data(), savedInvStdDev.Data(), contextIndex);

            // update savedMean, savedInvStdDev
            ElemType OneMinusExpAvgFactor = (ElemType)(1.0 - expAvgFactor);
            cblas_axpby((MKL_INT)runMean.GetNumElements(), (ElemType)expAvgFactor, savedMean.Data(), OneMinusExpAvgFactor, runMean.Data());

            // note savedInvStdDev currently hold variance of in.Data(), need to convert to InvStdDev and interpolate
            ElemType numReduced = (ElemType)(in.GetNumElements() / runVariance.GetNumElements());
            ElemType bcf = numReduced / (numReduced - 1);
            for (size_t i = 0; i < runVariance.GetNumElements(); i++)
            {
                ElemType& v = runVariance.Data()[i];
                ElemType& s = savedInvStdDev.Data()[i];
                v = v * OneMinusExpAvgFactor + bcf * s * (ElemType)expAvgFactor;
                s = (ElemType)1 / sqrt(s + (ElemType)epsilon);
            }
        }

        return true;
    }

    bool BackwardCoreMKL(const Mat& in, const Mat& srcGrad, Mat& grad, const Mat& scale,
        const Mat& savedMean, const Mat& savedInvStdDev, Mat& scaleGrad, Mat& biasGrad, bool accumulateDataGrad)
    {
        m_mklContext.Prepare(m_inOutT, m_spatial, srcGrad.GetNumCols(), ContextIndex_Backward);

        if (accumulateDataGrad)
        {
            if (!m_dataGradWorkspace)
                m_dataGradWorkspace = std::make_shared<Matrix<ElemType>>(0, 0, CPUDEVICE);

            m_dataGradWorkspace->SetValue(grad);
        }

        m_mklContext.Backward(in.Data(), srcGrad.Data(), grad.Data(), scale.Data(), savedMean.Data(), savedInvStdDev.Data(), scaleGrad.Data(), biasGrad.Data());

        if (accumulateDataGrad)
            cblas_axpby((MKL_INT)grad.GetNumElements(), (ElemType)1.0, m_dataGradWorkspace->Data(), (ElemType)1.0, grad.Data());

        return true;
    }
#endif
};

template class CntkBatchNormEngine<float>;
template class CntkBatchNormEngine<double>;

template <typename T> bool HasFlag(T src, T testFlag)
{
    return ((int)src & (int)testFlag) != 0;
}

template <class ElemType>
std::unique_ptr<BatchNormEngine<ElemType>> BatchNormEngine<ElemType>::Create(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                                                                             bool spatial, ImageLayoutKind imageLayout,
                                                                             BatchNormEngineKind enabledEngines)
{
    // Use CNTK as default batch norm engine.
    if (HasFlag(enabledEngines, BatchNormEngineKind::Cntk))
    {
        if (GetMathLibTraceLevel() > 0)
            fprintf(stderr, "Using CNTK batch normalization engine.\n");

        return std::make_unique<CntkBatchNormEngine<ElemType>>(deviceId, inOutT, spatial, imageLayout);
    }

    if (HasFlag(enabledEngines, BatchNormEngineKind::CuDnn))
    {
        if (GetMathLibTraceLevel() > 0)
            fprintf(stderr, "Using cuDNN batch normalization engine.\n");

        return CuDnnBatchNormEngineFactory<ElemType>::Create(deviceId, inOutT, spatial, imageLayout);
    }

    RuntimeError("Could not find appropriate batch normalization engine.");
}

template class BatchNormEngine<float>;
template class BatchNormEngine<double>;

}}}
