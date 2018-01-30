//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "BatchNormalizationEngine.h"
#include "CuDnnFactories.h"
#include "MklDnnCommon.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <class InoutType, class StatType>
void BatchNormEngine<InoutType, StatType>::Forward(const InoutMat& in, const StatMat& scale, const StatMat& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, StatMat& runMean, StatMat& runVariance,
                                        InoutMat& out, double epsilon, StatMat& savedMean, StatMat& savedInvStdDev)
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

template <class InoutType, class StatType>
void BatchNormEngine<InoutType, StatType>::Backward(const InoutMat& in, const InoutMat& srcGrad, InoutMat& grad, const StatMat& scale, double blendFactor,
                                         const StatMat& savedMean, const StatMat& savedInvStdDev, StatMat& scaleGrad, StatMat& biasGrad, bool accumulateDataGrad)
{
    assert(!savedMean.IsEmpty());
    assert(!savedInvStdDev.IsEmpty());
    EnsureCompatible();
    BackwardCore(in, srcGrad, grad, scale, blendFactor, savedMean, savedInvStdDev, scaleGrad, biasGrad, accumulateDataGrad);
}

template <class InoutType, class StatType>
class CntkBatchNormEngine : public BatchNormEngine<InoutType, StatType>
{
public:
    using Base = BatchNormEngine<InoutType, StatType>;
    using typename Base::InoutMat;
    using typename Base::StatMat;

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

    void ForwardCore(const InoutMat& in, const StatMat& scale, const StatMat& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, StatMat& runMean, StatMat& runVariance,
                     InoutMat& out, double epsilon, StatMat& savedMean, StatMat& savedInvStdDev) override
    {
#ifdef USE_MKL2017DNN
        if (in.GetCurrentMatrixLocation() == CPU &&
            std::is_same<InoutType, StatType>::value &&
            ForwardCoreMKL(*(const StatMat*)&in, scale, bias, inferenceOnly, expAvgFactor, runMean, runVariance, *(StatMat*)&out, epsilon, savedMean, savedInvStdDev))
            return;
#endif

        in.BatchNormalizationForward(scale, bias, inferenceOnly, expAvgFactor, blendFactor, runMean, runVariance, out, epsilon, savedMean, savedInvStdDev);
    }

    void BackwardCore(const InoutMat& in, const InoutMat& srcGrad, InoutMat& grad, const StatMat& scale, double blendFactor, const StatMat& savedMean, const StatMat& savedInvStdDev,
                      StatMat& scaleGrad, StatMat& biasGrad, bool accumulateDataGrad) override
    {
#ifdef USE_MKL2017DNN
        if (srcGrad.GetCurrentMatrixLocation() == CPU &&
            std::is_same<InoutType, StatType>::value &&
            BackwardCoreMKL(*(const StatMat*)&in, *(const StatMat*)&srcGrad, *(StatMat*)&grad, scale, savedMean, savedInvStdDev, scaleGrad, biasGrad, accumulateDataGrad))
            return;
#endif
        if (!accumulateDataGrad)
            grad.SetValue((InoutType)0);

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
            std::shared_ptr<Matrix<StatType>> mat;
            dnnResourceType_t resourceType;
            size_t numChannels;

            void Create(dnnResourceType_t rt, bool userToPrim, size_t n)
            {
                Clear();
                numChannels = n;
                mat = std::make_shared<Matrix<StatType>>(numChannels, 2, CPUDEVICE);
                isInput = userToPrim;
                resourceType = rt;
            }

            void PrepareForExecution(void* scale, void* bias, void* resources[dnnResourceNumber])
            {
                StatType* buffer = mat->Data();
                resources[resourceType] = buffer;
                if (isInput)
                {
                    memcpy(buffer, scale, sizeof(StatType) * numChannels);
                    memcpy(buffer + numChannels, bias, sizeof(StatType) * numChannels);
                }
            }

            void ConvertOutput(void* scale, void* bias)
            {
                if (isInput)
                    RuntimeError("Cannot execute output ResourceAdapter for input");

                StatType* buffer = mat->Data();
                memcpy(scale, buffer, sizeof(StatType) * numChannels);
                memcpy(bias, buffer + numChannels, sizeof(StatType) * numChannels);
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
            MKLDnnResourceAdapter<StatType> input;
            MKLDnnResourceAdapter<StatType> output;
            MKLScaleShiftAdapter scaleShift;
            std::shared_ptr<StatMat> varianceMat; // variance matrix used for converting InvStdDev

            dnnPrimitive_t primitive = nullptr;
            dnnPrimitiveAttributes_t attributes = nullptr;

            void Clear()
            {
                if (primitive) { dnnDelete<StatType>(primitive); primitive = nullptr; }
                input.Clear();
                scaleShift.Clear();
                output.Clear();
                if (attributes) { dnnPrimitiveAttributesDestroy<StatType>(attributes); attributes = nullptr; }
            }

            ~PrimitiveContext()
            {
                Clear();
            }
        } m_context[ContextIndex_Total];

        TensorShape m_shape;
        size_t m_numSamples;
        StatType m_epsilon;

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

        void Prepare(const TensorShape& shape, bool spatial, size_t numSamples, ContextIndex contextIndex, StatType epsilon = 0)
        {
            int flag = (1 << contextIndex);
            if (contextIndex == ContextIndex_Backward)
            {
                epsilon = HasPreparedFor(ContextIndex_ForwardTrain) ? m_epsilon : (StatType)DEFAULT_EPSILON;
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
                CHECK_MKL(dnnLayoutCreate<StatType>(&ltUserInput, inoutDim, inoutSizes, inoutStrides));
                CHECK_MKL(dnnLayoutCreate<StatType>(&ltUserOutput, inoutDim, inoutSizes, inoutStrides));
                CHECK_MKL(dnnPrimitiveAttributesCreate<StatType>(&ctx.attributes));
                CHECK_MKL(dnnBatchNormalizationCreateForward_v2<StatType>(
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
                CHECK_MKL(dnnLayoutCreate<StatType>(&ltUserInput, inoutDim, inoutSizes, inoutStrides));
                CHECK_MKL(dnnLayoutCreate<StatType>(&ltUserOutput, inoutDim, inoutSizes, inoutStrides));
                CHECK_MKL(dnnPrimitiveAttributesCreate<StatType>(&ctx.attributes));
                CHECK_MKL(dnnBatchNormalizationCreateBackward_v2<StatType>(
                    &ctx.primitive,
                    ctx.attributes,
                    ltUserInput,
                    m_epsilon,
                    dnnUseScaleShift));
                inputType = dnnResourceDiffDst;
                outputType = dnnResourceDiffSrc;
                scaleShiftType = dnnResourceDiffScaleShift;
                ctx.varianceMat = std::make_shared<StatMat>(numChannels, 1, CPUDEVICE);
                break;
            default:
                RuntimeError("Unexpected context type %d", (int)contextIndex);
            }

            CHECK_MKL(dnnLayoutCreateFromPrimitive<StatType>(&ltPrimInput, ctx.primitive, inputType));
            ctx.input.Create(ltUserInput, ltPrimInput, inputType, true);

            CHECK_MKL(dnnLayoutCreateFromPrimitive<StatType>(&ltPrimOutput, ctx.primitive, outputType));
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

            CHECK_MKL(dnnExecute<StatType>(ctx.primitive, resources));

            ctx.output.ConvertOutput(output);
        }

        void Backward(void* in, void* srcGrad, void* grad, void* scale, void* savedMean, void* savedInvStdDev, void* scaleGrad, void* biasGrad)
        {
            auto& ctx = m_context[ContextIndex_Backward];
            void* resources[dnnResourceNumber] = { 0 };

            ctx.input.PrepareForExecution(srcGrad, resources);
            ctx.output.PrepareForExecution(grad, resources);
            ctx.scaleShift.PrepareForExecution(scaleGrad, biasGrad, resources);

            std::shared_ptr<StatMat> scaleShiftMat;
            scaleShiftMat = std::make_shared<StatMat>(ctx.scaleShift.numChannels, 2, CPUDEVICE);
            memcpy(scaleShiftMat->Data(), scale, ctx.scaleShift.numChannels * sizeof(StatType));
            resources[dnnResourceScaleShift] = scaleShiftMat->Data();

            // convert from InvStdDev to variance
            for (size_t i = 0; i < ctx.scaleShift.numChannels; i++)
            {
                StatType& v = ctx.varianceMat->Data()[i];
                StatType& s = ((StatType*)savedInvStdDev)[i];
                v = (1 / (s * s) - m_epsilon);
            }

            resources[dnnResourceSrc] = in;
            resources[dnnResourceMean] = savedMean;
            resources[dnnResourceVariance] = ctx.varianceMat->Data();

            CHECK_MKL(dnnExecute<StatType>(ctx.primitive, resources));

            ctx.output.ConvertOutput(grad);
            ctx.scaleShift.ConvertOutput(scaleGrad, biasGrad);
        }
    };

    MKLBatchNormalizationContext m_mklContext;
    std::shared_ptr<StatMat> m_dataGradWorkspace;

    bool ForwardCoreMKL(const StatMat& in, const StatMat& scale, const StatMat& bias, bool inferenceOnly, double expAvgFactor, StatMat& runMean, StatMat& runVariance,
        StatMat& out, double epsilon, StatMat& savedMean, StatMat& savedInvStdDev)
    {
        ContextIndex contextIndex = inferenceOnly ?
            ContextIndex_ForwardInfer :
            ContextIndex_ForwardTrain;
        m_mklContext.Prepare(m_inOutT, m_spatial, in.GetNumCols(), contextIndex, (StatType)epsilon);

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
            StatType OneMinusExpAvgFactor = (StatType)(1.0 - expAvgFactor);
            cblas_axpby((MKL_INT)runMean.GetNumElements(), (StatType)expAvgFactor, savedMean.Data(), OneMinusExpAvgFactor, runMean.Data());

            // note savedInvStdDev currently hold variance of in.Data(), need to convert to InvStdDev and interpolate
            StatType numReduced = (StatType)(in.GetNumElements() / runVariance.GetNumElements());
            StatType bcf = numReduced / (numReduced - 1);
            for (size_t i = 0; i < runVariance.GetNumElements(); i++)
            {
                StatType& v = runVariance.Data()[i];
                StatType& s = savedInvStdDev.Data()[i];
                v = v * OneMinusExpAvgFactor + bcf * s * (StatType)expAvgFactor;
                s = (StatType)1 / sqrt(s + (StatType)epsilon);
            }
        }

        return true;
    }

    bool BackwardCoreMKL(const StatMat& in, const StatMat& srcGrad, StatMat& grad, const StatMat& scale,
        const StatMat& savedMean, const StatMat& savedInvStdDev, StatMat& scaleGrad, StatMat& biasGrad, bool accumulateDataGrad)
    {
        m_mklContext.Prepare(m_inOutT, m_spatial, srcGrad.GetNumCols(), ContextIndex_Backward);

        if (accumulateDataGrad)
        {
            if (!m_dataGradWorkspace)
                m_dataGradWorkspace = std::make_shared<Matrix<StatType>>(0, 0, CPUDEVICE);

            m_dataGradWorkspace->SetValue(grad);
        }

        m_mklContext.Backward(in.Data(), srcGrad.Data(), grad.Data(), scale.Data(), savedMean.Data(), savedInvStdDev.Data(), scaleGrad.Data(), biasGrad.Data());

        if (accumulateDataGrad)
            cblas_axpby((MKL_INT)grad.GetNumElements(), (StatType)1.0, m_dataGradWorkspace->Data(), (StatType)1.0, grad.Data());

        return true;
    }
#endif
};

template class CntkBatchNormEngine<float, float>;
template class CntkBatchNormEngine<double, double>;
template class CntkBatchNormEngine<half, float>;

template <typename T> bool HasFlag(T src, T testFlag)
{
    return ((int)src & (int)testFlag) != 0;
}

template <class InoutType, class StatType>
std::unique_ptr<BatchNormEngine<InoutType, StatType>> BatchNormEngine<InoutType, StatType>::Create(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
                                                                             bool spatial, ImageLayoutKind imageLayout,
                                                                             BatchNormEngineKind enabledEngines)
{
    // Use CNTK as default batch norm engine.
    if (HasFlag(enabledEngines, BatchNormEngineKind::Cntk))
    {
        if (GetMathLibTraceLevel() > 0)
            fprintf(stderr, "Using CNTK batch normalization engine.\n");

        return std::make_unique<CntkBatchNormEngine<InoutType, StatType>>(deviceId, inOutT, spatial, imageLayout);
    }

    if (HasFlag(enabledEngines, BatchNormEngineKind::CuDnn))
    {
        if (GetMathLibTraceLevel() > 0)
            fprintf(stderr, "Using cuDNN batch normalization engine.\n");

        return CuDnnBatchNormEngineFactory<InoutType, StatType>::Create(deviceId, inOutT, spatial, imageLayout);
    }

    RuntimeError("Could not find appropriate batch normalization engine.");
}

template class BatchNormEngine<float, float>;
template class BatchNormEngine<double, double>;
template class BatchNormEngine<half, float>;

}}}
