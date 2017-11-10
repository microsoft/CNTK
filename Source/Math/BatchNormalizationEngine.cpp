//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "BatchNormalizationEngine.h"
#include "CuDnnFactories.h"
#include "mkl_dnn.h"

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
        in.BatchNormalizationForward(scale, bias, inferenceOnly, expAvgFactor, blendFactor, runMean, runVariance, out, epsilon, savedMean, savedInvStdDev);
    }

    void BackwardCore(const Mat& in, const Mat& srcGrad, Mat& grad, const Mat& scale, double blendFactor, const Mat& savedMean, const Mat& savedInvStdDev,
                      Mat& scaleGrad, Mat& biasGrad, bool accumulateDataGrad) override
    {
        if (!accumulateDataGrad)
            grad.SetValue((ElemType)0);

        srcGrad.BatchNormalizationBackward(in, grad, scale, blendFactor, savedMean, savedInvStdDev, scaleGrad, biasGrad);
    }
};

template class CntkBatchNormEngine<float>;
template class CntkBatchNormEngine<double>;

template <class ElemType>
class MKL2017BatchNormEngine : public BatchNormEngine<ElemType>
{
public:
    using Base = BatchNormEngine<ElemType>;
    using typename Base::Mat;
    using Base::m_inOutT;

public:
    MKL2017BatchNormEngine(DEVICEID_TYPE deviceId, const TensorShape& inOutT,
							bool spatial, ImageLayoutKind imageLayout)
							: Base(deviceId, inOutT, spatial, imageLayout)
    {
    }

    ~MKL2017BatchNormEngine()
    {
        if (pBatchNormalizationFwd != NULL) {
            dnnDelete_F32(pBatchNormalizationFwd);
            pBatchNormalizationFwd = NULL;
        }

        delete[] inputSize;
        delete[] inputStrides;
        delete[] outputSize;
        delete[] outputStrides;

        if (lt_batchNorm_fwd_output != NULL) {
            dnnLayoutDelete_F32(lt_batchNorm_fwd_output);
            lt_batchNorm_fwd_output = NULL;
        }
        if (lt_batchNorm_fwd_input != NULL) {
            dnnLayoutDelete_F32(lt_batchNorm_fwd_input);
            lt_batchNorm_fwd_input = NULL;
        }
        if (lt_user_input != NULL) {
            dnnLayoutDelete_F32(lt_user_input);
            lt_user_input = NULL;
        }
        if (lt_user_output != NULL) {
            dnnLayoutDelete_F32(lt_user_output);
            lt_user_output = NULL;
        }
        if (scaleShift_buffer_l != NULL) {
            dnnLayoutDelete_F32(scaleShift_buffer_l);
            dnnReleaseBuffer_F32(scaleShift_buffer_);
            scaleShift_buffer_ = NULL;
            scaleShift_buffer_l = NULL;
        }

        if (lt_batchNorm_fwd_workspace != NULL) {
            dnnLayoutDelete_F32(lt_batchNorm_fwd_workspace);
            dnnReleaseBuffer_F32(workspace_buffer);
            workspace_buffer = NULL;
            lt_batchNorm_fwd_workspace = NULL;
        }
        if (cv_user_input_to_batchNorm_fwd_input != NULL)
        {
            dnnReleaseBuffer_F32(in_buffer);
            in_buffer = NULL;
            dnnDelete_F32(cv_user_input_to_batchNorm_fwd_input);
            cv_user_input_to_batchNorm_fwd_input = NULL;
        }
        if (batchNorm_fwd_output_to_user_output != NULL)
        {
            dnnReleaseBuffer_F32(out_buffer);
            out_buffer = NULL;
            dnnDelete_F32(batchNorm_fwd_output_to_user_output);
            batchNorm_fwd_output_to_user_output = NULL;
        }
    }

protected:
    using Base::m_deviceId;
    using Base::m_imageLayout;
    using Base::m_spatial;
    size_t* inputSize;
    size_t* inputStrides;
    size_t* outputSize;
    size_t* outputStrides;

    dnnPrimitive_t pBatchNormalizationFwd = NULL;
    dnnPrimitive_t pBatchNormalizationFwdTrain = NULL;
    dnnPrimitive_t cv_user_input_to_batchNorm_fwd_input = NULL;
    dnnPrimitive_t batchNorm_fwd_output_to_user_output = NULL;
    dnnPrimitive_t batchNormBwdScaleShift = NULL;
    dnnLayout_t lt_batchNorm_fwd_input;
    dnnLayout_t lt_batchNorm_fwd_output;
    dnnLayout_t lt_user_input;
    dnnLayout_t lt_user_output;
    dnnLayout_t lt_batchNorm_fwd_workspace;
    dnnLayout_t scaleShift_buffer_l = NULL;
    dnnLayout_t lt_user_input_scaleshift = nullptr;

    void *BatchNorm_res[dnnResourceNumber] = { 0 };
    void *workspace_buffer = NULL;
    void *user_input = NULL;
    void *user_output = NULL;
    void *scaleShift_buffer_;
    void *out_buffer;
    void *in_buffer;
	
	//For backprop 
	dnnPrimitive_t pBatchNormalizationBwd = NULL;
	dnnLayout_t lt_batchNorm_bwd_input;
	dnnLayout_t lt_batchNorm_bwd_outgrad;
	dnnLayout_t lt_batchNorm_bwd_scaleShift;
	dnnLayout_t lt_batchNorm_bwd_scaleShift_grad;
	dnnPrimitive_t cv_user_input_to_batchNorm_bwd_input = NULL;
	dnnPrimitive_t batchNorm_bwd_outgrad_to_user_output = NULL;

	void *outgrad_buffer;
	void *user_in_grad = NULL;
	void *user_out_grad = NULL;
	void *scaleShift_buffer;
	void *scaleShift_grad_buffer;
	float* saved_mean_buf;
	float* saved_Std_Dev;
	bool inference;
	double AvgFactor;
	double blendTimeConstant;

    void EnsureCompatible() override
    {

    }

    void ForwardCore(const Mat& in, const Mat& scale, const Mat& bias, bool inferenceOnly, double expAvgFactor, double blendFactor, Mat& runMean, Mat& runVariance,
        Mat& out, double epsilon, Mat& savedMean, Mat& savedInvStdDev) override
    {
        bool spatial = in.GetNumRows() != scale.GetNumRows();

        if (!spatial) {
            fprintf(stderr, "WARNING: MKL2017 Batch normalization does not support spatial == False. Using reference implementation now. \n");
            in.BatchNormalizationForward(scale, bias, inferenceOnly, expAvgFactor, blendFactor, runMean, runVariance, out, epsilon, savedMean, savedInvStdDev);
            return;
        }

        if (m_inOutT.size() != 3) {
            fprintf(stderr, "WARNING: MKL2017 Batch normalization supports 3D input tensors only. Using reference implementation now for %dD input tensor \n", (int)m_inOutT.size());
            in.BatchNormalizationForward(scale, bias, inferenceOnly, expAvgFactor, blendFactor, runMean, runVariance, out, epsilon, savedMean, savedInvStdDev);
            return;
        }

        float eps = static_cast<float>(epsilon);

        size_t n = in.GetNumCols(); //batch_size
        size_t iw = m_inOutT[0];    //input width
        size_t ih = m_inOutT[1];    //input height
        size_t ic = m_inOutT[2];    //input num channels                
        size_t dim = 4;
        outputSize = new size_t[dim];
        outputStrides = new size_t[dim];
        inputSize = new size_t[dim];
        inputStrides = new size_t[dim];
        user_input = in.Data();
        user_output = out.Data();
        inference = inferenceOnly;
        AvgFactor = expAvgFactor;
        blendTimeConstant = blendFactor;
        saved_mean_buf = reinterpret_cast<float*>(savedMean.Data());
        saved_Std_Dev = reinterpret_cast<float*>(savedInvStdDev.Data());

        float* buf_scale_shift = NULL;

        if (pBatchNormalizationFwd == NULL) {

            /*Create input strides for the user input layout*/
            inputSize[0] = iw;
            inputSize[1] = ih;
            inputSize[2] = ic;
            inputSize[3] = n;

            inputStrides[0] = 1;
            inputStrides[1] = iw;
            inputStrides[2] = iw * ih;
            inputStrides[3] = iw * ih * ic;

            assert(dnnLayoutCreate_F32(&lt_user_input, dim, inputSize, inputStrides) == E_SUCCESS);

            outputSize[0] = iw;
            outputSize[1] = ih;
            outputSize[2] = ic;
            outputSize[3] = n;

            outputStrides[0] = 1;
            outputStrides[1] = iw;
            outputStrides[2] = iw * ih;
            outputStrides[3] = iw * ih * ic;

            dnnPrimitiveAttributes_t primAttr = NULL;

            assert(dnnLayoutCreate_F32(&lt_user_output, dim, outputSize, outputStrides) == E_SUCCESS);
            assert(dnnBatchNormalizationCreateForward_v2_F32(&pBatchNormalizationFwd, primAttr, lt_user_input, eps, dnnUseInputMeanVariance | dnnUseScaleShift) == E_SUCCESS);
            assert(dnnBatchNormalizationCreateForward_v2_F32(&pBatchNormalizationFwdTrain, primAttr, lt_user_input, eps, dnnUseScaleShift) == E_SUCCESS);
            assert(dnnLayoutCreateFromPrimitive_F32(&lt_batchNorm_fwd_input, pBatchNormalizationFwd, dnnResourceSrc) == E_SUCCESS);

            if (!dnnLayoutCompare_F32(lt_batchNorm_fwd_input, lt_user_input)) {
                assert(dnnConversionCreate_F32(&cv_user_input_to_batchNorm_fwd_input, lt_user_input, lt_batchNorm_fwd_input) == E_SUCCESS);
                assert(dnnAllocateBuffer_F32(&in_buffer, lt_batchNorm_fwd_input) == E_SUCCESS);
                assert(dnnConversionExecute_F32(cv_user_input_to_batchNorm_fwd_input, user_input, in_buffer) == E_SUCCESS);
            }

            assert(dnnLayoutCreateFromPrimitive_F32(&scaleShift_buffer_l, pBatchNormalizationFwd, dnnResourceScaleShift) == E_SUCCESS);
            assert(dnnAllocateBuffer_F32(&scaleShift_buffer_, scaleShift_buffer_l) == E_SUCCESS);

            buf_scale_shift = static_cast<float*>(scaleShift_buffer_);

            for (int i = 0; i < ic; i++) {
                buf_scale_shift[i] = 1.0;
                buf_scale_shift[i + ic] = 0.0;
            }

            assert(dnnLayoutCreateFromPrimitive_F32(&lt_batchNorm_fwd_output, pBatchNormalizationFwd, dnnResourceDst) == E_SUCCESS);
            assert(dnnAllocateBuffer_F32((void**)&BatchNorm_res[dnnResourceDst], lt_user_output) == E_SUCCESS);

            if (!dnnLayoutCompare_F32(lt_batchNorm_fwd_output, lt_user_output)) {
                assert(dnnConversionCreate_F32(&batchNorm_fwd_output_to_user_output, lt_batchNorm_fwd_output, lt_user_output) == E_SUCCESS);
                assert(dnnAllocateBuffer_F32(&out_buffer, lt_batchNorm_fwd_output) == E_SUCCESS);
            }
        }

        for (int i = 0; i < ic; i++) {
            buf_scale_shift[i] = reinterpret_cast<float*>(scale.Data())[i];
            buf_scale_shift[i + ic] = reinterpret_cast<float*>(bias.Data())[i];
        }

        if (!dnnLayoutCompare_F32(lt_batchNorm_fwd_input, lt_user_input)) {
            BatchNorm_res[dnnResourceSrc] = (void*)in_buffer;
        }
        else {
            BatchNorm_res[dnnResourceSrc] = user_input;
        }

        BatchNorm_res[dnnResourceScaleShift] = scaleShift_buffer_;

        if (!dnnLayoutCompare_F32(lt_batchNorm_fwd_output, lt_user_output)) {
            BatchNorm_res[dnnResourceDst] = out_buffer;
        }
        else {
            BatchNorm_res[dnnResourceDst] = user_output;
        }

        if (inference) {
            BatchNorm_res[dnnResourceMean] = runMean.Data();
            BatchNorm_res[dnnResourceVariance] = runVariance.Data();
            assert(dnnExecute_F32(pBatchNormalizationFwd, BatchNorm_res) == E_SUCCESS);
        }
        else {
            BatchNorm_res[dnnResourceMean] = savedMean.Data();
            BatchNorm_res[dnnResourceVariance] = savedInvStdDev.Data();
            assert(dnnExecute_F32(pBatchNormalizationFwdTrain, BatchNorm_res) == E_SUCCESS);
        }

        if (!dnnLayoutCompare_F32(lt_batchNorm_fwd_output, lt_user_output)) {
            assert(dnnConversionExecute_F32(batchNorm_fwd_output_to_user_output, &BatchNorm_res[dnnResourceDst], out.Data()) == E_SUCCESS);
        }

    }
	void BackwardCore(const Mat& in, const Mat& srcGrad, Mat& grad, const Mat& scale, double blendFactor, const Mat& savedMean, const Mat& savedInvStdDev,
		Mat& scaleGrad, Mat& biasGrad, bool accumulateDataGrad) override
	{
		bool spatial = in.GetNumRows() != scale.GetNumRows();

		if (!spatial) {
			fprintf(stderr, "WARNING: MKL2017 Batch normalization does not support spatial == False. Using reference implementation now. \n");

			if (!accumulateDataGrad)
				grad.SetValue((ElemType)0);

			srcGrad.BatchNormalizationBackward(in, grad, scale, blendFactor, savedMean, savedInvStdDev, scaleGrad, biasGrad);
			return;
		}

		if (m_inOutT.size() != 3) {
			fprintf(stderr, "WARNING: MKL2017 Batch normalization supports 3D input tensors only. Using reference implementation now for %dD input tensor \n", (int)m_inOutT.size());

			if (!accumulateDataGrad)
				grad.SetValue((ElemType)0);

			srcGrad.BatchNormalizationBackward(in, grad, scale, blendFactor, savedMean, savedInvStdDev, scaleGrad, biasGrad);
			return;
		}

		//TODO: Pass eps from forward pass
		float eps = 1e-5;

		size_t n = in.GetNumCols(); //batch_size
		size_t iw = m_inOutT[0];    //input width
		size_t ih = m_inOutT[1];    //input height
		size_t ic = m_inOutT[2];    //input num channels
		size_t dim = 4;
		outputSize = new size_t[dim];
		outputStrides = new size_t[dim];
		inputSize = new size_t[dim];
		inputStrides = new size_t[dim];
		user_input = in.Data();
		user_in_grad = srcGrad.Data();
		user_out_grad = grad.Data();
		blendTimeConstant = blendFactor;
		saved_mean_buf = reinterpret_cast<float*>(savedMean.Data());
		saved_Std_Dev = reinterpret_cast<float*>(savedInvStdDev.Data());

		if (pBatchNormalizationBwd == NULL) {
			/*Create input strides for the user input layout*/
			inputSize[0] = iw;
			inputSize[1] = ih;
			inputSize[2] = ic;
			inputSize[3] = n;

			inputStrides[0] = 1;
			inputStrides[1] = iw;
			inputStrides[2] = iw * ih;
			inputStrides[3] = iw * ih * ic;

			assert(dnnLayoutCreate_F32(&lt_user_input, dim, inputSize, inputStrides) == E_SUCCESS);

			outputSize[0] = iw;
			outputSize[1] = ih;
			outputSize[2] = ic;
			outputSize[3] = n;

			outputStrides[0] = 1;
			outputStrides[1] = iw;
			outputStrides[2] = iw * ih;
			outputStrides[3] = iw * ih * ic;

			assert(dnnLayoutCreate_F32(&lt_user_output, dim, outputSize, outputStrides) == E_SUCCESS);
			dnnPrimitiveAttributes_t primAttr = NULL;

			//assert(dnnBatchNormalizationCreateBackward_v2_F32(&pBatchNormalizationBwd, primAttr, lt_user_input, eps, dnnUseInputMeanVariance | dnnUseScaleShift) == E_SUCCESS);
			assert(dnnBatchNormalizationCreateBackward_v2_F32(&pBatchNormalizationBwd, primAttr, lt_user_input, eps, dnnUseScaleShift) == E_SUCCESS);

			//Compare input layout with primitive layout
			assert(dnnLayoutCreateFromPrimitive_F32(&lt_batchNorm_bwd_input, pBatchNormalizationBwd, dnnResourceSrc) == E_SUCCESS);

			if (!dnnLayoutCompare_F32(lt_batchNorm_bwd_input, lt_user_input)) {
				assert(dnnConversionCreate_F32(&cv_user_input_to_batchNorm_bwd_input, lt_user_input, lt_batchNorm_bwd_input) == E_SUCCESS);
				assert(dnnAllocateBuffer_F32(&in_buffer, lt_batchNorm_bwd_input) == E_SUCCESS);
				assert(dnnConversionExecute_F32(cv_user_input_to_batchNorm_bwd_input, user_input, in_buffer) == E_SUCCESS);
			}

			//Compare out grad layout with primitive layout
			assert(dnnLayoutCreateFromPrimitive_F32(&lt_batchNorm_bwd_outgrad, pBatchNormalizationBwd, dnnResourceDiffDst) == E_SUCCESS);

			if (!dnnLayoutCompare_F32(lt_batchNorm_bwd_outgrad, lt_user_output)) {
				assert(dnnConversionCreate_F32(&batchNorm_bwd_outgrad_to_user_output, lt_user_output, lt_batchNorm_bwd_outgrad) == E_SUCCESS);
				assert(dnnAllocateBuffer_F32(&outgrad_buffer, lt_batchNorm_bwd_outgrad) == E_SUCCESS);
				assert(dnnConversionExecute_F32(batchNorm_bwd_outgrad_to_user_output, user_out_grad, outgrad_buffer) == E_SUCCESS);
			}

			//Initialize scaleshift buffer from primitive
			assert(dnnLayoutCreateFromPrimitive_F32(&lt_batchNorm_bwd_scaleShift, pBatchNormalizationBwd, dnnResourceScaleShift) == E_SUCCESS);
			assert(dnnAllocateBuffer_F32(&scaleShift_buffer, lt_batchNorm_bwd_scaleShift) == E_SUCCESS);

			//Initialize scaleshift grad buffer from primitive			
			assert(dnnLayoutCreateFromPrimitive_F32(&lt_batchNorm_bwd_scaleShift_grad, pBatchNormalizationBwd, dnnResourceDiffScaleShift) == E_SUCCESS);
			assert(dnnAllocateBuffer_F32(&scaleShift_grad_buffer, lt_batchNorm_bwd_scaleShift_grad) == E_SUCCESS);
		}

		if (!dnnLayoutCompare_F32(lt_batchNorm_bwd_input, lt_user_input)) {
			BatchNorm_res[dnnResourceSrc] = (void*)in_buffer;
		}
		else {
			BatchNorm_res[dnnResourceSrc] = user_input;
		}

		if (!dnnLayoutCompare_F32(lt_batchNorm_bwd_outgrad, lt_user_output)) {
			BatchNorm_res[dnnResourceDiffDst] = (void*)outgrad_buffer;
		}
		else {
			BatchNorm_res[dnnResourceDiffDst] = user_out_grad;
		}

		BatchNorm_res[dnnResourceMean] = savedMean.Data();
		BatchNorm_res[dnnResourceVariance] = savedInvStdDev.Data();
		BatchNorm_res[dnnResourceScaleShift] = scaleShift_buffer;
		BatchNorm_res[dnnResourceDiffSrc] = user_in_grad;
		BatchNorm_res[dnnResourceDiffScaleShift] = scaleShift_grad_buffer;

		assert(dnnExecute_F32(pBatchNormalizationBwd, BatchNorm_res) == E_SUCCESS);

	}

};

template class MKL2017BatchNormEngine<float>;
template class MKL2017BatchNormEngine<double>;

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
  
    if (HasFlag(enabledEngines, BatchNormEngineKind::MKL2017))
    {
      if (GetMathLibTraceLevel() > 0)
        fprintf(stderr, "Using MKL2017 batch normalization engine.\n");

      return std::make_unique<MKL2017BatchNormEngine<ElemType>>(deviceId, inOutT, spatial, imageLayout);
    }
  
    RuntimeError("Could not find appropriate batch normalization engine.");
}

template class BatchNormEngine<float>;
template class BatchNormEngine<double>;

}}}
