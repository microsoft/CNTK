//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "Matrix.h"
#include "GPUMatrix.h"
#include <typeinfo>
#include <typeindex>
#include "CuDnnCommon.h"

template <>
const char* CudaErrString<cudnnStatus_t>(cudnnStatus_t x)
{
    return cudnnGetErrorString(x);
}

namespace Microsoft { namespace MSR { namespace CNTK {

static bool IsGpu(DEVICEID_TYPE deviceId)
{
    return deviceId >= 0;
}

class CuDnnDropout
{
	CuDnn::ptr_t m_cudnn;
	unsigned long long m_seed = 0xdeadbeefull;
public:
	CuDnnDropout(float dropout = 0.0f, unsigned long long seed = 0xdeadbeefull)
		: m_dropoutDesc(nullptr), m_cudnn(CuDnn::Instance())
	{
		CUDNN_CALL(cudnnCreateDropoutDescriptor(&m_dropoutDesc));
		size_t stateSize;
		void *states;
		CUDNN_CALL(cudnnDropoutGetStatesSize(*m_cudnn, &stateSize));

		// bugbug: possible leak. Does CuDnn release this for us?
		CUDA_CALL(cudaMalloc(&states, stateSize));

		CUDNN_CALL(cudnnSetDropoutDescriptor(m_dropoutDesc,
			*m_cudnn,
			dropout,
			states,
			stateSize,
			seed));
	}

	~CuDnnDropout()
	{
		if (m_dropoutDesc != nullptr)
		{
			cudnnDestroyDropoutDescriptor(m_dropoutDesc);
			m_dropoutDesc = nullptr;
		}
	}

	operator cudnnDropoutDescriptor_t() const
	{
		return m_dropoutDesc;
	}

	DISABLE_COPY_AND_MOVE(CuDnnDropout);

private:
	cudnnDropoutDescriptor_t m_dropoutDesc;
};

class CuDnnRNN
{
	CuDnnDropout m_dropout;
public:
    CuDnnRNN(cudnnDataType_t dataType)
        : m_rnnDesc(nullptr)
    {
		CUDNN_CALL(cudnnCreateRNNDescriptor(&m_rnnDesc));

		// hard code these for now, expose other types later.
		cudnnRNNMode_t RNNMode = CUDNN_LSTM;
		int hiddenSize = 512;
		int seqLength = 512;
		int numLayers = 6;
		bool bidirectional = true;

		CUDNN_CALL(cudnnSetRNNDescriptor(m_rnnDesc,
			hiddenSize,
			seqLength,
			numLayers,
			m_dropout,
			CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
			bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
			RNNMode,
			dataType));
    }

    ~CuDnnRNN()
    {
        if (m_rnnDesc != nullptr)
        {
            cudnnDestroyRNNDescriptor(m_rnnDesc);
            m_rnnDesc = nullptr;
        }
    }

    operator cudnnRNNDescriptor_t() const
    {
        return m_rnnDesc;
    }

    DISABLE_COPY_AND_MOVE(CuDnnRNN);

private:
	cudnnRNNDescriptor_t m_rnnDesc;
};

class CuDnnFilter
{
	CuDnn::ptr_t m_cudnn;
public:
	CuDnnFilter(const CuDnnRNN& rnn, const cudnnTensorDescriptor_t *xDesc) :
		m_cudnn(CuDnn::Instance())
	{

		CUDNN_CALL(cudnnCreateFilterDescriptor(&m_filterDesc));

		size_t filterSize;
		CUDNN_CALL(cudnnGetRNNParamsSize(*m_cudnn, rnn, xDesc, &filterSize));

		int dimW[3];
		// bugbug: hard-wired for float
		dimW[0] = (int)(filterSize / sizeof(float));
		dimW[1] = 1;
		dimW[2] = 1;

		CUDNN_CALL(cudnnSetFilterNdDescriptor(m_filterDesc, CUDNN_DATA_FLOAT, CUDNN_TENSOR_NCHW, 3, dimW));
	}

	~CuDnnFilter()
	{
		if (m_filterDesc != nullptr)
		{
			cudnnDestroyFilterDescriptor(m_filterDesc);
			m_filterDesc = nullptr;
		}
	}

	operator cudnnFilterDescriptor_t() const
	{
		return m_filterDesc;
	}

	DISABLE_COPY_AND_MOVE(CuDnnFilter);

private:
	cudnnFilterDescriptor_t m_filterDesc;
};

template <class ElemType>
class CuDnnRNNExecutor
{
	CuDnn::ptr_t m_cudnn;
	cudnnDataType_t m_dataType;
	using Mat = Matrix<ElemType>;
public:
    CuDnnRNNExecutor(const TensorShape &inputShape, const TensorShape &outputShape, DEVICEID_TYPE deviceId) :
                           m_cudnn(CuDnn::Instance()),
                           m_dataType(CuDnnTensor::GetDataType<ElemType>())
    {
	}

protected:

    void EnsureCompatible() override
    {
        if (!IsGpu(m_deviceId))
            RuntimeError("cuDNN convolution engine supports GPU devices only.");
    }

    void EnsureRNNInitialized() override
    {
        if (m_rnnT == nullptr)
        {
			m_rnnT = std::make_unique<CuDnnRNN>(m_dataType);
        }
    }

    void ForwardCore(const TensorShape& in, const Mat& weights, TensorShape& out, Mat& workspace, Mat& reserve) override
    {	
		// get input data layout
		// source shape, stride is [inputSize, seqLength, miniBatch], [1, inputSize, inputSize*seqLength]
		// target shape, stride is [inputsize, miniBatch, seqLength], [1, inputSize*seqLength, inputSize]

		size_t inputSize = in.GetDim(0);
		size_t seqLength = in.GetDim(1);
		size_t miniBatch = in.GetDim(2);

		int dimX = { inputSize, miniBatch, 1 };
		int strideX = { 1, dimX[0] * dimX[1], dimX[0] }
		vector<cudnnTensorDescriptor_t> xDesc(seqLength);
		for (int i = 0; i < seqLength; i++) {
			cudnnErrCheck(cudnnCreateTensorDescriptor(&xDesc[i]));
			cudnnErrCheck(cudnnSetTensorNdDescriptor(xDesc[i], CUDNN_DATA_FLOAT, 3, dimS, strideX));
		}

		// get output data layout
		// source shape, stride is [outputSize, seqLength, miniBatch], [1, outputSize, outputSize*seqLength]
		// target shape, stride is [outputSize, miniBatch, seqLength], [1, outputSize*seqLength, outputSize]

		size_t outputSize = in.GetDim(0);
		if (in.GetDim(1) != seqLength)
			RuntimeError("CuDnn ForwardCore: Output sequence length doesn't match input sequence length");
		if (in.GetDim(2) != miniBatch)
			RuntimeError("CuDnn ForwardCore: Output minibatch size doesn't match input minibatch size");

		int dimY = { outputSize, miniBatch, 1 };
		int strideX = { 1, dimY[0] * dimY[1], dimY[0] }
		vector<cudnnTensorDescriptor_t> yDesc(seqLength);
		for (int i = 0; i < seqLength; i++) {
			cudnnErrCheck(cudnnCreateTensorDescriptor(&yDesc[i]));
			cudnnErrCheck(cudnnSetTensorNdDescriptor(yDesc[i], CUDNN_DATA_FLOAT, 3, dimY, strideY));
		}

		// ensure workspace and reserve are large enough
		{
			size_t workSize;
			size_t reserveSize;

			// Need for every pass
			CUDNN_CALL(cudnnGetRNNWorkspaceSize(m_cudnn, m_rnnT, xDesc, &workSize));
			// Only needed in training, can't be touched between passes.
			CUDNN_CALL(cudnnGetRNNTrainingReserveSize(m_cudnn, m_rnnT, xDesc, &reserveSize));

			// convert from bytes to ElemType
			workSize = (workSize + sizeof(ElemType) - 1) / (sizeof(ElemType));
			reserveSize = (reserveSize + sizeof(ElemType) - 1) / sizeof(ElemType);

			reserve.Resize(reserveSize);
			workspace.Resize(workSize);
		}

		CUDNN_CALL(cudnnRNNForwardTraining(m_cudnn,
			m_rnnT,
			xDesc.data(), ptr(in),
			0, nullptr,
			0, nullptr,
			wDesc, ptr(weights),
			yDesc.data(), ptr(out),
			0, nullptr,
			0, nullptr,
			ptr(workspace), workSize,
			ptr(reserveSpace), reserveSize));
    }

    void BackwardDataCore(const Mat& srcGrad, const Mat& kernel, Mat& grad, Mat& workspace) override
    {
        size_t batchSize = srcGrad.GetNumCols();
        // Find best algo and allocate temp buffer, if needed.
        auto finder = [this](int& calgo, cudnnConvolutionBwdDataAlgoPerf_t algoPerf[MaxAlgoCount]) -> cudnnStatus_t
        {
            return cudnnFindConvolutionBackwardDataAlgorithm(*m_cudnn, *m_kernelT, m_outT, *m_conv, m_inT, MaxAlgoCount, &calgo, algoPerf);
        };
        auto staticFinder = [this](cudnnConvolutionBwdDataAlgo_t& algo) -> cudnnStatus_t
        {
            return cudnnGetConvolutionBackwardDataAlgorithm(*m_cudnn, *m_kernelT, m_outT, *m_conv, m_inT, CUDNN_CONVOLUTION_BWD_DATA_NO_WORKSPACE, 0, &algo);
        };
        FindBestAlgo(batchSize, m_backDataAlgo, finder, staticFinder);
        if (m_backDataAlgo.Algo.memory > 0)
            workspace.Resize((m_backDataAlgo.Algo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
        // Compute gradients with respect to the output tensor (data).
        CUDNN_CALL(cudnnConvolutionBackwardData(*m_cudnn, &C::One, *m_kernelT, ptr(kernel), m_outT, ptr(srcGrad), *m_conv, m_backDataAlgo.Algo.algo,
                                                ptr(workspace), m_backDataAlgo.Algo.memory, &C::One, m_inT, ptr(grad)));
    }

    void BackwardKernelCore(const Mat& srcGrad, const Mat& in, Mat& kernelGrad, bool /*allowReuse*/, Mat& workspace) override
    {
        size_t batchSize = in.GetNumCols();
        // Find best algo and allocate temp buffer, if needed.
        auto finder = [this](int& calgo, cudnnConvolutionBwdFilterAlgoPerf_t algoPerf[MaxAlgoCount]) -> cudnnStatus_t
        {
            return cudnnFindConvolutionBackwardFilterAlgorithm(*m_cudnn, m_inT, m_outT, *m_conv, *m_kernelT, MaxAlgoCount, &calgo, algoPerf);
        };
        auto staticFinder = [this](cudnnConvolutionBwdFilterAlgo_t& algo) -> cudnnStatus_t
        {
            return cudnnGetConvolutionBackwardFilterAlgorithm(*m_cudnn, m_inT, m_outT, *m_conv, *m_kernelT, CUDNN_CONVOLUTION_BWD_FILTER_NO_WORKSPACE, 0, &algo);
			class CuDnnDropout
			{
				unsigned long long m_seed = 0xdeadbeefull;
			public:
				CuDnnDropout(float dropout = 0.0f, unsigned long long seed = 0xdeadbeefull)
					: m_dropoutDesc(nullptr)
				{
					CUDNN_CALL(cudnnCreateDropoutDescriptor(&m_dropoutDesc));
					size_t stateSize;
					void *states;
					CUDNN_CALL(cudnnDropoutGetStatesSize(CuDnn::Instance(), &stateSize));

					// bugbug: possible leak. Does CuDnn release this for us?
					CUDA_CALL(cudaMalloc(&states, stateSize));

					CUDA_CALL(cudnnSetDropoutDescriptor(m_dropoutDesc,
						CuDnn::Instance(),
						dropout,
						states,
						stateSize,
						seed));
				}

				~CuDnnDropout()
				{
					if (m_dropoutDesc != nullptr)
					{
						cudnnDestroyDropoutDescriptor(m_dropoutDesc);
						m_dropoutDesc = nullptr;
					}
				}

				operator cudnnDropoutDescriptor_t() const
				{
					return m_dropoutDesc;
				}

				DISABLE_COPY_AND_MOVE(CuDnnDropout);

			private:
				cudnnDropoutDescriptor_t m_dropoutDesc;
			};

		};
        FindBestAlgo(batchSize, m_backFiltAlgo, finder, staticFinder);
        if (m_backFiltAlgo.Algo.memory > 0)
            workspace.Resize((m_backFiltAlgo.Algo.memory + sizeof(ElemType) - 1) / sizeof(ElemType), 1);
        // Compute gradients with respect to the output tensor (data).
        CUDNN_CALL(cudnnConvolutionBackwardFilter(*m_cudnn, &C::One, m_inT, ptr(in), m_outT, ptr(srcGrad), *m_conv, m_backFiltAlgo.Algo.algo,
                                                  ptr(workspace), m_backFiltAlgo.Algo.memory, &C::One, *m_kernelT, ptr(kernelGrad)));
    }

private:
    using C = Consts<ElemType>;


    template <typename TAlgo, typename TFinder, typename TStaticFinder>
    void FindBestAlgo(size_t batchSize, TAlgo& algo, TFinder finder, TStaticFinder staticFinder)
    {
        if (!algo.NeedAutotuning(batchSize))
            return;
        m_inT.UpdateBatchSize(batchSize);
        m_outT.UpdateBatchSize(batchSize);
        using CuDnnAlgoT = decltype(TAlgo::Algo);
        CuDnnAlgoT algoPerf[MaxAlgoCount];
        int calgo = 0;
        cudnnStatus_t err = finder(calgo, algoPerf);
        // Alloc failed - usually means cuDNN runtime auto-tuner could not allocate workspace.
        // In such case, use static auto-tuner with no workspace.
        if (err == CUDNN_STATUS_ALLOC_FAILED)
        {
            decltype(CuDnnAlgoT::algo) noMemAlgo;
            CUDNN_CALL(staticFinder(noMemAlgo));
            algo.CurMBSize = batchSize;
            algo.Algo = algoPerf[0];
            algo.Algo.algo = noMemAlgo;
            algo.Algo.memory = 0;
            algo.Algo.status = CUDNN_STATUS_SUCCESS;
            algo.NoWorkspaceAlgo = noMemAlgo;
            return;
        }
        CUDNN_CALL(err);
        assert(calgo > 0);
        size_t inputSampleSize = m_geometry->InputShape().GetNumElements();
        size_t maxMem = m_maxTempMemSizeInSamples == 0 ? (std::numeric_limits<size_t>::max)() : inputSampleSize * m_maxTempMemSizeInSamples * sizeof(ElemType);
        // Find best (fastest) algorithm which satisfies workspace requirements.
        auto res = std::find_if(algoPerf, algoPerf + calgo,
            [=](const CuDnnAlgoT& cur)
            {
                return cur.status == CUDNN_STATUS_SUCCESS && cur.memory <= maxMem;
            });
        if (res == algoPerf + calgo)
            RuntimeError("cuDNN could not find suitable algorithm for the current convolution configuration.");
        algo.CurMBSize = batchSize;
        algo.Algo = *res;
        // Find fastest algorithm that does NOT require workspace. It is used as a fallback algo in Forward function.
        res = std::find_if(algoPerf, algoPerf + calgo,
            [](const CuDnnAlgoT& cur)
            {
                return cur.status == CUDNN_STATUS_SUCCESS && cur.memory == 0;
            });
        if (res == algoPerf + calgo)
        {
            // In theory, this should never happen.
            RuntimeError("cuDNN could not find no-workspace algorithm for the current convolution configuration.");
        }
        else
            algo.NoWorkspaceAlgo = (*res).algo;
    }

    static ElemType* ptr(Mat& src)
    {
        return src.Data();
    }
    static const ElemType* ptr(const Mat& src)
    {
        return src.Data();
    }

private:
    template <typename T>
    struct ConvAlgoInfo
    {
        using CuDnnAlgoT = decltype(T::algo);

        ConvAlgoInfo()
            : CurMBSize(0)
        {
            Algo.status = CUDNN_STATUS_NOT_INITIALIZED;
            NoWorkspaceAlgo = (CuDnnAlgoT)-1;
        }
        // Current mini-batch size, needed for re-computing statistics in auto-tuner.
        size_t CurMBSize;
        T Algo;
        CuDnnAlgoT NoWorkspaceAlgo;

        bool NeedAutotuning(size_t batchSize)
        {
            // Need to re-run auto-tuner in case minibatch size is increased.
            // If minibatch size is decreased we assume that previously selected algorithm requires less or the same amount of workspace.
            // This is done to avoid re-running auto-tuner every time in case minibatch size changes frequently (e.g. when distributed reading is enabled).
            // REVIEW alexeyk: potentially, this might cause some perf issues if better (faster) algo can be selected for a smaller mininbatch.
            // We also need to reset auto-tuning status at the beginning of each epoch but ComputationNode currently does not provide such notification.
            // We assume no other dimensions of tensors can change so we don't check it.
            // REVIEW alexeyk: review once we get response from NVIDIA.
            return (Algo.status != CUDNN_STATUS_SUCCESS || batchSize > CurMBSize);
        }
    };

    CuDnn::ptr_t m_cudnn;
    cudnnDataType_t m_dataType;
    CuDnnTensor m_inT;
    CuDnnTensor m_outT;
    // Convolution specific.
    std::unique_ptr<CuDnnKernel> m_kernelT;
    std::unique_ptr<CuDnnConv> m_conv;
    // Pooling specific.
    std::unique_ptr<CuDnnPool> m_pool;

    ConvAlgoInfo<cudnnConvolutionFwdAlgoPerf_t> m_fwdAlgo;
    ConvAlgoInfo<cudnnConvolutionBwdDataAlgoPerf_t> m_backDataAlgo;
    ConvAlgoInfo<cudnnConvolutionBwdFilterAlgoPerf_t> m_backFiltAlgo;
};

} } }
