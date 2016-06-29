#pragma once

#include "Matrix.h"
#include "GPUMatrix.h"
#include "TensorShape.h"
#include <typeinfo>
#include <typeindex>
#include "CuDnnCommon.h"
#include "RNNCommon.h"

namespace Microsoft { namespace MSR { namespace CNTK {

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

        fprintf(stderr, "CuDnnDropout()\n");
        CUDNN_CALL(cudnnSetDropoutDescriptor(m_dropoutDesc,
            *m_cudnn,
            dropout,
            states,
            stateSize,
            seed));
    }

    ~CuDnnDropout()
    {
        fprintf(stderr, "~CuDnnDropout()\n");
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
template <class ElemType>
class CuDnnRNN
{
private:
    cudnnDataType_t m_dataType;
    cudnnRNNDescriptor_t m_rnnDesc;
    CuDnnDropout m_dropout;
    RnnParameters m_rnnParameters;

    cudnnRNNMode_t GetMode()
    {
        if (m_rnnParameters.m_rnnMode == wstring(L"LSTM"))
            return cudnnRNNMode_t::CUDNN_LSTM;
        if (m_rnnParameters.m_rnnMode == wstring(L"GRU"))
            return cudnnRNNMode_t::CUDNN_GRU;
        if (m_rnnParameters.m_rnnMode == wstring(L"RNN_RELU"))
            return cudnnRNNMode_t::CUDNN_RNN_RELU;
        if (m_rnnParameters.m_rnnMode == wstring(L"RNN_TANH"))
            return cudnnRNNMode_t::CUDNN_RNN_TANH;
        InvalidArgument("RNN Mode set to %ls, but supported values are LSTM, GRU, RNN_RELU, RNN_TANH.", m_rnnParameters.m_rnnMode.c_str());
    }

public:
    CuDnnRNN(const RnnParameters& rnnParameters)
        : m_rnnDesc(nullptr), m_dropout(0.0f), m_rnnParameters(rnnParameters),
        m_dataType(CuDnnTensor::GetDataType<ElemType>())
    {
        fprintf(stderr, "CuDnnRNN()\n");
        CUDNN_CALL(cudnnCreateRNNDescriptor(&m_rnnDesc));
        CUDNN_CALL(cudnnSetRNNDescriptor(m_rnnDesc,
            (int)m_rnnParameters.m_hiddenSize,
            (int)m_rnnParameters.m_numLayers,
            m_dropout,
            CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
            m_rnnParameters.m_bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
            GetMode(),
            m_dataType));
    }

    ~CuDnnRNN()
    {
        fprintf(stderr, "~CuDnnRNN()\n");
        if (m_rnnDesc != nullptr)
        {
            cudnnDestroyRNNDescriptor(m_rnnDesc);
            m_rnnDesc = nullptr;
        }
    }

    bool IsCompatable(const RnnParameters& rnnParameters) const
    {
        return this->m_rnnParameters == rnnParameters;
    }

    operator cudnnRNNDescriptor_t() const
    {
        return m_rnnDesc;
    }

    bool isBidirectional() const { return m_rnnParameters.m_bidirectional; }

    size_t GetNumLayers() { return m_rnnParameters.m_numLayers; }
    size_t GetNumHidden() { return m_rnnParameters.m_hiddenSize; }

    DISABLE_COPY_AND_MOVE(CuDnnRNN);
};

template <class ElemType>
class CuDnnFilter
{
    cudnnDataType_t m_dataType;
    CuDnn::ptr_t m_cudnn;
    size_t m_filterSize;
public:
    CuDnnFilter(const CuDnnRNN<ElemType>& rnn, const cudnnTensorDescriptor_t& xDesc) :
        m_cudnn(CuDnn::Instance()), m_dataType(CuDnnTensor::GetDataType<ElemType>())
    {
        CUDNN_CALL(cudnnCreateFilterDescriptor(&m_filterDesc));
        try
        {
            size_t filterSize;
            CUDNN_CALL(cudnnGetRNNParamsSize(*m_cudnn, rnn, xDesc, &filterSize, m_dataType));

            size_t dataSize = 2; // CUDNN_DATA_HALF

            if (m_dataType == cudnnDataType_t::CUDNN_DATA_DOUBLE)
                dataSize = 8;
            else if (m_dataType == cudnnDataType_t::CUDNN_DATA_FLOAT)
                dataSize = 4;

            // convert from bytes to items
            m_filterSize = (filterSize + dataSize - 1) / dataSize;
            int dimW[3] = { (int)m_filterSize, 1, 1 };
            CUDNN_CALL(cudnnSetFilterNdDescriptor(m_filterDesc, m_dataType, CUDNN_TENSOR_NCHW, 3, dimW));
        }
        catch (exception e)
        {
            cudnnDestroyFilterDescriptor(m_filterDesc);
            m_filterDesc = nullptr;
            throw e;
        }
    }
    ~CuDnnFilter()
    {
        assert(m_filterDesc != nullptr);
        cudnnDestroyFilterDescriptor(m_filterDesc);
    }
    size_t GetSize() { return m_filterSize; }
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
    size_t m_xDim, m_yDim;
public:
    CuDnnRNNExecutor(size_t xDim, size_t yDim, const RnnParameters& rnnParameters ) :
        m_cudnn(CuDnn::Instance()),
        m_xDim(xDim), m_yDim(yDim),
        m_seqLength(0),
        m_dataType(CuDnnTensor::GetDataType<ElemType>()),
        m_BackwardDataCalledYet(false)
    {
        fprintf(stderr, "CuDnnRNNExecutor()\n");
        m_rnnT = std::make_unique<CuDnnRNN<ElemType>>(rnnParameters);
    }

    void ForwardCore(const GPUMatrix<ElemType>& weightsW, const GPUMatrix<ElemType>& inputX, GPUMatrix<ElemType>& outputY, const vector<size_t>& numSequencesForFrame, const RnnParameters& rnnParameters, GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace);
    void BackwardWeightsCore(const GPUMatrix<ElemType>& inputX, const GPUMatrix<ElemType>& outputY, GPUMatrix<ElemType>& dw, const RnnParameters& rnnParameters, GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace);
    void BackwardDataCore(const GPUMatrix<ElemType>& outputY, const GPUMatrix<ElemType>& outputDY, const GPUMatrix<ElemType>& w, GPUMatrix<ElemType>& dx, const RnnParameters& rnnParameters, GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace);

protected:
    std::unique_ptr<CuDnnFilter<ElemType>> wDesc;
    vector<cudnnTensorDescriptor_t> xDesc;
    vector<cudnnTensorDescriptor_t> yDesc;

private:
    static ElemType* ptr(GPUMatrix<ElemType>& src)
    {
        return src.Data();
    }
    static const ElemType* ptr(const GPUMatrix<ElemType>& src)
    {
        return src.Data();
    }

    void SetDescriptors(size_t dim, const vector<size_t>& numSequencesForFrame, vector<cudnnTensorDescriptor_t>& descriptors);

private:
    std::unique_ptr<CuDnnRNN<ElemType>> m_rnnT;
    bool m_BackwardDataCalledYet;
    size_t m_seqLength;
};

} } }