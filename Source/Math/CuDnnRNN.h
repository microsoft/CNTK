#pragma once

#include "Matrix.h"
#include "GPUMatrix.h"
#include "TensorShape.h"
#include <typeinfo>
#include <typeindex>
#include "CuDnnCommon.h"

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
template <class ElemType>
class CuDnnRNN
{
private:
    cudnnDataType_t m_dataType;
    cudnnRNNDescriptor_t m_rnnDesc;
    CuDnnDropout m_dropout;
    size_t m_numHidden;
    size_t m_numLayers;
public:
    CuDnnRNN(const size_t numLayers, const size_t numHidden)
        : m_rnnDesc(nullptr), m_dropout(0.0f), m_numHidden(numHidden), m_numLayers(numLayers),
        m_dataType(CuDnnTensor::GetDataType<ElemType>())
    {
        CUDNN_CALL(cudnnCreateRNNDescriptor(&m_rnnDesc));

        // hard code these for now, expose other types later.
        cudnnRNNMode_t RNNMode = CUDNN_LSTM;
        int seqLength = 31;
        bool bidirectional = true;

        CUDNN_CALL(cudnnSetRNNDescriptor(m_rnnDesc,
            (int)m_numHidden,
            seqLength,
            (int)m_numLayers,
            m_dropout,
            CUDNN_LINEAR_INPUT, // We can also skip the input matrix transformation
            bidirectional ? CUDNN_BIDIRECTIONAL : CUDNN_UNIDIRECTIONAL,
            RNNMode,
            m_dataType));
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

    size_t GetNumLayers() { return m_numLayers; }
    size_t GetNumHidden() { return m_numHidden; }

    DISABLE_COPY_AND_MOVE(CuDnnRNN);
};
template <class ElemType>
class CuDnnFilter
{
    cudnnDataType_t m_dataType;
    CuDnn::ptr_t m_cudnn;
    size_t m_filterSize;
public:
    CuDnnFilter(const CuDnnRNN<ElemType>& rnn, const cudnnTensorDescriptor_t *xDesc) :
        m_cudnn(CuDnn::Instance()), m_dataType(CuDnnTensor::GetDataType<ElemType>())
    {
        CUDNN_CALL(cudnnCreateFilterDescriptor(&m_filterDesc));
        try
        {
            size_t filterSize;
            CUDNN_CALL(cudnnGetRNNParamsSize(*m_cudnn, rnn, xDesc, &filterSize));

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
public:
    CuDnnRNNExecutor(const TensorShape &/*inputShape*/, const TensorShape &/*outputShape*/, const size_t numRows, const size_t numHidden) :
        m_cudnn(CuDnn::Instance()),
        m_dataType(CuDnnTensor::GetDataType<ElemType>()),
        workspace(0), reserve(0),
        m_BackwardDataCalledYet(false)
    {
        m_rnnT = std::make_unique<CuDnnRNN<ElemType>>(numRows, numHidden);
    }

    size_t GetWSize(cudnnTensorDescriptor_t *xDesc);

    void ForwardCore(const GPUMatrix<ElemType>& weightsW, const GPUMatrix<ElemType>& inputX, const TensorShape shapeX, GPUMatrix<ElemType>& outputY, const TensorShape shapeY);
    void BackwardWeightsCore(const GPUMatrix<ElemType>& inputX, const GPUMatrix<ElemType>& outputY, GPUMatrix<ElemType>& dw);
    void BackwardDataCore(const GPUMatrix<ElemType>& outputY, const GPUMatrix<ElemType>& outputDY, const GPUMatrix<ElemType>& w, GPUMatrix<ElemType>& dx);

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

private:
    std::unique_ptr<CuDnnRNN<ElemType>> m_rnnT;
    GPUMatrix<ElemType> workspace;
    GPUMatrix<ElemType> reserve;
    bool m_BackwardDataCalledYet;
};

} } }