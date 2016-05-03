//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "Matrix.h"
#include "GPUMatrix.h"
#include "TensorShape.h"
#include "TensorView.h"
#include <typeinfo>
#include <typeindex>
#include "CuDnnCommon.h"
#include "CuDnnRNN.h"

namespace Microsoft { namespace MSR { namespace CNTK {

#if 0
static bool IsGpu(DEVICEID_TYPE deviceId)
{
    return deviceId >= 0;
}
#endif

template<class ElemType>
class CuDnnTensorDescriptor
{
private:
    cudnnTensorDescriptor_t m_tensorDesc;
public:
    CuDnnTensorDescriptor(size_t hiddenSize, size_t miniBatch, size_t numLayers)
    {
        cudnnDataType_t m_dataType = CuDnnTensor::GetDataType<ElemType>();
        int dimA[3] = { (int)hiddenSize, (int)miniBatch, (int)numLayers };
        int strideA[3] = { 1, dimA[0], dimA[0] * dimA[1] };
        CUDNN_CALL(cudnnCreateTensorDescriptor(&m_tensorDesc));
        CUDNN_CALL(cudnnSetTensorNdDescriptor(m_tensorDesc, m_dataType, 3, dimA, strideA));
    }

    ~CuDnnTensorDescriptor()
    {
        cudnnDestroyTensorDescriptor(m_tensorDesc);
    }

    operator cudnnTensorDescriptor_t() const
    {
        return m_tensorDesc;
    }

    DISABLE_COPY_AND_MOVE(CuDnnTensorDescriptor);
};

template <class ElemType>
size_t CuDnnRNNExecutor<ElemType>::GetWSize(cudnnTensorDescriptor_t *xDesc)
{
    CuDnnFilter<ElemType> wFilter(*m_rnnT, xDesc);
    return wFilter.GetSize();
}

template <class ElemType>
void CuDnnRNNExecutor<ElemType>::ForwardCore(
    const GPUMatrix<ElemType>& weightsW,
    const GPUMatrix<ElemType>& inputX, const TensorShape shapeX, GPUMatrix<ElemType>& outputY, const TensorShape shapeY,
    GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace
    )
{
    // get input data layout
    // source shape, stride is [inputSize, seqLength, miniBatch], [1, inputSize, inputSize*seqLength]
    // target shape, stride is [inputsize, miniBatch, seqLength], [1, inputSize*seqLength, inputSize]

    // check for  in.GetShape().GetRank() == 4
    // check that in.getDim(3) == 1

    // compute the transposed tensor shape (in-place)
    size_t inputSize = shapeX.GetDim(0);
    size_t miniBatch = shapeX.GetDim(1);
    size_t seqLength = shapeX.GetDim(2);
    size_t T = shapeX.GetDim(3); // this is the number of time steps in each parallel sequence - make sure it is one
    if (T != 1)
        RuntimeError("RNN only works in frame mode");

    int dimX[3] = { (int)inputSize, (int)miniBatch, 1 };
    //int strideX[3] = { 1, dimX[0] * 31, dimX[0] };
    int strideX[3] = { 1, dimX[0], dimX[0] * dimX[1] };

    while (xDesc.size() < seqLength)
    {
        size_t i = xDesc.size();
        xDesc.push_back(cudnnTensorDescriptor_t());
        CUDNN_CALL(cudnnCreateTensorDescriptor(&xDesc[i]));
        CUDNN_CALL(cudnnSetTensorNdDescriptor(xDesc[i], CUDNN_DATA_FLOAT, 3, dimX, strideX));
    }

    // get output data layout
    // source shape, stride is [outputSize, seqLength, miniBatch], [1, outputSize, outputSize*seqLength]
    // target shape, stride is [outputSize, miniBatch, seqLength], [1, outputSize*seqLength, outputSize]

    size_t outputSize = shapeY.GetDim(0);
    if (outputSize != 2 * m_rnnT->GetNumHidden())
       InvalidArgument("CuDnn ForwardCore: Output leading dimension must be twice hidden size for bidirectional networks");
    if (shapeY.GetDim(1) != miniBatch)
        RuntimeError("CuDnn ForwardCore: Output minibatch size doesn't match input minibatch size");
    if (shapeY.GetDim(2) != seqLength)
        RuntimeError("CuDnn ForwardCore: Output sequence length doesn't match input sequence length");

    int dimY[3] = { (int)outputSize, (int)miniBatch, 1 };
    //int strideY[3] = { 1, dimY[0] * 31, dimY[0] };
    int strideY[3] = { 1, dimY[0], dimY[0] * dimY[1] };

    while (yDesc.size() < seqLength)
    {
        size_t i = yDesc.size();
        yDesc.push_back(cudnnTensorDescriptor_t());
        CUDNN_CALL(cudnnCreateTensorDescriptor(&yDesc[i]));
        CUDNN_CALL(cudnnSetTensorNdDescriptor(yDesc[i], CUDNN_DATA_FLOAT, 3, dimY, strideY));
    }

    // ensure workspace and reserve are large enough
    size_t workSize;
    size_t reserveSize;

    // Need for every pass
    CUDNN_CALL(cudnnGetRNNWorkspaceSize(*m_cudnn, *m_rnnT, xDesc.data(), &workSize));
    // Only needed in training, can't be touched between passes.
    CUDNN_CALL(cudnnGetRNNTrainingReserveSize(*m_cudnn, *m_rnnT, xDesc.data(), &reserveSize));

    // convert from bytes to ElemType
    workSize = (workSize + sizeof(ElemType) - 1) / (sizeof(ElemType));
    reserveSize = (reserveSize + sizeof(ElemType) - 1) / sizeof(ElemType);

    reserve.Resize(reserveSize, 1);
    workspace.Resize(workSize, 1);

    wDesc = make_unique<CuDnnFilter<ElemType>>(*m_rnnT, xDesc.data());
    if (wDesc->GetSize() != weightsW.GetNumRows())
        InvalidArgument("RNN needs %d parameters, but %d were allocated", wDesc->GetSize(), weightsW.GetNumRows());

    CUDNN_CALL(cudnnRNNForwardTraining(
        *m_cudnn, *m_rnnT,
        xDesc.data(), inputX.Data(),
        0, 0,
        0, 0,
        *wDesc, weightsW.Data(),
        yDesc.data(), outputY.Data(),
        0, 0,
        0, 0,
        workspace.Data(), workspace.GetNumElements()*sizeof(ElemType),
        reserve.Data(), reserve.GetNumElements()*sizeof(ElemType)));
    m_BackwardDataCalledYet = false;
}

template <class ElemType>
void CuDnnRNNExecutor<ElemType>::BackwardDataCore(
    const GPUMatrix<ElemType>& outputY, const GPUMatrix<ElemType>& outputDY, const GPUMatrix<ElemType>& weightsW, GPUMatrix<ElemType>& dx,
    GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace
    )
{
    if (!m_BackwardDataCalledYet)
    {
        CUDNN_CALL(cudnnRNNBackwardData(
            *m_cudnn, *m_rnnT,
            yDesc.data(), outputY.Data(),
            yDesc.data(), outputDY.Data(),
            0, 0,
            0, 0,
            *wDesc, weightsW.Data(),
            0, 0,
            0, 0,
            xDesc.data(), dx.Data(),
            0, 0,
            0, 0,
            workspace.Data(), workspace.GetNumElements()*sizeof(ElemType),
            reserve.Data(), reserve.GetNumElements()*sizeof(ElemType)));
    }
    m_BackwardDataCalledYet = true;
}

template <class ElemType>
void CuDnnRNNExecutor<ElemType>::BackwardWeightsCore(const GPUMatrix<ElemType>& inputX, const GPUMatrix<ElemType>& outputY, GPUMatrix<ElemType>& dw,
    GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace
    )
{
    if (!m_BackwardDataCalledYet)
        LogicError("out of order calling you have been very bad");
    CUDNN_CALL(cudnnRNNBackwardWeights(
        *m_cudnn, *m_rnnT,
        xDesc.data(), inputX.Data(),
        0, 0,
        yDesc.data(), outputY.Data(),
        workspace.Data(), workspace.GetNumElements()*sizeof(ElemType),
        *wDesc, dw.Data(),
        reserve.Data(), reserve.GetNumElements()*sizeof(ElemType)));
}

template class CuDnnRNNExecutor<double>;
template class CuDnnRNNExecutor<float>;

} } }