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
size_t CuDnnRNNExecutor<ElemType>::GetWSize()
{
    TensorShape temp_x((int) m_inputSize, (int)m_miniBatchSize, (int)m_rnnT->GetLength());
    SetXDesc(temp_x);
    CuDnnFilter<ElemType> wFilter(*m_rnnT, xDesc.data());
    return wFilter.GetSize();
}

template <class ElemType>
void CuDnnRNNExecutor<ElemType>::SetXDesc(const TensorShape& shapeX)
{
    // one time slice layout and stride
    int dimX1[3] = { (int)shapeX[0], (int)shapeX[1], 1 };
    int strideX[3] = { 1, dimX1[0], dimX1[0] * dimX1[1] };

    // create descriptors for the time slices
    for (size_t i = 0; i < shapeX[2]; i++)
    {
        if (xDesc.size() <= i)
        {
            xDesc.push_back(cudnnTensorDescriptor_t());
            CUDNN_CALL(cudnnCreateTensorDescriptor(&xDesc[i]));
        }
        CUDNN_CALL(cudnnSetTensorNdDescriptor(xDesc[i], CUDNN_DATA_FLOAT, 3, dimX1, strideX));
    }
}

template <class ElemType>
void CuDnnRNNExecutor<ElemType>::ForwardCore(
    const GPUMatrix<ElemType>& weightsW,
    const GPUMatrix<ElemType>& inputX, const TensorShape shapeX, GPUMatrix<ElemType>& outputY, const TensorShape shapeY,
    const RnnParameters& rnnParameters,
    const vector<size_t>& numSequencesForFrame, 
    GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace
    )
{
    // test that the RNN shape is correct
    if (!m_rnnT->IsCompatable(rnnParameters))
        LogicError("RNN Layout has changed during processing");
#if frame_mode
    // assume input, dims=[inputsize, miniBatch, seqLength], stride=[1, inputSize*seqLength, inputSize]
    // assume output is equivalent
    // n.b. we should test these assumptions

    // compute the transposed tensor shape (in-place)
    size_t miniBatch = shapeX.GetDim(1);
    size_t seqLength = shapeX.GetDim(2);
    size_t T = shapeX.GetDim(3); // this is the number of time steps in each parallel sequence - make sure it is one
    if (T != 1)
        RuntimeError("RNN only works in frame mode");

    if (m_rnnT->GetLength() != seqLength)
        m_rnnT->SetLength(seqLength);

    SetXDesc(shapeX);

    size_t outputSize = shapeY.GetDim(0);
    if (outputSize != (m_rnnT->isBidirectional()?2:1) * m_rnnT->GetNumHidden())
       InvalidArgument("CuDnn ForwardCore: Output leading dimension must be twice hidden size for bidirectional networks");
    if (shapeY.GetDim(1) != miniBatch)
        RuntimeError("CuDnn ForwardCore: Output minibatch size doesn't match input minibatch size");
    if (shapeY.GetDim(2) != seqLength)
        RuntimeError("CuDnn ForwardCore: Output sequence length doesn't match input sequence length");

    int dimY[3] = { (int)outputSize, (int)miniBatch, 1 };
    int strideY[3] = { 1, dimY[0], dimY[0] * dimY[1] };

    for (size_t i = 0; i < seqLength; i++)
    {
        if (yDesc.size() <= seqLength)
        {
            yDesc.push_back(cudnnTensorDescriptor_t());
            CUDNN_CALL(cudnnCreateTensorDescriptor(&yDesc[i]));
        }
        CUDNN_CALL(cudnnSetTensorNdDescriptor(yDesc[i], CUDNN_DATA_FLOAT, 3, dimY, strideY));
    }
#else
    // assume input, dims=[inputsize, ???], stride=[1, inputSize, ???]
    // assume output is equivalent
    // n.b. we should test these assumptions

    size_t seqLength = numSequencesForFrame.size();

    if (m_rnnT->GetLength() != seqLength)
        m_rnnT->SetLength(seqLength);

    { // set up the input descriptors
        for (size_t i = 0; i < numSequencesForFrame.size(); i++)
        {
            if (xDesc.size() <= i)
            {
                xDesc.push_back(cudnnTensorDescriptor_t());
                CUDNN_CALL(cudnnCreateTensorDescriptor(&xDesc[i]));
            }
            int dimX[3] = { (int)shapeX[0], (int)numSequencesForFrame[i], 1 };
            int strideX[3] = { 1, dimX[0], dimX[0] * dimX[1] };
            CUDNN_CALL(cudnnSetTensorNdDescriptor(xDesc[i], CUDNN_DATA_FLOAT, 3, dimX, strideX));
        }
    }

    size_t outputSize = shapeY.GetDim(0);
    if (outputSize != (m_rnnT->isBidirectional() ? 2 : 1) * m_rnnT->GetNumHidden())
        InvalidArgument("CuDnn ForwardCore: Output leading dimension must be twice hidden size for bidirectional networks");

    { // set up the output descriptors
        for (size_t i = 0; i < numSequencesForFrame.size(); i++)
        {
            if (yDesc.size() <= i)
            {
                yDesc.push_back(cudnnTensorDescriptor_t());
                CUDNN_CALL(cudnnCreateTensorDescriptor(&yDesc[i]));
            }
            int dimY[3] = { (int)outputSize, (int)numSequencesForFrame[i], 1 };
            int strideY[3] = { 1, dimY[0], dimY[0] * dimY[1] };
            CUDNN_CALL(cudnnSetTensorNdDescriptor(yDesc[i], CUDNN_DATA_FLOAT, 3, dimY, strideY));
        }
    }

#endif

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
        InvalidArgument("RNN needs %ld parameters, but %ld were allocated", wDesc->GetSize(), weightsW.GetNumRows());

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
    const RnnParameters& rnnParameters,
    GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace
    )
{
    // test that the RNN shape is correct
    if (!m_rnnT->IsCompatable(rnnParameters))
        LogicError("RNN Layout has changed during processing");

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
    const RnnParameters& rnnParameters,
    GPUMatrix<ElemType>& reserve, GPUMatrix<ElemType>& workspace
    )
{
    // test that the RNN shape is correct
    if (!m_rnnT->IsCompatable(rnnParameters))
        LogicError("RNN Layout has changed during processing");
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
