//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "GPUMatrix.h"
#include "CuDnnCommon.h"

namespace Microsoft { namespace MSR { namespace CNTK {

template <>
const float Consts<float>::One = 1;
template <>
const double Consts<double>::One = 1;
template <>
const float Consts<float>::Zero = 0;
template <>
const double Consts<double>::Zero = 0;

CuDnnTensor::CuDnnTensor(const TensorShape& src, cudnnDataType_t dataType)
    : m_tensor(nullptr)
{
    CUDNN_CALL(cudnnCreateTensorDescriptor(&m_tensor));
    // Set cuDNN tensor dimensions. cuDNN uses row-major format while TensorShape - column-major
    // so conversion is required. N dimension will be set to 1.
    const auto& stridesSrc = src.GetStrides();
    SmallVector<int> dims(src.GetRank() + 1);
    SmallVector<int> strides(stridesSrc.size() + 1);
    assert(dims.size() == strides.size());
    for (int i = 0; i < src.GetRank(); i++)
    {
        dims[dims.size() - 1 - i] = (int)src[i];
        strides[dims.size() - 1 - i] = (int)stridesSrc[i];
    }
    // Set "minibatch"(aka N) dimension.
    dims[0] = 1;
    strides[0] = strides[1] * dims[1];
    CUDNN_CALL(cudnnSetTensorNdDescriptor(m_tensor, dataType, (int)dims.size(), dims.data(), strides.data()));
}

CuDnnTensor::~CuDnnTensor()
{
    if (m_tensor != nullptr)
    {
        cudnnDestroyTensorDescriptor(m_tensor);
        m_tensor = nullptr;
    }
}

void CuDnnTensor::UpdateBatchSize(size_t batchSize)
{
    // Currently cuDNN supports only 2D and 3D convlutions anyway (so max 5D tensors).
    const int MaxDims = 5;
    int dims[MaxDims];
    int strides[MaxDims];
    int nbDims = 0;
    cudnnDataType_t dataType;
    // According to NVIDIA, Get/Set functions are very fast so it's safe to call them in a loop.
    CUDNN_CALL(cudnnGetTensorNdDescriptor(m_tensor, MaxDims, &dataType, &nbDims, dims, strides));
    assert(nbDims <= MaxDims);
    dims[0] = (int)batchSize;
    CUDNN_CALL(cudnnSetTensorNdDescriptor(m_tensor, dataType, nbDims, dims, strides));
}

template <typename ElemType>
cudnnDataType_t CuDnnTensor::GetDataType()
{
    if (typeid(ElemType) == typeid(float))
        return CUDNN_DATA_FLOAT;
    else if (typeid(ElemType) == typeid(double))
        return CUDNN_DATA_DOUBLE;
    else
        InvalidArgument("cuDNN engine currently supports only single and double precision data types.");
}

template cudnnDataType_t CuDnnTensor::GetDataType<float>();
template cudnnDataType_t CuDnnTensor::GetDataType<double>();

CuDnn::ptr_t CuDnn::Instance()
{
    auto createNew = []()
    {
        int deviceId;
        CUDA_CALL(cudaGetDevice(&deviceId));
        cudaDeviceProp props = {0};
        if (cudaGetDeviceProperties(&props, deviceId) != cudaSuccess || props.major < 3)
            RuntimeError("cuDNN requires device with compute capability 3.0 or higher.");
        cudnnHandle_t* cudnn = new cudnnHandle_t;
        CUDNN_CALL(cudnnCreate(cudnn));
        CUDNN_CALL(cudnnSetStream(*cudnn, GetStream()));
        return cudnn;
    };

    static std::shared_ptr<cudnnHandle_t> m_instance = std::shared_ptr<cudnnHandle_t>(createNew(), [](cudnnHandle_t* src)
    {
        assert(*src != nullptr);
        auto err = cudnnDestroy(*src);
        assert(err == CUDNN_STATUS_SUCCESS);
#ifdef NDEBUG
        UNUSED(err);
#endif
        delete src;
    });
    return m_instance;
}

} } }
