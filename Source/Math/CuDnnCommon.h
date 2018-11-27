//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "Basics.h"
#include "TensorShape.h"
#include <cudnn.h>
#if CUDNN_MAJOR < 5
#error CNTK requires the NVIDIA cuDNN library 5.0 or higher to build, cf. https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-Windows#cudnn or https://docs.microsoft.com/en-us/cognitive-toolkit/Setup-CNTK-on-Linux#cudnn for installation instructions.
#endif
#include <memory>
#include "half.hpp"

namespace Microsoft { namespace MSR { namespace CNTK {

class CuDnnTensor final
{
public:
    CuDnnTensor();
    CuDnnTensor(const TensorShape& src, cudnnDataType_t dataType);
    ~CuDnnTensor();

    void Set(const TensorShape& src, cudnnDataType_t dataType); 
    void UpdateBatchSize(size_t batchSize);

    operator cudnnTensorDescriptor_t() const { return m_tensor; }

    template <typename ElemType>
    static cudnnDataType_t GetDataType();

    DISABLE_COPY_AND_MOVE(CuDnnTensor);

private:
    cudnnTensorDescriptor_t m_tensor;
};

struct CuDnn final
{
    using ptr_t = std::shared_ptr<cudnnHandle_t>;
    static ptr_t Instance();

    DISABLE_COPY_AND_MOVE(CuDnn);
};

template <typename ElemType>
struct Consts
{
    static const ElemType Zero;
    static const ElemType One;
};

template <>
struct Consts<half>
{
    static const float Zero;
    static const float One;
};

} } }
