//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

// this header exposes MKL 2017's DNN functions, and MKL-DNN (https://github.com/01org/mkl-dnn)

#ifdef USE_MKL
#define USE_MKL2017DNN
#endif

#ifdef USE_MKL2017DNN

// These are headers from MKL 2017
#include "mkl_dnn.h"
#include "mkl_cblas.h"

// This header is from MKL-DNN
#ifdef USE_MKLDNN
#include "mkldnn.h"
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

template<typename T> inline
dnnError_t dnnLayoutCreate(
    dnnLayout_t* pLayout,
    size_t dimension,
    const size_t size[],
    const size_t strides[]);

template<> inline
dnnError_t dnnLayoutCreate<float>(
    dnnLayout_t* pLayout,
    size_t dimension,
    const size_t size[],
    const size_t strides[])
{
    return dnnLayoutCreate_F32(pLayout, dimension, size, strides);
}

template<> inline
dnnError_t dnnLayoutCreate<double>(
    dnnLayout_t* pLayout,
    size_t dimension,
    const size_t size[],
    const size_t strides[])
{
    return dnnLayoutCreate_F64(pLayout, dimension, size, strides);
}

template<typename T> inline
dnnError_t dnnLayoutCreateFromPrimitive(
    dnnLayout_t* pLayout,
    const dnnPrimitive_t primitive,
    dnnResourceType_t type);

template<> inline
dnnError_t dnnLayoutCreateFromPrimitive<float>(
    dnnLayout_t* pLayout,
    const dnnPrimitive_t primitive,
    dnnResourceType_t type)
{
    return dnnLayoutCreateFromPrimitive_F32(pLayout, primitive, type);
}

template<> inline
dnnError_t dnnLayoutCreateFromPrimitive<double>(
    dnnLayout_t* pLayout,
    const dnnPrimitive_t primitive,
    dnnResourceType_t type)
{
    return dnnLayoutCreateFromPrimitive_F64(pLayout, primitive, type);
}

template<typename T> inline
size_t dnnLayoutGetMemorySize(const dnnLayout_t layout);

template<> inline
size_t dnnLayoutGetMemorySize<float>(const dnnLayout_t layout) 
{
    return dnnLayoutGetMemorySize_F32(layout);
}
template<> inline
size_t dnnLayoutGetMemorySize<double>(const dnnLayout_t layout)
{
    return dnnLayoutGetMemorySize_F64(layout);
}

template<typename T> inline
int dnnLayoutCompare(const dnnLayout_t l1, const dnnLayout_t l2);

template<> inline
int dnnLayoutCompare<float>(const dnnLayout_t l1, const dnnLayout_t l2)
{
    return dnnLayoutCompare_F32(l1, l2);
}

template<> inline
int dnnLayoutCompare<double>(const dnnLayout_t l1, const dnnLayout_t l2)
{
    return dnnLayoutCompare_F64(l1, l2);
}

template<typename T> inline
dnnError_t dnnAllocateBuffer(void** pPtr, dnnLayout_t layout);

template<> inline
dnnError_t dnnAllocateBuffer<float>(void** pPtr, dnnLayout_t layout)
{
    return dnnAllocateBuffer_F32(pPtr, layout);
}

template<> inline
dnnError_t dnnAllocateBuffer<double>(void** pPtr, dnnLayout_t layout)
{
    return dnnAllocateBuffer_F64(pPtr, layout);
}

template<typename T> inline
dnnError_t dnnReleaseBuffer(void* ptr);

template<> inline
dnnError_t dnnReleaseBuffer<float>(void* ptr)
{
    return dnnReleaseBuffer_F32(ptr);
}
template<> inline
dnnError_t dnnReleaseBuffer<double>(void* ptr)
{
    return dnnReleaseBuffer_F64(ptr);
}

template<typename T> inline
dnnError_t dnnLayoutDelete(dnnLayout_t layout);
template<> inline
dnnError_t dnnLayoutDelete<float>(dnnLayout_t layout)
{
    return dnnLayoutDelete_F32(layout);
}
template<> inline
dnnError_t dnnLayoutDelete<double>(dnnLayout_t layout)
{
    return dnnLayoutDelete_F64(layout);
}

template<typename T> inline
dnnError_t dnnPrimitiveAttributesCreate(dnnPrimitiveAttributes_t* attributes);

template<> inline
dnnError_t dnnPrimitiveAttributesCreate<float>(dnnPrimitiveAttributes_t* attributes)
{
    return dnnPrimitiveAttributesCreate_F32(attributes);
}

template<> inline
dnnError_t dnnPrimitiveAttributesCreate<double>(dnnPrimitiveAttributes_t* attributes)
{
    return dnnPrimitiveAttributesCreate_F64(attributes);
}

template<typename T> inline
dnnError_t dnnPrimitiveAttributesDestroy(dnnPrimitiveAttributes_t attributes);

template<> inline
dnnError_t dnnPrimitiveAttributesDestroy<float>(dnnPrimitiveAttributes_t attributes)
{
    return dnnPrimitiveAttributesDestroy_F32(attributes);
}

template<> inline
dnnError_t dnnPrimitiveAttributesDestroy<double>(dnnPrimitiveAttributes_t attributes)
{
    return dnnPrimitiveAttributesDestroy_F64(attributes);
}

template<typename T> inline
dnnError_t dnnPrimitiveGetAttributes(dnnPrimitive_t primitive, dnnPrimitiveAttributes_t* attributes);

template<> inline
dnnError_t dnnPrimitiveGetAttributes<float>(dnnPrimitive_t primitive, dnnPrimitiveAttributes_t* attributes)
{
    return dnnPrimitiveGetAttributes_F32(primitive, attributes);
}

template<> inline
dnnError_t dnnPrimitiveGetAttributes<double>(dnnPrimitive_t primitive, dnnPrimitiveAttributes_t* attributes)
{
    return dnnPrimitiveGetAttributes_F64(primitive, attributes);
}

template<typename T> inline
dnnError_t dnnExecute(dnnPrimitive_t primitive, void* resources[]);

template<> inline
dnnError_t dnnExecute<float>(dnnPrimitive_t primitive, void* resources[])
{
    return dnnExecute_F32(primitive, resources);
}

template<> inline
dnnError_t dnnExecute<double>(dnnPrimitive_t primitive, void* resources[])
{
    return dnnExecute_F64(primitive, resources);
}

template<typename T> inline
dnnError_t dnnExecuteAsync(dnnPrimitive_t primitive, void* resources[]);

template<> inline
dnnError_t dnnExecuteAsync<float>(dnnPrimitive_t primitive, void* resources[])
{
    return dnnExecuteAsync_F32(primitive, resources);
}

template<> inline
dnnError_t dnnExecuteAsync<double>(dnnPrimitive_t primitive, void* resources[])
{
    return dnnExecuteAsync_F64(primitive, resources);
}

template<typename T> inline
dnnError_t dnnWaitFor(dnnPrimitive_t primitive);

template<> inline
dnnError_t dnnWaitFor<float>(dnnPrimitive_t primitive)
{
    return dnnWaitFor_F32(primitive);
}

template<> inline
dnnError_t dnnWaitFor<double>(dnnPrimitive_t primitive)
{
    return dnnWaitFor_F64(primitive);
}

template<typename T> inline
dnnError_t dnnDelete(dnnPrimitive_t primitive);

template<> inline
dnnError_t dnnDelete<float>(dnnPrimitive_t primitive)
{
    return dnnDelete_F32(primitive);
}

template<> inline
dnnError_t dnnDelete<double>(dnnPrimitive_t primitive)
{
    return dnnDelete_F64(primitive);
}

template<typename T> inline
dnnError_t dnnConversionCreate(dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to);

template<> inline
dnnError_t dnnConversionCreate<float>(dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to)
{
    return dnnConversionCreate_F32(pConversion, from, to);
}

template<> inline
dnnError_t dnnConversionCreate<double>(dnnPrimitive_t* pConversion, const dnnLayout_t from, const dnnLayout_t to)
{
    return dnnConversionCreate_F64(pConversion, from, to);
}

template<typename T> inline
dnnError_t dnnConversionExecute(dnnPrimitive_t conversion, void* from, void* to);

template<> inline
dnnError_t dnnConversionExecute<float>(dnnPrimitive_t conversion, void* from, void* to)
{
    return dnnConversionExecute_F32(conversion, from, to);
}

template<> inline
dnnError_t dnnConversionExecute<double>(dnnPrimitive_t conversion, void* from, void* to)
{
    return dnnConversionExecute_F64(conversion, from, to);
}

template<typename T> inline
dnnError_t dnnConvolutionCreateForward(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type);

template<> inline
dnnError_t dnnConvolutionCreateForward<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnConvolutionCreateForward_F32(
        pConvolution,
        attributes,
        algorithm,
        dimension,
        srcSize,
        dstSize,
        filterSize,
        convolutionStrides,
        inputOffset,
        border_type);
}

template<> inline
dnnError_t dnnConvolutionCreateForward<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnConvolutionCreateForward_F64(
        pConvolution,
        attributes,
        algorithm,
        dimension,
        srcSize,
        dstSize,
        filterSize,
        convolutionStrides,
        inputOffset,
        border_type);
}

template<typename T> inline
dnnError_t dnnConvolutionCreateForwardBias(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type);

template<> inline
dnnError_t dnnConvolutionCreateForwardBias<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnConvolutionCreateForwardBias_F32(
        pConvolution,
        attributes,
        algorithm,
        dimension,
        srcSize,
        dstSize,
        filterSize,
        convolutionStrides,
        inputOffset,
        border_type);
}

template<> inline
dnnError_t dnnConvolutionCreateForwardBias<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnConvolutionCreateForwardBias_F64(
        pConvolution,
        attributes,
        algorithm,
        dimension,
        srcSize,
        dstSize,
        filterSize,
        convolutionStrides,
        inputOffset,
        border_type);
}

template<typename T> inline
dnnError_t dnnConvolutionCreateBackwardData(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type);

template<> inline
dnnError_t dnnConvolutionCreateBackwardData<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnConvolutionCreateBackwardData_F32(
        pConvolution,
        attributes,
        algorithm,
        dimension,
        srcSize,
        dstSize,
        filterSize,
        convolutionStrides,
        inputOffset,
        border_type);
}

template<> inline
dnnError_t dnnConvolutionCreateBackwardData<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnConvolutionCreateBackwardData_F64(
        pConvolution,
        attributes,
        algorithm,
        dimension,
        srcSize,
        dstSize,
        filterSize,
        convolutionStrides,
        inputOffset,
        border_type);
}

template<typename T> inline
dnnError_t dnnConvolutionCreateBackwardFilter(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type);

template<> inline
dnnError_t dnnConvolutionCreateBackwardFilter<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnConvolutionCreateBackwardFilter_F32(
        pConvolution,
        attributes,
        algorithm,
        dimension,
        srcSize,
        dstSize,
        filterSize,
        convolutionStrides,
        inputOffset,
        border_type);
}

template<> inline
dnnError_t dnnConvolutionCreateBackwardFilter<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnConvolutionCreateBackwardFilter_F64(
        pConvolution,
        attributes,
        algorithm,
        dimension,
        srcSize,
        dstSize,
        filterSize,
        convolutionStrides,
        inputOffset,
        border_type);
}

template<typename T> inline
dnnError_t dnnConvolutionCreateBackwardBias(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t dstSize[]);

template<> inline
dnnError_t dnnConvolutionCreateBackwardBias<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t dstSize[])
{
    return dnnConvolutionCreateBackwardBias_F32(
        pConvolution, attributes, algorithm, dimension, dstSize);
}

template<> inline
dnnError_t dnnConvolutionCreateBackwardBias<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t dimension,
    const size_t dstSize[])
{
    return dnnConvolutionCreateBackwardBias_F64(
        pConvolution, attributes, algorithm, dimension, dstSize);
}

template<typename T> inline
dnnError_t dnnGroupsConvolutionCreateForward(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type);

template<> inline
dnnError_t dnnGroupsConvolutionCreateForward<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnGroupsConvolutionCreateForward_F32(
        pConvolution,
        attributes,
        algorithm,
        groups,
        dimension,
        srcSize,
        dstSize,
        filterSize,
        convolutionStrides,
        inputOffset,
        border_type);
}

template<> inline
dnnError_t dnnGroupsConvolutionCreateForward<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnGroupsConvolutionCreateForward_F64(
        pConvolution,
        attributes,
        algorithm,
        groups,
        dimension,
        srcSize,
        dstSize,
        filterSize,
        convolutionStrides,
        inputOffset,
        border_type);
}

template<typename T> inline
dnnError_t dnnGroupsConvolutionCreateForwardBias(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type);

template<> inline
dnnError_t dnnGroupsConvolutionCreateForwardBias<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnGroupsConvolutionCreateForwardBias_F32(
        pConvolution,
        attributes,
        algorithm,
        groups,
        dimension,
        srcSize,
        dstSize,
        filterSize,
        convolutionStrides,
        inputOffset,
        border_type);
}

template<> inline
dnnError_t dnnGroupsConvolutionCreateForwardBias<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnGroupsConvolutionCreateForwardBias_F64(
        pConvolution,
        attributes,
        algorithm,
        groups,
        dimension,
        srcSize,
        dstSize,
        filterSize,
        convolutionStrides,
        inputOffset,
        border_type);
}

template<typename T> inline
dnnError_t dnnGroupsConvolutionCreateBackwardData(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type);

template<> inline
dnnError_t dnnGroupsConvolutionCreateBackwardData<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnGroupsConvolutionCreateBackwardData_F32(
        pConvolution,
        attributes,
        algorithm,
        groups,
        dimension,
        srcSize,
        dstSize,
        filterSize,
        convolutionStrides,
        inputOffset,
        border_type);
}

template<> inline
dnnError_t dnnGroupsConvolutionCreateBackwardData<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnGroupsConvolutionCreateBackwardData_F64(
        pConvolution,
        attributes,
        algorithm,
        groups,
        dimension,
        srcSize,
        dstSize,
        filterSize,
        convolutionStrides,
        inputOffset,
        border_type);
}

template<typename T> inline
dnnError_t dnnGroupsConvolutionCreateBackwardFilter(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type);

template<> inline
dnnError_t dnnGroupsConvolutionCreateBackwardFilter<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnGroupsConvolutionCreateBackwardFilter_F32(
        pConvolution,
        attributes,
        algorithm,
        groups,
        dimension,
        srcSize,
        dstSize,
        filterSize,
        convolutionStrides,
        inputOffset,
        border_type);
}

template<> inline
dnnError_t dnnGroupsConvolutionCreateBackwardFilter<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t srcSize[],
    const size_t dstSize[],
    const size_t filterSize[],
    const size_t convolutionStrides[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnGroupsConvolutionCreateBackwardFilter_F64(
        pConvolution,
        attributes,
        algorithm,
        groups,
        dimension,
        srcSize,
        dstSize,
        filterSize,
        convolutionStrides,
        inputOffset,
        border_type);
}

template<typename T> inline
dnnError_t dnnGroupsConvolutionCreateBackwardBias(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t dstSize[]);

template<> inline
dnnError_t dnnGroupsConvolutionCreateBackwardBias<float>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t dstSize[])
{
    return dnnGroupsConvolutionCreateBackwardBias_F32(
        pConvolution, attributes, algorithm, groups, dimension, dstSize);
}

template<> inline
dnnError_t dnnGroupsConvolutionCreateBackwardBias<double>(
    dnnPrimitive_t* pConvolution,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t algorithm,
    size_t groups,
    size_t dimension,
    const size_t dstSize[])
{
    return dnnGroupsConvolutionCreateBackwardBias_F64(
        pConvolution, attributes, algorithm, groups, dimension, dstSize);
}

template<typename T> inline
dnnError_t dnnReLUCreateForward(
    dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    T negativeSlope);

template<> inline
dnnError_t dnnReLUCreateForward<float>(
    dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float negativeSlope)
{
    return dnnReLUCreateForward_F32(pRelu, attributes, dataLayout, negativeSlope);
}

template<> inline
dnnError_t dnnReLUCreateForward<double>(
    dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    double negativeSlope)
{
    return dnnReLUCreateForward_F64(pRelu, attributes, dataLayout, negativeSlope);
}

template<typename T> inline
dnnError_t dnnReLUCreateBackward(
    dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout,
    T negativeSlope);

template<> inline
dnnError_t dnnReLUCreateBackward<float>(
    dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout,
    float negativeSlope)
{
    return dnnReLUCreateBackward_F32(
        pRelu, attributes, diffLayout, dataLayout, negativeSlope);
}

template<> inline
dnnError_t dnnReLUCreateBackward<double>(
    dnnPrimitive_t* pRelu,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout,
    double negativeSlope)
{
    return dnnReLUCreateBackward_F64(
        pRelu, attributes, diffLayout, dataLayout, negativeSlope);
}

template<typename T> inline
dnnError_t dnnLRNCreateForward(
    dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    size_t kernel_size,
    T alpha,
    T beta,
    T k);

template<> inline
dnnError_t dnnLRNCreateForward<float>(
    dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    size_t kernel_size,
    float alpha,
    float beta,
    float k)
{
    return dnnLRNCreateForward_F32(
        pLrn, attributes, dataLayout, kernel_size, alpha, beta, k);
}

template<> inline
dnnError_t dnnLRNCreateForward<double>(
    dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    size_t kernel_size,
    double alpha,
    double beta,
    double k)
{
    return dnnLRNCreateForward_F64(
        pLrn, attributes, dataLayout, kernel_size, alpha, beta, k);
}

template<typename T> inline
dnnError_t dnnLRNCreateBackward(
    dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout,
    size_t kernel_size,
    T alpha,
    T beta,
    T k);

template<> inline
dnnError_t dnnLRNCreateBackward<float>(
    dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout,
    size_t kernel_size,
    float alpha,
    float beta,
    float k)
{
    return dnnLRNCreateBackward_F32(
        pLrn, attributes, diffLayout, dataLayout, kernel_size, alpha, beta, k);
}

template<> inline
dnnError_t dnnLRNCreateBackward<double>(
    dnnPrimitive_t* pLrn,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t diffLayout,
    const dnnLayout_t dataLayout,
    size_t kernel_size,
    double alpha,
    double beta,
    double k)
{
    return dnnLRNCreateBackward_F64(
        pLrn, attributes, diffLayout, dataLayout, kernel_size, alpha, beta, k);
}

template<typename T> inline
dnnError_t dnnPoolingCreateForward(
    dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t op,
    const dnnLayout_t srcLayout,
    const size_t kernelSize[],
    const size_t kernelStride[],
    const int inputOffset[],
    const dnnBorder_t border_type);

template<> inline
dnnError_t dnnPoolingCreateForward<float>(
    dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t op,
    const dnnLayout_t srcLayout,
    const size_t kernelSize[],
    const size_t kernelStride[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnPoolingCreateForward_F32(
        pPooling,
        attributes,
        op,
        srcLayout,
        kernelSize,
        kernelStride,
        inputOffset,
        border_type);
}

template<> inline
dnnError_t dnnPoolingCreateForward<double>(
    dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t op,
    const dnnLayout_t srcLayout,
    const size_t kernelSize[],
    const size_t kernelStride[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnPoolingCreateForward_F64(
        pPooling,
        attributes,
        op,
        srcLayout,
        kernelSize,
        kernelStride,
        inputOffset,
        border_type);
}

template<typename T> inline
dnnError_t dnnPoolingCreateBackward(
    dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t op,
    const dnnLayout_t srcLayout,
    const size_t kernelSize[],
    const size_t kernelStride[],
    const int inputOffset[],
    const dnnBorder_t border_type);

template<> inline
dnnError_t dnnPoolingCreateBackward<float>(
    dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t op,
    const dnnLayout_t srcLayout,
    const size_t kernelSize[],
    const size_t kernelStride[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnPoolingCreateBackward_F32(
        pPooling,
        attributes,
        op,
        srcLayout,
        kernelSize,
        kernelStride,
        inputOffset,
        border_type);
}

template<> inline
dnnError_t dnnPoolingCreateBackward<double>(
    dnnPrimitive_t* pPooling,
    dnnPrimitiveAttributes_t attributes,
    dnnAlgorithm_t op,
    const dnnLayout_t srcLayout,
    const size_t kernelSize[],
    const size_t kernelStride[],
    const int inputOffset[],
    const dnnBorder_t border_type)
{
    return dnnPoolingCreateBackward_F64(
        pPooling,
        attributes,
        op,
        srcLayout,
        kernelSize,
        kernelStride,
        inputOffset,
        border_type);
}

template<typename T> inline
dnnError_t dnnConcatCreate(
    dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes,
    const size_t N,
    dnnLayout_t src[]);

template<> inline
dnnError_t dnnConcatCreate<float>(
    dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes,
    const size_t N,
    dnnLayout_t src[])
{
    return dnnConcatCreate_F32(pConcat, attributes, N, src);
}

template<> inline
dnnError_t dnnConcatCreate<double>(
    dnnPrimitive_t* pConcat,
    dnnPrimitiveAttributes_t attributes,
    const size_t N,
    dnnLayout_t src[])
{
    return dnnConcatCreate_F64(pConcat, attributes, N, src);
}

template<typename T> inline
dnnError_t dnnSplitCreate(
    dnnPrimitive_t* pSplit,
    dnnPrimitiveAttributes_t attributes,
    const size_t N,
    dnnLayout_t src,
    size_t dst[]);

template<> inline
dnnError_t dnnSplitCreate<float>(
    dnnPrimitive_t* pSplit,
    dnnPrimitiveAttributes_t attributes,
    const size_t N,
    dnnLayout_t src,
    size_t dst[])
{
    return dnnSplitCreate_F32(pSplit, attributes, N, src, dst);
}

template<> inline
dnnError_t dnnSplitCreate<double>(
    dnnPrimitive_t* pSplit,
    dnnPrimitiveAttributes_t attributes,
    const size_t N,
    dnnLayout_t src,
    size_t dst[])
{
    return dnnSplitCreate_F64(pSplit, attributes, N, src, dst);
}

template<typename T> inline
dnnError_t dnnSumCreate(
    dnnPrimitive_t* pSum,
    dnnPrimitiveAttributes_t attributes,
    const size_t nSummands,
    dnnLayout_t layout,
    T* coefficients);

template<> inline
dnnError_t dnnSumCreate<float>(
    dnnPrimitive_t* pSum,
    dnnPrimitiveAttributes_t attributes,
    const size_t nSummands,
    dnnLayout_t layout,
    float* coefficients)
{
    return dnnSumCreate_F32(pSum, attributes, nSummands, layout, coefficients);
}

template<> inline
dnnError_t dnnSumCreate<double>(
    dnnPrimitive_t* pSum,
    dnnPrimitiveAttributes_t attributes,
    const size_t nSummands,
    dnnLayout_t layout,
    double* coefficients)
{
    return dnnSumCreate_F64(pSum, attributes, nSummands, layout, coefficients);
}

template<typename T> inline
dnnError_t dnnBatchNormalizationCreateForward(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    T eps);

template<> inline
dnnError_t dnnBatchNormalizationCreateForward<float>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps)
{
    return dnnBatchNormalizationCreateForward_F32(
        pBatchNormalization, attributes, dataLayout, eps);
}
template<> inline
dnnError_t dnnBatchNormalizationCreateForward<double>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    double eps)
{
    return dnnBatchNormalizationCreateForward_F64(
        pBatchNormalization, attributes, dataLayout, eps);
}

template<typename T> inline
dnnError_t dnnBatchNormalizationCreateBackwardData(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    T eps);

template<> inline
dnnError_t dnnBatchNormalizationCreateBackwardData<float>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps)
{
    return dnnBatchNormalizationCreateBackwardData_F32(
        pBatchNormalization, attributes, dataLayout, eps);
}

template<> inline
dnnError_t dnnBatchNormalizationCreateBackwardData<double>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    double eps)
{
    return dnnBatchNormalizationCreateBackwardData_F64(
        pBatchNormalization, attributes, dataLayout, eps);
}

template<typename T> inline
dnnError_t dnnBatchNormalizationCreateBackwardScaleShift(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    T eps);

template<> inline
dnnError_t
    dnnBatchNormalizationCreateBackwardScaleShift<float>(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout,
        float eps)
{
    return dnnBatchNormalizationCreateBackwardScaleShift_F32(
        pBatchNormalization, attributes, dataLayout, eps);
}
template<> inline
dnnError_t
    dnnBatchNormalizationCreateBackwardScaleShift<double>(
        dnnPrimitive_t* pBatchNormalization,
        dnnPrimitiveAttributes_t attributes,
        const dnnLayout_t dataLayout,
        double eps)
{
    return dnnBatchNormalizationCreateBackwardScaleShift_F64(
        pBatchNormalization, attributes, dataLayout, eps);
}

template<typename T> inline
dnnError_t dnnBatchNormalizationCreateForward_v2(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    T eps,
    unsigned int flags);

template<> inline
dnnError_t dnnBatchNormalizationCreateForward_v2<float>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps,
    unsigned int flags)
{
    return dnnBatchNormalizationCreateForward_v2_F32(
        pBatchNormalization, attributes, dataLayout, eps, flags);
}

template<> inline
dnnError_t dnnBatchNormalizationCreateForward_v2<double>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    double eps,
    unsigned int flags)
{
    return dnnBatchNormalizationCreateForward_v2_F64(
        pBatchNormalization, attributes, dataLayout, eps, flags);
}

template<typename T> inline
dnnError_t dnnBatchNormalizationCreateBackward_v2(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    T eps,
    unsigned int flags);

template<> inline
dnnError_t dnnBatchNormalizationCreateBackward_v2<float>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    float eps,
    unsigned int flags)
{
    return dnnBatchNormalizationCreateBackward_v2_F32(
        pBatchNormalization, attributes, dataLayout, eps, flags);
}

template<> inline
dnnError_t dnnBatchNormalizationCreateBackward_v2<double>(
    dnnPrimitive_t* pBatchNormalization,
    dnnPrimitiveAttributes_t attributes,
    const dnnLayout_t dataLayout,
    double eps,
    unsigned int flags)
{
    return dnnBatchNormalizationCreateBackward_v2_F64(
        pBatchNormalization, attributes, dataLayout, eps, flags);
}

template<typename T> inline
dnnError_t dnnInnerProductCreateForward(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels);

template<> inline
dnnError_t dnnInnerProductCreateForward<float>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels)
{
    return dnnInnerProductCreateForward_F32(
        pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}
template<> inline
dnnError_t dnnInnerProductCreateForward<double>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels)
{
    return dnnInnerProductCreateForward_F64(
        pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

template<typename T> inline
dnnError_t dnnInnerProductCreateForwardBias(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels);

template<> inline
dnnError_t dnnInnerProductCreateForwardBias<float>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels)
{
    return dnnInnerProductCreateForwardBias_F32(
        pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

template<> inline
dnnError_t dnnInnerProductCreateForwardBias<double>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels)
{
    return dnnInnerProductCreateForwardBias_F64(
        pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

template<typename T> inline
dnnError_t dnnInnerProductCreateBackwardData(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels);

template<> inline
dnnError_t dnnInnerProductCreateBackwardData<float>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels)
{
    return dnnInnerProductCreateBackwardData_F32(
        pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}
template<> inline
dnnError_t dnnInnerProductCreateBackwardData<double>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels)
{
    return dnnInnerProductCreateBackwardData_F64(
        pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

template<typename T> inline
dnnError_t dnnInnerProductCreateBackwardFilter(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels);

template<> inline
dnnError_t dnnInnerProductCreateBackwardFilter<float>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels)
{
    return dnnInnerProductCreateBackwardFilter_F32(
        pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}
template<> inline
dnnError_t dnnInnerProductCreateBackwardFilter<double>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[],
    size_t outputChannels)
{
    return dnnInnerProductCreateBackwardFilter_F64(
        pInnerProduct, attributes, dimensions, srcSize, outputChannels);
}

template<typename T> inline
dnnError_t dnnInnerProductCreateBackwardBias(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[]);

template<> inline
dnnError_t dnnInnerProductCreateBackwardBias<float>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[])
{
    return dnnInnerProductCreateBackwardBias_F32(
        pInnerProduct, attributes, dimensions, srcSize);
}

template<> inline
dnnError_t dnnInnerProductCreateBackwardBias<double>(
    dnnPrimitive_t* pInnerProduct,
    dnnPrimitiveAttributes_t attributes,
    size_t dimensions,
    const size_t srcSize[])
{
    return dnnInnerProductCreateBackwardBias_F64(
        pInnerProduct, attributes, dimensions, srcSize);
}

template<typename T> inline
void cblas_axpby(const MKL_INT n, const T alpha, const T* x, const T beta, T* y);

template<> inline
void cblas_axpby<float>(const MKL_INT n, const float alpha, const float* x, const float beta, float* y)
{
    cblas_saxpby(n, alpha, x, 1, beta, y, 1);
}

template<> inline
void cblas_axpby<double>(const MKL_INT n, const double alpha, const double* x, const double beta, double* y)
{
    cblas_daxpby(n, alpha, x, 1, beta, y, 1);
}

inline void CHECK_MKL(dnnError_t err)
{
    if (err != E_SUCCESS)
        RuntimeError("mkl err (%d)\n", err);
}

template<typename ElemType>
inline void dnnSafeDelete(dnnLayout_t& lt)
{
    if (lt)
    {
        dnnLayoutDelete<ElemType>(lt);
        lt = nullptr;
    }
}

template<typename ElemType>
inline void dnnSafeDelete(dnnPrimitive_t& prim)
{
    if (prim)
    {
        dnnDelete<ElemType>(prim);
        prim = nullptr;
    }
}

template<typename ElemType>
inline void dnnSafeDelete(void*& buffer)
{
    if (buffer)
    {
        dnnReleaseBuffer<ElemType>(buffer);
        buffer = nullptr;
    }
}

// adapter that converts data between user layout and primitive required layout
template<typename ElemType>
class MKLDnnResourceAdapter
{
    dnnLayout_t userLayout = nullptr;
    dnnLayout_t primLayout = nullptr;
    dnnPrimitive_t convertPrim = nullptr;
    bool isInput;
    void* tempBuffer = nullptr;
    dnnResourceType_t resourceType;
public:
    // consumes ltUser and ltPrim
    void Create(dnnLayout_t& ltUser, dnnLayout_t& ltPrim, dnnResourceType_t rt, bool userToPrim)
    {
        Clear();
        convertPrim = nullptr;
        tempBuffer = nullptr;
        isInput = userToPrim;
        resourceType = rt;
        if (!dnnLayoutCompare<ElemType>(ltUser, ltPrim))
        {
            userLayout = ltUser;
            primLayout = ltPrim;
            dnnLayout_t from = userToPrim ? ltUser : ltPrim;
            dnnLayout_t to = userToPrim ? ltPrim : ltUser;
            CHECK_MKL(dnnConversionCreate<ElemType>(&convertPrim, from, to));
            CHECK_MKL(dnnAllocateBuffer<ElemType>(&tempBuffer, ltPrim)); // always allocate temp buffer for primLayout
            ltUser = nullptr;
            ltPrim = nullptr;
        }
        dnnSafeDelete<ElemType>(ltUser);
        dnnSafeDelete<ElemType>(ltPrim);
    }

    void PrepareForExecution(void* userData, void* resources[dnnResourceNumber])
    {
        if (isInput)
        {
            if (convertPrim)
            {
                CHECK_MKL(dnnConversionExecute<ElemType>(convertPrim, userData, tempBuffer));
                resources[resourceType] = tempBuffer;
            }
            else
                resources[resourceType] = userData;
        }
        else
        {
            resources[resourceType] = convertPrim ? tempBuffer : userData;
        }
    }

    void ConvertOutput(void* userData)
    {
        if (isInput)
            RuntimeError("Cannot execute output ResourceAdapter for input");

        if (convertPrim)
            CHECK_MKL(dnnConversionExecute<ElemType>(convertPrim, tempBuffer, userData));
    }

    void Clear()
    {
        dnnSafeDelete<ElemType>(convertPrim);
        dnnSafeDelete<ElemType>(userLayout);
        dnnSafeDelete<ElemType>(primLayout);
        dnnSafeDelete<ElemType>(tempBuffer);
    }

    ~MKLDnnResourceAdapter()
    {
        Clear();
    }
};

} } }

#endif