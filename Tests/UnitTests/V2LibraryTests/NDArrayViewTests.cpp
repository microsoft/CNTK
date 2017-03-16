//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Common.h"
#include <functional>
#include <array>

using namespace CNTK;

namespace CNTK { namespace Test {

template <typename ElementType>
void CreateNDArrayViewOverStdArray()
{
    std::array<ElementType, 1> arrayData = { 3 };
    auto arrayDataView = MakeSharedObject<NDArrayView>(NDShape({}), arrayData);
    BOOST_TEST(arrayDataView->template DataBuffer<ElementType>() == arrayData.data(),
        "The DataBuffer of the NDArrayView does not match the original buffer it was created over");
}

template <typename ElementType>
NDArrayViewPtr GetClonedView(const NDShape viewShape, const NDArrayViewPtr dataView, const NDArrayViewPtr cpuDataView, const std::vector<ElementType> & data, const DeviceDescriptor& device)
{
    auto clonedView = dataView->DeepClone(false);
    ElementType* first = nullptr;
    const ElementType* second = cpuDataView->template DataBuffer<ElementType>();
    NDArrayViewPtr temp1CpuDataView, temp2CpuDataView;
    if ((device.Type() == DeviceKind::CPU))
    {

        BOOST_TEST(dataView->DataBuffer<ElementType>() == data.data(), "The DataBuffer of the NDArrayView does not match the original buffer it was created over");

        first = clonedView->WritableDataBuffer<ElementType>();
    }
    else
    {
        temp1CpuDataView = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), viewShape, DeviceDescriptor::CPUDevice());
        temp1CpuDataView->CopyFrom(*clonedView);

        first = temp1CpuDataView->WritableDataBuffer<ElementType>();
    }

    for (size_t i = 0; i < viewShape.TotalSize(); ++i)
    {
        BOOST_TEST(first[i] == second[i], "The contents of the clone do not match expected");
    }

    first[0] += 1;
    if ((device.Type() != DeviceKind::CPU))
        clonedView->CopyFrom(*temp1CpuDataView);

    if ((device.Type() == DeviceKind::CPU))
    {
        first = clonedView->WritableDataBuffer<ElementType>();
        second = dataView->DataBuffer<ElementType>();
    }
    else
    {
        temp1CpuDataView = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), viewShape, DeviceDescriptor::CPUDevice());
        temp1CpuDataView->CopyFrom(*clonedView);
        first = temp1CpuDataView->WritableDataBuffer<ElementType>();

        temp2CpuDataView = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), viewShape, DeviceDescriptor::CPUDevice());
        temp2CpuDataView->CopyFrom(*dataView);
        second = temp2CpuDataView->DataBuffer<ElementType>();
    }

    BOOST_TEST(first[0] == (second[0] + 1), "The clonedView's contents do not match expected");

    return clonedView;
}

template <typename ElementType>
NDArrayViewPtr GetAliasView(const NDArrayViewPtr clonedView, const NDArrayViewPtr dataView)
{
    auto aliasView = clonedView->Alias(true);
    const ElementType* aliasViewBuffer = aliasView->DataBuffer<ElementType>();
    const ElementType* clonedDataBuffer = clonedView->DataBuffer<ElementType>();

    auto errorMsg = "The buffers underlying the alias view and the view it is an alias of are different!";

    BOOST_TEST(aliasViewBuffer == clonedDataBuffer, errorMsg);

    clonedView->CopyFrom(*dataView);

    BOOST_TEST(aliasViewBuffer == clonedDataBuffer, errorMsg);

    return aliasView;
}

template <typename ElementType>
void TestReadonliness(const NDArrayViewPtr aliasView, const NDArrayViewPtr dataView){
    auto errorMsg = "Was incorrectly able to get a writable buffer pointer from a readonly view";
    // Should not be able to get the WritableDataBuffer for a read-only view
    VerifyException([&aliasView]() {
        ElementType* aliasViewBuffer = aliasView->WritableDataBuffer<ElementType>();
        aliasViewBuffer;
    }, errorMsg);

    // Should not be able to copy into a read-only view
    VerifyException([&aliasView, &dataView]() {
        aliasView->CopyFrom(*dataView);
    }, errorMsg);
}

template <typename ElementType>
void TestNDArrayView(size_t numAxes, const DeviceDescriptor& device)
{
    srand(1);

    size_t maxDimSize = 15;
    NDShape viewShape(numAxes);
    for (size_t i = 0; i < numAxes; ++i)
        viewShape[i] = (rand() % maxDimSize) + 1;

    CreateNDArrayViewOverStdArray<ElementType>();

    std::vector<ElementType> data(viewShape.TotalSize());
    ElementType scale = 19.0;
    ElementType offset = -4.0;
    for (size_t i = 0; i < viewShape.TotalSize(); ++i)
        data[i] = offset + ((((ElementType)rand()) / RAND_MAX) * scale);

    auto cpuDataView = MakeSharedObject<NDArrayView>(viewShape, data);
    BOOST_TEST((cpuDataView->template DataBuffer<ElementType>() == data.data()),
        "The DataBuffer of the NDArrayView does not match the original buffer it was created over");

    NDArrayViewPtr dataView;
    if ((device.Type() == DeviceKind::CPU))
        dataView = cpuDataView;
    else
    {
        dataView = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), viewShape, device);
        dataView->CopyFrom(*cpuDataView);
    }

    BOOST_TEST((dataView->Device() == device), "Device of NDArrayView does not match 'device' it was created on");

    auto clonedView = GetClonedView<ElementType>(viewShape, dataView, cpuDataView, data, device);

    auto aliasView = GetAliasView<ElementType>(clonedView, dataView);

    TestReadonliness<ElementType>(aliasView, dataView);
}

template <typename ElementType>
void TestSparseCSCArrayView(size_t numAxes, const DeviceDescriptor& device)
{
    srand(1);

    size_t maxDimSize = 15;
    NDShape viewShape(numAxes);
    for (size_t i = 0; i < numAxes; ++i)
        viewShape[i] = (rand() % maxDimSize) + 1;

    size_t numMatrixCols = (numAxes > 0) ? viewShape.SubShape(1).TotalSize() : 1;
    size_t numMatrixRows = (numAxes > 0) ? viewShape[0] : 1;

    std::vector<ElementType> referenceDenseData;
    std::vector<SparseIndexType> colsStarts;
    std::vector<SparseIndexType> rowIndices;
    std::vector<ElementType> nonZeroValues;
    size_t numNonZeroValues;
    std::tie(referenceDenseData, colsStarts, rowIndices, nonZeroValues, numNonZeroValues) = GenerateSequenceInCSC<ElementType>(numMatrixRows, numMatrixCols);

    NDArrayView sparseCSCArrayView(viewShape, colsStarts.data(), rowIndices.data(), nonZeroValues.data(), numNonZeroValues, device, true);

    // Copy it out to a dense matrix on the CPU and verify the data
    std::vector<ElementType> copiedDenseData(viewShape.TotalSize());
    NDArrayView denseCPUTensor(viewShape, copiedDenseData.data(), copiedDenseData.size(), DeviceDescriptor::CPUDevice());
    denseCPUTensor.CopyFrom(sparseCSCArrayView);

    BOOST_TEST(copiedDenseData == referenceDenseData, "The contents of the dense vector that the sparse NDArrayView is copied into do not match the expected values");

    NDArrayView emptySparseCSCArrayView(AsDataType<ElementType>(), StorageFormat::SparseCSC, viewShape, device);
    emptySparseCSCArrayView.CopyFrom(denseCPUTensor);
    NDArrayView newDenseCPUTensor(viewShape, copiedDenseData.data(), copiedDenseData.size(), DeviceDescriptor::CPUDevice());
    newDenseCPUTensor.CopyFrom(emptySparseCSCArrayView);

    BOOST_TEST(copiedDenseData == referenceDenseData, "The contents of the dense vector that the sparse NDArrayView is copied into do not match the expected values");
}

BOOST_AUTO_TEST_SUITE(NDArrayViewSuite)

BOOST_AUTO_TEST_CASE(CheckFloatNDArrayViewInCpu)
{
    if (ShouldRunOnCpu())
        TestNDArrayView<float>(2, DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(CheckNDArrayViewInGpu)
{
    if (ShouldRunOnGpu())
    {
        TestNDArrayView<float>(0, DeviceDescriptor::GPUDevice(0));
        TestNDArrayView<double>(4, DeviceDescriptor::GPUDevice(0));
    }
}

BOOST_AUTO_TEST_CASE(CheckCscArrayViewInGpu)
{
    if (ShouldRunOnGpu())
    {
        TestSparseCSCArrayView<float>(1, DeviceDescriptor::GPUDevice(0));
        TestSparseCSCArrayView<double>(4, DeviceDescriptor::GPUDevice(0));
    }
}

BOOST_AUTO_TEST_CASE(CheckCscArrayViewInCpu)
{
    if (ShouldRunOnCpu())
        TestSparseCSCArrayView<float>(2, DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_SUITE_END()

}}
