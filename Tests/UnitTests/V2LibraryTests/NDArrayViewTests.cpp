//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#include <inttypes.h>
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

    // Create an empty sparse CSC NDArrayView and the copy data from a dense NDArrayView on CPU.
    // Then copy the data in the sparse CSC one into another CPU dense one for checking results.
    NDArrayView emptySparseCSCArrayView(AsDataType<ElementType>(), StorageFormat::SparseCSC, viewShape, device);
    emptySparseCSCArrayView.CopyFrom(denseCPUTensor);
    NDArrayView newDenseCPUTensor(viewShape, copiedDenseData.data(), copiedDenseData.size(), DeviceDescriptor::CPUDevice());
    newDenseCPUTensor.CopyFrom(emptySparseCSCArrayView);

    BOOST_TEST(copiedDenseData == referenceDenseData, "The contents of the dense vector that the sparse NDArrayView is copied into do not match the expected values");
}

template <typename ElementType>
void TestSparseCSCDataBuffers(size_t numAxes, const DeviceDescriptor& device)
{
    size_t maxDimSize = 15;
    NDShape viewShape(numAxes);
    for (size_t i = 0; i < numAxes; ++i)
        viewShape[i] = (rand() % maxDimSize) + 1;

    size_t numMatrixCols = (numAxes > 0) ? viewShape.SubShape(1).TotalSize() : 1;
    size_t numMatrixRows = (numAxes > 0) ? viewShape[0] : 1;

    std::vector<ElementType> referenceDenseData;
    std::vector<SparseIndexType> expectedColsStarts;
    std::vector<SparseIndexType> expectedRowIndices;
    std::vector<ElementType> expectedNonZeroValues;
    size_t expectedNumNonZeroValues;
    std::tie(referenceDenseData, expectedColsStarts, expectedRowIndices, expectedNonZeroValues, expectedNumNonZeroValues) = GenerateSequenceInCSC<ElementType>(numMatrixRows, numMatrixCols);

    NDArrayView sparseCSCArrayView(viewShape, expectedColsStarts.data(), expectedRowIndices.data(), expectedNonZeroValues.data(), expectedNumNonZeroValues, device, true);

    const ElementType *outputNonZeroData;
    const SparseIndexType *outputColsStartsData;
    const SparseIndexType *outputRowIndicesData;
    size_t outputNumNonZeroData;
    std::tie(outputNonZeroData, outputColsStartsData, outputRowIndicesData, outputNumNonZeroData) = sparseCSCArrayView.SparseCSCDataBuffers<ElementType>();

    if (expectedNumNonZeroValues != outputNumNonZeroData)
        ReportFailure("The number of non-zero values does not match. expected: %" PRIu64 ", actual %" PRIu64 "\n", expectedNumNonZeroValues, outputNumNonZeroData);
    if (expectedNonZeroValues.size() != outputNumNonZeroData)
        ReportFailure("The number of non-zero values returned does not match that in the non-zero value buffers. expected: %" PRIu64 ", actual %" PRIu64 "\n", expectedNonZeroValues.size(), outputNumNonZeroData);

    if (device.Type() == DeviceKind::CPU)
    {
        for (size_t i = 0; i < expectedNonZeroValues.size(); i++)
            if (expectedNonZeroValues[i] != outputNonZeroData[i])
                ReportFailure("The non-zero value at position %" PRIu64 " does not match, expected: %f, actual: %f\n", i, expectedNonZeroValues[i] , outputNonZeroData[i]);
        for (size_t i=0; i < expectedColsStarts.size(); i++)
            if (expectedColsStarts[i] != outputColsStartsData[i])
                ReportFailure("The ColsStarts at position %" PRIu64 " does not match, expected: %f, actual: %f\n", i, expectedColsStarts[i], outputColsStartsData[i]);
        for (size_t i = 0; i < expectedRowIndices.size(); i++)
            if (expectedRowIndices[i] != outputRowIndicesData[i])
                ReportFailure("The RowIndices at position %" PRIu64 " does not match, expected: %f, actual: %f\n", i, expectedRowIndices[i], outputRowIndicesData[i]);
    }
    else
    {
        if (device.Type() != DeviceKind::GPU)
            ReportFailure("The device type of the NDArrayView is neither CPU nor GPU.");

        // The data buffers returned by SparseCSCDataBuffers() are on GPU, and there is no existing utility functions that compare data buffers between GPU and CPU (without adding
        // Cuda dependency here), we take a workaround here: first build another NDArrayView using the returned data buffers, copy it to another one on CPU but using dense format,
        // and then compare the data with the expected one.
        // Another limitation is here is NDArrayView::DeepClone() does not support the GPUSparseMatrix->CPUSparseMatrix, since Matrix::AssignValuesOf() has not implemented this feature
        // yet. This prevents from using AreEqual(NDArrayViewPtr, NDArrayViewPtr).
        NDArrayView viewFromOutput(viewShape, outputColsStartsData, outputRowIndicesData, outputNonZeroData, outputNumNonZeroData, device, true);

        // Copy it out to a dense matrix on the CPU and verify the data
        std::vector<ElementType> copiedDenseData(viewShape.TotalSize());
        NDArrayView denseCPUTensor(viewShape, copiedDenseData.data(), copiedDenseData.size(), DeviceDescriptor::CPUDevice());
        denseCPUTensor.CopyFrom(viewFromOutput);
        BOOST_TEST(copiedDenseData == referenceDenseData, "The contents of the dense vector that the sparse NDArrayView is copied into do not match the expected values");
    }
}

template <typename ElementType>
void TestDataBuffer(size_t numAxes, const DeviceDescriptor& device)
{
    size_t maxDimSize = 15;
    NDShape viewShape(numAxes);
    for (size_t i = 0; i < numAxes; ++i)
        viewShape[i] = (rand() % maxDimSize) + 1;

    std::vector<ElementType> data(viewShape.TotalSize());
    ElementType scale = 12.0;
    ElementType offset = -3.0;
    for (size_t i = 0; i < viewShape.TotalSize(); ++i)
        data[i] = offset + ((((ElementType)rand()) / RAND_MAX) * scale);

    auto cpuDataView = MakeSharedObject<NDArrayView>(viewShape, data.data(), data.size(), DeviceDescriptor::CPUDevice());
    NDArrayViewPtr dataView;
    if (device.Type() == DeviceKind::CPU)
    {
        dataView = cpuDataView;
    }
    else
    {
        BOOST_TEST((device.Type() == DeviceKind::GPU), "Device type must be CPU or GPU.");
        dataView = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), viewShape, device);
        dataView->CopyFrom(*cpuDataView);
    }

    auto dataBuffer = dataView->template DataBuffer<ElementType>();

    if ((device.Type() == DeviceKind::CPU))
    {
        BOOST_TEST(memcmp(dataBuffer, data.data(), data.size()), "DataBuffer of NDArrayView on CPU does not match the source data.");
    }
    else
    {
        // We cannot directly compare dataBuffer on GPU with the buffer on CPU. Instead, we construct an NDArrayView from the dataBuffer
        // and compare the views.
        auto dataViewFromOutput = MakeSharedObject<NDArrayView>(viewShape, dataBuffer, viewShape.TotalSize(), device);
        BOOST_TEST(AreEqual(dataViewFromOutput, dataView), "DataBuffer of NDArrayView on GPU does not match the source data.");
    }
}

struct NDArrayViewFixture
{
    NDArrayViewFixture()
    {
        srand(1);
    }
};

BOOST_FIXTURE_TEST_SUITE(NDArrayViewSuite, NDArrayViewFixture)

BOOST_AUTO_TEST_CASE(CheckFloatNDArrayViewInCpu)
{
    if (ShouldRunOnCpu())
        TestNDArrayView<float>(GenerateNumOfAxes(10), DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(CheckNDArrayViewInGpu)
{
    if (ShouldRunOnGpu())
    {
        TestNDArrayView<float>(0, DeviceDescriptor::GPUDevice(0));
        TestNDArrayView<double>(GenerateNumOfAxes(6), DeviceDescriptor::GPUDevice(0));
    }
}

BOOST_AUTO_TEST_CASE(CheckCscArrayViewInGpu)
{
    if (ShouldRunOnGpu())
    {
        TestSparseCSCArrayView<float>(1, DeviceDescriptor::GPUDevice(0));
        TestSparseCSCArrayView<double>(GenerateNumOfAxes(9), DeviceDescriptor::GPUDevice(0));
    }
}

BOOST_AUTO_TEST_CASE(CheckCscArrayViewInCpu)
{
    if (ShouldRunOnCpu())
    {
        TestSparseCSCArrayView<float>(2, DeviceDescriptor::CPUDevice());
        TestSparseCSCArrayView<float>(GenerateNumOfAxes(15), DeviceDescriptor::CPUDevice());
    }
}

BOOST_AUTO_TEST_CASE(CheckSparseCscDataBuffersInGpu)
{
    if (ShouldRunOnGpu())
    {
        TestSparseCSCDataBuffers<float>(1, DeviceDescriptor::GPUDevice(0));
        TestSparseCSCDataBuffers<double>(GenerateNumOfAxes(7), DeviceDescriptor::GPUDevice(0));
    }
}

BOOST_AUTO_TEST_CASE(CheckSparseCscDataBuffersInCpu)
{
    if (ShouldRunOnCpu())
    {
        TestSparseCSCDataBuffers<float>(2, DeviceDescriptor::CPUDevice());
        TestSparseCSCDataBuffers<float>(GenerateNumOfAxes(16), DeviceDescriptor::CPUDevice());
    }
}

BOOST_AUTO_TEST_CASE(CheckDataBufferInCpu)
{
    if (ShouldRunOnCpu())
        TestDataBuffer<float>(GenerateNumOfAxes(5), DeviceDescriptor::CPUDevice());
}

BOOST_AUTO_TEST_CASE(CheckDataBufferInGpu)
{
    if (ShouldRunOnGpu())
    {
        TestDataBuffer<float>(0, DeviceDescriptor::GPUDevice(0));
        TestDataBuffer<double>(GenerateNumOfAxes(11), DeviceDescriptor::GPUDevice(0));
    }
}


BOOST_AUTO_TEST_SUITE_END()

}}
