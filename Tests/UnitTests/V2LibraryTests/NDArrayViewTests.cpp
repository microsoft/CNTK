//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "CNTKLibrary.h"
#include "Common.h"
#include <functional>
#include <array>

using namespace CNTK;

template <typename ElementType>
void TestNDArrayView(size_t numAxes, const DeviceDescriptor& device)
{
    srand(1);

    size_t maxDimSize = 15;
    NDShape viewShape(numAxes);
    for (size_t i = 0; i < numAxes; ++i)
        viewShape[i] = (rand() % maxDimSize) + 1;

    // Create a NDArrayView over a std::array
    std::array<ElementType, 1> arrayData = { 3 };
    auto arrayDataView = MakeSharedObject<NDArrayView>(NDShape({}), arrayData);
    if (arrayDataView->template DataBuffer<ElementType>() != arrayData.data())
        throw std::runtime_error("The DataBuffer of the NDArrayView does not match the original buffer it was created over");

    std::vector<ElementType> data(viewShape.TotalSize());
    ElementType scale = 19.0;
    ElementType offset = -4.0;
    for (size_t i = 0; i < viewShape.TotalSize(); ++i)
        data[i] = offset + ((((ElementType)rand()) / RAND_MAX) * scale);

    auto cpuDataView = MakeSharedObject<NDArrayView>(viewShape, data);
    if (cpuDataView->template DataBuffer<ElementType>() != data.data())
        throw std::runtime_error("The DataBuffer of the NDArrayView does not match the original buffer it was created over");

    NDArrayViewPtr dataView;
    if ((device.Type() == DeviceKind::CPU))
        dataView = cpuDataView;
    else
    {
        dataView = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), viewShape, device);
        dataView->CopyFrom(*cpuDataView);
    }

    if (dataView->Device() != device)
        throw std::runtime_error("Device of NDArrayView does not match 'device' it was created on");

    // Test clone
    auto clonedView = dataView->DeepClone(false);
    ElementType* first = nullptr;
    const ElementType* second = cpuDataView->template DataBuffer<ElementType>();
    NDArrayViewPtr temp1CpuDataView, temp2CpuDataView;
    if ((device.Type() == DeviceKind::CPU))
    {
        if (dataView->DataBuffer<ElementType>() != data.data())
            throw std::runtime_error("The DataBuffer of the NDArrayView does not match the original buffer it was created over");

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
        if (first[i] != second[i])
            throw std::runtime_error("The contents of the clone do not match expected");
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

    if (first[0] != (second[0] + 1))
        throw std::runtime_error("The clonedView's contents do not match expected");

    // Test alias
    auto aliasView = clonedView->Alias(true);
    const ElementType* aliasViewBuffer = aliasView->DataBuffer<ElementType>();
    const ElementType* clonedDataBuffer = clonedView->DataBuffer<ElementType>();
    if (aliasViewBuffer != clonedDataBuffer)
        throw std::runtime_error("The buffers underlying the alias view and the view it is an alias of are different!");

    clonedView->CopyFrom(*dataView);
    if (aliasViewBuffer != clonedDataBuffer)
        throw std::runtime_error("The buffers underlying the alias view and the view it is an alias of are different!");

    // Test readonliness
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
void TestSparseCSCArrayView(size_t numAxes, const DeviceDescriptor& device)
{
    srand(1);

    size_t maxDimSize = 15;
    NDShape viewShape(numAxes);
    for (size_t i = 0; i < numAxes; ++i)
        viewShape[i] = (rand() % maxDimSize) + 1;

    size_t numMatrixCols = (numAxes > 0) ? viewShape.SubShape(1).TotalSize() : 1;
    size_t numMatrixRows = (numAxes > 0) ? viewShape[0] : 1;
    std::unique_ptr<int[]> colsStarts(new int[numMatrixCols + 1]);
    colsStarts[0] = 0;
    int numNonZeroValues = 0;
    for (size_t i = 1; i <= numMatrixCols; ++i)
    {
        int numValuesInCurrentCol = (rand() % numMatrixRows) + (rand() % 1);
        numNonZeroValues += numValuesInCurrentCol;
        colsStarts[i] = colsStarts[i - 1] + numValuesInCurrentCol;
    }

    // Now fill the actual values
    std::unique_ptr<ElementType[]> nonZeroValues(new ElementType[numNonZeroValues]);
    std::unique_ptr<int[]> rowIndices(new int[numNonZeroValues]);
    size_t nnzIndex = 0;
    std::vector<ElementType> referenceDenseData(viewShape.TotalSize(), 0);
    for (size_t j = 0; j < numMatrixCols; ++j)
    {
        size_t numRowsWithValuesInCurrentCol = colsStarts[j + 1] - colsStarts[j];
        size_t numValuesWritten = 0;
        std::unordered_set<int> rowsWrittenTo;
        while (numValuesWritten < numRowsWithValuesInCurrentCol)
        {
            int rowIndex = rand() % numMatrixRows;
            if (rowsWrittenTo.insert(rowIndex).second)
            {
                ElementType value = ((ElementType)rand()) / RAND_MAX;
                nonZeroValues[nnzIndex] = value;
                referenceDenseData[(j * numMatrixRows) + rowIndex] = value;
                rowIndices[nnzIndex] = rowIndex;
                numValuesWritten++;
                nnzIndex++;
            }
        }
    }

    NDArrayView sparseCSCArrayView(viewShape, colsStarts.get(), rowIndices.get(), nonZeroValues.get(), numNonZeroValues, device, true);

    // Copy it out to a dense matrix on the CPU and verify the data
    std::vector<ElementType> copiedDenseData(viewShape.TotalSize());
    NDArrayView denseCPUTensor(viewShape, copiedDenseData.data(), copiedDenseData.size(), DeviceDescriptor::CPUDevice());
    denseCPUTensor.CopyFrom(sparseCSCArrayView);
    if (copiedDenseData != referenceDenseData)
        throw std::runtime_error("The contents of the dense vector that the sparse NDArrayView is copied into do not match the expected values");

    NDArrayView emptySparseCSCArrayView(AsDataType<ElementType>(), StorageFormat::SparseCSC, viewShape, device);
    emptySparseCSCArrayView.CopyFrom(denseCPUTensor);
    NDArrayView newDenseCPUTensor(viewShape, copiedDenseData.data(), copiedDenseData.size(), DeviceDescriptor::CPUDevice());
    newDenseCPUTensor.CopyFrom(emptySparseCSCArrayView);
    if (copiedDenseData != referenceDenseData)
        throw std::runtime_error("The contents of the dense vector that the sparse NDArrayView is copied into do not match the expected values");
}

void NDArrayViewTests()
{
    TestNDArrayView<float>(2, DeviceDescriptor::CPUDevice());

    if (IsGPUAvailable())
    {
        TestNDArrayView<float>(0, DeviceDescriptor::GPUDevice(0));
        TestNDArrayView<double>(4, DeviceDescriptor::GPUDevice(0));

        TestSparseCSCArrayView<float>(1, DeviceDescriptor::GPUDevice(0));
        TestSparseCSCArrayView<double>(4, DeviceDescriptor::GPUDevice(0));
    }

    TestSparseCSCArrayView<float>(2, DeviceDescriptor::CPUDevice());
}
