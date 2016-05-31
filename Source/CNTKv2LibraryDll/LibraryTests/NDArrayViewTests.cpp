#include "CNTKLibrary.h"
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
    auto arrayDataView = new NDArrayView({}, arrayData);
    if (arrayDataView->DataBuffer<ElementType>() != arrayData.data())
        throw std::exception("The DataBuffer of the NDArrayView does not match the original buffer it was created over");

    std::vector<ElementType> data(viewShape.TotalSize());
    ElementType scale = 19.0;
    ElementType offset = -4.0;
    for (size_t i = 0; i < viewShape.TotalSize(); ++i)
        data[i] = offset + ((((ElementType)rand()) / RAND_MAX) * scale);

    auto cpuDataView = new NDArrayView(viewShape, data);
    if (cpuDataView->DataBuffer<ElementType>() != data.data())
        throw std::exception("The DataBuffer of the NDArrayView does not match the original buffer it was created over");

    NDArrayViewPtr dataView;
    if ((device.Type() == DeviceType::CPU))
        dataView = cpuDataView;
    else
    {
        dataView = new NDArrayView(GetDataType<ElementType>(), viewShape, device);
        dataView->CopyFrom(*cpuDataView);
    }

    if (dataView->Device() != device)
        throw std::exception("Device of NDArrayView does not match 'device' it was created on");

    // Test clone
    auto clonedView = dataView->DeepClone(false);
    ElementType* first = nullptr;
    const ElementType* second = cpuDataView->DataBuffer<ElementType>();
    NDArrayViewPtr temp1CpuDataView, temp2CpuDataView;
    if ((device.Type() == DeviceType::CPU))
    {
        if (dataView->DataBuffer<ElementType>() != data.data())
            throw std::exception("The DataBuffer of the NDArrayView does not match the original buffer it was created over");

        first = clonedView->WritableDataBuffer<ElementType>();
    }
    else
    {
        temp1CpuDataView = new NDArrayView(GetDataType<ElementType>(), viewShape, DeviceDescriptor::CPUDevice());
        temp1CpuDataView->CopyFrom(*clonedView);

        first = temp1CpuDataView->WritableDataBuffer<ElementType>();
    }

    for (size_t i = 0; i < viewShape.TotalSize(); ++i)
    {
        if (first[i] != second[i])
            throw std::exception("The contents of the clone do not match expected");
    }

    first[0] += 1;
    if ((device.Type() != DeviceType::CPU))
        clonedView->CopyFrom(*temp1CpuDataView);

    if ((device.Type() == DeviceType::CPU))
    {
        first = clonedView->WritableDataBuffer<ElementType>();
        second = dataView->DataBuffer<ElementType>();
    }
    else
    {
        temp1CpuDataView = new NDArrayView(GetDataType<ElementType>(), viewShape, DeviceDescriptor::CPUDevice());
        temp1CpuDataView->CopyFrom(*clonedView);
        first = temp1CpuDataView->WritableDataBuffer<ElementType>();

        temp2CpuDataView = new NDArrayView(GetDataType<ElementType>(), viewShape, DeviceDescriptor::CPUDevice());
        temp2CpuDataView->CopyFrom(*dataView);
        second = temp2CpuDataView->DataBuffer<ElementType>();
    }

    if (first[0] != (second[0] + 1))
        throw std::exception("The clonedView's contents do not match expected");

    // Test alias
    auto aliasView = clonedView->Alias(true);
    const ElementType* aliasViewBuffer = aliasView->DataBuffer<ElementType>();
    const ElementType* clonedDataBuffer = clonedView->DataBuffer<ElementType>();
    if (aliasViewBuffer != clonedDataBuffer)
        throw std::exception("The buffers underlying the alias view and the view it is an alias of are different!");

    clonedView->CopyFrom(*dataView);
    if (aliasViewBuffer != clonedDataBuffer)
        throw std::exception("The buffers underlying the alias view and the view it is an alias of are different!");

    // Test readonliness
    auto verifyException = [](const std::function<void()>& functionToTest) {
        bool error = false;
        try
        {
            functionToTest();
        }
        catch (const std::exception&)
        {
            error = true;
        }

        if (!error)
            throw std::exception("Was incorrectly able to get a writable buffer pointer from a readonly view");
    };

    // Should not be able to get the WritableDataBuffer for a read-only view
    verifyException([&aliasView]() {
        ElementType* aliasViewBuffer = aliasView->WritableDataBuffer<ElementType>();
        aliasViewBuffer;
    });

    // Should not be able to copy into a read-only view
    verifyException([&aliasView, &dataView]() {
        aliasView->CopyFrom(*dataView);
    });
}

void NDArrayViewTests()
{
    TestNDArrayView<float>(2, DeviceDescriptor::CPUDevice());
    TestNDArrayView<float>(0, DeviceDescriptor::GPUDevice(0));
    TestNDArrayView<double>(4, DeviceDescriptor::GPUDevice(0));
}
