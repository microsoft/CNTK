#include "CNTKLibrary.h"
#include "Common.h"

using namespace CNTK;

void TestReduceSum(const DeviceDescriptor& device)
{
    size_t numSequences = 7;
    size_t maxAllowedSequenceLength = 11;
    size_t dim = 23;

    auto sequenceLengths = GenerateSequenceLengths(numSequences, maxAllowedSequenceLength);
    auto sequences = GenerateSequences<float>(sequenceLengths, dim);
    ValuePtr sequencesValue = Value::Create({ dim }, sequences, device, true);

    // Test ReduceSum along a static axis
    {
        auto testReduceSum = [&sequences, &sequenceLengths, dim, sequencesValue, device](bool reduceAll)
        {
            size_t maxActualSequenceLength = sequencesValue->Data()->Shape()[1];
            size_t numSequences = sequencesValue->Data()->Shape()[2];

            auto inputVar = InputVariable({ dim }, DataType::Float, L"input");
            FunctionPtr reduceSumFunc;

            if (reduceAll)
                reduceSumFunc = ReduceSum(inputVar);
            else
                reduceSumFunc = ReduceSum(inputVar, Axis(0));

            NDShape outputShape;
            if (reduceAll)
                outputShape = {};
            else
                outputShape = reduceSumFunc->Output().Shape().AppendShape({ maxActualSequenceLength, numSequences });

            std::vector<float> outputData(outputShape.TotalSize());
            ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, outputData, false), reduceAll ? nullptr : sequencesValue->Mask()->DeepClone());

            std::unordered_map<Variable, ValuePtr> outputs = { { reduceSumFunc->Output(), outputValue } };
            reduceSumFunc->Forward({ { inputVar, sequencesValue } }, outputs, device);

            std::vector<float> expectedPerFrameTotals(maxActualSequenceLength * numSequences, 0.0f);
            float expectedTotal = 0.0f;
            for (size_t i = 0; i < numSequences; ++i)
            {
                size_t currentSequenceLength = sequenceLengths[i];
                for (size_t j = 0; j < currentSequenceLength; ++j)
                {
                    for (size_t k = 0; k < dim; ++k)
                    {
                        float value = sequences[i][(j * dim) + k];
                        expectedPerFrameTotals[(i * maxActualSequenceLength) + j] += value;
                        expectedTotal += value;
                    }
                }
            }

            if (reduceAll)
                FloatingPointVectorCompare(outputData, std::vector<float>({ expectedTotal }), "testReduceSum: Forward prop results do not match expected results");
            else
                FloatingPointVectorCompare(outputData, expectedPerFrameTotals, "testReduceSum: Forward prop results do not match expected results");
        };

        testReduceSum(true);
        testReduceSum(false);
    }

    // Test ReduceSum along a dynamic axis
    {
        auto testReduceSum = [&sequences, &sequenceLengths, dim, sequencesValue, device](const Axis& axis)
        {
            if (axis.IsStaticAxis())
                RuntimeError("Called the dynamic axis ReduceSum test with a static axis");

            size_t maxActualSequenceLength = sequencesValue->Data()->Shape()[1];
            size_t numSequences = sequencesValue->Data()->Shape()[2];

            auto inputVar = InputVariable({ dim }, DataType::Float, L"input");
            FunctionPtr reduceSumFunc = ReduceSum(inputVar, axis);

            NDShape maskShape = { ((axis == Axis::DefaultBatchAxis()) ? maxActualSequenceLength : 1), ((axis == Axis::DefaultBatchAxis()) ? 1 : numSequences) };
            NDShape outputShape = reduceSumFunc->Output().Shape().AppendShape(maskShape);

            std::vector<float> outputData(outputShape.TotalSize());
            auto maskPtr = MakeSharedObject<NDMask>(maskShape, device);
            ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, outputData, false), maskPtr);

            std::unordered_map<Variable, ValuePtr> outputs = { { reduceSumFunc->Output(), outputValue } };
            reduceSumFunc->Forward({ { inputVar, sequencesValue } }, outputs, device);

            std::vector<float> expectedTotals(outputShape.TotalSize(), 0.0f);
            for (size_t i = 0; i < numSequences; ++i)
            {
                size_t currentSequenceLength = sequenceLengths[i];
                for (size_t j = 0; j < currentSequenceLength; ++j)
                {
                    for (size_t k = 0; k < dim; ++k)
                    {
                        float value = sequences[i][(j * dim) + k];
                        if (axis == Axis::DefaultBatchAxis())
                            expectedTotals[(j * dim) + k] += value;
                        else
                            expectedTotals[(i * dim) + k] += value;;
                    }
                }
            }

            FloatingPointVectorCompare(outputData, expectedTotals, "testReduceSum: Forward prop results do not match expected results");
        };

        testReduceSum(Axis::DefaultDynamicAxis());
    }
}

void TestSlice(const DeviceDescriptor& device)
{
    size_t numSequences = 7;
    size_t maxAllowedSequenceLength = 11;
    size_t dim = 23;

    auto sequenceLengths = GenerateSequenceLengths(numSequences, maxAllowedSequenceLength);
    auto sequences = GenerateSequences<float>(sequenceLengths, dim);
    ValuePtr sequencesValue = Value::Create({ dim }, sequences, device, true);

    // Test slice along a static axis
    {
        auto testStaticAxisSlice = [&sequences, &sequenceLengths, dim, sequencesValue, device](int beginOffset, int endOffset)
        {
            size_t maxActualSequenceLength = sequencesValue->Data()->Shape()[1];
            size_t numSequences = sequencesValue->Data()->Shape()[2];

            auto inputVar = InputVariable({ dim }, DataType::Float, L"input");
            auto rowSliceFunc = Slice(inputVar, Axis(0), beginOffset, endOffset);

            NDShape outputShape = rowSliceFunc->Output().Shape().AppendShape({ maxActualSequenceLength, numSequences });
            std::vector<float> outputData(outputShape.TotalSize());
            ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, outputData, false), sequencesValue->Mask()->DeepClone());

            std::unordered_map<Variable, ValuePtr> outputs = { { rowSliceFunc->Output(), outputValue } };
            rowSliceFunc->Forward({ { inputVar, sequencesValue } }, outputs, device);

            size_t rowSliceStartOffset = (beginOffset >= 0) ? beginOffset : (dim + beginOffset);
            size_t sliceLength = endOffset - beginOffset;
            std::vector<float> expectedOutputValues(sliceLength * maxActualSequenceLength * numSequences);
            for (size_t i = 0; i < numSequences; ++i)
            {
                size_t currentSequenceLength = sequenceLengths[i];
                for (size_t j = 0; j < currentSequenceLength; ++j)
                {
                    for (size_t k = 0; k < sliceLength; ++k)
                        expectedOutputValues[(((i * maxActualSequenceLength) + j) * sliceLength) + k] = sequences[i][(j * dim) + k + rowSliceStartOffset];
                }
            }

            FloatingPointVectorCompare(outputData, expectedOutputValues, "testStaticAxisSlice: Forward prop results do not match expected results");
        };

        testStaticAxisSlice(3, 5);
        testStaticAxisSlice(-1, 0);
        testStaticAxisSlice(-3, -1);
    }

    // Test slice along a dynamic axis
    {
        auto testDynamicAxisSlice = [&sequences, &sequenceLengths, dim, sequencesValue, device](const Axis& axis, int beginOffset, int endOffset)
        {
            if (axis.IsStaticAxis())
                RuntimeError("Called the dynamic axis slice test with a static axis");

            size_t maxActualSequenceLength = sequencesValue->Data()->Shape()[1];
            size_t numSequences = sequencesValue->Data()->Shape()[2];

            size_t sliceLength = endOffset - beginOffset;

            auto inputVar = InputVariable({ dim }, DataType::Float, L"input");
            auto sliceFunc = Slice(inputVar, axis, beginOffset, endOffset);

            size_t outputSequenceAxisLength = (axis == Axis::DefaultDynamicAxis()) ? sliceLength : maxActualSequenceLength;
            size_t outputBatchAxisLength = (axis == Axis::DefaultBatchAxis()) ? sliceLength : numSequences;
            NDShape outputShape = sliceFunc->Output().Shape().AppendShape({ outputSequenceAxisLength, outputBatchAxisLength });
            std::vector<float> outputData(outputShape.TotalSize());
            ValuePtr outputValue = MakeSharedObject<Value>(MakeSharedObject<NDArrayView>(outputShape, outputData, false));

            std::unordered_map<Variable, ValuePtr> outputs = { { sliceFunc->Output(), outputValue } };
            sliceFunc->Forward({ { inputVar, sequencesValue } }, outputs, device);

            size_t startSequenceIdx = (axis == Axis::DefaultBatchAxis()) ? ((beginOffset >= 0) ? beginOffset : (numSequences + beginOffset)) : 0;
            size_t endSequenceIdx = (axis == Axis::DefaultBatchAxis()) ? ((endOffset > 0) ? endOffset : (numSequences + endOffset)) : numSequences;

            std::vector<float> expectedOutputValues(dim * outputSequenceAxisLength * outputBatchAxisLength);
            for (size_t i = startSequenceIdx; i < endSequenceIdx; ++i)
            {
                size_t currentSequenceLength = sequenceLengths[i];
                size_t startFrameIdx = (axis == Axis::DefaultDynamicAxis()) ? ((beginOffset >= 0) ? beginOffset : (currentSequenceLength + beginOffset)) : 0;
                size_t endFrameIdx = (axis == Axis::DefaultDynamicAxis()) ? ((endOffset > 0) ? endOffset : (currentSequenceLength + endOffset)) : currentSequenceLength;
                for (size_t j = startFrameIdx; j < endFrameIdx; ++j)
                {
                    for (size_t k = 0; k < dim; ++k)
                        expectedOutputValues[((((i - startSequenceIdx) * outputSequenceAxisLength) + (j - startFrameIdx)) * dim) + k] = sequences[i][(j * dim) + k];
                }
            }

            FloatingPointVectorCompare(outputData, expectedOutputValues, "testDynamicAxisSlice: Forward prop results do not match expected results");
        };

        testDynamicAxisSlice(Axis::DefaultDynamicAxis(), 0, 1);
        testDynamicAxisSlice(Axis::DefaultDynamicAxis(), -1, 0);
    }
}

void FunctionTests()
{
    TestSlice(DeviceDescriptor::CPUDevice());
    TestSlice(DeviceDescriptor::GPUDevice(0));

    TestReduceSum(DeviceDescriptor::CPUDevice());
    TestReduceSum(DeviceDescriptor::GPUDevice(0));
}
