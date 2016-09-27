#include "CNTKLibrary.h"
#include <functional>

using namespace CNTK;

void NDArrayViewTests();
void TensorTests();
void FeedForwardTests();
void RecurrentFunctionTests();
void TrainerTests();
void TestCifarResnet();
void FunctionTests();
void TrainLSTMSequenceClassifer();
void SerializationTests();
void LearnerTests();
void TrainSequenceToSequenceTranslator();
void EvalMultiThreadsWithNewNetwork(const DeviceDescriptor&, const int);
void DeviceSelectionTests();

int main()
{
    NDArrayViewTests();
    TensorTests();
    FunctionTests();

    FeedForwardTests();
    RecurrentFunctionTests();

    TrainerTests();
    SerializationTests();
    LearnerTests();

    TestCifarResnet();
    TrainLSTMSequenceClassifer();

    TrainSequenceToSequenceTranslator();

    // Test multi-threads evaluation
    fprintf(stderr, "Test multi-threaded evaluation on CPU.\n");
    EvalMultiThreadsWithNewNetwork(DeviceDescriptor::CPUDevice(), 2);
#ifndef CPUONLY
    fprintf(stderr, "Test multi-threaded evaluation on GPU\n");
    EvalMultiThreadsWithNewNetwork(DeviceDescriptor::GPUDevice(0), 2);
#endif

    DeviceSelectionTests();

    fprintf(stderr, "\nCNTKv2Library tests: Passed\n");
    fflush(stderr);
}
