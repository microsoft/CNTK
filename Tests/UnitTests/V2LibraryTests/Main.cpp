#include "CNTKLibrary.h"
#include <functional>

using namespace CNTK;

void NDArrayViewTests();
void TensorTests();
void FeedForwardTests();
void RecurrentFunctionTests();
void TrainerTests();
void TrainCifarResnet();
void FunctionTests();
void TrainLSTMSequenceClassifer();
void SerializationTests();
void LearnerTests();
void TrainSequenceToSequenceTranslator();
void TrainTruncatedLSTMAcousticModelClassifer();
void EvalMultiThreadsWithNewNetwork(const DeviceDescriptor&, const int);
void EvalMultiThreadsWithClone(const DeviceDescriptor&, const int);
void DeviceSelectionTests();

int main()
{
    // Lets disable automatic unpacking of PackedValue object to detect any accidental unpacking 
    // which will have a silent performance degradation otherwise
    Internal::DisableAutomaticUnpackingOfPackedValues();

    NDArrayViewTests();
    TensorTests();
    FunctionTests();

    SerializationTests();
    LearnerTests();

    TrainerTests();
    TrainCifarResnet();
    TrainLSTMSequenceClassifer();

    TrainSequenceToSequenceTranslator();
    TrainTruncatedLSTMAcousticModelClassifer();

    // Test multi-threads evaluation
    fprintf(stderr, "Test multi-threaded evaluation on CPU.\n");
    EvalMultiThreadsWithNewNetwork(DeviceDescriptor::CPUDevice(), 2);
#ifndef CPUONLY
    fprintf(stderr, "Test multi-threaded evaluation on GPU\n");
    EvalMultiThreadsWithNewNetwork(DeviceDescriptor::GPUDevice(0), 2);
#endif
    //// Test multi-threads evaluation
    //// Todo: Also test on GPUDevice()
    //EvalMultiThreadsWithNewNetwork(DeviceDescriptor::CPUDevice(), 2);
    EvalMultiThreadsWithClone(DeviceDescriptor::CPUDevice(), 1);

    fprintf(stderr, "Test device selection API\n");
    DeviceSelectionTests();

    fprintf(stderr, "\nCNTKv2Library tests: Passed\n");
    fflush(stderr);
}
