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
    // Todo: Also test on GPUDevice()
    EvalMultiThreadsWithNewNetwork(DeviceDescriptor::CPUDevice(), 2);

    fprintf(stderr, "\nCNTKv2Library tests: Passed\n");
    fflush(stderr);
}
