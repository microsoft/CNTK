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

int main()
{
    NDArrayViewTests();
    TensorTests();
    FunctionTests();

    FeedForwardTests();
    RecurrentFunctionTests();

    TrainerTests();
    TestCifarResnet();
    TrainLSTMSequenceClassifer();

    fprintf(stderr, "\nCNTKv2Library tests: Passed\n");
    fflush(stderr);
}
