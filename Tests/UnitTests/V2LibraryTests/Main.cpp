#include "CNTKLibrary.h"
#include <functional>

using namespace CNTK;

void NDArrayViewTests();
void TensorTests();
void FeedForwardTests();
void RecurrentFunctionTests();
void TrainerTests();
void TestCifarResnet();
void SerializationTests();
void LearnerTests();

int main()
{
    NDArrayViewTests();
    TensorTests();

    FeedForwardTests();
    RecurrentFunctionTests();

    TrainerTests();
    SerializationTests();
    LearnerTests();

    TestCifarResnet();

    fprintf(stderr, "\nCNTKv2Library tests: Passed\n");
    fflush(stderr);
}
