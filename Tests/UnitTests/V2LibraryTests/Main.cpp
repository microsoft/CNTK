#include "CNTKLibrary.h"
#include <functional>

void NDArrayViewTests();
void TensorTests();
void FeedForwardTests();
void RecurrentFunctionTests();
void TrainerTests();
void SerializationTests();

int main()
{
    NDArrayViewTests();
    TensorTests();
    FeedForwardTests();
    RecurrentFunctionTests();
    TrainerTests();
    SerializationTests();

    fprintf(stderr, "\nCNTKv2Library tests: Passed\n");
    fflush(stderr);
}
