#include "CNTKLibrary.h"
#include <functional>

void NDArrayViewTests();
void TensorTests();
void FeedForwardTests();
void RecurrentFunctionTests();
void LearnerTests();


int main()
{
    NDArrayViewTests();
    TensorTests();
    FeedForwardTests();
    RecurrentFunctionTests();
    LearnerTests();

    fprintf(stderr, "\nCNTKv2Library tests: Passed\n");
    fflush(stderr);
}
