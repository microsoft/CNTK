#include "CNTKLibrary.h"
#include <functional>

void NDArrayViewTests();
void TensorTests();
void FeedForwardTests();
void RecurrentFunctionTests();

int main()
{
    NDArrayViewTests();
    TensorTests();
    FeedForwardTests();
    RecurrentFunctionTests();
}
