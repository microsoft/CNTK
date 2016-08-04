
#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif

#include <vector>
#include "Common.h"
#include "Utils.h"
#include "Matrix.h"
#include "TensorView.h"
#include "CNTKLibrary.h"

using namespace std;

using Microsoft::MSR::CNTK::Matrix;
using Microsoft::MSR::CNTK::TensorView;
using Microsoft::MSR::CNTK::TensorShape;
using Microsoft::MSR::CNTK::MatrixType;

// TODO: fix "RuntimeError : ambiguous call to overloaded function",
// uncomment the line below and remove all the following using directives.
// using namespace CNTK;

using ::CNTK::NDArrayView;
using ::CNTK::NDArrayViewPtr;
using ::CNTK::MakeSharedObject;
using ::CNTK::SGDLearner;
using ::CNTK::NDShape;
using ::CNTK::DeviceDescriptor;
using ::CNTK::Value;
using ::CNTK::Variable;
using ::CNTK::ValuePtr;
using ::CNTK::AsDataType;
using ::CNTK::AsDeviceDescriptor;
using ::CNTK::AsCNTKImplDeviceId;
using ::CNTK::StorageFormat;

unsigned long Seed()
{
    static unsigned long seed = 0;
    return ++seed;
}

template <typename ElementType>
static NDArrayViewPtr AsNDArrayView(Matrix<ElementType>& matrix)
{
    NDShape shape = { matrix.GetNumRows(), matrix.GetNumCols() };
    if (matrix.GetMatrixType() == MatrixType::SPARSE)
    {
        // TODO: This is a temporary workaround that allows to wrap an existing sparse matrix in an NDViewArray.
        auto tensorView = new TensorView<ElementType>(std::shared_ptr<Matrix<ElementType>>(&matrix, [](Matrix<ElementType>*){}),
                                                      TensorShape(matrix.GetNumRows(), matrix.GetNumCols()));

        return MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), AsDeviceDescriptor(matrix.GetDeviceId()),
                                             StorageFormat::SparseCSC, shape, false, tensorView);
    }

    return MakeSharedObject<NDArrayView>(shape, matrix.Data(), matrix.GetNumElements(),
                                         AsDeviceDescriptor(matrix.GetDeviceId()));
}

template <typename ElementType>
void TestSGDLearner(const DeviceDescriptor& device)
{
    srand(Seed());

    auto deviceId = AsCNTKImplDeviceId(device);

    size_t maxDimSize = 100;

    NDShape viewShape = { (rand() % maxDimSize) + 1, (rand() % maxDimSize) + 1 };

    Variable parameter(viewShape, AsDataType<ElementType>(), L"");

    auto learner = SGDLearner({ parameter }, device);

    auto smoothedGradient = Matrix<ElementType>(viewShape[0], viewShape[1], deviceId);

    auto gradientMatrix = Matrix<ElementType>::RandomUniform(viewShape[0], viewShape[1], deviceId, 0.0f, 2.0f, Seed());
    auto parameterMatrix = Matrix<ElementType>::RandomUniform(viewShape[0], viewShape[1], deviceId, 0.0f, 2.0f, Seed());


    unordered_map<Variable, ValuePtr> parameters;
    unordered_map<Variable, const ValuePtr> gradients;

    parameters.insert(make_pair(parameter, MakeSharedObject<Value>(AsNDArrayView(parameterMatrix))));
    gradients.insert(make_pair(parameter, MakeSharedObject<Value>(AsNDArrayView(gradientMatrix))));


    auto learningRate = 0.5;
    auto momentum = 0.0;

    learner->SetLearningRate(learningRate);
    learner->Update(parameters, gradients, 1);

    smoothedGradient.NormalGrad(gradientMatrix, parameterMatrix, ElementType(learningRate), ElementType(momentum), false);

}

void LearnerTests()
{
    TestSGDLearner<float>(DeviceDescriptor::CPUDevice());
//#ifndef CPUONLY
//    TestTensorPlus<double>(4, 1, DeviceDescriptor::GPUDevice(0));
//    TestTensorPlus<float>(1, 3, DeviceDescriptor::GPUDevice(0));
//    TestTensorPlus<double>(2, 0, DeviceDescriptor::GPUDevice(0));
//    TestTensorPlus<float>(0, 0, DeviceDescriptor::GPUDevice(0));
//#endif
}
