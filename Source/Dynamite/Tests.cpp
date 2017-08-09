//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "CNTKLibraryHelpers.h"
#include "PlainTextDeseralizer.h"
#include "Layers.h"
#include "TimerUtility.h"
//#include "../Math/CommonMatrix.h"
// TODO: pull this from a header
enum ElementWiseOperator
{
    // nullary
    opConstOne, opNone,
    // unary (or binary with constant parameter)
    opCopy,
    opNegate, opNot, opAbs, opFloor, opReciprocal,
    opSigmoid, opTanh, opSqr, opSqrt, opExp, opLog, opLinearRectifier, opCosine, opSin, opExponentialLinearUnit, opStableSigmoid,
    // unary ops for use by Matrix class only (there is no TensorView implementation)
    opSigmoidDerivative, opLinearRectifierDerivative, opNegativeSine, opExponentialLinearUnitDerivative, opStableSigmoidDerivative,
    // binary
    opCopyIf, opCopyIfNot, opSum, opDifference, opElementwiseProduct, opElementwiseQuotient, opLogSum, opPow,
    opMax, opMin, opArgmax, opArgmin,
    opLess, opEqual, opGreater, opGreaterEqual, opNotEqual, opLessEqual, // Note: must obey this order: (sgn(a-b) == -1, 0, +1), (sgn(a-b) != -1, 0, +1)
    opAnd, opOr, opXor, opMaskNegative,
    opElementwiseProductWithSigmoidDerivativeFromOutput, opElementwiseProductWithTanhDerivativeFromOutput,
    opElementwiseProductWithLinearRectifierDerivativeFromOutput, opElementwiseProductWithLogDerivativeFromOutput,
    opElementwiseProductWithCosDerivative, opElementwiseProductWithSinDerivative,
    opElementwiseProductWithAbsDerivative, opElementwiseProductWithSqrtDerivative,
    opElementwiseProductWithReciprocalDerivative, opSqrOfDifference,
    opElementwiseProductWithExponentialLinearUnitDerivativeFromOutput,
    // binary ops for indexing
    // opIndex,
    // ternary
    opCond /*a ? b : c*/,
    opClip, /*clip a within interval b..c*/
    opElementwiseProductWithLogSumDerivative,
    opCopyIfEqual,
    opElementwiseProductWithExpOfDiff, /* a * exp(b - c) */
    opElementwiseProductWithQuotient, /* a * (b / c) */
    opElementwiseProductWithPowExponentDerivative, /* a * b * log(c) */
    opElementwiseProductWithPowBaseDerivative,  /* a * c * pow(b, c-1) */
                                                // Note: not all that's implemented in CNTK ComputationNodes has an opcode yet.
};


#include <cstdio>
#include <map>
#include <set>
#include <vector>

#define let const auto

using namespace CNTK;
using namespace std;

using namespace Dynamite;

#define Op(opCode) (pair<function<NDArrayViewPtr(const vector<NDArrayViewPtr>&)>, const char*>([=](const vector<NDArrayViewPtr>& argValues){ return NDArrayView::NumericOperation(argValues, 1.0, op##opCode); }, #opCode))

struct TensorViewTest
{
    pair<function<NDArrayViewPtr(const vector<NDArrayViewPtr>&)>, const char*> op;
    function<Variable(const vector<Variable>& args)> f;
    vector<NDShape> shapes;
};

// helper to create a random matrix
static NDArrayViewPtr RandomTensor(const NDShape& shape, double scale, unsigned long& seed, DataType dataType, const DeviceDescriptor& device)
{
    if (dataType == DataType::Float)
        return NDArrayView::RandomNormal<float> (shape, /*mean=*/0., /*stdDev=*/scale, seed++, device);
    else
        return NDArrayView::RandomNormal<double>(shape, /*mean=*/0., /*stdDev=*/scale, seed++, device);
}

// helper to compute average square error between two NDArrayViews
static double AvSqrErr(const NDArrayViewPtr& resVal, const NDArrayViewPtr& refVal, DataType dataType, const DeviceDescriptor& device)
{
    let sqrErr = NDArrayView::NumericOperation({ resVal, refVal }, 1.0 / refVal->Shape().TotalSize(), ElementWiseOperator::opSqrOfDifference, make_shared<NDArrayView>(dataType, NDShape{}, device), 0, ElementWiseOperator::opSum);
    return sqrErr->AsScalar<double>();
}

static double SumAll(const NDArrayViewPtr& x, DataType dataType, const DeviceDescriptor& device)
{
    let sum = NDArrayView::NumericOperation({ x }, 1.0, ElementWiseOperator::opCopy, make_shared<NDArrayView>(dataType, NDShape{}, device), 0, ElementWiseOperator::opSum);
    return sum->AsScalar<double>();
}

size_t DynamiteTest(size_t N, DataType dataType, const DeviceDescriptor& device)
{
    size_t numFailed = 0;
    unsigned long seed = 1;
    // for testing batching of the matrix product, we need a shared matrix
    let sharedMatrix = RandomTensor(NDShape{ 13, 42 }, 0.3, seed, dataType, device);
    let sharedMatrixVar = Parameter(sharedMatrix);
    vector<TensorViewTest> tests =
    {
        // matrix product
        { { [&](const vector<NDArrayViewPtr>& argValues) { return NDArrayView::MatrixProduct(false, sharedMatrix, false, argValues[1], false, 1.0, 1); }, "Times"          }, [&](const vector<Variable>& args) { return CNTK::Times (sharedMatrixVar, args[1]); },{ { 13, 42 },{ 42, 9 } } },
        { { [&](const vector<NDArrayViewPtr>& argValues) { return NDArrayView::MatrixProduct(false, argValues[0], false, argValues[1], false, 1.0, 1); }, "Times"          }, [&](const vector<Variable>& args) { return CNTK::Times         (args[0], args[1]); },{ { 13, 42 },{ 42, 9 } } },
        { { [&](const vector<NDArrayViewPtr>& argValues) { return NDArrayView::MatrixProduct(false, argValues[0], false, argValues[1], false, 1.0, 1); }, "Times"          }, [&](const vector<Variable>& args) { return CNTK::Times         (args[0], args[1]); },{ { 13, 42 },{ 42 } } },
        { { [&](const vector<NDArrayViewPtr>& argValues) { return NDArrayView::MatrixProduct(false, argValues[0], false, argValues[1], false, 1.0, 1); }, "Times"          }, [&](const vector<Variable>& args) { return CNTK::Times         (args[0], args[1]); },{ { 13, 42 },{ 42, 9, 5 } } },
        { { [&](const vector<NDArrayViewPtr>& argValues) { return NDArrayView::MatrixProduct(false, argValues[0], true,  argValues[1], false, 1.0, 1); }, "TransposeTimes" }, [&](const vector<Variable>& args) { return CNTK::TransposeTimes(args[0], args[1]); },{ { 42, 13 },{ 42, 9 } } },
        { { [&](const vector<NDArrayViewPtr>& argValues) { return NDArrayView::MatrixProduct(false, argValues[0], true,  argValues[1], false, 1.0, 1); }, "TransposeTimes" }, [&](const vector<Variable>& args) { return CNTK::TransposeTimes(args[0], args[1]); },{ { 42, 13 },{ 42 } } },
        { { [&](const vector<NDArrayViewPtr>& argValues) { return NDArrayView::MatrixProduct(false, argValues[0], true,  argValues[1], false, 1.0, 1); }, "TransposeTimes" }, [&](const vector<Variable>& args) { return CNTK::TransposeTimes(args[0], args[1]); },{ { 42, 13 },{ 42, 9, 3 } } },
        // ternary
        { Op(Clip                 ), [](const vector<Variable>& args) { return CNTK::Clip         (args[0], args[1], args[2]); }, { { 13, 42 }, { 13, 1 }, { 13, 1 } } },
        { Op(Cond                 ), [](const vector<Variable>& args) { return CNTK::ElementSelect(args[0], args[1], args[2]); }, { { 13, 42 }, { 13, 1 }, { 13, 1 } } },
        // binary
        { Op(Sum                  ), [](const vector<Variable>& args) { return CNTK::Plus         (args[0], args[1]); }, { { 13, 42 }, { 13, 42 } } },
        { Op(Difference           ), [](const vector<Variable>& args) { return CNTK::Minus        (args[0], args[1]); }, { { 13, 42 }, { 13, 1 } } },
        { Op(ElementwiseProduct   ), [](const vector<Variable>& args) { return CNTK::ElementTimes (args[0], args[1]); }, { { 13, 42 }, { 13, 1 } } },
        { Op(LogSum               ), [](const vector<Variable>& args) { return CNTK::LogAddExp    (args[0], args[1]); }, { { 13, 42 }, { 13, 1 } } },
        { Op(Pow                  ), [](const vector<Variable>& args) { return CNTK::Pow          (args[0], args[1]); }, { { 13, 42, 12 }, { 13, 1 } } },
        { Op(Equal                ), [](const vector<Variable>& args) { return CNTK::Equal        (args[0], args[1]); }, { { 13, 42 }, { 13, 1 } } },
        { Op(NotEqual             ), [](const vector<Variable>& args) { return CNTK::NotEqual     (args[0], args[1]); }, { { 13, 42 }, { 13, 42 } } },
        { Op(Less                 ), [](const vector<Variable>& args) { return CNTK::Less         (args[0], args[1]); }, { { 13, 42 }, { 13, 1 } } },
        { Op(LessEqual            ), [](const vector<Variable>& args) { return CNTK::LessEqual    (args[0], args[1]); }, { { 13, 42 }, { 13, 1 } } },
        { Op(Greater              ), [](const vector<Variable>& args) { return CNTK::Greater      (args[0], args[1]); }, { { 13, 42 }, { 13, 1 } } },
        { Op(GreaterEqual         ), [](const vector<Variable>& args) { return CNTK::GreaterEqual (args[0], args[1]); }, { { 13, 42 }, { 13, 1 } } },
        // unary
        { Op(LinearRectifier      ), [](const vector<Variable>& args) { return CNTK::ReLU         (args[0]         ); }, { { 13, 42 } } },
        { Op(Tanh                 ), [](const vector<Variable>& args) { return CNTK::Tanh         (args[0]         ); }, { { 13 } } },
        { Op(Log                  ), [](const vector<Variable>& args) { return CNTK::Log          (args[0]         ); }, { { 13, 42 } } },
        { Op(Exp                  ), [](const vector<Variable>& args) { return CNTK::Exp          (args[0]         ); }, { { 13, 42 } } },
        { Op(Cosine               ), [](const vector<Variable>& args) { return CNTK::Cos          (args[0]         ); }, { { 13, 42 } } },
        { Op(Sin                  ), [](const vector<Variable>& args) { return CNTK::Sin          (args[0]         ); }, { { 235, 13, 2 } } },
        { Op(Negate               ), [](const vector<Variable>& args) { return CNTK::Negate       (args[0]         ); }, { { 13 } } },
        { Op(Floor                ), [](const vector<Variable>& args) { return CNTK::Floor        (args[0]         ); }, { { 13, 42 } } },
        { Op(Abs                  ), [](const vector<Variable>& args) { return CNTK::Abs          (args[0]         ); }, { { 13, 42 } } },
        { Op(Sqrt                 ), [](const vector<Variable>& args) { return CNTK::Sqrt         (args[0]         ); }, { { 13, 42 } } },
        { Op(Reciprocal           ), [](const vector<Variable>& args) { return CNTK::Reciprocal   (args[0]         ); }, { { 13, 42 } } },
        { Op(ExponentialLinearUnit), [](const vector<Variable>& args) { return CNTK::ELU          (args[0]         ); }, { { 13, 42 } } },
        { Op(StableSigmoid        ), [](const vector<Variable>& args) { return CNTK::Sigmoid      (args[0]         ); }, { { 128 } } }
    };

    fprintf(stderr, "\n--- batch of %d. %s on %S\n\n", (int)N, CNTK::DataTypeName(dataType), device.AsString().c_str());
    for (let& test : tests)
    {
        NDArrayViewPtr refVal;
        Variable resVar, resVar1;
        vector<vector<NDArrayViewPtr>> allArgValues(N);
        vector<Variable> args;
        for (size_t n = 0; n < N; n++)
        {
            auto& argValues = allArgValues[n];
            for (let& shape : test.shapes)
                argValues.push_back(RandomTensor(shape, 0.3, seed, dataType, device));
        }
        for (size_t n = 0; n < N; n++)
        {
            let& argValues = allArgValues[n];
            // reference: TensorView op directly
            let refVal1 = test.op.first(argValues);
            if (n > 0)
                refVal = refVal + refVal1;// NDArrayView::NumericOperation({ refVal, refVal1 }, 1.0, ElementWiseOperator::opSum);
            else
                refVal = refVal1;
#if 0
            for (let& arg : argValues)
                arg->LogToFile(L"argVal", stderr);
            refVal1->LogToFile(L"resVal", stderr);
            refVal->LogToFile(L"sumVal", stderr);
#endif
        }
        for (size_t n = 0; n < N; n++)
        {
            let& argValues = allArgValues[n];
            // Dynamite:
            args.clear();
            for (let& argValue : argValues)
                args.push_back(Constant(argValue));
            if (n == 0)
            {
                fprintf(stderr, "%25s(", test.op.second);
                for (let& arg : args)
                    fprintf(stderr, " %S ", arg.Shape().AsString().c_str());
            }
            resVar1 = test.f(args);
            if (n > 0)
                resVar = resVar + resVar1;// CNTK::Plus(resVar, resVar1);
            else
                resVar = resVar1;
        }
        let resVal = resVar.Value(); // this triggers the batched evaluation
        fprintf(stderr, ") -> %S\n", resVal->AsString().c_str());
        let avSqrErr = AvSqrErr(resVal, refVal, dataType, device);
        if (avSqrErr > 1e-5)
        {
            fprintf(stderr, "################# FAILED: avSqrErr = %.2f\n", avSqrErr);
            numFailed++;
        }
        // gradient check
        if (dataType == DataType::Double)
        {
            let n = 0; // TODO: expand this to batching later
            let& argValues = allArgValues[n];
            let epsScale = 1e-4;
            for (size_t i = 0; i < argValues.size(); i++)
            {
                // we test SumAll(f(x,y))
                // compute original value
                args.clear();
                for (let& argValue : argValues)
                    args.push_back(Parameter(argValue)); // TODO: does Backward() really need to take Parameters? Why not any Variable? Can Constants take a gradient??
                let output = test.f(args);
                let input = Parameter(output.Owner()->Inputs()[i]); // get actual input and cast as Parameter (args[] may differ from actual since the test-case's lambda may ignore args[])
                if (output.Owner()->Inputs()[i] != args[i]) // lambda ignores this input
                    continue;
                let arg = Parameter(args[i]);
                Variable sumAll = ReduceSum(output, Axis::AllStaticAxes());
                let resVal = sumAll.Value(); // this triggers batched forward computation
                // compute perturbed output
                let eps = RandomTensor(arg.Shape(), epsScale /*/ arg.Shape().TotalSize()*/, seed, dataType, device);
                //eps->LogToFile(L"eps", stderr);
                auto perturbedArgs = args;
                perturbedArgs[i] = Constant(perturbedArgs[i].Value() + eps);
                Variable perturbedSumAll = ReduceSum(test.f(perturbedArgs), Axis::AllStaticAxes());
                let perturbedResVal = perturbedSumAll.Value();
                //resVal->LogToFile(L"resVal", stderr);
                //perturbedResVal->LogToFile(L"perturbedResVal", stderr);
                let perturbedDelta = (perturbedResVal - resVal)->AsScalar<double>();
                // compute gradient of sum over all elements of test.f(args) (=backprop a 1.0 into every element)
                unordered_map<Parameter, NDArrayViewPtr> gradients{ { arg, nullptr } };
                try
                {
                    sumAll.Backward(gradients); // this triggers batched backward computation
                }
                catch (...)
                {
                    continue; // some don't have a gradient implementation yet (don't log anything here; user will crash when trying to use it)
                }
                let gradientWrtInput = gradients[arg]; // get gradient for arg
                //gradientWrtInput->LogToFile(L"gradientWrtInput", stderr);
                // compute expected perturbed output based on gradient
                // gradientWrtInput[j,k] = slope of sum of all outputs w.r.t. changes of arg[j,k]
                let gradientBasedDelta = SumAll(gradientWrtInput * eps, dataType, device);
                let relErr = (perturbedDelta == gradientBasedDelta) ? 0 : fabs(((perturbedDelta - gradientBasedDelta) / perturbedDelta));
                if (relErr > 1e-5)
                    fprintf(stderr, "\t\t\t\tgradient[%d] err=%.10f%% (%.20f, %.20f)\n", (int)i, 100.0 * relErr, perturbedDelta, gradientBasedDelta);
            }
        }
    }
    return numFailed;
}

void RunDynamiteTests()
{
    size_t numFailed = 0;
    numFailed += DynamiteTest(1, DataType::Double, DeviceDescriptor::GPUDevice(0));
    numFailed += DynamiteTest(3, DataType::Double, DeviceDescriptor::CPUDevice());
    numFailed += DynamiteTest(3, DataType::Float, DeviceDescriptor::GPUDevice(0));
#if 0 // do this not every time
    numFailed += DynamiteTest(1, DataType::Float, DeviceDescriptor::GPUDevice(0));
    numFailed += DynamiteTest(1, DataType::Double, DeviceDescriptor::CPUDevice());
    numFailed += DynamiteTest(1, DataType::Float, DeviceDescriptor::CPUDevice());
#endif
    if (numFailed > 0)
        LogicError("RunDynamiteTests: %d tests failed.", (int)numFailed);
    exit(0);
}
