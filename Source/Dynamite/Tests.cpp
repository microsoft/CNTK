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

struct TensorViewTest
{
    pair<function<NDArrayViewPtr(const vector<NDArrayViewPtr>&)>, const char*> op;
    function<Variable(const vector<Variable>& args)> f;
    vector<NDShape> shapes;
};

// helper to create a random matrix
static NDArrayViewPtr RandomTestTensor(const NDShape& shape, double scale, const char* opName, size_t argIndex, unsigned long& seed, DataType dataType, const DeviceDescriptor& device)
{
    let randT = [&](double mean, double scale1)
    {
        if (dataType == DataType::Float)
            return NDArrayView::RandomNormal<float>(shape, mean, scale1, seed++, device);
        else
            return NDArrayView::RandomNormal<double>(shape, mean, scale1, seed++, device);
    };
    let constT = [&](double value)
    {
        return make_shared<NDArrayView>(value, dataType, shape, device);
    };
    auto res = randT(/*mean=*/0., /*stdDev=*/scale);
    // some special cases
    if (strstr(opName, "Log")) // Log requires positive numbers
    {
        res = NDArrayView::NumericOperation({ res }, 1.0, ElementWiseOperator::opAbs);
        res = NDArrayView::NumericOperation({ /*min=*/constT(1e-4), /*max=*/res, res }, 1.0, ElementWiseOperator::opClip);
    }
    else if (strcmp(opName, "Pow") == 0 && argIndex == 0) // Pow requires non-negative base
    {
        res = NDArrayView::NumericOperation({ res }, 1.0, ElementWiseOperator::opAbs);
    }
    else if (strcmp(opName, "Reciprocal") == 0) // Reciprocal should not use too small a number
    {
        res = NDArrayView::NumericOperation({ res }, 1.0, ElementWiseOperator::opAbs);
        res = NDArrayView::NumericOperation({ /*min=*/constT(1e-2), /*max=*/res, res }, 1.0, ElementWiseOperator::opClip);
    }
    return res;
}

// helper to compute average square error between two NDArrayViews
static double AvSqrErr(const NDArrayViewPtr& resVal, const NDArrayViewPtr& refVal, DataType dataType, const DeviceDescriptor& device)
{
    if (resVal->Shape() != refVal->Shape())
        LogicError("AvSqrErr: Result shape %S is different from expected shape %S", resVal->Shape().AsString().c_str(), refVal->Shape().AsString().c_str());
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
    let sharedMatrix = RandomTestTensor(NDShape{ 13, 42 }, 0.3, "Times", 0, seed, dataType, device);
    let sharedMatrixVar = Parameter(sharedMatrix);
#define Op(opCode) (pair<function<NDArrayViewPtr(const vector<NDArrayViewPtr>&)>, const char*>([=](const vector<NDArrayViewPtr>& argValues){ return NDArrayView::NumericOperation(argValues, 1.0, op##opCode); }, #opCode))
#define RedOp(redOpCode, shape, denom) (pair<function<NDArrayViewPtr(const vector<NDArrayViewPtr>&)>, const char*>([=](const vector<NDArrayViewPtr>& argValues){ return NDArrayView::NumericOperation(argValues, 1.0/denom, opCopy, make_shared<NDArrayView>(dataType, NDShape(shape), device), 0, op##redOpCode); }, "Reduce" #redOpCode))
#define OpWithRed(opCode, shape) (pair<function<NDArrayViewPtr(const vector<NDArrayViewPtr>&)>, const char*>([=](const vector<NDArrayViewPtr>& argValues){ return NDArrayView::NumericOperation(argValues, 1.0, op##opCode, make_shared<NDArrayView>(dataType, NDShape(shape), device), 0, opSum); }, #opCode "-ReduceSum"))
    vector<TensorViewTest> tests =
    {
        // dot product
        { OpWithRed(ElementwiseProduct, NDShape({ 1, 1, 42, 5 })), [](const vector<Variable>& args) { return CNTK::ReduceSum(CNTK::ReduceSum(CNTK::ElementTimes(args[0], args[1]), Axis(0)), Axis(1)); }, { { 13, 2, 42, 5 }, { 13, 2, 42, 5 } } },
        { OpWithRed(ElementwiseProduct, NDShape({ 1,    42    })), [](const vector<Variable>& args) { return                 CNTK::ReduceSum(CNTK::ElementTimes(args[0], args[1]), Axis(0));           }, { { 13,    42    }, { 13,    42    } } },
        { OpWithRed(ElementwiseProduct, NDShape({ 1           })), [](const vector<Variable>& args) { return                 CNTK::ReduceSum(CNTK::ElementTimes(args[0], args[1]), Axis(0));           }, { { 13           }, { 13           } } },
        // slicing, splicing, reshaping   --TODO: splicing, reshaping
        { { [&](const vector<NDArrayViewPtr>& argValues) { return argValues[0]->Slice           ({ 0, 3 }, {     13 }); }, "Index" }, [&](const vector<Variable>& args) { return CNTK::Index(args[0], 3); },{ { 13, 42 } } }, // index of rank > 1; also testing SlicedTensorView()
        { { [&](const vector<NDArrayViewPtr>& argValues) { return argValues[0]->SliceView       ({    1 }, {        }); }, "Index" }, [&](const vector<Variable>& args) { return CNTK::Index(args[0], 1); },{ { 13 } } }, // index of rank 1
        { { [&](const vector<NDArrayViewPtr>& argValues) { return argValues[0]->SliceView       ({ 0, 1 }, { 13,  4 }); }, "Slice" }, [&](const vector<Variable>& args) { return CNTK::Slice(args[0], { Axis(0), Axis(1) }, { 0, 1 }, { 13, 1+4 }); },{ { 13, 42 } } }, // multi-axis slice
        { { [&](const vector<NDArrayViewPtr>& argValues) { return argValues[0]->Slice           ({ 2, 0 }, {  3, 42 }); }, "Slice" }, [&](const vector<Variable>& args) { return CNTK::Slice(args[0], { Axis(0) }, { 2 }, { 2+3 }); },{ { 13, 42 } } }, // non-contiguous slice
        { { [&](const vector<NDArrayViewPtr>& argValues) { return argValues[0]->SliceView       ({ 0, 1 }, { 13,  4 }); }, "Slice" }, [&](const vector<Variable>& args) { return CNTK::Slice(args[0], { Axis(1) }, { 1 }, { 1+4 }); },{ { 13, 42 } } }, // contiguous slice of rank > 1
        { { [&](const vector<NDArrayViewPtr>& argValues) { return argValues[0]->Slice           ({ 0, 1 }, { 13,  4 }); }, "Slice" }, [&](const vector<Variable>& args) { return CNTK::Slice(args[0], { Axis(1) }, { 1 }, { 1+4 }); },{ { 13, 42 } } }, // same but testing SlicedTensorView() on the reference path
        { { [&](const vector<NDArrayViewPtr>& argValues) { return argValues[0]->SliceView       ({    1 }, {      3 }); }, "Slice" }, [&](const vector<Variable>& args) { return CNTK::Slice(args[0], { Axis(0) }, { 1 }, { 1+3 }); },{ { 13 } } }, // slice of rank 1
        // matrix product
        { { [&](const vector<NDArrayViewPtr>& argValues) { return NDArrayView::MatrixProduct(false, sharedMatrix, false, argValues[1], false, 1.0, 1); }, "Times_shared"   }, [&](const vector<Variable>& args) { return CNTK::Times (sharedMatrixVar, args[1]   ); },{ { 13, 42 },{ 42, 9 } } },
        { { [&](const vector<NDArrayViewPtr>& argValues) { return NDArrayView::MatrixProduct(false, argValues[0], false, argValues[1], false, 1.0, 1); }, "Times"          }, [&](const vector<Variable>& args) { return CNTK::Times         (args[0], args[1]   ); },{ { 13, 42 },{ 42, 9 } } },
        { { [&](const vector<NDArrayViewPtr>& argValues) { return NDArrayView::MatrixProduct(false, argValues[0], false, argValues[1], false, 1.0, 1); }, "Times"          }, [&](const vector<Variable>& args) { return CNTK::Times         (args[0], args[1]   ); },{ { 13, 42 },{ 42 } } },
        { { [&](const vector<NDArrayViewPtr>& argValues) { return NDArrayView::MatrixProduct(false, argValues[0], false, argValues[1], false, 1.0, 1); }, "Times"          }, [&](const vector<Variable>& args) { return CNTK::Times         (args[0], args[1], 0); },{ { 42 },{ 42 } } }, // should get batched
        { { [&](const vector<NDArrayViewPtr>& argValues) { return NDArrayView::MatrixProduct(false, argValues[0], false, argValues[1], false, 1.0, 1); }, "Times"          }, [&](const vector<Variable>& args) { return CNTK::Times         (args[0], args[1]   ); },{ { 13, 42 },{ 42, 9, 5 } } },
        { { [&](const vector<NDArrayViewPtr>& argValues) { return NDArrayView::MatrixProduct(false, argValues[0], true,  argValues[1], false, 1.0, 1); }, "TransposeTimes" }, [&](const vector<Variable>& args) { return CNTK::TransposeTimes(args[0], args[1]   ); },{ { 42, 13 },{ 42, 9 } } },
        { { [&](const vector<NDArrayViewPtr>& argValues) { return NDArrayView::MatrixProduct(false, argValues[0], true,  argValues[1], false, 1.0, 1); }, "TransposeTimes" }, [&](const vector<Variable>& args) { return CNTK::TransposeTimes(args[0], args[1]   ); },{ { 42, 13 },{ 42 } } },
        { { [&](const vector<NDArrayViewPtr>& argValues) { return NDArrayView::MatrixProduct(false, argValues[0], true,  argValues[1], false, 1.0, 1); }, "TransposeTimes" }, [&](const vector<Variable>& args) { return CNTK::TransposeTimes(args[0], args[1]   ); },{ { 42, 13 },{ 42, 9, 3 } } },
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
        { Op(StableSigmoid        ), [](const vector<Variable>& args) { return CNTK::Sigmoid      (args[0]         ); }, { { 128 } } },
        // reductions
        { RedOp(Sum,    NDShape({  1     }), 1 ), [](const vector<Variable>& args) { return CNTK::ReduceSum   (args[0], Axis(0)); }, { { 13 } } },
        { RedOp(Sum,    NDShape({ 13,  1 }), 1 ), [](const vector<Variable>& args) { return CNTK::ReduceSum   (args[0], Axis(1)); }, { { 13, 42 } } },
        { RedOp(Sum,    NDShape({  1, 42 }), 1 ), [](const vector<Variable>& args) { return CNTK::ReduceSum   (args[0], Axis(0)); }, { { 13, 42 } } },
        { RedOp(LogSum, NDShape({  1     }), 1 ), [](const vector<Variable>& args) { return CNTK::ReduceLogSum(args[0], Axis(0)); }, { { 13 } } },
        { RedOp(Sum,    NDShape({  1     }), 13), [](const vector<Variable>& args) { return CNTK::ReduceMean  (args[0], Axis(0)); }, { { 13 } } }
    };

    fprintf(stderr, "\n--- Running tests for batch of %d. %s on %S\n\n", (int)N, CNTK::DataTypeName(dataType), device.AsString().c_str());
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
                argValues.push_back(RandomTestTensor(shape, 0.3, test.op.second, argValues.size(), seed, dataType, device));
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
        bool testUsesAllArgs = true; // some tests ignore the arg; don't do gradient check on those
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
            {
                // aside: as a courtesy for the gradient check, determine whether the test function actually uses all the args given to it
                for (size_t i = 0; i < args.size(); i++)
                    testUsesAllArgs &= (resVar1.Owner()->Inputs()[i] == args[i]); // test lambda ignores this input
            }
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
        // we test BatchSum(f(x,y))
        if (dataType == DataType::Double && testUsesAllArgs)
        {
            let epsScale = 1e-6;
            // gradient check for every input
            for (size_t i = 0; i < allArgValues[0].size(); i++)
            {
                // in case of batching, compute the sum symbolically so we can test the batched gradient
                Variable originalBatchSumVar;  // sum over batch over ReduceSum(test.f(args))
                Variable perturbedBatchSumVar; // same but arg[i] perturbed by epsilon
                unordered_map<Parameter, NDArrayViewPtr> epsilons;  // [arg] -> epsilon used to perturb arg[i]
                unordered_map<Parameter, NDArrayViewPtr> gradients; // [arg] -> gradient for arg[i] goes here
                for (size_t n = 0; n < N; n++)
                {
                    let& argValues = allArgValues[n];

                    // original arguments
                    args.clear();
                    for (let& argValue : argValues)
                        args.push_back(Parameter(argValue)); // TODO: does Backward() really need to take Parameters? Why not any Variable? Can Constants take a gradient??
                    // perturbed arguments
                    let eps = RandomTestTensor(args[i].Shape(), epsScale, "eps", i, seed, dataType, device);
                    //eps->LogToFile(L"eps", stderr);
                    auto perturbedArgs = args;
                    perturbedArgs[i] = Constant(perturbedArgs[i].Value() + eps);

                    // compute test function, original and perturbed, and ReduceSum
                    Variable originalBatchSumVar1  = ReduceSum(test.f(args),          Axis::AllStaticAxes());
                    Variable perturbedBatchSumVar1 = ReduceSum(test.f(perturbedArgs), Axis::AllStaticAxes());

                    // batch sum, original and perturbed
                    originalBatchSumVar  = n == 0 ? originalBatchSumVar1  : (originalBatchSumVar  + originalBatchSumVar1);
                    perturbedBatchSumVar = n == 0 ? perturbedBatchSumVar1 : (perturbedBatchSumVar + perturbedBatchSumVar1);

                    // remember the epsilons and the gradients we need to compute
                    let arg = Parameter(args[i]); // we use this below as a key
                    epsilons[arg]  = eps;
                    gradients[arg] = nullptr;
                }

                // compute gradient of sum over all elements of test.f(args) (=backprop a 1.0 into every element)
                originalBatchSumVar.Backward(gradients); // this triggers batched backward computation

                // compute output perturbation based on those added epsilons
                let originalBatchSum  = originalBatchSumVar .Value()->AsScalar<double>();
                let perturbedBatchSum = perturbedBatchSumVar.Value()->AsScalar<double>();
                //originalBatchSumVar .Value()->LogToFile(L"originalBatchSum", stderr);
                //perturbedBatchSumVar.Value()->LogToFile(L"perturbedBatchSum", stderr);
                let perturbedDelta = perturbedBatchSum - originalBatchSum;

                // compute expected output perturbation based on gradients
                double gradientBasedDelta = 0;
                for (let& kv : gradients)
                {
                    let& arg = kv.first;
                    let eps = epsilons[arg];
                    let gradientWrtInput = gradients[arg]; // get gradient for arg
                    //gradientWrtInput->LogToFile(L"gradientWrtInput", stderr);
                    // compute expected perturbed output based on gradient
                    // gradientWrtInput[j,k] = slope of sum of all outputs w.r.t. changes of arg[j,k]
                    gradientBasedDelta += SumAll(gradientWrtInput * eps, dataType, device);
                }

                // check result
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
    numFailed += DynamiteTest(3, DataType::Double, DeviceDescriptor::CPUDevice());
    numFailed += DynamiteTest(1, DataType::Double, DeviceDescriptor::GPUDevice(0));
    numFailed += DynamiteTest(3, DataType::Float,  DeviceDescriptor::GPUDevice(0));
#if 1 // do this not every time
    numFailed += DynamiteTest(1, DataType::Float,  DeviceDescriptor::GPUDevice(0));
    numFailed += DynamiteTest(1, DataType::Double, DeviceDescriptor::CPUDevice());
    numFailed += DynamiteTest(1, DataType::Float,  DeviceDescriptor::CPUDevice());
#endif
    if (numFailed > 0)
        LogicError("RunDynamiteTests: %d tests failed.", (int)numFailed);
}
