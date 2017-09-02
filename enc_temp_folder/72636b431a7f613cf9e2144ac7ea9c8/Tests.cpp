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
static NDArrayViewPtr TestTensor(const NDShape& shape, double scale, const char* opName, size_t argIndex, DataType dataType, const DeviceDescriptor& device)
{
    static unsigned long seed = 1;
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
    auto res = scale > 0 ? randT(/*mean=*/0., /*stdDev=*/scale) : constT(scale); // (RandomNormal will fail for scale=0)
    // some special cases
    if (strstr(opName, "Log") && !strstr(opName, "LogSum")) // Log requires positive numbers -> use abs(x) + 0.1
    {
        res = NDArrayView::NumericOperation({ res }, 1.0, L"Abs");
        res = NDArrayView::NumericOperation({ constT(0.1) }, 1.0, L"Copy", nullptr, /*beta=*/1.0);
    }
    else if (strcmp(opName, "Pow") == 0 && argIndex == 0) // Pow requires non-negative base -> use abs(x) + 1
    {
        res = NDArrayView::NumericOperation({ res }, 1.0, L"Abs");
        res = NDArrayView::NumericOperation({ constT(1.0) }, 1.0, L"Copy", nullptr, /*beta=*/1.0);
    }
    else if (strcmp(opName, "Reciprocal") == 0) // Reciprocal should not use too small a number -> use abs(x) + 0.1
    {
        res = NDArrayView::NumericOperation({ res }, 1.0, L"Abs");
        res = NDArrayView::NumericOperation({ constT(0.1) }, 1.0, L"Copy", nullptr, /*beta=*/1.0);
    }
    else if (strcmp(opName, "Cond") == 0 && argIndex == 0) // Cond requires a flag as the condition
    {
        res = NDArrayView::NumericOperation({ res, constT(0.1) }, 1.0, L"Less"); // compare against some threshold
    }
    return res;
}
static Parameter TestParameter(const NDShape& shape, double scale, const char* opName, size_t argIndex, DataType dataType, const DeviceDescriptor& device)
{
    return Parameter(TestTensor(shape, scale, opName, argIndex, dataType, device));
}

// helper to compute average square error between two NDArrayViews
static double AvSqrErr(const NDArrayViewPtr& resVal, const NDArrayViewPtr& refVal, DataType dataType, const DeviceDescriptor& device)
{
    if (resVal->Shape() != refVal->Shape())
        LogicError("AvSqrErr: Result shape %S is different from expected shape %S", resVal->Shape().AsString().c_str(), refVal->Shape().AsString().c_str());
    let sqrErr = NDArrayView::NumericOperation({ resVal, refVal }, 1.0 / refVal->Shape().TotalSize(), L"SqrOfDifference", make_shared<NDArrayView>(dataType, NDShape{}, device), 0, L"Sum");
    return sqrErr->AsScalar<double>();
}

static double SumAll(const NDArrayViewPtr& x, DataType dataType, const DeviceDescriptor& device)
{
    let sum = NDArrayView::NumericOperation({ x }, 1.0, L"Copy", make_shared<NDArrayView>(dataType, NDShape{}, device), 0, L"Sum");
    return sum->AsScalar<double>();
}

size_t DynamiteTest(size_t N, DataType dataType, const DeviceDescriptor& device)
{
    // for testing batch normalization, we need shared several parameters
#define BN_SHAPE { 13, 42 }
    NDArrayViewPtr batchMean, batchInvStd;
    let bnRunningMean   = TestParameter(NDShape(BN_SHAPE), 0, "bnRunningMean",   0, dataType, device);
    let bnRunningInvStd = TestParameter(NDShape(BN_SHAPE), 0, "bnRunningInvStd", 0, dataType, device);
    let bnRunningCount  = TestParameter(NDShape{}        , 0, "bnRunningCount",  0, dataType, device);
    let batchNormFwd = [&](const vector<NDArrayViewPtr>& argValues) -> NDArrayViewPtr
    {
        //batchMean->LogToFile(L"batchMean", stderr);
        //batchInvStd->LogToFile(L"batchInvStd", stderr);
        return ((argValues[0] - batchMean) * batchInvStd) * argValues[1] + argValues[2];
    };
    // for testing splicing
    let doSplice = [&](const vector<NDArrayViewPtr>& argValues, size_t axis) -> NDArrayViewPtr
    {
        vector<size_t> totalShape(axis+1, 1); // total shape
        // first check all dims and determinethe shared shape
        size_t splicedDim = 0;
        for (let& val : argValues)
        {
            let& shape = val->Shape();
            if (shape.Rank() > totalShape.size())
                totalShape.resize(shape.Rank(), 1);
            for (size_t k = 0; k < shape.Rank(); k++)
            {
                if (totalShape[k] != shape[k] && totalShape[k] != 1 && shape[k] != 1) // shapes must match, considering broadcasting
                    InvalidArgument("doSplice: incompatible shapes");
                if (shape[k] != 1)
                    totalShape[k] = shape[k]; // collect the axis
            }
            splicedDim += axis < shape.Rank() ? shape[axis] : 1; // accumulate the total dimension for the spliced axis
        }
        // now implant the spliced dimension into totalShape
        totalShape[axis] = splicedDim;
        // allocate result
        let& val0 = argValues[0];
        let out = make_shared<NDArrayView>(0, val0->GetDataType(), totalShape, val0->Device());
        // copy all items one by one
        size_t sliceStart = 0;
        for (let& val : argValues)
        {
            let& shape = val->Shape();
            let sliceHeight = axis < shape.Rank() ? shape[axis] : 1;
            // slice in output
            auto startOffsets = vector<size_t>(totalShape.size(), 0);
            auto extents = totalShape;
            startOffsets[axis] = sliceStart;
            extents[axis] = sliceHeight;
            let outSlice = out->Slice(startOffsets, extents);
            // copy value
            // CopyFrom() does not presently support strides, so use NumericOperation. TODO: Fix this, and test it here.
            //val->LogToFile(L"val");
            NDArrayView::NumericOperation({ val }, 1.0, L"Copy", outSlice);
            sliceStart += sliceHeight;
        }
        //out->LogToFile(L"out");
        return out;
    };
    // definition of all tests
#define Op(opCode) (pair<function<NDArrayViewPtr(const vector<NDArrayViewPtr>&)>, const char*>([=](const vector<NDArrayViewPtr>& argValues){ return NDArrayView::NumericOperation(argValues, 1.0, L#opCode); }, #opCode))
#define RedOp(redOpCode, shape, denom) (pair<function<NDArrayViewPtr(const vector<NDArrayViewPtr>&)>, const char*>([=](const vector<NDArrayViewPtr>& argValues){ return NDArrayView::NumericOperation(argValues, 1.0/denom, L"Copy", make_shared<NDArrayView>(dataType, NDShape(shape), device), 0, L#redOpCode); }, "Reduce" #redOpCode))
#define OpWithRed(opCode, shape) (pair<function<NDArrayViewPtr(const vector<NDArrayViewPtr>&)>, const char*>([=](const vector<NDArrayViewPtr>& argValues){ return NDArrayView::NumericOperation(argValues, 1.0, L#opCode, make_shared<NDArrayView>(dataType, NDShape(shape), device), 0, L"Sum"); }, #opCode "|Reduce"))
    vector<TensorViewTest> tests =
    {
        // splicing. Uniform splicing along last dimension will use single-kernel Gather; otherwise use multiple copy ops. Test both, also batched.
        { { [&](const vector<NDArrayViewPtr>& argValues) { return doSplice(argValues, 0); }, "Splice" }, [&](const vector<Variable>& args) { return CNTK::Splice(args, Axis(0)); },{ { 2, 1 },{ 1, 3 },{ 1, 1 } } },          // messy shapes -> individual copy ops
        { { [&](const vector<NDArrayViewPtr>& argValues) { return doSplice(argValues, 2); }, "Splice" }, [&](const vector<Variable>& args) { return CNTK::Splice(args, Axis(2)); },{ { 2, 1 },{ 1, 3 },{ 1, 1 },{ 2, 3 } } }, // messy shapes, new axis
        { { [&](const vector<NDArrayViewPtr>& argValues) { return doSplice(argValues, 1); }, "Splice" }, [&](const vector<Variable>& args) { return CNTK::Splice(args, Axis(1)); },{ { 2, 1 },{ 1, 3 },{ 1, 1 } } },          // messy shapes
        { { [&](const vector<NDArrayViewPtr>& argValues) { return doSplice(argValues, 1); }, "Splice" }, [&](const vector<Variable>& args) { return CNTK::Splice(args, Axis(1)); },{ { 13, 42 },{ 13, 42 },{ 13, 42 } } },    // all same size -> optimized gather
        // BatchNorm. This is tricky, since it only makes sense when batching. Requires some special-casing in the test code below.
        { { [&](const vector<NDArrayViewPtr>& argValues) { return batchNormFwd(argValues); }, "BatchNormalization" }, [&](const vector<Variable>& args) { return CNTK::BatchNormalization(args[0], /*id=*/1, args[1], args[2], bnRunningMean, bnRunningInvStd, bnRunningCount, /*spatial=*/false, 0, 0, 0); },{ BN_SHAPE, BN_SHAPE, BN_SHAPE } },
        // dot product (both explicitly as InnerProduct() as well as composition, to verify the InnerProduct() optimization)
        { OpWithRed(ElementwiseProduct, NDShape({ 1,    42    })), [](const vector<Variable>& args) { return CNTK::InnerProduct(args[0], args[1], Axis(0)); }, { { 13,    42    }, { 13,    42    } } },
        { OpWithRed(ElementwiseProduct, NDShape({ 1           })), [](const vector<Variable>& args) { return CNTK::InnerProduct(args[0], args[1], Axis(0)); }, { { 13           }, { 13           } } },
        { OpWithRed(ElementwiseProduct, NDShape({ 1, 1, 42, 5 })), [](const vector<Variable>& args) { return CNTK::ReduceSum(CNTK::ReduceSum(CNTK::ElementTimes(args[0], args[1]), Axis(0)), Axis(1)); }, { { 13, 2, 42, 5 }, { 13, 2, 42, 5 } } },
        { OpWithRed(ElementwiseProduct, NDShape({ 1,    42    })), [](const vector<Variable>& args) { return                 CNTK::ReduceSum(CNTK::ElementTimes(args[0], args[1]), Axis(0));           }, { { 13,    42    }, { 13,    42    } } },
        { OpWithRed(ElementwiseProduct, NDShape({ 1           })), [](const vector<Variable>& args) { return                 CNTK::ReduceSum(CNTK::ElementTimes(args[0], args[1]), Axis(0));           }, { { 13           }, { 13           } } },
        // splicing. NDArrayView::Slice() and SliceView() differ in also slicing the matrix or not. Test both.
        // slicing, reshaping   --TODO: reshaping (should be easy)
        { { [&](const vector<NDArrayViewPtr>& argValues) { return argValues[0]->Slice           ({ 0, 3 }, {     13 }); }, "Index" }, [&](const vector<Variable>& args) { return CNTK::Index(args[0], 3); },{ { 13, 42 } } }, // index of rank > 1; also testing SlicedTensorView()
        { { [&](const vector<NDArrayViewPtr>& argValues) { return argValues[0]->SliceView       ({    1 }, {        }); }, "Index" }, [&](const vector<Variable>& args) { return CNTK::Index(args[0], 1); },{ { 13 } } }, // index of rank 1
        { { [&](const vector<NDArrayViewPtr>& argValues) { return argValues[0]->SliceView       ({ 0, 1 }, { 13,  4 }); }, "Slice" }, [&](const vector<Variable>& args) { return CNTK::Slice(args[0], { Axis(0), Axis(1) }, { 0, 1 }, { 13, 1+4 }); },{ { 13, 42 } } }, // multi-axis slice
        { { [&](const vector<NDArrayViewPtr>& argValues) { return argValues[0]->Slice           ({ 2, 0 }, {  3, 42 }); }, "Slice" }, [&](const vector<Variable>& args) { return CNTK::Slice(args[0], { Axis(0) }, { 2 }, { 2+3 }); },{ { 13, 42 } } }, // non-contiguous slice
        { { [&](const vector<NDArrayViewPtr>& argValues) { return argValues[0]->SliceView       ({ 0, 1 }, { 13,  4 }); }, "Slice" }, [&](const vector<Variable>& args) { return CNTK::Slice(args[0], { Axis(1) }, { 1 }, { 1+4 }); },{ { 13, 42 } } }, // contiguous slice of rank > 1
        { { [&](const vector<NDArrayViewPtr>& argValues) { return argValues[0]->Slice           ({ 0, 1 }, { 13,  4 }); }, "Slice" }, [&](const vector<Variable>& args) { return CNTK::Slice(args[0], { Axis(1) }, { 1 }, { 1+4 }); },{ { 13, 42 } } }, // same but testing SlicedTensorView() on the reference path
        { { [&](const vector<NDArrayViewPtr>& argValues) { return argValues[0]->SliceView       ({    1 }, {      3 }); }, "Slice" }, [&](const vector<Variable>& args) { return CNTK::Slice(args[0], { Axis(0) }, { 1 }, { 1+3 }); },{ { 13 } } }, // slice of rank 1
        // matrix product
        { { [&](const vector<NDArrayViewPtr>& argValues) { return NDArrayView::MatrixProduct(false, argValues[0], false, argValues[1], false, 1.0, 1); }, "Times_shared"   }, [&](const vector<Variable>& args) { return CNTK::Times         (args[0], args[1]   ); },{ { 13, 42 },{ 42, 9 } } },
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
    size_t numFailed = 0;
    for (let& test : tests) // loop over all tests
    {
        let aryness = test.shapes.size(); // number of arguments of this test
        let isBatchNorm = strstr(test.op.second, "BatchNormalization"); // BatchNormalization requires some special-casing
        // prepare example test tensors for all batch items
        vector<vector<NDArrayViewPtr>> allArgValues(N, vector<NDArrayViewPtr>(aryness)); // [batchIndex][argIndex]
        for (size_t n = 0; n < N; n++) // loop over samples
        {
            for (size_t j = 0; j < aryness; j++)
            {
                let isShared = // some tests share arguments across batch items
                    (isBatchNorm && j > 0) ||
                    (strstr(test.op.second, "Times_shared") && j == 0);
#define if_x(x) (x)? // helpers to make ternary expressions a bit more readable
#define else_x :
                allArgValues[n][j] =
                    if_x (n > 0 && isShared)
                        allArgValues[0][j] // some ops require args to be shared across the batch items
                    else_x
                        TestTensor(test.shapes[j], 0.3, test.op.second, j, dataType, device);
            }
        }
        // special case: for BatchNormalization reference, we manually compute mean and inv stddev here
        if (isBatchNorm)
        {
            if (N == 1) // BatchNormalization only makes sense when batching. Otherwise skip
                continue;
            // get the batch (first argument only; that's the data)
            vector<NDArrayViewPtr> batchItems;
            for (size_t n = 0; n < N; n++) // loop over samples
                batchItems.push_back(allArgValues[n][0]);
            let batch = NDArrayView::GatherBatch(batchItems, /*axis=*/(int)batchItems[0]->Shape().Rank());
            // create them in the right shape. This will inform the reduction.
            batchMean = make_shared<NDArrayView>(batchItems[0]->GetDataType(), batchItems[0]->GetStorageFormat(), batchItems[0]->Shape(), device);
            auto batchSqrMean = make_shared<NDArrayView>(batchItems[0]->GetDataType(), batchItems[0]->GetStorageFormat(), batchItems[0]->Shape(), device);
            let alpha = (double)batchMean->Shape().TotalSize() / (double)batch->Shape().TotalSize();
            NDArrayView::NumericOperation({ batch }, alpha, L"Copy", batchMean);
            NDArrayView::NumericOperation({ batch - batchMean }, alpha, L"Sqr",  batchSqrMean);
            let minusHalf = make_shared<NDArrayView>(-0.5, batchItems[0]->GetDataType(), NDShape{ }, device, true);
            batchInvStd = NDArrayView::NumericOperation({ batchSqrMean, minusHalf }, 1.0, L"Pow"); // x^{-0.5}
            //batchMean->LogToFile(L"batchMean", stderr);
            //batchInvStd->LogToFile(L"batchInvStd", stderr);
            //auto batchInvStdMin = make_shared<NDArrayView>(batchItems[0]->GetDataType(), batchItems[0]->GetStorageFormat(), NDShape{}, device);
            //NDArrayView::NumericOperation({ batchInvStd }, 1, L"Copy", batchInvStdMin, 0, L"Min");
            //batchInvStdMin->LogToFile(L"batchInvStdMin", stderr);
        }
        // reference computation (NDArrayView)
        NDArrayViewPtr refVal;
        for (size_t n = 0; n < N; n++) // aggregate over all samples in the MB with alternating sign
        {
            let refVal1 = test.op.first(allArgValues[n]);
            refVal =
                if_x (n == 0)
                    refVal1
                else_x if_x (n&1)
                    refVal - refVal1
                else_x
                    refVal + refVal1;
#if 0
            for (let& arg : argValues)
                arg->LogToFile(L"argVal", stderr);
            refVal1->LogToFile(L"resVal", stderr);
            refVal->LogToFile(L"sumVal", stderr);
#endif
        }
        // Dynamite arguments: prepare Dynamite Variables for all test of for all the samples in the MB
        vector<vector<Variable>> allArgs(N, vector<Variable>(aryness)); // [batchIndex][argIndex]
        for (size_t n = 0; n < N; n++)
        {
            for (size_t j = 0; j < aryness; j++)
            {
                allArgs[n][j] =
                    if_x (n > 0 && allArgValues[n][j] == allArgValues[0][j])
                        allArgs[0][j] // for ops that share args across batch items: share the Variables the same way
                    else_x
                        Parameter(allArgValues[n][j]); // all args are Parameterse so that we can take the gradient
            }
            if (n == 0) // logging
            {
                fprintf(stderr, "%25s(", test.op.second);
                for (let& arg : allArgs[n])
                    fprintf(stderr, " %S ", arg.Shape().AsString().c_str());
            }
        }
        // Dynamite computation. Result is sum with alternating sign over all batch items in this test.
        let functionUnderRest = [&](const vector<vector<Variable>>& testArgs) -> Variable
        {
            Variable res;
            for (size_t n = 0; n < N; n++) // aggregate over all samples in the MB
            {
                let itemRes = test.f(testArgs[n]);
                res =
                    if_x (n == 0)
                        itemRes
                    else_x if_x (n&1)
                        res - itemRes
                    else_x
                        res + itemRes;
            }
            return res;
        };
        let resVar = functionUnderRest(allArgs);
        let resVal = resVar.Value(); // this triggers the batched evaluation
        fprintf(stderr, ") -> %S\n", resVal->AsString().c_str());
        // compare reference result with Dynamite result
        let avSqrErr = AvSqrErr(resVal, refVal, dataType, device);
        if (avSqrErr > 1e-5)
        {
            fprintf(stderr, ">>>>>>>>>> FWD FAILED: avSqrErr = %.10f\n", avSqrErr);
            numFailed++;
        }
        // gradient check
        // We test the gradient for BatchSum(f(args)); that is d BatchSum(f(args))/d args.
        // We already have evaluated the test function. We compare two things:
        //  - numeric:  perturb the all inputs by an epsilon; compute the test function on that
        //  - symbolic: get all gradients from the test function and multiply with the same epsilon and add to the unperturbed result
        // We test each argument index separately, to make this test more informative.
        if (dataType == DataType::Double
            //&& !strstr(test.op.second, "Splice") // PUT THIS BACK once scatter works as well
            )
        {
            let epsScale = 1e-7;
            for (size_t argIndexUnderTest = 0; argIndexUnderTest < aryness; argIndexUnderTest++)
            {
                // some args are not differentiable: skip those
                if (strstr(test.op.second, "Clip") && argIndexUnderTest > 0)
                    continue;
                if (strstr(test.op.second, "Cond") && argIndexUnderTest == 0)
                    continue;
                // determine all gradients. That is args[*][argIndexUnderTest].
                // Note: args that are shared across the batch will only get a single entry in the gradients[] map
                unordered_map<Parameter, NDArrayViewPtr> gradients; // [argVariable] -> symbolic gradient for arg[n][i] goes here
                for (size_t n = 0; n < N; n++)
                    gradients[Parameter(allArgs[n][argIndexUnderTest])] = nullptr; // (Parameter() is just a type cast; it maintains object identity of its input Variable)
                // create epsilon tensors for every numeric gradient
                // Args that are shared will only get one epsilon.
                unordered_map<Variable, NDArrayViewPtr> epsilons; // [argVariable] -> numeric epsilon for a gradient
                for (let& kv : gradients)
                {
                    let& arg = kv.first;
                    epsilons[arg] = TestTensor(arg.Shape(), epsScale, "eps", 0, dataType, device);
                }
                // determine perturbed arguments for all inputs
                unordered_map<Variable, Variable> perturbedArgSet; // [argVariable] -> perturbed version of argument Variable
                for (let& kv : gradients)
                {
                    let& arg = kv.first;
                    perturbedArgSet[arg] = Constant(arg.Value() + epsilons[arg]);
                }
                vector<vector<Variable>> allPerturbedArgs(N, vector<Variable>(aryness)); // [batchIndex][argIndex]
                for (size_t n = 0; n < N; n++)
                {
                    for (size_t j = 0; j < aryness; j++)
                    {
                        auto arg = allArgs[n][j];
                        let iter = perturbedArgSet.find(arg);
                        if (iter != perturbedArgSet.end()) // (for args that have no gradient, we pass the original arg without epsilon)
                            arg = iter->second;
                        allPerturbedArgs[n][j] = arg;
                    }
                }
                // evaluate original (once again since we now also reduce to a scalar) as well as at the perturbed point
                let  originalBatchSumVar = ReduceSum(functionUnderRest(allArgs         ), Axis::AllStaticAxes())->Output();
                let perturbedBatchSumVar = ReduceSum(functionUnderRest(allPerturbedArgs), Axis::AllStaticAxes())->Output();
                // compute output perturbation due to those added epsilons
                let  originalBatchSum =  originalBatchSumVar.Value()->AsScalar<double>();
                let perturbedBatchSum = perturbedBatchSumVar.Value()->AsScalar<double>();
                //originalBatchSumVar .Value()->LogToFile(L"originalBatchSum", stderr);
                //perturbedBatchSumVar.Value()->LogToFile(L"perturbedBatchSum", stderr);
                let perturbationBasedDelta = perturbedBatchSum - originalBatchSum;

                // symbolic gradient: compute gradient of sum over all elements of test.f(args) (=backprop a 1.0 into every element)
                originalBatchSumVar.Backward(gradients); // this triggers batched backward computation
                // compute expected output perturbation based on gradients
                double gradientBasedDelta = 0;
                for (let& kv : gradients)
                {
                    let& arg = kv.first;
                    let gradientWrtInput = kv.second;
                    //gradientWrtInput->LogToFile(L"gradientWrtInput_" + to_wstring(argIndexUnderTest), stderr);
                    let eps = epsilons[arg]; // epsilon used to perturb
                    // compute expected perturbed output based on gradient
                    // gradientWrtInput[j,k] = slope of sum of all outputs w.r.t. changes of arg[j,k]
                    gradientBasedDelta += SumAll(gradientWrtInput * eps, dataType, device);
                }

                // check result
                let relErr = (perturbationBasedDelta == gradientBasedDelta) ? 0 : fabs(((perturbationBasedDelta - gradientBasedDelta) / perturbationBasedDelta));
                if (relErr > 1e-5)
                    fprintf(stderr, ">>>>>>>>>> BWD[%d] FAILED: err=%.10f%% (numeric=%.20f, symbolic=%.20f)\n", (int)argIndexUnderTest, 100.0 * relErr, perturbationBasedDelta, gradientBasedDelta);
                // Once in a while enable the following, to see whether there are some broken tests. We do have zeroes e.g. for Equal().
                //else if (gradientBasedDelta == 0)
                //    fprintf(stderr, ">>>>>>>>>> BWD[%d] IS 0??\n", (int)argIndexUnderTest);
            }
        }
    } // loop over tests
    return numFailed;
}

void RunDynamiteTests()
{
    size_t numFailed = 0;
    numFailed += DynamiteTest(1, DataType::Double, DeviceDescriptor::GPUDevice(0));
    numFailed += DynamiteTest(3, DataType::Double, DeviceDescriptor::GPUDevice(0));
    numFailed += DynamiteTest(3, DataType::Double, DeviceDescriptor::CPUDevice());
    numFailed += DynamiteTest(3, DataType::Float,  DeviceDescriptor::GPUDevice(0));
#if 1 // do this not every time
    numFailed += DynamiteTest(1, DataType::Float,  DeviceDescriptor::GPUDevice(0));
    numFailed += DynamiteTest(1, DataType::Double, DeviceDescriptor::CPUDevice());
    numFailed += DynamiteTest(1, DataType::Float,  DeviceDescriptor::CPUDevice());
#endif
    if (numFailed > 0)
        LogicError("RunDynamiteTests: %d tests failed.", (int)numFailed);
}
