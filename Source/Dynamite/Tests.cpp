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
    if ((strstr(opName, "Log") && !strstr(opName, "LogSum")) || strcmp(opName, "Sqrt") == 0) // Log and Sqrt require positive numbers -> use abs(x) + 0.1
    {
        res = NDArrayView::NumericOperation({ res }, 1.0, L"Abs");
        res += constT(0.1);
    }
    else if (strcmp(opName, "Pow") == 0 && argIndex == 0) // Pow requires non-negative base -> use abs(x) + 1
    {
        res = NDArrayView::NumericOperation({ res }, 1.0, L"Abs");
        res += constT(1.0);
    }
    else if (strcmp(opName, "Reciprocal") == 0) // Reciprocal should not use too small a number -> use abs(x) + 0.2
    {
        res = NDArrayView::NumericOperation({ res }, 1.0, L"Abs");
        res += constT(0.2);
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

static vector<Axis> AxisVector(const vector<size_t>& axisIndexVector) // create vector<Axis> from vector<axis index>
{
    return vector<Axis>(Transform(axisIndexVector, [&](size_t axisIndex) -> Axis { return Axis(axisIndex); }));
}

#ifndef _MSC_VER // gcc won't eat this with gazillion errors, so forget about it
size_t DynamiteTest(size_t N, DataType dataType, bool testStackingEnabled, const DeviceDescriptor& device);
#else
size_t DynamiteTest(size_t N, DataType dataType, bool testStackingEnabled, const DeviceDescriptor& device)
{
    // for testing batch normalization, we need shared several parameters
#define BN_SHAPE { 13 }
//#define BN_SHAPE { 13, 42 } // BUGBUG: Will fail because auto-batch only BatchNorms vectors presently. 
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
        NDShapeDimensions totalShape(axis+1, 1); // total shape
        // first check all dims and determinethe shared shape
        NDShapeDimension splicedDim = 0;
        for (let& val : argValues)
        {
            let& shape = val->Shape();
            if (shape.Rank() > totalShape.size())
            {
                auto paddedShape = MakeVector(totalShape);
                paddedShape.resize(shape.Rank(), 1); // (NDShapeDimensions is not resizable, so need to take a detour)
                totalShape = NDShapeDimensions(move(paddedShape));
            }
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
        NDShapeDimension sliceStart = 0;
        for (let& val : argValues)
        {
            let& shape = val->Shape();
            let sliceHeight = axis < shape.Rank() ? shape[axis] : 1;
            // slice in output
            auto startOffsets = NDShapeDimensions(totalShape.size(), 0);
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
    //  - Var* = CNTK function under test, implemented on Variables
    //  - Val* = reference operation, implemented on values (NDArrayView)
#define VarExpr(expr) ([&](const vector<Variable>& args) { return expr; })
#define ValExpr(expr) ([&](const vector<NDArrayViewPtr>& argValues) { return expr; })
#define ValOp(opCode) (pair<function<NDArrayViewPtr(const vector<NDArrayViewPtr>&)>, const char*>(ValExpr(NDArrayView::NumericOperation(argValues, 1.0, L#opCode)), #opCode))
#define ValRedOp(redOpCode, shape, denom) (pair<function<NDArrayViewPtr(const vector<NDArrayViewPtr>&)>, const char*>(ValExpr(NDArrayView::NumericOperation(argValues, 1.0/denom, L"Copy", make_shared<NDArrayView>(dataType, NDShape(shape), device), 0, L#redOpCode)), "Reduce" #redOpCode))
#define ValOpWithRed(opCode, shape) (pair<function<NDArrayViewPtr(const vector<NDArrayViewPtr>&)>, const char*>(ValExpr(NDArrayView::NumericOperation(argValues, 1.0, L#opCode, make_shared<NDArrayView>(dataType, NDShape(shape), device), 0, L"Sum")), #opCode "|Reduce"))
    vector<TensorViewTest> tests =
    {
        // transpose
        { { ValExpr(argValues[0]->AsTransposed(NDShapePermutation{ 0, 2, 1, 3 })), "Transpose" }, VarExpr(CNTK::Transpose(args[0], AxisVector({ 0, 2, 1, 3 }))),{ { 13, 42, 4, 2 } } }, // 2-axis permutation
        { { ValExpr(argValues[0]->AsTransposed(NDShapePermutation{ 1, 0       })), "Transpose" }, VarExpr(CNTK::Transpose(args[0])),{ { 13, 42 } } }, // basic transpose
        { { ValExpr(argValues[0]->AsTransposed(NDShapePermutation{ 1, 2, 3, 0 })), "Transpose" }, VarExpr(CNTK::Transpose(args[0], AxisVector({ 1, 2, 3, 0 }))),{ { 13, 42, 4, 2 } } }, // axis rotation
        // splicing. Uniform splicing along last dimension will use single-kernel Gather; otherwise use multiple copy ops. Test both, also batched.
        { { ValExpr(doSplice(argValues, 2)), "Splice" }, VarExpr(CNTK::Splice(args, Axis(2))),{ {  2,  1 },{  1,  3 },{  1,  1 },{ 2, 3 } } }, // messy shapes, new axis
        { { ValExpr(doSplice(argValues, 2)), "Splice" }, VarExpr(CNTK::Splice(args, Axis(2))),{ {  2,  1 },{  1,  3 },{  1,  1 },{ 2, 3 } } }, // messy shapes, new axis
        { { ValExpr(doSplice(argValues, 0)), "Splice" }, VarExpr(CNTK::Splice(args, Axis(0))),{ {  2,  1 },{  1,  3 },{  1,  1 }          } }, // messy shapes -> individual copy ops
        { { ValExpr(doSplice(argValues, 1)), "Splice" }, VarExpr(CNTK::Splice(args, Axis(1))),{ {  2,  1 },{  1,  3 },{  1,  1 }          } }, // messy shapes
        { { ValExpr(doSplice(argValues, 1)), "Splice" }, VarExpr(CNTK::Splice(args, Axis(1))),{ { 13, 42 },{ 13, 42 },{ 13, 42 }          } }, // all same size -> optimized gather
        // BatchNorm. This is tricky, since it only makes sense when batching. Requires some special-casing in the test code below.
        { { ValExpr(batchNormFwd(argValues)), "BatchNormalization" }, VarExpr(CNTK::BatchNormalization(args[0], /*id=*/1, args[1], args[2], bnRunningMean, bnRunningInvStd, bnRunningCount, /*spatial=*/false, 0, 0, 0)),{ BN_SHAPE, BN_SHAPE, BN_SHAPE } },
        // dot product (both explicitly as InnerProduct() as well as composition, to verify the InnerProduct() optimization)
        { ValOpWithRed(ElementwiseProduct, NDShape({ 1,    42    })), VarExpr(CNTK::InnerProduct(args[0], args[1], Axis(0))), { { 13, 42    }, { 13, 42 } } },
        { ValOpWithRed(ElementwiseProduct, NDShape({ 1           })), VarExpr(CNTK::InnerProduct(args[0], args[1], Axis(0))), { { 13        }, { 13     } } },
        { ValOpWithRed(ElementwiseProduct, NDShape({ 1, 1, 42, 5 })), VarExpr(CNTK::ReduceSum(CNTK::ReduceSum(CNTK::ElementTimes(args[0], args[1]), Axis(0)), Axis(1))), { { 13, 2, 42, 5 }, { 13, 2, 42, 5 } } },
        { ValOpWithRed(ElementwiseProduct, NDShape({ 1,    42    })), VarExpr(                CNTK::ReduceSum(CNTK::ElementTimes(args[0], args[1]), Axis(0))          ), { { 13,    42    }, { 13,    42    } } },
        { ValOpWithRed(ElementwiseProduct, NDShape({ 1           })), VarExpr(                CNTK::ReduceSum(CNTK::ElementTimes(args[0], args[1]), Axis(0))          ), { { 13           }, { 13           } } },
        // splicing. NDArrayView::Slice() and SliceView() differ in also slicing the matrix or not. Test both.
        // slicing, reshaping   --TODO: reshaping (should be easy)
        { { ValExpr(argValues[0]->Slice    (NDShapeDimensions{ 0, 3 }, NDShapeDimensions{ 13     })), "Index" }, VarExpr(CNTK::Index(args[0], 3)),{ { 13, 42 } } }, // index of rank > 1; also testing SlicedTensorView()
        { { ValExpr(argValues[0]->SliceView(NDShapeDimensions{    1 }, NDShapeDimensions{        })), "Index" }, VarExpr(CNTK::Index(args[0], 1)),{ { 13 } } }, // index of rank 1
        { { ValExpr(argValues[0]->SliceView(NDShapeDimensions{ 0, 1 }, NDShapeDimensions{ 13,  4 })), "Slice" }, VarExpr(CNTK::Slice(args[0], { Axis(0), Axis(1) }, { 0, 1 }, { 13, 1+4 })),{ { 13, 42 } } }, // multi-axis slice
        { { ValExpr(argValues[0]->Slice    (NDShapeDimensions{ 2, 0 }, NDShapeDimensions{  3, 42 })), "Slice" }, VarExpr(CNTK::Slice(args[0], { Axis(0)          }, { 2    }, { 2+3     })),{ { 13, 42 } } }, // non-contiguous slice
        { { ValExpr(argValues[0]->SliceView(NDShapeDimensions{ 0, 1 }, NDShapeDimensions{ 13,  4 })), "Slice" }, VarExpr(CNTK::Slice(args[0], { Axis(1)          }, { 1    }, { 1+4     })),{ { 13, 42 } } }, // contiguous slice of rank > 1
        { { ValExpr(argValues[0]->Slice    (NDShapeDimensions{ 0, 1 }, NDShapeDimensions{ 13,  4 })), "Slice" }, VarExpr(CNTK::Slice(args[0], { Axis(1)          }, { 1    }, { 1+4     })),{ { 13, 42 } } }, // same but testing SlicedTensorView() on the reference path
        { { ValExpr(argValues[0]->SliceView(NDShapeDimensions{    1 }, NDShapeDimensions{      3 })), "Slice" }, VarExpr(CNTK::Slice(args[0], { Axis(0)          }, { 1    }, { 1+3     })),{ { 13     } } }, // slice of rank 1
        // matrix product
        { { ValExpr(NDArrayView::MatrixProduct(false, argValues[0], false, argValues[1], false, 1.0, 1)), "Times_shared"   }, VarExpr(CNTK::Times         (args[0], args[1]   )),{ { 13, 42 },{ 42, 9    } } },
        { { ValExpr(NDArrayView::MatrixProduct(false, argValues[0], false, argValues[1], false, 1.0, 1)), "Times"          }, VarExpr(CNTK::Times         (args[0], args[1]   )),{ { 13, 42 },{ 42, 9    } } },
        { { ValExpr(NDArrayView::MatrixProduct(false, argValues[0], false, argValues[1], false, 1.0, 1)), "Times"          }, VarExpr(CNTK::Times         (args[0], args[1]   )),{ { 13, 42 },{ 42       } } },
        { { ValExpr(NDArrayView::MatrixProduct(false, argValues[0], false, argValues[1], false, 1.0, 1)), "Times"          }, VarExpr(CNTK::Times         (args[0], args[1], 0)),{ {     42 },{ 42       } } }, // should get batched
        { { ValExpr(NDArrayView::MatrixProduct(false, argValues[0], false, argValues[1], false, 1.0, 1)), "Times"          }, VarExpr(CNTK::Times         (args[0], args[1]   )),{ { 13, 42 },{ 42, 9, 5 } } },
        { { ValExpr(NDArrayView::MatrixProduct(false, argValues[0], true,  argValues[1], false, 1.0, 1)), "TransposeTimes" }, VarExpr(CNTK::TransposeTimes(args[0], args[1]   )),{ { 42, 13 },{ 42, 9    } } },
        { { ValExpr(NDArrayView::MatrixProduct(false, argValues[0], true,  argValues[1], false, 1.0, 1)), "TransposeTimes" }, VarExpr(CNTK::TransposeTimes(args[0], args[1]   )),{ { 42, 13 },{ 42       } } },
        { { ValExpr(NDArrayView::MatrixProduct(false, argValues[0], true,  argValues[1], false, 1.0, 1)), "TransposeTimes" }, VarExpr(CNTK::TransposeTimes(args[0], args[1]   )),{ { 42, 13 },{ 42, 9, 3 } } },
        // ternary
        { ValOp(Clip), VarExpr(CNTK::Clip         (args[2], args[0], args[1])), { { 13,  1 }, { 13, 1 }, { 13, 42 } } },
        { ValOp(Cond), VarExpr(CNTK::ElementSelect(args[0], args[1], args[2])), { { 13, 42 }, { 13, 1 }, { 13,  1 } } },
        // binary
        { ValOp(Sum               ), VarExpr(CNTK::Plus         (args[0], args[1])), { { 13, 42     }, { 13, 42 } } },
        { ValOp(Difference        ), VarExpr(CNTK::Minus        (args[0], args[1])), { { 13, 42     }, { 13,  1 } } },
        { ValOp(ElementwiseProduct), VarExpr(CNTK::ElementTimes (args[0], args[1])), { { 13, 42     }, { 13,  1 } } },
        { ValOp(LogSum            ), VarExpr(CNTK::LogAddExp    (args[0], args[1])), { { 13, 42     }, { 13,  1 } } },
        { ValOp(Pow               ), VarExpr(CNTK::Pow          (args[0], args[1])), { { 13, 42, 12 }, { 13,  1 } } },
        { ValOp(Equal             ), VarExpr(CNTK::Equal        (args[0], args[1])), { { 13, 42     }, { 13,  1 } } },
        { ValOp(NotEqual          ), VarExpr(CNTK::NotEqual     (args[0], args[1])), { { 13, 42     }, { 13, 42 } } },
        { ValOp(Less              ), VarExpr(CNTK::Less         (args[0], args[1])), { { 13, 42     }, { 13,  1 } } },
        { ValOp(LessEqual         ), VarExpr(CNTK::LessEqual    (args[0], args[1])), { { 13, 42     }, { 13,  1 } } },
        { ValOp(Greater           ), VarExpr(CNTK::Greater      (args[0], args[1])), { { 13, 42     }, { 13,  1 } } },
        { ValOp(GreaterEqual      ), VarExpr(CNTK::GreaterEqual (args[0], args[1])), { { 13, 42     }, { 13,  1 } } },
        // unary
        { ValOp(LinearRectifier      ), VarExpr(CNTK::ReLU      (args[0])), { {  13, 42    } } },
        { ValOp(Tanh                 ), VarExpr(CNTK::Tanh      (args[0])), { {  13        } } },
        { ValOp(Log                  ), VarExpr(CNTK::Log       (args[0])), { {  13, 42    } } },
        { ValOp(Exp                  ), VarExpr(CNTK::Exp       (args[0])), { {  13, 42    } } },
        { ValOp(Cosine               ), VarExpr(CNTK::Cos       (args[0])), { {  13, 42    } } },
        { ValOp(Sin                  ), VarExpr(CNTK::Sin       (args[0])), { { 235, 13, 2 } } },
        { ValOp(Negate               ), VarExpr(CNTK::Negate    (args[0])), { {  13        } } },
        { ValOp(Floor                ), VarExpr(CNTK::Floor     (args[0])), { {  13, 42    } } },
        { ValOp(Abs                  ), VarExpr(CNTK::Abs       (args[0])), { {  13, 42    } } },
        { ValOp(Sqrt                 ), VarExpr(CNTK::Sqrt      (args[0])), { {  13, 42    } } },
        { ValOp(Reciprocal           ), VarExpr(CNTK::Reciprocal(args[0])), { {  13, 42    } } },
        { ValOp(ExponentialLinearUnit), VarExpr(CNTK::ELU       (args[0])), { {  13, 42    } } },
        { ValOp(StableSigmoid        ), VarExpr(CNTK::Sigmoid   (args[0])), { { 128        } } },
        // reductions
        { ValRedOp(Sum,    NDShape({  1     }), 1 ), VarExpr(CNTK::ReduceSum   (args[0], Axis(0)          )), { { 13     } } },
        { ValRedOp(Sum,    NDShape({ 13,  1 }), 1 ), VarExpr(CNTK::ReduceSum   (args[0], Axis(1)          )), { { 13, 42 } } },
        { ValRedOp(Sum,    NDShape({ 13     }), 1 ), VarExpr(CNTK::ReduceSum   (args[0], Axis_DropLastAxis)), { { 13, 42 } } }, // removes the last axis
        { ValRedOp(Sum,    NDShape({  1, 42 }), 1 ), VarExpr(CNTK::ReduceSum   (args[0], Axis(0)          )), { { 13, 42 } } },
        { ValRedOp(LogSum, NDShape({  1     }), 1 ), VarExpr(CNTK::ReduceLogSum(args[0], Axis(0)          )), { { 13     } } },
        { ValRedOp(Sum,    NDShape({  1     }), 13), VarExpr(CNTK::ReduceMean  (args[0], Axis(0)          )), { { 13     } } }
    };

    fprintf(stderr, "\n--- Running tests for batch of %d. %s on %S.%s\n\n", (int)N, CNTK::DataTypeName(dataType), device.AsString().c_str(), N == 1 ? " Unbatched." : testStackingEnabled ? " Stacking." : " Batching.");
    let profiler = Function::CreateDynamicProfiler(1, L"test");
    let doGradientCheck = dataType == DataType::Double;
    size_t numFailed = 0;
    for (let& test : tests) // loop over all tests
    {
        let isTimes     = strstr(test.op.second, "Times")              != nullptr;
        let isSplice    = strstr(test.op.second, "Splice")             != nullptr;
        let isSlice     = strstr(test.op.second, "Slice")              != nullptr;
        let isBatchNorm = strstr(test.op.second, "BatchNormalization") != nullptr; // BatchNormalization requires some special-casing
        let isReduction = strstr(test.op.second, "Reduc")              != nullptr; // InnerProduct and Reduce

        let aryness = test.shapes.size(); // number of arguments of this test

        let testStacking = testStackingEnabled &&
                           !isSplice    &&  // splice uses funky shapes on output, don't touch the last axis
                           !isSlice     &&  // slice reference has baked-in result dimensions that are off
                           !isReduction &&  // some reduction ops test reduction along first axis, so can't mess with the other axes
                           !isBatchNorm;    // TODO: update BatchNorm test to test stacking (batch along last dim); currently it batches
        if (testStackingEnabled && !testStacking) // if stacking-test requested but not possible for this op then don't bother
            continue;

        // prepare example test tensors for all batch items
        vector<vector<NDArrayViewPtr>> allArgValues(N, vector<NDArrayViewPtr>(aryness)); // [batchIndex][argIndex]
        // determine max input rank, which is used to test varying the batch dimension if present (N>1 only)
        size_t maxRank = 0;
        for (size_t j = 0; j < aryness; j++)
        {
            if (isTimes && j == 0) // ignore the Times weight in determining this
                continue;
            let& argShape = test.shapes[j];
            maxRank = max(maxRank, argShape.Rank());
        }
        for (size_t n = 0; n < N; n++) // loop over samples
        {
            for (size_t j = 0; j < aryness; j++)
            {
                let isTimesWeight = isTimes && j == 0; // is first arg of Times op
                let isShared = // some tests share arguments across batch items
                    (isBatchNorm && j > 0) ||
                    (strstr(test.op.second, "Times_shared") && j == 0);
                auto argShape = test.shapes[j].Dimensions();
                // patch up the last dimension to test stacking
                if (testStacking)
                {
                    let rank = argShape.size();
                    if (n > 0 && !isShared && !isTimesWeight && rank == maxRank &&
                        (!isTimes || rank > 1) && // first axis of a Times input is not a batch axis
                        !argShape.empty() && argShape.back() != 1)     // 1 could be meant to be broadcast, so don't break that
                    {
                        argShape.back() += (NDShapeDimension)n; // increase the batch dimension
                    }
                }
                allArgValues[n][j] =
                    /*if*/ (n > 0 && isShared) ?
                        allArgValues[0][j] // some ops require args to be shared across the batch items
                    /*else*/:
                        TestTensor(argShape, 0.3, test.op.second, j, dataType, device);
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
            NDArrayView::NumericOperation({ batch - batchMean }, alpha, L"Sqr", batchSqrMean);
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
            auto refVal1 = test.op.first(allArgValues[n]);
            // when testing stacking, the batch dimensions are different, so we sum up all columns instead
            let& shape1 = refVal1->Shape();
            if (testStacking && shape1.Rank() > 0)
                refVal1 = NDArrayView::NumericOperation({ refVal1 }, /*alpha=*/1.0, L"Copy", shape1.SubShape(0, shape1.Rank() - 1).AppendShape({ 1 }));
            refVal =
                /*if*/ (n == 0) ?
                    refVal1
                /*else if*/: (n&1) ?
                    refVal - refVal1
                /*else*/:
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
                    /*if*/ (n > 0 && allArgValues[n][j] == allArgValues[0][j]) ?
                        allArgs[0][j] // for ops that share args across batch items: share the Variables the same way
                    /*else*/:
                        Parameter(allArgValues[n][j]); // all args are Parameterse so that we can take the gradient
            }
            if (n == 0) // logging
            {
                fprintf(stderr, "#%25s(", test.op.second);
                for (let& arg : allArgs[n])
                    fprintf(stderr, " %S ", arg.Shape().AsString().c_str());
            }
        }
        // Dynamite computation. Result is sum with alternating sign over all batch items in this test.
        let functionUnderRest = [&](const vector<vector<Variable>>& testArgs) -> Variable
        {
            let prevProfiler = Function::SetDynamicProfiler(profiler, false); // set to true to see the detailed log
            Variable res;
            for (size_t n = 0; n < N; n++) // aggregate over all samples in the MB
            {
                auto itemRes = test.f(testArgs[n]);
                let& shape1 = itemRes.Shape();
                if (testStacking && shape1.Rank() > 0)
                    itemRes = CNTK::ReduceSum(itemRes, Axis((int)shape1.Rank() - 1));
                res =
                    /*if*/ (n == 0) ?
                        itemRes
                    /*else if*/: (n&1) ?
                        res - itemRes
                    /*else*/:
                        res + itemRes;
            }
            Function::SetDynamicProfiler(prevProfiler);
            return res;
        };
        let resVar = functionUnderRest(allArgs);
        let resVal = resVar.Value(); // this triggers the batched evaluation
        fprintf(stderr, ") -> %S\n", resVal->AsString().c_str());
        // compare reference result with Dynamite result
        let avSqrErr = AvSqrErr(resVal, refVal, dataType, device);
        if (isnan(avSqrErr) || avSqrErr > 1e-5)
        {
            fprintf(stderr, ">>>>>>>>>> FWD FAILED: avSqrErr = %.10f\n", avSqrErr);
            resVal->LogToFile(L"result (Dynamite)");
            refVal->LogToFile(L"reference (NDArrayView)");
            numFailed++;
        }
        // gradient check
        // We test the gradient for BatchSum(f(args)); that is d BatchSum(f(args))/d args.
        // We already have evaluated the test function. We compare two things:
        //  - numeric:  perturb the all inputs by an epsilon; compute the test function on that
        //  - symbolic: get all gradients from the test function and multiply with the same epsilon and add to the unperturbed result
        // We test each argument index separately, to make this test more informative.
        if (doGradientCheck)
        {
            let epsScale = 1e-7;
            for (size_t argIndexUnderTest = 0; argIndexUnderTest < aryness; argIndexUnderTest++)
            {
                // some args are not differentiable: skip those
                if (strstr(test.op.second, "Clip") && argIndexUnderTest < 2) // note: arg order different for CNTK::Clip(). Here defined by opClip.
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
                let absErr = fabs(perturbationBasedDelta - gradientBasedDelta);
                let relErr = (perturbationBasedDelta == gradientBasedDelta) ? 0 : fabs(((perturbationBasedDelta - gradientBasedDelta) / perturbationBasedDelta));
                if (isnan(relErr) || (relErr > 1e-3 && absErr > 1e-10))
                {
                    fprintf(stderr, ">>>>>>>>>> BWD[%d] FAILED: err=%.10f%% (numeric=%.20f, symbolic=%.20f)\n", (int)argIndexUnderTest, 100.0 * relErr, perturbationBasedDelta, gradientBasedDelta);
                    numFailed++;
                }
                else if (relErr > 1e-5)
                    fprintf(stderr, "           BWD[%d] SOMEWHAT OFF: err=%.10f%% (numeric=%.20f, symbolic=%.20f)\n", (int)argIndexUnderTest, 100.0 * relErr, perturbationBasedDelta, gradientBasedDelta);
                // Once in a while enable the following, to see whether there are some broken tests. We do have zeroes e.g. for Equal().
                //else if (gradientBasedDelta == 0)
                //    fprintf(stderr, ">>>>>>>>>> BWD[%d] IS 0??\n", (int)argIndexUnderTest);
            }
        }
    } // loop over tests
    if (!doGradientCheck)
        fprintf(stderr, "Skipped gradient checks for Float precision.\n");
    return numFailed;
}
#endif

void RunDynamiteTests()
{
#if 1 // (interferes with logging for profiling and reprodible Parameter initialization)
    size_t numFailed = 0;
    size_t N = 7; // (make it odd, otherwise some stuff will cancel out in BatchNorm, causing huge rel error since it does not cancel out 100% numerically)
    numFailed += DynamiteTest(N, DataType::Double, /*testStacking=*/false, DeviceDescriptor::GPUDevice(0));
    numFailed += DynamiteTest(N, DataType::Double, /*testStacking=*/true,  DeviceDescriptor::GPUDevice(0));
#if 0 // only do a batched one on the GPU by default
    numFailed += DynamiteTest(1, DataType::Double, /*testStacking=*/false, DeviceDescriptor::GPUDevice(0));
    numFailed += DynamiteTest(N, DataType::Double, /*testStacking=*/false, DeviceDescriptor::CPUDevice());
    numFailed += DynamiteTest(N, DataType::Float,  /*testStacking=*/false, DeviceDescriptor::GPUDevice(0));
    numFailed += DynamiteTest(1, DataType::Float,  /*testStacking=*/false, DeviceDescriptor::GPUDevice(0));
    numFailed += DynamiteTest(1, DataType::Double, /*testStacking=*/false, DeviceDescriptor::CPUDevice());
    numFailed += DynamiteTest(1, DataType::Float,  /*testStacking=*/false, DeviceDescriptor::CPUDevice());
#endif
    if (numFailed > 0)
        LogicError("RunDynamiteTests: %d tests failed.", (int)numFailed);
#endif
}
