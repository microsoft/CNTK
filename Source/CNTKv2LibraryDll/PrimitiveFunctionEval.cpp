//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// The actual direct forward and backward computation of V2 Functions is containedin here.
// TODO: move most of Dynamite GetValue.cpp here

#include "stdafx.h"
#include "PrimitiveFunction.h"
#include "Utils.h"

#include <vector>
#include <string>

using namespace Microsoft::MSR::CNTK;
using namespace std;

namespace CNTK
{
    /*static*/ NDArrayViewPtr PrimitiveFunction::ComputeKnowableValue(PrimitiveOpType primitiveOp,  // execute this op
                     const vector<NDArrayViewPtr>& args, const Dictionary& attributes, // on these inputs --TODO: move attributes up
                     const NDShape& outputShape, NDArrayViewPtr&& out, // into this output (if null then create a new one)
                     const PrimitiveFunction& funcForErrMsg)
    {
        // first handle ops that do not create new data
        if (primitiveOp == PrimitiveOpType::StopGradient ||
            primitiveOp == PrimitiveOpType::Pass         ||
            primitiveOp == PrimitiveOpType::NoOp         ||
            primitiveOp == PrimitiveOpType::Reshape      ||
            primitiveOp == PrimitiveOpType::Slice)
        {
            if (out)
                LogicError("Variable '%S' Value(): An output buffer was passed for op %S that does not need one.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
            out = args[0];
            switch (primitiveOp)
            {
            // ops that do not copy data
            case PrimitiveOpType::StopGradient:
            case PrimitiveOpType::Pass:
            case PrimitiveOpType::NoOp:
                break;
            case PrimitiveOpType::Reshape:
                 if (out->Shape() != outputShape)
                     out = out->AsShape(outputShape);
                break;
            case PrimitiveOpType::Slice:
                {
                    auto axis       = attributes[PrimitiveFunction::AttributeNameAxis].Value<Axis>();
                    auto beginIndex = attributes[PrimitiveFunction::AttributeNameBeginIndex].Value<int>();
                    auto endIndex   = attributes[PrimitiveFunction::AttributeNameEndIndex].Value<int>();
                    NormalizeStaticAxis(axis, args[0]->Shape());
                    auto extent = out->Shape().Dimensions();
                    auto startOffset = vector<size_t>(extent.size(), 0);
                    auto axisIndex = axis.StaticAxisIndex();
                    if (startOffset[axisIndex] != beginIndex || extent[axisIndex] != endIndex - beginIndex)
                    {
                        startOffset[axisIndex] = beginIndex;
                        extent[axisIndex] = endIndex - beginIndex;
                        out = out->SliceView(startOffset, extent, true); // slice it
                    }
                }
                break;
            }
            return out;
        }

        // ops that generate output: allocate memory for the result unless memory was passed in
        if (!out)
            out = make_shared<NDArrayView>(args.front()->GetDataType(), outputShape, args.front()->Device());
        else if (out->Shape() != outputShape)
            LogicError("Variable '%S' Value(): The out buffer passed to op %S does not match outputShape.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
        // perform the operation
        auto op = Microsoft::MSR::CNTK::ElementWiseOperator::opNone;
        auto reductionOp = Microsoft::MSR::CNTK::ElementWiseOperator::opSum;
        switch (primitiveOp)
        {
            // elementwise ops are done outside, we just set the opcode
        case PrimitiveOpType::Plus:         op = Microsoft::MSR::CNTK::ElementWiseOperator::opSum;                break;
        case PrimitiveOpType::Minus:        op = Microsoft::MSR::CNTK::ElementWiseOperator::opDifference;         break;
        case PrimitiveOpType::ElementTimes: op = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProduct; break;
        case PrimitiveOpType::ReLU:         op = Microsoft::MSR::CNTK::ElementWiseOperator::opLinearRectifier;    break;
        case PrimitiveOpType::Tanh:         op = Microsoft::MSR::CNTK::ElementWiseOperator::opTanh;               break;
        case PrimitiveOpType::Sigmoid:      op = Microsoft::MSR::CNTK::ElementWiseOperator::opSigmoid;            break;
        case PrimitiveOpType::Log:          op = Microsoft::MSR::CNTK::ElementWiseOperator::opLog;                break;
        case PrimitiveOpType::Exp:          op = Microsoft::MSR::CNTK::ElementWiseOperator::opExp;                break;
        case PrimitiveOpType::Cos:          op = Microsoft::MSR::CNTK::ElementWiseOperator::opCosine;             break;
        case PrimitiveOpType::Sin:          op = Microsoft::MSR::CNTK::ElementWiseOperator::opSin;                break;
        case PrimitiveOpType::Negate:       op = Microsoft::MSR::CNTK::ElementWiseOperator::opNegate;             break;
            // reduction ops are also done outside, but set the reductionOp
        case PrimitiveOpType::ReduceElements:
            {
                op = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy;
                const auto& reductionOpName = attributes[PrimitiveFunction::AttributeNameReductionOpName].Value<wstring>();
                if (reductionOpName == PrimitiveFunction::InternalSumReductionOpName)
                    reductionOp = Microsoft::MSR::CNTK::ElementWiseOperator::opSum;
                else if (reductionOpName == PrimitiveFunction::InternalLogSumReductionOpName)
                    reductionOp = Microsoft::MSR::CNTK::ElementWiseOperator::opLogSum;
                else
                    //  PrimitiveFunction::InternalMeanReductionOpName
                    //  PrimitiveFunction::InternalMaxReductionOpName
                    //  PrimitiveFunction::InternalMinReductionOpName
                    //  PrimitiveFunction::InternalProdReductionOpName
                    LogicError("Variable '%S' Value(): Reduction op %S not yet implemented.", funcForErrMsg.AsString().c_str(), reductionOpName.c_str());
            }
            break;
            // non-elementwise ops are done here
        case PrimitiveOpType::Times:
        case PrimitiveOpType::TransposeTimes:
            out->MatrixProduct(false, args[0], primitiveOp == PrimitiveOpType::TransposeTimes, args[1], false, 1.0, attributes[PrimitiveFunction::AttributeNameOutputRank].Value<size_t>(), out);
            break;
        case PrimitiveOpType::Splice:
            {
                auto axis = attributes[PrimitiveFunction::AttributeNameAxis].Value<Axis>();
                size_t maxInputRank = args[0]->Shape().Rank();
                for (int i = 1; i < args.size(); i++)
                {
                    auto inputRank = args[i]->Shape().Rank();
                    if (maxInputRank < inputRank)
                        maxInputRank = inputRank;
                }
                NormalizeStaticAxis(axis, NDShape(maxInputRank));
                if (args.size() > 1)
                    NDArrayView::GatherBatch(args, axis.StaticAxisIndex(), out);
                else // only one: do nothing or at best reshape if a new axis is added
                {
                    // BUGBUG: This is a 'free' op, should be caught earlier.
                    out = args[0];
                    if (out->Shape() != outputShape)
                        out = out->AsShape(outputShape);
                }
            }
            break;
            // the following N-nary operations should be easy, mostly a matter of writing tests
            // unary operations to be completed
        case PrimitiveOpType::Sqrt:
        case PrimitiveOpType::Floor:
        case PrimitiveOpType::Abs:
        case PrimitiveOpType::Reciprocal:
        case PrimitiveOpType::ELU:
        case PrimitiveOpType::Pow:
        case PrimitiveOpType::Softmax:
        case PrimitiveOpType::Hardmax:
        case PrimitiveOpType::TransposeAxes:
        case PrimitiveOpType::LogSoftmax:
        case PrimitiveOpType::SumAll:
            LogicError("Variable '%S' Value(): Memoziation of unary operator %S not implemented yet.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
            // binary operations to be completed
        case PrimitiveOpType::Equal:
        case PrimitiveOpType::NotEqual:
        case PrimitiveOpType::Less:
        case PrimitiveOpType::LessEqual:
        case PrimitiveOpType::Greater:
        case PrimitiveOpType::GreaterEqual:
        case PrimitiveOpType::LogPlus:
        case PrimitiveOpType::Logistic:
        case PrimitiveOpType::CrossEntropyWithSoftmax:
        case PrimitiveOpType::ClassificationError:
        case PrimitiveOpType::SquaredError:
        case PrimitiveOpType::Gather:
                LogicError("Variable '%S' Value(): Memoziation of binary operator %S not implemented yet.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
            // ternary operations to be completed
        case PrimitiveOpType::Clip:
        case PrimitiveOpType::Select:
            LogicError("Variable '%S' Value(): Memoziation of ternary operator %S not implemented yet.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
            // dynamic-axis related operations are not supported as dynamic axes require Inputs and are therefore not applicable here
        case PrimitiveOpType::PackedIndex:
        case PrimitiveOpType::GatherPacked:
        case PrimitiveOpType::ScatterPacked:
        case PrimitiveOpType::PastValue:
        case PrimitiveOpType::FutureValue:
        case PrimitiveOpType::Where:
        case PrimitiveOpType::OptimizedRNNStack:
        case PrimitiveOpType::ReconcileDynamicAxis:
        case PrimitiveOpType::ToSequence:
        case PrimitiveOpType::ToSequenceLike:
        case PrimitiveOpType::UnpackSequence:
            RuntimeError("Variable '%S' Value(): Memoziation of dynamic-axis related operation %S is not possible as they imply unknown inputs.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
            // some operations are not supported because they do not apply
        case PrimitiveOpType::Combine:  // TODO: should be trivial to support, just need a test
        case PrimitiveOpType::Block:    // TODO: recursively invoke, needs a test and investigation whether blocks are always singleton copies
        case PrimitiveOpType::Assign:
            RuntimeError("Variable '%S' Value(): Memoziation of operation %S not applicable.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
            // the following operations are not TensorView, and may be implementable through relatively simple calls to Matrix
        case PrimitiveOpType::BatchNormalization:
        case PrimitiveOpType::CosDistance:
        case PrimitiveOpType::OneHot:
            LogicError("Variable '%S' Value(): Memoziation of operation %S not implemented yet.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
            // the following operations are not TensorView, and hence should be routed through V1 ComputationNodes
            // convolution family
        case PrimitiveOpType::Convolution:  // TODO: route these through TensorView
        case PrimitiveOpType::Pooling:
        case PrimitiveOpType::Unpooling:
        case PrimitiveOpType::ROIPooling:
            // random family
        case PrimitiveOpType::Dropout:
        case PrimitiveOpType::RandomSample:
        case PrimitiveOpType::RandomSampleInclusionFrequency:
            // special ops family
        case PrimitiveOpType::LambdaRank:
        case PrimitiveOpType::NDCG:
        case PrimitiveOpType::EditDistanceError:
        case PrimitiveOpType::LabelsToGraph:
        case PrimitiveOpType::ForwardBackward:
        case PrimitiveOpType::CosDistanceWithNegativeSamples:
            LogicError("Variable '%S' Value(): Memoziation of operation %S not implemented yet.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
        default:
            LogicError("Variable '%S' Value(): Memoziation of non-existent operation %S?", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
        }
        // most common case: elementwise ops are done here instead
        if (op != Microsoft::MSR::CNTK::ElementWiseOperator::opNone)
            NDArrayView::NumericOperation(args, 1.0, op, out, 0.0, reductionOp);
        return out;
    }

    // perform back into all inputs
    // Currently only supported for Splice.
    // TODO: Remove this, it's simple enough.
    /*static*/ void PrimitiveFunction::BackpropToAll(const NDArrayViewPtr& outputGradient,                  // incoming gradient from top...
                              PrimitiveOpType primitiveOp, const Dictionary& attributes,                    // ...goes through this backprop function...
                              const NDArrayViewPtr& outputValue, const vector<NDArrayViewPtr>& inputValues, // ...using these values from forward pass...
                              vector<NDArrayViewPtr>& inputGradients, double beta,                          // ...into here
                              const PrimitiveFunction& funcForErrMsg)
    {
        if (primitiveOp == PrimitiveOpType::Splice)
        {
            outputValue;  inputValues; // not used for Splice
            if (inputGradients.size() > 1)
                NDArrayView::ScatterBatch(outputGradient, inputGradients);
            else // only one: propagate by copying
                NDArrayView::NumericOperation({ outputGradient }, 1.0, opCopy, inputGradients.front(), 0.0, Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
        }
        else
            LogicError("Variable '%S' Value(): Bulk backpropagation for operation %S not implemented yet.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
    }

    // perform back propagation into an input
    // Gradient must have been allocated to the correct shape already.
    // If beta == 0 then gradient can be uninitialized memory.
    // For now only defined for functions with 1 output.
    // TODO: decide whether we pass raw pointers or shared pointers...
    /*static*/ void PrimitiveFunction::BackpropTo(const NDArrayView* outputGradient,                         // incoming gradient from top...
                              size_t i, PrimitiveOpType primitiveOp, const Dictionary& attributes,           // ...goes through this backprop function...
                              const NDArrayView* outputValue, const vector<const NDArrayView*>& inputValues, // ...using these values from forward pass...
                              const NDArrayViewPtr& gradient, double beta,                                   // ...into here. (Despite 'const', *gradient is the output.)
                              const PrimitiveFunction& funcForErrMsg)
    {
    #if 0   // TODO: bring this back once we have gradient functions that do not support beta
        if (beta == 0) // TODO: limit this to those ops that do not support beta
        {
            gradient->SetValue(0.0f);
            beta = 1;
        }
    #endif
        auto op1Arg  = Microsoft::MSR::CNTK::ElementWiseOperator::opNone; // this gets set for 1-argument TensorView ops for execution after the switch()
        auto op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opNone; // and this for 2-arg ops; all others execute inside the switch()
        const NDArrayView* arg1 = outputGradient;
        const NDArrayView* arg2 = nullptr;
        double alpha = 1;
        // NOTE: For now, this only implements the operators needed for the prototype
        switch (primitiveOp)
        {
            // binary operations with simple TensorView implementation
        case PrimitiveOpType::Plus:           op1Arg  = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy; break;
        case PrimitiveOpType::Minus:          op1Arg  = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy; alpha = i == 0 ? 1 : -1; break;
        case PrimitiveOpType::ElementTimes:   op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProduct; arg2 = inputValues[1 - i]; break;
            // Times family
        case PrimitiveOpType::Times:
        case PrimitiveOpType::TransposeTimes:
            arg2 = inputValues[1 - i];
            if (i == 0) // left input
                NDArrayView::MatrixProduct(/*transC=*/primitiveOp == PrimitiveOpType::TransposeTimes,
                                          /*A=*/const_cast<NDArrayView*>(arg1)->shared_from_this(), /*transA=*/false,
                                          /*B=*/const_cast<NDArrayView*>(arg2)->shared_from_this(), /*transB=*/true,  alpha, /*outputRank dummy=*/0, /*C=*/gradient, beta);
            else // right input
                NDArrayView::MatrixProduct(/*transC=*/false,
                                          /*A=*/const_cast<NDArrayView*>(arg2)->shared_from_this(), /*transA=*/primitiveOp != PrimitiveOpType::TransposeTimes,
                                          /*B=*/const_cast<NDArrayView*>(arg1)->shared_from_this(), /*transB=*/false, alpha, /*outputRank dummy=*/0, /*C=*/gradient, beta);
            break;
            // unary operations with simple TensorView implementation
        case PrimitiveOpType::ReLU:           op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithLinearRectifierDerivativeFromOutput; arg2 = outputValue; break;
            // no-op operations with simple TensorView implementation
            // NOTE: These do not need any data copy if there is only one consumer, which we won't know here. That case will be caught in the batched version.
        case PrimitiveOpType::NoOp:           op1Arg  = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy; break;
        case PrimitiveOpType::Reshape:        op1Arg  = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy; break;
            // gradients that are copies with broadcasting
        case PrimitiveOpType::ReduceElements:
            {
                const auto& reductionOpName = attributes[L"reductionOpName"/*PrimitiveFunction::AttributeNameReductionOpName*/].Value<wstring>();
                if (reductionOpName == L"Sum"/*PrimitiveFunction::InternalSumReductionOpName*/) // TODO: uncomment these symbols once we have access
                    op1Arg = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy;
                else if (reductionOpName == L"LogSum"/*PrimitiveFunction::InternalLogSumReductionOpName*/)
                    NDArrayView::NumericOperation({ const_cast<NDArrayView*>(outputGradient )->shared_from_this(),
                                                    const_cast<NDArrayView*>( inputValues[0])->shared_from_this(),
                                                    const_cast<NDArrayView*>(outputValue    )->shared_from_this() }, alpha,
                                                  Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithExpOfDiff,
                                                  gradient, beta,
                                                  Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
                else
                    //  PrimitiveFunction::InternalMeanReductionOpName
                    //  PrimitiveFunction::InternalMaxReductionOpName
                    //  PrimitiveFunction::InternalMinReductionOpName
                    //  PrimitiveFunction::InternalProdReductionOpName
                    LogicError("Variable '%S' Value(): Gradient of reduction op %S not yet implemented.", funcForErrMsg.AsString().c_str(), reductionOpName.c_str());
            }
            break;
            // hard stuff
        case PrimitiveOpType::Splice:
            {
                // TODO: allow to pass index as SIZE_MAX to denote all; but only allow that for splice.
                auto axis = attributes[L"axis"/*PrimitiveFunction::AttributeNameAxis*/].Value<Axis>();
                if (axis.StaticAxisIndex() != arg1->Shape().Rank() -1)
                    LogicError("NDArrayView::GatherBatch: Currently only splicing in a new slowest-changing axis is supported.");
                NDArrayView::NumericOperation({ arg1->IndexLastAxis(i) }, alpha,
                                              Microsoft::MSR::CNTK::ElementWiseOperator::opCopy, gradient, beta,
                                              Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
            }
            break;
        case PrimitiveOpType::Slice:
            {
                auto axis       = attributes[L"axis"      /*PrimitiveFunction::AttributeNameAxis*/      ].Value<Axis>();
                auto beginIndex = attributes[L"beginIndex"/*PrimitiveFunction::AttributeNameBeginIndex*/].Value<int>();
                auto endIndex   = attributes[L"endIndex"  /*PrimitiveFunction::AttributeNameEndIndex*/  ].Value<int>();
                auto extent = gradient->Shape().Dimensions();
                auto startOffset = vector<size_t>(extent.size(), 0);
                auto axisIndex = axis.StaticAxisIndex();
                if (startOffset[axisIndex] != beginIndex || extent[axisIndex] != endIndex - beginIndex)
                {
                    // backprop into a slice of 'gradient'
                    if (beta == 0) // if beta = 0 then we must explicitly initialize the entire gradient matrix, not just the slice
                        gradient->SetValue(0.0f);
                    startOffset[axisIndex] = beginIndex;
                    extent[axisIndex] = endIndex - beginIndex;
                    NDArrayView::NumericOperation({ const_cast<NDArrayView*>(arg1)->shared_from_this() }, alpha,
                                                  Microsoft::MSR::CNTK::ElementWiseOperator::opCopy, gradient->SliceView(startOffset, extent), beta,
                                                  Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
                }
                else
                    op1Arg = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy; // full slice actually: just copy (like a NoOp)
            }
            break;
        default:
            //fprintf(stderr, "NEEDS: %S\n", PrimitiveOpTypeName(primitiveOp).c_str());
            LogicError("Variable '%S' Value(): Backpropagation for operation %S not implemented yet.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
            //LogicError("Variable '%S' Value(): Backpropagation for non-existent operation %S?", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
        }
        // the simple TensorView operations are performed out here
        // TODO: we can eliminate the vector<> by passing a std::function, possibly?
        if (op1Arg != Microsoft::MSR::CNTK::ElementWiseOperator::opNone)
            NDArrayView::NumericOperation({ const_cast<NDArrayView*>(arg1)->shared_from_this() }, alpha,
                                          op1Arg, gradient, beta,
                                          Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
        else if (op2Args != Microsoft::MSR::CNTK::ElementWiseOperator::opNone)
            NDArrayView::NumericOperation({ const_cast<NDArrayView*>(arg1)->shared_from_this(), const_cast<NDArrayView*>(arg2)->shared_from_this() }, alpha,
                                          op2Args, gradient, beta,
                                          Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
    }

}
