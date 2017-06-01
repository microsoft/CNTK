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
    // BUGBUG: AsString() is called on 'this' (affects only error messages; once gone, change to static; also no need to be virtual)
    /*virtual*/ NDArrayViewPtr PrimitiveFunction::ComputeKnowableValue(PrimitiveOpType primitiveOp, 
        const vector<NDArrayViewPtr>& args, const Dictionary& attributes, const NDShape& outputShape, NDArrayViewPtr&& out) const
    {
        // first handle ops that do not create new data
        if (primitiveOp == PrimitiveOpType::StopGradient ||
            primitiveOp == PrimitiveOpType::Pass         ||
            primitiveOp == PrimitiveOpType::NoOp         ||
            primitiveOp == PrimitiveOpType::Reshape      ||
            primitiveOp == PrimitiveOpType::Slice)
        {
            if (out)
                LogicError("Variable '%S' Value(): An output buffer was passed for op %S that does not need one.", AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
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
            LogicError("Variable '%S' Value(): The out buffer passed to op %S does not match outputShape.", AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
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
                    LogicError("Variable '%S' Value(): Reduction op %S not yet implemented.", AsString().c_str(), reductionOpName.c_str());
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
            LogicError("Variable '%S' Value(): Memoziation of unary operator %S not implemented yet.", AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
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
                LogicError("Variable '%S' Value(): Memoziation of binary operator %S not implemented yet.", AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
            // ternary operations to be completed
        case PrimitiveOpType::Clip:
        case PrimitiveOpType::Select:
            LogicError("Variable '%S' Value(): Memoziation of ternary operator %S not implemented yet.", AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
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
            RuntimeError("Variable '%S' Value(): Memoziation of dynamic-axis related operation %S is not possible as they imply unknown inputs.", AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
            // some operations are not supported because they do not apply
        case PrimitiveOpType::Combine:  // TODO: should be trivial to support, just need a test
        case PrimitiveOpType::Block:    // TODO: recursively invoke, needs a test and investigation whether blocks are always singleton copies
        case PrimitiveOpType::Assign:
            RuntimeError("Variable '%S' Value(): Memoziation of operation %S not applicable.", AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
            // the following operations are not TensorView, and may be implementable through relatively simple calls to Matrix
        case PrimitiveOpType::BatchNormalization:
        case PrimitiveOpType::CosDistance:
        case PrimitiveOpType::OneHot:
            LogicError("Variable '%S' Value(): Memoziation of operation %S not implemented yet.", AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
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
            LogicError("Variable '%S' Value(): Memoziation of operation %S not implemented yet.", AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
        default:
            LogicError("Variable '%S' Value(): Memoziation of non-existent operation %S?", AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
        }
        // most common case: elementwise ops are done here instead
        if (op != Microsoft::MSR::CNTK::ElementWiseOperator::opNone)
            out->NumericOperation(args, 1.0, op, out, 0.0, reductionOp);
        return out;
    }
}
