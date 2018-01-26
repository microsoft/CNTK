//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// The actual direct forward and backward computation of V2 Functions is containedin here.
// TODO: remove the namespace before the elementwise operations; will shorten the code tremendously

#include "stdafx.h"
#include "PrimitiveFunction.h"
#include "Utils.h"
#include "TrainingNodes.h" // for RandomDistributionTypeXXX. TODO: move those out of the V1 header

#include <vector>
#include <string>

using namespace Microsoft::MSR::CNTK;
using namespace std;

namespace CNTK
{
    // helper to get a slice view, used for both forward and backward of Slice()
    // It returns a TensorView slice (which may have gaps due to strides, e.g. an auto-batched Slice()).
    static NDArrayViewPtr GetSliceView(const NDArrayViewPtr& arg, const Dictionary& attributes, const NDShape& outputShape, bool readOnly, const PrimitiveFunction& funcForErrMsg)
    {
        if (attributes.Size() == 1) // Index() --has no axis or endIndex parameter. and must drop the final axis
        {
            let index = attributes[PrimitiveFunction::AttributeNameBeginIndex].Value<int>();
            let& extent = outputShape.Dimensions(); // note: last dimension is missing; this will strip it in the output
            if (extent.size() + 1 != arg->Shape().Rank())
                LogicError("Variable '%S' Value(): The input and output rank for op Slice when indexing must differ by 1.", funcForErrMsg.AsString().c_str());
            auto startOffset = NDShapeDimensions(extent.size() + 1, 0);
            startOffset.back() = (size_t) index;
            return arg->Slice(startOffset, extent, NDShapeDimensions(), NDArrayView::SliceMode::View, readOnly); // slice it
        }
        else
        {
            if (attributes.Contains(PrimitiveFunction::AttributeNameAxisVec)) // vector of slices
            {
                let& axes         = AsVector<Axis>(attributes[PrimitiveFunction::AttributeNameAxisVec      ].Value<vector<DictionaryValue>>());
                let& beginIndices = AsVector<int> (attributes[PrimitiveFunction::AttributeNameBeginIndexVec].Value<vector<DictionaryValue>>());
                let& endIndices   = AsVector<int> (attributes[PrimitiveFunction::AttributeNameEndIndexVec  ].Value<vector<DictionaryValue>>());
                if (attributes.Contains(PrimitiveFunction::AttributeNameSliceStridesVec))
                    LogicError("Variable '%S' Value(): Strided slicing not yet implemented.", funcForErrMsg.AsString().c_str());
                auto extent = arg->Shape().Dimensions();
                auto startOffset = NDShapeDimensions(extent.size(), 0);
                for (size_t i = 0; i < axes.size(); i++)
                {
                    let axisIndex  = axes[i].StaticAxisIndex();
                    let beginIndex = beginIndices[i];
                    let endIndex   = endIndices[i];
                    startOffset[axisIndex] = beginIndex;
                    extent[axisIndex] = endIndex - beginIndex;
                }
                // TODO: use IndexLastAxis()
                if (extent != arg->Shape().Dimensions() || any_of(startOffset.begin(), startOffset.end(), [](NDShapeDimension v) { return v != 0; }))
                    return arg->Slice(startOffset, extent, NDShapeDimensions(), NDArrayView::SliceMode::View, readOnly); // slice it
            }
            else // single slice
            {
                let axis       = attributes[PrimitiveFunction::AttributeNameAxis].Value<Axis>();
                let beginIndex = attributes[PrimitiveFunction::AttributeNameBeginIndex].Value<int>();
                let endIndex   = attributes[PrimitiveFunction::AttributeNameEndIndex].Value<int>();
                if (attributes.Contains(PrimitiveFunction::AttributeNameSliceStrides))
                    LogicError("Variable '%S' Value(): Strided slicing not yet implemented.", funcForErrMsg.AsString().c_str());
                let axisIndex = axis.StaticAxisIndex();
                // TODO: implement a special version without std::vectors
                auto extent = arg->Shape().Dimensions();
                auto startOffset = NDShapeDimensions(extent.size(), 0);
                startOffset[axisIndex] = beginIndex;
                extent[axisIndex] = endIndex - beginIndex;
                if (extent != arg->Shape().Dimensions() || any_of(startOffset.begin(), startOffset.end(), [](NDShapeDimension v) { return v != 0; }))
                    return arg->Slice(startOffset, extent, NDShapeDimensions(), NDArrayView::SliceMode::View, readOnly); // slice it
            }
        }
        return arg;
    }

    // helper to parse Transpose axis arguments
    static NDShapePermutation TransposePermutationArg(const Dictionary& attributes)
    {
        if (attributes.Contains(PrimitiveFunction::AttributeNameAxisVec)) // two axes
            return NDShapePermutation(Transform(attributes[PrimitiveFunction::AttributeNameAxisVec].Value<std::vector<DictionaryValue>>(),
                                                [&](const DictionaryValue& arg) { return (NDShapePermutation::value_type)arg.Value<Axis>().StaticAxisIndex(); }));
        else
            return NDShapePermutation({ attributes[PrimitiveFunction::AttributeNameAxis2].Value<Axis>().StaticAxisIndex(),
                                        attributes[PrimitiveFunction::AttributeNameAxis1].Value<Axis>().StaticAxisIndex() });
    }

    // Performs a forward operation.
    // It is assumed that the inputs are immutable, and hence if the result can be expressed as a view (e.g. Reshape()),
    // a view into the an input is returned instead of a newly allocated buffer.
    // For Slice(), a view is returned even if the slice is not memory-contiguous; caller must fix this if it is a problem.
    // result remains compatible with potential subsequent Matrix-library operations.
    // Note: To support auto-batching, this function must only consider attributes when presence of an additional
    // batch axis makes no difference. That is, for example, not the case for ReduceElements over AllStaticAxes().
    // This is addressed as:
    //  - All relative axis arguments must have been normalized already in InferOutputs. Can't call NormalizeStaticAxis() here.
    //  - All reduction axes are already completely specified by the output shape. Can't look at the axis parameter.
    /*static*/ NDArrayViewPtr PrimitiveFunction::Forward(PrimitiveOpType primitiveOp, // execute this op
                     const Dictionary& attributes, bool isVolatile,                                // with these additional parameters
                     const vector<NDArrayViewPtr>& args,                                           // on these inputs
                     const NDShape& outputShape, NDArrayViewPtr&& out,                             // into this output (if null then create a new one)
                     const PrimitiveFunction& funcForErrMsg)
    {
        // Slice() can either create new data or not, so do it first
        let sliceView = primitiveOp == PrimitiveOpType::Slice ? GetSliceView(args[0], attributes, outputShape, /*readOnly=*/true, funcForErrMsg) : nullptr;

        // first handle ops that do not create new data
        // TODO: Transpose could be included here as well, if we pre-process inputs of ops that do not accept non-dense tensors.
        if (primitiveOp == PrimitiveOpType::StopGradient ||
            primitiveOp == PrimitiveOpType::Pass         ||
            primitiveOp == PrimitiveOpType::NoOp         ||
            primitiveOp == PrimitiveOpType::Reshape      ||
            (primitiveOp == PrimitiveOpType::Slice && !out /*&& sliceView->IsContiguous()*/)) // (should copy if output buffer provided or data not contiguous; we do that outside, as we cannot control mem alloc here)
        {
            if (out)
                LogicError("Variable '%S' Value(): An output buffer was passed for op %S that does not need one.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
            let arg = args[0];
            switch (primitiveOp)
            {
            case PrimitiveOpType::Reshape:
                if (arg->Shape() != outputShape)
                     return arg->AsShape(outputShape);
                break;
            case PrimitiveOpType::Slice:
                return sliceView;
            }
            // operation is a no-op: return original argument as is
            //arg->LogToFile(PrimitiveOpTypeName(primitiveOp), stderr);
            return arg;
        }

        // ops that generate output: allocate memory for the result unless memory was passed in
        if (!out)
            out = make_shared<NDArrayView>(args.front()->GetDataType(), outputShape, args.front()->Device());
        else if (out->Shape() != outputShape)
            LogicError("Variable '%S' Value(): The out buffer passed to op %S does not match outputShape.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
        // perform the operation
        auto op          = Microsoft::MSR::CNTK::ElementWiseOperator::opNone;
        auto reductionOp = Microsoft::MSR::CNTK::ElementWiseOperator::opSum;
        double alpha = 1; // changed in ReduceMean() to 1/#elements averaged over
        switch (primitiveOp)
        {
            // binary elementwise ops are done outside, we just set the opcode
        case PrimitiveOpType::Plus:          op = Microsoft::MSR::CNTK::ElementWiseOperator::opSum;                   break;
        case PrimitiveOpType::Minus:         op = Microsoft::MSR::CNTK::ElementWiseOperator::opDifference;            break;
        case PrimitiveOpType::ElementTimes:  op = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProduct;    break;
        case PrimitiveOpType::LogPlus:       op = Microsoft::MSR::CNTK::ElementWiseOperator::opLogSum;                break; // this is LogAddExp()
        case PrimitiveOpType::Pow:           op = Microsoft::MSR::CNTK::ElementWiseOperator::opPow;                   break;
        case PrimitiveOpType::Equal:         op = Microsoft::MSR::CNTK::ElementWiseOperator::opEqual;                 break;
        case PrimitiveOpType::NotEqual:      op = Microsoft::MSR::CNTK::ElementWiseOperator::opNotEqual;              break;
        case PrimitiveOpType::Less:          op = Microsoft::MSR::CNTK::ElementWiseOperator::opLess;                  break;
        case PrimitiveOpType::LessEqual:     op = Microsoft::MSR::CNTK::ElementWiseOperator::opLessEqual;             break;
        case PrimitiveOpType::Greater:       op = Microsoft::MSR::CNTK::ElementWiseOperator::opGreater;               break;
        case PrimitiveOpType::GreaterEqual:  op = Microsoft::MSR::CNTK::ElementWiseOperator::opGreaterEqual;          break;
            // unary elementwise ops as well are done outside, we just set the opcode
        case PrimitiveOpType::ReLU:          op = Microsoft::MSR::CNTK::ElementWiseOperator::opLinearRectifier;       break;
        case PrimitiveOpType::Tanh:          op = Microsoft::MSR::CNTK::ElementWiseOperator::opTanh;                  break;
        case PrimitiveOpType::Sigmoid:       op = Microsoft::MSR::CNTK::ElementWiseOperator::opSigmoid;               break;
        case PrimitiveOpType::Log:           op = Microsoft::MSR::CNTK::ElementWiseOperator::opLog;                   break;
        case PrimitiveOpType::Exp:           op = Microsoft::MSR::CNTK::ElementWiseOperator::opExp;                   break;
        case PrimitiveOpType::Cos:           op = Microsoft::MSR::CNTK::ElementWiseOperator::opCosine;                break;
        case PrimitiveOpType::Sin:           op = Microsoft::MSR::CNTK::ElementWiseOperator::opSin;                   break;
        case PrimitiveOpType::Negate:        op = Microsoft::MSR::CNTK::ElementWiseOperator::opNegate;                break;
        case PrimitiveOpType::Floor:         op = Microsoft::MSR::CNTK::ElementWiseOperator::opFloor;                 break;
        case PrimitiveOpType::Abs:           op = Microsoft::MSR::CNTK::ElementWiseOperator::opAbs;                   break;
        case PrimitiveOpType::Sqrt:          op = Microsoft::MSR::CNTK::ElementWiseOperator::opSqrt;                  break;
        case PrimitiveOpType::Reciprocal:    op = Microsoft::MSR::CNTK::ElementWiseOperator::opReciprocal;            break;
        case PrimitiveOpType::ELU:           op = Microsoft::MSR::CNTK::ElementWiseOperator::opExponentialLinearUnit; break;
        case PrimitiveOpType::StableSigmoid: op = Microsoft::MSR::CNTK::ElementWiseOperator::opStableSigmoid;         break;
        case PrimitiveOpType::ScaleAndShift:
        {
            let scale = attributes[PrimitiveFunction::AttributeNameScale].Value<double>();
            let shift = attributes[PrimitiveFunction::AttributeNameShift].Value<double>();
            // PERF BUGBUG: We don't have a single op to just add a constant. Shouldn't be hard, since the shapes are trivial.
            NDArrayView::NumericOperation(args, scale, Microsoft::MSR::CNTK::ElementWiseOperator::opCopy,     out);
            if (shift != 0)
                NDArrayView::NumericOperation({ },  shift, Microsoft::MSR::CNTK::ElementWiseOperator::opConstOne, out, 1.0);
        }
        break;
            // ternary operations
        case PrimitiveOpType::Select:        op = Microsoft::MSR::CNTK::ElementWiseOperator::opCond;   break;
        case PrimitiveOpType::ElementAffine: op = Microsoft::MSR::CNTK::ElementWiseOperator::opAxBplusC; break;
        case PrimitiveOpType::Clip:
            // special case since arg order of opClip is different
            NDArrayView::NumericOperation({ args[1], args[2], args[0] }, alpha, Microsoft::MSR::CNTK::ElementWiseOperator::opClip, out, 0.0, reductionOp);
            break;
            // nullary operations --these have no arguments, and generate data
        case PrimitiveOpType::RandomDistribution:
            {
                // TODOs:
                //  - how about gradient? Will it automatically know to not backprop into it?
                //  - maybe use this one to generate small constants?
                let& type = attributes[PrimitiveFunction::AttributeNameRandomDistributionType].Value<wstring>();
                if (type == Microsoft::MSR::CNTK::RandomDistributionTypeConstant)
                {
                    NOT_IMPLEMENTED; // TODO: use this for scalar constants
                }
                else
                {
                    // the shared_ptr<RNGState> was passed through as an opaque intptr_t, in a size_t
                    auto& rngState = *(RNGState*)attributes[PrimitiveFunction::AttributeNameRandomDistributionRNGHandle].Value<size_t>();
                    let& argsValueVec = attributes[PrimitiveFunction::AttributeNameRandomDistributionArgs].Value<vector<DictionaryValue>>(); // vector<double>
                    // TODO: add all the other types
                    //if (type == Microsoft::MSR:CNTK::RandomDistributionTypeUniform)
                    //{
                    //    result.SetUniformRandomValue(GetRNGHandle(), m_args[0], m_args[1]);
                    //    UpdateRngOffset(GetRngOffset() + result.GetNumElements());
                    //}
                    //else if (type == Microsoft::MSR:CNTK::RandomDistributionTypeNormal)
                    //{
                    //    result.SetGaussianRandomValue(GetRNGHandle(), m_args[0], m_args[1]);
                    //    UpdateRngOffset(GetRngOffset() + AsMultipleOf(result.GetNumElements(), 2));
                    //}
                    //else if (type == Microsoft::MSR:CNTK::RandomDistributionTypeGumbel)
                    //{
                    //    result.SetGumbelRandomValue(GetRNGHandle(), m_args[0], m_args[1]);
                    //    UpdateRngOffset(GetRngOffset() + result.GetNumElements());
                    //}
                    //else 
                    if (type == Microsoft::MSR::CNTK::RandomDistributionTypeBernoulli)
                        out->SetToRandomDistributionBernoulli(rngState, /*mean=*/argsValueVec[0].Value<double>(), /*scale=*/argsValueVec[1].Value<double>());
                    else
                        InvalidArgument("RandomDistribution: Unknown random distribution type '%S'", type.c_str());
                }
            }
            break;
            // Transpose
        case PrimitiveOpType::TransposeAxes:
            // TODO: I hope that one day this can be a see-through op (needs more solid support for non-dense objects)
            // we must copy, since non-consecutive tensors are not universally supported yet
            NDArrayView::NumericOperation({ args[0]->AsTransposed(TransposePermutationArg(attributes)) }, 1, Microsoft::MSR::CNTK::ElementWiseOperator::opCopy, out);
            break;
            // Slice if copy requested or needed
        case PrimitiveOpType::Slice:
            // The slice view has already completed, but we must copy the result over. The following op is the same as the shared one except for taking the slice view as its input.
            NDArrayView::NumericOperation({ sliceView }, alpha, Microsoft::MSR::CNTK::ElementWiseOperator::opCopy, out, 0.0, reductionOp);
            break;
            // reduction ops are also done outside, but set the reductionOp
        case PrimitiveOpType::ReduceElements:
            {
                op = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy; // note: reduction axes already fully specified via outputShape
                const auto& reductionOpName = attributes[PrimitiveFunction::AttributeNameReductionOpName].Value<wstring>();
                if (reductionOpName == PrimitiveFunction::InternalSumReductionOpName)
                    reductionOp = Microsoft::MSR::CNTK::ElementWiseOperator::opSum;
                else if (reductionOpName == PrimitiveFunction::InternalLogSumReductionOpName)
                    reductionOp = Microsoft::MSR::CNTK::ElementWiseOperator::opLogSum;
                else if (reductionOpName == PrimitiveFunction::InternalProdReductionOpName) // note: gradient still missing
                    reductionOp = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProduct;
                else if (reductionOpName == PrimitiveFunction::InternalMeanReductionOpName)
                {
                    reductionOp = Microsoft::MSR::CNTK::ElementWiseOperator::opSum;
                    alpha = (double)outputShape.TotalSize(/*check=*/false) / (double)args[0]->Shape().TotalSize(/*check=*/false);
                }
                else if (reductionOpName == PrimitiveFunction::InternalMaxReductionOpName) // note: gradient still missing
                    reductionOp = Microsoft::MSR::CNTK::ElementWiseOperator::opMax;
                else if (reductionOpName == PrimitiveFunction::InternalMinReductionOpName) // note: gradient still missing
                    reductionOp = Microsoft::MSR::CNTK::ElementWiseOperator::opMin;
                else if (reductionOpName == PrimitiveFunction::InternalArgmaxReductionOpName) // note: gradient still missing, whatever that is
                    reductionOp = Microsoft::MSR::CNTK::ElementWiseOperator::opArgmax;
                else if (reductionOpName == PrimitiveFunction::InternalArgminReductionOpName) // note: gradient still missing, whatever that is
                    reductionOp = Microsoft::MSR::CNTK::ElementWiseOperator::opArgmin;
                else
                    LogicError("Variable '%S' Value(): Unknown reduction op %S.", funcForErrMsg.AsString().c_str(), reductionOpName.c_str());
            }
            break;
            // non-elementwise ops are done here
        case PrimitiveOpType::Times:
        case PrimitiveOpType::TransposeTimes:
        case PrimitiveOpType::Affine:
        case PrimitiveOpType::TransposeAffine:
            NDArrayView::MatrixProduct(false,
                                       args[0], (primitiveOp == PrimitiveOpType::TransposeTimes || primitiveOp == PrimitiveOpType::TransposeAffine),
                                       args[1], false,
                                       1.0, attributes[PrimitiveFunction::AttributeNameOutputRank].Value<size_t>(), out);
            if (args.size() == 3) // affine
                NDArrayView::NumericOperation({ args[2] }, 1.0, Microsoft::MSR::CNTK::ElementWiseOperator::opCopy, out, 1.0);
            break;
        case PrimitiveOpType::Splice:
            // TODO: We may communicate from outside by not providing an Axis attribute that batching is along a new axis, and all shapes are the same.
            NDArrayView::GatherBatch(args, (size_t)attributes[PrimitiveFunction::AttributeNameAxis].Value<Axis>().StaticAxisIndex(), out);
            break;
        case PrimitiveOpType::BatchNormalization:
            // batch normalization is a little tricky
            {
                if (args.size() != 9)
                    LogicError("Variable '%S' Value(): Operation %S requires 3 additional arguments.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
                let& x      = args[0];
                let& scale  = args[1];
                let& bias   = args[2];
                auto& runSum    = args[3];
                auto& runSqrSum = args[4];
                auto& runCount  = args[5];
                // the following three are temps that carry over to backprop
                let& redBuf = args[6]; // mean buffer, also used for other ops later
                let& sigma  = args[7];
                let& xHat   = args[8]; // (x-mu)/sigma
                // things related to running stats
                let N = (double)x->Shape().TotalSize(/*check=*/false) / (double)bias->Shape().TotalSize(/*check=*/false);
                let isInferring = isVolatile; // this indicates inference mode
                let normalizationTimeConstant = attributes[PrimitiveFunction::AttributeNameNormalizationTimeConstant].Value<double>();
                let blendTimeConstant         = attributes[PrimitiveFunction::AttributeNameBlendTimeConstant].Value<double>(); // consider running stats to have this many samples
                if (N == 1 && !isInferring && blendTimeConstant == 0) // (observed in a test case; leave it in until stuff is solid)
                    fprintf(stderr, "WARNING: abnormal BatchNorm input with count=1\n"), fflush(stderr);
                double runningStatsWeight =
                    /*if*/ (isInferring || isinf(blendTimeConstant)) ?
                        1 // inference or only using running stats
                    /*else*/:
                        blendTimeConstant / (N + blendTimeConstant); // mix-in factor for running stats
                // mu and sigma^2
#undef LOG_BN
#ifdef LOG_BN
                fprintf(stderr, "BATCHNORM over %d, id=%d\n", (int)N, (int)attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>()), fflush(stderr);
                x->LogToFile(L"x |> BatchNorm");
#endif
                assert(N == (double)x->Shape().TotalSize(/*check=*/false) / (double)sigma->Shape().TotalSize(/*check=*/false)); // for now
                let& mu = redBuf;
                let& sigma2 = sigma; // we use sigma's location to first compute sigma^2
                // algorithm related to running stats vs. current MB stats
                //  1. update accumulators with new raw MB stats if needed
                //     As long as we are not inferring, compute the raw stats, then update running stats with it
                //  2. update running stats
                //  3. replace raw MB stats with smoothed MB stats if blend specified
                //     Interpolate from running stats to raw model
                //     BUGBUG: This will now bring in the current stats twice, unless we add code to adjust the weights, requiring GPU sync.
                //     Maybe it's OK, since the running stats move very slowly, so the new contribution is very small.
                //  4. in backprop: variance-estimation path should get weighted by the weight used for the raw MB data
                // --- step 1. compute new raw MB stats
                if (!isInferring) // actual MB stats (used in training only)
                {
                    NDArrayView::NumericOperation({ x     }, 1/ N   , Microsoft::MSR::CNTK::ElementWiseOperator::opCopy,            mu    );
                    NDArrayView::NumericOperation({ x, mu }, 1/ N   , Microsoft::MSR::CNTK::ElementWiseOperator::opSqrOfDifference, sigma2); // sigma^2
                    //NDArrayView::NumericOperation({ x, mu }, 1/(N-1), Microsoft::MSR::CNTK::ElementWiseOperator::opSqrOfDifference, sigma2); // sigma^2
                    // TODO: better use the unbiased estimate (a simple test showed very little impact, though)
#ifdef LOG_BN
                    mu    ->LogToFile(L"mu");
                    sigma2->LogToFile(L"sigma2");
#endif
                }
                // --- step 2. update running stats
                if (!isInferring)
                {
                    // statistics update:
                    //  - running accumulator with numer and denom
                    //  - new minibatch has N samples
                    //  - gets decayed by N samples, where N may vary
                    //  - we store 0th, 1st, and 2nd-order accumlators
                    //    NOTE: This is not what is documented. But we consider this internal, so it's OK.
                    //  - and convert them when needed
                    //  - accumulators are actually stored pre-multiplied by (1 - exp(-1 / normalizationTimeConstant)),
                    //    as this keeps the count accumulator around 1 (it converges to a tiny bit above 1
                    //    since we add multiple samples at once; if it was 1 sample at a time, it would be 1).
                    //    This way, runningMean is *nearly* the correct mean, and runningVar would be the variance if the mean was 0.
                    //    This makes debugging those values more meaningful, but is otherwise not criticial for anything.
                    // using:
                    //  - 1/count sum_j(x_j-mu)^2 = 1/count sum_j x_j^2 - 1/count sum_j mu^2 = 1/count sum_j x_j^2 - mu^2
                    //  - var = 1/count sqrSum - mu^2
                    //  - sqrSum = (var + mu^2) * count
                    //  - newSqrSum = (oldVar + oldMu^2) * oldVount + mbSqrSum
                    //  - mbSqrSum = (mbVar + mbMu^2) * N
                    // update:
                    //  - newCount  = oldCount  * exp(-N/T) + N
                    //  - newSum    = oldSum    * exp(-N/T) + N * mbMu
                    //  - newSqrSum = oldSqrSum * exp(-N/T) + N * (mbVar + mbMu^2)
                    // momentum-like formula: xsmooted = xold * exp(-1/T) + xnew * (1 - exp(-1/T))
#ifdef LOG_BN
                    runCount ->LogToFile(L"runCount (before)");
                    runSum   ->LogToFile(L"runSum (before)");
                    runSqrSum->LogToFile(L"runSqrSum (before)");
#endif
                    let decay      =     exp(-N / normalizationTimeConstant); // decay for N steps
                    let complement = 1 - exp(-1 / normalizationTimeConstant); // unit gain for 1 step
                    if (!runCount->IsReadOnly())
                        NDArrayView::NumericOperation({                }, N * complement, Microsoft::MSR::CNTK::ElementWiseOperator::opConstOne, runCount,  decay);
                    if (!runSum->IsReadOnly())
                        NDArrayView::NumericOperation({ mu             }, N * complement, Microsoft::MSR::CNTK::ElementWiseOperator::opCopy,     runSum,    decay);
                    if (!runSqrSum->IsReadOnly())
                        //NDArrayView::NumericOperation({ sigma2         }, N * complement, Microsoft::MSR::CNTK::ElementWiseOperator::opCopy,     runSqrSum, decay);
                        NDArrayView::NumericOperation({ mu, mu, sigma2 }, N * complement, Microsoft::MSR::CNTK::ElementWiseOperator::opAxBplusC, runSqrSum, decay);
                    // now the running stats have been updated with the new MB stats
#ifdef LOG_BN
                    runCount ->LogToFile(L"runCount");
                    runSum   ->LogToFile(L"runSum");
                    runSqrSum->LogToFile(L"runSqrSum");
#endif
                }
                let rawMBStatsWeight = 1.0 - runningStatsWeight; // actual contribution from raw MB stats. In inference, this is 0.
                let doBatchRenorm = runningStatsWeight != 0;
                if (doBatchRenorm) // this triggers Batch Renormalization
                {
                    // in Batch Renorm, we normalize against the running stats
                    // Batch Renormalization inserts an additional step:
                    //  - brn = subtract_mean >> var_norm >> StopGradient(affine_transform) >> mul{scale} >> add{bias}
                    //  - where affine_transform is:
                    //     - replace MB stddev by running stddev (*runningSigma/mbSigma)
                    //     - replace MB mean by running mean
                    //     - where these parameters are considered constants, no backprop through them
                    // Properties:
                    //  - ends up using final/smooth estimates for normalization
                    //  - while still producing gradients that have these properties:
                    //     - perpendicular to the input
                    //       <x_n, dL/dx_n> = 0
                    //     - zero mean
                    //       sum_n dL/dx_n = 0

                    // sigma    --remember that this is in-place, &sigma2==&sigma
                    //sigma2->LogToFile(L"sigma2 (local)");
                    NDArrayView::NumericOperation({ sigma2 }, 1.0, Microsoft::MSR::CNTK::ElementWiseOperator::opSqrt, sigma);
                    double epsilon = attributes[PrimitiveFunction::AttributeNameEpsilon].Value<double>();
                    if (epsilon > 0) // we add eps to sigma to avoid dividing by 0 or very small estimates
                        NDArrayView::NumericOperation({ }, epsilon, Microsoft::MSR::CNTK::ElementWiseOperator::opConstOne, sigma, /*beta=*/1.0); // sigma + eps (in-place)
#ifdef LOG_BN
                    sigma->LogToFile(L"sigma (local)");
#endif

                    // normalize input by MB statistics: xHat = (x-mu)/sigma
                    // This represents the first two steps of Batch Renorm.
                    // It also produces a unit-length direction vector used for projection in backprop.
                    NDArrayView::NumericOperation({ x, sigma, mu }, 1.0, Microsoft::MSR::CNTK::ElementWiseOperator::opAminusCoverB, xHat);  // (x-mu)/sigma
#ifdef LOG_BN
                    xHat->LogToFile(L"xHat");
#endif

                    // apply the batch renorm factors, to produce the output
                    //  - out = xHat * sigma_MB / sigma_running
                    //  - out += (- mu_MB + mu_running) / sigma_running
                    // undo MB sigma
                    NDArrayView::NumericOperation({ xHat, sigma }, 1.0, Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProduct, out);  // -> (x-mu) again
#ifdef LOG_BN
                    out->LogToFile(L"x-mu");
#endif
                    // compute running sigma   --this overwrites MB sigma
                    if (rawMBStatsWeight != 0) // if we should interpolate with MB stats, then need to recover it
                                               // BUGBUG: This does not exactly reproduce sigma2 due to epsilon.
                        NDArrayView::NumericOperation({ sigma }, 1.0, Microsoft::MSR::CNTK::ElementWiseOperator::opSqr, sigma2);
                    NDArrayView::NumericOperation({ runSqrSum, runCount },  runningStatsWeight, Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseQuotient,    sigma2, rawMBStatsWeight);
                    NDArrayView::NumericOperation({ runSum,    runCount }, -runningStatsWeight, Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseQuotientSqr, sigma2, 1.0);
#ifdef LOG_BN
                    sigma2->LogToFile(L"sigma2 (running)");
#endif
                    NDArrayView::NumericOperation({ sigma2 }, 1.0, Microsoft::MSR::CNTK::ElementWiseOperator::opSqrt, sigma);
                    if (epsilon > 0) // we add eps to sigma to avoid dividing by 0 or very small estimates
                        NDArrayView::NumericOperation({ }, epsilon, Microsoft::MSR::CNTK::ElementWiseOperator::opConstOne, sigma, /*beta=*/1.0); // sigma + eps (in-place)
#ifdef LOG_BN
                    sigma->LogToFile(L"sigma (running)");
#endif
                    // apply running sigma  --in-place
                    NDArrayView::NumericOperation({ out, sigma }, 1.0, Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseQuotient, out);  // (x-muMb)/sigmaRunning
#ifdef LOG_BN
                    out->LogToFile(L"(x-mu)/sigmaRunning");
#endif
                    // undo MB mu
                    NDArrayView::NumericOperation({ mu, sigma }, 1.0, Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseQuotient, out, 1.0);  // += muMb/sigmaRunning
#ifdef LOG_BN
                    out->LogToFile(L"(x-mu)/sigmaRunning + mu/sigmaRunning");
#endif
                    // compute running mu   --this overwrites MB mu
                    NDArrayView::NumericOperation({ runSum, runCount }, runningStatsWeight, Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseQuotient, mu, rawMBStatsWeight);
#ifdef LOG_BN
                    mu->LogToFile(L"mu (running)");
#endif
                    // apply running mu
                    NDArrayView::NumericOperation({ mu, sigma }, -1.0, Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseQuotient, out, 1.0);  // -= muRunning/sigmaRunning
#ifdef LOG_BN
                    out->LogToFile(L"(x-muRunning)/sigmaRunning");
#endif

                    // apply scale and bias (in-place)
                    NDArrayView::NumericOperation({ out, scale, bias }, 1.0, Microsoft::MSR::CNTK::ElementWiseOperator::opAxBplusC, out);
#ifdef LOG_BN
                    out->LogToFile(L"out (final)");
#endif

#ifdef LOG_BN
                    sigma->LogToFile(L"sigma (running), as passed to BackProp");
                    xHat->LogToFile(L"xHat (local), as passed to BackProp");
#endif
                }
                else
                {
                    if (runningStatsWeight != 0) // we got a contribution from the running stats
                    {
                        //runCount ->LogToFile(L"runCount");
                        //runSum   ->LogToFile(L"runSum");
                        //runSqrSum->LogToFile(L"runSqrSum");
                        NDArrayView::NumericOperation({ runSum,    runCount },  runningStatsWeight, Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseQuotient,    mu    , rawMBStatsWeight);
                        NDArrayView::NumericOperation({ runSqrSum, runCount },  runningStatsWeight, Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseQuotient,    sigma2, rawMBStatsWeight);
                        NDArrayView::NumericOperation({ runSum,    runCount }, -runningStatsWeight, Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseQuotientSqr, sigma2, 1.0);
                        // ^^ var = 1/count sqrSum - mu^2 = 1/count sqrSum - (sum/count)^2   --this subtracts mu^2
#ifdef LOG_BN
                        mu    ->LogToFile(L"mu'");
                        sigma2->LogToFile(L"sigma2'");
#endif
                    }
                    // now perform actual BatchNormalization based on the estimates of mu and sigma^2
                    // sigma (remember that this is in-place, &sigma2==&sigma)
                    NDArrayView::NumericOperation({ sigma2 }, 1.0, Microsoft::MSR::CNTK::ElementWiseOperator::opSqrt, sigma);
                    double epsilon = attributes[PrimitiveFunction::AttributeNameEpsilon].Value<double>();
                    if (epsilon > 0) // we add eps to sigma to avoid dividing by 0 or very small estimates
                        NDArrayView::NumericOperation({ }, epsilon, Microsoft::MSR::CNTK::ElementWiseOperator::opConstOne, sigma, /*beta=*/1.0); // sigma + eps (in-place)
                    // xHat = (x-mu)/sigma
                    NDArrayView::NumericOperation({ x, sigma, mu }, 1.0, Microsoft::MSR::CNTK::ElementWiseOperator::opAminusCoverB, xHat);  // (x-mu)/sigma
                    // apply scale and bias
                    NDArrayView::NumericOperation({ xHat, scale, bias }, 1.0, Microsoft::MSR::CNTK::ElementWiseOperator::opAxBplusC, out);
                }
            }
            break;
            // the following operations are not TensorView, but are implementable through relatively simple calls to Matrix
        case PrimitiveOpType::OneHot:
            NDArrayView::AsOneHot(args[0],
                                  (size_t)attributes[PrimitiveFunction::AttributeNameOneHotAxis].Value<Axis>().StaticAxisIndex(),
                                  out);
            break;
            // the following operations are not TensorView, and hence should be routed through V1 ComputationNodes
            // convolution family
        case PrimitiveOpType::Convolution:  // TODO: route these through TensorView
        case PrimitiveOpType::Pooling:
        case PrimitiveOpType::Unpooling:
        case PrimitiveOpType::ROIPooling:
            // random family
        case PrimitiveOpType::RandomSample:
        case PrimitiveOpType::RandomSampleInclusionFrequency:
            // --- ops below are those that do not apply to Dynamite and therefore are not supported for direct evaluation
            // dynamic-axis related operations are not supported as dynamic axes require Inputs and are therefore not applicable here
        case PrimitiveOpType::Gather:
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
            RuntimeError("Variable '%S' Value(): Memoziation of operation %S not applicable (applies only to static graphs).", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
            // special ops family
        // TODO: these are new, need to finish them
        case PrimitiveOpType::Sinh:
        case PrimitiveOpType::Cosh:
        case PrimitiveOpType::UnpackBatch:
        case PrimitiveOpType::ToBatch:
        case PrimitiveOpType::Asin:
        case PrimitiveOpType::Acos:
        case PrimitiveOpType::Pad:
        case PrimitiveOpType::Crop:
        // END new ops
        case PrimitiveOpType::LambdaRank:
        case PrimitiveOpType::NDCG:
        case PrimitiveOpType::EditDistanceError:
        case PrimitiveOpType::LabelsToGraph:
        case PrimitiveOpType::ForwardBackward:
        case PrimitiveOpType::CosDistanceWithNegativeSamples:
            LogicError("Variable '%S' Value(): Memoziation of operation %S not implemented yet.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
        case PrimitiveOpType::Softmax:
        case PrimitiveOpType::LogSoftmax:
        case PrimitiveOpType::Hardmax:
        case PrimitiveOpType::Dropout:     // should generate as a mul with a random output
        case PrimitiveOpType::CrossEntropyWithSoftmax:
        case PrimitiveOpType::ClassificationError:
        case PrimitiveOpType::Logistic:
        case PrimitiveOpType::SquaredError:
        case PrimitiveOpType::CosDistance: // should generate discretely
        case PrimitiveOpType::SumAll:      // never generated by API
            LogicError("Variable '%S' Value(): Memoziation of operation %S not supported on this level. This should be generated differently from the outer layer.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
        default:
            LogicError("Variable '%S' Value(): Memoziation of non-existent operation %S?", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
        }
        // most common case: elementwise ops are done here instead
        if (op != Microsoft::MSR::CNTK::ElementWiseOperator::opNone)
            NDArrayView::NumericOperation(args, alpha, op, out, 0.0, reductionOp);
        //out->LogToFile(PrimitiveOpTypeName(primitiveOp), stderr);
        return out;
    }

    // perform back propagation into an input
    // For now only defined for functions with 1 output.
    // Gradient must have been allocated to the correct shape already.
    // If beta == 0 then gradient can be uninitialized memory.
    // Important: Beta is meant to apply to the *entire* gradient tensor. Specifically, also for the case of Slice(),
    // beta == 0 will set the *entire* gradient tensor to 0, not just the slice.
    // Hence, when back-propagating into multiple slices of the same matrix, only pass beta=0 for the first one.
    // TODO: decide whether we pass raw pointers or shared pointers... Why are we passing naked pointers again?
    /*static*/ void PrimitiveFunction::BackpropTo(const NDArrayView* outputGradientValue,                    // incoming gradient from top...
                              size_t i, PrimitiveOpType primitiveOp, const Dictionary& attributes,           // ...goes through this backprop function...
                              const NDArrayView* outputValue, const vector<const NDArrayView*>& inputValues, // ...using these values from forward pass...
                              const NDArrayViewPtr& gradient, double beta,                                   // ...into here. (Despite 'const', *gradient is the output.)
                              const PrimitiveFunction& funcForErrMsg)
    {
        // The majority of operators are handled by shared code after the switch statement, based on the following op-code variables.
        // Special cases do operations inside the cases themselves, and leave the opcodes untouched.
        bool handled = false; // set this for gradients that do not use the shared code at the end
        bool op0 = false; // opcode to indicate that the gradient is zero
        auto op1Arg  = Microsoft::MSR::CNTK::ElementWiseOperator::opNone; // opcode for single-argument gradients
        auto op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opNone; // and this for 2-arg ops
        auto op3Args = Microsoft::MSR::CNTK::ElementWiseOperator::opNone; // and this for 3-arg ops; all others execute inside the switch()
        const NDArrayView* arg1 = outputGradientValue; // value of incoming gradient from top is the first arg of most gradient ops, so set it by default (some ops will change it)
        const NDArrayView* arg2 = nullptr;
        const NDArrayView* arg3 = nullptr;
        double alpha = 1;
        // NOTE: For now, this only implements the operators needed for the prototype
        switch (primitiveOp)
        {
            // binary operations with simple TensorView implementation
        case PrimitiveOpType::Plus:           op1Arg  = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy; break;
        case PrimitiveOpType::Minus:          op1Arg  = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy; alpha = i == 0 ? 1 : -1; break;
        case PrimitiveOpType::ElementTimes:   op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProduct; arg2 = inputValues[1 - i]; break;
        case PrimitiveOpType::LogPlus:        op3Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithLogSumDerivative; arg2 = inputValues[1 - i]; arg3 = inputValues[i]; break;
        case PrimitiveOpType::Pow:
            if (i == 0)
            {
                op3Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithPowBaseDerivative;
                arg2 = inputValues[0];
                arg3 = inputValues[1];
            }
            else
            {
                op3Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithPowExponentDerivative;
                arg2 = outputValue;
                arg3 = inputValues[0];
            }
            break;
            // Times family
        case PrimitiveOpType::Times:
        case PrimitiveOpType::TransposeTimes:
        case PrimitiveOpType::Affine:
        case PrimitiveOpType::TransposeAffine:
            if (i == 2) // Affine only: bias
                op1Arg = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy;
            else // weight and input (for Affine, the incoming bias is passed through Plus unmodified)
            {
                arg2 = inputValues[1 - i]; // arg1 is the incoming gradient from above
                if (i == 0) // left input
                    NDArrayView::MatrixProduct(/*transC=*/(primitiveOp == PrimitiveOpType::TransposeTimes || primitiveOp == PrimitiveOpType::TransposeAffine),
                                              /*A=*/const_cast<NDArrayView*>(arg1)->shared_from_this(), /*transA=*/false,
                                              /*B=*/const_cast<NDArrayView*>(arg2)->shared_from_this(), /*transB=*/true,  alpha, /*outputRank dummy=*/0, /*C=*/gradient, beta);
                else // right input
                    NDArrayView::MatrixProduct(/*transC=*/false,
                                              /*A=*/const_cast<NDArrayView*>(arg2)->shared_from_this(), /*transA=*/(primitiveOp != PrimitiveOpType::TransposeTimes && primitiveOp != PrimitiveOpType::TransposeAffine),
                                              /*B=*/const_cast<NDArrayView*>(arg1)->shared_from_this(), /*transB=*/false, alpha, /*outputRank dummy=*/0, /*C=*/gradient, beta);
                handled = true;
            }
            break;
            // unary operations with simple TensorView implementation
        case PrimitiveOpType::ReLU:          op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithLinearRectifierDerivativeFromOutput;       arg2 = outputValue; break;
        case PrimitiveOpType::Tanh:          op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithTanhDerivativeFromOutput;                  arg2 = outputValue; break;
        case PrimitiveOpType::Sigmoid:       op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithSigmoidDerivativeFromOutput;               arg2 = outputValue; break;
        case PrimitiveOpType::Log:           op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithLogDerivativeFromOutput;                   arg2 = outputValue; break;
        case PrimitiveOpType::Exp:           op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProduct;                                              arg2 = outputValue; break;
        case PrimitiveOpType::Cos:           op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithCosDerivative;                             arg2 = inputValues[0]; break;
        case PrimitiveOpType::Sin:           op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithSinDerivative;                             arg2 = inputValues[0]; break;
        case PrimitiveOpType::Negate:        op1Arg  = Microsoft::MSR::CNTK::ElementWiseOperator::opNegate;                                                          break;
        case PrimitiveOpType::Abs:           op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithAbsDerivative;                             arg2 = inputValues[0]; break;
        case PrimitiveOpType::Sqrt:          op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithSqrtDerivative;                            arg2 = outputValue; break;
        case PrimitiveOpType::Reciprocal:    op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithReciprocalDerivative;                      arg2 = outputValue; break;
        case PrimitiveOpType::ELU:           op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithExponentialLinearUnitDerivativeFromOutput; arg2 = outputValue; break;
        case PrimitiveOpType::StableSigmoid: op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithSigmoidDerivativeFromOutput;               arg2 = outputValue; break;
        case PrimitiveOpType::ScaleAndShift:
            // PERF BUGBUG: Without scale, this should not require a copy, just a view, like Plus.
            alpha = attributes[PrimitiveFunction::AttributeNameScale].Value<double>();
            op1Arg = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy;
            break;
        case PrimitiveOpType::TransposeAxes:
            // TODO: one day we may treat this as a see-through gradient, in Autobatch
            NDArrayView::NumericOperation({ outputGradientValue->AsTransposed(TransposePermutationArg(attributes), /*invert=*/true) }, 1, Microsoft::MSR::CNTK::ElementWiseOperator::opCopy, gradient, beta);
            handled = true;
            break;
            // gradients that are copies with broadcasting
            // no-op operations with simple TensorView implementation
            // NOTE: These do not need any data copy if there is only one consumer, which we won't know here. That case will be caught in the batched version.
        case PrimitiveOpType::NoOp:          op1Arg  = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy; break;
        case PrimitiveOpType::Reshape:
            // shapes won't match, so reshape one to match the other (we can't just opCopy with mismatching shapes, which may cause weird broadcasting weirdness that won't crash but produce garbage values)
            if (outputGradientValue->Shape() != gradient->Shape())
            {
                NDArrayView::NumericOperation({ outputGradientValue->AsShape(gradient->Shape()) }, alpha, // differs from shared code below in passing the reshaped outputGradientValue
                                              Microsoft::MSR::CNTK::ElementWiseOperator::opCopy, gradient, beta);
                handled = true;
            }
            else
                op1Arg  = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy; // (no shape change: they match already, use the the shared code below)
            break;
        case PrimitiveOpType::ReduceElements:
            {
                const auto& reductionOpName = attributes[PrimitiveFunction::AttributeNameReductionOpName].Value<wstring>();
                if (reductionOpName == PrimitiveFunction::InternalSumReductionOpName)
                    op1Arg = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy;
                else if (reductionOpName == PrimitiveFunction::InternalLogSumReductionOpName)
                {
                    NDArrayView::NumericOperation({ const_cast<NDArrayView*>(outputGradientValue)->shared_from_this(),
                                                    const_cast<NDArrayView*>( inputValues[0]    )->shared_from_this(),
                                                    const_cast<NDArrayView*>(outputValue        )->shared_from_this() }, alpha,
                                                  Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProductWithExpOfDiff,
                                                  gradient, beta,
                                                  Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
                    handled = true;
                }
                else if (reductionOpName == PrimitiveFunction::InternalMeanReductionOpName)
                {
                    op1Arg = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy;
                    alpha = (double)outputValue->Shape().TotalSize(/*check=*/false) / (double)inputValues[0]->Shape().TotalSize(/*check=*/false);
                }
                else
                    //  PrimitiveFunction::InternalMaxReductionOpName
                    //  PrimitiveFunction::InternalMinReductionOpName
                    //  PrimitiveFunction::InternalProdReductionOpName
                    LogicError("Variable '%S' Value(): Gradient of reduction op %S not yet implemented.", funcForErrMsg.AsString().c_str(), reductionOpName.c_str());
            }
            break;
            // hard stuff
        case PrimitiveOpType::Splice:
            {
                let& outputGradientShape = outputGradientValue->Shape(); // what's coming from the top (spliced)
                let axis = attributes[/*L"axis"*/PrimitiveFunction::AttributeNameAxis].Value<Axis>().StaticAxisIndex();
                auto outputGradientRank = outputGradientShape.Rank();
                if (axis >= outputGradientRank)
                    LogicError("Variable '%S' Value(): Splice axis %d exceeds outputGradientRank %d.", funcForErrMsg.AsString().c_str(), (int)axis, (int)outputGradientRank);
                if (i >= outputGradientShape[axis])
                    LogicError("Variable '%S' Value(): Input index %d exceeds dimension[%d]=%d.", funcForErrMsg.AsString().c_str(), (int)i, (int)axis, (int)outputGradientShape[axis]);
                // determine the slice of the incoming gradient that we should copy
                NDShapeDimensions startOffset(outputGradientRank, 0);
                NDShapeDimensions extent = outputGradientShape.Dimensions();
                let gradientRank = gradient->Shape().Rank();
                if (axis >= gradientRank) // this was a splice along a new axis: we can index directly
                {
                    startOffset[axis] = i;
                    let& lesserExtent = NDShapeDimensionsSpan(extent.begin(), extent.begin() + gradientRank); // chop off the trailing dims, which will lead to the slice to not have them either; which will in turn opCopy correctly
                    //extent.resize(gradientRank); // chop off the trailing dims, which will lead to the slice to not have them either; which will in turn opCopy correctly
                    // note: we are doing this here instead of in the shared code below (arg1=outputGradientSlice) only because arg1 is a naked pointer, but we must manage lifetime of 'outputGradientSlice'
                    NDArrayView::NumericOperation({ outputGradientValue->Slice(startOffset, lesserExtent, /*strides=*/NDShapeDimensions()) }, alpha,
                                                  Microsoft::MSR::CNTK::ElementWiseOperator::opCopy, gradient, beta);
                }
                else // splice along an existing axis: we must find out the start and extent by aggregation
                {
                    // Note that this is O(N^2) when back-propping into all inputs. This code, however, only runs
                    // if we don't use batched backprop. Batched backprop optimizes this using bulk Scatter propagation.
                    for (size_t j = 0; j < i; j++)
                    {
                        let& inputShape = inputValues[j]->Shape();
                        startOffset[axis] += axis < inputShape.Rank() ? inputShape[axis] : 1;
                    }
                    let& inputShape = inputValues[i]->Shape();
                    extent[axis] = axis < inputShape.Rank() ? inputShape[axis] : 1;
                    // note: we are doing this here instead of in the shared code below (arg1=outputGradientSlice) only because arg1 is a naked pointer, but we must manage lifetime of 'outputGradientSlice'
                    NDArrayView::NumericOperation({ outputGradientValue->Slice(startOffset, extent, /*strides=*/NDShapeDimensions()) }, alpha,
                                                  Microsoft::MSR::CNTK::ElementWiseOperator::opCopy, gradient, beta);
                }
                handled = true;
            }
            break;
        case PrimitiveOpType::Slice:
            // Backprop into the input slice of the input gradient: We can use forward-prop to determine the slice.
            if (beta == 0) // if beta = 0 then we must explicitly initialize the entire gradient matrix, not just the slice
                gradient->SetValue(0.0f);
            NDArrayView::NumericOperation({ const_cast<NDArrayView*>(outputGradientValue)->shared_from_this() }, alpha,
                                          Microsoft::MSR::CNTK::ElementWiseOperator::opCopy,
                                          GetSliceView(gradient, attributes, outputValue->Shape(), /*readOnly=*/false, funcForErrMsg),
                                          beta); // keep beta; although we just cleared it, beta=0 avoids the memory access
            // This ^^ op is the same as the shared one, except for the output slice.
            handled = true;
            break;
        case PrimitiveOpType::ElementAffine:
            if (i == 2) // bias is like Plus
                op1Arg = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy;
            else // factors are like ElementTimes
                op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProduct; arg2 = inputValues[1 - i];
            break;
        case PrimitiveOpType::Clip:
            if (i == 0)
            {
                // BUGBUG: This is not working, when implementing a clipped ReLU with it in MT.cpp. Different result from double-ReLU expression.
                op3Args = Microsoft::MSR::CNTK::ElementWiseOperator::opCopyIfEqual;
                arg1 = inputValues[0];
                arg2 = outputValue;
                arg3 = outputGradientValue;
            }
            else // the bounds have no gradient
                op0 = true;
            break;
        case PrimitiveOpType::Select:
            if (i == 0) // the condition has no gradient
                op0 = true;
            else
            {
                if (i == 1)
                    op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opCopyIf;
                else
                    op2Args = Microsoft::MSR::CNTK::ElementWiseOperator::opCopyIfNot;
                arg1 = inputValues[0];
                arg2 = outputGradientValue;
            }
            break;
            // primitives that have zero gradient (piecewise constants)
        case PrimitiveOpType::Equal:
        case PrimitiveOpType::NotEqual:
        case PrimitiveOpType::Less:
        case PrimitiveOpType::LessEqual:
        case PrimitiveOpType::Greater:
        case PrimitiveOpType::GreaterEqual:
        case PrimitiveOpType::Floor:
            op0 = true; // will be set to zero below
            break;
        case PrimitiveOpType::BatchNormalization:
            if (i == 0) // input argument
            {
                let& outGradVal = const_cast<NDArrayView*>(outputGradientValue)->shared_from_this(); // dL/dyi
                let& scale  = const_cast<NDArrayView*>(inputValues[1])->shared_from_this();
                //let& runSum = const_cast<NDArrayView*>(inputValues[3])->shared_from_this(); // for readOnly check
                let& sigma  = const_cast<NDArrayView*>(inputValues[7])->shared_from_this(); // sigma
                let& redBuf = const_cast<NDArrayView*>(inputValues[6])->shared_from_this(); // reduced-size buffer, allocated for mean
                let& xHat   = const_cast<NDArrayView*>(inputValues[8])->shared_from_this(); // (xi-mu)/sigma
                // [adapted from CNTK engine source:]
                // From the BN paper, dL/dxi is a sum of three terms: dL/dxi = t1 + t2 + t3
                // The formulas for dL/dBias and dL/dScale happen to occur as subexpressions in this gradient as well.
                // Leveraging this, this gradient can be simplified to:
                //   t1 = scale * dL/dyi * invStdDev
                //   t2 = mbStatsWeight * (-scale / m) * invStdDev * xiHat * dL/dScale
                //   t3 = mbStatsWeight * (-scale / m) * invStdDev * dL/dBias (for this one note that Reduce(xHat) == 0)
                // with
                //   xiHat = (xi - mean) * invStdDev
                //   dL/dBias = Reduce(dL/dyi)
                //   dL/dScale = Reduce(dL/dyi * xiHat)
                // gradient +=
                //     (scale / sigma) * outputGradientValue +
                //     (scale / sigma) * -1/N * (xHat * scaleGradient)
                //     (scale / sigma) * -1/N * biasGradient;
                // TODO: redundant with gradients[1] and [2]--how to cache this?
                // In case of fixed stats, we only have the first term. That'd be bad, since it no longer leaves the length of the input unchanged.
                //   ^^ BUGBUG: We need to check the volatile flag correctly, to make this test work.
                // what the gradients are, in words, of the normalization part:
                //  - bn = meanNorm >> lengthNorm >> mul_by_sqrt_N >> scale >> addBias
                //    with
                //     - lengthNorm(x') = x' / |x'|
                //     - meanNorm(x)    = x  - mean(x)         # == x'
                //  - gradient through lengthNorm(x'):
                //     - starting with gradient from top,
                //     - divide it by |x'|, and
                //     - "orthogonally reject" it on x' == only keep components perpendicular to x'
                //     - expressed as function composition:
                //       divide_by_length{x'} >> Sum(identity, orthogonal_projection_on{x'} >> negate)
                //  - gradient through meanNorm(x):
                //     - starting with gradient from top,
                //     - make it zero-mean
                //     - expressed as function composition:
                //       Sum(identity, mean >> negate)
                // In this gradient computation, xHat's role is a unit-length direction vector of (x-mu).

                //  - total gradient function G = d(meanNorm >> lengthNorm)(x)/dx:
                //       G = divide_by_length{x'} >> Sum(identity, orthogonal_projection_on{x'} >> negate) >> Sum(identity, mean >> negate)
                //         = Sum(divide_by_length{x'} >> identity, divide_by_length{x'} >> orthogonal_projection_on{x'} >> negate) >> Sum(identity, mean >> negate)
                //         = Sum(divide_by_length{x'} >> identity >> Sum(identity, mean >> negate),
                //               divide_by_length{x'} >> orthogonal_projection_on{x'} >> negate >> Sum(identity, mean >> negate))
                //         = Sum(divide_by_length{x'} >> identity >> identity,
                //               divide_by_length{x'} >> identity >> mean >> negate,
                //               divide_by_length{x'} >> orthogonal_projection_on{x'} >> negate >> identity),
                //               divide_by_length{x'} >> orthogonal_projection_on{x'} >> negate >> mean >> negate)
                //         = Sum(divide_by_length{x'},
                //               divide_by_length{x'} >> mean >> negate,
                //               divide_by_length{x'} >> orthogonal_projection_on{x'} >> negate),
                //               divide_by_length{x'} >> orthogonal_projection_on{x'} >> mean)
                //  - orthogonal_projection_on{x'}
                //         = lambda g: x' * (x'/|x'|)^T * g         # x' = meanNorm(x)               --is g a row or a column?

                //         = lambda g: x' * x'^T *  g / |x'|
                //         = lambda g: (x-Mx) * (x-Mx)^T *  g / |x'|        # M = eye(N,N)/N computes mean and broadcasts it
                //         = lambda g: [x x^T - M x x^T - x x^T M^T + M x x^T M^T]  *  g / |x'|        # M = eye(N,N)/N computes mean and broadcasts it

                // add first term to gradient. This is the main path.
                // outGradVal * (scale / sigma) == gradient through scale. This gives us the "gradient from top" of lengthNorm.
                //gradient->LogToFile(L"gradient at start, must be 0 for checks below to be correct");
                NDArrayView::NumericOperation({ outGradVal, scale, sigma }, 1.0, opElementwiseProductWithQuotient, gradient, beta);
                // add second term, that is, the path through the std dev estimator, which is
                // -(xHat * scaleGradient/N) * (scale / sigma)
                // scaled down by the weight that we actually use it, in case of blending
                // orthogonally project gradient from top onto x
                let N = (double)inputValues[0]->Shape().TotalSize(/*check=*/false) / (double)redBuf->Shape().TotalSize(/*check=*/false);
                //let blendTimeConstant = attributes[PrimitiveFunction::AttributeNameBlendTimeConstant].Value<double>(); // consider running stats to have this many samples
                //double runningStatsWeight =
                //    /*if*/ (isinf(blendTimeConstant)) ?
                //        1 // only using running stats
                //    /*else*/:
                //        blendTimeConstant / (N + blendTimeConstant); // mix-in factor for running stats
                //let doBatchRenorm = runningStatsWeight == 1; // this triggers Batch Renormalization
                let rawMBStatsWeight =
                //    /*if*/ (doBatchRenorm) ?
                        1.0         ;            // renorm: just use raw MB stats for backprop
                //    /*else*/:
                //        1.0 - runningStatsWeight; // actual contribution from raw MB stats
                //// note: scaleGradientAv and biasGradientAv both share a buffer with mu, since they are not needed at the same time
                //if (rawMBStatsWeight != 0)
                //{
                    // xHat has length sqrt(N)
#if 0               // check that xHat has length sqrt(N)
                    let xHatL2 = NDArrayView::MatrixProduct(false, xHat, false, xHat, true, /*alpha=*/1 / N, 1, nullptr);
                    xHatL2->LogToFile(L"xHatL2/sqrt(N)"); // diagonal should be all ones
#endif
                    //xHat->LogToFile(L"xHat", stderr, 10000);
                    //outGradVal->LogToFile(L"outGradVal", stderr, 10000);
                    //scale->LogToFile(L"scale", stderr, 10000);
                    //sigma->LogToFile(L"sigma", stderr, 10000);
                    //let projectionCoefficients1 = NDArrayView::NumericOperation({ outGradVal, xHat }, /*alpha=*/1 / sqrt(N), opElementwiseProduct, redBuf);
                    //projectionCoefficients1->LogToFile(L"projectionCoefficients1 of outGradVal on xHat");
                    // any use of xHat below must multiply with 1/sqrt(N) to make it unit-length
                    let projectionCoefficients = NDArrayView::NumericOperation({ outGradVal, xHat, scale, sigma }, /*alpha=*/1 / sqrt(N), opAxBxCoverD, redBuf);
                    //projectionCoefficients->LogToFile(L"projectionCoefficients of outGradVal on xHat");
                    // This gets the projection coefficients for projecting outGradVal onto (x-mu).
                    // Now multiply it with xHat, to get the projection of outGradVal onto (x-mu),
                    // and then subtract it from outGradVal. If rawMBStatsWeight==1 then that will leave only the contribution
                    // that is perpendicular to (x-mu).
                    // Note that xHat has norm sqrt(N), not 1. Hence we must divide by sqrt(N).
                    //NDArrayView::NumericOperation({ xHat, projectionCoefficients, scale, sigma }, /*alpha=*/-rawMBStatsWeight / sqrt(N), opAxBxCoverD, gradient, /*beta=*/1.0);
                    NDArrayView::NumericOperation({ xHat, projectionCoefficients }, /*alpha=*/-rawMBStatsWeight / sqrt(N), opElementwiseProduct, gradient, /*beta=*/1.0);
                    //gradient->LogToFile(L"gradient at input of var norm");
#if 0               // check whether gradient is indeed orthogonal
                    let x = const_cast<NDArrayView*>(inputValues[0])->shared_from_this();
                    x->LogToFile(L"x");
                    let mu = NDArrayView::NumericOperation({ x }, 1/ N   , Microsoft::MSR::CNTK::ElementWiseOperator::opCopy, redBuf);
                    mu->LogToFile(L"mu");
                    let xm = x - mu;
                    xm->LogToFile(L"xm");
                    let dotProds = NDArrayView::MatrixProduct(false, gradient, false, xm, true, 1.0, 1, nullptr);
                    dotProds->LogToFile(L"dotProds"); // diagonal should be all zeroes
#endif
                    // add third term, that is, the path through std dev and mean estimator, which is
                    // (scale / sigma) * -1/N * biasGradient
                    let biasGradient = NDArrayView::NumericOperation({ outGradVal }, /*alpha=*/1 / N, opCopy, redBuf);
                    // ^^ this computes the batch means of the incoming gradient
                    NDArrayView::NumericOperation({ biasGradient, scale, sigma }, -rawMBStatsWeight, opElementwiseProductWithQuotient, gradient, /*beta=*/1.0);
                    // This subtracts the batch mean, hence makes the gradient zero-mean.
                    // TODO: This seems to ignore the path through the variance normalizer. Is that already zero mean?
                //}
                handled = true;
            }
            else if (i == 1) // scale is a reduction over outputGradientValue * (x-mu)/sigma
            {
                // Note: In case of Batch Renormalization, this may be wrong, in that it should use the smoothed statistics.
                //       Hence, do not use learnable scale for Batch Renormalization. --TODO: add a check here
                let& xHat = inputValues[8]; // (x-mu)/sigma
                NDArrayView::NumericOperation({ const_cast<NDArrayView*>(outputGradientValue)->shared_from_this(),
                                                const_cast<NDArrayView*>(xHat)->shared_from_this() },
                                              /*alpha=*/1.0,
                                              Microsoft::MSR::CNTK::ElementWiseOperator::opElementwiseProduct, gradient, beta);
                //gradient->LogToFile(L"-> bn scale gradient", stderr);
                handled = true;
            }
            else if (i == 2) // bias is just a reduction over elements
            {
                op1Arg = Microsoft::MSR::CNTK::ElementWiseOperator::opCopy;
            }
            else
                op0 = true; // no gradients except for
            break;
        case PrimitiveOpType::OneHot: // should we just make it zero?
            LogicError("Variable '%S' Value(): Backpropagation for operation %S is not defined.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
        default:
            //fprintf(stderr, "NEEDS: %S\n", PrimitiveOpTypeName(primitiveOp).c_str());
            LogicError("Variable '%S' Value(): Backpropagation for operation %S not implemented yet.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
            //LogicError("Variable '%S' Value(): Backpropagation for non-existent operation %S?", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
        }
        // interpret the opcodes
        // An opcode must be set, or 'handled' must be set when the gradient was already completed in the case above.
        // TODO: we can eliminate the vector<> by passing a std::function, possibly?
        if (op0) // gradient is zero
        {
            if (beta == 0)
                gradient->SetValue(0.0f);
            else if (beta != 1) // will this ever be needed?
                LogicError("Variable '%S' Value(): Backpropagation with beta != 0 or 1 not implemented yet (operation %S).", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
        }
        else if (op1Arg != Microsoft::MSR::CNTK::ElementWiseOperator::opNone) // gradient is a TensorView op with one operand
            NDArrayView::NumericOperation({ const_cast<NDArrayView*>(arg1)->shared_from_this() }, alpha,
                op1Arg, gradient, beta,
                Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
        else if (op2Args != Microsoft::MSR::CNTK::ElementWiseOperator::opNone) // gradient is a TensorView op with two operands
            NDArrayView::NumericOperation({ const_cast<NDArrayView*>(arg1)->shared_from_this(), const_cast<NDArrayView*>(arg2)->shared_from_this() }, alpha,
                op2Args, gradient, beta,
                Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
        else if (op3Args != Microsoft::MSR::CNTK::ElementWiseOperator::opNone) // gradient is a TensorView op with three operands
            NDArrayView::NumericOperation({ const_cast<NDArrayView*>(arg1)->shared_from_this(), const_cast<NDArrayView*>(arg2)->shared_from_this(), const_cast<NDArrayView*>(arg3)->shared_from_this() }, alpha,
                op3Args, gradient, beta,
                Microsoft::MSR::CNTK::ElementWiseOperator::opSum);
        else if (!handled)
            LogicError("Variable '%S' Value(): Gradient for operation %S misses a handler.", funcForErrMsg.AsString().c_str(), PrimitiveOpTypeName(primitiveOp).c_str());
    }
}
