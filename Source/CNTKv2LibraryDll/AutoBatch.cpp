//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// The functions for automatically-batched evaluation of dynamic graphs (forward and backward) are contained here.

// BUGBUG: The redirection approach causes consistency problems, since redirects can end up being Outputs with no m_ownerFunction.
//         Alternative, less brittle approach: Introduce a new VariableKind::Redirect?
// BUGBUG: One gradient test sometimes fails. I think there is a SetValue(0) missing.

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "stdafx.h"
#include "PrimitiveFunction.h"
#include "CNTKLibrary.h"
#include "Variable.h"
#include "PrimitiveOpType.h"
#include "PrimitiveFunction.h"
#include "BlockFunction.h"
#include "CompositeFunction.h"
#include "CommonMatrix.h"
#include "Utils.h"

#include <unordered_map>
#include <vector>
#include <string>

//#define LOG_DETAILS   // if defined, log all forward and backward operations
#define LOG_STATS     // if defined, log statistics (#operations)
//#define NO_BATCHED_FORWARD  // if defined, don't batch forward
//#define NO_BATCHED_BACKPROP // if defined, don't do additional batching or any other extra optimization in backprop
#define DETAILED_STATS

#ifdef LOG_STATS
static size_t logMemoizeStatsPeriod = 50;
static size_t logMemoizeStatsCounter = logMemoizeStatsPeriod - 1; // counts up to logMemoizeStatsPeriod and wraps. We log if it is 0, starting with the second MB.
#else
static size_t logMemoizeStatsPeriod = SIZE_MAX;
static size_t logMemoizeStatsCounter = 1;
#endif
static bool ShouldLogMemoizeStats() { return logMemoizeStatsCounter < 4; }

using namespace Microsoft::MSR::CNTK;
using namespace std;

#define BarrierOp NoOp // for now, we use Alias() (=NoOp) to denote a Barrier(). Should become an op in its own right.

#pragma warning (disable: 4456) // until I fixed the shadowing

#define let const auto
#define fail_if(cond, err) (!!(cond) ? (LogicError(__FUNCTION__ ": " err),0) : 0)
#define Break fprintf(stderr, "") // use this inside a conditional to be able to set a breakpoint in Release code

namespace CNTK
{
    // Auto-batching merges operations with matching operations for saving GPU launches and boosting GPU occupancy.
    // Example: We have a batch of 2, and for each batch item, we compute c = a + b:
    // 
    //   c1 = a1 + b1
    //   c2 = a2 + b2
    // 
    // Assuming matching dimensions this is batched as
    // 
    //   c1_c2 = [a1, a2] + [b1, b2]
    //   c1 = c1_c2[0]
    //   c2 = c1_c2[1]
    // 
    // where an and bn, respectively, are batched along a new batch dimension. Alternatively, they could be batched like this:
    // 
    //   c1_c2 = [a1  + [b1
    //            a2]    b2]
    //   c1 = c1_c2[0:dim]
    //   c2 = c1_c2[dim:2*dim]
    // 
    // We call the first "batching" and the second "stacking."
    // a and b are the "args" of the + "op", and a1 and a2 are (arg batch) items of the first arg of the (batched) op.
    // 
    // The batching/stacking operation involves a memory copy to collate all arg batch items into a memory-consecutive
    // dense tensor. (This will generate a "gather" op under the hood, and the back-prop is handled by a "scatter" op.)
    // (In the future, this may be virtualized.)
    // As an optimization, the gather op is skipped if all items are already consecutive in memory.
    // This is often the case when they themselves are the result of a batched computation.
    // 
    // The batching/stacking axis is always last (slowest-changing) axis of the batched data; either a new one (batching) or the last one (stacking).
    // 
    // If all args to be batched are identical (all items are the same object), then, if possible, the item
    // is not copied, but instead virtually duplicated by means of broadcasting. E.g. if "b1 is b2" (in Python parlance),
    // then we can avoid one copy operation and write:
    // 
    //   c1_c2 = [a1, a2] + b1
    // 
    // where b1 will automatically be broadcast into the second dimension by the "+" operation.
    // This happens, for example, when adding a bias to each step of a sequence.
    // All presently supported operators support broadcasting.
    // 
    // If all of the args turn out to have this property, then the operation is computed unbatched but only once, and the result is shared:
    // 
    //   c1_c2 = a1 + b1
    //   c1 = c2 = c1_c2
    // 
    // All presently supported operators are batchable. They fall into one of the following classes:
    //  - elementwise:
    //     - all inputs share the same axis space
    //       Note: Any input/ouput of lower rank is interpreted as having its rank padded with singleton dimensions (dim=1).
    //     - computed output shares the same axis space (but may be Reshaped to form the final result)
    //     - hence, they are fully batchable/stackable
    //       Note: This is also true for Reductions, which are elementwise with the reduction dimension having dimension 1.
    //     - a special case are see-through ops (NoOp, Barrier, Reshape, Slice)
    //       These are short-circuited prior to auto-batched execution.
    //  - matrix-product class (Times(), TransposeTimes(), and Convolution):
    //     - the first arg is a weight tensor. It cannot have a batch axis.
    //       (One day we may support batch GEMM; and Convolution may already support it with non-shared weights.)
    //     - they are fully batchable/stackable
    //     - the batch axis may move
    //  - Block invocations:
    //     - block inputs must behave like elementwise, i.e. all inputs share the same axis space
    //       Note: Blocks may contain matrix-product class ops, but only with Parameters baked into the static graph
    //
    // Hence, the general conditions for batching/stacking are:
    //  - defintions:
    //     - non-batchable input := any of the following:
    //        - weight tensor of a matrix-product class operation. Note: It does not share the same axis space as the other input.
    //        - learnable parameters of BatchNorm. Note: These do share the same axis space with the other inputs, but are reduced.
    //     - batchable input := all other inputs
    //  - conditions for batching:
    //     - batch axis := maximum over ranks of all batchable inputs (Note that a Reshape's or matrix product's output may have a shifted batch axis)
    //     - all batchable inputs must have matching dimensions
    //     - all non-batchable inputs must have the same object identity
    //  - conditions for stacking:
    //     - batch axis := same as for batching, minus 1
    //     - batchable inputs: same as for batching, except dim[batch axis], if present, does not need to match
    //     - non-batchable inputs: same as for batching
    //     - batch axis must exist (no batchable input is a scalar)
    //     - the batch axis must not be reduced over
    //  - additional conditions for special cases:
    //     - sparse inputs must have batch axis = 1, due to current limitation of the underlying library
    //     - BatchNormalization:
    //        - all instances with the same bnId must be available
    //        - batch axis is specified by user. Must be either the last axis (stacking) or larger (batching).
    //     - OptimizedRNNStack (presently not supported):
    //        - the batch is not a tensor (non-uniform batch dimension across time steps)
    //
    // Algorithm for testing if two ops are batchable:
    //  - we use stacking, unless the following forces us to use batching:
    //     - no batchable input is a scalar
    //     - there is no reduction along the potential stacked batch axis (=max over ranks of batchable inputs - 1)
    //     - there is no batchable input that is a sparse vector (i.e., of rank 1). (Math library cannot stack sparse along axis 0.)
    //  - determine input and output batch axes, as given above depending on stacking/batching decision
    //  - for any sparse input, the batch axis must be 1, no matter we decided for stacking or batching
    //    Note: The information up to this point can be precomputed once and stored with the PrimitiveFunction.
    //  - operation, all attributes, and all inputs' DataType and IsSparse must match
    //  - batch axes must match
    //  - all batchable inputs must have matching dimensions up to the batch axis
    //  - special cases:
    //     - BatchNormalization also requires batch size == #instances
    //
    // Note: One can also imagine more elaborate stacking in different dimensions, and even in multiple dimensions
    // (like current CNTK MBLayout).
    //
    // Side note: The matrix-product class can be thought of as Reshapes combined with an ElementTimes with reduction:
    //  - A : [I x J]               -> [I x J x 1]
    //  - B : [J x T]               -> [1 x J x T]
    //  - C = A x B : [I x T] = ReduceSum(A.AsShape(I,J,1) * B.AsShape(1,J,T), Axis(1)).AsShape(I,T)
    // TODO: Actually try this!
    // A more complex formula of the same kind exists for convolution, when disregarding padding.
    // 
    // The conditions are tested in Schedule() and ExecuteBatchedOpAndSchedule().
    // The batching deciosion is made after output and Parameter shapes have been fully determined, as that
    // happens during graph building, before the computation begins. The already determined
    // shapes can directly drive reduction and reshape operations.
    // 
    // The following will go through every single operation and specify how it is batched.
    // This is designed in a way that even multiple batching, with multiple added batch axes, will behave correctly.
    // TODO: This is a TODO list for now with intent to be implemented this way.
    // 
    // The op-specific conditions are specified in the following initializer.
    // 
    // TODO: implement these V1 nodes somehow via TensorView
    //  - Convolution, Pooling, Unpooling
    //  - RandomDistribution  // for dropout
    //  - OptimizedRNNStack   // for MT
    //  - Hardmax. Can we have a OneHot-like op in TensorView?

    enum class OpSpecificConditionKind  :  size_t // the meanings of these are specified below
    {
        UnaryElementWise, Reducing, NoOp, Reshape, BinaryElementWise, TernaryElementWise, Pooling,
        Slice, Splice, Transpose,
        MatrixProduct, Convolution,
        Barrier, BatchNormalization, OptimizedRNNStack, RandomDistribution,
        BasicBlockInvocation,
        NotSupportedDynamicAxis, NotSupportedTempMem, NotSupportedStaticGraph, ToDo, Undefined
    };
    static map<OpSpecificConditionKind, vector<PrimitiveOpType>> opSpecificConditionInitializer
    {
        // unary element-wise TensorView operations; reduction TensorView operations
        // -------------------------------------------------------------------------
        //
        { OpSpecificConditionKind::UnaryElementWise, {
            PrimitiveOpType::Negate, PrimitiveOpType::Sigmoid, PrimitiveOpType::Tanh,
            PrimitiveOpType::ReLU, PrimitiveOpType::Exp, PrimitiveOpType::Log, PrimitiveOpType::Sqrt,
            PrimitiveOpType::Floor, PrimitiveOpType::Abs, PrimitiveOpType::Reciprocal,
            PrimitiveOpType::Sin, PrimitiveOpType::Cos, PrimitiveOpType::ELU,
            PrimitiveOpType::StableSigmoid
        }},
        { OpSpecificConditionKind::Reducing, {
            PrimitiveOpType::ReduceElements /*(=ReduceSum, ReduceLogSum, etc.)*/
        }},
        // 
        // Conditions for batching:
        //   All arg arguments must completely match in dimensions. The key for comparison is the full shape.
        // 
        // Conditions for stacking:
        //   All args must match except for their last dimension. The key for comparison is the shape less the last dimension.
        //   Reduction dimensions may not reduce along the last dimension (otherwise fall back to batching).
        // 
        // If all args are the same object, the op is executed only once, and the result is shared across the batch.
        // 
        // The same code works for both element-wise and reduction ops, as follows:
        //  - the output shape for the unbatched operations is known (and they are all identical across the batch)
        //  - the batched output shape is equal to the unbatched one plus the batch axis (unless we are stacking)
        //  - hence, reductions will be done as expected by TensorView.
        // 
        // The forward and backward code for reductions does not consider the axis properties, but instead relies
        // on the already determined output shape. This way it works after batching (whereas if it were to consult
        // the axis and found AllStaticAxes, it would reduce wrongly).
        // TODO: check all compute/backward code that it does not check/determines the shapes (but possibly checks them
        // where it does not conflict with batching).

        // No-ops
        // ------
        // 
        { OpSpecificConditionKind::NoOp, {
            /// PrimitiveOpType::NoOp, // NoOp currently used for Barrier
            PrimitiveOpType::Pass, PrimitiveOpType::StopGradient
        }},
        // 
        // No-ops are see-through in the code. They are short-circuited for auto-batching and therefore never be batched.

        // Reshape()
        // ---------
        // 
        { OpSpecificConditionKind::Reshape, {
            PrimitiveOpType::Reshape
        }},
        // 
        // Reshape is not a distinct operation, but short-circuited like a see-through op.
        // Reshape PrimitiveFunctions are skipped in the graph. Because all output shapes
        // have been inferred on the complete graph including Reshape, skipped Reshapes thus
        // themselves as operations producing results whose shapes do not match the shape
        // of the reshaped output. NDArrayView::AsShape() operations are inserted on the fly
        // where dimensions did not match.
        // TODO: In two cases in auto-batching, new explicit Reshape PrimitiveFunctions are
        //       presently created. That is not necessary, and even inefficient in backprop.
        //       These should be short-circuited as well.

        // Binary/ternary element-wise ops
        // -------------------------------
        // 
        { OpSpecificConditionKind::BinaryElementWise, {
            PrimitiveOpType::Plus, PrimitiveOpType::Minus, PrimitiveOpType::ElementTimes, PrimitiveOpType::LogPlus, PrimitiveOpType::Pow,
            PrimitiveOpType::Equal, PrimitiveOpType::NotEqual, PrimitiveOpType::Less,
            PrimitiveOpType::LessEqual, PrimitiveOpType::Greater, PrimitiveOpType::GreaterEqual
        }},
        { OpSpecificConditionKind::TernaryElementWise, {
            PrimitiveOpType::Clip, PrimitiveOpType::Select
        }},
        // 
        // Conditions:
        //   Same conditions as for unary element-wise ops must hold for all inputs.
        // 
        // When batching, all input shapes are, if needed, padded with singleton dimensions to have the same
        // rank, so that the batching/stacking dimension is at the same position across all inputs
        // and the output. Likewise, any such added singleton dimensions in the output are, if present,
        // removed after the op.

        // Matrix-product class
        // --------------------
        // 
        { OpSpecificConditionKind::MatrixProduct, {
            PrimitiveOpType::Times, PrimitiveOpType::TransposeTimes
        }},
        // 
        // Conditions (batching):
        //   The first input (the matrix) must be the same for all items.
        //   The second one has the same conditions as for unary element-wise ops.
        //   If expressable as ReduceSum(ElementTimes(x,y)) then matrix does not need to be the same.
        // 
        // Conditions (stacking):
        //   First input (matrix) like batching.
        //   Second one must match in all but the last dimension.
        // 
        // When batching, rank parameters that count from the right (mapRank?--TODO: track it down) must be
        // adjusted or resolved into absolute ones (that counts from the left).
        //
        // If expressable as ReduceSum(ElementTimes(x,y)) then the batched version will use a special Matrix operation.
        // ... TODO: Get the corresponding condition for the backprop path.
        // 
        // TODO: We should investigate stride-batched SGEMM. Then we can batch everything.

        // Convolution/pooling
        // -------------------
        //
        { OpSpecificConditionKind::Convolution, { // TODO: make this part of matrix-product class
            PrimitiveOpType::Convolution // includes ConvolutionTranspose()
        }},
        { OpSpecificConditionKind::Pooling, { // TODO: make this part of unary elementwise ops
            PrimitiveOpType::Pooling, PrimitiveOpType::Unpooling
        }},
        //
        // Conditions (binary):
        //   Same as for Matrix product.
        // 
        // Conditions (unary):
        //   Same as for unary element-wise ops.
        // 
        // This requires additional TensorView operation. Besides that, it can share code with matrix and unary.
        // The batch dimension just goes into the N dimension as usual.
        // Global pooling cannot be parameterized to exclude the batching/stacking axis.
        // So instead, Dynamite does not use the global flag (engine must throw an error); instead, it reshapes the value to a vector first.
        // TODO: double check what global pooling does. It goes to a scalar, right?
        // TODO: Are there additional arguments in the dict that are relative to the end? mapRank? That must be resolved first, or updated when batching.
        // 
        // TODO: How to hold the auto-tune state? Cache it indexed by shape except batch dim? Can be problematic for truly dynamic image stuff --check with Cha

        // Slice()
        // -------
        // 
        { OpSpecificConditionKind::Slice, {
            PrimitiveOpType::Slice
        }},
        // 
        // Conditions:
        //   false
        //
        // Slice() is an actual view, not a copy. Hence, it is not batched.
        // The result of Slice() is smaller than its input, so there is no point in gathering
        // the unsliced inputs, only to put a view on it.
        //
        // If we batch-Slice() an already batched input, this is only slightly suboptimal. Because slicing a batched
        // input always makes the result non-contiguous, it would have to be followed by a copy operation.
        // By not slicing batched inputs, instead we'd slice each item separately (which is merely a view)
        // and then gather them together. In both cases, the same number of bytes are copied.
        // However, in the former case, the copy is a rectangle, and it involves less overhead.
        //
        // Note: It is currently required that user-specified slices must always be memory-contiguous.
        // If not, then we may allocate temp memory in Memoize() in an unoptimized fashion, and
        // GatherBatch() will fail.

        // Splice()
        // --------
        // 
        { OpSpecificConditionKind::Splice, {
            PrimitiveOpType::Splice
        }},
        // 
        // Conditions:
        //   Same as for elementwise ops.
        //
        // A batched Splice first batches all operands, and then splices the batched operands.
        // The resulting batched Splice op is a strided one. Thus, it is not executed by the
        // one-kernel Gather operation (there are some special cases where we could combine
        // the batching and splicing into a single op). Instead, one strided copy-kernel (opCopy) per batched input
        // is made. Most of the time, we only splice a few items, such as 2; for those, this is certainly fine.
        // (If the original user-issued Splice consists of arguments of identical shapes, then all
        // involved Splice operations could be merged into a single one, followed by a reshape.
        // This is not presently done.)

        // TransposeAxes()
        // ---------------
        // 
        { OpSpecificConditionKind::Transpose, {
            PrimitiveOpType::TransposeAxes
        }},
        // 
        // Conditions (batching)):
        //   Same as for elementwise ops.
        // 
        // Conditions (stacking):
        //   Same as for elementwise ops.
        //   Must not transpose in stacking dimension.
        // 
        // This is like a unary element-wise op, except that the additional attributes dictionary
        // will be consulted for the axes to swap. The conditions guaranteed that the arguments
        // remain valid after batching/stacking.

        // Barrier()
        // ---------
        // 
        { OpSpecificConditionKind::Barrier, {
            PrimitiveOpType::BarrierOp
        }},
        //
        // Condition:
        //   true
        // 
        // A barrier means that all other ops available for batching get executed first, as long as there
        // is a barrier pending.
        // BUGBUG: We reuse the NoOp opcode, but Barrier has an additional attribute. This is not accounted for in the OpSpecificConditionKind.

        // BatchNormalization()
        // --------------------
        // 
        { OpSpecificConditionKind::BatchNormalization, {
            PrimitiveOpType::BatchNormalization
        }},
        // 
        // Condition (batching/stacking):
        //   A unique op identifier must match. (Dimensions are then required to match fully as well, otherwise it's an error.)
        // 
        // The items that belong together are identifed by a unique identifier that is owned by the batch-normalization Layer.
        // Each new layer gets a new id.
        //   
        // Unlike other ops, this operation is *required* to have all batch items available, similar
        // to Barrier(). If for some reason an operation with the same id gets broken into multiple
        // invocations within the same Batch::Map() call, an error is thrown (i.e. the full unique
        // id is (user id, Batch::Map invocation id)).
        // Note: This makes the threaded parallelization of batch ops mandatory.

        // OptimizedRNNStack()
        // -------------------
        // 
        { OpSpecificConditionKind::OptimizedRNNStack, {
            PrimitiveOpType::OptimizedRNNStack
        }},
        // 
        // Condition:
        //   Dimensions must match except for the last dimension (=the time axis).
        // 
        // Batching is done according to NVidia's data format.
        // This is presently not implemented.

        // RandomDistribution()
        // --------------------
        // 
        { OpSpecificConditionKind::RandomDistribution,{
            PrimitiveOpType::RandomDistribution // covers also the -Like version
        }},
        //
        // Condition (stacking):
        //   Initially: general.
        //   Future: true.
        //
        // This op is either nullary or unary (-Like() version).
        // 
        // FutureL This can always be stacked, independent of shapes, because all elements are independent.
        // The stacked operation must be followed by a reshape. TODO: That's why we won't stack all, since reshapes block re-batching.
        //
        // Each invocation gives rise to a new set of random numbers. This is used to implement Dropout.
        // This operation returns a Constant upon each invocation.
        // 
        // To share random numbers, e.g. across all time steps of a sequence for Dropout, users must manually compute it once.
        // Random numbers cannot be shared across batch items unless users explicitly compute them outside of Batch::Map().
        // The random numbers are lazily computed upon first Value() call, to allow for batching.
        // 
        // This is presently not implemented.
        // TODO: Does this need to hold on to a RNG state? How to do that?

        //
        // Invoke(isBasicBlock=true)
        // -------------------------
        //
        { OpSpecificConditionKind::BasicBlockInvocation, {
            PrimitiveOpType::Block
        }},
        //
        // Condition (batching):
        //   Like N-nary elementwise ops.
        //
        // Condition (stacking):
        //   False. To allow this, we'd need to analyze the content of the composite to line up the axes.
        //
        // Presently, Invoke(composite, isBasicBlock=true) is only allowed for composites that contain only
        // elementwise operations. That is, e.g. no Times or Convolution. That is because the underlying layer
        // does not support batch GEMM/batch Convolution at present. --NO: This has been relaxed already. TODO: Update this text.
        //
        // The inputs of the composite just get the batch axis appended. During computation, the composite is interpreted.
        // In each step, the batch axis of all inputs and the output are aligned as needed.

        // not supported: primitives that involve dynamic axes
        // ---------------------------------------------------
        // 
        { OpSpecificConditionKind::NotSupportedDynamicAxis, {
            PrimitiveOpType::Gather, PrimitiveOpType::Where, PrimitiveOpType::PackedIndex, PrimitiveOpType::GatherPacked, PrimitiveOpType::ScatterPacked,
            PrimitiveOpType::PastValue, PrimitiveOpType::FutureValue,
            PrimitiveOpType::ToSequence, PrimitiveOpType::ToSequenceLike, PrimitiveOpType::UnpackSequence, PrimitiveOpType::ReconcileDynamicAxis,
            PrimitiveOpType::SumAll
        }},
        // 
        // These operations are specifically meant for objects with dynamic axes.
        // Dynamite Variables cannot have dynamic axes. Dynamite does not support
        // these ops, or implements them with explicit loop unrolling.

        // not supported: primitives that require temporary memory: not supported as primitives
        // -------------------------------------------------------
        // 
        { OpSpecificConditionKind::NotSupportedTempMem, {
            PrimitiveOpType::Softmax, PrimitiveOpType::LogSoftmax, PrimitiveOpType::Hardmax,
            PrimitiveOpType::Dropout,
            PrimitiveOpType::CrossEntropyWithSoftmax, PrimitiveOpType::ClassificationError, PrimitiveOpType::Logistic,
            PrimitiveOpType::SquaredError, PrimitiveOpType::CosDistance
        }},
        // 
        // These operations exist as CNTK V2 PrimitiveOps, but cannot easily be realized in Dynamite since they
        // require temporary memory. For example, Softmax requires a temporary buffer, and Dropout requires to store the random mask.
        // Dynamite implements these operations on a higher level by explicit calls to other primitives
        // (e.g. LogSoftmax(x) = x - ReduceLogSum(x))
        // 
        // TODO: Dropout(x) = x * (BernoulliRandomLike(x, mean=p) * 1/(1-prob))
        // TODO: We critically need a way to express Hardmax. Is it ArgMax>>OneHot?
        // 
        // not supported: primitives that are specific to static graphs
        // ------------------------------------------------------------
        // 
        { OpSpecificConditionKind::NotSupportedStaticGraph, {
            PrimitiveOpType::Combine, PrimitiveOpType::Assign
        }},
        // 
        // These are specific to static graphs, and therefore do not apply to Dynamite.
        // As a consequence, all Dynamite operations have only one output. Code leverages that for now.
        // TODO: Is there value in having primitives with >1 output? E.g. an LSTM node?
        // 
        // Note: Block may be supported in the future, to inline static graphs into the dynamic graph.
        // This would allow to move some overhead for standard structures (e.g. GRU) from the user-side
        // code to Clone(), where we can optimize.
        // 
        // TODO: Maybe Block already works (but inefficiently); to be tested.
        // 
        // unclassified so far
        // -------------------
        // 
        { OpSpecificConditionKind::ToDo, {
            PrimitiveOpType::ROIPooling,   // need to find out how it works precisely--is it just like pooling?
            PrimitiveOpType::LambdaRank,
            PrimitiveOpType::NDCG,
            PrimitiveOpType::EditDistanceError,
            PrimitiveOpType::LabelsToGraph,
            PrimitiveOpType::ForwardBackward,
            PrimitiveOpType::CosDistanceWithNegativeSamples,
            PrimitiveOpType::OneHot,  // find out whether this is valuable, e.g. for Hardmax
            PrimitiveOpType::RandomSample,
            PrimitiveOpType::RandomSampleInclusionFrequency
        }}
    };

    // how graphs work in CNTK V2:
    //  - nodes := PrimitiveFunctions (incl. BlockFunction)
    //  - edges := Variables
    //  - net := CompositeFunction::m_allPrimitiveFunctions; duplicated for all refs to composites
    //  - output node: a node with additional ref to a net, created by calling Output() on a CompositeFunction
    // ownership:
    //  - nodes own edges: Functions hold shared_ptrs to m_inputs[] and m_outputs[]
    //  - edges do NOT own nodes
    //  - net owns full set of nodes
    //  - output node has a strong ref m_outputComposite to the CompositeFunction.
    //    This is injected when calling Output(), i.e. such an Output is really a different type w.r.t. ownership.

    // what we need to do:
    //  - operations that are computed in a batch:
    //     - Slice() ops (PrimitiveFunctions) batch the arguments
    //        - optimizes for the case that the arguments were already batched (they hold a m_lazySlice (pair<batchedOp, sliceIndex>))
    //     - a new PrimitiveFunction executes the batch immediately
    //     - the original operations get their m_value field filled with a slice into the batched op
    //        - this is done lazily; initially, they just remember a pair<batchedOp, sliceIndex>, as m_lazySlice
    //     - the batchedOp in the pair is also kept for batched backprop; it is a strong ref from Variable (cannot be in a cycle)
    //  - hence, we create N+1 new nodes:
    //     - the new batched op
    //     - Splice() for each of the N inputs
    // 'free' ops are always batched together and get executed first

using SliceRange = Internal::AutoBatchRedirection::SliceRange;

// ===========================================================================
// helper classes
// ===========================================================================

// ---------------------------------------------------------------------------
// cuda stats -- helper for profiling execution cost (CPU, GPU)
// ---------------------------------------------------------------------------

#ifdef DETAILED_STATS
class PCTimer // roll our own; high_resolution_timer is reportedly not high-resolution (0.1 us)
{
    LARGE_INTEGER freq, start;
    double total;
public:
    PCTimer() { if (!QueryPerformanceFrequency(&freq)) RuntimeError("PCTimer: QueryPerformanceFrequency failure"); } // count ticks per second
    void Start() { QueryPerformanceCounter(&start); }
    double Stop() // each read gives time elapsed since start, in seconds
    {
        LARGE_INTEGER end;
        QueryPerformanceCounter(&end);
        let elapsed = (end.QuadPart - start.QuadPart) / (double)freq.QuadPart;
        total += elapsed;
        return elapsed;
    }
    double Total() const { return total; }
};
struct CudaStats
{
    PrimitiveOpType op = PrimitiveOpType::UnknownOP;
    const wchar_t* opLabel = nullptr;
    size_t category = 0; // 1 = sparse, 2 = not an op
    size_t numInvocations = 0;
    size_t totalElements = 0;  // sum of all output elements, for a rough indication of utilization
    PCTimer timerLaunch;
    double cudaElapsed = 0;
};
vector<CudaStats> cudaStats;
// call this at start
// The interface is quite horrible w.r.t. device. We just need a flag.
CudaStats* BeginCudaStats(PrimitiveOpType op, const wchar_t* opLabel, size_t category = 0, size_t totalElements = 1, const DeviceDescriptor& device = DeviceDescriptor::CPUDevice())
{
    if (!ShouldLogMemoizeStats())
        return nullptr;
    cudaStats.resize(4 * (size_t)PrimitiveOpType::UnknownOP);
    auto* cudaStatsPtr = &cudaStats[(size_t)op * 4 + category];
    cudaStatsPtr->op = op; // (really only needed the first time)
    cudaStatsPtr->opLabel = opLabel;
    if (!cudaStatsPtr->opLabel)
        cudaStatsPtr->opLabel = PrimitiveOpTypeName(op).c_str();
    cudaStatsPtr->category = category;
    cudaStatsPtr->numInvocations++;
    cudaStatsPtr->totalElements += totalElements;
    if (logMemoizeStatsCounter & 1)
        NDArrayView::Sync(device); // reset CUDA timer
    cudaStatsPtr->timerLaunch.Start();
    return cudaStatsPtr;
}
// call this at end
void EndCudaStats(CudaStats* cudaStatsPtr, const DeviceDescriptor& device = DeviceDescriptor::CPUDevice())
{
    if (cudaStatsPtr)
    {
        cudaStatsPtr->timerLaunch.Stop();
        if (logMemoizeStatsCounter & 1)
            cudaStatsPtr->cudaElapsed += NDArrayView::Sync(device);
    }
}
// guard class to make it easy to use
struct CudaStatsGuard
{
    CudaStats* cudaStatsPtr;
    template <typename ...CtorArgTypes>
    CudaStatsGuard(CtorArgTypes&& ...ctorArgs) : cudaStatsPtr(BeginCudaStats(std::forward<CtorArgTypes>(ctorArgs)...)) { }
    void Stop() { EndCudaStats(cudaStatsPtr); cudaStatsPtr = nullptr; } // early stop; destructor does nothing
    ~CudaStatsGuard() { EndCudaStats(cudaStatsPtr); }
};
// and this at the end of when you want to dump the log
void ShowCudaStats()
{
    if (ShouldLogMemoizeStats())
    {
        for (size_t category = 0; category < 4; category++) // show non-sparse visually separated from sparse
        {
            double totalLaunch = 0;
            double totalExec = 0;
            wstring prefix = category == 1 ? L"sparse " : category == 2 ? L"free " : category == 3 ? L"batch " : L"";
            fprintf(stderr, "\n");
            for (let& s : cudaStats) if (s.category == category && s.numInvocations > 0)
            {
                fprintf(stderr, "-> %30S: %7.1f ms + %7.1f ms = (%9.6f + %9.6f) ms/call * %7d calls, %9.1f avsize/call -> %5.1f MBf\n", (prefix + s.opLabel).c_str(),
                        1000.0 * s.timerLaunch.Total(), 1000.0 * s.cudaElapsed,
                        1000.0 * s.timerLaunch.Total() / (double)s.numInvocations, 1000.0 * s.cudaElapsed / (double)s.numInvocations,
                        (int)s.numInvocations,
                        s.totalElements / (double)s.numInvocations,
                        s.totalElements * sizeof(float)/ (1024.*1024.));
                totalLaunch += s.timerLaunch.Total();
                totalExec += s.cudaElapsed;
            }
            if (prefix != L"batch ") // (batch ones are nested, so sum is meaningless)
                fprintf(stderr, "=> %Stotal launch + exec time: %.4f s + %.4f s\n", prefix.c_str(), totalLaunch, totalExec);
            fflush(stderr);
        }
        cudaStats.clear();
    }
    // control how often this is active
    logMemoizeStatsCounter++;
    if (logMemoizeStatsCounter == logMemoizeStatsPeriod)
        logMemoizeStatsCounter = 0;
}
// call this at start to eliminate potential carry-over from the previous minibatch
void ResetCudaStats()
{
    cudaStats.clear();
}
#else // no DETAILED_STATS. Makes very little runtime difference.
typedef void CudaStats;
CudaStats* BeginCudaStats(PrimitiveOpType op, const wchar_t* opLabel, size_t category = 0, size_t totalElements = 1, const DeviceDescriptor& device = DeviceDescriptor::CPUDevice()) { return nullptr; }
void EndCudaStats(CudaStats* cudaStatsPtr, const DeviceDescriptor& device = DeviceDescriptor::CPUDevice()) { }
struct CudaStatsGuard
{
    template <typename ...CtorArgTypes>
    CudaStatsGuard(CtorArgTypes&& ...ctorArgs) { }
    ~CudaStatsGuard() { }
};
// and this at the end of when you want to dump the log
void ShowCudaStats() { }
void ResetCudaStats() { }
#endif // DETAILED_STATS

// ---------------------------------------------------------------------------
// OpSpecificConditionKindTable -- singleton lookup table for the OSC codes
// ---------------------------------------------------------------------------

static const class OpSpecificConditionKindTable
{
    vector<OpSpecificConditionKind> oscTable; // [PrimitiveOpType]
public:
    OpSpecificConditionKindTable(const map<OpSpecificConditionKind, vector<PrimitiveOpType>>& init = opSpecificConditionInitializer) :
        oscTable((size_t)PrimitiveOpType::UnknownOP, OpSpecificConditionKind::Undefined)
    {
        // transfer the initializer table into a lookup table
        // Check for dups and missing entries.
        for (let& kv : init)
            for (let& op : kv.second)
                if (oscTable[(size_t)op] != OpSpecificConditionKind::Undefined)
                    LogicError("OpSpecificConditionKindTable: Duplicate entry for op %d", (int)op);
                else
                    oscTable[(size_t)op] = kv.first;
        for (size_t op = 0; op < oscTable.size(); op++)
            if (oscTable[op] == OpSpecificConditionKind::Undefined)
                LogicError("OpSpecificConditionKindTable: Must be updated to cover op %d", (int)op);
    }
    OpSpecificConditionKind operator[](PrimitiveOpType op) const { return oscTable[(size_t)op]; }
} g_oscTable;

// ---------------------------------------------------------------------------
// NDArrayViewArena -- helper class that implements efficient arena allocation for NDArrayView objects
// ---------------------------------------------------------------------------

// helper that is needed at a few places
// Reshape an NDArrayViewPtr in-place (replace by a new one) if its shape does not match.
static void ReplaceWithReshapedViewIfNeeded(NDArrayViewPtr& view, const NDShape& shape)
{
    if (view->Shape() != shape)
        view = view->AsShape(shape);
}

class NDArrayViewArena
{
    // allocate a new tensor in a large arena
    static NDArrayViewPtr s_currentArena;
    static size_t s_currentArenaUsed;
    static vector<unique_ptr<NDArrayView>> s_recycledArenas;
    static const size_t ARENASIZE = 64000000; // we allocate in this chunk size
public:
    // allocate an NDArrayView of a given shape, data type, and device
    // The returned memory region is a slice into a much larger NDArrayView; therefore,
    // this operation short-circuits CUDA and is very fast.
    // Sparse objects cannot be arena-allocated. Which is fine, since they are inputs or
    // gradients (of embeddings) that can be kept around across minibatches, and thus not part of batched computation.
    NDArrayViewPtr NewNDArrayView(const NDShape& shape, const DataType& dataType, StorageFormat format, const DeviceDescriptor& device)
    {
        let numElements = shape.TotalSize(/*check=*/false);
        // if too large, or sparse, then plain alloc
        if (numElements > ARENASIZE || format != StorageFormat::Dense)
        {
            //fprintf(stderr, "allocating %d elements of format %d, outside arena\n", (int)shape.TotalSize(/*check=*/false), (int)format);
            //CudaStatsGuard cudaStatsguard(PrimitiveOpType::Splice, L"single NewNDArrayView", 3, numElements);
            return make_shared<NDArrayView>(dataType, format, shape, device);
        }
        // If arena not large enough then waste its remainder and just allocate a fresh one.
        // This abandons the current m_arena. This will not cause a memory leak, however:
        // Since the slices into it that were returned before all hold a ref-count to that arena,
        // it will be deallocated automatically as soon the last slice goes away.
        // The deleter, though, will intercept that and move it into the recycledArenas array.
        // Next time we need a new arena, we will look there first and recycle one.
        // If the data type is different, we drop the current arena. We can't presently mix data types, so this is ah-OK.
        if (!s_currentArena || numElements > (ARENASIZE - s_currentArenaUsed) ||
            dataType != s_currentArena->GetDataType() || device != s_currentArena->Device())
        {
            //CudaStatsGuard cudaStatsguard(PrimitiveOpType::FutureValue, L"new arena NewNDArrayView", 3, numElements);
#if 0       // BUGBUG: This does not work for unknown reasons. The code below works, but once I uncomment the SetValue(),
            //         it produces a loss of 0. Somehow the buffer is still being referenced, likely by a CUDA
            //         operation with output in a different arena. However, SetValue() calls cudaMemset(), which runs on the NULL stream,
            //         so it should wait until whichever op is there has completed.
            s_currentArena.reset(); // abandon current one. If no references, then this will go into recycledArenas right here
            NDArrayView* viewPtr;
            if (!s_recycledArenas.empty()) // recycle one
            {
                viewPtr = s_recycledArenas.back().release(); // take back ownership
                s_recycledArenas.pop_back(); // pop empty one
            //    //fprintf(stderr, "..recycling arena of %d elements\n", (int)viewPtr->Shape().TotalSize(/*check=*/false));
                //viewPtr->SetValue(0.0); // BUGBUG: without this, we get funky results
                delete viewPtr;
            //    viewPtr = new NDArrayView(dataType, StorageFormat::Dense, NDShape{ ARENASIZE }, device);
            }
            //else
            {
                //fprintf(stderr, "allocating arena of %d elements\n", (int)ARENASIZE);
                viewPtr = new NDArrayView(dataType, StorageFormat::Dense, NDShape{ ARENASIZE }, device);
            }
            s_currentArena = shared_ptr<NDArrayView>(viewPtr, [](NDArrayView* viewPtr)
            {
                //fprintf(stderr, "..retiring arena of %d elements\n", (int)viewPtr->Shape().TotalSize(/*check=*/false));
                //delete viewPtr;
                s_recycledArenas.push_back(unique_ptr<NDArrayView>(viewPtr)); // don't release; rather keep it around
            });
#else
            //fprintf(stderr, "..allocating arena of %d elements\n", (int)ARENASIZE);
            s_currentArena = make_shared<NDArrayView>(dataType, StorageFormat::Dense, NDShape{ ARENASIZE }, device);
#endif
            s_currentArenaUsed = 0;
        }
        auto region = s_currentArena->SliceViewAsShape(s_currentArenaUsed, s_currentArenaUsed + numElements, shape);
        s_currentArenaUsed += numElements;
        return region;
    }
};

/*static*/ NDArrayViewPtr NDArrayViewArena::s_currentArena;
/*static*/ size_t NDArrayViewArena::s_currentArenaUsed = 0;
/*static*/ vector<unique_ptr<NDArrayView>> NDArrayViewArena::s_recycledArenas;

// ---------------------------------------------------------------------------
// RuntimeStatistics -- helper class for collecting runtime statistics, for
// diagnostics and debugging purposes
// ---------------------------------------------------------------------------

struct RuntimeStatistics
{
    // forward
    size_t numOpNodes = 0;
    size_t numInlinedBlocks = 0;
    size_t numShortCircuitedNodes = 0;
    size_t numLeafNodes = 0;
    size_t numDoneGatherOps = 0;
    size_t numDoneFreeOps = 0;
    size_t numDoneOtherOps = 0;
    size_t numBatchedLaunches = 0;
    size_t numCommonSubexpressionsEliminated = 0; // equivalent Functions (same op, same inputs) discovered
    // backward
    size_t numBackpropsThrough = 0;  // number of functions (after batching) that we backprop through to at least oneinput
    size_t numBackpropsToInputs = 0; // total number of inputs we backprop to
    size_t numBackpropGathers = 0;
    size_t numBackpropScatters = 0;
    size_t numBackpropSetZeroes = 0; // number of explicit SetValue(0) calls to clear a gradient
    size_t numAvoidedBackpropToMatrix = 0;
    size_t numBatchedBackpropToViews = 0; // number of gradients that turned out to be views and were short-circuited
    size_t numBatchedBackpropToCalls = 0; // number of gradients actually computed
};

// ---------------------------------------------------------------------------
// NonOwningFunctionList, NonOwningFunctionListBuilder -- helper classes:
// linked list over PrimitiveFunction objects, using m_autoBatchState.m_link.
// This is used in auto-batching instead of, say, a std::vector<> or std::set<>
// for performance reasons. It also does not hold shared_ptrs, since those
// have significant runtime overhead. We don't need them, since the lists
// we build here operate on existing structures without allocating additional
// resources to be tracked.
// ---------------------------------------------------------------------------

class NonOwningFunctionList
{
protected:
    PrimitiveFunction* head; // first item or nullptr
    size_t count;            // note: count is only in here for diagnostics; only needed in builder
public:
    NonOwningFunctionList() { clear(); }
    NonOwningFunctionList(PrimitiveFunction* f) : head(f), count(1) { }
    void operator=(const NonOwningFunctionList& other)
    {
        head  = other.head;
        count = other.count;
    }
    NonOwningFunctionList(const NonOwningFunctionList& other)
    {
        *this = other;
    }
    NonOwningFunctionList(NonOwningFunctionList&& other)
    {
        *this = other;
        other.clear();
    }
    PrimitiveFunction* front() const { return head; }
    bool empty() const { return !head; }
    size_t size() const { return count; }
    void clear()
    {
        head = nullptr;
        count = 0;
    }
    class FunctionListIterator
    {
        PrimitiveFunction* iter;
    public:
        FunctionListIterator(PrimitiveFunction* f) : iter(f) { }
        PrimitiveFunction* operator->() const { return iter; }
        operator PrimitiveFunction*() const { return iter; }
        PrimitiveFunction& operator*() const { return *iter; } // TODO: This is weird, figure this out
        PrimitiveFunction* operator++() { iter = iter->m_autoBatchState.m_link; return iter; }
        bool operator!=(const FunctionListIterator& other) { return iter != other.iter; }
    };
    FunctionListIterator begin() const { return front(); }
    FunctionListIterator end()   const { return nullptr; }
};
class NonOwningFunctionListBuilder : public NonOwningFunctionList // over PrimitiveFunction, using m_autoBatchState.m_link
{
    PrimitiveFunction* tail; // note: value undefined when list empty
public:
    NonOwningFunctionListBuilder() : NonOwningFunctionList() { }
    NonOwningFunctionListBuilder(PrimitiveFunction* f) : NonOwningFunctionList(f), tail(f) { f->m_autoBatchState.m_link = nullptr; }
    void push_back(PrimitiveFunction* f)
    {
        if (!head)
            head = f;
        else
            tail->m_autoBatchState.m_link = f;
        tail = f;
        count++;
        f->m_autoBatchState.m_link = nullptr;
    }
};

// ---------------------------------------------------------------------------
// VisitorTag -- helper for graph traversal
//  - call VisitorTag::Begin()
//  - in traversal: if (VisitorTag::Visited(node.m_visitedTag)) return;
// If you want this to nest, then remember the return value from Begin() and call End() with it.
// ---------------------------------------------------------------------------

class VisitorTag
{
    static size_t s_nextVisitTag; // unique id for a single non-nested visiting process
    size_t m_visitTag;
public:
    size_t Begin() // call this at start
    {
        let prevVisitTag = m_visitTag;
        // TODO: to make this thread-safe, use atomic increment
        m_visitTag = s_nextVisitTag++;
        return prevVisitTag;
    }
    bool Visited(size_t& tag) const // first time this is applied to 'tag', it will return false; and true otherwise
    {
        if (tag == m_visitTag)
            return true;
        tag = m_visitTag;
        return false;
    }
    // for nesting
    void End(size_t tag) { m_visitTag = tag; } // restore state before Begin() that returned 'tag'
};
/*static*/ size_t VisitorTag::s_nextVisitTag = 1;


// ===========================================================================
// DynamicProfiler -- helper for profiling dynamic batching of functions
// ===========================================================================

class DynamicProfiler : public enable_shared_from_this<DynamicProfiler>
{
public:
    DynamicProfiler(int verbosity, const wstring& name) :
        m_verbosity(verbosity), m_name(name)
    {
    }

    int Verbosity() const { return m_verbosity; }
    const wchar_t* Name() const { return m_name.c_str(); }

private:
    // config
    const int m_verbosity;
    const wstring m_name;
    // state
};

// TODO: make this thread local??  vvv
static shared_ptr<DynamicProfiler> m_currentProfiler; // current innermost active profiler, or empty

/*static*/ DynamicProfilerPtr Function::CreateDynamicProfiler(int verbosity, const wstring& name) { return MakeSharedObject<DynamicProfiler>(verbosity, name); }
// if 'outer' then enter a section that is profiled.
// When outside such a section, any call with 'outer'=false will have no effect.
// When inside, the effect is to change the name associated in the profiling output.
/*static*/ DynamicProfilerPtr Function::SetDynamicProfiler(const DynamicProfilerPtr& p, bool outer)
{
    auto prev = m_currentProfiler;
    if (outer || prev) // only set if verbosity>0 or there is already a profiler set (this is for inner lib functions)
        m_currentProfiler = p;
    return prev;
}
/*static*/ const DynamicProfilerPtr& PrimitiveFunction::CurrentDynamicProfiler() { return m_currentProfiler; }


// ===========================================================================
// AutoBatch -- autobatching happening inside here
// The auto-batching related functions are grouped inside a class, since they
// share quite a bit of state.
// ===========================================================================

class Variable::AutoBatch
{
    NDArrayViewArena m_arena; // helper to allocate NDArrayViews as slices into very large NDArrayView objects
    RuntimeStatistics m_stats;
    VisitorTag m_visitorTag; // helper for managing tree traversal (non-nested)

    // buffers e.g. for building NDArrayViewPtr vectors. Kept as class members to avoid repeated memory allocations.
    vector<NDArrayViewPtr>     m_inputValuesBuffer;
    vector<NDArrayViewPtr>     m_inputValuesBuffer2;
    vector<NDArrayViewPtr>     m_outputGradientsBuffer;
    vector<const NDArrayView*> m_inputValuesBufferRaw;
    vector<size_t>             m_dimsBuffer;
    template<class B> // B=vector<NDArrayViewPtr>
    B& BorrowBuffer(B& buffer, size_t batchSize)
    {
        if (buffer.capacity() < batchSize)
            buffer.reserve(batchSize * 2);
        buffer.resize(batchSize);
        return buffer;
    }

    // =======================================================================
    // forward-related functions
    // =======================================================================

    // predicate whether an op is only taking a view on its input
    // These are considered zero-cost, always batched whole-sale, and always done first.
    static bool IsViewOp(PrimitiveOpType op)
    {
        // if really needed, this can be done as a bit-test
        // TODO: The NoOps should never be tested here, right?
        //fail_if(IsAliasOp(op), "IsViewOp should never be asked about a no-op, should be short-circuited before");
        // ^^ Yes, they can be tested here while inlining of basic block during execution  --TODO: fix this, those should get short-circuited as well
        return
            op == PrimitiveOpType::StopGradient ||
            op == PrimitiveOpType::Pass         ||
            op == PrimitiveOpType::NoOp         ||
            op == PrimitiveOpType::BarrierOp    ||
            op == PrimitiveOpType::Reshape      ||
            op == PrimitiveOpType::Slice;
    }

    // predicate whether an op just passes through its input
    // This is used to decide whether we can short-circuit it in m_redirection.
    static bool IsAliasOp(PrimitiveOpType op)
    {
        // if really needed, this can be done as a bit-test
        return
            op == PrimitiveOpType::StopGradient ||
            op == PrimitiveOpType::Pass ||
            op == PrimitiveOpType::NoOp ||
            op == PrimitiveOpType::BarrierOp;
    }

    // predicate whether an op's gradient is a no-op (just copies the output gradient)
    // These are short-circuited in backprop.
    static bool IsGradientCopyingOp(PrimitiveOpType op, size_t inputIndex)
    {
        // if really needed, this can be done as a bit-test
        return
            op == PrimitiveOpType::StopGradient ||
            op == PrimitiveOpType::Pass         ||
            op == PrimitiveOpType::NoOp         ||
            op == PrimitiveOpType::BarrierOp    ||
            //op == PrimitiveOpType::Reshape      ||
            op == PrimitiveOpType::Plus         ||
            (op == PrimitiveOpType::Minus && inputIndex == 0);
            //(op == PrimitiveOpType::Plus && inputIndex == 0);
    }

    // helper to test whether a PrimitiveFunction is a barrier operation
    static bool IsBarrier(const PrimitiveFunction& f)
    {
        return f.m_op == PrimitiveOpType::BarrierOp && f.m_attributes.Size() > 0;
    }

    // predicate whether an op has an unbatchable first weigth parameter, like Times
    // TODO: Find a better name that eventually also includes Convolution.
    static bool IsMatrixProduct(PrimitiveOpType op)
    {
        return g_oscTable[op] == OpSpecificConditionKind::MatrixProduct;
    }

    // helper to check whether we should profile this function execution
    set<DynamicProfilerPtr> m_profilersUsed; // all used profilers will be registered for a given batched execution
    bool ShouldProfile(const PrimitiveFunction& f)
    {
#ifdef LOG_DETAILS
        f;
        let should = true;
#else
        let should = f.m_profiler && f.m_profiler->Verbosity() > 0;
#endif
        if (should)
            m_profilersUsed.insert(f.m_profiler); // this is slow but only used if any profiling output is printed, which is even slower
        return should;
    }

    // inlining of a composite 
    // This makes a deep copy of the graph below 'f', which must live in a composite owned by a BlockFunction; *not* the dynamic graph.
    // The actual PrimitiveFunction copy is made via the cloneFn. This way, this function can be used
    // for both the initial unbatched inlining (no basic block), as well for inlining basic blocks during execution.
    VisitorTag m_compositeVisitorTag; // helper for managing tree traversal through composite (not nested)
    template<class F>
    PrimitiveFunctionPtr RInlineComposite(PrimitiveFunction& f, const vector<Variable>& invocationArgs, const F& cloneFn) const
    {
        // if we already cloned this one then just return the clone
        if (m_compositeVisitorTag.Visited(f.m_autoBatchState.m_visitedTag))
        {
            let fInlined = f.m_autoBatchState.m_link; // we remembered the clone here
            if (!fInlined) // if we get here before this was filled in then we have discovered a cycle
                InvalidArgument("Invoke() cannot be used on composites with cycles.");
            return static_pointer_cast<PrimitiveFunction>(const_cast<PrimitiveFunction*>(fInlined)->shared_from_this()); // (shared_from_this() gives us a FunctionPtr, not PrimitiveFunctionPtr)
        }
f.m_autoBatchState.m_link = nullptr; // Bring into valid state so we can detect cycles. Gets overwritten once we are done cloning.

// clone f
// First clone the inputs;
let& inputs = f.m_inputs;
vector<Variable> inlinedInputs(inputs.size()); // (note: this is one malloc per cloned PrimitiveFunction. inlinedInputs then gets moved, not copied, into that function)
for (size_t i = 0; i < inputs.size(); i++)
{
    let& input = inputs[i];
    let& inputFields = GetInputFields(input);

    // --- case 0: a Variable that already has a value; that is, a Constant, Parameter, or any result of another Dynamite invocation
    if (inputFields.m_varKind == VariableKind::Constant || inputFields.m_varKind == VariableKind::Parameter || inputFields.m_value)
    {
        if (!inputFields.m_value)
        {
            input.Value(); // this is a Parameter for which we still need to run the initializer. This does that.
            fail_if(!inputFields.m_value, "Parameter/Constant has no Value()??");
        }
        inlinedInputs[i] = input;
    }
    // --- case 1: a Placeholder: substitute with invocation arg
    else if (inputFields.m_varKind == VariableKind::Placeholder)
    {
        let argIndex = inputFields.m_compositeArgumentIndex;
        fail_if(argIndex >= invocationArgs.size(), "no invocation arg lined was up with composite->Arguments[]??");
        inlinedInputs[i] = invocationArgs[argIndex];
    }
    // --- case 2: an Output Variable: clone the Function
    else
    {
        fail_if(inputFields.m_varKind != VariableKind::Output, "RInlineComposite encountered a non-output unexpectedly");
        let fInlinedPtr = RInlineComposite(*inputFields.Owner(), invocationArgs, cloneFn);
        inlinedInputs[i] = fInlinedPtr->m_outputs.front();
        inlinedInputs[i].m_acyclicOutputPrimitiveReference = fInlinedPtr;
        // ^^ the inlined input now holds a ref count to the function that generated it. This will move into and therefore be held by the newly inlined function below.
    }
}
// now create a new function and set up the outputs;
let fInlinedPtr = cloneFn(f, move(inlinedInputs));
// and finally remember where this function got redirected to.
f.m_autoBatchState.m_link = fInlinedPtr.get();
return fInlinedPtr;
    }

#define hashMultiplier ((size_t)1572869) // 4 1-bits. An arbitrarily picked prime from http://planetmath.org/goodhashtableprimes

#define Incorporate(newVal) (hash = hash * hashMultiplier + (size_t)(newVal))
#define IncorporateFieldsId(rFields) (Incorporate(((uintptr_t)&(rFields)) >> 6)) /* really want to divide by sizeof(VariableFields) */
    template<class VEC>
    static void IncorporateShape(size_t& hash, const VEC& shape)
    {
        Incorporate(shape.size());
        for (let dim : shape)
            Incorporate(dim);
    }
    // shapes and data types must match
    // BUGBUG: How about strides?
#define IncorporateType(fields) (IncorporateShape(hash, fields.m_shape.Dimensions()), Incorporate((fields.m_redirection.m_depthHint << 2) + (((size_t)fields.m_dataType) << 1) + (size_t)fields.m_isSparse))
    static size_t OpHash(const PrimitiveFunction& f)
    {
        size_t hash = f.m_autoBatchState.m_cachedOpHash;
        if (hash == SIZE_MAX)
        {
            let op = f.m_op;
            hash = 0;
            Incorporate(op);
            let& inputs = f.m_inputs;
            Incorporate(inputs.size());
            // special case BatchNormalization
            if (op == PrimitiveOpType::BatchNormalization)
            {
                // ids must match
                Incorporate(f.m_autoBatchState.m_batchNormId);
                fail_if(f.m_autoBatchState.m_batchNormId != f.m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>(), "m_batchNormId not initialized correctly??"); // TODO: remove once confirmed not flagging for a while
                // shape of first argument and object identities of all other arguments must match, otherwise it's an error
                IncorporateType(GetInputFields(inputs.front()));
                for (size_t i = 1; i < 6; i++)
                    IncorporateFieldsId(GetInputFields(inputs[i]));
            }
            else if (IsMatrixProduct(op)) // Times(): first arg must be the same object
            {
                IncorporateFieldsId(GetInputFields(inputs.front()));
                IncorporateType(GetInputFields(inputs.back()));
            }
            else // general case: all input dimensions must match (with exception of a few special cases)
            {
                for (let& input : inputs)
                    IncorporateType(GetInputFields(input));
            }
            f.m_autoBatchState.m_cachedOpHash = hash;
            //fprintf(stderr, "computed\n");
        }
        else
            //fprintf(stderr, "retrieved\n");
            fail_if(hash == SIZE_MAX - 1, "forgot to initialize m_cachedOpHash??");
        return hash;
    }

    // class to manage the set of ready operations (the schedule)
    class ReadyOps
    {
        NonOwningFunctionListBuilder m_viewOps;
        vector<NonOwningFunctionListBuilder> m_regularOps; // m_regularOps[] is a linked list
        //NonOwningFunctionListBuilder m_barrierOps; // TODO: currently dead
        // TODO: remove barrierPendingCounts
        //vector<size_t> m_barrierPendingCounts;  // [barrier id] number of consumers of a barrier id that are not yet ready
        vector<size_t> m_bnPendingCounts = vector<size_t>(16, 0);       // [bn id] number of pending (non-ready) BatchNormalization operations. We need 0, so why not allocate a few more already
        // TODO: This must be turned into something hashable.
        // test whether two PrimitiveFunctions can be executed as a single batched operation
        static bool AreBatchable(const PrimitiveFunction& a, const PrimitiveFunction& b) // TODO: change to & (NULL is not a valid value here)
        {
            //CudaStatsGuard cudaStatsGuard(PrimitiveOpType::Equal, L"AreBatchable()", 3);
            // check the hash first
#if 0
            if (a.m_op != b.m_op) // first-level hash :)
                return false;
            if (OpHash(a) != OpHash(b))
                return false;
#endif
            // first it must be the same operation
            let op = a.m_op;
            // free ops always get batched; even if they have different op-codes
            //fail_if(IsViewOp(op) && !IsBarrier(a), "should not get here for view ops or barrier ops");
            // op codes must match
            if (op != b.m_op)
                return false;
            // some operations have variable number of arguments, e.g. Splice(). Those can't batch if the number of inputs differ
            if (a.m_inputs.size() != b.m_inputs.size())
                return false;
            // special case BatchNormalization
            if (op == PrimitiveOpType::BatchNormalization)
            {
                // ids must match
                let aId = a.m_autoBatchState.m_batchNormId;
                let bId = b.m_autoBatchState.m_batchNormId;
                fail_if(aId != a.m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>() || bId != b.m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>(), "m_batchNormId not initialized correctly??"); // TODO: remove once confirmed not flagging for a while
                if (aId != bId)
                    return false;
                // shape of first argument and object identities of all other arguments must match, otherwise it's an error
                if (a.m_inputs.front().Shape() != b.m_inputs.front().Shape())
                    InvalidArgument("Primitive op '%S' encountered two instances of the same id %d with different shapes %S and %S.",
                                    PrimitiveOpTypeName(op).c_str(), (int)aId, a.m_inputs.front().Shape().AsString().c_str(), b.m_inputs.front().Shape().AsString().c_str());
                for (size_t i = 1; i < 6; i++)
                {
                    if (a.m_inputs[i].m_dataFields != b.m_inputs[i].m_dataFields)
                        InvalidArgument("Primitive op '%S' encountered two instances of the same id %d with different %d-th argument.",
                            PrimitiveOpTypeName(op).c_str(), (int)aId, (int)i);
                }
                return true; // these *must* be batched
            }
            // special case for Block
            if (op == PrimitiveOpType::Block)
            {
                // composites must match (object identity)
                let& aComposite = static_cast<const BlockFunction&>(a).Composite();
                let& bComposite = static_cast<const BlockFunction&>(b).Composite();
                if (CacheAndGetBatchableCompositeId(static_pointer_cast<CompositeFunction>(aComposite)) != CacheAndGetBatchableCompositeId(static_pointer_cast<CompositeFunction>(bComposite)))
                    return false;
            }
            // all input dimensions must match (with exception of a few special cases)
            //let opClass = g_oscTable[op]; // operation-specific auto-batching class
            //let isTimes = opClass == OpSpecificConditionKind::MatrixProduct;
            let isTimes = IsMatrixProduct(op);
            let numInputs = a.m_inputs.size();
            for (size_t i = 0; i < numInputs; i++)
            {
                let& aFields = GetInputFields(a.m_inputs[i]); // (we don't see through no-ops since the target shape is the right one to test)
                let& bFields = GetInputFields(b.m_inputs[i]);
                // there are a few special cases
                if (isTimes && i == 0)
                {
                    // for Times, the first arg must be the same object, not just the same shape
                    // TODO: a special case is a dot product, which we can write as ReduceSum(ElementTimes(a,b))
                    //       This would require to rewrite the graph though; can we do that?
                    if (&aFields != &bFields)
                        return false;
                }
                else
                {
                    // shapes and data types must match
                    // BUGBUG: How about strides?
#if 1
                    if (isTimes)
                    {
                        // special case for now: detect stacking only for Times
                        let rank = aFields.m_shape.Rank();
                        if (rank != bFields.m_shape.Rank())
                            return false;
                        for (size_t k = 0; k + 1 < rank; k++)
                        {
                            if (aFields.m_shape[k] != bFields.m_shape[k])
                                return false;
                        }
                    }
                    else
#endif
                    if (aFields.m_shape != bFields.m_shape)
                        return false;
                    if (aFields.m_dataType != bFields.m_dataType)
                        return false;
                    if (aFields.m_isSparse != bFields.m_isSparse)
                        return false;
                    // depth hint must match
                    if (aFields.m_redirection.m_depthHint != bFields.m_redirection.m_depthHint)
                        return false;
                    // sparse cannot be batched beyond rank 1
                    if (aFields.m_isSparse && aFields.m_shape.Rank() > 1)
                        return false;
                }
            }
            // attributes must also match
            if (a.m_attributes != b.m_attributes) // TODO: this could be an expensive operation; check that. We only need to compare for some ops; for most, attributes are already known from the shapes.
                return false;
            // all match: we can batch
            return true;
        }

        // determine and cache a unique id for every basic block composite
        // such that composites that are equivalent and therefore batchable can be detected easily
        static size_t CacheAndGetBatchableCompositeId(const CompositeFunctionPtr& compositePtr)
        {
            // TODO: move this class out of here
            class BatchableCompositeIdMatcher
            {
                // using weak_ptr so that we can detect whether a composite has been deleted, and remove it from the list,
                // and eventually recycle unique ids. TODO.
                vector<vector<weak_ptr<CompositeFunction>>> allComposites; // [m_batchableCompositeId]
                // Note: This handles the special case of Times() via AreBatchable().
                // Any embedded Times() operation must match object identity of the first argument.
                // That will never match if that argument is computed inside the basic block,
                // but it will work if it is a Parameter or a value computed outside that is the same object.
                static bool AreCompositesBatchable(const PrimitiveFunction& a, const PrimitiveFunction& b)
                {
                    if (&a == &b)
                        return true;
#if 0               // BUGBUG: I noticed a slight loss after I added batching of composites. It is not clear where it is from. Verify this!
                    return false;
#else
                    if (!AreBatchable(a, b))
                        return false;
                    for (size_t i = 0; i < a.m_inputs.size(); i++)
                    {
                        fail_if(a.m_inputs[i].Shape().IsUnknown(), "AreCompositesBatchable called for unknown shape??");
                        let& aInputFields = GetInputFields(a.m_inputs[i]);
                        let& bInputFields = GetInputFields(b.m_inputs[i]);
                        if (aInputFields.m_redirection.empty() != aInputFields.m_redirection.empty())
                            return false;
                        if (!aInputFields.m_redirection.empty())
                        {
                            // BUGBUG: we retraverse multiple paths for now. This will be rewritten and done properly.
                            // BUGBUG: We also must compare the actual graph link structure, not just the unrolled tree.
                            if (!AreBatchable(*aInputFields.m_redirection.m_function, *bInputFields.m_redirection.m_function))
                                return false;
                        }
                    }
                    return true;
#endif
                }
            public:
                size_t MatchAndRememberComposite(const CompositeFunctionPtr& compositePtr)
                {
                    // match new composite against all in the list by deep structure comparison
                    for (size_t i = 0; i < allComposites.size(); i++)
                    {
                        auto& otherList = allComposites[i];
                        if (otherList.empty()) // TODO: garbage collection and weak_ptr discovery should go in here
                            continue;
                        let otherCompositePtr = otherList.front().lock();
                        if (!otherCompositePtr)
                            continue; // TODO: keep looking for a non-expired one
                        let areBatchable = AreCompositesBatchable(static_cast<const PrimitiveFunction&>(*compositePtr->RootFunction()), static_cast<const PrimitiveFunction&>(*otherCompositePtr->RootFunction()));
                        // found a match
                        if (areBatchable)
                        {
                            otherList.push_back(compositePtr);
                            return i;
                        }
                    }
                    // none matched: create a new entry
                    allComposites.push_back(vector<weak_ptr<CompositeFunction>>(1, compositePtr));
                    return allComposites.size() - 1;
                 }
            };
            static BatchableCompositeIdMatcher matcher; // holding this in a static var, but using weak pointers, so this is lightweight
            if (compositePtr->m_batchableCompositeId == SIZE_MAX)
                compositePtr->m_batchableCompositeId = matcher.MatchAndRememberComposite(compositePtr);
            return compositePtr->m_batchableCompositeId;
        }
    public:
        // count an occurrence of a BatchNormalization with a given id
        void CountBatchNorm(size_t bnId)
        {
            fail_if(bnId == 0, "batch norm id should not be 0!");
            if (bnId >= m_bnPendingCounts.size())
                m_bnPendingCounts.resize(bnId * 10, 0);
            m_bnPendingCounts[bnId]++;
        }
        // schedule an operation that has been confirmed to be ready
        // This is called for nearly every unbatched PrimitiveFunction, and must therefore be blazingly fast.
        void Schedule(PrimitiveFunction& f)
        {
            //CudaStatsGuard cudaStatsGuard(PrimitiveOpType::ToSequence, L"Schedule()", 3, m_regularOps.size() * m_regularOps.size());
            let op = f.m_op;
            // special case BatchNormalization: we must account for all occurences
            if (op == PrimitiveOpType::BatchNormalization)
            {
                CudaStatsGuard cudaStatsGuard(PrimitiveOpType::BatchNormalization, L"Schedule(BN)", 3);
                let bnId = f.m_autoBatchState.m_batchNormId;
                fail_if(bnId != f.m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>(), "m_batchNormId not initialized correctly??"); // TODO: remove once confirmed not flagging for a while
                fail_if(m_bnPendingCounts[bnId] == 0, "m_bnPendingCounts decreased beyond initial count??");
                m_bnPendingCounts[bnId]--; // only those with pending count 0 are ready
            }
            // we manage two ready sets, since two common kinds are very simple
            //fail_if (IsBarrier(f), "m_barrierOps.push_back(f) should no longer be done"); // BUGBUG: We never get here since we now see through barriers for efficiency...
            if (IsViewOp(op))  // note: this is, with possibly a few exceptions, Slice()
                m_viewOps.push_back(&f); // (linked list)
            else
            {
                // this naive implementation just scans linearly
                // scan through all op sets to see if one is batchable with 'f'
                // So far this does not show up in profiling.
                for (auto& iter : m_regularOps) // (vector)
                {
                    if (AreBatchable(f, *iter.front()))
                    {
                        // Found another function that this is batchable with: add to batch.
                        iter.push_back(&f); // (note: This just adds to a linked list, and is therefore cheap.)
                        return;
                    }
                }
                // none fit: open a new set
                m_regularOps.push_back(NonOwningFunctionListBuilder(&f)); // (vector)
            }
        }
        // notify a function that an input has become available; schedule it when all inputs are now available
        // This is called for nearly every unbatched PrimitiveFunction, and must therefore be blazingly fast.
        void NotifyAnInputHasBecomeAvailable(PrimitiveFunction& f)
        {
            GetOutputFields(f); // ignoring return value; this just performs a consistency check
            fail_if(f.m_autoBatchState.m_pendingInputs == 0, "pending inputs already 0 yet we are executing it??");
            f.m_autoBatchState.m_pendingInputs--;
            // if it is now ready then schedule it
            if (f.m_autoBatchState.m_pendingInputs == 0)
                Schedule(f);
        }
        // test if no more ready ops
        bool empty() const { return m_viewOps.empty() && m_regularOps.empty() /*&& m_barrierOps.empty()*/; }
        //size_t size() const { return (m_viewOps.size() > 0) /*+  + (m_barrierOps.size() > 0)*/; } // TODO: What is this double +??
        size_t numBatchableOpsPending() const { return m_regularOps.size(); }
        // helper to check whether this is a BatchNormalization op that still has some instances pending
        // TODO: just return 'true'; or maybe rename to IsBatchNormPending()
        int GetBatchNormPending(const vector<NonOwningFunctionListBuilder>::const_iterator& iter)
        {
            let& f = *iter->front();
            if (f.m_op != PrimitiveOpType::BatchNormalization)
                return 0;
            let bnId = f.m_autoBatchState.m_batchNormId;
            fail_if(bnId != f.m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>(), "m_batchNormId not initialized correctly??"); // TODO: remove once confirmed not flagging for a while
            return m_bnPendingCounts[bnId] > 0;
            // TODO: get this from f.m_autoBatchState.m_batchNormId
        }
        // select the next batched op to execute
        NonOwningFunctionList pop_best()
        {
            CudaStatsGuard cudaStatsGuard(PrimitiveOpType::PastValue, L"pop_best()", 3, m_regularOps.size());
            //for (auto iter = m_regularOps.begin(); iter != m_regularOps.end(); iter++)
            //    if (iter->front()->Name() == L"vecSoftmaxMinus")
            //        fprintf(stderr, "### %S pending\n", iter->front()->Name().c_str()), fflush(stderr);

            // try all three queues, in priority order
            if (!m_viewOps.empty()) // view ops always go first, since they are free
                return move(m_viewOps);
            else //if (!m_regularOps.empty()) // regular ops
            {
                auto best = m_regularOps.begin();
                for (auto iter = best + 1; iter != m_regularOps.end(); iter++)
                {
                    // TODO: optimize this further, e.g. don't get autobatch state over again
                    int diff = 0;
                    // TODO: just say if IsBatchNormPending(iter) continue;
                    diff = -(GetBatchNormPending(iter) - GetBatchNormPending(best)); // BatchNormalization with pending inputs always loses
                    if (diff) goto got_diff;
                    diff = -((int)iter->front()->m_autoBatchState.m_depthHint - (int)best->front()->m_autoBatchState.m_depthHint); // barrier: higher depthHint gets batched last
                    if (diff) goto got_diff;
                    diff = (int)iter->size() - (int)best->size();
                got_diff:
                    if (diff > 0)
                        best = iter;
                }
                // special case BatchNormalization
                if (GetBatchNormPending(best)) // the only ready op is BN with some instances still pending -> error (I am not sure under which circumstances this may ever happen)
                    InvalidArgument("Primitive op '%S' with id %d must not be used in a recurrent loop (must not depend on its own output).",
                        PrimitiveOpTypeName(best->front()->m_op).c_str(), (int)best->front()->m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>());
                // and remove this one from the list
                NonOwningFunctionList out = *best; // since NonOwningFunctionListBuilder uses unmanaged pointers, we can just copy it
                m_regularOps.erase(best); // TODO: suboptimal complexity; but a list has the same problem. Priority queue?
                return out;
            }
            //else
            //    return move(m_barrierOps); // barriers only get returned when no other op is available
        }
    };
    ReadyOps m_schedule;

    // recursively traverse the tree hanging off a Variable and
    //  - prepare all nodes for batched execution
    //  - schedule all ready operations
    // TODO: Once we are in the main build, change all Function to PrimitiveFunction directly.
    // TODO: What to do with multi-valued functions? Which ones are there? What is Combine(), a barrier?
    // Caller must call m_visitorTag.Begin() first.
    // 'var' is an input; that is, what m_inputs[] points to, not undergone any redirect.
    void RBuildForwardGraphAndSchedule(const Variable& var, size_t depth)
    {
        auto& fields = GetInputFields(var);
        // return if this node was already visited
        if (m_visitorTag.Visited(fields.m_visitedTag))
            return;

        // initialize m_consumers chain of function that produces the values
        fields.m_consumers.clear();
        // if not visited yet then m_redirection is invalid and must be set up
        //fail_if(fields.m_redirection, "redirection set up here twice??"); // OK since ClonePrimitiveFunction initializes m_function to itself
        // BUGBUG: ^^ This logic is twisted. ClonePrimitiveFunction already "visits" the node. It should do it right and set the visitor flag.
        //         We should call RBuildForwardGraphAndSchedule() from inside the scheduler. But for that, we must separate out the schedule bit.

        // handle leaves
        // Leaves are Parameters, Constants, and also nodes that already have a value.
        if (fields.m_varKind == VariableKind::Input || fields.m_varKind == VariableKind::Placeholder)
            InvalidArgument("Dynamic Value() must not depend on Input or Placeholder.");
        let isParameterOrConstant = fields.m_varKind == VariableKind::Parameter || fields.m_varKind == VariableKind::Constant;
        if (fields.m_value || isParameterOrConstant)
        {
            if (isParameterOrConstant && !fields.m_value)
            {
                var.Value(); // this is a Parameter for which we still need to run the initializer. This does that.
                fail_if(!fields.m_value, "Parameter/Constant has no Value()??");
            }
            fields.m_redirection.reset();
            m_stats.numLeafNodes++;
            return;
        }

        // see through ops that do nothing
        auto& f = *fields.m_ownerFunction.lock().get(); // this is the only place we ever call lock(); after this, we just use the raw pointer (relying on immutability)
        let op = f.m_op;
        // unroll non-basic BlockFunctions (unless it is a basic block; then it is unrolled during execution)
        if (op == PrimitiveOpType::Block && !static_cast<BlockFunction&>(f).IsBasicBlock())
        {
            // make a deep copy of the block's root (PrimitiveFunction graph)
            // The Block is treated like a see-through op:
            //  - a deep copy of the Block is made (inlined), which connects to the Block node's inputs
            //  - the Block node, like a NoOp, gets its m_redirection set to point to the Block's copy.
            // Upon first call, the original Block function gets its dimensions inferred. I.e. it cannot be called twice with mismatching dimensions.
            // Besides that, the original Block function will remain unmodified.
            m_compositeVisitorTag.Begin();
            auto inlinedRootPtr = RInlineComposite(static_cast<PrimitiveFunction&>(*f.BlockRoot()),
                                                   f.m_inputs, /*cloneFn=*/ClonePrimitiveFunction);
            // This returns a shared_ptr to the deep copy of the root.
            // ^^ This currently does not inline nested blocks. Instead of cloning the BlockFunction, we should just unroll any nested non-basic-block right there.
            // prepare graph that we just unrolled
            // Note: We may just have pointed to yet another nested block, or a see-through op, which must be eliminated right here
            let& output = inlinedRootPtr->m_outputs.front();
            RBuildForwardGraphAndSchedule(output, depth + 1);
            // set up linkage in our overlaid structure
            fields.m_redirection = GetInputFields(output).m_redirection; // we redirect to whatever the inlined one redirects to
            // the ref count:
            //  - If the inlined block root is a redirect that holds a ref count, then that is a redirect itself.
            //    Hence, the root Function has been short-circuited, and no ref-count should be held to it (it gets freed right here).
            //  - If the inlined block is not a redirect (that is, m_function points to itself), then we must hold the ref count to it.
            if (!fields.m_redirection.m_functionHolder)
                fields.m_redirection.m_functionHolder = move(inlinedRootPtr); // keep a ref count to a potentially inlined function
            // For ref-counting, the root pointer of the deep copy is kept in redirectedFieldsHolder,
            // which goes into m_functionHolder of the Block's output Variable.
            // Note: When we get here, redirectedFieldsHolder may already hold a function.
            // It is OK to overwrite it, since the raw pointer in redirectedFieldsOwner also got overwritten,
            // and the original Function (which is a Block pre inlining) can be safely freed since it is
            // not referenced anymore (anything of interest has been copied in the inlining process).
            m_stats.numInlinedBlocks++;
            return;
        }

        // short-circuit see-through ops
        else if (IsAliasOp(op) || op == PrimitiveOpType::Reshape)
        {
            // TODO: what if input is non-continuous? Then Reshape should become a copy. Does this need to be addressed here?
            m_stats.numShortCircuitedNodes++;
            let& input = f.m_inputs.front();
            RBuildForwardGraphAndSchedule(input, depth + 1);
            auto& redirectedFields = GetInputFields(input);
            //fail_if(fields.m_redirection, "redirection set up here twice??");
            fields.m_redirection = redirectedFields.m_redirection; // we redirect to whatever the input redirects to
            // BUGBUG:!!!! How aboud the gradient?? We must know how to find the gradient!
            //        One solution is to redirect to the operation directly on top of the Parameter, not the parameter itself.
            if (fields.m_redirection.empty()) // redirects to leaves are currently not handled correctly (since redirects are based on Function, not Variable)  --TODO: change that
                InvalidArgument("Value(): See-through ops on leaves are currently not implemented.");
            // if a barrier then record the maximum depth hint encountered
            // Functions consuming this Variable are bound by the barrier (but not the function that produces the original value).
            if (IsBarrier(f)) // TODO: get this from a different attribute
                fields.m_redirection.m_depthHint = max(fields.m_redirection.m_depthHint, f.m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>());
            return;
        }

        // this op is not getting redirected
        if (f.m_outputs.size() > 1)
            InvalidArgument("Dynamic operations cannot have multiple outputs.");

        // set up linkage in our overlaid structure
        fields.m_redirection.reset(&f);          // pointer to ourselves, don't hold the ref count to ourselves (would create a cyclic graph)

        // special case BatchNormalization: we must account for all occurences before normalizing
        // We count all occurences during this initial tree traversal.
        // During batched execution, we hold back BatchNorm ops until the batch size equals
        // the number of occurences.
        // (We also cache the id here, to avoid repeated accesses to the attribute Dictionary.)
        if (op == PrimitiveOpType::BatchNormalization)
        {
            //if (!f.m_attributes.Contains(PrimitiveFunction::AttributeNameSyncId))
            //    InvalidArgument("Primitive op '%S' requires an id parameter. Please use the version that takes an id.",
            //        PrimitiveOpTypeName(f.m_op).c_str());
            let bnId = f.m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>();
            f.m_autoBatchState.m_batchNormId = bnId; // cached here since this is tested in inner loop
            m_schedule.CountBatchNorm(bnId);
        }
        else
            f.m_autoBatchState.m_batchNormId = 0;

        // determine how many inputs are pending; and also recurse and set up the consumer list
        size_t pendingInputs = 0;
        size_t maxDepthHint = 0;
        let& inputs = f.m_inputs;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            let& input = inputs[i];
            // recursively traverse
            RBuildForwardGraphAndSchedule(input, depth + 1);
            let& inputFields = GetInputFields(input);
            if (!inputFields.m_value) // (if input is a leaf, it has a value)
            {
                //if (inputFields.m_varKind == VariableKind::Placeholder) // Placeholder in Block inlining
                //    continue;
                fail_if(inputFields.m_redirection.empty(), "input has no value and is not a function either??");
                auto& outputFields = GetOutputFields(*inputFields.m_redirection.m_function);
                pendingInputs++;
                // record ourselves as a consumer of the input
                // Note that RBuildForwardGraphAndSchedule() will have reset this upon first visit of 'input'.
                // The recorded consumer is the function that physically produces things, not the redirect.
                outputFields.m_consumers.push_back({ &f, i });
                maxDepthHint = max(maxDepthHint, inputFields.m_redirection.m_depthHint);
            }
        }
        f.m_autoBatchState.m_pendingInputs = pendingInputs;
        // and some more initializations since we are here
        //f.m_autoBatchState.m_aliasHash = SIZE_MAX;        // aliasHash is used for detecting aliases (identical ops) in batched ops
        f.m_autoBatchState.m_depthHint = maxDepthHint;    // this controls batching priority; ops with higher depth hint will be held back longer
        f.m_autoBatchState.m_cachedOpHash = SIZE_MAX;
        // if none then operation is ready
        if (pendingInputs == 0)
            m_schedule.Schedule(f); // add to ready set
        m_stats.numOpNodes++;
    }

    // get the value of an input Variable, with full redirection
    // This will realize any lazy ops (slice, reshape).
    // A Variable's value is defined by its m_redirection.m_function->m_outputs.front(), followed by slice and/or reshape.
    static const NDArrayViewPtr& CacheAndGetValue(const Variable& v)
    {
        return CacheAndGetValue(GetInputFields(v));
    }
    static const NDArrayViewPtr& CacheAndGetValue(VariableFields& fields)
    {
        if (fields.m_value) // value is available (including cached values): use it
            return fields.m_value;
        fail_if(fields.m_redirection.empty(), "Variable unexpectedly has no value yet, nor is it a slice view into a batched op");
        // get the actual value from the function that computed it
        auto& functionFields = GetOutputFields(*fields.m_redirection.m_function);
        fail_if(&fields == &functionFields, "Variable unexpectedly has no value yet"); // avoid infinite recursion
        // function itself may be a redirect (a slice into a batched op)
        CacheAndGetValue(functionFields); // (calling ourselves on the source in case of re-redirection)
        RealizeVariableAsView(fields, functionFields.m_value);
        return fields.m_value;
    }
    // realize the lazy redirected value; that is, do the slice and reshape
    // This sets outputFields.m_value. inputFields must have m_value already set.
    static void RealizeVariableAsView(VariableFields& outputFields, NDArrayViewPtr from)
    {
        fail_if(!from, "Variable's input unexpectedly has no value yet");
        // optional implicit index and reshape
        let sliceRange = outputFields.m_redirection.m_sliceRange;
        if (!sliceRange.empty())
            from = from->SliceViewAsShape(sliceRange.BeginIndex(), sliceRange.EndIndex(), outputFields.m_shape);
        else // no slice
            ReplaceWithReshapedViewIfNeeded(from, outputFields.m_shape);
        outputFields.m_value = move(from);
    }
    // get the value that must already have been cached
    static const NDArrayViewPtr& GetCachedValue(const Variable& v)
    {
        let& value = GetInputFields(v).m_value;
        //fail_if(!value, "GetCachedValue: Variable unexpectedly has no value yet");
        return value;
    }
    // this gets the underlying NDArrayView that contains v's value, without realizing the value
    // The real value may be a view into this object that has not been realized yet.
    // Use this to find out properties, such as the device, type, and storage format.
    // Currently only used for batched BatchNorm.
    static const NDArrayView* GetValueObject(const Variable& v)
    {
        auto& fields = GetInputFields(v);
        if (fields.m_value)
            return fields.m_value.get();
        if (fields.m_redirection.empty()) // redirect to output of function that produces this value
            LogicError("GetValueObject() called where no value object exists (hit a leaf)??");
        let& output = fields.m_redirection.m_function->m_outputs.front(); // note: may be recursive
        if (&GetInputFields(output) == &fields)
            LogicError("GetValueObject() called where no value object exists (hit a self-ref)??");
        return GetValueObject(output);
    }
    // this gets the non-redirected data fields, which describe v's properties as an input
    // This returns the fields of a potentially virtual value that does not produce its own output but merely views another.
    static VariableFields& GetInputFields(const Variable& v)
    {
        return *v.m_dataFields;
    }
    // this the fields of the output of 'f', which describe where f's output goes, without redirection
    // This returns the fields where stuff is actually produced ("physical value").
    // ^^ BUGBUG: No, this may be a slice of a batched result.
    // This can only be used with functions that are not redirected.
    static VariableFields& GetOutputFields(const PrimitiveFunction& f)
    {
        auto& fields = *f.m_outputs.front().m_dataFields;
        //fail_if(!fields.m_redirection, "GetOutputFields() called on a function that is see-through or leaf??");
        return fields;
    }

    static void LogFunction(const PrimitiveFunction& f, const wchar_t* prefix = L"", size_t markIndex = SIZE_MAX)
    {
        let& inputs = f.m_inputs;
        let& output = f.m_outputs.front(); // BUGBUG: How to deal with multi-valued functions?
        let& outputShape = output.Shape();
        auto uid = f.Uid();
        let& name = f.Name();
        if (!name.empty())
            uid = name + L":" + uid;
        if (prefix && *prefix)
            fprintf(stderr, "[%S] ", prefix);
        fprintf(stderr, "%S%S = %S (", uid.c_str(), outputShape.AsString().c_str(), f.OpName().c_str());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            let& input = inputs[i];
            let& fields = GetInputFields(input); // (the fields that describe input as an input, e.g. its shape after potential see-through ops)
            // little helper function to fix up variable names by removing _Output_0
            // TODO: Once we support >1 output, this needs a bit more code.
            let GetVarName = [](const Variable& input) -> wstring
            {
                auto uid = input.Uid();
                if (uid.size() > 9 && wcscmp(uid.c_str() + uid.size() - 9, L"_Output_0") == 0)
                    uid.resize(uid.size() - 9);
                let& name = input.IsOutput() ? input.m_dataFields->m_redirection.m_function->Name() : input.Name();
                if (!name.empty())
                    uid = name + L":" + uid;
                return uid;
            };
            if (!fields.m_redirection.empty())
            {
                let& input1 = fields.m_redirection.m_function->m_outputs.front();
                fprintf(stderr, "%s%s%S%S", (i == 0) ? "" : ", ", (i == markIndex) ? "=>" : "", GetVarName(input1).c_str(), input1.Shape().AsString().c_str());
                let slice = fields.m_redirection.m_sliceRange;
                if (!slice.empty())
                {
                    if (slice.IsIndex())
                        fprintf(stderr, "[%d]", (int)fields.m_redirection.m_sliceRange.Index());
                    else
                        fprintf(stderr, "[%d:%d]", (int)fields.m_redirection.m_sliceRange.BeginIndex(), (int)(fields.m_redirection.m_sliceRange.EndIndex()));
                }
            }
            else
                fprintf(stderr, "%s%s%S%S", (i == 0) ? "" : ", ", (i == markIndex) ? "=>" : "", GetVarName(input).c_str(), input.Shape().AsString().c_str());
            if (i == 4 && inputs.size() > 6) // skip the middle ones
            {
                fprintf(stderr, ", ...+%d", (int)(inputs.size() - 6));
                i = inputs.size() - 2;
            }
        }
        let& attributes = f.m_attributes;
        if (attributes.Size() > 0)
        {
            for (let& kv : attributes)
            {
                fprintf(stderr, ", %S=", kv.first.c_str());
                let& val = kv.second;
                if (val.HasValue())
                {
                    switch (val.ValueType())
                    {
                    case DictionaryValue::Type::Bool:    fprintf(stderr, "%s",     val.Value<bool  >() ? "true" : "false"); break;
                    case DictionaryValue::Type::Int:     fprintf(stderr, "%d",     val.Value<int   >()); break;
                    case DictionaryValue::Type::SizeT:   fprintf(stderr, "%d",     (int)val.Value<size_t>()); break;
                    case DictionaryValue::Type::Float:   fprintf(stderr, "%f",     val.Value<float >()); break;
                    case DictionaryValue::Type::Double:  fprintf(stderr, "%f",     val.Value<double>()); break;
                    case DictionaryValue::Type::String:  fprintf(stderr, "\"%S\"", val.Value<wstring>().c_str()); break;
                    case DictionaryValue::Type::NDShape: fprintf(stderr, "%S",     val.Value<NDShape>().AsString().c_str()); break;
                    case DictionaryValue::Type::Axis:    fprintf(stderr, "%S",     val.Value<Axis   >().AsString().c_str()); break;
                    default: fprintf(stderr, "(type%d)", (int)val.ValueType()); break;
                    }
                }
                else
                    fprintf(stderr, "(empty)");
            }
        }
        if (!f.m_name.empty())
            fprintf(stderr, ", Name=\"%S\"", f.m_name.c_str());
        fprintf(stderr, ")\n");
    }
    // same but takes the prefix from the profiler if given
    static void LogFunction(const PrimitiveFunction& f, const DynamicProfilerPtr& profiler, size_t markIndex = SIZE_MAX)
    {
        LogFunction(f, profiler ? profiler->Name() : L"-", markIndex);
    }

    // clone a graph of PrimitiveFunctions of a composite of a basic block, and execute it as we go along
    // This is used to execute a basic block. It returns a new PrimitiveFunction(-Ptr), which will have its output's m_value set.
    // Logically, we execute the block's composite by replacing its Placeholders (on the fly) with the provided batched blockInputs[].
    // Practically, we completely clone this graph, in order to be able to backprop through it.
    // The clone will have changed dimensions that reflect the additional batch axis. The batch axis is shared across *all* operations,
    // and therefore must have been pre-determined to be larger than all ranks of all involved inputs and outputs
    // (also considering any potentially nested basic blocks).
    // Any input that does not have this shared batch axis is a non-batched input (all inputs the same).
    // Any Times operation inside the block requires an unbatched first argument (weight matrix);
    // therefore we carefully propagate the non-batchedness of intermediate results.
    // As a special case, execution can also be unbatched. This is indicated by batchAxis==SIZE_MAX.
    // Any nested blocks are executed inside here as well.
    // This function can be seen as a special case of ExecuteBatchedOpAndUpdateSchedule() that assumes
    // batched inputs and consistent batching for all ops, and therefore does not perform any additional (re-)batching.
    PrimitiveFunctionPtr InlineAndMemoizeBatchedBasicBlock(const BlockFunction& block, const vector<Variable>& invocationArgs, size_t batchAxis/*or SIZE_MAX*/, size_t batchSize)
    {
        // BUGBUG:
        //  - if block contains BatchNorm, then the implied barrier is not accounted for in batching.
        //    If user code is correct, it will compute the right thing. But we cannot presently check or account for it.
        //  - this does not short-circuit operations. We need to do the same to the composite as we do to the dynamic graph in forward-graph building.

        //if (any_of(block.m_inputs.begin(), block.m_inputs.end(), [](const Variable& arg) { return arg.m_dataFields->m_uniqueIdForDebugging == 243; }))
        //    Break;
        //if (block.m_inputs.size() == 4)
        //    Break;
        //if (block.m_inputs.size() == 4 && block.m_inputs.back().m_dataFields->m_uniqueIdForDebugging == 243)
        //    Break;
        //if (block.Composite()->m_outputs.front().m_dataFields->m_uniqueIdForDebugging == 288)
        //    Break;

        let prevVisitorTag = m_compositeVisitorTag.Begin(); // (for detecting which of the composite's PrimitiveFunctions have already been expanded)
        let fInlinedPtr = RInlineComposite(static_cast<PrimitiveFunction&>(*block.BlockRoot()), invocationArgs, [this, batchAxis, batchSize](PrimitiveFunction& f, vector<Variable>&& newInputs) -> PrimitiveFunctionPtr
        {
            // This lambda must apply the passed in the composite's PrimitiveFunction 'f' to 'newInputs'.
            // The newInputs are batched; that is, their shapes may mismatch those of f's inputs in that they may have an additional batch axis.

            // determine if any of the inputs have a batch axis
            // If any, then the output will get one as well.
            let anyBatchedInputs = any_of(newInputs.begin(), newInputs.end(), [batchAxis](const Variable& input) { return input.Shape().Rank() >= batchAxis; });

            if (IsMatrixProduct(f.m_op) && newInputs.front().Shape().Rank() >= batchAxis)
                InvalidArgument("Inside a basic block, the first argument of a matrix products cannot have a batch axis.");

            // clone and memoize this operation
            // We pass on batchAxis such that the output shape will be that of the original output shape of the Function in the
            // composite, augmented with the batch axis. However, if no input has a batch axis, then the output should have none.
            // PERF BUGBUG: newInputs is already a copy that we can just move
            // PERF BUGBUG: We should short-circuit the free ops for composites as well.
            let isFree = IsViewOp(f.m_op);
            return CreateAndMemoizeBatchedOp(f, /*move*/newInputs, anyBatchedInputs ? batchAxis : SIZE_MAX, batchSize, L"()"/*f0*/, isFree);
        });
        m_compositeVisitorTag.End(prevVisitorTag); // (restore)
        return fInlinedPtr;
    }

    // create a PrimitiveFunction, execute it right away, and prep result as an input
    // This first executes RawPrimitiveFunction() and MemoizeKnowableValueInArena(),
    // and then returns the result in the form of a Variable suitable as an input to the next PimitiveFunction
    // by setting its m_acyclicOutputPrimitiveReference field.
    // This is a commonly needed pattern in auto-batched execution. All ops generated in here are known to be acyclic.
    Variable CreateAndMemoizeOpAsInput(PrimitiveOpType op, vector<Variable>&& inputs, const NDShape& shape, Dictionary&& attributes, const wstring& name,
                                       const DynamicProfilerPtr& profiler, const wchar_t* logPrefix,
                                       bool isFree = false, bool spliceIsGather = false)
    {
        auto fPtr = CreateAndMemoizeOp(op, move(inputs), shape, move(attributes), name, profiler, logPrefix, isFree, spliceIsGather);
        let& output = fPtr->m_outputs.front();
        // To make the consumer of this hold a reference to this PrimitiveFunction, inject a strong ref to the copy of its Output.
        //input = output.CompositePreservingCopy(fPtr); // without the acyclic trick, this works as well, but is not quite right since fPtr is not a Composite
        Variable input = output;                           // copy the Variable object (which is mostly a shared_ptr tp VariableFields which are not copied but shared)
        input.m_acyclicOutputPrimitiveReference = move(fPtr); // and implant the reference to the Primitive that generated it, for reference countiung
        return input;
    }

    // create a PrimitiveFunction that is a batched version of a given op, and execute it right away
    // This is a wrapper around CreateAndMemoizeOp(). If a batchAxis is provided (!= SIZE_MAX), then the output will have that axis.
    // In that case, all inputs[*], unless not batched, must have the same batch axis (an example of a non-batched arg is the first arg of a matrix product).
    PrimitiveFunctionPtr CreateAndMemoizeBatchedOp(const PrimitiveFunction& f, const vector<Variable>& inputs, size_t batchAxis, size_t batchSize, const wchar_t* logPrefix, bool isFree)
    {
        // special case for basic blocks: need to clone the entire composite (for backprop)
        if (f.m_op == PrimitiveOpType::Block)
            return InlineAndMemoizeBatchedBasicBlock(static_cast<const BlockFunction&>(f), m_batchedInputs, batchAxis, batchSize);

        let& unbatchedOutputShape = f.m_outputs.front().Shape();
        const NDShape& shape = batchAxis != SIZE_MAX ? (batchAxis < unbatchedOutputShape.Rank() ? unbatchedOutputShape.SubShape(0, batchAxis) : unbatchedOutputShape).AppendAxis(batchAxis, batchSize) : unbatchedOutputShape;
        // TODO: This is a little inefficient                                  ^^
        Dictionary attributes;
        f.Attributes().ShallowCloneTo(attributes); // (this just copies the shared_ptr, not the content)
#if 1   // a sanity check whether we batched correctly. Can be removed once this stuff works.
        if (IsMatrixProduct(f.m_op))
            fail_if(inputs.front().Shape() != f.m_inputs.front().Shape(), "attempted to batch the weight matrix of a matrix product??");
#endif
        return CreateAndMemoizeOp(f.m_op, vector<Variable>(inputs), shape, move(attributes), f.m_name, f.m_profiler, L"*"/*f*/, isFree);
    }

    // create a PrimitiveFunction and execute it right away
    // This executes RawPrimitiveFunction() and MemoizeKnowableValueInArena().
    // This is a commonly needed pattern in auto-batched execution. All ops generated in here are known to be acyclic.
    PrimitiveFunctionPtr CreateAndMemoizeOp(PrimitiveOpType op, vector<Variable>&& inputs, const NDShape& shape, Dictionary&& attributes, const wstring& name,
                                            const DynamicProfilerPtr& profiler, const wchar_t* logPrefix,
                                            bool isFree = false, bool spliceIsGather = false)
    {
        // create the object
        CudaStatsGuard cudaStatsguard(PrimitiveOpType::Pass, L"RawPrimitiveFunction", 3);
        // PERF BUGBUG: This does not need to use MakeSharedObject(), make_shared() is fine, since functions created with this are internal-use only.
        auto fPtr = MakeSharedObject<PrimitiveFunction>(op, std::move(inputs), std::move(attributes), std::move(name));
        // unfortunately output initialization must be separated out since it requires s shared_ptr to f
        let& fInputs = fPtr->m_inputs;
        fPtr->InitOutput(OutputVariable(shape, fInputs.front().GetDataType(), vector<Axis>(),
                                        any_of(fInputs.begin(), fInputs.end(), [](const Variable& input) { return input.NeedsGradient(); }), // PERF BUGBUG: caller knows this already; should pass it in
                                        all_of(fInputs.begin(), fInputs.end(), [](const Variable& input) { return input.IsSparse();      }), // BUGBUG: This is not generally correct -> Caller knows this, just pass it in.
                                     wstring()));
        //if (fPtr->m_uniqueIdForDebugging == 194962)
        //    Break;
        //if (!fPtr->m_outputs.front().NeedsGradient())
        //    Break;
        FinishConstructingPrimitiveFunction(*fPtr, profiler, logPrefix);
        cudaStatsguard.Stop();

        // execute it
        MemoizeKnowableValueInArena(*fPtr, isFree, spliceIsGather);
        return fPtr;
    }

    // call this after constructing a PrimitiveFunction from inside here, to set up the additional fields
    static void FinishConstructingPrimitiveFunction(PrimitiveFunction& f, const DynamicProfilerPtr& profiler, const wchar_t* logPrefix)
    {
        f.m_profiler = profiler;
#ifdef LOG_DETAILS
        if (logPrefix)
            f.m_uid = logPrefix + f.m_inputs.front().Uid(); // propagate the name of the first input, for better logging
        // TODO: this has not been tested after refactoring
        //       Also we are not always using the correct input here. Fix this once I look at this again.
#endif
        // we must initialize the redirect for backprop
        auto& fields = *f.m_outputs.front().m_dataFields;
        fail_if(fields.m_redirection.m_functionHolder, "shouldn't functionHolder be empty here?");
        fields.m_redirection.reset(&f);         // we are computing stuff ourselves
    }

    // compute the value of 'f', storing it in the arena (unless 'isFree', which must be set when there is nothing to store)
    // The result is stored in f.m_outputs.front()'s m_value.
    const Variable& MemoizeKnowableValueInArena(PrimitiveFunction& f, bool isFree = false, bool spliceIsGather = false)
    {
        // fetch the NDArrayViewPtrs for all inputs
        let& inputs = f.m_inputs;
        CudaStatsGuard cudaStatsGuardPrepare(PrimitiveOpType::Pooling, L"Memoize: prepare", 3, inputs.size());
        auto& inputValues = BorrowBuffer(m_inputValuesBuffer, inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
            inputValues[i] = CacheAndGetValue(inputs[i]); // (if this is a redirect, then now we must resolve it)
        // allocate the output NDArrayViewPtr in the arena
        let& output = f.m_outputs.front(); // BUGBUG: How to deal with multi-valued functions?
        let& outputShape = output.Shape();
        // logging
        if (ShouldProfile(f))
            LogFunction(f, f.m_profiler);
        //if (f.m_op == PrimitiveOpType::ElementTimes)
        //    LogFunction(f, L"bf  ");
        auto outValue =
            /*if*/ (isFree) ?
                NDArrayViewPtr()
            /*else*/:
                m_arena.NewNDArrayView(outputShape, output.GetDataType(), output.IsSparse() ? StorageFormat::SparseCSC : StorageFormat::Dense, inputValues.front()->Device());
        cudaStatsGuardPrepare.Stop();
        CudaStats* cudaStatsPtr = nullptr;
        if (ShouldLogMemoizeStats())
        {
            let hasSparse = any_of(inputs.begin(), inputs.end(), [](const Variable& v) { return v.IsSparse(); });
            let logAsOp = (f.m_op == PrimitiveOpType::Splice && spliceIsGather) ? PrimitiveOpType::Gather : f.m_op; // gather ops are logged as op Gather (CNTK V2 Gather is not used by Dynamite)
            cudaStatsPtr = BeginCudaStats(logAsOp, nullptr, IsViewOp(f.m_op) ? 2 : hasSparse ? 1 : 0, outputShape.TotalSize(/*check=*/false), inputValues.front()->Device());
        }
        // execute it
        GetOutputFields(f).m_value = move(PrimitiveFunction::ComputeKnowableValue(f.m_op, inputValues, f.Attributes(), outputShape, move(outValue), f));
        // stats
        EndCudaStats(cudaStatsPtr, inputValues.front()->Device());
        let primitiveOp = f.m_op;
        if (isFree) // means we did not pass a data buffer for the result; any one we pass a buffer does actual work
            m_stats.numDoneFreeOps++;
        else if (primitiveOp == PrimitiveOpType::Splice && spliceIsGather)
            m_stats.numDoneGatherOps++;
        else
            m_stats.numDoneOtherOps++;
        return output;
    }

    // clone a PrimitiveFunction
    // Subroutine of inlining non-basic-block BlockFunctions during forward graph preparation.
    // Special consideration is given to nested BlockFunctions.
    static PrimitiveFunctionPtr ClonePrimitiveFunction(PrimitiveFunction& f, vector<Variable>&& newInputs)
    {
        CudaStatsGuard cudaStatsguard(PrimitiveOpType::Cos, L"ClonePrimitiveFunction", 3);
        // get the output shape
        let& outputs = f.m_outputs;
        if (outputs.size() != 1)
            InvalidArgument("Dynamic operations cannot have multiple outputs.");
        let& output = outputs.front();
        // clone it
        PrimitiveFunctionPtr fCloned;
        if (f.m_op == PrimitiveOpType::Block)
        {
            let& fAsBlock = static_cast<BlockFunction&>(f);
            fCloned = MakeSharedObject<BlockFunction>(fAsBlock.Composite(), move(newInputs), fAsBlock.IsBasicBlock(), wstring(fAsBlock.OpName()), wstring(f.Name()));
            // PERF BUGBUG: (?) If not basic block then we should expand it right here.
        }
        else
        {
            // clone the attributes
            Dictionary attributes;
            f.m_attributes.ShallowCloneTo(attributes); // note: shallow clone will not copy the map, just a shared_ptr. This only works if attributes are immutable, which is true inside auto-batch
            fCloned = MakeSharedObject<PrimitiveFunction>(f.m_op, move(newInputs), move(attributes), wstring(f.Name()));
        }
        //if (fCloned->m_uniqueIdForDebugging == 194962)
        //    Break;
        // PERF BUGBUG: This does not need to use MakeSharedObject(), make_shared() is fine, since functions created with this are internal-use only.
        // unfortunately output initialization must be separated out since it requires s shared_ptr to fCloned
        let& fClonedInputs = fCloned->m_inputs;
        fCloned->InitOutput(OutputVariable(output.Shape(), fClonedInputs.front().GetDataType(), vector<Axis>(), output.NeedsGradient(), output.IsSparse(), wstring()));
        // add additional initializations for auto-batch only
        FinishConstructingPrimitiveFunction(*fCloned, f.m_profiler, /*logPrefix=*/nullptr);
        return fCloned;
    }

    // notify consumers of 'f' that f's value is now available
    void NotifyOpsConsumersInputsAvailable(const PrimitiveFunction& f)
    {
        // notify consumers
        auto& fields = GetOutputFields(f);
        //fail_if(!fields.m_value && !fields.m_redirection, "NotifyOpsConsumersInputsAvailable: operation unexpectedly reveived no value");
#if 0   // this test is useful but fails for the root; enable for debugging where helpful
        if (!fields.m_consumers.size())
            LogicError("executed a function that shouldn't be executed");
#endif
        fields.m_consumers.ForAll([&](const std::pair<PrimitiveFunction*, size_t>& fi) { m_schedule.NotifyAnInputHasBecomeAvailable(*fi.first); });
        // clear consumer list (this operation is done) for good measure (not really necessary, we could just leave it dangling)
        //fields.m_consumers.clear();
        //fields.m_consumers.mangle(-3333); // leave a mark that this was reset nullptr;
    }

    // implant the the result of a function that was executed as a batched op
    // sliceRange = optional view into batched result, or SliceRange() if entire tensor
    void FinalizeBatchedOpAndUpdateSchedule(PrimitiveFunction& f, const PrimitiveFunctionPtr& batchedOp, SliceRange sliceRange = SliceRange())
    {
        //CudaStatsGuard cudaStatsguard(PrimitiveOpType::FutureValue, L"implant", 3);
        // we remember where we came from for backprop in this case
        auto& fields = GetOutputFields(f);
#if 1
        fields.m_redirection.reset(batchedOp, sliceRange);
        // note: This overwrites depthHint. That's fine since depthHint is only used for scheduling. But this function just got a value, hence will never be scheduled.
#else
        fields.m_redirection.m_function = batchedOp.get(); // this is the actual pointer used to traverse
        fields.m_redirection.m_functionHolder = batchedOp; // and also keep a ref count, since this may be a new object not referenced anywhere else yet
        // If m_functionHolder already contains a pointer, then overwriting it may free the object.
        // That is fine, since the -Holder is only concerned with holdinf a ref count for m_function
        // in case is it was a new function. Since m_function also gets overwritten, the object it used
        // to point to is no longer referenced.
        fields.m_redirection.m_sliceRange = sliceRange;
        // ^^ TODO: use reset() here. But how about depthHint?
#endif
        // semantically, this will compute as fields.m_value = out->IndexLastAxis(j);
        // but it gets deferred to save effort
        // update all ops' consumers and schedule them when possible
        NotifyOpsConsumersInputsAvailable(f);
    }

    // clone a result into all its aliases (which were determined by ShortCircuitBatchedOpDuplicatesAndUpdateSchedule())
    // TODO: We either duplicate m_value or m_redirection, and we know that upon call time; so split this function.
    void UpdateDuplicatesAndUpdateSchedule(PrimitiveFunction& f)
    {
        for (auto* dup = f.m_autoBatchState.m_aliasList; dup; dup = dup->m_autoBatchState.m_aliasList)
        {
            //CudaStatsGuard cudaStatsguard(PrimitiveOpType::FutureValue, L"implant", 3);
            GetOutputFields(*dup).m_value       = GetOutputFields(f).m_value;
            GetOutputFields(*dup).m_redirection = GetOutputFields(f).m_redirection;
            NotifyOpsConsumersInputsAvailable(*dup);
        }
    }

    static uintptr_t GetValueAddrForHash(const NDArrayViewPtr& value)
    {
        return
            /*if*/ value->IsSparse() ?  // DataBuffer() is only available for dense. --PERF BUGBUG: We can find a better hash for sparse data, e.g. its data buffer.
                (uintptr_t)&value
            /*else if*/: value->GetDataType() == DataType::Float ?
                (uintptr_t)value->DataBuffer<float>() / sizeof(float)
            /*else*/:
                (uintptr_t)value->DataBuffer<double>() / sizeof(double);
    }

    // get an offsettable address of m_value for hashing purposes
    static uintptr_t CacheAndGetValueAddrForHash(/*const*/ VariableFields& fields)
    {
#if 1   // This version compares object identity (physical fields, index). It is 6+ x faster, and seems to detect CSEs just the same.
        // we hash on object identity of the physical fields plus slice index
        uintptr_t hash = fields.m_valueAddrForHash;
        if (!hash)
        {
            // determine the Fields of the physical object and the slice
            SliceRange sliceRange;
            let* pfields = &fields;
            while (!pfields->m_redirection.empty())
            {
                let& outFields = GetOutputFields(*pfields->m_redirection.m_function);
                if (&outFields == pfields)
                    break;
                if (sliceRange.empty())
                    sliceRange = pfields->m_redirection.m_sliceRange; // note: it is guaranteed that there is only one index in the chain
                pfields = &outFields;
            }
            hash = 0;
            IncorporateFieldsId(*pfields);
            if (!sliceRange.empty())
            {
                // (Note: not sure if this hashing is done right. Using the same multiplier will put stuff on top of each other.)
                if (sliceRange.IsSlice())
                    hash += (sliceRange.BeginIndex() + hashMultiplier * sliceRange.Width()) * hashMultiplier;
                else
                    hash += (sliceRange.Index() + 1) * hashMultiplier; // SIZE_MAX -> 0, so we can skip the mul in the frequent case
            }
            fields.m_valueAddrForHash = hash;
        }
        return hash;
#else   // This version compares the starting address in GPU memory, and can therefore detect aliases coming through different routes.
        fail_if(true, "this must be updated to correctly use m_sliceRange width"); // we must compare the length as well
        auto addrForHash = fields.m_valueAddrForHash;
        if (!addrForHash)
        {
            if (fields.m_value) // if we have a m_value then get it from there
                addrForHash = GetValueAddrForHash(fields.m_value);
            else // we don't: get it from the redirect
            {
                fail_if(!fields.m_redirection, "CacheAndGetValueAddrForHash called for Variable with no value yet??");
                // get it from the redirect
                auto& outFields = GetOutputFields(*fields.m_redirection.m_function);
                fail_if(&fields == &outFields, "CacheAndGetValueAddrForHash called for Function with no value yet??");
                addrForHash = CacheAndGetValueAddrForHash(outFields); // recurse (result may be see-through into a batched slice)
                if (fields.m_redirection.m_index != SIZE_MAX && fields.m_redirection.m_index != 0 && !fields.m_isSparse) // correct for slice
                {
                    // (BUGBUG: We skip this for sparse, which will prevent CSE discovery. Needs a solution.)
                    // determine stride
                    let& shape = outFields.m_shape;
                    let stride = shape.TotalSize(/*check=*/false) / shape.Dimensions().back(); // BUGBUG: This must use the actual stride from the TensorView; this is a hack.
                    addrForHash += fields.m_redirection.m_index * stride;
                }
            }
            fields.m_valueAddrForHash = addrForHash;
            // WARNING: Expensive! Disable once it seems to work.
            //fail_if(!fields.m_isSparse && addrForHash != GetValueAddrForHash(CacheAndGetValue(fields)), "CacheAndGetValueAddrForHash computed wrong hash value");
            // Note: ^^ broken for sparse. May miss CSE.
        }
        return addrForHash;
#endif
    }

    // compute a hash value for f
    // This is called for nearly every unbatched PrimitiveFunction, and must therefore be blazingly fast.
    // TODO: instead of doing this lazily, should it be done in Schedule()?
    // TODO: ^^ no need to do this lazily actually. Only called once.
    static size_t ComputeAliasHash(PrimitiveFunction& f)
    {
        //if (f.m_autoBatchState.m_aliasHash != SIZE_MAX) // lazy  --TODO: no need; we don't need to even save this
        //    LogicError("ComputeAliasHash should never be called twice on the same object");
        //    //return;
        size_t hash = 0;
        // we only hash the argument identities; that is, their base pointer
        // We already know that opcode and shapes are identical.
        // If we really allow strides in the future, we may include them in the hash.
        let& inputs = f.m_inputs;
        let numInputs = inputs.size();
        //CudaStatsGuard cudaStatsGuard(PrimitiveOpType::ElementTimes, L"ComputeAliasHash()", 3, numInputs);
        for (size_t k = 0; k < numInputs; k++)
        {
            // TODO: just use object identity (follow redirects through !m_sliceRange) plus index
            let addrForHash = CacheAndGetValueAddrForHash(GetInputFields(inputs[k]));
            hash += (size_t)addrForHash;
            hash += (size_t)(addrForHash >> 8); // also hash the page number, in case allocation is page-aligned
            hash *= hashMultiplier;
        }
        //f.m_autoBatchState.m_aliasHash = hash;
        return hash;
    }

    // class to help deduplicating
    class DedupSet
    {
        static const size_t numBuckets = 65536;
        class Bucket
        {
            Bucket* m_nextBucket = nullptr;
            PrimitiveFunction* m_bucketList = nullptr; // list of items in this bucket, linked via m_bucketList
        public:
            // iterate over buckets themselves (note: not over the bucket *list*). This is used during cleanup.
            Bucket* ClearAndNext()
            {
                auto* next = m_nextBucket;
                m_nextBucket = nullptr;
                m_bucketList = nullptr; // abandon the list
                return next;
            }
            void CheckClean() const
            {
                fail_if(m_nextBucket || m_bucketList, "DedupSet bucket was not cleaned up upon last use");
            }
            // add an entry to the bucket list
            void PushFront(PrimitiveFunction* f, Bucket* &cleanupList)
            {
                if (!m_bucketList) // first time (bucket just became active): register the bucket for cleanup
                {
                    m_nextBucket = cleanupList;
                    cleanupList = this;
                }
                // insert f into the bucket
                f->m_autoBatchState.m_bucketList = m_bucketList;
                m_bucketList = f;
            }
            // use these to iterate over the entries
            PrimitiveFunction* Begin() const { return m_bucketList; }
            PrimitiveFunction* End() const { return nullptr; }
            void Next(PrimitiveFunction* &f) const { f = f->m_autoBatchState.m_bucketList; }
        };
        vector<Bucket> m_buckets = vector<Bucket>(numBuckets);
        Bucket* m_firstBucketInUse = nullptr; // for cleanup at the end: all active buckets are registered here
    public:
        // prepare for next use (clear out all entries)
        void Reset()
        {
            for (auto* bucket = m_firstBucketInUse; bucket; bucket = bucket->ClearAndNext())
                ;
            m_firstBucketInUse = nullptr;
            CheckClean();
        }
        // check that we are clean
        void CheckClean() const
        {
            fail_if(m_firstBucketInUse, "DedupSet was not cleaned up upon last use");
            //for (let& bucket : m_buckets) // expensive thorough check
            //    bucket.CheckClean();
        }
        // add to set and return nullptr, unless f is an alias of something that's already in the set; then return that
        PrimitiveFunction* FindDuplicateOrAddToSet(PrimitiveFunction* f)
        {
            let fHash = ComputeAliasHash(*f);
            let fIndex = fHash % numBuckets;
            auto* bucket = &m_buckets[fIndex]; // bucket for f
            // search for an alias in the list. Most of the same this will be a match, but we must confirm.
            let& inputs = f->m_inputs;
            let arity = inputs.size();
            for (auto* alias = bucket->Begin(); alias != bucket->End(); bucket->Next(alias)) // iterate over all entries with the same hash value
            {
                for (size_t k = 0; k < arity; k++)
                {
                    auto& fields      = GetInputFields(inputs[k]);
                    auto& aliasFields = GetInputFields(alias->m_inputs[k]);
                    if (&fields == &aliasFields)
                        continue;
                    // PERF BUGBUG: This vv is suboptimal as we force-realize the m_value, which is a slice. Alleviated by the hash though.
                    if (!CacheAndGetValue(fields)->IsAliasOf(CacheAndGetValue(aliasFields)))
                        goto try_next;
                }
                // all inputs are the same: f is a dup of 'alias'
                return alias; // was indeed an alias
            try_next:; // this bucket-list entry does not match
            }
            // no alias found: insert into the bucket
            bucket->PushFront(f, m_firstBucketInUse);
            return nullptr; // it was a new one
        }
    };
    DedupSet m_dedupSet; // TODO: make this a static thread local, to avoid reallocating the bucket list
    VisitorTag m_cseVisitorTag; // helper for CSE pre-check

    // pre-check whether the more expensive hash-table based CSE check is needed
    // If all inputs have either no dups or are all-dups, then there is no need for the more expensive hash-table based check.
    bool IsShortCircuitingBatchedOpDuplicatesNeeded(NonOwningFunctionList ops)
    {
        CudaStatsGuard cudaStatsGuard(PrimitiveOpType::NotEqual, L"CSE pre", 3, ops.size());
        let& f0 = *ops.front();
        let numArgs = f0.m_inputs.size();
        for (size_t i = 0; i < numArgs; i++)
        {
            m_cseVisitorTag.Begin();
            size_t numDups = 0;
            for (auto iter = ops.begin(); iter != ops.end(); ++iter) // create the batched tensors
            {
                if (m_visitorTag.Visited(GetInputFields(iter->m_inputs[i]).m_cseVisitedTag))
                    numDups++;
            }
            if (numDups != 0 && numDups != ops.size() - 1)
                return true; // pre-check discovered non-trivial duplicates
        }
        return false; // pre-check passed: no thorough check needed
    }

    // detect equivalent Functions and uniq them, redirecting dups to the first one
    // Returns a filtered list. Call this at start of batched execution.
    // TODO: Choose one name: ShortCircuit? Duplicate? Alias? Common subexpression?
    NonOwningFunctionList ShortCircuitBatchedOpDuplicatesAndUpdateSchedule(NonOwningFunctionList ops)
    {
        CudaStatsGuard cudaStatsGuard(PrimitiveOpType::ReconcileDynamicAxis, L"CSE", 3, ops.size());
        m_dedupSet.CheckClean(); // verify that we have cleaned up correctly
        NonOwningFunctionListBuilder filteredOps;
        for (auto iter = ops.begin(); iter != ops.end(); ) // create the batched tensors
        {
            auto& f = *iter;
            ++iter; // advance here, since we will reuse the m_link field
            // f has been taken out of the list, its m_link field is unused. If it is not a dup, then m_link will live in the filteredOps list instead.
#if 1
            auto* duplicate = m_dedupSet.FindDuplicateOrAddToSet(&f); // this is now O(1) in most cases
            if (!duplicate) // no matching one was found: start a new list, and return in filteredOps
            {
                // note that the above call has added f to the dedup table. It will be returned in the future for any alias we find.
                f.m_autoBatchState.m_aliasList = nullptr; // first entry in a potential list of duplicates
                filteredOps.push_back(&f); // now m_link is used again
            }
            else // duplicate: add f to the duplicate list (which must already be in filteredOps list)
            {
#if 1           // TODO: one day try if the #if 0 branch works. It should (but failed earlier on). Would allow to remove the brittle m_autoBatchState.m_aliasList.
                f.m_autoBatchState.m_link = (PrimitiveFunction*)-1; // (for good measure--it is not part of any m_link chain anymore)
                f.m_autoBatchState.m_aliasList = duplicate->m_autoBatchState.m_aliasList;
                duplicate->m_autoBatchState.m_aliasList = &f; // duplicate is the original anchor; it is the root of a linked list into the aliases
#else
                FinalizeBatchedOpAndUpdateSchedule(f, dynamic_pointer_cast<PrimitiveFunction>(jter->shared_from_this()));
#endif
                m_stats.numCommonSubexpressionsEliminated++;
            }
#else
            // PERF BUGBUG: This gives O(N^2) complexity. Fix this once I get correct behavior.
            //              For now, it seems using the hash with a linear search gets it fast enough, but we should use a proper hash table of course.
            let fHash = ComputeAliasHash(*f);
            f->m_autoBatchState.m_aliasHash = fHash;
            // PERF BUGBUG: no need to cache the hash, as it is only ever used on this very list
            //let fHash = f->m_autoBatchState.m_aliasHash;
            let& inputs = f->m_inputs;
            let arity = inputs.size();
            for (auto jter = filteredOps.begin(); jter != filteredOps.end(); ++jter)
            {
                let& jnputs = jter->m_inputs;
                if (jter->m_autoBatchState.m_aliasHash != fHash) // TODO: no need, we just fetch from hash table according to hash code (no need to check again, since very little clashes)
                    goto next_jter;
                for (size_t k = 0; k < jnputs.size(); k++)
                {
                    auto& fields = GetInputFields(inputs[k]);
                    auto& fjelds = GetInputFields(jnputs[k]);
                    if (&fields == &fjelds)
                        continue;
                    // PERF BUGBUG: This vv is suboptimal as we force-realize the m_value, which is a slice. Alleviated by the hash though.
                    if (!CacheAndGetValue(fields)->IsAliasOf(CacheAndGetValue(fjelds)))
                        goto next_jter;
                }
                // all inputs are the same: f is a dup of 'jter'
#if 1           // TODO: one day try if the #if 0 branch works. It should (but failed earlier on). Would allow to remove the brittle m_autoBatchState.m_aliasList.
                f->m_autoBatchState.m_aliasList = jter->m_autoBatchState.m_aliasList;
                jter->m_autoBatchState.m_aliasList = f; // jter is the original anchor; it is the root of a linked list into the aliases
#else
                FinalizeBatchedOpAndUpdateSchedule(f, dynamic_pointer_cast<PrimitiveFunction>(jter->shared_from_this()));
#endif
                m_stats.numCommonSubexpressionsEliminated++;
                goto break_iter; // gotcha! eliminated!
            next_jter:; // this one does not match
            }
            // no matching one was found
            f->m_autoBatchState.m_aliasList = nullptr; // first entry in a potential list of duplicates
            //f->m_autoBatchState.m_aliasHash = fHash; // TODO: later this comes out of the hash table itself
            filteredOps.push_back(f);
        break_iter:;
#endif
        }
        m_dedupSet.Reset(); // clean up after ourselves
        return filteredOps;
    }

    // determine the physical source and slice index of an input
    // If !originatingFunction then sliceRange.empty().
    static const struct { const PrimitiveFunction* originatingFunction; SliceRange sliceRange; } LazyPhysicalSlice(const VariableFields& fields)
    {
        auto function   = fields.m_redirection.m_function;
        auto sliceRange = SliceRange();
        // case 1: Placeholder or Constant
        if (!function)
            goto done;
        // case 2: one-level redirect
        sliceRange = fields.m_redirection.m_sliceRange;
        let& redirectedFields = GetOutputFields(*function);
        auto redirectedFunction = redirectedFields.m_redirection.m_function;
        if (redirectedFunction == function) // self: end of chain
            goto done;
        // case 3: multi-step redirections (these occur in composite inlining)
        // TODO: patch up the data structure upon first discovery
        for (;;)
        {
            function = redirectedFunction;
            let& redirectedSliceRange = redirectedFields.m_redirection.m_sliceRange;
            if (sliceRange.empty())
                sliceRange = redirectedSliceRange;
            else
                fail_if(!redirectedSliceRange.empty(), "LazyPhysicalSlice: hit a see-through slice??"); // multiple slicing not possible
            // TODO: ^^ multiple non-dropping slices could be composable here
            let& redirectedFields = GetOutputFields(*function);
            redirectedFunction = redirectedFields.m_redirection.m_function;
            if (redirectedFunction == function) // self: end of chain
                break;
        }
    done:
        fail_if(!function && !sliceRange.empty(), "LazyPhysicalSlice: sliceRange not empty if no redirect??");
        return{ function, sliceRange };
    }

    // subroutine to determine the max Rank() over f's m_inputs
    // Only use this for element-wise ops (and not, e.g., for Times or Convolution).
    size_t DetermineMaxElementwiseInputRank(const PrimitiveFunction& f)
    {
        let numArgs = f.m_inputs.size();
        size_t maxInputRank = f.m_inputs.front().Shape().Rank();
        for (size_t i = 1; i < numArgs; i++)
            maxInputRank = max(f.m_inputs[1].Shape().Rank(), maxInputRank);
        return maxInputRank;
    }

    // helper to determine the max Rank() over all inputs and outputs in composite's m_rootFunction
    // Auto-batching needs to know a common batch axis it can use throughout the entire BlockFunction.
    // It is determined as the max rank over all involved inputs and outputs.
    size_t CacheAndGetBasicBlockBatchAxis(const BlockFunction& block)
    {
        auto& composite = static_cast<CompositeFunction&>(*block.Composite());
        if (composite.m_basicBlockBatchAxis == SIZE_MAX) // not computed yet (has been reset in Invoke())
        {
            // start with the inputs
            size_t maxRank = DetermineMaxElementwiseInputRank(block);
            // now also maximize over all output shapes of all Functions inside the composite graph
            // BUGBUG: For expedience, we use PreorderTraverseFunctions(). However, this will incorrectly also
            //         traverse the weight arg of a MatrixProduct if it is the result of computation (a Function).
            //         For Matrix products whose weights are Parameters, the same error happens by using DetermineMaxElementwiseInputRank(),
            //         which only looks at the inputs of the composite and does not know whether an input is consumed by a matrix propduct.
            //         It is not really harmful, since in the worst case, the resulting batch axis higher than needed, which does not really cost anything.
            Function::PreorderTraverseFunctions(composite.RootFunction(), [&](const FunctionPtr& iter) // This is only done once per composite ever (across all minibatches), so it is OK to be slow.
            {
                let& f = static_cast<const PrimitiveFunction&>(*iter);
                if (f.m_op == PrimitiveOpType::Block) // nested block (may be shared, hence the value may already be there)
                    maxRank = max(maxRank, CacheAndGetBasicBlockBatchAxis(static_cast<const BlockFunction&>(f)));
                else
                {
                    // note: f may be Times or Convolution. Those require special-casing regarding their inputs, but not the output.
                    let& outputs = f.m_outputs;
                    if (outputs.size() != 1)
                        InvalidArgument("Invoke can only be used with composites that have a single output (this one contains a function with %d).", (int)outputs.size());
                    maxRank = max(maxRank, outputs.front().Shape().Rank());
                }
            });
            composite.m_basicBlockBatchAxis = maxRank;
        }
#if 1   // BUGBUG: This uses a hard-coded constant (12) to heuristically detect an uninitialized value. That is not general, so remove this once this stuff works.
        fail_if(composite.m_basicBlockBatchAxis == 0 || composite.m_basicBlockBatchAxis > 12, "m_basicBlockBatchAxis was not prepared??");
#endif
        return composite.m_basicBlockBatchAxis;
    }

    // temp variables for ExecuteBatchedOpAndUpdateSchedule(); keep outside to reuse the memory allocation
    vector<Variable> m_batchedInputs;
    vector<Variable> m_gatherArgsBuffer;

    // batch-execute a set of ops that are known to be batchable
    // For every batched operation, this generates a new PrimitiveFunction object for the op itself, and one
    // for a splice operation for each batched inputs.
    // I.e. this is not a full graph transform, but rather a graph augmentation, so that during backprop,
    // we can recover the batched operations, while the original graph does not get modified.
    // Any batched operation will generate its result in a dense tensor with a batch dimension.
    // The consumers of the original ops will get a back-reference in the m_redirection field.
    // If such a result is ever accessed individually, it will lead to a lazy NDArrayView::SliceView() call
    // (but no Splice Function object is used for this).
    void ExecuteBatchedOpAndUpdateSchedule(NonOwningFunctionList ops) // (note: NonOwningFunctionListBuilder is so small that it is best copied)
    {
        // get a representative op
        auto& f0 = *ops.front();
        let op = f0.m_op;
        let opClass = g_oscTable[op]; // operation-specific auto-batching class
        // fail on unsupported classes
        switch (opClass)
        {
        case OpSpecificConditionKind::Convolution:
            LogicError("ExecuteBatchedOpAndUpdateSchedule: Auto-batching of Convolution() not implemented yet.");
        case OpSpecificConditionKind::Pooling:
            LogicError("ExecuteBatchedOpAndUpdateSchedule: Auto-batching of Pooling ops not implemented yet.");
        case OpSpecificConditionKind::OptimizedRNNStack:
            LogicError("ExecuteBatchedOpAndUpdateSchedule: Auto-batching of OptimizedRNNStack() not implemented yet.");
        case OpSpecificConditionKind::RandomDistribution:
            LogicError("ExecuteBatchedOpAndUpdateSchedule: Auto-batching of RandomDistribution ops not implemented yet.");
        case OpSpecificConditionKind::NoOp:
        case OpSpecificConditionKind::Barrier:
            LogicError("ExecuteBatchedOpAndUpdateSchedule: Auto-batching of No-op attempted, should have been short-circuited before getting here.");
        case OpSpecificConditionKind::NotSupportedDynamicAxis:
            LogicError("ExecuteBatchedOpAndUpdateSchedule: Operations involving dynamic axes are not allowed in Dynamite (%S).", PrimitiveOpTypeName(op).c_str());
        case OpSpecificConditionKind::NotSupportedStaticGraph:
            LogicError("ExecuteBatchedOpAndUpdateSchedule: Operations specific for static graphs are not allowed in Dynamite (%S).", PrimitiveOpTypeName(op).c_str());
        case OpSpecificConditionKind::ToDo:
            LogicError("ExecuteBatchedOpAndUpdateSchedule: Auto-batching of this op not yet implemented (%S).", PrimitiveOpTypeName(op).c_str());
        case OpSpecificConditionKind::NotSupportedTempMem:
            LogicError("ExecuteBatchedOpAndUpdateSchedule: Primitive operations involving temp memory not supported in Dynamite. This op should have been implemented in a different way (%S).", PrimitiveOpTypeName(op).c_str());
        case OpSpecificConditionKind::Undefined:
            LogicError("ExecuteBatchedOpAndUpdateSchedule: Unexpected Undefined batching kind? (%S)", PrimitiveOpTypeName(op).c_str());
        }
        // different operation classes have to be treated differently in various aspects. These are all the special conditions:
        let isFree = IsViewOp(op); // (see-through ops except Slice() should have been short-circuited here except for a few boundary cases)
        let isTimes       = opClass == OpSpecificConditionKind::MatrixProduct;        // special case: first arg of matrix product cannot be batched --TODO: support batch GEMM --TODO: Convolution will also fall in this category
        let isBasicBlock  = opClass == OpSpecificConditionKind::BasicBlockInvocation; // special case: executing a basic block, where all ops are batched in lock-step
        let isElementWise = !isTimes && !isBasicBlock && opClass != OpSpecificConditionKind::Convolution;
        // "Element-wise" really means that the inputs and the output all share the same batch axis. Also e.g. for Splice. TODO: Rename?

        // common sub-expression elimination (CSE)
        // All common sub-expressions become available at the same time and show up in the same ops list.
        // All ops whose inputs are 100% aliases of another op compute the same thing.
        // Those will be removed from the list. The removed ones will have a result value implanted
        // that is a lazy view onto the non-removed one.
        // Batch norm must be excluded  since we must count samples as often as they appear in the batch statistics.
        // TODO: Merge the pre-check, CSE, and getting the batched args into one big loop.
        // BUGBUG: This must consider Block functions' composites.
        if (!isFree && op != PrimitiveOpType::BatchNormalization && ops.size() > 1 && IsShortCircuitingBatchedOpDuplicatesNeeded(ops))
            ops = ShortCircuitBatchedOpDuplicatesAndUpdateSchedule(ops);
        else
            for (auto iter = ops.begin(); iter != ops.end(); ++iter) // create the batched tensors
                iter->m_autoBatchState.m_aliasList = nullptr;
        // TODO: ^^ if the CSE can directly redirect, then this loop ^^ is no longer needed

        // perform the op
        let numBatchItems = ops.size();
        if (!isFree)
            m_stats.numBatchedLaunches++;
        let numArgs = f0.m_inputs.size();
        // is unbatched actually more expensive than batched?
        // We count CUDA launches, assuming all args are batched and require a Gather launch.
        // If not all args require that, we err on the side of not batching. BatchNorm must always be batched.
        let numCUDALaunchesInThisOp = 1; // TODO: for Block invocations, use the #nodes; or pick a number, such as 4
        let worthIt = numBatchItems * numCUDALaunchesInThisOp/*unbatched launches*/ > (numArgs + 1)/*batched launches*/ || op == PrimitiveOpType::BatchNormalization;
        //if (numBatchItems > 1 && !worthIt) // TODO: remove this message once I see it happen
        //    fprintf(stderr, "%S not worth it: %d vs. %d\n", PrimitiveOpTypeName(f0.m_op).c_str(), (int)numBatchItems, (int)numArgs + 1);
#ifdef NO_BATCHED_FORWARD
        auto doNaively = true;
#else
        let doNaively =
            isFree ||
            numBatchItems == 1 ||
            !worthIt;
#endif
        //fprintf(stderr, "%d %sexecuting %d instances of %S -> %S; %d batchable ops pending\n",
        //        isFree ? -1 : (int)m_stats.numBatchedLaunches,
        //        doNaively ? "" : "batch-",
        //        (int)numBatchItems, f0.OpName().c_str(), f0.m_outputs.front().Shape().AsString().c_str(),
        //        (int)m_schedule.numBatchableOpsPending());
        if (doNaively)
        {
            // for correctness testing of underlying mechanism, compute them without actual batching
            for (auto& f : ops)
            {
                // execute it
                if (f.m_op == PrimitiveOpType::Block)
                {
                    let fInlinedPtr = InlineAndMemoizeBatchedBasicBlock(static_cast<const BlockFunction&>(f), f.m_inputs, /*batchAxis=*/SIZE_MAX, /*batchSize=*/1/*dummy*/);
                    // implant the result
                    FinalizeBatchedOpAndUpdateSchedule(f, fInlinedPtr);
                }
                else
                {
                    MemoizeKnowableValueInArena(f, isFree);
                    // and notify consumers (which may schedule them)
                    NotifyOpsConsumersInputsAvailable(f);
                }
                // distribute value to all aliases
                UpdateDuplicatesAndUpdateSchedule(f);
            }
        }

        // === execute the batchable operations as a batch ===
        // This is where the magic happens.
        // Every resulting batched op consists of the following new operations:
        //  - a Splice() or Slice() for each input (e.g. 2 for a binary op)
        //  - a PrimitiveFunction that is the op itself
        //  - m_redirection entries that represent a "virtual" Slice() that is never created as a PrimitiveFunction object to saved mallocs.
        // As for resource management, m_redirection will hold a strong ref to the PrimitiveFunction;
        // and we will hack its input's m_outputComposite to hold a strong reference to the Splice() or Slice().
        // (This is a little ugly since m_outputComposite is meant to hold a CompositeFunction, but we misuse it
        // to hold a PrimitiveFunction.)
        else
        {
            // batch all arguments
            // TODO: see if this can be sped up using un-managed pointers (save lots of ref-counting); or even use GatherBatch lambda
            // The batch axis will be a new axis appended to all shapes.
            //  - For elementwise operations, its position must be the same for all inputs.
            //    The result's batch axis will also be in the same position.
            //    Thus, it may have padded singleton dims, e.g. for reductions or Index(), which may need to be removed.
            //    (This also holds for Slice(), if we one day decide to batch it.)
            //  - Splice() is like an elementwise op, but it may splice into a new axis.
            //    Hence, we preemptively insert one extra singleton axis into the inputs, and then append the batch axis.
            //  - For matrix products and convolution, the column axes is already sort of a batch axis. We append the batch axis to them.
            //    Our special matrix product may increase the number of axes. This is fine; the batch axis remains the respective trailing axis.
            //    TODO: Verify that mapRank is never counted from the back.
            m_batchedInputs.resize(numArgs);
            let& unbatchedOutputShape = f0.m_outputs.front().Shape();
            size_t i0;                   // index of first batched argument (1 for matrix product; 0 otherwise)
            size_t commonInputBatchAxis; // batch axis of all batched args (it is the same across all args)
            size_t outputBatchAxis;      // batch axis in output (may differ from commonInputBatchAxis for Times/Convolution)
            // TODO: We restrict this to a single shared batch axis, with exception of unbatched inputs (first Times arg).
            //       All batch axes must be the same across all inputs. Then we can always batch along those.
            //       If all shapes happen to be the same, then there is an optimization (batch into a new axis and then reshape).
            //       We probably want to make that a part of the Gather function, deep inside, and basically ignore here.
            //       So: Select a common batch axis for all, and batch along those.
            //       TODO: We still need a shift from inputs to output; Times() may shift the axis.
            if (isTimes)
            {
                // for matrix product, second arg is batched, and batch axes differ for arg and output. Just append one axis, respectively.
                commonInputBatchAxis = f0.m_inputs.back().Shape().Rank();
                outputBatchAxis      = unbatchedOutputShape      .Rank();
                i0 = 1; // first arg does not get batched (since that would no longer be a GEMM operation)  --PERF BUGBUG: implement batched GEMM as well, once we have virtual Gather!
            }
            else if (isBasicBlock)
            {
                outputBatchAxis = max(unbatchedOutputShape.Rank(), CacheAndGetBasicBlockBatchAxis(static_cast<const BlockFunction&>(f0)));
                commonInputBatchAxis = outputBatchAxis;
                i0 = 0; // all args participate in batching (non-elementwise ops are forbidden in basic blocks, with special-casing of Times)
            }
            else if (isElementWise)
            {
                // Elementwise ops may remove the last axis (e.g. InnerProduct()), or add axes (Splice, see-through Reshape).
                // So the batch axis must be the max over all inputs' and output's rank.
                // From the batching condition, we know that all input shapes are the same. So we only need to look at the first op instead.
                // TODO: We could also handle this by an implied AsShape() right before execution.
                outputBatchAxis = max(unbatchedOutputShape.Rank(), DetermineMaxElementwiseInputRank(f0));
                commonInputBatchAxis = outputBatchAxis;
                i0 = 0;                                 // all args participate in batching
            }
            else
                fail_if(true, "should not get here");

            // enable stacking; that is, along the last axis
            // This is tricky. E.g. must consider whether it's a scalar. For now we stack Times().
            // We must check the last axis over all inputs.
            let stackIt = isTimes && outputBatchAxis == 2; // TODO: correctly consider the mapRank
            if (stackIt)
            {
                outputBatchAxis--;
                commonInputBatchAxis--;
            }
            size_t batchSize = numBatchItems; // dimension of batch axis
            if (stackIt)
            {
                // this is for Times only for now
                batchSize = 0;
                for (let& f : ops) // create the batched tensors
                {
                    let& input = f.m_inputs.back();
                    batchSize += input.Shape().Dimensions().back();
                }
            }

            // create all the batched inputs by splicing along the batch axis
            // Special optimizations are taken if all elements are identical.
            bool anyBatchedInputs = false; // stays false if for all arguments, all respective batch-item arguments are identical
            if (i0 == 1) // Times(): the first arg has already been verified to be identical as part of the batchability condition
                m_batchedInputs[0] = f0.m_inputs.front();
            for (size_t i = i0; i < numArgs; i++)
            {
                // create splice args for this argument
                // allocate buffers to hold the arguments
                auto& gatherInputs = m_gatherArgsBuffer;
                gatherInputs.clear();
                if (gatherInputs.capacity() < numBatchItems)
                    gatherInputs.reserve(max(numBatchItems, 2 * gatherInputs.capacity()));
                // optimization: if all args are consecutive slices, then use a slice view instead
                let& pfields0 = GetInputFields(f0.m_inputs[i]);
                let isScalar = pfields0.m_shape.Rank() == 0;
                let redirectionPair0 = LazyPhysicalSlice(pfields0);
                let is0Redirected = redirectionPair0.originatingFunction != nullptr;
                // loop over all batched ops. Collect all inputs and classify if all the same or all consecutive.
                bool allSame = true;                                              // will be true if all are the same objects
                bool allConsecutiveSlices = !redirectionPair0.sliceRange.empty(); // will be true if all are consecutive index ops into the same batched result
                bool allSameShape = true;                                         // will be true if all shapes are the same
                CudaStatsGuard cudaStatsguard(PrimitiveOpType::Slice, L"gather batched args", 3, numBatchItems);
                size_t prevSliceEndIndex = 0; // running index
                for (let& f : ops) // create the batched tensors
                {
                    let& input = f.m_inputs[i];
                    let& pfields = GetInputFields(input);
                    let redirectionPair = LazyPhysicalSlice(pfields);
                    // optimization: if all args are the same, then don't batch
                    allSame = allSame &&
                        (&pfields == &pfields0 ||                           // same object
                         (is0Redirected && redirectionPair.originatingFunction == redirectionPair0.originatingFunction && redirectionPair.sliceRange == redirectionPair0.sliceRange)); // or same view
                    // ^^ we can remove this double check, and just compare m_function and m_sliceRange
                    // optimization: if all args are consecutive slices, then use a slice view instead
                    if (allConsecutiveSlices)
                    {
                        // optimization: if consecutive slices, then recover the original batched tensor
                        // TODO: Can we directly check the data buffer pointer?
                        fail_if(redirectionPair.sliceRange.IsSlice(), "stacking not yet supported");
                        allConsecutiveSlices =
                            redirectionPair0.originatingFunction == redirectionPair.originatingFunction && // function that creates the physical output (NULL if not a slice)
                            prevSliceEndIndex                    == redirectionPair.sliceRange.BeginIndex();
                        // TODO: Per Jon's suggestion, we can be a little loose here. For a variable-length
                        // scenario, we will loose entries in the middle. We can allow to keep a few around
                        // in garbage-in-garbage-out. If, say, there are additional 20% gap values, we just
                        // carry them forward, and ignore them when implanting the result.
                        prevSliceEndIndex = redirectionPair.sliceRange.EndIndex();
                    }
                    // optimization: if all shapes are the same then we batch, otherwise we stack (batching is faster in Gather)
                    allSameShape = allSameShape && (isScalar || pfields0.m_shape.Dimensions().back() == pfields.m_shape.Dimensions().back());
                    fail_if(allSameShape && pfields0.m_shape != pfields.m_shape, "shapes differ by more than the batch axis"); // (remove this check)
                    // append the input
                    gatherInputs.push_back(input);
                    // note: Variable is just two shared_ptrs, one being NULL; so this is cheap
                    // note: input is a regular Variable with regular ownwership rules (it does not come from inside here)
                }
                allConsecutiveSlices = allConsecutiveSlices && (prevSliceEndIndex == batchSize); // if consecutive so far then also check the end
                let allConsecutiveIndices = allConsecutiveSlices && allSameShape;
                if (!allSameShape)
                    Break;
                cudaStatsguard.Stop();
                // and splice
                if (allSame) // optimized case: all ops share the same operand: no need to batch them
                    // note: we assume strict broadcasting semantics here (if at least one input is actually batched)
                    m_batchedInputs[i] = gatherInputs[0];
                else if (allConsecutiveIndices) // they are consecutive slice views: can short-circuit as a slice view
                {
                    let& from  = redirectionPair0.originatingFunction;
                    fail_if(redirectionPair0.sliceRange.IsSlice(), "stacking not yet supported");
                    let  begin = redirectionPair0.sliceRange.Index();
                    let& output = from->m_outputs.front();
                    fail_if(!GetOutputFields(*from).m_value, "value not yet available??");
                    let& fromDims = output.Shape().Dimensions();
                    fail_if(fromDims.size() == 0, "slice view into batch has rank 0??");
                    let axis = fromDims.size() - 1;
                    if (begin == 0 && batchSize == fromDims.back()/*[axis]*/) // full range: just take it
                        m_batchedInputs[i] = output; // note: graph already has a strong ref to output elsewhere
                    else // sub-range: splice it by taking a slice view on the previously spliced batch
                    {
                        //CudaStatsGuard cudaStatsguard(PrimitiveOpType::Slice, L"re-batch", 3, numBatchItems);
                        // create a new PrimitiveFunction Slice()
                        vector<size_t> outputShape = fromDims; // determine output shape
                        outputShape.back()/*[axis]*/ = batchSize;
                        m_batchedInputs[i] = CreateAndMemoizeOpAsInput(PrimitiveOpType::Slice, vector<Variable>{ output }, outputShape,
                                                                       Dictionary(
                                                                           PrimitiveFunction::AttributeNameAxis       , Axis((int)axis) ,
                                                                           PrimitiveFunction::AttributeNameBeginIndex , (int)begin      ,
                                                                           PrimitiveFunction::AttributeNameEndIndex   , (int)(begin + batchSize)
                                                                       ),
                                                                       f0.m_name, f0.m_profiler, L"#"/*gatherInputs[0]*/,
                                                                       /*isFree=*/true);
                        // and that's our input to the batched operation
                        // TODO: Can this be done by directly messing with the Variable? Make a deep copy of the var object with begin/end slice?
                        //       Would avoid allocating a PrimitiveFunction, and make it easier to be short-circuited directly in backprop.
                    }
                    // if this op has a higher commonInputBatchAxis than the re-batched view, we must move the axis
                    // BUGBUG: (perf) Reshape incurs an unnecessary mem copy in Backprop
                    // BUGBUG: This seems never called in the MT case?
                    // TODO: Do this inside Gather, based on its output shape.
                    if (axis != commonInputBatchAxis)
                    {
                        CudaStatsGuard cudaStatsguard(PrimitiveOpType::Reshape, L"interm. reshape", 3, numBatchItems);
                        // TODO: Any way we can fold the reshape in using m_redirection? ... no need, this will become part of Gather.
                        let batchedInput = m_batchedInputs[i];
                        vector<size_t> outputShape = batchedInput.Shape().Dimensions(); // determine current shape
                        outputShape.insert(outputShape.end() - 1, commonInputBatchAxis - axis, 1);
                        // insert a Reshape() op
                        m_batchedInputs[i] = CreateAndMemoizeOpAsInput(PrimitiveOpType::Reshape, vector<Variable>{ batchedInput }, outputShape,
                                                                       Dictionary(),
                                                                       f0.m_name, f0.m_profiler, L"#,"/*gatherInputs[0]*/, /*isFree=*/true);
                        // and that's now really our input to the batched operation
                    }
                    anyBatchedInputs = true;
                }
                else // batch inputs are not consecutive: We must actually copy them together.
                {
                    CudaStatsGuard cudaStatsguard(PrimitiveOpType::Splice, L"batch", 3, numBatchItems);
                    let& input0Shape = CacheAndGetValue(gatherInputs[0])->Shape();
                    // create a new PrimitiveFunction Splice()
                    vector<size_t> outputShape; // determine output shape
                    outputShape.reserve(commonInputBatchAxis + 1);
                    outputShape = input0Shape.Dimensions();
                    //outputShape = gatherInputs[0]->Shape().Dimensions(); // TODO: no need to force-realize it here; should be done by MemoizeKnowableValueInArena()
                    outputShape.resize(commonInputBatchAxis, 1); // pad to commonInputBatchAxis
                    outputShape.push_back(batchSize);      // and add the batch axis
                    m_batchedInputs[i] = CreateAndMemoizeOpAsInput(PrimitiveOpType::Splice, vector<Variable>(gatherInputs), outputShape,
                                                                   Dictionary(PrimitiveFunction::AttributeNameAxis, Axis((int)commonInputBatchAxis)),
                                                                   f0.m_name, f0.m_profiler, L"#"/*gatherInputs[0]*/,
                                                                   /*isFree=*/false, /*spliceIsGather=*/true);
                    // and that's our input to the batched operation
                    anyBatchedInputs = true;
                }
                // release shared_ptrs asap, to take advantage of chances of memory reuse
                gatherInputs.clear();
            } // end for (i=...)
            // m_batchedInputs[] have been prepared. anyBatchedInputs has remained false if all had identical inputs across all batch items.

            // special case: BatchNormalization
            if (op == PrimitiveOpType::BatchNormalization)
            {
                // BatchNorm requires three additional parameters for the current mean and invStdDev, and the zero-mean/unit-variance intermediate. These must be kept for backprop.
                // This is sort of a hack for now. It is not, however, an efficiency problem since there are relatively few batched BatchNorm nodes in the graph.
                let& statShape = m_batchedInputs[1].Shape(); // note: This is guaranteed to have no batch axis, since they are identical across all instances in this batched op
                let device = GetValueObject(m_batchedInputs[0])->Device();
                let createParameter = [&](const NDShape& shape) -> Variable // helper to create a Parameter as if it had been initialized by RBuildForwardGraphAndSchedule()
                {
                    let p = Parameter(m_arena.NewNDArrayView(shape, m_batchedInputs[0].GetDataType(), StorageFormat::Dense, device));
                    auto& fields = GetInputFields(p);
                    //fields.m_redirection.m_function = nullptr; // it is already null after creation
                    fail_if(fields.m_redirection.m_function != nullptr, "Parameter redirect not initialized??");
                    //fields.m_redirection.m_index = SIZE_MAX;
                    // initialize m_consumers chain of function that produces the values
                    //fields.m_consumers.mangle(-5); // (temporarily, so that we can discover if this is ever used)
                    return p; // TODO: change to return Parameter...
                };
                m_batchedInputs.push_back(createParameter(               statShape)  );
                m_batchedInputs.push_back(createParameter(               statShape)  );
                m_batchedInputs.push_back(createParameter(m_batchedInputs[0].Shape()));
                anyBatchedInputs = true; // BUGBUG: If all operands are the same, then BatchNorm does not make sense (variance=0). Should we throw an error?
            }

            // execute the operation and implant the results
            // Batched inputs have been prepared in m_batchedInputs[].
            // If all inputs are identical then degrade to computing it only once (this is the easy case that we don't kick off the CSE machinery for).

            if (!anyBatchedInputs)
                for (size_t i = 0; i < numArgs; i++)
                    fail_if(&GetInputFields(m_batchedInputs[i]) != &GetInputFields(f0.m_inputs[i]), "all batch args the same, but not??");

            // >>> This is the actual batched op that we create and execute here. <<<
            // A new PrimitiveFunction is created for the batched op, so that we can backprop through it.
            // If f0 is a block, then actually an entire subgraph clone will be created here.
            // PERF BUGBUG: If all args are identical, we can degrade to a single op. Then we should not need to create a new PrimitiveFunction.
            //              Note: This is not covered by CSE, since CSE is only used for complex cases.
            auto batchedOp = CreateAndMemoizeBatchedOp(f0, m_batchedInputs, anyBatchedInputs ? outputBatchAxis : SIZE_MAX, batchSize, L"*"/*f0*/, /*isFree=*/false);

            // some stats and checks
            if (!anyBatchedInputs)
                m_stats.numCommonSubexpressionsEliminated += numBatchItems - 1;
            if (anyBatchedInputs)
                fail_if(batchedOp->m_outputs.front().Shape().Rank() != outputBatchAxis + 1, "outputBatchAxis was not predicted right");

            // in case of reducing operations (e.g. ReduceSum() and also Index()), additional singleton axes
            // may have been inserted to align the batch axes. Remove these if present.
            // BUGBUG: (perf) Reshape incurs an unnecessary mem copy in Backprop   --TODO: does it? Verify.
            if (anyBatchedInputs)
            {
                if (outputBatchAxis != unbatchedOutputShape.Rank()    && !stackIt/*for now*/)
                {
                    CudaStatsGuard cudaStatsguard(PrimitiveOpType::Reshape, L"interm. reshape", 3, numBatchItems);
                    // TODO: This should not be necessary anymore, we can let this be implicitly handled by a redirect.
                    fail_if(!isElementWise && !isBasicBlock, "output shape should only have additional singleton axes for elementwise operations or basic-block invocations");
                    // insert a Reshape() op to remove the axes
                    let batchedOutputShape = unbatchedOutputShape.AppendAxis(unbatchedOutputShape.Rank(), batchSize); // desired batched output shape without the singleton axes
                    fail_if(batchedOutputShape.TotalSize(/*check=*/false) != batchedOp->m_outputs.front().Shape().TotalSize(/*check=*/false), "output shape has unexpected axes that should be singletons but aren't");

                    Variable arg = batchedOp->m_outputs.front();
                    arg./*m_outputComposite*/m_acyclicOutputPrimitiveReference = batchedOp;

                    // Reshape() here does not need the properties at this level anymore; output shape is sufficient
                    let reshapeOp = CreateAndMemoizeOp(PrimitiveOpType::Reshape, vector<Variable>{ arg }, batchedOutputShape, Dictionary(), f0.m_name, f0.m_profiler, L"*,"/*arg*/, /*isFree=*/true);

                    batchedOp = reshapeOp; // this is the result that we redistribute from to the individual consumers
                }
            }

            // implant all results in the original unbatched operations (all as lazy/virtual references through m_redirection)
            // TODO: review this w.r.t. multi-output functions
            if (anyBatchedInputs && !stackIt)
            {
                size_t j = 0;
                for (auto& f : ops)
                {
                    // implant the result
                    FinalizeBatchedOpAndUpdateSchedule(f, batchedOp, SliceRange(j));
                    // and implant it to all aliases as well
                    UpdateDuplicatesAndUpdateSchedule(f);
                    // iterate the slice index
                    j++;
                }
            }
            else if (anyBatchedInputs && stackIt)
            {
                size_t prevSliceEnd = 0;
                for (auto& f : ops)
                {
                    let width = f.m_inputs.back().Shape().Dimensions().back();
                    // implant the result
                    let newSliceEnd = prevSliceEnd + width;
                    FinalizeBatchedOpAndUpdateSchedule(f, batchedOp, SliceRange(prevSliceEnd, newSliceEnd));
                    // and implant it to all aliases as well
                    UpdateDuplicatesAndUpdateSchedule(f);
                    // iterate the slice index
                    prevSliceEnd = newSliceEnd;
                }
            }
            else // no batched inputs
            {
                for (auto& f : ops)
                {
                    // TODO: review this w.r.t. multi-output functions
                    // implant the result
                    FinalizeBatchedOpAndUpdateSchedule(f, batchedOp);
                    // and implant it to all aliases as well
                    UpdateDuplicatesAndUpdateSchedule(f);
                }
            }
            // To keep the batchedOp ref count alive, FinalizeBatchedOpAndUpdateSchedule() saves the shared_ptr in all m_redirection.m_functionHolder.

            // release the ref counts on the batched inputs; but keep the vector's memory allocated
            m_batchedInputs.clear();
        }
    }

public:
    // -----------------------------------------------------------------------
    // BatchedForward() -- entry point for auto-batched implementation of PrimitiveFunction::Value()
    // -----------------------------------------------------------------------

    // Value(), computed with automatic batching
    // This (and BatchedBackward()) represent the graph via the following structures that overlay the CNTK API structures:
    //  - Variable:
    //     - never look at Variable itself unless it has no m_redirection.m_function, except for:
    //        - m_value: if not NULL then this is the cached value after applying m_sliceRange and a potential reshape
    //        - m_shape: the desired shape of this variable, if different from result of op/m_sliceRange
    //     - m_redirection.m_function     -> PrimitiveFunction (nullptr for leaves, that is, Parameter and Constant)
    //     - m_redirection.m_sliceRange    --if not empty then index or slice the last dimension with this
    //     - m_depthHint  --this value should only be consumed when there is no ready op that consumes a smaller m_depthHint (0=none specified)
    //  - PrimitiveFunction:
    //     - m_inputs[] -> Variable.m_redirection
    //     - m_output -> Variable.m_value, m_gradient
    //     - m_op, m_attributes
    // The value of a Variable is computed by this sequence:
    //  - execute m_function
    //  - get the result from m_function->m_output.m_dataFields->m_value (unless m_function->m_output == this)
    //  - slice it according to m_lazySlice if not (*,SIZE_MAX)
    //  - reshape it to match this Variable's m_shape
    //  - store it in this Variable's m_value field
    // Here, m_function may be not the same as Owner() but a redirect, e.g. to a higher-up see-through, or a batched operation.
    // Unlike the CNTK API structures, m_redirection is not immutable. I.e. we can optimize the graph.
    // Batching relies on this and replaces unbatched m_functionHolder with batched ones.
    // Naked pointers are used, assuming that ref counting is handled by the CNTK API structures.
    // This routine uses temporary fields that are assumed initialized in a specific way:
    //  - PrimitiveFunction::m_autoBatchState.m_pendingInputs:
    //     - #inputs that still need to be computed before a node's value can be computed
    //  - Variable::m_consumers:
    //     - set of consumers of this value. Used to count m_autoBatchState.m_pendingInputs.
    // plus more temp fields:
    //  - PrimitiveFunction::m_autoBatchState.m_link: pointer to next PrimitiveFunction in the same batchable op
    // And it leaves the following:
    //  - m_value: updated as desired
    //    TODO: values not needed by user or gradient should use scratch space
    //  - m_redirection: if a slice or view came from a batched operation, this points to it
    //     - Any newly created batched ops are referenced this way.
    //
    // Another take at writing this up:
    //  - we distinguish input and output Variables
    //  - output Variable = f_cuda(input Variables)          // "physical" variable, holds data (m_value contains data produced by the owner)
    //  - input Variable = f_seeThrough(output Variable)     // "virtual" variable, holds no data (m_value is a lazily cached view into something else)
    //    where f_seeThrough = Barrier >> Slice >> Reshape (those do not involve CUDA, and are thus mere CPU overhead)
    //    Slice ops here are only generated by the auto-batched, not by the user.
    //  - input Variable = output Variable if no see-through op between them
    //  - mapping from input to the outputs they depend on is done through Variable::m_redirect, wth
    //    output Variable = input Variable -> m_redirection -> m_function -> m_outputs.front()
    //    If no see-through ops, then m_outputs.front() points right back to input Variable we came from
    //  - muliple see-through ops can chain as a result of auto-batching
    //     - at most one Slice in a chain, since Slice (into a batched op) is onlty generated on top of a real
    //       CUDA op
    //     - in backprop, we further short-circuit these into one
    NDArrayViewPtr BatchedForward(const Variable& v)
    {
        auto& fields = GetInputFields(v);
        // if value already there then just return it
        if (fields.m_value)
            return fields.m_value;
#ifdef LOG_DETAILS
        Function::PreorderTraverseFunctions(v.OutputOwner(), [&](const FunctionPtr& f) { LogFunction(dynamic_cast<PrimitiveFunction&>(*f), L"r "); });
#endif
        ResetCudaStats();
        // phase 1 (all done in the same function call):
        //  - create our graph overlay, esp. short-circuit see-through ops
        //  - mark all nodes w.r.t. how many inputs they are waiting for before being computable
        //  - prepare and schedule first set
        auto* cudaStatsPtr = BeginCudaStats(PrimitiveOpType::LabelsToGraph/*misusing this for graph initializing*/, L"forward init", 3);
        m_visitorTag.Begin();
        RBuildForwardGraphAndSchedule(v, 0);
        EndCudaStats(cudaStatsPtr);
        // phase 2:
        //  - compute the entire graph
        while (!m_schedule.empty()) // main computation loop over all operations
        {
            // select the "best" amongst the scheduled op batches
            //  - scheduled = ready for computation, all inputs available
            //  - "best" = heuristically chosen to be best executed in batch
            auto opBatch = m_schedule.pop_best();
            // log (if barrier crossed)
            let& f0 = *opBatch.front();
            if (ShouldProfile(f0)) // profiling diagnostics
            {
                let& inputs = f0.m_inputs;
                for (size_t i = 0; i < inputs.size(); i++)
                {
                    let& input = inputs[i]; // we are lazy and only print the name if the barrier is the immediate input, so that we don't have to duplicate the traversal
                    let& inputFields = GetInputFields(input);
                    let depthHint = inputFields.m_redirection.m_depthHint;
                    if (depthHint != 0)
                    {
                        const wchar_t* name = nullptr;
                        if (!inputFields.m_redirection.empty())
                        {
                            let& f = *inputFields.m_redirection.m_function;
                            name = f.Name().c_str();
                        }
                        //fprintf(stderr, "\n[%S] --- %d (%S): %d pending\n\n", f0.m_profiler->Name(), (int)depthHint, (name && name[0]) ? name : L"?", (int)m_schedule.BarrierPendingCounts(depthHint));
                        fprintf(stderr, "\n[%S] --- %d (%S)\n\n", f0.m_profiler->Name(), (int)depthHint, (name && name[0]) ? name : L"?");
                    }
                }
            }
            // execute it, and also update all outputs' values and consumers, and the schedule
            CudaStatsGuard cudaStatsGuard(PrimitiveOpType::Block/*misusing this for actual op*/, L"forward execute", 3);
            ExecuteBatchedOpAndUpdateSchedule(opBatch);
        }
        CacheAndGetValue(fields); // force-flush a potential final lazily-indexed value
        fail_if(!fields.m_value, "BatchedForward process did not produce a value??");
        // log stats
        // TODO: clean this all up, also the SyncDevice() function which no longer does what its name says.
        ShowCudaStats();
#ifdef LOG_STATS
        fprintf(stderr, "BatchedForward: %d forward ops executed besides %d gathers, %d views, and %d CSEs, in nominally %d+%ds ops (%d inlined) on %d known values\n",
                (int)m_stats.numDoneOtherOps, (int)m_stats.numDoneGatherOps, (int)m_stats.numDoneFreeOps, (int)m_stats.numCommonSubexpressionsEliminated,
                (int)m_stats.numOpNodes, (int)m_stats.numShortCircuitedNodes, (int)m_stats.numInlinedBlocks,
                (int)m_stats.numLeafNodes);
#endif
#ifdef LOG_DETAILS
        size_t numOpNodes1 = 0;
        Function::PreorderTraverseFunctions(v.OutputOwner(), [&](const FunctionPtr&) { numOpNodes1++; });
        if (numOpNodes1 != m_stats.numOpNodes)
            fprintf(stderr, "BatchedForward: short-circuited %d aliases\n", (int)(numOpNodes1 - m_stats.numOpNodes));
#endif
        fail_if(fields.m_value->IsSparse() != fields.m_isSparse, "NDArrayView::m_sparse mismatches VariableFields::m_sparse??");
        return fields.m_value;
    }

    // =======================================================================
    // backward-related functions
    // =======================================================================

    static StorageFormat DetermineGradientStorageType(const PrimitiveFunction& f, size_t index)
    {
        //if (f.m_op == PrimitiveOpType::Times && f.m_inputs[1].Shape()[0] == 2000)
        //    fprintf(stderr, "%S --> %d\n", f.m_inputs[1].Shape().AsString().c_str(), (int)f.m_inputs[1].IsSparse());
        // Special case for DENSE * SPARSE -> DENSE, which leads to a SPARSE gradient for input0 (common for embedding).
        if (f.m_op == PrimitiveOpType::Times && index == 0 && f.m_inputs.back().IsSparse())
            return StorageFormat::SparseBlockCol;
        else
            return StorageFormat::Dense;
    }

    // return value of CacheAndGetGradientView()
    struct NewGradient
    {
        NDArrayViewPtr view; // view into gradient
        double beta;         // 0 or 1. If 0 then gradient is uninitialized virgin memory; if 1 then gradient already existed and must be added into.
    };

    // get the gradient view for a function's input, and allocate memory for m_gradient if needed
    // This lazily sets up the gradient for an input variable, and returns it.
    // The physical gradient object is held by the redirect, while the cached gradient object in the input itself may be a view.
    // Returns beta = 0 if gradient was newly created, otherwise 1.
    // BIG BUGBUG: What this atchitecture does not handle is see-through gradients (e.g. Plus). That must be fixed. It will change quite a bit.
    NewGradient CacheAndGetGradientView(const Variable& input, StorageFormat format = StorageFormat::Dense)
    {
        auto& inputFields = GetInputFields(input); // describes the input variable, e.g. shape. May be a redirect

        double beta = 1.0;

        // if cached gradient object exists then return it, & done.
        if (inputFields.m_gradient)
            return{ inputFields.m_gradient, beta };

        // no cached gradient
        // if we don't even have a physical one, then create the that one
        auto& gradFields = GetGradientFieldsForBackprop(inputFields); // describes the physical function output. This is where the gradient lives physically.
        if (!gradFields.m_gradient)
        {
            // create a new one
            // TODO: allocate parameters as separate objects; and allow user to pass buffers in
            gradFields.m_gradient = m_arena.NewNDArrayView(gradFields.m_shape, gradFields.m_dataType, format, gradFields.m_value->Device());
            beta = 0.0; // has not been initialized (random section in arena)
            // if there is no redirect, then writing to the physical one is the same as writing to the cached one. Then we are done.
            if (inputFields.m_gradient)
                return{ inputFields.m_gradient, beta };
        }

        // we now have a physical one, but no cached view
        // Reminder: cached view = physical  value |> Barrier >> Slice >> Reshape
        // We must do this backwards. 
        let sliceRange = inputFields.m_redirection.m_sliceRange;
        auto gradient = gradFields.m_gradient;
        // slice and reshape if needed
        if (!sliceRange.empty()) // it's a slice: gradient is a slice view into from's output gradient
        {
            if (beta == 0.0) // gradient is fresh: explicitly reset all (since we are slicing into the input gradientm, we cannot use the beta mechanism)
            {
                gradient->SetValue(0.0f);
                m_stats.numBackpropSetZeroes++;
                beta = 1.0;
            }
            gradient = gradient->SliceViewAsShape(sliceRange.BeginIndex(), sliceRange.EndIndex(), inputFields.m_shape);
        }
        else // no slice
            ReplaceWithReshapedViewIfNeeded(gradient, inputFields.m_shape);
        // implant the cached gradient
        inputFields.m_gradient = move(gradient);
        return{ inputFields.m_gradient, beta };
    }

    // recursively traverse the tree hanging off a Variable and build the m_consumer fields
    // This propagates in depth-first order like a naive backprop, but only for the purpose
    // of recording consumers of each node.
    //
    // Unlike forward prop, we...
    //  - can skip any branch that does not need a gradient (!m_needsGradient and StopGradient ops).
    //  - short-circuit into batched ops (m_redirection) so that we backprop through them instead
    //  - short-circuit sequences of see-throughs (see-through on top of slice into batched result)
    // Unlike forward prop, we should also fursther optimize certain backprop aggregations...
    //  - matrix products
    //  - splice (-> scatter)
    //
    // Backprop is orchestrated by first creating a reverse graph via m_consumers,
    // where output Variables know all the consumers that they receive gradients from.
    // The output Variables are the units of operation. More precisely, Variables that hold their own data.
    // See-through ops are short-circuited on the graph level, with reshape and slice happening on the way.
    // The short-circuiting *mutates* the graph in-place (m_redirection fields) where such situation is detected.
    //
    // During computation, we backprop from an output Variable through the function, into target
    // gradient NDArrayViews that represent the short-circuited
    //and perform the
    //// implied reshape
    //"input Variable" through the
    // see-through ops (Reshape << Slice << Barrier) into an "output Variable," from there
    // through a CUDA op (m_function), and from there into the next level of "input Variables."
    //
    // All nodes that were traversed have all input's m_consumers set up.
    // Caller must call m_visitorTag.Begin() first. This routine uses two visitor tags, one in the
    // function, and one in the fields.
    //
    // Each iteration updates the m_consumer fields of the inputs leading to this var.
    // Precondition: Call this only once per redirection target and once per leaf.
    void RBuildBackwardGraph(VariableFields& gradFields, bool userOwnsGradients = false)
    {
        fail_if(gradFields.m_varKind == VariableKind::Input || gradFields.m_varKind == VariableKind::Placeholder, "unexpectedly encountered an Input or a Placeholder??"); // (should have been caught in forward)
        fail_if(!gradFields.m_needsGradient, "unexpectedly encountered a node with m_needsGradient=false??");
        //fail_if(m_visitorTag.Visited(gradFields.m_visitedTag), "RBuildBackwardGraph called multiple times on the same node??"); // naw, can't test it this way

        // initialize the upwards graph links. These form the graph structure.
        gradFields.m_consumers.clear();
        // this arg will receive a gradient; reset it (later, we *accumulate* into it since nodes can receive gradients from multiple consumers)
        // note: must reset in case the user calls Backward() multiple times with different roots that share subgraphs.
        if (!userOwnsGradients) // user-owned gradients are those passed in by the user
            gradFields.m_gradient.reset();
        if (gradFields.m_gradient) // if one was passed in, clear it to zero   --TODO: use a dirty flag, and use beta=0 for this instead
        {
            gradFields.m_gradient->SetValue(0.0);
            m_stats.numBackpropSetZeroes++;
        }
        // done initializing this node. Now on to its inputs.

        // handle leaves
        if (gradFields.m_redirection.empty()) // has no owner/producer function: it's a Parameter or Constant
        {
            // this leaf will receive a gradient; zero it out if one is already present (in case user passes in the buffer)
            fail_if(gradFields.m_varKind != VariableKind::Parameter && gradFields.m_varKind != VariableKind::Constant, "backprop through a see-through op??");
            return;
        }

        // Determine the function that 'output' is the output of (seeing through see-through ops).
        // We back-propagated through the modified graph that is established by the m_redirection
        // fields. This means that if a value was not
        // actually computed from its true owner function, but rather a slice into the result of
        // a batched operation, then we traverse through that batched operation
        // instead. As a consequence, it is the batched operation that will be recorded as the
        // consumer of all inputs of the batched operation, rather than the original
        // unbatched operation. And as a consequence of that, back propagation will use
        // the same batching that was determined in forward computation. We do not need to rediscover it.
        auto& f = *gradFields.m_redirection.m_function;

        //if (f.m_op == PrimitiveOpType::Block)
        //    Break;
        //if (f.m_op == PrimitiveOpType::NoOp)
        //    fprintf(stderr, "->%d\n", (int)f.m_uniqueIdForDebugging), fflush(stderr);
        //
        //if (f.m_uniqueIdForDebugging == 368869)
        //    Break;

        fail_if(&GetOutputFields(f) != &gradFields, "RBuildBackwardGraph called on a redirection??");
        fail_if(!gradFields.m_value, "variable has no value yet??");

        fail_if(m_visitorTag.Visited(f.m_autoBatchState.m_visitedTag), "RBuildBackwardGraph: registering the same function twice??"); // should have been caught by gradFields' visitedTag

        fail_if(f.m_op == PrimitiveOpType::StopGradient, "unexpectedly encountered a StopGradient, which should have propagated m_needsGradient=false upwards"); // TODO: needsGradient handling

        // recursively process f's inputs, and register f as a consumer of all of its inputs
        let& inputs = f.m_inputs;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            auto& inputGradFields = GetGradientFieldsForBackprop(ResetInputGradient(inputs[i]), /*firstTimes=*/true); // this is where the gradient will be held   --TODO: also cache the reshaped/sliced gradient in input fields
            //if (inputGradFields.m_uniqueIdForDebugging == 243)
            //    Break;
            if (!inputGradFields.m_needsGradient) // TODO: use our own field for this. Interpret Constant and StopGradient. StopGradient output receives no gradient.
                continue; // skip inputs that receive no gradients
            // process recursively the inputs
            if (!m_visitorTag.Visited(inputGradFields.m_visitedTag))
                RBuildBackwardGraph(inputGradFields);
            // record ourselves as a consumer of the arg
            //if (inputGradFields.m_uniqueIdForDebugging == 243)
            //    Break;
            inputGradFields.m_consumers.push_back({ &f, i });
            m_stats.numBackpropsToInputs++;
        }
        // BUGBUG: Why is this count significantly higher (4x) than the number of batched forward ops? Are we missing batched forward ops?
        m_stats.numBackpropsThrough++;
    }

    // test whether 'fields' is a physical output of its m_function
    static bool ArePhysicalOutputFields(const VariableFields& fields)
    {
        return &GetOutputFields(*fields.m_redirection.m_function) == &fields;
    }

    // determine the dataFields that hold the gradient for an input (possibly through a redirection)
    // This is a complex one:
    //  - determine the output fields for an input
    //  - if first time (during graph building), then possibly monkey-patch input's m_redirection to short-circuit an additional slice into a batched output
    //    (and if not, then verify that this has been done)
    VariableFields& GetGradientFieldsForBackprop(const Variable& input, bool firstTime = false)
    {
        auto& inputFields = GetInputFields(input);
        return GetGradientFieldsForBackprop(inputFields, firstTime);
    }
    VariableFields& GetGradientFieldsForBackprop(VariableFields& inputFields, bool firstTime = false)
    {
        if (inputFields.m_redirection.empty()) // leaf
            return inputFields;
        auto& gradFields = GetOutputFields(*inputFields.m_redirection.m_function);
        //if (inputFields.m_redirection.m_function->m_uniqueIdForDebugging == 368869)
        //    Break;
        //if (gradFields.m_redirection.m_function->m_uniqueIdForDebugging == 368869)
        //    Break;
        fail_if(gradFields.m_redirection.empty(), "output Variable is a leaf??");
        //fail_if(inputFields.m_redirection.m_function->m_op == PrimitiveOpType::Block, "unexpanded Block invocation??");
        // short-circuit if needed
        if (ArePhysicalOutputFields(gradFields)) // a physical Variable
            return gradFields;
        // if not firstTime then we should not get here
        fail_if(!firstTime, "GetGradientFieldsForBackprop (not firstTime): hit a see-through slice??");
        // move up the m_redirection into 'input', overwriting the current one
        fail_if(!inputFields.m_redirection.m_sliceRange.empty(), "GetGradientFieldsForBackprop (firstTime): short-circuiting a see-through slice??"); // can only handle one slice per chain
        inputFields.m_redirection = gradFields.m_redirection; // replace current redirect with the down-stream one
        // and try again
        return GetGradientFieldsForBackprop(inputFields, firstTime/*=true*/); // (note: tail recursion)
    }

    // helper during initialization: reset m_gradients if it is redirected
    VariableFields& ResetInputGradient(const Variable& input)
    {
        auto& inputFields = GetInputFields(input);
        if (!inputFields.m_redirection.empty())
        {
            auto& gradFields = GetOutputFields(*inputFields.m_redirection.m_function);
            if (&gradFields != &inputFields)
                inputFields.m_gradient.reset();
        }
        return inputFields;
    }

    // helper to batch an array of NDArrayViews of the same rank along either the last or into a new axis
    // TODO: do this with a lambda so we can go straight into gatherBatchResultDims
    NDArrayViewPtr GatherBatchInArena(const vector<NDArrayViewPtr>& inputValues, size_t axis, size_t batchDim)
    {
        let& inputValue0 = *inputValues[0];
        let& inputShape = inputValue0.Shape().Dimensions();
        auto& gatherBatchResultDims = BorrowBuffer(m_dimsBuffer, inputShape.size()+1);
        gatherBatchResultDims.assign(inputShape.begin(), inputShape.end());
        fail_if(axis + 1 < gatherBatchResultDims.size(), "axis not trailing??");
        if (axis == gatherBatchResultDims.size())
            gatherBatchResultDims.push_back(inputValues.size()); // batching
        else
            gatherBatchResultDims[axis] = batchDim; // stacking
        auto out = m_arena.NewNDArrayView(gatherBatchResultDims, inputValue0.GetDataType(), inputValue0.GetStorageFormat(), inputValue0.Device());
        m_stats.numBackpropGathers++;
        return move(NDArrayView::GatherBatch(inputValues, (int)axis, move(out)));
    }

    typedef Internal::AutoBatchConsumer AutoBatchConsumer; // TODO: shouldn't this be done with using?

    // check whether an input's gradient is identical to the output's gradient, and can therefore be viewed
    // This only considers the nature of the operation; that is, its opcode, which input, and whether there's a shape change (indicating broadcasting).
    // It does not consider whether the gradient is already there.
    static bool IsOpGradientViewable(const PrimitiveOpType op, size_t inputIndex, const VariableFields& outputFields, const Variable& input)
    {
        let isPlusOrMinus =
            op == PrimitiveOpType::Plus ||
            (op == PrimitiveOpType::Minus && inputIndex == 0);
        bool doesOpAllowIt =
            isPlusOrMinus ||
            op == PrimitiveOpType::Reshape;  // explicit reshape, e.g. generated by auto-batcher  --TODO: change to implicit reshape instead
        if (doesOpAllowIt)
            // Verify that the op does not broadcast or reduce, which may happen for Plus or Minus.
            return
                !isPlusOrMinus ||                            // no need to check for broadcasting/reduction
                outputFields.m_shape == GetInputFields(input).m_shape; // Plus or Minus: must check
        else
            return false;
    }

    // back-propagate f's outputs' m_gradient to a specified input
    // This is the standard path for all unbatched ops.
    // This wraps the PrimitiveFunction's BackpropTo(), interfacing from vectors of Variable to vectors of NDArrayViewPtr.
    // Note that each input that is redirected should redirect the gradient into a slice in its lazy source.
    // If the target only has one consumer, pass viewAllowed. This will allow views for trivial gradients such as Plus.
    void BackpropToUnbatched(const AutoBatchConsumer& fi, bool viewAllowed)
    {
        let& f = *fi.first;
        let index = fi.second;
#ifdef LOG_DETAILS
        LogFunction(*f, L"bb ", index);
#endif
        // function's forward output and received gradient from top live here
        let& outputFields = GetOutputFields(f); // result of f lives here; hence also the gradient we back-propagate
        fail_if(!outputFields.m_value,    "unexpectedly ran into a function that has no m_value yet??");
        fail_if(!outputFields.m_gradient, "unexpectedly ran into a function that has no m_gradient yet??");

        // get the TensorViews for the forward inputs to this function
        // TODO: Can we know which ones are actually needed? We could save some time. This info would be shared with a memory-sharing mechanism.
        let& inputs = f.m_inputs;
        let numInputs = inputs.size();
        auto& inputValues = BorrowBuffer(m_inputValuesBufferRaw, numInputs);
        for (size_t i = 0; i < numInputs; i++)
            inputValues[i] = GetCachedValue(inputs[i]).get();

        // get the gradient view for the input whose gradient we desire
        // If it was newly created, then gradient.beta will be 0
        let& input = inputs[index];
        fail_if(!GetInputFields(input).m_needsGradient, "function unexpectedly does not need a gradient");

        // optimize for trivial gradients (e.g. Plus)
        let op = f.m_op;
        if (viewAllowed && IsOpGradientViewable(op, index, outputFields, input))
        {
            // TODO: Splice can also be viewable, but is tricky, as the Scatter optimization conflicts with it. A Splice gradient is only viewable of all its inputs'
            //       gradients are viewable; since the Scatter call currently cannot exclude slices.
            //       Also we need to set up an elaborate view, possibly determine the starting offset.
            //       Hence, Splice for now does not have a viewable gradient.
            //       Because of that, the gradient is indeed strictly a view, with possible reshape.
            // determine where the gradient should go
            // There are two places it goes: The immediate input's m_gradient (a view); and a potential redirect (to the physical location).
            // They may be the same pointer, different pointers to the same thing, or one could be a Reshape of the other. No slice possible.
            auto& inputFields = GetInputFields(input); // immediate input's gradient view
            auto& redirectedInputFields = inputFields.m_redirection.empty() ? inputFields : GetOutputFields(*inputFields.m_redirection.m_function); // physical gradient location
            fail_if(inputFields.m_gradient || redirectedInputFields.m_gradient, "function with viewable gradient unexpectedly already has a gradient??");
            auto outputGradientValue = outputFields.m_gradient; // incoming gradient from top. Our gradient is going to be a view of this.
            if (op == PrimitiveOpType::Reshape)
                outputGradientValue = outputGradientValue->AsShape(inputFields.m_shape); // an explicit Reshape (generated by auto-batch; other ops must have the same shape already) --TODO: do not generate this, use implicit reshape
            // implant the cached gradient into this input
            inputFields.m_gradient = move(outputGradientValue);
            // implant it into the redirect
            if (&inputFields != &redirectedInputFields)
            {
                // If input is a redirected slice, then necessarily the underlying object (=inputFields.m_redirection.m_function)
                // must have multiple consumers. Otherwise it would not be part of a batched operation, which is the only way
                // of creating redirected slices.
                fail_if(!inputFields.m_redirection.m_sliceRange.empty(), "redirected slice with single consumer shouldn't be a redirect in the first place");
                auto grad = inputFields.m_gradient;
                ReplaceWithReshapedViewIfNeeded(grad, redirectedInputFields.m_shape);
                redirectedInputFields.m_gradient = move(grad); // and the redirected location. If it is the same, then this will do nothing.
            }
            // Nota bene: This is a little scary. If this condition is not correct, and more gradients get accumulated
            // into this input, then those will get into the output as well, which would be incorrect. There is no good
            // way to ensure this.
            m_stats.numBatchedBackpropToViews++;
            return;
        }
        let gradient = CacheAndGetGradientView(input, DetermineGradientStorageType(f, 0));

        // compute gradients for the desired input
        // backprop into the input
        // BUGBUG: (perf) In case of Reshape we currently make a copy, which is not needed --> see-through the op, and backprop through a reshaped view into Reshape's argument gradient?
        PrimitiveFunction::BackpropTo(outputFields.m_gradient.get()/*incoming*/, index, op, f.m_attributes, outputFields.m_value.get(), inputValues, gradient.view/*target*/, gradient.beta, f);
        m_stats.numBatchedBackpropToCalls++;
#if 0   // debug the actual values
        fields.m_gradient->LogToFile(L"gradient", stderr);
#endif
    }

    // backprop into all inputs of a splice operation
    // This is done as a single CUDA launch into all inputs.
    // We must make sure we run this only once.
    // The first time this gradient gets pulled, we do it for all inputs
    // and remember that this has been done.
    void BackpropThroughSplice(PrimitiveFunction& f)
    {
        // if we pull this a second time, then don't propagate again
        // Note: This visited-tag is only used by this function.
        if (m_visitorTag.Visited(f.m_autoBatchState.m_visitedTag))
            return;
        // fast path: only one input (and not Splice, which we do in bulk)
        if (f.m_inputs.size() == 1)
            return BackpropToUnbatched({ &f, 0 }, /*viewAllowed=*/false);
        // Considerations:
        //  - For now we do optimize for consecutive inputs, because those would not
        //    have been transformed into a Splice operation in auto-batching.
        //    User's batch ops go here as well; we won't optimize for them for now.
        //    Hence, this is strictly a ScatterBatch operation.
        //  - It is possible that the Splice operation consumed the same input twice.
        //    This is currently handled via atomicAdd(), i.e. will have non-determinism.
        //    (A striding trick minimizes the probability of clashes.)
        //    Note that in this case, beta will be 1, since at least one of those
        //    gradients is not newly created, which is detected below and forces beta=1.
#if 0
        for (size_t index = 0; index < f.m_inputs.size(); index++)
        {
            BackpropToUnbatched({ f, index }, /*viewAllowed=*/false); // (this is a testing path only, so no need to worry about viewAllowed)
            m_stats.numBatchedBackpropToCalls--; // for now fake the call count to see the potential impact
        }
#else
#ifdef LOG_DETAILS
        LogFunction(*f, L"bb# ", SIZE_MAX);
#endif
        // function's forward output and received gradient from top live here
        let& outputFields = GetOutputFields(f); // result of f lives here; hence also the gradient we back-propagate
        fail_if(!outputFields.m_value,    "unexpectedly ran into a function that has no m_value yet??");
        fail_if(!outputFields.m_gradient, "unexpectedly ran into a function that has no m_gradient yet??");

        // The gradient of Splice is just copying all columns to the respective inputs.
        let& inputs =  f.m_inputs;
        let numInputs = inputs.size();
        auto& inputGradients = BorrowBuffer(m_inputValuesBuffer, numInputs);   // target locations to propagate the columns to (GetinputFields(input).m_gradient; no redirect unless it's a view)
        auto& inputGradientsToZeroOut = BorrowBuffer(m_inputValuesBuffer2, 0); // if we manually must reset gradients to zero, this is the list fo those
        bool allBetasZeroSoFar = true;
        for (size_t i = 0; i < numInputs; i++)
        {
            let& input = inputs[i];
            // create the gradient memory for this input. This sets both input->m_dataFields and the redirect if any
            let gradient = CacheAndGetGradientView(input);
            inputGradients[i] = gradient.view;
            // handle inconsistent betas
            if (gradient.beta != 0 && allBetasZeroSoFar)
            {
                // We were running under the assumption that all betas are zero, so we can use beta=0 below.
                // Now we must run with beta 1, and therefore manually reset all pevious ones.
                for (size_t i1 = 0; i1 < i; i1++) // these were all beta=0
                    inputGradientsToZeroOut.push_back(GetInputFields(inputs[i1]).m_gradient);
                allBetasZeroSoFar = false;
            }
            else if (gradient.beta == 0 && !allBetasZeroSoFar)
                inputGradientsToZeroOut.push_back(GetInputFields(input).m_gradient);
        }
        let beta = allBetasZeroSoFar ? 0.0 : 1.0; // if at least one is not zero, we must run qwith beta=1

        // manually reset all newly created ones when we are forced to use beta=1
        // TODO: Once we have virtual scatter, then scatter a ConstOne(alpha=0) into these in a single CUDA launch.
        //       E.g., the first time this is hit, it's for 515 items; that is, 515 CUDA launches (cudaMemsetAsync()).
        if (!inputGradientsToZeroOut.empty())
        {
            for (let& grad : inputGradientsToZeroOut)
                grad->SetValue(0.0f);
            m_stats.numBackpropSetZeroes += inputGradientsToZeroOut.size();
        }

        // backprop into all inputs
        let& outputGradient = outputFields.m_gradient; // this is the incoming batch of gradients
        NDArrayView::ScatterBatch(outputGradient, inputGradients, (size_t)f.m_attributes[PrimitiveFunction::AttributeNameAxis].Value<Axis>().StaticAxisIndex(), beta);
#endif
        m_stats.numBackpropScatters++;
    }

    // backprop into weight parameter of a Times op (input 0)
    // This can be batched into a single matrix product.
    void BackpropToMatrixWeight(const vector<pair<PrimitiveFunction*, size_t>>& consumers)
    {
#if 0
        for (auto& c : consumers)
            BackpropToUnbatched(c, /*viewAllowed=*/false); // view would not help, so we can pass false at no loss
#else
        // batch all outGrads, and batch all right inputs
        let numBatchItems = consumers.size();
        if (numBatchItems == 1) // fast path if only one
            return BackpropToUnbatched(consumers.front(), /*viewAllowed=*/false); // view would not help, so we can pass false at no loss
        // We compute
        //  leftGrad += sum_i outGrad_i @ right_i^T
        //            = (concat_i outGrad_i) @ (concat_i right_i)^T
        // where concat_i means to concatenate matrices along their trailing (batch) axis.
        // It has already been verified that all i have the same rank and dimensions except for a single reduction dimension.
        auto& timesOutGrads        = BorrowBuffer(m_inputValuesBuffer,  numBatchItems);
        auto& timesDataRightInputs = BorrowBuffer(m_inputValuesBuffer2, numBatchItems);
        let& f0 = *consumers.front().first;
#ifdef LOG_DETAILS
        LogFunction(f0, L"bb* ", 0);
#endif
        let& input0 = f0.m_inputs.front(); // all consumers share this weight, so it's OK to just get it from f0
        size_t batchDim = 0;
        for (size_t i = 0; i < numBatchItems; i++)
        {
            let &c = consumers[i];
            fail_if(c.second != 0, "wrong input??");
            let& f = *c.first;
            fail_if(&GetInputFields(f.m_inputs.front()) != &GetInputFields(input0), "batched matrix gradients do nto share the matrix??");
            let& outGrad = GetOutputFields(f).m_gradient;
            let& right   = GetInputFields(f.m_inputs.back()).m_value;
            timesOutGrads       [i] = outGrad; // incoming gradients from top
            timesDataRightInputs[i] = right;   // second arguments
            let numItems = outGrad->Shape().Dimensions().back();
            fail_if(numItems != right->Shape().Dimensions().back(), "batch dimension of two inputs not the same??");
            batchDim += numItems;
        }
        auto outGradBatch = GatherBatchInArena(timesOutGrads       , GetOutputFields(f0)                .m_shape.Rank() - 1, batchDim);
        auto rightBatch   = GatherBatchInArena(timesDataRightInputs, GetInputFields (f0.m_inputs.back()).m_shape.Rank() - 1, batchDim);
        m_stats.numAvoidedBackpropToMatrix += batchDim - 1; // these were saved

        // backprop into the left input from the batched outGrad and right
        auto& inputValues = BorrowBuffer(m_inputValuesBufferRaw, 2);
        inputValues[0] = nullptr;
        inputValues[1] = rightBatch.get();
        let gradient = CacheAndGetGradientView(input0, DetermineGradientStorageType(f0, 0));
        PrimitiveFunction::BackpropTo(/*outputGradient=*/outGradBatch.get(),      // incoming gradient from top...
                                      /*index=*/0, f0.m_op, f0.m_attributes,      // ...goes through this function...
                                      /*outputValue=*/nullptr, inputValues,       // ...using these values from forward pass...
                                      gradient.view, gradient.beta, f0);          // ...into here
        m_stats.numBatchedBackpropToCalls++;
#endif
    }

    // backprop gradient into 'var' by pulling all of its consumers (recursively)
    // This is the second function that does batching.
    // The vectors for building the lists are class members so that we reuse the malloc.
    // This is a subroutine of RAggregateGradientFromAllConsumers().
    vector<AutoBatchConsumer> m_spliceConsumers;       // Scatter optimization: backprop to all inputs at once using ScatterBatch
    vector<AutoBatchConsumer> m_matrixWeightConsumers; // matrix product optimization: sum -> inner dimension
    vector<AutoBatchConsumer> m_viewableConsumers;     // see-through gradients being aggregated -> Gather and Reduce
    vector<AutoBatchConsumer> m_otherConsumers;        // remaining incoming gradients for which we have no further optimization
    void ClearBuckets()
    {
        m_spliceConsumers      .clear();
        m_matrixWeightConsumers.clear();
        m_viewableConsumers    .clear();
        m_otherConsumers       .clear();
    }
    void DetermineAndAddToBucket (const AutoBatchConsumer& c)
    {
        let* f = c.first;
        let index = c.second;
        fail_if(f->m_outputs.size() != 1, "for now only functions with a single output are supported"); // (needs some more plumbing to fix this)
        // backprop into Times' matrix argument
        // BUGBUG: This currently does not capture single time steps that backprop into the same matrix as a batch.
        let IsMatrixGradient0Batchable = [](const PrimitiveFunction& f, const PrimitiveFunction& g) -> bool
        {
#if 0       // use 1 to disable batching of matrix gradients
            return false;
#else
            // we compute leftGrad += outGrad @ right^T
            let&   fOutShape = f.m_outputs.front().Shape().Dimensions();
            let&   gOutShape = g.m_outputs.front().Shape().Dimensions();
            let& fRightShape = f.m_inputs .back() .Shape().Dimensions();
            let& gRightShape = g.m_inputs .back() .Shape().Dimensions();
            let&   leftShape = f.m_inputs .front().Shape().Dimensions();
            let    outRank =   fOutShape.size();
            let   gOutRank =   gOutShape.size();
            let  rightRank = fRightShape.size();
            let gRightRank = gRightShape.size();
            let   leftRank =   leftShape.size();
            fail_if(leftShape != g.m_inputs.front().Shape().Dimensions(), "dimensions of matrix gradient don't match??");
            if (outRank != gOutRank || rightRank != gRightRank)
                return false; // rank not matching: stop batching right here (we could do better)
            // the center 'reductionRank' dimensions get reduced over
            if (outRank + rightRank - leftRank != 2) // if 2 then we reduce over a single batch axis
                return false; // this is not a batch gradient; back out
            fail_if(fOutShape.back() != fRightShape.back() || gOutShape.back() != gRightShape.back(), "inner dimensions of matrix gradient don't match??");
            // the two gradient ops match if all dimensions except for the batch dim match
            for (size_t k = 0; k < outRank - 1; k++) // check outGrad
                if (fOutShape[k] != gOutShape[k])
                    return false;
            for (size_t k = 0; k < rightRank - 1; k++) // check right
                if (fRightShape[k] != gRightShape[k])
                    return false;
            // gradient is batchable
            return true;
#endif
        };
        // Note: This needs to be enabled for column-sparse gradients to work!
        // BUGBUG: (perf) We should also have a special path for Reshape(), as to avoid the memory copy.
        //let opClass = g_oscTable[f->m_op]; // operation-specific auto-batching class
        // splice operation should use scatter
        // BUGBUG: Really only true if the Splice originated from a batching operation.
        //         User-specified Splice ops might have arguments that receive no gradient, e.g. constants.
        let op = f->m_op;
        if (op == PrimitiveOpType::Splice)
            m_spliceConsumers.push_back(c);
        // backpropagating into the first arg of a matrix product
        // These are shared.
        // We only collect matrix products that fully match the first one. --TODO: when do they not fully match?
        else if (IsMatrixProduct(op) && index == 0 &&
            (m_matrixWeightConsumers.empty() || (IsMatrixGradient0Batchable(*f, *m_matrixWeightConsumers.front().first))))
            m_matrixWeightConsumers.push_back(c);
        // backprop where gradients are just views that we can sum up
        //else if (f->m_op == PrimitiveOpType::Plus)
        //    return m_viewableConsumers;
        // all other
        else
            m_otherConsumers.push_back(c);
    };

    // compute a variable's outputs' gradient (var.m_gradient)
    // This operates on the PrimitiveFunction(s) that use this var's output value--its "consumers".
    // A variable knows all of its consumers. This function back-propagates from all consumers
    // into the variable's m_gradient field.
    // A consumer is a specific input of a PrimitiveFunction, specified as (function pointer, input index).
    // In the case of multiple consumers, the gradient is the sum.
    // This recursively traverses the graph upwards via the consumers chain.
    // Caller must call m_visitorTag.Begin() first.
    // This uses Variable::m_visitedTag for traversing the consumer tree upwards.
    // It uses PrimitiveFunction::m_visitedTag for bulk gradient of currently the
    // Splice op, which produces all inputs' gradients in a single go and must therefore only run once.
    // Normally, fields.m_gradient is NULL. The one exception is if the user supplies a buffer for a requested gradient.
    __declspec(noinline) void RAggregateGradientFromAllConsumers(VariableFields& fields)
    {
        if (m_visitorTag.Visited(fields.m_visitedTag))
            return;

        fail_if(fields.m_consumers.empty(), "root gradient not set up??"); // this is the root. It should already have been visited manually.

        fail_if(!fields.m_needsGradient, "backprop into variable that does not need gradient");

        // recursively realize all consumers' outputs' gradients
        // i.e. realize all that this gradient depends on
        fields.m_consumers.ForAll([&](const std::pair<PrimitiveFunction*, size_t>& fi)
        {
            auto& consumerGradFields = GetOutputFields(*fi.first);
            fail_if(!ArePhysicalOutputFields(consumerGradFields), "pulling gradient from a redirected output location??");
            RAggregateGradientFromAllConsumers(consumerGradFields);
        });

        // Now all consumers are ready to propagate into var's m_gradient.
        // The resulting gradient is the sum of all that's backpropped here,
        // and this is the only place where a variable's gradient ever gets aggregated.

        // create var's m_gradient (may be a slice view)
        // m_gradient may already exist for Parameters, and when it came through Splice.
        // Because of the latter, we cannot really test this here, and should just remove this check.
        //fail_if(var.Kind() != VariableKind::Parameter && fields.m_gradient, "non-Parameter variable unexpectedly already has a gradient"); // (sanity check; I am not sure actually, maybe too strict)

#ifdef NO_BATCHED_BACKPROP
        if (fields.m_consumers.size() == 1)
            BackpropToUnbatched(fields.m_consumers.front(), /*viewAllowed=*/!fields.m_gradient); // viewAllowed because there is only one, no aggregation needed
        else
            fields.m_consumers.ForAll([&](const std::pair<PrimitiveFunction*, size_t>& fi)
            {
                BackpropToUnbatched(fi, /*viewAllowed=*/false);
            });
#else
        // fast path: if only one consumer then there is nothing to batch
        // (with exception of Splice, which has a different optimization)
        if (fields.m_consumers.size() == 1 && fields.m_consumers.front().first->m_op != PrimitiveOpType::Splice)
        {
            BackpropToUnbatched(fields.m_consumers.front(), /*viewAllowed=*/!fields.m_gradient); // viewAllowed because there is only one, no aggregation needed
            return;
        }

        // auto-batched path
        // The forward prop has already batched data. However, for backprop, there can
        // be more batching across the multiple consumer's backprop operations.

        // At this point, we need to execute the BackpropTo() function for every
        // consumer (function, input), and then sum up the result.
        // The combination of the backprop functions and summing up their result
        // may be batchable, as for the matrix product.

        // Since a variable can be consumed by multiple different kinds of PrimitiveFunctions,
        // which each have a different gradient computation, we need to separate them out.
        // At present, we aim for weight gradients that are part of a matrix product
        // or a bias addition.
        // First sort all consumer gradients according to their operation.
        ClearBuckets();
        fields.m_consumers.ForAll([&](const std::pair<PrimitiveFunction*, size_t>& fi)
        {
            DetermineAndAddToBucket(fi);
        });

        // splice bucket
        // This input is pulling a gradient from a Splice operation.
        // Instead of producing just this one input's gradient, we use ScatterBatch to fill all of them.
        for (auto& c : m_spliceConsumers)
            BackpropThroughSplice(*c.first);

        // matrix-weight bucket
        if (!m_matrixWeightConsumers.empty())
            BackpropToMatrixWeight(m_matrixWeightConsumers);

        // summation bucket
        // ...

        // others bucket
        for (auto& c : m_otherConsumers)
            BackpropToUnbatched(c, /*viewAllowed=*/false);
#endif
    }

public:
    // -----------------------------------------------------------------------
    // BatchedBackward() -- entry point for auto-batched implementation of PrimitiveFunction::Backward()
    // -----------------------------------------------------------------------

#if 0 // helper for debugging, traversing the batched graph
    static void RCheck(const PrimitiveFunction* f, set<const Function*>& visited, size_t depth)
    {
        if (!visited.insert(f).second)
            return;
        if (f->m_op == PrimitiveOpType::Block)
            Break;
        if (f->m_op == PrimitiveOpType::NoOp)
            fprintf(stderr, "..%d\n", (int)f->m_uniqueIdForDebugging);
            //Break;
        for (let& input : f->m_inputs)
        {
            if (input.IsOutput() /*&& input.m_dataFields->m_needsGradient*/)
                //RCheck(input.OutputOwner().get(), visited, depth+1);
                RCheck(input.m_dataFields->m_redirection.m_function, visited, depth+1);
            if (input.m_dataFields->m_uniqueIdForDebugging == 243)
                Break;
        }
    }
    static void Check(const Variable& root)
    {
        set<const Function*> visited;
        RCheck(root.OutputOwner().get(), visited, 0);
        fprintf(stderr, "Check: %d Functions\n", (int)visited.size()), fflush(stderr);
    }
#endif

    // implant gradients into all variables
    // Unlike BatchedForward(), this is eager. If you call it twice, it's a completely new computation.
    // If you need multiple gradients, ask for them in a single go.
    void BatchedBackward(const Variable& root, unordered_map<Parameter, NDArrayViewPtr>& gradients)
    {
        if (!root.m_dataFields->m_needsGradient)
            LogicError("BatchedBackward: cannot compute gradient for root with m_needsGradient being False.");
        // BUGBUG: make sure some edge cases are done right:
        //  - root.m_needsGradient=false
        //  - gradients contains non-Parameters
        //  - root is a m_redirection
        // first get the forward computation, batching, etc. done if not yet
        BatchedForward(root);
        // if user passed NDArrayViewPtrs for the gradients, then implant those
        // If nulls are passed, then keep whatever is there. This way, the same buffers can be recycled.
        for (auto& kv : gradients)
        {
            let& param = kv.first;
            if (kv.second)
                param.m_dataFields->m_gradient = kv.second; // (if null then we will keep what's already there, or create them if null)
        }
        // --- Phase 1: form the inverted backprop graph
        //  - set up the m_consumer fields to form the inverted backprop graph
        //  - this will also short-circuit chains of see-through ops (in-place, as an equivalence transform)
        m_visitorTag.Begin();
        // first set it up for the Parameters for which we have requested a gradient
        // This way we won't backprop into gradients of Parameters that we did not ask for.  --TODO: implement this
        // BUGBUG:  --TODO: What did this BUGBUG comment want to tell me?? It's empty!
        for (auto& kv : gradients)
        {
            auto& gradFields = GetGradientFieldsForBackprop(ResetInputGradient(kv.first), /*firstTimes=*/true);
            if (m_visitorTag.Visited(gradFields.m_visitedTag)) // (note that the code does not require this; this is only to point out a likely user error)
                InvalidArgument("BatchedBackward: a Parameter was included more than once in gradients[]");
            RBuildBackwardGraph(gradFields, /*userOwnsGradients=*/true);
            // BUGBUG: ^^ userOwnsGradients won't work correctly if one Var in gradients[] is an input to another
        }
        // now build the graph. We use visited information for the gradients to infer our own needsGradient flag  --TODO: No, not done yet.
        auto& rootGradFields = GetGradientFieldsForBackprop(ResetInputGradient(root), /*firstTimes=*/true);
        if (!m_visitorTag.Visited(rootGradFields.m_visitedTag)) // (A crazy user may have passed root itself in gradients[]. That is OK.)
            RBuildBackwardGraph(rootGradFields);
        // sanity check
        for (auto& kv : gradients)
        {
            let& gradFields = GetGradientFieldsForBackprop(kv.first);
            if (gradFields.m_consumers.empty()) // if gradient's Parameter has no consumers, then it is not part of the root
                LogicError("BatchedBackward: requested gradient \"%S\" is not part of root.", kv.first.Name().c_str()); // TODO: or could it be due to StopGradient? What if StopGradient is used only sometimes?
            if (!gradFields.m_needsGradient) // (we could also just leave the gradient 0)
                LogicError("BatchedBackward: cannot compute gradient for variable with m_needsGradient being False."); // such as a Constant. Actually, why? User can certainly get that if desired.
        }
        // --- Phase 2: backprop through the inverted backprop graph
        //  - perform backprop operation, by depth-first traversal of inverted backprop graph starting from gradients
        m_visitorTag.Begin();
        // implant the first gradient. This "root: gradient is actually the leaf in the inverted backprop graph.
        // TODO: allow user to pass in the starting value
        // BUGBUG: we get a [1] here, but should be a scalar. This is a bug outside.
        //if (root.Value()->Shape() != NDShape{})
        //    LogicError("BatchedBackward: root must be a scalar, or root gradient must have been implanted already");
        rootGradFields.m_gradient = m_arena.NewNDArrayView(root.Shape(), root.GetDataType(), StorageFormat::Dense, root.Value()->Device());
        rootGradFields.m_gradient->SetValue(1.0f);
        m_visitorTag.Visited(rootGradFields.m_visitedTag); // done with this
        // perform backprop
        // This traverses the tree top-down, where each node pulls gradient(s) from its consumer(s).
        // This way we can optimize operations, such as a matrix product or gradient of GatherBatch().
        for (auto& kv : gradients)
        {
            auto& gradFields = GetGradientFieldsForBackprop(kv.first);
            RAggregateGradientFromAllConsumers(gradFields);
        }
        //fprintf(stderr, "Back-propagated through %d functions\n", (int)order.size());
        // implant the results into the map the user passed in
        for (auto& kv : gradients)
            kv.second = kv.first.m_dataFields->m_gradient;
        // note: we will leave the m_consumers fields dangling, and reset them upon next use
#ifdef LOG_STATS
        fprintf(stderr, "BatchedBackward: %d backprop computations and %d views besides %d gathers, %d scatters, %d set-zeroes, %d skipped matmuls for %d ops (total %d inputs)\n",
                (int)m_stats.numBatchedBackpropToCalls, (int)m_stats.numBatchedBackpropToViews,
                (int)m_stats.numBackpropGathers, (int)m_stats.numBackpropScatters, (int)m_stats.numBackpropSetZeroes, (int)m_stats.numAvoidedBackpropToMatrix,
                (int)m_stats.numBackpropsThrough, (int)m_stats.numBackpropsToInputs);
#endif
    }
}; // class

// ===========================================================================
// auto-batching entry points
// ===========================================================================

// this will become Variable::Value()
// Computes lazily the value of a node. Does nothing if called again.
NDArrayViewPtr PrimitiveFunction::BatchedForward() const
{
    //let sthis = dynamic_pointer_cast<PrimitiveFunction>(const_cast<PrimitiveFunction*>(this)->shared_from_this());
    //let comp = CompositeFunction::Create(sthis);
    //let res = comp->FilteredInputs<Variable>([](const Variable& v) { return v.m_dataFields->m_uniqueIdForDebugging == 243; });
    //res;
    auto autoBatcher = Variable::AutoBatch();
    return autoBatcher.BatchedForward(m_outputs.front());
}

// Perform backprop.
// TODO: CNTK grad() allows to pass multiple roots. Does that ever make sense in this context?
void PrimitiveFunction::BatchedBackward(std::unordered_map<Parameter, NDArrayViewPtr>& gradients) const
{
    auto autoBatcher = Variable::AutoBatch(); // has some internal state
    autoBatcher.BatchedBackward(m_outputs.front(), gradients);
}

// non-batched version of BatchedForward()
// This is actually not used except as a fallback for debugging.
// TODO: move to -Eval.cpp
void PrimitiveFunction::MemoizeKnowableValue() const
{
    if (m_outputs.size() != 1)
        LogicError("Variable '%S' Value(): Only Variables with one output can compute their Value for now.", AsString().c_str());
    const auto& output = m_outputs.front();
    if (output.m_dataFields->m_value) // already done
        return;
    // get the input values (recursively compute them if needed)
    if (m_inputs.empty())
        LogicError("Variable '%S' Value(): Only Variables with input arguments can compute their Value.", AsString().c_str());
    vector<NDArrayViewPtr> args(m_inputs.size());
    for (size_t i = 0; i < args.size(); i++)
        args[i] = m_inputs[i].Value();
    NDArrayViewPtr out;
    output.m_dataFields->m_value = move(ComputeKnowableValue(m_op, args, m_attributes, output.Shape(), move(out), *this));
}

// ===========================================================================
// Invoke() and our pieces of PrimitiveFunction and BlockFunction -- contained here for now
// ===========================================================================

// call this upon creation of the compositer
// TODO: Encapsulate this into the Invocable class.
void Function::InitCompositeForInvoke(const vector<Variable>& placeholders)
{
    let callee = dynamic_cast<CompositeFunction*>(this);
    if (!callee)
        InvalidArgument("InitForInvoke() must only be called on CompositeFunctions.");
    // implant the redirect into the placeholders
    for (size_t i = 0; i < placeholders.size(); i++)
    {
        let& p = placeholders[i]; // Placeholder or Parameter in callee
        if (!p.IsPlaceholder())
            InvalidArgument("InitForInvoke(): All argument items must be Placeholders.");
        p.m_dataFields->m_compositeArgumentIndex = i;     // when dynamically expanding this, we match up this Placeholder with the respective input[i]
    }
    // reset the cached batch axis (=max over all ranks involved, for batching the composite as elementwise
    // This is only used if isBasicBlock, but since.
    callee->m_basicBlockBatchAxis = SIZE_MAX;  // SIZE_MAX means value has not yet been determined. Once it is, it is cached here.

    callee->m_batchableCompositeId = SIZE_MAX; // composites with the same id are batchable
}

// argumentList = composite->Arguments() in a given order; Placeholders first, then all Parameters. Get updated upon determining shapes.
// operands     = what the arguments should prerent to be. Must currently include a copy of Parameters at the end.
Variable Invoke(const /*Composite*/FunctionPtr& callee, vector<Variable>& argumentList, const vector<Variable>& operands, bool isBasicBlock, bool& needToDetermineShapes, const std::wstring& name /*= std::wstring()*/)
{
    let composite = static_pointer_cast<CompositeFunction>(callee); // (static cast since caller must have called InitCompositeForInvoke() before, which checked the type)
#if 0   // This baloney, at least in the case of isBasicBlock.
    // TODO: To make use of this if !isBasicBlock, we must make sure that the static cloning machinery understands the shared composite.
    // BUGBUG: For now, the static support of Invoke is incomplete. Don't actually Clone() one of those babies.
    // This can be called in two situations:
    //  - during static building of a static graph
    //  - during dynamic imperative code
    // PERF BUGBUG: This is *not* nice, as we need to check the arguments all the time.
    if (any_of(operands.begin(), operands.end(), [](const Variable& arg) { return arg.Shape().IsUnknown(); }))
    {
        // If any Placeholder inputs, we are in static land -> inline the composite
        // Note: This will be slow, but it's OK since we are in static pre-processing land, not inside dynamic computation.
        let compositeArgs = composite->Arguments();
        if (compositeArgs.size() != operands.size())
            InvalidArgument("Invoke called with wrong number of operands (%d provided vs. %d expected", (int)operands.size(), (int)compositeArgs.size());
        unordered_map<Variable, Variable> replacementMap;
        for (size_t i = 0; i < operands.size(); i++)
            replacementMap[compositeArgs[i]] = operands[i];
        let clone = composite->Clone(ParameterCloningMethod::Share, replacementMap);
        let& outputs = clone->Outputs();
        if (outputs.size() != 1)
            InvalidArgument("Invoke can only be used with BlockFunctions that have a single output (this one has %d).", (int)outputs.size());
        return outputs.front();
    }
    else
#endif
    {
        // this leaves 'needToDetermineShapes' true until called for the first time with fully known shapes
        // This returns 'true' only once, and evaluates the IsUnknown test only once for a dynamic invocation
        // (but multiple times for initial invocations during construction of static graphs).
        bool determineShapesThisTime;
        if (needToDetermineShapes && all_of(operands.begin(), operands.end(), [](const Variable& arg) { return !arg.Shape().IsUnknown(); }))
        {
            needToDetermineShapes = false;
            determineShapesThisTime = true;
        }
        else
            determineShapesThisTime = false;
        // BUGBUG: Do we need this? m_inputs.emplace_back(std::move(inputVar.NonCompositePreservingCopy())); for the operands
        // TODO: Since we copy the operands, we could augment the Parameters here as well.
        let f = MakeSharedObject<BlockFunction>(composite, argumentList, vector<Variable>(operands), isBasicBlock, determineShapesThisTime, wstring(name));
        return f->FinalizeInvoke(/*shapeIsKnown=*/!needToDetermineShapes);
    }
}

// special short-circuited version where output is created outside
void PrimitiveFunction::InitOutput(Variable&& output)
{
    //std::call_once(m_outputsInitFlag, [this]() {});
    m_outputInitializingByThreadId = std::thread::id();
    fail_if(m_outputsInitFlag || !m_outputs.empty(), "InitOutput called twice??");
    m_outputsInitFlag++;
    output.SetOwner(static_pointer_cast<PrimitiveFunction>(shared_from_this()));
    // This really belongs inside the constructor, but we don't have the shared_ptr yet. Not nice this way.
    m_outputs.resize(1);
    m_outputs.front() = move(output);
}

// TODO: make the interface nicer. The composite should be locked away in a struct Invocable.
// This is for Dynamite only. We are (mis-)using the BlockFunction to represent a PrimitiveFunction that Dynamite can interpret.
// It is a valid PrimitiveFunction, but it shares the composite instead of owning it, and therefore not a valid BlockFunction for the static-graph machinery.
// TODO: Prevent the static machinery from tripping over this.
// It is a light-weight representation of 'callee' being called with 'operands.'
// 'Callee' is a BlockFunction holds a composite that represents its expression, where m_inputs consists of placeholders representing its inputs in right order, and m_outputs[0] holds the resulting shape/type.
// The resulting object is also a BlockFunction that points to the same composite, but m_inputs represents the invocation operands, and m_outputs[0] the result.
// Unlike a normal BlockFunction, however, it shares the composite (for Dynamite speed) with the callee, which makes it an invalid object for static CNTK.
// BUGBUG: This is really quite horrible, as it does not work with static graphs. We need a better implementation of this.
//         The main difference is that invoked blocks physically share their composites, to avoid duplicating them
//         (since the point of using Invoke() is speed). Also, we have quite a bit code dup in this function.
// A correct implementation could allow BlockFunction to lazily clone the composite (which is what Dynamite does).
// Special case: Invoke() may also be called while building a composite (static graph) that uses another.
// In that case, we cannot infer shapes yet. Instead, this will happen automatically when the outer composite
// is inferred. This is controlled by the determineShapes flag.
BlockFunction::BlockFunction(const CompositeFunctionPtr& callee, std::vector<Variable>& argumentList, std::vector<Variable>&& operands, bool isBasicBlock, bool determineShapes, wstring&& blockName) :
    PrimitiveFunction(PrimitiveOpType::Block, move(operands), Dictionary(), move(blockName)), m_composite(callee),
    m_compositeIsShared(true), m_isBasicBlock(isBasicBlock)
{
    // The very first time we pass the composite, we must set up its Placeholder m_compositeArgumentIndex fields.
    // The caller must pass in this flag. --TODO: encapsulate this in the Invocable class.
    if (!determineShapes)
        return;

    let& compositeOutputs = callee->RawOutputs(); // RawOutputs() forces callee.m_compositeOutputs to be initialized if not yet. It would initialize it to something is shape IsUnknown, though.
    if (compositeOutputs.size() != 1)
        InvalidArgument("Invoke can only be used with BlockFunctions that have a single output (this one has %d).", (int)compositeOutputs.size());

    // If this is the first call, then we are not done yet. We must:
    //  - initialize (reset) Dynamite-specific fields in the callee
    //  - enumerate all Placeholders and number them
    //  - infer the shapes, given the first actually supplied arguments
    // if the composite has no validated shape yet, then do this now
    // This updates the composite in-place, by replacing its Placeholders with new ones with shapes matching our operands.
    // Any subsequent invocation of this must have matching dimensions. It will otherwise fail inside Dynamite execution.
    // This is slow, but that's OK since it is done only once.

    // We determine the compositeOutputs by replacing the arguments of the composite with new placeholders with updated 
    // shape etc. information matching the corresponding mapped input

    // BUGBUG: Must handle the map here now that we pass it. Also must verify that the Placeholders are actually in the composite.

    // BUGBUG: Must verify that all Parameters are covered here.
    unordered_map<Variable, Variable> replacementMap;
    for (size_t i = 0; i < m_inputs.size(); i++)
    {
        let& compositeLeaf = argumentList[i]; // Placeholder or Parameter in composite
        let& input = m_inputs[i];             // what they should pretend to be
        if (compositeLeaf.IsParameter())
        {
            // for Parameters, supply an empty Variable
            if (input != compositeLeaf)
                LogicError("Invoke: Parameters should have passed as themselves.");
            // That's it. We just keep it in the list so that auto-batch can find them.
        }
        else if (compositeLeaf.IsPlaceholder())
        {
            // verify the mappings have been implanted
            fail_if(compositeLeaf.m_dataFields->m_compositeArgumentIndex != i, "m_compositeArgumentIndex not set up??"); // when dynamically expanding this, we match up this Placeholder with the respective input[i]
            // TODO: rethink the logic. If the input's shape IsUnknown, then why not directly return? Why even replace?
            if (input.IsInput())
                InvalidArgument("Invoke cannot work on Input variables, it is for dynamic networks only.");
            fail_if(input.Shape().IsUnknown(), "unknown input shapes at this point??");
            auto updatedCompositePlaceholder = PlaceholderLike(input); // we replace with a placeholder of the same type. This gives the composite the shape.
            updatedCompositePlaceholder.m_dataFields->m_compositeArgumentIndex = compositeLeaf.m_dataFields->m_compositeArgumentIndex;
            replacementMap.insert({ compositeLeaf, updatedCompositePlaceholder }); // replace with new Placeholder, which has a block mapping implanted
            // TODO: fix the interface. This should become a private method to Invocable
            argumentList[i] = updatedCompositePlaceholder;
        }
        else
            InvalidArgument("Invoke: argumentList can only contain Placeholders and Parameters.");
    }

    callee->ReplacePlaceholders(replacementMap); // This gives the composite the shape.
    // OUTDATED --The composite args' Placeholders now have a block mapping to the actual inputs.
    // BUGBUG: That's bad, since they are gone after this. There should be no block mapping.

    if (compositeOutputs.front().Shape().IsUnknown()) // or not?
        LogicError("Invoke with determineShapes=true must only be called with inputs with fully specified dimensions.");

    // Now the composite is fully type-inferred; ready for consumption by Dynamite.
}

// call this after construction from Invoke()
Variable BlockFunction::FinalizeInvoke(bool shapeIsKnown)
{
    // now set up the output variable. Clone the composite's one output Variable, then inject the mapping pointer. This following the pattern of InferOutputs().
    // ...EXCEPT we do not implant a Variable mapping, since the composite is shared. The composite does not know that it is part of a BlockFunction.
    let& compositeOutput = m_composite->m_outputs.front();
    Variable blockOutput =
        /*if*/ (!shapeIsKnown) ? // if we are being called while building a static graph
            OutputVariable(NDShape::Unknown(), DataType::Unknown, vector<Axis>(), Name())
        /*else*/:
            OutputVariable(compositeOutput.Shape(), compositeOutput.GetDataType(), vector<Axis>(), compositeOutput.NeedsGradient(), compositeOutput.IsSparse(), Name());
    InitOutput(move(blockOutput));
    // behave as if this was returning a Composite: implant a ref count. This will be taken over by the next consumer.
    return m_outputs.front().CompositePreservingCopy(static_pointer_cast<PrimitiveFunction>(shared_from_this()));
}

// helpers to make Variables behave like indexable arrays
Variable Variable::operator[](size_t index) const
{
    // TODO: this can be simplified by short-circuiting the CompositeFunction >> Output, and just returning the Output of the Primitive, like Invoke().
    return CNTK::Index(*this, index);
}

size_t Variable::size() const
{
    let& shape = Shape();
    if (shape.Rank() == 0)
        InvalidArgument("size: Variable is a scalar and thus has no length.");
    if (shape.IsUnknown())
        InvalidArgument("size: Variable has no known size yet.");
    return shape.Dimensions().back();
}

} // namespace CNTK
