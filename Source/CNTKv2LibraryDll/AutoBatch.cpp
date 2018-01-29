//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// The functions for automatically-batched evaluation of dynamic graphs (forward and backward) are contained here.

// BUGBUG: The redirection approach causes consistency problems, since redirects can end up being Outputs with no m_ownerFunction.
//         Alternative, less brittle approach: Introduce a new VariableKind::Redirect?

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
#include <mutex>
#include <thread>
#include <time.h>

using namespace Microsoft::MSR::CNTK;
using namespace std;

//#define LOG_DETAILS     // if defined, log all forward and backward operations
//#define LOG_STATS         // if defined, log high-level statistics (#operations)
#define DETAILED_STATS    // if defined, print detailed statistics (counts of function calls and operations)
//#define LOG_GPU         // if defined, profile the GPU (warning: this will disturb the CPU measurements)
#define NUM_MBS_TO_LOG 4

// these options allow to turn features on or off, for benchmarking
//#define NO_CSE              // if defined, disable CSE
//#define NO_BATCHED_FORWARD  // ifdefined, disable auto-batching
//#define NO_BARRIER          // ignore Barrier
//#define NO_BLOCK_INVOKE     // no static graph
//#define NO_LATE_INLINING    // no late inlining

//#define NO_BATCHED_BACKPROP // if defined, don't do additional batching or any other extra optimization in backprop

static const char* compilationOptionsAsString =
    "AutoBatch compilation options:"
#ifdef LOG_DETAILS
    " LOG_DETAILS"
#endif
#ifdef LOG_STATS
    " LOG_STATS"
#endif
#ifdef DETAILED_STATS
    " DETAILED_STATS"
#endif
#ifdef LOG_GPU
    " LOG_GPU"
#endif
    "\nAutoBatch feature-enabling options: "
#ifdef NO_CSE
    "NO_"
#endif
    "CSE "
#ifdef NO_BATCHED_FORWARD
    "NO_"
#endif
    "BATCHED_FORWARD "
#ifdef NO_BARRIER
    "NO_"
#endif
    "BARRIER "
#ifdef NO_BLOCK_INVOKE
    "NO_"
#endif
    "BLOCK_INVOKE "
#ifdef NO_LATE_INLINING
    "NO_"
#endif
    "LATE_INLINING "
#ifdef NO_BATCHED_BACKPROP
    "NO_"
#endif
    "BATCHED_BACKPROP "
    ;

#ifdef DETAILED_STATS
static size_t logMemoizeStatsPeriod = 500;
static size_t logMemoizeStatsCounter = logMemoizeStatsPeriod - 2; // counts up to logMemoizeStatsPeriod and wraps. We log if it is 0, starting with the second MB.
#else
static size_t logMemoizeStatsPeriod = SIZE_MAX;
static size_t logMemoizeStatsCounter = 1;
#endif
static inline bool ShouldLogMemoizeStats()     { return logMemoizeStatsCounter < NUM_MBS_TO_LOG; }
static inline bool ShouldLogMemoizeStatsCUDA() { return logMemoizeStatsCounter & 1; }
//static inline bool ShouldLogMemoizeStatsCUDA() { return true;  }
//#ifdef LOG_GPU
//static inline bool ShouldLogMemoizeStatsCUDA() { return true; }
//#else
//static inline bool ShouldLogMemoizeStatsCUDA() { return false; }
//#endif


#define BarrierOp NoOp // for now, we use Alias() (=NoOp) to denote a Barrier(). Should become an op in its own right.
#pragma warning (disable: 4456) // until I fixed the shadowing

#define let const auto
#define fail_if(cond, err) (!!(cond) ? (LogicError("%s: %s", __func__, err),0) : 0)
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
    //       Special case: Splice and Reshape may add axes. To make them fall into this category, we pad all inputs with singleton dims to output rank.
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
    //     - the batch axis must not be touched by the unbatched op. Ops that may are ReduceElements, ElementTimes (InnerProduct case), Slice, Splice, and TransposeAxes.
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
    //     - all batchable inputs are scalars
    //     - the batch axis is touched by the unbatched operation, e.g. sliced or reduced over
    //     - there is a batchable input that is a sparse vector (i.e., of rank 1). (Math library cannot stack sparse along axis 0.)
    //  - determine batch axis, as given above depending on stacking/batching decision
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
    //  - OptimizedRNNStack   // for MT
    //  - Hardmax. Can we have a OneHot-like op in TensorView?

    enum class OpSpecificConditionKind  :  size_t // the meanings of these are specified below
    {
        UnaryElementWise, Reducing, NoOp, ReshapeOneHot, BinaryElementWise, NaryElementWise, Pooling,
        Slice, Splice, Transpose,
        MatrixProduct, Convolution,
        Barrier, BatchNormalization, OptimizedRNNStack, Generative,
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
            PrimitiveOpType::StableSigmoid, PrimitiveOpType::ScaleAndShift,
        }},
        { OpSpecificConditionKind::Reducing, {
            PrimitiveOpType::ReduceElements /*(=ReduceSum, ReduceLogSum, etc.)*/,
            PrimitiveOpType::InverseStandardDeviation
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

        // Reshape(), OneHot()
        // -------------------
        // 
        { OpSpecificConditionKind::ReshapeOneHot, {
            PrimitiveOpType::Reshape,
            PrimitiveOpType::OneHot
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
        //
        // OneHot is a distinct operation, but it behaves, shape-wise, similar to Reshape
        // in that it can insert axes.

        // Binary/ternary element-wise ops
        // -------------------------------
        // 
        { OpSpecificConditionKind::BinaryElementWise, {
            PrimitiveOpType::Plus, PrimitiveOpType::Minus, PrimitiveOpType::ElementTimes, PrimitiveOpType::LogPlus, PrimitiveOpType::Pow,
            PrimitiveOpType::Equal, PrimitiveOpType::NotEqual, PrimitiveOpType::Less,
            PrimitiveOpType::LessEqual, PrimitiveOpType::Greater, PrimitiveOpType::GreaterEqual
        }},
        { OpSpecificConditionKind::NaryElementWise, {
            PrimitiveOpType::Clip, PrimitiveOpType::Select,
            PrimitiveOpType::ElementAffine,
            PrimitiveOpType::NormalizeDenormalize
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
            PrimitiveOpType::Times, PrimitiveOpType::TransposeTimes, PrimitiveOpType::Affine, PrimitiveOpType::TransposeAffine,
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
        { OpSpecificConditionKind::Generative,{
            PrimitiveOpType::RandomDistribution // covers also the -Like version and constants
        }},
        //
        // Condition (stacking):
        //   Attributes must match, including the random seed.
        //
        // This op is nullary. The unary (-Like) version is mapped immediately to a nullary version.
        // A potential FreeDimension in the -Like version's input is kept (static invocable only),
        // otherwise all dimensions and data type must be known for Dynamite.
        // This operation returns a new constant upon each invocation.
        //
        // If multiple instances are to share the same random numbers (e.g. Dropout within a sequence),
        // then user must pass the same 'rngState' parameter for all those invocation. Instances with
        // matching rngStates (and shape/other attributes) will be batched and CSE'ed.
        // Hence, they will be computed only once per Value() invocation.
        // However, any invocations that are not batchable (not ready at the same time) will cause
        // a new random tensor to be created.
        // BUGBUG: Naw, that won't work for dropout inside loops. Sigh. We need an explicit reset signal or something. Dang.
        //
        // This operation is also used to create scalar constants where needed, e.g. (x+1).
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
            // new ones after merge from master --TODO, sort and finish them
            PrimitiveOpType::Sinh,
            PrimitiveOpType::Cosh,
            PrimitiveOpType::UnpackBatch,
            PrimitiveOpType::ToBatch,
            PrimitiveOpType::Asin,
            PrimitiveOpType::Acos,
            PrimitiveOpType::Pad,
            PrimitiveOpType::Crop,
            // end new ones after merge from master
            PrimitiveOpType::ROIPooling,   // need to find out how it works precisely--is it just like pooling?
            PrimitiveOpType::LambdaRank,
            PrimitiveOpType::NDCG,
            PrimitiveOpType::EditDistanceError,
            PrimitiveOpType::LabelsToGraph,
            PrimitiveOpType::ForwardBackward,
            PrimitiveOpType::CosDistanceWithNegativeSamples,
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

    // thoughts on memory management:
    // Goal: do not keep NDArrayViews around that are not needed.
    // For example, in inference we can free NDArrayViews once their last consumer has consumed them.
    // 3 kinds of NDArrayViews:
    //  - values that user code may request
    //     - condition: user code holds an explicit reference to a Variable,
    //       which means that its value can be requested any time, and should therefore
    //       be kept once computed
    //  - values needed for back propagation
    //     - We know during forward prop which Variables are never needed in backprop
    //     - For those that may be used in backprop, we cannot know whether the user will
    //       actually ask for those gradients. But it is a reasonable assumption that
    //       all gradients will be requested. We rather should use explicit StopGradient()
    //       calls as an optimization hint
    //  - local temporary values not needed for gradients
    //     - Variables not needed for gradient computation, based on the specific function (e.g. + does not need the output values)
    //     - inference/"volatile" (will never receive a gradient)
    //     - StopGradient()
    //
    // For each, maintain its own arena; that is, 3 arenas
    //  - user-requested ones are few, but may live long, and therefore should not block deallocation of an arena
    //  - local temps can be freed more frequently
    //    Can they be maintained in a stack?
    //
    // Algorithm:
    //  - each time a Variable is consumed, and is known to not be needed by backprop, clear the Variable object
    //    The Variable and/or its associated NDArrayView data will remain alive if
    //     - any code holds a shared_ptr to the NDArrayView  --alive via ref count of NDArrayView instance, mostly that would be user code
    //     - any code holds a view (slice/reshape/alias) to the tensor data  --alive via ref count of the arena instance
    //     - other consumers that have the same Variable as an input --alive via internally-held ref count to the the Variable instance
    //     - explicit reference to the Variable in user code  --alive via externally-held ref count to the Variable instance
    // - at end of MemoizeInArena(),
    //   reset all Variables' shared_ptrs that fulfill one of the following conditions:
    //    - m_needsGradient flag is not set, or         // note: m_volatile is subsumed here already
    //    - not required for any gradient (considering also which gradients may never be requested)
    //      E.g. if gradient to inputs[0] needs output, but inputs[0].needsGradient == false, then it does not need output.
    //  - propagation rules for m_volatile and m_needsGradient:
    //     - m_volatile and m_needsGradient are *user-set* on Constants and Parameters (which can be frozen),
    //       and *inferred* through non-leaves as follows:
    //        - m_volatile      = any(input.m_volatile      for input in inputs)    // this Variable will never receive a gradient
    //        - m_needsGradient = any(input.m_needsGradient for input in inputs)    // this Variable needs to receive a gradient
    //                            and !m_volatile
    //                            and m_op != StopGradient

// ===========================================================================
// helper classes
// ===========================================================================

// ---------------------------------------------------------------------------
// cuda stats -- helper for profiling execution cost (CPU, GPU)
// ---------------------------------------------------------------------------

#ifdef DETAILED_STATS
class PCTimer // roll our own; high_resolution_timer is reportedly not high-resolution (0.1 us)
{
    double total;
#ifdef _WIN32
    LARGE_INTEGER freq, start;
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
#else // Linux
    timespec start;
public:
    // TODO: implement this for gcc/unix if needed
    PCTimer() { } // count ticks per second
    void Start() { clock_gettime(CLOCK_REALTIME/*HREAD_CPUTIME_ID*/, &start); }
    double Stop() // each read gives time elapsed since start, in seconds
    {
        timespec present;
        clock_gettime(CLOCK_REALTIME/*THREAD_CPUTIME_ID*/, &present);
	timespec temp;
	if (present.tv_nsec < start.tv_nsec)
        {
	  temp.tv_sec  = present.tv_sec - start.tv_sec - 1;
	  temp.tv_nsec = 1000000000 + present.tv_nsec - start.tv_nsec;
	}
        else
        {
	  temp.tv_sec  = present.tv_sec - start.tv_sec;
	  temp.tv_nsec = present.tv_nsec - start.tv_nsec;
	}
	double elapsed = temp.tv_sec + temp.tv_nsec * 1e-9;
        //fprintf(stderr, "TIMEDIFF: %d.%d - %d.%d = %.4f ms", (int)present.tv_sec, (int)present.tv_nsec, (int)start.tv_sec, (int)start.tv_nsec, elapsed * 1000.0);
        total += elapsed;
        return elapsed;
    }
#endif
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
    if (ShouldLogMemoizeStatsCUDA())
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
        if (ShouldLogMemoizeStatsCUDA())
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
                fprintf(stderr, "=> %Stotal launch + exec time: %.4f ms + %.4f ms\n", prefix.c_str(), 1000.0 *
                        totalLaunch, 1000.0 * totalExec);
            fflush(stderr);
        }
        cudaStats.clear();
    }
    // control how often this is active
}
// call this at start to eliminate potential carry-over from the previous minibatch
void ResetCudaStats(bool updateCounter)
{
    cudaStats.clear();
    if (updateCounter)
    {
        logMemoizeStatsCounter++;
        if (logMemoizeStatsCounter == logMemoizeStatsPeriod)
            logMemoizeStatsCounter = 0;
    }
}
#else // no DETAILED_STATS. Makes very little runtime difference.
typedef void CudaStats;
CudaStats* BeginCudaStats(PrimitiveOpType op, const wchar_t* opLabel, size_t category = 0, size_t totalElements = 1, const DeviceDescriptor& device = DeviceDescriptor::CPUDevice()) { return nullptr; }
void EndCudaStats(CudaStats* cudaStatsPtr, const DeviceDescriptor& device = DeviceDescriptor::CPUDevice()) { }
struct CudaStatsGuard
{
    template <typename ...CtorArgTypes>
    CudaStatsGuard(CtorArgTypes&& ...ctorArgs) { }
    void Stop() { }
    ~CudaStatsGuard() { }
};
// and this at the end of when you want to dump the log
void ShowCudaStats() { }
void ResetCudaStats(bool) { }
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
// BUGBUGBUGBUG: This class is a prototype hack, full of subtle bugs. Redo it.
// ---------------------------------------------------------------------------

class NDArrayViewArena
{
    // MemoryBlock -- represents one consecutive range of allocated memory
    struct MemoryBlock
    {
        MatrixBasePtr m_sob;    // reference to underlying storage object, to hold ref count
        uint8_t* m_data;        // data pointer, points inside m_sob's memory
        size_t   m_size;        // number of bytes
        MemoryBlock(const MatrixBasePtr& sob, uint8_t* data, size_t size) : m_sob(sob), m_data(data), m_size(size) {} // note: size in bytes
        // for set support --borrowed from Marian toolkit
        size_t   size() const { return m_size; } // note: bytes, not elements
        uint8_t* data() const { return m_data; }
        uint8_t* begin() const { return m_data; }
        uint8_t* end() const { return m_data + m_size; }
        bool operator<(const MemoryBlock& other) const // this defines the ordering of the set
        {
            return (m_size < other.size()) || (m_size == other.size() && m_data < other.data());
        }
        bool operator==(const MemoryBlock& other) const
        {
            return m_data == other.data() && m_size == other.size();
        }
        template<typename ElementType>
        static MemoryBlock Create(const MatrixBasePtr& sob, size_t firstIndex, size_t numElements)
        {
            let& mat = (const Matrix<ElementType>&)(*sob);
            auto* p = mat.Data();
            p += firstIndex;
            return MemoryBlock(sob, (uint8_t*)p, numElements * sizeof(ElementType));
        }
        static MemoryBlock Create(const MatrixBasePtr& sob, DataType dataType, size_t firstIndex, size_t numElements)
        {
            switch (dataType)
            {
            case DataType::Float:  return Create<float >(sob, firstIndex, numElements); break;
            case DataType::Double: return Create<double>(sob, firstIndex, numElements); break;
            default: LogicError("Unsupported DataType %s", DataTypeName(dataType));
            }
        }
        static MemoryBlock Create(DataType dataType, size_t numElements) // reduced version for lower_bound() argument
        {
            switch (dataType)
            {
            case DataType::Float:  return MemoryBlock(nullptr, nullptr, numElements * sizeof(float )); break;
            case DataType::Double: return MemoryBlock(nullptr, nullptr, numElements * sizeof(double)); break;
            default: LogicError("Unsupported DataType %s", DataTypeName(dataType));
            }
        }
    };
    struct Deleter : public IBaseMatrixStorageExternalBufferDeleter
    {
        //NDArrayViewArena& m_arena;
        MemoryBlock m_memoryBlock;
        Deleter(MemoryBlock&& memoryBlock, NDArrayViewArena& /*arena*/) :
            m_memoryBlock(move(memoryBlock)) //, m_arena(arena)
        {}
        virtual void Delete(void*) override // called when the BaseMatrixStorage instance is destructed
        {
            NDArrayViewArena::/*m_arena.*/RecycleMemoryBlock(move(m_memoryBlock));
            delete this; // this will deref the underlying arena, so that eventually it can be freed
            // TODO: merge the matrix and the Deleter into a single object that is MakeShared controlled
        }
    };

    static void CheckGaps()
    {
        size_t totalRecyclable = 0;
        for (let& b : s_recycledMemoryBlocks)
            totalRecyclable += b.size();
        if (totalRecyclable != s_totalGaps)
            Break;
    }
    static void RecycleMemoryBlock(MemoryBlock&& memoryBlock) // callback from Deleter()
    {
        lock_guard<recursive_mutex> guard(s_mutex);
        CheckGaps();
        s_totalAllocated -= memoryBlock.size();
        s_totalGaps += memoryBlock.size();
        s_recycledMemoryBlocks.emplace(move(memoryBlock)); // we move this gap into the gap set
        CheckGaps();
        // consolidate rigth away --TODO: currently this is highly inefficient; do it right
        if (s_recycledMemoryBlocks.size() > 1)
        {
            // consolidate by sorting by address, then merging
            s_mergeBuffer.assign(s_recycledMemoryBlocks.begin(), s_recycledMemoryBlocks.end());
            sort(s_mergeBuffer.begin(), s_mergeBuffer.end(), [](const MemoryBlock& a, const MemoryBlock& b) { return a.begin() < b.begin(); });
            size_t j = 1;
            for (size_t i = 1; i < s_mergeBuffer.size(); i++)
            {
                if (s_mergeBuffer[j - 1].end() == s_mergeBuffer[i].begin())
                    s_mergeBuffer[j - 1].m_size += s_mergeBuffer[i].size();
                else if (j != i)
                    s_mergeBuffer[j++] = s_mergeBuffer[i];
                else
                    j++;
            }
            if (j < s_recycledMemoryBlocks.size())
            {
                s_recycledMemoryBlocks.clear();
                for (size_t i = 0; i < j; i++)
                    s_recycledMemoryBlocks.insert(move(s_mergeBuffer[i]));
            }
            s_mergeBuffer.clear(); // release all ref counts
            CheckGaps();
        }
    }
    // allocate a new tensor in a large arena
    static recursive_mutex s_mutex;
    static const size_t numStorageFormats = 3; // we index the arrays below by [(size_t)storageFormat]
    static array<MatrixBasePtr                 , numStorageFormats> s_currentArenas;          // currently active arena (for the given storage format)
    static array<DataType                      , numStorageFormats> s_currentArenaDataTypes;  // == s_currentArena's dataType
    static array<DeviceDescriptor              , numStorageFormats> s_currentArenaDevices;    // == s_currentArena's device
    static array<size_t                        , numStorageFormats> s_currentArenaSizes;      // == s_currentArena's number of elements (== s_currentArena->GetNumElements())
    // allocation state
    static array<size_t                        , numStorageFormats> s_currentArenaUseds;      // allocation cursor. Elements below this are already allocated.
    // arenas no longer referenced get remembered here for reuse, to avoid GPU syncs
    static array<vector<unique_ptr<MatrixBase>>, numStorageFormats> s_recycledArenass;
    static set<MemoryBlock> s_recycledMemoryBlocks; // free set
    static const NDShapeDimension defaultDenseArenaSize = 256000000; // we allocate in this chunk size (dense only)

    static size_t s_totalAllocated; // diagnostics
    static size_t s_totalGaps;
    static vector<MemoryBlock> s_mergeBuffer;

    static bool IsMatrixOfDataType(const MatrixBase& matrix, DataType dataType) // helper for checking the DataType of the MatrixBase object
    {
        switch (dataType)
        {
        case DataType::Float:  return dynamic_cast<const Matrix<float>*>(&matrix) != nullptr;
        case DataType::Double: return dynamic_cast<const Matrix<float>*>(&matrix) != nullptr;
        default: LogicError("GetMatrixType: Unsupported element type.");
        }
    }
    MatrixBasePtr WrapStorageRangeAsMatrix(MemoryBlock&& memoryBlock, DataType dataType)
    {
        let* data = memoryBlock.data();
        let deviceId = memoryBlock.m_sob->GetDeviceId();
        let sizeInBytes = memoryBlock.size();
        auto* deleter = new Deleter(move(memoryBlock), *this);
        // PERF BUGBUG: ^^ This is a plain malloc(), not good. Use our allocator, and/or merge with Matrix object itself.
        switch (dataType)
        {
        case DataType::Float:  return MakeSharedObject<Matrix<float >>(/*rows=*/1, /*cols=*/sizeInBytes / sizeof(float ), (float *)data, deviceId, matrixFlagDontOwnBuffer, /*nnz=*/0, deleter);
        case DataType::Double: return MakeSharedObject<Matrix<double>>(/*rows=*/1, /*cols=*/sizeInBytes / sizeof(double), (double*)data, deviceId, matrixFlagDontOwnBuffer, /*nnz=*/0, deleter);
        default: LogicError("Unsupported DataType %s", DataTypeName(dataType));
        }       
    }
public:
    // allocate an NDArrayView of a given shape, data type, and device
    // The returned memory region is a slice into a much larger NDArrayView; therefore,
    // this operation short-circuits CUDA and is very fast.
    // Sparse objects cannot be arena-allocated. Which is fine, since they are inputs or
    // gradients (of embeddings) that can be kept around across minibatches, and thus not part of batched computation.
    // For sparse objects, it is assumed that the first axis is sparse.
    NDArrayViewPtr NewNDArrayView(const NDShape& shape, const DataType& dataType, StorageFormat storageFormat, const DeviceDescriptor& device)
    {
        lock_guard<recursive_mutex> guard(s_mutex);
        let isSparse = IsSparseStorageFormat(storageFormat);
        let formatAsIndex = (size_t)storageFormat;
        fail_if(formatAsIndex >= s_currentArenas.size(), "unexpected storageFormat int value??");
        auto& s_currentArena         = s_currentArenas        [formatAsIndex];
        auto& s_currentArenaSize     = s_currentArenaSizes    [formatAsIndex];
        auto& s_currentArenaUsed     = s_currentArenaUseds    [formatAsIndex];
        auto& s_currentArenaDataType = s_currentArenaDataTypes[formatAsIndex];
        auto& s_currentArenaDevice   = s_currentArenaDevices  [formatAsIndex];
        auto& s_recycledArenas       = s_recycledArenass      [formatAsIndex];
        // TODO: This ^^ calls for a struct!

        // for now, we only support one data type/device. If it changes,. we tear down the entire thing.
        if (dataType != s_currentArenaDataType || device != s_currentArenaDevice)
        {
            // tear down everything (we use ref-counting, so existing objects will continue to exist)
            // subtle BUGBUG: if we still have pending views of wrong type, they will go back into the recycling area
            s_currentArena.reset();
            s_currentArenaSize = 0;
            s_currentArenaUsed = 0;
            s_currentArenaDataType = dataType;
            s_currentArenaDevice = device;
            s_recycledArenas.clear();
            s_recycledMemoryBlocks.clear();
        }

        let numElements = shape.TotalSize(/*check=*/false);

        // --- sparse uses a different strategy
        // Sparse:
        //  Arena allocation is not possible, because  sparse matrices cannot be appended to.
        //  Sparse outputs are used rarely, and sparse CSC matrices are small, so for sparse, we just cache individual objects.
        if (isSparse)
        {
            // allocation size is arena size for dense, and the actual required size for sparse
            //if (shape.Rank() < 1)
            ////if (shape.Rank() != 1 && shape.Rank() != 2)
            //    InvalidArgument("NewNDArrayView(): Currently, only sparse vectors and matrices are supported (no tensors of higher ranks)."), fflush(stderr);
            // TODO: This is where we decide which dimensions are sparse. Currently only the first axis, hard-coded.
            let numRows = shape[0];
            let numCols = shape.Rank() > 1 ? shape.TotalSize() / numRows : 1;
            let matrixStorageFormat = AsCNTKImplMatrixFormat(storageFormat);
            // reuse an existing object if possible; otherwise create a new one
            MatrixBase* matrixPtr = nullptr;
            for (auto iter = s_recycledArenas.begin(); iter != s_recycledArenas.end(); ++iter)
            {
                let& thisMatrixPtr = *iter;
                if (thisMatrixPtr->GetDeviceId() == AsCNTKImplDeviceId(device) &&
                    IsMatrixOfDataType(*thisMatrixPtr, dataType) &&
                    thisMatrixPtr->GetFormat() == matrixStorageFormat)
                {
                    matrixPtr = iter->release();  // take back ownership from the unique_ptr
                    s_recycledArenas.erase(iter); // remove from recycling buffer
                    //fprintf(stderr, "@@ reactivating %s arena matrix of %d elements\n", isSparse ? "sparse" : "dense", (int)matrixPtr->GetNumElements()), fflush(stderr);
                    matrixPtr->Resize1(numRows, numCols, /*reserveNzElems=*/numCols, /*growOnly=*/true);
                    break;
                }
            }
            if (!matrixPtr) // no reusable found: create a new one
            {
                CudaStatsGuard cudaStatsguard(PrimitiveOpType::FutureValue, L"new arena NewNDArrayView", 3, numElements);
                try
                {
                    //fprintf(stderr, "@@ allocating %s arena matrix of %d elements (recycle buffer has %d entries)\n", isSparse ? "sparse" : "dense", (int)requiredDenseArenaSize, (int)s_recycledArenas.size()), fflush(stderr);
                    switch (dataType)
                    {
                    case DataType::Float:  matrixPtr = new Matrix<float >(numRows, numCols, AsCNTKImplDeviceId(device), MatrixType::SPARSE, matrixStorageFormat, /*nnz=*/numCols); break;
                    case DataType::Double: matrixPtr = new Matrix<double>(numRows, numCols, AsCNTKImplDeviceId(device), MatrixType::SPARSE, matrixStorageFormat, /*nnz=*/numCols); break;
                    default: LogicError("Unsupported DataType %s", DataTypeName(dataType));
                    }
                }
                catch (const exception& e)
                {
                    fprintf(stderr, "NewNDArrayView: Out of memory allocating sparse matrix: %s.\n", e.what());
                    throw;
                }
            }
            // create a shared ptr out of it with a destructor
            s_currentArena = shared_ptr<MatrixBase>(matrixPtr,
                [&s_recycledArenas](MatrixBase* matrixPtr)
                {
                    lock_guard<recursive_mutex> guard(s_mutex);
                    //fprintf(stderr, "@@ retiring %s arena of %d elements\n", matrixPtr->GetMatrixType() == MatrixType::SPARSE ? "sparse" : "dense", (int)matrixPtr->GetNumElements()), fflush(stderr);
                    // check the sob's ref count; if > 1 then there are other views into it, we cannot recycle it. It's a workaround.
                    if (matrixPtr->GetNumViews() == 1)
                    {
                        if (matrixPtr->GetMatrixType() == MatrixType::SPARSE)
                            matrixPtr->Reset(); // this resets the sparse matrix, but does not release its (small) memory
                        s_recycledArenas.push_back(unique_ptr<MatrixBase>(matrixPtr)); // don't release; rather keep it around
                    }
                    else
                    {
                        // if it still has a view, then we won't recycle it, just destruct the view, so that the storage object will be freed when the last view is done
                        // I think I have never observed this.
                        static bool errorShown = false;
                        if (!errorShown)
                        {
                            fprintf(stderr, "WARNING: NewNDArrayView cannot recycle arena because it is still used by Matrix object not under the control of the arena allocator. This message will only be shown once.\n"), fflush(stderr);
                            errorShown = true;
                        }
                        delete matrixPtr;
                    }
                });
            auto region = MakeSharedObject<NDArrayView>(dataType, shape, /*begin*/0, /*end*/numElements, s_currentArena);
            return region;
        }

        // --- dense

        let DebugSnapShot = [&]()
        {
            size_t totalRecyclable = 0;
            size_t maxRecyclable = 0;
            for (let& b : s_recycledMemoryBlocks)
            {
                totalRecyclable += b.size();
                maxRecyclable = max(maxRecyclable, b.size());
            }
            fprintf(stderr, "NDArrayView snapshot: %d recyclable gaps, max %d bytes, total %d bytes, %d recycable arenas, %f total alloc, %f total gaps\n",
                    (int)s_recycledMemoryBlocks.size(), (int)maxRecyclable, (int)totalRecyclable, (int)s_recycledArenas.size(), (double)s_totalAllocated, (double)s_totalGaps);
            fflush(stderr);
            fail_if(totalRecyclable != s_totalGaps, "s_totalGaps out of sync with gap list??");
        };

        static size_t debugCounter = 0;
        if (debugCounter++ % 10000 == 0)
            DebugSnapShot();

        // try to allocate from existing gaps first
        let desiredMemoryBlock = MemoryBlock::Create(dataType, numElements);
        auto iter = lower_bound(s_recycledMemoryBlocks.begin(), s_recycledMemoryBlocks.end(), desiredMemoryBlock);
        if (iter != s_recycledMemoryBlocks.end()) // found one
        {
            CheckGaps();
            s_totalGaps -= iter->size(); // check it out
            // for unused bytes, we create a new recycled memory block
            let unusedBytes = iter->size() - desiredMemoryBlock.size();
            //if (unusedBytes > 0)
            //    fprintf(stderr, "splitting off %d unused bytes\n", (int)unusedBytes);
            if (unusedBytes > 0)
            {
                s_recycledMemoryBlocks.emplace(iter->m_sob, iter->data() + desiredMemoryBlock.size(), unusedBytes);
                s_totalGaps += unusedBytes;
            }
            s_totalAllocated += desiredMemoryBlock.size();
            let matrixViewPtr = WrapStorageRangeAsMatrix(MemoryBlock(iter->m_sob, iter->data(), desiredMemoryBlock.size()), dataType);
            //fprintf(stderr, "reused %d bytes\n", (int)desiredMemoryBlock.size());
            auto region = MakeSharedObject<NDArrayView>(dataType, shape, /*begin*/0, /*end*/numElements, matrixViewPtr);
            s_recycledMemoryBlocks.erase(iter);
            CheckGaps();
            return region;
        }

        //  If arena not large enough then waste its remainder and just allocate a fresh one.
        //  This abandons the current m_arena. This will not cause a memory leak, however:
        //  Since the slices into it that were returned before all hold a ref-count to that arena,
        //  it will be deallocated automatically as soon the last slice goes away.
        //  The deleter, though, will intercept that and retire it into the recycledArenas array.
        //  Next time we need a new arena, we will look there first and recycle one.
        //  If the data type is different, we drop the current arena. CNTK can'really presently mix dataTypes properly anyway, so this is ah-OK.
        if (!s_currentArena                                         ||
            numElements > (s_currentArenaSize - s_currentArenaUsed))
        {
            let requiredDenseArenaSize = max(defaultDenseArenaSize, numElements);
            s_currentArena.reset(); // abandon current one. If no references, then this will put itself into recycledArenas right here
            // get hold of an arena; either by recycling an existing one, or creating a new one
            MatrixBase* matrixPtr = nullptr;
            for (auto iter = s_recycledArenas.begin(); iter != s_recycledArenas.end(); ++iter)
            {
                let& thisMatrixPtr = *iter;
                if (requiredDenseArenaSize <= thisMatrixPtr->GetNumElements()  &&
                    thisMatrixPtr->GetDeviceId() == AsCNTKImplDeviceId(device) &&
                    IsMatrixOfDataType(*thisMatrixPtr, dataType))
                {
                    matrixPtr = iter->release();  // take back ownership from the unique_ptr
                    s_recycledArenas.erase(iter); // remove from recycling buffer
                    //fprintf(stderr, "@@ reactivating %s arena matrix of %d elements\n", isSparse ? "sparse" : "dense", (int)matrixPtr->GetNumElements()), fflush(stderr);
                    break;
                }
            }
            // allocation size is arena size for dense, and the actual required size for sparse
            let numCols = requiredDenseArenaSize + 1; // we add 1 so that we won't accidentally merge two consecutive arenas when merging gaps
            if (!matrixPtr) // create a new one
            {
                CudaStatsGuard cudaStatsguard(PrimitiveOpType::FutureValue, L"new arena NewNDArrayView", 3, numElements);
                try
                {
                    //fprintf(stderr, "@@ allocating %s arena matrix of %d elements (recycle buffer has %d entries)\n", isSparse ? "sparse" : "dense", (int)requiredDenseArenaSize, (int)s_recycledArenas.size()), fflush(stderr);
                    switch (dataType)
                    {
                    case DataType::Float:  matrixPtr = new Matrix<float >(1, numCols, AsCNTKImplDeviceId(device)); break;
                    case DataType::Double: matrixPtr = new Matrix<double>(1, numCols, AsCNTKImplDeviceId(device)); break;
                    default: LogicError("Unsupported DataType %s", DataTypeName(dataType));
                    }
                }
                catch (const exception& e)
                {
                    fprintf(stderr, "NewNDArrayView: Out of memory allocating %d elements: %s.\n", (int)numElements, e.what());
                    DebugSnapShot();
                    throw;
                }
            }
            s_currentArena = shared_ptr<MatrixBase>(matrixPtr,
                [&s_recycledArenas](MatrixBase* matrixPtr)
                {
                    lock_guard<recursive_mutex> guard(s_mutex);
                    //fprintf(stderr, "@@ retiring %s arena of %d elements\n", matrixPtr->GetMatrixType() == MatrixType::SPARSE ? "sparse" : "dense", (int)matrixPtr->GetNumElements()), fflush(stderr);
                    // check the sob's ref count; if > 1 then there are other views into it, we cannot recycle it. It's a workaround.
                    if (matrixPtr->GetNumViews() == 1)
                    {
                        if (matrixPtr->GetMatrixType() == MatrixType::SPARSE)
                            matrixPtr->Reset(); // this resets the sparse matrix, but does not release its (small) memory
                        s_recycledArenas.push_back(unique_ptr<MatrixBase>(matrixPtr)); // don't release; rather keep it around
                    }
                    else
                    {
                        static bool errorShown = false;
                        if (!errorShown)
                        {
                            fprintf(stderr, "WARNING: NewNDArrayView cannot recycle arena because it is still used by Matrix object not under the control of the arena allocator. This message will only be shown once.\n"), fflush(stderr);
                            errorShown = true;
                        }
                        delete matrixPtr;
                    }
                });
            s_currentArenaDataType = dataType;
            s_currentArenaDevice = device;
            s_currentArenaUsed = 0;
            s_currentArenaSize = s_currentArena->GetNumElements();
        }
        // We create a new matrix storage object that wraps a pointer into the arena.
        // Any view into such object will reference the same underlying storage object.
        // The storage object has a custom deleter. We use that to track gaps.
        let matrixViewPtr = WrapStorageRangeAsMatrix(MemoryBlock::Create(s_currentArena, dataType, /*firstIndex=*/s_currentArenaUsed, numElements), dataType);
        auto region = MakeSharedObject<NDArrayView>(dataType, shape, /*begin*/0, /*end*/numElements, matrixViewPtr);
        s_currentArenaUsed += numElements;
        s_totalAllocated += MemoryBlock::Create(dataType, numElements).size(); // (unnecessarily slow, but this is for diags only)
        return region;
    }
};

/*static*/ recursive_mutex NDArrayViewArena::s_mutex;
/*static*/ array<MatrixBasePtr                     , NDArrayViewArena::numStorageFormats> NDArrayViewArena::s_currentArenas;
/*static*/ array<DataType                          , NDArrayViewArena::numStorageFormats> NDArrayViewArena::s_currentArenaDataTypes;// = { DataType::Unknown, DataType::Unknown };
/*static*/ array<DeviceDescriptor                  , NDArrayViewArena::numStorageFormats> NDArrayViewArena::s_currentArenaDevices = { DeviceDescriptor::CPUDevice(), DeviceDescriptor::CPUDevice(), DeviceDescriptor::CPUDevice() };
/*static*/ array<size_t                            , NDArrayViewArena::numStorageFormats> NDArrayViewArena::s_currentArenaSizes;
/*static*/ array<size_t                            , NDArrayViewArena::numStorageFormats> NDArrayViewArena::s_currentArenaUseds;
/*static*/ array<vector<unique_ptr<MatrixBase>>    , NDArrayViewArena::numStorageFormats> NDArrayViewArena::s_recycledArenass;
/*static*/ set<NDArrayViewArena::MemoryBlock> NDArrayViewArena::s_recycledMemoryBlocks;
/*static*/ size_t NDArrayViewArena::s_totalAllocated = 0;
/*static*/ size_t NDArrayViewArena::s_totalGaps = 0;
/*static*/ vector<NDArrayViewArena::MemoryBlock> NDArrayViewArena::s_mergeBuffer;

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
    const int m_verbosity;
    const wstring m_name;
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

// a few helper functions--sort these
static const NDShapeDimension ABSENT_FREE_DIMENSION = (NDShapeDimension)(-1);
static inline NDShapeDimensions ReplaceFreeDim(const NDShapeDimensions& shape, NDShapeDimension batchDimValue);

// ===========================================================================
// Memoizer -- this performs all actual computations
// This uses a parallel thread in order to mask CUDA launch overhead.
// All actual operations that create/update Variable::m_value and m_gradient
// are routed through this class, by calling SubmitForward()
// ...and the respective backward function once we are there
// Before accessing an actual m_value field, call Join().
// ===========================================================================

inline VariableFields& PrimitiveFunction::GetOutputFields() const { return VariableFields::FromVariable(m_outputs.front()); }

// this gets the non-redirected data fields, which describe v's properties as an input
// This returns the fields of a potentially virtual value that does not produce its own output but merely views another.
static inline VariableFields& GetInputFields(const InternalVariable& v) { return VariableFields::FromVariable(v); }

// this gets the fields of the output of 'f', which describe where f's output goes, without redirection
// It is *not* following a redirect.
// If f's result is the result of a batched result, then this function returns
// the slice (or potentially where the slice would go), not the original place.
static inline VariableFields& GetOutputFields(const PrimitiveFunction& f) { return f.GetOutputFields(); }

// helper that is needed at a few places
// Reshape an NDArrayViewPtr in-place (replace by a new one) if its shape does not match.
// Thread-safety: This is called by MTCacheAndGetValue() and backprop. Eventually this should move inside the background thread.
static void ReplaceWithReshapedViewIfNeeded(NDArrayViewPtr& view, const NDShape& shape)
{
    if (view->Shape() != shape)
        view = view->AsShape(shape);
}

#if 0
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
static const NDArrayView* GetValueObject(const InternalVariable& v)
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
#endif

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

// predicate whether an op is a matrix product
// These have special treatment for the bacthing of the weight matrix.
static bool IsTimesOp(PrimitiveOpType op)
{
    return
        op == PrimitiveOpType::Times ||
        op == PrimitiveOpType::TransposeTimes ||
        op == PrimitiveOpType::Affine ||
        op == PrimitiveOpType::TransposeAffine;
}

#if 0
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
#endif

template<class B> // B=vector<NDArrayViewPtr>
B& BorrowBuffer(B& buffer, size_t batchSize)
{
    if (buffer.capacity() < batchSize)
        buffer.reserve(batchSize * 2);
    buffer.resize(batchSize);
    return buffer;
}

// a Win32-like Event type
class Event
{
    mutex m_mutex;
    condition_variable m_condition;
    bool m_state;
public:
    Event() : m_state(true) {}
    void Set()
    {
        lock_guard<mutex> guard(m_mutex);
        if (!m_state)
        {
            m_state = true; // remember that Event is in flagged state, for consumer that may be waiting in the futur
            m_condition.notify_all(); // notify consumer that may be waiting *right now*
        }
    }
    void Wait()
    {
        unique_lock<mutex> lk(m_mutex);
        m_condition.wait(lk, [this]() { return m_state; }); // releases the lock while waiting
        m_state = false;
    }
};

// simple helper class for running a worker thread
// Once the run function throws an exception, the instance must be destructed.
class WorkerThread
{
    bool m_enableThreading = true; // set to false for emulating behavior via in-thread immediate computaton
    thread m_thread;
    volatile bool m_terminateRequest; // set by WorkerThread destructor; bg thread must return asap, destructor will wait
    Event m_consumerStateChanged, m_workerStateChanged;
    void MTNotifyConsumerOfWorkerStateChange()
    {
        //fprintf(stderr, __FUNCTION__ " %d\n", __LINE__), fflush(stderr);
        m_workerStateChanged.Set(); // my state has changed
        //fprintf(stderr, __FUNCTION__ " %d\n", __LINE__), fflush(stderr);
    }
    void MTWaitForConsumerStateChange()
    {
        //fprintf(stderr, __FUNCTION__ " %d\n", __LINE__), fflush(stderr);
        m_consumerStateChanged.Wait();
        //fprintf(stderr, __FUNCTION__ " %d\n", __LINE__), fflush(stderr);
    }
    void MTThreadProc()
    {
        MTNotifyConsumerOfWorkerStateChange(); // hello world!
        try
        {
            // main worker loop
            for (;;)
            {
                if (m_terminateRequest)          // object destructor was called: return without further ado
                    return;
                let hasWorkPending = MTDoWork(); // may not throw
                if (hasWorkPending)              // if there is more known work to do, do that right away
                    continue;
                // queue has run empty: wait for replenishment
                MTNotifyConsumerOfWorkerStateChange(); // tell consumer that the worker is out of work
                MTWaitForConsumerStateChange();  // wait for consumer to tell us that something may have been done
            }
        }
        // uncaught exception. Note: MTDoWork() must not throw.
        catch (const exception& e)
        {
            fprintf(stderr, "Uncaught exception in background thread: %s\n", e.what()), fflush(stderr);
            abort(); // tear down the program immediately
        }
        catch (...)
        {
            fprintf(stderr, "Uncaught exception in background thread\n"), fflush(stderr);
            abort(); // tear down the program immediately
        }
        // cannot get here
    }
    void NonThreadedProc() // emulation if threading not enabled
    {
        while (MTDoWork()) // run all pending tasks
            ;
    }
public:
    ~WorkerThread()
    {
        m_terminateRequest = true;           // signal the thread to stop
        NotifyWorkerOfConsumerStateChange(); // if worker is in wait state, then get it out
        if (m_thread.joinable())             // and wait until it's gone (if it was not in wait state, it may still in the midst of processing one item)
            m_thread.join();
        fprintf(stderr, "WorkerThread: bg thread is gone\n"), fflush(stderr);
    }
    void Reset()
    {
        if (!m_enableThreading)
            return;
        if (!m_thread.joinable()) // thread does not exist yet
        {
            m_terminateRequest = false;
            m_thread = thread([this]() { MTThreadProc(); });
            fprintf(stderr, "WorkerThread: bg thread started\n"), fflush(stderr);
        }
    }
    virtual bool MTDoWork() noexcept = 0; // override this. Do one limited package of work. Returns true if it should be called again immediately
    void NotifyWorkerOfConsumerStateChange() // tell the worker that state has (may have) changed by main thread
    {
        //fprintf(stderr, __FUNCTION__ " %d\n", __LINE__), fflush(stderr);
        m_consumerStateChanged.Set();
        //fprintf(stderr, __FUNCTION__ " %d\n", __LINE__), fflush(stderr);
    }
    void WaitForWorkerStateChange() // wait for worker to change the state. Rethrows exception if the thread died
    {
        if (!m_enableThreading)
            return NonThreadedProc();
        //fprintf(stderr, __FUNCTION__ " %d\n", __LINE__), fflush(stderr);
        m_workerStateChanged.Wait();
        //fprintf(stderr, __FUNCTION__ " %d\n", __LINE__), fflush(stderr);
    }
};

// helper class like std::deque but thread-safe
template<typename T>
class ConcurrentQueue
{
    mutable recursive_mutex m_mutex; // guards m_queue
    deque<T> m_queue;
public:
    size_t Size() const noexcept { lock_guard<recursive_mutex> guard(m_mutex); return m_queue.size(); }
    bool Empty() const noexcept { lock_guard<recursive_mutex> guard(m_mutex); return m_queue.empty(); }
    void Clear() noexcept { lock_guard<recursive_mutex> guard(m_mutex); m_queue.clear (); }
    void EmplaceBack(T&& item) { lock_guard<recursive_mutex> guard(m_mutex); return m_queue.emplace_back(move(item)); }
    const T* Front() const noexcept // returns null if empty. Once done, use PopFront() to complete the operation.
    {
        lock_guard<recursive_mutex> guard(m_mutex);
        if (m_queue.empty())
            return nullptr;
        return &m_queue.front();
    }
    void PopFront() noexcept { lock_guard<recursive_mutex> guard(m_mutex); m_queue.pop_front(); }
};

class InternalVariable::Memoizer
{
    // one work item as queued
    struct WorkItem
    {
        Memoizer* us;
        size_t backpropTo;         // backprop: which input to backprop into; forward: SentinelForForward
        static const size_t SentinelForForward = SIZE_MAX;
        PrimitiveFunctionPtr fPtr; // the function to execute
        bool isFree;
        bool logSpliceAsGather;
    };
    // worker thread with callback and queue
    class MemoizeWorkerThread : public WorkerThread
    {
        typedef WorkerThread Base;
        ConcurrentQueue<WorkItem> m_queue;
        volatile bool m_hasException;     // set by MTDoWork() in case of exception
        exception_ptr m_exception;        // if any  --TODO: does this need to be volatile?
        // callback. Process one work item in each call.
        bool MTDoWork() noexcept override
        {
            if (!m_hasException)
            {
                try
                {
                    let* item = m_queue.Front();
                    if (!item)
                        return false;
                    // do some work. This may throw an exception upon error.
                    item->us->MTProcessNextItem(*item);
                    m_queue.PopFront();
                    // only now remove it, so that consumer sees it disappear only after it has been done
                    return !m_queue.Empty();
                }
                catch (...)
                {
                    // newly caught exception: save it
                    m_exception = current_exception();
                    m_hasException = true; // not sure if exception_ptr can be assigned atomically, so use this as a guard
                }
            }
            // an exception has been caught, or was already present
            // we get here in three occasions:
            //  - a newly caught exception from MTProcessNextItem()
            //  - an error occured in a work item, but fg thread then submitted more work before seeing the flag
            //  - fg thread's Reset() may have set this. The queue can only be shrunk in the worker thread, so do this here.
            m_queue.Clear(); // what's already there, clear it
            // Note: fg thread might just have submitted another item before m_hasException was set.
            // That is OK. Worker thread will do one more round, only to find the queue empty already.
            return false;
        }
    public:
        // before submitting the first thing in a run, call Reset()
        // If there is still a dangling operation, it will be terminated, and its error code ignored.
        void Reset()
        {
            m_hasException = true; // if still some bg activity then tell MTDoWork() to skip all of that
            NotifyWorkerOfConsumerStateChange(); // (not sure if it is needed, but won't harm)
            while (Pending() > 0)  // and wait until all operations have ceased.
                WaitForWorkerStateChange(); // note: This will re-throw any exception from the worker
            // The m_queue items are removed after bg thread has complete them, so at this point we are sure the bg thread is idle.
            // now reset the thread state.
            Base::Reset(); // reset thread. This really just starts it if it is not there yet.
            m_hasException = false; // if still some bg activity then skip all of that
            m_exception = nullptr;
        }
        // after the last submission, and before accessing any resulting m_value, call Join()
        // In case of an error, this will forward the error.
        void Join()
        {
            //fprintf(stderr, "### QUEUE SIZE=%d\n", (int)s_workerThread.Pending()), fflush(stderr);
            NotifyWorkerOfConsumerStateChange();
            // wait until all done
            while (Pending() > 0)
                WaitForWorkerStateChange(); // note: This will re-throw any exception from the worker
            // if error then pass to caller
            if (m_hasException)
                rethrow_exception(m_exception);
        }
        // main operation, call for each work item
        void Submit(WorkItem&& item)
        {
            if (m_hasException) // if in error state, ignore it
                return;
            m_queue.EmplaceBack(move(item));
            NotifyWorkerOfConsumerStateChange();
        }
        size_t Pending() const { return m_queue.Size(); }
    };
    // methods that run on worker thread
    // get the value of an input Variable, with full redirection
    // This will realize any lazy ops (slice, reshape).
    // A Variable's value is defined by its m_redirection.m_function->m_outputs.front(), followed by slice and/or reshape.
    // Thread-safety: Currently called by Memoizer::Forward(), BatchedForward() after Join()
    //                 TODO: CSE still uses this in a validation check. Remove once confirmed.
public: // for call in BatchedForward()  --TODO: clean this up
    static const NDArrayViewPtr& MTCacheAndGetValue(VariableFields& fields)
    {
        if (!fields.m_value) // value is not available: create and cache it
        {
            fail_if(fields.m_redirection.empty(), "Variable unexpectedly has no value yet, nor is it a slice view into a batched op");
            // get the actual value from the function that computed it
            auto& functionFields = GetOutputFields(*fields.m_redirection.m_function);
            fail_if(&fields == &functionFields, "Variable unexpectedly has no value yet"); // avoid infinite recursion
            // function itself may be a redirect (a slice into a batched op)
            MTCacheAndGetValue(functionFields); // (calling ourselves on the source in case of re-redirection)
            // realize the lazy redirected value; that is, do the slice and reshape
            // This sets fields.m_value. inputFields must have m_value already set.
            fail_if(!functionFields.m_value, "Variable's input unexpectedly has no value yet");
            // optional implicit index and reshape
            let sliceRange = fields.m_redirection.m_sliceRange;
            if (!sliceRange.empty())
                fields.m_value = move(functionFields.m_value->SliceViewAsShape(sliceRange.BeginIndex(), sliceRange.EndIndex(), fields.m_shape));
            else // no slice
            {
                fields.m_value = functionFields.m_value;
                ReplaceWithReshapedViewIfNeeded(fields.m_value, fields.m_shape);
            }
        }
        return fields.m_value; // return the (now) cached value
    }
private:
    static const NDArrayViewPtr& MTCacheAndGetValue(const Variable& v)
    {
        return MTCacheAndGetValue(GetInputFields(v));
    }
    vector<NDArrayViewPtr> m_inputValuesBuffer; // buffer for extracted NDArrayViews of inputs
    void MTForward(PrimitiveFunction& f, bool isFree, bool logSpliceAsGather)
    {
        // fetch the NDArrayViewPtrs for all inputs
        let& inputs = f.m_inputs;
        CudaStatsGuard cudaStatsGuardPrepareI(PrimitiveOpType::Pooling, L"Memoize: MT prep inputs", 3, inputs.size());
        auto& inputValues = BorrowBuffer(m_inputValuesBuffer, inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            // special treatment for BatchNorm which needs additional temp buffers, which we keep track of as additional inputs
            // TODO: disentangle xHat (remove scale/bias and keep xHat as the main output) from the temp buffers
            if (f.m_op == PrimitiveOpType::BatchNormalization && i >= 6)
                GetInputFields(inputs[i]).m_value = m_arena.NewNDArrayView(inputs[i].Shape(), inputs[i].GetDataType(), StorageFormat::Dense, inputValues.front()->Device());
            // fetch the m_value from the input
            inputValues[i] = MTCacheAndGetValue(inputs[i]); // (if this is a redirect, then now we must resolve it)
        }
        cudaStatsGuardPrepareI.Stop();
        CudaStatsGuard cudaStatsGuardPrepareO(PrimitiveOpType::Log, L"Memoize: MT prep output", 3, inputs.size());
        // allocate the output NDArrayViewPtr in the arena
        let& output = f.m_outputs.front(); // BUGBUG: How to deal with multi-valued functions?
        let& outputShape = output.Shape();
        let& outputDevice = /*if*/ (f.m_op == PrimitiveOpType::RandomDistribution) ? // special case: generators have no inputs; they must have an init device instead
                                AsDeviceDescriptor((*(RNGState*)f.m_attributes[PrimitiveFunction::AttributeNameRandomDistributionRNGHandle].Value<size_t>())->DeviceId())
                            /*else*/ :
                                inputValues.front()->Device();
        auto outValue =
            /*if*/ (isFree) ?
                NDArrayViewPtr()
            /*else*/:
                m_arena.NewNDArrayView(outputShape, output.GetDataType(),
                                       output.IsSparse() ? StorageFormat::SparseCSC : StorageFormat::Dense,
                                       outputDevice);
        cudaStatsGuardPrepareO.Stop();
        // logging
        if (Memoizer::ShouldProfile(f))
            Memoizer::LogFunction(f, f.m_profiler);
        //if (f.m_op == PrimitiveOpType::ElementTimes)
        //    LogFunction(f, L"bf  ");
        CudaStats* cudaStatsPtr = nullptr;
        if (ShouldLogMemoizeStats())
        {
            let hasSparse = any_of(inputs.begin(), inputs.end(), [](const Variable& v) { return v.IsSparse(); });
            let logAsOp = (f.m_op == PrimitiveOpType::Splice && logSpliceAsGather) ? PrimitiveOpType::Gather : f.m_op; // gather ops are logged as op Gather (CNTK V2 Gather is not used by Dynamite)
            cudaStatsPtr = BeginCudaStats(logAsOp, nullptr, IsViewOp(f.m_op) ? 2 : hasSparse ? 1 : 0, outputShape.TotalSize(/*check=*/false), outputDevice);
        }
        //if (f.m_op == PrimitiveOpType::Slice && !any_of(inputs.begin(), inputs.end(), [](const Variable& v) { return v.IsSparse(); }))
        //    Break;
        // execute it
        //if (output.m_dataFields->m_uniqueIdForDebugging == 125515)
        //    Break;
        auto res = PrimitiveFunction::Forward(f.m_op, f.Attributes(), output.IsVolatile(), inputValues, outputShape, move(outValue), f);
#if 0
        if (output.m_dataFields->m_uniqueIdForDebugging == 125515)
        {
            for (size_t i = 0; i < inputs.size(); i++)
                inputValues[i]->LogToFile(L"###" + std::to_wstring(i), stderr, 800);
            res->LogToFile(L"###->", stderr, 800);
            fflush(stderr);
        }
#endif
        EndCudaStats(cudaStatsPtr, outputDevice);
        // special case: a Slice op is non-contiguous. We must copy.
        // TODO: Do this lazily for inputs of ops that cannot handle non-contiguous data, or for externally requested values.
        if (f.m_op == PrimitiveOpType::Slice && !res->IsContiguous())
        {
            let hasSparse = any_of(inputs.begin(), inputs.end(), [](const Variable& v) { return v.IsSparse(); });
            CudaStatsGuard cudaStatsGuard(f.m_op, nullptr, hasSparse ? 1 : 0, outputShape.TotalSize(/*check=*/false));
            res = move(NDArrayView::NumericOperation({ move(res) }, 1.0, Microsoft::MSR::CNTK::ElementWiseOperator::opCopy,
                                                     m_arena.NewNDArrayView(outputShape, output.GetDataType(),
                                                                            output.IsSparse() ? StorageFormat::SparseCSC : StorageFormat::Dense,
                                                                            inputValues.front()->Device())));
        }
#if 0
        Memoizer::LogFunction(f, f.m_profiler);
        res->LogToFile(f.Name());
#endif
        GetOutputFields(f).m_value = move(res);
        // we can free the inputs if they are no longer needed
        // TODO: For now this only helps for inference. For training, we need a more elaborate check.
#if 0
        // BUGBUG: (1) still has an ElementTimes with isVolatile; (2) crashes in CUDA
        for (auto& input : f.m_inputs)
        {
            if (!input.NeedsGradient())
                input.Reset();
        }
#endif
    }
    void MTProcessNextItem(const WorkItem& item)
    {
        if (item.backpropTo == WorkItem::SentinelForForward)
            MTForward(*item.fPtr, item.isFree, item.logSpliceAsGather);
        else
            LogicError("MTProcessNextItem: backprop not yet implemented");
    }
public:
    Memoizer()
    {
        // BUGBUG: The bg thread can theoretically be used from multiple consuming threads.
        //         But they share the queue. Hence, if an exception is thrown, it is not clear
        //         how the queue should be cleaned up, and how all consumers receive the exception.
        //         So for now, use this only from one thread.
    }
    ~Memoizer()
    {
        s_workerThread.Reset(); // wait for bg thread to flush all pending work
    }
    // submit a Function evaluation
    void SubmitForward(PrimitiveFunction& f, bool isFree, bool logSpliceAsGather)
    {
        s_workerThread.Submit(WorkItem
        {
            this,
            /*backpropTo=*/ WorkItem::SentinelForForward,
            static_pointer_cast<PrimitiveFunction>(f.shared_from_this()),
            isFree,
            logSpliceAsGather
        });
#if 0   // for testing: simulate sync operation via the bg thread
        s_workerThread.Join();
#endif
    }
    // enclose all usage in these two
    void Begin()
    {
        s_workerThread.Reset();
    }
    void End() // sync with thread and catch its error state if any
    {
        s_workerThread.Join(); // will throw any caught error
    }
    // temporarily, the arena is also used from outside
    // Once we go fully multi-threaded, then this should no longer be exposed.
    NDArrayViewArena& Arena() { return m_arena; }

    // helper to check whether we should profile this function execution
public:
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

    // profiling-related
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
        fprintf(stderr, "%S^%d%S = %S (", uid.c_str(), (int)GetInputFields(output).m_uniqueIdForDebugging, outputShape.AsString().c_str(), f.OpName().c_str());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            let& input = inputs[i];
            let& fields = GetInputFields(input); // (the fields that describe input as an input, e.g. its shape after potential see-through ops)
            // little helper function to fix up variable names by removing _Output_0
            // TODO: Once we support >1 output, this needs a bit more code.
            let GetVarName = [](const InternalVariable& input) -> wstring
            {
                auto uid = input.Uid();
                if (uid.size() > 9 && wcscmp(uid.c_str() + uid.size() - 9, L"_Output_0") == 0)
                    uid.resize(uid.size() - 9);
                let& inputFields = GetInputFields(input);
                let& name = !inputFields.m_redirection.empty() ? inputFields.m_redirection.m_function->Name() : input.Name();
                if (!name.empty())
                    uid = name + L":" + uid;
                let uidForDebugging = !inputFields.m_redirection.empty() ? GetOutputFields(*inputFields.m_redirection.m_function).m_uniqueIdForDebugging : GetInputFields(input).m_uniqueIdForDebugging;
                uid += L"^" + to_wstring(uidForDebugging);
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
                if (input1.Shape() != fields.m_shape)
                    fprintf(stderr, "r%S", input.Shape().AsString().c_str());
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
private:
    static MemoizeWorkerThread s_workerThread;
    NDArrayViewArena m_arena; // helper to allocate NDArrayViews as slices into very large NDArrayView objects
    // TODO: arena is all-static; change to s_arena, and make Arena class itself not static?
    set<DynamicProfilerPtr> m_profilersUsed; // all used profilers will be registered for a given batched execution
};
/*static*/ InternalVariable::Memoizer::MemoizeWorkerThread InternalVariable::Memoizer::s_workerThread;

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
    PrimitiveFunction& front() { return *begin(); }
    const PrimitiveFunction& front() const { return *begin(); }
    bool empty() const { return !head; }
    size_t size() const { return count; }
    void clear()
    {
        head = nullptr;
        count = 0;
    }
    template<typename PrimitiveFunctionType>
    class FunctionListIterator : public std::iterator<forward_iterator_tag, PrimitiveFunctionType>
    {
        PrimitiveFunctionType* iter;
    public:
        FunctionListIterator(PrimitiveFunctionType* f) : iter(f) { }
        PrimitiveFunctionType* operator->() const { return iter; }
        operator PrimitiveFunctionType*() const { return iter; }
        PrimitiveFunctionType& operator*() const { return *iter; }
        FunctionListIterator operator++() { auto cur = *this; operator++(1); return cur; }
        FunctionListIterator operator++(int) { iter = iter->m_autoBatchState.m_link; return *this; }
        //bool operator==(const FunctionListIterator& other) { return iter == other.iter; }
        bool operator!=(const FunctionListIterator& other) { return iter != other.iter; }
        //difference_type operator-(const FunctionListIterator& other) const { fail_if(true, "x"); }
    };
    typedef FunctionListIterator<PrimitiveFunction> iterator;
    typedef FunctionListIterator<PrimitiveFunction const> const_iterator;
    typedef PrimitiveFunction& value_type;
    iterator       begin()        { return head; }
    iterator       end()          { return nullptr; }
    const_iterator begin()  const { return head; }
    const_iterator end()    const { return nullptr; }
    const_iterator cbegin() const { return head; }
    const_iterator cend()   const { return nullptr; }
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
// //If you want this to nest, then remember the return value from Begin() and call End() with it.
// If you want this to nest, then create one on the stack, and pass it around by value (it is small enough).
// ---------------------------------------------------------------------------

class VisitorTag
{
    static std::atomic_size_t s_nextVisitTag; // unique id for a single non-nested visiting process
    size_t m_visitTag;
public:
    VisitorTag() { Begin(); } // constructor used when creating a VisitorTag on the stack
    void Begin() // call this at start when using existing objects
    {
        m_visitTag = s_nextVisitTag++; // note: atomic = thread-safe
    }
    bool Visited(size_t& tag) const // first time this is applied to 'tag', it will return false; and true otherwise
    {
        if (tag == m_visitTag)
            return true;
        tag = m_visitTag;
        return false;
    }
};
/*static*/ std::atomic_size_t VisitorTag::s_nextVisitTag{ 1 };


// ===========================================================================
// AutoBatch -- autobatching happening inside here
// The auto-batching related functions are grouped inside a class, since they
// share quite a bit of state.
// ===========================================================================

class InternalVariable::AutoBatch
{
    using SliceRange = Internal::AutoBatchRedirection::SliceRange;
    using StackingMode = PrimitiveFunction::StackingMode;

    Memoizer m_memoizer;
    RuntimeStatistics m_stats;
    VisitorTag m_visitorTag; // helper for managing tree traversal (non-nested)

    // buffers e.g. for building NDArrayViewPtr vectors. Kept as class members to avoid repeated memory allocations.
    vector<NDArrayViewPtr>     m_inputValuesBuffer;
    vector<NDArrayViewPtr>     m_inputValuesBuffer2;
    vector<NDArrayViewPtr>     m_outputGradientsBuffer;
    vector<const NDArrayView*> m_inputValuesBufferRaw;
    vector<size_t>             m_dimsBuffer;

    // =======================================================================
    // forward-related functions
    // =======================================================================

    // helper to test whether a PrimitiveFunction is a barrier operation
    friend class Invocable;
    static bool IsBarrier(const PrimitiveFunction& f)
    {
        return f.m_op == PrimitiveOpType::BarrierOp && f.m_attributes.Size() > 0;
    }

    // inlining of a composite 
    // This makes a deep copy of the graph below 'f', which must live in a composite owned by a BlockFunction; *not* the dynamic graph.
    // The actual PrimitiveFunction copy is made via the cloneFn. This way, this function can be used
    // for both the initial unbatched inlining (no basic block), as well for inlining basic blocks during execution.
    // This function checks whether all substituted Placeholders have the same batch dimension (FreeDimension).
    //  - At the root of a composite, pass invocationArgsFreeDim=0; inside here, pass the value as is.
    //  - Upon return, it will contain the one batch dim value that is shared across all substituted placeholders.
    // Any not-yet-initialized constants or parameters are initialized here,
    // unless isShapeIsKnown == false. This is the case when inlining during construction
    // of a static graph.
    template<class F>
    static PrimitiveFunctionPtr RInlineComposite(PrimitiveFunction& clonee, const Function::InputsVectorType& invocationArgs,
                                          /*in/out*/NDShapeDimension& invocationArgsFreeDim, /*out*/ NDShapeDimension& inlinedFreeDim,
                                          const F& cloneFn, VisitorTag compositeVisitorTag, bool shapeIsKnown)
    {
        // if we already cloned this one then just return the clone
        if (compositeVisitorTag.Visited(clonee.m_autoBatchState.m_visitedTag))
        {
            let fInlined = clonee.m_autoBatchState.m_link; // we remembered the clone here
            if (!fInlined) // if we get here before this was filled in then we have discovered a cycle
                InvalidArgument("Invoke() cannot be used on composites with cycles.");
            inlinedFreeDim = clonee.m_autoBatchState.m_batchDim;
            return static_pointer_cast<PrimitiveFunction>(const_cast<PrimitiveFunction*>(fInlined)->shared_from_this()); // (shared_from_this() gives us a FunctionPtr, not PrimitiveFunctionPtr)
        }
        clonee.m_autoBatchState.m_link = nullptr; // Bring into valid state so we can detect cycles. Gets overwritten once we are done cloning.

        // clone clonee
        // The clonee lives inside a composite, not in a Dynamic graph.
        // First clone the clonee's inputs;
        inlinedFreeDim = ABSENT_FREE_DIMENSION; // resulting batch dim
        let& cloneeInputs = clonee.m_inputs;
        Function::InputsVectorType inlinedInputs(cloneeInputs.size(), Variable()); // (note: this is one malloc per cloned PrimitiveFunction. inlinedInputs then gets moved, not copied, into that function)
        for (size_t i = 0; i < cloneeInputs.size(); i++)
        {
            let& cloneeInput = cloneeInputs[i];
            let& cloneeInputFields = GetInputFields(cloneeInput);
            auto thisFreeDim = ABSENT_FREE_DIMENSION;

            // --- case 0: a Variable that already has a value; that is, a Constant, Parameter, or any result of another Dynamite invocation
            if (cloneeInputFields.m_varKind == VariableKind::Constant || cloneeInputFields.m_varKind == VariableKind::Parameter || cloneeInputFields.m_value)
            {
                if (!cloneeInputFields.m_value && shapeIsKnown)
                {
                    // TODO: why is this needed? Is this the right place to do it?
                    cloneeInput.Value(); // this is a Parameter for which we still need to run the initializer. This does that.
                    fail_if(!cloneeInputFields.m_value, "Parameter/Constant has no Value()??");
                }
                // Note: By not touching inlinedFreeDim, we are not handling the case that constants may have a batch axis.
                //       However, this only affects Constants that live inside the clonee, which cannot actually have one.
                inlinedInputs[i] = cloneeInput;
            }
            // --- case 1: a Placeholder: substitute with invocation arg
            else if (cloneeInputFields.m_varKind == VariableKind::Placeholder)
            {
                let argIndex = cloneeInputFields.m_compositeArgumentIndex;
                fail_if(argIndex >= invocationArgs.size(), "no invocation arg lined was up with clonee->Arguments[]??");
                let& operand = invocationArgs[argIndex];
                // cloneeInput = placeholder; operand = what it should pretend to be
                // Typecheck.
                // PERF BUGBUG: This typecheck is done repeatedly for the same argument. Can we save that effort?
                let& placeholderDims = cloneeInput.Shape().Dimensions();
                let& operandDims = operand.Shape().Dimensions();
                let placeholderRank = placeholderDims.size();
                let operandRank = operandDims.size();
                let placeholderHasFreeDimension = (placeholderRank > 0 && placeholderDims.back() == NDShape::FreeDimension);
                let itemRank = placeholderRank - placeholderHasFreeDimension;
                //if (placeholderHasFreeDimension)
                //    Break;
                // itemRank = shape components that must match
                if (operandRank < itemRank)
                    InvalidArgument("Invoke: Operand shape %S has too few axes to match placeholder's shape %S.", operand.Shape().AsString().c_str(), cloneeInput.Shape().AsString().c_str());
                for (size_t k = 0; k < itemRank; k++)
                    if (operandDims[k] != placeholderDims[k])
                        InvalidArgument("Invoke: Operand shape %S incompatible with placeholder's shape %S.", operand.Shape().AsString().c_str(), cloneeInput.Shape().AsString().c_str());
                // determine the free dimension
                let freeAxis = itemRank;
                thisFreeDim = (placeholderHasFreeDimension && freeAxis < operandRank) ? operandDims[freeAxis] : ABSENT_FREE_DIMENSION;
                if (invocationArgsFreeDim == ABSENT_FREE_DIMENSION)
                    invocationArgsFreeDim = thisFreeDim;
                else if (thisFreeDim != ABSENT_FREE_DIMENSION && invocationArgsFreeDim != thisFreeDim)
                    InvalidArgument("Invoke: Incompatible replacement for FreeDimension %d (previous: %d) for placeholder's shape %S.", (int)thisFreeDim, (int)invocationArgsFreeDim, cloneeInput.Shape().AsString().c_str());
                // OK!
                inlinedInputs[i] = operand;
            }
            // --- case 2: an Output Variable: clone the Function
            else
            {
                fail_if(cloneeInputFields.m_varKind != VariableKind::Output, "RInlineComposite encountered a non-output unexpectedly");
                let fInlinedPtr = RInlineComposite(*cloneeInputFields.Owner(), invocationArgs, /*in/out*/ invocationArgsFreeDim, /*out*/ thisFreeDim, cloneFn, compositeVisitorTag, shapeIsKnown);
                inlinedInputs[i] = Variable(fInlinedPtr->m_outputs.front(), ConstFunctionPtr(), fInlinedPtr);
                //inlinedInputs[i].m_acyclicOutputPrimitiveReference = fInlinedPtr;
                // ^^ the inlined input now holds a ref count to the function that generated it. This will move into and therefore be held by the newly inlined function below.
            }
            // update inlinedFreeDim
            // This is the batch dimension that the result should get.
            // For arguments that have a batch dimension, it must match. It is possible that argument have none.
            // For example, in x - ReduceMean(x), the second arg has no batch dim.
            if (inlinedFreeDim == ABSENT_FREE_DIMENSION) // first encounter
                inlinedFreeDim = thisFreeDim;
            else if (thisFreeDim != ABSENT_FREE_DIMENSION && inlinedFreeDim != thisFreeDim)
                InvalidArgument("Invoke: Inconsistent replacement for FreeDimension %d (previous: %d) for placeholder's shape %S.", (int)thisFreeDim, (int)inlinedFreeDim, cloneeInput.Shape().AsString().c_str());
        }
        // now create a new function and set up the outputs
        // 'inlinedFreeDim' will replace the FreeDimension.
        let fInlinedPtr = cloneFn(clonee, move(inlinedInputs), inlinedFreeDim);
        // and finally remember where this function got redirected to.
        clonee.m_autoBatchState.m_link = fInlinedPtr.get();
        clonee.m_autoBatchState.m_batchDim = inlinedFreeDim;
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
            else if (IsTimesOp(op)) // Times(): first arg must be the same object
            {
                IncorporateFieldsId(GetInputFields(inputs[0]));
                IncorporateType(GetInputFields(inputs[1]));
                if (inputs.size() == 3) // Affine
                    IncorporateType(GetInputFields(inputs[2]));
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
        vector<size_t> m_bnPendingCounts = vector<size_t>(1000, 0);       // [bn id] number of pending (non-ready) BatchNormalization operations. We need 0, so why not allocate a few more already
        // TODO: This must be turned into something hashable.
        // test whether two PrimitiveFunctions can be executed as a single batched operation
        static bool AreBatchable(const PrimitiveFunction& a, const PrimitiveFunction& b) // TODO: change to & (NULL is not a valid value here)
        {
            //CudaStatsGuard cudaStatsGuard(PrimitiveOpType::Equal, L"AreBatchable()", 3);
            // check the hash first  --turns out this is slower
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
            // batch axis must match. Note that m_batchAxis reflects a decision on stacking vs. batching.
            let batchAxis = a.m_autoBatchState.m_batchAxis;
            fail_if(batchAxis == SIZE_MAX - 2, "m_batchAxis not initialized??");
            if (batchAxis != b.m_autoBatchState.m_batchAxis)
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
                // TODO: Analyze the interplay with batchAxis.
                let& aFields = GetInputFields(a.m_inputs.front());
                let& bFields = GetInputFields(b.m_inputs.front());
                // shapes and data types must match
                // BUGBUG: How about strides?
                let rank = aFields.m_shape.Rank();
                if (rank != bFields.m_shape.Rank())
                    if (aId == 0)
                        return false;
                    else
                        InvalidArgument("Primitive op '%S' encountered two instances of the same id %d with different-rank shapes %S and %S.",
                                        PrimitiveOpTypeName(op).c_str(), (int)aId, a.m_inputs.front().Shape().AsString().c_str(), b.m_inputs.front().Shape().AsString().c_str());
                // shapes must have same rank and match up to the batch axis
                // If the batch axis is not outside the shape, then this is the stacking case.
                for (size_t k = 0; k < rank && k < batchAxis; k++)
                {
                    if (aFields.m_shape[k] != bFields.m_shape[k])
                        if (aId == 0)
                            return false;
                        else
                            InvalidArgument("Primitive op '%S' encountered two instances of the same id %d with different shapes %S and %S.",
                                            PrimitiveOpTypeName(op).c_str(), (int)aId, a.m_inputs.front().Shape().AsString().c_str(), b.m_inputs.front().Shape().AsString().c_str());
                }
                // check the remaining parameters (stats and "un-stats")--requires object identity
                for (size_t i = 1; i < 6; i++)
                {
                    if (a.m_inputs[i].m_dataFields != b.m_inputs[i].m_dataFields)
                        if (aId == 0)
                            return false;
                        else
                            InvalidArgument("Primitive op '%S' encountered two instances of the same id %d with different %d-th argument.",
                                PrimitiveOpTypeName(op).c_str(), (int)aId, (int)i);
                    fail_if(a.m_inputs[i].m_dataFields->m_shape.Rank() > batchAxis, "BatchNorm: shape of a statistic parameter exceeds batch axis??");
                }
                return true; // these *must* be batched
            }
#if 0       // for debugging: disable all batching except for BatchNorm were it is required
            static bool firstTime = true;
            if (firstTime)
            {
                fprintf(stderr, "NO BACTHING/STACKING except BN\n");
                firstTime = false;
            }
            return false;
#else
            // BUGBUG: Somehow this specific operation changes the result if stacked.
            //         Classification as STACKING only happens for sparse.
            //if (op == PrimitiveOpType::ElementTimes && a.m_attributes.Size() > 0 && a.m_autoBatchState.m_stacking == StackingMode::STACKING)
            //    return false;
            // special case for Block
            let isBlock = (op == PrimitiveOpType::Block);
            if (isBlock)
            {
                // composites must match (object identity or fully equivalent graph)
                let& aComposite = static_cast<const BlockFunction&>(a).Composite();
                let& bComposite = static_cast<const BlockFunction&>(b).Composite();
                if (CacheAndGetBatchableCompositeId(static_pointer_cast<CompositeFunction>(aComposite)) != CacheAndGetBatchableCompositeId(static_pointer_cast<CompositeFunction>(bComposite)))
                    return false;
            }
            // all input dimensions must match (with exception of a few special cases)
            let ShapeAndDataTypeMatches = [&](const VariableFields& aFields, const VariableFields& bFields)
            {
                // BUGBUG: How about strides? Intuitively, strides should be OK, since auto-batching involves a copy that undoes the stride.
                let rank = aFields.m_shape.Rank();
                if (rank != bFields.m_shape.Rank())
                    return false;
#if 1
                // shapes must have same rank and match up to the batch axis
                // If the batch axis is not outside the shape, then this is the stacking case.
                for (size_t k = 0; k < rank && k < batchAxis; k++)
                    if (aFields.m_shape[k] != bFields.m_shape[k])
                        return false;
#else
                // this branch disables stacking (unless special rare cases where it is forced)
                if (aFields.m_shape != bFields.m_shape)
                    return false;
#endif
                if (aFields.m_dataType != bFields.m_dataType)
                    return false;
                return true;
            };
            let isTimes = IsTimesOp(op);
            let numInputs = a.m_inputs.size();
            for (size_t i = 0; i < numInputs; i++)
            {
                let& aFields = GetInputFields(a.m_inputs[i]); // (we don't see through no-ops since the target shape is the right one to test)
                let& bFields = GetInputFields(b.m_inputs[i]);
                // there are a few special cases
                if ((isTimes && i == 0) ||
                    (isBlock && (aFields.m_varKind == VariableKind::Parameter || bFields.m_varKind == VariableKind::Parameter))) // BUGBUG: unnecessarily restrictive
                {
                    // for Times, the first arg must be the same object, not just the same shape
                    // TODO: a special case is a dot product, which we can write as ReduceSum(ElementTimes(a,b))
                    //       This would require to rewrite the graph though; can we do that?
                    // for Block, the built-in references to Parameters must match by object, since for now, we cannot substitute them. with a batched version
                    if (&aFields != &bFields)
                        return false;
                }
                else
                {
                    // shapes and data types must match
                    if (!ShapeAndDataTypeMatches(aFields, bFields))
                        return false;
                    if (aFields.m_isSparse != bFields.m_isSparse)
                        return false;
                    // depth hint must match
#ifndef NO_BARRIER
                    if (aFields.m_redirection.m_depthHint != bFields.m_redirection.m_depthHint)
                        return false;
#endif
#if 0
                    // sparse cannot be batched beyond rank 1
                    if (aFields.m_isSparse && rank > 1)
                        return false;
#endif
                }
            }
            // special case: random generation has no inputs, so we must check the output shape
            if (op == PrimitiveOpType::RandomDistribution)
            {
                let& aFields = GetOutputFields(a);
                let& bFields = GetOutputFields(b);
                if (!ShapeAndDataTypeMatches(aFields, bFields))
                    return false;
            }
            // attributes must also match
            if (a.m_attributes != b.m_attributes) // TODO: this could be an expensive operation; check that. We only need to compare for some ops; for most, attributes are already known from the shapes.
                return false;
            // all match: we can batch
            return true;
#endif
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
                vector<vector<weak_ptr<CompositeFunction>>> allComposites; // [m_basicBlockInfo.m_batchableCompositeId]
                // Note: This function implicitly handles the special case of Times() via AreBatchable().
                // Any embedded Times() operation must match object identity of the first argument.
                // That will never match if that argument is computed inside the basic block,
                // but it will work if it is a Parameter or a value computed outside that is the same object.
                // This runs only once, so it can take its time.
                static bool AreCompositesBatchable(const CompositeFunctionPtr& compositePtr, const CompositeFunctionPtr& otherCompositePtr)
                {
                    fail_if(compositePtr == otherCompositePtr, "AreCompositesBatchable comparing with itself??");
#if 0               // BUGBUG: I noticed a slight loss after I added batching of composites. It is not clear where it is from. Verify this!
                    //         Actually it seems that Parameters did not get replaced by batched versions.
                    return false;
#else
                    // parameters must be the same
                    // BUGBUG: This is too restrictive, but for now, the late inlining code cannot handle batching of Parameters.
                    //         So for now, require them to be identical, not just matching shape.
                    //         BUGBUG: ^^ Somehow the code seemed (seems) to indeed handle it, but I cannot see how it did that; I think it is buggy.
                    let aParameters =      compositePtr->Parameters();
                    let bParameters = otherCompositePtr->Parameters();
                    if (aParameters != bParameters)
                        return false;
                    // we must compare the two composites' graphs
                    vector<PrimitiveFunctionPtr> aNodes, bNodes;
                    Function::PreorderTraverseFunctions(     compositePtr->RootFunction(), [&](const FunctionPtr& fPtr) { aNodes.push_back(static_pointer_cast<PrimitiveFunction>(fPtr)); });
                    Function::PreorderTraverseFunctions(otherCompositePtr->RootFunction(), [&](const FunctionPtr& fPtr) { bNodes.push_back(static_pointer_cast<PrimitiveFunction>(fPtr)); });
                    if (aNodes.size() != bNodes.size())
                        return false;
                    // BUGBUG: This compares the original graph, not the short-circuited one. Is that sufficient?
                    for (size_t i = 0; i < aNodes.size(); i++)
                    {
                        let& a = *aNodes[i];
                        let& b = *bNodes[i];
                        //fail_if(a.m_inputs[i].Shape().IsUnknown(), "AreCompositesBatchable called for unknown shape??");
                        if (!AreBatchable(a, b))
                            return false;
                    }
                    //for (size_t i = 0; i < a.m_inputs.size(); i++)
                    //{
                    //    let& aInputFields = GetInputFields(a.m_inputs[i]);
                    //    let& bInputFields = GetInputFields(b.m_inputs[i]);
                    //    if (aInputFields.m_redirection.empty() != aInputFields.m_redirection.empty())
                    //        return false;
                    //    if (!aInputFields.m_redirection.empty())
                    //    {
                    //        // BUGBUG: we retraverse multiple paths for now. This will be rewritten and done properly.
                    //        // BUGBUG: We also must compare the actual graph link structure, not just the unrolled tree.
                    //        // BIG BUGBUG: AreBatchable considers m_batchAxis etc., which have not been set up for Primitives inside the composite.
                    //        //             ^^ Isn't this addressed in MatchAndRememberComposite(), which traverses the composite to set these?
                    //        if (!AreBatchable(*aInputFields.m_redirection.m_function, *bInputFields.m_redirection.m_function))
                    //            return false;
                    //    }
                    //}
                    return true;
#endif
                }
            public:
                size_t MatchAndRememberComposite(const CompositeFunctionPtr& compositePtr)
                {
                    // populate all composite's m_freeAxis etc. fields
                    let freeAxis = compositePtr->m_basicBlockInfo.m_freeAxis;
                    Function::PreorderTraverseFunctions(compositePtr->RootFunction(), [freeAxis](const FunctionPtr& fPtr)
                    {
                        auto& f = *dynamic_pointer_cast<PrimitiveFunction>(fPtr);
                        let batchAxisAndDim = DetermineBatchAxisAndDim(f);
                        f.m_autoBatchState.m_stacking  = batchAxisAndDim.stacking;
                        f.m_autoBatchState.m_batchAxis = batchAxisAndDim.batchAxis;
                        f.m_autoBatchState.m_batchDim  = batchAxisAndDim.batchDim;
                    });
                    // match new composite against all in the list by deep structure comparison
                    for (size_t i = 0; i < allComposites.size(); i++)
                    {
                        auto& otherList = allComposites[i];
                        if (otherList.empty()) // TODO: garbage collection and weak_ptr discovery should go in here
                            continue;
                        let otherCompositePtr = otherList.front().lock();
                        if (!otherCompositePtr)
                            continue; // TODO: keep looking for a non-expired one
                        //if (compositePtr->Name() == L"dense.normWeight5" || otherCompositePtr->Name() == L"dense.normWeight5")
                        //    Break;
                        let areBatchable = AreCompositesBatchable(compositePtr, otherCompositePtr);
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
            if (compositePtr->m_basicBlockInfo.m_batchableCompositeId == SIZE_MAX)
                compositePtr->m_basicBlockInfo.m_batchableCompositeId = matcher.MatchAndRememberComposite(compositePtr);
            return compositePtr->m_basicBlockInfo.m_batchableCompositeId;
        }
        // helper to determine the batch axis and its dimension
        // We decide stacking vs. batching here. There are two possible return values:
        //  - batching: axis = max rank, dim = 1 (since axis is outside of all shapes)
        //  - stacking: axis = max rank - 1; dim = max over those dims (inputs that do not have that axis can be considered virtually padded to 1. Effectively they are just skipped.)
        // The decision is purely based on the operation and input ranks. Input shapes (other than their rank) must not be considered.
        // BUGBUG: For Splice into a new axis, we must account for that new axis.
        // Note that this is also called for composites, where the last dimension may be FreeDimension. Should still work.
        struct DetermineBatchAxisAndDimResult { StackingMode stacking; size_t batchAxis; size_t batchDim; };
        static DetermineBatchAxisAndDimResult DetermineBatchAxisAndDim(const PrimitiveFunction& f)
#if 0   // temporarily disable STACKING
        {
            auto res = DetermineBatchAxisAndDim1(f);
            if (get<0>(res) == StackingMode::STACKING_BUT_MAY_BATCH)
            {
                get<0>(res) = StackingMode::BATCHING;
                get<1>(res)++;
                get<2>(res) = 1;
            }
            return res;
        }
        static DetermineBatchAxisAndDimResult DetermineBatchAxisAndDim1(const PrimitiveFunction& f)
#endif
        {
            // helper to read out a dimension
            let getLastDim = [](const Variable& input, size_t axis, StackingMode modeIfStacking) -> DetermineBatchAxisAndDimResult
            {
                let& inputShape = input.Shape().Dimensions();
                let inputRank = inputShape.size();
                if (axis == inputRank)
                    return{ StackingMode::BATCHING, axis, 1 };
                else if (axis + 1 == inputRank)
                    return{ modeIfStacking, axis, inputShape.back() };
                else
                    LogicError("getLastDim: batch axis is at wrong position");
            };
            // We use stacking, unless the following forces us to use batching:
            //  - all batchable input are scalars
            //  - the batch axis is touched by the unbatched operation, e.g. sliced or reduced over
            //  - there is abatchable input that is a sparse vector (i.e., of rank 1). (Math library cannot stack sparse along axis 0.)
            let op = f.m_op;
            if (op == PrimitiveOpType::BatchNormalization)
            {
                // BUGBUG: Get this from a parameter. For now only support vectors, batch axis is always 1.
                return getLastDim(f.m_inputs.front(), 1, StackingMode::STACKING);
            }
            else if (IsTimesOp(op)) // Times: stacking if input has a batch dimension already
            {
                let& inputs = f.m_inputs;
                let& output = GetOutputFields(f);
                let& left  = inputs[0];
                let& right = inputs[1];
                // for Affine there is also inputs[2], but we can ignore it, as its shape must be same or broadcast into the result of the matrix product
                let leftRank  = left .Shape().Rank();
                let rightRank = right.Shape().Rank();
                let reductionRankX2 = leftRank + rightRank - output.m_shape.Rank();
                // TODO: check this w.r.t. TransposeTimes
                let reductionRank = reductionRankX2 / 2;
                fail_if(reductionRank * 2 != reductionRankX2, "DetermineBatchAxisAndDim: reductionRank for matrix-product class not determined correctly");
                let outputRank = leftRank  - reductionRank;
                let mapRank    = rightRank - reductionRank;
#if 0
                if (right.IsSparse() && rightRank > 2)
                    fprintf(stderr, "DetermineBatchAxisAndDim: Sparse matrix product currently only supports vectors or matrices.\n");
                    //InvalidArgument("DetermineBatchAxisAndDim: Sparse matrix product currently only supports vectors or matrices.");
#endif
                // Note: We could batch Times ops that have the same sequence length. For now, those would be forced to be stacking.
                // Stacking is fine though in this case. Since there is no funky broadcasting involved, it is equally efficient (same kernel dims).
                return getLastDim(right, mapRank == 0/*single vector; no batch dim*/ ? reductionRank : rightRank - 1, StackingMode::STACKING);
            }
            else if (op == PrimitiveOpType::RandomDistribution) // RandomDistribution: never stack; batchAxis determined by output shape
            {
                let& outputFields = GetOutputFields(f);
                // batch axis is appended to output shape
                // Note that we will never actually batch, since this op cannot be batched. But it can be CSE'd.
                return{ StackingMode::BATCHING, outputFields.m_shape.Rank(), 1 };
            }
            // determine maxRank and lastDim over all batchable inputs
            // We leverage the fact that for any operation with multiple batchable inputs, the batch axis
            // can be shared (and therefore must be the max over the rank).
            // For example select(cond, x, y) may have a broadcasting 'cond' with less axes than x or y,
            // but the result always has a rank that is the max over all.
            // Note that Times(W,x) has only one batchable input, so this holds here as well.
            // TODO: Re-understand and validate the above for Splice().
            size_t maxRank = 0;
            NDShapeDimension lastDim = 1; // dimension of last axis encountered (actually max over them, due to broadcasting)
            bool hasSparse = false;
            let updateRankDimSparse = [&](const VariableFields& inputFields)
            {
                let& inputShape = inputFields.m_shape.Dimensions();
                let inputRank = inputShape.size();
                if (inputRank > maxRank)
                {
                    maxRank = inputRank;
                    lastDim = inputShape.back(); // previous lastDim was for the wrong axis; start over
                }
                else if (inputRank == maxRank && inputRank > 0)
                    lastDim = max(lastDim, inputShape.back()); // account for broadcasting
            };
            for (let& input: f.m_inputs) // (Note that this loop would be incorrect for matrix-product class ops. Those are already special-cased above.)
            {
                let& inputFields = GetInputFields(input);
                hasSparse = hasSparse || inputFields.m_isSparse;
                updateRankDimSparse(inputFields);
            }
            if (op == PrimitiveOpType::Block)
            {
                let composite = static_cast<const CompositeFunction*>(static_cast<const BlockFunction&>(f).Composite().get());
                //if (composite->Name() == L"doToOutput")
                //    Break;
                let freeAxis = composite->m_basicBlockInfo.m_freeAxis;
                if (freeAxis == SIZE_MAX) // if basic block has no free dimension, then we won't batch in the first place' but formally, let's say we batch it
                    return{ StackingMode::BATCHING, maxRank, 1 };
                else if (freeAxis == maxRank) // argument has no FreeDimension specified: stack, but dim on stacking axis is implied 1
                    return{ StackingMode::STACKING_BUT_MAY_BATCH, maxRank - 1, lastDim }; // BUGBUG: This seems to work, but I am not sure why.

                    //return{ StackingMode::STACKING_BUT_MAY_BATCH, freeAxis, 1 };
                else if (freeAxis + 1 == maxRank) // argument has a axis to fill the FreeDimension: use that
                    return{ StackingMode::STACKING_BUT_MAY_BATCH, freeAxis, lastDim };
                InvalidArgument("DetermineBatchAxis: The rank of an argument to a basic block invocation exceeds the declared free-axis position.");
            }
            if ((op == PrimitiveOpType::Splice && !hasSparse) || op == PrimitiveOpType::Reshape /*|| op == PrimitiveOpType::OneHot*/)
            {
                // BUGBUG: Without the !hasSparse condition, this will cause it to fail for an unbatched Splice() of sparse vectors.
                //         Think this through. Maybe now it no longer works for actual batching/stacking.
                //         Or maybe this condition is also wrong for dense?
                // BUGBUG: Is this correct for Reshape()?
                //         If Reshape does not touch the last axis, we can STACK, otherwise not, I think.
                //         Same for OneHot.
                updateRankDimSparse(GetOutputFields(f));
            }
            // sparse inputs can only be batched/stacked in axis 1  --TODO: No need, and already seems to work without this constraint.
#if 0
            if (hasSparse && (maxRank < 1 || maxRank > 2))
                fprintf(stderr, "DetermineBatchAxis: A sparse input with rank > 2 was encountered, which is not presently supported.\n");
                //InvalidArgument("DetermineBatchAxis: A sparse input with rank > 2 was encountered, which is not presently supported.");
#endif
            // decide stacking vs. batching
            let stackingAxis = maxRank - 1; // (gcc requires me to do this before the 'goto')
            if (maxRank == 0) // can only stack if inputs are not scalars
                goto mustBatch;
            switch (op) // does the unbatched op touch the (potential) stacking axis?
            {
            case PrimitiveOpType::ElementTimes:
                if (!f.m_attributes.Contains(PrimitiveFunction::AttributeNameAxis)) // if InnerProduct then treat like ReduceElements
                    break;
                // fall-through to ReduceElements for InnerProduct case
            case PrimitiveOpType::Splice:
            case PrimitiveOpType::ReduceElements:
                {
                    let& axis = f.m_attributes[PrimitiveFunction::AttributeNameAxis].Value<Axis>();
                    if (!axis.IsStaticAxis() || stackingAxis == (size_t)axis.StaticAxisIndex(/*checked=*/false))
                        goto mustBatch; // touched. Can't stack, must batch.
                }
                break;
            case PrimitiveOpType::Slice: // it's a view op, but we can get here in postprocessing basic-block composites
                // TODO: Once we short-circuit in static graphs, we should not get here at all.
                if (f.m_attributes.Contains(PrimitiveFunction::AttributeNameAxisVec)) // vector of slices
                {
                    let& axes = AsVector<Axis>(f.m_attributes[PrimitiveFunction::AttributeNameAxisVec].Value<vector<DictionaryValue>>());
                    for (size_t i = 0; i < axes.size(); i++)
                    {
                        let axisIndex = axes[i].StaticAxisIndex();
                        if (stackingAxis == (size_t)axisIndex)
                            goto mustBatch; // touched. Can't stack, must batch.
                    }
                }
                else // single slice
                {
                    let axis = f.m_attributes[PrimitiveFunction::AttributeNameAxis].Value<Axis>();
                    if (stackingAxis == (size_t)axis.StaticAxisIndex(/*checked=*/false))
                        goto mustBatch; // touched. Can't stack, must batch.
                }
                break;
            case PrimitiveOpType::OneHot: // TODO   --I think this can be removed. OneHotOp always adds one axis, so in input batching, we cater to it
                // BUGBUG: This is not complete. Think this through. For now, only used for scalar constant to vector expansion.
                return{ StackingMode::BATCHING, maxRank, 1 };
            case PrimitiveOpType::TransposeAxes:
                {
                    size_t maxAxisIndex = 0; // determine the largest axis involved in transposition
                    if (f.m_attributes.Contains(PrimitiveFunction::AttributeNameAxisVec))
                    {
                        const auto& perm = f.m_attributes[PrimitiveFunction::AttributeNameAxisVec].Value<std::vector<DictionaryValue>>();
                        for (size_t i = 0; i < perm.size(); i++)
                        {
                            let axisIndex = perm[i].Value<Axis>().StaticAxisIndex();
                            if (axisIndex != i && axisIndex > maxAxisIndex) // record largest axis that is involved in any axis swap
                                maxAxisIndex = axisIndex;
                        }
                    }
                    else
                    {
                        maxAxisIndex = max(f.m_attributes[PrimitiveFunction::AttributeNameAxis1].Value<Axis>().StaticAxisIndex(),
                                           f.m_attributes[PrimitiveFunction::AttributeNameAxis2].Value<Axis>().StaticAxisIndex());
                    }
                    if (maxAxisIndex >= stackingAxis) // transposing into the batch axis: can't stack
                        goto mustBatch;
                    break;
                }
            }
            // no problem case detected: we can stack
            if (!hasSparse)
                return{ StackingMode::STACKING_BUT_MAY_BATCH, stackingAxis, lastDim };
            else if (maxRank >= 2) // sparse matrices must be stacked along the last axis of the dense axes
            //else if (maxRank == 2) // sparse matrices must be stacked along axis 1
                return{ StackingMode::STACKING, stackingAxis, lastDim };
            else if (maxRank == 1) // sparse vectors must be batched along axis 1
                return{ StackingMode::BATCHING, maxRank, 1 };
            else
                LogicError("should not get here??");
        mustBatch: // problem case: we cannot batch
            // We get here if the operation touches the stacking axis, or if it is a scalar op.
            //if (hasSparse)
            //    Break;
            if (hasSparse && maxRank != 1)
                InvalidArgument("DetermineBatchAxis: Sparse matrices cannot be batched.");
            return{ StackingMode::BATCHING, maxRank, 1 }; // if we batch, the batch dimension is 1
        }

    public:
        // count an occurrence of a BatchNormalization with a given id
        void CountBatchNorm(size_t bnId)
        {
            //fail_if(bnId == 0, "batch norm id should not be 0!");
            if (bnId >= m_bnPendingCounts.size())
                m_bnPendingCounts.resize(bnId * 10, 0);
            m_bnPendingCounts[bnId]++;
        }
        // schedule an operation that has been confirmed to be ready
        // This is called for nearly every unbatched PrimitiveFunction, and must therefore be blazingly fast.
        void Schedule(PrimitiveFunction& f)
        {
            CudaStatsGuard cudaStatsGuard(PrimitiveOpType::ToSequence, L"Schedule()", 3, m_regularOps.size() * m_regularOps.size());
            let op = f.m_op;
            // special case BatchNormalization: we must account for all occurences
            if (op == PrimitiveOpType::BatchNormalization)
            {
                CudaStatsGuard cudaStatsGuard(PrimitiveOpType::BatchNormalization, L"Schedule(BN)", 3);
                let bnId = f.m_autoBatchState.m_batchNormId;
                fail_if(bnId != f.m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>(), "m_batchNormId not initialized correctly??"); // TODO: remove once confirmed not flagging for a while
                m_bnPendingCounts[bnId]--; // only those with pending count 0 are ready (exception: bnId == 0, caters to running-stats-BN only)
            }
            else if (op == PrimitiveOpType::Block)
            {
                // we also must count BatchNorm instances embedded inside basic-block composites
                let* composite = static_cast<const CompositeFunction*>(static_cast<const BlockFunction&>(f).Composite().get());
                for (let bnId : *composite->m_basicBlockInfo.m_batchNormIds)
                    m_bnPendingCounts[bnId]--;
            }
            // we manage two ready sets, since two common kinds are very simple
            //fail_if (IsBarrier(f), "m_barrierOps.push_back(f) should no longer be done"); // BUGBUG: We never get here since we now see through barriers for efficiency...
            // TODO: Do we even need to route view ops through the scheduler?
            if (IsViewOp(op))  // note: this is, with possibly a few exceptions, Slice()
                m_viewOps.push_back(&f); // (linked list)
            else
            {
                // determine stacking vs. batching, and corresponding m_batchAxis
                // TODO: Can this be done during graph construction? (multi-threadable)
                let batchAxisAndDim = DetermineBatchAxisAndDim(f);
                f.m_autoBatchState.m_stacking  = batchAxisAndDim.stacking;
                f.m_autoBatchState.m_batchAxis = batchAxisAndDim.batchAxis;
                f.m_autoBatchState.m_batchDim  = batchAxisAndDim.batchDim;
                fail_if(f.m_autoBatchState.m_batchDim == NDShape::FreeDimension, "FreeDimension made it into batchDim??");
                fail_if(f.m_autoBatchState.m_stacking == StackingMode::BATCHING && f.m_autoBatchState.m_batchDim != 1, "batching but batchDim != 1??");
                // this naive implementation just scans linearly
                // scan through all op sets to see if one is batchable with 'f'
                // So far this does not show up in profiling.
                for (auto& iter : m_regularOps) // (vector)
                {
                    if (AreBatchable(f, iter.front()))
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
        // This handles the special case of bnId == 0, for which this constraint is not applied.
        bool AnyBatchNormPending(const vector<NonOwningFunctionListBuilder>::const_iterator& iter)
        {
            let& f = iter->front();
            if (f.m_op == PrimitiveOpType::BatchNormalization)
            {
                let bnId = f.m_autoBatchState.m_batchNormId;
                fail_if(bnId != f.m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>(), "m_batchNormId not initialized correctly??"); // TODO: remove once confirmed not flagging for a while
                return bnId != 0 && m_bnPendingCounts[bnId] > 0;
            }
            else if (f.m_op == PrimitiveOpType::Block)
            {
                // we also must count BatchNorm instances embedded inside basic-block composites
                let* composite = static_cast<const CompositeFunction*>(static_cast<const BlockFunction&>(f).Composite().get());
                for (let bnId : *composite->m_basicBlockInfo.m_batchNormIds)
                    if (bnId != 0 && m_bnPendingCounts[bnId] > 0)
                        return true;
            }
            return false;
        }
        // select the next batched op to execute
        NonOwningFunctionList pop_best()
        {
            CudaStatsGuard cudaStatsGuard(PrimitiveOpType::PastValue, L"pop_best()", 3, m_regularOps.size());
            //for (auto iter = m_regularOps.begin(); iter != m_regularOps.end(); iter++)
            //    if (iter->front()->Name() == L"vecSoftmaxMinus")
            //        fprintf(stderr, "### %S pending\n", iter->front()->Name().c_str()), fflush(stderr);

            // try both queues, in priority order
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
                    diff = -((int)AnyBatchNormPending(iter) - (int)AnyBatchNormPending(best)); // BatchNormalization with pending inputs always loses
                    if (diff) goto got_diff;
#ifndef NO_BARRIER  // TODO: the existence of the Barrier op may already influence this
                    diff = -((int)iter->front().m_autoBatchState.m_depthHint - (int)best->front().m_autoBatchState.m_depthHint); // barrier: higher depthHint gets batched last
                    if (diff) goto got_diff;
#endif
                    diff = (int)iter->size() - (int)best->size();
                got_diff:
                    if (diff > 0)
                        best = iter;
                }
                // special case BatchNormalization
                if (AnyBatchNormPending(best))
                {
                    double blendTimeConstant = (*best).front().m_attributes[PrimitiveFunction::AttributeNameBlendTimeConstant].Value<double>();
                    if (blendTimeConstant == 0) // the only ready op is BN with some instances still pending -> error (I am not sure under which circumstances this may ever happen)
                        InvalidArgument("Primitive op '%S' with id %d must not be used in a recurrent loop (must not depend on its own output).",
                            PrimitiveOpTypeName(best->front().m_op).c_str(), (int)best->front().m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>());
                }
                // and remove this one from the list
                NonOwningFunctionList out = *best; // since NonOwningFunctionListBuilder uses unmanaged pointers, we can just copy it
                auto last = m_regularOps.end() - 1;
                if (best != last)
                    *best = move(*last);
                m_regularOps.pop_back();
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
    void RPrepareForwardGraphAndSchedule(const InternalVariable& var, size_t depth)
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
        //         We should call RPrepareForwardGraphAndSchedule() from inside the scheduler. But for that, we must separate out the schedule bit.
        // TODO: Move towards long-lived m_redirection, so that we can short-circuit things during graph building.

        // handle leaves
        // Leaves are Parameters, Constants, and also nodes that already have a value.
        if (fields.m_varKind == VariableKind::Input || fields.m_varKind == VariableKind::Placeholder)
            InvalidArgument("Dynamic Value() must not depend on Input or Placeholder.");
        let isParameterOrConstant = fields.m_varKind == VariableKind::Parameter || fields.m_varKind == VariableKind::Constant;
        if (isParameterOrConstant)
        {
            if (!fields.m_value)
            {
                var.Value(); // this is a Parameter for which we still need to run the initializer. This does that.
                fail_if(!fields.m_value, "Parameter/Constant has no Value()??");
            }
            fields.m_redirection.reset();
            m_stats.numLeafNodes++;
            return;
        }
        // above code deals with Constand and Parameter variables, below with Output variables

        if (fields.m_value) // this is a Variable that has been computed before
            return;

        // see through ops that do nothing
        auto& f = *fields.m_ownerFunction.lock().get(); // this is the only place we ever call lock(); after this, we just use the raw pointer (relying on immutability)
        let op = f.m_op;
        // unroll non-basic BlockFunctions (if it is a basic block, then it is unrolled during execution later)
        if (op == PrimitiveOpType::Block && !static_cast<BlockFunction&>(f).IsBasicBlock())
        {
#if 1
            // inlining is now done during graph construction
            fail_if(true, "non-basic-block inlining not done during invocation??");
#else
            // TODO: this is done outside now, so we can remove this code
            // make a deep copy of the block's root (PrimitiveFunction graph)
            // The Block is treated like a see-through op:
            //  - a deep copy of the Block is made (inlined), which connects to the Block node's inputs
            //  - the Block node, like a NoOp, gets its m_redirection set to point to the Block's copy.
            // Upon first call, the original Block function gets its dimensions inferred. I.e. it cannot be called twice with mismatching dimensions.
            // Besides that, the original Block function will remain unmodified.
            NDShapeDimension invocationArgsFreeDim = ABSENT_FREE_DIMENSION;
            NDShapeDimension inputsBatchDimDummy;
            auto inlinedRootPtr = RInlineComposite(static_cast<PrimitiveFunction&>(*f.BlockRoot()),
                                                   f.m_inputs, invocationArgsFreeDim, inputsBatchDimDummy,
                                                   /*cloneFn=*/ClonePrimitiveFunction,
                                                   VisitorTag(), /*shapeIsKnown=*/true);
            // This returns a shared_ptr to the deep copy of the root.
            // ^^ This currently does not inline nested blocks. Instead of cloning the BlockFunction, we should just unroll any nested non-basic-block right there.
            // prepare graph that we just unrolled
            // Note: We may just have pointed to yet another nested block, or a see-through op, which must be eliminated right here
            let& output = inlinedRootPtr->m_outputs.front();
            RPrepareForwardGraphAndSchedule(output, depth + 1);
            // set up linkage in our overlaid structure
            let& redirectedFields = GetInputFields(output);
            fields.m_redirection = redirectedFields.m_redirection; // we redirect to whatever the inlined one redirects to
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
#endif
        }

        // short-circuit see-through ops
        else if (IsAliasOp(op) || op == PrimitiveOpType::Reshape)
        {
            // TODO: what if input is non-continuous? Then Reshape should become a copy. Does this need to be addressed here?
            m_stats.numShortCircuitedNodes++;
            let& input = f.m_inputs.front();
            RPrepareForwardGraphAndSchedule(input, depth + 1);
            let &redirectedFields = GetInputFields(input);
            //fail_if(fields.m_redirection, "redirection set up here twice??");
            fields.m_redirection = redirectedFields.m_redirection; // we redirect to whatever the input redirects to
            // BUGBUG:!!!! How about the gradient?? We must know how to find the gradient!
            //        One solution is to redirect to the operation directly on top of the Parameter, not the parameter itself.
            if (fields.m_redirection.empty()) // redirects to leaves are currently not handled correctly (since redirects are based on Function, not Variable)  --TODO: change that
                InvalidArgument("Value(): See-through ops on leaves are currently not implemented.");
            // if a barrier then record the maximum depth hint encountered
            // Functions consuming this Variable are bound by the barrier (but not the function that produces the original value).
            if (IsBarrier(f)) // TODO: get this from a different attribute
                fields.m_redirection.m_depthHint = max(fields.m_redirection.m_depthHint, f.m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>());
            return;
        }

        // above code deals with redirected ops; below with non-redirected ones
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
        if (op == PrimitiveOpType::Block)
        {
            // we also must count BatchNorm instances embedded inside basic-block composites
            let* composite = static_cast<const CompositeFunction*>(static_cast<const BlockFunction&>(f).Composite().get());
            for (let bnId : *composite->m_basicBlockInfo.m_batchNormIds)
                m_schedule.CountBatchNorm(bnId);
        }

        // determine how many inputs are pending; and also recurse and set up the consumer list
        size_t pendingInputs = 0;
        size_t maxDepthHint = 0;
        let& inputs = f.m_inputs;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            let& input = inputs[i];
            // recursively traverse
            RPrepareForwardGraphAndSchedule(input, depth + 1);
            auto& inputFields = GetInputFields(input);
            auto& outputFields = inputFields.m_redirection.empty() ? inputFields : GetOutputFields(*inputFields.m_redirection.m_function);
            if (!outputFields.m_value) // (if input is a leaf, or the result of a prior invocation, it has a value)
            {
                pendingInputs++;
                // record ourselves as a consumer of the input
                // Note that RPrepareForwardGraphAndSchedule() will have reset this upon first visit of 'input'.
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

    // execute a basic block.
    // This clones a block's composite graph of PrimitiveFunctions, and evaluates it as it goes along.
    // Called only by InlineAndMemoizeBatchedBasicBlock() and the equivalent non-batched case in ExecuteBatchedOpAndUpdateSchedule().
    // This returns a new PrimitiveFunction(-Ptr) (so we can backprop through it), which will have its output's m_value set.
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
    PrimitiveFunctionPtr InlineAndMemoizeBatchedBasicBlock(const BlockFunction& block, const Function::InputsVectorType& invocationArgs, size_t batchAxis/*or SIZE_MAX*/, NDShapeDimension batchSize)
    {
        CudaStatsGuard cudaStatsguard(block.m_op, L"invoke basic block", 3, block.m_outputs.front().Shape().TotalSize(/*check=*/false));
        // BUGBUG:
        //  - if block contains BatchNorm, then the implied barrier is not accounted for in batching.
        //    If user code is correct, it will compute the right thing. But we cannot presently check or account for it.
        //  - this does not short-circuit operations. We need to do the same to the composite as we do to the dynamic graph in forward-graph building.

        // lambda (cloneFn) that RInlineComposite() calls for each cloned function in the composite
        let cloneFn = [this, batchAxis, batchSize](PrimitiveFunction& f, Function::InputsVectorType&& newInputs, NDShapeDimension compositeBatchDim) -> PrimitiveFunctionPtr
        {
            // This lambda must apply the passed in the composite's PrimitiveFunction 'f' to 'newInputs'.
            // The newInputs are batched; that is, their shapes may mismatch those of f's inputs in that they may have an additional batch axis.

            // determine if any of the inputs have a batch axis
            // If any, then the output will get one as well.
            let anyBatchedInputs = any_of(newInputs.begin(), newInputs.end(), [batchAxis](const Variable& input) { return input.Shape().Rank() >= batchAxis; });

            // clone and memoize this operation
            // We pass on batchAxis such that the output shape will be that of the original output shape of the Function in the
            // composite, augmented with the batch axis. However, if no input has a batch axis, then the output should have none.
            // PERF BUGBUG: newInputs is already a copy that we can just move
            // PERF BUGBUG: We should short-circuit the free ops for composites as well.
            let isFree = IsViewOp(f.m_op);
            return CreateAndMemoizeBatchedOp(f, move(newInputs), compositeBatchDim, anyBatchedInputs ? batchAxis : SIZE_MAX, batchSize, L"()"/*f0*/, isFree);
        };

        NDShapeDimension invocationArgsFreeDim = ABSENT_FREE_DIMENSION;
        NDShapeDimension compositeBatchDim;
        //let prevVisitorTag = m_compositeVisitorTag.Begin(); // (for detecting which of the composite's PrimitiveFunctions have already been expanded)
        let fInlinedPtr = RInlineComposite(static_cast<PrimitiveFunction&>(*block.BlockRoot()), invocationArgs,
                                           invocationArgsFreeDim, compositeBatchDim,
                                           cloneFn, VisitorTag(), /*shapeIsKnown=*/true);
        //m_compositeVisitorTag.End(prevVisitorTag); // (restore)
        return fInlinedPtr;
    }

    // create a PrimitiveFunction, execute it right away, and prep result as an input
    // This first executes RawPrimitiveFunction() and MemoizeInArena(),
    // and then returns the result in the form of a Variable suitable as an input to the next PimitiveFunction
    // by setting its m_acyclicOutputPrimitiveReference field.
    // This is a commonly needed pattern in auto-batched execution. All ops generated in here are known to be acyclic.
    template<typename ShapeType>
    Variable CreateAndMemoizeOpAsInput(PrimitiveOpType op, Function::InputsVectorType&& inputs, Dictionary&& attributes/*, const wstring& name*/,
                                       const ShapeType& shape, DataType dataType, bool isSparse,
                                       const DynamicProfilerPtr& profiler, const wchar_t* logPrefix,
                                       bool isFree = false, bool logSpliceAsGather = false)
    {
        auto fPtr = CreateAndMemoizeOp(op, move(inputs), move(attributes), shape, dataType, isSparse/*, name*/, profiler, logPrefix, isFree, logSpliceAsGather);
        let& output = fPtr->m_outputs.front();
        // To make the consumer of this hold a reference to this PrimitiveFunction, inject a strong ref to the copy of its Output.
        //input = output.CompositePreservingCopy(fPtr);         // without the acyclic trick, this works as well, but is not quite right since fPtr is not a Composite
        Variable input = Variable(output, FunctionPtr(), fPtr); // copy the Variable object (which is mostly a shared_ptr tp VariableFields which are not copied but shared)
        // TODO: Add the move overload to Variable()
        //input.m_acyclicOutputPrimitiveReference = move(fPtr); // and implant the reference to the Primitive that generated it, for reference countiung
        return input;
    }

    // create a PrimitiveFunction that is a batched version of a given function ('clonee'), and execute it right away
    // This is a wrapper around CreateAndMemoizeOp().
    // If a batchAxis is provided (!= SIZE_MAX), then the output will have that axis.
    // In that case, all inputs[*], unless not batched, must have the same batch axis (an example of a non-batched arg is the first arg of a matrix product).
    // Note that compositeBatchDim is only valid if we are cloning from a composite.
    // TODO: pass 'inputs' as rvalue ref, and move it below.
    PrimitiveFunctionPtr CreateAndMemoizeBatchedOp(const PrimitiveFunction& clonee, Function::InputsVectorType&& inputs, NDShapeDimension compositeBatchDim, size_t batchAxis, NDShapeDimension batchSize, const wchar_t* logPrefix, bool isFree)
    {
        // special case for basic blocks: need to clone the entire composite (for backprop)
        if (clonee.m_op == PrimitiveOpType::Block)
            return InlineAndMemoizeBatchedBasicBlock(static_cast<const BlockFunction&>(clonee), inputs, batchAxis, batchSize);

        // get the unbatched output shape, considering the case that 'clonee' lives inside a composite
        let& output = clonee.m_outputs.front();
        let& cloneeOutputShape = output.Shape();
        let& cloneeOutputDims = cloneeOutputShape.Dimensions();
        let mustReplaceFreeDimension = (!cloneeOutputDims.empty() && cloneeOutputDims.back() == NDShape::FreeDimension);
        if (mustReplaceFreeDimension)
            VerifyFreeDimensionReplacement(clonee.m_inputs, inputs, compositeBatchDim);

        let& unbatchedOutputShape = mustReplaceFreeDimension ? NDShape(ReplaceFreeDim(cloneeOutputDims, compositeBatchDim)) : cloneeOutputShape;

        const NDShape& shape = batchAxis != SIZE_MAX ? (batchAxis < unbatchedOutputShape.Rank() ? unbatchedOutputShape.SubShape(0, batchAxis) : unbatchedOutputShape).AppendAxis(batchAxis, batchSize) : unbatchedOutputShape;
        Dictionary attributes;
        clonee.Attributes().ShallowCloneTo(attributes); // (this just copies the shared_ptr, not the content)
#if 1   // a sanity check whether we batched correctly. Can be removed once this stuff works.
        if (IsTimesOp(clonee.m_op))
            fail_if(inputs.front().Shape() != clonee.m_inputs.front().Shape(), "attempted to batch the weight matrix of a matrix product??");
#endif
        return CreateAndMemoizeOp(clonee.m_op, move(inputs), move(attributes), shape, output.GetDataType(), output.IsSparse()/*, clonee.m_name*/, clonee.m_profiler, L"*"/*clonee*/, isFree);
    }

    // create a PrimitiveFunction and execute it right away
    // This executes RawPrimitiveFunction() and MemoizeInArena().
    // This is a commonly needed pattern in auto-batched execution. All ops generated in here are known to be acyclic.
    template<typename ShapeType>
    PrimitiveFunctionPtr CreateAndMemoizeOp(PrimitiveOpType op, Function::InputsVectorType&& inputs, Dictionary&& attributes,
                                            const ShapeType& shape, DataType dataType, bool isSparse/*, const wstring& name*/,
                                            const DynamicProfilerPtr& profiler, const wchar_t* logPrefix,
                                            bool isFree = false, bool logSpliceAsGather = false)
    {
        // create the object
        CudaStatsGuard cudaStatsguard(PrimitiveOpType::Pass, L"RawPrimitiveFunction", 3);
        auto fPtr = MakeSharedObject<PrimitiveFunction>(op, move(inputs), move(attributes), wstring()/*dummy for overload resolution; move(name)*/);
        // unfortunately output initialization must be separated out since it requires s shared_ptr to f
        let& fInputs = fPtr->m_inputs;
        let isVolatile    = any_of(fInputs.begin(), fInputs.end(), [](const Variable& input) { return input.IsVolatile();    }); // PERF BUGBUG: caller knows this already; should pass it in
        let needsGradient = !isVolatile &&
                            any_of(fInputs.begin(), fInputs.end(), [](const Variable& input) { return input.NeedsGradient(); }); // PERF BUGBUG: caller knows this already; should pass it in
        fPtr->InitOutput(OutputVariable(NDShape(shape)/*it's a &&*/, dataType, needsGradient, isSparse, isVolatile));
        //if (fPtr->m_uniqueIdForDebugging == 194962)
        //    Break;
        //if (!fPtr->m_outputs.front().NeedsGradient())
        //    Break;
        FinishConstructingPrimitiveFunction(*fPtr, profiler, logPrefix);
        cudaStatsguard.Stop();

        // execute it
        MemoizeInArena(*fPtr, isFree, logSpliceAsGather);
        return fPtr;
    }

    // call this after constructing a PrimitiveFunction from inside here, to set up the additional fields
    static __forceinline void FinishConstructingPrimitiveFunction(PrimitiveFunction& f, const DynamicProfilerPtr& profiler, const wchar_t* logPrefix)
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
    // This is submitted into the background Memoizer thread, which will eventually store the result in f.m_outputs.front()'s m_value.
    // Note: This breaks the immutability principle, in that it frees inputs that are known to be no longer needed.
    const void MemoizeInArena(PrimitiveFunction& f, bool isFree = false, bool logSpliceAsGather = false)
    {
        m_memoizer.SubmitForward(f, isFree, logSpliceAsGather);
        // stats
        if (isFree) // means we did not pass a data buffer for the result; any one we pass a buffer does actual work
            m_stats.numDoneFreeOps++;
        else if (f.m_op == PrimitiveOpType::Splice && logSpliceAsGather)
            m_stats.numDoneGatherOps++;
        else
            m_stats.numDoneOtherOps++;
    }

    // helper to verify that we figured out the FreeDimension replacement for a clonee correctly
    static void VerifyFreeDimensionReplacement(const Function::InputsVectorType& cloneeInputs, const Function::InputsVectorType& operands, NDShapeDimension newInputsFreeDim)
    {
        let arity = cloneeInputs.size();
        fail_if(arity > 0 && newInputsFreeDim == 0, "composite has batch dim but operands do not, and it passed typecheck??");
        for (size_t i = 0; i < arity; i++)
        {
            let& cloneeInput = cloneeInputs[i];
            let& operand = operands[i];
            let& cloneeInputDims = cloneeInput.Shape().Dimensions();
            let rank = cloneeInputDims.size();
            if (rank == 0 || cloneeInputDims.back() != NDShape::FreeDimension)
                continue;
            let freeAxis = rank - 1; // placeholder has a free axis, at this position
            let& operandDims = operand.Shape().Dimensions();
            // now test the condition we are after
            fail_if(freeAxis < operandDims.size()  && operandDims[freeAxis] != newInputsFreeDim, "newInputsFreeDim does not match the dimension passed for the placeholder's FreeDimension");
        }
    }

    static Dictionary ShallowCloneDictionary(const Dictionary& other)
    {
        Dictionary res;
        other.ShallowCloneTo(res); // note: shallow clone will not copy the map, just a shared_ptr. This only works if attributes are immutable, which is true inside auto-batch
        return res;
    }

    // this saves an instruction when checking the size of a vector
    template<class VectorType>
    static bool VecSizeIs(const VectorType& vec, size_t s)
    {
        return vec.end() == vec.begin() + s;
    }

    // simplified version avoiding moving unnecessary parameters around (esp. strings)
    static InternalVariable OutputVariable(NDShape&& shape, ::CNTK::DataType dataType, bool needsGradient, bool isSparse, bool isVolatile)
    {
        return InternalVariable(move(shape), VariableKind::Output, dataType, needsGradient, isSparse, isVolatile);
    }

    // clone a PrimitiveFunction
    // Subroutine of early inlining non-basic-block BlockFunctions during forward graph preparation.
    // These are non-batched. The output shape is determined here by either stripping the FreeDimension if no input has one, or replacing it.
    // Special consideration is given to nested BlockFunctions.
    // This runs in early inlining, and is thus time-critical. This function was at one point measured to take 10% of total Forward time,
    // so the code below has been hand-optimized by single-stepping through disassembly.
    static PrimitiveFunctionPtr ClonePrimitiveFunction(PrimitiveFunction& clonee, Function::InputsVectorType&& newInputs, size_t newInputsFreeDim)
    {
        CudaStatsGuard cudaStatsguard(PrimitiveOpType::Cos, L"ClonePrimitiveFunction", 3);
        // clone it
        PrimitiveFunctionPtr fCloned =
            /*if*/ (clonee.m_op == PrimitiveOpType::Block) ?                     // PERF BUGBUG: vvv short-circuit this as well
                MakeSharedObject<BlockFunction>(static_cast<BlockFunction&>(clonee).Composite(), MakeVector(newInputs), static_cast<BlockFunction&>(clonee).IsBasicBlock(), wstring(), wstring()/*static_cast<BlockFunction&>(clonee).OpName()), wstring(clonee.Name())*/)
            /*else*/:
                MakeSharedObject<PrimitiveFunction>(clonee.m_op, move(newInputs), move(ShallowCloneDictionary(clonee.m_attributes)), wstring()/*dummy for overload resolution*/);// , wstring(clonee.Name()));
        // Note: We can use make_shared since no shared_ptrs to these clones are ever exposed across the DLL boundary.
        //if (fCloned->m_uniqueIdForDebugging == 20000)
        //    fprintf(stderr, "");
        // unfortunately output initialization must be separated out since it requires s shared_ptr to fCloned
        // get the output shape
        let& outputs = clonee.m_outputs;
        if (!VecSizeIs(outputs, 1))
        //if (outputs.size() != 1)
            InvalidArgument("Dynamic operations cannot have multiple outputs.");
        let& output = outputs.front();
        let shapeIsKnown = !output.Shape().IsUnknown();
        //if (!shapeIsKnown)
        //    Break;
        // if output has a FreeDimension (which represents the batch axis) then replace (or drop) it
        let& outputDims = output.Shape().Dimensions();
        let mustReplaceFreeDimension = shapeIsKnown && !outputDims.empty() && outputDims.back() == NDShape::FreeDimension;
        fail_if(mustReplaceFreeDimension && newInputsFreeDim == 0, "composite has batch dim but operands do not, and it passed typecheck??");
#if 0
        // check that we got it right
        if (mustReplaceFreeDimension)
            VerifyFreeDimensionReplacement(clonee.m_inputs, fCloned->m_inputs, newInputsFreeDim);
#endif
        // initialize the output
        //let dataType = fCloned->m_inputs.front().GetDataType();
        fail_if(shapeIsKnown && output.GetDataType() == DataType::Unknown, "ClonePrimitiveFunction: output has no determined data type yet??");
        let isVolatile = any_of(fCloned->m_inputs.begin(), fCloned->m_inputs.end(), [](const Variable& input) { return input.IsVolatile(); });
        if (mustReplaceFreeDimension)
            fCloned->InitOutput(InternalVariable(NDShape(ReplaceFreeDim(outputDims, newInputsFreeDim)), VariableKind::Output, output.GetDataType(), output.NeedsGradient() && !isVolatile, output.IsSparse(), isVolatile));
        else
            fCloned->InitOutput(InternalVariable(NDShape(output.Shape()), VariableKind::Output, output.GetDataType(), output.NeedsGradient() && !isVolatile, output.IsSparse(), isVolatile));
        // Note: Somehow, OutputVariable() above does not get inlined, even with __forceinline.
        // add additional initializations for auto-batch only
        FinishConstructingPrimitiveFunction(*fCloned, clonee.m_profiler, /*logPrefix=*/nullptr);
        return fCloned;
    }

    // notify consumers of 'f' that f's value is now available
    void NotifyOpsConsumersInputsAvailable(const PrimitiveFunction& f)
    {
        // notify consumers
        auto& fields = GetOutputFields(f);
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
        fields.m_redirection.reset(batchedOp, sliceRange);
        // note: This overwrites depthHint. That's fine since depthHint is only used for scheduling. But this function just got a value, hence will never be scheduled.
        // Semantically, this will compute as fields.m_value = out[sliceRange]; but it gets deferred to save effort.
        // update all ops' consumers and schedule them when possible
        NotifyOpsConsumersInputsAvailable(f);
    }

    // clone a result into all its aliases (which were determined by ShortCircuitBatchedOpDuplicatesAndUpdateSchedule())
    void UpdateDuplicatesAndUpdateSchedule(PrimitiveFunction& f)
    {
        for (auto* dup = f.m_autoBatchState.m_aliasList; dup; dup = dup->m_autoBatchState.m_aliasList)
        {
            //CudaStatsGuard cudaStatsguard(PrimitiveOpType::FutureValue, L"implant", 3);
            //GetOutputFields(*dup).m_value       = GetOutputFields(f).m_value;
            fail_if(GetOutputFields(f).m_redirection.empty(), "CSE-redistributing non-function output??");
            GetOutputFields(*dup).m_redirection = GetOutputFields(f).m_redirection;
            NotifyOpsConsumersInputsAvailable(*dup);
        }
    }

#if 0 // note: This will not work after moving CUDA launching to background thread
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
#endif

    // given fields, determine the base fields and slice range
    static inline pair<const VariableFields*, SliceRange> DetermineSliceAndBase(VariableFields& fields)
    {
        pair<const VariableFields*, SliceRange> res;
        auto& pfields    = res.first;
        auto& sliceRange = res.second;
        pfields = &fields;
        while (!pfields->m_redirection.empty())
        {
            let& outFields = GetOutputFields(*pfields->m_redirection.m_function);
            if (&outFields == pfields)
                break;
            if (sliceRange.empty())
                sliceRange = pfields->m_redirection.m_sliceRange; // note: it is guaranteed that there is only one index in the chain
            pfields = &outFields;
        }
        return res;
    }

    // get a hash for a Variable
    // This is based on object identity (physical fields ptr, slice range).
    // Note: An earlier version compared actual GPU address. Was 6 x slower and did not hash better.
    static uintptr_t CacheAndGetCSEVariableHash(/*const*/ VariableFields& fields)
    {
        // we hash on object identity of the physical fields plus slice index
        uintptr_t hash = fields.m_valueAddrForHash;
        if (!hash)
        {
            // determine the Fields of the physical object and the slice
            let res = DetermineSliceAndBase(fields);
            let* pfields    = res.first;
            let& sliceRange = res.second;
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
    }

    // compute a hash value for f
    // This is called for nearly every unbatched PrimitiveFunction, and must therefore be blazingly fast.
    // Note: This is only called once. Done here because this is for CSE, which is not always fully evaluated.
    static size_t ComputeCSEAliasHash(PrimitiveFunction& f)
    {
        size_t hash = 0;
        // we only hash the argument identities; that is, their base pointer
        // We already know that opcode and shapes are identical.
        // If we really allow strides in the future, we may include them in the hash.
        let& inputs = f.m_inputs;
        let numInputs = inputs.size();
        //CudaStatsGuard cudaStatsGuard(PrimitiveOpType::ElementTimes, L"ComputeCSEAliasHash()", 3, numInputs);
        for (size_t k = 0; k < numInputs; k++)
        {
            // TODO: just use object identity (follow redirects through !m_sliceRange) plus index
            let addrForHash = CacheAndGetCSEVariableHash(GetInputFields(inputs[k]));
            hash += (size_t)addrForHash;
            hash += (size_t)(addrForHash >> 8); // also hash the page number, in case allocation is page-aligned
            hash *= hashMultiplier;
        }
        //f.m_autoBatchState.m_aliasHash = hash;
        return hash;
    }

    // class to help deduplicating for CSE
    class CSEDedupSet
    {
        static const size_t numBuckets = 65536;
        class HashBucket
        {
            HashBucket* m_nextBucket = nullptr;
            PrimitiveFunction* m_bucketList = nullptr; // list of items in this bucket, linked via m_bucketList
        public:
            // iterate over buckets themselves (note: not over the bucket *list*). This is used during cleanup.
            HashBucket* ClearAndNext()
            {
                auto* next = m_nextBucket;
                m_nextBucket = nullptr;
                m_bucketList = nullptr; // abandon the list
                return next;
            }
            void CheckClean() const
            {
                fail_if(m_nextBucket || m_bucketList, "CSEDedupSet bucket was not cleaned up upon last use");
            }
            // add an entry to the bucket list
            void PushFront(PrimitiveFunction* f, HashBucket* &cleanupList)
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
        vector<HashBucket> m_buckets = vector<HashBucket>(numBuckets);
        HashBucket* m_firstBucketInUse = nullptr; // for cleanup at the end: all active buckets are registered here
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
            fail_if(m_firstBucketInUse, "CSEDedupSet was not cleaned up upon last use");
            //for (let& bucket : m_buckets) // expensive thorough check
            //    bucket.CheckClean();
        }
        // add to set and return nullptr, unless f is an alias of something that's already in the set; then return that
        PrimitiveFunction* FindDuplicateOrAddToSet(PrimitiveFunction* f)
        {
            let fHash = ComputeCSEAliasHash(*f);
            let fIndex = fHash % numBuckets;
            auto* bucket = &m_buckets[fIndex]; // bucket for f
            // search for an alias in the list. Most of the time this will be a match, but we must confirm.
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
#if 0               // old version that calls MTCacheAndGetValue(), which is now forbidden (owned by memoize thread)
                    bool d1 = DetermineSliceAndBase(fields) != DetermineSliceAndBase(aliasFields);
                    bool d2 = !Memoizer::MTCacheAndGetValue(fields)->IsAliasOf(Memoizer::MTCacheAndGetValue(aliasFields));
                    fail_if(d1 != d2, "cse wrong detection");
#endif
                    // check object identity by what slice it occupies into the main storage arena (the shape is already known to be the same)
                    if (DetermineSliceAndBase(fields) != DetermineSliceAndBase(aliasFields))
                        goto try_next;
                    //if (!DetermineSliceAndBase(fields).second.empty())
                    //    Break;
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
    CSEDedupSet m_cseDedupSet; // TODO: make this a static thread local, to avoid reallocating the bucket list
    VisitorTag m_cseVisitorTag; // helper for CSE pre-check

    // pre-check whether the more expensive hash-table based CSE check is needed
    // If all inputs have either no dups or are all-dups, then there is no need for the more expensive hash-table based check.
    bool IsShortCircuitingBatchedOpDuplicatesNeeded(NonOwningFunctionList ops)
    {
        CudaStatsGuard cudaStatsGuard(PrimitiveOpType::NotEqual, L"CSE pre", 3, ops.size());
        let& f0 = ops.front();
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
        m_cseDedupSet.CheckClean(); // verify that we have cleaned up correctly
        NonOwningFunctionListBuilder filteredOps;
        for (auto iter = ops.begin(); iter != ops.end(); ) // create the batched tensors
        {
            auto& f = *iter;
            ++iter; // advance here, since we will reuse the m_link field
            // f has been taken out of the list, its m_link field is unused. If it is not a dup, then m_link will live in the filteredOps list instead.
#if 1
            auto* duplicate = m_cseDedupSet.FindDuplicateOrAddToSet(&f); // this is now O(1) in most cases
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
            let fHash = ComputeCSEAliasHash(*f);
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
                    if (!MTCacheAndGetValue(fields)->IsAliasOf(MTCacheAndGetValue(fjelds)))
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
        m_cseDedupSet.Reset(); // clean up after ourselves
        return filteredOps;
    }

    // determine the physical source and slice index of an input
    // If !originatingFunction then sliceRange.empty().
    struct LazyPhysicalSliceReturnType { const PrimitiveFunction* originatingFunction; SliceRange sliceRange; };
    static const LazyPhysicalSliceReturnType LazyPhysicalSlice(const VariableFields& fields)
    {
        auto function   = fields.m_redirection.m_function;
        auto sliceRange = SliceRange();
        // case 1: Placeholder or Constant
        if (function)
        {
            // case 2: one-level redirect
            sliceRange = fields.m_redirection.m_sliceRange;
            let& redirectedFields = GetOutputFields(*function);
            auto redirectedFunction = redirectedFields.m_redirection.m_function;
            while (redirectedFunction != function) // not self: not end of chain
            {
                // case 3: multi-step redirections (these occur in composite inlining)
                // TODO: patch up the data structure upon first discovery
                function = redirectedFunction;
                let& redirectedSliceRange = redirectedFields.m_redirection.m_sliceRange;
                if (sliceRange.empty())
                    sliceRange = redirectedSliceRange;
                else
                    fail_if(!redirectedSliceRange.empty(), "LazyPhysicalSlice: hit a see-through slice??"); // multiple slicing not possible
                // TODO: ^^ multiple non-dropping slices could be composable here
                let& redirectedFields = GetOutputFields(*function);
                redirectedFunction = redirectedFields.m_redirection.m_function;
                //if (redirectedFunction == function) // self: end of chain
                //    break;
            }
        }
        fail_if(!function && !sliceRange.empty(), "LazyPhysicalSlice: sliceRange not empty if no redirect??");
        return{ function, sliceRange };
    }

    // subroutine to determine the max Rank() over f's m_inputs
    // This function is currently only used in an error check.
    size_t DetermineMaxElementwiseInputRank(const PrimitiveFunction& f)
    {
        if (IsTimesOp(f.m_op)) // special case for Times and its Affine and Transpose- variants
            return f.m_inputs[1].Shape().Rank();
        size_t maxInputRank = 0;
        for (let& input : f.m_inputs)
            maxInputRank = max(input.Shape().Rank(), maxInputRank);
        //let numArgs = f.m_inputs.size();
        //size_t maxInputRank = f.m_inputs.front().Shape().Rank();
        //for (size_t i = 1; i < numArgs; i++)
        //    maxInputRank = max(f.m_inputs[i].Shape().Rank(), maxInputRank);
        return maxInputRank;
    }

#if 0
    // helper to determine the max Rank() over all inputs and outputs in composite's m_rootFunction
    // Auto-batching needs to know a common batch axis it can use throughout the entire BlockFunction.
    // It is determined as the max rank over all involved inputs and outputs.
    size_t CacheAndGetBasicBlockBatchAxis(const BlockFunction& block)
    {
        auto& composite = static_cast<CompositeFunction&>(*block.Composite());
        if (composite.m_basicBlockInfo.m_freeAxis == SIZE_MAX) // not computed yet (has been reset in Invoke())
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
            composite.m_basicBlockInfo.m_freeAxis = maxRank;
        }
#if 1   // BUGBUG: This uses a hard-coded constant (12) to heuristically detect an uninitialized value. That is not general, so remove this once this stuff works.
        fail_if(composite.m_basicBlockInfo.m_freeAxis == 0 || composite.m_basicBlockInfo.m_freeAxis > 12, "m_basicBlockInfo.m_freeAxis was not prepared??");
#endif
        return composite.m_basicBlockInfo.m_freeAxis;
    }
#endif

    // helper to create a batched input given a list of operations
    // The result will have outputShape[batchAxis] == batchDim, with one exception:
    // If all inputs are identical and that can be represented by broadcasting (which is only possible if BATCHING).
    // In that case, the output will have no batchAxis (which implies it is 1, which means 'broadcast along the batchAxis' to all ops).
    // The stackingMode passed here does not affect the meaning of batchAxis and batchDim; but rather their constraints.
    vector<NDShapeDimension> m_CreateBatchedInputFor_shapeBuffer; // local variable, for which we keep the allocation across invocation
    inline Variable CreateBatchedInputFor(NonOwningFunctionList ops, size_t batchAxis, NDShapeDimension batchDim, StackingMode stackingMode,
                                          size_t i, /*in/out*/bool& anyBatchedInputs)
    {
        let& f0 = ops.front();
        // special case: for matrix-product class operations, the first argument is non-batchable
        // It has already been verified to be identical as part of the batchability condition.
        if (IsTimesOp(f0.m_op) && i == 0)
        {
            return f0.m_inputs.front();
        }

        // create splice args for this argument
        let numBatchItems = ops.size();
        CudaStatsGuard cudaStatsguard(PrimitiveOpType::Slice, L"gather batched args", 3, numBatchItems);

        // first determine special cases that can be optimized
        // If all args are consecutive slices, then use a slice view instead. If all objects are identical, then don't even use a batch.
        let& input0 = f0.m_inputs[i];
        let& input0Fields = GetInputFields(input0);
        let isScalar = input0Fields.m_shape.Rank() == 0;
        let redirectionPair0 = LazyPhysicalSlice(input0Fields);
        let is0Redirected = redirectionPair0.originatingFunction != nullptr;
        bool allSame = true;                                              // will be true if all are the same objects
        bool allConsecutiveSlices = !redirectionPair0.sliceRange.empty(); // will be true if all are consecutive index ops into the same batched result
        size_t prevSliceEndIndex = allConsecutiveSlices ? redirectionPair0.sliceRange.BeginIndex() : SIZE_MAX/*not used*/; // running index
        //if (input0Fields.m_uniqueIdForDebugging == 26932)
        //    Break;
        for (let& f : ops) // create the batched tensors
        {
            if (!allSame && !allConsecutiveSlices)
                break;
            let& input = f.m_inputs[i];
            let& inputFields = GetInputFields(input);
            fail_if(inputFields.m_shape.Rank() > batchAxis + (stackingMode == StackingMode::STACKING), "batch axis too small??");
            let redirectionPair = LazyPhysicalSlice(inputFields);
            // optimization: if all args are the same, then don't batch
            allSame = allSame &&
                      (&inputFields == &input0Fields ||                           // same object (also covers the case of leaves)
                       (is0Redirected && redirectionPair.originatingFunction == redirectionPair0.originatingFunction && redirectionPair.sliceRange == redirectionPair0.sliceRange)); // or same view
            // Note: If we remove is0Redirected, then this comparison will not be correct for leaves.
            // optimization: if all args are consecutive slices, then use a slice view instead
            if (allConsecutiveSlices)
            {
                // optimization: if consecutive slices, then recover the original batched tensor
                // TODO: Can we directly check the data buffer pointer?
                allConsecutiveSlices =
                    redirectionPair0.originatingFunction == redirectionPair.originatingFunction && // function that creates the physical output (NULL if not a slice)
                    prevSliceEndIndex                    == redirectionPair.sliceRange.BeginIndex();
                // TODO: Per Jon's suggestion, we could be a little loose here. For a variable-length
                // scenario, we will loose entries in the middle. We can allow to keep a few around
                // in garbage-in-garbage-out. If, say, there are additional 20% gap values, we just
                // carry them forward, and ignore them when implanting the result.
                // If this input is broadcasting in the stacking axis, then the stacked version cannot be realized as a view.
                if (stackingMode == StackingMode::STACKING &&
                    (batchAxis < inputFields.m_shape.Rank() ? inputFields.m_shape[batchAxis] : 1) != f.m_autoBatchState.m_batchDim)
                {
                    allConsecutiveSlices = false;
                }
                prevSliceEndIndex = redirectionPair.sliceRange.EndIndex();
            }
        }
        fail_if(allSame && allConsecutiveSlices && numBatchItems > 1, "allSame and allConsecutiveSlices are mututally exclusive");
        //if (stackingMode == StackingMode::STACKING)
        //    Break;
        if (allSame && stackingMode == StackingMode::STACKING) // allSame broadcasting is only OK if the input has no extent in stacking direction
        {
            // Consider stacking of b + [X X Y Y Y Z Z], where b, X, Y, Z have matching vector dimension D.
            // In this case, concatenating X, X, Y, Y, Y, Z and Z gives a 7*D vector. b cannot be added to that.
            // (While for BATCHING, we'd get a [D x 7] tensor, to which b can be added without problem.)
            allSame = batchAxis >= input0Fields.m_shape.Rank() || // does not live in the batch axis
                      input0Fields.m_shape[batchAxis] == 1;       // lives there but broadcasts
        }
        cudaStatsguard.Stop();
        // create the batched input
        anyBatchedInputs |= !allSame;
        if (allSame) // optimized case: all ops share the same operand: no need to batch them
        {
            // note: we assume strict broadcasting semantics here (if at least one input is actually batched)
            return f0.m_inputs[i];
        }
        else if (allConsecutiveSlices) // they are consecutive slice views into the same batched tensor: can short-circuit slice+gather as a single slice view
        {
            //if (stackingMode == StackingMode::STACKING)
            //    Break;
            let beginIndex = redirectionPair0.sliceRange.BeginIndex(); // this is the slice's entire consecutive index range covered
            let endIndex   = prevSliceEndIndex;
            let& from = redirectionPair0.originatingFunction; // and this is the function that we take this consecutive slice into
            let& fromOutput = from->m_outputs.front();        // and its output...
            //fail_if(!GetOutputFields(*from).m_value, "value not yet available??");
            let& fromOutputDims = fromOutput.Shape().Dimensions();  // ...and its shape dimensions
            fail_if(fromOutputDims.size() == 0, "slice view into batch has rank 0??");
            let fromOutputRank = fromOutputDims.size() - 1;          // the slice of from is taken along this axis
            Variable batchedInput = Variable(fromOutput, ConstFunctionPtr(), static_pointer_cast<PrimitiveFunction const>(from->shared_from_this())); // our return value
            if (beginIndex == 0 && endIndex == fromOutputDims.back()/*[fromOutputRank]*/) // full range: just take it, no need to slice at all
            {
                // BUGBUG: ^^ This does not cover STACKING of a previously BATCHED result
                batchedInput; // this is what we want here
            }
            else // sub-range: splice it by taking a slice view on the previously spliced batch
            {
                //CudaStatsGuard cudaStatsguard(PrimitiveOpType::Slice, L"re-batch", 3, numBatchItems);
                // create a new PrimitiveFunction Slice()
                auto outputShape = fromOutputDims; // determine output shape
                outputShape.back()/*[fromOutputRank]*/ = endIndex - beginIndex; // narrow it down to the range of the slice we are taking
                batchedInput = CreateAndMemoizeOpAsInput(PrimitiveOpType::Slice, Function::InputsVectorType(nullptr/*1*/, batchedInput),
                                                         Dictionary(
                                                             PrimitiveFunction::AttributeNameAxis,       Axis((int)fromOutputRank),
                                                             PrimitiveFunction::AttributeNameBeginIndex, (int)beginIndex,
                                                             PrimitiveFunction::AttributeNameEndIndex,   (int)endIndex
                                                         ),
                                                         outputShape, batchedInput.GetDataType(), batchedInput.IsSparse(),
                                                         //f0.m_name,
                                                         f0.m_profiler, L"#"/*gatherInputs[0]*/,
                                                         /*isFree=*/true);
                // and that's our input to the batched operation
                // TODO: Can this be done by directly messing with the Variable? Make a deep copy of the var object with begin/end slice?
                //       Would avoid allocating a PrimitiveFunction, and make it easier to be short-circuited directly in backprop.
            }
            // the last axis of batchedInput is the batch axis, which coincides with the last input axis for STACKING, and is a new one for BATCHING.
            fail_if(batchedInput.Shape().Dimensions().back() != batchDim, "batchedInput does not have expected last dimension");

            // We now have a batched input, but its shape is based on the originating output,
            // not considering potential Reshapes expressed as the shape of the input that redirects there.
            // It is also possible that the batchAxis is different.
            // May we also have a stacking/batching switch?
            // To address this:
            //  - determine the expected batched input shape
            //  - if it does not match, then Reshape
            // How to determine the expected input shape:
            //  - notes:
            //     - all m_inputs' shapes (after potential Reshape) are identical except for last a axis when STACKING
            //     - the batchedInput's last dim is batchDim. All other axes match the pre-Reshape originating output shape, which we don't want.
            //  - take first input shape
            //  - if STACKING, strip its last axis and add batchDim in its place (i.e. replace it)
            //  - if BATCHING, append batchDim to its shape
#if 1
            // to avoid mem copies, we first check directly whether the shape is already correct
            let& input0Dims = input0Fields.m_shape.Dimensions();
            let expectedBatchedInputRank = batchAxis + 1;
            fail_if(expectedBatchedInputRank != input0Dims.size() + (stackingMode == StackingMode::BATCHING), "expectedBatchedInputRank not as expected??");
            let& batchedInputDims = batchedInput.Shape().Dimensions(); // shape of the batched input at this point
            bool shapeOK = true;
            shapeOK = shapeOK && (batchedInputDims.size() == expectedBatchedInputRank); // rank must match
            for (size_t k = 0; shapeOK && k < batchAxis; k++)
                shapeOK = shapeOK && (input0Dims[k] == batchedInputDims[k]); // dimensions before batchAxis must match
            // dimensions do not match: there was an additional Reshape that we must replicate
            if (!shapeOK)
            {
                CudaStatsGuard cudaStatsguard(PrimitiveOpType::Reshape, L"intermediate reshape", 3, numBatchItems);
                auto& expectedBatchedInputShapeVec = m_CreateBatchedInputFor_shapeBuffer;
                expectedBatchedInputShapeVec.assign(input0Dims.begin(), input0Dims.begin() + batchAxis); // (skips last axis if STACKING)
                expectedBatchedInputShapeVec.push_back(batchDim);
                // insert a Reshape() op
                let batchedInputTotalSizePre = batchedInput.Shape().TotalSize();
                let batchedInputDataType = batchedInput.GetDataType();
                let batchedInputIsSparse = batchedInput.IsSparse();
                batchedInput = CreateAndMemoizeOpAsInput(PrimitiveOpType::Reshape, Function::InputsVectorType(nullptr/*1*/, move(batchedInput)), Dictionary(),
                                                         expectedBatchedInputShapeVec, batchedInputDataType, batchedInputIsSparse,
                                                         //f0.m_name,
                                                         f0.m_profiler, L"#,"/*gatherInputs[0]*/, /*isFree=*/true);
                let batchedInputTotalSizePost = batchedInput.Shape().TotalSize();
                fail_if(batchedInputTotalSizePre != batchedInputTotalSizePost, "post-slice reshape did not produce correct dimensions??"); // ...Reshape should not even work then, well...
            }
#else
            // old version, which is garbage:  --TODO: delete this block once tested with old MT.cpp model
            // if this op has a a different batchAxis than the re-batched view, we must adjust the axis as to fulfill this function's post-condition
            // Complex corner cases arise when going back and forth betwen STACKING and BATCHING.
            // BUGBUG: (perf) Reshape incurs an unnecessary mem copy in Backprop
            // BUGBUG: This seems never called in the MT case?
            // TODO: Do this inside Gather, based on its output shape.
            if (fromOutputRank != batchAxis)
            {
                CudaStatsGuard cudaStatsguard(PrimitiveOpType::Reshape, L"intermediate reshape", 3, numBatchItems);
                // TODO: Any way we can fold the reshape in using m_redirection? ... no need, this will become part of Gather.
                // start with actual shape we have, and make it the one we want
                let& batchedInputDims = batchedInput.Shape().Dimensions();
                if (batchAxis < fromOutputRank)
                {
                    if (batchAxis + 1 != fromOutputRank)
                        LogicError("stacking axis not adjacent to batch axis??"); // TODO: Can this legally happen?
                    fail_if(batchedInputDims[batchAxis] * batchedInputDims[batchAxis + 1] != batchDim, "batch axis cannot be absorbed into stacking axis??");
                    auto outputShape = batchedInputDims.BackPopped();
                    outputShape.back() = batchDim;
                    // insert a Reshape() op
                    batchedInput = CreateAndMemoizeOpAsInput(PrimitiveOpType::Reshape, Function::InputsVectorType(nullptr/*1*/, move(batchedInput)), outputShape,
                                                             Dictionary(),
                                                             //f0.m_name,
                                                             f0.m_profiler, L"#,"/*gatherInputs[0]*/, /*isFree=*/true);
                }
                else // batchAxis > fromOutputRank
                {
                    auto& outputShapeVec = m_CreateBatchedInputFor_shapeBuffer;
                    outputShapeVec.assign(batchedInputDims.begin(), batchedInputDims.end());
                    //auto outputShapeVec = MakeVector(batchedInputDims); // PERF BUGBUG--can we do better than this? E.g. use a shared vector class member for this?
                    if (outputShapeVec.back() == batchDim)
                        outputShapeVec.insert(outputShapeVec.end() - 1, batchAxis - fromOutputRank, 1);
                    else
                    {
//                        if (batchAxis != fromOutputRank + 1)    // batchAxis > fromOutputRank
//#if 0                       // happens when disabling Barrier??
//                            // BUGBUG: Will crash later. Fix this.
//                            // Observed: [3072*11] -> [1 x 3072 x 11], with expected Placeholder [3072 x FreeDim]
//                            // op = Block
//                            // batchAxis==2 vs. [3027 x FreeDim] is already mismatching
//                            // --> batchAxis is not determined correctly for Block. Reshape?
//                            outputShapeVec.insert(outputShapeVec.begin() + fromOutputRank, batchAxis - (fromOutputRank + 1), 1);
//#else
//                            LogicError("batch axis not adjacent to stacking axis??"); // TODO: Can this legally happen? If yes, then we can just insert ones.
//#endif
                        fail_if(outputShapeVec.back() / batchDim * batchDim != outputShapeVec.back(), "stacking axis cannot be converted to batching axis??");
                        outputShapeVec.back() /= batchDim;
                        if (outputShapeVec.size() < batchAxis)    // batchAxis > fromOutputRank
                            // Example: input is 10 512-dim vectors stacked -> [5120]
                            //          but batchAxis == 2 --> [512 x 1 x 10]
                            outputShapeVec.resize(batchAxis, 1);
                        outputShapeVec.push_back(batchDim);
                    }
                    // insert a Reshape() op
                    batchedInput = CreateAndMemoizeOpAsInput(PrimitiveOpType::Reshape, Function::InputsVectorType(nullptr/*1*/, move(batchedInput)), outputShapeVec,
                                                             Dictionary(),
                                                             //f0.m_name,
                                                             f0.m_profiler, L"#,"/*gatherInputs[0]*/, /*isFree=*/true);
                }
                // and that's now really our input to the batched operation
            }
#endif
            fail_if(batchedInput.Shape().Rank() != batchAxis + 1 || batchedInput.Shape().Dimensions().back() != batchDim, "CreateBatchedInputFor() post-condition not fulfilled??"); // with all this axis mess, verify this function's post-condition
            return batchedInput;
        }
        else // batch inputs are not consecutive: We must actually copy them together.
        {
            //if (stackingMode == StackingMode::STACKING)
            //    Break;
            // create the arguments to the gather operation
            let createArgumentFn = [&](const PrimitiveFunction& f) -> Variable
            {
                let& input = f.m_inputs[i];
                let& inputFields = GetInputFields(input);
                if (stackingMode == StackingMode::STACKING &&
                    (batchAxis < inputFields.m_shape.Rank() ? inputFields.m_shape[batchAxis] : 1) != f.m_autoBatchState.m_batchDim)
                {
                    // if this input broadcasts in the batch-axis dimension, we must manually unroll it.
                    // This implements broadcasting with non-uniform dimensions in the stacking case.
                    auto& broadcastShape = m_CreateBatchedInputFor_shapeBuffer; // reuse a shared object to avoid malloc
                    let& inputDims = inputFields.m_shape.Dimensions();
                    broadcastShape.assign(inputDims.begin(), inputDims.end());
                    broadcastShape.resize(batchAxis, 1);
                    broadcastShape.push_back(f.m_autoBatchState.m_batchDim); // this is the shape we want
                    // insert a ReduceElements op, which in fact ignores its axes and therefore can also be used to broadcast
                    return CreateAndMemoizeOpAsInput(PrimitiveOpType::ReduceElements, Function::InputsVectorType(nullptr/*1*/, input),
                                                     Dictionary(PrimitiveFunction::AttributeNameReductionOpName, PrimitiveFunction::InternalSumReductionOpName),
                                                     broadcastShape, input.GetDataType(), input.IsSparse(),
                                                     //f0.m_name,
                                                     f0.m_profiler, L"#,"/*gatherInputs[0]*/, /*isFree=*/false);
                    // Note that at this point, the inputs to the Gather operation will have inconsistent
                    // rank; those we expanded here have a batch axis, while the unexpanded may not.
                    // Gather can handle that.
                }
                else
                    return input;
                // note: Variable is just three shared_ptrs, one being NULL; so this is cheap
                // note: input is a regular Variable with regular ownwership rules (it does not come from inside here)
            };
            Function::InputsVectorType gatherInputs(Transform(ops, createArgumentFn));
            CudaStatsGuard cudaStatsguard(PrimitiveOpType::Splice, L"batch", 3, numBatchItems);
            let& input0Shape    = gatherInputs.front().Shape();
            let  input0IsSparse = gatherInputs.front().IsSparse();
            let  input0DataType = gatherInputs.front().GetDataType();
#if 0       // TODO: remove this. I used this earlier. Not sure why.
            let& input0Shape1 = MTCacheAndGetValue(gatherInputs[0])->Shape();
            fail_if(input0Shape != input0Shape1, "shape not set?");
#endif
            // create a new PrimitiveFunction Splice()
            vector<NDShapeDimension> batchedInputShape; // determine output shape   --TODO: use a vector<NDShapeDimension> in this class
            batchedInputShape.reserve(batchAxis + 1);
            batchedInputShape = vector<NDShapeDimension>(input0Shape.Dimensions());
            //batchedInputShape = gatherInputs[0]->Shape().Dimensions(); // TODO: no need to force-realize it here; should be done by MemoizeInArena()
            batchedInputShape.resize(batchAxis, 1); // pad to batchAxis
            batchedInputShape.push_back(batchDim);  // and add the batch axis
            //fail_if(batchedInputShape[batchAxis] != batchDim, "CreateBatchedInputFor() post=condition not fulfilled??"); // (no need to check; post-condition is obviously fulfilled by construction)
            // BUGBUG: vv The move(gatherInputs) is ineffective because ultimately, the copy (not move) constructor of PrimtiveFunction ends up being called.
            //if (f0.m_op == PrimitiveOpType::ElementTimes && f0.m_attributes.Size() > 0 && stackingMode == StackingMode::STACKING) // we were batched for batching
            //    Break;
            return CreateAndMemoizeOpAsInput(PrimitiveOpType::Splice, move(gatherInputs),
                                             Dictionary(PrimitiveFunction::AttributeNameAxis, Axis((int)batchAxis)),
                                             batchedInputShape, input0DataType, input0IsSparse,
                                             //f0.m_name,
                                             f0.m_profiler, L"#"/*gatherInputs[0]*/,
                                             /*isFree=*/false, /*logSpliceAsGather=*/true);
        }
    }

    // batch-execute a set of ops that are known to be batchable
    // For every batched operation, this generates a new PrimitiveFunction object for the op itself, and one
    // for a splice operation for each batched inputs.
    // I.e. this is not a full graph transform, but rather a graph augmentation, so that during backprop,
    // we can recover the batched operations, while the original graph does not get modified.
    // Any batched operation will generate its result in a dense tensor with a batch dimension.
    // The consumers of the original ops will get a back-reference in the m_redirection field.
    // If such a result is ever accessed individually, it will lead to a lazy NDArrayView::SliceView() call
    // (but no Splice Function object is used for this).
    vector<Variable> m_ExecuteBatchedOpAndUpdateSchedule_batchedInputsBuffer; // local variable, for which we keep the allocation across invocation
    void ExecuteBatchedOpAndUpdateSchedule(NonOwningFunctionList ops) // (note: NonOwningFunctionListBuilder is so small that it is best copied)
    {
        // get a representative op
        let& f0 = ops.front();
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
#ifndef NO_CSE
        if (!isFree && op != PrimitiveOpType::BatchNormalization && ops.size() > 1 && IsShortCircuitingBatchedOpDuplicatesNeeded(ops))
            ops = ShortCircuitBatchedOpDuplicatesAndUpdateSchedule(ops);
        else
#endif
            for (auto iter = ops.begin(); iter != ops.end(); ++iter) // create the batched tensors
                iter->m_autoBatchState.m_aliasList = nullptr;
        // TODO: ^^ if the CSE can directly redirect, then this loop ^^ is no longer needed

        // perform the op
        let numBatchItems = ops.size();
        if (!isFree)
            m_stats.numBatchedLaunches++;
        let numArgs = f0.m_inputs.size();
        //if (GetOutputFields(f0).m_uniqueIdForDebugging == 241565 || GetOutputFields(f0).m_uniqueIdForDebugging == 241579 || GetOutputFields(f0).m_uniqueIdForDebugging == 241593 || GetOutputFields(f0).m_uniqueIdForDebugging == 241607)
        //    Break;
        //if (f0.m_op == PrimitiveOpType::RandomDistribution)
        //    Break;

        // special case: under certain circumstances, we don't actually execute an op batched

        // is unbatched actually more expensive than batched?
        // We count CUDA launches, assuming all args are batched and require a Gather launch.
        // If not all args require that, we err on the side of not batching. BatchNorm must always be batched.
        let isSparseSplice = op == PrimitiveOpType::Splice && f0.m_outputs.front().IsSparse();
        let numCUDALaunchesInThisOp = 1; // TODO: for Block invocations, use the #nodes; or pick a number, such as 4
        let worthIt = true;// numBatchItems * numCUDALaunchesInThisOp/*unbatched launches*/ > (numArgs + 1)/*batched launches*/ || op == PrimitiveOpType::BatchNormalization;
        // BUGBUG: setting worthIt := true elicits a problem with output batch axis
        //if (numBatchItems > 1 && !worthIt) // TODO: remove this message once I see it happen
        //    fprintf(stderr, "%S not worth it: %d vs. %d\n", PrimitiveOpTypeName(f0.m_op).c_str(), (int)numBatchItems, (int)numArgs + 1);
#ifdef NO_BATCHED_FORWARD
        auto doNaively = true;
#else
        let doNaively =
            isFree             ||
            isSparseSplice     || // we only have one axis to work with for splicing sparse data
            numBatchItems == 1 ||
            !worthIt;
#endif
        //fprintf(stderr, "%d %sexecuting %d instances of %S -> %S; %d batchable ops pending\n",
        //        isFree ? -1 : (int)m_stats.numBatchedLaunches,
        //        doNaively ? "" : "batch-",
        //        (int)numBatchItems, f0.OpName().c_str(), f0.m_outputs.front().Shape().AsString().c_str(),
        //        (int)m_schedule.numBatchableOpsPending());
        if (doNaively && f0.m_op != PrimitiveOpType::BatchNormalization) // BN can't be non-batched
        {
            // for correctness testing of underlying mechanism, compute them without actual batching
            for (auto& f : ops)
            {
                // execute it
                if (f.m_op == PrimitiveOpType::Block)
                {
                    //if (f.m_uniqueIdForDebugging == 52152)
                    //    Break;
                    let fInlinedPtr = InlineAndMemoizeBatchedBasicBlock(static_cast<const BlockFunction&>(f), f.m_inputs, /*batchAxis=*/SIZE_MAX, /*batchSize=*/1/*dummy*/);
                    // implant the result
                    FinalizeBatchedOpAndUpdateSchedule(f, fInlinedPtr);
                }
                else
                {
                    MemoizeInArena(f, isFree);
                    // and notify consumers (which may schedule them)
                    NotifyOpsConsumersInputsAvailable(f);
                }
                // distribute value to all aliases
                UpdateDuplicatesAndUpdateSchedule(f);
            }
            return; // and done
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
        auto& batchedInputs = m_ExecuteBatchedOpAndUpdateSchedule_batchedInputsBuffer; // (this is semantically a local var; we use a class var to avoid reallocation across invocations)
        batchedInputs.resize(numArgs);
        let& unbatchedOutputShape = f0.m_outputs.front().Shape();
        let i0 = (size_t)isTimes;    // index of first batchable argument (1 for matrix-product class; 0 otherwise)
        // TODO: Make this work logically, then speed this up!
        // determine stacking vs. batching
        //  - we must batch if the operation does not allow stacking. AreBatchable takes this into account.
        //  - if the op allows stacking, we still want to back off to batching if the operation allows it. It will be more efficient.
        //  - we stack only if it is allowed and required
        //if (f0.m_op == PrimitiveOpType::RandomDistribution)
        //    Break;
        //if (f0.m_uniqueIdForDebugging == 4342)
        //    Break;
        auto stackingMode         = f0.m_autoBatchState.m_stacking;  // we batch or stack?
        auto commonInputBatchAxis = f0.m_autoBatchState.m_batchAxis; // we batch along this axis
        NDShapeDimension batchSize;
        //if (f0.m_op == PrimitiveOpType::ElementTimes && f0.m_attributes.Size() > 0 && stackingMode == StackingMode::STACKING) // we were batched for batching
        //    Break;
        if (stackingMode == StackingMode::BATCHING) // we were batched for batching
            batchSize = (NDShapeDimension)numBatchItems;
        else // we were batched for stacking. Use batching if possible.
        {
            // determine the output batch size
            bool allBatchDimsTheSame = true;
            auto batchDim0 = f0.m_autoBatchState.m_batchDim;
            batchSize = 0;
            for (let& f : ops) // create the batched tensors
            {
                let batchDim = f.m_autoBatchState.m_batchDim;
                batchSize += batchDim;
                allBatchDimsTheSame = allBatchDimsTheSame && batchDim == batchDim0;
            }
            if (allBatchDimsTheSame && stackingMode == StackingMode::STACKING_BUT_MAY_BATCH) // we don't need to stack
            {
                stackingMode = StackingMode::BATCHING;
                commonInputBatchAxis++;
                batchSize = (NDShapeDimension)numBatchItems;
            }
            else
                stackingMode = StackingMode::STACKING;
            //if (stackingMode == StackingMode::STACKING)
            //    fprintf(stderr, "STACKING op %S\n", PrimitiveOpTypeName(f0.m_op).c_str());
        }
        fail_if(stackingMode == StackingMode::STACKING_BUT_MAY_BATCH, "StackingMode::STACKING_BUT_MAY_BATCH should have been decided by now??");
        auto outputBatchAxis = commonInputBatchAxis;
        if (isTimes) // Times family: may shift the batch axis
            outputBatchAxis += f0.m_outputs.front().Shape().Rank() - f0.m_inputs[1].Shape().Rank();
        else if (f0.m_op == PrimitiveOpType::OneHot) // will shift the batch axis
            outputBatchAxis += f0.m_outputs.front().Shape().Rank() - f0.m_inputs[0].Shape().Rank(); // should be by 1, but this expression is more universal
        // TODO: ^^ get to a point where this is universally true
        //       I think for that we need to determine the max input rank.
        if (!isTimes &&
            op != PrimitiveOpType::Splice && op != PrimitiveOpType::Reshape && op != PrimitiveOpType::OneHot && 
            op != PrimitiveOpType::RandomDistribution && op != PrimitiveOpType::Block &&
            f0.m_outputs.front().Shape().Rank() > DetermineMaxElementwiseInputRank(f0))
            LogicError("elementwise op that increases rank??");
        //if (op == PrimitiveOpType::Block && stackingMode == StackingMode::BATCHING)
        //    Break;
        //if (op == PrimitiveOpType::Block && f0.m_inputs.empty())
        //    Break;

        // create all batchedInputs[] by splicing along the batch axis
        // Special optimizations are taken if all elements are identical.
        bool anyBatchedInputs = false; // stays false if for all arguments, all respective batch-item arguments are identical
        for (size_t i = 0; i < numArgs; i++) // if isTimes then skip the first arg
        {
            batchedInputs[i] = CreateBatchedInputFor(ops, commonInputBatchAxis, batchSize, stackingMode,
                                                       i, /*in/out*/anyBatchedInputs);
        }
        // anyBatchedInputs will still be false if all had identical inputs across all batch items.
        // BUGBUG: ^^ We batch arguments of basic blocks as well, which is not necessary snce they are never replaced.

        // special case: BatchNormalization
        if (op == PrimitiveOpType::BatchNormalization)
        {
            // BatchNorm requires three additional parameters for the current mean and invStdDev, and the zero-mean/unit-variance intermediate. These must be kept for backprop.
            // This is sort of a hack for now. It is not, however, an efficiency problem since there are relatively few batched BatchNorm nodes in the graph.
            // Note that we cannot just move creation of these outside because one depends on the actual size.
            let dataType = batchedInputs[0].GetDataType();
            let createParameter = [&](const NDShape& shape) -> Variable // helper to create a Parameter as if it had been initialized by RPrepareForwardGraphAndSchedule()
            {
                // BUGBUG: We construct a Parameter object as an InternalVariable. Better construct a true Parameter, in case Parameter ever becomes a distinct type with extra members.
                let p = InternalVariable(shape, VariableKind::Parameter, dataType, /*value=*/nullptr, /*needsGradient=*/false, /*dynamicAxes=*/{}, /*name=*/wstring(), /*uid=*/wstring());
                return Variable(p, nullptr, nullptr);
            };
            let& statShape = batchedInputs[1].Shape(); // note: This is guaranteed to have no batch axis, since they are identical across all instances in this batched op
            batchedInputs.push_back(createParameter(               statShape  ));
            batchedInputs.push_back(createParameter(               statShape  ));
            batchedInputs.push_back(createParameter(batchedInputs[0].Shape()));
            anyBatchedInputs = true; // Note: Even if all operands are the same, we may still have a sequence axis to normalize over
        }

        // execute the operation and implant the results
        // Batched inputs have been prepared in batchedInputs[].
        // If all inputs are identical then degrade to computing it only once (this is the easy case that we don't kick off the CSE machinery for).

        if (!anyBatchedInputs)
            for (size_t i = 0; i < numArgs; i++)
                fail_if(&GetInputFields(batchedInputs[i]) != &GetInputFields(f0.m_inputs[i]), "all batch args the same, but not??");

        //if (f0.m_op == PrimitiveOpType::Block)
        //{
        //    for (let& f : ops)
        //    {
        //        let name = static_cast<const BlockFunction&>(f).Composite()->Name();
        //        if (name.size() > 6 && name.substr(0, 6) == L"dense.")
        //            Break;
        //        if (name == L"dense.normWeight3")
        //            Break;
        //    }
        //}
        // >>> This is the actual batched op that we create and execute here. <<<
        // A new PrimitiveFunction is created for the batched op, so that we can backprop through it.
        // If f0 is a block, then actually an entire subgraph clone will be created here.
        // PERF BUGBUG: If all args are identical, we can degrade to a single op. Then we should not need to create a new PrimitiveFunction.
        //              Note: This is not covered by CSE, since CSE is only used for complex cases.
        //if (f0.m_uniqueIdForDebugging == 23286)
        //    Break;
        // TODO: If !anyBatchedInputs then why even clone?
        auto batchedOp = CreateAndMemoizeBatchedOp(f0, Function::InputsVectorType(move(batchedInputs)), /*compositeBatchDim=*/ABSENT_FREE_DIMENSION/*dummy*/, anyBatchedInputs ? outputBatchAxis : SIZE_MAX, batchSize, L"*"/*f0*/, /*isFree=*/false);
        //if (batchedOp->m_uniqueIdForDebugging == 10634)
        //    Break;

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
            let unbatchedOutputRank = unbatchedOutputShape.Rank() - (stackingMode == StackingMode::STACKING); // (stacking axis is included in unbatched shape, while batching axis is not)
            if (outputBatchAxis != unbatchedOutputRank)
            {
                CudaStatsGuard cudaStatsguard(PrimitiveOpType::Reshape, L"interm. reshape", 3, numBatchItems);
                // TODO: An explicit Reshape should be avoiable now, since we could let this be implicitly handled by a redirect.
                fail_if(!isElementWise && !isBasicBlock, "output shape should only have additional singleton axes for elementwise operations or basic-block invocations");
                // insert a Reshape() op to remove the axes
                let batchedOutputShape = unbatchedOutputShape.SubShape(0, unbatchedOutputRank).AppendAxis(unbatchedOutputRank, batchSize); // desired batched output shape without the singleton axes
                fail_if(batchedOutputShape.TotalSize(/*check=*/false) != batchedOp->m_outputs.front().Shape().TotalSize(/*check=*/false), "output shape has unexpected axes that should be singletons but aren't");

                Variable arg = Variable(batchedOp->m_outputs.front(), ConstFunctionPtr(), batchedOp); // TODO: use a proper constructor here
                //arg./*m_outputComposite*/m_acyclicOutputPrimitiveReference = batchedOp;

                // Reshape() here does not need the properties at this level anymore; output shape is sufficient
                let argDataType = arg.GetDataType();
                let argIsSparse = arg.IsSparse();
                let reshapeOp = CreateAndMemoizeOp(PrimitiveOpType::Reshape,
                                                   Function::InputsVectorType(nullptr/*1*/, move(arg)), Dictionary(),
                                                   batchedOutputShape, argDataType, argIsSparse/*, f0.m_name*/,
                                                   f0.m_profiler, L"*,"/*arg*/, /*isFree=*/true);

                batchedOp = reshapeOp; // this is the result that we redistribute from to the individual consumers
            }
        }

        // implant all results in the original unbatched operations (all as lazy/virtual references through m_redirection)
        // TODO: review this w.r.t. multi-output functions
        size_t sliceBegin = 0;
        SliceRange sliceRange;
        for (auto& f : ops)
        {
            // implant the result
            if (anyBatchedInputs) // if no batched inputs then we only produced a single shared reuslt that must just be copied
            {
                if (stackingMode == StackingMode::BATCHING) // if batching then create an index
                {
                    sliceRange = SliceRange(sliceBegin);
                    sliceBegin++;
                }
                else // if stacking then create a slice
                {
                    sliceRange = SliceRange(sliceBegin, sliceBegin + f.m_autoBatchState.m_batchDim);
                    sliceBegin = sliceRange.EndIndex();
                }
            }
            FinalizeBatchedOpAndUpdateSchedule(f, batchedOp, sliceRange);
            // and implant it to all aliases as well
            UpdateDuplicatesAndUpdateSchedule(f);
        }
        // To keep the batchedOp ref count alive, FinalizeBatchedOpAndUpdateSchedule() saves the shared_ptr in all m_redirection.m_functionHolder.

        // release the ref counts on the batched inputs; but keep the vector's memory allocated
        // TODO: CreateAndMemoizeBatchedOp() already moves out the ref counts. Do we still need this? --TODO: double-check that they have been moved out
        batchedInputs.clear();
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
    NDArrayViewPtr BatchedForward(const InternalVariable& v)
    {
#if 1   // show compilation options
        static bool compilationOptionsShown = false;
        if (!compilationOptionsShown)
        {
            compilationOptionsShown = true;
            fprintf(stderr, "\n\n%s\n\n", compilationOptionsAsString), fflush(stderr);
        }
#endif
        auto& fields = GetInputFields(v);
        // if value already there then just return it
        if (fields.m_value)
            return fields.m_value;
#ifdef LOG_DETAILS
        Function::PreorderTraverseFunctions(v.OutputOwner(), [&](const FunctionPtr& f) { Memoizer::LogFunction(dynamic_cast<PrimitiveFunction&>(*f), L"r "); });
#endif
        ResetCudaStats(/*updateCounter=*/true);
        CudaStatsGuard cudaStatsGuardForward(PrimitiveOpType::ForwardBackward/*misusing this for actual op*/, L"batched FORWARD", 3);
        // phase 1 (all done in the same function call):
        //  - create our graph overlay, esp. short-circuit see-through ops
        //  - mark all nodes w.r.t. how many inputs they are waiting for before being computable
        //  - prepare and schedule first set
        CudaStatsGuard cudaStatsGuardInit(PrimitiveOpType::LabelsToGraph/*misusing this for graph initializing*/, L"forward init", 3);
        m_visitorTag.Begin();
        RPrepareForwardGraphAndSchedule(v, 0);
        cudaStatsGuardInit.Stop();
        // phase 2:
        //  - compute the entire graph
        m_memoizer.Begin(); // sync the worker thread at this point in time, for error handling
        while (!m_schedule.empty()) // main computation loop over all operations
        {
            // select the "best" amongst the scheduled op batches
            //  - scheduled = ready for computation, all inputs available
            //  - "best" = heuristically chosen to be best executed in batch
            auto opBatch = m_schedule.pop_best();
            // log (if barrier crossed)
            let& f0 = opBatch.front();
            if (m_memoizer.ShouldProfile(f0)) // profiling diagnostics
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
            ExecuteBatchedOpAndUpdateSchedule(opBatch);
        }
        cudaStatsGuardForward.Stop(); // this measures the auto-batching process, which submits actual computation to a bg thread
        CudaStatsGuard cudaStatsGuardCalc(PrimitiveOpType::Assign/*misusing this for actual op*/, L"forward thread hangover", 3);

        // let CUDA submission thread complete its submitting work (but the CUDA ops themselves do not need to be complete)
        // Until this is completed, the m_value fields are not filled in yet.
        m_memoizer.End();
        // now all m_values that were submitted for are filled in
        // the final m_value may be a reshape, which the bg thread would not have filled in yet, as those are done lazily. Do it now if needed.
        Memoizer::MTCacheAndGetValue(fields); // (note: this is a view op that costs nearly nothing) --TODO: It may make a copy in the future, if the result is not memory-contiguous.

        cudaStatsGuardCalc.Stop(); // this measures the bg thread; specifically, how much longer it needs after the fg thread has submitted the last item

        fail_if(!fields.m_value, "BatchedForward process did not produce a value??");
#if 0   // this syncs the GPU before backprop, wasting CPU cycles
        CudaStatsGuard cudaStatsGuardGPU(PrimitiveOpType::Exp/*misusing this for actual op*/, L"batched forward gpu", 3);
        if (fields.m_shape.TotalSize() == 1) // BUGBUG: we use AsScalar() to force a GPU sync, but that won't work for logging intermediate results
            fields.m_value->AsScalar<float>();
        cudaStatsGuardGPU.Stop();
#endif
        // log stats
        // TODO: clean this all up, also the SyncDevice() function which no longer does what its name says.
        ShowCudaStats();
#ifdef LOG_STATS
        fprintf(stderr, "BatchedForward:  %d fw calcs + %d gathers + %d views, %d CSEs, in nominally %d+%ds ops (%d inlined) on %d known values\n",
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
        if ((f.m_op == PrimitiveOpType::Times || f.m_op == PrimitiveOpType::Affine) && index == 0 && f.m_inputs[1].IsSparse())
            return StorageFormat::SparseBlockCol;  // BUGBUG? how about TransposeTimes? Not sparse?
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
            // TODO: allocate parameter gradients as separate objects
            gradFields.m_gradient = m_memoizer.Arena().NewNDArrayView(gradFields.m_shape, gradFields.m_dataType, format, gradFields.m_value->Device());
            // BUGBUG: gradFields.m_value may not exist, once we optimize memory. Device should be passed in (borrowing from the incoming gradient from top)
            beta = 0.0; // has not been initialized (random section in arena)
            // if there is no redirect, then writing to the physical one is the same as writing to the cached one. Then we are done.
            if (inputFields.m_gradient)
                return{ inputFields.m_gradient, beta };
        }

        // we now have a physical one, but no cached view
        // Reminder: cached view = physical value |> Barrier >> Slice >> Reshape
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
    //
    // Pass 'userOwnsGradients' for the leaves. For those, the gradient NDArrayView will not be cleared/released.
    // Instead, the existing NDArrayView will be kept and its value reset to 0 (unless beta = 1).
    void RPrepareBackwardGraph(VariableFields& gradFields, bool userOwnsGradients = false, double beta = 0)
    {
        fail_if(gradFields.m_varKind == VariableKind::Input || gradFields.m_varKind == VariableKind::Placeholder, "unexpectedly encountered an Input or a Placeholder??"); // (should have been caught in forward)
        fail_if(!gradFields.m_needsGradient, "unexpectedly encountered a node with m_needsGradient=false??");
        //fail_if(m_visitorTag.Visited(gradFields.m_visitedTag), "RPrepareBackwardGraph called multiple times on the same node??"); // naw, can't test it this way

        // initialize the upwards graph links. These form the graph structure.
        gradFields.m_consumers.clear();
        // this arg will receive a gradient; reset it (later, we *accumulate* into it since nodes can receive gradients from multiple consumers)
        // note: must reset in case the user calls Backward() multiple times with different roots that share subgraphs.
        if (!userOwnsGradients)            // user-owned gradients are those passed in by the user
            gradFields.m_gradient.reset(); // not user owned: release any NDArrayView that may still be hanging here
        else if (gradFields.m_gradient)    // for user-owned leaves, if user passed in an existing NDArrayView, clear its value to zero
        {
            //gradFields.m_gradient->LogToFile(gradFields.m_name.get());
            if (beta == 0) // reset the values, unless beta == 1
            {
                gradFields.m_gradient->SetValue(0.0);
                m_stats.numBackpropSetZeroes++;
            }
            else if (beta != 1) // scale it. TODO: Implement it if ever needed. Simple but needs a test case.
                LogicError("RPrepareBackwardGraph: Beta != 1 and != 0 is currently not supported.");
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

        fail_if(&GetOutputFields(f) != &gradFields, "RPrepareBackwardGraph called on a redirection??");
        fail_if(!gradFields.m_value, "variable has no value yet??");

        fail_if(m_visitorTag.Visited(f.m_autoBatchState.m_visitedTag), "RPrepareBackwardGraph: registering the same function twice??"); // should have been caught by gradFields' visitedTag

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
                RPrepareBackwardGraph(inputGradFields);
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
    VariableFields& ResetInputGradient(const InternalVariable& input)
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
    // Only used for backprop so far.
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
        auto out = m_memoizer.Arena().NewNDArrayView(gatherBatchResultDims, inputValue0.GetDataType(), inputValue0.GetStorageFormat(), inputValue0.Device());
        m_stats.numBackpropGathers++;

        CudaStats* cudaStatsPtr = nullptr;
        let& outDevice = out->Device();
        if (ShouldLogMemoizeStats())
            cudaStatsPtr = BeginCudaStats(PrimitiveOpType::Gather, nullptr, /*category=*/out->IsSparse() ? 1 : 0, out->Shape().TotalSize(/*check=*/false), outDevice);

        auto res = NDArrayView::GatherBatch(inputValues, (int)axis, move(out));

        EndCudaStats(cudaStatsPtr, outDevice);

        return res;
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
    // This is the standard path for all ops which have no further batching beyond the forward pass.
    // This wraps the PrimitiveFunction's BackpropTo(), interfacing from vectors of Variable to vectors of NDArrayViewPtr.
    // Note that each input that is redirected should redirect the gradient into a slice in its lazy source.
    // If the target only has one consumer, pass viewAllowed. This will allow views for trivial gradients such as Plus.
    void BackpropToUnbatched(const AutoBatchConsumer& fi, bool viewAllowed)
    {
        let& f = *fi.first;
        let index = fi.second;
#ifdef LOG_DETAILS
        Memoizer::LogFunction(f, L"bb ", index);
#endif
        // function's forward output and received gradient from top live here
        let& outputFields = GetOutputFields(f); // result of f lives here; hence also the gradient we back-propagate
        fail_if(!outputFields.m_value,    "unexpectedly ran into a function that has no m_value yet??");
        fail_if(!outputFields.m_gradient, "unexpectedly ran into a function that has no m_gradient yet??");

        // the input's gradient will live on the same device as the incoming gradient from top
        let& inputDevice = outputFields.m_gradient->Device();

        // get the TensorViews for the forward inputs to this function
        // TODO: Can we know which ones are actually needed? We could save some time. This info would be shared with a memory-sharing mechanism.
        let& inputs = f.m_inputs;
        let numInputs = inputs.size();
        auto& inputValues = BorrowBuffer(m_inputValuesBufferRaw, numInputs);
        for (size_t i = 0; i < numInputs; i++)
            inputValues[i] = GetInputFields(inputs[i]).m_value.get();

        // get the gradient view for the input whose gradient we desire
        // If it was newly created, then gradient.beta will be 0
        let& input = inputs[index];
        fail_if(!GetInputFields(input).m_needsGradient, "function unexpectedly does not need a gradient");

        // optimize for trivial gradients (e.g. Plus)
        let op = f.m_op;
        if (viewAllowed && IsOpGradientViewable(op, index, outputFields, input))
        {
            CudaStats* cudaStatsPtr = nullptr;
            if (ShouldLogMemoizeStats())
            {
                bool logSpliceAsGather = false; // TODO
                let logAsOp = (f.m_op == PrimitiveOpType::Splice && logSpliceAsGather) ? PrimitiveOpType::Gather : f.m_op; // gather ops are logged as op Gather (CNTK V2 Gather is not used by Dynamite)
                cudaStatsPtr = BeginCudaStats(logAsOp, nullptr, /*category=viewOp*/2, input.Shape().TotalSize(/*check=*/false), inputDevice);
            }

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
                // Sanity check: If input is a redirected slice, then necessarily the underlying object (=inputFields.m_redirection.m_function)
                // must have multiple consumers. Otherwise it would not be part of a batched operation, which is the only way
                // of creating redirected slices.
                // An exception is BatchNorm, which uses the batching mechanism even for e.g. a single-sequence batch.
                fail_if(!inputFields.m_redirection.m_sliceRange.empty() && inputFields.m_redirection.m_function->m_op != PrimitiveOpType::BatchNormalization, "redirected slice with single consumer shouldn't be a redirect in the first place");
                auto grad = inputFields.m_gradient;
                ReplaceWithReshapedViewIfNeeded(grad, redirectedInputFields.m_shape);
                redirectedInputFields.m_gradient = move(grad); // and the redirected location. If it is the same, then this will do nothing.
            }
            // Nota bene: This is a little scary. If this condition is not correct: When more gradients get accumulated
            // into this input, then those will get into the output as well, which would be incorrect. There is no good
            // way to ensure this.

            EndCudaStats(cudaStatsPtr, inputDevice);

            m_stats.numBatchedBackpropToViews++;
            return;
        }
        let gradient = CacheAndGetGradientView(input, DetermineGradientStorageType(f, index));
        // BUGBUG: This ^^ should use inputDevice. Currently it uses input[index]'s device. But that input may not exist, once we optimize. The one that always exists is outputGradientValue.

        // compute gradients for the desired input
        // backprop into the input
        // BUGBUG: (perf) In case of Reshape we currently make a copy, which is not needed --> see-through the op, and backprop through a reshaped view into Reshape's argument gradient?
        CudaStats* cudaStatsPtr = nullptr;
        if (ShouldLogMemoizeStats())
        {
            bool logSpliceAsGather = false; // TODO
            let isSparse = DetermineGradientStorageType(f, index) != StorageFormat::Dense;
            let logAsOp = (f.m_op == PrimitiveOpType::Splice && logSpliceAsGather) ? PrimitiveOpType::Gather : f.m_op; // gather ops are logged as op Gather (CNTK V2 Gather is not used by Dynamite)
            cudaStatsPtr = BeginCudaStats(logAsOp, nullptr, /*category=*/isSparse ? 1 : 0, input.Shape().TotalSize(/*check=*/false), inputDevice);
        }

        PrimitiveFunction::BackpropTo(outputFields.m_gradient.get()/*incoming*/, index, op, f.m_attributes, outputFields.m_value.get(), inputValues, gradient.view/*target*/, gradient.beta, f);
        m_stats.numBatchedBackpropToCalls++;

        EndCudaStats(cudaStatsPtr, inputDevice);
#if 0   // debug the actual values
        Memoizer::LogFunction(f, f.m_profiler);
        gradient.view->LogToFile((input.Name() == L"" ? f.OpName() : input.Name()) + L"_" + to_wstring(index), stderr);
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
        Memoizer::LogFunction(f, L"bb# ", SIZE_MAX);
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
        CudaStats* cudaStatsPtr = nullptr;
        if (ShouldLogMemoizeStats())
        {
            bool logSpliceAsGather = true; // TODO
            let logAsOp = (f.m_op == PrimitiveOpType::Splice && logSpliceAsGather) ? PrimitiveOpType::ScatterPacked : f.m_op; // gather ops are logged as op Gather (CNTK V2 Gather is not used by Dynamite)
            cudaStatsPtr = BeginCudaStats(logAsOp, nullptr, /*category=dense*/0, outputFields.m_shape.TotalSize(/*check=*/false), outputFields.m_gradient->Device());
        }

        let& outputGradient = outputFields.m_gradient; // this is the incoming batch of gradients
        NDArrayView::ScatterBatch(outputGradient, inputGradients, (size_t)f.m_attributes[PrimitiveFunction::AttributeNameAxis].Value<Axis>().StaticAxisIndex(), beta);

        EndCudaStats(cudaStatsPtr, outputFields.m_gradient->Device());
#endif
        m_stats.numBackpropScatters++;
    }

    // helper to determine the outputRank of the transform
    static size_t DetermineTimesOutputRank(const PrimitiveFunction& f)
    {
        fail_if(!IsTimesOp(f.m_op), "DetermineTimesOutputRank unexpectedly called for non-Times operation");
        let&   resShape = f.m_outputs.front().Shape().Dimensions();
        let&  leftShape = f.m_inputs[0].Shape().Dimensions();
        let& rightShape = f.m_inputs[1].Shape().Dimensions();
        let    resRank = resShape.size(); // output rank must match
        let   leftRank = leftShape.size();
        let  rightRank = rightShape.size();
        // matrix-product dimensions for y = W * x:
        //               [ output dimensions , reduction dimensions , map dimensions ]
        // left =    W : [ output dimensions x reduction dimensions                  ]
        // right =   x : [                     reduction dimensions x map dimensions ]
        // res =     y : [ output dimensions                        x map dimensions ]
        return (leftRank + rightRank - resRank) / 2; // #dimenions over which matrix product reduces
        // Note: since we only deal with ranks, and not actual dimensions, this is correct for TransposeTimes aswell.
    };

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
        Memoizer::LogFunction(f0, L"bb* ", 0);
#endif
        let& input0 = f0.m_inputs[0]; // all consumers share this weight, so it's OK to just get it from f0
        let outputRank0 = DetermineTimesOutputRank(f0);
        let FlattenMapAxes = [&](const NDArrayViewPtr& arg) -> NDArrayViewPtr // helper to flatten
        {
            let& shape = arg->Shape();
            let rank = shape.Rank();
            fail_if(rank <= outputRank0, "FlattenMapAxes: input has too few axes??");
            if (rank <= outputRank0 + 1) // if it has no max axes to flatten, return
                return arg;
            // it has multiple axes
            auto flattenedShape = shape.SubShape(0, outputRank0 + 1);
            for (size_t k = outputRank0 + 1; k < rank; k++)
                flattenedShape[outputRank0] *= shape[k];
            return arg->AsShape(flattenedShape);
        };
        size_t batchDim = 0;
        for (size_t i = 0; i < numBatchItems; i++)
        {
            let &c = consumers[i];
            fail_if(c.second != 0, "wrong input??");
            let& f = *c.first;
            if (i > 0) // sanity checks
            {
                fail_if(&GetInputFields(f.m_inputs.front()) != &GetInputFields(input0), "batched matrix gradients do not share the matrix??");
                let outputRank = DetermineTimesOutputRank(f);
                fail_if(outputRank != outputRank0, "BackpropToMatrixWeight() called on incompatibly-shaped operations?");
            }
            let& outGrad = GetOutputFields(f).m_gradient;
            let& right   = GetInputFields(f.m_inputs[1]).m_value;
            // to be able to batch gradient ops with different map rank, we flatten all to a single map dimension
            timesOutGrads       [i] = move(FlattenMapAxes(outGrad)); // incoming gradients from top
            timesDataRightInputs[i] = move(FlattenMapAxes(right  )); // second arguments
            let numItems = timesOutGrads[i]->Shape().Dimensions().back();
            fail_if(numItems != timesDataRightInputs[i]->Shape().Dimensions().back(), "batch dimension of two inputs not the same??");
            batchDim += numItems;
        }
        auto outGradBatch = GatherBatchInArena(timesOutGrads       , outputRank0, batchDim);
        auto rightBatch   = GatherBatchInArena(timesDataRightInputs, outputRank0, batchDim);
        m_stats.numAvoidedBackpropToMatrix += batchDim - 1; // these were saved

        // backprop into the left input from the batched outGrad and right
        auto& inputValues = BorrowBuffer(m_inputValuesBufferRaw, 2);
        inputValues[0] = nullptr;
        inputValues[1] = rightBatch.get();
        let gradient = CacheAndGetGradientView(input0, DetermineGradientStorageType(f0, 0));

        CudaStats* cudaStatsPtr = nullptr;
        if (ShouldLogMemoizeStats())
            cudaStatsPtr = BeginCudaStats(f0.m_op, nullptr, /*category=*/gradient.view->IsSparse() ? 1 : 0, input0.Shape().TotalSize(/*check=*/false), gradient.view->Device());

        PrimitiveFunction::BackpropTo(/*outputGradient=*/outGradBatch.get(),      // incoming gradient from top...
                                      /*index=*/0, f0.m_op, f0.m_attributes,      // ...goes through this function...
                                      /*outputValue=*/nullptr, inputValues,       // ...using these values from forward pass...
                                      gradient.view, gradient.beta, f0);          // ...into here
        m_stats.numBatchedBackpropToCalls++;

        EndCudaStats(cudaStatsPtr, gradient.view->Device());

#if 0   // debug the actual values
        Memoizer::LogFunction(f0, f0.m_profiler);
        gradient.view->LogToFile((input0.Name() == L"" ? f0.OpName() : input0.Name()) + L"_" + to_wstring(0), stderr);
#endif
#endif
    }

    // backprop gradient into 'var' by pulling all of its consumers (recursively)
    // This is the second function that does batching.
    // The vectors for building the lists are class members so that we reuse the malloc.
    // This is a subroutine of RAggregateGradientFromAllConsumers().
    vector<AutoBatchConsumer> m_spliceConsumers;             // Scatter optimization: backprop to all inputs at once using ScatterBatch
    vector<AutoBatchConsumer> m_matrixWeightConsumers[2][2]; // [isTransposed][isSparse] matrix product optimization: sum -> inner dimension
    vector<AutoBatchConsumer> m_viewableConsumers;           // see-through gradients being aggregated -> Gather and Reduce
    vector<AutoBatchConsumer> m_otherConsumers;              // remaining incoming gradients for which we have no further optimization
    void ClearBuckets()
    {
        m_spliceConsumers            .clear();
        m_matrixWeightConsumers[0][0].clear();
        m_matrixWeightConsumers[0][1].clear();
        m_matrixWeightConsumers[1][0].clear();
        m_matrixWeightConsumers[0][1].clear();
        m_viewableConsumers          .clear();
        m_otherConsumers             .clear();
    }
    void DetermineAndAddToBucket (const AutoBatchConsumer& c)
    {
        let* f = c.first;
        let index = c.second;
        fail_if(f->m_outputs.size() != 1, "for now only functions with a single output are supported"); // (needs some more plumbing to fix this)
        // backprop into Times' matrix argument
        // BUGBUG: This currently does not capture single time steps that backprop into the same matrix as a batch.
        // BUGBUG: (perf) We should also have a special path for Reshape(), as to avoid the memory copy.
        // splice operation should use scatter
        // BUGBUG: Really only true if the Splice originated from a batching operation.
        //         User-specified Splice ops might have arguments that receive no gradient, e.g. constants.
        let op = f->m_op;
        if (op == PrimitiveOpType::Splice)
        {
            m_spliceConsumers.push_back(c);
            return;
        }
        // backpropagating into the first arg of a matrix product
        // These are shared. This is how it works:
        //  - y = W x ; b = W a
        //  - dL/dW = dL/dy dy/dx + dL/db db/da
        //          = dL/dy x' + dL/db a'
        //          = [dL/dy dL/db] [x' ;
        //                           a']
        //          = [dL/dy dL/db] [x a]'
        // Algorithm:
        //  - for each (outputGradient, input),
        //     - the first outputRank dimension(s) get flattened and match the dims of W, flattened
        //     - the remaining dimension(s), the "map" axes, match between each (outputGradient, input)
        //        - flatten them -> one axis
        //        - splice all outputGradients and all inputs, respectively, along that axis
        //        - then multiply them
        //  - for this to work, all operations...
        //     - must interpret W the same way
        //        - same outputRank --> for now we will simply forbid using inconsistent outputRanks on the same matrix
        //     - remember these ops have not been batched in forward, possibly due to
        //        - complex map axes that were not obvious to match for batching,
        //          e.g. W * [13 x 42 x 5] and W * [13 x 100 x 200] which we'd not batch in forward, but can easily batch in backward
        //        - dependencies, e.g. same matrix in a recurrent loop
        // We only collect matrix products that fully match the first one. --TODO: when do they ever not fully match? Below we now enforce that they do.
        let VerifyMatrixGradient0Batchable = [](const PrimitiveFunction& f, const PrimitiveFunction& g)
        {
            // must be the same op (Times vs. TransposeTimes)
            fail_if(f.m_op != g.m_op, "VerifyMatrixGradient0Batchable: should not mix operations");
            // we compute leftGrad += resGrad @ right^T
            let  reductionRank = DetermineTimesOutputRank(f); // #dimenions over which matrix product reduces
            let gReductionRank = DetermineTimesOutputRank(g);
            if (reductionRank != gReductionRank) // this is not strictly required, but makes everything much easier, and never happens anyway
                InvalidArgument("VerifyMatrixGradient0Batchable: higher-order weight tensors of Times() and Affine() must presently be interpreted in the same way (same output rank)");
        };
        if (IsTimesOp(op) && index == 0)
        {
            //Memoizer::LogFunction(*f, L"mb", index), fflush(stderr);
            bool isTransposed = (op == PrimitiveOpType::TransposeTimes || op == PrimitiveOpType::TransposeAffine);
            bool isSparse = f->m_inputs[1].IsSparse();
            auto& consumers = m_matrixWeightConsumers[isTransposed][isSparse]; // there are 4 categories that are not batchable across
            if (!consumers.empty()) // verify that we got here with correct dimensions
                VerifyMatrixGradient0Batchable(*f, *consumers.front().first);
            consumers.push_back(c);
            return;
        }
        // backprop where gradients are just views that we can sum up
        //else if (f->m_op == PrimitiveOpType::Plus)
        //    return m_viewableConsumers;
        // all other
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
    void RAggregateGradientFromAllConsumers(VariableFields& fields)
    {
        if (m_visitorTag.Visited(fields.m_visitedTag))
            return;
        //if (fields.m_uniqueIdForDebugging == 5526)
        //    Break;

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
        // BUGBUG: This will fail in case of mixed dense/sparse gradients into a matrix.
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

        // summation bucket
        // ...not used yet

        // others bucket
        for (auto& c : m_otherConsumers)
            BackpropToUnbatched(c, /*viewAllowed=*/false);

        // matrix-weight buckets
        // Matrix-weight gradients may be column-sparse or dense.
        // If a matrix weight receives both kinds of gradients, we do the dense ones first,
        // since we have an op to add a sparse matrix into a dense one, but not vice versa.
        // The use case is tied embeddings in MT, where the same embedding is applied to sparse inputs
        // and is also used as a Softmax projection, which is transposed and dense.
        for (size_t isSparse = 0; isSparse < 2; isSparse++) // watch out: all dense must be before sparse
            for (size_t isTransposed = 0; isTransposed < 2; isTransposed++)
            {
                let& consumers = m_matrixWeightConsumers[isTransposed][isSparse];
                if (!consumers.empty())
                    BackpropToMatrixWeight(consumers);
            }
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
    // If you need multiple gradients, ask for them in a single go to avoid duplicate computation.
    // Beta can be 1 or 0. If 0, gradients are overwritten, and if 1, added to.
    void BatchedBackward(const InternalVariable& root, unordered_map<Parameter, NDArrayViewPtr>& gradients, double beta)
    {
        if (!root.m_dataFields->m_needsGradient)
            LogicError("BatchedBackward: cannot compute gradient for root with m_needsGradient being False.");
        // BUGBUG: make sure some edge cases are done right:
        //  - root.m_needsGradient=false
        //  - gradients contains non-Parameters
        //  - root is a m_redirection
        // first get the forward computation, batching, etc. done if not yet
        BatchedForward(root);
        // note: if BatchedForward runs out of GPU RAM, then this call will have thrown an exception

        ResetCudaStats(/*updateCounter=*/false); // false means keep the same logging dis/enable conditions as Forward
        CudaStatsGuard cudaStatsGuardBackward(PrimitiveOpType::ForwardBackward/*misusing this for actual op*/, L"batched BACKWARD", 3);

        // if user passed NDArrayViewPtrs for the gradients, then implant those
        // If nulls are passed, then the existing gradient memory will be reused, if any, and gradients[] will be updated to return that.
        // If gradients[] gets initialized with nulls, and then never changed outside, one will keep reusing the existing buffers.
        for (auto& kv : gradients)
        {
            let& param = kv.first;
            if (kv.second)
                param.m_dataFields->m_gradient = kv.second; // (if null then we will keep what's already there, or create them if null)
        }
        // --- Phase 1: form the inverted backprop graph
        //  - set up the m_consumer fields to form the inverted backprop graph
        //  - this will also short-circuit chains of see-through ops (in-place, as an equivalence transform)
        // Note: We may consider to do this phase concurrently with BatchedForward's memoization thread.
        // This will require to not call End() in BatchedForward(). We'd need to verify that RPrepareBackwardGraph()
        // does not access m_value fields (nor can any other part in this phase), as those are populated by the bg thread.
        // For the Marian scenario, doing this concurrently would not save much time, but that may be different for a real auto-batched scenario.
        CudaStatsGuard cudaStatsGuardInit(PrimitiveOpType::LabelsToGraph/*misusing this for graph initializing*/, L"backward init", 3);
        m_visitorTag.Begin();
        // first set it up for the Parameters for which we have requested a gradient
        // This way we won't backprop into gradients of Parameters that we did not ask for.  --TODO: implement this
        for (auto& kv : gradients)
        {
            auto& gradFields = GetGradientFieldsForBackprop(ResetInputGradient(kv.first), /*firstTimes=*/true);
            if (m_visitorTag.Visited(gradFields.m_visitedTag)) // (note that the code does not require this; this is only to point out a likely user error)
                InvalidArgument("BatchedBackward: a Parameter was included more than once in gradients[]");
            RPrepareBackwardGraph(gradFields, /*userOwnsGradients=*/true, beta);
            // BUGBUG: ^^ userOwnsGradients won't work correctly if one Var in gradients[] is an input to another
        }
        // now build the backward graph. We use visited information for the gradients to infer our own needsGradient flag  --TODO: No, not done yet.
        auto& rootGradFields = GetGradientFieldsForBackprop(ResetInputGradient(root), /*firstTimes=*/true);
        if (!m_visitorTag.Visited(rootGradFields.m_visitedTag)) // (A crazy user may have passed root itself in gradients[]. That is OK.)
            RPrepareBackwardGraph(rootGradFields);
        // sanity check
        for (auto& kv : gradients)
        {
            let& gradFields = GetGradientFieldsForBackprop(kv.first);
            if (gradFields.m_consumers.empty()) // if gradient's Parameter has no consumers, then it is not part of the root
                LogicError("BatchedBackward: requested gradient \"%S\" is not part of root.", kv.first.Name().c_str()); // TODO: or could it be due to StopGradient? What if StopGradient is used only sometimes?
            if (!gradFields.m_needsGradient) // (we could also just leave the gradient 0)
                LogicError("BatchedBackward: cannot compute gradient for variable with m_needsGradient being False."); // such as a Constant. Actually, why? User can certainly get that if desired.
        }
        cudaStatsGuardInit.Stop();
        // --- Phase 2: backprop through the inverted backprop graph
        //  - perform backprop operation, by depth-first traversal of inverted backprop graph starting from gradients
        m_visitorTag.Begin();
        // implant the first gradient. This "root: gradient is actually the leaf in the inverted backprop graph.
        // TODO: allow user to pass in the starting value
        // BUGBUG: we get a [1] here, but should be a scalar. This is a bug outside.
        //if (root.Value()->Shape() != NDShape{})
        //    LogicError("BatchedBackward: root must be a scalar, or root gradient must have been implanted already");
        rootGradFields.m_gradient = m_memoizer.Arena().NewNDArrayView(root.Shape(), root.GetDataType(), StorageFormat::Dense, root.Value()->Device());
        rootGradFields.m_gradient->SetValue(1.0f);
        m_visitorTag.Visited(rootGradFields.m_visitedTag); // done with this
        // perform backprop
        // This traverses the tree top-down, where each node pulls gradient(s) from its consumer(s).
        // This way we can optimize operations, such as a matrix product or gradient of GatherBatch().
        m_memoizer.Begin(); // reset bg thread
        for (auto& kv : gradients)
        {
            auto& gradFields = GetGradientFieldsForBackprop(kv.first);
            RAggregateGradientFromAllConsumers(gradFields);
        }
        //fprintf(stderr, "Back-propagated through %d functions\n", (int)order.size());
        // implant the results into the map the user passed in
        for (auto& kv : gradients)
            kv.second = kv.first.m_dataFields->m_gradient;

        cudaStatsGuardBackward.Stop(); // this measures the backprop process, which submits actual computation to GPU, with occasional (undesired) internal syncs

        CudaStatsGuard cudaStatsGuardCalc(PrimitiveOpType::Assign/*misusing this for actual op*/, L"backward thread hangover", 3);

        m_memoizer.End(); // let CUDA submission thread complete its submitting work
        // At this point, no more CUDA submissions will be made, but the CUDA ops themselves should still be ongoing.

        cudaStatsGuardCalc.Stop(); // this measures the bg thread; specifically, how much longer it needs after the fg thread has submitted the last item

        // log stats
        ShowCudaStats();

        // note: we will leave the m_consumers fields dangling, and reset them upon next call to Backward() (which users are not meant to do)
#ifdef LOG_STATS
        fprintf(stderr, "BatchedBackward: %d bp calcs + %d views + %d gathers + %d scatters + %d set-zeroes, %d skipped matmuls for %d ops (total %d inputs)\n",
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
    auto autoBatcher = InternalVariable::AutoBatch();
    return autoBatcher.BatchedForward(m_outputs.front());
}

// Perform backprop.
// TODO: CNTK grad() allows to pass multiple roots. Does that ever make sense in this context?
void PrimitiveFunction::BatchedBackward(std::unordered_map<Parameter, NDArrayViewPtr>& gradients, double beta) const
{
    auto autoBatcher = InternalVariable::AutoBatch(); // has some internal state
    autoBatcher.BatchedBackward(m_outputs.front(), gradients, beta);
}

// non-batched version of BatchedForward()
// This is actually not used except as a fallback for debugging.
// TODO: move to -Eval.cpp
void PrimitiveFunction::Forward() const
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
    output.m_dataFields->m_value = move(Forward(m_op, m_attributes, output.IsVolatile(), args, output.Shape(), move(out), *this));
}

// ===========================================================================
// Invocable and our pieces of PrimitiveFunction and BlockFunction -- contained here for now
// ===========================================================================

// special short-circuited version where output is created outside
void PrimitiveFunction::InitOutput(InternalVariable&& output)
{
    //std::call_once(m_outputsInitFlag, [this]() {});
#ifndef DYNAMITE_ONLY
    m_outputInitializingByThreadId = std::thread::id();
#endif
    fail_if(m_outputsInitFlag || !m_outputs.empty(), "InitOutput called twice??");
    m_outputsInitFlag++;
    output.SetOwner(static_pointer_cast<PrimitiveFunction>(shared_from_this()));
    // This really belongs inside the constructor, but we don't have the shared_ptr yet. Not nice this way.
    //m_outputs.resize(1);
    //m_outputs.front() = move(output);
    m_outputs.assign(nullptr/*1*/, move(output));
}

// Static invocation
//  - a static graph can be invoked with input arguments that substitute for its Placeholders
//  - this is done via the Invocable class
//  - invocation:
//     - an invocation translates into a single node in the graph, which links to the static graph via m_composite
//     - multiple invocations share the same underlying static graph
//     - static graphs can be called both during dynamic computation ("dynamic invocation"), as well as during construction of other static graphs ("static invocation")
//     - upon first dynamic invocation, the static graph's shapes are determined
//        - to support stacking, the last axis of each input may be variable. It is stored and inferred in the static graph as FreeDimension.
//        - at static-graph construction time, input shapes are not known. For now, user must tell at that location what the stacking axis is.
//          E.g. a static graph Times(W,_) can be applied to inputs with variable #columns. Pass freeAxis = 1 here to allow stacking them.
//     - static graphs can be invoked in two modes:
//        - early inlining (isBasicBlock=false): static graph gets inlined into the dynamic graph before any batching decisions
//          Allows for fine-grained auto-batching, as if the static graph's operations were just dynamically unrolled.
//        - late inlining (isBasicBlock=true): static graph gets treated like a PrimitiveFunction for purpose of batching.
//          Allows to cut overhead of the auto-batching algorithm, and also acts as an auto-batching constraint.
//  - variable batch dimension (FreeDimension):
//     - static graphs get their shapes inferred only once, upon first invocation
//       This is to save time. Due to this, early inlining of a static graph is more restrictive than regular dynamic invocation.
//     - static graph can only have one free dimension, in the "free axis" which must be the last. This is implemented via CNTK's FreeDimension mechanism.
//     - user must declare the free axis location for all arguments
//     - upon invocation, either all or no actual argument must have that axis.  --TODO: I think they no longer need to.
//       If they do, they must all have the same dimension in their respective axis, which the output will also have. If not, the output will not have the axis.
//        - early inlining:
//           - allowed: f([I x *], [J x K x *]) -> [N x *],
//                      invocable as f([I x T], [J x K x T]) -> [N x T]
//                      or as f([I], [J x K]) -> [N]
//           - not supported: f([I x *], [J x K]) -> [N x ?] (second arg has no free dimension)
//           - not supported: invoke as f([I x T], [J x K]) -> [N x ?] (second arg has no free dimension)
//             TODO: This ^^ is actually doable with just a little more code.
//        - late inlining:
//           - the batch axis must be the same for all args, and no intermediate op may touch that axis
//             TODO: We could relax this, but for now it simplifies things a lot. (TODO: revisit later to see if it really does)
//           - allowed: f([I x *], [J x *]) -> [N x *]
//           - not supported: f([I x *], [J x K x *]) -> [N x *] (batch dim not at same location)
//  - batching
//     - any invocation with free dimension can use stacking or batching
//        - batching if all inputs have matching dimension in the respective free axes. Batching adds one more axis.
//          { [I x J X T], [I x J x T] } -> [I x J x T x 2], with T=batch axis
//          In this situation, the actual execution will see tensors with one axis to the right of the free axis. TODO: So don't use back() on operands!
//        - stacking if free-dimension values differ
//          { [I x J X T1], [I x J x T2] } -> [I x J x (T1+T2)]
//        - none at all if not batchable
//     - batchability per stacking condition: two ops are batchable if their inputs have matching dimensions except for the free axis


/*Internal::*/Invocable::Invocable(size_t arity, size_t freeAxis, bool isBasicBlock, const function<Variable(const vector<Variable>&)>& lambda, std::wstring name) :
    m_arity(arity), m_isBasicBlock(isBasicBlock)
{
#ifdef NO_BLOCK_INVOKE // for debugging, we can disable static invocation altogether
    m_operands.resize(m_arity);
    m_lambdaRememberedForDebugging = lambda; // just remember the lambda, and done
    m_nameRememberedForDebugging = name;
    return;
#else

    // -- create the composite
    // allocate m_argumentList/m_operands and populate the Placeholder section (later we will add Parameters)
    for (size_t i = 0; i < m_arity; i++)
    {
        let arg = PlaceholderVariable();
        // implant the redirect into the placeholder
        arg.m_dataFields->m_compositeArgumentIndex = i;     // when dynamically expanding this, we match up this Placeholder with the respective input[i]
        m_argumentList.push_back(arg);
        m_argumentFreeAxes.push_back(freeAxis);
    }
    // invoke the lambda with Placeholders as arguments
    // This builds the graph as a CompositeFunction.
    FunctionPtr fPtr = lambda(m_argumentList);
    // note: the graph is built by calling the lambda on Placeholders
    // We must pass in the Placeholders and remember them, since the fPtr itself will not remember their ordering.
    if (!name.empty())
        fPtr = Alias(fPtr, name);
#if 0
    fprintf(stderr, "Invocable('%S'):\n", name.c_str());
    for (let& p : fPtr->Parameters())
        fprintf(stderr, "    %S : %S\n", p.Name().c_str(), p.Shape().AsString().c_str());
#endif
    auto fCompositePtr = dynamic_pointer_cast<CompositeFunction>(fPtr);
    if (!fCompositePtr) // If the static graphs has no Placeholder (e.g. weight norm), then it won't be a composite. Make it one.
        fCompositePtr = dynamic_pointer_cast<CompositeFunction>(CompositeFunction::Create(dynamic_pointer_cast<PrimitiveFunction>(fPtr), fPtr->Name(), /*uid=*/wstring()));
    // precompute some additional info for basic blocks
    // Some of these are used for composites that are declared as basic blocks, but also for those
    // invoked by basic blocks even if not declared as such. Hence, we must initialize this always.
    // (we probably only need m_freeAxis; the others are only used for sceduling).
    // TODO: Review how these propagate into nested composites. Do we need to copy them in below, and can freeAxis move around?
    //        Seems this is done in the cached comparison of composites.
    fCompositePtr->m_basicBlockInfo.m_freeAxis = m_argumentFreeAxes.empty() ? SIZE_MAX : freeAxis; // remember the one axis used for all ops inside the basic block
    fCompositePtr->m_basicBlockInfo.m_batchableCompositeId = SIZE_MAX; // composites with the same id are batchable
    fCompositePtr->m_basicBlockInfo.m_batchNormIds = make_unique<vector<size_t>>();
    fCompositePtr->m_basicBlockInfo.m_depthHint = 0;
    Function::PreorderTraverseFunctions(fCompositePtr->RootFunction(), [&](const FunctionPtr& fPtr)
    {
        let& f = *dynamic_pointer_cast<PrimitiveFunction>(fPtr);
        let op = f.OpType();
        if (op == PrimitiveOpType::BatchNormalization)
        {
            let bnId = f.Attributes()[PrimitiveFunction::AttributeNameSyncId].Value<size_t>();
            fCompositePtr->m_basicBlockInfo.m_batchNormIds->push_back(bnId);
        }
        if (InternalVariable::AutoBatch::IsBarrier(f))
            fCompositePtr->m_basicBlockInfo.m_depthHint = max(fCompositePtr->m_basicBlockInfo.m_depthHint, f.Attributes()[PrimitiveFunction::AttributeNameSyncId].Value<size_t>());
    }, /*traverseInsideBlockFunction=*/ true);
    m_composite = move(fCompositePtr);

    // -- prep this class for Invoke(). Complete the m_argsMap pairs by including all learnable Parameters in it as well.
    // This is needed so that the auto-batcher can see all Parameters that are inside, without having to traverse it.
    m_operands.resize(m_argumentList.size());
#if 0 // instead we explicitly expand it in InputsImpl()
    for (let& p : m_composite->Parameters())
    {
        m_argumentList.push_back(p);            // presently also must pass all Parameters
        m_operands.push_back(p);                // we prepopulate the operands here, these are not changed afterwards
        // BUGBUG: This is not correct! We must batch the parameters. How is that accomplished?
        m_argumentFreeAxes.push_back(SIZE_MAX); // make sure we can index these elements, too
    }
#endif
    m_stillNeedsToInferShapes = true;
#endif // NO_BLOCK_INVOKE
}

// determine "the" free dimension of an invocation, using a hack (fixed axis := 1)
// The result is either the dim (if all inputs have a batch dim) or ABSENT_FREE_DIMENSION (if no input has a batch dim), or an error.
// BUGBUG: This constraint is no longer needed. Also, tis should use the correct axes.
// This function has the heuristics built in that the batch axis is hard-coded as 1.
template<typename InputsVectorType>
static NDShapeDimension DetermineInvokeFreeDim(const InputsVectorType& inputs)
{
    let freeAxis = 1; // TODO: this must be parameterized somehow
    NDShapeDimension invocationArgsFreeDim = 0; // 0 means no input at all
    for (let& input : inputs)
    {
        if (input.IsParameter()) // placeholders have no batch dimension
            continue;
        let& shape = input.Shape().Dimensions();
        let rank = shape.size();
        let thisFreeDim = freeAxis < rank ? shape[freeAxis] : ABSENT_FREE_DIMENSION;
        if (invocationArgsFreeDim == 0) // first encounter
            invocationArgsFreeDim = thisFreeDim;
        else if (invocationArgsFreeDim != thisFreeDim)
            InvalidArgument("Invoke: Inconsistent replacement for FreeDimension %d (previous: %d) for placeholder's shape %S.", (int)thisFreeDim, (int)invocationArgsFreeDim, input.Shape().AsString().c_str());
    }
    return invocationArgsFreeDim;
}

// helper to replace actual the batch dimension of inputShape by FreeDimension
static inline NDShapeDimensions ReplaceWithFreeDim(const NDShapeDimensions& inShape)
{
    // BUGBUG: We must honor the declared freeAxis here.
    let rank = inShape.size();
    if (rank < 1 || rank > 2)
        InvalidArgument("Invoke: Shapes with rank < 1 or > 2 currently not supported here.");
    if (rank == 1)
    {
        auto shape = MakeVector(inShape); // PERF BUGBUG: Do this without malloc
        shape.push_back(NDShape::FreeDimension);
        return NDShapeDimensions(shape);
    }
    else
    {
        auto shape = inShape;
        shape.back() = NDShape::FreeDimension;
        return shape;
    }
}

// helper to replace the batch dimension of inputShape by the given value
// The batch dim must be present. If the given value is ABSENT_FREE_DIMENSION, it strips the axis.
static inline NDShapeDimensions ReplaceFreeDim(const NDShapeDimensions& inShape, NDShapeDimension batchDimValue)
{
    fail_if(batchDimValue == NDShape::FreeDimension, "use ReplaceWithFreeDim() instead");
    fail_if(inShape.back() != NDShape::FreeDimension, "ReplaceFreeDim called on shape that has no FreeDimension??");
    //let rank = shape.size();
    if (batchDimValue == ABSENT_FREE_DIMENSION)
    {
        //shape.resize(rank - 1); // no batch dimension: drop the axis
        //auto shape = MakeVector(inShape); // PERF BUGBUG: Do this without malloc --> SubShape()
        //shape.pop_back(); // no batch dimension: drop the axis
        //return shape;
        return inShape.BackPopped(); // this creates a copy, but all uses of this copy it on
    }
    else
    {
        auto shape = inShape;
        shape.back() = batchDimValue;
        return shape;
    }
}

// called to invoke a static function inside a dynamic graph, or as part of building another static graph
// The operands must have been implanted in m_operands before calling this, and will be cleared at the end.

// For Dynamite only, we are (mis-)using the BlockFunction to represent a PrimitiveFunction that Dynamite can interpret.
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
Variable /*Internal::*/Invocable::DoInvoke() const // note: caller must call SetOperand() first to set the operands
{
#ifdef NO_BLOCK_INVOKE // for debugging, we can disable static invocation altogether
    return m_lambdaRememberedForDebugging(m_operands);
#else
    // To invoke it, we place the arguments into the m_argsMap array next to the corresponding Placeholder.
    // We leave the Parameters in the m_argsMap array untouched (they are at the end).
    // After the call, we destruct the argument as to not accidentally keep a reference to the argument around.
    //  m_argumentList = composite->Arguments() in a given order; Placeholders first, then all Parameters. Get updated upon determining shapes.
    //  m_operands     = what the arguments should prerent to be

    // BUGBUG: Do we need this? m_inputs.emplace_back(std::move(inputVar.NonCompositePreservingCopy())); for the operands  --TODO: <-- what does this comment refer to??
    // TODO: Since we copy the operands, we could augment the Parameters here as well.
    let callee = static_pointer_cast<CompositeFunction>(m_composite); // (static cast since caller must have called InitCompositeForInvoke() before, which checked the type)
#ifndef NO_LATE_INLINING
    let isBasicBlock = m_isBasicBlock;
#else
    let isBasicBlock = false;
#endif

    // this leaves 'm_stillNeedsToInferShapes' true until called for the first time with fully known shapes
    // This returns 'true' only once, and evaluates the IsUnknown test only once for a dynamic invocation
    // (but multiple times for initial invocations during construction of static graphs).
    if (m_stillNeedsToInferShapes && all_of(m_operands.begin(), m_operands.end(), [](const Variable& arg) { return !arg.Shape().IsUnknown(); }))
    {
        m_stillNeedsToInferShapes = false;
        // The very first time we pass the composite, we must set up its Placeholder m_compositeArgumentIndex fields.
        // The caller must pass in this flag.
        // Note: We only ever get here once during the life of an Invocation; so this is the fast path.

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

        // determine the batch dimension
        let batchDim = DetermineInvokeFreeDim(m_operands);

        // BUGBUG: Must verify that all Parameters are covered here.
        unordered_map<Variable, Variable> replacementMap;
        for (size_t i = 0; i < m_operands.size(); i++)
        {
            let& compositeLeaf = m_argumentList[i]; // Placeholder or Parameter in composite
            let& operand = m_operands[i];           // what they should pretend to be
            if (compositeLeaf.IsParameter())
            {
                // for Parameters, supply an empty Variable
                if (operand != compositeLeaf)
                    LogicError("Invoke: Parameters should have passed as themselves.");
                // That's it. We just keep it in the list so that auto-batch can find them.
            }
            else if (compositeLeaf.IsPlaceholder())
            {
                // verify the mappings have been implanted
                fail_if(compositeLeaf.m_dataFields->m_compositeArgumentIndex != i, "m_compositeArgumentIndex not set up??"); // when dynamically expanding this, we match up this Placeholder with the respective operand [i]
                // TODO: rethink the logic. If the operand's shape IsUnknown, then why not directly return? Why even replace?
                if (operand.IsInput())
                    InvalidArgument("Invoke cannot work on Input variables, it is for dynamic networks only.");
                fail_if(operand.Shape().IsUnknown(), "unknown operand shapes at this point??");
                // we replace with a placeholder of the same type. This gives the composite the shape.
                // to allow for varying sequence axis, we replace the last axis with FreeDimension, and infer with that
                // If the operand does not have that axis, then we create it.
                fail_if(batchDim == 0, "no batchDim determined??");
                // TODO: check against the current Placeholder
                auto updatedCompositePlaceholder = PlaceholderVariable(ReplaceWithFreeDim(operand.Shape().Dimensions()),
                    operand.GetDataType(), operand.Name(), operand.DynamicAxes(), operand.NeedsGradient(), operand.IsSparse());
                //auto updatedCompositePlaceholder = PlaceholderLike(operand);
                updatedCompositePlaceholder.m_dataFields->m_compositeArgumentIndex = compositeLeaf.m_dataFields->m_compositeArgumentIndex;
                replacementMap.insert({ compositeLeaf, updatedCompositePlaceholder }); // replace with new Placeholder, which has a block mapping implanted
                // TODO: fix the interface. This should become a private method to Invocable
                m_argumentList[i] = updatedCompositePlaceholder;
            }
            else
                InvalidArgument("Invoke: argumentList can only contain Placeholders and Parameters.");
        }

        if (!replacementMap.empty())
            callee->ReplacePlaceholders(replacementMap); // This gives the composite the shape.
        // OUTDATED --The composite args' Placeholders now have a block mapping to the actual inputs.
        // BUGBUG: That's bad, since they are gone after this. There should be no block mapping.

        // if batchDim != ABSENT_FREE_DIMENSION, the resulting composite output likely has a batch axis as well, which is also FreeDimension
        if (compositeOutputs.front().Shape().IsUnknown()) // or not?
            LogicError("Invoke with determineShapes=true must not be called with inputs with Placeholder dimensions.");

#if 0
        fprintf(stderr, "BlockFunction('%S') : %S\n", callee->Name().c_str(), compositeOutputs.front().Shape().AsString().c_str());
        for (let& p : Parameters())
            fprintf(stderr, "    %S : %S\n", p.Name().c_str(), p.Shape().AsString().c_str());
#endif
        // Now the composite is fully type-inferred; ready for consumption by Dynamite.
    }

    FunctionPtr f;
    if (isBasicBlock)
    {
        // basic block: we generate a Block operation that is batched as a whole
        f = MakeSharedObject<BlockFunction>(callee, Function::InputsVectorType(m_operands), isBasicBlock);
        static_pointer_cast<BlockFunction>(f)->FinalizeInvoke(m_argumentList, /*shapeIsKnown=*/!m_stillNeedsToInferShapes);
    }
    else
    {
        //fprintf(stderr, "EARLY UNROLLED\n");
        // not a basic block: we inline the static graph right here
        // The difference to having user code unroll explicitly is that cloning is much cheaper due to short-circuiting.
        NDShapeDimension invocationArgsFreeDim = ABSENT_FREE_DIMENSION;
        NDShapeDimension inputsBatchDimDummy;
        f = InternalVariable::AutoBatch::RInlineComposite(static_cast<PrimitiveFunction&>(*callee->RootFunction()),
                                                          Function::InputsVectorType(m_operands), invocationArgsFreeDim, inputsBatchDimDummy,
                                                          /*cloneFn=*/InternalVariable::AutoBatch::ClonePrimitiveFunction,
                                                          VisitorTag(), /*shapeIsKnown=*/!m_stillNeedsToInferShapes);
    }
    // release references to the arguments in m_operands
    for (size_t i = 0; i < m_arity; i++)
        SetOperand(i, m_noArg);
    // behave as if this was returning a Composite: implant a ref count to ourselves. This will be taken over by the next consumer.
    return f->Output(/*init=*/ false);
#endif // NO_BLOCK_INVOKE
}

// call this after construction from Invoke()
void BlockFunction::FinalizeInvoke(const vector<Variable>& argumentList, bool shapeIsKnown)
{
    // now set up the output variable. Clone the composite's one output Variable, then inject the mapping pointer. This following the pattern of InferOutputs().
    // ...EXCEPT we do not implant a Variable mapping, since the composite is shared. The composite does not know that it is part of a BlockFunction.
    let& compositeOutput = m_composite->m_outputs.front();
    NDShapeDimensions outputShape;
    let freeAxis = 1; // TODO: this must be parameterized somehow
    // TODO: This should use the same heuristics as during inlining. But we don't know the placeholders, do we?
    if (shapeIsKnown && compositeOutput.Shape().Rank() > freeAxis && compositeOutput.Shape().Dimensions().back() == NDShape::FreeDimension)
    {
        // if input shape is known and composite output has a FreeDimension in the batch axis, then replace it with the actual value
        let inputsFreeDim = DetermineInvokeFreeDim(m_inputs);
        // type-check the inputs
        let arity = m_inputs.size();
        for (size_t k = 0; k < arity; k++)
        {
            let& input = argumentList[k];
            if (!input.IsPlaceholder())
                continue;
            let& operand = m_inputs[k];
#if 1   // this code is duplicated from RInlineComposite()--share this
            // input = placeholder; operand = what it should pretend to be
            // Typecheck.
            // PERF BUGBUG: This typecheck is done repeatedly for the same argument. Can we save that effort?
            let& placeholderDims = input.Shape().Dimensions();
            let& operandDims = operand.Shape().Dimensions();
            let placeholderRank = placeholderDims.size();
            let operandRank = operandDims.size();
            let placeholderHasFreeDimension = (placeholderRank > 0 && placeholderDims.back() == NDShape::FreeDimension);
            let itemRank = placeholderRank - placeholderHasFreeDimension;
            //if (placeholderHasFreeDimension)
            //    Break;
            // itemRank = shape components that must match
            if (operandRank < itemRank || operandRank > placeholderRank)
                InvalidArgument("Invoke: Argument shape %S too short to match placeholder's shape %S.", operand.Shape().AsString().c_str(), input.Shape().AsString().c_str());
            for (size_t k = 0; k < itemRank; k++)
                if (operandDims[k] != placeholderDims[k])
                    InvalidArgument("Invoke: Argument shape %S incompatible with placeholder's shape %S.", operand.Shape().AsString().c_str(), input.Shape().AsString().c_str());
            // deal with invocationArgsFreeDim
            let thisFreeDim = (placeholderHasFreeDimension && operandRank == placeholderRank) ? operandDims.back() : ABSENT_FREE_DIMENSION;
            if (inputsFreeDim != thisFreeDim)
                InvalidArgument("Invoke: Inconsistent replacement for FreeDimension %d (previous: %d) for placeholder's shape %S.", (int)thisFreeDim, (int)inputsFreeDim, input.Shape().AsString().c_str());
            // BUGBUG: ^^ I think we can now handle inconsistent ones.
#endif
        }
        outputShape = ReplaceFreeDim(compositeOutput.Shape().Dimensions(), inputsFreeDim);
    }
    let isVolatile = any_of(m_inputs.begin(), m_inputs.end(), [](const Variable& input) { return input.IsVolatile(); });
    InternalVariable blockOutput =
        /*if*/ (!shapeIsKnown) ? // if we are being called while building a static graph
            OutputVariable(NDShape::Unknown(), /*remaining args are dummy:*/ DataType::Unknown, vector<Axis>(), /*needsGradient=*/true, /*isSparse=*/false, isVolatile, Name())
        /*else*/:
            OutputVariable(outputShape.empty() ? compositeOutput.Shape() : NDShape(outputShape), compositeOutput.GetDataType(), vector<Axis>(), compositeOutput.NeedsGradient() && !isVolatile, compositeOutput.IsSparse(), isVolatile, Name());
    fail_if(blockOutput.Shape().HasFreeDimension(), "Invoke: still has FreeDimension??");
    InitOutput(move(blockOutput));
    // DELETE THIS after this commit
    //// behave as if this was returning a Composite: implant a ref count to ourselves. This will be taken over by the next consumer.
    //// TODO: do not create this, instead do it at call site (there is only one)
    //return Variable(m_outputs.front(), static_pointer_cast<PrimitiveFunction>(shared_from_this()), ConstPrimitiveFunctionPtr());
    //// TODO: Maybe this can instead keep the primitive directly?
    ////return m_outputs.front().CompositePreservingCopy(static_pointer_cast<PrimitiveFunction>(shared_from_this()));
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
    if (shape.IsUnknown()) // BUGBUG: FreeDimension
        InvalidArgument("size: Variable has no known size yet.");
    return shape.Dimensions().back();
}

} // namespace CNTK
