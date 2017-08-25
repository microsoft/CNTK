//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// The functions for automatically-batched evaluation of dynamic graphs (forward and backward) are contained here.

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "stdafx.h"
#include "PrimitiveFunction.h"
#include "CNTKLibrary.h"
#include "Variable.h"
#include "PrimitiveOpType.h"
#include "PrimitiveFunction.h"
#include "CommonMatrix.h"
#include "Utils.h"

#include <unordered_map>
#include <vector>
#include <string>

//#define LOG_DETAILS   // if defined, log all forward and backward operations
#define LOG_STATS     // if defined, log statistics (#operations)
//#define NO_BATCHED_FORWARD  // if defined, don't batch forward
//#define NO_BATCHED_BACKPROP // if defined, don't do batched backprop

static size_t logMemoizeStatsPeriod = 10;
static size_t logMemoizeStatsCounter = 0; // counts up to logMemoizeStatsPeriod and wraps. We log if it is 0.

using namespace Microsoft::MSR::CNTK;
using namespace std;

#define BarrierOp NoOp // for now, we use Alias() (=NoOp) to denote a Barrier(). Should become an op in its own right.

#pragma warning (disable: 4456) // until I fixed the shadowing

#define let const auto
#define fail_if(cond, err) (!!(cond) ? (LogicError(__FUNCTION__ ": " err),0) : 0)
#define BreakPoint fprintf(stderr, "") // use this inside a conditional to be able to set a breakpoint in Release code

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
    // As an optimization, the gather op is skipped if all items are already consecutive in memory.
    // This is often the case when they themselves are the result of a batched computation.
    // 
    // The concatenation axis is always last (slowest-changing) axis of the batched data; either a new one (batching) or the last one (stacking).
    // 
    // For now, we use batching and not stacking, since the Index proxy only stores an index and not a range.
    // I.e. all args must have identical shapes including the last axis. To support stacking along the last axis,
    // we'd need to store a range. This will allow mismatching last dimensions (e.g. stacking vectors of different
    // dimensions), but not all primitive operations can handle such changed dimensions.
    // One can also imagine more elaborate stacking in different dimensions, and even in multiple dimensions
    // (like current CNTK MBLayout).
    // 
    // If all args to be batched are identical (all items are the same object), then, if possible, the item
    // is not copied, but instead virtually duplicated by means of broadcasting. E.g. if "b1 is b2" (in Python parlance),
    // then we can avoid one copy operation and write:
    // 
    //   c1_c2 = [a1, a2] + b1
    // 
    // where b1 will automatically be broadcast into the second dimension by the "+" operation.
    // This happens, for example, when adding a bias to each step of a sequence.
    // This optimization can only be done for operators that support broadcasting.
    // 
    // If all args have this property, then the operation is computed unbatched but only once, and the result is shared:
    // 
    //   c1_c2 = a1 + b1
    //   c1 = c2 = c1_c2
    // 
    // When comparing dimensions, trailing singleton dimensions can be ignored if possible.
    // This is not implemented presently, since it makes the comparison operation more complex.
    // 
    // The batching operation happens after output and Parameter shapes have been fully determined, as that
    // happens during graph building, before the computation kicks in. The already determined
    // shapes can directly drive reduction and reshape operations.
    // 
    // The following will go through every single operation and specify how it is batched.
    // This is designed in a way that even multiple batching, with multiple added batch axes, will behave correctly.
    // TODO: This is a TODO list for now with intent to be implemented this way.
    // 
    // TODO: Needed redesign:
    //  - Lazy index must contain a range. So it cannot be a pair<>... which is better anyway.
    //  - Lazy index must contain an optional built-in reshape.
    //    This may be possible without actually storing the shape (which would be a malloc()),
    //    since the target dimension is already known when we have access to the unbatched result.
    //    We should add NDArrayView::SliceViewAsShape(), and always use that.
    // 
    // general condition
    // -----------------
    // 
    // This condition applies to all operations, and will be modified with additional op-specific conditions.
    // There is only one condition that is not an AND with these; which is a relaxation of shape match for Reshape().
    // 
    // Conditions for batching:
    //   All arg arguments must completely match in dimensions. The key for comparison is the full shape.
    // 
    // Conditions for stacking:
    //   All args must match except for their last dimension. The key for comparison is the shape less the last dimension.
    //   Reduction dimensions may not reduce along the last dimension (otherwise fall back to batching).
    //
    // The conditions are tested in Schedule() and ExecuteBatchedOpAndSchedule().
    //
    // The op-specific conditions are specified in the following initializer.
    // 
    // TODO: implement these V1 nodes somehow via TensorView
    //  - Convolution, Pooling, Unpooling
    //  - RandomDistribution  // for dropout
    //  - BatchNormalization  // for MT
    //  - OptimizedRNNStack   // for MT
    // 
    // TODO:
    //  - figure out Hardmax. Can we have a OneHot-like op in TensorView?

    enum class OpSpecificConditionKind  :  size_t // the meanings of these are specified below
    {
        UnaryElementWise, Reducing, NoOp, Reshape, BinaryElementWise, TernaryElementWise, Pooling,
        Slice, Splice, Transpose,
        MatrixProduct, Convolution,
        Barrier, BatchNormalization, OptimizedRNNStack, RandomDistribution,
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
        // Conditions:
        //   All input items must be consecutive in memory.
        //   All output shapes must match.
        //   (The input shapes do not need to match. It is already guaranteed that they are all reshapable into the same output shape.)
        // 
        // If all items are already consecutive in memory, then we just slice-view and reshape it, and generate the individual lazy slices into that.
        // (TODO: check that gather does not fail if input shapes do not fullt match.)
        // This way the reshaped outputs will still be recognizable as consecutive by the next op.
        // If they are not consecutive, then don't gather them, but instead reshape them individually (i.e. don't consider them batchable).
        // This is done because reshaping does not involve the GPU, only meta information, and since they are not consecutive,
        // reshaping them individually will not prevent a consecutive set from being recognized.
        // 
        // TODO: Can this be merged with No-ops? Can it be see-through?

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
        // When batching, all inputs must be padded with singleton dimensions to have the same
        // rank, so that the batching/stacking dimension is at a matching position.

        // Matrix products
        // ---------------
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
        { OpSpecificConditionKind::Convolution, {
            PrimitiveOpType::Convolution // includes ConvolutionTranspose()
        }},
        { OpSpecificConditionKind::Pooling, {
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
        // Slice() is an actual view, not a copy. Hence, we do not batch it.
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
        // Conditions (batching)):
        //   WRONG
        //   All inputs must have matching shapes, respectively (this is like an N-nary elementwise op).
        // 
        // Conditions (stacking):
        //   WRONG
        //   Must not splice along stacking dimension.
        //   Stacking dimension must have matching shapes across items.

        // TransposeAxes()
        // ---------------
        // 
        { OpSpecificConditionKind::Transpose, {
            PrimitiveOpType::TransposeAxes
        }},
        // 
        // Conditions (batching)):
        //   All inputs must have matching shapes.
        // 
        // Conditions (stacking):
        //   All inputs must have matching shapes except for last axis.
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
        // TODO: Does this need to hold on to a RNG state? How to do that?

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
            PrimitiveOpType::Combine, PrimitiveOpType::Block, PrimitiveOpType::Assign
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

// ===========================================================================
// helper classes
// ===========================================================================

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

class NDArrayViewArena
{
    // allocate a new tensor in a large arena
    static NDArrayViewPtr s_currentArena;
    static size_t s_currentArenaUsed;
    static const size_t ARENASIZE = 64000000; // we allocate in this chunk size
public:
    // allocate an NDArrayView of a given shape, data type, and device
    // The returned memory region is a slice into a much larger NDArrayView; therefore,
    // this operation short-circuits CUDA and is very fast.
    // Sparse objects cannot be arena-allocated. Which is fine, since they are inputs or
    // gradients (of embeddings) that can be kept around across minibatches, and thus not part of batched computation.
    NDArrayViewPtr NewNDArrayView(const NDShape& shape, const DataType& dataType, StorageFormat format, const DeviceDescriptor& device)
    {
        let numElements = shape.TotalSize();
        // if too large, or sparse, then plain alloc
        if (numElements > ARENASIZE || format != StorageFormat::Dense)
            return make_shared<NDArrayView>(dataType, format, shape, device);
        // If arena not large enough then waste its remainder and just allocate a fresh one.
        // This abandons the current m_arena. This will not cause a memory leak, however:
        // Since the slices into it that were returned before all hold a ref-count to that arena,
        // it will be deallocated automatically as soon the last slice goes away.
        // If the data type is different, we drop the current arena. We can't presently mix data types, so this is OK.
        if (!s_currentArena || numElements > (ARENASIZE - s_currentArenaUsed) ||
            dataType != s_currentArena->GetDataType() || device != s_currentArena->Device())
        {
            s_currentArena = make_shared<NDArrayView>(dataType, StorageFormat::Dense, NDShape{ ARENASIZE }, device);
            s_currentArenaUsed = 0;
        }
        vector<size_t> startOffset{ s_currentArenaUsed };
        vector<size_t> extent{ numElements };
        //NDArrayViewPtr region = s_currentArena->SliceView(startOffset, extent); // SliceView() adjusts the MatrixView
        NDArrayViewPtr region = s_currentArena->Slice(startOffset, extent);  // BUGBUG: fails in DistributedLearner
        s_currentArenaUsed += numElements;
        if (region->Shape() == shape)
            return region;
        else
            return region->AsShape(shape);
    }
};

/*static*/ NDArrayViewPtr NDArrayViewArena::s_currentArena;
/*static*/ size_t NDArrayViewArena::s_currentArenaUsed = 0;

// ---------------------------------------------------------------------------
// RuntimeStatistics -- helper class for collecting runtime statistics, for
// diagnostics and debugging purposes
// ---------------------------------------------------------------------------

struct RuntimeStatistics
{
    // forward
    size_t numOpNodes = 0;
    size_t numLeafNodes = 0;
    size_t numDoneSpliceOps = 0;
    size_t numDoneFreeOps = 0;
    size_t numDoneOtherOps = 0;
    // backward
    size_t numBackpropsToInputs = 0;
    size_t numBackpropGathers = 0;
    size_t numBackpropScatters = 0;
    size_t numBatchedBackpropToCalls = 0;
};

// ---------------------------------------------------------------------------
// NonOwningFunctionList, NonOwningFunctionListBuilder -- helper classes:
// linked list over PrimitiveFunction objects, using m_link.
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
        PrimitiveFunction& operator*() const { return *iter; } // TODO: This is weird, figure this out
        PrimitiveFunction* operator++() { iter = iter->m_link; return iter; }
        bool operator!=(const FunctionListIterator& other) { return iter != other.iter; }
    };
    FunctionListIterator begin() const { return front(); }
    FunctionListIterator end()   const { return nullptr; }
};
class NonOwningFunctionListBuilder : public NonOwningFunctionList // over PrimitiveFunction, using m_link
{
    PrimitiveFunction* tail; // note: value undefined when list empty
public:
    NonOwningFunctionListBuilder() : NonOwningFunctionList() { }
    NonOwningFunctionListBuilder(PrimitiveFunction* f) : NonOwningFunctionList(f), tail(f) { f->m_link = nullptr; }
    void push_back(PrimitiveFunction* f)
    {
        if (!head)
            head = f;
        else
            tail->m_link = f;
        tail = f;
        count++;
        f->m_link = nullptr;
    }
};

// ---------------------------------------------------------------------------
// VisitorTag -- helper for graph traversal
//  - call VisitorTag::Begin()
//  - in traversal: if (VisitorTag::Visited(node.m_visitedTag)) return;
// This does not nest!
// ---------------------------------------------------------------------------

class VisitorTag
{
    static size_t s_nextVisitTag; // unique id for a single non-nested visiting process
    size_t m_visitTag;
public:
    void Begin() // call this at start
    {
        // TODO: to make this thread-safe, use atomic increment
        m_visitTag = s_nextVisitTag++;
    }
    bool Visited(size_t& tag)
    {
        if (tag == m_visitTag)
            return true;
        tag = m_visitTag;
        return false;
    }
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
        fail_if(IsAlias(op), "IsViewOp should never be asked about a no-op, should be short-circuited before");
        return
            op == PrimitiveOpType::StopGradient ||
            op == PrimitiveOpType::Pass         ||
            op == PrimitiveOpType::NoOp         ||
            op == PrimitiveOpType::BarrierOp    ||
            op == PrimitiveOpType::Reshape      ||
            op == PrimitiveOpType::Slice;
    }

    // predicate whether an op just passes through its input
    // This is used to decide whether we can short-circuit it in SeeThroughNoOps().
    static bool IsAlias(PrimitiveOpType op)
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

    // helper to test whether a FunctionPtr is a barrier operation
    template <typename FunctionPtr> // PrimitiveFunctionPtr or PrimitiveFunction*
    static bool IsBarrier(const FunctionPtr& f)
    {
        return f->m_op == PrimitiveOpType::BarrierOp && f->m_attributes.Size() > 0;
    }

    // see through no-ops, such as barrier, Pass, or StopGradient
    // Use this for ANY access to PrimitiveFunction::m_inputs EXCEPT not needed (and possibly wrong one day) when directly getting the shape.
    // This function also determines the top-most barrier id that this input may depend on.
    static const Variable& SeeThroughNoOps(const vector<Variable>& inputs, size_t index, size_t& topBarrierId)
    {
        let& input = inputs[index];
        let& fields = *input.m_dataFields;
        // lazy index: not an alias
        if (fields.m_lazyIndex.first)
            return input;
        // does not have an owner: not an alias
        let f = fields.Owner();
        if (!f)
            return input;
        // op is not an alias
        if (!IsAlias(f->m_op))
            return input;
        // it is an alias (including barrier): register barrier id and see right through
        if (topBarrierId == SIZE_MAX && IsBarrier(f))
            topBarrierId = f->m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>();
        return SeeThroughNoOps(f->m_inputs, 0, topBarrierId); // (all aliases are unary functions)
    }
    static const Variable& SeeThroughNoOps(const vector<Variable>& inputs, size_t index)
    {
        size_t barrierId = 0; // (not passing SIZE_MAX will short-circuit the test)
        return SeeThroughNoOps(inputs, index, barrierId);
    }

    // helper to check whether we should profile this function execution
    set<DynamicProfilerPtr> m_profilersUsed; // all used profilers will be registered for a given batched execution
    bool ShouldProfile(const PrimitiveFunction* f)
    {
#ifdef LOG_DETAILS
        f;
        let should = true;
#else
        let should = f->m_profiler && f->m_profiler->Verbosity() > 0;
#endif
        if (should)
            m_profilersUsed.insert(f->m_profiler);
        return should;
    }

    // class to manage the set of ready operations (the schedule)
    class ReadyOps
    {
        NonOwningFunctionListBuilder m_viewOps;
        vector<NonOwningFunctionListBuilder> m_regularOps; // m_regularOps[] is a linked list
        NonOwningFunctionListBuilder m_barrierOps; // TODO: currently dead
        vector<size_t> m_barrierPendingCounts;  // [barrier id] number of consumers of a barrier id that are not yet ready
        vector<size_t> m_bnPendingCounts;       // [bn id] number of pending (non-ready) BatchNormalization operations
        // TODO: This must be turned into something hashable.
        // test whether two PrimitiveFunctions can be executed as a single batched operation
        static bool AreBatchable(const PrimitiveFunction* a, const PrimitiveFunction* b)
        {
            // first it must be the same operation
            let op = a->m_op;
            let opClass = g_oscTable[op]; // operation-specific auto-batching class
            // free ops always get batched; even if they have different op-codes
            if (IsViewOp(op) && !IsBarrier(a))
                LogicError("should not get here for view ops or barrier ops");
            // op codes must match
            if (op != b->m_op)
                return false;
            // priority must match (depending on barrier or not)
            if (a->m_priority != b->m_priority)
                return false;
            // some operations have variable number of arguments. Those cannot be batched, e.g. Splice().
            if (a->m_inputs.size() != b->m_inputs.size())
                return false;
            // special case BatchNormalization
            if (op == PrimitiveOpType::BatchNormalization)
            {
                let aId = a->m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>();
                let bId = b->m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>();
                if (aId != bId)
                    return false;
                // shape of first argument and object identities of all other arguments must match, otherwise it's an error
                if (a->m_inputs[0].Shape() != b->m_inputs[0].Shape())
                    InvalidArgument("Primitive op '%S' encountered two instances of the same id %d with different shapes %S and %S.",
                        PrimitiveOpTypeName(op).c_str(), (int)aId, a->m_inputs[0].Shape().AsString().c_str(), b->m_inputs[0].Shape().AsString().c_str());
                for (size_t i = 1; i < 6; i++)
                {
                    if (a->m_inputs[i].m_dataFields != b->m_inputs[i].m_dataFields)
                        InvalidArgument("Primitive op '%S' encountered two instances of the same id %d with different %d-th argument.",
                            PrimitiveOpTypeName(op).c_str(), (int)aId, (int)i);
                }
                return true; // these *must* be batched
            }
            // all input dimensions must match (with exception of a few special cases)
            for (size_t i = 0; i < a->m_inputs.size(); i++)
            {
                // see through no-ops for testing the input
                size_t aBarrierId = SIZE_MAX;
                size_t bBarrierId = SIZE_MAX;
                /*let& aInput =*/ SeeThroughNoOps(a->m_inputs, i, aBarrierId);
                /*let& bInput =*/ SeeThroughNoOps(b->m_inputs, i, bBarrierId);
                // barrier id, if any, must match
                if (aBarrierId != bBarrierId)
                    return false;
                // there are a few special cases
                if (opClass == OpSpecificConditionKind::MatrixProduct && i == 0)
                {
                    // for Times, the first arg must be the same object, not just the same shape
                    // TODO: a special case is a dot product, which we can write as ReduceSum(ElementTimes(a,b))
                    //       This would require to rewrite the graph though; can we do that?
                    if (a->m_inputs[i].m_dataFields != b->m_inputs[i].m_dataFields)
                        return false;
                }
                else
                {
                    // shapes must match (we don't see through no-ops since the target shape is the right one to test)
                    if (a->m_inputs[i].Shape() != b->m_inputs[i].Shape())
                        return false;
                }
            }
            // attributes must also match
            if (a->m_attributes != b->m_attributes)
                return false;
            // all match: we can batch
            return true;
        }
    public:
        // count an occurrence of a barrier with a given id
        void CountBarrier(size_t barrierId)
        {
            if (barrierId == SIZE_MAX)
                return;
            if (barrierId >= m_barrierPendingCounts.size())
                m_barrierPendingCounts.resize(barrierId * 10, 0);
            m_barrierPendingCounts[barrierId]++;
        }
        int BarrierPendingCounts(size_t barrierId) const { return barrierId != SIZE_MAX ? (int)m_barrierPendingCounts[barrierId]: -1; } // this is used for logging
        // count an occurrence of a BatchNormalization with a given id
        void CountBatchNorm(size_t bnId)
        {
            if (bnId >= m_bnPendingCounts.size())
                m_bnPendingCounts.resize(bnId * 10, 0);
            m_bnPendingCounts[bnId]++;
        }
        // schedule an operation that has been confirmed ready
        void Schedule(PrimitiveFunction* f)
        {
            let op = f->m_op;
            // special case BatchNormalization: we must account for all occurences
            if (op == PrimitiveOpType::BatchNormalization)
            {
                let bnId = f->m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>();
                fail_if(m_bnPendingCounts[bnId] == 0, "m_bnPendingCounts decreased more than increased??");
                m_bnPendingCounts[bnId]--; // only those with pending count 0 are ready
            }
            // we manage three ready sets, since two common kinds are very simple
            if (IsBarrier(f))
                // BUGBUG: We never get here since we now see through barriers for efficiency...
                LogicError("m_barrierOps.push_back(f) should no longer be done"); // m_barrierOps.push_back(f);
            else if (IsViewOp(op))
                m_viewOps.push_back(f);
            else
            {
                // determine the priority. This is for Barriers after short-circuiting NoOps...
                // This is highly inefficient, always reaching through the Owner pointer. Needs a flag.
                // TODO: Once we have fixed the barrier, this goes away.
                int pri = 0; // normal priority
                let& inputs = f->m_inputs;
                for (size_t i = 0; i < inputs.size(); i++)
                {
                    size_t barrierId = SIZE_MAX;
                    SeeThroughNoOps(inputs, i, barrierId);
                    if (barrierId != SIZE_MAX)
                    {
                        fail_if(m_barrierPendingCounts[barrierId] == 0, "barrierPendingCounts decreased more than increased??");
                        m_barrierPendingCounts[barrierId]--;
                    }
                    let& input = inputs[i];
                    if (input.IsOutput() && IsBarrier(input.OutputOwner()))
                    {
                        pri = -1; // lower priority: Can only execute when anything else of normal priority (not depending on a barrier) is gone
                        break;
                    }
                }
                f->m_priority = pri;
                // this naive implementation just scans linearly
                // scan through all op sets to see if one is batchable with 'f'
                // So far this does not show up in profiling.
                for (auto iter = m_regularOps.begin(); iter != m_regularOps.end(); iter++)
                {
                    if (AreBatchable(f, iter->front()))
                    {
                        iter->push_back(f);
                        return;
                    }
                }
                // none fit: open a new set
                m_regularOps.push_back(NonOwningFunctionListBuilder(f));
            }
        }
        // notify a function that an input has become available; schedule it when all inputs are now available
        void NotifyInputAvailable(PrimitiveFunction* f)
        {
            if (f->m_pendingInputs <= 0)
                LogicError("NotifyInputAvailable: pending inputs already 0 yet we are executing it");
            f->m_pendingInputs--;
            // if it is now ready then schedule it
            if (f->m_pendingInputs == 0)
                Schedule(f);
        }
        // test if no more ready ops
        bool empty() const { return m_viewOps.empty() && m_regularOps.empty() && m_barrierOps.empty(); }
        size_t size() const { return (m_viewOps.size() > 0) +  + (m_barrierOps.size() > 0); }
        size_t numBatchableOpsPending() const { return m_regularOps.size(); }
        // helper to determine how many barrier ops are unfulfilled
        template <typename IteratorType>
        int GetBarrierGap(const IteratorType& iter)
        {
            let& f = iter->front();
            let batchSize = (int)iter->size();
            let& inputs = f->m_inputs;
            int gap = 0;
            // TODO: This is highly inefficient; we should remember somewhere whether a function depends on a barrier
            for (size_t i = 0; i < inputs.size(); i++)
            {
                size_t barrierId = SIZE_MAX;
                SeeThroughNoOps(inputs, i, barrierId); // TODO: this is inefficient; better have a second version that stops once it found the first barrier
                if (barrierId == SIZE_MAX)
                    continue;
                let thisGap = (int)m_barrierPendingCounts[barrierId]; // how many outstanding (not ready) barrier consumers do we have?
                if (thisGap > gap)
                    gap = thisGap; // determine the largest gap
            }
            return gap;
        }
        // helper to check whether this is a BatchNormalization op that still has some instances pending
        int GetBatchNormPending(const vector<NonOwningFunctionListBuilder>::const_iterator& iter)
        {
            let& f = iter->front();
            if (f->m_op != PrimitiveOpType::BatchNormalization)
                return 0;
            else
                return m_bnPendingCounts[f->m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>()] > 0;
        }
        // select the next batched op to execute
        NonOwningFunctionList pop_best()
        {
            // try all three queues, in priority order
            if (!m_viewOps.empty()) // view ops always go first, since they are free
                return move(m_viewOps);
            else if (!m_regularOps.empty()) // regular ops
            {
                auto best = m_regularOps.begin();
                for (auto iter = best + 1; iter != m_regularOps.end(); iter++)
                {
                    // barrier is realized through priority
                    int diff = 0;
                    if (diff == 0)
                        diff = -(GetBarrierGap(iter) - GetBarrierGap(best)); // lower gap is better
                    if (diff == 0)
                        diff = -(GetBatchNormPending(iter) - GetBatchNormPending(best)); // BatchNormalization with pending inputs always loses
                    if (diff == 0)
                        diff = iter->front()->m_priority - best->front()->m_priority;
                    if (diff == 0)
                        diff = (int)iter->size() - (int)best->size();
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
            else
                return move(m_barrierOps); // barriers only get returned when no other op is available
        }
    };
    ReadyOps m_schedule;

    // recursively traverse the tree hanging off a Variable and
    //  - prepare all nodes for batched execution
    //  - schedule all ready operations
    // TODO: Once we are in the main build, change all Function to PrimitiveFunction directly.
    // TODO: What to do with multi-valued functions? Which ones are there? What is Combine(), a barrier?
    // This function assumes that
    //  - it only runs once
    //  - m_value must not have been set (don't call this function if it has)
    //  - m_pendingInputs has been initialized to -1 by the constructor
    // Caller must call m_visitorTag.Begin() first.
    void RInitForScheduling(const Variable& var)
    {
        auto& fields = *var.m_dataFields;
        // return if already visit
        if (m_visitorTag.Visited(fields.m_visitedTag))
            return;
        // some sanity checks
        if (fields.m_value)
            LogicError("RInitForScheduling() should not have been called on variables that already have a value.");
        if (fields.m_varKind == VariableKind::Input || fields.m_varKind == VariableKind::Placeholder)
            LogicError("Value() depends on Input or Placeholder, it is not knowable.");
        // initialize m_consumers chain
        fields.m_consumers.first.first = nullptr;
        fields.m_consumers.second.clear();
        // handle leaves
        if (fields.m_varKind == VariableKind::Parameter || fields.m_varKind == VariableKind::Constant)
        {
            if (!fields.m_value)
                var.Value(); // this initializes it
            if (!fields.m_value)
                LogicError("Parameter/Constant has no Value??");
            m_stats.numLeafNodes++;
            return;
        }
        // not a leaf
        auto& f = *fields.m_ownerFunction.lock();
        // special case BatchNormalization: we must account for all occurences before normalizing
        if (f.m_op == PrimitiveOpType::BatchNormalization)
        {
            if (!f.m_attributes.Contains(PrimitiveFunction::AttributeNameSyncId))
                InvalidArgument("Primitive op '%S' requires an id parameter. Please use the version that takes an id.",
                    PrimitiveOpTypeName(f.m_op).c_str());
            let bnId = f.m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>();
            m_schedule.CountBatchNorm(bnId);
        }
        // determine how many inputs are pending; and also recurse and set up the consumer list
        size_t pendingInputs = 0;
        let& inputs = f.m_inputs;
        for (size_t i = 0; i < inputs.size(); i++)
        {
            size_t barrierId = SIZE_MAX;
            let& input = SeeThroughNoOps(inputs, i, barrierId);
            m_schedule.CountBarrier(barrierId);
            auto& fields = *input.m_dataFields;
            // recursively traverse
            if (!fields.m_value)
            {
                RInitForScheduling(input);
                if (!fields.m_value) // (in case of a Parameter, we now may have a value)
                {
                    pendingInputs++;
                    // record ourselves as a consumer of the input
                    // Note that RInitForScheduling() will have reset this upon first visit of 'input'.
                    if (!fields.m_consumers.first.first) // optimized for main case of 1 consumer. No std::vector in that case.
                        fields.m_consumers.first = make_pair(&f, i); // note: we don't need i for forward; can optimize
                    else
                        fields.m_consumers.second.push_back(make_pair(&f, i));
                }
            }
            else
                m_stats.numLeafNodes++;
        }
        f.m_pendingInputs = (int)pendingInputs;
        // if none then operation is ready
        if (pendingInputs == 0)
            m_schedule.Schedule(&f); // add to ready set
        m_stats.numOpNodes++;
    }

    // return the m_value field of a variable, but possibly realizing it lazily if it is an index operation
    // TODO: Generalize this to allow implicit reshape to target. Then this can be use for slice, reshape, and slice>>reshape.
    static const NDArrayViewPtr& LazilyIndexedValue(const Variable& v)
    {
        auto& fields = *v.m_dataFields;
        if (fields.m_value)
            return fields.m_value;
        fail_if(!fields.m_lazyIndex.first, "variable unexpectedly has no value yet, nor is it a slice view into a batched op");
        // the PrimitiveFunction does not own its output, it is a slice view into another
        let& from = LazilyIndexedValue(fields.m_lazyIndex.first->m_outputs[0]);
        let index = fields.m_lazyIndex.second;
        // TODO: Allow an implicit Reshape() here, so that we can use the same mechanism for slice and reshape.
        if (index == SIZE_MAX) // special sentinel value that means "don't slice, actually"
            fields.m_value = from;
        else
            fields.m_value = from->IndexLastAxis(index);
        fail_if(fields.m_shape != fields.m_value->Shape(), "variable shape different from its value??");
        return fields.m_value;
    }

    static void LogFunction(const PrimitiveFunction& f, const wchar_t* prefix = L"", size_t markIndex = SIZE_MAX)
    {
        let& inputs = f.m_inputs;
        let& output = f.m_outputs[0]; // BUGBUG: How to deal with multi-valued functions?
        let& outputShape = output.Shape();
        auto uid = f.Uid();
        let& name = f.Name();
        if (!name.empty())
            uid = name + L":" + uid;
        if (prefix && *prefix)
            fprintf(stderr, "[%S]  ", prefix);
        fprintf(stderr, "%S%S = %S (", uid.c_str(), outputShape.AsString().c_str(), f.OpName().c_str());
        for (size_t i = 0; i < inputs.size(); i++)
        {
            let& input = SeeThroughNoOps(inputs, i);
            let& fields = *input.m_dataFields;
            // little helper function to fix up variable names by removing _Output_0
            // TODO: Once we support >1 output, this needs a bit more code.
            let GetVarName = [](const Variable& input) -> wstring
            {
                auto uid = input.Uid();
                if (uid.size() > 9 && wcscmp(uid.c_str() + uid.size() - 9, L"_Output_0") == 0)
                    uid.resize(uid.size() - 9);
                let& name = input.IsOutput() ? input.Owner()->Name() : input.Name();
                if (!name.empty())
                    uid = name + L":" + uid;
                return uid;
            };
            if (fields.m_lazyIndex.first)
            {
                let& input1 = fields.m_lazyIndex.first->m_outputs[0];
                fprintf(stderr, "%s%s%S%S[%d]", (i == 0) ? "" : ", ", (i == markIndex) ? "=>" : "", GetVarName(input1).c_str(), input1.Shape().AsString().c_str(), (int)fields.m_lazyIndex.second);
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

    // for memoization statistics
    class PCTimer // roll our own; high_resolution_timer is reportedly not high-resolution (0.1 us)
    {
        LARGE_INTEGER freq, start;
        double total;
    public:
        PCTimer() { if (!QueryPerformanceFrequency(&freq)) RuntimeError("auto_timer: QueryPerformanceFrequency failure"); } // count ticks per second
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
        bool hasSparse = false;
        size_t numInvocations = 0;
        size_t totalElements = 0;  // sum of all output elements, for a rough indication of utilization
        PCTimer timerLaunch;
        PCTimer timerRun;
        PCTimer timerSync; // measure a dummy sync
    };
    vector<CudaStats> cudaStats;

    // compute the value of 'f', storing it in the arena (unless 'isFree', which must be set when there is nothing to store)
    const Variable& MemoizeKnowableValueInArena(PrimitiveFunction& f, bool isFree = false)
    {
        if (f.m_outputs.size() != 1)
            LogicError("MemoizeKnowableValueInArena: only functions with 1 output are supported");
        // fetch the NDArrayViewPtrs for all inputs
        let& inputs = f.m_inputs;
        auto& inputValues = BorrowBuffer(m_inputValuesBuffer, inputs.size());
        for (size_t i = 0; i < inputs.size(); i++)
            inputValues[i] = LazilyIndexedValue(SeeThroughNoOps(inputs, i)); // (if this is a lazy slice, then now we must resolve it)\
        // allocate the output NDArrayViewPtr in the arena
        let& output = f.m_outputs[0]; // BUGBUG: How to deal with multi-valued functions?
        let& outputShape = output.Shape();
        // logging
        if (ShouldProfile(&f))
            LogFunction(f, f.m_profiler);
        //if (f.m_op == PrimitiveOpType::ElementTimes)
        //    LogFunction(f, L"[bf]  ");
        auto outValue = isFree
            ? nullptr
            : m_arena.NewNDArrayView(outputShape, output.GetDataType(), output.IsSparse() ? StorageFormat::SparseCSC : StorageFormat::Dense, inputValues[0]->Device());
        CudaStats* cudaStatsPtr = nullptr;
        if (logMemoizeStatsCounter == 0)
        {
            cudaStats.resize(2 * (size_t)PrimitiveOpType::UnknownOP);
            let hasSparse = any_of(inputs.begin(), inputs.end(), [](const Variable& v) { return v.IsSparse(); });
            cudaStatsPtr = &cudaStats[(size_t)f.m_op + (hasSparse ? 1 : 0)];
            cudaStatsPtr->op = f.m_op; // (really only needed the first time)
            cudaStatsPtr->hasSparse = hasSparse;
            cudaStatsPtr->numInvocations++;
            cudaStatsPtr->totalElements += outputShape.TotalSize();
            NDArrayView::Sync(inputValues[0]->Device());
            cudaStatsPtr->timerLaunch.Start();
        }
        // execute it
        output.m_dataFields->m_value = move(PrimitiveFunction::ComputeKnowableValue(f.m_op, inputValues, f.Attributes(), outputShape, move(outValue), f));
        if (logMemoizeStatsCounter == 0)
        {
            cudaStatsPtr->timerLaunch.Stop();
            cudaStatsPtr->timerRun.Start();
            NDArrayView::Sync(inputValues[0]->Device());
            cudaStatsPtr->timerRun.Stop();
            cudaStatsPtr->timerSync.Start(); // measure the overhead of just syncing the GPU
            NDArrayView::Sync(inputValues[0]->Device());
            cudaStatsPtr->timerSync.Stop();
        }
#if 0   // run multiple times to get a feel for the runtime cost
        for (size_t i = 1; i < 5; i++)
        {
            outValue =
                  isFree ? nullptr
                //: !output.IsSparse() ? output.m_dataFields->m_value;
                : m_arena.NewNDArrayView(outputShape, output.GetDataType(), output.IsSparse() ? StorageFormat::SparseCSC : StorageFormat::Dense, inputValues[0]->Device());
            output.m_dataFields->m_value = move(PrimitiveFunction::ComputeKnowableValue(f.m_op, inputValues, f.Attributes(), outputShape, move(outValue), f));
        }
#endif
        // stats
        let primitiveOp = f.m_op;
        if (isFree) // means we did not pass a data buffer for the result; any one we pass a buffer does actual work
            m_stats.numDoneFreeOps++;
        else if (primitiveOp == PrimitiveOpType::Splice)
            m_stats.numDoneSpliceOps++;
        else
            m_stats.numDoneOtherOps++;
        return output;
    }

    static void ResetPendingToIdle(PrimitiveFunction& f)
    {
        if (f.m_pendingInputs != 0)
            LogicError("ResetPendingToIdle: pendingINputs is not 0, so we should not have gotten here");
        f.m_pendingInputs = -1; // unknown
    }

    // temp variables for ExecuteBatchedOpAndUpdateSchedule(); keep outside to reuse the memory allocation
    vector<Variable> m_batchedInputs;
    vector<Variable> m_spliceArgsBuffer;
    size_t m_numBatchedLaunches = 0; // (for statistics only)

    // batch-execute a set of ops that are known to be batchable
    // For every batched operation, this generates a new PrimitiveFunction object for the op itself, and one
    // for a splice operation for each batched inputs.
    // I.e. this is not a full graph transform, but rather a graph augmentation, so that during backprop,
    // we can recover the batched operations, while the original graph does not get modified.
    // Any batched operation will generate its result in a dense tensor with a batch dimension.
    // The consumers of the original ops will get a back-reference in the m_lazyIndex field.
    // If such a result is ever accessed individually, it will lead to a lazy NDArrayView::SliceView() call
    // (but no Splice Function object is used for this).
    // All ops passed to this function must get their m_pendingInputs changed from 0 to -1 (newly created batched ones also will have -1).
    void ExecuteBatchedOpAndUpdateSchedule(NonOwningFunctionList ops) // (note: NonOwningFunctionListBuilder is so small that it is best copied)
    {
        // TODO: need to handle ops that have >1 output, such as Combine(). Just don't batch them ever? Combine() is just a see-through anyway.
        // get a representative op
        auto& f0 = *ops.front();
        let op = f0.m_op;
        let batchSize = ops.size();
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
        let isFree = IsViewOp(op);
        let isTimes       = opClass == OpSpecificConditionKind::MatrixProduct; // is special-cased
        let isElementWise = opClass != OpSpecificConditionKind::MatrixProduct && opClass != OpSpecificConditionKind::Convolution;

        // perform the op
        if (!isFree)
            m_numBatchedLaunches++;
        let numArgs = f0.m_inputs.size();
#ifdef NO_BATCHED_FORWARD
        auto doNaively = true;
#else
        let doNaively =
            isFree ||
            op == PrimitiveOpType::Splice ||
            batchSize == 1;
#endif
        //fprintf(stderr, "%d %sexecuting %d instances of %S -> %S; %d batchable ops pending\n",
        //        isFree ? -1 : (int)m_numBatchedLaunches,
        //        doNaively ? "" : "batch-",
        //        (int)batchSize, f0.OpName().c_str(), f0.m_outputs[0].Shape().AsString().c_str(),
        //        (int)m_schedule.numBatchableOpsPending());
        if (doNaively)
        {
            // for correctness testing of underlying mechanism, compute them without actual batching
            for (auto op = ops.begin(); op != ops.end(); ++op)
            {
                // execute it
                MemoizeKnowableValueInArena(*op, isFree);
                // reset state
                ResetPendingToIdle(*op);
                // TODO: realize splice ops that are index ops as a m_lazyIndex at this point
                //if (f0.m_op == PrimitiveOpType::Slice)
                //{
                //    // inject the lazyIndex
                //}
#if 0           // test effect of the unbatched sparse times
                if (f0.m_op == PrimitiveOpType::Times)
                    op->ComputeKnowableValue(op->m_op, m_inputValuesBuffer, op->Attributes(), op->m_outputs[0].Shape(), move(out));
#endif
            }
        }
        // execute the batchable operations as a batch
        // Every resulting batched op consists of the following new operations:
        //  - a Splice() or Slice() for each input (e.g. 2 for a binary op)
        //  - a PrimitiveFunction that is the op itself
        //  - m_lazyIndex entries that represent a "virtual" Slice() that is never created as a PrimitiveFunction object to saved mallocs.
        // As for resource management, m_lazyIndex will hold a strong ref to the PrimitiveFunction;
        // and we will hack its SeeThroughNoOps(m_inputs,i).m_outputComposite to hold a strong reference to the Splice() or Slice().
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
            //  - For global pooling, we need to watch out. TODO!
            m_batchedInputs.resize(numArgs);
            size_t batchAxis = 0;
            size_t i0 = isTimes ? 1 : 0;
            for (size_t i = i0; i < numArgs; i++)
            {
                let rank = f0.m_inputs[i].Shape().Rank();
                if (rank > batchAxis)
                    batchAxis = rank;
            }
            fail_if(isTimes && (numArgs != 2 || batchAxis != f0.m_inputs.back().Shape().Rank()), "batchAxis incorrect for isTimes"); // the loop above also works for the isTimes case
            if (opClass == OpSpecificConditionKind::Splice)
                batchAxis++; // Splice may create a new axis at the end; so just add one from the start
            // determine the position of the resulting batch axis
            let& unbatchedOutputShape = f0.m_outputs[0].Shape();
            let outputBatchAxis = isElementWise ? batchAxis : unbatchedOutputShape.Rank();

            // create all the batched inputs by splicing along the batch axis
            // Special optimizations are taken if all elements are identical.
            bool anyBatchedInputs = false;
            if (i0 == 1) // Times(): matrix must be identical
                m_batchedInputs[0] = SeeThroughNoOps(f0.m_inputs, 0);
            for (size_t i = i0; i < numArgs; i++)
            {
                // create splice args for this argument
                // allocate buffers to hold the arguments
                auto& spliceInputs = m_spliceArgsBuffer; // TODO rename to gatherInputs and m_gatherArgsBuffer
                fail_if(!spliceInputs.empty(), "spliceInputs was left not empty"); // previous use must have cleared it
                if (spliceInputs.capacity() < batchSize)
                    spliceInputs.reserve(max(batchSize, 2 * spliceInputs.capacity()));
                // optimization: if all args are consecutive slices, then use a slice view instead
                let* pfields0 = SeeThroughNoOps(f0.m_inputs, i).m_dataFields.get();
                let& lazyIndex0 = pfields0->m_lazyIndex; // for 'allConsecutiveSlices' test
                let is0LazyIndex = (bool)lazyIndex0.first;
                // loop over all batched ops
                // BUGBUG: How about NoOp? (used for Barrier) Also Alias and Reshape actually
                //         Seems if we can carry on a bacth, we should run them once; otherwise don't batch.
                bool allSame = true;
                bool allConsecutiveSlices = is0LazyIndex && lazyIndex0.second != SIZE_MAX; // to be consecutive, it must be a slice to start with
                size_t j = 0;
                for (auto op = ops.begin(); op != ops.end(); ++op, j++) // create the batched tensors
                {
                    let& input = SeeThroughNoOps(op->m_inputs, i);
                    let* pfields = input.m_dataFields.get();
                    let& lazyIndex = pfields->m_lazyIndex;
                    // optimization: if all args are the same, then don't batch
                    allSame = allSame &&
                        (pfields == pfields0 ||                 // same
                         (is0LazyIndex && lazyIndex == lazyIndex0)); // or same view  --TODO: could this need to be checked recursively?
                    // optimization: if all args are consecutive slices, then use a slice view instead
                    if (allConsecutiveSlices)
                    {
                        // optimization: if consecutive slices, then recover the original batched tensor
                        allConsecutiveSlices = allConsecutiveSlices &&
                            lazyIndex.first  == lazyIndex0.first    &&
                            lazyIndex.second == lazyIndex0.second + j;
                        // TODO: Per Jon's suggestion, we can be a little loose here. For a variable-length
                        // scenario, we will loose entries in the middle. We can allow to keep a few around
                        // in garbage-in-garbage-out. If, say, there are additional 20% gap values, we just
                        // carry them forward, and ignore them when implanting the result.
                    }
                    // append the input
                    spliceInputs.push_back(input);
                    // note: Variable is just two shared_ptrs, one being NULL; so this is cheap
                    // note: input is a regular Variable with regular ownwership rules (it does not come from inside here)
                }
                // and splice
                if (allSame) // optimized case: all ops share the same operand: no need to batch them
                    // note: we assume strict broadcasting semantics here (if at least one input is actually batched)
                    m_batchedInputs[i] = spliceInputs[0];
                else if (allConsecutiveSlices) // they are consecutive: can short-circuit as a slice view
                {
                    let& from  = lazyIndex0.first;
                    let  begin = lazyIndex0.second;
                    let& output = from->m_outputs[0];
                    fail_if(!output.m_dataFields->m_value, "value not yet available??");
                    let& fromDims = output.Shape().Dimensions();
                    fail_if(fromDims.size() == 0, "slice view into batch has rank 0??");
                    let axis = fromDims.size() - 1;
                    if (begin == 0 && j == fromDims[axis]) // full range: just take it
                        m_batchedInputs[i] = output; // note: graph already has a strong ref to output elsewhere
                    else // sub-range: splice it by taking a slice view on the previously spliced batch
                    {
                        // create a new PrimitiveFunction Slice()
                        vector<size_t> outputShape = fromDims; // determine output shape
                        outputShape[axis] = j;
                        auto additionalProperties = Dictionary(); // create additional arguments
                        additionalProperties[PrimitiveFunction::AttributeNameAxis      ] = Axis((int)axis);
                        additionalProperties[PrimitiveFunction::AttributeNameBeginIndex] = (int)begin;
                        additionalProperties[PrimitiveFunction::AttributeNameEndIndex  ] = (int)(begin + j);
                        let spliceOp = PrimitiveFunction::RawPrimitiveFunction(PrimitiveOpType::Slice, vector<Variable>{ output }, outputShape, move(additionalProperties), f0.m_name);
                        spliceOp->m_profiler = f0.m_profiler;
#ifdef LOG_DETAILS
                        spliceOp->m_uid = L"#" + spliceInputs[0].Uid();
#endif
                        // and execute it
                        let& output = MemoizeKnowableValueInArena(*spliceOp, /*isFree=*/true);
                        // and that's our input to the batched operation
                        //m_batchedInputs[i] = output.CompositePreservingCopy(spliceOp);
                        m_batchedInputs[i] = output;
                        m_batchedInputs[i].m_outputComposite = spliceOp;
                    }
                    // if this op has a higher batchAxis than the re-batched view, we must move the axis
                    // BUGBUG: (perf) Reshape incurs an unnecessary mem copy in Backprop
                    if (axis != batchAxis)
                    {
                        let batchedInput = m_batchedInputs[i];
                        vector<size_t> outputShape = batchedInput.Shape().Dimensions(); // determine current shape
                        outputShape.insert(outputShape.end() - 1, batchAxis - axis, 1);
                        // insert a Reshape() op
                        auto additionalProperties = Dictionary(); // create additional arguments
                        // Reshape() here does not need the properties at this level anymore; output shape is sufficient
                        //additionalProperties[PrimitiveFunction::AttributeNameNewShape]  = NDShape(outputShape);
                        //additionalProperties[PrimitiveFunction::AttributeNameBeginAxis] = Axis((int)0);
                        //additionalProperties[PrimitiveFunction::AttributeNameEndAxis]   = Axis((int)axis);
                        let reshapeOp = PrimitiveFunction::RawPrimitiveFunction(PrimitiveOpType::Reshape, vector<Variable>{ batchedInput }, outputShape, move(additionalProperties), f0.m_name);
                        reshapeOp->m_profiler = f0.m_profiler;
#ifdef LOG_DETAILS
                        reshapeOp->m_uid = L"#," + spliceInputs[0].Uid();
#endif
                        // and execute it
                        let& output = MemoizeKnowableValueInArena(*reshapeOp, /*isFree=*/true);
                        // and that's now really our input to the batched operation
                        //m_batchedInputs[i] = output.CompositePreservingCopy(reshapeOp);
                        m_batchedInputs[i] = output;
                        m_batchedInputs[i].m_outputComposite = reshapeOp;
                    }
                    anyBatchedInputs = true;
                }
                else
                {
                    // create a new PrimitiveFunction Splice()
                    vector<size_t> outputShape; // determine output shape
                    outputShape.reserve(batchAxis + 1);
                    outputShape = LazilyIndexedValue(spliceInputs[0])->Shape().Dimensions();
                    outputShape.resize(batchAxis, 1);           // pad to batchAxis
                    outputShape.push_back(spliceInputs.size()); // and add the batch axis
                    if (outputShape == NDShape{ 512, 36, 108 })
                        BreakPoint;
                    auto additionalProperties = Dictionary();   // create additional arguments
                    additionalProperties[PrimitiveFunction::AttributeNameAxis] = Axis((int)batchAxis);
                    let spliceOp = PrimitiveFunction::RawPrimitiveFunction(PrimitiveOpType::Splice, vector<Variable>(spliceInputs), outputShape, move(additionalProperties), f0.m_name);
                    spliceOp->m_profiler = f0.m_profiler;
#ifdef LOG_DETAILS
                    spliceOp->m_uid = L"#" + spliceInputs[0].Uid();
#endif
                    // and execute it
                    let& output = MemoizeKnowableValueInArena(*spliceOp);
                    // and that's our input to the batched operation
                    // To make sure we hold a reference to this PrimitiveFunction, inject a strong ref to the spliceOp into the copy of its Output.
                    // Note that we abuse the composite field for a non-composite, which works because it is just a FunctionPtr, and we own it.
                    //m_batchedInputs[i] = output.CompositePreservingCopy(spliceOp);
                    m_batchedInputs[i] = output;
                    m_batchedInputs[i].m_outputComposite = spliceOp;
                    anyBatchedInputs = true;
                }
                // release shared_ptrs asap
                spliceInputs.clear();
            }
            // special case: BatchNormalization
            if (op == PrimitiveOpType::BatchNormalization)
            {
                // BatchNorm requires three additional parameters for the current mean and invStdDev, and the zero-mean/unit-variance intermediate. These must be kept for backprop.
                let& statShape = m_batchedInputs[1].Shape(); // note: This is guaranteed to have no batch axis, since they are identical across all instances in this batched op
                m_batchedInputs.push_back(Parameter(m_arena.NewNDArrayView(statShape, m_batchedInputs[0].GetDataType(), StorageFormat::Dense, m_batchedInputs[0].m_dataFields->m_value->Device())));
                m_batchedInputs.push_back(Parameter(m_arena.NewNDArrayView(statShape, m_batchedInputs[0].GetDataType(), StorageFormat::Dense, m_batchedInputs[0].m_dataFields->m_value->Device())));
                m_batchedInputs.push_back(Parameter(m_arena.NewNDArrayView(m_batchedInputs[0].Shape(), m_batchedInputs[0].GetDataType(), StorageFormat::Dense, m_batchedInputs[0].m_dataFields->m_value->Device())));
                anyBatchedInputs = true; // BUGBUG: If all operands are the same, then BatchNorm does not make sense (variance=0). Should we throw an error?
            }
            // execute the operation and implant the results
            // BUGBUG: The newly created PrimitiveFunction objects must get their consumer chain set up.
            PrimitiveFunctionPtr batchedOp;
            Dictionary attributes(f0.Attributes());
            if (anyBatchedInputs)
            {
                // create a new PrimitiveFunction for the batched op
                // This is the actual batched op that we create here.
                // Batched inputs have been prepared in m_batchedInputs[].

#if 0
                // handle the special case of Index()
                // This is presently dead code since we don't batch Slice().
                // It operates on the trailing dimension. We convert it to a Slice() op instead that does not drop the dimension.
                // The additional axis is reshaped away further below.
                if (op == PrimitiveOpType::Slice && attributes.Size() == 1)
                {
                    let axis = unbatchedOutputShape.Rank(); // (this axis is disappeared by the Index() operation)
                    let beginIndex = attributes[PrimitiveFunction::AttributeNameBeginIndex].Value<int>();
                    let endIndex = beginIndex + 1;
                    // inject the two additional parameters. With this parameter set it's a Splice().
                    attributes[PrimitiveFunction::AttributeNameAxis] = Axis((int)axis);
                    attributes[PrimitiveFunction::AttributeNameEndIndex] = (int)(endIndex);
                }
#endif

                let expectedOutputShape = unbatchedOutputShape.AppendAxis(outputBatchAxis, batchSize);
                batchedOp = PrimitiveFunction::RawPrimitiveFunction(f0.m_op, vector<Variable>(m_batchedInputs), expectedOutputShape, move(attributes), f0.m_name);
                batchedOp->m_profiler = f0.m_profiler;
                // Note: We could move(m_batchedInputs), but don't, since then we would have to reallocate m_batchedInputs for the next operation, so makes no difference.
                fail_if(batchedOp->m_outputs[0].Shape().Rank() != outputBatchAxis + 1, "outputBatchAxis was not predicted right");
#ifdef LOG_DETAILS
                batchedOp->m_uid = L"*" + f0.Uid();
#endif
            }
            else
            {
                // all inputs identical: compute it only once
                batchedOp = PrimitiveFunction::RawPrimitiveFunction(f0.m_op, vector<Variable>(f0.m_inputs), f0.m_outputs[0].Shape(), move(attributes), f0.m_name);
                batchedOp->m_profiler = f0.m_profiler;
#ifdef LOG_DETAILS
                batchedOp->m_uid = L"." + f0.Uid();
#endif
                // TODO: the following is a little more efficient, but creates a cycle, so we should exclude the lazy index for the first op
                //batchedOp = f0.shared_from_this();
            }

            // execute it
            // This is the actual batched execution of the op.
            MemoizeKnowableValueInArena(*batchedOp);

            // in case of reducing operations (e.g. ReduceSum() and also Index()), additional singleton axes
            // may have been inserted to align the batch axes. Remove these if present.
            // BUGBUG: (perf) Reshape incurs an unnecessary mem copy in Backprop
            if (anyBatchedInputs)
            {
                if (outputBatchAxis != unbatchedOutputShape.Rank())
                {
                    fail_if(!isElementWise, "output shape should only have additional singleton axes for elementwise operations");
                    // insert a Reshape() op to remove the axes
                    let batchedOutputShape = unbatchedOutputShape.AppendAxis(unbatchedOutputShape.Rank(), batchSize); // desired batched output shape without the singleton axes
                    fail_if(batchedOutputShape.TotalSize() != batchedOp->m_outputs[0].Shape().TotalSize(), "output shape has unexpected axes that should be singletons but aren't");

                    Variable arg = batchedOp->m_outputs[0];
                    arg.m_outputComposite = batchedOp;

                    auto additionalProperties = Dictionary(); // create additional arguments
                    // Reshape() here does not need the properties at this level anymore; output shape is sufficient
                    //additionalProperties[PrimitiveFunction::AttributeNameNewShape]  = NDShape(outputShape);
                    //additionalProperties[PrimitiveFunction::AttributeNameBeginAxis] = Axis((int)0);
                    //additionalProperties[PrimitiveFunction::AttributeNameEndAxis]   = Axis((int)axis);
                    let reshapeOp = PrimitiveFunction::RawPrimitiveFunction(PrimitiveOpType::Reshape, vector<Variable>{ arg }, batchedOutputShape, move(additionalProperties), f0.m_name);
                    reshapeOp->m_profiler = f0.m_profiler;
#ifdef LOG_DETAILS
                    reshapeOp->m_uid = L"*," + arg.Uid();
#endif
                    // and execute it
                    MemoizeKnowableValueInArena(*reshapeOp, /*isFree=*/true);

                    batchedOp = reshapeOp; // this is the result that we redistribute from to the individual consumers
                }
            }

            // implant all results (all as lazy/virtual references through m_lazyIndex)
            size_t j = anyBatchedInputs ? 0 : SIZE_MAX;
            for (auto op = ops.begin(); op != ops.end(); ++op)
            {
                // TODO: review this w.r.t. multi-output functions
                // we remember where we came from for backprop in this case
                auto& fields = *op->m_outputs[0].m_dataFields;
                fields.m_lazyIndex = make_pair(batchedOp, j);
                // semantically, this will compute as fields.m_value = out->IndexLastAxis(j);
                // but it gets deferred to save effort
                if (j != SIZE_MAX) // SIZE_MAX means don't slice
                    j++;
                // TODO: set up batchedOp.m_consumers
                // reset state
                ResetPendingToIdle(*op);
            }

            // release the ref counts on the batched inputs; but keep the vector's memory allocated
            m_batchedInputs.clear();
        }

        // update all ops' consumers and schedule them when possible
        // BUGBUG: Consumer chain here should have been migrated to the batched op; and notifed from there.
        for (auto op = ops.begin(); op != ops.end(); ++op)
        {
            for (let& output : op->m_outputs)
            {
                // notify consumers
                auto& fields = *output.m_dataFields;
                auto& c = fields.m_consumers.first; // first consumer (this is a special optimization to avoid a malloc in case of 1 consumer)
                if (c.first)
                    m_schedule.NotifyInputAvailable(c.first);
                for (auto& c : fields.m_consumers.second) // all other consumers
                    m_schedule.NotifyInputAvailable(c.first);
                // clear consumer list (this operation is done)
                fields.m_consumers.first.first = nullptr;
                fields.m_consumers.second.clear();
            }
        }
    }

public:
    // -----------------------------------------------------------------------
    // BatchedForward() -- entry point for auto-batched implementation of PrimitiveFunction::Value()
    // -----------------------------------------------------------------------

    // Value(), computed with automatic batching
    // This routine uses temporary fields that are assumed initialized in a specific way:
    //  - PrimitiveFunction::m_pendingInputs:
    //     - #inputs that still need to be computed before a node's value can be computed
    //     - also used as a 'visited' flag during traversal
    //     - upon entry and exit of this function, this must be -1 (idle)
    //  - Variable::m_consumers:
    //     - set of consumers of this value. Used to count m_pendingInputs.
    //     - must be empty upon entry and exit
    // plus more temp fields:
    //  - PrimitiveFunction::m_link: pointer to next PrimitiveFunction in the same batchable op
    // And it leaves the following:
    //  - m_value: updated as desired
    //    TODO: values not needed by user or gradient should use scratch space
    //  - m_lazyIndex: if a slice or view came from a batched operation, this points to it
    //     - Any newly created batched ops are referenced this way.
    NDArrayViewPtr BatchedForward(const Variable& v)
    {
        auto& fields = *v.m_dataFields;
        // if value already there then just return it
        if (fields.m_value)
            return fields.m_value;
#ifdef LOG_DETAILS
        Function::PreorderTraverseFunctions(v.OutputOwner(), [&](const FunctionPtr& f) { LogFunction(dynamic_cast<PrimitiveFunction&>(*f), L"[r] "); });
#endif
        // mark all nodes w.r.t. how many inputs they are waiting for before being computable
        //if (!fields.m_value)
        //{
            // prepare and schedule first set
            m_visitorTag.Begin();
            RInitForScheduling(v);
            // compute the entire graph
            while (!m_schedule.empty())
            {
                // select the best amongst the scheduled ops
                auto opBatch = m_schedule.pop_best();
                // log (if barrier crossed)
                let f = opBatch.front();
                if (ShouldProfile(f))
                {
                    let& inputs = f->m_inputs;
                    for (size_t i = 0; i < inputs.size(); i++)
                    {
                        size_t barrierId = SIZE_MAX;
                        SeeThroughNoOps(inputs, i, barrierId);
                        if (barrierId != SIZE_MAX)
                        {
                            let& input = inputs[i]; // we are lazy and only print the name if the barrier is the immediate input, so that we don't have to duplicate the traversal in SeeThroughNoOps()
                            const wchar_t* name = nullptr;
                            size_t id = SIZE_MAX;
                            if (input.IsOutput())
                            {
                                let& f = input.OutputOwner();
                                name = f->Name().c_str();
                                id = f->m_attributes[PrimitiveFunction::AttributeNameSyncId].Value<size_t>();
                            }
                            fprintf(stderr, "\n[%S] --- %d (%S): %d pending\n\n", f->m_profiler->Name(), (int)id, (name && name[0]) ? name : L"?", (int)m_schedule.BarrierPendingCounts(id));
                        }
                    }
                }
                // execute it, and also update all outputs' values and consumers, and the schedule
                ExecuteBatchedOpAndUpdateSchedule(opBatch);
            }
            fail_if(!fields.m_value, "BatchedForward process did not produce a value??");
            // log stats
            if (logMemoizeStatsCounter == 0)
            {
                double totalLaunch = 0;
                double totalExec = 0;
                for (let& s : cudaStats) if (s.numInvocations)
                {
                    let prefix = s.hasSparse ? L"sparse " : L"";
                    let name = PrimitiveOpTypeName(s.op);
                    let execTime = s.timerRun.Total() - s.timerSync.Total();
                    fprintf(stderr, "-> %30S: %7.4f s + %7.4f s = (%9.6f + %9.6f) ms/node * %5d nodes, %9.2f elements/node\n", (prefix + name).c_str(),
                            s.timerLaunch.Total(), execTime,
                            1000.0 * s.timerLaunch.Total() / (double)s.numInvocations, 1000.0 * execTime / (double)s.numInvocations,
                            (int)s.numInvocations,
                            s.totalElements / (double)s.numInvocations);
                    totalLaunch += s.timerLaunch.Total();
                    totalExec   += execTime;
                }
                fprintf(stderr, "-> total launch + exec time: %.4f s + %.4f s\n", totalLaunch, totalExec);
            }
            logMemoizeStatsCounter++;
            if (logMemoizeStatsCounter == logMemoizeStatsPeriod)
                logMemoizeStatsCounter = 0;
        //}
        LazilyIndexedValue(v); // force-flush a potential final lazily-indexed value
#ifdef LOG_STATS
        fprintf(stderr, "BatchedForward: %d forward ops executed besides %d splices and %d views, in nominally %d PrimitiveFunctions on %d known values\n",
                (int)m_stats.numDoneOtherOps, (int)m_stats.numDoneSpliceOps, (int)m_stats.numDoneFreeOps, (int)m_stats.numOpNodes, (int)m_stats.numLeafNodes);
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
        if (f.m_op == PrimitiveOpType::Times && index == 0 && f.m_inputs[1].IsSparse())
            return StorageFormat::SparseBlockCol;
        else
            return StorageFormat::Dense;
    }

    // allocate memory for m_gradient
    // This lazily creates the m_gradient NDArrayView, which may live in a batched op.
    // Returns beta = 0 if gradient was newly created, otherwise 1
    double LazilyCreateLazilyIndexedGradient(const Variable& v, StorageFormat format = StorageFormat::Dense)
    {
        auto& fields = *v.m_dataFields;
        // if gradient exists then return it
        double beta;
        if (fields.m_gradient)
            beta = 1.0;
        else
        {
            // create new gradient
#ifndef NO_BATCHED_BACKPROP
            // if this op draws from a batched op, then the gradient lives in there as well; we return a view onto it
            if (fields.m_lazyIndex.first)
            {
                let& from  = fields.m_lazyIndex.first;
                let  index = fields.m_lazyIndex.second;
                let& fromOutput = from->m_outputs[0];
                beta = LazilyCreateLazilyIndexedGradient(fromOutput);
                let& fromGradient = fromOutput.m_dataFields->m_gradient;
                if (index == SIZE_MAX) // special sentinel value that means "don't slice, actually"
                    fields.m_gradient = fromGradient;
                else // it's a slice: gradient is a slice view into from's output gradient
                {
                    if (beta == 0.0) // gradient is fresh: explicitly reset all (since we are slicing into the input gradientm, we cannot use the beta mechanism)
                    {
                        fromGradient->SetValue(0.0f);
                        beta = 1.0;
                    }
                    fields.m_gradient = fromGradient->IndexLastAxis(index);
                }
            }
            else
#endif
            {
                // create a new one
                // TODO: allocate parameters as separate objects; and allow user to pass buffers in
                fields.m_gradient = m_arena.NewNDArrayView(fields.m_shape, fields.m_dataType, format, fields.m_value->Device());
                beta = 0.0; // has not been initialized (random section in arena)
            }
        }
        return beta;
    }

    // determine the input to backprop a gradient into
    // Normally that is SeeThroughNoOps(f.m_inputs, index).
    // However, some gradients are just copies, so we can see through them.
    // This saves memory and allows easier discovery of optimizable patterns.
    // Note that every single time one iterates over m_inputs for gradients, this function must be used.
    // Note: THIS IS NOT LEVERAGED YET, and could be removed if we don't leverage it.
    const Variable& GetShortCircuitedGradientInput(const PrimitiveFunction& f, size_t index)
    {
        let& input = SeeThroughNoOps(f.m_inputs, index);
#ifndef NO_BATCHED_BACKPROP
        // if the input is a result of a batched operation, then traverse into that instead
        if (input.m_dataFields->m_lazyIndex.first)
            return input.m_dataFields->m_lazyIndex.first->m_outputs[0];
#endif
        return input;
    }

    // recursively traverse the tree hanging off a Variable and build the m_consumer fields
    // This propagates in depth-first order like a naive backprop, but only for the purpose
    // of recording consumers of each node.
    // Unlike forward prop, we...
    //  - can skip any branch that does not need a gradient (!m_needsGradient and StopGradient ops).
    //  - short-circuit into batched ops (m_lazyIndex) so that we backprop through them instead
    // All nodes that were traversed have all input's m_consumers set up and their m_pendingInputs set to 0.
    // Caller must call m_visitorTag.Begin() first.
    void RDetermineConsumersForBackward(const Variable& var)
    {
        auto& fields = *var.m_dataFields;

        if (fields.m_varKind == VariableKind::Parameter || fields.m_varKind == VariableKind::Constant)
            return; // reached a leaf

        fail_if(!fields.m_value, "variable has no value yet??");
        fail_if(!fields.m_needsGradient, "unexpectedly encountered a node with m_needsGradient=false??");
        fail_if(fields.m_varKind == VariableKind::Input || fields.m_varKind == VariableKind::Placeholder, "unexpectedly encountered an Input or a Placeholder??");

        // Determine the function that 'var' is an output of.
        // If 'var' has the m_lazyIndex field set, it means that its value was not
        // actually computed from its true owner function; but rather a slice into the result of
        // a batched operation. In that case, we traverse through that batched operation
        // instead. As a consequence, it is the batched operation that will be recorded as the
        // consumer of all inputs of the batched operation, rather than the original
        // unbatched operation. And as a consequence of that, back propagation will use
        // the same batching that was determined in forward computation.
        auto& outFields = *var.m_dataFields;
#ifndef NO_BATCHED_BACKPROP
        auto&f = outFields.m_lazyIndex.first       // if var was computed via batched op
              ? *outFields.m_lazyIndex.first       // then backprop into the batched op
              : *outFields.m_ownerFunction.lock(); // otherwise traverse the original op
#else
        auto&f = outFields.m_ownerFunction.lock();
#endif
        // return if we already visited the function
        if (m_visitorTag.Visited(f.m_visitedTag))
            return;

        fail_if(f.m_op == PrimitiveOpType::StopGradient, "unexpectedly encountered a StopGradient, which should have propagated m_needsGradient=false upwards");

        // recurse and set up the consumer list
        let& inputs = f.m_inputs;
        for (size_t i = 0; i < inputs.size(); i++)
        {
#if 1
            // Note: Using this function may be misleading.
            // We are not short-circuiting the input here, but rather determine which function
            // it came from.
            let& input = GetShortCircuitedGradientInput(f, i);
#else
            let* inputp = &SeeThroughNoOps(inputs, i);
#ifndef NO_BATCHED_BACKPROP
            // if the input is a result of a batched operation, then traverse into that instead
            if (inputp->m_dataFields->m_lazyIndex.first)
                inputp = &inputp->m_dataFields->m_lazyIndex.first->m_outputs[0];
#endif
            let& input = *inputp;
#endif
            auto& fields = *input.m_dataFields;
            if (!fields.m_needsGradient)
                continue; // skip inputs that receive no gradients
            // this input will receive a gradient; reset it (later, we *accumulate* into it since nodes can receive gradients from multiple consumers)
            // Note that BatchedBackward() returns shared_ptrs to the gradient values, so they won't get lost.
            // BUGBUG: (But they get reallocated over again, and will hold the entire arena!!) BUGBUG!
            // BUGBUG: we must not kill the gradient buffers passed by the user
            fields.m_gradient.reset();
            // record ourselves as a consumer of the input
            // Remember that any input that is a lazyIndex has been redirected to its lazy source,
            // i.e. it is the lazy source that will pull this gradient.
            if (!fields.m_consumers.first.first)
            {
                fields.m_consumers.first = make_pair(&f, i);
                // now process recursively the inputs
                RDetermineConsumersForBackward(input);
            }
            else
                fields.m_consumers.second.push_back(make_pair(&f, i));
            m_stats.numBackpropsToInputs++;
        }
        f.m_pendingInputs = 0; // used as a visited flag
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
            gatherBatchResultDims.push_back(inputValues.size());
        else
            gatherBatchResultDims[axis] = batchDim;
        auto out = m_arena.NewNDArrayView(gatherBatchResultDims, inputValue0.GetDataType(), inputValue0.GetStorageFormat(), inputValue0.Device());
        m_stats.numBackpropGathers++;
        return move(NDArrayView::GatherBatch(inputValues, (int)axis, move(out)));
    }

    // back-propagate f's outputs' m_gradient to a specified input
    // This is the standard path for all unbatched ops.
    // This wraps the PrimitiveFunction's BackpropTo(), interfacing from vectors of Variable to vectors of NDArrayViewPtr.
    // Note that each input that is lazy should redirect into a slice in its lazy source.
    void BackpropToUnbatched(PrimitiveFunction* f, size_t index)
    {
#ifdef LOG_DETAILS
        LogFunction(*f, L"[bb] ", index);
#endif
        let& inputs =  f->m_inputs;
        auto& input = SeeThroughNoOps(inputs, index);
        auto& fields = *input.m_dataFields;
        fail_if(!fields.m_needsGradient, "function unexpectedly does not need a gradient");
        // get the TensorViews for everything we may compute the gradient from
        let& outputs = f->m_outputs;
        fail_if(outputs.size() != 1, "only functions with 1 output are currently supported");
        let& outputFields = *outputs[0].m_dataFields;
#ifndef NO_BATCHED_BACKPROP
        fail_if(outputFields.m_lazyIndex.first, "unexpectedly ran into a function that does not own its output"); // we don't backprop through unbatched ops
#endif
        fail_if(!outputFields.m_value,    "unexpectedly ran into a function that has no m_value yet??");
        fail_if(!outputFields.m_gradient, "unexpectedly ran into a function that has no m_gradient yet??");
        let* outputValue    = outputFields.m_value   .get();
        let* outputGradient = outputFields.m_gradient.get();

        let numInputs = inputs.size();
        auto& inputValues = BorrowBuffer(m_inputValuesBufferRaw, numInputs);
        for (size_t i = 0; i < numInputs; i++)
        {
            let& input1 = SeeThroughNoOps(inputs, i);
            let& fields = *input1.m_dataFields;
            fail_if(!fields.m_value, "unexpectedly ran into a function that has no m_value yet??");
            inputValues[i] = fields.m_value.get();
        }

        // compute gradients for the desired input
        // Get or create m_gradient as the desired gradient's TensorView.
        // If the input is a lazyIndex, then the gradient is a view into the lazy source.
        let beta = LazilyCreateLazilyIndexedGradient(input, DetermineGradientStorageType(*f, 0));
        // backprop into the input
        // BUGBUG: (perf) In case of Reshape we currently make a copy, which is not needed --> see-through the op, and backprop through a reshaped view into Reshape's argument gradient?
        PrimitiveFunction::BackpropTo(outputGradient/*incoming*/, index, f->m_op, f->m_attributes, outputValue, inputValues, fields.m_gradient/*target*/, beta, *f);
        m_stats.numBatchedBackpropToCalls++;
#if 0   // debug the actual values
        fields.m_gradient->LogToFile(L"gradient", stderr);
#endif
    }

    // backprop into an input of a splice operation
    // This is done as a single CUDA launch into all inputs.
    // This is a little tricky. We backprop into all inputs.
    // So we must make sure we run this only once.
    // The first time this gradient gets puled, we do it for all inputs
    // and remember that this has been done.
    void BackpropToSplice(PrimitiveFunction* f)
    {
        // if we pull this a second time, then don't propagate again
        if (m_visitorTag.Visited(f->m_visitedTag))
            return;
        // fast path: only one input (and not Splice, which we do in bulk)
        if (f->m_inputs.size() == 1)
            return BackpropToUnbatched(f, 0);
        // Considerations:
        //  - For now we do optimize for consecutive inputs, because those would not
        //    have been transformed into a Splice operation in auto-batching.
        //    User's batch ops go here as well; we won't optimize for them for now.
        //    Hence, this is strictly a ScatterBatch operation.
        //  - It is possible that the Splice operation consumed the same input twice.
        //    This is currently handled via atomicAdd(), i.e. will have non-determinism.
        //    (A striding trick minimizes the probability of clashes.)
#if 0
        for (size_t index = 0; index < f->m_inputs.size(); index++)
        {
            BackpropToUnbatched(f, index);
            m_stats.numBatchedBackpropToCalls--; // for now fake the call count to see the potentail impact
        }
#else
#ifdef LOG_DETAILS
        LogFunction(*f, L"[bb#] ", SIZE_MAX);
#endif
        // The gradient of Splice is just copying all columns to the respective inputs.
        let& inputs =  f->m_inputs;
        // get the TensorViews for everything we may compute the gradient from
        let& output = f->m_outputs[0];
        let& outputFields = *output.m_dataFields;
#ifndef NO_BATCHED_BACKPROP
        fail_if(outputFields.m_lazyIndex.first, "unexpectedly ran into a function that does not own its output"); // we don't backprop through unbatched ops
#endif
        fail_if(!outputFields.m_value,    "unexpectedly ran into a function that has no m_value yet??");
        fail_if(!outputFields.m_gradient, "unexpectedly ran into a function that has no m_gradient yet??");
        let& outputGradient = outputFields.m_gradient; // this is the incoming batch of gradients

        let numInputs = inputs.size();
        auto& inputGradients = BorrowBuffer(m_inputValuesBuffer, numInputs); // target locations to propagate the columns to
        bool allBetasZero = true;
        for (size_t i = 0; i < numInputs; i++)
        {
            let& input = SeeThroughNoOps(inputs, i);
            // create the gradient memory for this input
            let beta = LazilyCreateLazilyIndexedGradient(input);
            let& fields = *input.m_dataFields;
            inputGradients[i] = fields.m_gradient;
            // handle inconsistent betas
            if (beta != 0 && allBetasZero)
            {
                // We were running under the assumption that all betas are zero, so we can use beta=0 below.
                // Now we must run with beta 1, and therefore manually reset all pevious ones.
                for (size_t i1 = 0; i1 < i; i1++) // these were all beta=0
                    SeeThroughNoOps(inputs, i1).m_dataFields->m_gradient->SetValue(0.0f);
                allBetasZero = false;
            }
            else if (beta == 0 && !allBetasZero)
                fields.m_gradient->SetValue(0.0f);
        }
        let beta = allBetasZero ? 0.0 : 1.0; // if at least one is not zero, we must run qwith beta=1

        // backprop into all inputs
        NDArrayView::ScatterBatch(outputGradient, inputGradients, beta);
#endif
        m_stats.numBackpropScatters++;
    }

    // backprop into weight parameter of a Times op (SeeThroughNoOps(inputs, 0))
    // This can be batched into a single matrix product.
    void BackpropToMatrixWeight(vector<pair<PrimitiveFunction*, size_t>>& consumers)
    {
#if 0
        for (auto& c : consumers)
            BackpropToUnbatched(c.first, c.second);
#else
        // batch all outGrads, and batch all right inputs
        let numBatchItems = consumers.size();
        if (numBatchItems == 1) // fast path if only one (and the last dim is not guaranteed to be a batch dim)
            // ^^ This comment makes no sense. The batching condition is hosed (too restrictive).
            return BackpropToUnbatched(consumers.front().first, consumers.front().second);
        // We compute
        //  leftGrad += sum_i outGrad_i @ right_i^T
        //            = (concat_i outGrad_i) @ (concat_i right_i)^T
        // where concat_i means to concatenate matrices along their trailing (batch) axis.
        // It has already been verified that all i have the same rank and dimensions except for a single reduction dimension.
        // ^^ This may be bogus.
        auto& timesOutGrads        = BorrowBuffer(m_inputValuesBuffer,     numBatchItems);
        auto& timesDataRightInputs = BorrowBuffer(m_outputGradientsBuffer, numBatchItems);
        let& f0 = *consumers.front().first;
#ifdef LOG_DETAILS
        LogFunction(f0, L"[bb*] ", 0);
#endif
        let& input0 = SeeThroughNoOps(f0.m_inputs, 0);
        size_t batchDim = 0;
        for (size_t i = 0; i < numBatchItems; i++)
        {
            let &c = consumers[i];
            fail_if(c.second != 0, "wrong input??");
            let& outGrad = c.first->m_outputs[0].m_dataFields->m_gradient;
            let& right = SeeThroughNoOps(c.first->m_inputs, 1).m_dataFields->m_value;
            timesOutGrads       [i] = outGrad;
            timesDataRightInputs[i] = right;
            let numItems = outGrad->Shape().Dimensions().back();
            fail_if(numItems != right->Shape().Dimensions().back(), "batch dimension of two inputs not the same??");
            batchDim += numItems;
        }
        auto outGradBatch = GatherBatchInArena(timesOutGrads       , f0.m_outputs[0].Shape().Rank() - 1, batchDim);
        auto rightBatch   = GatherBatchInArena(timesDataRightInputs, f0.m_inputs[1]. Shape().Rank() - 1, batchDim);

        // backprop into the left input from the batched outGrad and right
        auto& inputValues = BorrowBuffer(m_inputValuesBufferRaw, 2);
        inputValues[0] = nullptr;
        inputValues[1] = rightBatch.get();
        let beta = LazilyCreateLazilyIndexedGradient(input0, DetermineGradientStorageType(f0, 0));
        PrimitiveFunction::BackpropTo(/*outputGradient=*/outGradBatch.get(),      // incoming gradient from top...
                                      /*index=*/0, f0.m_op, f0.m_attributes,      // ...goes through this function...
                                      /*outputValue=*/nullptr, inputValues,       // ...using these values from forward pass...
                                      input0.m_dataFields->m_gradient, beta, f0); // ...into here
        m_stats.numBatchedBackpropToCalls++;
#endif
    }

    // backprop gradient into 'var' by pulling all of its consumers (recursively)
    // This is the second function that does batching.
    // The vectors for building the lists are class members so that we reuse the malloc.
    // This is a subroutine of RAggregateGradientFromAllConsumers().
    vector<pair<PrimitiveFunction*, size_t>> m_spliceConsumers;
    vector<pair<PrimitiveFunction*, size_t>> m_matrixWeightConsumers;
    vector<pair<PrimitiveFunction*, size_t>> m_summandConsumers;
    vector<pair<PrimitiveFunction*, size_t>> m_otherConsumers;
    void DetermineAndAddToBucket (const pair<PrimitiveFunction*, size_t>& c, bool isFirstCall = false)
    {
        if (isFirstCall) // first time pass true here
        {
            m_spliceConsumers      .clear();
            m_matrixWeightConsumers.clear();
            m_summandConsumers     .clear();
            m_otherConsumers       .clear();
        }
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
            let&   fOutShape = f.m_outputs[0].Shape().Dimensions();
            let&   gOutShape = g.m_outputs[0].Shape().Dimensions();
            let& fRightShape = f.m_inputs[1].Shape().Dimensions();
            let& gRightShape = g.m_inputs[1].Shape().Dimensions();
            let&   leftShape = f.m_inputs[0].Shape().Dimensions();
            let    outRank =   fOutShape.size();
            let   gOutRank =   gOutShape.size();
            let  rightRank = fRightShape.size();
            let gRightRank = gRightShape.size();
            let   leftRank =   leftShape.size();
            fail_if(leftShape != g.m_inputs[0].Shape().Dimensions(), "dimensions of matrix gradient don't match??");
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
#ifndef NO_BATCHED_FORWARD  // (the backward batching disable flag does not work somehow)
        // Note: This needs to be enabled for column-sparse gradients to work!
        // BUGBUG: (perf) We should also have a special path for Reshape(), as to avoid the memory copy.
        let opClass = g_oscTable[f->m_op]; // operation-specific auto-batching class
        // splice operation must use scatter
        if (opClass == OpSpecificConditionKind::Splice)
            m_spliceConsumers.push_back(c);
        // matrix product
        // We only collect matrix products with fully matching dimensions.
        else if (opClass == OpSpecificConditionKind::MatrixProduct && index == 0 &&
            (m_matrixWeightConsumers.empty() || (IsMatrixGradient0Batchable(*f, *m_matrixWeightConsumers.back().first))))
            m_matrixWeightConsumers.push_back(c);
        // backprop into either of Plus' arguments
        //else if (f->m_op == PrimitiveOpType::Plus)
        //    return m_summandConsumers;
        // all other
        else
#endif
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
    __declspec(noinline) void RAggregateGradientFromAllConsumers(const Variable& var)
    {
        let& fields = *var.m_dataFields;
        if (m_visitorTag.Visited(fields.m_visitedTag))
            return;

        auto& c = fields.m_consumers.first;
        // reached a "leaf" in the revesre tree; i.e. we hit the root
        if (!c.first)
            return;

        fail_if(!fields.m_needsGradient, "backprop into variable that does not need gradient");

        // recursively realize all consumers' outputs' gradients
        for (let& output : c.first->m_outputs)
            RAggregateGradientFromAllConsumers(output);
        for (auto& c : fields.m_consumers.second)
            for (let& output : c.first->m_outputs)
                RAggregateGradientFromAllConsumers(output);
        // Now all consumers are ready to propagate into var's m_gradient.
        // The resulting gradient is the sum of all that's backpropped here,
        // and this is the only place where a variable's gradient ever gets aggregated.

        // create var's m_gradient (may be a slice view)
        // m_gradient may already exist for Parameters, and when it came through Splice.
        // Because of the latter, we cannot really test this here, and should just remove this check.
        //fail_if(var.Kind() != VariableKind::Parameter && fields.m_gradient, "non-Parameter variable unexpectedly already has a gradient"); // (sanity check; I am not sure actually, maybe too strict)

        // fast path: only one consumer, nothing to batch
        // (with exception of Splice, which has a different optimization)
        if (fields.m_consumers.second.empty() && fields.m_consumers.first.first->m_op != PrimitiveOpType::Splice)
        {
            BackpropToUnbatched(c.first, c.second);
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
        DetermineAndAddToBucket(c, /*isFirstCall=*/true);
        for (auto& c : fields.m_consumers.second)
            DetermineAndAddToBucket(c);

        // splice bucket
        for (auto& c : m_spliceConsumers)
            BackpropToSplice(c.first);

        // matrix-weight bucket
        if (!m_matrixWeightConsumers.empty())
            BackpropToMatrixWeight(m_matrixWeightConsumers);

        // summation bucket
        // ...

        // others bucket
        for (auto& c : m_otherConsumers)
            BackpropToUnbatched(c.first, c.second);
    }

public:
    // -----------------------------------------------------------------------
    // BatchedBackward() -- entry point for auto-batched implementation of PrimitiveFunction::Backward()
    // -----------------------------------------------------------------------

    // implant gradients into all variables
    // Unlike BatchedForward(), this is eager. If you call it twice, it's a completely new computation.
    // If you need multiple gradients, ask for them in a single go.
    void BatchedBackward(const Variable& root, unordered_map<Parameter, NDArrayViewPtr>& gradients)
    {
        if (!root.m_dataFields->m_needsGradient)
            LogicError("BatchedBackward: cannot compute gradient for root with m_needsGradient being False.");
        // BUGBUG: make sure some edge cases are done right:
        //  - root.m_needsGradient=false
        //  - gradients contains root
        //  - root is a m_lazyIndex
        // first get the forward computation, batching, etc. done if not yet
        BatchedForward(root);
        // set up the m_consumer fields, which BatchedBackward() will work off
        m_visitorTag.Begin();
        RDetermineConsumersForBackward(root); // (gotta improve the name of these things)
        // implant the first gradient
        // TODO: allow user to pass in the starting value
        // BUGBUG: we get a [1] here, but should be a scalar. This is a bug outside.
        //if (root.Value()->Shape() != NDShape{})
        //    LogicError("BatchedBackward: root must be a scalar, or root gradient must have been implanted already");
        root.m_dataFields->m_gradient = m_arena.NewNDArrayView(root.Shape(), root.GetDataType(), StorageFormat::Dense, root.Value()->Device());
        root.m_dataFields->m_gradient->SetValue(1.0f);
        // if user passed NDArrayViewPtrs for the gradients, then keep using them
        // This way, the same buffers can be recycled.
        for (auto& kv : gradients)
        {
            if (kv.second)
                kv.second->SetValue(0.0f); // BUGBUG: inefficient; better reset to 0 lazily
            kv.first.m_dataFields->m_gradient = kv.second; // (if null then these will be set inside)
        }
        // BUGBUG: how to reset m_pendingInputs when there is no gradient on that path?
        // perform backprop
        // This traverses the tree top-down, where each node pulls gradient(s) from its consumer(s).
        // This way we can optimize operations, such as a matrix product or gradient of GatherBatch().
        m_visitorTag.Begin();
        for (auto& kv : gradients)
        {
            let& param = kv.first;
            let& fields = *param.m_dataFields;
            if (!fields.m_consumers.first.first) // if no consumer entry, we did not reach this gradient
                LogicError("BatchedBackward: a requested gradient is not part of root."); // TODO: or could it be due to StopGradient? What if StopGradient is used only sometimes?
            if (!fields.m_needsGradient) // (we could also just leafve the gradient 0)
                LogicError("BatchedBackward: cannot compute gradient for variable with m_needsGradient being False.");
            RAggregateGradientFromAllConsumers(param);
        }
        //fprintf(stderr, "Back-propagated through %d functions\n", (int)order.size());
        // implant the results into the map the user passed in
        for (auto& kv : gradients)
            kv.second = kv.first.m_dataFields->m_gradient;
        for (auto& kv : gradients)
        {
            let& param = kv.first;
            auto& fields = *param.m_dataFields;
            fields.m_consumers.first.first = nullptr;
            fields.m_consumers.second.clear();
        }
#ifdef LOG_STATS
        fprintf(stderr, "BatchedBackward: %d backprop computations besides %d gathers and %d scatters executed in nominal %d post-batching ops\n",
                (int)m_stats.numBatchedBackpropToCalls, (int)m_stats.numBackpropGathers, (int)m_stats.numBackpropScatters, (int)m_stats.numBackpropsToInputs);
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
    auto autoBatcher = Variable::AutoBatch();
    return autoBatcher.BatchedForward(m_outputs[0]);
}

// Perform backprop.
// TODO: CNTK grad() allows to pass multiple roots. Does that ever make sense in this context?
void PrimitiveFunction::BatchedBackward(std::unordered_map<Parameter, NDArrayViewPtr>& gradients) const
{
    auto autoBatcher = Variable::AutoBatch(); // has some internal state
    autoBatcher.BatchedBackward(m_outputs[0], gradients);
}

// non-batched version of BatchedForward()
// This is actually not used except as a fallback for debugging.
void PrimitiveFunction::MemoizeKnowableValue() const
{
    if (m_outputs.size() != 1)
        LogicError("Variable '%S' Value(): Only Variables with one output can compute their Value for now.", AsString().c_str());
    const auto& output = m_outputs[0];
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

} // namespace CNTK
