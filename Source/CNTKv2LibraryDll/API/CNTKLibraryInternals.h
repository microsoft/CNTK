//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Contains internals used for defining the CNTKLibrary.h APIs
//

#pragma once

#ifdef SWIG
#define final
#define explicit
#define static_assert(condition, message)
#define __attribute__(x)
#endif

#ifdef _WIN32
#ifdef CNTKV2LIBRARYDLL
#define CNTK_API __declspec(dllexport)
#else
#define CNTK_API __declspec(dllimport)
#endif
#define _SCL_SECURE_NO_WARNINGS
#else // no DLLs on Linux
#define CNTK_API
#endif

#include <memory>
#include <vector>
#include <array>
#include <stdarg.h>
#include <assert.h>
#include <atomic>
#include <type_traits>
#include <unordered_set>
#include <unordered_map>
#include <stdlib.h>
#include <string.h>
#include "CNTKHelperTypes.h"

#pragma warning(disable: 4702 4127)

// Forward declarations
namespace Microsoft { namespace MSR { namespace CNTK {
    struct MatrixBase;

    template <typename ElemType>
    class Matrix;

    template <typename ElemType>
    class TensorView;

    struct TensorShape;

    class ComputationNetwork;
    typedef std::shared_ptr<ComputationNetwork> ComputationNetworkPtr;

    template <typename ElemType>
    class ComputationNetworkBuilder;

    template <typename ElementType>
    class ComputationNode;

    class ComputationNodeBase;
    typedef std::shared_ptr<ComputationNodeBase> ComputationNodeBasePtr;

    struct GpuData;

    class RNGHandle;
}}}

// TODO: The following should be reconciled with the equivalent code in the CNTK implementation

#ifndef _MSC_VER
#define _countof(_Array) (sizeof(_Array) / sizeof(_Array[0]))
static inline wchar_t* _wcsdup(const wchar_t *s)
{
    return ::wcsdup(s);
}
#endif

namespace CNTK
{

#define UNUSED(x) (void)(x) // for variables that are, e.g., only used in _DEBUG builds

#ifdef _MSC_VER
#define __declspec_noreturn __declspec(noreturn)
#else
#define __declspec_noreturn __attribute__((noreturn))
#endif

// Some projects require only some generic data types/interfaces from this file, and do not want to link explicitly to CNTKv2Library.
// In this case they have to define CNTK_HEADERONLY_DEFINITIONS before including CNTKLibrary.h
#ifndef CNTK_HEADERONLY_DEFINITIONS

#pragma warning(push)
#pragma warning(disable : 4996)
#ifndef _MSC_VER // TODO: what is the correct trigger for gcc?
    template <class E>
    __declspec_noreturn void ThrowFormatted(const char* format, ...) __attribute__((format(printf, 1, 2)));
#endif

    template <class E>
    CNTK_API __declspec_noreturn void ThrowFormatted(const char* format, ...);

#pragma warning(pop)

#endif

    // RuntimeError - throw a std::runtime_error with a formatted error string
#ifndef _MSC_VER // gcc __attribute__((format(printf())) does not percolate through variadic templates; so must go the macro route
#ifndef RuntimeError
#define RuntimeError ThrowFormatted<std::runtime_error>
#endif
#ifndef LogicError
#define LogicError ThrowFormatted<std::logic_error>
#endif
#ifndef InvalidArgument
#define InvalidArgument ThrowFormatted<std::invalid_argument>
#endif
#else
    template <class... _Types>
    __declspec_noreturn inline void RuntimeError(const char* format, _Types&&... _Args)
    {
        ThrowFormatted<std::runtime_error>(format, std::forward<_Types>(_Args)...);
    }
    template <class... _Types>
    __declspec_noreturn inline void LogicError(const char* format, _Types&&... _Args)
    {
        ThrowFormatted<std::logic_error>(format, std::forward<_Types>(_Args)...);
    }
    template <class... _Types>
    __declspec_noreturn inline void InvalidArgument(const char* format, _Types&&... _Args)
    {
        ThrowFormatted<std::invalid_argument>(format, std::forward<_Types>(_Args)...);
    }
#endif

#ifndef NOT_IMPLEMENTED
#define NOT_IMPLEMENTED                                                                                                               \
    {                                                                                                                                 \
        fprintf(stderr, "Inside File: %s  Line: %d  Function: %s  -> Feature Not Implemented.\n", __FILE__, __LINE__, __FUNCTION__);  \
        CNTK::LogicError("Inside File: %s  Line: %d  Function: %s  -> Feature Not Implemented.\n", __FILE__, __LINE__, __FUNCTION__); \
    }
#endif
}

namespace CNTK
{
    // Forward declarations
    class Utils;
    class NDShape;
    class PrimitiveFunction;
    class CompositeFunction;
    class DynamicProfiler;
    typedef std::shared_ptr<DynamicProfiler> DynamicProfilerPtr;
    class BlockFunction;
    class Function;
    class Variable;
    class Parameter;
    class Axis;
    class DeviceDescriptor;
    enum class PrimitiveOpType : unsigned char;
    enum class DataType : unsigned char;

    struct MinibatchInfo;
    struct MinibatchData;

    class Serializer;

    // Forward declarations
    class NDArrayView;
    typedef std::shared_ptr<NDArrayView> NDArrayViewPtr;

    class NDMask;
    typedef std::shared_ptr<NDMask> NDMaskPtr;

    class Value;
    typedef std::shared_ptr<Value> ValuePtr;

    class Function;
    typedef std::shared_ptr<Function> FunctionPtr;
    typedef std::shared_ptr<Function const> ConstFunctionPtr;

    class PrimitiveFunction;
    typedef std::shared_ptr<PrimitiveFunction> PrimitiveFunctionPtr;
    typedef std::shared_ptr<PrimitiveFunction const> ConstPrimitiveFunctionPtr;

    class Learner;
    typedef std::shared_ptr<Learner> LearnerPtr;

    class Learners;
    typedef std::shared_ptr<Learners> LearnersPtr;

    class Dictionary;
    typedef std::shared_ptr<Dictionary> DictionaryPtr;

    class MinibatchSource;
    typedef std::shared_ptr<MinibatchSource> MinibatchSourcePtr;

    class DistributedCommunicator;
    typedef std::shared_ptr<DistributedCommunicator> DistributedCommunicatorPtr;

    class QuantizedDistributedCommunicator;
    typedef std::shared_ptr<QuantizedDistributedCommunicator> QuantizedDistributedCommunicatorPtr;

    class DistributedLearner;
    typedef std::shared_ptr<DistributedLearner> DistributedLearnerPtr;

    struct VariableFields;
    //typedef std::shared_ptr<VariableFields> VariableFieldsPtr;
    typedef strong_shared_ptr<VariableFields> VariableFieldsPtr;

    class TrainingSession;
    typedef std::shared_ptr<TrainingSession> TrainingSessionPtr;

    class Evaluator;
    typedef std::shared_ptr<Evaluator> EvaluatorPtr;

    class Trainer;
    typedef std::shared_ptr<Trainer> TrainerPtr;

    class ProgressWriter;
    typedef std::shared_ptr<ProgressWriter> ProgressWriterPtr;

    class Accumulator;
    typedef std::shared_ptr<Accumulator> AccumulatorPtr;

    class UserFunctionFactory;
    typedef std::shared_ptr<UserFunctionFactory> UserFunctionFactoryPtr;

    class PackedValue;
    typedef std::shared_ptr<PackedValue> PackedValuePtr;
    typedef std::weak_ptr<PackedValue> PackedValueWeakPtr;

    struct MinibatchSourceConfig;

    typedef std::shared_ptr<Microsoft::MSR::CNTK::RNGHandle> RNGState;

#ifndef CNTK_HEADERONLY_DEFINITIONS

    namespace Internal
    {
        CNTK_API FunctionPtr IsWithin(const Variable& operand, int offset, const std::wstring& name = std::wstring());
        CNTK_API FunctionPtr PackedIndex(const Variable& operand, const Variable& index, const std::wstring& name = std::wstring());
        CNTK_API FunctionPtr GatherPacked(const Variable& operand, const Variable& packedIndex, const std::wstring& name = std::wstring());
        CNTK_API FunctionPtr ScatterPacked(const Variable& operand, const Variable& packedIndex, const Variable& condition, const std::wstring& name = std::wstring());
        CNTK_API FunctionPtr ZeroesWithDynamicAxesLike(const Variable& operand);
        CNTK_API FunctionPtr Where(const Variable& condition, const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, const std::wstring& name = L"");
        CNTK_API FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::wstring& name = L"");
        CNTK_API FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, const std::wstring& name = L"");
        CNTK_API FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::wstring& name = L"");
        CNTK_API FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, const std::wstring& name = L"");
        CNTK_API FunctionPtr Index(const Variable& operand, int index, const std::wstring& name = std::wstring());
        CNTK_API FunctionPtr Slice(const Variable& operand, const std::vector<Axis>& axis, const std::vector<int>& beginIndex, const std::vector<int>& endIndex, const std::vector<int>& strides, const std::wstring& name = L"");
        CNTK_API FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const Axis& axis, const std::wstring& name = L"");
        CNTK_API FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const Axis& axis, bool keepReducedDimensions, const std::wstring& name = L"");
        CNTK_API FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const std::vector<Axis>& axes, const std::wstring& name = L"");
        CNTK_API FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const std::vector<Axis>& axes, bool keepReducedDimensions, const std::wstring& name = L"");
        CNTK_API FunctionPtr CosineDistanceWithNegativeSamples(const Variable& leftOperand, const Variable& rightOperand, const Variable& shiftWindow, const Variable& numberOfNegativeSamples, const std::wstring& name = L"");
        CNTK_API FunctionPtr Convolution(const Variable& convolutionMap, const Variable& operand, const NDShape& strides, const std::vector<bool>& sharing, const std::vector<bool>& autoPadding,
                                         const NDShape& dilation, bool transpose, const NDShape& outputShape, size_t maxTempMemSizeInSamples, const std::wstring& name = L"");
        CNTK_API FunctionPtr SpatialConvolution(const Variable& convolutionMap, const Variable& operand, const NDShape& strides, const std::vector<bool>& sharing,
                                                const std::vector<bool>& autoPadding, const NDShape& dilation, size_t maxTempMemSizeInSamples, const std::wstring& name = L"");
        CNTK_API FunctionPtr GroupConvolution(const Variable& convolutionMap, const Variable& operand, const NDShape& strides, const std::vector<bool>& sharing,
                                              const std::vector<bool>& autoPadding, const NDShape& dilation, size_t groups, size_t maxTempMemSizeInSamples,
                                              const std::wstring& name = L"");
                                         

        // This is meant for debugging purposes only and is very likely to be deprecated in the future.
        CNTK_API void SaveAsLegacyModel(const FunctionPtr& rootFunction, const std::wstring& modelFile);

        CNTK_API size_t NewUniqueId();

        CNTK_API size_t GenerateRandomSeed(bool perWorkerLocalValue = false);

        // Internal hooks for testing and higher-level bindings
        // These should not be directly called by C++ API users
        CNTK_API void EnableReversingTensorShapesInErrorMessages();
        CNTK_API bool IsReversingTensorShapesInErrorMessagesEnabled();

        CNTK_API void AlwaysAllowSettingDefaultDevice();
        bool IsSettingDefaultDeviceAlwaysAllowed();

        CNTK_API void AllowRenamingFunctions();
        bool IsRenamingFunctionsAllowed();

        CNTK_API void SetAutomaticUnpackingOfPackedValues(bool disable);
        CNTK_API bool IsAutomaticUnpackingOfPackedValuesDisabled();

        CNTK_API void SetComputationNetworkTraceLevel(int traceLevel);
        int GetComputationNetworkTraceLevel();

        CNTK_API void SetGPUMemoryAllocationTraceLevel(int traceLevel);

        CNTK_API void SetMathLibTraceLevel(int traceLevel);

        CNTK_API void ForceDeterministicAlgorithms();
        CNTK_API bool ShouldForceDeterministicAlgorithms();

        CNTK_API void EnableSynchronousGPUKernelExecution();
        CNTK_API bool IsSynchronousGPUKernelExecutionEnabled();

        CNTK_API void UseSparseGradientAggregationInDataParallelSGD(bool enable);
        CNTK_API bool ShouldUseSparseGradientAggregationInDataParallelSGD();

        CNTK_API unsigned long GetRandomSeed();
        CNTK_API void SetFixedRandomSeed(unsigned long value);
        CNTK_API bool IsRandomSeedFixed();
        // If SetFixedRandomSeed has been called before, this will clear the 'fixed' flag.
        CNTK_API void ResetRandomSeed(unsigned long value = 0);

        CNTK_API void EnableForwardValuesSharing();
        CNTK_API void DisableForwardValuesSharing();

        CNTK_API void EnableGradientAccumulationOptimization();
        CNTK_API void DisableGradientAccumulationOptimization();

        static const uint64_t DefaultProfilerBufferSize = 32 * 1024 * 1024;
        CNTK_API void StartProfiler(const std::wstring& profilerDir = L"profiler", bool profilerSyncGpu = false, size_t profilerBufferSize = DefaultProfilerBufferSize);
        CNTK_API void EnableProfiler();
        CNTK_API void DisableProfiler();
        CNTK_API void StopProfiler();

        CNTK_API bool AreEquivalent(const ::CNTK::FunctionPtr& f1, const ::CNTK::FunctionPtr& f2);
        CNTK_API bool AreEquivalent(const ::CNTK::Variable& v1, const ::CNTK::Variable& v2, bool allowParameterAndConstantsEquivalence = false);

        CNTK_API bool AreEqual(const ::CNTK::NDArrayView& view1, const ::CNTK::NDArrayView& view2, double relativeTolerance = 0.0, double absoluteTolerance = 0.0);
        CNTK_API bool AreEqual(const ::CNTK::Value& value1, const ::CNTK::Value& value2, double relativeTolerance = 0.0, double absoluteTolerance = 0.0);

        CNTK_API size_t DefaultPackThresholdSizeInBytes();

        // This is an internal API, needed for testing.
        CNTK_API Dictionary ToDictionary(const MinibatchSourceConfig& dict);

#ifndef SWIG
        /// Convenience constructor that should be used by foreign language bindings.
        /// This is the Proper declaration understood by a real C++ compiler.
        LearnerPtr UniversalLearner(const std::vector<::CNTK::Parameter>& parameters, const std::vector<std::pair<::CNTK::Variable, ::CNTK::FunctionPtr> >& updates);
#else
        /// Convenience constructor that should be used by foreign language bindings.
        /// Workaround declaration for SWIG.
        /// This is for now necessary because it has been elusive to find an equivalent of
        /// %template() std::vector<std::pair<CNTK::Variable, std::shared_ptr<CNTK::Function>>>;
        /// which will generate correct code (i.e. code that will accept a list of tuples in the foreign language)
        /// when the proper declaration is processed by SWIG.
        LearnerPtr UniversalLearner(const std::vector<CNTK::Parameter>& parameters, const std::vector<std::pair<CNTK::Variable, CNTK::FunctionPtr> >& updates);
#endif

        CNTK_API void PrintBuiltInfo();
        CNTK_API void PrintGpuInfo(const std::vector<Microsoft::MSR::CNTK::GpuData>& gpusData);

        class VariableResolver;

        ///
        /// Returns true if num CPU Threads was set.
        ///
        bool MaxNumCPUThreadsSet();

        ///
        /// TensorBoardFileWriter allows collecting various metrics (e.g. loss/error etc.) as the training progresses,
        /// so that they can be analyzed in TensorBoard.
        /// It also provides an option to serialize the model being trained, so that it can also be visualized.
        /// The class is NOT thread-safe: it is assumed that only one thread is using each instance.
        ///
        class TensorBoardFileWriter final
        {
        public:
            ///
            /// Construct a TensorBoardFileWriter to log metrics as files in the given directory.
            /// An optional model argument allows serializing the model as well, so that it can be visualized
            /// in an external tool.
            ///
            CNTK_API explicit TensorBoardFileWriter(const std::wstring& dir, const FunctionPtr& modelToVisualize = nullptr);

            ///
            /// Construct a TensorBoardFileWriter to log metrics as files in the given directory.
            /// An network argument allows serializing the model as well, so that it can be visualized in an external tool.
            ///
            CNTK_API explicit TensorBoardFileWriter(const std::wstring& dir, const ::Microsoft::MSR::CNTK::ComputationNetworkPtr& modelToVisualize = nullptr);

            ///
            /// Destruct the TensorBoardFileWriter and close any open files.
            ///
            CNTK_API ~TensorBoardFileWriter() { Close(); }

            ///
            /// Record a value of some metric at a particular step.
            /// For example, to record average value of a loss function for the n-th minibatch, one could call this:
            ///     WriteValue("mb_avg_loss", lossValue, minibatchIdx);
            ///
            CNTK_API void WriteValue(const std::wstring& name, float value, uint64_t step);

#ifndef CNTK_UWP // doesn't support UWP due to compatibablity of opencv libs
            ///
            /// Record an image for a CNTK NDArrayViewPtr at a particular step.
            ///
            CNTK_API void WriteImage(const std::wstring& name, NDArrayViewPtr NDPtr, uint64_t step);
#endif

            ///
            /// Flushes any outstanding records to disk. Returns true on success, false otherwise.
            ///
            CNTK_API bool Flush();

            ///
            /// Flushes any outstanding records to disk and closes a currently open underlying file.
            /// Subsequent calls to WriteValue will open a new file. Returns true on success, false otherwise.
            ///
            CNTK_API bool Close();

        private:
            void Init();
            void WriteModel();
            void WriteRecord(const std::string& data);
            void WriteVersion(time_t time);

            // Disable copy-construction and assignment.
            TensorBoardFileWriter(const TensorBoardFileWriter& other) = delete;
            TensorBoardFileWriter& operator=(const TensorBoardFileWriter& other) = delete;

            const FunctionPtr m_model;
            const std::wstring m_dir;
            FILE* m_file;
            std::wstring m_fileName;
        };

        // SWIG callback wrapper for the UDF deserialization.
        class UDFDeserializeCallbackWrapper
        {
        public:
            virtual FunctionPtr operator()(const std::vector<Variable>&, const std::wstring&, const Dictionary&) const = 0;
            virtual ~UDFDeserializeCallbackWrapper() = default;
        };

        typedef std::shared_ptr<UDFDeserializeCallbackWrapper> UDFDeserializeCallbackWrapperPtr;

        CNTK_API void RegisterUDFDeserializeCallbackWrapper(UDFDeserializeCallbackWrapperPtr callbackPtr);


        CNTK_API bool IsNativeUserFunctionRegistered(const std::wstring& uniqueOpName);

        // A stripped-down version of boost::optional.
        // TODO: replace by std::optional, once it's fully supported by VS.
        template <class T>
        class Optional
        {
        public:

            Optional() = default;

            Optional& operator= (T value)
            {
                m_initialized = true;
                m_value = value;
                return *this;
            }

            void Reset()
            {
                m_initialized = false;
            }

            bool IsInitialized() const
            {
                return m_initialized;
            }

            T Get() const
            {
                if (IsInitialized())
                    return m_value;
                RuntimeError("Optional value is not initialized.");
            }

            Optional(const Optional&) = default; Optional& operator=(const Optional&) = default;
            Optional(Optional&&) = delete; Optional& operator=(Optional&&) = delete;
        private:
             T m_value;
             bool m_initialized { false };
        };

        // space for holding either a TensorView<float> or TensorView<double> (inside NDArrayView)
        struct TensorViewUnion
        {
            char buf[240]; // must be the same size as TensorView<float/double>
        };

        // Dynamite
        struct AutoBatchRedirection // redirect this value to a different owner function. Also allow for lazy Index operation.
        {
            // There are three cases for m_function and the holder:
            //  - no redirect:         m_functionHolder == empty  ; m_function == nullptr
            //  - redirect into owner: m_functionHolder == empty  ; m_function == m_owner
            //  - redirect elsewhere:  m_functionHolder == object ; m_function == m_functionHolder.get()
            // I.e. in the 'elsewhere' case, this instance owns m_function, holding the ref-count in m_functionHolder.
            PrimitiveFunction*   m_function = nullptr; // function that actually produces the value for this
            PrimitiveFunctionPtr m_functionHolder;     // holds shared_ptr to owner if it was added by auto-batching
            class SliceRange
            {
                // TODO: change to NDShapeDimension... which is not known here :(
                size_t m_begin = SIZE_MAX - 1;      // and we take this slice on the way (SIZE_MAX if none)
                size_t m_width = SIZE_MAX - 1;      // of this width (stacking case). SIZE_MAX if drop axis (batching case).
            public:
                void reset()                         { m_begin =        m_width = SIZE_MAX;    } // reset to no slice at all
                void reset(size_t index)             { m_begin = index; m_width = SIZE_MAX;    } // reset to index (with dropping axis)
                void reset(size_t begin, size_t end) { m_begin = begin; m_width = end - begin; } // reset to slice (not dropping axis)
                explicit SliceRange()                         { reset(); }
                explicit SliceRange(size_t index)             { reset(index); }
                explicit SliceRange(size_t begin, size_t end) { reset(begin, end); }
                bool empty() const { return m_begin == SIZE_MAX; }                  // empty (it's either index or slice)
                bool IsIndex()  const { return m_begin != SIZE_MAX && !IsSlice(); } // index means to index then drop the axis
                bool IsSlice()  const { return m_width != SIZE_MAX; }               // slice means to take a range (axis is not dropped)
                size_t Index()      const { return m_begin; } // note: these assume you have already checked the type
                size_t BeginIndex() const { return m_begin; }
                size_t EndIndex()   const { return m_begin + (IsSlice() ? m_width : 1); } // this can be used for both IsIndex and IsSlice
                size_t Width()      const { return m_width; }
                bool operator==(const SliceRange& other) const { return m_begin == other.m_begin && m_width == other.m_width; }
            } m_sliceRange;
            size_t m_depthHint = 0;                  // this redirection skipped a Barrier with this depthHint
            bool empty() const { return m_function == nullptr; }
            void reset(PrimitiveFunction* f = nullptr) // reset to non-redirected state or a simple non-owned function
            {
                m_function = f;
                // (the following is not really needed if m_function is empty)
                m_functionHolder.reset();
                m_sliceRange.reset();
                m_depthHint = 0;
            }
            void reset(PrimitiveFunctionPtr&& fPtr, SliceRange sliceRange = SliceRange()) // redirect to a function which we will own (with or without slice range)
            {
                m_function = fPtr.get();
                m_functionHolder = move(fPtr); // retain the shared_ptr, and thus the ref-count
                m_sliceRange = sliceRange;
                m_depthHint = 0;
            }
            void reset(const PrimitiveFunctionPtr& fPtr, SliceRange sliceRange = SliceRange()) // redirect to a function which we will own
            {
                reset(move(PrimitiveFunctionPtr(fPtr)), sliceRange);
            }
        };

        // optimized for main case of 1 consumer. No std::vector in that case.
        // Note: We may want to generalize this class.
        typedef std::pair<PrimitiveFunction*, size_t> AutoBatchConsumer;
#if 1
        class AutoBatchConsumers
        {
        public:
            //AutoBatchConsumers() { first.first = (PrimitiveFunction*)-1; } // this initialization can be removed once this is debugged (or once we replaced this horrible construct)
            size_t size() const { return m_numElements; }
            bool empty() const { return size() == 0; }
            void clear() { m_numElements = 0; m_secondary.clear(); }
            const AutoBatchConsumer& front() const { return m_primary.front(); }
            //void reset(AutoBatchConsumer&& fi) { first = std::move(fi); second.clear(); } // reset to one
            void push_back(AutoBatchConsumer&& fi)
            {
                // optimized for main case of few consumers. No std::vector in that case.
                if (m_numElements < m_primary.size()) // watch out: array::size() is the capacity
                    m_primary[m_numElements] = std::move(fi);
                else
                    m_secondary.emplace_back(std::move(fi));
                m_numElements++;
            }
            template<class F>
            void ForAll(const F& f) const
            {
                if (m_numElements <= m_primary.size())
                {
                    for (size_t i = 0; i < m_numElements; i++) // first few consumers
                        f(m_primary[i]);
                }
                else
                {
                    for (size_t i = 0; i < m_primary.size(); i++) // first few consumers
                        f(m_primary[i]);
                    for (auto& c : m_secondary) // all other consumers
                        f(c);
                }
            }
        private:
            size_t m_numElements = 0; // number of elements
            std::array<AutoBatchConsumer, 4> m_primary;  // the first few are stored without malloc. 4 seems a good choice.
            std::vector<AutoBatchConsumer> m_secondary;  // additional ones go into a vector object
        };
#else // somehow I cannot get the above to work, too tired it seems
        struct AutoBatchConsumers : private std::pair<AutoBatchConsumer, std::vector<AutoBatchConsumer>>
        {
            //AutoBatchConsumers() { first.first = (PrimitiveFunction*)-1; } // this initialization can be removed once this is debugged (or once we replaced this horrible construct)
            size_t size() const { return (first.first ? 1 : 0) + second.size(); }
            bool empty() const { return first.first == nullptr; }
            void clear() { first.first = nullptr; second.clear(); }
            const AutoBatchConsumer& front() const { return first; }
            //void reset(AutoBatchConsumer&& fi) { first = std::move(fi); second.clear(); } // reset to one
            void push_back(AutoBatchConsumer&& fi)
            {
                if (!first.first) // optimized for main case of 1 consumer. No std::vector in that case.
                    first = std::move(fi); // note: we don't need i for forward; can optimize
                else
                    second.emplace_back(fi);
            }
            template<class F>
            void ForAll(const F& f) const
            {
                if (first.first)
                    f(first);
                for (auto& c : second) // all other consumers
                    f(c);
            }
        };
#endif
    } // Internal

    // Forward-declare test fixtures, so that they can be used as friends.
    namespace Test
    {
        struct DeviceSelectionTestFixture;
    }

#endif
}
