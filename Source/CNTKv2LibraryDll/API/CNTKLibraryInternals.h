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

#pragma warning(disable: 4702 4127)

// Forward declarations
namespace Microsoft { namespace MSR { namespace CNTK {
    struct MatrixBase;

    template <typename ElemType>
    class Matrix;

    template <typename ElemType>
    class TensorView;

    class ComputationNetwork;
    typedef std::shared_ptr<ComputationNetwork> ComputationNetworkPtr;

    template <typename ElemType>
    class ComputationNetworkBuilder;

    template <typename ElementType>
    class ComputationNode;

    class ComputationNodeBase;
    typedef std::shared_ptr<ComputationNodeBase> ComputationNodeBasePtr;

    struct GpuData;
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
#define NOT_IMPLEMENTED                                                                                                              \
    {                                                                                                                                \
        fprintf(stderr, "Inside File: %s  Line: %d  Function: %s  -> Feature Not Implemented.\n", __FILE__, __LINE__, __FUNCTION__); \
        CNTK::LogicError("Inside File: %s  Line: %d  Function: %s  -> Feature Not Implemented.\n", __FILE__, __LINE__, __FUNCTION__);      \
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
    class BlockFunction;
    class Function;
    class Variable;
    class Parameter;
    class Axis;
    class DeviceDescriptor;
    enum class PrimitiveOpType : unsigned int;
    enum class DataType : unsigned int;

    struct MinibatchInfo;
    struct MinibatchData;

    class Serializer;

    // Similar to make_shared except that it associates a custom deleter with the shared_ptr to ensure
    // that objects are deleted on the same side of the library DLL where they are allocated
    template <typename T, typename ...CtorArgTypes>
    inline std::shared_ptr<T> MakeSharedObject(CtorArgTypes&& ...ctorArgs)
    {
        auto objPtr = new T(std::forward<CtorArgTypes>(ctorArgs)...);
        return std::shared_ptr<T>(objPtr, [](T* ptr) { delete ptr; });
    }

    // Forward declarations
    class NDArrayView;
    typedef std::shared_ptr<NDArrayView> NDArrayViewPtr;

    class NDMask;
    typedef std::shared_ptr<NDMask> NDMaskPtr;

    class Value;
    typedef std::shared_ptr<Value> ValuePtr;

    class Function;
    typedef std::shared_ptr<Function> FunctionPtr;

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
    typedef std::shared_ptr<VariableFields> VariableFieldsPtr;

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

#ifndef CNTK_HEADERONLY_DEFINITIONS

    namespace Internal
    {
        CNTK_API FunctionPtr IsWithin(const Variable& operand, int offset, const std::wstring& name = L"");
        CNTK_API FunctionPtr PackedIndex(const Variable& operand, const Variable& index, const std::wstring& name = L"");
        CNTK_API FunctionPtr GatherPacked(const Variable& operand, const Variable& packedIndex, const std::wstring& name = L"");
        CNTK_API FunctionPtr ScatterPacked(const Variable& operand, const Variable& packedIndex, const Variable& condition, const std::wstring& name = L"");
        CNTK_API FunctionPtr ZeroesWithDynamicAxesLike(const Variable& operand);
        CNTK_API FunctionPtr Where(const Variable& condition, const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, const std::wstring& name = L"");
        CNTK_API FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::wstring& name = L"");
        CNTK_API FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, const std::wstring& name = L"");
        CNTK_API FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::wstring& name = L"");
        CNTK_API FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, const std::wstring& name = L"");
        CNTK_API FunctionPtr Slice(const Variable& operand, const std::vector<Axis>& axis, const std::vector<int>& beginIndex, const std::vector<int>& endIndex, const std::vector<int>& strides, const std::wstring& name = L"");
        CNTK_API FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const Axis& axis, const std::wstring& name = L"");
        CNTK_API FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const Axis& axis, bool keepReducedDimensions, const std::wstring& name = L"");
        CNTK_API FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const std::vector<Axis>& axes, const std::wstring& name = L"");
        CNTK_API FunctionPtr ReduceElements(const Variable& operand, const std::wstring& reductionOpName, const std::vector<Axis>& axes, bool keepReducedDimensions, const std::wstring& name = L"");
        CNTK_API FunctionPtr CosineDistanceWithNegativeSamples(const Variable& leftOperand, const Variable& rightOperand, const Variable& shiftWindow, const Variable& numberOfNegativeSamples, const std::wstring& name = L"");
        CNTK_API FunctionPtr Convolution(const Variable& convolutionMap, const Variable& operand, const NDShape& strides, const std::vector<bool>& sharing, const std::vector<bool>& autoPadding,
                                         const NDShape& dilation, bool transpose, const NDShape& outputShape, size_t groups, size_t maxTempMemSizeInSamples, const std::wstring& name = L"");
        CNTK_API FunctionPtr Convolution(const Variable& convolutionMap, const Variable& operand, const NDShape& strides, const std::vector<bool>& sharing, const std::vector<size_t>& lowerPad,
                                         const std::vector<size_t>& upperPad, const NDShape& dilation, bool transpose, const NDShape& outputShape, size_t groups, size_t maxTempMemSizeInSamples, const std::wstring& name = L"");
        CNTK_API FunctionPtr ConvolutionSequenceShape(const Variable& convolutionMap, const Variable& operand, const NDShape& strides, const std::vector<bool>& sharing, const std::vector<bool>& autoPadding,
                                                      const NDShape& dilation, bool transpose, const NDShape& outputShape, size_t groups, size_t maxTempMemSizeInSamples, const std::wstring& name = L"");
        CNTK_API FunctionPtr SpatialConvolution(const Variable& convolutionMap, const Variable& operand, const NDShape& strides, const std::vector<bool>& sharing,
                                                const std::vector<bool>& autoPadding, const NDShape& dilation, size_t maxTempMemSizeInSamples, const std::wstring& name = L"");
        CNTK_API FunctionPtr SpatialConvolutionSequenceShape(const Variable& convolutionMap, const Variable& operand, const NDShape& strides, const std::vector<bool>& sharing,
                                                             const std::vector<bool>& autoPadding,const NDShape& dilation, size_t maxTempMemSizeInSamples, const std::wstring& name = L"");
        CNTK_API FunctionPtr MatMul(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
        CNTK_API FunctionPtr Gemm(const Variable& operandA, const Variable& operandB, const Variable& operandC, float alpha = 1.0, float beta = 1.0, bool transA = false, bool transB = false, const std::wstring& name = L"");
        CNTK_API FunctionPtr Unsqueeze(const Variable& operand, const std::vector<Axis>& axes, const std::wstring& name = L"");

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

        CNTK_API void EnableNodeTiming();
        CNTK_API void DisableNodeTimeing();

        CNTK_API void EnableCPUEvalOptimization();
        CNTK_API void DisableCPUEvalOptimization();

        CNTK_API void SetMPIPackThreshold(size_t packThesholdInBytes);
        CNTK_API size_t GetMPIPackThreshold();

        CNTK_API bool AreEquivalent(const ::CNTK::FunctionPtr& f1, const ::CNTK::FunctionPtr& f2);
        CNTK_API bool AreEquivalent(const ::CNTK::Variable& v1, const ::CNTK::Variable& v2, bool allowParameterAndConstantsEquivalence = false);

        CNTK_API bool AreEqual(const ::CNTK::NDArrayView& view1, const ::CNTK::NDArrayView& view2, double relativeTolerance = 0.0, double absoluteTolerance = 0.0);
        CNTK_API bool AreEqual(const ::CNTK::Value& value1, const ::CNTK::Value& value2, double relativeTolerance = 0.0, double absoluteTolerance = 0.0);


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
    }

    // Forward-declare test fixtures, so that they can be used as friends.
    namespace Test
    {
        struct DeviceSelectionTestFixture;
    }

#endif
}
