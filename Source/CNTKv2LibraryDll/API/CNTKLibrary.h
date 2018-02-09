//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// This is the main header of the CNTK library API containing the entire public API definition.
//

#pragma once

#include <memory>
#include <vector>
#include <array>
#include <stdarg.h>
#include <assert.h>
#include <map>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <sstream>
#include <iosfwd>
#include <algorithm>
#include <mutex>
#include <future>
#include <cstddef>
#include <cmath>

#define DYNAMITE_ONLY // define this to remove some code paths that can never be executed. Note: Currently not testing without it.

#ifdef SWIG
#define final
#define explicit
#define static_assert(condition, message)
#endif

#include "CNTKLibraryInternals.h"

// undef max in the rest of the file to avoid conflicts with the max macro defined in windows.h.
#pragma push_macro("max")
#undef max

namespace CNTK
{
    ///
    /// Enumeration type denoting data type of symbolic data entities or actual data.
    ///
    enum class DataType : unsigned char
    {
        Unknown = 0,
        Float = 1,
        Double = 2,
        UChar = 3, // So far only used internally in deserializers.

        /* TODO:
        Bit,
        Char,
        Short,
        UShort,
        Int,
        UInt,
        Long,
        ULong,
        Float8,
        Float16,
        Complex,
        String,
        */
    };

    ///
    /// Get the 'DataType' corresponding to the ElementType template type argument.
    ///
    template <typename ElementType>
    inline DataType AsDataType()
    {
        if (std::is_same<ElementType, float>())
            return DataType::Float;
        else if (std::is_same<ElementType, double>())
            return DataType::Double;
        else
            NOT_IMPLEMENTED;
    }

    inline const char* DataTypeName(DataType dataType)
    {
        if (dataType == DataType::Float)
            return "Float";
        else if (dataType == DataType::Double)
            return "Double";
        else
            LogicError("Unknown DataType.");
    }

    inline size_t DataTypeSize(DataType dataType)
    {
        if (dataType == DataType::Float)
            return sizeof(float);
        else if (dataType == DataType::Double)
            return sizeof(double);
        else
            LogicError("Unknown DataType.");
    }

    ///
    /// Enumeration type denoting the format of storage underlying an instance of a NDArrayView.
    ///
    enum class StorageFormat
    {
        Dense,
        SparseCSC,
        SparseBlockCol,
    };

    inline bool IsSparseStorageFormat(StorageFormat storageFormat)
    {
        return (storageFormat != StorageFormat::Dense);
    }

    ///
    /// Enumeration type denoting the type of a compute device.
    ///
    enum class DeviceKind : unsigned int
    {
        CPU,
        GPU,
        // TODO: FPGA
    };

    inline const wchar_t* DeviceKindName(DeviceKind deviceKind)
    {
        switch (deviceKind)
        {
        case DeviceKind::CPU:
            return L"CPU";
        case DeviceKind::GPU:
            return L"GPU";
        default:
            LogicError("Unknown DeviceKind.");
        }
    }

    ///
    /// Enumeration type representing logging verbosity levels.
    ///
    enum class TraceLevel : unsigned int
    {
        Error = 0,
        Warning = 1,
        Info = 2
    };

    ///
    /// Denotes a multi-dimensional rectangular shape.
    ///
    typedef unsigned int NDShapeDimension;
    //typedef std::vector<NDShapeDimension> NDShapeDimensions;
    //typedef std::vector<NDShapeDimension> NDShapeDimensionsSpan;
    typedef FixedVectorWithBuffer<NDShapeDimension,4> NDShapeDimensions;
    typedef NDShapeDimensions::Span NDShapeDimensionsSpan;
    typedef FixedVectorWithBuffer<NDShapeDimension, 4> NDShapePermutation; // use as arg to NDArrayView::AsTransposed()
    class NDShape final
    {
        friend bool operator==(const NDShape& first, const NDShape& second);
        friend class PrimitiveFunction;

        static const NDShapeDimension SentinelDimValueForUnknownShape = (NDShapeDimension)-2;
    public:

        ///
        /// A placeholder value to use for an axis whose dimension is unknown and is to be inferred by the system.
        ///
        static constexpr NDShapeDimension InferredDimension{ (NDShapeDimension)-1 };

        ///
        /// A placeholder value to use for an axis whose dimension is unbound and is only determined
        /// when actual data is bound to the variable. Note that since the actual dimension is bound
        /// from actual minibatch data, the dimension can vary across different evaluations.
        ///
        static constexpr NDShapeDimension FreeDimension{ (NDShapeDimension)-3 };

        ///
        /// A placeholder shape to use to denote an unknown shape
        ///
        inline static const NDShape& Unknown()
        {
            const static NDShape unknown(1, SentinelDimValueForUnknownShape);
            return unknown;
        }

    public:
        ///
        /// Construct a NDShape with 0 axes, which denotes a scalar.
        ///
        NDShape() {}

        ///
        /// Construct a NDShape instance with the specified rank and dimensionality in each axis.
        ///
        explicit NDShape(size_t numAxes, NDShapeDimension dimension = InferredDimension)
            : m_shapeDims(numAxes, dimension)
        {}

        ///
        /// Construct a NDShape instance with specified dimensions.
        ///
        NDShape(const NDShapeDimensionsSpan& dimensions)
            : m_shapeDims(dimensions)
        {}
        NDShape(const std::vector<NDShapeDimension>& dimensions)
            : m_shapeDims(dimensions)
        {}
        NDShape(const std::vector<size_t>& dimensions)
            : m_shapeDims(Transform(dimensions, [](size_t dim) { return (NDShapeDimension)dim; }))
        {}
        template<typename IteratorType WHERE_IS_ITERATOR(IteratorType)>
        NDShape(const IteratorType& beginIter, const IteratorType& endIter)
            : m_shapeDims(Transform(Span<IteratorType>(beginIter, endIter), [](size_t dim) { return (NDShapeDimension)dim; }))
        {}

        ///
        /// Construct a NDShape instance with specified dimensions.
        ///
        NDShape(NDShapeDimensions&& dimensions)
            : m_shapeDims(std::move(dimensions))
        {}

        ///
        /// Construct a NDShape instance with specified dimensions.
        ///
        template<typename T>
        NDShape(const std::initializer_list<T>& dimensions)
            : m_shapeDims(dimensions)
        {}

        ///
        /// Returns the dimensions of 'this' shape as a std::vector<size_t>
        ///
#ifndef SWIG
        const NDShapeDimensions/*auto*/& Dimensions() const { return m_shapeDims; }
        //const auto& Dimensions() const { return m_shapeDims; }
#endif

        ///
        /// Returns a boolean indicating if 'this' shape is the special Unknown shape
        ///
        //bool IsUnknown() const { return (*this == NDShape::Unknown()); }
        bool IsUnknown() const
        {
            return
                m_shapeDims.end() == m_shapeDims.begin() + 1 &&
                m_shapeDims.front() == SentinelDimValueForUnknownShape;
        }

        ///
        /// Returns the rank of 'this' shape.
        ///
        size_t Rank() const { return m_shapeDims.size(); }

        ///
        /// Returns a reference to dimension size for the specified axis.
        ///
#ifndef SWIG
        NDShapeDimension/*auto*/& operator[](size_t axisId)
        //auto& operator[](size_t axisId)
        {
            return m_shapeDims.at(axisId);
        }
#endif

        ///
        /// Returns the dimension size for the specified axis.
        ///
#ifndef SWIG
        NDShapeDimension/*auto*/ operator[](size_t axisId) const
        //auto operator[](size_t axisId) const
        {
            return m_shapeDims.at(axisId);
        }
#endif

        ///
        /// Creates and returns a new NDShape instance with the same dimensions as 'this' shape's specified axis range [beginAxisId, endAxisId).
        ///
        NDShape SubShape(size_t beginAxisId = 0, size_t endAxisId = SIZE_MAX) const
        {
            endAxisId = (endAxisId == SIZE_MAX) ? Rank() : endAxisId;
            if ((endAxisId < beginAxisId) || (endAxisId > Rank()))
                InvalidArgument("NDShape::SubShape: The specified endAxisId (%zu) must not exceed the rank (%zu) of 'this' NDShape and must be >= than the specified beginAxisId (%zu)", endAxisId, Rank(), beginAxisId);

            NDShapeDimensions subShapeDims(m_shapeDims.begin() + beginAxisId, m_shapeDims.begin() + endAxisId);
            return std::move(subShapeDims);
        }

        ///
        /// Returns a boolean value indicating if the dimension size for any of the axes of 'this' shape is unknown/inferred (aka == NDShape::InferredDimension).
        ///
        bool HasInferredDimension() const
        {
            return (std::find(m_shapeDims.begin(), m_shapeDims.end(), InferredDimension) != m_shapeDims.end());
        }

        ///
        /// Returns a boolean value indicating if the dimension size for any of the axes of 'this' shape is free (aka == NDShape::FreeDimension).
        ///
        bool HasFreeDimension() const
        {
            return (std::find(m_shapeDims.begin(), m_shapeDims.end(), FreeDimension) != m_shapeDims.end());
        }

        ///
        /// Returns a boolean value indicating if the dimension size for any of the axes of 'this' shape is free or inferred
        /// i.e. (== NDShape::FreeDimension or == NDShape::InferredDimension).
        ///
        bool HasUnboundDimension() const
        {
            return HasFreeDimension() || HasInferredDimension();
        }

        ///
        /// Returns the total size of the rectangular shape that 'this' shape denotes.
        ///
        NDShapeDimension TotalSize(bool check = true) const
        {
            if (check && HasUnboundDimension())
                RuntimeError("NDShape::TotalSize: TotalSize cannot be determined for a NDShape '%S' with one or more dimensions being InferredDimension or FreeDimension.", AsString().c_str());

            const auto& dims = m_shapeDims;
            size_t rank = dims.size();
            if (rank == 0) // this function must be fast
                return 1;
            NDShapeDimension totalSize = dims.front();
            if (rank > 1)
                for (size_t k = 1; k < rank; k++)
                    totalSize *= dims[k];

            return totalSize;
        }

        ///
        /// Creates and returns a new shape constructed by appending the dimensions of the specified 'shape' to 'this' shape's dimensions.
        ///
        NDShape AppendShape(const NDShape& shape) const
        {
            std::vector<size_t> newShapeDims(Rank() + shape.Rank());
            std::copy(m_shapeDims.begin(), m_shapeDims.end(), newShapeDims.begin());
            std::copy(shape.m_shapeDims.begin(), shape.m_shapeDims.end(), newShapeDims.begin() + m_shapeDims.size());

            return NDShape(newShapeDims);
        }

        ///
        /// Creates and returns a new shape constructed by appending an axis of the given dimension, padding if needed.
        ///
        NDShape AppendAxis(size_t axisIndex, NDShapeDimension dim) const
        {
            if (axisIndex < Rank())
                LogicError("AppendAxis: invalid axisIndex.");
            std::vector<size_t> newShapeDims(axisIndex + 1);
            std::copy(m_shapeDims.begin(), m_shapeDims.end(), newShapeDims.begin());
            std::fill(newShapeDims.begin() + Rank(), newShapeDims.begin() + axisIndex, 1);
            newShapeDims[axisIndex] = dim;
            return NDShape(newShapeDims);
        }

        ///
        /// Create a string representation of 'this' NDShape for display/printing purposes
        ///
        std::wstring AsString() const
        {
            if (IsUnknown())
            {
                return L"[???]";
            }
            else
            {
                bool reverseShape = Internal::IsReversingTensorShapesInErrorMessagesEnabled();
                auto displayShape = *this;
                if (reverseShape)
                {
                    for (size_t i = 0, j = Rank() - 1; i < Rank(); ++i, --j)
                        displayShape[i] = (*this)[j];
                }

                std::wstringstream wStrStream;
                wStrStream << L"[";
                for (size_t i = 0; i < Rank(); i++)
                {
                    if (i != 0)
                        wStrStream << L" x ";

                    if (displayShape[i] == InferredDimension)
                        wStrStream << "?";
                    else if (displayShape[i] == FreeDimension)
                        wStrStream << "*";
                    else
                        wStrStream << displayShape[i];
                }
                wStrStream << L"]";
                return wStrStream.str();
            }
        }

    private:
        NDShapeDimensions m_shapeDims;
    };

    inline bool operator==(const NDShape& first, const NDShape& second)
    {
        return first.m_shapeDims == second.m_shapeDims;
    }

    inline bool operator!=(const NDShape& first, const NDShape& second)
    {
        return !(first == second);
    }

    typedef int SparseIndexType;

    static const unsigned long SentinelValueForAutoSelectRandomSeed = std::numeric_limits<unsigned long>::max() - 2; // An arbitrary choice of sentinel value

    ///
    /// Describes an input stream: its name, element type, storage, etc.
    ///
    struct StreamInformation
    {
        StreamInformation() : m_id(0), m_storageFormat(StorageFormat::Dense), m_elementType(DataType::Unknown),
            m_sampleLayout(NDShape::Unknown())
        {}

        std::wstring m_name;           // Unique name of the stream
        size_t m_id;                   // Unique identifier of the stream
        StorageFormat m_storageFormat; // Storage format of the stream
        DataType m_elementType;        // Element type of the stream
        NDShape m_sampleLayout;        // Layout of the sample for the stream
        bool m_definesMbSize = false;  // Flag indicating whether this stream defines minibatch size.

        std::wstring AsString() const
        {
            return m_name + L"(" + m_sampleLayout.AsString() + L")";
        }
    };

    inline bool operator==(const StreamInformation& left, const StreamInformation& right)
    {
        return ((left.m_id == right.m_id) &&
            (left.m_name == right.m_name) &&
            (left.m_storageFormat == right.m_storageFormat) &&
            (left.m_elementType == right.m_elementType) &&
            (left.m_sampleLayout == right.m_sampleLayout));
    }

    // Some projects require only some generic data types/interfaces from this file, and do not want to link explicitly to CNTKv2Library.
    // In this case they have to define CNTK_HEADERONLY_DEFINITIONS before including CNTKLibrary.h
#ifndef CNTK_HEADERONLY_DEFINITIONS
    ///
    /// Checked mode enables additional runtime verification such as:
    /// - Tracking NaN occurrences in sequence gaps.
    /// - Function graph verification after binding of free static axes to actual values at runtime
    /// 
    /// Enabling checked mode incurs additional runtime costs and is meant to be used as a debugging aid.
    ///
    CNTK_API void SetCheckedMode(bool enable);
    bool GetCheckedMode();


    ///
    /// Specifies global logging verbosity level.
    ///
    CNTK_API void SetTraceLevel(TraceLevel value);

    ///
    /// Returns current logging verbosity level.
    ///
    CNTK_API TraceLevel GetTraceLevel();

    /// A collection of additional information needed for the distributed trainer to aggregate the gradients
    struct MinibatchInfo
    {
        bool atEndOfData;
        bool atEndOfSweep;
        size_t numberOfSamples;
        NDArrayViewPtr trainingLossValue;
        NDArrayViewPtr evalCriterionValue;

        bool IsEmpty() const { return numberOfSamples == 0; }
    };

    ///
    /// Additional GPU-specific device information.
    ///
    struct GPUProperties final
    {
        unsigned int deviceId;
        int versionMajor;
        int versionMinor;
        int cudaCores;
        std::string name;
        size_t totalMemory;
    };

    ///
    /// Denotes a compute device instance.
    ///
    class DeviceDescriptor final
    {
        friend bool operator==(const DeviceDescriptor& first, const DeviceDescriptor& second);

        friend struct Test::DeviceSelectionTestFixture;

        static std::mutex s_mutex;
        static bool s_defaultDeviceFrozen;
        static std::unique_ptr<DeviceDescriptor> s_defaultDevice;
        static std::vector<DeviceDescriptor> s_excludedDevices;
        static std::vector<DeviceDescriptor> s_allDevices;
        static std::vector<GPUProperties> s_gpuProperties;
    public:
        ///
        /// Returns the Id of 'this' device.
        ///
        unsigned int Id() const { return m_deviceId; }

        ///
        /// Returns the DeviceKind of 'this' device.
        ///
        DeviceKind Type() const { return m_deviceType; }

        ///
        /// Returns true if another CNTK process already holds an exclusive lock on this device.
        ///
        CNTK_API bool IsLocked() const;

        ///
        /// Static method to get the descriptor of the CPU device on the local system.
        ///
        static DeviceDescriptor CPUDevice() { return{ 0, DeviceKind::CPU }; }

        ///
        /// Static method to get the descriptor of the GPU device on the local system with the specified CUDA device ID.
        ///
        CNTK_API static DeviceDescriptor GPUDevice(unsigned int deviceId);

        ///
        /// This static method freezes the default device of the current CNTK process disallowing further changes
        /// through calls to TrySetDefaultDevice. This default device will used for all CNTK operations
        /// where a device needs to be specified and where none was explicitly provided. If no default device has been specified
        /// with a call to TrySetDefaultDevice, on the first invocation, this methods will auto-select one
        /// of the available (non-locked) devices as the default. Returns a descriptor of the globally default device.
        ///
        CNTK_API static DeviceDescriptor UseDefaultDevice();

        ///
        /// This static method tries to set the specified device as the globally default, optionally acquiring an exclusive
        /// (cooperative) lock on the device (GPU). The default device can only be changed if it has not yet been frozen by being
        /// implicitly used in any previous CNTK operation.
        ///
        /// CNTK uses cooperative locking for the device access, whereby only a single process can acquire
        /// a device lock. This locking mechanism allows CNTK processes to avoid device oversubscription only if they collectively
        /// choose so. In other words, the device locked by one CNTK process, can still be accessed by another CNTK process without
        /// acquiring any locks (i.e, the existing device lock can be ignored by other CNTK processes). This cooperative
        /// locking mechanism does not guarantee any kind of exclusive access to the device. The proper way to ensure exclusivity
        /// is to use tools provided by NVIDIA (nvidia smi).
        ///
        /// This methods returns false if
        /// * the specified device appears in the list of excluded devices;
        /// * 'acquireDeviceLock' is true and another process already holds a lock on the device;
        /// * 'acquireDeviceLock' is true and 'newDefaultDevice' corresponds to a CPU device (which cannot be locked).
        ///
        CNTK_API static bool TrySetDefaultDevice(const DeviceDescriptor& newDefaultDevice, bool acquireDeviceLock = false);

        ///
        /// Static method to retrieve additional properties (total memory, number of CUDA cores, etc.) for the specified GPU
        /// device. This method will raise an exception if a CPU device is specified as an argument.
        ///
        CNTK_API static const GPUProperties& GetGPUProperties(const DeviceDescriptor& device);

        ///
        /// Static method to specify a list of excluded devices that cannot be used as globally default (neither auto-selected
        /// nor explicitly specified by 'TrySetDefaultDevice'). For example, to avoid auto-selecting the CPU, invoke
        /// 'SetExcludedDevices({DeviceDescriptor::CPUDevice()})'. However, after the default device has been selected and
        /// frozen (by a call to UseDefaultDevice()), invoking this methods has no effect, it becomes essentially a no-op.
        ///
        CNTK_API static void SetExcludedDevices(const std::vector<DeviceDescriptor>& excluded);

        ///
        /// Static method to get a list of descriptors of all available/supported devices.
        ///
        CNTK_API static const std::vector<DeviceDescriptor>& AllDevices();

        ///
        /// Return a string summary of this device
        ///
        CNTK_API std::wstring AsString() const;

    private:
        DeviceDescriptor(unsigned int deviceId, DeviceKind deviceType)
            : m_deviceId(deviceId), m_deviceType(deviceType)
        {}

        /// Resets static properties, needed for unit-tests.
        CNTK_API static void Reset();

    private:
        unsigned int m_deviceId;
        DeviceKind m_deviceType;
    };

    inline bool operator==(const DeviceDescriptor& left, const DeviceDescriptor& right)
    {
        return ((left.Type() == right.Type()) && (left.Id() == right.Id()));
    }

    inline bool operator!=(const DeviceDescriptor& left, const DeviceDescriptor& right)
    {
        return !(left == right);
    }

    ///
    /// An interface denoting an entity that can serialize its state into a Dictionary.
    ///
    class IDictionarySerializable
    {
    public:
        ///
        /// Save the state of 'this' object into a dictionary.
        ///
        CNTK_API virtual Dictionary Serialize() const = 0;

        ///
        /// Returns the current version (e.g. model, checkpoint version) of the class
        /// implementing this interface. The version number must be incremented each time
        /// when Serialize() implementation is modified (for instance, when a new field is added to
        /// the class that needs to be captured in the dictionary generated by the Serialize method).
        ///
        CNTK_API virtual size_t CurrentVersion() const = 0;

    protected:
        virtual ~IDictionarySerializable() {}
    };

    ///
    /// Denotes a multi-dimensional writable or read-only array of elemental values.
    /// This type denotes a view and there may be multiple simultaneous views of the data underlying a NDArrayView instance.
    /// The underlying data is stored in sparse or dense format, and is located on a specific device.
    /// The actual underlying storage is either external or internal in which case its lifetime is managed through reference counting.
    ///
    class NDArrayView final : public std::enable_shared_from_this<NDArrayView>
    {
        friend class CompositeFunction;
        friend class Utils;
        friend class LearnerBase;
        friend class InternalVariable;
        friend class Value;
        friend class Accumulator;
        friend class PackedValue;
        friend class MPICommunicatorImpl;
        friend class BlockMomentumDistributedLearner;
        friend class Internal::VariableResolver;
        friend class Trainer;

        template <typename T, typename ...CtorArgTypes>
        friend inline std::shared_ptr<T> MakeSharedObject(CtorArgTypes&& ...ctorArgs);

    public:
        ///
        /// Construct a NDArrayView with the specified 'dataBuffer' as the backing storage.
        /// The 'dataBuffer' must have been allocated on the specified 'device', must be at least
        /// as large as the total size of the specified 'viewShape' and must outlive the created NDArrayView object.
        ///
        CNTK_API NDArrayView(::CNTK::DataType dataType, const NDShape& viewShape, void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device, bool readOnly = false);

        ///
        /// Construct a read-only NDArrayView with the specified 'dataBuffer' as the backing storage.
        /// The 'dataBuffer' must have been allocated on the specified 'device', must be at least
        /// as large as the total size of the specified 'viewShape' and must outlive the created NDArrayView object.
        ///
        NDArrayView(::CNTK::DataType dataType, const NDShape& viewShape, const void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device)
            : NDArrayView(dataType, viewShape, const_cast<void*>(dataBuffer), bufferSizeInBytes, device, /*readOnly =*/ true)
        {}

        ///
        /// Construct a NDArrayView with newly allocated sparse storage in SparseCSC format on the specified 'device' and initialize its contents
        /// with the specified Sparse CSC format data.
        ///
        template <typename ElementType>
        CNTK_API NDArrayView(const NDShape& viewShape, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const ElementType* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly = false);

        ///
        /// Construct a NDArrayView over newly allocated storage in the specified format on the specified 'device'.
        ///
        CNTK_API NDArrayView(::CNTK::DataType dataType, ::CNTK::StorageFormat storageType, const NDShape& viewShape, const DeviceDescriptor& device);

        ///
        /// Construct a NDArrayView over newly allocated dense storage on the specified 'device'.
        ///
        NDArrayView(::CNTK::DataType dataType, const NDShape& viewShape, const DeviceDescriptor& device)
            : NDArrayView(dataType, StorageFormat::Dense, viewShape, device)
        {}

        ///
        /// Construct a NDArrayView with the specified 'dataBuffer' as the backing storage.
        /// The 'dataBuffer' must have been allocated on the specified 'device', must be at least
        /// as large as the total size of the specified 'viewShape' and must outlive the created NDArrayView object.
        ///
        template <typename ElementType>
        NDArrayView(const NDShape& viewShape, ElementType* dataBuffer, size_t numBufferElements, const DeviceDescriptor& device, bool readOnly = false)
            : NDArrayView(AsDataType<ElementType>(), viewShape, dataBuffer, numBufferElements * sizeof(ElementType), device, readOnly)
        {}

        ///
        /// Construct a read-only NDArrayView with the specified 'dataBuffer' as the backing storage.
        /// The 'dataBuffer' must have been allocated on the specified 'device', must be at least
        /// as large as the total size of the specified 'viewShape' and must outlive the created NDArrayView object.
        ///
        template <typename ElementType>
        NDArrayView(const NDShape& viewShape, const ElementType* dataBuffer, size_t numBufferElements, const DeviceDescriptor& device)
            : NDArrayView(AsDataType<ElementType>(), viewShape, dataBuffer, numBufferElements * sizeof(ElementType), device)
        {}

        ///
        /// Construct a NDArrayView with the buffer underlying the specified std::vector or std::array being the underlying storage.
        /// The container must be at least as large as the total size of the specified 'viewShape' and should outlive the created NDArrayView object.
        ///
        template <typename ContainerType, typename std::enable_if<std::is_same<ContainerType, std::vector<typename ContainerType::value_type>>::value ||
                                                                  std::is_same<ContainerType, std::array<typename ContainerType::value_type, sizeof(ContainerType) / sizeof(typename ContainerType::value_type)>>::value>::type* = nullptr>
        NDArrayView(const NDShape& viewShape, ContainerType& sourceContainer, bool readOnly = false)
            : NDArrayView(viewShape, sourceContainer.data(), sourceContainer.size(), DeviceDescriptor::CPUDevice(), readOnly)
        {}

        ///
        /// Construct a read-only NDArrayView with the buffer underlying the specified std::vector or std::array being the underlying storage.
        /// The container must be the same size as the total size of the specified 'viewShape' and should outlive the created NDArrayView object.
        ///
        template <typename ContainerType, typename std::enable_if<std::is_same<ContainerType, std::vector<typename ContainerType::value_type>>::value ||
                                                                  std::is_same<ContainerType, std::array<typename ContainerType::value_type, sizeof(ContainerType) / sizeof(typename ContainerType::value_type)>>::value>::type* = nullptr>
        NDArrayView(const NDShape& viewShape, const ContainerType& sourceContainer)
            : NDArrayView(viewShape, sourceContainer.data(), sourceContainer.size(), DeviceDescriptor::CPUDevice())
        {
            if (sourceContainer.size() != viewShape.TotalSize())
                InvalidArgument("The size (%zu) of the STL container does not match the size (%u) of the specified viewShape '%S'.",
                                sourceContainer.size(), viewShape.TotalSize(), viewShape.AsString().c_str());
        }

        ///
        /// Construct a NDArrayView over newly allocated dense storage on the specified device and
        /// assign the specified value to each element of the view.
        ///
        template <typename ElementType>
        explicit NDArrayView(const ElementType& value, const NDShape& viewShape = { 1 }, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), bool readOnly = false)
            : NDArrayView(AsDataType<ElementType>(), viewShape, device)
        {
            SetValue(value);
            m_isReadOnly = readOnly;
        }

        ///
        /// Construct a NDArrayView over newly allocated dense storage on the specified device and assign the specified value to each element of the view.
        /// The specified value is cast to the specified DataType.
        /// BUGBUG: Shape should be { } (a scalar), not a 1-dimensional vector.
        ///
        explicit NDArrayView(double value, DataType dataType = DataType::Float, const NDShape& viewShape = { 1 }, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), bool readOnly = false)
            : NDArrayView(dataType, viewShape, device)
        {
            switch (m_dataType)
            {
            case DataType::Float:
                SetValue((float)value);
                break;
            case DataType::Double:
                SetValue(value);
                break;
            default:
                LogicError("Unsupported DataType %s.", DataTypeName(m_dataType));
                break;
            }

            m_isReadOnly = readOnly;
        }

        ///
        /// Construct a NDArrayView over newly allocated dense storage on the specified device and assign the values
        /// pointed to by 'valuePtr' to the view. The buffer must match the specified viewShape.
        /// The specified value is cast to the specified DataType.
        ///
        NDArrayView(const double* data, DataType dataType, const NDShape& viewShape, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), bool readOnly = false)
            : NDArrayView(dataType, viewShape, device)
        {
            SetValue(data, m_viewShape.TotalSize());
            m_isReadOnly = readOnly;
        }

        ///
        /// Destruct 'this' NDArrayView object
        ///
        CNTK_API ~NDArrayView();

        ///
        /// Returns a writable device pointer to the data buffer underlying 'this' view
        /// Throws an exception if 'this' view is read-only
        ///
        template <typename ElementType>
        CNTK_API ElementType* WritableDataBuffer();

        ///
        /// Returns a read-only pointer to the data buffer underlying 'this' view
        /// The buffer may be a GPU buffer that cannot be directly accessed.
        ///
        template <typename ElementType>
        CNTK_API const ElementType* DataBuffer() const;

        ///
        /// Copies the data buffer to the given vector
        /// The resulting vector will have a size equal to the number of elements.
        /// For now, this is only implemented for dense data.
        /// If the result type differs from DataType, it will be type-cast without warning.
        ///
        template <typename ResultType>
        CNTK_API void CopyDataTo(std::vector<ResultType>& outputBuffer) const;

        ///
        /// Returns a read-only pointer to the data buffers in sparse CSC format underlying 'this' view
        /// The buffers may be a GPU buffer that cannot be directly accessed.
        ///
        template <typename ElementType>
        CNTK_API std::tuple<const ElementType *, const SparseIndexType*, const SparseIndexType*, size_t> SparseCSCDataBuffers() const;

        ///
        /// Returns a read-only pointer to the data buffers in sparse block column format underlying 'this' view
        /// The buffers may be a GPU buffer that cannot be directly accessed.
        /// 
        template <typename ElementType>
        CNTK_API std::tuple<const void*, const SparseIndexType*, const SparseIndexType*, size_t, size_t, size_t> SparseBlockColumnDataBuffers() const;

        ///
        /// adjusts the sparse block column matrix with the new Col2BlockId
        /// For each column, if new Col2BlockId contains valid index, a corresponding block exists at the index
        /// if old col2BlockId[i] contains value at that column, it would be copied over; otherwise the block would be filled with zeros
        ///
        CNTK_API void AdjustSparseBlockColumn(const SparseIndexType* cpuCol2BlockId, size_t numBlocks, bool useBlockId2Col);

        ///
        /// Returns the descriptor of the device that 'this' view resides on
        ///
        DeviceDescriptor Device() const { return m_device; }

        ///
        /// Returns the data type of 'this' view's contents.
        ///
        DataType GetDataType() const { return m_dataType; }

        ///
        /// Returns the storage format of 'this' view.
        ///
        StorageFormat GetStorageFormat() const { return m_storageFormat; }

        ///
        /// Returns the shape 'this' view.
        ///
        const NDShape& Shape() const { return m_viewShape; }

        ///
        /// Returns a boolean indicating if 'this' view contains data in sparse storage format.
        ///
        bool IsSparse() const
        {
            return (GetStorageFormat() != StorageFormat::Dense);
        }

        ///
        /// Returns a boolean indicating if 'this' view is read-only.
        ///
        bool IsReadOnly() const { return m_isReadOnly; }

        // TODO: The set methods should be offered in template from
        ///
        /// Fill 'this' NDArrayView with the specified value. The underlying DataType of 'this' view should be either DataType::Float or DataType::Double.
        ///
        CNTK_API void SetValue(float value);

        ///
        /// Fill 'this' NDArrayView with the specified value. The underlying DataType of 'this' view should be DataType::Double.
        ///
        CNTK_API void SetValue(double value);

    private:
        ///
        /// Copy the specified buffer to 'this' NDArrayView. Data is type-cast to the actual type.
        ///
        CNTK_API void SetValue(const double* data, size_t size);
    public:

        ///
        /// Creates a new NDArrayView with newly allocated storage on the specified device and copies 'this' view's contents into the newly allocated view.
        ///
        CNTK_API NDArrayViewPtr DeepClone(const DeviceDescriptor& device, bool readOnly = false) const;

        ///
        /// Creates a new NDArrayView with newly allocated storage on the same device as 'this' view and copies 'this' view's contents into the newly allocated view.
        ///
        inline NDArrayViewPtr DeepClone(bool readOnly) const
        {
            return DeepClone(this->Device(), readOnly);
        }

        ///
        /// Creates a new NDArrayView with newly allocated storage on the same device as 'this' view and copies 'this' view's contents into the newly allocated view.
        ///
        inline NDArrayViewPtr DeepClone() const
        {
            return DeepClone(this->IsReadOnly());
        }

        ///
        /// Creates a new NDArrayView which is an alias of 'this' view; i.e. a new view of the same shape as 'this' over the same underlying data.
        ///
        CNTK_API NDArrayViewPtr Alias(bool readOnly = false) const;

        ///
        /// Tests whether an NDArrayView is a full alias of another.
        ///
        CNTK_API bool IsAliasOf(const NDArrayViewPtr& other) const;

        ///
        /// Performs a numeric operation, returning NDArrayView(beta * *this + alpha * op(inputs)).
        /// The result is shaped according to broadcasting rules.
        /// If 'out' is not povided, a new object with correct shape will be created.
        /// For beta=0, 'out' may contain uninitialized/undefined values.
        ///
        CNTK_API static NDArrayViewPtr NumericOperation(const std::vector<NDArrayViewPtr>& inputs, double alpha, int op, NDArrayViewPtr out = nullptr, double beta = 0, int reductionOp = -1);
        CNTK_API static NDArrayViewPtr NumericOperation(const std::vector<NDArrayViewPtr>& inputs, double alpha, const std::wstring& op, NDArrayViewPtr out = nullptr, double beta = 0, const std::wstring& reductionOp = L"Sum");
        CNTK_API static NDArrayViewPtr NumericOperation(const std::vector<NDArrayViewPtr>& inputs, double alpha, int op, const NDShape& outShape, int reductionOp = -1);
        CNTK_API static NDArrayViewPtr NumericOperation(const std::vector<NDArrayViewPtr>& inputs, double alpha, const std::wstring& op, const NDShape& outShape, const std::wstring& reductionOp = L"Sum");

        ///
        /// Performs the matrix product, returning NDArrayView(beta * *this + alpha * trC(alpha * trA(inputA) * trB(inputB)))
        /// where trX is transposition if transX is true.
        /// For beta=0, *this may contain uninitialized/undefined values.
        ///
        CNTK_API static NDArrayViewPtr MatrixProduct(bool transC, const NDArrayViewPtr& inputA, bool transA, const NDArrayViewPtr& inputB, bool transB, double alpha, size_t outputRank, NDArrayViewPtr out = nullptr, double beta = 0);

        ///
        /// Converts a tensor of indices to one-hot representation
        /// The output view must be supplied, and can be dense or sparse.
        ///
        CNTK_API static NDArrayViewPtr AsOneHot(NDArrayViewPtr arg, size_t axis, NDArrayViewPtr out);

        ///
        /// Batch all inputs into a single tensor, along the last or a newly created axis.
        /// Specifying an axis outside the valid range will insert such an axis; a negative value will shift the axis indices in the result.
        /// When gathering along a new axis, all inputs must have identical dimensions. Gathering along the last axis requires
        /// all axis but the last to match.
        /// If out is not provided, a new object is created.
        ///
        CNTK_API static NDArrayViewPtr GatherBatch(const std::vector<NDArrayViewPtr>& inputs, size_t axis, NDArrayViewPtr out = nullptr);

        ///
        /// Scatter a single tensor into all inputs along the slowest-changing axis.
        /// This is the inverse to GatherBatch().
        ///
        CNTK_API static void ScatterBatch(const NDArrayViewPtr& input, std::vector<NDArrayViewPtr>& outputs, size_t axis, double beta = 0);

        ///
        /// Creates a new NDArrayView which is an alias of a slice of 'this' view; i.e. a new view over the underlying data
        /// corresponding to the specified slice of 'this' view.
        /// The view is compatible with Matrix operations. Hence, it must be contiguous in memory, it cannot incur strides.
        /// (If you need strides, use Slice().)
        /// If extent[] has less axes than the object, those axes are dropped from the result, assuming extents of assumed 1.
        /// This expresses the common case of indexing the batch (=trailing) axis.
        /// If the tensor is sparse, the leading axis (which is the sparse one) cannot be slice-viewed.
        ///
        CNTK_API NDArrayViewPtr SliceView(const NDShapeDimensionsSpan& startOffset, const NDShapeDimensionsSpan& extent, bool readOnly = false) const
        {
            return Slice(startOffset, extent, NDShapeDimensions(), SliceMode::ContiguousView, readOnly);
        }

        ///
        /// Creates a new NDArrayView which is an alias of a slice of 'this' view if possible, or else a copy.
        /// Caller can specify whether slice must be contiguous in memory, and whether a copy may be made to ensure this.
        /// The default is View; use SliceView() if you require Matrix-compatible a contiguous view.
        /// Only operations backed by TensorView allow non-contiguous slice views.
        /// (To always make a copy, use SliceCopy().)
        /// Copying is not supported for sparse storage at present.
        /// If extent[] has less axes than the object, those axes are dropped from the result, assuming extents of assumed 1.
        ///
        enum class SliceMode
        {
            View,                 // tensor-level view. Cheapest. TensorView compatible (e.g. NumericOperation). Underlying Matrix object is not changed.
            ContiguousView,       // tensor and matrix view. Fails if not -contiguous. Matrix compatible (e.g. MatrixProduct).
            ContiguousViewOrCopy, // like ContiguousView but makes a copy if not memory-contiguous. Matrix compatible, but not always a view.
        };
        CNTK_API NDArrayViewPtr Slice(const NDShapeDimensionsSpan& startOffset, const NDShapeDimensionsSpan& extent, const NDShapeDimensionsSpan& strides = NDShapeDimensions(),
                                      SliceMode sliceMode = SliceMode::View, bool readOnly = false) const;

        ///
        /// Creates a new NDArrayView which is an alias of a slice along the last axis of 'this' view, and reshaped to target.
        /// This is a fast-path version of a commonly occurring combination of SliceView() and AsShape() where the reshape does not incur extra runtime cost.
        ///
        CNTK_API NDArrayViewPtr SliceViewAsShape(size_t beginIndex, size_t endIndex, const NDShape& shape, bool readOnly = false) const;

        ///
        /// Tests if the tensor slice is memory-contiguous (no gaps due to strides).
        /// TODO: Remove again if this is not actually needed.
        ///
        CNTK_API bool IsContiguous() const;

        ///
        /// Same as Slice(), but always makes a copy. Non-contiguous slices and strides are allowed (except for sparse input).
        ///
        NDArrayViewPtr SliceCopy(const NDShapeDimensionsSpan& startOffset, const NDShapeDimensionsSpan& extent, const NDShapeDimensionsSpan& strides = NDShapeDimensions(), bool readOnly = false) const
        {
            return Slice(startOffset, extent, strides, SliceMode::View, readOnly)->DeepClone();
        }

        ///
        /// Creates a new NDArrayView which is a view that indexes the last axis.
        /// The axis itself is dropped in the returned view.
        /// If the tensor is sparse, it must have at least two axes (since the leading axis, which is the sparse one, cannot be slice-viewed).
        /// BUGBUG: This is covered by SliceView() with too short extent[], so remove this function and use SliceView() instead.
        ///
        CNTK_API NDArrayViewPtr IndexLastAxis(size_t index, bool readOnly = false) const;

#if 0
        ///
        /// vector/iteration interface
        ///
        // Not ideal. We operate on shared_ptrs, so one would need to say "for (x : *s)". TODO: Remove this again, but reuse this for Variable.
        class iterator : public std::iterator<std::random_access_iterator_tag, NDArrayViewPtr>
        {
            iterator(const std::shared_ptr<const NDArrayView> ths, size_t index) : m_value(ths), m_currentIndex(index) { }
        public:
            iterator(const NDArrayView*   ths, size_t index) : m_value(ths->shared_from_this()), m_currentIndex(index) { }
            iterator(      NDArrayView*   ths, size_t index) : m_value(ths->shared_from_this()), m_currentIndex(index) { }
            iterator operator++() { auto cur = *this; m_currentIndex++; return cur; }
            iterator operator++(int) { m_currentIndex++; return *this; }
            NDArrayViewPtr operator*() const { return m_value->IndexLastAxis(m_currentIndex); }
            //auto operator->() const { return m_value->IndexLastAxis(m_currentIndex); }
            bool operator==(const iterator& other) const { return m_value == m_value && m_currentIndex == other.m_currentIndex; }
            bool operator!=(const iterator& other) const { return !operator==(other); }
            iterator operator+(difference_type offset) const { return iterator(m_value, (size_t)((ptrdiff_t)m_currentIndex + offset)); }
            iterator operator-(difference_type offset) const { return iterator(m_value, (size_t)((ptrdiff_t)m_currentIndex - offset)); }
            difference_type operator-(const iterator& other) const { return m_currentIndex - other.m_currentIndex; }
        private:
            std::shared_ptr<const NDArrayView> m_value;
            size_t m_currentIndex; // current position
        };
        typedef iterator const_iterator;
        typedef NDArrayViewPtr value_type;
        const_iterator cbegin() const { return const_iterator(this, 0); }
        const_iterator cend()   const { return const_iterator(this, size()); }
        const_iterator begin()  const { return cbegin(); }
        const_iterator end()    const { return cend(); }
        iterator       begin()        { return iterator(this, 0); }
        iterator       end()          { return iterator(this, size()); }
#endif
        bool empty()  const { return Shape().Dimensions().begin() == Shape().Dimensions().end(); }
        size_t size() const
        {
            if (empty())
                InvalidArgument("size() cannot be applied to scalars");
            return Shape().Dimensions().back();
        }
        NDArrayViewPtr operator[](size_t index) const { return IndexLastAxis(index, IsReadOnly()); }

        ///
        /// Creates a new NDArrayView which is an alias of 'this' view but with a new shape.
        ///
        CNTK_API NDArrayViewPtr AsShape(const NDShape& newShape) const;

        ///
        /// Creates a new NDArrayView which is an alias of 'this' view but with axes permuted.
        /// The resulting object is no longer contiguous in memory, and therefore may not be usable for all operations.
        /// permutation[i] denotes which original axis will become axis i: newShape[i] <- shape[permutation[i]]
        /// If 'invert' then revert the meaning.
        ///
        CNTK_API NDArrayViewPtr AsTransposed(const NDShapePermutation& permutation, bool inverted = false) const;

        ///
        /// Copies the contents of the 'source' NDArrayView to 'this' view.
        /// The shapes of the 'source' view and 'this' view must be identical.
        ///
        CNTK_API void CopyFrom(const NDArrayView& source);

        ///
        /// Change the device of 'this' NDArrayView to the specified device
        ///
        CNTK_API void ChangeDevice(const DeviceDescriptor& device);

        ///
        /// Static method to construct a new NDArrayView object whose contents are drawn from a normal distribution with the specified mean and standard deviation..
        ///
        template <typename ElementType>
        CNTK_API static NDArrayViewPtr RandomNormal(const NDShape& shape, double mean, double stdDev, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Static method to construct a new NDArrayView object whose contents are drawn from a uniform distribution in the specified value range.
        ///
        template <typename ElementType>
        CNTK_API static NDArrayViewPtr RandomUniform(const NDShape& shape, double rangeStart, double rangeEnd, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Fill an instance with random values in { 0, scale }, with the proportion of 0 being equal to 'mean'.
        /// Scale != 1 is useful implementing the dropout mask (mean = keep rate = 1 - drop rate).
        /// The rngState must be initialized by LazilyCreateRNGState(). This function only does something
        /// upon first call, so you can chain them: SetToRandomDistributionBernoulli(LazilyCreateRNGState(rngState, device), 0.5).
        ///
        CNTK_API void SetToRandomDistributionBernoulli(RNGState& rngState, double mean, double scale = 1.0);

        // internal helper to initialize the rngState once
        CNTK_API static RNGState& LazilyCreateRNGState(RNGState& rngState, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());

        ///
        /// If the value stored is a scalar, returns it. Otherwise, throws an error.
        ///
        template<typename ElementType>
        CNTK_API ElementType AsScalar() const;

        ///
        /// Return a string summary of the NDArrayView.
        ///
        CNTK_API std::wstring AsString() const;

        ///
        /// Log the value of the tensor to a file
        /// This is for debugging purposes.
        ///
        CNTK_API void LogToFile(const std::wstring& name, FILE* f = stderr, size_t maxItems = 6, bool columnMajor = true) const;

        ///
        /// For debugging/timing: Synchronize the device. That is, wait until any pending operation has been completed.
        /// This will return the aggregate time in seconds since the last Sync() (for a max #kernel launches inbetween).
        ///
        CNTK_API static double Sync(const DeviceDescriptor& device);

        ///
        /// An interface to provide an efficient allocator for NDArrayViews.
        /// Currently not used by NDArrayView itself, but by auto-batching. In the future, we can pass this to functions in here as well.
        ///
        /*interface*/ struct IAllocator
        {
            virtual NDArrayViewPtr New(const NDShape& shape, const DataType& dataType, StorageFormat storageFormat, const DeviceDescriptor& device) = 0;
        };

    private:
        // Disallow copy and move construction and assignment
        NDArrayView(const NDArrayView&) = delete; NDArrayView& operator=(const NDArrayView&) = delete; NDArrayView& operator=(NDArrayView&&) = delete; NDArrayView(NDArrayView&& other) = delete;

    private:
        static const size_t AutoSelectRowColSplitPoint = SIZE_MAX;

    private:
    public: // public for MakeSharedObject() only. TODO: Remove once we know how to do that right.
        CNTK_API NDArrayView(::CNTK::DataType dataType, const NDShape& viewShape, bool readOnly, const std::shared_ptr<Microsoft::MSR::CNTK::MatrixBase>& storageObject);
        CNTK_API NDArrayView(::CNTK::DataType dataType, const NDShape& viewShape, size_t beginElement, size_t endElement, const std::shared_ptr<Microsoft::MSR::CNTK::MatrixBase>& storageObject);
        CNTK_API NDArrayView(::CNTK::DataType dataType, const Microsoft::MSR::CNTK::TensorShape& tensorShape, bool readOnly, const std::shared_ptr<Microsoft::MSR::CNTK::MatrixBase>& storageObject, bool sobTypeAlreadyVerified = false);
    private:
        CNTK_API NDArrayViewPtr Reviewed(const Microsoft::MSR::CNTK::TensorShape& tensorShape, bool readOnly) const;

        //template <typename ElementType>
        //static typename Microsoft::MSR::CNTK::Matrix<ElementType>::MatrixPtr GetMatrixImpl(const Microsoft::MSR::CNTK::TensorView<ElementType>& tensorView, size_t rowColSplitPoint);

        template <typename ElementType>
        typename Microsoft::MSR::CNTK::Matrix<ElementType>::ConstMatrixPtr GetMatrix(size_t rowColSplitPoint = AutoSelectRowColSplitPoint) const;

        template <typename ElementType>
        typename Microsoft::MSR::CNTK::Matrix<ElementType>::MatrixPtr GetWritableMatrix(size_t rowColSplitPoint = AutoSelectRowColSplitPoint);

        template<typename ElementType>
        friend class TensorViewPtrArrayRef;
        template<typename ElementType>
        friend class WritableTensorViewPtrArrayRef;
        template <typename ElementType>
        const Microsoft::MSR::CNTK::TensorView<ElementType>& NativeTensorView() const;
        template <typename ElementType>
        Microsoft::MSR::CNTK::TensorView<ElementType>& WritableNativeTensorView();
        template <typename ElementType, size_t N>
        friend /*static*/ void NativeNumericOperation(const std::array<NDArrayView*, N>& args, int opInt, int reductionOpInt, double alpha, double beta);

        const Microsoft::MSR::CNTK::TensorShape& GetTensorShape() const;

    private:
        ::CNTK::DataType m_dataType;
        DeviceDescriptor m_device;
        ::CNTK::StorageFormat m_storageFormat;
        NDShape m_viewShape;
        bool m_isReadOnly;

        Internal::TensorViewUnion m_tensorViewUnion;
        //union
        //{
        //    TensorView<float>  m_tensorViewFloat;
        //    TensorView<double> m_tensorViewDouble;
        //} m_tensorViews;
        //std::shared_ptr<void> m_tensorViewPtr; // Microsoft::MSR::CNTK::TensorView<ElemType>*
    public:
        // temporary debugging aid for identifying objects
        unsigned int m_uniqueIdForDebugging = GetUniqueId(); static unsigned int GetUniqueId() { static unsigned int id = 0; return ++id; }
    };

    ///
    /// Some operator overloads for NDArrayView.
    /// Note that the non-in-place ones allocate memory, while the in-place ones don't, but *= and /= cannot reduce.
    ///
    CNTK_API NDArrayViewPtr operator+(const NDArrayViewPtr& leftOperand, const NDArrayViewPtr& rightOperand);
    CNTK_API NDArrayViewPtr operator-(const NDArrayViewPtr& leftOperand, const NDArrayViewPtr& rightOperand);
    CNTK_API NDArrayViewPtr operator*(const NDArrayViewPtr& leftOperand, const NDArrayViewPtr& rightOperand);
    CNTK_API NDArrayViewPtr operator/(const NDArrayViewPtr& leftOperand, const NDArrayViewPtr& rightOperand);
    CNTK_API NDArrayViewPtr& operator+=(NDArrayViewPtr& leftOperand, const NDArrayViewPtr& rightOperand);
    CNTK_API NDArrayViewPtr& operator-=(NDArrayViewPtr& leftOperand, const NDArrayViewPtr& rightOperand);
    CNTK_API NDArrayViewPtr& operator*=(NDArrayViewPtr& leftOperand, const NDArrayViewPtr& rightOperand);
    CNTK_API NDArrayViewPtr& operator/=(NDArrayViewPtr& leftOperand, const NDArrayViewPtr& rightOperand);

    enum class MaskKind : char
    {
        Invalid = 0,
        Valid = 1,
        SequenceBegin = 2,
    };

    ///
    /// Denotes a multi-dimensional mask used for specifying specific sections of a NDArrayView object as masked/invalid.
    /// This type denotes a view and there may be multiple simultaneous views of the data underlying a NDMask instance.
    ///
    class NDMask final : public std::enable_shared_from_this<NDMask>
    {
        friend class CompositeFunction;

        template <typename T, typename ...CtorArgTypes>
        friend inline std::shared_ptr<T> MakeSharedObject(CtorArgTypes&& ...ctorArgs);

    public:
        ///
        /// Construct a new Mask object of specified shape
        ///
        CNTK_API explicit NDMask(const NDShape& shape, const DeviceDescriptor& device = DeviceDescriptor::CPUDevice());

        ///
        /// Destruct 'this' NDMask object
        ///
        CNTK_API ~NDMask();

        ///
        /// Mask out (i.e. mark Invalid) the specified sub-section of 'this' mask
        ///
        void InvalidateSection(const std::vector<size_t>& sectionOffset, const NDShape& sectionShape)
        {
            MarkSectionAs(sectionOffset, sectionShape, MaskKind::Invalid);
        }

        ///
        /// Mark the specified position in 'this' mask as sequence begin
        ///
        void MarkSequenceBegin(const std::vector<size_t>& offset)
        {
            NDShape sectionShape = NDShape(Shape().Rank(), 1);
            MarkSectionAs(offset, sectionShape, MaskKind::SequenceBegin);
        }

        ///
        /// Mark the specified sub-section of 'this' mask as sequence begin
        ///
        void MarkSequenceBegin(const std::vector<size_t>& offset, const NDShape& sectionShape)
        {
            MarkSectionAs(offset, sectionShape, MaskKind::SequenceBegin);
        }

        ///
        /// Clear the mask; i.e. unmask or mark Valid all currently masked (i.e. Invalid) values
        ///
        CNTK_API void Clear();

        ///
        /// Returns the number of masked/invalid values
        ///
        CNTK_API size_t MaskedCount() const;

        ///
        /// Returns the descriptor of the device that 'this' mask resides on
        ///
        DeviceDescriptor Device() const { return m_device; }

        ///
        /// Returns the shape 'this' mask.
        ///
        const NDShape& Shape() const { return m_maskShape; }

        ///
        /// Returns a read-only pointer to the data buffer underlying 'this' Mask object
        ///
        CNTK_API const MaskKind* DataBuffer() const;

        ///
        /// Creates a new NDMask with newly allocated storage on the specified device and copies 'this' mask's contents into the newly allocated view.
        ///
        CNTK_API NDMaskPtr DeepClone(const DeviceDescriptor& device) const;

        ///
        /// Creates a new NDMask with newly allocated storage on the same device as 'this' mask and copies 'this' mask's contents into the newly allocated mask.
        ///
        NDMaskPtr DeepClone() const
        {
            return DeepClone(this->Device());
        }

        ///
        /// Creates a new NDMask which is an alias of 'this' mask.
        ///
        CNTK_API NDMaskPtr Alias() const;

        ///
        /// Copies the contents of the 'source' NDMask to 'this' mask.
        /// The shapes of the 'source' mask and 'this' mask must be identical.
        ///
        CNTK_API void CopyFrom(const NDMask& source);

    private:
    public: // public for MakeSharedObject() only. TODO: Remove once we know how to do that right.
        NDMask(const NDShape& shape, Microsoft::MSR::CNTK::Matrix<char>* matrix);
    private:

        CNTK_API void MarkSectionAs(const std::vector<size_t>& sectionOffset, const NDShape& sectionShape, MaskKind maskKind);

        Microsoft::MSR::CNTK::Matrix<char>* GetMatrix() const;

        // Disallow copy and move construction and assignment
        NDMask(const NDMask&) = delete; NDMask& operator=(const NDMask&) = delete; NDMask& operator=(NDMask&&) = delete; NDMask(NDMask&& other) = delete;

    private:
        DeviceDescriptor m_device;
        NDShape m_maskShape;

        std::shared_ptr<Microsoft::MSR::CNTK::Matrix<char>> m_matrixView;
        //strong_shared_ptr<Microsoft::MSR::CNTK::Matrix<char>> m_matrixView;
    };

    ///
    /// Denotes an Axis of a Variable and is used for specifying the axes parameters of certain Functions such as reductions.
    /// Besides the static axes corresponding to each of the axes of the Variable's shape, Variables of kind 'Input' and any
    /// 'Output' Variables dependent on an 'Input' Variable also have 2 additional dynamic axes whose dimensions are known only
    /// when the Variable is bound to actual data during compute (viz. sequence axis and batch axis denoting the axis along which
    /// multiple sequences are batched)
    ///
    class Axis final
    {
        friend bool operator==(const Axis& first, const Axis& second);

        CNTK_API static const std::wstring StaticAxisNamePrefix;

        CNTK_API static const int SentinelStaticAxisIndexValueForDynamicAxes;
        CNTK_API static const int SentinelStaticAxisIndexValueForAllStaticAxes;
        CNTK_API static const int SentinelStaticAxisIndexValueForUnknownAxes;
        CNTK_API static const int SentinelEndStaticAxisIndexValue;
        CNTK_API static const int SentinelStaticAxisIndexValueForAllAxes;

        class UniqueDynamicAxesNames
        {
        public:
            CNTK_API bool RegisterAxisName(const std::wstring& axisName);
            CNTK_API const std::wstring& NewUniqueDynamicAxisName(const std::wstring& axisNamePrefix);

        private:
            std::unordered_set<std::wstring> m_allKnownDynamicAxisNames;
            std::mutex m_mutex;
        };

        CNTK_API static UniqueDynamicAxesNames s_uniqueDynamicAxisNames;

    public:
        CNTK_API static const std::vector<Axis>& DefaultInputVariableDynamicAxes();

        ///
        /// Axis object representing unknown dynamic axes
        ///
        CNTK_API static const std::vector<Axis>& UnknownDynamicAxes();

        ///
        /// Check whether Axis vector represents the unknown dynamic axes
        ///
        CNTK_API static bool IsUnknownDynamicAxes(const std::vector<Axis>& axes)
        {
            return axes.size() == 1 && axes[0].m_staticAxisIdx == SentinelStaticAxisIndexValueForUnknownAxes;
        }

    public:
        ///
        /// Construct an Axis object denoting a static axis with the specified index.
        ///
        explicit Axis(int staticAxisIdx)
            : m_staticAxisIdx(staticAxisIdx), m_isOrderedDynamicAxis(false)
        {
        }

        ///
        /// Construct a dynamic axis with the specified name.
        ///
        explicit Axis(const std::wstring& name, bool isOrderedDynamicAxis = true)
            : m_staticAxisIdx(SentinelStaticAxisIndexValueForDynamicAxes), m_name(name), m_isOrderedDynamicAxis(isOrderedDynamicAxis)
        {
            RegisterAxisName(name);
        }

        ///
        /// Returns a boolean indicating if 'this' Axis corresponds to a static axis
        ///
        bool IsStaticAxis() const
        {
            return ((m_staticAxisIdx != SentinelStaticAxisIndexValueForDynamicAxes) &&
                    (m_staticAxisIdx != SentinelStaticAxisIndexValueForAllStaticAxes) &&
                    (m_staticAxisIdx != SentinelStaticAxisIndexValueForUnknownAxes) &&
                    (m_staticAxisIdx != SentinelStaticAxisIndexValueForAllAxes));
        }

        ///
        /// Returns a boolean indicating if 'this' Axis corresponds to a dynamic axis
        ///
        bool IsDynamicAxis() const
        {
            return (m_staticAxisIdx == SentinelStaticAxisIndexValueForDynamicAxes);
        }

        ///
        /// Indicate whether 'this' Axis is a batch axis.
        ///
        bool IsBatchAxis() const
        {
            //TODO: Do we assume there is only one batch axis in the whole system?
            return (this->IsDynamicAxis() && !this->IsSequenceAxis());
        }

        ///
        /// Indicate whether 'this' Axis is a sequence axis.
        ///
        bool IsSequenceAxis() const
        {
            return (this->IsDynamicAxis() && this->m_isOrderedDynamicAxis);
        }

        ///
        /// Returns a boolean indicating if 'this' Axis is ordered; i.e. if there is an ordering between the dimensions along this axis.
        ///
        bool IsOrdered() const { return IsStaticAxis() || m_isOrderedDynamicAxis; }

        ///
        /// Returns the axis index if 'this' Axis is a static axis. Throws an exception otherwise if checked == true.
        ///
        int StaticAxisIndex(bool checked = true) const
        {
            if (checked && !IsStaticAxis())
                InvalidArgument("Cannot query the static axis index for a non-static axis");

            return m_staticAxisIdx;
        }

        ///
        /// Axis object representing the default dynamic axis.
        ///
        CNTK_API static const Axis& DefaultDynamicAxis();

        ///
        /// Axis object representing the sequence axis (ordered dynamic axis) of an
        /// operand whose dynamic axes have not yet been inferred/resolved (i.e. are unknown).
        /// This automatically resolves to the actual sequence dynamic axis of the operand that
        /// it is specified for, when the dynamic axes of the operand are resolved.
        ///
        CNTK_API static const Axis& OperandSequenceAxis();

        ///
        /// Axis object representing the batch axis.
        ///
        CNTK_API static const Axis& DefaultBatchAxis();

        ///
        /// Axis object representing all the static axes of an operand
        ///
        CNTK_API static const Axis& AllStaticAxes();

        ///
        /// Axis object representing all static and dynamic axes of an operand
        ///
        CNTK_API static const Axis& AllAxes();

        ///
        /// Returns a new unique Dynamic axis
        ///
        static Axis NewUniqueDynamicAxis(const std::wstring& axisNamePrefix, bool isOrderedDynamicAxis = true)
        {
            return Axis(s_uniqueDynamicAxisNames.NewUniqueDynamicAxisName(axisNamePrefix), isOrderedDynamicAxis);
        }

        ///
        /// Returns an axis object representing the default end static axis.
        /// This is used as the default value for the end axis for some operators like Reshape.
        ///
        static Axis EndStaticAxis()
        {
            return Axis(SentinelEndStaticAxisIndexValue);
        }

        ///
        /// Name of 'this' axis
        ///
        const std::wstring& Name() const
        {
            if (m_name.empty())
            {
                if (IsStaticAxis())
                    m_name = StaticAxisNamePrefix + std::to_wstring(m_staticAxisIdx);
                else if (m_staticAxisIdx == SentinelStaticAxisIndexValueForAllStaticAxes)
                    m_name = L"AllStaticAxes";
                else if (m_staticAxisIdx == SentinelStaticAxisIndexValueForUnknownAxes)
                    m_name = L"UnknownAxes";
                else if (m_staticAxisIdx == SentinelStaticAxisIndexValueForAllAxes)
                    m_name = L"AllAxes";
                else if (m_staticAxisIdx == SentinelStaticAxisIndexValueForDynamicAxes)
                    m_name = StaticAxisNamePrefix + L"DynamicAxisSentinel";
                else
                    LogicError("Unknown sentinel value for Axis");
            }
            return m_name.get();
        }

        ///
        /// Returns a string representation for this Axis.
        ///
        CNTK_API std::wstring AsString() const;

        ///
        /// Default constructor; results in an invalid axis object.
        ///
        Axis()
            : m_staticAxisIdx(SentinelStaticAxisIndexValueForDynamicAxes)
        {}

    private:
        CNTK_API void RegisterAxisName(const std::wstring& axisName);

    private:
        int m_staticAxisIdx;
        mutable OptionalString m_name; // (initialized lazily, since not needed most of the time) --TODO: Needed for some hash. Is that often?
        bool m_isOrderedDynamicAxis;
    };

    inline bool operator==(const Axis& first, const Axis& second)
    {
        if (first.IsDynamicAxis() != second.IsDynamicAxis())
            return false;

        if (!first.IsDynamicAxis())
            return first.StaticAxisIndex(/*checked =*/ false) == second.StaticAxisIndex(/*checked =*/ false);
        else
            return first.Name() == second.Name();
    }

    inline bool operator!=(const Axis& first, const Axis& second)
    {
        return !(first == second);
    }
}

namespace std {
    template <> struct hash<::CNTK::Axis>
    {
        size_t operator()(const ::CNTK::Axis& x) const
        {
            return std::hash<std::wstring>()(x.Name());
        }
    };
}


namespace CNTK
{
    template <typename T>
    class TrainingParameterSchedule;
    ///
    /// A serializable value represents one of:
    /// a) Boolean
    /// b) Signed and unsigned long integer
    /// c) Single and double precision floating point values
    /// d) NDShape
    /// e) Axis
    /// f) vector<DictionaryValue>
    /// g) Dictionary
    /// h) NDArrayView
    /// i) TrainingParameterSchedule<double>
    ///
    /// TODO: We need to have native support for DictionaryValue<vector> and DictionaryValue<NDArrayView>.
    class DictionaryValue final
    {
        friend class Serializer;
    public:
        enum class Type : unsigned int
        {
            None,
            Bool,
            Int,
            SizeT,
            Float,
            Double,
            String,
            NDShape,
            Axis,
            Vector,
            Dictionary,
            NDArrayView,
            TrainingParameterSchedule
        };

        static const char* TypeName(Type type)
        {
            switch (type)
            {
            case Type::None:
                return "None";
            case Type::Bool:
                return "Bool";
            case Type::Int:
                return "Int";
            case Type::SizeT:
                return "SizeT";
            case Type::Float:
                return "Float";
            case Type::Double:
                return "Double";
            case Type::String:
                return "String";
            case Type::NDShape:
                return "NDShape";
            case Type::Axis:
                return "Axis";
            case Type::Vector:
                return "Vector";
            case Type::Dictionary:
                return "Dictionary";
            case Type::NDArrayView:
                return "NDArrayView";
            case Type::TrainingParameterSchedule:
                return "TrainingParameterSchedule";
            default:
                LogicError("Unknown DictionaryValue::Type.");
            }
        }

    public:
        DictionaryValue() : m_valueType(Type::None)
        {
        }

        DictionaryValue(bool value) : m_valueType(GetValueType<bool>())
        {
            m_data.m_boolean = value;
        }

        DictionaryValue(int value) : m_valueType(GetValueType<int>())
        {
            m_data.m_int = value;
        }

        DictionaryValue(size_t value) : m_valueType(GetValueType<size_t>())
        {
            m_data.m_sizeT = value;
        }

        DictionaryValue(float value) : m_valueType(GetValueType<float>())
        {
            m_data.m_float = value;
        }

        DictionaryValue(double value) : m_valueType(GetValueType<double>())
        {
            m_data.m_double = value;
        }

        DictionaryValue(const wchar_t* value)
            : DictionaryValue(std::wstring(value))
        {}

        // Due to SWIG we had to flatten this template for vector<DictionaryValue>
        DictionaryValue(const std::vector<::CNTK::DictionaryValue>& value) : m_valueType(GetValueType<std::vector<::CNTK::DictionaryValue>>())
        {
            AllocateDataPtr(value);
        }

        template <typename T>
        DictionaryValue(const T& value) : m_valueType(GetValueType<T>())
        {
            static_assert((std::is_same<T, NDShape>::value ||
                std::is_same<T, Axis>::value ||
                std::is_same<T, std::wstring>::value ||
                std::is_same<T, std::vector<DictionaryValue>>::value ||
                std::is_same<T, Dictionary>::value ||
                std::is_same<T, TrainingParameterSchedule<double>>::value ||
                std::is_same<T, NDArrayView>::value),
                "Unsupported ValueType");

            AllocateDataPtr(value);
        }

        DictionaryValue(const DictionaryValue& other) : m_valueType(Type::Bool)
        {
            // The m_valueType must have been set to a non-ptr type to prevent an attempt to interpret
            // the underlying underlying uninitialized value as a ptr and free it.
            *this = other;
        }

        DictionaryValue(DictionaryValue&& other) : m_valueType(Type::Bool)
        {
            // The m_valueType must have been set to a non-ptr type to prevent an attempt to interpret
            // the underlying underlying uninitialized value as a ptr and free it.
            *this = std::move(other);
        }
        DictionaryValue& operator=(const DictionaryValue& other)
        {
            if (this != &other)
            {
                FreeDataPtr();

                m_valueType = other.m_valueType;
                m_data = other.m_data;

                if (other.m_valueType == Type::String)
                    AllocateDataPtr(other.Value<std::wstring>());
                else if (other.m_valueType == Type::NDShape)
                    AllocateDataPtr(other.Value<NDShape>());
                else if (other.m_valueType == Type::Axis)
                    AllocateDataPtr(other.Value<Axis>());
                else if (other.m_valueType == Type::Vector)
                    AllocateDataPtr(other.Value<std::vector<DictionaryValue>>());
                else if (other.m_valueType == Type::Dictionary)
                    AllocateDataPtr(other.Value<Dictionary>());
                else if (other.m_valueType == Type::NDArrayView)
                    AllocateDataPtr(other.Value<NDArrayView>());
                else if (other.m_valueType == Type::TrainingParameterSchedule)
                    AllocateDataPtr(other.Value<TrainingParameterSchedule<double>>());
            }

            return *this;
        }

        DictionaryValue& operator=(DictionaryValue&& other)
        {
            FreeDataPtr();

            m_valueType = other.m_valueType;
            m_data = other.m_data;

            if (other.m_valueType == Type::String ||
                other.m_valueType == Type::NDShape ||
                other.m_valueType == Type::Axis ||
                other.m_valueType == Type::Vector ||
                other.m_valueType == Type::Dictionary ||
                other.m_valueType == Type::TrainingParameterSchedule ||
                other.m_valueType == Type::NDArrayView)
            {
                other.m_data.m_ptr = nullptr;
            }

            other.m_valueType = Type::None;

            return *this;
        }
        ~DictionaryValue()
        {
            FreeDataPtr();
        }

        template <typename T, typename std::enable_if<std::is_same<T, bool>::value>::type* = nullptr>
        const T& Value() const
        {
            VerifyType<T>();
            return m_data.m_boolean;
        }

        template <typename T, typename std::enable_if<std::is_same<T, bool>::value>::type* = nullptr>
        T& Value()
        {
            VerifyType<T>();
            return m_data.m_boolean;
        }

        template <typename T, typename std::enable_if<std::is_same<T, int>::value>::type* = nullptr>
        const T& Value() const
        {
            VerifyType<T>();
            return m_data.m_int;
        }

        template <typename T, typename std::enable_if<std::is_same<T, int>::value>::type* = nullptr>
        T& Value()
        {
            VerifyType<T>();
            return m_data.m_int;
        }

        template <typename T, typename std::enable_if<std::is_same<T, size_t>::value>::type* = nullptr>
        const T& Value() const
        {
            VerifyType<T>();
            return m_data.m_sizeT;
        }

        template <typename T, typename std::enable_if<std::is_same<T, size_t>::value>::type* = nullptr>
        T& Value()
        {
            VerifyType<T>();
            return m_data.m_sizeT;
        }

        template <typename T, typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr>
        const T& Value() const
        {
            VerifyType<T>();
            return m_data.m_float;
        }

        template <typename T, typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr>
        T& Value()
        {
            VerifyType<T>();
            return m_data.m_float;
        }

        template <typename T, typename std::enable_if<std::is_same<T, double>::value>::type* = nullptr>
        const T& Value() const
        {
            VerifyType<T>();
            return m_data.m_double;
        }

        template <typename T, typename std::enable_if<std::is_same<T, double>::value>::type* = nullptr>
        T& Value()
        {
            VerifyType<T>();
            return m_data.m_double;
        }

        template <typename T, typename std::enable_if<std::is_same<T, NDShape>::value ||
            std::is_same<T, Axis>::value ||
            std::is_same<T, std::wstring>::value ||
            std::is_same<T, std::vector<DictionaryValue>>::value ||
            std::is_same<T, Dictionary>::value ||
            std::is_same<T, TrainingParameterSchedule<double>>::value ||
            std::is_same<T, NDArrayView>::value>::type* = nullptr>
            const T& Value() const
        {
            VerifyType<T>();
            return *(reinterpret_cast<T*>(m_data.m_ptr));
        }

        template <typename T, typename std::enable_if<std::is_same<T, NDShape>::value ||
            std::is_same<T, Axis>::value ||
            std::is_same<T, std::wstring>::value ||
            std::is_same<T, std::vector<DictionaryValue>>::value ||
            std::is_same<T, Dictionary>::value ||
            std::is_same<T, TrainingParameterSchedule<double>>::value ||
            std::is_same<T, NDArrayView>::value>::type* = nullptr>
            T& Value()
        {
            VerifyType<T>();
            return *(reinterpret_cast<T*>(m_data.m_ptr));
        }

        bool HasValue() const
        {
            return m_valueType != Type::None;
        }

        Type ValueType() const
        {
            return m_valueType;
        }

        CNTK_API bool operator==(const DictionaryValue& other) const;
        CNTK_API bool operator!=(const DictionaryValue& other) const;

        friend CNTK_API std::istream& operator>>(std::istream& stream, DictionaryValue& us);
        friend CNTK_API std::ostream& operator<<(std::ostream& stream, const DictionaryValue& us);

        CNTK_API void Save(const std::wstring& filename);
        CNTK_API static DictionaryValue Load(const std::wstring& filename);

    private:
        template <typename T>
        static Type GetValueType()
        {
            static_assert((std::is_same<T, bool>::value ||
                           std::is_same<T, int>::value ||
                           std::is_same<T, size_t>::value ||
                           std::is_same<T, float>::value ||
                           std::is_same<T, double>::value ||
                           std::is_same<T, std::wstring>::value ||
                           std::is_same<T, NDShape>::value ||
                           std::is_same<T, Axis>::value ||
                           std::is_same<T, std::vector<DictionaryValue>>::value ||
                           std::is_same<T, Dictionary>::value ||
                           std::is_same<T, TrainingParameterSchedule<double>>::value ||
                           std::is_same<T, NDArrayView>::value),
                           "Unsupported ValueType");

            if (std::is_same<T, bool>::value)                                      return Type::Bool;
            if (std::is_same<T, int>::value)                                       return Type::Int;
            if (std::is_same<T, size_t>::value)                                    return Type::SizeT;
            if (std::is_same<T, float>::value)                                     return Type::Float;
            if (std::is_same<T, double>::value)                                    return Type::Double;
            if (std::is_same<T, std::wstring>::value)                              return Type::String;
            if (std::is_same<T, NDShape>::value)                                   return Type::NDShape;
            if (std::is_same<T, Axis>::value)                                      return Type::Axis;
            if (std::is_same<T, std::vector<DictionaryValue>>::value)              return Type::Vector;
            if (std::is_same<T, Dictionary>::value)                                return Type::Dictionary;
            if (std::is_same<T, NDArrayView>::value)                               return Type::NDArrayView;
            if (std::is_same<T, TrainingParameterSchedule<double>>::value)          return Type::TrainingParameterSchedule;
        }

        template <typename T>
        void VerifyType() const
        {
            if (GetValueType<T>() != m_valueType)
                RuntimeError("Reading a DictionaryValue as the wrong type; Reading as type %s when actual type is %s", typeid(T).name(), DictionaryValue::TypeName(m_valueType));
        }

        template <typename T>
        CNTK_API void AllocateDataPtr(const T& value);

        template <typename T>
        CNTK_API void FreePtrAsType();

        CNTK_API void FreeDataPtr()
        {
            if (m_valueType == Type::String)
                FreePtrAsType<std::wstring>();
            else if (m_valueType == Type::NDShape)
                FreePtrAsType<NDShape>();
            else if (m_valueType == Type::Axis)
                FreePtrAsType<Axis>();
            else if (m_valueType == Type::Vector)
                FreePtrAsType<std::vector<DictionaryValue>>();
            else if (m_valueType == Type::Dictionary)
                FreePtrAsType<Dictionary>();
            else if (m_valueType == Type::NDArrayView)
                FreePtrAsType<NDArrayView>();
            else if (m_valueType == Type::TrainingParameterSchedule)
                FreePtrAsType<TrainingParameterSchedule<double>>();
        }

        Type m_valueType;

        union ValueData
        {
            bool m_boolean;
            int m_int;
            size_t m_sizeT;
            float m_float;
            double m_double;
            void* m_ptr;
            // these are for use in debugging only
            std::wstring* m_ptrAsString;
            NDShape* m_ptrAsNDShape;
            Axis* m_ptrAsAxis;
            std::vector<DictionaryValue>* m_ptrAsVector;
            Dictionary* m_ptrAsDictionary;
            NDArrayView* m_ptrAsNDArrayView;
            TrainingParameterSchedule<double>* m_ptrAsTrainingParameterSchedule;
        } m_data;

         static const size_t s_version;
    };

    ///
    /// A type denoting a dictionary (keyed by Unicode strings) of serializable values (dynamically typed).
    ///
    class Dictionary
    {
        friend inline void AddConfigString(std::wstringstream& s, const DictionaryValue& value, size_t numIndentationSpaces);
        friend class CompositeMinibatchSource;
        friend class Serializer;
    public:
        CNTK_API Dictionary();
        CNTK_API ~Dictionary();

        template<typename... Args>
        Dictionary(const std::wstring& key, const DictionaryValue& value, Args... args) : Dictionary()
        {
            Add(key, value, args...);
        }

        CNTK_API Dictionary(const Dictionary&);
        CNTK_API Dictionary& operator=(const Dictionary&);

        CNTK_API Dictionary(Dictionary&& other);
        CNTK_API Dictionary& operator=(Dictionary&& other);

        CNTK_API void ShallowCloneTo(Dictionary& to) const;

        CNTK_API DictionaryValue& operator[](const wchar_t* key);
        DictionaryValue& operator[](const std::wstring& key)
        {
            return operator[](key.c_str());
        }

        CNTK_API const DictionaryValue& operator[](const wchar_t* key) const;

        const DictionaryValue& operator[](const std::wstring& key) const
        {
            return operator[](key.c_str());
        }

        template<typename T>
        const T& GetOrElse(const std::wstring& key, const T& elseVal) const
        {
            // TODO: This can be done more efficiently using find().
            if (Contains(key))
            {
                const DictionaryValue& val = operator[](key);
                return val.Value<T>();
            }
            else
                return elseVal;
        }

        CNTK_API bool Contains(const wchar_t* key) const;

        bool Contains(const std::wstring& key) const
        {
            return Contains(key.c_str());
        }

        CNTK_API void Add(const Dictionary& other);

        void Add(const std::wstring& key, const DictionaryValue& value)
        {
            operator[](key.c_str()) = value;
        }

        template<typename... Args>
        void Add(const std::wstring& key, const DictionaryValue& value, Args... args)
        {
            Add(key, value); //insert one
            Add(args...);    //recurse
        }

        CNTK_API bool operator==(const Dictionary& other) const;
        CNTK_API bool operator!=(const Dictionary& other) const;

        typedef std::unordered_map<std::wstring, DictionaryValue>::iterator DictionaryIterator;
        typedef std::unordered_map<std::wstring, DictionaryValue>::const_iterator ConstDictionaryIterator;

        DictionaryIterator begin() { return GetDictionaryData().begin(); }
        DictionaryIterator end()   { return GetDictionaryData().end(); }
        ConstDictionaryIterator begin() const { return GetDictionaryData().begin(); }
        ConstDictionaryIterator end()   const { return GetDictionaryData().end(); }
        ConstDictionaryIterator cbegin() const { return GetDictionaryData().cbegin(); }
        ConstDictionaryIterator cend()   const { return GetDictionaryData().cend(); }

        size_t Size() const { return m_dictionaryData ? m_dictionaryData->size() : 0;  }

        std::unordered_set<std::wstring> Keys()
        {
            std::unordered_set<std::wstring> keys;
            if (m_dictionaryData)
                for (const auto& kv : *m_dictionaryData)
                    keys.insert(kv.first);
            return keys;
        }

        friend CNTK_API std::istream& operator>>(std::istream& stream, Dictionary& us);
        friend CNTK_API std::ostream& operator<<(std::ostream& stream, const Dictionary& us);

        CNTK_API void Save(const std::wstring& filename);
        CNTK_API static Dictionary Load(const std::wstring& filename);

    private: public: // TODO: This must be public to define the FixedSizePoolStorage. How to avoid this?
        struct SharableDict : public enable_strong_shared_ptr<std::unordered_map<std::wstring, DictionaryValue>>, public std::unordered_map<std::wstring, DictionaryValue>
        {
            template<size_t N> friend class FixedSizePoolStorage;
            SharableDict() { }
            SharableDict(const SharableDict& otherMap) : std::unordered_map<std::wstring, DictionaryValue>(otherMap) { }
        };
    private:
        //std::shared_ptr<SharableDict> m_dictionaryData;
        strong_shared_ptr<SharableDict> m_dictionaryData;
        CNTK_API const SharableDict& GetDictionaryData() const;
        CNTK_API       SharableDict& GetDictionaryData();
        static const size_t s_version = 1;// TODO: check this: latest master does not initialize this
    };

    ///
    /// Enumeration type denoting the kind of a symbolic Variable object
    ///
    enum class VariableKind : unsigned char
    {
        Input = 0,
        Output = 1,
        Parameter = 2,
        Constant = 3,
        Placeholder = 4,
    };

    inline const wchar_t* VariableKindName(VariableKind variableKind)
    {
        switch (variableKind)
        {
        case VariableKind::Input:
            return L"Input";
        case VariableKind::Output:
            return L"Output";
        case VariableKind::Parameter:
            return L"Parameter";
        case VariableKind::Constant:
            return L"Constant";
        case VariableKind::Placeholder:
            return L"Placeholder";
        default:
            LogicError("Unknown VariableKind.");
        }
    }

    namespace Internal
    {
        inline std::wstring GenerateUid(std::wstring&& prefix)
        {
            prefix += std::to_wstring(Internal::NewUniqueId());
            return prefix;
        }

        inline std::wstring GenerateUid(const wchar_t * prefix)
        {
            // We chop the prefix to first 2 chars, in order to allow triggering small-string optimization.
            // This is safe (w.r.t. being unique) since a unique count is always appended.
            size_t len = prefix[0] == 0 ? 0 : (prefix[1] == 0 ? 1 : 2);
            return GenerateUid(std::wstring(prefix, len));
        }

        inline std::wstring GenerateUid(VariableKind varKind)
        {
            return GenerateUid(VariableKindName(varKind));
        }

        inline std::wstring GenerateUid(const std::wstring& prefix)
        {
#if 0
            return GenerateUid(prefix.c_str()); // will chop prefix to 2 chars
#else
            return GenerateUid(std::wstring(prefix));
#endif
        }
    }

    typedef Dictionary ParameterInitializer;

    // Forward declarations
    inline Variable PlaceholderVariable(const NDShape& shape, ::CNTK::DataType dataType, const std::wstring& name, const std::vector<Axis>& dynamicAxes = Axis::UnknownDynamicAxes(), bool needsGradient = false, bool isSparse = false);
    inline Variable InputVariable(const NDShape& shape, bool isSparse, ::CNTK::DataType dataType, bool needsGradient, const std::wstring& name, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
    inline class InternalVariable OutputVariable(const NDShape& shape, ::CNTK::DataType dataType, const std::vector<Axis>& dynamicAxes, bool needsGradient, bool isSparse, bool isVolatile, const std::wstring& name = std::wstring());

    ///
    /// Denotes a symbolic entity corresponding to the inputs and outputs of a Function.
    /// Variable and InternalVariable are symbolic and do not represent the actual values,
    /// except for Dynamite, which they represent both a deferred and a realized value.
    /// Also, Variable type is a value type and copies of a Variable object are aliases of the
    /// source Variable object itself and have the same identity.
    ///
    /// The InternalVariable is merely a reference (a shared pointer) to VariableFields,
    /// which contain the actual content. A Variable, in contrast, is an InternalVariable
    /// that holds a ref count to its owning function (in case of an Output).
    /// Variable is to InternalVariable as CompositeFunction to PrimitiveFunction.
    ///
    /// The ownership hierarchy is:
    ///  - InternalVariable : VariableFieldsPtr (shared)
    ///  - Function output : InternalVariable (owned)
    ///  - Variable (user-visible) : (InternalVariable (owned), FunctionPtr (if output; shared))
    ///  - Function inputs : Variable
    /// A Function holds a reference to its output's InternalVariable fields.
    /// When an output is surfaced to users, the ownerships is reversed: Instead of an InternalVariable
    /// (owned), users receive a Variable, which is a tuple that owns both the InternalVariable
    /// and its owning Function.
    /// When such a Variable is used as the input of a Function, the Function also remembers
    /// the full Variable (again, that's an output InternalVariable and its owner).
    /// This way, the graph is fully owned, either by holding its root Function, or a Variable
    /// that holds both the root Function's output and the root Function itself.
    ///
    class InternalVariable : private IDictionarySerializable
    {
        friend bool operator==(const InternalVariable& first, const InternalVariable& second);
        friend class Function;
        friend class CompositeFunction;
        friend class BlockFunction;
        friend class Trainer;
        friend class PrimitiveFunction;
        friend class Utils;
        friend struct VariableFields;
        friend class Invocable;

        template <typename T>
        friend struct std::hash;

        friend class Internal::VariableResolver;

    public:

        ///
        /// Create an 'Output' variable aliasing the output of the specified Function
        /// Throws an exception if called for a Function instance with multiple outputs
        ///
        //CNTK_API InternalVariable(const FunctionPtr& function);
        //CNTK_API InternalVariable(FunctionPtr&& function);

        ///
        /// Implicit conversion to a FunctionPtr; creates a pass through primitive Function
        /// This returns a new CompositeFunction that wraps Owner().   --TODO: ^^is this comment correct?
        ///
        //CNTK_API operator FunctionPtr() const;

        ///
        /// Default constructor for creating an invalid/null InternalVariable instance.
        /// Required for use in an external std::vector container.
        ///
        CNTK_API InternalVariable();
        CNTK_API ~InternalVariable();
        CNTK_API InternalVariable(const InternalVariable&);
        CNTK_API InternalVariable(InternalVariable&&);
        CNTK_API InternalVariable& operator=(const InternalVariable&);
        CNTK_API InternalVariable& operator=(InternalVariable&&);

        ///
        /// Returns the shape of 'this' variable
        ///
        CNTK_API const NDShape& Shape() const;

        ///
        /// Returns the dynamic axes of 'this' variable
        ///
        CNTK_API const std::vector<Axis>& DynamicAxes() const;

        ///
        /// Returns the VariableKind of 'this' variable
        ///
        CNTK_API VariableKind Kind() const;

        ///
        /// Returns a boolean value indicating if 'this' variable denotes sparse data
        ///
        CNTK_API bool IsSparse() const;

        ///
        /// Returns a boolean value indicating if 'this' variable is an Input
        ///
        bool IsInput() const { return Kind() == VariableKind::Input; }

        ///
        /// Returns a boolean value indicating if 'this' variable is an Output
        ///
        bool IsOutput() const { return Kind() == VariableKind::Output; }

        ///
        /// Returns a boolean value indicating if 'this' variable is a Parameter
        ///
        bool IsParameter() const { return Kind() == VariableKind::Parameter; }

        ///
        /// Returns a boolean value indicating if 'this' variable is a Constant
        ///
        bool IsConstant() const { return Kind() == VariableKind::Constant; }

        ///
        /// Returns a boolean value indicating if 'this' variable is a Placeholder
        ///
        bool IsPlaceholder() const { return Kind() == VariableKind::Placeholder; }

        ///
        /// Returns the name of 'this' variable
        ///
        CNTK_API const std::wstring& Name() const;

        ///
        /// Debug helper that updates the name of 'this' variable
        /// This is temporary.
        ///
        CNTK_API void DebugUpdateName(const std::wstring& newName);

        ///
        /// Returns the internally generated unique name of the variable
        /// The uid may be generated on demand upon first call.
        ///
        CNTK_API const std::wstring& Uid() const;

        ///
        /// Returns the a strong reference to the Function object which 'this' variable is an output of.
        /// Returns null when called for a InternalVariable that is not of 'Output' VariableKind.
        ///
        CNTK_API FunctionPtr Owner() const;

        ///
        /// Returns the DataType of the data that 'this' InternalVariable symbolically represents
        ///
        CNTK_API DataType GetDataType() const;

        ///
        /// Returns a boolean value indicating if gradient computation is enabled for this variable.
        ///
        CNTK_API bool NeedsGradient() const;

        ///
        /// Returns a boolean value indicating if this Variable is flagged for inference only (leaf),
        /// or has been inferred as such (non-leaf).
        /// This Variable and any Variable depending on it will have NeedsGradient() return false.
        ///
        CNTK_API bool IsVolatile() const;

        ///
        /// Returns a string representation for this variable.
        ///
        CNTK_API std::wstring AsString() const;

        ///
        /// Returns this InternalVariable's timestamp.
        /// Timestamps are used to determine if a InternalVariable's value is up to date and, if not, computations that depend on this InternalVariable's value will be re-executed.
        ///
        CNTK_API size_t CurrentValueTimeStamp() const;

        ///
        /// In Dynamite, the value of this node is knowable. This computes and returns it.
        /// The requirement is that this InternalVariable is the output of a Function that only
        /// depends on Constants and Parameters but not on Inputs and Placeholders.
        ///
        CNTK_API NDArrayViewPtr Value() const;

        ///
        /// In Dynamite, this computes the gradient of this InternalVariable w.r.t. given leaves.
        /// Beta can be 0 or 1. If 1, gradients are added to.
        /// TODO: Function::grad() allows to pass multiple roots. Does that ever make sense in this context?
        ///
        CNTK_API void Backward(std::unordered_map<Parameter, NDArrayViewPtr>& gradients, double beta = 0.0) const;

        ///
        /// Test whether a variable is non-empty.
        /// Use this to test for a default-constructed variable to denote "None".
        ///
        operator bool() const { return (bool)m_dataFields; }

        ///
        /// For debugging: Return the unique id
        ///
        CNTK_API unsigned int UniqueIdForDebugging() const;
    protected:
        class AutoBatch;
        class Memoizer;
    protected:
#ifdef SWIGPYTHON
    public:
#endif
        CNTK_API InternalVariable(const NDShape& shape, VariableKind varType, ::CNTK::DataType dataType, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, const std::wstring& name, const std::wstring& uid)
            : InternalVariable(shape, varType, dataType, value, needsGradient, dynamicAxes, /*isSparse =*/ false, /*isVolatile=*/false, name, uid)
        {}

    protected:
        //CNTK_API NDArrayViewPtr Value() const;
        CNTK_API void SetValue(const NDArrayViewPtr& value);

        CNTK_API InternalVariable Clone() const;

        void Reset();

    private:
#ifdef SWIGPYTHON
    public:
#endif
        // TODO: get rid of all these overloads
        CNTK_API InternalVariable(const NDShape& shape, bool isSparse, ::CNTK::DataType dataType, bool needsGradient, const std::wstring& name, const std::vector<Axis>& dynamicAxes, const std::wstring& uid)
            : InternalVariable(shape, VariableKind::Input, dataType, nullptr, needsGradient, dynamicAxes, isSparse, /*isVolatile=*/false, name, uid)
        {}

        // TODO: This should be a private but if not made public, the python bindings build complains about an unresolved external
        // Probably due the above ctor being a public method in SWIG codegen
    public:
        CNTK_API InternalVariable(const NDShape& shape, VariableKind varType, ::CNTK::DataType dataType, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, bool isVolatile, const std::wstring& name, const std::wstring& uid);

        // simplified version for Dynamite
        CNTK_API InternalVariable(NDShape&& shape, VariableKind varType, ::CNTK::DataType dataType, bool needsGradient, bool isSparse, bool isVolatile);

    private:
        PrimitiveFunctionPtr OutputOwner() const; // for Outputs only; can never return null
        bool OwnerIs(const Function* f) const; // faster than saying Owner() == ...

        //CNTK_API const InternalVariable& BlockFunctionVariableMapping() const; // [fseide] has been moved to BlockFunction

        CNTK_API virtual Dictionary Serialize() const override;

        virtual size_t CurrentVersion() const override { return s_serializationVersion; }

        template <typename ElementType>
        static NDArrayViewPtr CreateValueFromParameterInitializer(const NDShape& shape, const ParameterInitializer& initConfig, const DeviceDescriptor& device);

        CNTK_API static InternalVariable Deserialize(const Dictionary& dictionary, const ::CNTK::DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());

        void SetOwner(const std::weak_ptr<PrimitiveFunction>& ownerFunction);
        void SetOwner(std::weak_ptr<PrimitiveFunction>&& ownerFunction);

    private:
#if defined(SWIGCSHARP) || defined(SWIGJAVA)
    public:
        // TODO: a better way to get hash value?
        size_t GetHashValue()
        {
            return std::hash<const void *>()(m_dataFields.get());
        }
#endif

    protected:
        VariableFieldsPtr m_dataFields;
        static const size_t s_serializationVersion = 1;
    };

    // this is the exported version
    class Variable : public InternalVariable
    {
        //friend bool operator==(const Variable& first, const Variable& second);
        friend class Function;
        friend class CompositeFunction;
        friend class BlockFunction;
        friend class Trainer;
        friend class PrimitiveFunction;
        friend class Utils;
        friend struct VariableFields;
        friend class Invocable;

        //template <typename T>
        //friend struct std::hash;

        friend class Internal::VariableResolver;

#ifndef SWIG
    private:
        friend inline Variable PlaceholderVariable(const NDShape& shape, ::CNTK::DataType dataType, const std::wstring& name, const std::vector<Axis>& dynamicAxes /*=Axis::UnknownDynamicAxes()*/, bool needsGradient /*= false*/, bool isSparse /*= false*/);
        friend inline Variable InputVariable(const NDShape& shape, bool isSparse, ::CNTK::DataType dataType, bool needsGradient, const std::wstring& name, const std::vector<Axis>& dynamicAxes /*= Axis::DefaultInputVariableDynamicAxes()*/);
        friend inline InternalVariable OutputVariable(const NDShape& shape, ::CNTK::DataType dataType, const std::vector<Axis>& dynamicAxes, bool needsGradient, bool isSparse, bool isVolatile, const std::wstring& name /*= L""*/);
#endif

    public:

        ///
        /// Construct from InternalVariable
        ///
        // The direct construction from InternalVariable should not happen, but
        // not all code has been updated. That code uses the ", true" version, to
        // make it easy to detect.
        //CNTK_API Variable(const InternalVariable& variable);
        //CNTK_API Variable(InternalVariable&& variable);
        CNTK_API Variable(const InternalVariable& variable, bool); // special version, to be fixed some day
        CNTK_API Variable(InternalVariable&& variable, bool);
        CNTK_API Variable(const Parameter& variable);
        CNTK_API Variable(Parameter&& variable);
        CNTK_API Variable(const class Constant& variable);
        CNTK_API Variable(Constant&& variable);

        ///
        /// Create an 'Output' variable aliasing the output of the specified Function
        /// The resulting Variable holds a ref count to the function.
        /// Throws an exception if called for a Function instance with multiple outputs.
        ///
        CNTK_API Variable(const FunctionPtr& function);
        CNTK_API Variable(FunctionPtr&& function);

        ///
        /// Implicit conversion to a FunctionPtr; creates a pass through primitive Function
        /// This returns a new CompositeFunction that wraps Owner().   --TODO: ^^is this comment correct?
        ///
        CNTK_API operator FunctionPtr() const;

        ///
        /// Default constructor for creating an invalid/null Variable instance.
        /// Required for use in a std::vector container.
        ///
        CNTK_API Variable();
        CNTK_API ~Variable();
        CNTK_API Variable(const Variable&);
        CNTK_API Variable(Variable&&);
        CNTK_API Variable& operator=(const Variable&);
        CNTK_API Variable& operator=(Variable&&);

    protected:
#ifdef SWIGPYTHON
    public:
#endif
        CNTK_API Variable(const NDShape& shape, VariableKind varType, ::CNTK::DataType dataType, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, const std::wstring& name, const std::wstring& uid)
            : Variable(shape, varType, dataType, value, needsGradient, dynamicAxes, /*isSparse =*/ false, name, uid)
        {}

    protected:
        //CNTK_API NDArrayViewPtr Value() const;
        //CNTK_API void SetValue(const NDArrayViewPtr& value);

    private:
#ifdef SWIGPYTHON
    public:
#endif
        CNTK_API Variable(const NDShape& shape, bool isSparse, ::CNTK::DataType dataType, bool needsGradient, const std::wstring& name, const std::vector<Axis>& dynamicAxes, const std::wstring& uid)
            : Variable(shape, VariableKind::Input, dataType, nullptr, needsGradient, dynamicAxes, isSparse, name, uid)
        {}

        // TODO: This should be a private but if not made public, the python bindings build complains about an unresolved external
        // Probably due the above ctor being a public method in SWIG codegen
    public:
        CNTK_API Variable(const NDShape& shape, VariableKind varType, ::CNTK::DataType dataType, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, const std::wstring& name, const std::wstring& uid);

        // simplified version for Dynamite
        CNTK_API Variable(NDShape&& shape, VariableKind varType, ::CNTK::DataType dataType, bool needsGradient, bool isSparse);

        CNTK_API Variable Clone() const;

        ///
        /// Index the last axis. This maps to Slice().
        ///
        CNTK_API Variable operator[](size_t index) const;

        ///
        /// Length of the last axis.
        ///
        CNTK_API size_t size() const;

    protected:
        Variable(const InternalVariable& other, const ConstFunctionPtr& composite, const ConstPrimitiveFunctionPtr& primitive);
        Variable(const InternalVariable& other, ConstFunctionPtr&& composite, const ConstPrimitiveFunctionPtr& primitive);

        //Variable CompositePreservingCopy(const ConstFunctionPtr& composite) const;
        //Variable CompositePreservingCopy(ConstFunctionPtr&& composite) const;
        //Variable NonCompositePreservingCopy() const;
        Variable CompositePreservingCopy(const ConstFunctionPtr& composite) const
        {
            return Variable((const InternalVariable&)*this, composite, m_acyclicOutputPrimitiveReference);
        }
        Variable CompositePreservingCopy(ConstFunctionPtr&& composite) const
        {
            return Variable((const InternalVariable&)*this, std::move(composite), m_acyclicOutputPrimitiveReference);
        }
        Variable NonCompositePreservingCopy() const
        {
            return Variable((const InternalVariable&)*this, ConstFunctionPtr(), m_acyclicOutputPrimitiveReference);
        }
        static Variable CompositePreservingCopy(const InternalVariable& other, ConstFunctionPtr&& composite);

        void Reset();

    private:
#if defined(SWIGCSHARP) || defined(SWIGJAVA)
    //public:
    //    // TODO: a better way to get hash value?
    //    size_t GetHashValue()
    //    {
    //        return std::hash<const void *>()(m_dataFields.get());
    //    }
#endif

    protected:
    public: // for now
        ConstFunctionPtr m_outputComposite; // Outputs() returns copies with this set.
    protected:
#ifdef DYNAMITE_ONLY
        // Note: This ^^ is called outputComposite, but there is no assumption that it actually is a composite. Maybe we can even merge this with vv.
#endif
        friend class InternalVariable::AutoBatch; // TODO: remove this, and instead have the correct constructor that sets this field up automatically
        ConstPrimitiveFunctionPtr m_acyclicOutputPrimitiveReference; // Output: ref to Primitive if known to be acyclic.
        // for debugging:
        const struct { NDShapeDimension dims[4]; }* m_shapeDims = nullptr; // keep a reference to underlying VariableFields that shows nicely in the debugger
    };

    // TODO: Variable equality should be based on uids.
    inline bool operator==(const InternalVariable& first, const InternalVariable& second)
    {
        return first.m_dataFields == second.m_dataFields;
    }

    inline bool operator!=(const InternalVariable& first, const InternalVariable& second)
    {
        return !(first == second);
    }

    inline bool operator==(const Variable& first, const Variable& second)
    {
        return (const InternalVariable&) first == (const InternalVariable&)second;
    }

    inline bool operator!=(const Variable& first, const Variable& second)
    {
        return !(first == second);
    }

    inline bool operator==(const Variable& first, const InternalVariable& second)
    {
        return (const InternalVariable&)first == second;
    }

    inline bool operator!=(const Variable& first, const InternalVariable& second)
    {
        return !(first == second);
    }

    inline bool operator==(const InternalVariable& first, const Variable& second)
    {
        return first == (const InternalVariable&)second;
    }

    inline bool operator!=(const InternalVariable& first, const Variable& second)
    {
        return !(first == second);
    }

    ///
    /// Create a Placeholder variable to be used as a temporary/placeholder input to a Function.
    /// All placeholder inputs of a Function must be replaced with non-placeholder Variables before Forward evaluation of the Function.
    ///
    inline Variable PlaceholderVariable(const NDShape& shape, ::CNTK::DataType dataType, const std::wstring& name, const std::vector<Axis>& dynamicAxes /*=Axis::UnknownDynamicAxes()*/, bool needsGradient /*= false*/, bool isSparse /*= false*/)
    {
        auto varKind = VariableKind::Placeholder;
        return Variable(shape, varKind, dataType, nullptr, needsGradient, dynamicAxes, isSparse, name, Internal::GenerateUid(varKind));
    }

    ///
    /// Create a Placeholder variable to be used as a temporary/placeholder input to a Function.
    /// All placeholder inputs of a Function must be replaced with non-placeholder Variables before Forward evaluation of the Function.
    ///
    inline Variable PlaceholderVariable(const NDShape& shape, const std::wstring& name, const std::vector<Axis>& dynamicAxes)
    {
        return PlaceholderVariable(shape, DataType::Unknown, name, dynamicAxes);
    }

    ///
    /// Create a Placeholder variable to be used as a temporary/placeholder input to a Function.
    /// All placeholder inputs of a Function must be replaced with non-placeholder Variables before Forward evaluation of the Function.
    ///
    inline Variable PlaceholderVariable(const NDShape& shape, const std::vector<Axis>& dynamicAxes = Axis::UnknownDynamicAxes())
    {
        return PlaceholderVariable(shape, L"", dynamicAxes);
    }

    ///
    /// Create a Placeholder variable to be used as a temporary/placeholder input to a Function.
    /// All placeholder inputs of a Function must be replaced with non-placeholder Variables before Forward evaluation of the Function.
    ///
    inline Variable PlaceholderVariable(const std::wstring& name = std::wstring())
    {
        return PlaceholderVariable(NDShape::Unknown(), name, Axis::UnknownDynamicAxes());
    }

    ///
    /// Create an 'Input' Variable denoting sparse data and specify if gradients are to be computed for this input
    ///
    inline Variable InputVariable(const NDShape& shape, bool isSparse, ::CNTK::DataType dataType, bool needsGradient, const std::wstring& name /*= L""*/, const std::vector<Axis>& dynamicAxes /*= Axis::DefaultInputVariableDynamicAxes()*/)
    {
        return Variable(shape, isSparse, dataType, needsGradient, name, dynamicAxes, Internal::GenerateUid(VariableKind::Input));
    }

    ///
    /// Create an 'Input' Variable and specify if gradients are to be computed for this input
    ///
    inline Variable InputVariable(const NDShape& shape, ::CNTK::DataType dataType, bool needsGradient, const std::wstring& name = std::wstring(), const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes())
    {
        return InputVariable(shape, /*isSparse =*/ false, dataType, needsGradient, name, dynamicAxes);
    }

    ///
    /// Create an 'Input' Variable.
    ///
    inline Variable InputVariable(const NDShape& shape, DataType dataType, const std::wstring& name, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes())
    {
        return InputVariable(shape, dataType, /*needsGradient =*/ false, name, dynamicAxes);
    }

    ///
    /// Create an 'Input' Variable.
    ///
    inline Variable InputVariable(const NDShape& shape, DataType dataType, const wchar_t* name, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes())
    {
        return InputVariable(shape, dataType, std::wstring(name), dynamicAxes);
    }

    ///
    /// Create an 'Input' Variable.
    ///
    inline Variable InputVariable(const NDShape& shape, DataType dataType, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes())
    {
        return InputVariable(shape, dataType, L"", dynamicAxes);
    }

    ///
    /// Create an 'Input' Variable denoting sparse data.
    ///
    inline Variable InputVariable(const NDShape& shape, bool isSparse, ::CNTK::DataType dataType, const std::wstring& name, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes())
    {
        return InputVariable(shape, isSparse, dataType, /*needsGradient =*/ false, name, dynamicAxes);
    }

    ///
    /// Create an 'Input' Variable denoting sparse data.
    ///
    inline Variable InputVariable(const NDShape& shape, bool isSparse, ::CNTK::DataType dataType, const wchar_t* name, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes())
    {
        return InputVariable(shape, isSparse, dataType, std::wstring(name), dynamicAxes);
    }

    ///
    /// Create an 'Input' Variable denoting sparse data.
    ///
    inline Variable InputVariable(const NDShape& shape, bool isSparse, ::CNTK::DataType dataType, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes())
    {
        return InputVariable(shape, isSparse, dataType, L"", dynamicAxes);
    }

#if 0 // OutputVariables are internal, so always call the full constructor
    ///
    /// Create an 'Output' variable
    ///
    inline InternalVariable OutputVariable(const NDShape& shape, ::CNTK::DataType dataType, const std::vector<Axis>& dynamicAxes, const std::wstring& name = std::wstring())
    {
        return OutputVariable(shape, dataType, dynamicAxes, /*needsGradient =*/ true, name);
    }

    ///
    /// Create an 'Output' variable
    ///
    inline InternalVariable OutputVariable(const NDShape& shape, ::CNTK::DataType dataType, const std::vector<Axis>& dynamicAxes, bool needsGradient, const std::wstring& name /*= L""*/)
    {
        return InternalVariable(shape, VariableKind::Output, dataType, nullptr, needsGradient, dynamicAxes, /*isSparse =*/ false, name, std::wstring());// Internal::GenerateUid(VariableKind::Output));
    }
#endif

    ///
    /// Create an 'Output' variable
    ///
    // TODO: we should reorder the arguments, to separate type from gradient flags
    inline InternalVariable OutputVariable(const NDShape& shape, ::CNTK::DataType dataType, const std::vector<Axis>& dynamicAxes, bool needsGradient, bool isSparse, bool isVolatile, const std::wstring& name /*= L""*/)
    {
        return InternalVariable(shape, VariableKind::Output, dataType, nullptr, needsGradient, dynamicAxes, isSparse, isVolatile, name, std::wstring());// Internal::GenerateUid(VariableKind::Output));
    }

    static const int SentinelValueForInferParamInitRank = std::numeric_limits<int>::max();
    static const int DefaultParamInitScale = 1;
    static const int DefaultParamInitOutputRank = 1;
    static const int DefaultParamInitFilterRank = 0;

    CNTK_API ParameterInitializer ConstantInitializer(double value = 0.0);
    CNTK_API ParameterInitializer UniformInitializer(double scale, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
    CNTK_API ParameterInitializer NormalInitializer(double scale, int outputRank = SentinelValueForInferParamInitRank, int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
    CNTK_API ParameterInitializer XavierInitializer(double scale = DefaultParamInitScale, int outputRank = SentinelValueForInferParamInitRank, int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
    CNTK_API ParameterInitializer GlorotUniformInitializer(double scale = DefaultParamInitScale, int outputRank = SentinelValueForInferParamInitRank, int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
    CNTK_API ParameterInitializer GlorotNormalInitializer(double scale = DefaultParamInitScale, int outputRank = SentinelValueForInferParamInitRank, int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
    CNTK_API ParameterInitializer HeUniformInitializer(double scale = DefaultParamInitScale, int outputRank = SentinelValueForInferParamInitRank, int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
    CNTK_API ParameterInitializer HeNormalInitializer(double scale = DefaultParamInitScale, int outputRank = SentinelValueForInferParamInitRank, int filterRank = SentinelValueForInferParamInitRank, unsigned long seed = SentinelValueForAutoSelectRandomSeed);
    CNTK_API ParameterInitializer BilinearInitializer(size_t kernelWidth, size_t kernelHeight);
    CNTK_API ParameterInitializer RandomInitializerWithRank(const ParameterInitializer& initializer, int outputRank, int filterRank);
    CNTK_API ParameterInitializer TruncatedNormalInitializer(double scale = DefaultParamInitScale, unsigned long seed = SentinelValueForAutoSelectRandomSeed);

    ///
    /// Denotes Parameter inputs of a Function.
    ///
    class Parameter final : public InternalVariable
    {
        template <typename T>
        friend struct std::hash;

        friend class Internal::VariableResolver;

    public:
        ///
        /// Construct a parameter whose initial contents are a copy of the specified 'value'
        ///
        explicit Parameter(const NDArrayViewPtr& value, const std::wstring& name = std::wstring())
            : Parameter(value, name, Internal::GenerateUid(VariableKind::Parameter))
        {}
        // note: unlike other Variables, we must not generate the uid on demand for parameters, to ensure they are always the same

        // TODO: Constructor to move a specified NDArrayView value

        ///
        /// Construct a parameter of specified shape whose contents are initialized with the specified 'initValue'
        ///
        template<typename ElemType>
        Parameter(const NDShape& shape, ElemType initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = std::wstring())
            : Parameter(shape, AsDataType<ElemType>(), ConstantInitializer(initValue), device, name)
        {}

        ///
        /// Construct a constant of specified shape whose contents are initialized with the specified 'initValue'
        ///
        Parameter(const NDShape& shape, DataType dataType, double initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = std::wstring())
            : Parameter(shape, dataType, ConstantInitializer(initValue), device, name)
        {}

        ///
        /// Construct a constant of specified shape whose contents are initialized using the specified initializer
        ///
        CNTK_API Parameter(const NDShape& shape, DataType dataType, const ParameterInitializer& initializer, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = std::wstring());

        ///
        /// DownCast a Variable to a Parameter. Only allowed if the VariableKind is Parameter and throws an exception otherwise.
        ///
        explicit Parameter(const InternalVariable& variable)
            : InternalVariable(variable)
        {
            if (!IsParameter())
                InvalidArgument("A non-parameter Variable '%S' cannot be converted to a Parameter.", variable.AsString().c_str());
        }

        ///
        /// Get the value of 'this' parameter
        ///
        // TODO: Is this needed? Why not call base directly? It is public.
        NDArrayViewPtr Value() const
        {
            return InternalVariable::Value();
        }

        ///
        /// Copies the contents of the 'value' NDArrayView into the view backing 'this'
        /// parameter's value. The shapes of both views must be identical.
        ///
        CNTK_API void SetValue(const NDArrayViewPtr& value)
        {
            InternalVariable::SetValue(value);
            RecordValueUpdate();
        }

        CNTK_API void RecordValueUpdate();

        ///
        /// Redirect an existing Parameter's m_value field to be a reference to another.
        /// This is a bad hack for debugging only, not fully supported everywhere.
        /// TODO: If this is not used anymore then remove it.
        ///
        //CNTK_API void TieValueWith(const Parameter& other);

    private:
        explicit Parameter(const NDArrayViewPtr& value, const std::wstring& name, const std::wstring& uid)
            : InternalVariable(value->Shape(), VariableKind::Parameter, value->GetDataType(), value, true, {}, name, uid)
        {
            if (value->IsReadOnly())
                InvalidArgument("Parameter cannot be constructed from a read-only NDArrayView value; you can create a non read-only clone of the value and use that instead!");
        }
    };

    // Implementation note: The InternalVariable type is a value type and not polymorphic in nature.
    // However we have a couple of derivatives of the type to extend the base interface and thus we ensure that the derived types do not have additional fields.
    // This check is weak in that the derives types may sneak in some additional fields if the base type had some padding at the end, without changing the object size
    // but it should be good enough for catching any accidental addition of fields.
    static_assert(sizeof(Parameter) == sizeof(InternalVariable), "The Parameter type should not have any data fields beyond what its base type 'Variable' has.");

    ///
    /// Denotes Constant inputs of a Function.
    ///
    class Constant final : public InternalVariable
    {
        template <typename T>
        friend struct std::hash;

        friend class Internal::VariableResolver;

    public:
        ///
        /// Construct a Constant whose initial contents are a copy of the specified value
        ///
        Constant(const NDArrayViewPtr& value, const std::wstring& name = std::wstring())
            : Constant(value, /*isVolatile=*/false, name, std::wstring())// Internal::GenerateUid(VariableKind::Constant))
        {}

        ///
        /// Construct a Constant whose initial contents are a copy of the specified value
        ///
        Constant(const NDArrayViewPtr& value, bool isVolatile, const std::wstring& name = std::wstring())
            : Constant(value, isVolatile, name, std::wstring())// Internal::GenerateUid(VariableKind::Constant))
        {}

        // TODO: Constructor to move a specified NDArrayView value

        ///
        /// Construct a constant of specified shape whose contents are initialized with the specified 'initValue'
        ///
        template<typename ElemType>
        Constant(const NDShape& shape, ElemType initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = std::wstring())
            : Constant(shape, AsDataType<ElemType>(), /*isVolatile=*/false, ConstantInitializer(initValue), device, name)
        {}

        ///
        /// Construct a constant of specified shape whose contents are initialized with the specified 'initValue'
        ///
        Constant(const NDShape& shape, DataType dataType, double initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = std::wstring())
            : Constant(shape, dataType, /*isVolatile=*/false, ConstantInitializer(initValue), device, name)
        {}

        ///
        /// Construct a constant of specified shape whose contents are initialized with the specified 'initValue'
        /// This overload has the isVolatile flag.
        ///
        Constant(const NDShape& shape, DataType dataType, bool isVolatile, double initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = std::wstring())
            : Constant(shape, dataType, isVolatile, ConstantInitializer(initValue), device, name)
        {}

        ///
        /// Create a clone of 'this' constant with the specified DataType.
        /// This only supports converting from a lower precision type to a higher precision type (e.g. DataType::Float to DataType::Double)
        ///
        CNTK_API Constant CloneAs(DataType dataType) const;

        ///
        /// Create a scalar constant. The specified value is cast to the specified DataType
        ///
        static inline ::CNTK::Constant Scalar(::CNTK::DataType dataType, double value, const ::CNTK::DeviceDescriptor& device = DeviceDescriptor::CPUDevice())
        {
            return Constant({}, dataType, value, device);
        }

        ///
        /// Create a scalar constant. The specified value is cast to the specified DataType
        ///
        template<typename ElementType>
        static inline ::CNTK::Constant Scalar(ElementType value, const ::CNTK::DeviceDescriptor& device = DeviceDescriptor::CPUDevice())
        {
            return Constant({}, value, device);
        }

        ///
        /// DownCast a Variable to a Constant. Only allowed if the VariableKind is Constant and throws an exception otherwise.
        ///
        explicit Constant(const InternalVariable& variable)
            : InternalVariable(variable)
        {
            if (!IsConstant())
                InvalidArgument("A non-constant Variable '%S' being converted to a Constant.", variable.AsString().c_str());
        }

        ///
        /// Get the value of 'this' Constant
        ///
        // Cf. Parameter: We may not need this method, since base is public.
        NDArrayViewPtr Value() const
        {
            return InternalVariable::Value();
        }

        ///
        /// Copies the contents of the 'value' NDArrayView into the view backing 'this'
        /// Constant's value. The shapes of both views must be identical.
        ///
        CNTK_API void SetValue(const NDArrayViewPtr& value);
        CNTK_API void RecordValueUpdate();

    private:
        Constant(const NDArrayViewPtr& value, bool isVolatile, const std::wstring& name, const std::wstring& uid)
            : InternalVariable(value->Shape(), VariableKind::Constant, value->GetDataType(), value, /*needsGradient=*/false, {}, value->GetStorageFormat() != StorageFormat::Dense, isVolatile, name, uid)
        {}

        ///
        /// Construct a constant of specified shape whose contents are initialized using the specified initializer
        ///
        CNTK_API Constant(const NDShape& shape, DataType dataType, bool isVolatile, const ParameterInitializer& initializer, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = std::wstring());
    };

    // Implementation note: The Variable type is a value type and not polymorphic in nature.
    // However we have a couple of derivatives of the type to extend the base interface and thus we ensure that the derived types do not have additional fields.
    // This check is weak in that the derives types may sneak in some additional fields if the base type had some padding at the end, without changing the object size
    // but it should be good enough for catching any accidental addition of fields.
    static_assert(sizeof(Constant) == sizeof(InternalVariable), "The Constant type should not have any data fields beyond what its base type 'Variable' has.");
}

namespace std {

    template <> struct hash<::CNTK::NDShape>
    {
        size_t operator()(const ::CNTK::NDShape& x) const
        {
            return std::hash<std::wstring>()(x.AsString());
        }
    };

    // TODO: Variable hash should be based on uid.
    // TODO: Actually not; uids are expensive. The pointer is fine.
    template <> struct hash<::CNTK::Variable>
    {
        size_t operator()(const ::CNTK::Variable& x) const
        {
            return std::hash<const void*>()(x.m_dataFields.get());
        }
    };

    template <> struct hash<::CNTK::InternalVariable>
    {
        size_t operator()(const ::CNTK::InternalVariable& x) const
        {
            return std::hash<const void*>()(x.m_dataFields.get());
        }
    };

    template <> struct hash<::CNTK::Parameter>
    {
        size_t operator()(const ::CNTK::Parameter& x) const
        {
            return std::hash<::CNTK::InternalVariable>()(x);
        }
    };

    template <> struct hash<::CNTK::Constant>
    {
        size_t operator()(const ::CNTK::Constant& x) const
        {
            return std::hash<::CNTK::InternalVariable>()(x);
        }
    };
}

namespace CNTK
{
    ///
    /// Denotes a multi-dimensional array with an optional mask and is the actual data fed into or produced from a computation.
    /// The mask is typically lower dimensionality than the data, meaning data is masked in coarse individual sample units where
    /// sample shape is data.Shape().SubShape(0, data.Shape().Rank() - mask.Shape().Rank)
    /// Also, note that the size of the data's trailing mask.Shape().Rank() dimensions must match the mask shape dimensions.
    ///
    class Value : public std::enable_shared_from_this<Value>
    {
        friend class Utils;

    public:

        ///
        /// a special index for one hot to indicate zero vector
        ///
        static const size_t OneHotSkip = (size_t)-1;

        ///
        /// A multi-dimensional value with no mask.
        ///
        explicit CNTK_API Value(const NDArrayViewPtr& data);

        ///
        /// A multi-dimensional value with an associated mask.
        ///
        CNTK_API Value(const NDArrayViewPtr& data, const NDMaskPtr& mask);

        ///
        /// Create a new Value object containing a collection of variable length sequences.
        /// The sequenceStartFlags argument allows specifying for each sequence whether that sequence is a
        /// a new sequence or continuation of a previous sequence at the same index in the
        /// sequences vector from a previous call to this method.
        /// The created Value object contains a copy of the specified 'sequences' data.
        ///
        template <typename ElementType>
        CNTK_API static ValuePtr Create(const NDShape& sampleShape, const std::vector<std::vector<ElementType>>& sequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly = false);

        ///
        /// Create a new Value object containing a collection of variable length sequences.
        /// The created Value object contains a copy of the specified 'sequences' data.
        ///
        template <typename ElementType>
        static ValuePtr Create(const NDShape& sampleShape, const std::vector<std::vector<ElementType>>& sequences, const DeviceDescriptor& device, bool readOnly = false)
        {
            return Create(sampleShape, sequences, {}, device, readOnly);
        }

        ///
        /// Create a new Value object containing a collection of variable length sequences.
        ///
        CNTK_API static ValuePtr Create(const NDShape& sampleShape, const std::vector<NDArrayViewPtr>& sequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly, bool createNewCopy);

        ///
        /// Create a new Value object containing a collection of variable length sequences.
        /// The created Value object contains a copy of the specified 'sequences' data.
        ///
        static ValuePtr Create(const NDShape& sampleShape, const std::vector<NDArrayViewPtr>& sequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly = false)
        {
            return Create(sampleShape, sequences, sequenceStartFlags, device, readOnly, /*createNewCopy =*/ false);
        }

        ///
        /// Create a new Value object containing a collection of variable length sequences.
        /// The created Value object contains a copy of the specified 'sequences' data.
        ///
        static ValuePtr Create(const NDShape& sampleShape, const std::vector<NDArrayViewPtr>& sequences, const DeviceDescriptor& device, bool readOnly = false)
        {
            return Create(sampleShape, sequences, {}, device, readOnly);
        }

        ///
        /// Create a new Value object containing a collection of variable length sequences of one hot vectors
        /// The created Value object contains a copy of the specified 'sequences' data.
        ///
        template <typename ElementType>
        CNTK_API static ValuePtr Create(const NDShape& sampleShape, const std::vector<std::vector<size_t>>& oneHotSequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly = false);

        ///
        /// Create a new Value object containing a collection of variable length sequences of one hot vectors
        /// The created Value object contains a copy of the specified 'sequences' data.
        ///
        template <typename ElementType>
        static ValuePtr Create(const NDShape& sampleShape, const std::vector<std::vector<size_t>>& oneHotSequences, const DeviceDescriptor& device, bool readOnly = false)
        {
            return Create<ElementType>(sampleShape, oneHotSequences, {}, device, readOnly);
        }

        ///
        /// Create a new Value object containing a collection of variable length sequences of one hot vectors
        /// The created Value object contains a copy of the specified 'sequences' data.
        ///
        template <typename ElementType>
        static ValuePtr Create(size_t dimension, const std::vector<std::vector<size_t>>& oneHotSequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly = false)
        {
            return Create<ElementType>(NDShape({ dimension }), oneHotSequences, sequenceStartFlags, device, readOnly);
        }

        ///
        /// Create a new Value object containing a collection of variable length sequences of one hot vectors
        /// The created Value object contains a copy of the specified 'sequences' data.
        ///
        template <typename ElementType>
        static ValuePtr Create(size_t dimension, const std::vector<std::vector<size_t>>& oneHotSequences, const DeviceDescriptor& device, bool readOnly = false)
        {
            return Create<ElementType>(dimension, oneHotSequences, {}, device, readOnly);
        }

        ///
        /// Creates a new Value object containing a batch of samples.
        /// The number of samples in the batch is the number of elements in batch divided by the size of shape (A runtime error occurs if the remainder is not zero).
        /// The created Value object contains a copy of the specified data in batch.
        /// Parameters:
        ///     sampleShape: the tensor shape of the Value object.
        ///     batchData: the data to be contained in the Value object.
        ///     device: on which device the Value object should be created.
        ///     readOnly: the Value object is read-only if this flag is true.
        ///
        template <typename ElementType>
        CNTK_API static ValuePtr CreateBatch(const NDShape& sampleShape, const std::vector<ElementType>& batchData, const DeviceDescriptor& device, bool readOnly = false);

        ///
        /// Creates a new Value object containing a sequence of samples.
        /// The created Value object contains a copy of the specified sequence data.
        /// The sequenceStartFlag specifies wehther this sequence is a new sequence or continuation of a previous sequence at the same index in the sequences list from a previous call to this method.The sequence length is the number of elements in sequence divided by the size of shape.
        /// (A runtime error occurs if the remainder is not zero).
        /// Parameters:
        ///     sampleShape: the tensor shape of the Value.
        ///     sequenceData: the data to be contained in the Value.
        ///     sequenceStartFlag: true indicates that it is a new sequence. false means a continuation of a previous sequence.
        ///     device: on which device the Value object should be created.
        ///     readOnly: the Value is read-only if this flag is true.
        ///
        template <typename ElementType>
        CNTK_API static ValuePtr CreateSequence(const NDShape& sampleShape, const std::vector<ElementType>& sequenceData, bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly = false);

        ///
        /// Creates a new Value object containing a sequence of samples.
        /// The created Value object contains a copy of the specified data in sequence.
        /// The sequence length is the number of elements in sequence divided by the size of shape(A runtime error occurs if the remainder is not zero).
        /// The created sequece is a new sequence.
        /// Parameters:
        ///     sampleShape: the tensor shape of the Value.
        ///     sequenceData: the data to be contained in the Value.
        ///     device: on which device the Value object should be created.
        ///     readOnly: the Value is read-only if this flag is true.
        ///
        template <typename ElementType>
        static ValuePtr CreateSequence(const NDShape& sampleShape, const std::vector<ElementType>& sequenceData, const DeviceDescriptor& device, bool readOnly = false)
        {
            return CreateSequence(sampleShape, sequenceData, true, device, readOnly);
        }

        ///
        /// Creates a new Value object containing a batch of variable length sequences.
        /// The created Value object contains a copy of the specified data in batchOfSequences.
        /// The number of sequences in the batch is the size of batchOfSequences.
        /// The length of each sequence is the number of elements in the corresponding sequence of batchOfSequences divided by the size of shape.
        /// (A runtime error occurs if the remainder is not zero).
        /// Parameters:
        ///     sampleShape: the tensor shape of the Value.
        ///     batchOfSequences: the data to be stored in the Value.The outer vector represents a collection of sequences with variable length, and the inner vector represents each individual sequence.
        ///     sequenceStartFlags: A collection of boolean value. Each element represent whether the correspoinding sequence in batchOfSequences is a new sequence (in case of true) or a continuation of a previous sequence (in case of false).
        ///     device: on which device the Value should be created.
        ///     readOnly: the Value is read-only if this flag is true.
        ///
        template <typename ElementType>
        static ValuePtr CreateBatchOfSequences(const NDShape& sampleShape, const std::vector<std::vector<ElementType>>& batchOfSequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly = false)
        {
            return Create(sampleShape, batchOfSequences, sequenceStartFlags, device, readOnly);
        }

        ///
        /// Creates a new Value object containing a batch of variable length sequences.
        /// The created Value object contains a copy of the specified data in batchOfSequences.
        /// The number of sequences in the batch is the size of batchOfSequences.
        /// The length of each sequence is the number of elements in the corresponding sequence of batchOfSequences divided by the size of shape.
        /// (A runtime error occurs if the remainder is not zero).
        /// Each sequence in batchOfSequences is a new sequence.
        /// Parameters:
        ///     sampleShape: the tensor shape of the Value.
        ///     batchOfSequences: the data to be stored in the Value.The outer vector represents a collection of sequences with variable length, and the inner vector represents each individual sequence.
        ///     device: on which device the Value should be created.
        ///     readOnly: the Value is read-only if this flag is true.
        ///
        template <typename ElementType>
        static ValuePtr CreateBatchOfSequences(const NDShape& sampleShape, const std::vector<std::vector<ElementType>>& batchOfSequences, const DeviceDescriptor& device, bool readOnly = false)
        {
            return Create(sampleShape, batchOfSequences, {}, device, readOnly);
        }

        ///
        /// Creates a new Value object containing a batch of samples.
        /// Each sample is represented by an index value that points to the non-zero value in the one-hot vector of dimension elements.
        /// The number of samples in the batch is the number of elements in batch.
        /// Parameters:
        ///     ElementType: data type of the created Value object. Currently, float and double are supported.
        ///     dimension: the size of dimension of the one-hot vector.
        ///     batchData: the collection of indexes representing the batch of samples.
        ///     device: on which device the Value object should be created.
        ///     readOnly: the Value is read-only if this flag is true.
        ///
        template <typename ElementType>
        CNTK_API static ValuePtr CreateBatch(size_t dimension, const std::vector<size_t>& batchData, const DeviceDescriptor& device, bool readOnly = false);

        ///
        /// Creates a new Value object containing a sequence of samples.
        /// Each sample is represented by an index value that points to the non-zero value in the one-hot vector of dimension elements.
        /// The sequenceStartFlag specifies wehther this sequence is a new sequence or continuation of a previous sequence at the same index in the sequences list from a previous call to this method.
        /// The sequence length is the number of elements in sequence.
        /// Parameters:
        ///     ElementType: data type of the created Value object.Currently, float and double are supported.
        ///     dimension: the size of dimension of the one-hot vector.
        ///     sequenceData: the collection of indexes representing the sequence of samples.
        ///     sequenceStartFlag: true indicates that it is a new sequence. false means a continuation of a previous sequence.
        ///     device: on which device the Value object should be created.
        ///     readOnly: the Value is read-only if this flag is true.
        ///
        template <typename ElementType>
        CNTK_API static ValuePtr CreateSequence(size_t dimension, const std::vector<size_t>& sequenceData, bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly = false);

        ///
        /// Creates a new Value object containing a sequence of samples.
        /// Each sample is represented by an index value that points to the non-zero value in the one-hot vector of dimension elements.
        /// The sequence length is the number of elements in sequence.
        /// The created sequence is a new sequence.
        /// Parameters:
        ///     ElementType: data type of the created Value object.Currently, float and double are supported.
        ///     dimension: the size of dimension of the one-hot vector.
        ///     sequenceData: the collection of indexes representing the sequence of samples.
        ///     device: on which device the Value object should be created.
        ///     readOnly: the Value is read-only if this flag is true.
        ///
        template <typename ElementType>
        static ValuePtr CreateSequence(size_t dimension, const std::vector<size_t>& sequenceData, const DeviceDescriptor& device, bool readOnly = false)
        {
            return CreateSequence<ElementType>(dimension, sequenceData, true, device, readOnly);
        }

        ///
        /// Creates a new Value object containing a batch of variable length sequences.
        /// Each sample is represented by an index value that points to the non-zero value in the one-hot vector of dimension elements.
        /// The number of sequences is the number of elements in the outer vector of batchOfSequences.
        /// The length of each sequence is the number of elements of the corresponding sequence in the inner vector of batchOfSequences.
        /// Parameters:
        ///     ElementType: data type of the created Value object.Currently, float and double are supported.
        ///     dimension: the size of dimension of the one-hot vector.
        ///     batchOfSequences: the collection of indexes representing sequences of samples.The outer vector represents a collection of sequences with variable length, and the inner vector represents each individual sequence.
        ///     sequenceStartFlags: A collection of boolean value.Each element represent whether the correspoinding sequence in batchOfSequences is a new sequence(in case of true) or a continuation of a previous sequence(in case of false).
        ///     device: on which device the Value object should be created.
        ///     readOnly: the Value is read-only if this flag is true.
        ///
        template <typename ElementType>
        static ValuePtr CreateBatchOfSequences(size_t dimension, const std::vector<std::vector<size_t>>& batchOfSequences, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device, bool readOnly = false)
        {
            return Create<ElementType>(dimension, batchOfSequences, sequenceStartFlags, device, readOnly);
        }

        ///
        /// Creates a new Value object containing a batch of variable length sequences.
        /// Each sample is represented by an index value that points to the non-zero value in the one-hot vector of dimension elements.
        /// The number of sequences is the number of elements in the outer vector of batchOfSequences.
        /// The length of each sequence is the number of elements of the corresponding sequence in the inner vector of batchOfSequences.
        /// Each sequence in batchOfSequences is a new sequence.
        /// Parameters:
        ///     ElementType: data type of the created Value object.Currently, float and double are supported.
        ///     dimension: the size of dimension of the one-hot vector.
        ///     batchOfSequences: the collection of indexes representing sequences of samples.The outer vector represents a collection of sequences with variable length, and the inner vector represents each individual sequence.
        ///     device: on which device the Value object should be created.
        ///     readOnly: the Value is read-only if this flag is true.
        ///
        template <typename ElementType>
        static ValuePtr CreateBatchOfSequences(size_t dimension, const std::vector<std::vector<size_t>>& batchOfSequences, const DeviceDescriptor& device, bool readOnly = false)
        {
            return Create<ElementType>(dimension, batchOfSequences, {}, device, readOnly);
        }

        ///
        /// Creates a new Value object containing a sequence of samples.
        /// The sequence is represented by CSC sparse input format (http://docs.nvidia.com/cuda/cusparse/#compressed-sparse-column-format-csc)
        /// The sequenceStartFlag specifies wehther this sequence is a new sequence or continuation of a previous sequence at the same index in the sequences list from a previous call to this method.
        /// The sequence length is determined by the number of rows of the sparse matrix.
        /// Parameters:
        ///     ElementType: data type of the created Value object.Currently, float and double are supported.
        ///     sampleShape: the tensor shape. For sparse input, the tensor shape leading dimensionality must be the same as the total size of the tensor shape.
        ///     sequenceLength: the sequence length.
        ///     sequenceData: the collection of indexes representing the sequence of samples.
        ///     sequenceStartFlag : true indicates that it is a new sequence. false means a continuation of a previous sequence.
        ///     device : on which device the Value object should be created.
        ///     readOnly : the Value is read - only if this flag is true.
        ///
        template <typename ElementType>
        CNTK_API static ValuePtr CreateSequence(const NDShape& sampleShape, size_t sequenceLength, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const ElementType* nonZeroValues, size_t numNonZeroValues, bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly = false);

        ///
        /// Creates a new Value object containing a sequence of samples.
        /// This method does not have paraemter sequenceStartFlag, and thus the sequence is always a new sequence.
        /// All other parameters are same as the method above.
        ///
        template <typename ElementType>
        static ValuePtr CreateSequence(const NDShape& sampleShape, size_t sequenceLength, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const ElementType* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly = false)
        {
            return CreateSequence(sampleShape, sequenceLength, colStarts, rowIndices, nonZeroValues, numNonZeroValues, true, device, readOnly);
        }

        template <typename ElementType>
        static ValuePtr CreateSequence(size_t dimension, size_t sequenceLength, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const ElementType* nonZeroValues, size_t numNonZeroValues, bool sequenceStartFlag, const DeviceDescriptor& device, bool readOnly = false)
        {
            auto sampleShape = NDShape({dimension});
            return CreateSequence(sampleShape, sequenceLength, colStarts, rowIndices, nonZeroValues, numNonZeroValues, sequenceStartFlag, device, readOnly);
        }

        template <typename ElementType>
        static ValuePtr CreateSequence(size_t dimension, size_t sequenceLength, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const ElementType* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly = false)
        {
            auto sampleShape = NDShape({ dimension });
            return CreateSequence(sampleShape, sequenceLength, colStarts, rowIndices, nonZeroValues, numNonZeroValues, true, device, readOnly);
        }

        ///
        /// Destruct 'this' Value object.
        ///
        virtual ~Value();

        ///
        /// Returns the descriptor of the device that 'this' Value resides on
        ///
        virtual DeviceDescriptor Device() const { return Data()->Device(); }

        ///
        /// Returns the data type of 'this' Value's contents.
        ///
        virtual DataType GetDataType() const { return Data()->GetDataType(); }

        ///
        /// Returns the storage format of 'this' Value.
        ///
        virtual StorageFormat GetStorageFormat() const { return Data()->GetStorageFormat(); }

        ///
        /// Returns the shape 'this' Value.
        ///
        virtual const NDShape& Shape() const { return Data()->Shape(); }

        ///
        /// Returns a boolean indicating if 'this' Value contains data in sparse storage format.
        ///
        bool IsSparse() const { return (GetStorageFormat() != StorageFormat::Dense); }

        ///
        /// Returns a boolean indicating if 'this' Value is read-only.
        ///
        virtual bool IsReadOnly() const { return Data()->IsReadOnly(); }

        ///
        /// Returns the number of masked/invalid values
        ///
        virtual size_t MaskedCount() const { return m_mask ? m_mask->MaskedCount() : 0; }

        ///
        /// Returns the NDArrayView object corresponding to the data contents of 'this value object.
        ///
        virtual NDArrayViewPtr Data() const;

        ///
        /// Returns the NDMask object corresponding to the mask associated with 'this value object.
        ///
        virtual NDMaskPtr Mask() const;

        ///
        /// Creates a new Value with newly allocated storage on the same device as 'this' Value and copies 'this' Value's contents into the newly allocated Value.
        ///
        virtual ValuePtr DeepClone(bool readOnly) const;

        ///
        /// Creates a new Value with newly allocated storage on the same device as 'this' Value and copies 'this' Value's contents into the newly allocated Value.
        ///
        ValuePtr DeepClone() const { return DeepClone(IsReadOnly()); }

        ///
        /// Creates a new Value which is an alias of 'this' Value.
        ///
        virtual ValuePtr Alias(bool readOnly = false) const;

        ///
        /// Copies the contents of the 'source' Value to 'this' Value.
        /// The shapes of the 'source' Value's data and mask must be identical to 'this' Value's data and mask.
        ///
        virtual void CopyFrom(const Value& source);

        virtual void Erase();

        ///
        /// Unpacks sequences in 'this' Value as a vector of NDArrayView objects, each represeting a sequence in the
        /// batch of sequences that 'this' Value object contains data for.
        /// Besides the NDArrayView objects (that represent contents of each sequence), this method also returns
        /// the sequence start information for each sequence, which indicates whether that sequence is the start of
        /// a new sequence or a continuation of a previous one
        ///
        std::pair<std::vector<NDArrayViewPtr>, std::vector<bool>> UnpackVariableValue(const Variable& variable, bool sequenceSegmentsAllowed, const DeviceDescriptor& device)
        {
            // PackedValue should be automatically unpacked when accessing Data() and Mask().
            size_t numOfSequences;
            size_t maxSequenceLen;
            NDShape actualVariableShape;
            std::tie(maxSequenceLen, numOfSequences) = GetSequenceAndBatchLength(variable, &actualVariableShape);

            std::vector<std::ptrdiff_t> sequenceBeginIndices(numOfSequences, 0);
            std::vector<size_t> sequenceLengths(numOfSequences, maxSequenceLen);
            GetSequenceStartsAndLengths(Mask(), sequenceBeginIndices, sequenceLengths, variable.DynamicAxes().size());

            auto valueShapeWithSequenceAndBatchAxes = actualVariableShape.AppendShape(NDShape({ maxSequenceLen , numOfSequences }));
            auto valueData = Data()->AsShape(valueShapeWithSequenceAndBatchAxes);
            if (valueData->Device() != device)
                valueData = valueData->DeepClone(device, valueData->IsReadOnly());

            std::vector<NDArrayViewPtr> sequences(numOfSequences);
            std::vector<bool> sequenceStartFlags(numOfSequences);
            for (size_t i = 0; i < numOfSequences; ++i)
            {
                if (!sequenceSegmentsAllowed && (sequenceBeginIndices[i] != 0))
                    RuntimeError("Value::UnpackVariableValue: Only Value objects containing the entire sequence (no segments) are supported.");

                NDShapeDimensions offset(valueShapeWithSequenceAndBatchAxes.Rank(), 0);
                offset.back() = (NDShapeDimension)i;

                NDShapeDimensions extent(valueShapeWithSequenceAndBatchAxes.Rank() - 1, NDShape::InferredDimension);
                extent.back() = (NDShapeDimension)sequenceLengths[i];

                //sequences[i] = valueData->SliceView(offset, extent, valueData->IsReadOnly());
                sequences[i] = valueData->Slice(offset, extent, NDShapeDimensions(), NDArrayView::SliceMode::View, valueData->IsReadOnly());
                sequenceStartFlags[i] = (sequenceBeginIndices[i] == 0);
            }

            return{ sequences , sequenceStartFlags };
        }

        ///
        /// Unpacks sequences in 'this' Value as a vector of NDArrayView objects, each represeting a sequence in the
        /// batch of sequences that 'this' Value object contains data for.
        /// Besides the NDArrayView objects (that represent contents of each sequence), this method also returns
        /// the sequence start information for each sequence, which indicates whether that sequence is the start of
        /// a new sequence or a continuation of a previous one
        ///
        std::vector<NDArrayViewPtr> UnpackVariableValue(const Variable& variable, const DeviceDescriptor& device)
        {
            return UnpackVariableValue(variable, /* sequenceSegmentsAllowed = */false, device).first;
        }

        ///
        /// Copy the data stored in the Value object to the buffer 'sequences' as a collection of variable length sequences.
        /// The sequence buffer will be resized if necessary.
        /// The Value should have the same tensor shape as outputVariable.
        ///
        template <typename ElementType>
        void CopyVariableValueTo(const Variable& outputVariable, std::vector<std::vector<ElementType>>& sequences)
        {
            ResizeOutputBuffer(outputVariable, sequences);
            CopyVariableValueToVector<ElementType>(outputVariable, sequences);
        }

        ///
        /// Copy the data stored in the Value object to the buffer 'sequences' as a collection of variable length sequences.
        /// The output data is in one-hot format.
        /// The sequence buffer will be resized if ncessary.
        /// The Value should have the same tensor shape as outputVariable.
        ///
        void CopyVariableValueTo(const Variable& outputVariable, std::vector<std::vector<size_t>>& sequences)
        {
            auto dataType = GetDataType();

            ResizeOutputBuffer(outputVariable, sequences);
            if (dataType == DataType::Float)
            {
                CopyVariableValueToVector<float>(outputVariable, sequences);
            }
            else if (dataType == DataType::Double)
            {
                CopyVariableValueToVector<double>(outputVariable, sequences);
            }
        }

        ///
        /// Copy the data stored in 'this' Value object to the buffers representing a sequence in CSC sparse format.
        /// The sequence buffer will be resized if necessary.
        /// The Value should have the same tensor shape as outputVariable.
        /// On return, the sequenceLength is set to the length of the sequence stored in 'this' Value,
        /// and the colStarts, rowIndices and nonZeroValues contain the data of column indexes, row indexes and non-zero values,
        /// and the numNonZeroValues is set to number of non-zero values contained in 'this' Value.
        ///
        template <typename ElementType>
        void CopyVariableValueTo(const Variable& outputVariable, size_t& sequenceLength, std::vector<SparseIndexType>& colStarts, std::vector<SparseIndexType>& rowIndices, std::vector<ElementType>& nonZeroValues, size_t& numNonZeroValues)
        {
            size_t numColsInMatrix;
            std::tie(sequenceLength, numColsInMatrix, numNonZeroValues) = ValidateSparseCSCAndGetIndexBufferSizes<ElementType>(outputVariable);

            // resize output vectors.
            colStarts.resize(numColsInMatrix);
            rowIndices.resize(numNonZeroValues);
            nonZeroValues.resize(numNonZeroValues);

            CopyVariableValueToCSCSparse(sequenceLength, colStarts, rowIndices, nonZeroValues, numNonZeroValues);
        }

        ///
        /// If the value stored is a scalar, returns it. Otherwise, throws an error.
        ///
        template<typename ElementType>
        ElementType AsScalar() const;

        ///
        /// Returns whether this object has been invalidated (by another forward and/or backward pass)
        ///
        CNTK_API virtual bool IsValid() const;

        ///
        /// Returns a string summary of this Value object
        ///
        CNTK_API std::wstring AsString() const;

    private:
        template <typename ElementType>
        static void AppendSparseSequenceData(const NDArrayViewPtr& sequenceData, std::vector<SparseIndexType>& colStarts, std::vector<SparseIndexType>& rowIndices, std::vector<char>& nonZeroValues, size_t maxSequenceLengthInCols);

        ///
        /// Copy the data stored in 'this' Value object to the buffer 'sequences' as a collection of variable length sequences.
        /// The output data is in the dense format.
        /// Assumption: The 'sequences' buffer has been resized to match the number of sequences and the length of each sequence stored in the Value object.
        /// The resizing is done by ResizeOutputBuffer() and needs to be done on the heap of the caller.
        ///
        template <typename ElementType>
        void CopyVariableValueToVector(const Variable& outputVariable, std::vector<std::vector<ElementType>>& sequences);

        ///
        /// Copy the data stored in 'this' Value object to the buffer 'sequences' as a collection of variable length sequences.
        /// The output data is in the one-hot format.
        /// The resizing is done by ResizeOutputBuffer() and needs to be done on the heap of the caller.
        /// Assumption: The 'sequences' buffer has been resized to match the number of sequences and the length of each sequence stored in the Value object.
        ///
        template <typename ElementType>
        void CopyVariableValueToVector(const Variable& outputVariable, std::vector<std::vector<size_t>>& sequences);

        template <typename ValueType, typename DestType>
        void CopyVariableValueToImpl(const Variable& outputVariable, std::vector<std::vector<DestType>>& sequences);

    private:
        CNTK_API std::pair<size_t, size_t> GetSequenceAndBatchLength(const Variable& outputVariable, NDShape* inferredVarShape = nullptr);

        template <typename ElementType>
        CNTK_API std::tuple<size_t, size_t, size_t> ValidateSparseCSCAndGetIndexBufferSizes(const Variable& outputVariable);

        template <typename ElementType>
        CNTK_API void CopyVariableValueToCSCSparse(size_t sequenceLength, std::vector<SparseIndexType>& colStarts, std::vector<SparseIndexType>& rowIndices, std::vector<ElementType>& nonZeroValues, size_t& numNonZeroValues);

        CNTK_API static void GetSequenceStartsAndLengths(const NDMaskPtr& mask, std::vector<ptrdiff_t>& sequenceBeginIndices, std::vector<size_t>& sequenceLengths, size_t numDynamicAxes);

        ///
        /// Resize the 'sequences' buffer if needed.
        /// It should be kept in the header file, as the memory should be allocated at the caller side, not the CNTKLibarary.dll side.
        /// outputVariable defines tensor, the sequence axis and the batch axis.
        /// The 'sequences' is the output buffer which is used to store data from 'this' Value. On return, its size and the size of its each element are adjusted
        /// to match the number of sequences and the length of each sequence stored in the Value object.
        ///
        template <typename ElementType>
        void ResizeOutputBuffer(const Variable& outputVariable, std::vector<std::vector<ElementType>>& sequences)
        {
            if (outputVariable.Shape().IsUnknown())
                RuntimeError("The outputVariable '%S' shape '%S' is unknown shape.",
                              outputVariable.AsString().c_str(), outputVariable.Shape().AsString().c_str());

            // Make sure that inferredVarShape has correct rank in order to avoid any rank resize during inferring free dimension.
            NDShape inferredVarShape = outputVariable.Shape();
            size_t numOfSequences;
            size_t maxSequenceLen;
            // Verify compatibility of 'this' value and outputVariable, get sequence and batch length, and get the inferred shape if the variable has a free dimension.
            std::tie(maxSequenceLen, numOfSequences) = GetSequenceAndBatchLength(outputVariable, &inferredVarShape);
            if (outputVariable.Shape().Rank() != inferredVarShape.Rank())
                RuntimeError("The shape of outputVariable has a different rank after inferring unbound dimensions.");

            // Calculate the number of elements is needed to represent a sample in output buffer.
            // For dense output, it is the total size of the shape.
            // For one-hot output, only 1 index is needed to represent the sample.
            size_t outputSizeOfSample = (std::is_same<ElementType, size_t>::value) ? 1 : inferredVarShape.TotalSize();

            // resize the output buffer size to reflect the number of sequences in output.
            sequences.resize(numOfSequences);

            // Check whether each sequence has enough space allocated and resize if necessary.
            std::vector<ptrdiff_t> sequenceBeginIndices(numOfSequences, 0);
            std::vector<size_t> sequenceLengths(numOfSequences, maxSequenceLen);
            GetSequenceStartsAndLengths(Mask(), sequenceBeginIndices, sequenceLengths, outputVariable.DynamicAxes().size());
            for (auto seqIndex = 0; seqIndex < numOfSequences; seqIndex++)
            {
                if (sequenceBeginIndices[seqIndex] != 0)
                    RuntimeError("Currently, only sequence starting with SequenceBegin is supported.");

                sequences[seqIndex].resize(sequenceLengths[seqIndex] * outputSizeOfSample);
            }
        }

        // Disallow copy and move construction and assignment
        Value(const Value&) = delete; Value& operator=(const Value&) = delete; Value(Value&&) = delete; Value& operator=(Value&&) = delete;

    protected:
        mutable NDArrayViewPtr m_data;
        mutable NDMaskPtr m_mask;
    };

    ///
    /// Encapsulates the internal computation state of a Function computed as part of the 'Forward' call on a Function
    /// that must be passed to a subsequent 'Backward' call on the same Function to backpropagate gradient values
    /// for the same computation backwards through the Function
    ///
    class BackPropState : public std::enable_shared_from_this<BackPropState>
    {
    public:
        ///
        /// Constructs a BackPropState object
        /// The function and computeDevice parameters record the Function and compute device that 'this' BackPropState corresponds to
        /// The forwardPropValuesToSave is an optional map of forward compute values saved for later use during back propagation of gradients
        /// in the backward call that 'this' BackPropState object is used.
        ///
        BackPropState(const FunctionPtr& function, const DeviceDescriptor& computeDevice, const std::unordered_map<Variable, ValuePtr>& forwardPropValuesToSave = {})
            : m_function(function), m_forwardComputeDevice(computeDevice), m_savedForwardPropValues(forwardPropValuesToSave)
        {}

        ///
        /// Destructor
        ///
        virtual ~BackPropState() {}

        ///
        /// Returns the Function that 'this' BackPropState belongs to
        ///
        FunctionPtr Function() const { return m_function; }

        ///
        /// Returns the DeviceDescriptor that the forward call, that created 'this' BackPropState, was executed on
        ///
        DeviceDescriptor Device() const { return m_forwardComputeDevice; }

        ///
        /// Returns the forward prop values saved when constructing 'this' BackPropState state
        /// for later use during back propagation of gradients in a backward call that 'this' BackPropState object is used.
        ///
        const std::unordered_map<Variable, ValuePtr>& SavedForwardPropValues() const { return m_savedForwardPropValues; }

    protected:
        FunctionPtr m_function;
        DeviceDescriptor m_forwardComputeDevice;
        std::unordered_map<Variable, ValuePtr> m_savedForwardPropValues;
    };
    typedef std::shared_ptr<BackPropState> BackPropStatePtr;

    ///
    /// How are Parameters handled when cloning a Function
    ///
    enum class ParameterCloningMethod
    {
        ///
        /// Parameters are shared between the Function being cloned and the new clone
        ///
        Share,

        ///
        /// New learnable Parameters are created and initialized with the current values of the
        /// corresponding Parameters of the Function being cloned
        ///
        Clone,

        ///
        /// Parameters are cloned and made immutable; i.e. Constants in the new clone
        /// (e.g. for use as a fixed feature extractor)
        ///
        Freeze,

        ///
        /// Internal use only
        ///
        Invalid,
    };

    ///
    /// Defines a signature of the deserialize callback for user defined functions,
    /// that needs to be provided to Function::Load to inflate user defined functions in the model.
    /// This callback reconstructs a user defined function given its inputs, name and a dictionary
    /// containing its state.
    ///
    typedef std::function<FunctionPtr(const std::vector<Variable>& /*inputs*/,
        const std::wstring& /*name*/,
        const Dictionary& /*dictionary*/)> UDFDeserializeCallback;

    typedef std::shared_ptr<UDFDeserializeCallback> UDFDeserializeCallbackPtr;

    static auto NoOp = [] (const std::vector<Variable>&, const std::wstring&, const Dictionary&)
    {
        return nullptr;
    };

    ///
    /// Represents a function (optionally differentiable w.r.t. its inputs)
    /// A Function denotes a symbolic computation with zero or more input arguments and one or more outputs.
    /// A Function may be primitive or composite (comprised of other Function instances whose inputs and outputs are wired together).
    /// A Function effectively is a computation graph composed of other primitive Functions (denoting computation) as nodes and Variable objects
    /// (denoting data) as the edges and leaves of the graph.
    /// Function class inherits from  IDictionarySerializable to allow derived 'Function' types to specify custom serialization procedure.
    ///
    class Function : public std::enable_shared_from_this<Function>, public IDictionarySerializable
    {
        friend class CompositeFunction;
        friend class PrimitiveFunction;
        friend class BlockFunction;
        friend class UDFUtils;
        friend class Trainer;
        friend class InternalVariable::AutoBatch;
        friend class InternalVariable::Memoizer;

        friend Variable GetCorrespondingOutputVariableFromClone(const Variable&, const FunctionPtr&, const FunctionPtr&);
        friend bool Internal::IsNativeUserFunctionRegistered(const std::wstring& uniqueOpName);

    public:
        typedef FixedVectorWithBuffer<Variable, 2> InputsVectorType;
        typedef FixedVectorWithBuffer<InternalVariable, 1> OutputsVectorType;

        ///
        /// Computes and stores the values of specified variables in the 'outputs' map, using provided 'inputs' values corresponding
        /// to each leaf variable of the Function of VariableKind 'Input'.
        /// The variables specified in the 'outputs' map denote the subset of 'this' Function's output variables that the caller wants to obtain values of.
        /// Callers may specify the storage to be used for storing the 'outputs' Values or pass null in which case the implementation allocates the actual storage
        /// for the 'outputs' for which the ValuePtr mapping was left null by the caller. If a null Value was specified, the implementation created Value objects
        /// are temporary and only guaranteed to be valid until the next Forward/Backward call. You must explicitly clone the temporay Values if they need to be accessed later.
        /// The optional 'outputsToRetainBackwardStateFor' parameter specifies the subset of the Function's output variables for which gradients will be specified
        /// in a subsequent Backward call for backpropagation.
        /// The method returns a BackPropState object containing all intermediate variable values needed during backpropagation of gradients from the
        /// 'outputsToRetainBackwardStateFor' outputs of the Function to any of the inputs of the Function, in a subsequent Backward call.
        /// Note that the returned BackPropState instance also stores a reference to the supplied 'inputs' Values and generated 'outputs' Values
        /// and the user is responsible for ensuring that the contents of the inputs and outputs are unchanged until after any uses of the BackPropState instance
        /// for backpropagating gradients through this Function.
        ///
        CNTK_API BackPropStatePtr Forward(const std::unordered_map<Variable, ValuePtr>& arguments,
                                          std::unordered_map<Variable, ValuePtr>& outputs,
                                          const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice(),
                                          const std::unordered_set<Variable>& outputsToRetainBackwardStateFor = {},
                                          const std::unordered_set<Variable>& inputsToExcludeGradientsFor = {});

        ///
        /// Backpropagates supplied 'rootGradientValues' for one or more of the output variables of the Function, to produce gradient Values
        /// corresponding to the specified set of input variables in 'backPropagatedGradientValuesForInputs'.
        /// Callers may specify the actual storage to be used for storing the 'backPropagatedGradientValuesForInputs' Values or leave them to be null
        /// in which case the implementation allocates the actual storage for storing the gradients. If a null Value was specified, the implementation created Value objects
        /// are temporary and only guaranteed to be valid until the next Forward/Backward call. You must explicitly clone the temporay Values if they need to be accessed later.
        /// In case an existing storage is specified, the gradients are aggregated with existing values in the specified storage.
        /// The 'state' parameter is an instance of an BackPropState instance obtained from a previous call to the Forward method on 'this; Function for the
        /// computation that this gradient backpropagation corresponds to.
        ///
        CNTK_API virtual void Backward(const BackPropStatePtr& state,
                                       const std::unordered_map<Variable, ValuePtr>& rootGradientValues,
                                       std::unordered_map<Variable, ValuePtr>& backPropagatedGradientValuesForInputs);

    protected:
        ///
        /// Computes and stores the values of specified variables in the 'outputs' map, using provided 'inputs' values for each input of the Function.
        /// The variables specified in the 'outputs' map denote the subset of 'this' Function's output variables that the caller wants to obtain values of.
        /// Callers may specify the storage to be used for storing the 'outputs' Values or pass null in which case the implementation allocates the actual storage
        /// for the 'outputs' for which the ValuePtr mapping was left null by the caller.  If a null Value was specified, the implementation created Value objects
        /// are temporary and only guaranteed to be valid until the next Forward/Backward call. You must explicitly clone the temporay Values if they need to be accessed later.
        /// The optional 'outputsToRetainBackwardStateFor' parameter specifies the subset of the Function's output variables for which gradients will be specified
        /// in a subsequent Backward call for backpropagation.
        /// The method returns a BackPropState object containing all intermediate variable values needed during backpropagation of gradients from the
        /// 'outputsToRetainBackwardStateFor' outputs of the Function to any of the inputs of the Function, in a subsequent Backward call.
        /// Note that the returned BackPropState instance also stores a reference to the supplied 'inputs' Values and generated 'outputs' Values
        /// and the user is responsible for ensuring that the contents of the inputs and outputs are unchanged until after any uses of the BackPropState instance
        /// for backpropagating gradients through this Function.
        /// User defined Functions that derive from the Function type must implement this method.
        ///
        virtual BackPropStatePtr Forward(const std::vector<ValuePtr>& inputValues,
                                         std::unordered_map<Variable, ValuePtr>& outputs,
                                         const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice(),
                                         const std::unordered_set<Variable>& outputsToRetainBackwardStateFor = {}) = 0;

        ///
        /// Infers the shape, data type and dynamic axes of the outputs of 'this' function based on the
        /// Function's inputs, and returns Output Variable objects containing the inferred information
        /// Result cannot exceed the max number of outputs (128).
        /// The passed "outputs" vector should also reserve 128 elements in order to not cause memory allocation during
        /// crossing of dll boundary.
        ///
        CNTK_API virtual OutputsVectorType InferOutputs() = 0;

        ///
        /// Returns the name of the module (dll/so) containing this function. For native functions registered through
        /// a call to 'RegisterNativeUserFunction', unless overridden, this method return the value of the 'moduleName'
        /// argument.
        ///
        CNTK_API virtual std::wstring ModuleName() const;

        ///
        /// Returns the name of the method which should be invoked to deserialize this function. For native functions
        /// registered through a call to 'RegisterNativeUserFunction', unless overridden, this method return the value
        /// of the 'factoryMethodName' argument. If overridden, it must have the same signature as the factory method.
        ///
        CNTK_API virtual std::wstring DeserializeMethodName() const;

    public:

        // Optional overrides

        ///
        /// Destruct this Function.
        ///
        CNTK_API virtual ~Function();

        ///
        /// Returns the name of the operation that this Function denotes
        ///
        CNTK_API virtual const std::wstring& OpName() const;

        ///
        /// This method needs to be explicitly overriden in subclasses.
        ///
        CNTK_API virtual size_t CurrentVersion() const override { NOT_IMPLEMENTED; }

        ///
        /// Generates a dictionary that captures the state of the Function graph underlying this Function.
        ///
        CNTK_API virtual Dictionary Serialize() const override { return Attributes(); }

        ///
        /// Creates a clone of this Function instance, using the specified 'inputs' that are inputs of the clone to be constructed.
        ///
        CNTK_API virtual FunctionPtr Clone(const std::vector<Variable>& /*clonedInputs*/) { NOT_IMPLEMENTED; }

    public:
        ///
        /// Compute the gradients of the output of this Function, w.r.t. the specified input variables in 'gradients'
        /// at the specified 'arguments' values for the Function inputs
        ///
        CNTK_API void Gradients(const std::unordered_map<Variable, ValuePtr>& arguments,
                                std::unordered_map<Variable, ValuePtr>& gradients,
                                std::unordered_map<Variable, ValuePtr>& outputsToEvaluate,
                                const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        CNTK_API void Gradients(const std::unordered_map<Variable, ValuePtr>& arguments,
                                Variable& gradientRoot,
                                std::unordered_map<Variable, ValuePtr>& gradients,
                                std::unordered_map<Variable, ValuePtr>& outputsToEvaluate,
                                const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Compute the gradients of the output of this Function, w.r.t. the specified input variables in 'gradients'
        /// at the specified 'arguments' values for the Function inputs
        ///
        void Gradients(const std::unordered_map<Variable, ValuePtr>& arguments,
                       std::unordered_map<Variable, ValuePtr>& gradients,
                       const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice())
        {
            std::unordered_map<Variable, ValuePtr> outputsToEvaluate = {};
            return Gradients(arguments, gradients, outputsToEvaluate, computeDevice);
        }

        ///
        /// Performs forward computation, i.e. evaluation, on the computaion graph using provided 'input' and stores the results in the 'outputs' map.
        /// It is same as Forward, but without storing and returning information needed for backpropagation.
        ///
        CNTK_API void Evaluate(const std::unordered_map<Variable, ValuePtr>& arguments,
                               std::unordered_map<Variable, ValuePtr>& outputs,
                               const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Clones 'this' Function. The parameters of the Function are either cloned, shared or frozen as specified by the parameterCloneMethod argument and
        /// any variable replacements requested are applied in the cloned Function instance.
        ///
        CNTK_API FunctionPtr Clone(ParameterCloningMethod parameterCloneMethod = ParameterCloningMethod::Clone, const std::unordered_map<Variable, Variable>& replacements = {}) const;

        ///
        /// Deserializes a Function from the model dictionary, using the specified UDF deserializer to
        //  reconstruct user defined functions if the model contains any (in which case an exception will be raised
        /// if deserializer was omitted). If there are no user defined functions in the model, deserializer is ignored.
        ///
        CNTK_API static FunctionPtr Deserialize(const Dictionary& dictionary,
                                                const ::CNTK::DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());

    public:
        ///
        /// Returns the name of 'this' Function.
        ///
        const std::wstring& Name() const { return m_name.get(); }

        ///
        /// Sets the name of 'this' Function.
        /// Setting the name of a Function is only allowed if the Function does not already have a name.
        /// Calling this method, when 'this' Function already has a name, results in an exception.
        ///
        CNTK_API void SetName(const std::wstring& name);

        ///
        /// Returns the internally generated unique name of the Function
        /// The uid may be generated lazily upon first call.
        ///
        CNTK_API const std::wstring& Uid() const;

        ///
        /// Returns the primitive Function at the root of the graph of Functions underlying this Function.
        /// If 'this' Function itself is a primitive Function then (this->RootFunction() == this).
        ///
        FunctionPtr RootFunction() const
        {
            auto res = (m_rootFunction == nullptr) ? const_cast<Function*>(this)->shared_from_this() : m_rootFunction;
            assert(res->IsPrimitive()); // TODO: remove this test again once I understand this fully
#if 1
            if (!res->IsPrimitive())
                LogicError("RootFunction must be PrimitiveFunction");
#endif
            return res;
        }

        ///
        /// Returns a boolean indicating if this Function is a composite Function
        ///
        bool IsComposite() const { return (m_rootFunction != nullptr); }

        ///
        /// Returns a boolean indicating if this Function is a primitive Function
        ///
        bool IsPrimitive() const { return !IsComposite(); }

        ///
        /// Returns a boolean indicating if this Function is a block Function which is basically
        /// a composite encapsulated as an opaque block which appears as a primitive during traversing
        /// the graph of Functions that this block is part of.
        ///
        CNTK_API bool IsBlock() const;

        ///
        /// Returns the root of the Function graph underlying this block Function.
        /// Throws an exception if this is not a block Function
        ///
        CNTK_API FunctionPtr BlockRoot() const;

        ///
        /// Returns the mapping from the arguments of the composite underlying this block Function
        /// to the Variables that they are bound to in the outer graph of Functions that this
        /// block Function is part of.
        ///
        std::vector<std::pair<Variable, Variable>> BlockArgumentsMapping() const
        {
            return *BlockArgumentsMappingImpl().get();
        }

        ///
        /// Returns all inputs of 'this' Function.
        /// Note that inputs here denotes all Variables that feed into this Function including any
        /// Parameter/Constant Variables that are children of 'this' Function.
        ///
        std::vector<Variable> Inputs(bool pythonOperandOrder = false) const
        {
            return *(InputsImpl(pythonOperandOrder).get());
        }

        ///
        /// Returns the Output variable of 'this' Function. Throws an exception of 'this' Function has more that one output.
        ///
        CNTK_API Variable Output(bool init = true) const;

        ///
        /// Returns a vector consisting of all Output variables of 'this' Function.
        ///
        std::vector<Variable> Outputs() const
        {
            return *(OutputsImpl().get());
        }

        ///
        /// Returns a set comprising of all input variables of 'this' Function's variables that are not of kind 'Parameter' or 'Constant'.
        ///
        std::vector<Variable> Arguments(bool rowMajor = false) const
        {
            return FilteredInputs<Variable>([](const Variable& var) {
                return IsArgument(var);
            }, rowMajor);
        }

        ///
        /// Returns the set of all Parameter variables of 'this' Function.
        ///
        std::vector<Parameter> Parameters() const
        {
            return FilteredInputs<Parameter>([](const Variable& var) {
                return var.IsParameter();
            });
        }

        ///
        /// Returns the set of all Constant variables of 'this' Function.
        ///
        std::vector<Constant> Constants() const
        {
            return FilteredInputs<Constant>([](const Variable& var) {
                return var.IsConstant();
            });
        }

        ///
        /// Returns the set of all Placeholder variables of 'this' Function.
        ///
        std::vector<Variable> Placeholders() const
        {
            return FilteredInputs<Variable>([](const Variable& var) {
                return var.IsPlaceholder();
            });
        }

        ///
        /// Find a function with the given name in the Function graph underlying 'this' Function.
        /// If more than one function with the same name, an exception is thrown.
        /// If nestedSearchInsideBlockFunction is true, all functions inside block functions are also searched for the given name.
        ///
        FunctionPtr FindByName(const std::wstring& name, bool nestedSearchInsideBlockFunction = false)
        {
            FunctionPtr  foundFunction = nullptr;
            PreorderTraverseFunctions(RootFunction(), [&foundFunction, &name, this](const FunctionPtr& function) {
                if (name.compare(function->Name()) == 0)
                {
                    if (foundFunction != nullptr)
                        RuntimeError("FindByName: Multiple functions with the name '%S' are found in the Function graph underlying 'this' Function.", name.c_str());
                    else
                        foundFunction = function;
                }
            }, nestedSearchInsideBlockFunction);

            return foundFunction;
        }

        ///
        /// Find a list of functions with the given name in the Function graph underlying 'this' Function.
        /// If nestedSearchInsideBlockFunction is true, all functions inside block functions are also searched for the given name.
        ///
        std::vector<FunctionPtr> FindAllWithName(const std::wstring& name, bool nestedSearchInsideBlockFunction = false)
        {
            std::vector<FunctionPtr> foundFunctions;
            PreorderTraverseFunctions(RootFunction(), [&foundFunctions, &name](const FunctionPtr& function) {
                if (name.compare(function->Name()) == 0)
                   foundFunctions.push_back(function);
            }, nestedSearchInsideBlockFunction);

            return foundFunctions;
        }

        /// Returns the dictionary of attributes of 'this' Function
        ///
        const Dictionary& Attributes() const { return m_attributes; }

        ///
        /// In-place replace specified placeholders in the Function graph with the specified replacements in the map
        ///
        CNTK_API FunctionPtr ReplacePlaceholders(const std::unordered_map<Variable, Variable>& placeholderReplacements);

        ///
        /// In-place replace the only placeholder in the Function graph with the specified replacements in the map
        /// Throws an exception if 'this' Function has multiple placeholders
        ///
        CNTK_API FunctionPtr ReplacePlaceholder(const Variable& placeholderReplacement);

        ///
        CNTK_API void Save(std::vector<char> &vectorBuf);

        ///
        /// Save this Function graph into a model file.
        ///
        CNTK_API void Save(const std::wstring& filepath);

        ///
        /// Restore the models parameters (in-place) from a model file
        ///
        CNTK_API void Restore(const std::wstring& filepath);

        ///
        /// Load a Function from a model file
        ///
        CNTK_API static FunctionPtr Load(const std::wstring& filepath,
                                         const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Load a Function from a memory buffer
        ///
        CNTK_API static FunctionPtr Load(const char* buffer, size_t length,
                                         const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Load a Function from an istream. The legacy V1 model is not supported.
        ///
        CNTK_API static FunctionPtr Load(std::istream& inputStream,
                                         const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Returns a string representation of this Function
        ///
        CNTK_API std::wstring AsString(bool doNotInferOutputs = true) const;

        ///
        /// Allows to change a function attribute. Currently supported:
        ///
        /// * 'dropoutRate' with the corresponding float or double value. Modifies the dropout rate
        /// of a dropout function (can only be invoked on a function instance returned from
        /// the Dropout() method or a primitive dropout function returned from FindByName()).
        ///
        /// * 'rngSeed' with the corresponding int or size_t value. Modifies the seed of a stateful function,
        /// i.e., Dropout, RandomSample or RandomSampleInclusionFrequency (can only be invoked on a
        /// function instance returned from the Dropout(), RandomSample(), RandomSampleInclusionFrequency()
        /// method or a corresponding primitive function returned from FindByName()).
        ///
        CNTK_API void SetAttribute(const std::wstring& name, const DictionaryValue& value);

        ///
        /// Maximum number of outputs that is currently supported.
        ///
        static const int MaxNumOutputs = 64;

    public:

        ///
        /// Registers a native user-defined Function that can be subsequently instantiated using the Function::NativeUserFunction method.
        ///
        // TODO: Do we need an Unregister to unload the module?
        CNTK_API static void RegisterNativeUserFunction(const std::wstring& uniqueOpId, const std::wstring& moduleName, const std::wstring& factoryMethodName);

        ///
        /// Create an instance of a user-defined Function type registered using Function::RegisterNativeUserFunction method.
        ///
        CNTK_API static FunctionPtr NativeUserFunction(const std::wstring& opId, const std::vector<Variable>& operands, const Dictionary& functionConfig, const std::wstring& userFunctionInstanceName = L"");

        ///
        /// Register a callback function to be invoked when deserializing a user-defined Function with the corresponding op name.
        /// When loading a model, CNTK will try to automatically reconstruct user-defined Functions (for native functions, CNTK will
        /// invoke the same factory method, the Function op name was registered with). This method allows to override
        /// default user-defined Function deserialization behavior by specifying an op name and the corresponding callback that should be invoked
        /// to inflate the Function object.
        ///
        CNTK_API static void RegisterUDFDeserializeCallback(const std::wstring& uniqueOpName, const UDFDeserializeCallback& deserializer);

        static UDFDeserializeCallbackPtr GetUDFDeserializeCallback(const std::wstring& uniqueOpName);

    protected:
        static bool IsArgument(const Variable& var)
        {
            return (var.IsInput() || var.IsPlaceholder() || var.IsOutput());
        }

        ///
        /// Protected constructor for user-derived 'Function' types to specify the actual input and output variables for the (primitive) Function instance.
        ///
        CNTK_API Function(const std::vector<Variable>& inputs, Dictionary&& functionConfig, const std::wstring& name = std::wstring(), const std::wstring& uid = Internal::GenerateUid(L"UserDefinedFunction"));

        template <typename FunctionType>
        static void PreorderTraverseFunctions(const FunctionPtr& rootFunction, const FunctionType& functor, bool traverseInsideBlockFunction = false)
        {
            std::unordered_set<FunctionPtr> visitedFunctions;
            PreorderTraverseFunctions(rootFunction, visitedFunctions, functor, traverseInsideBlockFunction);
        }

        // Recursively traverses the Function graph underlying the 'rootFunction' invoking the provided functor for all visited nodes in the graph.
        template <typename FunctionType>
        static void PreorderTraverseFunctions(const FunctionPtr& rootFunction, std::unordered_set<FunctionPtr>& visitedFunctions, const FunctionType& functor, bool traverseInsideBlockFunction = false)
        {
            visitedFunctions.insert(rootFunction);
            functor(rootFunction);

            if (traverseInsideBlockFunction && rootFunction->IsBlock())
                PreorderTraverseFunctions(rootFunction->BlockRoot(), visitedFunctions, functor, traverseInsideBlockFunction);

            std::vector<Variable> rootFunctionInputs = rootFunction->Inputs();
            for (const auto& rootInput : rootFunctionInputs)
            {
                if (rootInput.IsOutput() && visitedFunctions.find(rootInput.Owner()) == visitedFunctions.end())
                {
                    const auto& function = rootInput.Owner();
                    PreorderTraverseFunctions(function, visitedFunctions, functor, traverseInsideBlockFunction);
                }
            }
        }

        /// Restores the state of the 'this' Function in place using the provided dictionary.
        /// Structurally, 'this' Function graph has to be identical to the state captured in the dictionary.
        CNTK_API virtual void RestoreFromCheckpoint(const Dictionary& dictionary);

        ///
        /// Notifies the Function of any placeholder replacements
        ///
        CNTK_API virtual void OnPlaceholdersReplaced(const std::unordered_map<Variable, Variable>& placeholderReplacements,
                                                     std::unordered_set<Variable>& replacedPlaceholders);

    protected:
        static bool ValidateOrUpdateOutput(const InternalVariable& output, const InternalVariable& newOutput, bool alwaysUpdate);

        // Returns a outputs without ref-counting the owner.
        friend class Variable;
#ifndef SWIG
        CNTK_API const Span<InternalVariable*>/*auto*/ RawOutputs() const { const_cast<Function*>(this)->InitOutputs(); return MakeSpan(m_outputs); }
        //CNTK_API const auto RawOutputs() const { const auto& outputs = const_cast<Function*>(this)->InitOutputs(); return MakeSpan(outputs); }
#endif

    private:
        CNTK_API std::shared_ptr<std::vector<std::pair<Variable, Variable>>> BlockArgumentsMappingImpl() const;

        // Lazily initialize the Function's outputs on first invocation. Returns m_outputs.
        CNTK_API OutputsVectorType& InitOutputs();

        template <typename VariableType, typename FilterFunction>
        std::vector<VariableType> FilteredInputs(FilterFunction&& filterFunc, bool rowMajor = false) const
        {
            std::vector<VariableType> filteredInputs;
            std::unordered_set<Variable> uniqueFilteredInputs;
            auto inputs = Inputs(rowMajor);
            for (auto inputVar : inputs)
            {
                if (filterFunc(inputVar) && (uniqueFilteredInputs.find(inputVar) == uniqueFilteredInputs.end()))
                {
                    uniqueFilteredInputs.insert(inputVar);
                    filteredInputs.push_back(VariableType(inputVar));
                }
            }

            return filteredInputs;
        }

        CNTK_API std::shared_ptr<std::vector<Variable>> InputsImpl(bool pythonOperandOrder = false) const;
        CNTK_API std::shared_ptr<std::vector<Variable>> OutputsImpl() const;

        void ValidateOrUpdateOutputs();
        void ValidateOrUpdateOutputs(std::unordered_map<const Function*, size_t>& visitedFunctions, bool& recurrentNodeOutputModified, std::vector<InternalVariable>& buffer);

        static void ReplacePlaceholderInPlace(Variable& var,
                                              const std::unordered_map<Variable, Variable>& placeholderReplacements,
                                              std::unordered_set<Variable>& replacedPlaceholders);

        void ReplacePlaceholdersInPlace(const std::unordered_map<Variable, Variable>& placeholderReplacements,
                                        std::unordered_set<const Function*>& visitedFunctions,
                                        std::unordered_set<Variable>& replacedPlaceholders);

        static FunctionPtr Clone(const FunctionPtr& clonee,
                                 ParameterCloningMethod parameterCloneMethod,
                                 const std::unordered_map<Variable, Variable>& replacements,
                                 std::unordered_map<const Function*, FunctionPtr>& cloneMap,
                                 std::unordered_map<Variable, Variable>& leafVariablesCloneMap,
                                 std::unordered_map<Variable, Variable>& placeholderReplacements);

        // Disallow copy and move construction and assignment
        Function(const Function&) = delete; Function(Function&&) = delete; Function& operator=(const Function&) = delete; Function& operator=(Function&&) = delete;

    public:
        CNTK_API Function(const std::vector<Variable>& inputs, const std::wstring& name = std::wstring());

    public: // for Dynamite
        CNTK_API static DynamicProfilerPtr CreateDynamicProfiler(int verbosity, const std::wstring& name);
        CNTK_API static DynamicProfilerPtr SetDynamicProfiler(const DynamicProfilerPtr&, bool outer = false);

    private:
        bool IsNative() const { return m_native; }
        Dictionary SerializeNativeImpl() const;
        static FunctionPtr DeserializeNativeImpl(const std::vector<Variable>& inputs, const std::wstring& name, const Dictionary& dict);

#ifdef SWIGPYTHON
    public:
        void SetNative(bool native) { m_native = native; }
#endif

    private:
        //CNTK_API Function(const std::vector<Variable>& inputs, const Dictionary& functionConfig, const FunctionPtr& rootFunction, const std::wstring& name, const std::wstring& uid);
        CNTK_API Function(const std::vector<Variable>& inputs, Dictionary&& functionConfig, const FunctionPtr& rootFunction, const std::wstring& name, const std::wstring& uid);
        CNTK_API Function(const Variable& input0, const Variable& input1, Dictionary&& functionConfig, const std::wstring& name);
        CNTK_API Function(const Variable& input0, Dictionary&& functionConfig, const std::wstring& name);
        // move constructor where everything is prepared outside; used in auto-batching
        CNTK_API Function(InputsVectorType&& inputs, Dictionary&& functionConfig, const std::wstring& name);
        Function(InputsVectorType&& inputs, Dictionary&& functionConfig/*, std::wstring&& name, std::wstring&& uid*/);

        // --- data members ---

        //std::vector<Variable> m_inputs; // primitives: direct input variables; composites: empty (Inputs() determines all leaves on the fly); block: all leaves as if it was a composite
        InputsVectorType m_inputs; // primitives: direct input variables; composites: empty (Inputs() determines all leaves on the fly); block: all leaves as if it was a composite
        OutputsVectorType m_outputs;
#ifdef DYNAMITE_ONLY
        unsigned int m_outputsInitFlag = 0;
#else
        std::once_flag m_outputsInitFlag = 0;
        std::thread::id m_outputInitializingByThreadId;
#endif

        FunctionPtr m_rootFunction; // This is a PrimitiveFunctionPtr for composites, or a nullptr for PrimitiveFunction instances
        Dictionary m_attributes;

        // secondary stuff
        OptionalString m_name;
        mutable OptionalString m_uid;
        //std::unordered_set<std::wstring> m_dirtyAttributes;
        bool m_dirtyAttributeDropoutRate = false;
        bool m_dirtyAttributeRngSeed = false;

        // user-function related
        bool m_native = true;
        static UserFunctionFactoryPtr s_userFunctionFactory;

        // serialization-related
        static const size_t s_serializationVersion = 1;
    };

    ///
    /// Create an instance of the CNTK built-in elementwise negate operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Negate(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Unary negation operator corresponding to the Negate operation
    ///
    inline FunctionPtr operator-(const Variable& operand)
    {
        return Negate(operand);
    }

    ///
    /// Create an instance of the CNTK built-in elementwise sigmoid operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Sigmoid(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise tanh operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Tanh(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise asin operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Asin(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise sine operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Sin(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise acos operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Acos(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise cosine operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Cos(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise cosh operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Cosh(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise sinh operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Sinh(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise linear rectifier operation with the specified input operand.
    ///
    CNTK_API FunctionPtr ReLU(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise exp operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Exp(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise log operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Log(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise square operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Square(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise square-root operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Sqrt(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise round operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Round(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise floor operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Floor(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise ceil operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Ceil(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise abs operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Abs(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise reciprocal operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Reciprocal(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in softmax operation on specified tensor input operand
    ///
    CNTK_API FunctionPtr Softmax(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in softmax operation on specified axis on a
    /// specified tensor input operand
    ///
    CNTK_API FunctionPtr Softmax(const Variable& operand, const Axis& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in hardmax operation on specified tensor input operand
    ///
    CNTK_API FunctionPtr Hardmax(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in transpose dimensions operation on specified tensor input operand
    ///
    CNTK_API FunctionPtr TransposeAxes(const Variable& operand, const Axis& axis1, const Axis& axis2, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in transpose operation on the specified 1D or 2D input operand
    ///
    CNTK_API FunctionPtr Transpose(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in transpose operation on the specified input operand using the specified permutation
    ///
    CNTK_API FunctionPtr Transpose(const Variable& operand, const std::vector<Axis>& permutation,  const std::wstring& name = L"");

    ///
    /// Create an instance of the index operation on specified tensor input operand
    /// Index() is like Slice() for a single axis, which is then dropped.
    /// This is a temporary stop-gap until Christoph's GetItem() is merged.
    ///
    CNTK_API FunctionPtr Index(const Variable& operand, int index, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the slice operation on specified tensor input operand
    ///
    CNTK_API FunctionPtr Slice(const Variable& operand, const std::vector<Axis>& axis, const std::vector<int>& beginIndex, const std::vector<int>& endIndex, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the slice operation on specified tensor input operand
    ///
    CNTK_API FunctionPtr Slice(const Variable& operand, const Axis& axis, int beginIndex, int endIndex, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the slice operation on specified tensor input operand
    ///
    CNTK_API FunctionPtr Slice(const Variable& operand, const std::vector<Axis>& axis, const std::vector<int>& beginIndex, const std::vector<int>& endIndex, const std::vector<int>& strides, const std::wstring& name = L"");
\
    /// Create an instance of attach dynamic axis operation that convert the input's first static axis to dynamic axis.
    /// Only batch axis is supported now.
    ///
    CNTK_API FunctionPtr ToBatch(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of detach dynamic axis operation that convert the input's last dynamic axis to static axis.
    /// Only batch axis is supported now.
    ///
    CNTK_API FunctionPtr UnpackBatch(const Variable& operand, const std::wstring& name);

    enum class PaddingMode
    {
        CONSTANTPAD = 0, // the default, fill the padding cells with 0
        REFLECTPAD = 1, // Padding with reflect mode
        SYMMETRICPAD = 2, // Padding with symmetric mode
    };

    CNTK_API FunctionPtr Pad(const Variable& operand, PaddingMode mode, const std::vector<size_t>& head, const std::vector<size_t>& foot, double constantValue = 0, const std::wstring& name = L"");

    ///
    /// Create an instance of the random_sample operation on specified sampling weights input vector
    ///
    CNTK_API FunctionPtr RandomSample(const Variable& operand, size_t numSamples, bool allowDuplicates, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the random_sample_inclusion_frequency operation on specified sampling weights input vector
    ///
    CNTK_API FunctionPtr RandomSampleInclusionFrequency(const Variable& operand, size_t numSamples, bool allowDuplicates, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the dropout operation on specified tensor input operand
    ///
    CNTK_API FunctionPtr Dropout(const Variable& operand, double dropoutRate, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of a uniform random operation. This produces random numbers with the specified shape (no dynamic axes), uniformly distributed in [low, high)
    ///
    CNTK_API FunctionPtr UniformRandom(const NDShape& shape, DataType dataType, double low=0.0, double high=1.0, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const std::wstring& name = L"");

    ///
    /// Create an instance of a uniform random operation. This produces random numbers with the shape and dynamic axes specified by the operand, uniformly distributed in [low, high)
    ///
    CNTK_API FunctionPtr UniformRandomLike(const Variable& operand, double low = 0.0, double high = 1.0, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const std::wstring& name = L"");

    ///
    /// Create an instance of a normal random operation. This produces random numbers with the specified shape (no dynamic axes), normally distributed with the specified mean and standard deviation (scale)
    ///
    CNTK_API FunctionPtr NormalRandom(const NDShape& shape, DataType dataType, double mean = 0.0, double scale = 1.0, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const std::wstring& name = L"");

    ///
    /// Create an instance of a normal random operation. This produces random numbers with the shape and dynamic axes specified by the operand, normally distributed with the specified mean and standard deviation (scale)
    ///
    CNTK_API FunctionPtr NormalRandomLike(const Variable& operand, double mean = 0.0, double scale = 1.0, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const std::wstring& name = L"");

    ///
    /// Create an instance of a Gumbel random operation. This produces random numbers with the specified shape (no dynamic axes), distributed according to the Gumbel distribution with the specified location and scale
    ///
    CNTK_API FunctionPtr GumbelRandom(const NDShape& shape, DataType dataType, double loc = 0.0, double scale = 1.0, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const std::wstring& name = L"");

    ///
    /// Create an instance of a Gumbel random operation. This produces random numbers with the shape and dynamic axes specified by the operand, distributed according to the Gumbel distribution with the specified location and scale
    ///
    CNTK_API FunctionPtr GumbelRandomLike(const Variable& operand, double loc = 0.0, double scale = 1.0, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const std::wstring& name = L"");

    ///
    /// Create an instance of a Bernoulli random operation. This produces random numbers with the specified shape (no dynamic axes).
    /// The probability of emitting a 1 is 'mean'.
    ///
    CNTK_API FunctionPtr BernoulliRandom(const NDShape& shape, DataType dataType, double mean = 0.5, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const std::wstring& name = L"");

    ///
    /// Create an instance of a Bernoulli random operation. This produces random numbers with the shape and dynamic axes specified by the operand.
    /// The probability of emitting a 1 is 'mean'.
    ///
    CNTK_API FunctionPtr BernoulliRandomLike(const Variable& operand, double mean = 0.5, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const std::wstring& name = L"");

    ///
    /// Create an instance of a Bernoulli random operation. This produces random numbers with the specified shape (no dynamic axes).
    /// This version passes back and forth a state variable for the random-number generator to persist its state. This is for use with Dymamite.
    /// The probability of emitting a 1 is 'mean'.
    /// The rngState must be initialized by LazilyCreateRNGState(). This function only does something
    /// upon first call, so you can chain them: BernoulliRandom(LazilyCreateRNGState(rngState, device), shape, 0.5).
    ///
    CNTK_API FunctionPtr BernoulliRandom(RNGState& rngState, const NDShape& shape, DataType dataType, double mean = 0.5, double scale = 1.0, const std::wstring& name = L"");

    ///
    /// Create an instance of the reshape operation on specified tensor input operand
    ///
    CNTK_API FunctionPtr Reshape(const Variable& operand, const NDShape& replacementShape, const Axis& beginAxis, const Axis& endAxis, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the reshape operation on specified tensor input operand
    ///
    inline FunctionPtr Reshape(const Variable& operand, const NDShape& newShape, const std::wstring& name = std::wstring())
    {
        return Reshape(operand, newShape, Axis(0), Axis::EndStaticAxis(), name);
    }

    ///
    /// Create an instance of the CNTK built-in elementwise tensor addition operation with the specified input operands.
    ///
    CNTK_API FunctionPtr Plus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = std::wstring());

    ///
    /// Binary addition operator corresponding to the Plus operation
    ///
    inline FunctionPtr operator+(const Variable& leftOperand, const Variable& rightOperand)
    {
        return Plus(leftOperand, rightOperand);
    }

    ///
    /// Create an instance of the CNTK built-in elementwise tensor subtraction operation with the specified input operands.
    ///
    CNTK_API FunctionPtr Minus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = std::wstring());

    ///
    /// Binary minus operator corresponding to the Minus operation
    ///
    inline FunctionPtr operator-(const Variable& leftOperand, const Variable& rightOperand)
    {
        return Minus(leftOperand, rightOperand);
    }

    ///
    /// Create an instance of the CNTK operator to compute (x * scale + shift) where scale and shift are scalar constants.
    /// Use this form to avoid creating intermediate Constant() objects for this common form of expression.
    ///
    CNTK_API FunctionPtr ScaleAndShift(const Variable& x, double scale, double shift, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise tensor operation that computes the log of the sum of the exponentials of the specified input operands.
    ///
    CNTK_API FunctionPtr LogAddExp(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise tensor operation that computes the leftOperand raised to the power of the right operand.
    ///
    CNTK_API FunctionPtr Pow(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise multiplication operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr ElementTimes(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in inner-product operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr InnerProduct(const Variable& leftOperand, const Variable& rightOperand, const Axis& axis, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK operation equivalent to (ElementTimes(leftFactor, rightFactor) + additiveTerm).
    ///
    CNTK_API FunctionPtr ElementAffine(const Variable& leftFactor, const Variable& rightFactor, const Variable& additiveTerm, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise division operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr ElementDivide(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise equality comparison operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr Equal(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise not-equal comparison operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr NotEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise less than comparison operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr Less(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise less than or equal to comparison operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr LessEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise greater than comparison operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr Greater(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise greater than or equal to comparison operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr GreaterEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in tensor multiplication operation with the specified input operands.
    /// TODO: Specify the constraints on the shapes of the operands.
    /// TODO: Document inferInputRankToMap
    ///

    // special values for Times inferInputRankToMap
    enum : int
    {
        TimesNoInferredInputRank                        = -1, // the default, do not infer left operand input rank from right operand
        TimesReduceSequenceAxisWithoutInferredInputRank = -2, // reduce sequence axis. Currently only support cases like (m x k) x (k) -> (m) for sequences
    };

    CNTK_API FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank, int inferInputRankToMap, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in tensor multiplication operation with the specified input operands.
    /// TODO: Specify the constraints on the shapes of the operands.
    /// TODO: Document inferInputRankToMap
    ///
    inline FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank, const std::wstring& name = std::wstring())
    {
        return Times(leftOperand, rightOperand, outputRank, TimesNoInferredInputRank, name);
    }

    ///
    /// Create an instance of the CNTK built-in tensor multiplication operation with the specified input operands.
    /// TODO: Specify the constraints on the shapes of the operands.
    /// TODO: Document inferInputRankToMap
    ///
    inline FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = std::wstring())
    {
        return Times(leftOperand, rightOperand, /*outputRank =*/ 1, name);
    }

    CNTK_API FunctionPtr Affine(const Variable& W, const Variable& x, const Variable& b, size_t outputRank, int inferInputRankToMap, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK operation equivalent to Times(W, x) + b.
    /// This operator needs half the memory compared to writing it out explicitly.
    ///
    inline FunctionPtr Affine(const Variable& W, const Variable& x, const Variable& b, size_t outputRank, const std::wstring& name = std::wstring())
    {
        return Affine(W, x, b, outputRank, TimesNoInferredInputRank, name);
    }

    ///
    /// Create an instance of the CNTK built-in tensor multiplication operation with the specified input operands.
    /// TODO: Specify the constraints on the shapes of the operands.
    /// TODO: Document inferInputRankToMap
    ///
    inline FunctionPtr Affine(const Variable& W, const Variable& x, const Variable& b, const std::wstring& name = std::wstring())
    {
        return Affine(W, x, b, /*outputRank =*/ 1, name);
    }

    ///
    /// Create an instance of the CNTK built-in matrix multiplication operation with the transpose of the left input operand
    /// and the specified right operand. Only accepts left operands of ranks 1 or 2.
    /// TODO: Specify the constraints on the shapes of the operands.
    ///
    CNTK_API FunctionPtr TransposeTimes(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in matrix multiplication operation with the transpose of the left input operand
    /// and the specified right operand. Only accepts left operands of ranks 1 or 2.
    /// TODO: Specify the constraints on the shapes of the operands.
    ///
    inline FunctionPtr TransposeTimes(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = std::wstring())
    {
        return TransposeTimes(leftOperand, rightOperand, /*outputRank =*/ 1, name);
    }

    CNTK_API FunctionPtr TransposeAffine(const Variable& W, const Variable& x, const Variable& b, size_t outputRank, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK operation equivalent to TransposeTimes(W, x) + b.
    /// This operator needs half the memory compared to writing it out explicitly.
    ///
    inline FunctionPtr TransposeAffine(const Variable& W, const Variable& x, const Variable& b, const std::wstring& name = std::wstring())
    {
        return TransposeAffine(W, x, b, /*outputRank =*/ 1, name);
    }

    ///
    /// Create an instance of the CNTK built-in operation to compute the cosine distance for the specified input operands.
    ///
    CNTK_API FunctionPtr CosineDistance(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operation to compute the cosine distance with negative samplesfor the specified input operands.
    ///
    CNTK_API FunctionPtr CosineDistanceWithNegativeSamples(const Variable& leftOperand, const Variable& rightOperand, size_t shiftWindow, size_t numberOfNegativeSamples, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operation to compute binary cross-entropy for specified input operands.
    ///
    CNTK_API FunctionPtr BinaryCrossEntropy(const Variable& prediction, const Variable& targets, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operation to compute weighted binary cross-entropy for specified input operands.
    ///
    CNTK_API FunctionPtr WeightedBinaryCrossEntropy(const Variable& prediction, const Variable& targets, const Variable& weights, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operation to compute squared-error for specified input operands.
    ///
    CNTK_API FunctionPtr SquaredError(const Variable& prediction, const Variable& targets, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operation to compute cross-entropy with softmax for specified input operands.
    ///
    CNTK_API FunctionPtr CrossEntropyWithSoftmax(const Variable& prediction, const Variable& labels, const Axis& axis, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operation to compute cross-entropy with softmax for specified input operands.
    ///
    inline FunctionPtr CrossEntropyWithSoftmax(const Variable& prediction, const Variable& labels, const std::wstring& name = std::wstring())
    {
        return CrossEntropyWithSoftmax(prediction, labels, Axis(0), name);
    }

    ///
    /// Create an instance of the CNTK built-in operation for computing the edit distance error for specified operands.
    ///
    CNTK_API FunctionPtr EditDistanceError(const Variable& prediction, const Variable& labels, float substitutionPenalty, float deletionPenalty, float insertionPenalty, bool squashInputs, const std::vector<size_t>& tokensToIgnore, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operation for computing the forwardbackward for specified operands.
    ///
    CNTK_API FunctionPtr ForwardBackward(const Variable& graph, const Variable& features, size_t blankTokenId, int delayConstraint, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operation for computing the labels to graph for input operands.
    ///
    CNTK_API FunctionPtr LabelsToGraph(const Variable& labels, const std::wstring& name = std::wstring());


    ///
    /// Create an instance of the CNTK built-in operation for computing the classification prediction error for specified operands.
    ///
    CNTK_API FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, size_t topN, const Axis& axis, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operation for computing the classification prediction error for specified operands.
    ///
    inline FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, size_t topN, const std::wstring& name = std::wstring())
    {
        return ClassificationError(prediction, labels, topN, Axis(0), name);
    }

    ///
    /// Create an instance of the CNTK built-in operation for computing the classification prediction error for specified operands.
    ///
    inline FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, const Axis& axis, const std::wstring& name = std::wstring())
    {
        return ClassificationError(prediction, labels, /*topN =*/ 1, axis, name);
    }

    ///
    /// Create an instance of the CNTK built-in operation for computing the classification prediction error for specified operands.
    ///
    inline FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, const std::wstring& name = std::wstring())
    {
        return ClassificationError(prediction, labels, Axis(0), name);
    }


    ///
    /// Create an instance of the CNTK built-in noise contrastive estimation loss for specified operands. 
    ///
    CNTK_API FunctionPtr NCELoss(const Variable& weights, const Variable& biases, const Variable& inputs, const Variable& labels, 
        const Constant& noiseWeights, size_t numSamples, bool allowDuplicates=true, unsigned long seed = SentinelValueForAutoSelectRandomSeed,
        const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in LambdaRank loss an effective proxy for optimizing the NDCG metric
    ///
    CNTK_API FunctionPtr LambdaRank(const Variable& prediction, const Variable& gains, const Variable& groupId, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operation for evaluating the NDCG at 1 metric
    ///
    CNTK_API FunctionPtr NDCGAt1(const Variable& prediction, const Variable& gains, const Variable& groupId, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operation for getting the past value along the lone dynamic axis of the specified operand.
    /// Throws an exception of the operand has more than one dynamic axis.
    ///
    CNTK_API FunctionPtr PastValue(const Variable& operand, const Variable& initialState, size_t offset = 1, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operation for getting the past value along the lone dynamic axis of the specified operand.
    /// This overload uses an initial state value of 0.
    /// Throws an exception of the operand has more than one dynamic axis.
    ///
    inline FunctionPtr PastValue(const Variable& operand, size_t offset = 1, const std::wstring& name = std::wstring())
    {
        static const auto defaultInitialState = Constant::Scalar(0.0f);
        return PastValue(operand, defaultInitialState, offset, name);
    }

    ///
    /// Create an instance of the CNTK built-in operation for getting the future value along the lone dynamic axis of the specified operand.
    /// Throws an exception of the operand has more than one dynamic axis.
    ///
    CNTK_API FunctionPtr FutureValue(const Variable& operand, const Variable& initialState, size_t offset = 1, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operation for getting the future value along the lone dynamic axis of the specified operand.
    /// This overload uses an initial state value of 0.
    /// Throws an exception of the operand has more than one dynamic axis.
    ///
    inline FunctionPtr FutureValue(const Variable& operand, size_t offset = 1, const std::wstring& name = std::wstring())
    {
        static const auto defaultInitialState = Constant::Scalar(0.0f);
        return FutureValue(operand, defaultInitialState, offset, name);
    }

    ///
    /// Create an instance of the CNTK built-in operation to get the one_hot tensor on specified input along the specified axis
    ///
    CNTK_API FunctionPtr OneHotOp(const Variable& operand, size_t numClass, bool outputSparse, const Axis& axis, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operation to get a tensor that is gathered from reference tensor by indices.
    ///
    CNTK_API FunctionPtr GatherOp(const Variable& indices, const Variable& reference, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in sum reduction operation on specified tensor input operand along the specified axis
    ///
#define Axis_DropLastAxis (Axis(-1)) // TODO: make this a CNTK construct, Axis::DropLastAxis(), with a special sentinel; or an official flag to ReduceXXX()
    CNTK_API FunctionPtr ReduceSum(const Variable& operand, const Axis& axis, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in LogSum reduction operation on specified tensor input operand along the specified axis
    ///
    CNTK_API FunctionPtr ReduceLogSum(const Variable& operand, const Axis& axis, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in Mean reduction operation on specified tensor input operand along the specified axis
    ///
    CNTK_API FunctionPtr ReduceMean(const Variable& operand, const Axis& axis, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in Max reduction operation on specified tensor input operand along the specified axis
    ///
    CNTK_API FunctionPtr ReduceMax(const Variable& operand, const Axis& axis, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in Min reduction operation on specified tensor input operand along the specified axis
    ///
    CNTK_API FunctionPtr ReduceMin(const Variable& operand, const Axis& axis, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in Prod reduction operation on specified tensor input operand along the specified axis
    ///
    CNTK_API FunctionPtr ReduceProd(const Variable& operand, const Axis& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK to peration that computes Pow(ReduceMean(Sqr(x - mean)) + epsilon, -0.5) along the specified axis.
    /// This uses significantly less memory than the explicit expression.
    ///
    CNTK_API FunctionPtr ReduceInvStdDev(const Variable& x, const Variable& mean, const Variable& epsilon, const Axis& axis, const std::wstring& name = L"");

    // multiple axes reduction below:

    ///
    /// Create an instance of the CNTK built-in sum reduction operation on specified tensor input operand along the specified axes
    ///
    CNTK_API FunctionPtr ReduceSum(const Variable& operand, const std::vector<Axis>& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in LogSum reduction operation on specified tensor input operand along the specified axes
    ///
    CNTK_API FunctionPtr ReduceLogSum(const Variable& operand, const std::vector<Axis>& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in Mean reduction operation on specified tensor input operand along the specified axes
    ///
    CNTK_API FunctionPtr ReduceMean(const Variable& operand, const std::vector<Axis>& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in Max reduction operation on specified tensor input operand along the specified axes
    ///
    CNTK_API FunctionPtr ReduceMax(const Variable& operand, const std::vector<Axis>& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in Min reduction operation on specified tensor input operand along the specified axes
    ///
    CNTK_API FunctionPtr ReduceMin(const Variable& operand, const std::vector<Axis>& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in Prod reduction operation on specified tensor input operand along the specified axes
    ///
    CNTK_API FunctionPtr ReduceProd(const Variable& operand, const std::vector<Axis>& axis, const std::wstring& name = L"");
    ///
    /// Per dimension mean-variance normalization of the specified input operand.
    ///

    ///
    /// Create an instance of the CNTK operation that computes Pow(ReduceMean(Sqr(x - mean)) + epsilon, -0.5) along the specified axis.
    /// This uses significantly less memory than the explicit expression.
    ///
    CNTK_API FunctionPtr ReduceInvStdDev(const Variable& x, const Variable& mean, const Variable& epsilon, const std::vector<Axis>& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK operation that computes (x - x0) * s0 * s1 + x1.
    /// This is useful as part of normalization operations like LayerNormalization.
    ///
    CNTK_API FunctionPtr NormalizeDenormalize(const Variable& x, const Variable& x0, const Variable& s0, const Variable& s1, const Variable& x1, const std::wstring& name = L"");

    CNTK_API FunctionPtr PerDimMeanVarianceNormalize(const Variable& operand, const Variable& mean, const Variable& invStdDev, const std::wstring& name = std::wstring());

    ///
    /// Per dimension mean-variance normalization of the specified input operand.
    ///
    inline FunctionPtr PerDimMeanVarianceNormalize(const Variable& operand, const NDArrayViewPtr& mean, const NDArrayViewPtr& invStdDev, const std::wstring& name = std::wstring())
    {
        Constant meanVar(mean);
        Constant invStdDevVar(invStdDev);

        return PerDimMeanVarianceNormalize(operand, meanVar, invStdDevVar, name);
    }

    ///
    /// Convolution
    ///
    CNTK_API FunctionPtr Convolution(const Variable& convolutionMap,
                                     const Variable& operand,
                                     const NDShape& strides = { 1 },
                                     const std::vector<bool>& sharing = { true },
                                     const std::vector<bool>& autoPadding = { true },
                                     const NDShape& dilation = { 1 },
                                     size_t reductionRank = 1,
                                     size_t groups = 1,
                                     size_t maxTempMemSizeInSamples = 0,
                                     const std::wstring& name = std::wstring());

    ///
    /// Convolution transpose
    ///
    CNTK_API FunctionPtr ConvolutionTranspose(const Variable& convolutionMap,
                                              const Variable& operand,
                                              const NDShape& strides = { 1 },
                                              const std::vector<bool>& sharing = { true },
                                              const std::vector<bool>& autoPadding = { true },
                                              const NDShape& outputShape = { 0 },
                                              const NDShape& dilation = { 1 },
                                              size_t reductionRank = 1,
                                              size_t maxTempMemSizeInSamples = 0,
                                              const std::wstring& name = std::wstring());

    ///
    /// Pooling type.
    ///
    enum class PoolingType
    {
        Max,
        Average,
    };

    ///
    /// Create an instance of the CNTK built-in ROI pooling operation on specified tensor input operands with the specified output shape
    ///
    CNTK_API FunctionPtr ROIPooling(const Variable& operand,
                                    const Variable& rois,
                                    PoolingType poolingType,
                                    const NDShape& roiOutputShape,
                                    double spatialScale,
                                    const std::wstring& name/* = L""*/);

    ///
    /// TODO:
    ///
    CNTK_API FunctionPtr Pooling(const Variable& operand,
                                 PoolingType poolingType,
                                 const NDShape& poolingWindowShape,
                                 const NDShape& strides = {1},
                                 const std::vector<bool>& autoPadding = {false},
                                 const bool ceilOutDim = false,
                                 const bool includePad = false,
                                 const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in Unpooling operation on specified tensor input operands with the specified type and shape
    ///
    CNTK_API FunctionPtr Unpooling(const Variable& operand,
                                   const Variable& poolingInput,
                                   PoolingType UnpoolingType,
                                   const NDShape& UnpoolingWindowShape,
                                   const NDShape& strides = { 1 },
                                   const std::vector<bool>& autoPadding = { false },
                                   const std::wstring& name = std::wstring());

    ///
    /// TODO:
    ///
    CNTK_API FunctionPtr BatchNormalization(const Variable& operand,
                                            const Variable& scale,
                                            const Variable& bias,
                                            const Variable& runningMean,
                                            const Variable& runningInvStd,
                                            const Variable& runningCount,
                                            bool spatial,
                                            double normalizationTimeConstant = 0,
                                            double blendTimeConstant = 0,
                                            double epsilon = 0.00001,
                                            bool useCuDNNEngine = true,
                                            const std::wstring& name = std::wstring());

    ///
    /// BatchNormalization, Dynamite version. This variant requires a unique id for cross-sequence synchronization.
    ///
    CNTK_API FunctionPtr BatchNormalization(const Variable& operand,
                                            size_t id,
                                            const Variable& scale,
                                            const Variable& bias,
                                            const Variable& runningMean,
                                            const Variable& runningInvStd,
                                            const Variable& runningCount,
                                            bool spatial,
                                            double normalizationTimeConstant = 0,
                                            double blendTimeConstant = 0,
                                            double epsilon = 0.00001,
                                            const std::wstring& name = std::wstring());

    /// Create an instance of the CNTK built-in OptimizedRNNStack operation on specified input operands
    ///
    CNTK_API FunctionPtr OptimizedRNNStack(const Variable& operand, const Variable& weights, size_t hiddenSize, size_t numLayers, bool bidirectional = false, const std::wstring& recurrentOp = L"lstm", const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise clip operation on the tensor operand
    ///
    CNTK_API FunctionPtr Clip(const Variable& operand, const Variable& min, const Variable& max, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise choice operation using a condition tensor for specified tensor operands.
    ///
    CNTK_API FunctionPtr ElementSelect(const Variable& condition, const Variable& thenOperand, const Variable& elseOperand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in splice operation to splice together all the specified tensor operands into a single output tensor
    ///
    CNTK_API FunctionPtr Splice(const std::vector<Variable>& operands, const Axis& axis, const std::wstring& name = std::wstring());

    ///
    /// Create a new Function instance which just combines the outputs of the specified list of 'operands' Functions such that the 'Outputs' of the
    /// new 'Function' are union of the 'Outputs' of each of the specified 'operands' Functions.
    /// E.g. When creating a classification model, typically the CrossEntropy loss Function and the ClassificationError Function comprise the two roots
    /// of the computation graph which can be "Combine"d to create a single Function with 2 outputs; viz. CrossEntropy loss and ClassificationError output.
    ///
    CNTK_API FunctionPtr Combine(const std::vector<Variable>& operands, const std::wstring& name = std::wstring());

    ///
    /// Creates a new Function instance which is just an alias of the specified operand.
    ///
    CNTK_API FunctionPtr Alias(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Creates a new Function instance which is just an alias of the specified operand, and enforces all values to be synchronized in auto-batching.
    ///
    CNTK_API FunctionPtr BatchSync(const Variable& operand, size_t id, const std::wstring& name = std::wstring());

    ///
    /// Creates a Block Function that encapsulates a composite to create an opaque Function object that
    /// appears as any other primitive Function
    ///
    CNTK_API FunctionPtr AsBlock(FunctionPtr&& composite, const std::vector<std::pair<Variable, Variable>>& argumentsMap, const std::wstring& blockOpName, const std::wstring& blockName = std::wstring());

    // TODO: Move this to Internal. Not trivial since Variable is not known inside the Internal header.
    class Invocable
    {
        const size_t m_arity; // actual number of function arguments. Note: m_argumentList/m_operands contain additional leaves, so its size() is not sufficient.
        const bool m_isBasicBlock;
        mutable std::vector<Variable> m_argumentList; // contains the Variables in the composite. May be updated upon determining the shapes with PlaceholderLikes.
        std::vector<size_t> m_argumentFreeAxes;       // axis where the FreeDimension goes, for each argument
        mutable std::vector<Variable> m_operands;     // these are overwritten upon each call
        FunctionPtr m_composite; // Note: multiple calls to Invoke() assume that the composite does not change; so encapsulate it here.
        mutable bool m_stillNeedsToInferShapes;

        // for debugging: allow to directly call the lambda, without any Composite involved
        std::function<Variable(const std::vector<Variable>&)> m_lambdaRememberedForDebugging;
        std::wstring m_nameRememberedForDebugging;

        CNTK_API Invocable(size_t arity, size_t freeAxis, bool isBasicBlock, const std::function<Variable(const std::vector<Variable>&)>& f, std::wstring name);
        void CheckArity(size_t arity) const
        {
            if (m_arity != arity)
                LogicError("Invocable: It was attempted to invoke a %d-nary function with %d arguments.", (int)m_arity, (int)arity);
        }
        void SetOperand(size_t argIndex, const Variable& argVal) const
        {
            m_operands[argIndex] = argVal;
        }
        Variable m_noArg; // dummy for clearing out the args map
        CNTK_API Variable DoInvoke() const; // note: caller must call SetOperand() first to set the operands
        //Variable Invoke(const FunctionPtr& composite, std::vector<Variable>& argumentList, const std::vector<Variable>& operands, bool isBasicBlock, bool& determineShapes, const std::wstring& name = std::wstring()) const;
        // TODO: ^^ merge Invoke() into DoInvoke()
    public:
        Invocable(bool isBasicBlock, size_t freeAxis, const std::function<Variable(                                                 )>& f, std::wstring name) : Invocable(0, freeAxis, isBasicBlock, [=](const std::vector<Variable>& args) { args; return f(                   ); }, name) { }
        Invocable(bool isBasicBlock, size_t freeAxis, const std::function<Variable(const Variable&                                  )>& f, std::wstring name) : Invocable(1, freeAxis, isBasicBlock, [=](const std::vector<Variable>& args) { return f(args[0]                  ); }, name) { }
        Invocable(bool isBasicBlock, size_t freeAxis, const std::function<Variable(const Variable&, const Variable&                 )>& f, std::wstring name) : Invocable(2, freeAxis, isBasicBlock, [=](const std::vector<Variable>& args) { return f(args[0], args[1]         ); }, name) { }
        Invocable(bool isBasicBlock, size_t freeAxis, const std::function<Variable(const Variable&, const Variable&, const Variable&)>& f, std::wstring name) : Invocable(3, freeAxis, isBasicBlock, [=](const std::vector<Variable>& args) { return f(args[0], args[1], args[2]); }, name) { }
        Variable operator()() const
        {
            CheckArity(0);
            return DoInvoke();
        }
        Variable operator()(const Variable& x1) const
        {
            CheckArity(1);
            SetOperand(0, x1);
            return DoInvoke();
        }
        Variable operator()(const Variable& x1, const Variable& x2) const
        {
            CheckArity(2);
            SetOperand(0, x1);
            SetOperand(1, x2);
            return DoInvoke();
        }
        Variable operator()(const Variable& x1, const Variable& x2, const Variable& x3) const
        {
            CheckArity(3);
            SetOperand(0, x1);
            SetOperand(1, x2);
            SetOperand(2, x3);
            return DoInvoke();
        }
    };

    ///
    /// Creates a new Function instance which output its input as it is and previent any gradient contribution from its output.
    ///
    CNTK_API FunctionPtr StopGradient(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Assign the value in operand to ref and return the new value, ref need to be the same layout as operand.
    /// During forward pass, ref will get the new value after the forward or backward pass finish, so that any part of
    /// the graph that depend on ref will get the old value. To get the new value, use the one returned by
    /// the assign node.The reason for that is to make ``assign`` have a deterministic behavior.
    /// During inference the value of ref wull be updated after the forward pass and during training the value
    /// of ref will be updated after backprop.
    ///
    CNTK_API FunctionPtr Assign(Variable& ref, const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Creates a composite Function that has the specified rootFunction as its root.
    /// The composite denotes a higher-level Function encapsulating the entire graph
    /// of Functions underlying the specified rootFunction.
    ///
    /// TODO: This is not something that external users should do. Move to an appropriate place. 
    CNTK_API FunctionPtr AsComposite(const /*Primitive*/FunctionPtr& rootFunction, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise exponential linear unit operation with the specified input operand.
    ///
    CNTK_API FunctionPtr ELU(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise scaled exponential linear unit operation with the specified input operand.
    ///
    CNTK_API FunctionPtr SELU(const Variable& operand, double scale = 1.0507009873554804934193349852946, double alpha = 1.6732632423543772848170429916717, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise leaky linear rectifier operation with the specified input operand.
    ///
    CNTK_API FunctionPtr LeakyReLU(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise parametric rectified linear Unit operation
    /// with the specified input operand and learning parameter alpha.
    ///
    CNTK_API FunctionPtr PReLU(const Variable& alpha, const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in elementwise softplus operation
    ///
    CNTK_API FunctionPtr Softplus(const Variable& operand, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in argmax operation on specified tensor input operand along the specified axis
    ///
    CNTK_API FunctionPtr Argmax(const Variable& operand, const Axis& axis, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in argmin on specified tensor input operand along the specified axis
    ///
    CNTK_API FunctionPtr Argmin(const Variable& operand, const Axis& axis, const std::wstring& name = std::wstring());
 
    ///
    /// Create an instance of the CNTK built-in operator for converting the specified tensor operand into a sequence
    ///
    CNTK_API FunctionPtr ToSequence(const Variable& operand, const std::wstring& sequenceAxisNamePrefix, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operator for converting the specified tensor operand into a sequence
    /// This overload allows specifying an additional operand containing the lengths of individual sequences
    ///
    CNTK_API FunctionPtr ToSequence(const Variable& operand, const Variable& sequenceLengths, const std::wstring& sequenceAxisNamePrefix, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operator for converting the specified tensor operand into a sequence
    /// This overload allows specifying an additional 'dynamicAxesLike' operand which is used to obtain the lengths of the
    /// generated sequences; the dynamic axes of the generated sequence are required to match the dynamic axes of the 'dynamicAxesLike' operand.
    ///
    CNTK_API FunctionPtr ToSequenceLike(const Variable& operand, const Variable& dynamicAxesLike, const std::wstring& name = std::wstring());

    ///
    /// Create an instance of the CNTK built-in operator for reconciling the dynamic axes of the specified tensor operands.
    /// The output of the returned Function has the sample layout of the left operand and the dynamic axes of the axesAsOperand.
    /// It also performs a runtime check to ensure that the  dynamic axes layouts of the 2 operands indeed match.
    ///
    CNTK_API FunctionPtr ReconcileDynamicAxes(const Variable& operand, const Variable& axesAsOperand, const std::wstring& name = std::wstring());

    namespace Sequence
    {
        CNTK_API FunctionPtr IsFirst(const Variable& operand, const std::wstring& name = std::wstring());
        CNTK_API FunctionPtr IsLast(const Variable& operand, const std::wstring& name = std::wstring());

        CNTK_API FunctionPtr Slice(const Variable& operand, int beginIndex, int endIndex, const std::wstring& name = std::wstring());

        ///
        /// Create an instance of the CNTK built-in sum reduction operation on specified tensor input operand along the operands lone dynamic sequence axis
        ///
        CNTK_API FunctionPtr ReduceSum(const Variable& operand, const std::wstring& name = std::wstring());

        CNTK_API FunctionPtr ReduceMax(const Variable& operand, const std::wstring& name = std::wstring());

        CNTK_API FunctionPtr Softmax(const Variable& operand, const std::wstring& name = std::wstring());

        CNTK_API FunctionPtr First(const Variable& operand, const std::wstring& name = std::wstring());
        CNTK_API FunctionPtr Last(const Variable& operand, const std::wstring& name = std::wstring());

        CNTK_API FunctionPtr Where(const Variable& condition, const std::wstring& name = std::wstring());
        CNTK_API FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::wstring& name = std::wstring());
        CNTK_API FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, const std::wstring& name = std::wstring());

        CNTK_API FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::wstring& name = std::wstring());
        CNTK_API FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, const std::wstring& name = std::wstring());

        CNTK_API FunctionPtr BroadcastAs(const Variable& operand, const Variable& broadcastAs, const std::wstring& name = std::wstring());

        ///
        /// Create an instance of the CNTK built-in operator for unpacking the specified sequence operand along
        /// the most significant static axis [-1] and padding any gaps with the specified padding value.
        /// If supressMaskOutput is false, the returned Function has 2 outputs; viz. the unpacked non-sequence data and a mask
        /// denoting the gaps in the unpacked output due to differences across lengths of the sequences in the operand.
        ///
        CNTK_API FunctionPtr Unpack(const Variable& operand, double paddingValue, bool supressMaskOutput, const std::wstring& name = std::wstring());
    }

    ///
    /// A collection of key-value pairs that represents a training parameter schedule
    /// (e.g., learning rate, momentum schedule) in terms of the number of samples.
    /// This class is designed to simplify Learner's factory methods and provides a number of
    /// convenience constructors to allow easy conversion from a single value, a vector of values
    /// and a list of pairs to the training schedule. For example, a learning rate schedule
    /// { { 10, 0.5 }, { 100, 0.3 }, { 20, 0.2 } } indicates that the rate of 0.5 should be
    /// used for the first 10 samples, followed by 0.3 for the next 100 samples, and then 0.2 for
    /// the remaining 20 samples or until the end of training if it takes longer.
    ///
    template <typename T>
    class TrainingParameterSchedule : public IDictionarySerializable
    {
    public:
        /// A special value that can be used for the epochSize to indicate that the schedule is sweep-based.
        ///
        static const size_t FullDataSweep = 0;
        ///
        /// A special value that can be used for the minibatchSize to indicate that the reference minibatch size is not specified.
        ///
        static const size_t IgnoredMinibatchSize = 0;
        ///
        /// Create a schedule with a constant parameter value.
        /// @param value a single value to populate the schedule
        /// @param minibatchSize a minibatch size that the @e value specifies for.
        ///
        CNTK_API TrainingParameterSchedule(T value, size_t minibatchSize = IgnoredMinibatchSize);

#ifndef SWIG
        ///
        /// Create a schedule where the parameter changes its value every 'epochSize' samples:
        /// schedule[0] is used for the first 'epochSize' samples, schedule[1] -- for the second,
        /// and so on. The last value is then used repeatedly until the end of training. 
        /// @e minibatchSize is the a minibatch size that each schedule[i] specifies for.
        ///
        CNTK_API TrainingParameterSchedule(const std::vector<T>& schedule, size_t epochSize = FullDataSweep, size_t minibatchSize = IgnoredMinibatchSize);
#endif

        ///
        /// Create a schedule using the list of key-value pairs, where the key specifies
        /// the number of epochs the parameter should maintain the corresponding value,
        /// (which 'epochSize' samples in each epoch). The value from the last pair is used
        /// repeatedly until the end of training. For example, {{1, 0.05}, {2, 0.1}, {1, 0.005}}
        /// and epochSize = 100, corresponds to a schedule where the value of '0.05' is used for
        /// the first 100 samples, then '0.1' is used for the second 200 samples,
        /// after which the values is switched to '0.005'.
        /// @e minibatchSize is the a minibatch size that each schedule[i] specifies for.
        ///
        CNTK_API TrainingParameterSchedule(const std::vector<std::pair<size_t, T>>& schedule, size_t epochSize = FullDataSweep, size_t minibatchSize = IgnoredMinibatchSize);


        ///
        /// Returns a value corresponding to the absolute sample (or sweep)
        /// count from the beginning of training.
        ///
        CNTK_API const T& operator[](size_t count) const;

        bool IsSweepBased() const { return m_epochSize == FullDataSweep; }

        CNTK_API virtual ~TrainingParameterSchedule();

        CNTK_API TrainingParameterSchedule(const TrainingParameterSchedule<T>&);
        CNTK_API TrainingParameterSchedule(TrainingParameterSchedule<T>&&);
        CNTK_API TrainingParameterSchedule<T>& operator=(const TrainingParameterSchedule<T>&);
        CNTK_API TrainingParameterSchedule<T>& operator=(TrainingParameterSchedule<T>&&);

        CNTK_API virtual Dictionary Serialize() const override;

        virtual size_t CurrentVersion() const override { return s_serializationVersion; }

        CNTK_API static TrainingParameterSchedule<T> Deserialize(const Dictionary& dictionary);

        CNTK_API bool operator==(const TrainingParameterSchedule<T>& right)
        {
            return this->m_schedule == right.m_schedule
                && this->m_epochSize == right.m_epochSize
                && this->m_minibatchSize == right.m_minibatchSize;
        }

        CNTK_API TrainingParameterSchedule<T>& Transform(std::function<T(const T&)> func);

        CNTK_API size_t GetMinibatchSize() const { return m_minibatchSize; }
        CNTK_API void SetMinibatchSize(size_t minibatchSize) { m_minibatchSize = minibatchSize; }
    private:

        friend class Learner;

        CNTK_API void ConstructSchedule(const std::vector<std::pair<size_t, T>>& schedule);

        CNTK_API TrainingParameterSchedule(const Dictionary& dictionary);

        ///Version history:
        ///1 --- initial version.
        ///2 --- removed UnitType and intrudoce reference minibath size
        static const size_t s_serializationVersion = 2;

    protected:
        std::map<size_t, T> m_schedule;
        //TODO: enable reference mb size for each rate
        size_t m_minibatchSize; ///< reference design minibatch size the training parameter schedule are targeting at
        size_t m_epochSize;
    };

    typedef TrainingParameterSchedule<double> LearningRateSchedule;
    typedef TrainingParameterSchedule<double> MomentumSchedule;
    typedef TrainingParameterSchedule<size_t> MinibatchSizeSchedule;
    
    
    //The following are for convenient usages:
    template<typename T>
    TrainingParameterSchedule<T> TrainingParameterPerSampleSchedule(const std::vector<T>& schedule, size_t epochSize = TrainingParameterSchedule<T>::FullDataSweep)
    {
        return TrainingParameterSchedule<T>(schedule, epochSize, 1);
    }

    template<typename T>
    TrainingParameterSchedule<T> TrainingParameterPerSampleSchedule(const std::vector<std::pair<size_t, T>>& schedule, size_t epochSize = TrainingParameterSchedule<T>::FullDataSweep)
    {
        return TrainingParameterSchedule<T>(schedule, epochSize, 1);
    }

    template<typename T>
    TrainingParameterSchedule<T> TrainingParameterPerSampleSchedule(T value)
    {
        return TrainingParameterSchedule<T>(value, 1);
    }

    ///Compute the momentum from time constant.
    ///For backward compatability only. *Will be deprecated*.
    inline double MomentumFromTimeConstant(double momTC)
    {
        return momTC == 0.0 ? 0 : exp(-1.0 / momTC);
    }

    ///Construct MomentumAsTimeCosntantSchedule. *Will be deprecated*.
    CNTK_API MomentumSchedule MomentumAsTimeConstantSchedule(double time_constant);
    ///Construct MomentumAsTimeCosntantSchedule. *Will be deprecated*.
    CNTK_API MomentumSchedule MomentumAsTimeConstantSchedule(const MomentumSchedule& schedule); 
    ///Construct MomentumAsTimeCosntantSchedule. *Will be deprecated*.
    CNTK_API MomentumSchedule MomentumAsTimeConstantSchedule(const std::vector<double>& schedule, size_t epoch_size = MomentumSchedule::FullDataSweep);
    ///Construct MomentumAsTimeCosntantSchedule. *Will be deprecated*.
    CNTK_API MomentumSchedule MomentumAsTimeConstantSchedule(const std::vector<std::pair<size_t, double>>& schedule,  size_t epoch_size = MomentumSchedule::FullDataSweep);


    ///
    /// A collection of additional options that affect parameter updates and
    /// are applicable for all standard learners
    ///
    struct AdditionalLearningOptions 
    //TODO: replace the struct option with dictOptions
    {
        double l1RegularizationWeight = 0.0;
        double l2RegularizationWeight = 0.0;
        TrainingParameterSchedule<double> gaussianNoiseInjectionStdDev = 0.0;
        double gradientClippingThresholdPerSample = std::numeric_limits<double>::infinity();
        bool gradientClippingWithTruncation = true;

        Dictionary dictOptions;
    };

    ///
    /// Returns true if by default momentum is applied in the unit-gain fashion.
    ///
    CNTK_API bool DefaultUnitGainValue();

    ///
    /// Sets globally default unit-gain flag value.
    ///
    CNTK_API void SetDefaultUnitGainValue(bool value);

    ///
    /// Abstraction for learning a subset of parameters of a learnable Function using first order gradient values.
    /// For example momentum, AdaGrad, RMSProp, etc. are different types of learners with their own algorithms for
    /// learning parameter values using first order gradients.
    ///
    class Learner
    {
    public:
        ///
        /// A key that is associated with MinibatchSize.
        ///
        CNTK_API static const std::wstring MinibatchSizeKey;
        ///
        /// A special value that can be used for the minibatchSize to indicate that the reference minibatch size is not specified.
        ///
        CNTK_API static const size_t IgnoredMinibatchSize;

    public:
        //
        // Method to update the parameters associated with this learner. By returning false, this method indicates that
        // learning has stopped for all of the parameters associated with this learner
        //
        virtual bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, size_t trainingSampleCount, bool sweepEnd = false) = 0;

        ///
        /// Returns the set of parameters associated with this learner.
        ///
        virtual const std::vector<Parameter>& Parameters() const { return m_parameters; }

        ///
        /// Optionally overridable method to checkpoint the learner's state.
        ///
        virtual Dictionary CreateCheckpoint() { return Dictionary(); }

        ///
        /// Optionally overridable method to restore the learner's state from a previous checkpoint.
        ///
        virtual void RestoreFromCheckpoint(const Dictionary&) { NOT_IMPLEMENTED }

        ///
        /// Destruct this Learner.
        ///
        virtual ~Learner() {}

        ///
        /// This method needs to be explicitly overriden in subclasses.
        ///
        virtual size_t CurrentVersion() const { NOT_IMPLEMENTED }

        ///
        /// Sets a new learning rate overriding the schedule parameter used to construct this learner.
        /// The new schedule is adjusted to be relative to the current number of elapsed samples/sweeps:
        /// the 0 offset in the new schedule corresponds to the current value of elapsed samples/sweeps,
        /// and it takes effect from the current position in the training process onwards.
        ///
        CNTK_API virtual void ResetLearningRate(const LearningRateSchedule& learningRateSchedule);

        ///
        /// Resets smoothed gradients.
        ///
        virtual void ResetSmoothedGradients() = 0;

        ///
        /// Returns current learning rate.
        ///
        virtual double LearningRate() const
        {
            return GetCurrentTrainingParameterValue<double>(m_learningRateSchedule);
        }

        size_t TotalNumberOfSamplesSeen() const
        {
            return m_sampleCount;
        }

        ///
        /// Specifies progress writers that should be used to report any relevant stats.
        ///
        void AddProgressWriters(const std::vector<ProgressWriterPtr>& progressWriters)
        {
            m_progressWriters.insert(progressWriters.begin(), progressWriters.end());
        }

        CNTK_API Dictionary& GetOptions() { return m_additionalOptions.dictOptions; }
        CNTK_API const Dictionary& GetOptions() const { return m_additionalOptions.dictOptions; }

        ///In the litature, usually the learner hyper-parameters, such as the learning rates and other hyper-parameters (such as those 
        ///in momentum SGD or ADAM), are chosen for the specified minibatch size. However, for efficient implementation and for distributed training,
        ///CNTK can vary the actual minibatch sizes for better computational efficiency. Therefore CNTK allows users to set
        ///the reference minibatch size. CNTK will try its best to adjust the learning hyper-parameters internally to match the
        ///behavior of the learning parameters with the specified specified minibatch size while the actual minibatch size
        ///can vary for better computational performance. If minibatchSize is set to 0, CNTK will apply the hyper-parameters
        ///over the whole minibatch as it is without any underlying scaling. 
        ///Note the underlying TrainingParameterSchedule's reference minibatch size setting can over this reference minibatch size
        ///setting and be specialized to its own reference minibatch size. However, this is only suggested for advanced
        ///users.
        CNTK_API void SetMinibatchSize(std::size_t minibatchSize) { GetOptions().Add(MinibatchSizeKey, minibatchSize); }
        CNTK_API std::size_t GetMinibatchSize() const { return GetOptions().GetOrElse(MinibatchSizeKey, IgnoredMinibatchSize); }

        CNTK_API void SetLearningRateSchedule(const LearningRateSchedule& learningRateSchedule) { m_learningRateSchedule = learningRateSchedule; }
        CNTK_API const LearningRateSchedule& GetLearningRateSchedule() const { return m_learningRateSchedule; }

        ///Return whether the learning schedule indicates a literature compatible mode to use mean gradient and potentially other adjustment of the parameters if necessary.
        template<typename T>
        static bool IsCompatibleMode(const TrainingParameterSchedule<T>& schedule)
        {
            return schedule.GetMinibatchSize() == IgnoredMinibatchSize;
        }

        ///
        ///Return whether the learner is in literature compatible mode to use mean gradient and the adjustment 
        ///of the parameters if necessary.
        ///
        CNTK_API bool IsCompatibleMode() const
        {
            if (GetOptions().Contains(MinibatchSizeKey))
            {
                return GetMinibatchSize() == IgnoredMinibatchSize;
            }
            else
                //if the learner minbiatch size is not set, by default it is not in compatible mode.
                return false;
        }

    protected:
        ///
        /// Retrieves and returns current value from the training parameter schedule.
        ///
        template <typename ElementType>
        ElementType GetCurrentTrainingParameterValue(const TrainingParameterSchedule<ElementType>& schedule) const
        {
            auto count = schedule.IsSweepBased() ? m_sweepCount : m_sampleCount;
            return schedule[count];
        }

        Learner(const std::vector<Parameter>& parameters, const LearningRateSchedule& learningRateSchedule, const AdditionalLearningOptions& additionalOptions = AdditionalLearningOptions())
            : m_parameters(parameters),
            m_learningRateSchedule(learningRateSchedule),
            m_additionalOptions(additionalOptions),
            m_sampleCount(0),
            m_minibatchCount(0),
            m_sweepCount(0)
        {}

        std::vector<Parameter> m_parameters;
        LearningRateSchedule m_learningRateSchedule;
        size_t m_sampleCount;
        size_t m_minibatchCount;
        size_t m_sweepCount;
        AdditionalLearningOptions m_additionalOptions;
        mutable std::unordered_set<ProgressWriterPtr> m_progressWriters;
    };

    ///
    /// Create an instance of the CNTK built-in SGD learner.
    ///
    CNTK_API LearnerPtr SGDLearner(const std::vector<Parameter>& parameters,
                                   const LearningRateSchedule& learningRateSchedule,
                                   AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

    ///
    /// Create an instance of the CNTK built-in Momentum SGD learner.
    ///
    CNTK_API LearnerPtr MomentumSGDLearner(const std::vector<Parameter>& parameters,
                                           const LearningRateSchedule& learningRateSchedule,
                                           const MomentumSchedule& momentumSchedule,
                                           bool unitGain = DefaultUnitGainValue(),
                                           AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

    ///
    /// Create an instance of the CNTK built-in Nesterov's accelerated SGD learner.
    ///
    CNTK_API LearnerPtr NesterovLearner(const std::vector<Parameter>& parameters,
                                        const LearningRateSchedule& learningRateSchedule,
                                        const MomentumSchedule& momentumSchedule,
                                        bool unitGain = DefaultUnitGainValue(),
                                        AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

    static MomentumSchedule DefaultVarianceMomentum = MomentumAsTimeConstantSchedule(2 * 3600 * 100);

    ///
    /// Create an instance of FSAdaGrad learner as the original paper.
    ///
    CNTK_API LearnerPtr FSAdaGradLearner(const std::vector<Parameter>& parameters,
                                         const LearningRateSchedule& learningRateSchedule,
                                         const MomentumSchedule& momentumSchedule,
                                         bool unitGain = DefaultUnitGainValue(),
                                         const MomentumSchedule& varianceMomentumSchedule = DefaultVarianceMomentum,
                                         AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

    ///
    /// Create an instance of Adam learner as the original paper.
    ///
    CNTK_API LearnerPtr AdamLearner(const std::vector<Parameter>& parameters,
                                    const LearningRateSchedule& learningRateSchedule,
                                    const MomentumSchedule& momentumSchedule,
                                    bool unitGain = DefaultUnitGainValue(),
                                    const MomentumSchedule& varianceMomentumSchedule = DefaultVarianceMomentum,
                                    double epsilon = 1e-8,
                                    bool adamax = false,
                                    AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

    ///
    /// Create an instance of the CNTK built-in AdaGrad learner.
    ///
    CNTK_API LearnerPtr AdaGradLearner(const std::vector<Parameter>& parameters,
                                       const LearningRateSchedule& learningRateSchedule,
                                       bool needAveMultiplier = true,
                                       AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

    ///
    /// Create an instance of the CNTK built-in RMSProp learner.
    ///
    CNTK_API LearnerPtr RMSPropLearner(const std::vector<Parameter>& parameters,
                                       const LearningRateSchedule& learningRateSchedule,
                                       double gamma,
                                       double inc,
                                       double dec,
                                       double max,
                                       double min,
                                       bool needAveMultiplier = true,
                                       AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

    ///
    /// Create an instance of the CNTK built-in AdaDelta learner.
    ///
    CNTK_API LearnerPtr AdaDeltaLearner(const std::vector<Parameter>& parameters,
                                        const LearningRateSchedule& learningRateSchedule,
                                        double rho = 0.95,
                                        double epsilon = 1e-8,
                                        AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

    ///
    /// A shorthand for the type of a function that takes a Parameter and a Variable as arguments and returns a Function.
    /// It can be used with UniversalLearner.
    ///
    typedef std::function<FunctionPtr(Parameter, Variable)> ParameterUpdateFunctor;

    ///
    /// Create an instance of a learner whose update is given by the specified factory which returns a CNTK FunctionPtr.
    ///
    CNTK_API LearnerPtr UniversalLearner(const std::vector<Parameter>& parameters, const ParameterUpdateFunctor& func);

    ///
    /// Create an instance of a learner by specifying the parameters , gradients and update function. Return a CNTK FunctionPtr.
    ///
    CNTK_API LearnerPtr UniversalLearner(const std::vector<Parameter>& parameters, const std::vector<Variable>& gradients, FunctionPtr updateFunc);

    ///
    /// Distributed Learner.
    ///
    class DistributedLearner : public Learner
    {
    public:
        ///
        /// Returns the distributed communicator used in the distributed learner
        ///
        CNTK_API virtual DistributedCommunicatorPtr GetCommunicator() const
        {
            return m_communicator;
        }

        bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, size_t minibatchSampleCount, bool sweepEnd = false) override
        {
            MinibatchInfo info{ false, sweepEnd, minibatchSampleCount };
            return Update(gradientValues, info);
        }

        virtual void ResetLearningRate(const LearningRateSchedule& learningRateSchedule)
        {
            m_learner->ResetLearningRate(learningRateSchedule);
        }

        virtual double LearningRate() const
        {
            return m_learner->LearningRate();
        }

        void ResetSmoothedGradients() override
        {
            m_learner->ResetSmoothedGradients();
        }

        //
        // Returns the total number of samples needed for warmup.
        // After reaching this number of samples the learner switches to the distributed mode.
        // Warm up is useful for
        //
        virtual size_t ParallelizationAfter()
        {
            return m_distributeAfterSamples;
        }

        //
        // Method to update the parameters associated with this learner. By returning false, this method indicates that
        // learning has stopped for all of the parameters associated with this learner
        //
        CNTK_API virtual bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, MinibatchInfo& minibatch) = 0;

        //
        // In distributed mode all built-in minibatch sources return a minibatch decimated by the number of workers.
        // Some distributed methods (i.e. BlockMomentum) require each worker to run with original/not decimated minibatch size.
        // This method is used by the training session to adapt minibatch size before asking the minibatch source for data.
        // The function returns the scale factor for the minibatch size.
        //
        virtual size_t MinibatchSizeScaleFactor()
        {
            return 1;
        }

    protected:
        DistributedLearner(DistributedCommunicatorPtr communicator, LearnerPtr learner, size_t distributeAfterSamples)
            : Learner(learner? learner->Parameters() : std::vector<Parameter>(),
                      LearningRateSchedule(0)),
              m_learner(learner),
              m_communicator(communicator),
              m_distributeAfterSamples(distributeAfterSamples)
        {
            if (!m_learner)
                InvalidArgument("Learner passed to a Distributed learner ctor must not be null.");

            if (!m_communicator)
                InvalidArgument("Communicator passed to a Distributed learner ctor must not be null.");
        }

        const LearnerPtr m_learner;
        const DistributedCommunicatorPtr m_communicator;
        const size_t m_distributeAfterSamples;

        // Disallow copy and move construction and assignment
        DistributedLearner(const DistributedLearner&) = delete; DistributedLearner& operator=(const DistributedLearner&) = delete; DistributedLearner& operator=(DistributedLearner&&) = delete; DistributedLearner(DistributedLearner&&) = delete;
    };

    CNTK_API DistributedLearnerPtr CreateDataParallelDistributedLearner(DistributedCommunicatorPtr communicator, LearnerPtr learner, size_t distributeAfterSamples, bool useAsyncBufferedParameterUpdate = false);

    CNTK_API DistributedLearnerPtr CreateQuantizedDataParallelDistributedLearner(QuantizedDistributedCommunicatorPtr communicator, LearnerPtr learner, size_t distributeAfterSamples, bool useAsyncBufferedParameterUpdate = false);

    CNTK_API DistributedLearnerPtr CreateBlockMomentumDistributedLearner(
        DistributedCommunicatorPtr communicator,
        LearnerPtr learner,
        size_t distributeAfterSamples,
        size_t blockSize,
        double blockMomentumAsTimeConstant,
        bool useNestrovMomentum = true,
        bool resetSGDMomentumAfterAggregation = true,
        double blockLearningRate = 1.0);

    CNTK_API DistributedLearnerPtr CreateBlockMomentumDistributedLearner(
        DistributedCommunicatorPtr communicator,
        LearnerPtr learner,
        size_t distributeAfterSamples,
        size_t blockSize,
        bool useNestrovMomentum = true,
        bool resetSGDMomentumAfterAggregation = true,
        double blockLearningRate = 1.0);

    ///
    /// Evaluator is a top-level abstraction for evaluating a model's performance with specified error criterion.
    ///
    class Evaluator : public std::enable_shared_from_this<Evaluator>
    {
    public:
        ///
        /// Tests the model on the specified batch of samples using the evaluation Function specified during construction of the Evaluator
        /// Returns the average evaluation criterion value per sample for the tested minibatch of samples
        ///
        CNTK_API double TestMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// An overload of the TestMinibatch above that takes a map of variables and their values (as its first argument).
        ///
        CNTK_API double TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// An overload of the TestMinibatch above that takes a map of output variables.
        ///
        CNTK_API double TestMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// An overload of the TestMinibatch above that takes a map of output variables.
        ///
        CNTK_API double TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Evaluation Function that is used as for the criterion for evaluating the trained model's quality.
        ///
        FunctionPtr EvaluationFunction() const { return m_evaluationFunction; }

        ///
        /// Writes the summary of test progress and resets the accumulators.
        ///
        CNTK_API void SummarizeTestProgress();

        ///
        /// Progress writers.
        ///
        CNTK_API const std::unordered_set<ProgressWriterPtr>& ProgressWriters() const
        {
            return m_progressWriters;
        }

        CNTK_API virtual ~Evaluator() {}

    private:
        template <typename T1, typename ...CtorArgTypes>
        friend std::shared_ptr<T1> MakeSharedObject(CtorArgTypes&& ...ctorArgs);

        friend class TrainingSession;

        // Returns true if testing should be continued in a distributed mode.
        // Aggregated error and sample count can be retrieved using 'result' parameter.
        bool TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::pair<ValuePtr, size_t>& result, const DeviceDescriptor& computeDevice, bool distributed = false);
        bool TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, std::pair<ValuePtr, size_t>& result, const DeviceDescriptor& computeDevice, bool distributed = false);

        std::pair<ValuePtr, size_t> TestLocalMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice);

        void UpdateTestProgress(size_t numSamples, const ValuePtr& evalCriterion, const DeviceDescriptor& computeDevice);

    protected:
    public: // public for MakeSharedObject() only. TODO: Remove once we know how to do that right.
        Evaluator(const FunctionPtr& evaluationFunction, const std::vector<ProgressWriterPtr>& progressWriters = {}, bool initializeCombined = true);
    protected:

        // Helper functions.
        std::vector<Variable> GetCombinedEvalFunctionArgs() const;
        static size_t GetSampleCount(const Variable& var, const ValuePtr& value);
        static std::unordered_map<Variable, ValuePtr> GetInputs(const std::unordered_map<Variable, MinibatchData>& arguments);


        // The combined eval function can be reset by the derived classes.
        void SetCombinedEvalFunction(FunctionPtr combinedEvalFunction)
        {
            if (m_combinedEvalFunction != nullptr)
                LogicError("Combined eval function has already been initialized.");

            if (!combinedEvalFunction)
                InvalidArgument("Combined eval function is not allowed to be null.");

            m_combinedEvalFunction = combinedEvalFunction;
        }

        FunctionPtr m_evaluationFunction;
        FunctionPtr m_aggregatedEvaluationFunction;
        std::unordered_set<ProgressWriterPtr> m_progressWriters;

    private:
        const AccumulatorPtr m_aggregatedTestEvalCriterionValue;
        Variable             m_testSampleCountVar;
        FunctionPtr          m_combinedEvalFunction;
    };

    ///
    /// Construct an Evaluator for the specified eval function.
    ///
    CNTK_API EvaluatorPtr CreateEvaluator(const FunctionPtr& evaluationFunction, const std::vector<ProgressWriterPtr>& progressWriters = {});

    ///
    /// Trainer is the top-level abstraction responsible for the orchestration of the training of a model
    /// using the specified learners and training data either explicitly supplied as Value objects or from
    /// a MinibatchSource object.
    ///
    class Trainer : public Evaluator
    {
    public:
        ///
        /// Optimize model parameters using the specified 'arguments' minibatch of training samples.
        /// Returns false if all parameter learners indicate end of learning (through their Update method's return value).
        ///
        CNTK_API bool TrainMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// An overload of the TrainMinibatch above that takes a map of variables and their values (as its first argument).
        ///
        CNTK_API bool TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Optimize model parameters using the specified 'arguments' minibatch of training samples.
        /// The variables specified in the 'outputsToFetch' map denote the subset of 'this' Function's output variables that the caller wants to obtain values of.
        /// Callers may specify the storage to be used for storing the 'outputs' Values or pass null in which case the implementation allocates the actual storage
        /// for the 'outputs' for which the ValuePtr mapping was left null by the caller.
        /// Returns false if all parameter learners indicate end of learning (through their Update method's return value).
        ///
        CNTK_API bool TrainMinibatch(const std::unordered_map<Variable, MinibatchData>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// An overload of the TrainMinibatch above that takes a map of variables and their values (as its first argument).
        ///
        CNTK_API bool TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Checkpoint the model and other Trainer state at the specified file location
        ///
        CNTK_API void SaveCheckpoint(const std::wstring& filePath, Dictionary externalState = Dictionary());

        ///
        /// Restore the model and trainer state from a previously saved model and checkpoint from the specified file location
        ///
        CNTK_API Dictionary RestoreFromCheckpoint(const std::wstring& filePath);

        ///
        /// Model being trained by 'this' Trainer.
        ///
        FunctionPtr Model() const { return m_model; }

        ///
        /// Loss Function that is used as the optimization criterion for learning the model's parameters.
        ///
        FunctionPtr LossFunction() const { return m_lossFunction; }

        ///
        /// Returns the average training loss per sample for the last minibatch trained.
        ///
        CNTK_API double PreviousMinibatchLossAverage() const;

        ///
        /// Returns the average evaluation criterion value per sample for the last minibatch trained.
        ///
        CNTK_API double PreviousMinibatchEvaluationAverage() const;

        ///
        /// Returns the number of samples in the last minibatch trained with.
        ///
        size_t PreviousMinibatchSampleCount() const { return m_prevMinibatchNumSamples; }

        ///
        /// Learners associated with this Trainer for updating the model's parameters using computed gradients.
        ///
        CNTK_API const std::vector<LearnerPtr>& ParameterLearners() const;

        ///
        /// Total number of samples seen from the beginning of the training.
        ///
        CNTK_API size_t TotalNumberOfSamplesSeen() const;

        ///
        /// Writes the summary of training progress and resets the accumulators.
        ///
        CNTK_API void SummarizeTrainingProgress();

    private:
        template <typename T1, typename ...CtorArgTypes>
        friend std::shared_ptr<T1> MakeSharedObject(CtorArgTypes&& ...ctorArgs);

        friend class TrainingSession;

    public: // public for MakeSharedObject() only. TODO: Remove once we know how to do that right.
        Trainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const std::vector<LearnerPtr>& parameterLearners,
                const std::vector<ProgressWriterPtr>& progressWriters = {});
        Trainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const FunctionPtr& evaluationFunction, const std::vector<LearnerPtr>& parameterLearners,
                const std::vector<ProgressWriterPtr>& progressWriters = {});
    private:

        void ExecuteForwardBackward(
            const std::unordered_map<Variable, ValuePtr>& arguments,
            std::unordered_map<Variable, ValuePtr>& outputsToFetch,
            const DeviceDescriptor& computeDevice,
            std::unordered_map<Variable, ValuePtr>& parameterGradients);

        bool TrainLocalMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, bool sweepEnd, const DeviceDescriptor& computeDevice);
        bool TrainDistributedMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, bool sweepEnd, const DeviceDescriptor& computeDevice);

        void Save(const std::wstring& modelFilePath, const std::vector<DictionaryValue>& learnerState,
            const Dictionary& externalState, const Dictionary& distributedState = {});

        void UpdateTrainingProgress(size_t numSamples, const ValuePtr& loss, const ValuePtr& evalCriterion, const DeviceDescriptor& computeDevice);
        void AddProgressWriters(const std::vector<ProgressWriterPtr>& progressWriters);

        FunctionPtr m_model;
        FunctionPtr m_combinedTrainingFunction;
        FunctionPtr m_lossFunction;
        FunctionPtr m_aggregatedLossFunction;
        Variable    m_trainingSampleCountVar;
        LearnersPtr m_parameterLearners;
        std::unordered_set<Parameter> m_learnerParameters;
        std::unordered_set<Variable> m_modelParametersNotCoveredByLearners;
        bool        m_distributed;
        ValuePtr    m_rootGradientValue;

        size_t   m_prevMinibatchNumSamples;
        ValuePtr m_prevMinibatchAggregateTrainingLossValue;
        ValuePtr m_prevMinibatchAggregateEvalCriterionValue;

        AccumulatorPtr m_aggregatedTrainingLossValue;
        AccumulatorPtr m_aggregatedTrainingEvalCriterionValue;

        size_t m_prevDistributedTotalNumSamples;
    };

    ///
    /// Construct a Trainer to train the specified 'model' with the specified 'trainingLoss' Variable as the training criterion
    /// and using the specified set of 'parameterLearners' for updating the model's parameters using computed gradients.
    ///
    CNTK_API TrainerPtr CreateTrainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const std::vector<LearnerPtr>& parameterLearners,
                                      const std::vector<ProgressWriterPtr>& progressWriters = {});

    ///
    /// Construct a Trainer to train the specified 'model' with the specified 'trainingLoss' as the training criterion,
    /// the specified 'evaluationFunction' as the criterion for evaluating the trained model's quality, and using the specified set
    /// of 'parameterLearners' for updating the model's parameters using computed gradients.
    ///
    CNTK_API TrainerPtr CreateTrainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const FunctionPtr& evaluationFunction, const std::vector<LearnerPtr>& parameterLearners,
                                      const std::vector<ProgressWriterPtr>& progressWriters = {});
}

namespace std {
    template <> struct hash<::CNTK::StreamInformation>
    {
        size_t operator()(const ::CNTK::StreamInformation& x) const
        {
            return std::hash<size_t>()(x.m_id);
        }
    };
}

namespace CNTK
{
    ///
    /// A struct that combines the minibatch meta-data with the actual minibatch data.
    /// The former includes the number of sequences and samples in the minibatch,
    /// as well as the sweep-end flag, which is set to true to indicate that the minibatch
    /// concludes a data sweep (i.e, it's the last minibatch at the end of the sweep).
    ///
    struct MinibatchData
    {
        MinibatchData() : MinibatchData(nullptr)
        {}

        // a convenience constructor to allow passing ValuePtr arguments in place
        // of MinibatchData parameter (e.g., in Trainer::TrainMinibatch)
        MinibatchData(ValuePtr value) : MinibatchData(value, 0)
        {}

        MinibatchData(ValuePtr value, size_t numSamples, bool sweepEnd = false)
            : MinibatchData(value, numSamples, numSamples, sweepEnd)
        {}

        MinibatchData(ValuePtr value, size_t numSequences, size_t numSamples, bool sweepEnd)
            : data(value), numberOfSequences(numSequences), numberOfSamples(numSamples), sweepEnd(sweepEnd)
        {}

        std::wstring AsString() const
        {
            std::wstringstream wss;
            wss << L"MinibatchData(data=" << data->AsString() << L", samples=" << numberOfSamples << L", seqs=" << numberOfSequences << L")";
            return wss.str();
        }

        ValuePtr data;
        size_t numberOfSequences;
        size_t numberOfSamples;
        bool sweepEnd;
    };

    ///
    /// Abstraction for generating minibatches of samples for training/evaluation.
    ///
    class MinibatchSource : public std::enable_shared_from_this<MinibatchSource>
    {
    public:
        CNTK_API static const size_t InfinitelyRepeat;
        CNTK_API static const size_t FullDataSweep;
        CNTK_API static const size_t DefaultRandomizationWindowInChunks;

    public:
        ///
        /// Describes the streams 'this' MinibatchSource produces.
        ///
        virtual const std::unordered_set<StreamInformation>& StreamInfos() = 0;

        ///
        /// Reads a minibatch that contains data for all input streams.
        /// The minibatch size is specified in terms of #samples and/or #sequences for the primary input stream; value of 0 for #samples/#sequences means unspecified.
        /// In case the size is specified in terms of both #sequences and #samples, the smaller of the 2 is taken.
        /// An empty map is returned when the MinibatchSource has no more data to return.
        ///
        CNTK_API const std::unordered_map<StreamInformation, MinibatchData>& GetNextMinibatch(
            size_t minibatchSizeInSequences,
            size_t minibatchSizeInSamples,
            const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Same as above but allows to specify partition of data in a distributed environment.
        /// Depending on the number of workers the data is split in different partitions,
        /// and depending on the worker rank, only a particular partition is read.
        ///
        CNTK_API virtual const std::unordered_map<StreamInformation, MinibatchData>& GetNextMinibatch(
            size_t minibatchSizeInSequences,
            size_t minibatchSizeInSamples,
            size_t numberOfWorkers,
            size_t workerRank,
            const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice()) = 0;

        ///
        /// Gets the current reader position in samples.
        /// This is useful for logging the number of samples consumed by a training so far.
        ///
        CNTK_API virtual size_t GetCurrentSamplePosition() const = 0;

        ///
        /// Gets the total number of samples in corpus.
        /// This is useful for logging the number of samples consumed by a training so far.
        ///
        CNTK_API virtual size_t GetFullDataSweepSize() const = 0;

        ///
        /// Destruct this MinibatchSource.
        ///
        virtual ~MinibatchSource() {}

        ///
        /// Optionally overridable method to checkpoint the MinibatchSource's state.
        ///
        virtual Dictionary GetCheckpointState() const
        {
            return Dictionary();
        }

        ///
        /// Optionally overridable method to restore the MinibatchSource's state from a previous checkpoint.
        ///
        virtual void RestoreFromCheckpoint(const Dictionary& /*checkpoint*/) {}

        virtual bool IsInfinite()
        {
            return false;
        }

    public:
        ///
        /// Gets the description of the stream with given name.
        /// Throws an exception of there are none or multiple streams with this same name.
        ///
        CNTK_API const StreamInformation& StreamInfo(const std::wstring& streamName);

        ///
        /// Gets the description of the stream that matches the attributes (Shape, DataType and StorageFormat) of the specified Variable object
        /// Throws an exception if there are none or multiple streams matching the Variable's attributes
        ///
        CNTK_API const StreamInformation& StreamInfo(const Variable& variableToMatch);

        ///
        /// Reads a minibatch that contains data for all input streams.
        /// The minibatch size is specified terms of #samples for the primary input stream.
        /// An empty map is returned when the MinibatchSource has no more data to return.
        ///
        CNTK_API const std::unordered_map<StreamInformation, MinibatchData>& GetNextMinibatch(size_t minibatchSizeInSamples, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());

        // Disallow copy and move construction and assignment
        MinibatchSource(const MinibatchSource&) = delete; MinibatchSource(MinibatchSource&&) = delete; MinibatchSource& operator=(const MinibatchSource&) = delete; MinibatchSource& operator=(MinibatchSource&&) = delete;

    protected:
        MinibatchSource() {}
    };

    typedef Dictionary Deserializer;

    ///
    /// A configuration required to instantiate the CNTK built-in composite minibatch source.
    ///
    struct MinibatchSourceConfig
    {
        ///
        /// Creates a new minibatch source configuration, with enabled randomization and
        /// the randomization window set to DefaultRandomizationWindowInChunks when 'randomize' is
        /// 'true' (default).
        ///
        CNTK_API MinibatchSourceConfig(const std::vector<Deserializer>& deserializers, bool randomize = true);

        ///
        /// The maximum number of input samples (not 'label samples') the reader can produce
        /// (the default value is InfinitelyRepeat). After this number has been reached, the reader
        /// returns empty minibatches on subsequent calls to GetNextMinibatch(). 'maxSweeps' and 'maxSamples'
        /// are mutually exclusive, an exception will be raised if both have non-default values.
        ///
        size_t maxSamples{ MinibatchSource::InfinitelyRepeat };

        ///
        /// The maximum allowed number of sweeps over the input dataset. After this number has been reached,
        /// the reader returns empty minibatches on subsequent calls to GetNextMinibatch().
        /// 'maxSweeps' and 'maxSamples' are mutually exclusive, an exception will be raised if both have
        /// non-default values.
        ///
        size_t maxSweeps{ MinibatchSource::InfinitelyRepeat };

        ///
        /// Size of the randomization window in chunks, non-zero value enables randomization.
        /// 'randomizationWindowInChunks' and 'randomizationWindowInSamples' are mutually exclusive,
        /// an exception will be raised if both have non-zero values.
        ///
        size_t randomizationWindowInChunks{ MinibatchSource::DefaultRandomizationWindowInChunks };

        ///
        /// Size of the randomization window in samples, non-zero value enables randomization.
        /// 'randomizationWindowInChunks' and 'randomizationWindowInSamples' are mutually exclusive,
        /// an exception will be raised if both have non-zero values.
        ///
        size_t randomizationWindowInSamples{ 0 };

        ///
        /// Initial randomization seed value (incremented every sweep when the input data is re-randomized).
        ///
        size_t randomizationSeed{ 0 };

        ///
        /// Output verbosity level.
        ///
        TraceLevel traceLevel{ GetTraceLevel() };

        ///
        /// Truncation length in samples, non-zero value enables the truncation (only applicable for BPTT,
        /// cannot be used in frame mode, an exception will be raised if frame mode is enabled and the
        /// truncation length is non-zero).
        ///
        size_t truncationLength{ 0 };

        ///
        /// Switches the frame mode on and off. If the frame mode is enabled the input data will be processed
        /// as individual frames ignoring all sequence information (this option cannot be used for BPTT,
        /// an exception will be raised if frame mode is enabled and the truncation length is non-zero).
        ///
        bool isFrameModeEnabled{ false };

        ///
        /// Specifies if the deserialization should be done on a single or multiple threads.
        /// Defaults to 'auto' (multithreading is disabled unless ImageDeserializer is present
        /// in the deserializers list). 'false' and 'true' faithfully turn the multithreading off/on.
        ///
        Internal::Optional<bool> isMultithreaded;

        ///
        /// Specifies if the minibatch should be prefetched on a parallel thread.
        ///
        bool enableMinibatchPrefetch{ true };

        ///
        /// Deserializers to be used in the composite reader.
        ///
        std::vector<Deserializer> deserializers;
    };

    ///
    /// Instantiate the CNTK built-in composite minibatch source.
    ///
    CNTK_API MinibatchSourcePtr CreateCompositeMinibatchSource(const MinibatchSourceConfig& configuration);

    struct StreamConfiguration
    {
        StreamConfiguration(const std::wstring& streamName, size_t dim, bool isSparse = false, const std::wstring& streamAlias = L"", bool definesMbSize = false)
            : m_streamName(streamName), m_dim(dim), m_isSparse(isSparse), m_streamAlias(streamAlias), m_definesMbSize(definesMbSize)
        {}

        std::wstring m_streamName;
        size_t m_dim;
        bool m_isSparse;
        std::wstring m_streamAlias;
        bool m_definesMbSize;
    };

    struct HTKFeatureConfiguration
    {
        HTKFeatureConfiguration(const std::wstring& streamName, const std::wstring& scp, size_t dim, size_t left, size_t right, bool broadcast, bool definesMbSize = false)
            : m_streamName(streamName), m_dim(dim), m_scp(scp), m_left(left), m_right(right), m_broadcast(broadcast), m_definesMbSize(definesMbSize)
        {}

        std::wstring m_streamName;
        std::wstring m_scp;
        size_t m_dim;
        size_t m_left;
        size_t m_right;
        bool m_broadcast;
        bool m_definesMbSize;
    };

    typedef Dictionary ImageTransform;

    ///
    /// Create a crop transform with the specified options to be used with a reader
    ///
    CNTK_API ImageTransform ReaderCrop(const wchar_t* cropType = L"center",
        std::pair<int, int> cropSize = std::make_pair(0, 0),
        std::pair<float, float> sideRatio = std::make_pair(0.0f, 0.0f),
        std::pair<float, float> areaRatio = std::make_pair(0.0f, 0.0f),
        std::pair<float, float> aspectRatio = std::make_pair(1.0f, 1.0f),
        const wchar_t* jitterType = L"none");

    ///
    /// Create a scale transform with the specified options to be used with a reader
    ///
    CNTK_API ImageTransform ReaderScale(int width,
        int height, int channels, const wchar_t* interpolations = L"linear",
        const wchar_t* scaleMode = L"fill", int padValue = -1);

    ///
    /// Create a mean subtraction transform with the specified options to be used with a reader
    ///
    CNTK_API ImageTransform ReaderMean(const wchar_t* meanFile);

    ///
    /// Create a color transform with the specified options to be used with a reader
    ///
    CNTK_API ImageTransform ReaderColor(float brightnessRadius = 0.0f,
        float contrastRadius = 0.0f, float saturationRadius = 0.0f);

    ///
    /// Create an ImageDeserializer with the specified options
    ///
    CNTK_API  Deserializer ImageDeserializer(const std::wstring& fileName, const std::wstring& labelStreamName, size_t numLabels, const std::wstring& imageStreamName, const std::vector<ImageTransform>& transforms = {});

    ///
    /// Create a Base64ImageDeserializer with the specified options
    ///
    CNTK_API  Deserializer Base64ImageDeserializer(const std::wstring& fileName, const std::wstring& labelStreamName, size_t numLabels, const std::wstring& imageStreamName, const std::vector<ImageTransform>& transforms = {});

    ///
    /// Create a CTFDeserializer with the specified options
    ///
    CNTK_API  Deserializer CTFDeserializer(const std::wstring& fileName, const std::vector<StreamConfiguration>& streams);

    ///
    /// Create a CBFDeserializer with the specified options
    ///
    CNTK_API  Deserializer CBFDeserializer(const std::wstring& fileName, const std::vector<StreamConfiguration>& streams = {});

    ///
    /// Create an HTKFeatureDeserializer with the specified options
    ///
    CNTK_API  Deserializer HTKFeatureDeserializer(const std::vector<HTKFeatureConfiguration>& streams);

    ///
    /// Create an HTKMLFDeserializer with the specified options
    ///
    CNTK_API  Deserializer HTKMLFDeserializer(const std::wstring& streamName, const std::wstring& labelMappingFile, size_t dimension, const std::vector<std::wstring>& mlfFiles, bool phoneBoundaries = false);

    ///
    /// Instantiate the CNTK built-in text format minibatch source
    ///
    inline MinibatchSourcePtr TextFormatMinibatchSource(const std::wstring& dataFilePath, const std::vector<StreamConfiguration>& streamConfigs,
        size_t epochSize = MinibatchSource::InfinitelyRepeat,
        bool randomize = true,
        size_t randomizationWindow = MinibatchSource::DefaultRandomizationWindowInChunks,
        bool sampleBasedRandomizationWindow = false)
    {
        MinibatchSourceConfig config({ CTFDeserializer(dataFilePath, streamConfigs) }, randomize);
        config.maxSamples = epochSize;

        if (randomize)
        {
            if (sampleBasedRandomizationWindow)
                config.randomizationWindowInSamples = randomizationWindow;
            else
                config.randomizationWindowInChunks = randomizationWindow;
        }

        return CreateCompositeMinibatchSource(config);
    }

    ///
    /// Compute the per dimension means and variances for each of the specified streams using data from the specified minibatchSource.
    ///
    CNTK_API void ComputeInputPerDimMeansAndInvStdDevs(const MinibatchSourcePtr& minibatchSource,
        std::unordered_map<StreamInformation, std::pair<NDArrayViewPtr, NDArrayViewPtr>>& computedMeanAndVariances,
        const DeviceDescriptor& device = DeviceDescriptor::CPUDevice());

    ///
    /// Set the process-wide setting for maximum number of CPU threads to be used by any individual compute operation
    /// Note that this is a per compute operation limit and if the user performs multiple compute operations concurrently
    /// by launching multiple threads and performing a compute operation inside, it will result in each of those concurrently
    /// executing operations to use the specified number of CPU threads limit.
    ///
    CNTK_API void SetMaxNumCPUThreads(size_t numCPUThreads);

    ///
    /// Returns the current process-wide setting for maximum number of CPU threads to be used by any individual compute operation
    ///
    CNTK_API size_t GetMaxNumCPUThreads();

    struct DistributedWorkerDescriptor
    {
        size_t m_globalRank;
        std::wstring m_hostId;

        bool IsMain() const
        {
            return m_globalRank == 0;
        }
    };

    inline bool operator==(const DistributedWorkerDescriptor& left, const DistributedWorkerDescriptor& right)
    {
        return ((left.m_globalRank == right.m_globalRank) && (left.m_hostId == right.m_hostId));
    }
}

namespace std
{
    template <> struct hash<::CNTK::DistributedWorkerDescriptor>
    {
        size_t operator()(const ::CNTK::DistributedWorkerDescriptor& x) const
        {
            return std::hash<size_t>()(x.m_globalRank);
        }
    };
}

namespace CNTK
{
    ///
    /// A communicator interface exposing communication primitives that serve as building blocks
    /// for distributed training.
    ///
    class DistributedCommunicator
    {
    public:
        CNTK_API virtual const std::unordered_set<DistributedWorkerDescriptor>& Workers() const = 0;

        CNTK_API virtual const DistributedWorkerDescriptor& CurrentWorker() const = 0;

        // Creates a new distributed communicator comprising of a subset of the workers in this communicator
        CNTK_API virtual DistributedCommunicatorPtr SubGroup(const std::unordered_set<DistributedWorkerDescriptor>& subGroupWorkers) const = 0;

        // A collective communication API to concatenate values across each worker of this communicator. The concatenated values are only sent to the specified workers; for all others the returned Values are null
        CNTK_API virtual void Concatenate(
            const std::vector<ValuePtr>& values,
            std::vector<ValuePtr>& outputValues,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) = 0;

        CNTK_API virtual void Concatenate(
            const std::vector<NDArrayViewPtr>& input,
            std::vector<NDArrayViewPtr>& output,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) = 0;

        // Gathers the inputs from a subset of workers on the main worker.
        CNTK_API virtual void Gather(
            const Dictionary& input,
            std::vector<DictionaryPtr>& output,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) = 0;

        // A collective communication API to aggregate values across each worker of this communicator.
        // The aggregated values are only sent to the specified workers; for all others the returned Values are null
        CNTK_API virtual void AggregateInPlace(
            const std::vector<NDArrayViewPtr>& values,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) = 0;

        CNTK_API virtual void AllReduceSparseBlockColumn(
            std::vector<NDArrayViewPtr>&) = 0;
        //{
        //    LogicError("This function should not be reached.");
        //}

        CNTK_API virtual void Aggregate(
            const std::vector<NDArrayViewPtr>& values,
            std::vector<NDArrayViewPtr>& outputValues,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) = 0;

        virtual ~DistributedCommunicator() {}

        // TODO: Currently this is a workaround to free static MPIWrapper, it will go away soon.
        // Should be called before executable exits.
        CNTK_API static void Finalize();

        // a barrier to sync all ranks that calls WaitAll() underneath
        CNTK_API virtual void Barrier() = 0;

    protected:
        DistributedCommunicator() {};
    };

    ///
    /// A distributed communicator that supports quantized aggreagtion of values.
    ///
    class QuantizedDistributedCommunicator : public DistributedCommunicator
    {
    public:
        // A collective communication API to perform quantized aggregation of values across all workers of this communicator
        CNTK_API virtual void QuantizedAggregate(
            const std::vector<NDArrayViewPtr>& inValues,
            const std::vector<NDArrayViewPtr>& valueQuantizationResidues,
            const std::vector<NDArrayViewPtr>& stripeQuantizationResidues,
            std::vector<NDArrayViewPtr>& aggregatedOutputs,
            std::vector<NDArrayViewPtr>& newQuantizationResidues,
            std::vector<NDArrayViewPtr>& newStripeQuantizationResidues,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) = 0;

        CNTK_API virtual void QuantizedAggregateInPlace(
            std::vector<NDArrayViewPtr>& inValues,
            std::vector<NDArrayViewPtr>& valueQuantizationResidues,
            std::vector<NDArrayViewPtr>& stripeQuantizationResidues,
            const std::unordered_set<DistributedWorkerDescriptor>& sendToWorkers) = 0;

        CNTK_API virtual void AllReduceSparseBlockColumn(
            std::vector<NDArrayViewPtr>&) = 0;
    protected:
        QuantizedDistributedCommunicator() {};
    };

    ///
    /// Built-in MPI-based communicator.
    ///
    CNTK_API DistributedCommunicatorPtr MPICommunicator(size_t packThresholdSizeInBytes = Internal::DefaultPackThresholdSizeInBytes());

    ///
    /// Distributed communicator that allows quantized aggregations.
    ///
    CNTK_API QuantizedDistributedCommunicatorPtr QuantizedMPICommunicator(bool zeroThresholdFor1Bit, bool useQuantizationForSelfStripe, size_t numQuantizationBits);

    ///
    /// Cross validation configuration
    ///
    struct CrossValidationConfig
    {
    public:
        /// Cross validation configuration.
        /// crossValidationSource: a minibatch source that will be used for cross validation.
        /// crossValidationSchedule : a minibatch size schedule for cross validation.
        /// crossValidationFrequencyInSamples: frequency in samples when to perform cross validation.
        ///
        CNTK_API CrossValidationConfig(const MinibatchSourcePtr& crossValidationSource,
            const MinibatchSizeSchedule& crossValidationSchedule = MinibatchSizeSchedule(64),
            size_t crossValidationFrequencyInSamples = std::numeric_limits<size_t>::max(),
            size_t maxSamples = std::numeric_limits<size_t>::max(),
            const std::unordered_map<Variable, StreamInformation>& inputVarToStream = {});

    private:
        friend class TrainingSession;
        const MinibatchSourcePtr m_source;
        const MinibatchSizeSchedule m_mbSize;
        const size_t m_frequency;
        const size_t m_maxSamples;
        const std::unordered_map<Variable, StreamInformation> m_varToStream;
    };

    ///
    /// Checkpoint configuration
    ///
    struct CheckpointConfig
    {
    public:
        ///
        /// Checkpoint configuration.
        /// checkPointFileName: a file name where the checkpoint will be stored.
        /// checkpointFrequencyInSamples: frequency in samples when to perform checkpointing.
        /// restoreFromCheckpointIfExists: if flag is set, the training session will try to restore before training.
        /// preserveAllCheckpoints: if flag is set, all checkpoints will be preserved.
        ///
        CNTK_API CheckpointConfig(
            const std::wstring& checkPointFileName,
            size_t checkpointFrequencyInSamples = std::numeric_limits<size_t>::max(),
            bool restoreFromCheckpointIfExists = true,
            bool preserveAllCheckpoints = false);

    private:
        friend class TrainingSession;
        const std::wstring m_fileName;
        const bool m_restore;
        const bool m_preserveAll;
        const size_t m_frequency;
    };

    ///
    /// Test configuration
    ///
    struct TestConfig
    {
    public:
        /// Test configuration.
        /// source : a minibatch source that will be used for test
        /// schedule : a minibatch size schedule
        ///
        CNTK_API TestConfig(const MinibatchSourcePtr& source,
            const MinibatchSizeSchedule& schedule = MinibatchSizeSchedule(64),
            const std::unordered_map<Variable, StreamInformation>& inputVarToStream = {});

    private:
        friend class TrainingSession;
        const MinibatchSourcePtr m_source;
        const MinibatchSizeSchedule m_mbSize;
        const std::unordered_map<Variable, StreamInformation> m_varToStream;
    };

    ///
    /// Base abstract class that represents a training session.
    /// Derived classes can redefine different aspects of training, overriding base virtual methods (GetMinibatchSize, OnMinibatchStart, etc.)
    ///
    class TrainingSession
    {
        struct PeriodicAction
        {
            size_t frequency;
            size_t currentIndex;
            size_t sampleCountWhenLastCalled;
            std::function<bool(size_t currentIndex, const DeviceDescriptor&)> action;
        };

    public:
        ///
        /// Constructor of the training session:
        /// trainer : an instance of a trainer
        /// trainingSource: minibatch source
        /// minibatchSizeSchedule: mb size schedule
        /// inputVarToStream: var to stream mapping
        /// maxNumTrainingSamples: max number of training samples
        /// progress : a training configuration
        ///
        CNTK_API TrainingSession(
            const TrainerPtr& trainer,
            const MinibatchSourcePtr& trainingSource,
            const MinibatchSizeSchedule& minibatchSizeSchedule,
            const std::unordered_map<Variable, StreamInformation>& inputVarToStream,
            size_t maxNumTrainingSamples,
            size_t progressFrequency,
            const CheckpointConfig& checkpointing,
            const CrossValidationConfig& crossValidation,
            const TestConfig& test);

        ///
        /// Runs the session.
        ///
        CNTK_API void Train(const DeviceDescriptor& computeDevice);

        ///
        /// Restores a session from a checkpoint.
        ///
        CNTK_API void RestoreFromCheckpoint(const std::wstring& checkpointFileName);

        CNTK_API virtual ~TrainingSession() {}

    public:
        ///
        /// Optionally overridable, called each time before a new minibatch is requested from the minibatch source
        /// during training (from Run method).
        ///
        virtual size_t GetMinibatchSize()
        {
            return m_mbSize[Trainer()->TotalNumberOfSamplesSeen()];
        }

        ///
        /// Optionally overridable callback that is invoked before each minibatch.
        ///
        CNTK_API virtual void OnMinibatchStart() {};

        ///
        /// Optionally overridable callback that is invoked after each minibatch.
        /// If return value is false, the training will be stopped.
        ///
        CNTK_API virtual bool OnMinibatchEnd() { return true; };

        ///
        /// Optionally overridable callback that is invoked before each checkpoint.
        ///
        CNTK_API virtual void OnCheckpointStart(size_t /*checkpointIndex*/) {};

        ///
        /// Optionally overridable callback that is invoked after each checkpoint.
        ///
        CNTK_API virtual void OnCheckpointEnd(size_t /*checkpointIndex*/) {};

        ///
        /// Optionally overridable callback that is invoked before each cross validation.
        ///
        CNTK_API virtual void OnCrossValidationStart(size_t /*validationIndex*/) {};

        ///
        /// Optionally overridable callback that is invoked after each cross validation.
        /// If return value is false, the training will be stopped.
        ///
        CNTK_API virtual bool OnCrossValidationEnd(size_t /*validationIndex*/, double /*averageError*/, size_t /*numberOfSamples*/, size_t /*numberOfMinibatches*/)
        {
            return true;
        }

    protected:
        ///
        /// Accessors.
        ///
        TrainerPtr Trainer() const { return m_trainer; }

    private:
        /// Disallow copy and move construction and assignment
        TrainingSession(const TrainingSession&) = delete; TrainingSession& operator=(const TrainingSession&) = delete; TrainingSession& operator=(TrainingSession&&) = delete; TrainingSession(TrainingSession&&) = delete;

        // Auxilary functions.
        void GetNextMinibatch(const MinibatchSourcePtr& source,
            std::unordered_map<Variable, ValuePtr>& minibatch,
            const std::unordered_map<Variable, StreamInformation>& inputVarToStream,
            size_t maxMbSize, size_t workerRank, size_t numberOfWorkers, const DeviceDescriptor& computeDevice);
        void GetTrainingMinibatch(std::unordered_map<Variable, ValuePtr>& minibatch, size_t maxMbSize, const DeviceDescriptor& computeDevice);
        void GetCrossValidationMinibatch(std::unordered_map<Variable, ValuePtr>& minibatch, size_t maxMbSize, const DeviceDescriptor& computeDevice);

        void RestoreFromCheckpoint();
        void SaveCheckpoint(size_t currentIndex);
        void SaveFinalCheckpoint();

        bool CrossValidate(size_t currentIndex, const DeviceDescriptor& computeDevice);
        void ReportProgress(size_t currentIndex);
        void Test(const DeviceDescriptor& computeDevice);

        size_t m_parallelAfterSamples;
        size_t m_workerRank;
        size_t m_numberOfWorkers;

        // Scaler for the minibatch size in distributed mode.
        size_t m_mbSizeScaleFactor;

        std::vector<PeriodicAction> m_actions;

        // Training.
        TrainerPtr m_trainer;
        const MinibatchSourcePtr m_source;
        const MinibatchSizeSchedule m_mbSize;
        const std::unordered_map<Variable, StreamInformation> m_varToStream;
        const size_t m_maxNumSamples;
        const size_t m_progressFrequency;

        // Additional configuration.
        CheckpointConfig m_checkpoint;
        CrossValidationConfig m_cv;
        TestConfig m_test;
    };

    ///
    /// Base class for all classes that want to record training/evaluation progress.
    ///
    class ProgressWriter
    {
    public:
        ///
        /// Constructor.
        ///
        /// The frequency arguments control a schedule on which the training/evaluation progress updates are written.
        /// The frequency value of 0 specifies geometric schedule, i.e. write progress after 1, 2, 4, 8, 16... updates.
        /// The frequency value other than zero specifies arithmetic schedule, i.e. write progress after each
        /// 'frequency' updates.
        ///
        /// The firstUpdatesToWrite arguments only apply on arithemetic schedule. If specified, the first
        /// 'firstUpdatesToWrite' updates will be written explicitly before using an arithmetic schedule.
        ///
        // TODO: Encapsulate (freq, firstToWrite) as an update schedule type.
        CNTK_API ProgressWriter(size_t trainingUpdateWriteFrequency, size_t trainingFirstUpdatesToWrite,
                                size_t testUpdateWriteFrequency, size_t testFirstUpdatesToWrite,
                                size_t distributedSyncUpdateWriteFrequency, size_t distributedSyncFirstUpdatesToWrite);

        ///
        /// Destructor.
        ///
        CNTK_API virtual ~ProgressWriter();

        ///
        /// Actually outputs information about the update in training progress. Overridable in derived classes.
        ///
        CNTK_API virtual void OnWriteTrainingUpdate(const std::pair<size_t, size_t>& /*samples*/,
                                                    const std::pair<size_t, size_t>& /*updates*/,
                                                    const std::pair<double, double>& /*aggregateLoss*/,
                                                    const std::pair<double, double>& /*aggregateMetric*/) {};

        ///
        /// Actually outputs information about the update in evaluation progress.  Overridable in derived classes.
        ///
        CNTK_API virtual void OnWriteTestUpdate(const std::pair<size_t, size_t>& /*samples*/,
                                                const std::pair<size_t, size_t>& /*updates*/,
                                                const std::pair<double, double>& /*aggregateMetric*/) {};

        /// Actually outputs information about the synchronization across parallel workers
        /// in distributed training. Overridable in derived classes.
        ///
        CNTK_API virtual void OnWriteDistributedSyncUpdate(const std::pair<size_t, size_t>& /*samples*/,
                                                           const std::pair<size_t, size_t>& /*updates*/,
                                                           const std::pair<double, double>& /*aggregateMetric*/) {};

        ///
        /// Called after each training update, regardless whether the actual write is needed.
        ///
        CNTK_API virtual void OnTrainingUpdateEnd() {};

        ///
        /// Actually outputs information about the summary of training progress.  Overridable in derived classes.
        ///
        CNTK_API virtual void OnWriteTrainingSummary(size_t /*samples*/, size_t /*updates*/, size_t /*summaries*/,
                                                     double /*aggregateLoss*/, double /*aggregateMetric*/,
                                                     size_t /*elapsedMilliseconds*/) {};

        ///
        /// Actually outputs information about the summary of evaluation progress.  Overridable in derived classes.
        ///
        CNTK_API virtual void OnWriteTestSummary(size_t /*samples*/, size_t /*updates*/, size_t /*summaries*/,
                                                 double /*aggregateMetric*/, size_t /*elapsedMilliseconds*/) {};

        ///
        /// Writes out the string key together with the specified value.
        ///
        CNTK_API virtual void Write(const std::wstring& /*key*/, double /*value*/) {};

        ///
        /// Returns the total number of training progress updates received by the progress writer.
        ///
        CNTK_API size_t TotalTrainingUpdates() const;

        ///
        /// Returns the total number of evaluation progress updates received by the progress writer.
        ///
        CNTK_API size_t TotalTestUpdates() const;

        ///
        /// Updates the writer with the accumulated loss/metric since the start of training.
        ///
        void UpdateTraining(size_t numSamples, const ValuePtr& accumulatedLoss, const ValuePtr& accumulatedMetric);

        ///
        /// Updates the writer with the accumulated metric since the start of evaluation.
        ///
        void UpdateTest(size_t numSamples, const ValuePtr& accumulatedMetric);

        ///
        /// Updates the writer with the accumulated metric since the start of evaluation.
        ///
        void UpdateDistributedSync(size_t numSamples, const ValuePtr& accumulatedMetric);

        ///
        /// Writes a summary of training progress since the last call to this function.
        ///
        void WriteTrainingSummary(const ValuePtr& accumulatedLoss, const ValuePtr& accumulatedMetric);

        ///
        /// Writes a summary of evaluation progress since the last call to this function.
        ///
        void WriteTestSummary(const ValuePtr& accumulatedMetric);

    private:
        // Disallow copy and move construction and assignment
        ProgressWriter(const ProgressWriter&) = delete; ProgressWriter(ProgressWriter&&) = delete; ProgressWriter& operator=(const ProgressWriter&) = delete; ProgressWriter& operator=(ProgressWriter&&) = delete;

        class Impl;
        std::unique_ptr<Impl> m_training;
        std::unique_ptr<Impl> m_test;
        std::unique_ptr<Impl> m_distributedSync;
    };

    /// Creates an instance of the training session class. Parameters match the parameters of the TrainingSession constructor.
    ///
    CNTK_API TrainingSessionPtr CreateTrainingSession(
        const TrainerPtr& trainer,
        const MinibatchSourcePtr& trainingSource,
        const MinibatchSizeSchedule& minibatchSizeSchedule,
        const std::unordered_map<Variable, StreamInformation>& inputVarToStream,
        size_t maxNumTrainingSamples,
        size_t progressFrequency,
        const CheckpointConfig& checkpointing = { L"" },
        const CrossValidationConfig& crossValidation = { nullptr },
        const TestConfig& test = { nullptr });

    ///
    /// Creates an instance of crop node, which crops one of its inputs along spatial dimensions only.
    /// The size of the crop rectangle is determined by another input node.
    /// The offset of the crop rectangle is either given explicitly or computed automatically by examining the network.
    ///

    ///
    /// Creates an instance of crop node with explicitly specified crop offsets.
    /// nodeInput: input node to be cropped.
    /// nodeReferent: input node which determines the spatial size of output.
    /// offsetX, offsetY: offset values in pixel which determine the position of crop rectangle.
    ///
    CNTK_API FunctionPtr Crop(const Variable& nodeInput, const Variable& nodeReferent, size_t offsetX, size_t offsetY, const std::wstring& name = L"");

    ///
    /// Creates an instance of crop node with automatically computed crop offsets.
    /// nodeInput: input node to be cropped.
    /// nodeReferent: input node which determines the spatial size of output.
    ///
    CNTK_API FunctionPtr Crop(const Variable& nodeInput, const Variable& nodeReferent, const std::wstring& name = L"");

    ///
    /// Creates an instance of crop node with automatically computed crop offsets and specified ancestor nodes.
    /// This is used in cases when input nodes do not have common ancestor in the network.
    /// nodeInput: input node to be cropped.
    /// nodeReferent: input node which determines the spatial size of output.
    /// ancestorInput: ancestor of nodeInput.
    /// ancestorReferent: ancestor of nodeReferent which is treated as equal to ancestorInput for the purpose of computing crop offsets.
    ///
    CNTK_API FunctionPtr Crop(const Variable& nodeInput, const Variable& nodeReferent, const Variable& ancestorInput, const Variable& ancestorReferent, const std::wstring& name = L"");

#endif // !CNTK_HEADERONLY_DEFINITIONS
}

// restore saved macro definition
#pragma pop_macro("max")
