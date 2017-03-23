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

#ifdef SWIG
#define final
#define explicit
#define static_assert(condition, message)
#endif

#include "CNTKLibraryInternals.h"

namespace CNTK
{
    ///
    /// Enumeration type denoting data type of symbolic data entities or actual data.
    ///
    enum class DataType : unsigned int
    {
        Unknown = 0,
        Float = 1,
        Double = 2,

        /* TODO:
        Bit,
        Char,
        UChar,
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
        /// CNTK uses a cooperative synchronization for the device access, whereby only a single process can acquire 
        /// a device lock. However, if exclusivity is not required, the same device can still be accessed without acquiring 
        /// any locks (in which case, any existing lock corresponding to the device will be ignored).
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

        std::wstring AsString() const
        {
            std::wstring str = DeviceKindName(Type());
            if (Type() == DeviceKind::GPU)
                str = str + L"[" + std::to_wstring(Id()) + L"]";

            return str;
        }

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
    /// Denotes a multi-dimensional rectangular shape.
    ///
    class NDShape final
    {
        friend bool operator==(const NDShape& first, const NDShape& second);
        friend class PrimitiveFunction;

        static const size_t SentinelDimValueForUnknownShape = (size_t)-2;
    public:

        ///
        /// A placeholder value to use for an axis whose dimension is unknown and is to be inferred by the system.
        ///
        static const size_t InferredDimension = (size_t)-1;

        ///
        /// A placeholder shape to use to denote an unknown shape
        ///
        CNTK_API static const NDShape Unknown;

    public:
        ///
        /// Construct a NDShape with 0 axes, which denotes a scalar.
        ///
        NDShape() {}

        ///
        /// Construct a NDShape instance with the specified rank and dimensionality in each axis.
        ///
        explicit NDShape(size_t numAxes, size_t dimension = InferredDimension)
            : m_shapeDims(numAxes, dimension)
        {}

        ///
        /// Construct a NDShape instance with specified dimensions.
        ///
        NDShape(const std::vector<size_t>& dimensions)
            : m_shapeDims(dimensions)
        {}

        ///
        /// Construct a NDShape instance with specified dimensions.
        ///
        NDShape(const std::initializer_list<size_t>& dimensions)
            : m_shapeDims(dimensions)
        {}

        ///
        /// Returns the dimensions of 'this' shape as a std::vector<size_t>
        ///
        const std::vector<size_t>& Dimensions() const { return m_shapeDims; }

        ///
        /// Returns a boolean indicating if 'this' shape is the special Unknown shape
        ///
        bool IsUnknown() const { return (*this == NDShape::Unknown); }

        ///
        /// Returns the rank of 'this' shape.
        ///
        size_t Rank() const { return m_shapeDims.size(); }

        ///
        /// Returns a reference to dimension size for the specified axis.
        ///
        size_t& operator[](size_t axisId) { return m_shapeDims[axisId]; }

        ///
        /// Returns the dimension size for the specified axis.
        ///
        size_t operator[](size_t axisId) const { return m_shapeDims[axisId]; }

        ///
        /// Creates and returns a new NDShape instance with the same dimensions as 'this' shape's specified axis range [beginAxisId, endAxisId).
        ///
        NDShape SubShape(size_t beginAxisId = 0, size_t endAxisId = SIZE_MAX) const
        {
            endAxisId = (endAxisId == SIZE_MAX) ? Rank() : endAxisId;
            if ((endAxisId < beginAxisId) || (endAxisId > Rank()))
                InvalidArgument("NDShape::SubShape: The specified endAxisId (%zu) must not exceed the rank (%zu) of 'this' NDShape and must be >= than the specified beginAxisId (%zu)", endAxisId, Rank(), beginAxisId);

            std::vector<size_t> subShapeDims(m_shapeDims.begin() + beginAxisId, m_shapeDims.begin() + endAxisId);
            return subShapeDims;
        }

        ///
        /// Returns a boolean value indicating if the dimension size for any of the axes of 'this' shape is unknown/inferred (aka == NDShape::InferredDimension).
        ///
        bool HasInferredDimension() const
        {
            return (std::find(m_shapeDims.begin(), m_shapeDims.end(), (size_t)InferredDimension) != m_shapeDims.end());
        }

        ///
        /// Returns the total size of the rectangular shape that 'this' shape denotes.
        ///
        size_t TotalSize() const
        {
            if (HasInferredDimension())
                RuntimeError("NDShape::TotalSize: TotalSize cannot be determined for a NDShape '%S' with one or more dimensions being InferredDimension.", AsString().c_str());

            size_t totalSize = 1;
            for (auto dim : m_shapeDims)
                totalSize *= dim;

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

            return newShapeDims;
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

                    if (displayShape[i] != InferredDimension)
                        wStrStream << displayShape[i];
                    else
                        wStrStream << "?";
                }
                wStrStream << L"]";
                return wStrStream.str();
            }
        }

    private:
        std::vector<size_t> m_shapeDims;
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
        friend class Variable;
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

        /// Construct a read-only NDArrayView with the specified 'dataBuffer' as the backing storage.
        /// The 'dataBuffer' must have been allocated on the specified 'device', must be at least
        /// as large as the total size of the specified 'viewShape' and must outlive the created NDArrayView object.
        ///
        NDArrayView(::CNTK::DataType dataType, const NDShape& viewShape, const void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device)
            : NDArrayView(dataType, viewShape, const_cast<void*>(dataBuffer), bufferSizeInBytes, device, /*readOnly =*/ true)
        {}

        ///
        /// Construct a NDArrayView with newly allocated sparse storage in SparseCSC format on the specified 'device' and initialize its contents
        // with the specified Sparse CSC format data.
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
                InvalidArgument("The size (%zu) of the STL container does not match the size (%zu) of the specified viewShape '%S'.",
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
        /// Destruct 'this' NDArrayView object
        ///
        CNTK_API ~NDArrayView();

        ///
        /// Returns a writable pointer to the data buffer underlying 'this' view
        /// Throws an exception if 'this' view is read-only
        /// 
        template <typename ElementType>
        CNTK_API ElementType* WritableDataBuffer();

        ///
        /// Returns a read-only pointer to the data buffer underlying 'this' view
        /// 
        template <typename ElementType>
        CNTK_API const ElementType* DataBuffer() const;

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
        /// Fill 'this' NDArrayView with the specified value. The underlying DataType of 'this' view should be DataType::Float.
        ///
        CNTK_API void SetValue(float value);

        ///
        /// Fill 'this' NDArrayView with the specified value. The underlying DataType of 'this' view should be DataType::Double.
        ///
        CNTK_API void SetValue(double value);

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
        /// Creates a new NDArrayView which is an alias of a slice of 'this' view; i.e. a new view over the underlying data
        /// corresponding to the specified slice of 'this' view.
        ///
        CNTK_API NDArrayViewPtr SliceView(const std::vector<size_t>& startOffset, const std::vector<size_t>& extent, bool readOnly = false) const;

        ///
        /// Creates a new NDArrayView which is an alias of 'this' view but with a new shape.
        ///
        CNTK_API NDArrayViewPtr AsShape(const NDShape& newShape) const;

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
        /// If the value stored is a scalar, returns it. Otherwise, throws an error.
        ///
        template<typename ElementType>
        ElementType AsScalar() const;

    private:
        // Disallow copy and move construction and assignment
        NDArrayView(const NDArrayView&) = delete; NDArrayView& operator=(const NDArrayView&) = delete; NDArrayView& operator=(NDArrayView&&) = delete; NDArrayView(NDArrayView&& other) = delete;

    private:
        static const size_t AutoSelectRowColSplitPoint = SIZE_MAX;

    private:
        CNTK_API NDArrayView(::CNTK::DataType dataType, const DeviceDescriptor& device, ::CNTK::StorageFormat storageType, const NDShape& viewShape, bool readOnly, void* tensorView);

        template <typename ElementType>
        static std::shared_ptr<Microsoft::MSR::CNTK::Matrix<ElementType>> GetMatrixImpl(const Microsoft::MSR::CNTK::TensorView<ElementType>* tensorView, size_t rowColSplitPoint);

        template <typename ElementType>
        std::shared_ptr<const Microsoft::MSR::CNTK::Matrix<ElementType>> GetMatrix(size_t rowColSplitPoint = AutoSelectRowColSplitPoint) const;

        template <typename ElementType>
        std::shared_ptr<Microsoft::MSR::CNTK::Matrix<ElementType>> GetWritableMatrix(size_t rowColSplitPoint = AutoSelectRowColSplitPoint);

        template <typename ElementType>
        const Microsoft::MSR::CNTK::TensorView<ElementType>* GetTensorView() const;

        template <typename ElementType>
        Microsoft::MSR::CNTK::TensorView<ElementType>* GetWritableTensorView();

    private:
        ::CNTK::DataType m_dataType;
        DeviceDescriptor m_device;
        ::CNTK::StorageFormat m_storageFormat;
        NDShape m_viewShape;
        bool m_isReadOnly;

        std::shared_ptr<void> m_tensorView; // Microsoft::MSR::CNTK::TensorView<ElemType>*
    };

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
        /// Creates a new NDArrayView with newly allocated storage on the specified device and copies 'this' view's contents into the newly allocated view.
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
        NDMask(const NDShape& shape, Microsoft::MSR::CNTK::Matrix<char>* matrix);

        CNTK_API void MarkSectionAs(const std::vector<size_t>& sectionOffset, const NDShape& sectionShape, MaskKind maskKind);

        Microsoft::MSR::CNTK::Matrix<char>* GetMatrix() const;

        // Disallow copy and move construction and assignment
        NDMask(const NDMask&) = delete; NDMask& operator=(const NDMask&) = delete; NDMask& operator=(NDMask&&) = delete; NDMask(NDMask&& other) = delete;

    private:
        DeviceDescriptor m_device;
        NDShape m_maskShape;

        std::shared_ptr<Microsoft::MSR::CNTK::Matrix<char>> m_matrixView;
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

    public:
        ///
        /// Construct an Axis object denoting a static axis with the specified index.
        ///
        explicit Axis(int staticAxisIdx)
            : m_staticAxisIdx(staticAxisIdx), m_isOrderedDynamicAxis(false)
        {
            m_name = StaticAxisNamePrefix + std::to_wstring(staticAxisIdx);
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
        const std::wstring& Name() const { return m_name; }

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
        std::wstring m_name;
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
        } m_data;

         static const size_t s_version = 1;
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

        CNTK_API Dictionary(const Dictionary&);
        CNTK_API Dictionary& operator=(const Dictionary&);

        CNTK_API Dictionary(Dictionary&& other);
        CNTK_API Dictionary& operator=(Dictionary&& other);

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

        DictionaryIterator begin() const { return m_dictionaryData->begin(); }
        ConstDictionaryIterator cbegin() const { return m_dictionaryData->cbegin(); }
        DictionaryIterator end() const { return m_dictionaryData->end(); }
        ConstDictionaryIterator cend() const { return m_dictionaryData->cend(); }

        size_t Size() const { return m_dictionaryData->size();  }

        std::unordered_set<std::wstring> Keys() 
        { 
            std::unordered_set<std::wstring> keys;
            for (const auto& kv : *m_dictionaryData)
                keys.insert(kv.first);
            return keys;
        }

        friend CNTK_API std::istream& operator>>(std::istream& stream, Dictionary& us);
        friend CNTK_API std::ostream& operator<<(std::ostream& stream, const Dictionary& us);

        CNTK_API void Save(const std::wstring& filename);
        CNTK_API static Dictionary Load(const std::wstring& filename);

    private:
        std::shared_ptr<std::unordered_map<std::wstring, DictionaryValue>> m_dictionaryData;
        static const size_t s_version = 1;
    };

    ///
    /// Enumeration type denoting the kind of a symbolic Variable object
    ///
    enum class VariableKind : unsigned int
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
            return prefix + std::to_wstring(Internal::NewUniqueId());
        }

        inline std::wstring GenerateUid(VariableKind varKind)
        {
            return GenerateUid(std::wstring(VariableKindName(varKind)));
        }

        inline std::wstring GenerateUid(const std::wstring& prefix)
        {
            return GenerateUid(std::wstring(prefix));
        }
    }

    typedef Dictionary ParameterInitializer;

    // Forward declarations
    inline Variable PlaceholderVariable(const NDShape& shape, ::CNTK::DataType dataType, const std::wstring& name, const std::vector<Axis>& dynamicAxes = Axis::UnknownDynamicAxes());
    inline Variable InputVariable(const NDShape& shape, bool isSparse, ::CNTK::DataType dataType, bool needsGradient, const std::wstring& name, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes());
    inline Variable OutputVariable(const NDShape& shape, ::CNTK::DataType dataType, const std::vector<Axis>& dynamicAxes, bool needsGradient, const std::wstring& name = L"");

    ///
    /// Denotes a symbolic entity corresponding to the inputs and outputs of a Function.
    /// A Variable is symbolic and does not represent the actual values.
    /// Also, Variable type is a value type and copies of a Variable object are aliases of the
    /// source Variable object itself and have the same identity.
    ///
    class Variable : private IDictionarySerializable
    {
        friend bool operator==(const Variable& first, const Variable& second);
        friend class Function;
        friend class CompositeFunction;
        friend class BlockFunction;
        friend class Trainer;
        friend class PrimitiveFunction;

        template <typename T>
        friend struct std::hash;

        friend class Internal::VariableResolver;

#ifndef SWIG
    private:
        friend inline Variable PlaceholderVariable(const NDShape& shape, ::CNTK::DataType dataType, const std::wstring& name, const std::vector<Axis>& dynamicAxes);
        friend inline Variable InputVariable(const NDShape& shape, bool isSparse, ::CNTK::DataType dataType, bool needsGradient, const std::wstring& name, const std::vector<Axis>& dynamicAxes /*= Axis::DefaultInputVariableDynamicAxes()*/);
        friend inline Variable OutputVariable(const NDShape& shape, ::CNTK::DataType dataType, const std::vector<Axis>& dynamicAxes, bool needsGradient, const std::wstring& name /*= L""*/);
#endif

    public:

        ///
        /// Create an 'Output' variable aliasing the output of the specified Function
        /// Throws an exception if called for a Function instance with multiple outputs
        ///
        CNTK_API Variable(const FunctionPtr& function);

        ///
        /// Implicit conversion to a FunctionPtr; creates a pass through primitive Function
        ///
        CNTK_API operator FunctionPtr() const;

        /// 
        /// Default constructor for creating an invalid/null Variable instance. 
        /// Required for use in a std::vector container.
        /// 
        Variable() {}

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
        /// Returns the internally generated unique name of the variable
        ///
        CNTK_API const std::wstring& Uid() const;

        ///
        /// Returns the Function object which 'this' variable is an output of.
        /// Returns null when called for a Variable that is not of 'Output' VariableKind.
        ///
        CNTK_API FunctionPtr Owner() const;

        ///
        /// Returns the DataType of the data that 'this' Variable symbolically represents
        ///
        CNTK_API DataType GetDataType() const;

        ///
        /// Returns a boolean value indicating if gradient computation is enabled for this variable.
        ///
        CNTK_API bool NeedsGradient() const;

        ///
        /// Returns a string representation for this variable.
        ///
        CNTK_API std::wstring AsString() const;

    protected:
#ifdef SWIG
    public:
#endif
        Variable(const NDShape& shape, VariableKind varType, ::CNTK::DataType dataType, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, const std::wstring& name, const std::wstring& uid)
            : Variable(shape, varType, dataType, value, needsGradient, dynamicAxes, /*isSparse =*/ false, name, uid)
        {}

    protected:
        CNTK_API NDArrayViewPtr Value() const;
        CNTK_API void SetValue(const NDArrayViewPtr& value);

    private:
#ifdef SWIG
    public:
#endif
        Variable(const NDShape& shape, bool isSparse, ::CNTK::DataType dataType, bool needsGradient, const std::wstring& name, const std::vector<Axis>& dynamicAxes, const std::wstring& uid)
            : Variable(shape, VariableKind::Input, dataType, nullptr, needsGradient, dynamicAxes, isSparse, name, uid)
        {}

        // TODO: This should be a private but if not made public, the python bindings build complains about an unresolved external
        // Probably due the above ctor being a public method in SWIG codegen
    public:
        CNTK_API Variable(const NDShape& shape, VariableKind varType, ::CNTK::DataType dataType, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, const std::wstring& name, const std::wstring& uid);

private:
        CNTK_API const Variable& BlockFunctionVariableMapping() const;

        CNTK_API Variable Clone() const;

        CNTK_API virtual Dictionary Serialize() const override;

        virtual size_t CurrentVersion() const override { return s_serializationVersion; }

        template <typename ElementType>
        static NDArrayViewPtr CreateValueFromParameterInitializer(const NDShape& shape, const ParameterInitializer& initConfig, const DeviceDescriptor& device);

        CNTK_API static Variable Deserialize(const Dictionary& dictionary, const ::CNTK::DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());

        void SetOwner(const std::weak_ptr<Function>& ownerFunction);

        Variable CompositePreservingCopy(const std::shared_ptr<const Function>& composite) const;

        Variable NonCompositePreservingCopy() const;

    private:
#ifdef SWIGCSHARP
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

    private:
        std::shared_ptr<const Function> m_outputComposite; // Currently needed for outputs.
    };

    // TODO: Variable equality should be based on uids.
    inline bool operator==(const Variable& first, const Variable& second)
    {
        return first.m_dataFields == second.m_dataFields;
    }

    inline bool operator!=(const Variable& first, const Variable& second)
    {
        return !(first == second);
    }

    ///
    /// Create a Placeholder variable to be used as a temporary/placeholder input to a Function.
    /// All placeholder inputs of a Function must be replaced with non-placeholder Variables before Forward evaluation of the Function.
    ///
    inline Variable PlaceholderVariable(const NDShape& shape, ::CNTK::DataType dataType, const std::wstring& name, const std::vector<Axis>& dynamicAxes)
    {
        auto varKind = VariableKind::Placeholder;
        return Variable(shape, varKind, dataType, nullptr, false, dynamicAxes, name, Internal::GenerateUid(varKind));
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
    inline Variable PlaceholderVariable(const std::wstring& name = L"")
    {
        return PlaceholderVariable(NDShape::Unknown, name, Axis::UnknownDynamicAxes());
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
    inline Variable InputVariable(const NDShape& shape, ::CNTK::DataType dataType, bool needsGradient, const std::wstring& name = L"", const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes())
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

    ///
    /// Create an 'Output' variable
    ///
    inline Variable OutputVariable(const NDShape& shape, ::CNTK::DataType dataType, const std::vector<Axis>& dynamicAxes, const std::wstring& name = L"")
    {
        return OutputVariable(shape, dataType, dynamicAxes, /*needsGradient =*/ true, name);
    }

    ///
    /// Create an 'Output' variable
    ///
    inline Variable OutputVariable(const NDShape& shape, ::CNTK::DataType dataType, const std::vector<Axis>& dynamicAxes, bool needsGradient, const std::wstring& name /*= L""*/)
    {
        return Variable(shape, VariableKind::Output, dataType, nullptr, needsGradient, dynamicAxes, /*isSparse =*/ false, name, Internal::GenerateUid(VariableKind::Output));
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

    ///
    /// Denotes Parameter inputs of a Function.
    ///
    class Parameter final : public Variable
    {
        template <typename T>
        friend struct std::hash;

        friend class Internal::VariableResolver;

    public:
        ///
        /// Construct a parameter whose initial contents are a copy of the specified 'value'
        ///
        explicit Parameter(const NDArrayViewPtr& value, const std::wstring& name = L"")
            : Parameter(value, name, Internal::GenerateUid(VariableKind::Parameter))
        {}

        // TODO: Constructor to move a specified NDArrayView value

        ///
        /// Construct a parameter of specified shape whose contents are initialized with the specified 'initValue'
        ///
        template<typename ElemType>
        Parameter(const NDShape& shape, ElemType initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"")
            : Parameter(shape, AsDataType<ElemType>(), ConstantInitializer(initValue), device, name)
        {}

        ///
        /// Construct a constant of specified shape whose contents are initialized with the specified 'initValue'
        ///
        Parameter(const NDShape& shape, DataType dataType, double initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"")
            : Parameter(shape, dataType, ConstantInitializer(initValue), device, name)
        {}

        ///
        /// Construct a constant of specified shape whose contents are initialized using the specified initializer
        ///
        CNTK_API Parameter(const NDShape& shape, DataType dataType, const ParameterInitializer& initializer, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"");

        ///
        /// DownCast a Variable to a Parameter. Only allowed if the VariableKind is Parameter and throws an exception otherwise.
        ///
        explicit Parameter(const Variable& variable)
            : Variable(variable)
        {
            if (!IsParameter())
                InvalidArgument("A non-parameter Variable '%S' cannot be converted to a Parameter.", variable.AsString().c_str());
        }

        ///
        /// Get the value of 'this' parameter
        ///
        NDArrayViewPtr Value() const
        {
            return Variable::Value();
        }

        ///
        /// Copies the contents of the 'value' NDArrayView into the view backing 'this' 
        /// parameter's value. The shapes of both views must be identical.
        ///
        void SetValue(const NDArrayViewPtr& value)
        {
            Variable::SetValue(value);
            RecordValueUpdate();
        }

        CNTK_API size_t CurrentValueTimeStamp() const;

        CNTK_API void RecordValueUpdate();

    private:
        explicit Parameter(const NDArrayViewPtr& value, const std::wstring& name, const std::wstring& uid)
            : Variable(value->Shape(), VariableKind::Parameter, value->GetDataType(), value, true, {}, name, uid)
        {
            if (value->IsReadOnly())
                InvalidArgument("Parameter cannot be constructed from a read-only NDArrayView value; you can create a non read-only clone of the value and use that instead!");
        }
    };

    // Implementation note: The Variable type is a value type and not polymorphic in nature. 
    // However we have a couple of derivatives of the type to extend the base interface and thus we ensure that the derived types do not have additional fields.
    // This check is weak in that the derives types may sneak in some additional fields if the base type had some padding at the end, without changing the object size
    // but it should be good enough for catching any accidental addition of fields.
    static_assert(sizeof(Parameter) == sizeof(Variable), "The Parameter type should not have any data fields beyond what its base type 'Variable' has.");

    ///
    /// Denotes Constant inputs of a Function.
    ///
    class Constant final : public Variable
    {
        template <typename T>
        friend struct std::hash;

        friend class Internal::VariableResolver;

    public:
        ///
        /// Construct a Constant whose initial contents are a copy of the specified value
        ///
        Constant(const NDArrayViewPtr& value, const std::wstring& name = L"")
            : Constant(value, name, Internal::GenerateUid(VariableKind::Constant))
        {}

        // TODO: Constructor to move a specified NDArrayView value

        ///
        /// Construct a constant of specified shape whose contents are initialized with the specified 'initValue'
        ///
        template<typename ElemType>
        Constant(const NDShape& shape, ElemType initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"")
            : Constant(shape, AsDataType<ElemType>(), ConstantInitializer(initValue), device, name)
        {}

        ///
        /// Construct a constant of specified shape whose contents are initialized with the specified 'initValue'
        ///
        Constant(const NDShape& shape, DataType dataType, double initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"")
            : Constant(shape, dataType, ConstantInitializer(initValue), device, name)
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
        explicit Constant(const Variable& variable)
            : Variable(variable)
        {
            if (!IsConstant())
                InvalidArgument("A non-constant Variable '%S' being converted to a Constant.", variable.AsString().c_str());
        }

        ///
        /// Get the value of 'this' Constant
        ///
        NDArrayViewPtr Value() const
        {
            return Variable::Value();
        }

    private:
        Constant(const NDArrayViewPtr& value, const std::wstring& name, const std::wstring& uid)
            : Variable(value->Shape(), VariableKind::Constant, value->GetDataType(), value, false, {}, name, uid)
        {}

        ///
        /// Construct a constant of specified shape whose contents are initialized using the specified initializer
        ///
        CNTK_API Constant(const NDShape& shape, DataType dataType, const ParameterInitializer& initializer, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"");
    };

    // Implementation note: The Variable type is a value type and not polymorphic in nature. 
    // However we have a couple of derivatives of the type to extend the base interface and thus we ensure that the derived types do not have additional fields.
    // This check is weak in that the derives types may sneak in some additional fields if the base type had some padding at the end, without changing the object size
    // but it should be good enough for catching any accidental addiiton of fields.
    static_assert(sizeof(Constant) == sizeof(Variable), "The Constant type should not have any data fields beyond what its base type 'Variable' has.");
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
    template <> struct hash<::CNTK::Variable>
    {
        size_t operator()(const ::CNTK::Variable& x) const
        {
            return std::hash<const void*>()(x.m_dataFields.get());
        }
    };

    template <> struct hash<::CNTK::Parameter>
    {
        size_t operator()(const ::CNTK::Parameter& x) const
        {
            return std::hash<::CNTK::Variable>()(x);
        }
    };

    template <> struct hash<::CNTK::Constant>
    {
        size_t operator()(const ::CNTK::Constant& x) const
        {
            return std::hash<::CNTK::Variable>()(x);
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
        virtual DeviceDescriptor Device() const { return m_data->Device(); }

        ///
        /// Returns the data type of 'this' Value's contents.
        ///
        virtual DataType GetDataType() const { return m_data->GetDataType(); }

        ///
        /// Returns the storage format of 'this' Value.
        ///
        virtual StorageFormat GetStorageFormat() const { return m_data->GetStorageFormat(); }

        ///
        /// Returns the shape 'this' Value.
        ///
        virtual const NDShape& Shape() const { return m_data->Shape(); }

        ///
        /// Returns a boolean indicating if 'this' Value contains data in sparse storage format.
        ///
        bool IsSparse() const { return (GetStorageFormat() != StorageFormat::Dense); }

        ///
        /// Returns a boolean indicating if 'this' Value is read-only.
        ///
        virtual bool IsReadOnly() const { return m_data->IsReadOnly(); }

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
            std::tie(maxSequenceLen, numOfSequences) = GetSequenceAndBatchLength(variable);

            std::vector<std::ptrdiff_t> sequenceBeginIndices(numOfSequences, 0);
            std::vector<size_t> sequenceLengths(numOfSequences, maxSequenceLen);
            GetSequenceStartsAndLengths(Mask(), sequenceBeginIndices, sequenceLengths, variable.DynamicAxes().size());

            auto valueShapeWithSequenceAndBatchAxes = variable.Shape().AppendShape(NDShape({ maxSequenceLen , numOfSequences }));
            auto valueData = Data()->AsShape(valueShapeWithSequenceAndBatchAxes);
            if (valueData->Device() != device)
                valueData = valueData->DeepClone(device, valueData->IsReadOnly());

            std::vector<NDArrayViewPtr> sequences(numOfSequences);
            std::vector<bool> sequenceStartFlags(numOfSequences);
            for (size_t i = 0; i < numOfSequences; ++i)
            {
                if (!sequenceSegmentsAllowed && (sequenceBeginIndices[i] != 0))
                    RuntimeError("Value::UnpackVariableValue: Only Value objects containing the entire sequence (no segments) are supported.");

                std::vector<size_t> offset(valueShapeWithSequenceAndBatchAxes.Rank(), 0);
                offset.back() = i;

                std::vector<size_t> extent(valueShapeWithSequenceAndBatchAxes.Rank() - 1, NDShape::InferredDimension);
                extent.back() = sequenceLengths[i];

                sequences[i] = valueData->SliceView(offset, extent, valueData->IsReadOnly());
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
            if (outputVariable.GetDataType() != GetDataType())
                InvalidArgument("The outputVariable '%S' has a different data type than the Value object.", outputVariable.AsString().c_str());

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
            if (outputVariable.GetDataType() != dataType)
                InvalidArgument("The outputVariable '%S' has a different data type than the Value object.", outputVariable.AsString().c_str());

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
        /// If the value stored is a scalar, returns it. Otherwise, throws an error.
        ///
        template<typename ElementType>
        ElementType AsScalar() const;

    private:
        template <typename ElementType>
        static void AppendSparseSequenceData(const NDArrayViewPtr& sequenceData, std::vector<SparseIndexType>& colStarts, std::vector<SparseIndexType>& rowIndices, std::vector<char>& nonZeroValues, size_t maxSequenceLength);

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

        virtual std::pair<size_t, size_t> GetSequenceAndBatchLength(const Variable& outputVariable);

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
            auto shape = outputVariable.Shape();
            if (shape == NDShape::Unknown || shape.HasInferredDimension())
                RuntimeError("The outputVariable '%S' shape '%S' is unknown shape or has inferred dimension for at least one axis.",
                              outputVariable.AsString().c_str(), shape.AsString().c_str());

            size_t numOfSequences;
            size_t maxSequenceLen;
            std::tie(maxSequenceLen, numOfSequences) = GetSequenceAndBatchLength(outputVariable);

            // Calculate the number of elements is needed to represent a sample in output buffer.
            // For dense output, it is the total size of the shape.
            // For one-hot output, only 1 index is needed to represent the sample.
            size_t outputSizeOfSample = (std::is_same<ElementType, size_t>::value) ? 1 : shape.TotalSize();

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
        friend class Trainer;

        friend Variable GetCorrespondingOutputVariableFromClone(const Variable&, const FunctionPtr&, const FunctionPtr&);

    public:

        ///
        /// Computes and stores the values of specified variables in the 'outputs' map, using provided 'inputs' values corresponding
        /// to each leaf variable of the Function of VariableKind 'Input'.
        /// The variables specified in the 'outputs' map denote the subset of 'this' Function's output variables that the caller wants to obtain values of. 
        /// Callers may specify the storage to be used for storing the 'outputs' Values or pass null in which case the implementation allocates the actual storage
        /// for the 'outputs' for which the ValuePtr mapping was left null by the caller.
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
        /// in which case the implementation allocates the actual storage for storing the gradients.
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
        /// for the 'outputs' for which the ValuePtr mapping was left null by the caller.
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
        CNTK_API virtual void InferOutputs(std::vector<Variable>& outputs) = 0;

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
        CNTK_API virtual Dictionary Serialize() const override { return Dictionary(); }

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
        /// Deserializes a Function from the dictionary.
        /// TODO: add a second overload with a 'Function builder' parameter that would allow hooking
        /// user-defined op-codes with custom functionality.
        ///
        CNTK_API static FunctionPtr Deserialize(const Dictionary& dictionary, const ::CNTK::DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());

    public:
        ///
        /// Returns the name of 'this' Function.
        ///
        const std::wstring& Name() const { return m_name; }

        ///
        /// Sets the name of 'this' Function.
        /// Setting the name of a Function is only allowed if the Function does not already have a name.
        /// Calling this method, when 'this' Function already has a name, results in an exception.
        ///
        CNTK_API void SetName(const std::wstring& name);

        ///
        /// Returns the internally generated unique name of the Function
        ///
        const std::wstring& Uid() const { return m_uid; }

        ///
        /// Returns the primitive Function at the root of the graph of Functions underlying this Function.
        /// If 'this' Function itself is a primitive Function then (this->RootFunction() == this).
        ///
        FunctionPtr RootFunction() const
        {
            return (m_rootFunction == nullptr) ? const_cast<Function*>(this)->shared_from_this() : m_rootFunction;
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
        /// Returns all Input variables of 'this' Function.
        ///
        std::vector<Variable> Inputs(bool pythonOperandOrder = false) const
        {
            return *(InputsImpl(pythonOperandOrder).get());
        }

        ///
        /// Returns the Output variable of 'this' Function. Throws an exception of 'this' Function has more that one output.
        ///
        Variable Output() const
        {
            auto outputs = Outputs();
            if (outputs.size() > 1)
                RuntimeError("A Function instance '%S' with more than one output cannot be implicitly converted to a Variable.", AsString().c_str());

            return outputs[0];
        }

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
        /// Save this Function graph into a model file.
        ///
        CNTK_API void SaveModel(const std::wstring& modelFile);

        ///
        /// Restore the models parameters (in-place) from a model file
        ///
        CNTK_API void RestoreModel(const std::wstring& modelFilePath);

        ///
        /// Load a Function from a model file
        ///
        CNTK_API static FunctionPtr LoadModel(const std::wstring& modelFile, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Load a Function from a memory buffer
        ///
        CNTK_API static FunctionPtr LoadModel(char *modelBuffer, size_t modelBufferLength, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Load a Function from an istream. The legacy V1 model is not supported.
        ///
        CNTK_API static FunctionPtr LoadModel(std::istream& inputStream, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Prints the entire graph underlying this Function to stderr
        ///
        CNTK_API void PrintGraph() const;

        ///
        /// Returns a string representation of this Function
        ///
        CNTK_API std::wstring AsString(bool doNotInferOutputs = true) const;

        ///
        /// Maximum number of outputs that is currently supported.
        ///
        static const int MaxNumOutputs = 64;

    protected:
        static bool IsArgument(const Variable& var)
        {
            return (var.IsInput() || var.IsPlaceholder() || var.IsOutput());
        }

        ///
        /// Protected constructor for derived 'Function' types to specify the actual input and output variables for the (primitive) Function instance.
        ///
        CNTK_API Function(const std::vector<Variable>& inputs, Dictionary&& functionConfig, const std::wstring& name = L"", const std::wstring& uid = Internal::GenerateUid(L"UserDefinedFunction"));

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
        static bool ValidateOrUpdateOutput(const Variable& output, const Variable& newOutput, bool alwaysUpdate);

        // Returns a outputs without ref-counting the owner.
        CNTK_API std::vector<Variable>& RawOutputs() const;

    private:
        CNTK_API std::shared_ptr<std::vector<std::pair<Variable, Variable>>> BlockArgumentsMappingImpl() const;

        // Lazily initialize the Function's outputs on first invocation
        CNTK_API std::vector<Variable>& InitOutputs();

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

        void ValidateOrUpdateOutputs(std::unordered_map<const Function*, size_t>& visitedFunctions, bool& recurrentNodeOutputModified, std::vector<Variable>& buffer);

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
        CNTK_API Function(const std::vector<Variable>& inputs, const std::wstring& name = L"", const std::wstring& uid = Internal::GenerateUid(L"UserDefinedFunction"));

    private:
        CNTK_API Function(const std::vector<Variable>& inputs, Dictionary&& functionConfig, const FunctionPtr& rootFunction, const std::wstring& name, const std::wstring& uid);

        std::vector<Variable> m_inputs;
        std::once_flag m_outputsInitFlag;
        std::vector<Variable> m_outputs;

        FunctionPtr m_rootFunction; // nullptr for primitive Function instances
        std::wstring m_name;
        std::wstring m_uid;
        Dictionary m_attributes;
    };

    ///
    /// Create an instance of the CNTK built-in elementwise negate operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Negate(const Variable& operand, const std::wstring& name = L"");

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
    CNTK_API FunctionPtr Sigmoid(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise tanh operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Tanh(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise sine operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Sin(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise cosine operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Cos(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise linear rectifier operation with the specified input operand.
    ///
    CNTK_API FunctionPtr ReLU(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise exp operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Exp(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise log operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Log(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise square operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Square(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise square-root operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Sqrt(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise round operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Round(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise floor operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Floor(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise ceil operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Ceil(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise abs operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Abs(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise reciprocal operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Reciprocal(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in softmax operation on specified tensor input operand
    /// TODO: this Softmax() needs to support specifying the axis
    ///
    CNTK_API FunctionPtr Softmax(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in hardmax operation on specified tensor input operand
    ///
    CNTK_API FunctionPtr Hardmax(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in transpose dimensions operation on specified tensor input operand
    ///
    CNTK_API FunctionPtr TransposeAxes(const Variable& operand, const Axis& axis1, const Axis& axis2, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in transpose operation on the specified 1D or 2D input operand
    ///
    CNTK_API FunctionPtr Transpose(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the slice operation on specified tensor input operand
    ///
    CNTK_API FunctionPtr Slice(const Variable& operand, const std::vector<Axis>& axis, const std::vector<int>& beginIndex, const std::vector<int>& endIndex, const std::wstring& name = L"");

    ///
    /// Create an instance of the random_sample operation on specified sampling weights input vector
    ///
    CNTK_API FunctionPtr RandomSample(const Variable& operand, size_t numSamples, bool allowDuplicates, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const std::wstring& name = L"");

    ///
    /// Create an instance of the random_sample_inclusion_frequency operation on specified sampling weights input vector
    ///
    CNTK_API FunctionPtr RandomSampleInclusionFrequency(const Variable& operand, size_t numSamples, bool allowDuplicates, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const std::wstring& name = L"");

    ///
    /// Create an instance of the dropout operation on specified tensor input operand
    ///
    CNTK_API FunctionPtr Dropout(const Variable& operand, double dropoutRate, unsigned long seed = SentinelValueForAutoSelectRandomSeed, const std::wstring& name = L"");

    ///
    /// Create an instance of the reshape operation on specified tensor input operand
    ///
    CNTK_API FunctionPtr Reshape(const Variable& operand, const NDShape& replacementShape, const Axis& beginAxis, const Axis& endAxis, const std::wstring& name = L"");

    ///
    /// Create an instance of the reshape operation on specified tensor input operand
    ///
    inline FunctionPtr Reshape(const Variable& operand, const NDShape& newShape, const std::wstring& name = L"")
    {
        return Reshape(operand, newShape, Axis(0), Axis::EndStaticAxis(), name);
    }

    ///
    /// Create an instance of the CNTK built-in elementwise tensor addition operation with the specified input operands.
    ///
    CNTK_API FunctionPtr Plus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

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
    CNTK_API FunctionPtr Minus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

    ///
    /// Binary minus operator corresponding to the Minus operation
    ///
    inline FunctionPtr operator-(const Variable& leftOperand, const Variable& rightOperand)
    {
        return Minus(leftOperand, rightOperand);
    }

    /// Create an instance of the CNTK built-in elementwise tensor operation that computes the log of the sum of the exponentials of the specified input operands.
    ///
    CNTK_API FunctionPtr LogAddExp(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise multiplication operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr ElementTimes(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise division operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr ElementDivide(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise equality comparison operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr Equal(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise not-equal comparison operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr NotEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise less than comparison operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr Less(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise less than or equal to comparison operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr LessEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise greater than comparison operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr Greater(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise greater than or equal to comparison operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr GreaterEqual(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

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

    CNTK_API FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank, int inferInputRankToMap, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in tensor multiplication operation with the specified input operands.
    /// TODO: Specify the constraints on the shapes of the operands.
    /// TODO: Document inferInputRankToMap
    ///
    inline FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank, const std::wstring& name = L"")
    {
        return Times(leftOperand, rightOperand, outputRank, TimesNoInferredInputRank, name);
    }

    ///
    /// Create an instance of the CNTK built-in tensor multiplication operation with the specified input operands.
    /// TODO: Specify the constraints on the shapes of the operands.
    /// TODO: Document inferInputRankToMap
    ///
    inline FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"")
    {
        return Times(leftOperand, rightOperand, /*outputRank =*/ 1, name);
    }

    ///
    /// Create an instance of the CNTK built-in matrix multiplication operation with the transpose of the left input operand
    /// and the specified right operand. Only accepts left operands of ranks 1 or 2.
    /// TODO: Specify the constraints on the shapes of the operands.
    ///
    CNTK_API FunctionPtr TransposeTimes(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in matrix multiplication operation with the transpose of the left input operand
    /// and the specified right operand. Only accepts left operands of ranks 1 or 2.
    /// TODO: Specify the constraints on the shapes of the operands.
    ///
    inline FunctionPtr TransposeTimes(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"")
    {
        return TransposeTimes(leftOperand, rightOperand, /*outputRank =*/ 1, name);
    }


    ///
    /// Create an instance of the CNTK built-in operation to compute the cosine distance for the specified input operands.
    ///
    CNTK_API FunctionPtr CosineDistance(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation to compute the cosine distance with negative samplesfor the specified input operands.
    ///
    CNTK_API FunctionPtr CosineDistanceWithNegativeSamples(const Variable& leftOperand, const Variable& rightOperand, size_t shiftWindow, size_t numberOfNegativeSamples, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation to compute binary cross-entropy for specified input operands.
    ///
    CNTK_API FunctionPtr BinaryCrossEntropy(const Variable& prediction, const Variable& targets, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation to compute weighted binary cross-entropy for specified input operands.
    ///
    CNTK_API FunctionPtr WeightedBinaryCrossEntropy(const Variable& prediction, const Variable& targets, const Variable& weights, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation to compute squared-error for specified input operands.
    ///
    CNTK_API FunctionPtr SquaredError(const Variable& prediction, const Variable& targets, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation to compute cross-entropy with softmax for specified input operands.
    ///
    CNTK_API FunctionPtr CrossEntropyWithSoftmax(const Variable& prediction, const Variable& labels, const Axis& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation to compute cross-entropy with softmax for specified input operands.
    ///
    inline FunctionPtr CrossEntropyWithSoftmax(const Variable& prediction, const Variable& labels, const std::wstring& name = L"")
    {
        return CrossEntropyWithSoftmax(prediction, labels, Axis(0), name);
    }

    ///
    /// Create an instance of the CNTK built-in operation for computing the edit distance error for specified operands.
    ///
    CNTK_API FunctionPtr EditDistanceError(const Variable& prediction, const Variable& labels, float substitutionPenalty, float deletionPenalty, float insertionPenalty, bool squashInputs, const std::vector<size_t>& tokensToIgnore, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation for computing the forwardbackward for specified operands.
    ///
    CNTK_API FunctionPtr ForwardBackward(const Variable& graph, const Variable& features, size_t blankTokenId, int delayConstraint, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation for computing the labels to graph for input operands.
    ///
    CNTK_API FunctionPtr LabelsToGraph(const Variable& labels, const std::wstring& name = L"");


    ///
    /// Create an instance of the CNTK built-in operation for computing the classification prediction error for specified operands.
    ///
    CNTK_API FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, size_t topN, const Axis& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation for computing the classification prediction error for specified operands.
    ///
    inline FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, size_t topN, const std::wstring& name = L"")
    {
        return ClassificationError(prediction, labels, topN, Axis(0), name);
    }

    ///
    /// Create an instance of the CNTK built-in operation for computing the classification prediction error for specified operands.
    ///
    inline FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, const Axis& axis, const std::wstring& name = L"")
    {
        return ClassificationError(prediction, labels, /*topN =*/ 1, axis, name);
    }

    ///
    /// Create an instance of the CNTK built-in operation for computing the classification prediction error for specified operands.
    ///
    inline FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, const std::wstring& name = L"")
    {
        return ClassificationError(prediction, labels, Axis(0), name);
    }

    ///
    /// Create an instance of the CNTK built-in LambdaRank loss an effective proxy for optimizing the NDCG metric
    ///
    CNTK_API FunctionPtr LambdaRank(const Variable& prediction, const Variable& gains, const Variable& groupId, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation for evaluating the NDCG at 1 metric
    ///
    CNTK_API FunctionPtr NDCGAt1(const Variable& prediction, const Variable& gains, const Variable& groupId, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation for getting the past value along the lone dynamic axis of the specified operand.
    /// Throws an exception of the operand has more than one dynamic axis.
    ///
    CNTK_API FunctionPtr PastValue(const Variable& operand, const Variable& initialState, size_t offset = 1, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation for getting the past value along the lone dynamic axis of the specified operand.
    /// This overload uses an initial state value of 0.
    /// Throws an exception of the operand has more than one dynamic axis.
    ///
    inline FunctionPtr PastValue(const Variable& operand, size_t offset = 1, const std::wstring& name = L"")
    {
        static const auto defaultInitialState = Constant::Scalar(0.0f);
        return PastValue(operand, defaultInitialState, offset, name);
    }

    ///
    /// Create an instance of the CNTK built-in operation for getting the future value along the lone dynamic axis of the specified operand.
    /// Throws an exception of the operand has more than one dynamic axis.
    ///
    CNTK_API FunctionPtr FutureValue(const Variable& operand, const Variable& initialState, size_t offset = 1, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation for getting the future value along the lone dynamic axis of the specified operand.
    /// This overload uses an initial state value of 0.
    /// Throws an exception of the operand has more than one dynamic axis.
    ///
    inline FunctionPtr FutureValue(const Variable& operand, size_t offset = 1, const std::wstring& name = L"")
    {
        static const auto defaultInitialState = Constant::Scalar(0.0f);
        return FutureValue(operand, defaultInitialState, offset, name);
    }

    CNTK_API FunctionPtr OneHotOp(const Variable& operand, size_t numClass, bool outputSparse, Axis& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in sum reduction operation on specified tensor input operand along all the axes
    ///
    CNTK_API FunctionPtr ReduceSum(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in sum reduction operation on specified tensor input operand along the specified axis
    ///
    CNTK_API FunctionPtr ReduceSum(const Variable& operand, const Axis& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in LogSum reduction operation on specified tensor input operand along the specified axis
    ///
    CNTK_API FunctionPtr ReduceLogSum(const Variable& operand, const Axis& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in Mean reduction operation on specified tensor input operand along the specified axis
    ///
    CNTK_API FunctionPtr ReduceMean(const Variable& operand, const Axis& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in Max reduction operation on specified tensor input operand along the specified axis
    ///
    CNTK_API FunctionPtr ReduceMax(const Variable& operand, const Axis& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in Min reduction operation on specified tensor input operand along the specified axis
    ///
    CNTK_API FunctionPtr ReduceMin(const Variable& operand, const Axis& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in Prod reduction operation on specified tensor input operand along the specified axis
    ///
    CNTK_API FunctionPtr ReduceProd(const Variable& operand, const Axis& axis, const std::wstring& name = L"");

    ///
    /// Per dimension mean-variance normalization of the specified input operand.
    ///
    CNTK_API FunctionPtr PerDimMeanVarianceNormalize(const Variable& operand, const NDArrayViewPtr& mean, const NDArrayViewPtr& invStdDev, const std::wstring& name = L"");


    ///
    /// Convolution 
    ///
    CNTK_API FunctionPtr Convolution(const Variable& convolutionMap,
                                     const Variable& operand, 
                                     const NDShape& strides = { 1 },
                                     const std::vector<bool>& sharing = { true },
                                     const std::vector<bool>& autoPadding = { true },
                                     size_t maxTempMemSizeInSamples = 0, 
                                     const std::wstring& name = L"");

    ///
    /// Convolution transpose
    ///
    CNTK_API FunctionPtr ConvolutionTranspose(const Variable& convolutionMap,
        const Variable& operand,
        const NDShape& strides = { 1 },
        const std::vector<bool>& sharing = { true },
        const std::vector<bool>& autoPadding = { true },
        const NDShape& outputShape = { 0 },
        size_t maxTempMemSizeInSamples = 0,
        const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in ROI pooling operation on specified tensor input operands with the specified output shape
    ///
    CNTK_API FunctionPtr ROIPooling(const Variable& convolutionMap, const Variable& rois, const NDShape& roiOutputShape, const std::wstring& name = L"");

    ///
    /// TODO:
    ///
    enum class PoolingType
    {
        Max,
        Average,
    };

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
                                 const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in Unpooling operation on specified tensor input operands with the specified type and shape
    ///
    CNTK_API FunctionPtr Unpooling(const Variable& operand,
                                   const Variable& poolingInput,
                                   PoolingType UnpoolingType,
                                   const NDShape& UnpoolingWindowShape,
                                   const NDShape& strides = { 1 },
                                   const std::vector<bool>& autoPadding = { false },
                                   const std::wstring& name = L"");

    ///
    /// TODO:
    ///
    // TODO: Do we need a separate "spatial" parameter or can it be inferred from the tensor dimensions
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
                                            const std::wstring& name = L"");

    /// Create an instance of the CNTK built-in OptimizedRNNStack operation on specified input operands
    ///
    CNTK_API FunctionPtr OptimizedRNNStack(const Variable& operand, const Variable& weights, size_t hiddenSize, size_t numLayers, bool bidirectional = false, const std::wstring& recurrentOp = L"lstm", const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise clip operation on the tensor operand
    ///
    CNTK_API FunctionPtr Clip(const Variable& operand, const Variable& min, const Variable& max, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise choice operation using a condition tensor for specified tensor operands.
    ///
    CNTK_API FunctionPtr ElementSelect(const Variable& condition, const Variable& thenOperand, const Variable& elseOperand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in splice operation to splice together all the specified tensor operands into a single output tensor
    ///
    CNTK_API FunctionPtr Splice(const std::vector<Variable>& operands, const Axis& axis, const std::wstring& name = L"");

    ///
    /// Create a new Function instance which just combines the outputs of the specified list of 'operands' Functions such that the 'Outputs' of the 
    /// new 'Function' are union of the 'Outputs' of each of the specified 'operands' Functions.
    /// E.g. When creating a classification model, typically the CrossEntropy loss Function and the ClassificationError Function comprise the two roots
    /// of the computation graph which can be "Combine"d to create a single Function with 2 outputs; viz. CrossEntropy loss and ClassificationError output.
    ///
    CNTK_API FunctionPtr Combine(const std::vector<Variable>& operands, const std::wstring& name = L"");

    ///
    /// Creates a new Function instance which is just an alias of the specified operand.
    ///
    CNTK_API FunctionPtr Alias(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Creates a Block Function that encapsulates a composite to create an opaque Function object that
    /// appears as any other primitive Function
    ///
    CNTK_API FunctionPtr AsBlock(FunctionPtr&& composite, const std::vector<std::pair<Variable, Variable>>& argumentsMap, const std::wstring& blockOpName, const std::wstring& blockName = L"");

    ///
    /// Creates a new Function instance which output its input as it is and previent any gradient contribution from its output. 
    ///
    CNTK_API FunctionPtr StopGradient(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Creates a composite Function that has the specified rootFunction as its root.
    /// The composite denotes a higher-level Function encapsulating the entire graph
    /// of Functions underlying the specified rootFunction.
    ///
    CNTK_API FunctionPtr AsComposite(const FunctionPtr& rootFunction, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise exponential linear unit operation with the specified input operand.
    ///
    CNTK_API FunctionPtr ELU(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise leaky linear rectifier operation with the specified input operand.
    ///
    CNTK_API FunctionPtr LeakyReLU(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise parametric rectified linear Unit operation 
    /// with the specified input operand and learning parameter alpha.
    ///
    CNTK_API FunctionPtr PReLU(const Variable& alpha, const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise softplus operation 
    ///
    CNTK_API FunctionPtr Softplus(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in argmax operation on specified tensor input operand along the specified axis
    ///
    CNTK_API FunctionPtr Argmax(const Variable& operand, const Axis& axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in argmin on specified tensor input operand along the specified axis
    ///
    CNTK_API FunctionPtr Argmin(const Variable& operand, const Axis& axis, const std::wstring& name = L"");

    namespace Sequence
    {
        CNTK_API FunctionPtr IsFirst(const Variable& operand, const std::wstring& name = L"");
        CNTK_API FunctionPtr IsLast(const Variable& operand, const std::wstring& name = L"");

        CNTK_API FunctionPtr Slice(const Variable& operand, int beginIndex, int endIndex, const std::wstring& name = L"");

        ///
        /// Create an instance of the CNTK built-in sum reduction operation on specified tensor input operand along the operands lone dynamic sequence axis
        ///
        CNTK_API FunctionPtr ReduceSum(const Variable& operand, const std::wstring& name = L"");

        CNTK_API FunctionPtr First(const Variable& operand, const std::wstring& name = L"");
        CNTK_API FunctionPtr Last(const Variable& operand, const std::wstring& name = L"");

        CNTK_API FunctionPtr Where(const Variable& condition, const std::wstring& name = L"");
        CNTK_API FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::wstring& name = L"");
        CNTK_API FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, const std::wstring& name = L"");

        CNTK_API FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::wstring& name = L"");
        CNTK_API FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::pair<size_t, int>& newDerivedSequenceAxisScalingAndAdditiveFactor, const std::wstring& name = L"");

        CNTK_API FunctionPtr BroadcastAs(const Variable& operand, const Variable& broadcastAs, const std::wstring& name = L"");
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
        ///
        /// Indicates whether the values in the schedule are specified on the per-sample or 
        /// per-minibatch basis.
        ///
        enum class UnitType : unsigned int
        {
            Sample = 0,
            Minibatch = 1,
        };

        ///
        /// A special value that can be used for the epochSize to indicate that the schedule is sweep-based.
        ///
        static const size_t FullDataSweep = 0;

        ///
        /// Create a schedule with a constant parameter value.
        ///
        CNTK_API TrainingParameterSchedule(T value, UnitType unit);

        ///
        /// Create a schedule where the parameter changes its value every 'epochSize' samples:
        /// schedule[0] is used for the first 'epochSize' samples, schedule[1] -- for the second,
        /// and so on. The last value is then used repeatedly until the end of training.
        ///
        CNTK_API TrainingParameterSchedule(const std::vector<T>& schedule, UnitType unit, size_t epochSize = FullDataSweep);

        ///
        /// Create a schedule using the list of key-value pairs, where the key specifies 
        /// the number of epochs the parameter should maintain the corresponding value,
        /// (which 'epochSize' samples in each epoch). The value from the last pair is used 
        /// repeatedly until the end of training. For example, {{1, 0.05}, {2, 0.1}, {1, 0.005}} 
        /// and epochSize = 100, corresponds to a schedule where the value of '0.05' is used for 
        /// the first 100 samples, then '0.1' is used for the second 200 samples, 
        /// after which the values is switched to '0.005'.
        ///
        CNTK_API TrainingParameterSchedule(const std::vector<std::pair<size_t, T>>& schedule, UnitType unit, size_t epochSize = FullDataSweep);

        ///
        /// Returns a value corresponding to the absolute sample (or sweep) 
        /// count from the beginning of training.
        ///
        CNTK_API const T& operator[](size_t count) const;

        ///
        /// Returns the unit type for 'this' training parameter schedule. 
        /// In case when the values are specified on the per-Minibatch basis, they are
        /// re-scaled by the learner using the actual minibatch size in samples.
        ///
        UnitType Unit() const { return m_unit; }

        bool IsSweepBased() const { return m_epochSize == FullDataSweep; }

        CNTK_API virtual ~TrainingParameterSchedule();

        CNTK_API TrainingParameterSchedule(const TrainingParameterSchedule<T>&); 
        CNTK_API TrainingParameterSchedule(TrainingParameterSchedule<T>&&); 
        CNTK_API TrainingParameterSchedule<T>& operator=(const TrainingParameterSchedule<T>&); 
        CNTK_API TrainingParameterSchedule<T>& operator=(TrainingParameterSchedule<T>&&);

        CNTK_API virtual Dictionary Serialize() const override;

        virtual size_t CurrentVersion() const override { return s_serializationVersion; }

        CNTK_API static TrainingParameterSchedule<T> Deserialize(const Dictionary& dictionary);

    private:

        friend class Learner;

        CNTK_API void ConstructSchedule(const std::vector<std::pair<size_t, T>>& schedule);

        CNTK_API TrainingParameterSchedule(const Dictionary& dictionary);

        static const size_t s_serializationVersion = 1;

    protected:           
        std::map<size_t, T> m_schedule;
        UnitType m_unit;
        size_t m_epochSize;
    };

    template <typename T, typename TrainingParameterSchedule<T>::UnitType U>
    class TrainingParameterPerUnitSchedule : public TrainingParameterSchedule<T>
    {
    public:
        TrainingParameterPerUnitSchedule(T value)
            : TrainingParameterSchedule<T>::TrainingParameterSchedule(value, U)
        { }

        TrainingParameterPerUnitSchedule(const std::vector<T>& schedule, 
                                         size_t epochSize = TrainingParameterSchedule<T>::FullDataSweep)
            : TrainingParameterSchedule<T>::TrainingParameterSchedule(schedule, U, epochSize)
        { }


        TrainingParameterPerUnitSchedule(const std::vector<std::pair<size_t, T>>& schedule, 
                                         size_t epochSize = TrainingParameterSchedule<T>::FullDataSweep)
            : TrainingParameterSchedule<T>::TrainingParameterSchedule(schedule, U, epochSize)
        { }

#ifdef SWIG // for Python interop (adds indexer)
        const T __getitem__(size_t count) const
        {
            return TrainingParameterSchedule<T>::operator[](count);
        }
#endif
    };

// Swig does not understand type aliasing.
#ifndef SWIG
    ///
    /// Training parameter schedule with per-sample values.
    ///
    template <typename T>
    using TrainingParameterPerSampleSchedule = TrainingParameterPerUnitSchedule<T, TrainingParameterSchedule<T>::UnitType::Sample>;

    ///
    /// Training parameter schedule with per-minibatch values.
    ///
    template <typename T>
    using TrainingParameterPerMinibatchSchedule = TrainingParameterPerUnitSchedule<T, TrainingParameterSchedule<T>::UnitType::Minibatch>;

    typedef TrainingParameterPerSampleSchedule<double> LearningRatePerSampleSchedule;
    typedef TrainingParameterPerMinibatchSchedule<double> LearningRatePerMinibatchSchedule;

    typedef TrainingParameterPerSampleSchedule<double> MomentumPerSampleSchedule;
    typedef TrainingParameterPerMinibatchSchedule<double> MomentumPerMinibatchSchedule;
#endif

    typedef TrainingParameterPerUnitSchedule<size_t, TrainingParameterSchedule<size_t>::UnitType::Sample> MinibatchSizeSchedule;
    typedef TrainingParameterSchedule<double> LearningRateSchedule;
    typedef TrainingParameterSchedule<double> MomentumSchedule;

    ///
    /// This class allows to specify momentum as time constant in place of momentum per sample in 
    /// all of Learners factory methods. The specified values are then automatically converted into 
    /// per sample values.
    ///
    class MomentumAsTimeConstantSchedule: public TrainingParameterSchedule<double>
    {
    public:
        MomentumAsTimeConstantSchedule(double value) 
            : TrainingParameterSchedule<double>::TrainingParameterSchedule(value, UnitType::Sample)
        { 
            ConvertToPerSampleValues();
        }
        
        MomentumAsTimeConstantSchedule(const std::vector<double>& schedule, size_t epochSize = FullDataSweep) 
            : TrainingParameterSchedule<double>::TrainingParameterSchedule(schedule, UnitType::Sample, epochSize) 
        { 
            ConvertToPerSampleValues();
        }
        
        MomentumAsTimeConstantSchedule(const std::vector<std::pair<size_t, double>>& schedule, size_t epochSize = FullDataSweep) 
            : TrainingParameterSchedule<double>::TrainingParameterSchedule(schedule, UnitType::Sample, epochSize)
        { 
            ConvertToPerSampleValues();
        }

#ifdef SWIG // for Python interop (adds indexer)
        const double __getitem__(size_t count) const
        {
            return operator[](count);
        }
#endif

    private:
        CNTK_API void ConvertToPerSampleValues();
    };

    ///
    /// A collection of additional options that affect parameter updates and 
    /// are applicable for all standard learners 
    ///
    struct AdditionalLearningOptions
    {
        double l1RegularizationWeight = 0.0;
        double l2RegularizationWeight = 0.0;
#ifdef SWIG //for python interop (swig does not fully support "using")
        TrainingParameterPerUnitSchedule<double, TrainingParameterSchedule<double>::UnitType::Minibatch> gaussianNoiseInjectionStdDev = 0.0;
#else
        TrainingParameterPerMinibatchSchedule<double> gaussianNoiseInjectionStdDev = 0.0;
#endif
        double gradientClippingThresholdPerSample = std::numeric_limits<double>::infinity();
        bool gradientClippingWithTruncation = true;
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

    protected:
        ///
        /// Retrieves and returns current value from the training parameter schedule.
        ///
        template <typename ElementType>
        ElementType GetCurrentTrainingParameterValue(const TrainingParameterSchedule<ElementType>& schedule) const
        {
            if (schedule.IsSweepBased())
            {
                return schedule[m_sweepCount];
            }
            else
            {
                return schedule[m_sampleCount];
            }
        }

        Learner(const std::vector<Parameter>& parameters, const LearningRateSchedule& learningRateSchedule)
            : m_parameters(parameters),
            m_learningRateSchedule(learningRateSchedule),
            m_sampleCount(0),
            m_minibatchCount(0),
            m_sweepCount(0)
        {}

        std::vector<Parameter> m_parameters;
        LearningRateSchedule m_learningRateSchedule;
        size_t m_sampleCount;
        size_t m_minibatchCount;
        size_t m_sweepCount;
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
                                        double rho = 0.95,
                                        double epsilon = 1e-8,
                                        AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

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

    protected:
        DistributedLearner(DistributedCommunicatorPtr communicator, LearnerPtr learner, size_t distributeAfterSamples)
            : Learner(learner? learner->Parameters() : std::vector<Parameter>(),
                      LearningRateSchedule(0, LearningRateSchedule::UnitType::Sample)),
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
    /// Describes an input stream: its name, element type, storage, etc.
    ///
    struct StreamInformation
    {
        std::wstring m_name;           // Unique name of the stream
        size_t m_id;                   // Unique identifier of the stream
        StorageFormat m_storageFormat; // Storage format of the stream
        DataType m_elementType;        // Element type of the stream
        NDShape m_sampleLayout;        // Layout of the sample for the stream

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

        CNTK_API virtual ~Evaluator() {}

    private:
        template <typename T1, typename ...CtorArgTypes>
        friend std::shared_ptr<T1> MakeSharedObject(CtorArgTypes&& ...ctorArgs);

        friend class TrainingSession;

        // Returns aggregated evaluation criterion value and sample count.
        std::pair<ValuePtr, size_t> TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice, bool distributed);
        std::pair<ValuePtr, size_t> TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice, bool distributed);

        std::pair<ValuePtr, size_t> TestLocalMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice);

        void UpdateTestProgress(size_t numSamples, const ValuePtr& evalCriterion, const DeviceDescriptor& computeDevice);

    protected:
        Evaluator(const FunctionPtr& evaluationFunction, const std::vector<ProgressWriterPtr>& progressWriters = {}, bool initializeCombined = true);

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
        /// Total number of samples seen from the begining of the training.
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

        Trainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const std::vector<LearnerPtr>& parameterLearners,
                const std::vector<ProgressWriterPtr>& progressWriters = {});
        Trainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const FunctionPtr& evaluationFunction, const std::vector<LearnerPtr>& parameterLearners,
                const std::vector<ProgressWriterPtr>& progressWriters = {});

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

        // TODO: Workaround for back compat. Should not be used and will be removed in the next version.
        friend CNTK_API void ::CNTK::Internal::AddProgressWriters(const TrainerPtr&, const std::vector<ProgressWriterPtr>&);

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
        /// The minibatch size is specified terms of #samples and/or #sequences for the primary input stream; value of 0 for #samples/#sequences means unspecified.
        /// In case the size is specified in terms of both #sequences and #samples, the smaller of the 2 is taken.
        /// An empty map is returned when the MinibatchSource has no more data to return.
        ///
        CNTK_API const std::unordered_map<StreamInformation, MinibatchData>& GetNextMinibatch(
            size_t minibatchSizeInSequences,
            size_t minibatchSizeInSamples,
            const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Same as above but allows to specify partition of data in a distributed environment.
        /// Depending on the number of workers the data is splitted in different partitions,
        /// and depending on the worker rank, only a particular partition is read.
        ///
        CNTK_API virtual const std::unordered_map<StreamInformation, MinibatchData>& GetNextMinibatch(
            size_t minibatchSizeInSequences,
            size_t minibatchSizeInSamples,
            size_t numberOfWorkers,
            size_t workerRank,
            const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice()) = 0;

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
        // TODO: This is general enough and be hoisted out once there are specific use-cases outside of
        // configuring a MinibatchSource.
        enum TraceLevel : unsigned int
        {
            Error = 0,
            Warning = 1,
            Info = 2
        };

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
        size_t maxSamples { MinibatchSource::InfinitelyRepeat };

        ///
        /// The maximum allowed number of sweeps over the input dataset. After this number has been reached, 
        /// the reader returns empty minibatches on subsequent calls to GetNextMinibatch().
        /// 'maxSweeps' and 'maxSamples' are mutually exclusive, an exception will be raised if both have 
        /// non-default values.
        /// 
        size_t maxSweeps { MinibatchSource::InfinitelyRepeat };

        ///
        /// Size of the randomization window in chunks, non-zero value enables randomization. 
        /// 'randomizationWindowInChunks' and 'randomizationWindowInSamples' are mutually exclusive,
        /// an exception will be raised if both have non-zero values.
        ///
        size_t randomizationWindowInChunks { MinibatchSource::DefaultRandomizationWindowInChunks };

        ///
        /// Size of the randomization window in samples, non-zero value enables randomization. 
        /// 'randomizationWindowInChunks' and 'randomizationWindowInSamples' are mutually exclusive,
        /// an exception will be raised if both have non-zero values.
        ///
        size_t randomizationWindowInSamples { 0 };

        ///
        /// Output verbosity level.
        ///
        TraceLevel traceLevel { TraceLevel::Warning };

        ///
        /// Truncation length in samples, non-zero value enables the truncation (only applicable for BPTT,
        /// cannot be used in frame mode, an exception will be raised if frame mode is enabled and the 
        /// truncation length is non-zero).
        ///
        size_t truncationLength { 0 };

        ///
        /// Switches the frame mode on and off. If the frame mode is enabled the input data will be processed
        /// as individual frames ignoring all sequence information (this option cannot be used for BPTT,
        /// an exception will be raised if frame mode is enabled and the truncation length is non-zero).
        ///
        bool isFrameModeEnabled { false };

        ///
        /// Specifies if the deserialization should be done on a single or multiple threads.
        ///
        bool isMultithreaded { false };

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
        StreamConfiguration(const std::wstring& streamName, size_t dim, bool isSparse = false, const std::wstring& streamAlias = L"")
            : m_streamName(streamName), m_dim(dim), m_isSparse(isSparse), m_streamAlias(streamAlias)
        {}

        std::wstring m_streamName;
        size_t m_dim;
        bool m_isSparse;
        std::wstring m_streamAlias;
    };

    struct HTKFeatureConfiguration
    {
        HTKFeatureConfiguration(const std::wstring& streamName, const std::wstring& scp, size_t dim, size_t left, size_t right, bool broadcast)
            : m_streamName(streamName), m_dim(dim), m_scp(scp), m_left(left), m_right(right), m_broadcast(broadcast)
        {}

        std::wstring m_streamName;
        std::wstring m_scp;
        size_t m_dim;
        size_t m_left;
        size_t m_right;
        bool m_broadcast;
    };

    typedef Dictionary ImageTransform;

    /// 
    /// Create a crop transform with the specified options to be used with a reader
    /// 
    CNTK_API ImageTransform ReaderCrop(const wchar_t* cropType = L"center",
        int cropSize = 0, float sideRatio = 0.0f, float areaRatio = 0.0f,
        float aspectRatio = 1.0f, const wchar_t* jitterType = L"none");

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
    /// Create an CTFDeserializer with the specified options
    /// 
    CNTK_API  Deserializer CTFDeserializer(const std::wstring& fileName, const std::vector<StreamConfiguration>& streams);

    /// 
    /// Create an HTKFeatureDeserializer with the specified options
    /// 
    CNTK_API  Deserializer HTKFeatureDeserializer(const std::vector<HTKFeatureConfiguration>& streams);

    /// 
    /// Create an HTKMLFDeserializer with the specified options
    /// 
    CNTK_API  Deserializer HTKMLFDeserializer(const std::wstring& streamName, const std::wstring& labelMappingFile, size_t dimension, const std::vector<std::wstring>& mlfFiles);

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
            size_t crossValidationFrequencyInSamples = std::numeric_limits<size_t>::max());

    private:
        friend class TrainingSession;
        const MinibatchSourcePtr m_source;
        const MinibatchSizeSchedule m_mbSize;
        const size_t m_frequency;
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
            const MinibatchSizeSchedule& schedule = MinibatchSizeSchedule(64));

    private:
        friend class TrainingSession;
        const MinibatchSourcePtr m_source;
        const MinibatchSizeSchedule m_mbSize;
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

        /// !!! DEPRECATED !!!
        /// Constructor of the training session: 
        /// trainingSource : a minibatch source that will be used for training
        /// trainer : an instance of a trainer
        /// modelInputsToMinibatchSourceMapping : mapping between the input node of the model and the corresponding stream
        /// minibatchSizeSchedule : a minibatch size schedule used for training
        /// checkpointFrequencyInSamples : an approximate number of global samples processed accross the workers
        ///    after which the checkpoint is taken. Should be positive number if the checkpoint file is specified.
        /// checkpointFilename : a file name of the checkpoint file, if empty, the checkpointing is disabled.
        /// crossValidationSource: a minibatch source that will be used for cross validation.
        /// crossValidationSchedule : a minibatch size schedule for cross validation.
        /// restoreFromCheckpointIfExists: flag, indicating whether perform restore of the training session from the checkpoint before the start of the training.
        /// keepExistingCheckpoints: flag, indicating whether to store all checkpoints, by default only the last checkpoint is preserved
        /// maxNumberOfTrainingSamples : max number of samples after which the training should be stopped
        /// progressFrequency : an approximate number of global samples processed accross the workers
        ///    after which the summary of metrics is reported using the progress_printer
        ///
        CNTK_API TrainingSession(
            const MinibatchSourcePtr& trainingSource,
            const TrainerPtr& trainer,
            const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
            const MinibatchSizeSchedule& minibatchSizeSchedule,
            size_t checkpointFrequencyInSamples,
            const std::wstring& checkPointFileName,
            const MinibatchSourcePtr& crossValidationSource = nullptr,
            const MinibatchSizeSchedule& crossValidationSchedule = MinibatchSizeSchedule(1),
            size_t crossValidationFrequencyInSamples = std::numeric_limits<size_t>::max(),
            bool restoreFromCheckpointIfExists = true,
            bool keepExistingCheckpoints = false,
            size_t maxNumberOfTrainingSamples = std::numeric_limits<size_t>::max(),
            size_t progressFrequency = std::numeric_limits<size_t>::max(),
            const std::vector<ProgressWriterPtr>& progressWriters = {});

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
        void GetNextMinibatch(const MinibatchSourcePtr& source, std::unordered_map<Variable, ValuePtr>& minibatch, size_t maxMbSize, size_t workerRank, size_t numberOfWorkers, const DeviceDescriptor& computeDevice);
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
    /// !!! DEPRECATED !!!
    /// Creates an instance of the training session class. Parameters match the paramters of the TrainingSession constructor.
    ///
    CNTK_API TrainingSessionPtr CreateBasicTrainingSession(
        const MinibatchSourcePtr& trainingSource,
        const TrainerPtr& trainer,
        const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
        const MinibatchSizeSchedule& minibatchSizeSchedule,
        size_t checkpointFrequencyInSamples,
        const std::wstring& checkPointFileName,
        const MinibatchSourcePtr& crossValidationSource = nullptr,
        const MinibatchSizeSchedule& crossValidationSchedule = MinibatchSizeSchedule(1),
        size_t crossValidationFrequencyInSamples = std::numeric_limits<size_t>::max(),
        bool restoreFromCheckpointIfExists = true,
        bool keepExistingCheckpoints = false,
        size_t maxNumberOfTrainingSamples = std::numeric_limits<size_t>::max(),
        size_t progressFrequency = std::numeric_limits<size_t>::max(),
        const std::vector<ProgressWriterPtr>& progressWriters = {});

    CNTK_API void PrintBuiltInfo();

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
        CNTK_API ProgressWriter(size_t trainingUpdateWriteFrequency, size_t trainingFirstUpdatesToWrite,
                                size_t testUpdateWriteFrequency, size_t testFirstUpdatesToWrite);

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
        const CheckpointConfig& checkpointing,
        const CrossValidationConfig& crossValidation,
        const TestConfig& test);
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
