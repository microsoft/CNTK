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
#include<algorithm>
#include <mutex>


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
    enum class DataType
    {
        Unknown,
        Float,
        Double,

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
            LogicError("Unknown DataType");
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
    enum class DeviceKind
    {
        CPU,
        GPU,
        // TODO: FPGA
    };

    ///
    /// Denotes a compute device instance.
    ///
    class DeviceDescriptor final
    {
        friend bool operator==(const DeviceDescriptor& first, const DeviceDescriptor& second);

        static std::atomic<bool> s_defaultDeviceFrozen;
        static std::shared_ptr<DeviceDescriptor> s_defaultDevice;
        static std::shared_ptr<std::vector<DeviceDescriptor>> s_allDevices;
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
        /// Static method to get the descriptor of the CPU device on the local system.
        ///
        static DeviceDescriptor CPUDevice() { return{ 0, DeviceKind::CPU }; }

        ///
        /// Static method to get the descriptor of the GPU device on the local system with the specified CUDA device ID.
        ///
        CNTK_API static DeviceDescriptor GPUDevice(unsigned int deviceId);

        ///
        /// Static method to get the descriptor of the default device for the current process.
        /// This device is used for all CNTK operations where a device needs to be specified and one is not explicitly specified.
        ///
        CNTK_API static DeviceDescriptor DefaultDevice();

        ///
        /// Static method to get the descriptor of the default device for the current process.
        /// This device is used for all CNTK operations where a device needs to be specified and one is not explicitly specified.
        /// Additionally after this method gets executed for the first time, it freezes the default device of the process disallowing
        /// changing the default device by further calls to SetDefaultDevice.
        ///
        CNTK_API static DeviceDescriptor UseDefaultDevice();

        ///
        /// The default device can only be changed if it has not yet been implicitly used by any previous operation in the CNTK library.
        ///
        CNTK_API static void SetDefaultDevice(const DeviceDescriptor& newDefaultDevice);

        ///
        /// Static method to get the descriptor of the best available device.
        ///
        CNTK_API static DeviceDescriptor BestDevice();

        ///
        /// Static method to get a list of descriptors of all available/supported devices.
        ///
        CNTK_API static const std::vector<DeviceDescriptor>& AllDevices();

    private:
        DeviceDescriptor(unsigned int deviceId, DeviceKind deviceType)
            : m_deviceId(deviceId), m_deviceType(deviceType)
        {}

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
                InvalidArgument("NDShape::SubShape : The specified endAxisId (%d) cannot exceed the rank (%d) of 'this' NDShape and must be >= than the specified beginAxisId (%d)", (int)endAxisId, (int)Rank(), (int)beginAxisId);

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
                RuntimeError("NDShape::TotalSize : TotalSize cannot be determined for a NDShape with one or more dimensions being InferredDimension");

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
            std::wstringstream wStrStream;
            wStrStream << L"[";
            for (size_t i = 0; i < Rank(); i++)
            {
                if (i != 0)
                    wStrStream << L" x ";

                wStrStream << m_shapeDims[i];
            }

            wStrStream << L"]";
            return wStrStream.str();
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

    static const unsigned long DefaultRandomSeed = 1;
    ///
    /// Denotes a multi-dimensional writable or read-only array of elemental values.
    /// This type denotes a view and there may be multiple simultaneous views of the data underlying a NDArrayView instance.
    /// The underlying data is stored in sparse or dense format, and is located on a specific device.
    /// The actual underlying storage is either external or internal in which case its lifetime is managed through reference counting.
    ///
    class NDArrayView final : public std::enable_shared_from_this<NDArrayView>
    {
        friend class CompositeFunction;
        friend class LearnerBase;
        friend class Variable;
        friend class PackedValue;

        template <typename T, typename ...CtorArgTypes>
        friend inline std::shared_ptr<T> MakeSharedObject(CtorArgTypes&& ...ctorArgs);
    public:
        ///
        /// Construct a NDArrayView with the specified 'dataBuffer' as the backing storage.
        /// The 'dataBuffer' must have been allocated on the specified 'device', must be at least
        /// as large as the total size of the specified 'viewShape' and must outlive the created NDArrayView object.
        ///
        CNTK_API NDArrayView(CNTK::DataType dataType, const NDShape& viewShape, void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device, bool readOnly = false);

        /// Construct a read-only NDArrayView with the specified 'dataBuffer' as the backing storage.
        /// The 'dataBuffer' must have been allocated on the specified 'device', must be at least
        /// as large as the total size of the specified 'viewShape' and must outlive the created NDArrayView object.
        ///
        NDArrayView(CNTK::DataType dataType, const NDShape& viewShape, const void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device)
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
        CNTK_API NDArrayView(CNTK::DataType dataType, CNTK::StorageFormat storageType, const NDShape& viewShape, const DeviceDescriptor& device);

        ///
        /// Construct a NDArrayView over newly allocated dense storage on the specified 'device'.
        ///
        NDArrayView(CNTK::DataType dataType, const NDShape& viewShape, const DeviceDescriptor& device)
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
        /// Construct a NDArrayView with the buffer underlying the specified std::vector or std::aray being the underlying storage.
        /// The container must be at least as large as the total size of the specified 'viewShape' and should outlive the created NDArrayView object.
        ///
        template <typename ContainerType, typename std::enable_if<std::is_same<ContainerType, std::vector<typename ContainerType::value_type>>::value ||
                                                                  std::is_same<ContainerType, std::array<typename ContainerType::value_type, sizeof(ContainerType) / sizeof(typename ContainerType::value_type)>>::value>::type* = nullptr>
        NDArrayView(const NDShape& viewShape, ContainerType& sourceContainer, bool readOnly = false)
            : NDArrayView(viewShape, sourceContainer.data(), sourceContainer.size(), DeviceDescriptor::CPUDevice(), readOnly)
        {}

        ///
        /// Construct a read-only NDArrayView with the buffer underlying the specified std::vector or std::aray being the underlying storage.
        /// The container must be the same size as the total size of the specified 'viewShape' and should outlive the created NDArrayView object.
        ///
        template <typename ContainerType, typename std::enable_if<std::is_same<ContainerType, std::vector<typename ContainerType::value_type>>::value ||
                                                                  std::is_same<ContainerType, std::array<typename ContainerType::value_type, sizeof(ContainerType) / sizeof(typename ContainerType::value_type)>>::value>::type* = nullptr>
        NDArrayView(const NDShape& viewShape, const ContainerType& sourceContainer)
            : NDArrayView(viewShape, sourceContainer.data(), sourceContainer.size(), DeviceDescriptor::CPUDevice())
        {
            if (sourceContainer.size() != viewShape.TotalSize())
                InvalidArgument("The size of the STL container does not match the size of the specified viewShape");
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
                LogicError("Unsupported DataType %s", DataTypeName(m_dataType));
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
        inline NDArrayViewPtr DeepClone(bool readOnly = false) const
        {
            return DeepClone(this->Device(), readOnly);
        }

        ///
        /// Creates a new NDArrayView which is an alias of 'this' view; i.e. a new view of the same shape as 'this' over the same underlying data.
        ///
        CNTK_API NDArrayViewPtr Alias(bool readOnly = false) const;

        ///
        /// Copies the contents of the 'source' NDArrayView to 'this' view.
        /// The shapes of the 'source' view and 'this' view must be identical.
        ///
        CNTK_API void CopyFrom(const NDArrayView& source);

        ///
        /// Static method to construct a new NDArrayView object whose contents are drawn from a normal distribution with the specified mean and standard deviation..
        ///
        template <typename ElementType>
        CNTK_API static NDArrayViewPtr RandomNormal(const NDShape& shape, double mean, double stdDev, unsigned long seed = DefaultRandomSeed, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Static method to construct a new NDArrayView object whose contents are drawn from a uniform distribution in the specified value range.
        ///
        template <typename ElementType>
        CNTK_API static NDArrayViewPtr RandomUniform(const NDShape& shape, double rangeStart, double rangeEnd, unsigned long seed = DefaultRandomSeed, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());

    private:
        // Disallow copy and move construction and assignment
        NDArrayView(const NDArrayView&) = delete; NDArrayView& operator=(const NDArrayView&) = delete; NDArrayView& operator=(NDArrayView&&) = delete; NDArrayView(NDArrayView&& other) = delete;

    private:
        static const size_t AutoSelectRowColSplitPoint = SIZE_MAX;

    private:
        CNTK_API NDArrayView(CNTK::DataType dataType, const DeviceDescriptor& device, CNTK::StorageFormat storageType, const NDShape& viewShape, bool readOnly, void* tensorView);


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
        CNTK::DataType m_dataType;
        DeviceDescriptor m_device;
        CNTK::StorageFormat m_storageFormat;
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
        CNTK_API explicit NDMask(const NDShape& shape, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice());

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
    /// Denotes a multi-dimensional array with an optional mask and is the actual data fed into or produced from a computation.
    /// The mask is typically lower dimensionality than the data, meaning data is masked in coarse individual sample units where
    /// sample shape is data.Shape().SubShape(0, data.Shape().Rank() - mask.Shape().Rank)
    /// Also, note that the size of the data's trailing mask.Shape().Rank() dimensions must match the mask shape dimensions.
    /// 
    class Value : public std::enable_shared_from_this<Value>
    {
    public:
        ///
        /// A multi-dimensional value with no mask.
        ///
        CNTK_API Value(const NDArrayViewPtr& data);

        ///
        /// A multi-dimensional value with an associated mask.
        ///
        CNTK_API Value(const NDArrayViewPtr& data, const NDMaskPtr& mask);

        ///
        /// Create a new Value object containing a collection of variable length sequences.
        /// The created Value object contains a copy of the specified 'sequences' data.
        ///
        template <typename ElementType>
        CNTK_API static ValuePtr Create(const NDShape& sampleShape, const std::vector<std::vector<ElementType>>& sequences, const DeviceDescriptor& device, bool readOnly = false);

        ///
        /// Create a new Value object containing a collection of variable length sequences of one hot vectors
        /// The created Value object contains a copy of the specified 'sequences' data.
        ///
        template <typename ElementType>
        CNTK_API static ValuePtr Create(size_t vocabularySize, const std::vector<std::vector<size_t>>& oneHotSequences, const DeviceDescriptor& device, bool readOnly = false);

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
        bool IsSparse() const
        {
            return (GetStorageFormat() != StorageFormat::Dense);
        }

        ///
        /// Returns a boolean indicating if 'this' Value is read-only.
        ///
        virtual bool IsReadOnly() const { return m_data->IsReadOnly(); }

        ///
        /// Returns the number of masked/invalid values
        ///
        virtual size_t MaskedCount() const 
        {
            return m_mask ? m_mask->MaskedCount() : 0;
        }

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
        virtual ValuePtr DeepClone(bool readOnly = false) const;

        ///
        /// Creates a new Value which is an alias of 'this' Value.
        ///
        virtual ValuePtr Alias(bool readOnly = false) const;

        ///
        /// Copies the contents of the 'source' Value to 'this' Value.
        /// The shapes of the 'source' Value's data and mask must be identical to 'this' Value's data and mask.
        ///
        virtual void CopyFrom(const Value& source);

    private:
        // Disallow copy and move construction and assignment
        Value(const Value&) = delete; Value& operator=(const Value&) = delete; Value(Value&&) = delete; Value& operator=(Value&&) = delete;

    protected:
        mutable NDArrayViewPtr m_data;
        mutable NDMaskPtr m_mask;
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
        CNTK_API static const std::wstring StaticAxisNamePrefix;

        CNTK_API static const int SentinelStaticAxisIndexValueForDynamicAxes;
        static const int SentinelStaticAxisIndexValueForAllStaticAxes;
        static const int SentinelStaticAxisIndexValueForUnknownAxes;

        class UniqueDynamicAxesNames
        {
        public:
            bool RegisterAxisName(const std::wstring& axisName)
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                return m_allKnownDynamicAxisNames.insert(axisName).second;
            }

            const std::wstring& NewUniqueDynamicAxisName(const std::wstring& axisNamePrefix)
            {
                std::unique_lock<std::mutex> lock(m_mutex);
                if (m_allKnownDynamicAxisNames.find(axisNamePrefix) == m_allKnownDynamicAxisNames.end())
                {
                    m_allKnownDynamicAxisNames.insert(axisNamePrefix);
                    return axisNamePrefix;
                }

                for (size_t i = 1;; i++)
                {
                    auto newDynamicAxisName = axisNamePrefix + std::to_wstring(i);
                    if (m_allKnownDynamicAxisNames.find(newDynamicAxisName) == m_allKnownDynamicAxisNames.end())
                    {
                        m_allKnownDynamicAxisNames.insert(newDynamicAxisName);
                        return *m_allKnownDynamicAxisNames.find(newDynamicAxisName);
                    }
                }
            }

        private:
            std::unordered_set<std::wstring> m_allKnownDynamicAxisNames;
            std::mutex m_mutex;
        };

        CNTK_API static UniqueDynamicAxesNames s_uniqueDynamicAxisNames;

    public:
        CNTK_API static const std::vector<Axis> DefaultInputVariableDynamicAxes;

        ///
        /// Axis object representing unknown dynamic axes
        ///
        CNTK_API static const std::vector<Axis> UnknownDynamicAxes;

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
        bool IsStaticAxis() const { return m_staticAxisIdx != SentinelStaticAxisIndexValueForDynamicAxes; }

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
        /// Axis object representing the batch axis.
        ///
        CNTK_API static const Axis& DefaultBatchAxis();

        ///
        /// Axis object representing all the static axes of an operand
        ///
        CNTK_API static const Axis& AllStaticAxes();

        ///
        /// Returns a new unique Dynamic axis
        ///
        static Axis NewUniqueDynamicAxis(const std::wstring& axisNamePrefix, bool isOrderedDynamicAxis = true)
        {
            return Axis(s_uniqueDynamicAxisNames.NewUniqueDynamicAxisName(axisNamePrefix), isOrderedDynamicAxis);
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
        if (first.IsStaticAxis() != second.IsStaticAxis())
            return false;

        if (first.IsStaticAxis())
            return first.StaticAxisIndex() == second.StaticAxisIndex();
        else
            return first.Name() == second.Name();
    }

    inline bool operator!=(const Axis& first, const Axis& second)
    {
        return !(first == second);
    }
}

namespace std {
    template <> struct hash<CNTK::Axis>
    {
        size_t operator()(const CNTK::Axis& x) const
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
                LogicError("Unknown DictionaryValue::Type");
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
        DictionaryValue(const std::vector<CNTK::DictionaryValue>& value) : m_valueType(GetValueType<std::vector<CNTK::DictionaryValue>>())
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
            else if (m_valueType == Type::Dictionary)
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

        const size_t version = 1;
    };

    ///
    /// A type denoting a dictionary (keyed by Unicode strings) of serializable values (dynamically typed).
    ///
    class Dictionary final
    {
        friend inline void AddConfigString(std::wstringstream& s, const DictionaryValue& value, size_t numIndentationSpaces);
        friend class CompositeMinibatchSource;
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

        CNTK_API DictionaryValue operator[](const wchar_t* key) const;

        DictionaryValue operator[](const std::wstring& key) const
        {
            return operator[](key.c_str());
        }

        CNTK_API bool Contains(const wchar_t* key) const;

        bool Contains(const std::wstring& key) const
        {
            return Contains(key.c_str());
        }

        CNTK_API void Add(const Dictionary& other);

        CNTK_API bool operator==(const Dictionary& other) const;
        CNTK_API bool operator!=(const Dictionary& other) const;

        friend CNTK_API std::istream& operator>>(std::istream& stream, Dictionary& us);
        friend CNTK_API std::ostream& operator<<(std::ostream& stream, const Dictionary& us);

    private:
        std::shared_ptr<std::unordered_map<std::wstring, DictionaryValue>> m_dictionaryData;
        const size_t version = 1;
    };

    ///
    /// Enumeration type denoting the kind of a symbolic Variable object
    ///
    enum class VariableKind
    {
        Input,
        Output,
        Parameter,
        Constant,
        Placeholder
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
            LogicError("Unknown VariableKind");
        }
    }

    namespace Internal
    {
        inline std::wstring GenerateUid(VariableKind varKind)
        {
            return std::wstring(VariableKindName(varKind)) + std::to_wstring(Internal::NewUniqueId());
        }
    }

    typedef Dictionary ParameterInitializer;

    // Forward declarations
    inline Variable PlaceholderVariable(const NDShape& shape, const std::wstring& name, const std::vector<Axis>& dynamicAxes = Axis::UnknownDynamicAxes);
    inline Variable InputVariable(const NDShape& shape, bool isSparse, CNTK::DataType dataType, bool needsGradient, const std::wstring& name, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes);
    inline Variable OutputVariable(const NDShape& shape, CNTK::DataType dataType, Function* ownerFunction, const std::vector<Axis>& dynamicAxes, const std::wstring& name = L"");

    ///
    /// Denotes a symbolic entity corresponding to the inputs and outputs of a Function.
    /// A Variable is symbolic and does not represent the actual values.
    /// Also, Variable type is a value type and copies of a Variable object are aliases of the
    /// source Variable object itself and have the same identity.
    ///
    class Variable
    {
        friend bool operator==(const Variable& first, const Variable& second);
        friend class Function;
        friend class CompositeFunction;
        friend class PrimitiveFunction;

        template <typename T>
        friend struct std::hash;

        template <typename ElementType>
        friend Variable GetVariable(const Microsoft::MSR::CNTK::ComputationNodeBasePtr& node,
                                    std::unordered_map<Microsoft::MSR::CNTK::ComputationNodeBasePtr, Variable>& nodeToVariableMap,
                                    std::unordered_map<Variable, Variable>& placeholderReplacements,
                                    std::unordered_set<FunctionPtr>& allPrimitiveFunctions);

#ifndef SWIG
    private:
        friend inline Variable PlaceholderVariable(const NDShape& shape, const std::wstring& name, const std::vector<Axis>& dynamicAxes);
        friend inline Variable InputVariable(const NDShape& shape, bool isSparse, CNTK::DataType dataType, bool needsGradient, const std::wstring& name, const std::vector<Axis>& dynamicAxes /*= Axis::DefaultInputVariableDynamicAxes*/);
        friend inline Variable OutputVariable(const NDShape& shape, CNTK::DataType dataType, Function* ownerFunction, const std::vector<Axis>& dynamicAxes, const std::wstring& name /*= L""*/);
#endif

    public:

        ///
        /// Create an 'Output' variable aliasing the output of the specified Function
        /// Throws an exception if called for a Function instance with multiple outputs
        ///
        CNTK_API Variable(const FunctionPtr& function);

        ///
        /// Implicit conversion to a FunctionPtr; creates a pass through primitive function
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
        const NDShape& Shape() const { return m_dataFields->m_shape; }

        ///
        /// Returns the dynamic axes of 'this' variable
        ///
        const std::vector<Axis>& DynamicAxes() const { return m_dataFields->m_dynamicAxes; }

        ///
        /// Returns the VariableKind of 'this' variable
        ///
        VariableKind Kind() const { return m_dataFields->m_varKind; }

        ///
        /// Returns a boolean value indicating if 'this' variable denotes sparse data
        ///
        bool IsSparse() const { return m_dataFields->m_isSparse; }

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
        const std::wstring& Name() const { return m_dataFields->m_name; }

        ///
        /// Returns the internally generated unique name of the variable
        ///
        const std::wstring& Uid() const { return m_dataFields->m_uid; }

        ///
        /// Returns the Function object which 'this' variable is an ouptut of.
        /// Returns null when called for a Variable that is not of 'Output' VariableKind.
        ///
        CNTK_API FunctionPtr Owner() const;

        ///
        /// Returns the DataType of the data that 'this' Variable symbolically represents
        ///
        DataType GetDataType() const { return m_dataFields->m_dataType; }

        ///
        /// Returns a boolean value indicating if gradient computation is enabled for this variable.
        ///
        bool NeedsGradient() const { return m_dataFields->m_needsGradient; }

    protected:
#ifdef SWIG
    public:
#endif
        Variable(const NDShape& shape, VariableKind varType, CNTK::DataType dataType, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, const std::wstring& name, const std::wstring& uid)
            : Variable(shape, varType, dataType, nullptr, value, needsGradient, dynamicAxes, /*isSparse =*/ false, name, uid)
        {}

    protected:
        CNTK_API NDArrayViewPtr Value() const;

    private:
#ifdef SWIG
    public:
#endif
        Variable(const NDShape& shape, bool isSparse, CNTK::DataType dataType, bool needsGradient, const std::wstring& name, const std::vector<Axis>& dynamicAxes, const std::wstring& uid)
            : Variable(shape, VariableKind::Input, dataType, nullptr, nullptr, needsGradient, dynamicAxes, isSparse, name, uid)
        {}

        // TODO: This should be a private but if not made public, the python bindings build complains about an unresolved external
        // Probably due the above ctor being a public method in SWIG codegen
    public:
        CNTK_API Variable(const NDShape& shape, VariableKind varType, CNTK::DataType dataType, Function* ownerFunction, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, const std::wstring& name, const std::wstring& uid);

private:
        Variable Clone() const
        {
            Variable clonedVariable;
            clonedVariable.m_dataFields = m_dataFields->Clone();

            return clonedVariable;
        }

        template <typename ElementType>
        static NDArrayViewPtr CreateValueFromParameterInitializer(const NDShape& shape, const ParameterInitializer& initConfig, const DeviceDescriptor& device);

    private:

        struct VariableFields final : public std::enable_shared_from_this<VariableFields>
        {
            friend class CompositeFunction;

            NDShape m_shape;
            VariableKind m_varKind;
            CNTK::DataType m_dataType;
            Function* const m_ownerFunction; // Variable does not keep the Function alive
            NDArrayViewPtr m_value;
            std::unique_ptr<ParameterInitializer> m_valueInitializer;
            std::unique_ptr<DeviceDescriptor> m_valueInitializationDevice;
            bool m_needsGradient;
            std::wstring m_name;
            std::vector<Axis> m_dynamicAxes;
            bool m_isSparse;
            std::wstring m_uid;

            VariableFields(const NDShape& shape, VariableKind varType, CNTK::DataType type, Function* ownerFunction, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, const std::wstring& name, const std::wstring& uid)
                : m_shape(shape), m_varKind(varType), m_dataType(type), m_ownerFunction(ownerFunction), m_value(value), m_needsGradient(needsGradient), m_dynamicAxes(dynamicAxes), m_isSparse(isSparse), m_name(name), m_uid(uid)
            {
                if (value && (type != value->GetDataType()))
                    InvalidArgument("The DataType of the Parameter/Constant Variable does not match the DataType of the associated Value");

                // Validate that each of the dynamic axes are unique
                std::unordered_set<Axis> uniqueDynamicAxis;
                for (auto& currentDynamicAxis : dynamicAxes)
                {
                    auto retVal = uniqueDynamicAxis.insert(currentDynamicAxis);
                    if (!retVal.second)
                        InvalidArgument("Dynamic axis named %S is specified more than once for Variable object", currentDynamicAxis.Name().c_str());
                }
            }

            std::shared_ptr<VariableFields> Clone() const
            {
                if (m_ownerFunction != nullptr)
                    InvalidArgument("Output variables cannot be cloned");

                return MakeSharedObject<VariableFields>(m_shape,
                                                        m_varKind,
                                                        m_dataType,
                                                        m_ownerFunction,
                                                        (m_value) ? m_value->DeepClone() : nullptr,
                                                        m_needsGradient,
                                                        m_dynamicAxes,
                                                        m_isSparse,
                                                        m_name,
                                                        Internal::GenerateUid(m_varKind));
            }

            CNTK_API void SetValueInitialization(const ParameterInitializer& initializationConfig, const DeviceDescriptor& device);

        private:
            // Disallow copy and move construction and assignment
            VariableFields(const VariableFields&) = delete; VariableFields& operator=(const VariableFields& other) = delete; VariableFields(VariableFields&&) = delete; VariableFields& operator=(VariableFields&&) = delete;
        };
        typedef std::shared_ptr<VariableFields> VariableFieldsPtr;

    protected:
        VariableFieldsPtr m_dataFields;
    };

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
    inline Variable PlaceholderVariable(const NDShape& shape, const std::wstring& name, const std::vector<Axis>& dynamicAxes)
    {
        auto varKind = VariableKind::Placeholder;
        return Variable(shape, varKind, DataType::Unknown, nullptr, false, dynamicAxes, name, Internal::GenerateUid(varKind));
    }

    ///
    /// Create a Placeholder variable to be used as a temporary/placeholder input to a Function.
    /// All placeholder inputs of a Function must be replaced with non-placeholder Variables before Forward evaluation of the Function.
    ///
    inline Variable PlaceholderVariable(const NDShape& shape, const std::vector<Axis>& dynamicAxes = Axis::UnknownDynamicAxes)
    {
        return PlaceholderVariable(shape, L"", dynamicAxes);
    }

    ///
    /// Create a Placeholder variable to be used as a temporary/placeholder input to a Function.
    /// All placeholder inputs of a Function must be replaced with non-placeholder Variables before Forward evaluation of the Function.
    ///
    inline Variable PlaceholderVariable(const std::wstring& name = L"")
    {
        return PlaceholderVariable(NDShape::Unknown, name, Axis::UnknownDynamicAxes);
    }

    ///
    /// Create an 'Input' Variable denoting sparse data and specify if gradients are to be computed for this input
    ///
    inline Variable InputVariable(const NDShape& shape, bool isSparse, CNTK::DataType dataType, bool needsGradient, const std::wstring& name /*= L""*/, const std::vector<Axis>& dynamicAxes /*= Axis::DefaultInputVariableDynamicAxes*/)
    {
        return Variable(shape, isSparse, dataType, needsGradient, name, dynamicAxes, Internal::GenerateUid(VariableKind::Input));
    }

    ///
    /// Create an 'Input' Variable and specify if gradients are to be computed for this input
    ///
    inline Variable InputVariable(const NDShape& shape, CNTK::DataType dataType, bool needsGradient, const std::wstring& name = L"", const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes)
    {
        return InputVariable(shape, /*isSparse =*/ false, dataType, needsGradient, name, dynamicAxes);
    }

    ///
    /// Create an 'Input' Variable.
    ///
    inline Variable InputVariable(const NDShape& shape, DataType dataType, const std::wstring& name, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes)
    {
        return InputVariable(shape, dataType, /*needsGradient =*/ false, name, dynamicAxes);
    }

    ///
    /// Create an 'Input' Variable.
    ///
    inline Variable InputVariable(const NDShape& shape, DataType dataType, const wchar_t* name, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes)
    {
        return InputVariable(shape, dataType, std::wstring(name), dynamicAxes);
    }

    ///
    /// Create an 'Input' Variable.
    ///
    inline Variable InputVariable(const NDShape& shape, DataType dataType, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes)
    {
        return InputVariable(shape, dataType, L"", dynamicAxes);
    }

    ///
    /// Create an 'Input' Variable denoting sparse data.
    ///
    inline Variable InputVariable(const NDShape& shape, bool isSparse, CNTK::DataType dataType, const std::wstring& name, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes)
    {
        return InputVariable(shape, isSparse, dataType, /*needsGradient =*/ false, name, dynamicAxes);
    }

    ///
    /// Create an 'Input' Variable denoting sparse data.
    ///
    inline Variable InputVariable(const NDShape& shape, bool isSparse, CNTK::DataType dataType, const wchar_t* name, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes)
    {
        return InputVariable(shape, isSparse, dataType, std::wstring(name), dynamicAxes);
    }

    ///
    /// Create an 'Input' Variable denoting sparse data.
    ///
    inline Variable InputVariable(const NDShape& shape, bool isSparse, CNTK::DataType dataType, const std::vector<Axis>& dynamicAxes = Axis::DefaultInputVariableDynamicAxes)
    {
        return InputVariable(shape, isSparse, dataType, L"", dynamicAxes);
    }

    ///
    /// Create an 'Output' variable
    ///
    inline Variable OutputVariable(const NDShape& shape, CNTK::DataType dataType, Function* ownerFunction, const std::vector<Axis>& dynamicAxes, const std::wstring& name /*= L""*/)
    {
        return Variable(shape, VariableKind::Output, dataType, ownerFunction, nullptr, /*needsGradient =*/ false, dynamicAxes, /*isSparse =*/ false, name, Internal::GenerateUid(VariableKind::Output));
    }

    static const int DefaultParamInitScale = 1;
    static const int DefaultParamInitOutputRank = 1;
    static const int DefaultParamInitFilterRank = 0;

    CNTK_API ParameterInitializer ConstantInitializer(double value = 0.0);
    CNTK_API ParameterInitializer UniformInitializer(double scale = DefaultParamInitScale, unsigned long seed = DefaultRandomSeed);
    CNTK_API ParameterInitializer GaussianInitializer(int outputRank = DefaultParamInitOutputRank, int filterRank = DefaultParamInitFilterRank, double scale = DefaultParamInitScale, unsigned long seed = DefaultRandomSeed);
    CNTK_API ParameterInitializer XavierInitializer(int outputRank = DefaultParamInitOutputRank, int filterRank = DefaultParamInitFilterRank, double scale = DefaultParamInitScale, unsigned long seed = DefaultRandomSeed);
    CNTK_API ParameterInitializer GlorotUniformInitializer(int outputRank = DefaultParamInitOutputRank, int filterRank = DefaultParamInitFilterRank, double scale = DefaultParamInitScale, unsigned long seed = DefaultRandomSeed);
    CNTK_API ParameterInitializer GlorotNormalInitializer(int outputRank = DefaultParamInitOutputRank, int filterRank = DefaultParamInitFilterRank, double scale = DefaultParamInitScale, unsigned long seed = DefaultRandomSeed);
    CNTK_API ParameterInitializer HeUniformInitializer(int outputRank = DefaultParamInitOutputRank, int filterRank = DefaultParamInitFilterRank, double scale = DefaultParamInitScale, unsigned long seed = DefaultRandomSeed);
    CNTK_API ParameterInitializer HeNormalInitializer(int outputRank = DefaultParamInitOutputRank, int filterRank = DefaultParamInitFilterRank, double scale = DefaultParamInitScale, unsigned long seed = DefaultRandomSeed);
    CNTK_API ParameterInitializer BilinearInitializer(size_t kernelWidth, size_t kernelHeight);

    ///
    /// Denotes Parameter inputs of a Function.
    ///
    class Parameter final : public Variable
    {
        template <typename T>
        friend struct std::hash;

        template <typename ElementType>
        friend Variable GetVariable(const Microsoft::MSR::CNTK::ComputationNodeBasePtr& node,
                                    std::unordered_map<Microsoft::MSR::CNTK::ComputationNodeBasePtr, Variable>& nodeToVariableMap,
                                    std::unordered_map<Variable, Variable>& placeholderReplacements,
                                    std::unordered_set<FunctionPtr>& allPrimitiveFunctions);

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
            : Variable(shape, VariableKind::Parameter, AsDataType<ElemType>(), MakeSharedObject<NDArrayView>(initValue, shape, device), true, {}, name, Internal::GenerateUid(VariableKind::Parameter))
        {}

        ///
        /// Construct a constant of specified shape whose contents are initialized with the specified 'initValue'
        ///
        Parameter(const NDShape& shape, DataType dataType, double initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"")
            : Variable(shape, VariableKind::Parameter, dataType, MakeSharedObject<NDArrayView>(initValue, dataType, shape, device), true, {}, name, Internal::GenerateUid(VariableKind::Parameter))
        {}

        ///
        /// Construct a constant of specified shape whose contents are initialized using the specified initializer
        ///
        Parameter(const NDShape& shape, DataType dataType, const ParameterInitializer& initializer, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"")
            : Variable(shape, VariableKind::Parameter, dataType, nullptr, true, {}, name, Internal::GenerateUid(VariableKind::Parameter))
        {
            m_dataFields->SetValueInitialization(initializer, device);
        }

        ///
        /// DownCast a Variable to a Parameter. Only allowed if the VariableKind is Parameter and throws an exception otherwise.
        ///
        explicit Parameter(const Variable& variable)
            : Variable(variable)
        {
            if (!IsParameter())
                InvalidArgument("A non-parameter Variable being converted to a Parameter");
        }

        ///
        /// Get the value of 'this' parameter
        ///
        NDArrayViewPtr Value() const
        {
            return Variable::Value();
        }

    private:
        explicit Parameter(const NDArrayViewPtr& value, const std::wstring& name, const std::wstring& uid)
            : Variable(value->Shape(), VariableKind::Parameter, value->GetDataType(), value->DeepClone(), true, {}, name, uid)
        {}
    };

    // Implementation note: The Variable type is a value type and not polymorphic in nature. 
    // However we have a couple of derivatives of the type to extend the base interface and thus we ensure that the derived types do not have additional fields.
    // This check is weak in that the derives types may sneak in some additional fields if the base type had some padding at the end, without changing the object size
    // but it should be good enough for catching any accidental additon of fields.
    static_assert(sizeof(Parameter) == sizeof(Variable), "The Parameter type should not have any data fields beyond what it's base type 'Variable' has.");

    ///
    /// Denotes Constant inputs of a Function.
    ///
    class Constant final : public Variable
    {
        template <typename T>
        friend struct std::hash;

        template <typename ElementType>
        friend Variable GetVariable(const Microsoft::MSR::CNTK::ComputationNodeBasePtr& node,
                                    std::unordered_map<Microsoft::MSR::CNTK::ComputationNodeBasePtr, Variable>& nodeToVariableMap,
                                    std::unordered_map<Variable, Variable>& placeholderReplacements,
                                    std::unordered_set<FunctionPtr>& allPrimitiveFunctions);

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
            : Variable(shape, VariableKind::Constant, AsDataType<ElemType>(), MakeSharedObject<NDArrayView>(initValue, shape, device), false, {}, name, Internal::GenerateUid(VariableKind::Constant))
        {}

        ///
        /// Construct a constant of specified shape whose contents are initialized with the specified 'initValue'
        ///
        Constant(const NDShape& shape, DataType dataType, double initValue, const DeviceDescriptor& device = DeviceDescriptor::UseDefaultDevice(), const std::wstring& name = L"")
            : Variable(shape, VariableKind::Constant, dataType, MakeSharedObject<NDArrayView>(initValue, dataType, shape, device), false, {}, name, Internal::GenerateUid(VariableKind::Constant))
        {}

        ///
        /// Create a scalar constant. The specified value is cast to the specified DataType
        ///
        static inline CNTK::Constant Scalar(CNTK::DataType dataType, double value, const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::CPUDevice())
        {
            return Constant({}, dataType, value, device);
        }

        ///
        /// Create a scalar constant. The specified value is cast to the specified DataType
        ///
        template<typename ElementType>
        static inline CNTK::Constant Scalar(ElementType value, const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::CPUDevice())
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
                InvalidArgument("A non-constant Variable being converted to a Constant");
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
            : Variable(value->Shape(), VariableKind::Constant, value->GetDataType(), value->DeepClone(true), false, {}, name, uid)
        {}
    };

    // Implementation note: The Variable type is a value type and not polymorphic in nature. 
    // However we have a couple of derivatives of the type to extend the base interface and thus we ensure that the derived types do not have additional fields.
    // This check is weak in that the derives types may sneak in some additional fields if the base type had some padding at the end, without changing the object size
    // but it should be good enough for catching any accidental additon of fields.
    static_assert(sizeof(Constant) == sizeof(Variable), "The Constant type should not have any data fields beyond what it's base type 'Variable' has.");
}

namespace std {
    
    template <> struct hash<CNTK::NDShape>
    {
        size_t operator()(const CNTK::NDShape& x) const
        {
            return std::hash<std::wstring>()(x.AsString());
        }
    };

    
    template <> struct hash<CNTK::Variable>
    {
        size_t operator()(const CNTK::Variable& x) const
        {
            return std::hash<const void*>()(x.m_dataFields.get());
        }
    };

    template <> struct hash<CNTK::Parameter>
    {
        size_t operator()(const CNTK::Parameter& x) const
        {
            return std::hash<CNTK::Variable>()(x);
        }
    };

    template <> struct hash<CNTK::Constant>
    {
        size_t operator()(const CNTK::Constant& x) const
        {
            return std::hash<CNTK::Variable>()(x);
        }
    };
}

namespace CNTK
{
    ///
    /// Encapsulates the internal computation state of a Function computed as part of the 'Forward' call on a Function
    /// that must be passed to a subsequent 'Backward' call on the same Function to backpropagate gradient values
    /// for the same computation backwards through the Function
    ///
    class BackPropState : public std::enable_shared_from_this<BackPropState>
    {
    public:
        ///
        /// Returns the Function that 'this' BackPropState belongs to
        ///
        FunctionPtr Function() const { return m_function; }
        virtual ~BackPropState() {}

    protected:
        BackPropState(const FunctionPtr& function) : m_function(function) {}

    protected:
        FunctionPtr m_function;
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
        /// New learnable Parameters are created and initialied with the current values of the
        /// corresponding Parameters of the Function being cloned
        ///
        Clone,

        ///
        /// Parameters are cloned and made immutable; i.e. Constants in the new clone 
        /// (e.g. for use as a fixed feature extractor)
        ///
        Freeze,
    };

    ///
    /// Represents a function (optionally differentiable w.r.t. its inputs)
    /// A Function denotes a symbolic computation with zero or more input arguments and one or more outputs. 
    /// A Function may be primitive or composite (comprised of other function instances whose inputs and outputs are wired together).
    /// A Function effectively is a computation graph composed of other primitive Functions (denoting computation) as nodes and Variable objects
    /// (denoting data) as the edges and leaves of the graph.
    ///
    class Function : public std::enable_shared_from_this<Function>
    {
        friend class CompositeFunction;
        friend class Trainer;

    public:
        ///
        /// Computes and stores the values of speficied variables in the 'outputs' map, using provided 'inputs' values corresponding
        /// to each leaf variable of the function of VariableKind 'Input'.
        /// The variables specified in the 'outputs' map denote the subset of 'this' Function's output variables that the caller wants to obtain values of. 
        /// Callers may specify the storage to be used for storing the 'outputs' Values or pass null in which case the implementation allocates the actual storage
        /// for the 'outputs' for which the ValuePtr mapping was left null by the caller.
        /// The optional 'outputsToRetainBackwardStateFor' parameter specifies the subset of the Function's output variables for which gradients will be specified
        /// in a subsequent Backward call for backpropagation.
        /// The method returns a BackPropState object containing all intermediate variable values needed during backpropagation of gradients from the 
        /// 'outputsToRetainBackwardStateFor' outputs of the function to any of the inputs of the Function, in a subsequent Backward call.
        /// Note that the returned BackPropState instance also stores a reference to the supplied 'inputs' Values and generated 'outputs' Values
        /// and the user is responsible for ensuring that the contents of the inputs and outputs are unchanged until after any uses of the BackPropState instance
        /// for backpropagating gradients through this function.
        ///
        virtual BackPropStatePtr Forward(const std::unordered_map<Variable, ValuePtr>& arguments,
                                         std::unordered_map<Variable, ValuePtr>& outputs,
                                         const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice(),
                                         const std::unordered_set<Variable>& outputsToRetainBackwardStateFor = {}) = 0;

        ///
        /// Backpropagates supplied 'rootGradientValues' for one or more of the output variables of the Function, to produce gradient Values
        /// corresponding to the specified set of input variables in 'backPropagatedGradientValuesForInputs'.
        /// Callers may specify the actual storage to be used for storing the 'backPropagatedGradientValuesForInputs' Values or leave them to be null
        /// in which case the implementation allocates the actual storage for storing the gradients.
        /// In case an existing storage is specified, the gradients are aggregated with existing values in the specified storage.
        /// The 'state' parameter is an instance of an BackPropState instance obtained from a previous call to the Forward method on 'this; Function for the 
        /// computation that this gradient backpropagation corresponds to.
        ///
        virtual void Backward(const BackPropStatePtr& state,
                              const std::unordered_map<Variable, ValuePtr>& rootGradientValues,
                              std::unordered_map<Variable, ValuePtr>& backPropagatedGradientValuesForInputs) = 0;

        ///
        /// Returns the name of the operation that this Function denotes
        ///
        virtual const std::wstring& OpName() = 0;

    public:

        // Optional overrides

        ///
        /// Destruct this Function.
        ///
        virtual ~Function() {}

        ///
        /// Clones 'this' Function. The parameters of the Function are either cloned, shared or frozen as specified by the parameterCloneMethod argument and
        /// any variable replacements requested are applied in the cloned Function instance.
        ///
        CNTK_API FunctionPtr Clone(ParameterCloningMethod parameterCloneMethod = ParameterCloningMethod::Clone, const std::unordered_map<Variable, Variable>& replacements = {}) const;

    public:
        ///
        /// Returns the name of 'this' variable.
        ///
        const std::wstring& Name() const { return m_name; }

        ///
        /// Returns the primitive Function at the root of the graph of Functions underlying this Function.
        /// If 'this' Function itself is a primitive function then (this->RootFunction() == this).
        ///
        FunctionPtr RootFunction() const
        {
            return (m_rootFunction == nullptr) ? const_cast<Function*>(this)->shared_from_this() : m_rootFunction;
        }

        ///
        /// Returns all Input variables of 'this' Function.
        ///
        std::vector<Variable> Inputs() const
        {
            return *(InputsImpl().get());
        }

        ///
        /// Returns the Output variable of 'this' Function. Throws an exception of 'this' Function has more that one output.
        ///
        Variable Output() const
        {
            if (m_outputs.size() > 1)
                RuntimeError("A Function instance with more than one output cannot be implicitly converted to a Variable");

            return m_outputs[0];
        }

        ///
        /// Returns a vector consisting of all Output variables of 'this' Function.
        ///
        const std::vector<Variable>& Outputs() const { return m_outputs; }

        ///
        /// Returns a set comprising of all input variables of 'this' Function's variables that are not of kind 'Parameter' or 'Constant'.
        ///
        std::vector<Variable> Arguments() const
        {
            return FilteredInputs<Variable>([](const Variable& var) {
                return (var.IsInput() || var.IsOutput());
            });
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
        /// Returns the set of all Constant variables of 'this' Function.
        ///
        std::vector<Variable> Placeholders() const
        {
            return FilteredInputs<Variable>([](const Variable& var) {
                return var.IsPlaceholder();
            });
        }

        ///
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
        /// Restore the models parameters from a saved model file
        ///
        CNTK_API void RestoreFromLegacyModel(const std::wstring& modelFilePath);

    private:

        template <typename VariableType, typename FilterFunction>
        std::vector<VariableType> FilteredInputs(FilterFunction&& filterFunc) const
        {
            std::vector<VariableType> filteredInputs;
            std::unordered_set<Variable> uniqueFilteredInputs;
            auto inputs = Inputs();
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

        CNTK_API std::shared_ptr<std::vector<Variable>> InputsImpl() const;

        void ValidateOrUpdateOutputs(std::unordered_map<const Function*, size_t>& visitedFunctions);

        virtual void ReplacePlaceholdersInPlace(const std::unordered_map<Variable, Variable>& placeholderReplacements,
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

    protected:
        ///
        /// Protected constructor for derived 'Function' types to specify the actual input and output variables for the Function instance.
        ///
        Function(const std::vector<Variable>& inputs, const std::vector<Variable>& outputs, Dictionary&& functionConfig, const FunctionPtr& rootFunction = nullptr, const std::wstring& name = L"")
            : m_rootFunction(rootFunction), m_name(name), m_attributes(std::move(functionConfig))
        {
            for (auto inputVar : inputs)
            {
                m_inputs.push_back(inputVar);

                if (!inputVar.IsInput() &&
                    !inputVar.IsOutput() &&
                    !inputVar.IsParameter() &&
                    !inputVar.IsConstant() &&
                    !inputVar.IsPlaceholder())
                {
                    InvalidArgument("Function input has invalid VariableKind!");
                }
            }

            std::unordered_set<Variable> uniqueOutputs;
            for (auto outputVar : outputs)
            {
                if (uniqueOutputs.find(outputVar) != uniqueOutputs.end())
                    RuntimeError("Same variable appears multiple times in the outputs vector passed to Function constructor");

                m_outputs.push_back(outputVar);
                uniqueOutputs.insert(outputVar);
            }
        }

    private:

        std::vector<Variable> m_inputs;
        std::vector<Variable> m_outputs;

        FunctionPtr m_rootFunction; // nullptr for primitive function instances
        std::wstring m_name;
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
    CNTK_API FunctionPtr Slice(const Variable& operand, const Axis& axis, int beginIndex, int endIndex, const std::wstring& name = L"");

    ///
    /// Create an instance of the dropout operation on specified tensor input operand
    ///
    // TODO: The initial random seed should be specifiable
    CNTK_API FunctionPtr Dropout(const Variable& operand, double dropoutRate, const std::wstring& name = L"");

    ///
    /// Create an instance of the reshape operation on specified tensor input operand
    ///
    CNTK_API FunctionPtr Reshape(const Variable& operand, const NDShape& newShape, const std::wstring& name = L"");

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
    CNTK_API FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank = 1, int inferInputRankToMap = -1, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in matrix multiplication operation with the transpose of the left input operand
    /// and the specified right operand. Only accepts left operands of ranks 1 or 2.
    /// TODO: Specify the constraints on the shapes of the operands.
    ///
    CNTK_API FunctionPtr TransposeTimes(const Variable& leftOperand, const Variable& rightOperand, size_t outputRank = 1, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation to compute squared-error for specified input operands.
    ///
    CNTK_API FunctionPtr SquaredError(const Variable& prediction, const Variable& targets, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation to compute cross-entropy with softmax for specified input operands.
    ///
    CNTK_API FunctionPtr CrossEntropyWithSoftmax(const Variable& prediction, const Variable& labels, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation for computing the classification prediction error for specified operands.
    ///
    CNTK_API FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, size_t topN, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation for computing the classification prediction error for specified operands.
    ///
    inline FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, const std::wstring& name = L"")
    {
        return ClassificationError(prediction, labels, /*topN =*/ 1, name);
    }

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
    /// Per dimension mean-variance normalization of the specified input operand.
    ///
    CNTK_API FunctionPtr PerDimMeanVarianceNormalize(const Variable& operand, const NDArrayViewPtr& mean, const NDArrayViewPtr& invStdDev, const std::wstring& name = L"");

    ///
    /// TODO:
    ///
    CNTK_API FunctionPtr Convolution(const Variable& convolutionMap,
                                     const Variable& operand,
                                     const NDShape& strides = {1},
                                     const std::vector<bool>& sharing = {true},
                                     const std::vector<bool>& autoPadding = {true},
                                     const NDShape& lowerPad = {0},
                                     const NDShape& upperPad = {0},
                                     bool transpose = false,
                                     size_t maxTempMemSizeInSamples = 0,
                                     const std::wstring& name = L"");

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
                                 const NDShape& lowerPad = {0},
                                 const NDShape& upperPad = {0},
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
                                            bool spatial,
                                            double normalizationTimeConstant = 0,
                                            double blendTimeConstant = 0,
                                            double epsilon = 0.00001,
                                            bool useCuDNNEngine = false,
                                            const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise clip operation on the tensor operand
    ///
    CNTK_API FunctionPtr Clip(const Variable& operand, const Variable& min, const Variable& max, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise choice operation using a condition tensor for specified tensor operands.
    ///
    CNTK_API FunctionPtr ElementSelect(const Variable& condition, const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

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

    namespace Sequence
    {
        CNTK_API FunctionPtr IsFirst(const Variable& operand, const std::wstring& name = L"");
        CNTK_API FunctionPtr IsLast(const Variable& operand, const std::wstring& name = L"");

        CNTK_API FunctionPtr First(const Variable& operand, const std::wstring& name = L"");
        CNTK_API FunctionPtr Last(const Variable& operand, const std::wstring& name = L"");

        CNTK_API FunctionPtr Where(const Variable& condition, const std::wstring& name = L"");
        CNTK_API FunctionPtr Gather(const Variable& operand, const Variable& condition, const std::wstring& name = L"");
        CNTK_API FunctionPtr Scatter(const Variable& operand, const Variable& condition, const std::wstring& name = L"");

        CNTK_API FunctionPtr BroadcastAs(const Variable& operand, const Variable& broadcastAs, const std::wstring& name = L"");
    }

    ///
    /// Load a legacy CNTK v1 format model
    ///
    CNTK_API FunctionPtr LoadLegacyModel(DataType dataType, const std::wstring& modelFile, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

    /// 
    /// Save a Composite Function instance to a file in CNTK legacy model format
    ///
    CNTK_API void SaveAsLegacyModel(const FunctionPtr& rootFunction, const std::wstring& modelFile);

    ///
    /// Abstraction for learning a subset of parameters of a learnable function using first order gradient values
    /// For e.g momentum, AdaGrad, RMSProp etc. are different types of learners with their own algorithms for
    /// learning parameter values using first order gradients.
    ///
    class Learner : public std::enable_shared_from_this<Learner>
    {
        static const std::wstring LearningRateAttributeName;

    public:
        //
        // Method to update the parameters associated with this learner. By returning false, this method indicates that
        // learning has stopped for all of the parameters associated with this learner
        //
        virtual bool Update(const std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, size_t trainingSampleCount) = 0;

        ///
        /// Returns the set of parameters associated with this learner.
        ///
        const std::unordered_set<Parameter>& Parameters() const { return m_parameters; }

        ///
        /// Optionally overridable method to checkpoint the learner's state.
        ///
        // TODO: move the following two methods into ISerializable interface, make 
        // Learner (and all other entities that need checkpointing capability) implement it.
        virtual Dictionary GetCheckpointState() const 
        {
            Dictionary baseCheckpointState;
            baseCheckpointState[LearningRateAttributeName] = m_learningRate;

            return baseCheckpointState;
        }

        ///
        /// Optionally overridable method to restore the learner's state from a previous checkpoint.
        ///
        virtual void RestoreFromCheckpoint(const Dictionary& checkpoint) 
        {
            if (checkpoint.Contains(LearningRateAttributeName))
                m_learningRate = checkpoint[LearningRateAttributeName].Value<double>();
        }

        ///
        /// Destruct this Learner.
        ///
        virtual ~Learner() {}

        virtual void ResetLearningRate(double learningRate) { m_learningRate = learningRate; }
        virtual double LearningRate() const { return m_learningRate; }

    protected:
        Learner(const std::vector<Parameter>& parameters, double learningRate)
            : m_parameters(parameters.begin(), parameters.end()), m_learningRate(learningRate)
        {}

        std::unordered_set<Parameter> m_parameters;
        double m_learningRate;
    };

    ///
    /// A collection of key-value pairs that represents a training parameter schedule in 
    /// terms of the number of processed samples (e.g., learning rate and momentum schedules). 
    /// This class is designed to simplify Learner's factory methods and provides a number of 
    /// convenience constructors to allow easy conversion from a single value, a vector of values 
    /// and a list of pairs to the training schedule. For example, a learning rate schedule 
    /// { { 10, 0.5 }, { 100, 0.3 }, { 20, 0.2 } } indicates that the rate of 0.5 should be
    /// used for the first 10 units (equivalently, samples if the default unit = 1 is used)
    /// followed by 0.3 for the next 100 units, and then 0.2 for the remaining 20 units or 
    /// until the end of training if it takes longer.
    ///
    template <typename T>
    class TrainingParameterSchedule
    {
    public:
        ///
        /// Create a schedule with a constant parameter value.
        ///
        CNTK_API TrainingParameterSchedule(T value);

        ///
        /// Create a schedule where the parameter changes its value every 'unit' samples:
        /// schedule[0] is used for the first 'unit' samples, schedule[1] -- for the second,
        /// and so on. The last value is then used repeatedly until the end of training.
        ///
        CNTK_API TrainingParameterSchedule(const std::vector<T>& schedule, size_t unit = 1);

        ///
        /// Create a schedule using the list of key-value pairs, where the key specifies 
        /// the number of 'units' the parameter should maintain the corresponding value.
        /// The value from the last pair is used repeatedly until the end of training.
        /// For example, {{1, 0.05}, {2, 0.1}, {1, 0.005}} and unit = 100, corresponds to 
        /// a schedule where the value of '0.05' is used for the first 100 samples, then
        /// '0.1' is used for the second 200 samples, after which the values is switched
        /// to '0.005'.
        ///
        CNTK_API TrainingParameterSchedule(const std::vector<std::pair<size_t, T>>& schedule, size_t unit = 1);

        ///
        /// Returns a value corresponding to the absolute sample count from the beginning of training.
        ///
        CNTK_API virtual const T& operator[](size_t sampleCount) const;

        CNTK_API virtual ~TrainingParameterSchedule();

        CNTK_API TrainingParameterSchedule(const TrainingParameterSchedule<T>&); 
        CNTK_API TrainingParameterSchedule(TrainingParameterSchedule<T>&&); 
        CNTK_API TrainingParameterSchedule<T>& operator=(const TrainingParameterSchedule<T>&); 
        CNTK_API TrainingParameterSchedule<T>& operator=(TrainingParameterSchedule<T>&&);

    private:
        CNTK_API void ConstructSchedule(const std::vector<std::pair<size_t, T>>& schedule);

    protected:           
        std::map<size_t, T> m_schedule;
        size_t m_unit;
    };

    typedef TrainingParameterSchedule<double> LearningRatesPerSample;
    typedef TrainingParameterSchedule<double> MomentumValuesPerSample;

    ///
    /// This class allows to specify momentum as time constant in place of momentum per sample in 
    /// all of Learners factory methods. The specified values are then automatically converted into 
    /// per sample values.
    /// 
    class MomentumValuesAsTimeConstants: public MomentumValuesPerSample
    {
    public:
        MomentumValuesAsTimeConstants(double value) 
            : MomentumValuesPerSample(value)
        { 
            ConvertToPerSampleValues();
        }
        
        MomentumValuesAsTimeConstants(const std::vector<double>& schedule, size_t unit = 1) 
            : MomentumValuesPerSample(schedule, unit)
        { 
            ConvertToPerSampleValues();
        }
        
        MomentumValuesAsTimeConstants(const std::vector<std::pair<size_t, double>>& schedule, size_t unit = 1) 
            : MomentumValuesPerSample(schedule, unit)
        { 
            ConvertToPerSampleValues();
        }

    private:
        CNTK_API void ConvertToPerSampleValues();
    };

    /// A collection of additional options that affect parameter updates and 
    /// are applicable for all standard learners 
    struct AdditionalLearningOptions
    {
        double l1RegularizationWeight = 0.0;
        double l2RegularizationWeight = 0.0;
        double gaussianNoiseInjectionStdDev = 0.0;
        double gradientClippingThresholdPerSample = std::numeric_limits<double>::infinity();
        bool gradientClippingWithTruncation = true;
    };

    ///
    /// Create an instance of the CNTK built-in SGD learner.
    ///
    CNTK_API LearnerPtr SGDLearner(const std::vector<Parameter>& parameters,
                                   const LearningRatesPerSample& learningRates,
                                   AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

    ///
    /// Create an instance of the CNTK built-in Momentum SGD learner.
    ///
    CNTK_API LearnerPtr MomentumSGDLearner(const std::vector<Parameter>& parameters,
                                           const LearningRatesPerSample& learningRates,
                                           const MomentumValuesPerSample& momentumValues,
                                           AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

    ///
    /// Create an instance of the CNTK built-in Nesterov's accelerated SGD learner.
    ///
    CNTK_API LearnerPtr NesterovLearner(const std::vector<Parameter>& parameters,
                                        const LearningRatesPerSample& learningRates,
                                        const MomentumValuesPerSample& momentumValues,
                                        AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

    ///
    /// Create an instance of the CNTK built-in FSAdaGrad (improved AdaGrad) learner.
    ///
    CNTK_API LearnerPtr FSAdaGradLearner(const std::vector<Parameter>& parameters,
                                         const LearningRatesPerSample& learningRates,
                                         const MomentumValuesPerSample& momentumValues,
                                         const double targetAdagradAvDenom = 0.0025, // 1/400 magic constant 
                                         const size_t adagradT = 2 * 3600 * 100,
                                         AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

    ///
    /// Create an instance of the CNTK built-in AdaGrad learner.
    ///
    CNTK_API LearnerPtr AdaGradLearner(const std::vector<Parameter>& parameters,
                                       const LearningRatesPerSample& learningRates,
                                       bool needAveMultiplier = true,
                                       AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

    ///
    /// Create an instance of the CNTK built-in RMSProp learner.
    ///
    CNTK_API LearnerPtr RMSPropLearner(const std::vector<Parameter>& parameters,
                                       const LearningRatesPerSample& learningRates,
                                       double gamma,
                                       double inc,
                                       double dec,
                                       double max,
                                       double min,
                                       bool needAveMultiplier = true,
                                       AdditionalLearningOptions additionalOptions = AdditionalLearningOptions());

    ///
    /// Trainer is the top-level abstraction responsible for the orchestration of the training of a model
    /// using the specified learners and training data either explicitly supplied as Value objects or from
    /// a MinibatchSource object.
    ///
    class Trainer
    {
    public:
        ///
        /// Construct a Trainer to train the specified 'model' with the specified 'trainingLoss' Variable as the training criterion
        /// and using the specified set of 'parameterLearners' for updating the model's parameters using computed gradients.
        ///
        CNTK_API Trainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const std::unordered_set<LearnerPtr>& parameterLearners);

        ///
        /// Construct a Trainer to train the specified 'model' with the specified 'trainingLoss' as the training criterion,
        /// the specified 'evaluationFunction' as the criterion for evaluating the trained model's quality, and using the specified set
        /// of 'parameterLearners' for updating the model's parameters using computed gradients.
        ///
        // TODO: Add overload for multiple evaluation criterion
        CNTK_API Trainer(const FunctionPtr& model, const FunctionPtr& lossFunction, const FunctionPtr& evaluationFunction, const std::unordered_set<LearnerPtr>& parameterLearners);

        ///
        /// Optimize model parameters using the specified 'arguments' minibatch of training samples.
        /// Returns false if all parameter learners indicate end of learning (through their Update method's return value).
        ///
        CNTK_API bool TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Optimize model parameters using the specified 'arguments' minibatch of training samples.
        /// The variables specified in the 'outputsToFetch' map denote the subset of 'this' Function's output variables that the caller wants to obtain values of. 
        /// Callers may specify the storage to be used for storing the 'outputs' Values or pass null in which case the implementation allocates the actual storage
        /// for the 'outputs' for which the ValuePtr mapping was left null by the caller.
        /// Returns false if all parameter learners indicate end of learning (through their Update method's return value).
        ///
        CNTK_API bool TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, std::unordered_map<Variable, ValuePtr>& outputsToFetch, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Test the model on the specified batch of samples using the evaluation Function specified during construction of the Trainer
        /// Returns the average evaluation criterion value per sample for the tested minibatch of samples
        ///
        CNTK_API double TestMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice = DeviceDescriptor::UseDefaultDevice());

        ///
        /// Checkpoint the model and other Trainer state at the specified file location
        ///
        CNTK_API void SaveCheckpoint(const std::wstring& modelFilePath);

        ///
        /// Restore the model and trainer state from a previously saved model and checkpoint from the specified file location
        ///
        CNTK_API void RestoreFromCheckpoint(const std::wstring& modelFilePath);

        ///
        /// Model being trained by 'this' Trainer.
        ///
        FunctionPtr Model() const { return m_model; }

        ///
        /// Loss function that is used as the optimization criterion for learning the model's parameters.
        ///
        FunctionPtr LossFunction() const { return m_lossFunction; }

        ///
        /// Evaluation Function that is used as for the criterion for evaluating the trained model's quality.
        ///
        FunctionPtr EvaluationFunction() const { return m_evaluationFunction; }

        ///
        /// Returns the average training loss per sample for the last minibatch trained.
        ///
        CNTK_API double PreviousMinibatchLossAverage() const;

        ///
        /// Returns the average evaluation criterion value per sample for the last minibatch trained.
        ///
        CNTK_API double PreviousMinibatchEvaluationAverage() const;

        ///
        /// Returns the number of samples in the last minibatch trained with
        ///
        size_t PreviousMinibatchSampleCount() const { return m_prevMinibatchNumSamples; }

        ///
        /// Learners associated with this Trainer for updating the model's parameters using computed gradients.
        ///
        const std::unordered_set<LearnerPtr>& ParameterLearners() const { return m_parameterLearners; }

    private:
        FunctionPtr m_combinedTrainingFunction;
        FunctionPtr m_model;
        FunctionPtr m_lossFunction;
        FunctionPtr m_aggregatedLossFunction;
        FunctionPtr m_evaluationFunction;
        FunctionPtr m_aggregatedEvaluationFunction;
        Variable    m_trainingSampleCountVar;
        Variable    m_testSampleCountVar;

        std::unordered_set<LearnerPtr> m_parameterLearners;

        size_t m_prevMinibatchNumSamples;
        ValuePtr m_prevMinibatchAggregateTrainingLossValue;
        ValuePtr m_prevMinibatchAggregateEvalCriterionValue;
    };

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
    };

    inline bool operator==(const StreamInformation& left, const StreamInformation& right)
    {
        return ((left.m_id == right.m_id) &&
                (left.m_name == right.m_name) &&
                (left.m_storageFormat == right.m_storageFormat) &&
                (left.m_elementType == right.m_elementType) &&
                (left.m_sampleLayout == right.m_sampleLayout));
    }
}

namespace std {
    template <> struct hash<CNTK::StreamInformation>
    {
        size_t operator()(const CNTK::StreamInformation& x) const
        {
            return std::hash<size_t>()(x.m_id);
        }
    };
}

namespace CNTK
{
    struct MinibatchData
    {
        size_t m_numSequences;
        size_t m_numSamples;
        ValuePtr m_data;
    };

    ///
    /// Abstraction for generating minibatches of samples for training/evaluation.
    ///
    class MinibatchSource : public std::enable_shared_from_this<MinibatchSource>
    {
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
        virtual const std::unordered_map<StreamInformation, MinibatchData>& GetNextMinibatch(size_t minibatchSizeInSamples,
                                                                                             size_t minibatchSizeInSequences,
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

    ///
    /// Instantiate the CNTK built-in composite minibatch source.
    ///
    CNTK_API MinibatchSourcePtr CreateCompositeMinibatchSource(const Dictionary& configuration);

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

    /// 
    /// Instantiate the CNTK built-in test format minibatch source
    ///
    inline MinibatchSourcePtr TextFormatMinibatchSource(const std::wstring& dataFilePath, const std::vector<StreamConfiguration>& streamConfigs, size_t epochSize = SIZE_MAX, bool randomize = true)
    {
        CNTK::Dictionary minibatchSourceConfiguration;
        minibatchSourceConfiguration[L"epochSize"] = epochSize;

        if (randomize)
            minibatchSourceConfiguration[L"randomize"] = true;

        CNTK::Dictionary deserializerConfiguration;
        deserializerConfiguration[L"type"] = L"CNTKTextFormatDeserializer";
        deserializerConfiguration[L"file"] = dataFilePath;

        CNTK::Dictionary inputStreamsConfig;
        for (auto streamConfig : streamConfigs)
        {
            std::wstring streamName = streamConfig.m_streamName;
            size_t streamDim = streamConfig.m_dim;
            bool isSparse = streamConfig.m_isSparse;
            std::wstring streamAlias = streamConfig.m_streamAlias;

            CNTK::Dictionary inputStreamConfig;
            inputStreamConfig[L"dim"] = streamDim;
            inputStreamConfig[L"format"] = isSparse ? L"sparse" : L"dense";
            if (!streamAlias.empty())
                inputStreamConfig[L"alias"] = streamAlias;

            inputStreamsConfig[streamName] = inputStreamConfig;
        }

        deserializerConfiguration[L"input"] = inputStreamsConfig;
        minibatchSourceConfiguration[L"deserializers"] = std::vector<CNTK::DictionaryValue>({ deserializerConfiguration });

        return CreateCompositeMinibatchSource(minibatchSourceConfiguration);
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
}
