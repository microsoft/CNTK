//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// This is the main header of the CNTK library API containing the entire public API definition. 
//

#pragma once

#include "CNTKLibraryInternals.h"

#include <memory>
#include <vector>
#include <array>
#include <stdarg.h>
#include <assert.h>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <sstream>
#include<algorithm>

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
    public:
        ///
        /// Returns the Id of 'this' device.
        ///
        int Id() const { return m_deviceId; }

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
        static DeviceDescriptor GPUDevice(unsigned int deviceId) { return{ deviceId, DeviceKind::GPU }; }

        ///
        /// Static method to get the descriptor of the default device for the current process.
        /// This device is used for all CNTK operations where a device needs to be specified and one is not explicitly specified.
        ///
        CNTK_API static DeviceDescriptor DefaultDevice();

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
    public:

        ///
        /// A placeholder value to use for an axis whose dimension is unknown and is to be inferred by the system.
        ///
        static const size_t InferredDimension = (size_t)-1;

    public:
        ///
        /// Construct a NDShape with 0 axes, which denotes a scalar.
        ///
        NDShape() {}

        ///
        /// Contruct a NDShape instance with the specified number of axes and dimensionality in each axis.
        ///
        explicit NDShape(size_t numAxes, size_t dimension = InferredDimension)
            : m_shapeDims(numAxes, dimension)
        {}

        ///
        /// Contruct a NDShape instance with specified dimensions.
        ///
        NDShape(const std::vector<size_t>& dimensions)
            : m_shapeDims(dimensions)
        {}

        ///
        /// Contruct a NDShape instance with specified dimensions.
        ///
        NDShape(const std::initializer_list<size_t>& dimensions)
            : m_shapeDims(dimensions)
        {}

        ///
        /// Returns the dimensions of 'this' shape as a std::vector<size_t>
        ///
        const std::vector<size_t>& Dimensions() const { return m_shapeDims; }

        ///
        /// Returns the number of axes of 'this' shape.
        ///
        size_t NumAxes() const { return m_shapeDims.size(); }

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
            endAxisId = (endAxisId == SIZE_MAX) ? NumAxes() : endAxisId;
            if ((endAxisId < beginAxisId) || (endAxisId > NumAxes()))
                InvalidArgument("NDShape::SubShape : The specified endAxisId (%d) cannot exceed the number of axes (%d) of 'this' NDShape and must be >= than the specified beginAxisId (%d)", (int)endAxisId, (int)NumAxes(), (int)beginAxisId);

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
        /// Creates and returns a new shape contructed by appending the dimensions of the specified 'shape' to 'this' shape's dimensions.
        ///
        NDShape AppendShape(const NDShape& shape) const
        {
            std::vector<size_t> newShapeDims(NumAxes() + shape.NumAxes());
            std::copy(m_shapeDims.begin(), m_shapeDims.end(), newShapeDims.begin());
            std::copy(shape.m_shapeDims.begin(), shape.m_shapeDims.end(), newShapeDims.begin() + m_shapeDims.size());

            return newShapeDims;
        }

        ///
        /// Create a string representation of 'this' NDShape for display/printing purposes
        ///
        std::wstring AsString() const
        {
            std::wstringstream wStrStream(L"{");
            for (size_t i = 0; i < NumAxes(); i++)
            {
                if (i != 0)
                    wStrStream << L", ";

                wStrStream << m_shapeDims[i];
            }

            wStrStream << L"}";
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
        explicit NDArrayView(const ElementType& value, const NDShape& viewShape = { 1 }, const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice(), bool readOnly = false)
            : NDArrayView(AsDataType<ElementType>(), viewShape, device)
        {
            SetValue(value);
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
        /// Creates a new NDArrayView with newly allocated storage on the same device as 'this' view and copies 'this' view's contents into the newly allocated view.
        ///
        CNTK_API NDArrayViewPtr DeepClone(bool readOnly = false) const;

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
        CNTK_API static NDArrayViewPtr RandomNormal(const NDShape& shape, double mean, double stdDev, unsigned long seed = 1, const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice());

        ///
        /// Static method to construct a new NDArrayView object whose contents are drawn from a uniform distribution in the specified value range.
        ///
        template <typename ElementType>
        CNTK_API static NDArrayViewPtr RandomUniform(const NDShape& shape, double rangeStart, double rangeEnd, unsigned long seed = 1, const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice());

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
        CNTK_API explicit NDMask(const NDShape& shape, const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice());

        ///
        /// Destruct 'this' NDMask object
        ///
        CNTK_API ~NDMask();

        ///
        /// Mask out the specified sub-section of 'this' mask
        ///
        CNTK_API void MaskSection(const std::vector<size_t>& sectionOffset, const NDShape& sectionShape);

        ///
        /// Clear the mask; i.e. unmask all currently masked values
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
        CNTK_API const char* DataBuffer() const;

        ///
        /// Creates a new NDMask with newly allocated storage on the same device as 'this' mask and copies 'this' mask's contents into the newly allocated mask.
        ///
        CNTK_API NDMaskPtr DeepClone() const;

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
    /// sample shape is data.Shape().SubShape(0, data.Shape().NumAxes() - mask.Shape().NumAxes)
    /// Also, note that the size of the data's trailing mask.Shape().NumAxes() dimensions must match the mask shape dimensions.
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
        CNTK_API virtual ~Value();

        ///
        /// Returns the NDArrayView object corresponding to the data contents of 'this value object.
        ///
        CNTK_API virtual NDArrayViewPtr Data() const;

        ///
        /// Returns the NDMask object corresponding to the mask associated with 'this value object.
        ///
        CNTK_API virtual NDMaskPtr Mask() const;

        ///
        /// Creates a new Value with newly allocated storage on the same device as 'this' Value and copies 'this' Value's contents into the newly allocated Value.
        ///
        CNTK_API virtual ValuePtr DeepClone(bool readOnly = false) const;

        ///
        /// Creates a new Value which is an alias of 'this' Value.
        ///
        CNTK_API virtual ValuePtr Alias(bool readOnly = false) const;

        ///
        /// Copies the contents of the 'source' Value to 'this' Value.
        /// The shapes of the 'source' Value's data and mask must be identical to 'this' Value's data and mask.
        ///
        CNTK_API virtual void CopyFrom(const Value& source);

    private:
        // Disallow copy and move construction and assignment
        Value(const Value&) = delete; Value& operator=(const Value&) = delete; Value(Value&&) = delete; Value& operator=(Value&&) = delete;

    private:
        NDArrayViewPtr m_data;
        NDMaskPtr m_mask;
    };

    ///
    /// Denotes an Axis of a Variable and is used for specifying the axes parameters of certain Functions such as reductions.
    /// Besides the static axes corresponding to each of the axes of the Variable's shape, Input and Output Variables
    /// also have one or more dynamic axes (corresponding to the sequence dimensions) and one implicit batch axis denoting the axes 
    /// along which multiple sequences are batched in the Values corresponding to the variable when performing computations.
    ///
    class Axis final
    {
    public:
        ///
        /// Construct an Axis object denoting a static axis with the specified index.
        ///
        Axis(size_t staticAxisIdx)
            : m_staticAxisIdx(staticAxisIdx)
        {
            const wchar_t* staticAxisNamePrefix = L"staticAxis_";
            m_name = staticAxisNamePrefix + std::to_wstring(staticAxisIdx);
        }

        ///
        /// Construct a dynamic axis with the specified name.
        ///
        Axis(const std::wstring& name)
            : m_staticAxisIdx(SIZE_MAX), m_name(name)
        {
        }

        ///
        /// Returns a boolean indicating if 'this' Axis corresponds to a static axis
        ///
        bool IsStaticAxis() const { return m_staticAxisIdx == SIZE_MAX; }

        ///
        /// Returns the axis index if 'this' Axis is a static axis. Throws an exception otherwise.
        ///
        size_t StaticAxisIndex() const
        {
            if (!IsStaticAxis())
                InvalidArgument("Cannot query the static axis index for a non-static axis");

            return m_staticAxisIdx;
        }

        ///
        /// Static Axis object representing the default dynamic axis.
        ///
        CNTK_API static const Axis& DefaultDynamicAxis();

        ///
        /// Static Axis object representing the batch axis.
        ///
        CNTK_API static const Axis& BatchAxis();

        ///
        /// Special Axis object denoting all the axes of the Value object in whose context it is used.
        ///
        CNTK_API static const Axis& AllAxes();

        ///
        /// Name of 'this' axis
        ///
        const std::wstring& Name() const { return m_name; }

        ///
        /// Default constructor; results in an invalid axis object.
        ///
        Axis()
            : m_staticAxisIdx(SIZE_MAX)
        {}

    private:
        size_t m_staticAxisIdx;
        std::wstring m_name;
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

        template <typename T>
        friend struct std::hash;

    public:
        ///
        /// Create an 'Input' Variable.
        ///
        Variable(const NDShape& shape, CNTK::DataType dataType, const wchar_t* name = L"")
            : Variable(shape, dataType, std::wstring(name))
        {}

        ///
        /// Create an 'Input' Variable.
        ///
        Variable(const NDShape& shape, CNTK::DataType dataType, const std::wstring& name = L"")
            : Variable(shape, VariableKind::Input, dataType, nullptr, nullptr, false, { Axis::DefaultDynamicAxis() }, false, name)
        {}

        ///
        /// Create an 'Input' Variable denoting sparse data.
        ///
        Variable(const NDShape& shape, bool isSparse, CNTK::DataType dataType, const std::wstring& name = L"")
            : Variable(shape, VariableKind::Input, dataType, nullptr, nullptr, false, { Axis::DefaultDynamicAxis() }, isSparse, name)
        {}

        ///
        /// Create an 'Input' Variable and specify if gradients are to be computed for this input
        ///
        Variable(const NDShape& shape, CNTK::DataType dataType, bool needsGradient, const std::wstring& name = L"")
            : Variable(shape, VariableKind::Input, dataType, nullptr, nullptr, needsGradient, { Axis::DefaultDynamicAxis() }, false, name)
        {}

        ///
        /// Create an 'Input' Variable denoting sparse data and specify if gradients are to be computed for this input
        ///
        Variable(const NDShape& shape, bool isSparse, CNTK::DataType dataType, bool needsGradient, const std::wstring& name = L"")
            : Variable(shape, VariableKind::Input, dataType, nullptr, nullptr, needsGradient, { Axis::DefaultDynamicAxis() }, isSparse, name)
        {}

        ///
        /// Create an 'Output' variable
        ///
        Variable(const NDShape& shape, CNTK::DataType dataType, Function* ownerFunction, const std::vector<Axis>& dynamicAxes, const std::wstring& name = L"")
            : Variable(shape, VariableKind::Output, dataType, ownerFunction, nullptr, false, dynamicAxes, false, name)
        {}

        ///
        /// Create an 'Output' variable aliasing the output of the specified Function
        /// Throws an exception if called for a Function instance with multiple outputs
        ///
        CNTK_API Variable(const FunctionPtr& function);

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
        bool IsSparse() const { return (m_dataFields->m_isSparse); }

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
        Variable(const NDShape& shape, VariableKind varType, CNTK::DataType dataType, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, const std::wstring& name)
            : Variable(shape, varType, dataType, nullptr, value, needsGradient, dynamicAxes, false, name)
        {}

        NDArrayViewPtr Value() const
        {
            assert(m_dataFields->m_value != nullptr);
            return m_dataFields->m_value;
        }

    private:
        Variable(const NDShape& shape, VariableKind varType, CNTK::DataType dataType, Function* ownerFunction, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, const std::wstring& name)
            : m_dataFields(MakeSharedObject<VariableFields>(shape, varType, dataType, ownerFunction, value, needsGradient, dynamicAxes, isSparse, name))
        {}

    private:

        struct VariableFields final : public std::enable_shared_from_this<VariableFields>
        {
            NDShape m_shape;
            VariableKind m_varKind;
            CNTK::DataType m_dataType;
            Function* m_ownerFunction; // Variable does not keep the Function alive
            NDArrayViewPtr m_value;
            bool m_needsGradient;
            std::wstring m_name;
            std::vector<Axis> m_dynamicAxes;
            bool m_isSparse;

            VariableFields(const NDShape& shape, VariableKind varType, CNTK::DataType type, Function* ownerFunction, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, const std::wstring& name)
                : m_shape(shape), m_varKind(varType), m_dataType(type), m_ownerFunction(ownerFunction), m_value(value), m_needsGradient(needsGradient), m_dynamicAxes(dynamicAxes), m_isSparse(isSparse), m_name(name)
            {
            }

        private:
            // Disallow copy and move construction and assignment
            VariableFields(const VariableFields&) = delete; VariableFields& operator=(const VariableFields& other) = delete; VariableFields(VariableFields&&) = delete; VariableFields& operator=(VariableFields&&) = delete;
        };
        typedef std::shared_ptr<VariableFields> VariableFieldsPtr;

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
    /// Denotes Parameter inputs of a Function.
    ///
    class Parameter final : public Variable
    {
        template <typename T>
        friend struct std::hash;

    public:
        ///
        /// Construct a parameter whose initial contents are a copy of the specified 'value'
        ///
        explicit Parameter(const NDArrayViewPtr& value, const std::wstring& name = L"")
            : Variable(value->Shape(), VariableKind::Parameter, value->GetDataType(), value->DeepClone(), true, {}, name)
        {}

        // TODO: Constructor to move a specified NDArrayView value

        ///
        /// Construct a parameter of specified shape whose contents are initialized with the specified 'initValue'
        ///
        template<typename ElemType>
        Parameter(const NDShape& shape, ElemType initValue, const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice(), const std::wstring& name = L"")
            : Variable(shape, VariableKind::Parameter, AsDataType<ElemType>(), MakeSharedObject<NDArrayView>(initValue, shape, device), true, {}, name)
        {}

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

    public:
        ///
        /// Contruct a Constant whose initial contents are a copy of the specified value
        ///
        Constant(const NDArrayViewPtr& value, const std::wstring& name = L"")
            : Variable(value->Shape(), VariableKind::Constant, value->GetDataType(), value->DeepClone(true), false, {}, name)
        {}

        // TODO: Constructor to move a specified NDArrayView value

        ///
        /// Construct a constant of specified shape whose contents are initialized with the specified 'initValue'
        ///
        template<typename ElemType>
        Constant(const NDShape& shape, ElemType initValue, const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice(), const std::wstring& name = L"")
            : Variable(shape, VariableKind::Constant, AsDataType<ElemType>(), MakeSharedObject<NDArrayView>(initValue, shape, device), false, {}, name)
        {}

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
    };

    // Implementation note: The Variable type is a value type and not polymorphic in nature. 
    // However we have a couple of derivatives of the type to extend the base interface and thus we ensure that the derived types do not have additional fields.
    // This check is weak in that the derives types may sneak in some additional fields if the base type had some padding at the end, without changing the object size
    // but it should be good enough for catching any accidental additon of fields.
    static_assert(sizeof(Constant) == sizeof(Variable), "The Constant type should not have any data fields beyond what it's base type 'Variable' has.");

    ///
    /// Denotes a Placeholder input to a Function.
    /// All placeholder inputs of a Function must be replaced with non-placeholder Variables before Forward evaluation of the Function.
    ///
    class Placeholder final : public Variable
    {
        template <typename T>
        friend struct std::hash;
        
        friend class Function;

    public:
        ///
        /// Contruct a Placeholder with the specified NDShape
        ///
        explicit Placeholder(const NDShape& shape, const std::wstring& name = L"")
            : Variable(shape, VariableKind::Placeholder, DataType::Unknown, nullptr, false, {Axis::DefaultDynamicAxis()}, name)
        {}

        ///
        /// DownCast a Variable to a Placeholder. Only allowed if the VariableKind is Placeholder and throws an exception otherwise.
        ///
        explicit Placeholder(const Variable& variable)
            : Variable(variable)
        {
            if (!IsPlaceholder())
                InvalidArgument("A non-placeholder Variable being converted to a Placeholder");
        }
    };

    static_assert(sizeof(Placeholder) == sizeof(Variable), "The Placeholder type should not have any data fields beyond what it's base type 'Variable' has.");
}

namespace std {
    template <> struct hash<CNTK::Axis>
    {
        size_t operator()(const CNTK::Axis& x) const
        {
            return std::hash<std::wstring>()(x.Name());
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

    template <> struct hash<CNTK::Placeholder>
    {
        size_t operator()(const CNTK::Placeholder& x) const
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
    /// Represents a function (optionally differentiable w.r.t. its inputs)
    /// A Function denotes a symbolic computation with zero or more input arguments and one or more outputs. 
    /// A Function may be primitive or composite (comprised of other function instances whose inputs and outputs are wired together).
    /// A Function effectively is a computation graph composed of other primitive Functions (denoting computation) as nodes and Variable objects
    /// (denoting data) as the edges and leaves of the graph.
    ///
    class Function : public std::enable_shared_from_this<Function>
    {
        friend class CompositeFunction;

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
        CNTK_API virtual BackPropStatePtr Forward(const std::unordered_map<Variable, ValuePtr>& arguments,
                                                  std::unordered_map<Variable, ValuePtr>& outputs,
                                                  const DeviceDescriptor& computeDevice = DeviceDescriptor::DefaultDevice(),
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
        CNTK_API virtual void Backward(const BackPropStatePtr& state,
            const std::unordered_map<Variable, ValuePtr>& rootGradientValues,
                                       std::unordered_map<Variable, ValuePtr>& backPropagatedGradientValuesForInputs) = 0;

    public:

        // Optional overrides

        ///
        /// Destruct this Function.
        ///
        virtual ~Function() {}

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
        std::unordered_set<Variable> Arguments() const
        {
            return FilteredInputs<Variable>([](const Variable& var) {
                return (var.IsInput() || var.IsOutput());
            });
        }

        ///
        /// Returns the set of all Parameter variables of 'this' Function.
        ///
        std::unordered_set<Parameter> Parameters() const
        {
            return FilteredInputs<Parameter>([](const Variable& var) {
                return var.IsParameter();
            });
        }

        ///
        /// Returns the set of all Constant variables of 'this' Function.
        ///
        std::unordered_set<Constant> Constants() const
        {
            return FilteredInputs<Constant>([](const Variable& var) {
                return var.IsConstant();
            });
        }

        ///
        /// Returns the set of all Constant variables of 'this' Function.
        ///
        std::unordered_set<Placeholder> Placeholders() const
        {
            return FilteredInputs<Placeholder>([](const Variable& var) {
                return var.IsPlaceholder();
            });
        }

        CNTK_API FunctionPtr ReplacePlaceholders(const std::unordered_map<Placeholder, Variable>& placeholderReplacements);

    private:

        template <typename VariableType, typename FilterFunction>
        std::unordered_set<VariableType> FilteredInputs(FilterFunction&& filterFunc) const
        {
            std::unordered_set<VariableType> filteredInputs;
            auto inputs = Inputs();
            for (auto inputVar : inputs)
            {
                if (filterFunc(inputVar))
                    filteredInputs.insert(VariableType(inputVar));
            }

            return filteredInputs;

        }

        CNTK_API std::shared_ptr<std::vector<Variable>> InputsImpl() const;

        virtual void ReplacePlaceholders(const std::unordered_map<Placeholder, Variable>& placeholderReplacements,
                                         std::unordered_set<const Function*>& visitedFunctions,
                                         std::unordered_set<Placeholder>& replacedPlaceholders);

        // Disallow copy and move construction and assignment
        Function(const Function&) = delete; Function(Function&&) = delete; Function& operator=(const Function&) = delete; Function& operator=(Function&&) = delete;

    protected:
        ///
        /// Protected constructor for derived 'Function' types to specify the actual input and output variables for the Function instance.
        ///
        Function(const std::vector<Variable>& inputs, const std::vector<Variable>& outputs, const FunctionPtr& rootFunction = nullptr, const std::wstring& name = L"")
            : m_rootFunction(rootFunction), m_name(name)
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

                switch (outputVar.Kind())
                {
                case VariableKind::Output:
                    m_outputs.push_back(outputVar);
                    uniqueOutputs.insert(outputVar);
                    break;
                default:
                    InvalidArgument("Function output has invalid VariableKind!");
                    break;
                }
            }
        }

    private:

        std::vector<Variable> m_inputs;
        std::vector<Variable> m_outputs;

        FunctionPtr m_rootFunction; // nullptr for primitive function instances
        std::wstring m_name;
    };

    ///
    /// Create an instance of the CNTK built-in elementwise negate operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Negate(const Variable& operand, const std::wstring& name = L"");

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
    /// Create an instance of the CNTK built-in elementwise tensor addition operation with the specified input operands.
    ///
    CNTK_API FunctionPtr Plus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise tensor subtraction operation with the specified input operands.
    ///
    CNTK_API FunctionPtr Minus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");
    
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
    /// Create an instance of the CNTK built-in matrix multiplication operation with the specified input operands.
    /// TODO: Specify the constraints on the shapes of the operands.
    ///
    CNTK_API FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

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
    CNTK_API FunctionPtr ClassificationError(const Variable& prediction, const Variable& labels, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation for getting the past value along the lone dynamic axis of the specified operand.
    /// Throws an exception of the operand has more than one dynamic axis.
    ///
    CNTK_API FunctionPtr PastValue(const Variable& initialState, const Variable& operand, size_t stepSize, const std::wstring& name = L"");

    //CNTK_API FunctionPtr PastValue(const Variable& initialState, const Variable& operand, Axis axis, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation for getting the future value along the lone dynamic axis of the specified operand.
    /// Throws an exception of the operand has more than one dynamic axis.
    ///
    CNTK_API FunctionPtr FutureValue(const Variable& initialState, const Variable& operand, size_t stepSize, const std::wstring& name = L"");


    ///
    /// Create an instance of the CNTK built-in sum reduction operation on specified tensor input operand along all the axes
    ///
    CNTK_API FunctionPtr ReduceSum(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create a new Function instance which just combines the outputs of the specified list of 'operands' Functions such that the 'Outputs' of the 
    /// new 'Function' are union of the 'Outputs' of each of the specified 'operands' Functions.
    /// E.g. When creating a classification model, typically the CrossEntropy loss Function and the ClassificationError Function comprise the two roots
    /// of the computation graph which can be "Combine"d to create a single Function with 2 outputs; viz. CrossEntropy loss and ClassificationError output.
    ///
    CNTK_API FunctionPtr Combine(const std::vector<FunctionPtr>& operands, const std::wstring& name = L"");

    ///
    /// Load a legacy CNTK v1 format model
    ///
    template <typename ElementType>
    CNTK_API FunctionPtr LoadLegacyModel(const std::wstring& modelFile, const DeviceDescriptor& computeDevice = DeviceDescriptor::DefaultDevice());

    /// 
    /// Save a Composite Function instance to a file in CNTK legacy model format
    ///
    template <typename ElementType>
    CNTK_API void SaveAsLegacyModel(const FunctionPtr& rootFunction, const std::wstring& modelFile);

    ///
    /// A serializable value represents one of:
    /// a) Boolean
    /// b) Signed long integer
    /// c) Single and double precision floating point values
    /// d) NDShape
    /// e) vector<DictionaryValue>
    ///
    /// TODO: We need to have native support for DictionaryValue<vector> and DictionaryValue<NDArrayView>.
    class DictionaryValue final
    {
    public:
        enum class Type : unsigned int
        {
            None,
            Bool,
            SizeT,
            Float,
            Double,
            String,
            NDShape,
            Vector,
            Dictionary,
        };

        static const char* TypeName(Type type)
        {
            switch (type)
            {
            case Type::None:
                return "None";
            case Type::Bool:
                return "Bool";
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
            case Type::Vector:
                return "Vector";
            case Type::Dictionary:
                return "Dictionary";
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
        template <typename T>
        DictionaryValue(const T& value) : m_valueType(GetValueType<T>())
        {
            static_assert(std::is_same<T, NDShape>::value ||
                std::is_same<T, std::wstring>::value ||
                std::is_same<T, std::vector<DictionaryValue>>::value ||
                std::is_same<T, Dictionary>::value,
                          "Unsupported ValueType");

            AllocateDataPtr(value);
        }

        DictionaryValue(const DictionaryValue& other) : m_valueType(Type::Bool)
        {
            // The m_valueType must have been set to a non-ptr type to prevent an attempt to interpret
            // the underlying underlying uninitialized value as a ptr and free it.
            *this = other;
        }

        DictionaryValue& operator=(const DictionaryValue& other)
        {
            if (this != &other)
            {
                FreeDataPtr();

                m_valueType = other.m_valueType;
                m_data = other.m_data;

                if (other.m_valueType == Type::String)
                    AllocateDataPtr(other.GetValue<std::wstring>());
                else if (other.m_valueType == Type::NDShape)
                    AllocateDataPtr(other.GetValue<NDShape>());
                else if (other.m_valueType == Type::Vector)
                    AllocateDataPtr(other.GetValue<std::vector<DictionaryValue>>());
                else if (other.m_valueType == Type::Dictionary)
                    AllocateDataPtr(other.GetValue<Dictionary>());
            }

            return *this;
        }

        ~DictionaryValue()
        {
            FreeDataPtr();
        }

        template <typename T, typename std::enable_if<std::is_same<T, bool>::value>::type* = nullptr>
        const T& GetValue() const
        {
            VerifyType<T>();
            return m_data.m_boolean;
        }

        template <typename T, typename std::enable_if<std::is_same<T, size_t>::value>::type* = nullptr>
        const T& GetValue() const
        {
            VerifyType<T>();
            return m_data.m_sizeT;
        }

        template <typename T, typename std::enable_if<std::is_same<T, float>::value>::type* = nullptr>
        const T& GetValue() const
        {
            VerifyType<T>();
            return m_data.m_float;
        }

        template <typename T, typename std::enable_if<std::is_same<T, double>::value>::type* = nullptr>
        const T& GetValue() const
        {
            VerifyType<T>();
            return m_data.m_double;
        }

        template <typename T, typename std::enable_if<std::is_same<T, NDShape>::value ||
            std::is_same<T, std::wstring>::value ||
            std::is_same<T, std::vector<DictionaryValue>>::value ||
            std::is_same<T, Dictionary>::value>::type* = nullptr>
        const T& GetValue() const
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

        friend CNTK_API Microsoft::MSR::CNTK::File& operator>>(Microsoft::MSR::CNTK::File& stream, DictionaryValue& us);
        friend CNTK_API Microsoft::MSR::CNTK::File& operator<<(Microsoft::MSR::CNTK::File& stream, const DictionaryValue& us);

    private:
        template <typename T>
        static Type GetValueType()
        {
            static_assert(std::is_same<T, bool>::value ||
                          std::is_same<T, size_t>::value ||
                          std::is_same<T, float>::value ||
                          std::is_same<T, double>::value ||
                std::is_same<T, std::wstring>::value ||
                          std::is_same<T, NDShape>::value ||
                std::is_same<T, std::vector<DictionaryValue>>::value ||
                std::is_same<T, Dictionary>::value,
                          "Unsupported ValueType");

            if (std::is_same<T, bool>::value)                                      return Type::Bool;
            if (std::is_same<T, size_t>::value)                                    return Type::SizeT;
            if (std::is_same<T, float>::value)                                     return Type::Float;
            if (std::is_same<T, double>::value)                                    return Type::Double;
            if (std::is_same<T, std::wstring>::value)                              return Type::String;
            if (std::is_same<T, NDShape>::value)                                   return Type::NDShape;
            if (std::is_same<T, std::vector<DictionaryValue>>::value)              return Type::Vector;
            if (std::is_same<T, Dictionary>::value)                                return Type::Dictionary;
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
            else if (m_valueType == Type::Vector)
                FreePtrAsType<std::vector<DictionaryValue>>();
            else if (m_valueType == Type::Dictionary)
                FreePtrAsType<Dictionary>();
        }

        Type m_valueType;

        union ValueData
        {
            bool m_boolean;
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


        friend CNTK_API Microsoft::MSR::CNTK::File& operator>>(Microsoft::MSR::CNTK::File& stream, Dictionary& us);
        friend CNTK_API Microsoft::MSR::CNTK::File& operator<<(Microsoft::MSR::CNTK::File& stream, const Dictionary& us);

    private:
        std::shared_ptr<std::unordered_map<std::wstring, DictionaryValue>> m_dictionaryData;
        const size_t version = 1;
    };

    ///
    /// Abstraction for learning a subset of parameters of a learnable function using first order gradient values
    /// For e.g momentum, AdaGrad, RMSProp etc. are different types of learners with their own algorithms for
    /// learning parameter values using first order gradients.
    ///
    class Learner : public std::enable_shared_from_this<Learner>
    {
    public:
        //
        // Method to update the parameters associated with this learner. By returning false, this method indicates that
        // learning has stopped for all of the parameters associated with this learner
        //
        CNTK_API virtual bool Update(const std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, size_t trainingSampleCount) = 0;

        ///
        /// Returns the set of parameters associated with this learner.
        ///
        const std::unordered_set<Parameter>& Parameters() const { return m_parameters; }

        ///
        /// Optionally overridable method to checkpoint the learner's state.
        ///
        // TODO: move the following two methods into ISerializable interface, make 
        // Learner (and all other entities that need checkpointing capability) implement it.
        CNTK_API virtual Dictionary GetCheckpointState() const { return Dictionary(); }

        ///
        /// Optionally overridable method to restore the learner's state from a previous checkpoint.
        ///
        CNTK_API virtual void RestoreFromCheckpoint(const Dictionary& /*checkpoint*/) {}

        virtual ~Learner() {}

    protected:
        Learner(const std::unordered_set<Parameter>& parameters)
            : m_parameters(parameters)
        {}

        std::unordered_set<Parameter> m_parameters;

    };

    ///
    /// Create an instance of the CNTK built-in SGD learner.
    ///
    /// TODO: add additional SGD parameters here (a collection of learning rate values)
    CNTK_API LearnerPtr SGDLearner(const std::unordered_set<Parameter>& parameters, double learningRatePerSample);

    ///
    /// Create an instance of the CNTK built-in Momentum SGD learner.
    ///
    /// TODO: add additional Momentum parameters here (a collection of momentum rate values)
    CNTK_API LearnerPtr MomentumSGDLearner(const std::unordered_set<Parameter>& parameters);

    ///
    /// Create an instance of the CNTK built-in Nesterov's accelerated SGD learner.
    ///
    CNTK_API LearnerPtr NesterovLearner(const std::unordered_set<Parameter>& parameters);

    ///
    /// Create an instance of the CNTK built-in AdaGrad learner.
    ///
    CNTK_API LearnerPtr AdaGradLearner(const std::unordered_set<Parameter>& parameters, bool needAveMultiplier = true);

    ///
    /// Create an instance of the CNTK built-in FSAdaGrad (improved AdaGrad) learner.
    ///
    CNTK_API LearnerPtr FSAdaGradLearner(const std::unordered_set<Parameter>& parameters);

    ///
    /// Create an instance of the CNTK built-in RMSProp learner.
    ///
    CNTK_API LearnerPtr RMSPropLearner(const std::unordered_set<Parameter>& parameters,
                                       double gamma,
                                       double inc,
                                       double dec,
                                       double max,
                                       double min,
                                       bool needAveMultiplier = true);

    ///
    /// Trainer is the top-level abstraction responsible for the orchestration of the training of a model
    /// using the specified learners and training data either explicilty supplied as Value objects or from
    /// a MinibatchSource object.
    ///
    class Trainer
    {
    public:
        ///
        /// Construct a Trainer to train the specified 'model' with the specified 'trainingLoss' Variable as the training criterion
        /// and using the specified set of 'parameterLearners' for updating the model's parameters using computed gradients.
        ///
        CNTK_API Trainer(const FunctionPtr& model, const Variable& trainingLoss, const std::unordered_set<LearnerPtr>& parameterLearners);

        ///
        /// Optimize model parameters using the specified 'arguments' minibatch of training samples.
        /// Returns false if all parameter learners indicate end of learning (through their Update method's return value).
        ///
        CNTK_API bool TrainMinibatch(const std::unordered_map<Variable, ValuePtr>& arguments, const DeviceDescriptor& computeDevice = DeviceDescriptor::DefaultDevice());

        ///
        /// Model being trained by 'this' Trainer.
        ///
        FunctionPtr Model() const { return m_model; }

        ///
        /// Variable of the Trainer's model representing the training loss that is used as the optimization 
        /// criterion for learning the model's parameters.
        ///
        Variable TrainingLossVariable() const { return m_trainingLossVar; }

        ///
        /// Returns the Value of the training loss variable of the model corresponding to the last minibatch trained with
        ///
        ValuePtr PreviousMinibatchTrainingLossValue() const { return m_prevMinibatchTrainingLossValue; }

        ///
        /// Learners associated with this Trainer for updating the model's parameters using computed gradients.
        ///
        const std::unordered_set<LearnerPtr>& ParameterLearners() const { return m_parameterLearners; }

    private:
        FunctionPtr m_model;
        Variable m_trainingLossVar;
        ValuePtr m_prevMinibatchTrainingLossValue;
        std::unordered_set<LearnerPtr> m_parameterLearners;
    };

    ///
    /// Describes an input stream: its name, element type, storage, etc.
    ///
    struct StreamInfo
    {
        std::wstring m_name;           // Unique name of the stream
        size_t m_id;                   // Unique identifier of the stream
        StorageFormat m_storageFormat; // Storage format of the stream
        DataType m_elementType;        // Element type of the stream
        NDShape m_sampleLayout;        // Layout of the sample for the stream
    };

    inline bool operator==(const StreamInfo& left, const StreamInfo& right)
    {
        return (left.m_id == right.m_id);
    }
}

namespace std {
    template <> struct hash<CNTK::StreamInfo>
    {
        size_t operator()(const CNTK::StreamInfo& x) const
        {
            return std::hash<size_t>()(x.m_id);
        }
    };
}

namespace CNTK
{
    ///
    /// Abstraction for generating minbatches of samples for training/evaluation.
    ///
    class MinibatchSource : public std::enable_shared_from_this<MinibatchSource>
    {
    public:
        ///
        /// Describes the streams 'this' MinibatchSource produces.
        ///
        virtual const std::unordered_set<StreamInfo>& StreamInfos() = 0;

        ///
        /// Reads a minibatch that contains data across all input streams.
        /// The minibatchData argument specifies the desired minibatch size for each stream of the reader and the actual returned size is the min across all streams.
        /// The return value of false indciates that the reader will no longer return any further data.
        ///
        virtual bool GetNextMinibatch(std::unordered_map<StreamInfo, std::pair<size_t, ValuePtr>>& minibatchData) = 0;

        // TODO: Methods to save and restore from checkpoints

        // Disallow copy and move construction and assignment
        MinibatchSource(const MinibatchSource&) = delete; MinibatchSource(MinibatchSource&&) = delete; MinibatchSource& operator=(const MinibatchSource&) = delete; MinibatchSource& operator=(MinibatchSource&&) = delete;

    protected:
        MinibatchSource() {}
    };

    ///
    /// Instantiate the CNTK built-in composite minibatch source.
    ///
    CNTK_API MinibatchSourcePtr CreateCompositeMinibatchSource(const Dictionary& configuration);
}
