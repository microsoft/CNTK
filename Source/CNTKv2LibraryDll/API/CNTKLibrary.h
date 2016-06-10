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
    enum class DeviceType
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
        int Id() const
        {
            return m_deviceId;
        }

        ///
        /// Returns the DeviceType of 'this' device.
        ///
        DeviceType Type() const
        {
            return m_deviceType;
        }

        ///
        /// Static method to get the descriptor of the CPU device on the local system.
        ///
        static DeviceDescriptor CPUDevice()
        {
            return{ 0, DeviceType::CPU };
        }

        ///
        /// Static method to get the descriptor of the GPU device on the local system with the specified CUDA device ID.
        ///
        static DeviceDescriptor GPUDevice(unsigned int deviceId)
        {
            return{ deviceId, DeviceType::GPU };
        }

        ///
        /// Static method to get the descriptor of the default device for the current process.
        /// This device is used for all CNTK operations where a device needs to be specified and one is not explicitly specified.
        ///
        CNTK_API static DeviceDescriptor DefaultDevice();

    private:
        DeviceDescriptor(unsigned int deviceId, DeviceType deviceType)
            : m_deviceId(deviceId), m_deviceType(deviceType)
        {
        }

    private:
        unsigned int m_deviceId;
        DeviceType m_deviceType;
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
        /// Contruct a NDShape instance with the specified number of axes and dimensionality in each axis.
        ///
        explicit NDShape(size_t numAxes, size_t dimension = InferredDimension)
            : m_shapeDims(numAxes, dimension)
        {}

        ///
        /// Contruct a NDShape instance with specified dimensions.
        ///
        NDShape(const std::vector<size_t>& dimensions)
            : m_shapeDims(_Internal::_SimpleVector<size_t>::CreateSimpleVector(dimensions))
        {
        }

        ///
        /// Contruct a NDShape instance with specified dimensions.
        ///
        NDShape(const std::initializer_list<size_t>& dimensions)
            : m_shapeDims(_Internal::_SimpleVector<size_t>::CreateSimpleVector(dimensions))
        {}

        ///
        /// Returns the number of axes of 'this' shape.
        ///
        size_t NumAxes() const
        {
            return m_shapeDims.Size();
        }

        ///
        /// Returns a reference to dimension size for the specified axis.
        ///
        size_t& operator[](size_t axisId)
        {
            return m_shapeDims[axisId];
        }

        ///
        /// Returns the dimension size for the specified axis.
        ///
        size_t operator[](size_t axisId) const
        {
            return m_shapeDims[axisId];
        }

        ///
        /// Creates and returns a new NDShape instance with the same dimensions as 'this' shape's specified axis range.
        ///
        NDShape SubShape(size_t startAxisId = 0, size_t endAxisIdExclusive = SIZE_MAX) const
        {
            endAxisIdExclusive = (endAxisIdExclusive == SIZE_MAX) ? NumAxes() : endAxisIdExclusive;
            if ((endAxisIdExclusive < startAxisId) || (endAxisIdExclusive > NumAxes()))
                InvalidArgument("NDShape::SubShape : The specified endAxisId cannot exceed the number of axes of 'this' NDShape and must be >= than the specified startAxisId");

            NDShape subShape(endAxisIdExclusive - startAxisId);
            for (size_t i = 0; i < subShape.NumAxes(); ++i)
                subShape[i] = m_shapeDims[startAxisId + i];

            return subShape;
        }

        ///
        /// Returns a boolean value indicating if the dimension size for any of the axes of 'this' shape is unknown/inferred (aka == NDShape::InferredDimension).
        ///
        bool HasInferredDimension() const
        {
            for (size_t i = 0; i < NumAxes(); ++i)
            {
                if (m_shapeDims[i] == InferredDimension)
                    return true;
            }

            return false;
        }

        ///
        /// Returns the total size of the rectangular shape that 'this' shape denotes.
        ///
        size_t TotalSize() const
        {
            if (HasInferredDimension())
                RuntimeError("NDShape::TotalSize : TotalSize cannot be determined for a NDShape with one or more dimensions being InferredDimension");

            size_t numAxes = NumAxes();
            size_t totalSize = 1;
            for (size_t i = 0; i < numAxes; ++i)
                totalSize *= m_shapeDims[i];

            return totalSize;
        }

        ///
        /// Creates and returns a new shape contructed by appending the dimensions of the specified 'shape' to 'this' shape's dimensions.
        ///
        NDShape AppendShape(const NDShape& shape) const
        {
            NDShape newShape(NumAxes() + shape.NumAxes());

            std::copy(m_shapeDims.Data(), m_shapeDims.Data() + m_shapeDims.Size(), newShape.m_shapeDims.Data());
            std::copy(shape.m_shapeDims.Data(), shape.m_shapeDims.Data() + shape.m_shapeDims.Size(), newShape.m_shapeDims.Data() + m_shapeDims.Size());

            return newShape;
        }

    private:
        _Internal::_SimpleVector<size_t> m_shapeDims;
    };

    inline bool operator==(const NDShape& first, const NDShape& second)
    {
        return first.m_shapeDims == second.m_shapeDims;
    }

    inline bool operator!=(const NDShape& first, const NDShape& second)
    {
        return !(first == second);
    }

#pragma warning(push)
#pragma warning(disable : 4251 4275)

    typedef int SparseIndexType;

    ///
    /// Denotes a multi-dimensional writable or read-only array of elemental values.
    /// This type denotes a view and there may be multiple simultaneous views of the data underlying a NDArrayView instance.
    /// The underlying data is stored in sparse or dense format, and is located on a specific device.
    /// The actual underlying storage is either external or internal in which case its lifetime is managed through reference counting.
    ///
    class CNTK_API NDArrayView final : public _Internal::_ReferenceCounter
    {
        friend class CompositeFunction;
        friend class Learner;

    public:
        ///
        /// Construct a NDArrayView with the specified 'dataBuffer' as the backing storage.
        /// The 'dataBuffer' must have been allocated on the specified 'device', must be at least
        /// as large as the total size of the specified 'viewShape' and must outlive the created NDArrayView object.
        ///
        NDArrayView(CNTK::DataType dataType, const NDShape& viewShape, void* dataBuffer, size_t bufferSizeInBytes, const DeviceDescriptor& device, bool readOnly = false);

        ///
        /// Construct a NDArrayView with newly allocated sparse storage in SparseCSC format on the specified 'device' and initialize its contents
        // with the specified Sparse CSC format data.
        ///
        template <typename ElementType>
        NDArrayView(const NDShape& viewShape, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const ElementType* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly = false);

        ///
        /// Construct a NDArrayView over newly allocated storage in the specified format on the specified 'device'.
        ///
        NDArrayView(CNTK::DataType dataType, CNTK::StorageFormat storageType, const NDShape& viewShape, const DeviceDescriptor& device);

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
        /// Construct a NDArrayView with the buffer underlying the specified std::vector or std::aray being the underlying storage.
        /// The conatiner must be at least as large as the total size of the specified 'viewShape' and should outlive the created NDArrayView object.
        ///
        template <typename ContainerType, typename std::enable_if<std::is_same<ContainerType, std::vector<typename ContainerType::value_type>>::value ||
                                                                  std::is_same<ContainerType, std::array<typename ContainerType::value_type, sizeof(ContainerType) / sizeof(typename ContainerType::value_type)>>::value>::type* = nullptr>
        NDArrayView(const NDShape& viewShape, ContainerType& sourceContainer, bool readOnly = false)
            : NDArrayView(viewShape, sourceContainer.data(), sourceContainer.size(), DeviceDescriptor::CPUDevice(), readOnly)
        {}

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
        /// Destruct 'this' view object
        /// 
        ~NDArrayView();

        ///
        /// Returns a writable pointer to the data buffer underlying 'this' view
        /// Throws an exception if 'this' view is read-only
        /// 
        template <typename ElementType>
        ElementType* WritableDataBuffer();

        ///
        /// Returns a read-only pointer to the data buffer underlying 'this' view
        /// 
        template <typename ElementType>
        const ElementType* DataBuffer() const;

        ///
        /// Returns the descriptor of the device that 'this' view resides on
        ///
        DeviceDescriptor Device() const
        {
            return m_device;
        }

        ///
        /// Returns the data type of 'this' view's contents.
        ///
        DataType GetDataType() const
        {
            return m_dataType;
        }

        ///
        /// Returns the storage format of 'this' view.
        ///
        StorageFormat GetStorageFormat() const
        {
            return m_storageFormat;
        }

        ///
        /// Returns the shape 'this' view.
        ///
        NDShape Shape() const
        {
            return m_viewShape;
        }

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
        bool IsReadOnly() const
        {
            return m_isReadOnly;
        }

        ///
        /// Creates a new NDArrayView with newly allocated storage on the same device as 'this' view and copies 'this' view's contents into the newly allocated view.
        ///
        NDArrayViewPtr DeepClone(bool readOnly = false) const;

        ///
        /// Creates a new NDArrayView which is an alias of 'this' view.
        ///
        NDArrayViewPtr Alias(bool readOnly = false) const;

        ///
        /// Copies the contents of the 'source' NDArrayView to 'this' view.
        /// The shapes of the 'source' view and 'this' view must be identical.
        ///
        void CopyFrom(const NDArrayView& source);

        ///
        /// Static method to construct a new NDArrayView object whose contents are drawn from a normal distribution with the specified mean and standard deviation..
        ///
        template <typename ElementType>
        static NDArrayViewPtr RandomNormal(const NDShape& shape, double mean, double stdDev, unsigned long seed = 1, const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice());

        ///
        /// Static method to construct a new NDArrayView object whose contents are drawn from a uniform distribution in the specified value range.
        ///
        template <typename ElementType>
        static NDArrayViewPtr RandomUniform(const NDShape& shape, double rangeStart, double rangeEnd, unsigned long seed = 1, const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice());

    private:
        // Disallow copy construction and assignment
        NDArrayView(const NDArrayView&) = delete;
        NDArrayView& operator=(const NDArrayView&) = delete;

        // Disallow move construction and assignment
        NDArrayView& operator=(NDArrayView&&) = delete;
        NDArrayView(NDArrayView&& other) = delete;

    private:
        static const size_t AutoSelectRowColSplitPoint = SIZE_MAX;

    private:
        NDArrayView(CNTK::DataType dataType, const DeviceDescriptor& device, CNTK::StorageFormat storageType, const NDShape& viewShape, bool readOnly, void* tensorView);

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

        void SetValue(float value);
        void SetValue(double value);

    private:
        CNTK::DataType m_dataType;
        DeviceDescriptor m_device;
        CNTK::StorageFormat m_storageFormat;
        NDShape m_viewShape;
        bool m_isReadOnly;

        void* m_tensorView;
    };

    ///
    /// Denotes a multi-dimensional mask used for specifying specific sections of a NDArrayView object as masked/invalid.
    /// This type denotes a view and there may be multiple simultaneous views of the data underlying a NDMask instance.
    ///
    class CNTK_API NDMask final : public _Internal::_ReferenceCounter
    {
        friend class CompositeFunction;

    public:
        ///
        /// Construct a new Mask object of specified shape
        /// 
        explicit NDMask(const NDShape& shape, const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice());

        ///
        /// Destruct 'this' mask object
        /// 
        ~NDMask();

        ///
        /// Mask out the specified sub-section of 'this' mask
        ///
        void MaskSection(const std::vector<size_t>& sectionOffset, const NDShape& sectionShape);

        ///
        /// Clear the mask; i.e. unmask all currently masked values
        ///
        void Clear();

        ///
        /// Returns the descriptor of the device that 'this' mask resides on
        ///
        DeviceDescriptor Device() const
        {
            return m_device;
        }

        ///
        /// Returns the shape 'this' mask.
        ///
        NDShape Shape() const
        {
            return m_maskShape;
        }

        ///
        /// Creates a new NDMask with newly allocated storage on the same device as 'this' mask and copies 'this' mask's contents into the newly allocated mask.
        ///
        NDMaskPtr DeepClone() const;

        ///
        /// Creates a new NDMask which is an alias of 'this' mask.
        ///
        NDMaskPtr Alias() const;

        ///
        /// Copies the contents of the 'source' NDMask to 'this' mask.
        /// The shapes of the 'source' mask and 'this' mask must be identical.
        ///
        void CopyFrom(const NDMask& source);

    private:
        NDMask(const NDShape& shape, Microsoft::MSR::CNTK::Matrix<char>* matrix);
        Microsoft::MSR::CNTK::Matrix<char>* GetMatrix() const;

        // Disallow copy construction and assignment
        NDMask(const NDMask&) = delete;
        NDMask& operator=(const NDMask&) = delete;

        // Disallow move construction and assignment
        NDMask& operator=(NDMask&&) = delete;
        NDMask(NDMask&& other) = delete;

    private:
        DeviceDescriptor m_device;
        NDShape m_maskShape;

        Microsoft::MSR::CNTK::Matrix<char>* m_matrixView;
    };

    /// 
    /// Denotes a multi-dimensional array with an optional mask and is the actual data fed into or produced from a computation.
    /// The mask is typically lower dimensionailty than the data, meaning data is masked in coarse individual sample units where
    /// sample shape is data.Shape().SubShape(0, data.Shape().NumAxes() - mask.Shape().NumAxes)
    /// Also, note that the size of the data's trailing mask.Shape().NumAxes() dimensions must match the mask shape dimensions.
    /// 
    class CNTK_API Value : public _Internal::_ReferenceCounter
    {
    public:
        ///
        /// A multi-dimensional value with no mask.
        ///
        Value(const NDArrayViewPtr& data);

        ///
        /// A multi-dimensional value with an associated mask.
        ///
        Value(const NDArrayViewPtr& data, const NDMaskPtr& mask);

        ///
        /// Create a new Value object containing a collection of variable length sequences.
        /// The created Value object contains a copy of the specified 'sequences' data.
        ///
        template <typename ElementType>
        static ValuePtr Create(const NDShape& sampleShape, const std::vector<std::vector<ElementType>>& sequences, const DeviceDescriptor& device, bool readOnly = false);

        ///
        /// Create a new Value object containing a collection of variable length sequences of one hot vectors
        /// The created Value object contains a copy of the specified 'sequences' data.
        ///
        template <typename ElementType>
        static ValuePtr Create(size_t vocabularySize, const std::vector<std::vector<size_t>>& oneHotSequences, const DeviceDescriptor& device, bool readOnly = false);

        ///
        /// Destruct 'this' Value object.
        ///
        virtual ~Value();

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
        // Disallow copy construction and assignment
        Value(const Value&) = delete;
        Value& operator=(const Value&) = delete;

        // Disallow move assignment and copy
        Value(Value&&) = delete;
        Value& operator=(Value&&) = delete;

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
            std::wstring tempName = staticAxisNamePrefix;
            tempName = tempName + std::to_wstring(staticAxisIdx);
            m_name = CopyString(tempName.c_str());
        }

        ///
        /// Construct a dynamic axis with the specified name.
        ///
        Axis(const std::wstring& name)
            : m_staticAxisIdx(SIZE_MAX)
        {
            m_name = CopyString(name.c_str());
        }

        ///
        /// Copy constructor.
        ///
        Axis(const Axis& other)
            : m_staticAxisIdx(SIZE_MAX), m_name(nullptr)
        {
            *this = other;
        }

        ///
        /// Copy assignment.
        ///
        Axis& operator=(const Axis& other)
        {
            if (this != &other)
            {
                delete[] m_name;

                m_staticAxisIdx = other.m_staticAxisIdx;
                m_name = (other.m_name != nullptr) ? CopyString(other.m_name) : other.m_name;
            }

            return *this;
        }

        ///
        /// Move constructor.
        ///
        Axis(Axis&& other)
            : m_staticAxisIdx(SIZE_MAX), m_name(nullptr)
        {
            *this = std::move(other);
        }

        ///
        /// Move assignment.
        ///
        Axis& operator=(Axis&& other)
        {
            assert(this != &other);

            delete[] m_name;

            m_staticAxisIdx = other.m_staticAxisIdx;
            m_name = other.m_name;

            other.m_staticAxisIdx = SIZE_MAX;
            other.m_name = nullptr;

            return *this;
        }

        ///
        /// Returns a boolean indicating if 'this' Axis corresponds to a static axis
        ///
        bool IsStaticAxis() const
        {
            return m_staticAxisIdx == SIZE_MAX;
        }

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
        static Axis DefaultDynamicAxis;

        ///
        /// Static Axis object representing the batch axis.
        ///
        static Axis BatchAxis;

        ///
        /// Special Axis object denoting all the axes of the Value object in whose context it is used.
        ///
        static Axis AllAxes;

        ///
        /// Name of 'this' axis
        ///
        std::wstring Name() const
        {
            return m_name;
        }

        ///
        /// Destructor
        ///
        ~Axis()
        {
            delete[] m_name;
        }

        ///
        /// Default constructor; results in an invalid axis object.
        ///
        Axis()
            : m_staticAxisIdx(SIZE_MAX), m_name(nullptr)
        {
        }

    private:
        size_t m_staticAxisIdx;
        wchar_t* m_name;
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
    ///
    class CNTK_API Variable
    {
        friend bool operator==(const Variable& first, const Variable& second);
        friend class Function;

        template <typename T>
        friend struct std::hash;

    public:
        ///
        /// Create an 'Input' Variable.
        ///
        Variable(const NDShape& shape, CNTK::DataType dataType, const std::wstring& name = L"")
            : Variable(shape, VariableKind::Input, dataType, nullptr, nullptr, false, { Axis::DefaultDynamicAxis }, false, name)
        {
        }

        ///
        /// Create an 'Input' Variable denoting sparse data.
        ///
        Variable(const NDShape& shape, bool isSparse, CNTK::DataType dataType, const std::wstring& name = L"")
            : Variable(shape, VariableKind::Input, dataType, nullptr, nullptr, false, { Axis::DefaultDynamicAxis }, isSparse, name)
        {
        }

        ///
        /// Create an 'Input' Variable and specify if gradients are to be computed for this input
        ///
        Variable(const NDShape& shape, CNTK::DataType dataType, bool needsGradient, const std::wstring& name = L"")
            : Variable(shape, VariableKind::Input, dataType, nullptr, nullptr, needsGradient, { Axis::DefaultDynamicAxis }, false, name)
        {
        }

        ///
        /// Create an 'Input' Variable denoting sparse data and specify if gradients are to be computed for this input
        ///
        Variable(const NDShape& shape, bool isSparse, CNTK::DataType dataType, bool needsGradient, const std::wstring& name = L"")
            : Variable(shape, VariableKind::Input, dataType, nullptr, nullptr, needsGradient, { Axis::DefaultDynamicAxis }, isSparse, name)
        {
        }

        ///
        /// Create an 'Output' variable
        ///
        Variable(const NDShape& shape, CNTK::DataType dataType, Function* ownerFunction, const std::vector<Axis>& dynamicAxes, const std::wstring& name = L"")
            : Variable(shape, VariableKind::Output, dataType, ownerFunction, nullptr, false, dynamicAxes, false, name)
        {
        }

        ///
        /// Create an 'Output' variable aliasing the output of the specified Function
        /// Throws an exception if called for a Function instance with multiple outputs
        ///
        Variable(const FunctionPtr& function);

        ///
        /// Returns the shape of 'this' variable
        ///
        NDShape Shape() const
        {
            return m_dataFields->m_shape;
        }

        ///
        /// Returns the dynamic axes of 'this' variable
        ///
        std::vector<Axis> DynamicAxes() const
        {
            return m_dataFields->m_dynamicAxes;
        }

        ///
        /// Returns the VariableKind of 'this' variable
        ///
        VariableKind Kind() const
        {
            return m_dataFields->m_varKind;
        }

        ///
        /// Returns a boolean value indicating if 'this' variable denotes sparse data
        ///
        bool IsSparseInput() const
        {
            return (Kind() == VariableKind::Input) && (m_dataFields->m_isSparse);
        }

        ///
        /// Returns a boolean value indicating if 'this' variable is a Parameter
        ///
        bool IsParameter() const
        {
            return Kind() == VariableKind::Parameter;
        }

        ///
        /// Returns a boolean value indicating if 'this' variable is a Constant
        ///
        bool IsConstant() const
        {
            return Kind() == VariableKind::Constant;
        }

        ///
        /// Returns a boolean value indicating if 'this' variable is a Placeholder
        ///
        bool IsPlaceholder() const
        {
            return Kind() == VariableKind::Placeholder;
        }

        ///
        /// Returns the name of 'this' variable
        ///
        std::wstring Name() const
        {
            return (m_dataFields->m_name == nullptr) ? L"" : m_dataFields->m_name;
        }

        ///
        /// Returns the Function object which 'this' variable is an ouptut of.
        /// Returns null when called for a Variable that is not of 'Output' VariableKind.
        ///
        FunctionPtr Owner() const
        {
            return m_dataFields->m_ownerFunction;
        }

        ///
        /// Returns the DataType of the data that 'this' Variable symbolically represents
        ///
        DataType GetDataType() const
        {
            return m_dataFields->m_dataType;
        }

        Variable()
        {
        }

        ///
        /// Returns a boolean value indicating if gradient computation is enabled for this variable.
        ///
        bool NeedsGradient() const
        {
            return m_dataFields->m_needsGradient;
        }

    protected:
        Variable(const NDShape& shape, VariableKind varType, CNTK::DataType dataType, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, const std::wstring& name)
            : Variable(shape, varType, dataType, nullptr, value, needsGradient, dynamicAxes, false, name)
        {
        }

        NDArrayViewPtr Value() const
        {
            assert(m_dataFields->m_value != nullptr);
            return m_dataFields->m_value;
        }

    private:
        Variable(const NDShape& shape, VariableKind varType, CNTK::DataType dataType, Function* ownerFunction, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, const std::wstring& name)
            : m_dataFields(new _VariableFields(shape, varType, dataType, ownerFunction, value, needsGradient, dynamicAxes, isSparse, (name == L"") ? nullptr : name.c_str()), [](_Internal::_ReferenceCounter* ptr) { delete ptr; })
        {
        }

    private:

        struct _VariableFields final : public _Internal::_ReferenceCounter
        {
            NDShape m_shape;
            VariableKind m_varKind;
            CNTK::DataType m_dataType;
            Function* m_ownerFunction; // Variable does not keep the Function alive
            NDArrayViewPtr m_value;
            bool m_needsGradient;
            wchar_t* m_name;
            _Internal::_SimpleVector<Axis> m_dynamicAxes;
            bool m_isSparse;

            _VariableFields(const NDShape& shape, VariableKind varType, CNTK::DataType type, Function* ownerFunction, const NDArrayViewPtr& value, bool needsGradient, const std::vector<Axis>& dynamicAxes, bool isSparse, const wchar_t* name)
                : m_shape(shape), m_varKind(varType), m_dataType(type), m_ownerFunction(ownerFunction), m_value(value), m_needsGradient(needsGradient), m_dynamicAxes(_Internal::_SimpleVector<Axis>::CreateSimpleVector(dynamicAxes)), m_isSparse(isSparse), m_name(nullptr)
            {
                if (name != nullptr)
                    m_name = CopyString(name);
            }

            ~_VariableFields()
            {
                delete[] m_name;
            }

        private:
            // Disallow copy construction and assignment
            _VariableFields(const _VariableFields&) = delete;
            _VariableFields& operator=(const _VariableFields& other) = delete;

            // Disallow move construction and assignment
            _VariableFields(_VariableFields&&) = delete;
            _VariableFields& operator=(_VariableFields&&) = delete;
        };
        typedef _Internal::_ReferenceCounterSharedPtr<_VariableFields> _VariableFieldsPtr;

        _VariableFieldsPtr m_dataFields;
    };

    inline bool operator==(const Variable& first, const Variable& second)
    {
        return first.m_dataFields == second.m_dataFields;
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
        {
        }

        // TODO: Constructor to move a specified NDArrayView value

        ///
        /// Construct a parameter of specified shape whose contents are initialized with the specified 'initValue'
        ///
        template<typename ElemType>
        Parameter(const NDShape& shape, ElemType initValue, const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice(), const std::wstring& name = L"")
            : Variable(shape, VariableKind::Parameter, AsDataType<ElemType>(), new NDArrayView(initValue, shape, device), true, {}, name)
        {
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
    };

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
        {
        }

        // TODO: Constructor to move a specified NDArrayView value

        ///
        /// Construct a constant of specified shape whose contents are initialized with the specified 'initValue'
        ///
        template<typename ElemType>
        Constant(const NDShape& shape, ElemType initValue, const DeviceDescriptor& device = DeviceDescriptor::DefaultDevice(), const std::wstring& name = L"")
            : Variable(shape, VariableKind::Constant, AsDataType<ElemType>(), new NDArrayView(initValue, shape, device), false, {}, name)
        {
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
    };

    static_assert(sizeof(Constant) == sizeof(Variable), "The Constant type should not have any data fields beyond what it's base type 'Variable' has.");

    ///
    /// Denotes a Placeholder input to a Function.
    /// All placeholder inputs of a Function must be replaced with non-placeholder Variables before Forward evaluation of the Function.
    ///
    class CNTK_API Placeholder final : public Variable
    {
        template <typename T>
        friend struct std::hash;
        
        friend class Function;

    public:
        ///
        /// Contruct a Placeholder with the specified NDShape
        ///
        explicit Placeholder(const NDShape& shape, const std::wstring& name = L"")
            : Variable(shape, VariableKind::Placeholder, DataType::Unknown, nullptr, false, {Axis::DefaultDynamicAxis}, name)
        {
        }

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
#pragma warning(pop)
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
            return std::hash<const void*>()(x.m_dataFields);
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
    /// that must be passed to a a subsequent 'Backward' call on the same Function to backpropgate gradient values
    /// for the same computation backwards through the Function
    ///
    class BackPropState : public _Internal::_ReferenceCounter
    {
    public:
        ///
        /// Returns the Function that 'this' BackPropState belongs to
        ///
        FunctionPtr Function() const { return m_function; }

    protected:
        BackPropState(const FunctionPtr& function) : m_function(function) {}

    private:
        virtual void _ForceRTTIGeneration() final
        {
            LogicError("This is an internal method that is never supposed to be called");
        }

    protected:
        FunctionPtr m_function;
    };
    typedef _Internal::_ReferenceCounterSharedPtr<BackPropState> BackPropStatePtr;

#pragma warning(push)
#pragma warning(disable : 4251 4275)

    ///
    /// Represents a function (optionally differentiable)
    /// A Function is a symbolic entity with zero or more input arguments and one or more outputs. 
    /// A Function may be primitive or composite (comprised of other function instances whose inputs and outputs are wired together).
    /// A Function effectively is an arbitrary computation graph composed of other primitive Functions, where Variable objects
    /// for the edges and leaves of the graph.
    ///
    class CNTK_API Function : public _Internal::_ReferenceCounter
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
        /// 'outputsToRetainBackwardStateFor'outputs of the function to any of the inputs of the Function, in a subsequent Backward call.
        /// Note that the returned BackPropState instance also stores a reference to the supplied 'inputs' Values and generated 'outputs' Values
        /// and the user is responsible for ensuring that the contents of the inputs and outputs are unchanged until after any uses of the BackPropState instance
        /// for backpropagating gradients through this function.
        ///
        BackPropStatePtr Forward(const std::unordered_map<Variable, const ValuePtr>& arguments,
                                 std::unordered_map<Variable, ValuePtr>& outputs,
                                 const DeviceDescriptor& computeDevice = DeviceDescriptor::DefaultDevice(),
                                 const std::unordered_set<Variable>& outputsToRetainBackwardStateFor = {})
        {
            auto abisSafeArgumentsMap = _Internal::_SimpleMap<Variable, const ValuePtr>::CreateSimpleMap(arguments);
            auto abisSafeOutputsMap = _Internal::_SimpleMap<Variable, ValuePtr>::CreateSimpleMap(outputs);
            auto abisSafeOutputsToRetainBackwardStateFor = _Internal::_SimpleSet<Variable>::CreateSimpleSet(outputsToRetainBackwardStateFor);

            auto backPropState = Forward(abisSafeArgumentsMap, abisSafeOutputsMap, abisSafeOutputsToRetainBackwardStateFor, computeDevice);

            // Copy over the ValuePtr values in outputs
            for (auto iter = outputs.begin(); iter != outputs.end(); ++iter)
                outputs[iter->first] = abisSafeOutputsMap[iter->first];

            return backPropState;
        }

        ///
        /// Backpropagates supplied 'rootGradientValues' for one or more of the output variables of the Function, to produce gradient Values
        /// corresponding to the specified set of input variables in 'backPropagatedGradientValuesForInputs'.
        /// Callers may specify the actual storage to be used for storing the 'backPropagatedGradientValuesForInputs' Values or leave them to be null
        /// in which case the implementation allocates the actual storage for storing the gradients.
        /// In case an existing storage is specified, the gradients are aggregated with existing values in the specified storage.
        /// The 'state' parameter is an instance of an BackPropState instance obtained from a previous call to the Forward method on 'this; Function for the 
        /// computation that this gradient backpropagation corresponds to.
        ///
        void Backward(const BackPropStatePtr& state,
                      const std::unordered_map<Variable, const ValuePtr>& rootGradientValues,
                      std::unordered_map<Variable, ValuePtr>& backPropagatedGradientValuesForInputs)
        {
            auto abisSafeRootGradientValuesMap = _Internal::_SimpleMap<Variable, const ValuePtr>::CreateSimpleMap(rootGradientValues);
            auto abisSafeBackPropagatedGradientValuesForInputs = _Internal::_SimpleMap<Variable, ValuePtr>::CreateSimpleMap(backPropagatedGradientValuesForInputs);

            Backward(state, abisSafeRootGradientValuesMap, abisSafeBackPropagatedGradientValuesForInputs);

            // Copy over the ValuePtr values in backPropagatedGradientValuesForInputs
            for (auto iter = backPropagatedGradientValuesForInputs.begin(); iter != backPropagatedGradientValuesForInputs.end(); ++iter)
                backPropagatedGradientValuesForInputs[iter->first] = abisSafeBackPropagatedGradientValuesForInputs[iter->first];
        }

    protected:
        // Mandatory methods to be overriden by new 'Function' types.

        virtual BackPropStatePtr Forward(const _Internal::_SimpleMap<Variable, const ValuePtr>& arguments,
                                         _Internal::_SimpleMap<Variable, ValuePtr>& outputs,
                                         const _Internal::_SimpleSet<Variable>& outputsToRetainBackwardStateFor,
                                         const DeviceDescriptor& computeDevice) = 0;

        virtual void Backward(const BackPropStatePtr& state,
                              const _Internal::_SimpleMap<Variable, const ValuePtr>& rootGradientValues,
                              _Internal::_SimpleMap<Variable, ValuePtr>& backPropagatedGradientValuesForInputs) = 0;

    public:

        // Optional overrides

        ///
        /// Destruct this Function.
        ///
        virtual ~Function()
        {
            delete[] m_name;
        }

    public:
        ///
        /// Returns the name of 'this' variable.
        ///
        std::wstring Name() const
        {
            return (m_name == nullptr) ? L"" : m_name;
        }

        ///
        /// Returns the primitive Function at the root of the graph of Functions underlying this Function.
        /// If 'this' Function itself is a primitive function then (this->RootFunction() == this).
        ///
        FunctionPtr RootFunction() const
        {
            return (m_rootFunction == nullptr) ? const_cast<Function*>(this) : m_rootFunction.GetPtr();
        }

        ///
        /// Returns all Input variables of 'this' Function.
        ///
        std::vector<Variable> Inputs() const
        {
            return _Inputs();
        }

        ///
        /// Returns the Output variable of 'this' Function. Throws an exception of 'this' Function has more that one output.
        ///
        Variable Output() const
        {
            if (m_outputs.Size() > 1)
                RuntimeError("A Fuction instance with more than one output cannot be implicitly converted to a Variable");

            return m_outputs[0];
        }

        ///
        /// Returns a vector consisting of all Output variables of 'this' Function.
        ///
        std::vector<Variable> Outputs() const
        {
            return m_outputs;
        }

        ///
        /// Returns a set comprising of all input variables of 'this' Function  variables that are not of kind 'Parameter' or 'Constant'.
        ///
        std::unordered_set<Variable> Arguments() const
        {
            return FilteredInputs<Variable>([](const Variable& var) {
                return ((var.Kind() == VariableKind::Input) || (var.Kind() == VariableKind::Output));
            });
        }

        ///
        /// Returns the set of all Parameter variables of 'this' Function.
        ///
        std::unordered_set<Parameter> Parameters() const
        {
            return FilteredInputs<Parameter>([](const Variable& var) {
                return (var.Kind() == VariableKind::Parameter);
            });
        }

        ///
        /// Returns the set of all Constant variables of 'this' Function.
        ///
        std::unordered_set<Constant> Constants() const
        {
            return FilteredInputs<Constant>([](const Variable& var) {
                return (var.Kind() == VariableKind::Constant);
            });
        }

        ///
        /// Returns the set of all Constant variables of 'this' Function.
        ///
        std::unordered_set<Placeholder> Placeholders() const
        {
            return FilteredInputs<Placeholder>([](const Variable& var) {
                return (var.Kind() == VariableKind::Placeholder);
            });
        }

        FunctionPtr ReplacePlaceholders(const std::unordered_map<Placeholder, Variable>& placeholderReplacements)
        {
            // Cannot be called on primitive functions
            if (RootFunction() == nullptr)
                InvalidArgument("ReplacePlaceholders should never be called on primitive functions");

            _Internal::_SimpleSet<const Function*> visitedFunctions;
            _Internal::_SimpleSet<Placeholder> replacedPlaceholders;
            auto abiSafePlaceholderReplacementsMap = _Internal::_SimpleMap<Placeholder, Variable>::CreateSimpleMap(placeholderReplacements);
            _ReplacePlaceholders(abiSafePlaceholderReplacementsMap, visitedFunctions, replacedPlaceholders);

            if (abiSafePlaceholderReplacementsMap.Keys() != replacedPlaceholders)
                InvalidArgument("At least one of the placeholders specified for replacement was not found in the function");

            return this;
        }

    private:

        template <typename VariableType, typename FilterFunction>
        std::unordered_set<VariableType> FilteredInputs(FilterFunction&& filterFunc) const
        {
            std::unordered_set<VariableType> filteredInputs;
            auto inputs = Inputs();
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                if (filterFunc(inputs[i]))
                    filteredInputs.insert(VariableType(inputs[i]));
            }

            return filteredInputs;

        }

        _Internal::_SimpleVector<Variable> _Inputs() const;
        virtual void _ReplacePlaceholders(const _Internal::_SimpleMap<Placeholder, Variable>& placeholderReplacements, _Internal::_SimpleSet<const Function*>& visitedFunctions, _Internal::_SimpleSet<Placeholder>& replacedPlaceholders);

        // Disallow copy and move construction and assignment
        Function(const Function&) = delete;
        Function(Function&&) = delete;
        Function& operator=(const Function&) = delete;
        Function& operator=(Function&&) = delete;

    protected:
        ///
        /// Protected constructor for derived 'Function' types to specify the actual input and output variables for the Function instance.
        /// All 'inputs' specified must be Variables of type Constant, Parameter or Input.
        ///
        Function(const std::vector<Variable>& inputs, const std::vector<Variable>& outputs, const FunctionPtr& rootFunction = nullptr, const std::wstring& name = L"")
            : m_rootFunction(rootFunction), m_name(nullptr)
        {
            for (size_t i = 0; i < inputs.size(); ++i)
            {
                m_inputs.PushBack(inputs[i]);

                if ((inputs[i].Kind() != VariableKind::Input) &&
                    (inputs[i].Kind() != VariableKind::Output) &&
                    (inputs[i].Kind() != VariableKind::Parameter) &&
                    (inputs[i].Kind() != VariableKind::Constant) &&
                    (inputs[i].Kind() != VariableKind::Placeholder))
                {
                    InvalidArgument("Function input has invalid VariableKind!");
                }
            }

            _Internal::_SimpleSet<Variable> uniqueOutputs;
            for (size_t i = 0; i < outputs.size(); ++i)
            {
                if (uniqueOutputs.Contains(outputs[i]))
                    RuntimeError("Same variable appears multiple times in the outputs vector passed to Function constructor");

                switch (outputs[i].Kind())
                {
                case VariableKind::Output:
                    m_outputs.PushBack(outputs[i]);
                    uniqueOutputs.Insert(outputs[i]);
                    break;
                default:
                    InvalidArgument("Function output has invalid VariableKind!");
                    break;
                }
            }

            if (name != L"")
                m_name = CopyString(name.c_str());
        }

    private:

        _Internal::_SimpleVector<Variable> m_inputs;
        _Internal::_SimpleVector<Variable> m_outputs;

        FunctionPtr m_rootFunction;
        wchar_t* m_name;
    };
#pragma warning(pop)

    CNTK_API FunctionPtr _Combine(const _Internal::_SimpleVector<FunctionPtr>& operands, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in matrix multiplication operation with the specified input operands.
    /// TODO: Specify the constraints on the shapes of the operands.
    ///
    CNTK_API FunctionPtr Times(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise tensor addition operation with the specified input operands.
    ///
    CNTK_API FunctionPtr Plus(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise sigmoid operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Sigmoid(const Variable& operand, const std::wstring& name = L"");
    
    ///
    /// Create an instance of the CNTK built-in elementwise tanh operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Tanh(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation to compute cross-entropy with softmax for specified input operands.
    ///
    CNTK_API FunctionPtr CrossEntropyWithSoftmax(const Variable& output, const Variable& labels, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in operation for computing the classification prediction error for specified operands.
    ///
    CNTK_API FunctionPtr PredictionError(const Variable& prediction, const Variable& labels, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in elementwise exp operation with the specified input operand.
    ///
    CNTK_API FunctionPtr Exp(const Variable& operand, const std::wstring& name = L"");

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
    /// Create an instance of the CNTK built-in elementwise multiplication operation on specified tensor input operands.
    ///
    CNTK_API FunctionPtr ElementTimes(const Variable& leftOperand, const Variable& rightOperand, const std::wstring& name = L"");

    ///
    /// Create an instance of the CNTK built-in sum reduction operation on specified tensor input operand along all the axes
    ///
    CNTK_API FunctionPtr ReduceSum(const Variable& operand, const std::wstring& name = L"");

    ///
    /// Create a new Function instance which just combines the outputs of the specified list of 'operands' Functions such that the 'Outputs' of the 
    /// new 'Function' are union of the 'Outputs' of each of the specified 'operands' Functions.
    ///
    inline FunctionPtr Combine(const std::initializer_list<FunctionPtr>& operands, const std::wstring& name = L"")
    {
        auto operandVector = _Internal::_SimpleVector<FunctionPtr>::CreateSimpleVector(operands);
        return _Combine(operandVector, name);
    }


    // Abstraction for learning a subset of parameters of a learnable function using first order gradient values
    // For e.g momentum, AdaGrad, RmsProp etc. are different types of learners with their own algorithms for 
    // learning parameter values using first order gradients.
    class Learner : public _Internal::_ReferenceCounter
    {
    public:
        // Method to update the parameters associated with this learner. By returning false, this method indicates that
        // learning has stopped for all of the parameters associated with this learner
        bool Update(std::unordered_map<Variable, ValuePtr>& parameters,
            const std::unordered_map<Variable, const ValuePtr>& gradients,
            size_t trainingSampleCount)
        {
            auto abisSafeParametersMap = _Internal::_SimpleMap<Variable, ValuePtr>::CreateSimpleMap(parameters);
            auto abisSafeGradientsMap = _Internal::_SimpleMap<Variable, const ValuePtr>::CreateSimpleMap(gradients);
            bool result = Update(abisSafeParametersMap, abisSafeGradientsMap, trainingSampleCount);

            for (auto iter : parameters)
            {
                parameters[iter.first] = abisSafeParametersMap[iter.first];
            }

            return result;
        }

        std::unordered_set<Variable> Parameters() const { return m_parameters; }

// TODO: the following methods are needed for backwards compatibility until sgd.cpp is updated to v2.
#pragma region _temporary_back_compat
        virtual double GetLearningRate() const = 0;
        virtual double GetMomentum() const = 0;
        virtual void SetLearningRate(double value) = 0;
        virtual void SetMomentum(double value) = 0;

        template <typename ElementType>
        std::list<std::shared_ptr<Microsoft::MSR::CNTK::Matrix<ElementType>>> GetSmoothedGradientsMatrices()
        {
            std::list<std::shared_ptr<Microsoft::MSR::CNTK::Matrix<ElementType>>> list;
            auto gradients = SmoothedGradients();
            for (size_t i = 0; i < gradients.Size(); ++i)
            {
                list.push_back(GetWritableMatrix<ElementType>(gradients[i]))
            }
            return list;
        }

    protected:

        virtual _Internal::_SimpleVector<ValuePtr>  SmoothedGradients() const = 0;

#pragma endregion _temporary_back_compat

    protected:
        Learner(const std::unordered_set<Variable>& parameters)
            : m_parameters(_Internal::_SimpleSet<Variable>::CreateSimpleSet(parameters))
        {   
        }

        template <typename ElementType>
        static std::shared_ptr<const Microsoft::MSR::CNTK::Matrix<ElementType>> GetMatrix(const NDArrayViewPtr arrayView)
        {
            return arrayView->GetMatrix<ElementType>();
        }

        template <typename ElementType>
        static std::shared_ptr<Microsoft::MSR::CNTK::Matrix<ElementType>> GetWritableMatrix(NDArrayViewPtr arrayView)
        {
            return arrayView->GetWritableMatrix<ElementType>();
        }

        template <typename ElementType>
        static const Microsoft::MSR::CNTK::TensorView<ElementType>* GetTensorView(const NDArrayViewPtr arrayView)
        {
            return arrayView->GetTensorView<ElementType>();
        }

        template <typename ElementType>
        static Microsoft::MSR::CNTK::TensorView<ElementType>* GetWritableTensorView(NDArrayViewPtr arrayView)
        {
            return arrayView->GetWritableTensorView<ElementType>();
        }


        virtual bool Update(const _Internal::_SimpleMap<Variable, ValuePtr>& parameters,
            const _Internal::_SimpleMap<Variable, const ValuePtr>& gradients,
            size_t trainingSampleCount) = 0;

        _Internal::_SimpleSet<Variable>  m_parameters;
    };

    // Methods to instantiate CNTK built-in learners
    LearnerPtr SGDLearner(const std::unordered_set<Variable>& parameters, 
        double learningRatePerSample, double momentumPerSample, bool useNesterovAcceleration = false);

    LearnerPtr AdaGradLearner(const std::unordered_set<Variable>& parameters, 
        double learningRatePerSample, bool needAveMultiplier = true);

    LearnerPtr FSAdaGradLearner(const std::unordered_set<Variable>& parameters,
        double learningRatePerSample, double momentumPerSample);

    LearnerPtr RmsPropLearner(const std::unordered_set<Variable>& parameters, 
        double learningRatePerSample, double gamma, double inc, double dec, double max, double min, bool needAveMultiplier = true);
}
