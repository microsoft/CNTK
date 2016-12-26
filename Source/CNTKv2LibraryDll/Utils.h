//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "CNTKLibrary.h"
#include "CommonMatrix.h"
#include "TensorShape.h"
#include <string>
#include "Config.h"
#include "Reader.h"
#include "ConvolutionEngine.h"

namespace CNTK
{
    // Forward declarations
    class Dictionary;

    // Helper to get the size of an element of the specified DataType
    inline size_t ElementSize(DataType dataType)
    {
        if (dataType == DataType::Float)
            return sizeof(float);
        else if (dataType == DataType::Double)
            return sizeof(double);
        else
            NOT_IMPLEMENTED;
    }

    inline DEVICEID_TYPE AsCNTKImplDeviceId(const DeviceDescriptor& device)
    {
        if (device.Type() == DeviceKind::CPU)
            return CPUDEVICE;
        else if (device.Type() == DeviceKind::GPU)
            return device.Id();
        else
            NOT_IMPLEMENTED;
    }

    inline Microsoft::MSR::CNTK::MatrixFormat AsCNTKImplMatrixFormat(StorageFormat storageFormat)
    {
        if (storageFormat == StorageFormat::Dense)
            return Microsoft::MSR::CNTK::MatrixFormat::matrixFormatDense;
        else if (storageFormat == StorageFormat::SparseCSC)
            return Microsoft::MSR::CNTK::MatrixFormat::matrixFormatSparseCSC;
        else if (storageFormat == StorageFormat::SparseBlockCol)
            return Microsoft::MSR::CNTK::MatrixFormat::matrixFormatSparseBlockCol;
        else
            NOT_IMPLEMENTED;
    }

    inline StorageFormat AsStorageFormat(Microsoft::MSR::CNTK::MatrixFormat matrixFormat)
    {
        if (matrixFormat == Microsoft::MSR::CNTK::MatrixFormat::matrixFormatDense)
            return StorageFormat::Dense;
        else if (matrixFormat == Microsoft::MSR::CNTK::MatrixFormat::matrixFormatSparseCSC)
            return StorageFormat::SparseCSC;
        else if (matrixFormat == Microsoft::MSR::CNTK::MatrixFormat::matrixFormatSparseBlockCol)
            return StorageFormat::SparseBlockCol;
        else
            NOT_IMPLEMENTED;
    }

    inline DeviceDescriptor AsDeviceDescriptor(DEVICEID_TYPE deviceId)
    {
        if (deviceId == CPUDEVICE)
            return DeviceDescriptor::CPUDevice();
        else
            return DeviceDescriptor::GPUDevice(deviceId);
    }

    inline NDShape AsNDShape(const Microsoft::MSR::CNTK::TensorShape& tensorShape, bool allowNonFlattenableTensorShapes = false)
    {
        if (!allowNonFlattenableTensorShapes)
    {
        // The TensorShape should be flattenable to 1D
        for (size_t i = 1; i < tensorShape.GetRank(); ++i)
        {
            if (!tensorShape.CanFlatten(i))
                InvalidArgument("AsNDShape() can only be called for TensorShapes that can be flattened to 1D");
        }
        }

        return std::vector<size_t>(tensorShape.GetDims().begin(), tensorShape.GetDims().end());
    }

    inline DataType AsDataType(Microsoft::MSR::CNTK::ElementType readerDataType)
    {
        switch (readerDataType)
        {
        case Microsoft::MSR::CNTK::ElementType::tfloat:
            return DataType::Float;
        case Microsoft::MSR::CNTK::ElementType::tdouble:
            return DataType::Double;
        default:
            LogicError("Unsupported ElementType from CNTK Reader");
        }
    }

    inline StorageFormat AsStorageFormat(Microsoft::MSR::CNTK::StorageType readerStorageType)
    {
        switch (readerStorageType)
        {
        case Microsoft::MSR::CNTK::StorageType::dense:
            return StorageFormat::Dense;
        case Microsoft::MSR::CNTK::StorageType::sparse_csc:
            return StorageFormat::SparseCSC;
        default:
            LogicError("Unsupported StorageType from CNTK Reader");
        }
    }

    inline Microsoft::MSR::CNTK::TensorShape AsTensorShape(const NDShape& viewShape)
    {
        const size_t maxNumAxesSupportedByTensorView = 12;
        if (viewShape.Rank() > maxNumAxesSupportedByTensorView)
            LogicError("The number of requested axes exceeds the currently supported limit");

        // TensorShape is required to be at least 1D
        size_t minRankSize = 1;
        Microsoft::MSR::CNTK::SmallVector<size_t> tensorViewShape(std::max<size_t>(minRankSize, viewShape.Rank()));
        for (size_t i = 0; i < tensorViewShape.size(); ++i)
            tensorViewShape[i] = (i < viewShape.Rank()) ? viewShape[i] : 1;

        return tensorViewShape;
    }

    inline Microsoft::MSR::CNTK::TensorShape AsTensorViewShape(const Microsoft::MSR::CNTK::TensorShape& viewShape)
    {
        // For TensorView shapes we pad the TensorShape to be at least rank 2
        return viewShape.PadRank(std::max<size_t>(2, viewShape.GetRank()));
    }

    inline Microsoft::MSR::CNTK::TensorShape AsTensorViewShape(const NDShape& viewShape)
    {
        return AsTensorViewShape(AsTensorShape(viewShape));
    }

    inline std::wstring AsStringForErrorReporting(const NDShape& shape)
    {
        bool invertShape = Internal::IsReversingTensorShapesInErrorMessagesEnabled();
        auto displayShape = shape;
        if (invertShape)
        {
            for (size_t i = 0, j = shape.Rank() - 1; i < shape.Rank(); ++i, --j)
                displayShape[i] = shape[j];
        }

        return displayShape.AsString();
    }

    inline std::pair<size_t, size_t> GetMatrixDimensions(const NDShape& viewShape)
    {
        // Ensure none of the shape dimensions are unknown
        if (viewShape.HasInferredDimension())
            InvalidArgument("Cannot create an NDArrayView using a view shape that has unknown dimensions for any of its axes!");

        size_t matrixRowSize = (viewShape.Rank() > 0) ? viewShape[0] : 1;
        size_t matrixColSize = (viewShape.Rank() > 0) ? viewShape.SubShape(1).TotalSize() : 1;

        return{ matrixRowSize, matrixColSize };
    }

    inline bool IsSparseInput(const Variable& var)
    {
        return var.IsInput() && var.IsSparse();
    }


    inline void AddIndentation(std::wstringstream& s, size_t numIndentationSpaces)
    {
        for (size_t i = 0; i < numIndentationSpaces; ++i)
            s << L" ";
    }

    static const size_t perLevelIndentSize = 4;
    inline void AddConfigString(std::wstringstream& s, const std::wstring& key, const DictionaryValue& value, size_t numIndentationSpaces);
    inline void AddConfigString(std::wstringstream& s, const DictionaryValue& value, size_t numIndentationSpaces)
    {
        switch (value.ValueType())
        {
        case DictionaryValue::Type::Bool:
            s << value.Value<bool>();
            break;
        case DictionaryValue::Type::Float:
            s << value.Value<float>();
            break;
        case DictionaryValue::Type::Double:
            s << value.Value<double>();
            break;
        case DictionaryValue::Type::String:
            s << value.Value<std::wstring>();
            break;
        case DictionaryValue::Type::Int:
            s << value.Value<int>();
            break;
        case DictionaryValue::Type::SizeT:
            s << value.Value<size_t>();
            break;
        case DictionaryValue::Type::Vector:
        {
            const auto& valueVector = value.Value<std::vector<DictionaryValue>>();
            s << L"(" << std::endl;
            AddIndentation(s, numIndentationSpaces + perLevelIndentSize);
            bool isFirst = true;
            for (const auto& val : valueVector)
            {
                if (!isFirst)
                    s << L":";
                else
                    isFirst = false;

                AddConfigString(s, val, numIndentationSpaces + perLevelIndentSize);
            }
            AddIndentation(s, numIndentationSpaces);
            s << L")";
            break;
        }
        case DictionaryValue::Type::Dictionary:
        {
            const auto& valueDictionary = value.Value<Dictionary>();
            s << L"[" << std::endl;
            for (const auto& keyValuePair : *(valueDictionary.m_dictionaryData))
            {
                AddConfigString(s, keyValuePair.first, keyValuePair.second, numIndentationSpaces + perLevelIndentSize);
            }
            AddIndentation(s, numIndentationSpaces);
            s << L"]";
            break;
        }
        default:
            LogicError("Unsupported DictionaryValue type");
        }
    }

    inline void AddConfigString(std::wstringstream& s, const std::wstring& key, const DictionaryValue& value, size_t numIndentationSpaces)
    {
        AddIndentation(s, numIndentationSpaces);
        s << key << L" = ";
        AddConfigString(s, value, numIndentationSpaces);
        s << std::endl;
    }

    template <typename T>
    inline std::vector<DictionaryValue> AsDictionaryValueVector(const std::vector<T>& elementVector)
    {
        static_assert(std::is_same<T, bool>::value ||
                      std::is_same<T, size_t>::value ||
                      std::is_same<T, float>::value ||
                      std::is_same<T, double>::value ||
                      std::is_same<T, Axis>::value ||
                      std::is_same<T, std::wstring>::value,
                      "Unsupported ValueType");

        std::vector<DictionaryValue> dictionaryValueVector;
        for (auto value : elementVector)
            dictionaryValueVector.push_back(value);

        return dictionaryValueVector;
    }

    template <typename T>
    inline std::vector<T> AsVector(const std::vector<DictionaryValue>& dictionaryValueVector)
    {
        static_assert(std::is_same<T, bool>::value ||
                      std::is_same<T, size_t>::value ||
                      std::is_same<T, float>::value ||
                      std::is_same<T, double>::value ||
                      std::is_same<T, Axis>::value ||
                      std::is_same<T, std::wstring>::value,
                      "Unsupported ValueType");

        std::vector<T> elementVector;
        for (auto value : dictionaryValueVector)
            elementVector.push_back(value.Value<T>());

        return elementVector;
    }

    inline PoolingType AsPoolingType(Microsoft::MSR::CNTK::PoolKind cntkPoolingKind)
    {
        switch (cntkPoolingKind)
        {
        case Microsoft::MSR::CNTK::PoolKind::Average:
            return PoolingType::Average;
        case Microsoft::MSR::CNTK::PoolKind::Max:
            return PoolingType::Max;
        default:
            LogicError("Unknown pooling type");
        }
    }

    inline Microsoft::MSR::CNTK::PoolKind AsCNTKPoolKind(PoolingType poolingType)
    {
        switch (poolingType)
        {
        case PoolingType::Average:
            return Microsoft::MSR::CNTK::PoolKind::Average;
        case PoolingType::Max:
            return Microsoft::MSR::CNTK::PoolKind::Max;
        default:
            LogicError("Unknown pooling type");
        }
    }

    static size_t const CNTKInternalIdxValueForAllStaticAxes = 0;
    inline Axis AsAxis(int CNTKInternalAxisIdx)
    {
        if (CNTKInternalAxisIdx == CNTKInternalIdxValueForAllStaticAxes)
            return Axis::AllStaticAxes();

        return Axis(CNTKInternalAxisIdx - 1);
    }

    inline int AsCNTKInternalAxisIdx(const Axis& axis)
    {
        if (axis == Axis::AllStaticAxes())
            return CNTKInternalIdxValueForAllStaticAxes;

        if (!axis.IsStaticAxis())
            LogicError("Only Axis that represent static indices can be converted to a CNTK internal axis index");

        return (int)(axis.StaticAxisIndex() + 1);
    }

    inline std::pair<NDShape, NDShape> GetConvolutionOutputMapCountAndKernelShape(const NDShape& convolutionMapShape, const NDShape& operandShape)
    {
        NDShape kernelShape = convolutionMapShape.SubShape(0, operandShape.Rank());
        auto outputMapCount = convolutionMapShape.SubShape(kernelShape.Rank());
        NDShape paddedOutputMapCount(operandShape.Rank(), 1);
        for (size_t i = 0; i < outputMapCount.Rank(); ++i)
            paddedOutputMapCount[paddedOutputMapCount.Rank() - 1 - i] = outputMapCount[outputMapCount.Rank() - 1 - i];

        return{ paddedOutputMapCount, kernelShape };
    }

    template <typename SourceElementType, typename TargetElementType>
    inline TargetElementType* Copy(const SourceElementType* src, size_t srcSize)
    {
        // Cast to double
        TargetElementType* castValue = new TargetElementType[srcSize];
        for (size_t i = 0; i < srcSize; ++i)
            castValue[i] = (TargetElementType)src[i];

        return castValue;
    }

    inline NDArrayViewPtr CloneAsDataType(const NDArrayViewPtr& source, DataType targetDataType, bool readOnly)
    {
        if (source->Device() != DeviceDescriptor::CPUDevice())
            LogicError("CloneAsDataType currently does not support non-CPU source NDArrayView objects");

        auto sourceDataType = source->GetDataType();
        if (sourceDataType == targetDataType)
            LogicError("CloneAsDataType: Source and target DataTypes are same");

        if (targetDataType != DataType::Double)
            LogicError("CloneAsDataType: Only Double target DataType is supported");

        auto sourceShape = source->Shape();
        auto sourceSize = sourceShape.TotalSize();

        // Cast to double
        double* castValue = Copy<float, double>(source->DataBuffer<float>(), sourceSize);
        return MakeSharedObject<NDArrayView>(sourceShape, castValue, sourceSize, DeviceDescriptor::CPUDevice(), readOnly);
    }

    template <typename T>
    inline std::string Typename(const T* = nullptr)
    {
        auto name = typeid(T).name(); 
        if (strncmp(name, "class ", 6) == 0)
        {
            // On Windows, the type name contains "class" prefix. 
            // Return the actual name, omitting the prefix.
            return &name[6];
        }
        return name;
    }

    inline std::wstring ParanthesizedName(const std::wstring& name)
    {
        if (name.empty())
            return name;

        return L"(" + name + L")";
    }

    static const std::wstring UidPrefix = L"__v2libuid__";
    static const std::wstring NamePrefix = L"__v2libname__";

    inline std::wstring CNTKInternalNodeNameFromUidAndName(const std::wstring& uid, const std::wstring& name)
    {
        return UidPrefix + uid + NamePrefix + name;
    }

    inline std::pair<std::wstring, std::wstring> UidAndNameFromCNTKInternalNodeName(const std::wstring& CNTKInternalNodeName)
    {
        std::wstring uid, name;
        auto uidPrefixBeginPos = CNTKInternalNodeName.find(UidPrefix);
        if (uidPrefixBeginPos != std::wstring::npos)
        {
            auto uidBeginPos = uidPrefixBeginPos + UidPrefix.length();
            auto namePrefixBeginPos = CNTKInternalNodeName.find(NamePrefix, uidBeginPos);
            if (namePrefixBeginPos == std::wstring::npos)
                LogicError("CNTK internal node name found to contain uid but not name!");

            auto nameBeginPos = namePrefixBeginPos + NamePrefix.length();
            uid = CNTKInternalNodeName.substr(uidBeginPos, namePrefixBeginPos - uidBeginPos);
            name = CNTKInternalNodeName.substr(nameBeginPos);
        }

        return{ uid, name };
    }

    inline std::pair<std::wstring, std::wstring> UidAndNameFromCNTKInternalNodeName(const std::wstring& CNTKInternalNodeName, VariableKind varKind)
    {
        std::wstring uid, name;
        std::tie(uid, name) = UidAndNameFromCNTKInternalNodeName(CNTKInternalNodeName);
        if (uid == L"")
        {
            name = CNTKInternalNodeName;
            uid = Internal::GenerateUid(varKind);
        }

        return{ uid, name };
    }

    std::pair<std::wstring, std::wstring> UidAndNameFromCNTKInternalNodeName(const std::wstring& CNTKInternalNodeName, const PrimitiveOpType& opType);

    inline std::vector<Axis> GetDerivedDynamicAxes(const Axis& sourceAxis, size_t multiplicativeFactor, int additiveFactor)
    {
        if (!sourceAxis.IsDynamicAxis())
            LogicError("Only dynamic axes can be derived from to create new dynamic axes!");

        if ((multiplicativeFactor == 0) && (additiveFactor == 0))
            LogicError("Zero size dynamic axes are not allowed!");

        // If we slice off exactly one frame off of the source axis, then we effectively delete this axis
        if ((multiplicativeFactor == 0) && (additiveFactor == 1))
            return {};

        if ((multiplicativeFactor == 1) && (additiveFactor == 0))
            return {sourceAxis};

        std::wstring derivedDynamicAxisName = sourceAxis.Name();
        if (multiplicativeFactor > 0)
        {
            derivedDynamicAxisName += L"_times_" + std::to_wstring(multiplicativeFactor);
            if (additiveFactor > 0)
                derivedDynamicAxisName += L"_plus_" + std::to_wstring(additiveFactor);
            else
                derivedDynamicAxisName += L"_minus_" + std::to_wstring(-additiveFactor);
        }
        else
        {
            assert(additiveFactor > 0);
            derivedDynamicAxisName += L"_fixedSliceOf_" + std::to_wstring(additiveFactor);
        }

        return{ Axis(derivedDynamicAxisName, sourceAxis.IsOrdered()) };
    }

    inline Axis& NormalizeStaticAxis(Axis& axis, const NDShape& operandShape)
    {
        if (axis != Axis::AllStaticAxes())
        {
            assert(axis.IsStaticAxis());
            assert(operandShape != NDShape::Unknown);

            if (axis == Axis::EndStaticAxis())
                axis = Axis((int)operandShape.Rank());
            else if (axis.StaticAxisIndex() < 0)
                axis = Axis((int)operandShape.Rank() + axis.StaticAxisIndex());
        }

        return axis;
    }

    inline void VerifyStaticAxis(const Axis& axis, const NDShape& operandShape)
    {
        assert(axis.IsStaticAxis());
        assert(axis.StaticAxisIndex() >= 0);

        if (axis.StaticAxisIndex() >= (int)operandShape.Rank())
            InvalidArgument("The specified axis index (%d) exceeds the static #axes (%d) of the corresponding operand", (int)axis.StaticAxisIndex(), (int)operandShape.Rank());
    }

    std::shared_ptr<std::fstream> GetFstream(const std::wstring& filePath, bool readOnly);
    int GetFileDescriptor(const std::wstring& filePath, bool readOnly);

    std::string ToString(const std::wstring& wstring);
    std::wstring ToWString(const std::string& string);
    // Helper class to manage a collection of learners.
    class Learners
    {
    public:
        explicit Learners(const std::vector<LearnerPtr>& learners);

        bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, size_t trainingSampleCount);
        bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, MinibatchInfo& minibatchInfo);

        std::vector<DictionaryValue> CreateCheckpoint();

        void RestoreFromCheckpoint(const std::vector<DictionaryValue>&);

        const std::vector<LearnerPtr>& ParameterLearners() const
        {
            return m_learners;
        }

        std::unordered_set<Parameter> GetParameters() const
        {
            std::unordered_set<Parameter> result;
            for (auto l : m_learners)
            {
                const auto& p = l->Parameters();
                result.insert(p.begin(), p.end());
            }
            return result;
        }

        bool IsDistributed() const
        {
            return m_isDistributed;
        }

    private:
        void GetLearnerGradients(LearnerPtr learner, const std::unordered_map<Parameter, NDArrayViewPtr>& allGradients, std::unordered_map<Parameter, NDArrayViewPtr>& learnerGradients);
        void CheckDistributedLearners();

        std::vector<LearnerPtr> m_learners;
        bool m_isDistributed;
    };

    class Utils
    {
    public:
        template <typename ElementType>
        static std::pair<std::shared_ptr<const Microsoft::MSR::CNTK::Matrix<ElementType>>, Microsoft::MSR::CNTK::MBLayoutPtr> GetCNTKImplMatrixAndMBLayoutFromValueObject(const Variable& var, const ValuePtr& value);

        template <typename ElementType>
        static ValuePtr GetValueObjectFromCNTKImplMatrixAndMBLayout(const NDShape& sampleShape, const Microsoft::MSR::CNTK::Matrix<ElementType>& matrix, const Microsoft::MSR::CNTK::MBLayoutPtr& layout, bool readOnly = true);

        template <typename ElementType>
        static ValuePtr GetValueObjectFromCNTKImplMatrixAndMBLayout(const Variable& var, const Microsoft::MSR::CNTK::Matrix<ElementType>& matrix, const Microsoft::MSR::CNTK::MBLayoutPtr& layout, bool readOnly = true);
    };

    template <typename NamedType>
    inline std::wstring NamedListString(const std::vector<NamedType>& namedList)
    {
        std::wstring namedListString;
        for (auto namedObject : namedList)
        {
            if (!namedListString.empty())
                namedListString += L", ";

            namedListString += namedObject.Name();
        }

        return namedListString;
    }
}
