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
#include "ConvolutionEngine.h"
#include "ReshapingNodes.h"

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

    template <typename T>
    inline bool IsObjectExpired(std::weak_ptr<T> ptrToObject)
    {
        if ((ptrToObject.owner_before(std::weak_ptr<T>{}) || std::weak_ptr<T>{}.owner_before(ptrToObject)) && ptrToObject.expired())
            return true;
        else
            return false;
    }

    inline DEVICEID_TYPE AsCNTKImplDeviceId(const DeviceDescriptor& device)
    {
        if (device.Type() == DeviceKind::CPU)
            return CPUDEVICE;
        if (device.Type() == DeviceKind::GPU)
            return device.Id();

        LogicError("Invalid device type (%u).", (unsigned int)device.Type());
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
                InvalidArgument("AsNDShape() can only be called for TensorShapes that can be flattened to 1D.");
        }
        }

        return std::vector<size_t>(tensorShape.GetDims().begin(), tensorShape.GetDims().end());
    }

    inline Microsoft::MSR::CNTK::TensorShape AsTensorShape(const NDShape& viewShape)
    {
        const size_t maxNumAxesSupportedByTensorView = 12;
        if (viewShape.Rank() > maxNumAxesSupportedByTensorView)
            LogicError("The number (%d) of requested axes exceeds the currently supported limit (%d)", (int)viewShape.Rank(), (int)maxNumAxesSupportedByTensorView);

        // TensorShape is required to be at least 1D
        size_t minRankSize = 0;
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

    inline std::pair<size_t, size_t> GetMatrixDimensions(const NDShape& viewShape)
    {
        // Ensure none of the shape dimensions are unknown
        if (viewShape.HasUnboundDimension())
            InvalidArgument("Cannot create an NDArrayView using a view shape '%S' that has unknown dimensions for any of its axes.", viewShape.AsString().c_str());

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
                      std::is_same<T, int>::value ||
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
                      std::is_same<T, int>::value || 
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

    static int const CNTKInternalIdxValueForAllStaticAxes = Microsoft::MSR::CNTK::ReduceElementsNode<float>::CNTKInternalIdxValueForAllStaticAxes;
    static int const CNTKInternalIdxValueForAllAxes = Microsoft::MSR::CNTK::ReduceElementsNode<float>::CNTKInternalIdxValueForAllAxes;
    static int const CNTKInternalIdxValueForSequenceAxis = Microsoft::MSR::CNTK::ReduceElementsNode<float>::CNTKInternalIdxValueForSequenceAxis;
    static int const CNTKInternalIdxValueForBatchAxis = Microsoft::MSR::CNTK::ReduceElementsNode<float>::CNTKInternalIdxValueForBatchAxis;

    inline Axis AsAxis(int CNTKInternalAxisIdx)
    {
        if (CNTKInternalAxisIdx == CNTKInternalIdxValueForAllStaticAxes)
            return Axis::AllStaticAxes();

        if (CNTKInternalAxisIdx == CNTKInternalIdxValueForAllAxes)
            return Axis::AllAxes();

        if (CNTKInternalAxisIdx == CNTKInternalIdxValueForSequenceAxis)
            return Axis::OperandSequenceAxis();

        if (CNTKInternalAxisIdx == CNTKInternalIdxValueForBatchAxis)
            return Axis::DefaultBatchAxis();

        return Axis(CNTKInternalAxisIdx - 1);
    }

    inline std::vector<Axis> AsAxis(std::vector<int> CNTKInternalAxis)
    {
        std::vector<Axis> retAxisVec; 
        for (auto& axisIdx : CNTKInternalAxis)
            retAxisVec.push_back(AsAxis(axisIdx));
        return retAxisVec;
    }

    inline int AsCNTKInternalAxisIdx(const Axis& axis)
    {
        if (axis == Axis::AllStaticAxes())
            return CNTKInternalIdxValueForAllStaticAxes;

        if (axis == Axis::AllAxes())
            return CNTKInternalIdxValueForAllAxes;

        if (axis.IsDynamicAxis() && axis.IsOrdered())
            return CNTKInternalIdxValueForSequenceAxis;

        if (axis == Axis::DefaultBatchAxis())
            return CNTKInternalIdxValueForBatchAxis;

        if (!axis.IsStaticAxis())
            LogicError("Only Static Axes can be converted to a CNTK internal axis index");

        return (int)(axis.StaticAxisIndex() + 1);
    }

    inline std::vector<int> AsCNTKInternalAxisIdx(const std::vector<Axis>& axisVec)
    {
        std::vector<int> retAxisVec; 
        for (auto& axis : axisVec)
            retAxisVec.push_back(AsCNTKInternalAxisIdx(axis)); 
        return retAxisVec; 
    }

    inline std::pair<NDShape, NDShape> GetConvolutionOutputMapCountAndKernelShape(const NDShape& convolutionMapShape, const NDShape& operandShape, bool transpose)
    {
        NDShape kernelShape = convolutionMapShape.SubShape(0, operandShape.Rank());
        auto outputMapCount = convolutionMapShape.SubShape(kernelShape.Rank());
        auto shapeRank = operandShape.Rank(); 
        NDShape paddedOutputMapCount;
        if (shapeRank > outputMapCount.Rank())
            paddedOutputMapCount = NDShape(shapeRank - outputMapCount.Rank(), 1);

        paddedOutputMapCount = paddedOutputMapCount.AppendShape(outputMapCount);

        if (transpose && (shapeRank > 0) && (paddedOutputMapCount[shapeRank - 1] == NDShape::InferredDimension))  // convolution transpose, the mapCount in depth is derived from operandShape 
        {
            if (operandShape[shapeRank - 1] == NDShape::FreeDimension)
                InvalidArgument("Deconvolution: Output map count cannot be inferred from operand shape '%S' free dimension.", operandShape.AsString().c_str());

            paddedOutputMapCount[shapeRank - 1] = operandShape[shapeRank - 1];
        }

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

    // 'generateMangledNames' = true is used if we want to emit mangled names for the internal CNTK v1 nodes so that when
    // saving the model in V1 format and loading it back, we can retrieve the original V2 Variable/Function UID and Name.
    inline std::wstring CNTKInternalNodeNameFromUidAndName(const std::wstring& uid, const std::wstring& name, bool generateMangledNames = false)
    {
        if (generateMangledNames)
            return UidPrefix + uid + NamePrefix + name;
        else
            return uid;
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
            LogicError("Only dynamic axes can be derived from, to create new dynamic axes!");

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

    inline Axis& NormalizeStaticAxis(Axis& axis, size_t rank)
    {
        if (axis == Axis::EndStaticAxis())
            axis = Axis((int)rank);
        else if (axis.StaticAxisIndex() < 0)
        {
            auto normalizedAxis = Axis((int)rank + axis.StaticAxisIndex());
            if (normalizedAxis.StaticAxisIndex() < 0)
                InvalidArgument("Axis '%S' is out of bounds for the rank '%zd' it applies to.", axis.AsString().c_str(), rank);
            else
                axis = normalizedAxis;
        }
        return axis;
    }

    inline Axis& NormalizeStaticAxis(Axis& axis, const NDShape& operandShape)
    {
        if (axis != Axis::AllStaticAxes() && axis != Axis::AllAxes())
        {
            assert(axis.IsStaticAxis());
            assert(!operandShape.IsUnknown());
            axis = NormalizeStaticAxis(axis, operandShape.Rank());
        }
        return axis;
    }

    inline Axis& NormalizeAxis(Axis& axis, const Variable& operand)
    {
        if (axis.IsDynamicAxis())
        {
            auto operandDynamicAxes = operand.DynamicAxes();
            if (axis == Axis::OperandSequenceAxis() && (operandDynamicAxes != Axis::UnknownDynamicAxes()))
            {
                auto numOrderedDynamicAxes = std::count_if(operandDynamicAxes.begin(), operandDynamicAxes.end(), [](const Axis& axis) { return axis.IsOrdered(); });
                if (numOrderedDynamicAxes != 1)
                    InvalidArgument("Axis::OperandSequenceAxis() sentinel cannot be resolved if the operand '%S' has no sequence axis or > 1 ordered dynamic axes.", operand.AsString().c_str());

                axis = *std::find_if(operandDynamicAxes.begin(), operandDynamicAxes.end(), [](const Axis& axis) { return axis.IsOrdered(); });
            }

            return axis;
        }
        else
            return NormalizeStaticAxis(axis, operand.Shape());
    }

    inline void VerifyStaticAxis(const Axis& axis, const NDShape& operandShape)
    {
        assert(axis.IsStaticAxis());
        assert(axis.StaticAxisIndex() >= 0);

        if (axis.StaticAxisIndex() >= (int)operandShape.Rank())
            InvalidArgument("The specified axis index (%d) exceeds the #static axes (%d) of the corresponding operand (shape='%S)",
                            (int)axis.StaticAxisIndex(), (int)operandShape.Rank(), operandShape.AsString().c_str());
    }

    bool IsFirstOutputOfMultiOutputFunction(const Variable& var);
    inline  bool IsConstantScalar(const Variable& var)
    {
        return var.IsConstant() && (var.Shape().TotalSize() == 1);
    }

    inline Variable PlaceholderLike(const Variable& var)
    {
        return PlaceholderVariable(var.Shape(), var.GetDataType(), var.Name(), var.DynamicAxes());
    }

    std::vector<Axis> DynamicAxesFromInternalDynamicAxisName(const std::wstring& internalDynamicAxisName);

    // Construct the dynamic axis name to be used internally for the CNTK InputNodes
    std::wstring InternalDynamicAxisNameFromDynamicAxes(const std::vector<Axis>& dynamicAxes);

    std::shared_ptr<std::fstream> GetFstream(const std::wstring& filePath, bool readOnly);
    int GetFileDescriptor(const std::wstring& filePath, bool readOnly);

    std::string ToString(const std::wstring& wstring);
    std::wstring ToWString(const std::string& string);


    std::pair<size_t, size_t> GetNumTimeStepsAndSequences(const NDShape& maskShape, size_t numDynamicAxes);

    inline size_t ShapeRowColSplitPoint(const NDShape& varShape, bool isSparse, bool noDynamicAxes)
    {
        if (isSparse || noDynamicAxes)
            return std::min<size_t>(varShape.Rank(), 1);
        else
            return varShape.Rank();
    }

    inline size_t VariableRowColSplitPoint(const Variable& var)
    {
        return ShapeRowColSplitPoint(var.Shape(), var.IsSparse(), var.DynamicAxes().empty());
    }

    bool IsPackedValue(const ValuePtr& value);

    inline NDShape GetVariableShape(const NDShape& varShape, const Microsoft::MSR::CNTK::TensorShape& computationNodeShape)
    {
        auto fullyDefinedVarShape = varShape;
        if (computationNodeShape.GetRank() < fullyDefinedVarShape.Rank())
            LogicError("Computation node tensor shape '%s' must not be of lower rank than variable shape '%S'.", ((std::string)computationNodeShape).c_str(), fullyDefinedVarShape.AsString().c_str());

        for (size_t i = 0; i < fullyDefinedVarShape.Rank(); ++i)
        {
            if ((fullyDefinedVarShape[i] == NDShape::FreeDimension) || (fullyDefinedVarShape[i] == NDShape::InferredDimension))
                fullyDefinedVarShape[i] = computationNodeShape.GetDim(i);
            else if (fullyDefinedVarShape[i] != computationNodeShape.GetDim(i))
                LogicError("Computation node tensor shape '%s' does not match variable shape '%S'.", ((std::string)computationNodeShape).c_str(), fullyDefinedVarShape.AsString().c_str());
        }

        for (size_t i = fullyDefinedVarShape.Rank(); i < computationNodeShape.GetRank(); ++i)
        {
            if (computationNodeShape.GetDim(i) != 1)
                LogicError("Computation node tensor shape '%s' does not match variable shape '%S'.", ((std::string)computationNodeShape).c_str(), fullyDefinedVarShape.AsString().c_str());
        }

        return fullyDefinedVarShape;
    }

    std::vector<Axis> GetSqueezableAxes(const NDShape& inputShape);

    NDShape GetSqueezedShape(const NDShape& inputShape, const std::vector<Axis>& axes);

    NDShape GetSqueezedShape(const NDShape& inputShape);

    NDShape GetSqueezedShape(const NDShape& inputShape, const Dictionary& squeezeConfig);

    NDMaskPtr CreateMask(const std::vector<size_t>& sequenceLengths, const std::vector<bool>& sequenceStartFlags = {}, const DeviceDescriptor& device = DeviceDescriptor::CPUDevice());

    double ReductionIdentityValue(const std::wstring& reductionOpName);

    // Helper class to manage a collection of learners.
    class Learners
    {
    public:
        explicit Learners(const std::vector<LearnerPtr>& learners);

        bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, size_t trainingSampleCount, bool sweepEnd);
        bool Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, MinibatchInfo& minibatchInfo);

        std::vector<DictionaryValue> CreateCheckpoint();

        void RestoreFromCheckpoint(const std::vector<DictionaryValue>&);

        const std::vector<LearnerPtr>& ParameterLearners() const
        {
            return m_learners;
        }

        const LearnerPtr& GetMetricAggregatingLearner() const;

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

        std::function<void(NDArrayViewPtr&, NDArrayViewPtr&)> DoAggregateMetricsIfNeededLambda;
        
    private:
        void GetLearnerGradients(LearnerPtr learner, const std::unordered_map<Parameter, NDArrayViewPtr>& allGradients, std::unordered_map<Parameter, NDArrayViewPtr>& learnerGradients);
        void CheckDistributedLearners();

        std::vector<LearnerPtr> m_learners;
        bool m_isDistributed;
        LearnerPtr m_metricAggregatingLearner;
    };

    class Utils
    {
    public:
        static Axis NewDynamicAxisDerivedFromOperand(const std::wstring& axisNamePrefix, const Variable& operand)
        {
            std::function<Variable(const Variable&)> GetActualSourceVariable;
            GetActualSourceVariable = [&GetActualSourceVariable](const Variable& var) -> Variable {
                if (var.BlockFunctionVariableMapping() == Variable())
                    return var;
                else
                    return GetActualSourceVariable(var.BlockFunctionVariableMapping());
            };

            auto whereNodeConditionSourceVar = GetActualSourceVariable(operand);
            return Axis(axisNamePrefix + whereNodeConditionSourceVar.Uid());
        }
        static void VerifyVariableValueCompatibility(const Variable& var, const ValuePtr& value, NDShape* inferredVarShape = nullptr);

        template <typename ElementType>
        static std::pair<std::shared_ptr<const Microsoft::MSR::CNTK::Matrix<ElementType>>, Microsoft::MSR::CNTK::MBLayoutPtr>
        GetCNTKImplMatrixAndMBLayoutFromValueObject(const Variable& var, const ValuePtr& value, NDShape* inferredVarShape,
                                                    const std::shared_ptr<Microsoft::MSR::CNTK::Matrix<ElementType>>& outputMatrixStorage,
                                                    const std::shared_ptr<Microsoft::MSR::CNTK::Matrix<ElementType>>& tempIndicesStorage);

        template <typename ElementType>
        static std::pair<std::shared_ptr<const Microsoft::MSR::CNTK::Matrix<ElementType>>, Microsoft::MSR::CNTK::MBLayoutPtr>
        GetCNTKImplMatrixAndMBLayoutFromValueObject(const Variable& var, const ValuePtr& value, NDShape* inferredVarShape = nullptr)
        {
            auto nullSharedPtr = std::shared_ptr<Microsoft::MSR::CNTK::Matrix<ElementType>>(nullptr);
            return GetCNTKImplMatrixAndMBLayoutFromValueObject(var, value, inferredVarShape, nullSharedPtr, nullSharedPtr);
        }

        template <typename ElementType>
        static ValuePtr GetValueObjectFromCNTKImplMatrixAndMBLayout(const NDShape& sampleShape, const std::vector<Axis>& sampleDynamicAxes, const Microsoft::MSR::CNTK::Matrix<ElementType>& matrix, const Microsoft::MSR::CNTK::MBLayoutPtr& layout, bool readOnly = true);

        template <typename ElementType>
        static ValuePtr GetValueObjectFromCNTKImplMatrixAndMBLayout(const Variable& var, const Microsoft::MSR::CNTK::ComputationNodeBasePtr& computationNode, const Microsoft::MSR::CNTK::Matrix<ElementType>& matrix, const Microsoft::MSR::CNTK::MBLayoutPtr& layout, bool readOnly = true);
    };

    template <typename Container>
    inline std::wstring NamedListString(const Container& namedList)
    {
        std::wstringstream wss;
        bool first = true;
        for (auto namedObject : namedList)
        {
            if (!first)
                wss << L", ";

            wss << namedObject.AsString();
            first = false;
        }

        return wss.str();
    }

    class Accumulator : public Value
    {
    public:
        Accumulator() : Value(nullptr), m_numUpdates(0), m_isInitialized(false) {}

        void Update(const ValuePtr& delta, const DeviceDescriptor& device);
        void Reset();
        bool IsInitialized() { return m_isInitialized; }
    private:
        void ResetToZero();

        bool m_isInitialized;
        size_t   m_numUpdates;
    };

    std::wstring DynamicAxesAsString(const std::vector<Axis>& da, bool rowMajor = false);

    template <typename T> //T can be Variable or StreamInfo
    static bool IsAtSweepEnd(const std::unordered_map<T, MinibatchData>& arguments)
    {
        if (arguments.empty()) return true;

        return std::any_of(arguments.begin(), arguments.end(), [](const std::pair<const T, MinibatchData>& kv)
        {
            return kv.second.sweepEnd;
        });
    }

    // half is V1 ElemType, so specialize here instead of in CNTKLibrary.h
    template<>
    inline DataType AsDataType<half>()
    {
        return DataType::Float16;
    }
}
