//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#if defined(_MSC_VER) || defined(_CODECVT_H)
#include <codecvt>
#else
#include <cstdlib>
#include <clocale>
#endif
#include "CNTKLibrary.h"
#include "Utils.h"
#include "Serialization.h"
#include <fcntl.h>
#include "PrimitiveFunction.h"
#include "RecurrentNodes.h"
#include "Value.h"
#include "CompositeFunction.h"

using namespace std;
using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    template<typename T>
    T* CreateDataPtr(const T& value)
    {
        return new T(value);
    }

    template <>
    NDArrayView* CreateDataPtr<NDArrayView>(const NDArrayView& value)
    {
        // TODO: replace this copy with an alias to value.
        NDArrayView* viewPtr = new NDArrayView(value.GetDataType(), value.Shape(), DeviceDescriptor::CPUDevice());
        viewPtr->CopyFrom(value);
        return viewPtr;
    }

    template <typename T>
    void DictionaryValue::AllocateDataPtr(const T& value)
    {
        static_assert(is_same<T, NDShape>::value ||
                      is_same<T, Axis>::value ||
                      is_same<T, wstring>::value ||
                      is_same<T, vector<DictionaryValue>>::value ||
                      is_same<T, Dictionary>::value ||
                      is_same<T, NDArrayView>::value,
                      "AllocateDataPtr called with invalid type");
        m_data.m_ptr = CreateDataPtr<T>(value);
    }

    template <typename T>
    void DictionaryValue::FreePtrAsType()
    {
        T* typedPtr = reinterpret_cast<T*>(m_data.m_ptr);
        delete typedPtr;

        m_data.m_ptr = nullptr;
    }


    bool DictionaryValue::operator==(const DictionaryValue& other) const
    {
        if (this == &other)
        {
            return true;
        }

        if (m_valueType != other.m_valueType)
        {
            return false;
        }
        
        switch (m_valueType)
        {
        case DictionaryValue::Type::Bool:
            return (m_data.m_boolean == other.m_data.m_boolean);
        case DictionaryValue::Type::Int:
            return (m_data.m_int == other.m_data.m_int);
        case DictionaryValue::Type::SizeT:
            return (m_data.m_sizeT == other.m_data.m_sizeT);
        case DictionaryValue::Type::Float:
            return (m_data.m_float == other.m_data.m_float);
        case DictionaryValue::Type::Double:
            return (m_data.m_double == other.m_data.m_double);
        case DictionaryValue::Type::String:
        {
            wstring* strPtr1 = reinterpret_cast<wstring*>(m_data.m_ptr);
            wstring* strPtr2 = reinterpret_cast<wstring*>(other.m_data.m_ptr);
            return (*strPtr1 == *strPtr2);
        }
        case DictionaryValue::Type::NDShape:
        {
            NDShape* shapePtr1 = reinterpret_cast<NDShape*>(m_data.m_ptr);
            NDShape* shapePtr2 = reinterpret_cast<NDShape*>(other.m_data.m_ptr);
            return (*shapePtr1 == *shapePtr2);
        }
        case DictionaryValue::Type::Axis:
        {
            Axis* axisPtr1 = reinterpret_cast<Axis*>(m_data.m_ptr);
            Axis* axisPtr2 = reinterpret_cast<Axis*>(other.m_data.m_ptr);
            return (*axisPtr1 == *axisPtr2);
        }
        case DictionaryValue::Type::Vector:
        {   
            vector<DictionaryValue>* vectorPtr1 = reinterpret_cast<vector<DictionaryValue>*>(m_data.m_ptr);
            vector<DictionaryValue>* vectorPtr2 = reinterpret_cast<vector<DictionaryValue>*>(other.m_data.m_ptr);
            return (*vectorPtr1 == *vectorPtr2);
        }
        case DictionaryValue::Type::Dictionary:
        {
            Dictionary* dictPtr1 = reinterpret_cast<Dictionary*>(m_data.m_ptr);
            Dictionary* dictPtr2 = reinterpret_cast<Dictionary*>(other.m_data.m_ptr);
            return (*dictPtr1 == *dictPtr2);
        }
        case DictionaryValue::Type::NDArrayView:
        {
            NDArrayView* viewPtr1 = reinterpret_cast<NDArrayView*>(m_data.m_ptr);
            NDArrayView* viewPtr2 = reinterpret_cast<NDArrayView*>(other.m_data.m_ptr);

            return Internal::AreEqual(*viewPtr1, *viewPtr2);
        }
        default:
            NOT_IMPLEMENTED;
        }
    }
    
    bool DictionaryValue::operator!=(const DictionaryValue& other) const
    {
        return !(*this == other);    
    }

    Dictionary::Dictionary()
        : m_dictionaryData(new unordered_map <wstring, DictionaryValue>)
    {
    }

    Dictionary::~Dictionary()
    {
    }

    Dictionary::Dictionary(const Dictionary& other)
    {
        *this = other;
    }

    Dictionary& Dictionary::operator=(const Dictionary& other)
    {
        assert(this != &other);
        m_dictionaryData.reset(new unordered_map<wstring, DictionaryValue>(*(other.m_dictionaryData)));
        return *this;
    }

    Dictionary::Dictionary(Dictionary&& other)
        : m_dictionaryData(nullptr)
    {
        *this = move(other);
    }

    Dictionary& Dictionary::operator=(Dictionary&& other)
    {
        assert(this != &other);

        m_dictionaryData = other.m_dictionaryData;
        other.m_dictionaryData = nullptr;

        return *this;
    }

    DictionaryValue& Dictionary::operator[](const wchar_t* key)
    {
        return (*m_dictionaryData)[key];
    }

    const DictionaryValue& Dictionary::operator[](const wchar_t* key) const
    {
        return m_dictionaryData->at(key);
    }

    bool Dictionary::Contains(const wchar_t* key) const
    {
        return (m_dictionaryData->find(key) != m_dictionaryData->end());
    }

    void Dictionary::Add(const Dictionary& other)
    {
        for (auto& kv : *(other.m_dictionaryData))
        {
            if (Contains(kv.first))
                InvalidArgument("Dictionary::Add: Already contains an entry with key %S being added from the 'other' dictionary", kv.first.c_str());

            (*this)[kv.first] = kv.second;
        }
    }

    bool Dictionary::operator==(const Dictionary& other) const
    {
        if (this == &other)
        {
            return true;
        }

        if (m_dictionaryData->size() != other.m_dictionaryData->size())
        {
            return false;
        }
        
        for (const auto& kv : *m_dictionaryData)
        {
            auto result = other.m_dictionaryData->find(kv.first);
            if (result == other.m_dictionaryData->end() || kv.second != result->second)
            {
                return false;
            }
        }

        return true;
    }
    
    bool Dictionary::operator!=(const Dictionary& other) const
    {
        return !(*this == other);    
    }

    std::pair<std::wstring, std::wstring> UidAndNameFromCNTKInternalNodeName(const std::wstring& CNTKInternalNodeName, const PrimitiveOpType& opType)
    {
        std::wstring uid, name;
        std::tie(uid, name) = UidAndNameFromCNTKInternalNodeName(CNTKInternalNodeName);
        if (uid == L"")
        {
            name = CNTKInternalNodeName;
            uid = GenerateUid(opType);
        }

        return{ uid, name };
    }

    template <typename T>
    TrainingParameterSchedule<T>::TrainingParameterSchedule(T value, UnitType unit) 
        : m_schedule({ make_pair(0, value) }), m_unit(unit), m_epochSize(FullDataSweep)
    {
    }

    template <typename T>
    TrainingParameterSchedule<T>::TrainingParameterSchedule(const vector<T>& schedule, UnitType unit, size_t epochSize) 
        : m_unit(unit), m_epochSize(epochSize)
    {
        std::vector<std::pair<size_t, T>> s(schedule.size());
        for (auto i = 0; i < schedule.size(); ++i)
        {
            s[i].first = 1;
            s[i].second = schedule[i];
        }
        ConstructSchedule(s);
    }

    template <typename T>
    TrainingParameterSchedule<T>::TrainingParameterSchedule(const vector<std::pair<size_t, T>>& schedule, UnitType unit, size_t epochSize)
        : m_unit(unit), m_epochSize(epochSize)
    {
        ConstructSchedule(schedule);
    }

    template <typename T>
    void TrainingParameterSchedule<T>::ConstructSchedule(const std::vector<std::pair<size_t, T>>& schedule)
    {
        // In case of the FullDataSweep, the scheduling unit is just 1 sweep, 
        // otherwise, it's the epoch size in samples.
        const auto unitSize = (m_epochSize == FullDataSweep) ? 1 : m_epochSize;

        if (schedule.size() == 0)
            RuntimeError("TrainingParameterSchedule::ConstructSchedule : schedule is empty.");

        size_t unitCount = 0;
        for (int i = 0; i < schedule.size(); ++i)
        {
            const auto& pair = schedule[i];
            // Unit count for all, but last element must be non-zero.
            if (i < (schedule.size() - 1) && pair.first == 0)
                RuntimeError("TrainingParameterSchedule::ConstructSchedule : unit count in the 'schedule' argument must not be 0.");

            unitCount += (pair.first != 0) ? pair.first : 1;
            m_schedule[unitSize * unitCount] = pair.second;
        }
    }

    template <typename T>
    /*virtual*/ TrainingParameterSchedule<T>::~TrainingParameterSchedule()
    {
    }

    // Returns the element whose key is greater than the required unit count 
    // or the last element if no such key exists.
    template <typename T>
    /*virtual*/ const T& TrainingParameterSchedule<T>::operator[](size_t count) const
    {
        assert(m_schedule.size() > 0);
        auto it = m_schedule.upper_bound(count);
        if (it == m_schedule.end())
        {
            --it;
        }
        return it->second;
    }

    template <typename T>
    TrainingParameterSchedule<T>::TrainingParameterSchedule(const TrainingParameterSchedule<T>&) = default;

    // cannot be defaulted due to a bug in VS2013 (https://connect.microsoft.com/VisualStudio/feedback/details/1255564)
    template <typename T>
    TrainingParameterSchedule<T>::TrainingParameterSchedule(TrainingParameterSchedule<T>&& that)
        :m_schedule(move(that.m_schedule)), m_unit(that.m_unit), m_epochSize(that.m_epochSize)
    {
    }

    template <typename T>
    TrainingParameterSchedule<T>& TrainingParameterSchedule<T>::operator=(const TrainingParameterSchedule<T>&) = default;

    // cannot be defaulted due to a bug in VS2013 (https://connect.microsoft.com/VisualStudio/feedback/details/1255564)
    template <typename T>
    TrainingParameterSchedule<T>& TrainingParameterSchedule<T>::operator=(TrainingParameterSchedule<T>&& that)
    {
        m_schedule = move(that.m_schedule);
        m_epochSize = that.m_epochSize;
        m_unit = that.m_unit;
        return *this;
    }

    static const std::wstring s_trainingParameterScheduleTypeValue = L"TrainingParameterSchedule";

    template <typename T>
    /*virtual*/ Dictionary TrainingParameterSchedule<T>::Serialize() const
    {
        Dictionary schedule;
        for (const auto& it : m_schedule)
        {
            schedule[std::to_wstring(it.first)] = DictionaryValue(it.second);
        }
        Dictionary dict;
        dict[versionKey] = CurrentVersion();
        dict[typeKey] = s_trainingParameterScheduleTypeValue;
        dict[epochSizeKey] = m_epochSize;
        dict[unitKey] = static_cast<size_t>(m_unit);
        dict[scheduleKey] = schedule;
        return dict;
    }

     template <typename T>
    /*static*/ TrainingParameterSchedule<T>  TrainingParameterSchedule<T>::Deserialize(const Dictionary& dict)
    {
        static const vector<std::wstring> s_requiredDictionaryKeys = { typeKey, unitKey, epochSizeKey, scheduleKey };

        ValidateDictionary<TrainingParameterSchedule<T>>(dict, s_requiredDictionaryKeys, s_trainingParameterScheduleTypeValue, s_serializationVersion);

        return TrainingParameterSchedule<T>(dict);
    }

    template <typename T>
    TrainingParameterSchedule<T>::TrainingParameterSchedule(const Dictionary& dictionary)
    {
        m_unit = UnitType(dictionary[unitKey].Value<size_t>());
        m_epochSize = dictionary[epochSizeKey].Value<size_t>();
        Dictionary schedule = dictionary[scheduleKey].Value<Dictionary>();
        for (const auto& kv : schedule)
        {
            m_schedule[std::stoll(kv.first)] = kv.second.Value<T>();
        }
    }

    void MomentumAsTimeConstantSchedule::ConvertToPerSampleValues()
    {
        for (auto& it : m_schedule)
        {
            double momTC = it.second;
            double momPS = momTC == 0.0 ? 0 : exp(-1.0 / momTC);
            it.second = momPS;
        }
    }

    std::shared_ptr<std::fstream> GetFstream(const std::wstring& filePath, bool readOnly)
    {
        if (!readOnly)
        {
            msra::files::make_intermediate_dirs(filePath.c_str());
        }

        std::shared_ptr<std::fstream> stream;
        std::ios_base::openmode mode = std::ios_base::binary | (readOnly ? std::ios_base::in : std::ios_base::out);
#ifdef _MSC_VER
        stream = std::make_shared<std::fstream>(filePath, mode);
#else
        stream = std::make_shared<std::fstream>(wtocharpath(filePath.c_str()).c_str(), mode);
#endif
        stream->exceptions(std::ios_base::badbit);
        if (stream->fail())
            RuntimeError("Cannot open file '%S' for %s.", filePath.c_str(), (readOnly ? "reading" : "writing"));

        return stream;
    }

    int GetFileDescriptor(const std::wstring& filePath, bool readOnly)
    {
        if (!readOnly)
        {
            msra::files::make_intermediate_dirs(filePath.c_str());
        }

        auto mode = (readOnly ? O_RDONLY : ( O_CREAT | O_WRONLY));
        int fd;
#ifdef _MSC_VER
        mode = mode | O_BINARY;
        fd = _wopen(filePath.c_str(), mode, 0644);
#else
        fd = open(ToString(filePath).c_str(), mode, 0644);
#endif
        if (fd < 0)
            RuntimeError("Cannot open file '%S' for %s.", filePath.c_str(), (readOnly ? "reading" : "writing"));

        return fd;
    }

    std::string ToString(const std::wstring& wstring)
    {
#ifdef _MSC_VER
        std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
        return converter.to_bytes(wstring);
#else
        const auto length = wstring.length() * sizeof(std::wstring::value_type) + 1;
        char buf[length];
        const auto res = std::wcstombs(buf, wstring.c_str(), sizeof(buf));
        return (res >= 0) ? buf : "";
#endif
    }

    std::wstring ToWString(const std::string& string)
    {
#ifdef _MSC_VER
        std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
        return converter.from_bytes(string);
#else
        const auto length = string.length() + 1;
        wchar_t buf[length];
        const auto res = std::mbstowcs(buf, string.c_str(),  sizeof(buf));
        return (res >= 0) ? buf : L"";
#endif
    }

    bool IsFirstOutputOfMultiOutputFunction(const Variable& var)
    {
        if (!var.IsOutput())
            return false;

        auto owner = var.Owner();
        return (var == owner->Outputs()[0]) && (owner->Outputs().size() > 1);
    }

    std::vector<Axis> DynamicAxesFromInternalDynamicAxisName(const std::wstring& internalDynamicAxisName)
    {
        std::vector<Axis> inputVarDynamicAxes;
        if (internalDynamicAxisName.substr(0, ComputationNodeBase::DefaultDynamicAxisName.length()) == ComputationNodeBase::DefaultDynamicAxisName)
            inputVarDynamicAxes = { Axis::DefaultDynamicAxis(), Axis::DefaultBatchAxis() };
        else if (internalDynamicAxisName.substr(0, ComputationNodeBase::DefaultNoSequenceAxisName.length()) == ComputationNodeBase::DefaultNoSequenceAxisName)
            inputVarDynamicAxes = { Axis::DefaultBatchAxis() };
        else
            inputVarDynamicAxes = { Axis(internalDynamicAxisName), Axis::DefaultBatchAxis() };

        return inputVarDynamicAxes;
    }

    // Construct the dynamic axis name to be used internally for the CNTK InputNodes
    std::wstring InternalDynamicAxisNameFromDynamicAxes(const std::vector<Axis>& dynamicAxes)
    {
        if (dynamicAxes.empty())
            LogicError("Empty dynamic axes set");

        if (dynamicAxes == std::vector<Axis>({ Axis::DefaultBatchAxis() }))
            return ComputationNodeBase::DefaultNoSequenceAxisName;
        else if (dynamicAxes == std::vector<Axis>({ Axis::DefaultDynamicAxis(), Axis::DefaultBatchAxis() }))
            return ComputationNodeBase::DefaultDynamicAxisName;
        else
            return dynamicAxes[0].Name();
    }

    bool IsPackedValue(const ValuePtr& value)
    {
        auto packedValue = dynamic_pointer_cast<PackedValue>(value);
        return (packedValue != nullptr) && packedValue->IsPacked();
    }
    std::pair<size_t, size_t> GetNumTimeStepsAndSequences(const NDShape& maskShape, size_t numDynamicAxes) 
    {
        size_t maxNumTimeSteps = 1;
        size_t numSequences = 1;
        if (maskShape.Rank() > 1)
        {
            // since only 2 axes are supported at the moment, sequence axis should be the first and batch axis -- the second.
            // sequence axis dimension determines the maximum number of time steps (= maximum sequence length),
            // batch axis dimension -- the number of sequences (= 'training units') in a batch.
            maxNumTimeSteps = maskShape[0];
            numSequences = maskShape[1];
        }
        else if (maskShape.Rank() > 0)
        {
            if (numDynamicAxes > 1)
            {
                maxNumTimeSteps = maskShape[0];
            }
            else
            {
                // there's only one axis (the default batch axis).
                numSequences = maskShape[0];
            }
        }

        return std::pair<size_t, size_t>(maxNumTimeSteps, numSequences);
    }

    /*static*/ void Utils::VerifyVariableValueCompatibility(const Variable& var, const ValuePtr& value, NDShape* inferredVarShape)
    {
        bool allowFreeDimensionsInVar = true;

        if (var.GetDataType() != value->GetDataType())
            LogicError("The Variable '%S' DataType %s does not match the corresponding Value's DataType %s", var.AsString().c_str(), DataTypeName(var.GetDataType()), DataTypeName(value->GetDataType()));

        // TODO: Is supplying dense data for an Input variable tagged as sparse, a fatal error even for packed value objects?
        bool isPackedValue = IsPackedValue(value);
        if (!isPackedValue)
        {
            if (IsSparseInput(var) && !value->IsSparse())
                InvalidArgument("Dense input data supplied for sparse input Variable '%S'.", var.AsString().c_str());

            if (IsSparseInput(var) && (value->GetStorageFormat() != StorageFormat::SparseCSC))
                InvalidArgument("Sparse Input data for Variable '%S' must be in SparseCSC format.", var.AsString().c_str());
        }

        auto varShape = var.Shape();
        auto valueShape = value->Shape();

        auto numDynamicAxes = var.DynamicAxes().size();
        if (numDynamicAxes > 2)
            LogicError("More than 2 dynamic axis for a variable '%S' is currently unsupported", var.AsString().c_str());

        // max(2, numDynamicAxes) is needed for some backcompat scenarios, where even when there are no sequence axes
        // the user can pass a value object with a dim of 1 for the sequence axis.
        // TODO: try and remove support for this in the future, change the condition below to
        // valueShape.Rank() - varShape.Rank() <=  var.DynamicAxes().size()
        size_t maxAddionalValueAxes = std::max<size_t>(2, numDynamicAxes);

        // For packed values, we sometimes have the reader return the matrix with a flattened sample layout
        if (isPackedValue &&
            ((valueShape.Rank() < varShape.Rank()) || (valueShape.SubShape(0, varShape.Rank()) != varShape)) &&
            (valueShape.SubShape(1).Rank() <= maxAddionalValueAxes))
        {
            // If the leading dim of the value shape is same as the total size of the varShape,
            // lets expand the leading dim to varShape for the purposes of the rest of the validation
            if (allowFreeDimensionsInVar && varShape.HasFreeDimension())
            {
                auto newVarShape = varShape;
                for (size_t i = 0; i < newVarShape.Rank(); ++i)
                    if (newVarShape[i] == NDShape::FreeDimension)
                        newVarShape[i] = NDShape::InferredDimension;

                PrimitiveFunction::ReshapeOutputShape({ valueShape[0] }, newVarShape, Axis(0), Axis(1), /*inferDimensions =*/ true);
                valueShape = newVarShape.AppendShape(valueShape.SubShape(1));
            }
            else if (valueShape[0] == varShape.TotalSize())
                valueShape = varShape.AppendShape(valueShape.SubShape(1));
        }

        if (valueShape.Rank() < varShape.Rank())
            InvalidArgument("Value's rank (%d) should be >= the Variable's rank (%d); Variable = '%S', Value shape = '%S'.", 
                            (int)valueShape.Rank(), (int)varShape.Rank(), var.AsString().c_str(), valueShape.AsString().c_str());

        if (valueShape.Rank() > (varShape.Rank() + maxAddionalValueAxes))
            InvalidArgument("Value rank (%d) should be larger than the Variable rank (%d) at most by number of dynamic axes (%d); Variable = '%S', Value shape = '%S'.",
                            (int)valueShape.Rank(), (int)varShape.Rank(), (int)numDynamicAxes, var.AsString().c_str(), valueShape.AsString().c_str());

        if (valueShape.Rank() > (varShape.Rank() + numDynamicAxes))
        {
            for (size_t i = 0; i < (maxAddionalValueAxes - numDynamicAxes); ++i)
            {
                if (valueShape[varShape.Rank() + i] != 1)
                    InvalidArgument("Value rank (%d) should be larger than the Variable rank (%d) at most by number of dynamic axes (%d); Variable = '%S', Value shape = '%S'.",
                                    (int)valueShape.Rank(), (int)varShape.Rank(), (int)numDynamicAxes, var.AsString().c_str(), valueShape.AsString().c_str());
            }
        }

        auto valueVarSubshape = valueShape.SubShape(0, varShape.Rank());
        if (valueVarSubshape != varShape)
        {
            for (size_t i = 0; i < varShape.Rank(); ++i)
            {
                if (allowFreeDimensionsInVar && (varShape[i] == NDShape::FreeDimension))
                    varShape[i] = valueVarSubshape[i];
                else if (varShape[i] != valueVarSubshape[i])
                {
                    InvalidArgument("The %s dimensions of the Value shape '%S' do not match the Variable '%S' shape '%S'.",
                                    Internal::IsReversingTensorShapesInErrorMessagesEnabled() ? "trailing" : "leading",
                                    valueShape.AsString().c_str(),
                                    var.AsString().c_str(),
                                    varShape.AsString().c_str());
                }
            }
        }

        if (!isPackedValue)
        {
            auto mask = value->Mask();
            if ((mask != nullptr) && ((varShape.Rank() + mask->Shape().Rank()) != valueShape.Rank()))
            {
                InvalidArgument("Invalid Value object: sum of the rank (%d) of the mask and Variable rank (%d) does not equal "
                                "the Value's rank (%d); Variable = '%S', Value shape = '%S'.",
                                (int)mask->Shape().Rank(), (int)varShape.Rank(), (int)valueShape.Rank(), var.AsString().c_str(), valueShape.AsString().c_str());
            }
        }

        if (inferredVarShape)
        {
            if (varShape.HasFreeDimension())
                InvalidArgument("At least one of the free dimensions of Variable '%S' could not be resolved from the supplied value.", varShape.AsString().c_str());

            *inferredVarShape = varShape;
        }
    }

    template <typename ElementType>
    std::pair<std::shared_ptr<const Matrix<ElementType>>, MBLayoutPtr> Utils::GetCNTKImplMatrixAndMBLayoutFromValueObject(const Variable& var, const ValuePtr& value, NDShape* inferredVarShape,
                                                                                                                          const std::shared_ptr<Matrix<ElementType>>& outputMatrixStorage,
                                                                                                                          const std::shared_ptr<Matrix<ElementType>>& tempIndicesStorage)
    {
        VerifyVariableValueCompatibility(var, value, inferredVarShape);

        if (AsDataType<ElementType>() != value->GetDataType())
            LogicError("The specified ElementType %s does not match the Value object's DataType %s for Variable '%S'",
                        typeid(ElementType).name(), DataTypeName(value->GetDataType()), var.AsString().c_str());

        auto packedValue = dynamic_cast<PackedValue*>(value.get());
        if (packedValue && packedValue->IsPacked())
        {
            auto packedMatrixAndLayout = packedValue->PackedData<ElementType>();
            if (!var.DynamicAxes().empty() && (packedMatrixAndLayout.second == nullptr))
            {
                auto layout = std::make_shared<MBLayout>();
                layout->InitAsFrameMode(1);
                packedMatrixAndLayout.second = layout;
            }

            return packedMatrixAndLayout;
        }

        auto valueShape = value->Shape();
        auto varShape = inferredVarShape ? *inferredVarShape : valueShape.SubShape(0, var.Shape().Rank());
        auto numDynamicAxes = var.DynamicAxes().size();
        auto mask = value->Mask();

        if (numDynamicAxes == 0)
            return{ value->Data()->GetMatrix<ElementType>(), nullptr };

        size_t maxNumTimeSteps, numSequences;
        std::tie(maxNumTimeSteps, numSequences) = GetNumTimeStepsAndSequences(valueShape.SubShape(varShape.Rank()), numDynamicAxes);

        if ((numSequences == 1) || (maxNumTimeSteps == 1))
        {
            // The data need not be shuffled
            std::shared_ptr<const Matrix<ElementType>> matrixData = value->Data()->GetMatrix<ElementType>(VariableRowColSplitPoint(var));
            auto layout = std::make_shared<MBLayout>();
            if (!mask)
            {
                if (maxNumTimeSteps == 1)
                    layout->InitAsFrameMode(numSequences);
                else
                {
                    layout->Init(numSequences, maxNumTimeSteps);
                    layout->AddSequence(0, 0, 0, maxNumTimeSteps);
                }
            }
            else
            {
                layout->Init(numSequences, maxNumTimeSteps);

                std::vector<ptrdiff_t> sequenceBeginIndices(numSequences, 0);
                std::vector<size_t> sequenceLengths(numSequences, maxNumTimeSteps);
                Value::GetSequenceStartsAndLengths(mask, sequenceBeginIndices, sequenceLengths, numDynamicAxes);

                for (size_t i = 0; i < numSequences; ++i)
                    layout->AddSequence(i, i, sequenceBeginIndices[i], sequenceLengths[i]);
            }

            return{ matrixData, layout };
        }
        else
        {
            std::vector<ptrdiff_t> sequenceBeginIndices(numSequences, 0);
            std::vector<size_t> sequenceLengths(numSequences, maxNumTimeSteps);
            if (mask != nullptr)
                Value::GetSequenceStartsAndLengths(mask, sequenceBeginIndices, sequenceLengths, numDynamicAxes);

            bool hasTruncatedSequences = std::find_if(sequenceBeginIndices.begin(), sequenceBeginIndices.end(), [](const int& val) { return (val < 0); }) != sequenceBeginIndices.end();

            auto layout = std::make_shared<MBLayout>();
            std::vector<std::pair<size_t, size_t>> placement;
            if (!hasTruncatedSequences)
            {
                std::vector<MBLayout::SequenceInfo> sequences;
                for (size_t i = 0; i < numSequences; ++i)
                    sequences.push_back({ i, SIZE_MAX, sequenceBeginIndices[i], sequenceLengths[i] });

                std::vector<size_t> rowAllocations;
                layout->InitAsPackedSequences(sequences, placement, rowAllocations);
            }
            else
            {
                layout->Init(numSequences, maxNumTimeSteps);

                // We cannot pack as some of the sequences are truncated and thus all sequences have to be
                // kept in their original parallel streams
                placement.resize(numSequences);
                for (size_t i = 0; i < numSequences; ++i)
                {
                    layout->AddSequence(i, i, sequenceBeginIndices[i], sequenceLengths[i]);

                    // Add the gap if there is one
                    if (sequenceLengths[i] < maxNumTimeSteps)
                        layout->AddSequence(GAP_SEQUENCE_ID, i, sequenceLengths[i], maxNumTimeSteps);

                    placement[i] = std::make_pair(i, 0);
                }
            }

            if (maxNumTimeSteps != layout->GetNumTimeSteps())
                LogicError("The number (%d) of time steps in the packed MBLayout does not match the longest sequence's length (%d) in the Value object", (int)maxNumTimeSteps, (int)layout->GetNumTimeSteps());

            if (numSequences != layout->GetNumSequences())
                LogicError("The number (%d) of sequences in the packed MBLayout does not match the sequence count (%d) in the Value object.", (int)numSequences, (int)layout->GetNumSequences());

            // The data needs to be rearranged since CNTK requires sequences to be interleaved across timesteps
            // Now generate the gather indices
            auto numColsPerSample = varShape.SubShape(VariableRowColSplitPoint(var)).TotalSize();
            std::shared_ptr<Matrix<ElementType>> matrixData = outputMatrixStorage;
            auto matrixDataNumRows = varShape.TotalSize() / numColsPerSample;
            auto matrixDataNumCols = layout->GetNumCols() * numColsPerSample;
            auto matrixType = value->IsSparse() ? MatrixType::SPARSE : MatrixType::DENSE;
            auto matrixFormat = AsCNTKImplMatrixFormat(value->GetStorageFormat());
            if (!matrixData)
                matrixData = std::make_shared<Matrix<ElementType>>(matrixDataNumRows, matrixDataNumCols, AsCNTKImplDeviceId(value->Device()), matrixType, matrixFormat);
            else
            {
                matrixData->SwitchToMatrixType(matrixType, matrixFormat, /*keepValues=*/false);
                matrixData->Resize(matrixDataNumRows, matrixDataNumCols);
            }

            std::vector<size_t> sequencesShorterThanLongestSequence;
            for (size_t i = 0; i < numSequences; ++i)
                if (sequenceLengths[i] != maxNumTimeSteps)
                    sequencesShorterThanLongestSequence.push_back(i);

            // Set the source location for all gaps to be the last step of the first sequence that is shorter than the longest sequence in the batch
            size_t sourceColIdxForInvalidColumns = sequencesShorterThanLongestSequence.empty() ? 0 : (((sequencesShorterThanLongestSequence[0] + 1) * maxNumTimeSteps) - 1);
            std::vector<ElementType> gatherIndicesVector(matrixData->GetNumCols(), (ElementType)sourceColIdxForInvalidColumns);
            for (size_t i = 0; i < numSequences; ++i)
            {
                size_t targetParallelStreamIdx = placement[i].first;
                size_t targetStartIdxInParallelStream = placement[i].second;
                for (size_t j = 0; j < sequenceLengths[i]; ++j)
                    for (size_t k = 0; k < numColsPerSample; ++k)
                        gatherIndicesVector[((((targetStartIdxInParallelStream + j) * layout->GetNumParallelSequences()) + targetParallelStreamIdx) * numColsPerSample) + k] = (ElementType)((((i * maxNumTimeSteps) + j) * numColsPerSample) + k);
            }

            auto gatherIdxMatrix = tempIndicesStorage;
            if (!gatherIdxMatrix)
                gatherIdxMatrix = std::make_shared<Matrix<ElementType>>(1, gatherIndicesVector.size(), gatherIndicesVector.data(), AsCNTKImplDeviceId(value->Device()));
            else
                gatherIdxMatrix->SetValue(1, gatherIndicesVector.size(), AsCNTKImplDeviceId(value->Device()), gatherIndicesVector.data());

            matrixData->DoGatherColumnsOf(0, *gatherIdxMatrix, *(value->Data()->GetMatrix<ElementType>(VariableRowColSplitPoint(var))), 1);
            return{ matrixData, layout };
        }
    }

    template <typename ElementType>
    ValuePtr Utils::GetValueObjectFromCNTKImplMatrixAndMBLayout(const NDShape& sampleShape, const std::vector<Axis>& sampleDynamicAxes, const Matrix<ElementType>& matrix, const MBLayoutPtr& layout, bool readOnly /*= true*/)
    {
        auto CreateMask = [](const MBLayoutPtr& layout, const DeviceDescriptor& device) {
            std::vector<bool> sequenceBeginFlags;
            std::vector<size_t> sequenceLengths;
            std::vector<size_t> sequencesShorterThanLongestSequence;

            size_t maxNumTimeSteps = layout->GetNumTimeSteps();
            size_t numSequences = layout->GetNumSequences();
            auto& layoutSequences = layout->GetAllSequences();

            size_t sequenceIdx = 0;
            bool allSequencesStartInThisMB = true;
            bool allSequencesSameLength = true;
            for (auto sequenceInfo : layoutSequences)
            {
                if (sequenceInfo.seqId != GAP_SEQUENCE_ID)
                {
                    auto currentSequenceBeginIdx = std::max<ptrdiff_t>(0, sequenceInfo.tBegin);
                    auto currentSequenceEndIdx = std::min(maxNumTimeSteps, sequenceInfo.tEnd);
                    auto currentSequenceLength = (currentSequenceEndIdx - currentSequenceBeginIdx);
                    auto isCurrentSequenceBeginningInsideThisMB = sequenceInfo.tBegin >= 0;

                    allSequencesStartInThisMB = allSequencesStartInThisMB && isCurrentSequenceBeginningInsideThisMB;
                    allSequencesSameLength = allSequencesSameLength && (currentSequenceLength == maxNumTimeSteps);

                    sequenceBeginFlags.push_back(isCurrentSequenceBeginningInsideThisMB);
                    sequenceLengths.push_back(currentSequenceLength);

                    if (currentSequenceLength != maxNumTimeSteps)
                        sequencesShorterThanLongestSequence.push_back(sequenceIdx);

                    sequenceIdx++;
                }
            }

            if (!allSequencesStartInThisMB && (numSequences != layout->GetNumParallelSequences()))
                LogicError("Cannot create an unpacked Value object from packed data where one or more sequences are truncated");

            bool maskNeeded = !allSequencesSameLength || !allSequencesStartInThisMB;

            NDMaskPtr mask;
            if (maskNeeded)
            {
                mask = MakeSharedObject<NDMask>(NDShape({ maxNumTimeSteps, numSequences }), DeviceDescriptor::CPUDevice());
                for (size_t i = 0; i < numSequences; ++i)
                    if (sequenceBeginFlags[i])
                        mask->MarkSequenceBegin({ 0, i });

                for (auto shortSequenceIdx : sequencesShorterThanLongestSequence)
                    mask->InvalidateSection({ sequenceLengths[shortSequenceIdx], shortSequenceIdx }, { NDShape::InferredDimension, 1 });
            }

            return mask;
        };

        // No data shuffling needed if no layout or the layout has just one time-step or just one sequence
        NDMaskPtr mask;
        if (layout != nullptr)
            mask = CreateMask(layout, AsDeviceDescriptor(matrix.GetDeviceId()));

        // Reshuffle to data to unpack and uninterleave the CNTK form packed data
        auto unpackedTensorView = ComputationNode<ElementType>::Unpack(AsTensorShape(sampleShape), matrix, layout, /*batchMajor=*/ false, /*gapPadValue=*/ nullptr);
        auto dataShape = PackedValue::GetUnpackedShape(sampleShape, sampleDynamicAxes, layout);
        auto data = MakeSharedObject<NDArrayView>(AsDataType<ElementType>(), AsDeviceDescriptor(matrix.GetDeviceId()), AsStorageFormat(matrix.GetFormat()), dataShape, readOnly, new TensorView<ElementType>(unpackedTensorView, AsTensorViewShape(dataShape)));
        return MakeSharedObject<Value>(data, mask);
    }

    template <typename ElementType>
    ValuePtr Utils::GetValueObjectFromCNTKImplMatrixAndMBLayout(const Variable& var, const ComputationNodeBasePtr& computationNode, const Matrix<ElementType>& matrix, const MBLayoutPtr& layout, bool readOnly /*= true*/)
    {
        if (var.DynamicAxes().size() > 2)
            LogicError("More than 2 dynamic axes for a variable '%S' is currently unsupported", var.AsString().c_str());

        if (AsDataType<ElementType>() != var.GetDataType())
            LogicError("The specified ElementType %s of Variable '%S' does not match the DataType %s", typeid(ElementType).name(), var.AsString().c_str(), DataTypeName(var.GetDataType()));

        if ((layout != nullptr) && (matrix.GetNumRows() != var.Shape().TotalSize()))
            LogicError("Unexpected matrix layout: The number (%d) of rows in the matrix does not match the sample size (%d) of the Variable '%S'", (int)matrix.GetNumRows(), (int)var.Shape().TotalSize(), var.AsString().c_str());

        auto varShape = var.Shape();
        if (computationNode)
            varShape = GetVariableShape(var.Shape(), computationNode->GetSampleLayout());

        return GetValueObjectFromCNTKImplMatrixAndMBLayout(varShape, var.DynamicAxes(), matrix, layout, readOnly);
    }

    NDMaskPtr CreateMask(const std::vector<size_t>& sequenceLengths, const std::vector<bool>& sequenceStartFlags, const DeviceDescriptor& device)
    {
        size_t numSequences = sequenceLengths.size();

        if (!sequenceStartFlags.empty() && (sequenceStartFlags.size() != numSequences))
            InvalidArgument("Value::Create:: The number (%zu) of sequence start flags does not match the number (%zu) of sequences,",
                sequenceStartFlags.size(), numSequences);

        std::vector<bool> actualStarts = sequenceStartFlags;
        if (actualStarts.empty())
            actualStarts.resize(numSequences, true);

        size_t maxSequenceLength = 0;
        for (size_t i = 0; i < numSequences; ++i)
            maxSequenceLength = std::max(maxSequenceLength, sequenceLengths[i]);

        bool needsMask = (std::find(actualStarts.begin(), actualStarts.end(), false) != actualStarts.end());
        needsMask = needsMask || (std::find_if(sequenceLengths.begin(), sequenceLengths.end(), [maxSequenceLength](const size_t& currentSequenceLength) {
            return (currentSequenceLength != maxSequenceLength);
        }) != sequenceLengths.end());

        // If needed, create a mask to account for variability in lengths of specified sequences
        NDMaskPtr deviceValueMask;
        if (needsMask)
        {
            NDShape valueMaskShape = { maxSequenceLength, numSequences };
            deviceValueMask = MakeSharedObject<NDMask>(valueMaskShape, device);
            for (size_t i = 0; i < numSequences; ++i)
            {
                if (actualStarts[i])
                    deviceValueMask->MarkSequenceBegin({ 0, i });
                deviceValueMask->InvalidateSection({ sequenceLengths[i], i }, { NDShape::InferredDimension, 1 });
            }
        }

        return deviceValueMask;
    }

    double ReductionIdentityValue(const std::wstring& reductionOpName)
    {
        auto reductionOpEnumValue = ReduceElementsNode<double>::ReductionOpEnumValue(reductionOpName);
        return ReduceElementsNode<double>::NeutralValue(reductionOpEnumValue);
    }

    template void DictionaryValue::AllocateDataPtr<NDShape>(const NDShape& value);
    template void DictionaryValue::AllocateDataPtr<Axis>(const Axis& value);
    template void DictionaryValue::AllocateDataPtr<vector<DictionaryValue>>(const vector<DictionaryValue>& value);
    template void DictionaryValue::AllocateDataPtr<wstring>(const wstring& value);
    template void DictionaryValue::AllocateDataPtr<Dictionary>(const Dictionary& value);
    template void DictionaryValue::AllocateDataPtr<NDArrayView>(const NDArrayView& value);

    template void DictionaryValue::FreePtrAsType<NDShape>();
    template void DictionaryValue::FreePtrAsType<Axis>();
    template void DictionaryValue::FreePtrAsType<vector<DictionaryValue>>();
    template void DictionaryValue::FreePtrAsType<wstring>();
    template void DictionaryValue::FreePtrAsType<Dictionary>();
    template void DictionaryValue::FreePtrAsType<NDArrayView>();

    template class TrainingParameterSchedule<double>;
    template class TrainingParameterSchedule<size_t>;

    Learners::Learners(const std::vector<LearnerPtr>& learners) :
        m_learners(learners),
        m_isDistributed(false)
    {
        if (learners.empty())
            InvalidArgument("These must be at least one learner.");

        std::unordered_set<Parameter> learnerParameters;
        for (const auto& learner : m_learners)
        {
            if (dynamic_pointer_cast<DistributedLearner>(learner) != nullptr)
                m_isDistributed = true;

            const auto& currentLearnerParameters = learner->Parameters();
            for (const auto& parameter : currentLearnerParameters)
            {
                auto insertRetVal = learnerParameters.insert(parameter);
                if (!insertRetVal.second)
                    InvalidArgument("Parameter '%S' is covered by 2 different learners", parameter.AsString().c_str());
            }
        }

        if (m_isDistributed)
            CheckDistributedLearners();
    }

    void Learners::CheckDistributedLearners()
    {
        for (const auto& learner : m_learners)
        {
            if (dynamic_pointer_cast<DistributedLearner>(learner) == nullptr)
                InvalidArgument("Cannot use a non-distributed learner for some parameters together with a distributed learner for other parameters, in a single Trainer.");
        }
    }

    void Learners::GetLearnerGradients(LearnerPtr learner, const std::unordered_map<Parameter, NDArrayViewPtr>& allGradients, std::unordered_map<Parameter, NDArrayViewPtr>& learnerGradients)
    {
        const auto& learnerParameters = learner->Parameters();
        for (const auto& parameter : learnerParameters)
        {
            auto value = allGradients.find(parameter);
            if (value == allGradients.end())
                LogicError("Learner contains parameter '%S' that does not exists in the model.", parameter.AsString().c_str());

            learnerGradients[parameter] = value->second;
        }
    }

    bool Learners::Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, size_t sampleInMinibatch, bool sweepEnd)
    {
        bool anyUpdatesPerformed = false;
        for (auto learner : m_learners)
        {
            std::unordered_map<Parameter, NDArrayViewPtr> learnerGradients;
            GetLearnerGradients(learner, gradientValues, learnerGradients);
            anyUpdatesPerformed |= learner->Update(learnerGradients, sampleInMinibatch, sweepEnd);
        }
        return anyUpdatesPerformed;
    }

    bool Learners::Update(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, MinibatchInfo& minibatch)
    {
        bool anyUpdatesPerformed = false;
        for (auto l : m_learners)
        {
            auto learner = dynamic_pointer_cast<DistributedLearner>(l);
            std::unordered_map<Parameter, NDArrayViewPtr> learnerGradients;
            GetLearnerGradients(learner, gradientValues, learnerGradients);
            anyUpdatesPerformed |= learner->Update(learnerGradients, minibatch);
        }
        return anyUpdatesPerformed;
    }

    std::vector<DictionaryValue> Learners::CreateCheckpoint()
    {
        std::vector<DictionaryValue> state;
        for (auto l : m_learners)
            state.push_back(l->CreateCheckpoint());
        return state;
    }

    void Learners::RestoreFromCheckpoint(const std::vector<DictionaryValue>& state)
    {
        if (m_learners.size() != state.size())
            RuntimeError("RestoreFromCheckpoint: Number of learners (%zu) does not match learner count in the checkpoint (%zu).", m_learners.size(), state.size());

        for (size_t i = 0; i < m_learners.size(); ++i)
        {
            m_learners[i]->RestoreFromCheckpoint(state[i].Value<Dictionary>());
        }
    }

    template std::pair<std::shared_ptr<const Matrix<float>>, MBLayoutPtr> Utils::GetCNTKImplMatrixAndMBLayoutFromValueObject<float>(const Variable& var, const ValuePtr& value, NDShape* inferredVarShape);
    template std::pair<std::shared_ptr<const Matrix<double>>, MBLayoutPtr> Utils::GetCNTKImplMatrixAndMBLayoutFromValueObject<double>(const Variable& var, const ValuePtr& value, NDShape* inferredVarShape);

    template ValuePtr Utils::GetValueObjectFromCNTKImplMatrixAndMBLayout<float>(const NDShape& sampleShape, const std::vector<Axis>& sampleDynamicAxes, const Matrix<float>& matrix, const MBLayoutPtr& layout, bool readOnly /*= true*/);
    template ValuePtr Utils::GetValueObjectFromCNTKImplMatrixAndMBLayout<double>(const NDShape& sampleShape, const std::vector<Axis>& sampleDynamicAxes, const Matrix<double>& matrix, const MBLayoutPtr& layout, bool readOnly /*= true*/);

    template ValuePtr Utils::GetValueObjectFromCNTKImplMatrixAndMBLayout<float>(const Variable& var, const ComputationNodeBasePtr& computationNode, const Matrix<float>& matrix, const MBLayoutPtr& layout, bool readOnly /*= true*/);
    template ValuePtr Utils::GetValueObjectFromCNTKImplMatrixAndMBLayout<double>(const Variable& var, const ComputationNodeBasePtr& computationNode, const Matrix<double>& matrix, const MBLayoutPtr& layout, bool readOnly /*= true*/);

    void Accumulator::Update(const ValuePtr& delta, const DeviceDescriptor& device)
    {
        if (!delta)
            InvalidArgument("Attempting to accumulate a null Value.");

        bool copied = false;
        if (m_isUninitialized ||
            GetDataType() != delta->GetDataType() ||
            Shape() != delta->Shape() ||
            Device() != device ||
            Mask() != delta->Mask())
        {
            copied = true;
            m_data = MakeSharedObject<NDArrayView>(delta->GetDataType(), delta->Shape(), device);
            m_mask = delta->Mask();
            ResetToZero();
            m_isUninitialized = false;
        }

        if (delta->GetDataType() == DataType::Float)
            Data()->GetWritableTensorView<float>()->AddCopyOf(*delta->Data()->GetTensorView<float>());
        else
            Data()->GetWritableTensorView<double>()->AddCopyOf(*delta->Data()->GetTensorView<double>());

        if (copied && m_numUpdates != 0)
            RuntimeError("Accumulation values are created when accumulated num updates not zero");

        m_numUpdates++;
    }

    void Accumulator::Reset()
    {
        ResetToZero();
        m_numUpdates = 0;
    }

    void Accumulator::ResetToZero()
    {
        if (m_isUninitialized)
            return;

        if (GetDataType() == DataType::Float)
            Data()->SetValue(0.0f);
        else
            Data()->SetValue(0.0);
    }

    std::wstring DynamicAxesAsString(const std::vector<Axis>& axes, bool rowMajor)
    {
        auto da = axes;
        if (da.size() == 0)
            return L"[]";
        std::wstringstream wss;
        wss << "[";
        if (da == Axis::UnknownDynamicAxes())
            wss << "???";
        else
        {
            if (rowMajor)
                std::reverse(da.begin(), da.end());
            bool first = true;
            for (auto d : da)
            {
                wss << (first ? "" : ", ");
                if (d == Axis::DefaultBatchAxis())
                    wss << "#";
                else if (d == Axis::DefaultDynamicAxis())
                    wss << "*";
                else
                    wss << d.Name();
                first = false;
            }
        }
        wss << "]";
        return wss.str();
    }
}
