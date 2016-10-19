//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "Serialization.h"
#include "Function.h"

using namespace std;

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
        for (auto kv : *(other.m_dictionaryData))
        {
            if (Contains(kv.first))
                InvalidArgument("Dictionary::Add: This dictionary already contains an entry with key %S that is being attempted to add from the 'other' dinctionary", kv.first.c_str());

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
    TrainingParameterSchedule<T>::TrainingParameterSchedule(T value) 
        : m_schedule({ make_pair(0, value) }), m_unit(1)
    {
    }

    template <typename T>
    TrainingParameterSchedule<T>::TrainingParameterSchedule(const vector<T>& schedule, size_t unit) 
        : m_unit(unit)
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
    TrainingParameterSchedule<T>::TrainingParameterSchedule(const vector<std::pair<size_t, T>>& schedule, size_t unit)
        : m_unit(unit)
    {
        ConstructSchedule(schedule);
    }

    template <typename T>
    void TrainingParameterSchedule<T>::ConstructSchedule(const std::vector<std::pair<size_t, T>>& schedule)
    {
        // TODO: 0 will be used to mean "the entire sweep"
        if (m_unit == 0)
            RuntimeError("TrainingParameterSchedule::ConstructSchedule : 'unit' cannot be 0.");

        if (schedule.size() == 0)
            RuntimeError("TrainingParameterSchedule::ConstructSchedule : schedule is empty.");

        size_t count = 0;
        for (int i = 0; i < schedule.size(); ++i)
        {
            const auto& pair = schedule[i];
            // Unit count for all, but last element must be non-zero.
            if (i < (schedule.size() - 1) && pair.first == 0)
                RuntimeError("TrainingParameterSchedule::ConstructSchedule : unit count cannot be 0.");

            count += pair.first;
            m_schedule[m_unit * count] = pair.second;
        }
    }

    template <typename T>
    /*virtual*/ TrainingParameterSchedule<T>::~TrainingParameterSchedule()
    {
    }

    // Returns the element whose key is greater than the required sample count 
    // or the last element if no such key exists.
    template <typename T>
    /*virtual*/ const T& TrainingParameterSchedule<T>::operator[](size_t sampleCount) const
    {
        assert(m_schedule.size() > 0);
        auto it = m_schedule.upper_bound(sampleCount);
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
        :m_schedule(move(that.m_schedule)), m_unit(that.m_unit)
    {
    }

    template <typename T>
    TrainingParameterSchedule<T>& TrainingParameterSchedule<T>::operator=(const TrainingParameterSchedule<T>&) = default;

    // cannot be defaulted due to a bug in VS2013 (https://connect.microsoft.com/VisualStudio/feedback/details/1255564)
    template <typename T>
    TrainingParameterSchedule<T>& TrainingParameterSchedule<T>::operator=(TrainingParameterSchedule<T>&& that)
    {
        m_schedule = move(that.m_schedule);
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
        dict[unitKey] = m_unit;
        dict[scheduleKey] = schedule;
        return dict;
    }

     template <typename T>
    /*static*/ TrainingParameterSchedule<T>  TrainingParameterSchedule<T>::Deserialize(const Dictionary& dict)
    {
        static const vector<std::wstring> s_requiredDictionaryKeys = { typeKey, unitKey, scheduleKey };

        ValidateDictionary<TrainingParameterSchedule<T>>(dict, s_requiredDictionaryKeys, s_trainingParameterScheduleTypeValue, s_serializationVersion);

        return TrainingParameterSchedule<T>(dict);
    }

    template <typename T>
    TrainingParameterSchedule<T>::TrainingParameterSchedule(const Dictionary& dictionary)
    {
        m_unit = dictionary[unitKey].Value<size_t>();
        Dictionary schedule = dictionary[scheduleKey].Value<Dictionary>();
        vector<std::wstring> keys = schedule.Keys();
        std::map<size_t, T> map;
        for (const auto& key : keys)
        {
            m_schedule[std::stoll(key)] = schedule[key].Value<T>();
        }
    }

    void MomentumValuesAsTimeConstants::ConvertToPerSampleValues()
    {
        for (auto& it : m_schedule)
        {
            double momTC = it.second;
            double momPS = momTC == 0.0 ? 0 : exp(-1.0 / momTC);
            it.second = momPS;
        }
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
}
