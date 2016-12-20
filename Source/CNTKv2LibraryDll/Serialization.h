//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"

namespace CNTK
{
    const std::wstring versionKey = L"version";
    const std::wstring typeKey = L"type";
    const std::wstring uidKey = L"uid";
    const std::wstring kindKey = L"kind";
    const std::wstring dataTypeKey = L"data_type";
    const std::wstring dynamicAxisKey = L"dynamic_axis";
    const std::wstring isSparseKey = L"is_sparse";
    const std::wstring nameKey = L"name";
    const std::wstring needsGradientKey = L"needs_gradient";
    const std::wstring shapeKey = L"shape";
    const std::wstring valueKey = L"value";
    const std::wstring opKey = L"op";
    const std::wstring attributesKey = L"attributes";
    const std::wstring inputsKey = L"inputs";
    const std::wstring rootKey = L"root";
    const std::wstring functionsKey = L"primitive_functions";
    const std::wstring sampleCountKey = L"sample_count";
    const std::wstring minibatchCountKey = L"minibatchCount";
    const std::wstring unitKey = L"unit";
    const std::wstring epochSizeKey = L"epoch_size";
    const std::wstring scheduleKey = L"schedule";
    const std::wstring learningRateScheduleKey = L"learnig_rate_schedule";
    const std::wstring stateKey = L"state";
    const std::wstring rngSeedKey = L"rng_seed";
    const std::wstring rngOffsetKey = L"rng_offset";
    const std::wstring blockFunctionCompositeKey = L"block_function_composite";
    const std::wstring blockFunctionOpNameKey = L"block_function_op_name";
    const std::wstring blockFunctionCompositeArgumentsMapKey = L"block_function_composite_arguments_map";

    template <typename T> 
    inline std::string GetVersionsString(size_t currentVersion, size_t dictVersion)
    {
        std::stringstream info;
        info << "Current " << Typename<T>() << " version = " << currentVersion 
             << ", Dictionary version = " << dictVersion;
        return info.str();
    }

    inline size_t GetVersion(const Dictionary& dict)
    {
        if (!dict.Contains(versionKey))
        {
             LogicError("Required key '%ls' is not found in the dictionary.", versionKey.c_str());
        } 
        return dict[versionKey].Value<size_t>();
    }

    template <typename T>
    inline void ValidateType(const Dictionary& dict, const std::wstring& typeValue, size_t currentVersion)
    {
        if (!dict.Contains(typeKey))
        {
            const auto& version = GetVersion(dict);
            LogicError("Required key '%ls' is not found in the dictionary "
                            "(%s).", typeKey.c_str(), GetVersionsString<T>(currentVersion, version).c_str());
        } 

        const auto& type = dict[typeKey].Value<std::wstring>();
        if (type != typeValue) 
        {
            const auto& version = GetVersion(dict);
            LogicError("Unexpected '%ls':'%ls' in place of '%ls':'%ls' "
                        "(%s).", typeKey.c_str(), type.c_str(), typeKey.c_str(), typeValue.c_str(), GetVersionsString<T>(currentVersion, version).c_str());
        }
    }

    // Make sure that the dictionary contains all required keys, and if it does, return version value
    // from the dictionary.
    template <typename T>
    inline size_t ValidateDictionary(const Dictionary& dict, const std::vector<std::wstring>& requiredKeys, const std::wstring& typeValue, size_t currentVersion)
    { 
        const auto& version = GetVersion(dict);

        for (const auto& key : requiredKeys)
        {
            if (!dict.Contains(key))
            {
                 LogicError("Required key '%ls' is not found in the dictionary "
                            "(%s).", key.c_str(), GetVersionsString<T>(currentVersion, version).c_str());
            }
        }

        ValidateType<T>(dict, typeValue, currentVersion);

        return version;
    }
}
