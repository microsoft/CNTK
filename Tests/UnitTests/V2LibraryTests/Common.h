#pragma once

#include <exception>
#include <algorithm>
#include "CNTKLibrary.h"

static const double relativeTolerance = 0.001f;
static const double absoluteTolerance = 0.000001f;

template <typename ElementType>
inline void FloatingPointVectorCompare(const std::vector<ElementType>& first, const std::vector<ElementType>& second, const char* message)
{
    for (size_t i = 0; i < first.size(); ++i)
    {
        ElementType leftVal = first[i];
        ElementType rightVal = second[i];
        ElementType allowedTolerance = (std::max<ElementType>)((ElementType)absoluteTolerance, ((ElementType)relativeTolerance) * leftVal);
        if (std::abs(leftVal - rightVal) > allowedTolerance)
            throw std::runtime_error(message);
    }
}

#pragma warning(push)
#pragma warning(disable: 4996)

template <typename ElementType>
inline void SaveAndReloadModel(CNTK::FunctionPtr& functionPtr, const std::vector<CNTK::Variable*>& variables, const CNTK::DeviceDescriptor& device)
{
    static std::wstring s_tempModelPath = L"feedForward.net";

    if ((_wunlink(s_tempModelPath.c_str()) != 0) && (errno != ENOENT))
        RuntimeError("Error deleting file '%ls': %s", s_tempModelPath.c_str(), strerror(errno));

    std::unordered_map<std::wstring, Variable*> inputVarNames;
    std::unordered_map<std::wstring, Variable*> outputVarNames;

    for (auto varPtr : variables)
    {
        auto retVal = varPtr->IsOutput() ? outputVarNames.insert({ varPtr->Owner()->Name(), varPtr }) : inputVarNames.insert({ varPtr->Name(), varPtr });
        if (!retVal.second)
            RuntimeError("SaveAndReloadModel: Multiple variables having same name cannot be restored after save and reload");
    }

    SaveAsLegacyModel<ElementType>(functionPtr, s_tempModelPath);
    functionPtr = LoadLegacyModel<ElementType>(s_tempModelPath, device);

    if (_wunlink(s_tempModelPath.c_str()) != 0)
        RuntimeError("Error deleting file '%ls': %s", s_tempModelPath.c_str(), strerror(errno));

    auto inputs = functionPtr->Inputs();
    for (auto inputVarInfo : inputVarNames)
    {
        auto newInputVar = *(std::find_if(inputs.begin(), inputs.end(), [inputVarInfo](const Variable& var) {
            return (var.Name() == inputVarInfo.first);
        }));

        *(inputVarInfo.second) = newInputVar;
    }

    auto outputs = functionPtr->Outputs();
    for (auto outputVarInfo : outputVarNames)
    {
        auto newOutputVar = *(std::find_if(outputs.begin(), outputs.end(), [outputVarInfo](const Variable& var) {
            return (var.Owner()->Name() == outputVarInfo.first);
        }));

        *(outputVarInfo.second) = newOutputVar;
    }
}

#pragma warning(pop)
