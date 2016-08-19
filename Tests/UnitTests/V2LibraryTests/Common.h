#pragma once
#include <exception>
#include <algorithm>
#include "CNTKLibrary.h"
#include <functional>
#include <fstream>
#include <random>

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

static std::mt19937_64 rng(0);

#pragma warning(push)
#pragma warning(disable: 4996)

#ifndef _MSC_VER
#include <unistd.h>
static inline std::string wtocharpath(const wchar_t *p)
{
    size_t len = wcslen(p);
    std::string buf;
    buf.resize(2 * len + 1);            // max: 1 wchar => 2 mb chars
    ::wcstombs(&buf[0], p, buf.size()); // note: technically it is forbidden to stomp over std::strings 0 terminator, but it is known to work in all implementations
    buf.resize(strlen(&buf[0]));        // set size correctly for shorter strings
    return buf;
}

static inline int _wunlink(const wchar_t *p)
{
    return unlink(wtocharpath(p).c_str());
}

static inline FILE *_wfopen(const wchar_t *path, const wchar_t *mode)
{
    return fopen(wtocharpath(path).c_str(), wtocharpath(mode).c_str());
}

#endif

template <typename ElementType>
inline void SaveAndReloadModel(CNTK::FunctionPtr& functionPtr, const std::vector<CNTK::Variable*>& variables, const CNTK::DeviceDescriptor& device)
{
    static std::wstring s_tempModelPath = L"feedForward.net";

    if ((_wunlink(s_tempModelPath.c_str()) != 0) && (errno != ENOENT))
        std::runtime_error("Error deleting temp model file 'feedForward.net'");

    std::unordered_map<std::wstring, CNTK::Variable*> inputVarNames;
    std::unordered_map<std::wstring, CNTK::Variable*> outputVarNames;

    for (auto varPtr : variables)
    {
        auto retVal = varPtr->IsOutput() ? outputVarNames.insert({ varPtr->Owner()->Name(), varPtr }) : inputVarNames.insert({ varPtr->Name(), varPtr });
        if (!retVal.second)
            std::runtime_error("SaveAndReloadModel: Multiple variables having same name cannot be restored after save and reload");
    }

    CNTK::SaveAsLegacyModel<ElementType>(functionPtr, s_tempModelPath);
    functionPtr = CNTK::LoadLegacyModel<ElementType>(s_tempModelPath, device);

    if (_wunlink(s_tempModelPath.c_str()) != 0)
         std::runtime_error("Error deleting temp model file 'feedForward.net'");

    auto inputs = functionPtr->Inputs();
    for (auto inputVarInfo : inputVarNames)
    {
        auto newInputVar = *(std::find_if(inputs.begin(), inputs.end(), [inputVarInfo](const CNTK::Variable& var) {
            return (var.Name() == inputVarInfo.first);
        }));

        *(inputVarInfo.second) = newInputVar;
    }

    auto outputs = functionPtr->Outputs();
    for (auto outputVarInfo : outputVarNames)
    {
        auto newOutputVar = *(std::find_if(outputs.begin(), outputs.end(), [outputVarInfo](const CNTK::Variable& var) {
            return (var.Owner()->Name() == outputVarInfo.first);
        }));

        *(outputVarInfo.second) = newOutputVar;
    }
}

inline CNTK::FunctionPtr FullyConnectedDNNLayer(CNTK::Variable input, size_t outputDim, const CNTK::DeviceDescriptor& device, const std::function<CNTK::FunctionPtr(const CNTK::FunctionPtr&)>& nonLinearity)
{
    assert(input.Shape().NumAxes() == 1);
    size_t inputDim = input.Shape()[0];

    auto timesParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ outputDim, inputDim }, -0.05, 0.05, 1, device));
    auto timesFunction = CNTK::Times(timesParam, input);

    auto plusParam = CNTK::Parameter(CNTK::NDArrayView::RandomUniform<float>({ outputDim }, -0.05, 0.05, 1, device));
    auto plusFunction = CNTK::Plus(plusParam, timesFunction);

    return nonLinearity(plusFunction);
}

inline float PrevMinibatchTrainingLossValue(const CNTK::Trainer& trainer)
{
    float trainLossValue = 0.0;
    auto prevMBTrainingLossValue = trainer.PreviousMinibatchTrainingLossValue()->Data();
    CNTK::NDArrayView cpuTrainLossValue(prevMBTrainingLossValue->Shape(), &trainLossValue, 1, CNTK::DeviceDescriptor::CPUDevice());
    cpuTrainLossValue.CopyFrom(*prevMBTrainingLossValue);

    return trainLossValue;
}

#pragma warning(pop)

inline CNTK::NDShape CreateShape(size_t numAxes, size_t maxDimSize)
{
    CNTK::NDShape shape(numAxes);
    for (size_t i = 0; i < numAxes; ++i)
    {
        shape[i] = (rng() % maxDimSize) + 1;
    }

    return shape;
}

inline void OpenStream(std::fstream& stream, const std::wstring& filename, bool readonly)
{
    if (filename.empty())
        std::runtime_error("File: filename is empty");

    std::ios_base::openmode mode = std::ios_base::binary;
    mode = mode | (readonly ? std::ios_base::in : std::ios_base::out);

    #ifdef _MSC_VER
    stream.open(filename.c_str(), mode);
    #else
    stream.open(wtocharpath(filename.c_str()).c_str(), mode);
    #endif
    stream.exceptions(std::ios_base::failbit | std::ios_base::badbit);  
}