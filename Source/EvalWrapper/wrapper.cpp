//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include <windows.h>
#include <vcclr.h>
#include <string>
#include <utility>
#include <msclr\marshal_cppstd.h>

#include "Eval.h"

#using <System.dll>
#using <System.Collections.dll>

using namespace System;
using namespace System::Collections::Generic;
using namespace System::Collections;

using namespace Microsoft::MSR::CNTK;

namespace Microsoft {
namespace MSR {
namespace CNTK {
// Used for retrieving the model appropriate for the element type (float / double)
template<typename ElemType>
using GetEvalProc = void(*)(IEvaluateModel<ElemType>**);

/// Managed wrapper for the native evaluation model
template<typename ElemType>
public ref class IEvaluateModelManaged
{
    typedef std::pair<std::wstring, std::vector<ElemType>*> MapEntry;

public:
    /// <summary>Initializes a new instance of the <see cref="IEvaluateModelManaged"> class.</summary>
    /// <param name="funcName">Factory function name for retrieving the native model from the dll.</param>
    IEvaluateModelManaged(String^ funcName)
    {
        pin_ptr<const WCHAR> dllname = PtrToStringChars("evaldll.dll");
        auto hModule = LoadLibrary(dllname);

        msclr::interop::marshal_context context;
        const std::string func = context.marshal_as<std::string>(funcName);
        auto procAddress = GetProcAddress(hModule, func.c_str());
        auto getEvalProc = (GetEvalProc<ElemType>)procAddress;
        pin_ptr <IEvaluateModel<ElemType>*> p_eval = &m_eval;
        getEvalProc(p_eval);
    }

    /// <summary>Initializes the model evaluation library with a CNTK configuration</summary>
    void Init(String^ config)
    {
        msclr::interop::marshal_context context;
        const std::string stdConfig = context.marshal_as<std::string>(config);

        m_eval->Init(stdConfig);
    }

    /// <summary>Destroys the model evaluation object</summary>
    void Destroy()
    {
        m_eval->Destroy();
    }

    /// <summary>Loads a model file</summary>
    /// <param name="modelFileName">The model file name to load</param>
    void LoadModel(String^ modelFileName)
    {
        pin_ptr<const WCHAR> stdModelPath = PtrToStringChars(modelFileName);
        m_eval->LoadModel(stdModelPath);
    }

    /// <summary>Evaluates the model against input data and retrieves the output layer data</summary>
    /// <param name="inputs"></param>
    /// <param name="outputs"></param>
    void Evaluate(Dictionary<String^, List<ElemType>^>^ inputs, Dictionary<String^, List<ElemType>^>^ outputs)
    {
        std::map<std::wstring, std::vector<ElemType>*> stdInputs;
        std::map<std::wstring, std::vector<ElemType>*> stdOutputs;

        for each (auto item in inputs)
        {
            pin_ptr<const WCHAR> key = PtrToStringChars(item.Key);
            stdInputs.insert(MapEntry(key, CopyList(item.Value)));
        }

        for each (auto item in outputs)
        {
            pin_ptr<const WCHAR> key = PtrToStringChars(item.Key);
            stdOutputs.insert(MapEntry(key, CopyList(item.Value)));
        }

        m_eval->Evaluate(stdInputs, stdOutputs);

        auto enumerator = outputs->Keys->GetEnumerator();
        for (std::map<std::wstring, std::vector<ElemType>*>::iterator ii = stdOutputs.begin(), e = stdOutputs.end(); ii != e; ii++)
        {
            // Retreive the layer key
            enumerator.MoveNext();
            String^ key = enumerator.Current;

            std::vector<ElemType> &refVector = *((*ii).second);
            int index = 0;

            // Copy output to CLI structure
            for (std::vector<ElemType>::iterator ii = refVector.begin(), e = refVector.end(); ii != e; ii++)
            {
                outputs[key][index++] = *ii;
            }
        }

        // Release the memory used
        for (std::map<std::wstring, std::vector<ElemType>*>::iterator ii = stdInputs.begin(), e = stdInputs.end(); ii != e; ii++)
        {
            delete (*ii).second;
        }

        for (std::map<std::wstring, std::vector<ElemType>*>::iterator ii = stdOutputs.begin(), e = stdOutputs.end(); ii != e; ii++)
        {
            delete (*ii).second;
        }
    }

    /// <summary>Evaluates the model against input data and retrieves the output layer data</summary>
    /// <param name="inputs"></param>
    /// <param name="outputKey"></param>
    /// <param name="outputSize"></param>
    /// <returns>Results for specified layer</returns>
    List<ElemType>^ Evaluate(Dictionary<String^, List<ElemType>^>^ inputs, String^ outputKey, int outputSize)
    {
        std::map<std::wstring, std::vector<ElemType>*> stdInputs;
        std::map<std::wstring, std::vector<ElemType>*> stdOutputs;

        // Prepare input
        for each (auto item in inputs)
        {
            pin_ptr<const WCHAR> key = PtrToStringChars(item.Key);
            stdInputs.insert(MapEntry(key, CopyList(item.Value)));
        }

        // Prepare output buffer
        pin_ptr<const WCHAR> key = PtrToStringChars(outputKey);
        stdOutputs.insert(MapEntry(key, new std::vector<ElemType>(outputSize)));

        // Perform evaluation
        m_eval->Evaluate(stdInputs, stdOutputs);

        // Copy output to CLI structure
        List<ElemType>^ output = gcnew List<ElemType>();
        std::vector<ElemType>* vector = stdOutputs.begin()->second;
        int count = 0;
        for (std::vector<ElemType>::iterator ii = (*vector).begin(), e = (*vector).end(); ii != e && count < outputSize; ii++, count++)
        {
            output->Add(*ii);
        }

        // Release the used memory
        for (std::map<std::wstring, std::vector<ElemType>*>::iterator ii = stdInputs.begin(), e = stdInputs.end(); ii != e; ii++)
        {
            delete (*ii).second;
        }

        return output;
    }

private:
    // Native model evaluation instance
    IEvaluateModel<ElemType> *m_eval;

    /// <summary>Copies a list of element types from a CLI structure to a native structure
    /// <param name="list">The CLI list of items</param>
    /// <returns>A native vector of items</returns>
    std::vector<ElemType>* CopyList(List<ElemType>^ list)
    {
        std::vector<ElemType>* lower = new std::vector<ElemType>();
        for each (ElemType item in list)
        {
            lower->push_back(item);
        }

        return lower;
    }
};

/// <summary>Managed float-specific model evaluation class</summary>
/// <remarks>This class is necessary due to how generics and templates work in CLR</remarks>
public ref class IEvaluateModelManagedF : IEvaluateModelManaged<float>
{
public:
    IEvaluateModelManagedF::IEvaluateModelManagedF()
        : IEvaluateModelManaged("GetEvalF")
    {
    }
};

/// <summary>Managed double-specific model evaluation class</summary>
/// <remarks>This class is necessary due to how generics and templates work in CLR</remarks>
public ref class IEvaluateModelManagedD : IEvaluateModelManaged<double>
{
public:
    IEvaluateModelManagedD::IEvaluateModelManagedD()
        : IEvaluateModelManaged("GetEvalD")
    {
    }
};

// This method tricks the compiler into emitting the methods of the classes
// Refer to https://msdn.microsoft.com/en-us/library/ms177213.aspx for an
// explanation to this insanity
void emit()
{
    IEvaluateModelManagedF f;
    f.Init("");
    f.Evaluate(nullptr, nullptr);
    f.Evaluate(nullptr, "", 0);
    f.LoadModel("");
    f.Destroy();

    IEvaluateModelManagedD d;
    d.Init("");
    d.Evaluate(nullptr, nullptr);
    d.Evaluate(nullptr, "", 0);
    d.LoadModel("");
    d.Destroy();
}
}
}
}