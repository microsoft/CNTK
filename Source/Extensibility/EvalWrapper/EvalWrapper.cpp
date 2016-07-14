//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// EvalWrapper.cpp -- Managed code wrapping the native EvaluateModel interface
//

#include <windows.h>
#include <vcclr.h>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <msclr\marshal_cppstd.h>

#include "CNTKException.h"
#include "EvalCommon.h"
#include "Eval.h"

#using <System.dll>
#using <System.Collections.dll>

using namespace std;
using namespace System;
using namespace System::Collections::Generic;
using namespace System::Collections;
using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Extensibility { namespace Managed {

// Used for retrieving the model appropriate for the element type (float / double)
template<typename ElemType>
using GetEvalProc = void(*)(IEvaluateModel<ElemType>**);

/// Managed wrapper for the native evaluation model
template<typename ElemType>
public ref class IEvaluateModelManaged : IDisposable
{
    typedef std::pair<std::wstring, std::vector<ElemType>*> MapEntry;

public:
    /// <summary>Initializes a new instance of the <see cref="IEvaluateModelManaged"> class.</summary>
    /// <param name="funcName">Factory function name for retrieving the native model from the dll.</param>
    IEvaluateModelManaged(String^ funcName)
    {
        pin_ptr<const WCHAR> dllname = PtrToStringChars("evaldll.dll");
        auto hModule = LoadLibrary(dllname);
        if (hModule == nullptr)
        {
            throw gcnew CNTKException(System::String::Format("Cannot find library: {0}", gcnew String(dllname)));
        }

        try
        {
            msclr::interop::marshal_context context;
            const std::string func = context.marshal_as<std::string>(funcName);
            auto procAddress = GetProcAddress(hModule, func.c_str());
            auto getEvalProc = (GetEvalProc<ElemType>)procAddress;
            pin_ptr <IEvaluateModel<ElemType>*> p_eval = &m_eval;
            getEvalProc(p_eval);
        }
        catch (const exception& ex)
        {
            throw gcnew CNTKException(gcnew System::String(ex.what()));
        }
    }

    /// <summary>Initializes the model evaluation library with a CNTK configuration</summary>
    /// <param name="config">Model configuration entries</param>
    void Init(String^ config)
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        msclr::interop::marshal_context context;
        const std::string stdConfig = context.marshal_as<std::string>(config);

        try
        {
            m_eval->Init(stdConfig);
        }
        catch (const exception& ex)
        {
            throw GetCustomException(ex);
        }
    }

    /// <summary>Creates a network based on the network description in the configuration</summary>
    /// <param name="networkDescription">The configuration file containing the network description</param>
    void CreateNetwork(String^ networkDescription)
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        msclr::interop::marshal_context context;
        const std::string stdNetworkDescription = context.marshal_as<std::string>(networkDescription);

        try
        {
            m_eval->CreateNetwork(stdNetworkDescription);
        }
        catch (const exception& ex)
        {
            throw GetCustomException(ex);
        }
    }

    /// <summary>Creates a network based on the network description in the configuration</summary>
    /// <param name="networkDescription">The configuration file containing the network description</param>
    /// <param name="outputNodeNames">The output list of nodes (replaces the model's list of output nodes)</param>
    void CreateNetwork(String^ networkDescription, List<String^>^ outputNodeNames)
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        String^ outputNodeNamesProperty = outputNodeNames != nullptr ? String::Concat("outputNodeNames=", String::Join(":", outputNodeNames)) : "";
        String^ newNetworkConfig = String::Format("{0}\n{1}", outputNodeNamesProperty, networkDescription);
        this->CreateNetwork(newNetworkConfig);
    }

    /// <summary>Creates a network based on the network description in the configuration</summary>
    /// <param name="networkDescription">The configuration file containing the network description</param>
    /// <param name="deviceId">The device ID to specify for the network</param>
    void CreateNetwork(String^ networkDescription, int deviceId)
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        this->CreateNetwork(networkDescription, deviceId, nullptr);
    }

    /// <summary>Creates a network based on the network description in the configuration</summary>
    /// <param name="networkDescription">The configuration file containing the network description</param>
    /// <param name="deviceId">The device ID to specify for the network</param>
    /// <param name="outputNodeNames">The output list of nodes (replaces the model's list of output nodes)</param>
    void CreateNetwork(String^ networkDescription, int deviceId, List<String^>^ outputNodeNames)
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        String^ outputNodeNamesProperty = outputNodeNames != nullptr ? String::Concat("outputNodeNames=", String::Join(":", outputNodeNames)) : "";
        String^ newNetworkConfig = String::Format("deviceId={0}\n{1}\n{2}", deviceId, outputNodeNamesProperty, networkDescription);
        this->CreateNetwork(newNetworkConfig);
    }

    /// <summary>Evaluates the model using a single forward feed pass and retrieves the output layer data</summary>
    /// <param name="outputKey"></param>
    /// <param name="outputSize"></param>
    /// <returns>Results for specified layer</returns>
    __declspec(deprecated) List<ElemType>^ Evaluate(String^ outputKey, int outputSize)
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        std::map<std::wstring, std::vector<ElemType>*> stdOutputs;


        try
        {
            std::vector<shared_ptr<std::vector<ElemType>>> sharedOutputVectors;

            List<ElemType>^ outputs = gcnew List<ElemType>(outputSize);
            for (int i = 0; i < outputSize; i++)
            {
                outputs->Add(*(gcnew ElemType));
            }

            Dictionary<String^, List<ElemType>^>^ outputMap = gcnew Dictionary<String^, List<ElemType>^>();
            outputMap->Add(outputKey, outputs);

            for each (auto item in outputMap)
            {
                pin_ptr<const WCHAR> key = PtrToStringChars(item.Key);
                shared_ptr<std::vector<ElemType>> ptr = CopyList(item.Value);
                sharedOutputVectors.push_back(ptr);
                stdOutputs.insert(MapEntry(key, ptr.get()));
            }

            try
            {
                m_eval->Evaluate(stdOutputs);
            }
            catch (const exception& ex)
            {
                throw GetCustomException(ex);
            }

            auto enumerator = outputMap->Keys->GetEnumerator();
            for (auto& map_item : stdOutputs)
            {
                // Retrieve the layer key
                enumerator.MoveNext();
                String^ key = enumerator.Current;

                std::vector<ElemType> &refVec = *(map_item.second);
                int index = 0;

                // Copy output to CLI structure
                for (auto& vec : refVec)
                {
                    outputMap[key][index++] = vec;
                }
            }

            return outputMap[outputKey];
        }
        catch (Exception^)
        {
            throw;
        }
    }

    /// <summary>Evaluates the model using a single forward feed pass and retrieves the output layer data</summary>
    /// <param name="outputKey"></param>
    /// <param name="outputSize"></param>
    /// <returns>Results for specified layer</returns>
    List<ElemType>^ Evaluate(String^ outputKey)
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        std::map<std::wstring, std::vector<ElemType>*> stdOutputs;

        try
        {
            std::vector<shared_ptr<std::vector<ElemType>>> sharedOutputVectors;
            int outputSize = GetNodeDimensions(NodeGroup::Output)[outputKey];

            List<ElemType>^ outputs = gcnew List<ElemType>(outputSize);
            for (int i = 0; i < outputSize; i++)
            {
                outputs->Add(*(gcnew ElemType));
            }

            Dictionary<String^, List<ElemType>^>^ outputMap = gcnew Dictionary<String^, List<ElemType>^>();
            outputMap->Add(outputKey, outputs);

            for each (auto item in outputMap)
            {
                pin_ptr<const WCHAR> key = PtrToStringChars(item.Key);
                shared_ptr<std::vector<ElemType>> ptr = CopyList(item.Value);
                sharedOutputVectors.push_back(ptr);
                stdOutputs.insert(MapEntry(key, ptr.get()));
            }

            try
            {
                m_eval->Evaluate(stdOutputs);
            }
            catch (const exception& ex)
            {
                throw GetCustomException(ex);
            }

            auto enumerator = outputMap->Keys->GetEnumerator();
            for (auto& map_item : stdOutputs)
            {
                // Retrieve the layer key
                enumerator.MoveNext();
                String^ key = enumerator.Current;

                std::vector<ElemType> &refVec = *(map_item.second);
                int index = 0;

                // Copy output to CLI structure
                for (auto& vec : refVec)
                {
                    outputMap[key][index++] = vec;
                }
            }

            return outputMap[outputKey];
        }
        catch (Exception^)
        {
            throw;
        }
    }

    /// <summary>Evaluates the model against input data and retrieves the output layer data</summary>
    /// <param name="inputs"></param>
    /// <param name="outputs"></param>
    void Evaluate(Dictionary<String^, List<ElemType>^>^ inputs, Dictionary<String^, List<ElemType>^>^ outputs)
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        std::map<std::wstring, std::vector<ElemType>*> stdInputs;
        std::map<std::wstring, std::vector<ElemType>*> stdOutputs;

        try
        {
            std::vector<shared_ptr<std::vector<ElemType>>> sharedInputVectors;
            std::vector<shared_ptr<std::vector<ElemType>>> sharedOutputVectors;

            for each (auto item in inputs)
            {
                pin_ptr<const WCHAR> key = PtrToStringChars(item.Key);
                shared_ptr<std::vector<ElemType>> ptr = CopyList(item.Value);
                sharedInputVectors.push_back(ptr);
                stdInputs.insert(MapEntry(key, ptr.get()));
            }

            for each (auto item in outputs)
            {
                pin_ptr<const WCHAR> key = PtrToStringChars(item.Key);
                shared_ptr<std::vector<ElemType>> ptr = CopyList(item.Value);
                sharedOutputVectors.push_back(ptr);
                stdOutputs.insert(MapEntry(key, ptr.get()));
            }

            try
            {
                m_eval->Evaluate(stdInputs, stdOutputs);
            }
            catch (const exception& ex)
            {
                throw GetCustomException(ex);
            }

            auto enumerator = outputs->Keys->GetEnumerator();
            for (auto& map_item : stdOutputs)
            {
                // Retrieve the layer key
                enumerator.MoveNext();
                String^ key = enumerator.Current;

                std::vector<ElemType> &refVec = *(map_item.second);
                int index = 0;

                // Copy output to CLI structure
                for (auto& vec : refVec)
                {
                    outputs[key][index++] = vec;
                }
            }
        }
        catch (Exception^)
        {
            throw;
        }
    }

    /// <summary>Evaluates the model against input data and retrieves the output layer data</summary>
    /// <param name="inputs"></param>
    /// <param name="outputKey"></param>
    /// <param name="outputSize"></param>
    /// <returns>Results for specified layer</returns>
    __declspec(deprecated) List<ElemType>^ Evaluate(Dictionary<String^, List<ElemType>^>^ inputs, String^ outputKey, int outputSize)
    {
        List<ElemType>^ outputs = gcnew List<ElemType>(outputSize);
        for (int i = 0; i < outputSize; i++)
        {
            outputs->Add(*(gcnew ElemType));
        }

        Dictionary<String^, List<ElemType>^>^ outputMap = gcnew Dictionary<String^, List<ElemType>^>();
        outputMap->Add(outputKey, outputs);

        Evaluate(inputs, outputMap);

        return outputMap[outputKey];
    }

    /// <summary>Evaluates the model against input data and retrieves the desired output layer data</summary>
    /// <param name="inputs"></param>
    /// <param name="outputKey"></param>
    /// <returns>Results for requested layer</returns>
    List<ElemType>^ Evaluate(Dictionary<String^, List<ElemType>^>^ inputs, String^ outputKey)
    {
        auto outDims = GetNodeDimensions(NodeGroup::Output);
        int outputSize = outDims[outputKey];

        List<ElemType>^ outputs = gcnew List<ElemType>(outputSize);
        for (int i = 0; i < outputSize; i++)
        {
            outputs->Add(*(gcnew ElemType));
        }

        Dictionary<String^, List<ElemType>^>^ outputMap = gcnew Dictionary<String^, List<ElemType>^>();
        outputMap->Add(outputKey, outputs);

        Evaluate(inputs, outputMap);

        return outputMap[outputKey];
    }

    /// <summary>Returns the layer(s) and associated dimensions for the specified node group
    /// <param name="nodeGroup">The node type to query for</param>
    /// <returns>A dictionary mapping layer names to their dimension</returns>
    Dictionary<String^, int>^ GetNodeDimensions(NodeGroup nodeGroup)
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        std::map<std::wstring, size_t> stdDims;

        try
        {
            Microsoft::MSR::CNTK::NodeGroup gr(GetNodeGroup(nodeGroup));
            m_eval->GetNodeDimensions(stdDims, gr);
        }
        catch (const exception& ex)
        {
            throw GetCustomException(ex);
        }

        Dictionary<String^, int>^ dims = gcnew Dictionary<String^, int>();

        for (auto& map_item : stdDims)
        {
            String^ key = gcnew String(map_item.first.c_str());
            int dim = static_cast<int>(map_item.second);
            dims->Add(key, dim);
        }

        return dims;
    }

    ~IEvaluateModelManaged()
    {
        if (m_eval == nullptr)
        {
            return;
        }

        this->!IEvaluateModelManaged();
    }

protected:
    !IEvaluateModelManaged()
    {
        if (m_eval != nullptr)
        {
            m_eval->Destroy();
            m_eval = nullptr;
        }
    }

private:
    // Native model evaluation instance
    IEvaluateModel<ElemType> *m_eval;

    /// <summary>Copies a list of element types from a CLI structure to a native structure</summary>
    /// <param name="list">The CLI list of items</param>
    /// <returns>A native vector of items</returns>
    shared_ptr<std::vector<ElemType>> CopyList(List<ElemType>^ list)
    {
        shared_ptr<std::vector<ElemType>> lower(new std::vector<ElemType>());
        if (list != nullptr)
        {
            for each (ElemType item in list)
            {
                lower->push_back(item);
            }
        }
        return lower;
    }

    /// <summary> Throws a CLR exception based on a native exception</summary>
    /// <param name="ex">The native exception to throw as a CLR exception</param>
    /// <returns>A CLR exception</returns>
    CNTKException^ GetCustomException(const exception& ex)
    {
        // Determine the appropriate exception and initialize it with the exception payload
        if (typeid(ex) == typeid(ExceptionWithCallStack<runtime_error>))
        {
            ExceptionWithCallStack<runtime_error>& rich = dynamic_cast<ExceptionWithCallStack<runtime_error>&>((runtime_error&)ex);
            return gcnew CNTKRuntimeException(gcnew System::String(rich.what()), gcnew System::String(rich.CallStack()));
        }
        else if (typeid(ex) == typeid(ExceptionWithCallStack<logic_error>))
        {
            ExceptionWithCallStack<logic_error>& rich = dynamic_cast<ExceptionWithCallStack<logic_error>&>((logic_error&)ex);
            return gcnew CNTKLogicErrorException(gcnew System::String(ex.what()), gcnew System::String(rich.CallStack()));
        }
        else if (typeid(ex) == typeid(ExceptionWithCallStack<invalid_argument>))
        {
            ExceptionWithCallStack<invalid_argument>& rich = dynamic_cast<ExceptionWithCallStack<invalid_argument>&>((invalid_argument&)ex);
            return gcnew CNTKInvalidArgumentException(gcnew System::String(ex.what()), gcnew System::String(rich.CallStack()));
        }
        else if (typeid(ex) == typeid(bad_alloc))
        {
            return gcnew CNTKBadAllocException(gcnew System::String(ex.what()));
        }
        else
        {
            return gcnew CNTKException(gcnew System::String(ex.what()));
        }
    }

    /// <summary Converts a managed (CLI) enum NodeGroup to a native NodeGroup
    /// <param name="nodeGroup">The managed (CLI) NodeGroup to convert to native</param>
    Microsoft::MSR::CNTK::NodeGroup GetNodeGroup(NodeGroup nodeGroup)
    {
        switch ((int)nodeGroup)
        {
        case Microsoft::MSR::CNTK::NodeGroup::nodeInput:
            return Microsoft::MSR::CNTK::NodeGroup::nodeInput;
        case Microsoft::MSR::CNTK::NodeGroup::nodeOutput:
            return Microsoft::MSR::CNTK::NodeGroup::nodeOutput;
        case Microsoft::MSR::CNTK::NodeGroup::nodeSpecified:
            return Microsoft::MSR::CNTK::NodeGroup::nodeSpecified;
        default:
            throw gcnew CNTKRuntimeException(String::Format("Cannot convert native NodeGroup with value: {0} to corresponding managed NodeGroup.",(int)nodeGroup), "");
        }
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
// explanation to this behavior
void emit()
{
    Dictionary<String^, List<float>^>^ nullDictF = nullptr;
    Dictionary<String^, List<double>^>^ nullDictD = nullptr;

    IEvaluateModelManagedF f;
    f.Init("");
    f.Evaluate(nullptr, nullDictF);
    f.Evaluate(nullptr, "");
    f.Evaluate("");
    f.CreateNetwork("");
    f.CreateNetwork("", 0);
    f.CreateNetwork("", nullptr);
    f.CreateNetwork("", 0, nullptr);
    f.GetNodeDimensions(NodeGroup::Specified);

    IEvaluateModelManagedD d;
    d.Init("");
    d.Evaluate(nullptr, nullDictD);
    d.Evaluate(nullptr, "");
    d.Evaluate("");
    d.CreateNetwork("");
    d.CreateNetwork("", 0);
    d.CreateNetwork("", nullptr);
    d.CreateNetwork("", 0,nullptr);
    d.GetNodeDimensions(NodeGroup::Specified);

    // Deprecated code, hush warnings locally only
#pragma warning(push)
#pragma warning(disable: 4996)
    f.Evaluate(nullptr, "", 0);
    f.Evaluate("", 0);
    d.Evaluate(nullptr, "", 0);
    d.Evaluate("", 0);
#pragma warning(pop)
}

}}}}}
