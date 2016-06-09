//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// EvalExtendedWrapper.cpp -- Managed code wrapping the native EvaluateExtendedModel interface
//

#include <windows.h>
#include <vcclr.h>
#include <string>
#include <utility>
#include <vector>
#include <memory>
#include <msclr\marshal_cppstd.h>

#include "CNTKException.h"
#include "Eval.h"

#using <System.dll>
#using <System.Collections.dll>

using namespace std;
using namespace System;
using namespace System::Collections::Generic;
using namespace System::Collections;
using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Extensibility { namespace Managed {

// Used for retrieving the appropriate model for the element type (float / double)
template<typename ElemType>
using GetEvalProc = void(*)(IEvaluateModelExtended<ElemType>**);

/// Enumeration for the types of nodes
public enum class NodeGroup
{
    nodeInput,  // an input node
    nodeOutput, // an output node
    nodeSpecified
};

//
// A buffer to keep data for all samples in a (variable length) sequence 
// from a single input or output.
// This is used for both dense and sparse data.
//
generic<class ElemType>
public ref class ValueBuffer
{
public:
    ValueBuffer(int size)
    {
        m_buffer = gcnew array<ElemType>(size);
        m_indices = gcnew List<int>();
        m_colIndices = gcnew List<int>();
    }

    //
    // All elements of a sequence, concatenated.
    // For dense inputs, the number of samples is given by the the length of
    // this vector / product of tensor dimensions. E.g. for a tensor of dimension
    // [2,2] and 12 elements in the buffer, the number of samples is 3.
    // For sparse inputs, the number of samples is indicated by the m_colIndices field.
    //
    array<ElemType>^ m_buffer;

    // In case of sparse data, the following is also used. Otherwise, the 
    // contents are ignored.

    // E.g. a sequence of three sparse vectors with 2 / 4 / 2 non-zero values
    // could be represented as the following:
    // colIdx:  0   2       6   8
    //          v   v       v   v
    // indices  1 3 2 3 5 6 2 7
    // buffer   0 1 2 3 4 5 6 7

    //
    // For every element in buffer, an entry in this array gives its position.
    // For every vector the entries must be ascending.
    //
    List<int>^ m_indices;

    //
    // Contains numberOfsamples + 1 indices into the buffer. The first entry
    // is always 0. The last entry points after the last element.
    // See http://docs.nvidia.com/cuda/cusparse/#compressed-sparse-column-format-csc
    //
    List<int>^ m_colIndices;
};

//
// Meta data
//
public ref struct VariableLayout
{
    enum class DataType
    {
        Float32,
        Float64
    };

    enum class StorageType
    {
        Undetermined,
        Dense,
        Sparse,
    };

    // Name of the input
    String^ m_name;

    DataType m_dataType;

    StorageType m_storageType;

    // Dimension of the tensor, flattened to 1 dimension, for one entry on the dynamic axis.
    // E.g. for a tensor [2,3,*] this would be 6.
    int m_numElements;
};

public ref class VariableSchema : List<VariableLayout^>
{
public:
    generic<typename ElemType>
    List<ValueBuffer<ElemType>^>^ CreateBuffers(List<int>^ maxLengths)
    {
        if (maxLengths->Count != this->Count)
        {
            throw gcnew CNTKRuntimeException("Expected max lengths for all variables.", String::Empty);
        }

        List<ValueBuffer<ElemType>^>^ buffers = gcnew List<ValueBuffer<ElemType>^>(this->Count);
        for (int i = 0; i < this->Count; i++)
        {
            buffers->Add(gcnew ValueBuffer<ElemType>(this[i]->m_numElements * maxLengths[i]));
        }

        return buffers;
    }

    // Creates minimum size buffers based on schema
    generic<typename ElemType>
        List<ValueBuffer<ElemType>^>^ CreateBuffers()
        {
            List<ValueBuffer<ElemType>^>^ buffers = gcnew List<ValueBuffer<ElemType>^>(this->Count);
            for (int i = 0; i < this->Count; i++)
            {
                buffers->Add(gcnew ValueBuffer<ElemType>(this[i]->m_numElements));
            }

            return buffers;
        }
};

/*
template <typename ElemType>
using Variables = List<ValueBuffer<ElemType>^>^;

template <typename ElemType>
using ValueBufferNat = Microsoft::MSR::CNTK::ValueBuffer<ElemType>;

template <typename ElemType>
using VariablesNat = std::vector<ValueBufferNat<ElemType>>;

using VariableSchema = List<VariableLayout^>^;
*/

/// Managed wrapper for the native evaluation model
template<typename ElemType>
public ref class IEvaluateModelExtendedManaged : IDisposable
{
    typedef std::pair<std::wstring, std::vector<ElemType>*> MapEntry;

public:
    /// <summary>Initializes a new instance of the <see cref="IEvaluateModelExtendedManaged"> class.</summary>
    /// <param name="funcName">Factory function name for retrieving the native model from the dll.</param>
    IEvaluateModelExtendedManaged(String^ funcName)
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
            pin_ptr <IEvaluateModelExtended<ElemType>*> p_eval = &m_eval;
            getEvalProc(p_eval);
        }
        catch (const exception& ex)
        {
            throw gcnew CNTKException(gcnew System::String(ex.what()));
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

    //
    // GetOutputSchema - retrieve information about tensor shapes and memory layout of the outputs for this
    // model.
    //
    VariableSchema^ GetOutputSchema()
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        auto outputLayout = m_eval->GetOutputSchema();

        auto outputSchema = gcnew VariableSchema();
        for (auto& lay : outputLayout)
        {
            VariableLayout^ layout = gcnew VariableLayout();
            layout->m_name = gcnew String(lay.m_name.c_str());
            layout->m_dataType = GetDataType(lay.m_dataType);
            layout->m_numElements = lay.m_numElements;

            outputSchema->Add(layout);
        }
        return outputSchema;
    }

    //
    // Allocate internal state for calling ForwardPass(). The call restricts the network (inputs and outputs)
    // to the functions represented by the output name.
    //
    void StartForwardEvaluation(List<String^>^ outputs)
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        std::vector<wstring> outputNodeNames;
        msclr::interop::marshal_context context;

        for each (String^ output in outputs)
        {
            //std::wstring name = context.marshal_as<std::wstring>(output);
            outputNodeNames.push_back(context.marshal_as<std::wstring>(output));
        }

        m_eval->StartForwardEvaluation(outputNodeNames);
    }

    //
    // GetInputSchema - 
    //
    List<VariableLayout^>^ GetInputSchema()
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        auto inputLayout = m_eval->GetInputSchema();

        auto inputSchema = gcnew List<VariableLayout^>();
        for (auto& lay : inputLayout)
        {
            VariableLayout^ layout = gcnew VariableLayout();
            layout->m_name = gcnew String(lay.m_name.c_str());
            layout->m_dataType = GetDataType(lay.m_dataType);
            layout->m_numElements = lay.m_numElements;

            inputSchema->Add(layout);
        }
        return inputSchema;
    }

    //
    // Evaluate - Evaluate (perform a forward pass for) a single unit using the model with the given inputs and 
    // outputs.
    // The layout and shape of the data in inputs vector must match the schema returned by GetInputLayouts.
    // This method is not reentrant, as the forward pass keeps internal state.
    // outputId - output to compute values for. See GetOutputLayouts()
    // inputs - vector of input buffers, one for every input as given by GetInputLayouts()
    // outputs - map from node name to output vector, outputs vectors need to be preallocated by caller, sizing 
    // will happen during evaluation.
    // Called after StartForwardEvaluation()
    //
    void ForwardPass(List<ValueBuffer<ElemType>^>^ inputs, List<ValueBuffer<ElemType>^>^ outputs)
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        std::vector<shared_ptr<std::vector<ElemType>>> stdSharedBufferInputs;
        std::vector<shared_ptr<std::vector<int>>> stdSharedIndicesInputs;
        std::vector<shared_ptr<std::vector<int>>> stdSharedColIndicesInputs;
        std::vector<shared_ptr<std::vector<ElemType>>> stdSharedBufferOutputs;
        std::vector<shared_ptr<std::vector<int>>> stdSharedIndicesOutputs;
        std::vector<shared_ptr<std::vector<int>>> stdSharedColIndicesOutputs;

        try
        {
            std::vector<Microsoft::MSR::CNTK::ValueBuffer<ElemType>> stdInputs;
            std::vector<Microsoft::MSR::CNTK::ValueBuffer<ElemType>> stdOutputs;

            // TODO: Pin the memories prior to passing the pointer to the native side
            for each (auto item in inputs)
            {
                shared_ptr<std::vector<ElemType>> ptrBuffer = CopyList<ElemType>(item->m_buffer);
                stdSharedBufferInputs.push_back(ptrBuffer);
                shared_ptr<std::vector<int>> ptrIndices = CopyList<int>(item->m_indices);
                stdSharedIndicesInputs.push_back(ptrIndices);
                shared_ptr<std::vector<int>> ptrColIndices = CopyList<int>(item->m_colIndices);
                stdSharedColIndicesInputs.push_back(ptrColIndices);
                stdInputs.push_back({*ptrBuffer.get(), *ptrIndices.get(), *ptrColIndices.get()});
            }

            for each (auto item in outputs)
            {
                shared_ptr<std::vector<ElemType>> ptrBuffer = CopyList(item->m_buffer);
                stdSharedBufferOutputs.push_back(ptrBuffer);
                shared_ptr<std::vector<int>> ptrIndices = CopyList<int>(item->m_indices);
                stdSharedIndicesOutputs.push_back(ptrIndices);
                shared_ptr<std::vector<int>> ptrColIndices = CopyList<int>(item->m_colIndices);
                stdSharedColIndicesOutputs.push_back(ptrColIndices);
                stdOutputs.push_back({*ptrBuffer.get(), *ptrIndices.get(), *ptrColIndices.get()});
            }

            try
            {
                m_eval->ForwardPass(stdInputs, stdOutputs);
            }
            catch (const exception& ex)
            {
                throw GetCustomException(ex);
            }

            // Once memory is pinned, the output values should already be in the outputs list
            for (int varIndex = 0; varIndex < stdOutputs.size(); varIndex++)
            {
                for (int bufIndex = 0; bufIndex < stdOutputs[varIndex].m_buffer.size(); bufIndex++)
                {
                    auto out = stdOutputs[varIndex].m_buffer[bufIndex];
                    outputs[varIndex]->m_buffer[bufIndex] = out;
                }
            }
        }
        catch (Exception^)
        {
            throw;
        }
    }

    ~IEvaluateModelExtendedManaged()
    {
        if (m_eval == nullptr)
        {
            return;
        }

        this->!IEvaluateModelExtendedManaged();
    }

protected:
    !IEvaluateModelExtendedManaged()
    {
        if (m_eval != nullptr)
        {
            m_eval->Destroy();
            m_eval = nullptr;
        }
    }

private:
    // Native model evaluation instance
    IEvaluateModelExtended<ElemType> *m_eval;

    /// <summary>Copies a list of element types from a CLI structure to a native structure</summary>
    /// <param name="list">The CLI list of items</param>
    /// <returns>A native vector of items</returns>
    template<typename ElemType>
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

    template<typename ElemType>
    shared_ptr<std::vector<ElemType>> CopyList(array<ElemType>^ list)
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
            throw gcnew CNTKRuntimeException(String::Format("Cannot convert native NodeGroup with value: {0} to corresponding managed NodeGroup.", (int)nodeGroup), "");
        }
    }

    VariableLayout::DataType GetDataType(Microsoft::MSR::CNTK::VariableLayout::DataType dataType)
    {
        switch ((int)dataType)
        {
        case VariableLayout::DataType::Float32:
            return VariableLayout::DataType::Float32;
        case VariableLayout::DataType::Float64:
            return VariableLayout::DataType::Float64;
        default:
            throw gcnew CNTKRuntimeException(String::Format("Cannot convert native DataType with value: {0} to corresponding managed DataType.", (int)dataType), "");
        }
    }

    VariableLayout::StorageType GetStorageType(Microsoft::MSR::CNTK::VariableLayout::StorageType storageType)
    {
        switch ((int)storageType)
        {
        case VariableLayout::StorageType::Dense:
            return VariableLayout::StorageType::Dense;
        case VariableLayout::StorageType::Sparse:
            return VariableLayout::StorageType::Sparse;
        case VariableLayout::StorageType::Undetermined:
            return VariableLayout::StorageType::Undetermined;
        default:
            throw gcnew CNTKRuntimeException(String::Format("Cannot convert native StorageType with value: {0} to corresponding managed StorageType.", (int)storageType), "");
        }
    }
};

/// <summary>Managed float-specific model evaluation class</summary>
/// <remarks>This class is necessary due to how generics and templates work in CLR</remarks>
public ref class IEvaluateModelExtendedManagedF : IEvaluateModelExtendedManaged<float>
{
public:
    IEvaluateModelExtendedManagedF::IEvaluateModelExtendedManagedF()
        : IEvaluateModelExtendedManaged("GetEvalExtendedF")
    {
    }
};

/// <summary>Managed double-specific model evaluation class</summary>
/// <remarks>This class is necessary due to how generics and templates work in CLR</remarks>
public ref class IEvaluateModelExtendedManagedD : IEvaluateModelExtendedManaged<double>
{
public:
    IEvaluateModelExtendedManagedD::IEvaluateModelExtendedManagedD()
        : IEvaluateModelExtendedManaged("GetEvalExtendedD")
    {
    }
};


// This method tricks the compiler into emitting the methods of the classes
// Refer to https://msdn.microsoft.com/en-us/library/ms177213.aspx for an
// explanation to this behavior
void emitExtended()
{
    IEvaluateModelExtendedManagedF f;
    f.CreateNetwork("");
    f.GetOutputSchema();
    f.GetInputSchema();
    f.StartForwardEvaluation(nullptr);
    f.ForwardPass(nullptr, nullptr);

    IEvaluateModelExtendedManagedD d;
    d.CreateNetwork("");
    d.GetOutputSchema();
    d.GetInputSchema();
    d.StartForwardEvaluation(nullptr);
    d.ForwardPass(nullptr, nullptr);
}

}}}}}
