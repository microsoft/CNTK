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
#pragma warning(push)
#pragma warning(disable : 4793) // Function compiled as native
#include "Basics.h"
#include "ScriptableObjects.h"
#pragma warning(pop)
#include "EvalCommon.h"
#include "Eval.h"

#using <System.dll>
#using <System.Collections.dll>
#using <System.IO.dll>
#using <System.Reflection.dll>

using namespace std;
using namespace System;
using namespace System::Collections::Generic;
using namespace System::Collections;
using namespace System::Runtime::InteropServices;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Extensibility { namespace Managed {

namespace Native = Microsoft::MSR::CNTK;

// Used for retrieving the appropriate model for the element type (float / double)
template<typename ElemType>
using GetEvalProc = void(*)(IEvaluateModelExtended<ElemType>**);

//
// A buffer to keep data for all samples in a (variable length) sequence 
// from a single input or output.
// This is used for both dense and sparse data.
//
generic<class ElemType>
public ref class ValueBuffer
    {
    public:
        ValueBuffer()
        {
            Size = 0;
        }

        //
        // Init for Dense
        //
        ValueBuffer(int bufferSize)
        {
            Buffer = gcnew cli::array<ElemType>(bufferSize);
            Size = bufferSize;
        }

        //
        // Init for Sparse
        //
        ValueBuffer(int bufferSize, int colIndicesSize)
        {
            Buffer = gcnew cli::array<ElemType>(bufferSize);
            Indices = gcnew cli::array<int>(bufferSize);
            ColIndices = gcnew cli::array<int>(colIndicesSize);
            Size = colIndicesSize;
        }

        //
        // For dense, this is the length of Buffer (in nr. of ElemTypes).
        // For sparse, this is the length of ColIndices (i.e. the number of columns + 1).
        // This allows Buffer / Indices / ColIndices to be larger than Size to avoid
        // reallocation.
        //
        property int Size;

        //
        // All elements of a sequence, concatenated.
        // For dense inputs, the number of samples is given by the the length of
        // this vector / product of tensor dimensions. E.g. for a tensor of dimension
        // [2,2] and 12 elements in the buffer, the number of samples is 3.
        // For sparse inputs, the number of samples is indicated by the ColIndices field.
        //
        property cli::array<ElemType>^ Buffer;

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
        property cli::array<int>^ Indices;

        //
        // Contains numberOfsamples + 1 indices into the buffer. The first entry
        // is always 0. The last entry points after the last element.
        // See http://docs.nvidia.com/cuda/cusparse/#compressed-sparse-column-format-csc
        //
        property cli::array<int>^ ColIndices;


        // TODO: Should it have a read-only StorageType property?
    };

//
// Meta data
//
public ref struct VariableLayout
{
    // Name of the input
    property String^ Name;

    property DataType DataType;

    property StorageType StorageType;

    // Dimension of the tensor, flattened to 1 dimension, for one entry on the dynamic axis.
    // E.g. for a tensor [2,3,*] this would be 6.
    property int NumElements;
};

public ref class VariableSchema : List<VariableLayout^>
{
public:
    generic<typename ElemType>
        cli::array<ValueBuffer<ElemType>^>^ CreateBuffers(... cli::array<int>^ maxLengths)
        {
            if (maxLengths->Length == 0)
            {
                maxLengths = gcnew cli::array<int>(this->Count);
                for (int i = 0; i<maxLengths->Length; i++)
                {
                    maxLengths[i] = 1;
                }
            }

            if (maxLengths->Length != this->Count)
            {
                throw gcnew CNTKRuntimeException("Expected max lengths for all variables.", String::Empty);
            }

            cli::array<ValueBuffer<ElemType>^>^ buffers = gcnew cli::array<ValueBuffer<ElemType>^>(this->Count);
            for (int i = 0; i < this->Count; i++)
            {
                buffers[i] = gcnew ValueBuffer<ElemType>(this[i]->NumElements * maxLengths[i]);
            }

            return buffers;
        }
};

/// Managed wrapper for the native evaluation model
template<typename ElemType>
public ref class ModelEvaluationExtended : IDisposable
{
    typedef std::pair<std::wstring, std::vector<ElemType>*> MapEntry;
    typedef std::shared_ptr<Native::ValueBuffer<ElemType, Native::VectorRef>> ValueBufferPtr;

public:
    /// <summary>Initializes a new instance of the <see cref="ModelEvaluationExtended"> class.</summary>
    /// <param name="funcName">Factory function name for retrieving the native model from the dll.</param>
    ModelEvaluationExtended(String^ funcName)
    {
        try
        {
            pin_ptr <IEvaluateModelExtended<ElemType>*> p_eval = &m_eval;
            GetEvalExtended<ElemType>(p_eval);
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
    // GetInputSchema - retrieve information about tensor shapes and memory layout for this model.
    //
    VariableSchema^ GetInputSchema()
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        return ConvertNativeSchemaToManaged(m_eval->GetInputSchema());
    }

    //
    // GetOutputSchema - retrieve information about tensor shapes and memory layout for this model.
    //
    VariableSchema^ GetOutputSchema()
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        return ConvertNativeSchemaToManaged(m_eval->GetOutputSchema());
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
            outputNodeNames.push_back(context.marshal_as<std::wstring>(output));
        }

        try
        {
            m_eval->StartForwardEvaluation(outputNodeNames);
        }
        catch (const exception& ex)
        {
            throw GetCustomException(ex);
        }
    }

    //
    // Forward Pass - Evaluate (perform a forward pass for) a single unit using the model with the given inputs and 
    // outputs.
    // The layout and shape of the data in inputs vector must match the schema returned by GetInputLayouts.
    // This method is not reentrant, as the forward pass keeps internal state.
    // outputId - output to compute values for. See GetOutputLayouts()
    // inputs - vector of input buffers, one for every input as given by GetInputLayouts()
    // outputs - map from node name to output vector, outputs vectors need to be preallocated by caller
    // Called after StartForwardEvaluation()
    //
    void ForwardPass(cli::array<ValueBuffer<ElemType>^>^ inputs, cli::array<ValueBuffer<ElemType>^>^ outputs)
    {
        ForwardPass(inputs, outputs, true);
    }

    //
    // Same as above, and
    // resetRNN - flags whether to reset memory cells of RNN. 
    //
    void ForwardPass(cli::array<ValueBuffer<ElemType>^>^ inputs, cli::array<ValueBuffer<ElemType>^>^ outputs, bool resetRNN)
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        // Hold all buffers that should be pinned during native operations
        List<GCHandle>^ pinnedGCHandleList = gcnew List<GCHandle>;

        try
        {
            Native::ValueRefs<ElemType> stdInputs;
            Native::ValueRefs<ElemType> stdOutputs;

            // Map the managed space into the native space, results will be written directly into the managed memory space
            // https://msdn.microsoft.com/en-us/library/1dz8byfh.aspx
            TransferVectorsToValueBuffers(inputs, stdInputs, pinnedGCHandleList, StorageType::Sparse);
            TransferVectorsToValueBuffers(outputs, stdOutputs, pinnedGCHandleList, StorageType::Dense);

            try
            {
                m_eval->ForwardPass(stdInputs, stdOutputs, resetRNN);

                // Update actual output size.
                for (int i = 0; i < outputs->Length; ++i)
                {
                    outputs[i]->Size = (int)stdOutputs[i].m_buffer.m_size;
                }
            }
            catch (const exception& ex)
            {
                throw GetCustomException(ex);
            }
        }
        catch (Exception^)
        {
            throw;
        }
        finally
        {
            for each (auto h in pinnedGCHandleList)
            {
                h.Free();
            }
        }
    }

    ~ModelEvaluationExtended()
    {
        if (m_eval == nullptr)
        {
            return;
        }

        this->!ModelEvaluationExtended();
    }

protected:
    !ModelEvaluationExtended()
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
        else if (dynamic_cast<const ScriptableObjects::ScriptingException*>(&ex) != nullptr) // Includes derived classes
        {
            const auto& err = dynamic_cast<const ScriptableObjects::ScriptingException&>(ex);
            return gcnew CNTKLogicErrorException(gcnew System::String(::msra::strfun::_strprintf<wchar_t>(L"%ls\n%ls", msra::strfun::utf16(err.what()).c_str(), err.GetError(L"").c_str()).c_str()), nullptr);
        }
        else
        {
            return gcnew CNTKException(gcnew System::String(ex.what()));
        }
    }

    /// <summary Converts a managed (CLI) enum NodeGroup to a native NodeGroup
    /// <param name="nodeGroup">The managed (CLI) NodeGroup to convert to native</param>
    Native::NodeGroup GetNodeGroup(NodeGroup nodeGroup)
    {
        switch ((int)nodeGroup)
        {
        case Native::NodeGroup::nodeInput:
            return Native::NodeGroup::nodeInput;
        case Native::NodeGroup::nodeOutput:
            return Native::NodeGroup::nodeOutput;
        case Native::NodeGroup::nodeSpecified:
            return Native::NodeGroup::nodeSpecified;
        default:
            throw gcnew CNTKRuntimeException(String::Format("Cannot convert native NodeGroup with value: {0} to corresponding managed NodeGroup.", (int)nodeGroup), "");
        }
    }

    DataType GetDataType(Microsoft::MSR::CNTK::VariableLayout::DataType dataType)
    {
        switch ((int)dataType)
        {
        case DataType::Float32:
            return DataType::Float32;
        case DataType::Float64:
            return DataType::Float64;
        default:
            throw gcnew CNTKRuntimeException(String::Format("Cannot convert native DataType with value: {0} to corresponding managed DataType.", (int)dataType), "");
        }
    }

    StorageType GetStorageType(Microsoft::MSR::CNTK::VariableLayout::StorageType storageType)
    {
        switch ((int)storageType)
        {
        case StorageType::Dense:
            return StorageType::Dense;
        case StorageType::Sparse:
            return StorageType::Sparse;
        case StorageType::Unknown:
            return StorageType::Unknown;
        default:
            throw gcnew CNTKRuntimeException(String::Format("Cannot convert native StorageType with value: {0} to corresponding managed StorageType.", (int)storageType), "");
        }
    }

    void PinBuffer(cli::array<ElemType>^ itemBuffer, List<GCHandle>^ pinnedGCHandleList, Native::ValueBuffer<ElemType, Native::VectorRef>* vb, StorageType storageType, int bufferSize)
    {
        GCHandle h = GCHandle::Alloc(itemBuffer, GCHandleType::Pinned);
        pinnedGCHandleList->Add(h);
        ElemType* pp = reinterpret_cast<ElemType *>(h.AddrOfPinnedObject().ToPointer());
        vb->m_buffer.InitFrom(pp, bufferSize, storageType == StorageType::Sparse ? bufferSize : 0);
    }

    void PinIndices(cli::array<int>^ itemBuffer, List<GCHandle>^ pinnedGCHandleList, Native::ValueBuffer<ElemType, Native::VectorRef>* vb, StorageType storageType, int bufferSize)
    {
        GCHandle h = GCHandle::Alloc(itemBuffer, GCHandleType::Pinned);
        pinnedGCHandleList->Add(h);
        int* pp = reinterpret_cast<int *>(h.AddrOfPinnedObject().ToPointer());
        vb->m_indices.InitFrom(pp, bufferSize, storageType == StorageType::Sparse ? bufferSize : 0);
    }

    void PinColIndices(cli::array<int>^ itemBuffer, List<GCHandle>^ pinnedGCHandleList, Native::ValueBuffer<ElemType, Native::VectorRef>* vb, StorageType storageType, int bufferSize)
    {
        GCHandle h = GCHandle::Alloc(itemBuffer, GCHandleType::Pinned);
        pinnedGCHandleList->Add(h);
        int* pp = reinterpret_cast<int *>(h.AddrOfPinnedObject().ToPointer());
        vb->m_colIndices.InitFrom(pp, bufferSize, storageType == StorageType::Sparse ? bufferSize : 0);
    }

    void TransferVectorsToValueBuffers(cli::array<ValueBuffer<ElemType>^>^ list, Native::ValueRefs<ElemType>& valueRefs, List<GCHandle>^ pinnedGCHandleList, StorageType storageType)
    {
        for each (auto item in list)
        {
            Native::ValueBuffer<ElemType, Native::VectorRef> vb;

            int numElements = item->Size;
            int bufferSize = item->ColIndices != nullptr ? item->ColIndices[item->Size - 1] : item->Size;

            // Buffer is required
            if (item->Buffer == nullptr)
            {
                throw gcnew CNTKRuntimeException("Invalid buffer (empty) for argument into ForwardPass", String::Empty);
            }

            PinBuffer(item->Buffer, pinnedGCHandleList, &vb, storageType, bufferSize);

            if (item->Indices != nullptr)
            {
                PinIndices(item->Indices, pinnedGCHandleList, &vb, storageType, bufferSize);
            }

            if (item->ColIndices != nullptr)
            {
                PinColIndices(item->ColIndices, pinnedGCHandleList, &vb, storageType, numElements);
            }

            valueRefs.push_back(vb);
        }
    }

    //
    // ConvertNativeSchemaToManaged - Converts a native schema to a manged one
    //
    VariableSchema^ ConvertNativeSchemaToManaged(Native::VariableSchema layouts)
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        auto schema = gcnew VariableSchema();
        for (auto& lay : layouts)
        {
            VariableLayout^ varlayout = gcnew VariableLayout();
            varlayout->Name = gcnew String(lay.m_name.c_str());
            varlayout->DataType = GetDataType(lay.m_dataType);
            varlayout->NumElements = static_cast<int>(lay.m_numElements);
            varlayout->StorageType = GetStorageType(lay.m_storageType);

            schema->Add(varlayout);
        }
        return schema;
    }
};

/// <summary>Managed float-specific model evaluation class</summary>
/// <remarks>This class is necessary due to how generics and templates work in CLR</remarks>
public ref class ModelEvaluationExtendedF : ModelEvaluationExtended<float>
{
public:
    ModelEvaluationExtendedF::ModelEvaluationExtendedF()
        : ModelEvaluationExtended("GetEvalExtendedF")
    {
    }
};

/// <summary>Managed double-specific model evaluation class</summary>
/// <remarks>This class is necessary due to how generics and templates work in CLR</remarks>
public ref class ModelEvaluationExtendedD : ModelEvaluationExtended<double>
{
public:
    ModelEvaluationExtendedD::ModelEvaluationExtendedD()
        : ModelEvaluationExtended("GetEvalExtendedD")
    {
    }
};


// This method tricks the compiler into emitting the methods of the classes
// Refer to https://msdn.microsoft.com/en-us/library/ms177213.aspx for an
// explanation to this behavior
void EmitExtended()
{
    ModelEvaluationExtendedF f;
    f.CreateNetwork("");
    f.GetInputSchema();
    f.GetOutputSchema();
    f.StartForwardEvaluation(nullptr);
    f.ForwardPass(nullptr, nullptr);

    ModelEvaluationExtendedD d;
    d.CreateNetwork("");
    d.GetInputSchema();
    d.GetOutputSchema();
    d.StartForwardEvaluation(nullptr);
    d.ForwardPass(nullptr, nullptr);

    VariableSchema sc;
    sc.CreateBuffers<float>();
    sc.CreateBuffers<double>();
}

}}}}}
