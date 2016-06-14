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
#include "EvalCommon.h"
#include "Eval.h"

#using <System.dll>
#using <System.Collections.dll>

using namespace std;
using namespace System;
using namespace System::Collections::Generic;
using namespace System::Collections;

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
        }

        ValueBuffer(int bufferSize)
        {
            m_buffer = gcnew array<ElemType>(bufferSize);
            m_size = bufferSize;
        }

        ValueBuffer(int bufferSize, int indicesSize, int colIndicesSize) : ValueBuffer(bufferSize)
        {
            m_indices = gcnew array<int>(indicesSize);
            m_colIndices = gcnew array<int>(colIndicesSize);
        }
    
        // Returns the current allocated size for the buffer or 0 if empty or no buffer allocated
        property int Size
        {
            int get() 
            {
                return m_buffer == nullptr ? 0 : m_buffer->Length < m_size ? m_buffer->Length : m_size;
            }
            void set(int newSize)
            {
                if (newSize > m_buffer->Length)
                {
                    throw gcnew CNTKRuntimeException("Cannot expand dynamically, instead re-assign a new Buffer.", String::Empty);
                }
                m_size = newSize;
            }
        }

        //
        // All elements of a sequence, concatenated.
        // For dense inputs, the number of samples is given by the the length of
        // this vector / product of tensor dimensions. E.g. for a tensor of dimension
        // [2,2] and 12 elements in the buffer, the number of samples is 3.
        // For sparse inputs, the number of samples is indicated by the m_colIndices field.
        //
        property array<ElemType>^ Buffer
        {
            array<ElemType>^ get()
            {
                return m_buffer;
            }

            void set(array<ElemType>^ buffer)
            {
                m_buffer = buffer;
                m_size = m_buffer == nullptr ? 0 : m_buffer->Length;
            }
        }

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
        property array<int>^ Indices 
        { 
            array<int>^ get() 
            {
                return m_indices;
            }

            void set(array<int>^ indices)
            {
                m_indices = indices;
            }
        }

        //
        // Contains numberOfsamples + 1 indices into the buffer. The first entry
        // is always 0. The last entry points after the last element.
        // See http://docs.nvidia.com/cuda/cusparse/#compressed-sparse-column-format-csc
        //
        property array<int>^ ColIndices {
            array<int>^ get() 
            {
                return m_colIndices;
            }

            void set(array<int>^ colIndices)
            {
                m_colIndices = colIndices;
            }
        }

    private:

    //
    // All elements of a sequence, concatenated.
    // For dense inputs, the number of samples is given by the the length of
    // this vector / product of tensor dimensions. E.g. for a tensor of dimension
    // [2,2] and 12 elements in the buffer, the number of samples is 3.
    // For sparse inputs, the number of samples is indicated by the m_colIndices field.
    //
    array<ElemType>^ m_buffer;
    int m_size;

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
    array<int>^ m_indices;

    //
    // Contains numberOfsamples + 1 indices into the buffer. The first entry
    // is always 0. The last entry points after the last element.
    // See http://docs.nvidia.com/cuda/cusparse/#compressed-sparse-column-format-csc
    //
    array<int>^ m_colIndices;
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
    List<ValueBuffer<ElemType>^>^ CreateBuffers(List<int>^ maxLengths)
    {
        if (maxLengths->Count != this->Count)
        {
            throw gcnew CNTKRuntimeException("Expected max lengths for all variables.", String::Empty);
        }

        auto buffers = gcnew List<ValueBuffer<ElemType>^>(this->Count);
        for (int i = 0; i < this->Count; i++)
        {
            buffers->Add(gcnew ValueBuffer<ElemType>(this[i]->NumElements * maxLengths[i]));
        }

        return buffers;
    }

    // Creates minimum size buffers based on schema
    generic<typename ElemType>
    List<ValueBuffer<ElemType>^>^ CreateBuffers()
    {
        auto maxLengths = gcnew List<int>(this->Count);
        for (int i = 0; i < this->Count; i++)
        {
            maxLengths->Add(1);
        }

        return this->CreateBuffers<ElemType>(maxLengths);
    }
};

/// Managed wrapper for the native evaluation model
template<typename ElemType>
public ref class ModelEvaluationExtended : IDisposable
{
    typedef std::pair<std::wstring, std::vector<ElemType>*> MapEntry;

public:
    /// <summary>Initializes a new instance of the <see cref="ModelEvaluationExtended"> class.</summary>
    /// <param name="funcName">Factory function name for retrieving the native model from the dll.</param>
    ModelEvaluationExtended(String^ funcName)
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
    // GetSchema - retrieve information about tensor shapes and memory layout for this model.
    //
    VariableSchema^ GetSchema(NodeGroup nodeGroup)
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        if (nodeGroup != NodeGroup::Input && nodeGroup != NodeGroup::Output)
        {
            throw gcnew CNTKInvalidArgumentException("The nodeGroup argument is invalid.", String::Empty);
        }

        auto layouts = nodeGroup == NodeGroup::Input ? m_eval->GetInputSchema() : m_eval->GetOutputSchema();

        auto schema = gcnew VariableSchema();
        for (auto& lay : layouts)
        {
            VariableLayout^ varlayout = gcnew VariableLayout();
            varlayout->Name = gcnew String(lay.m_name.c_str());
            varlayout->DataType = GetDataType(lay.m_dataType);
            varlayout->NumElements = lay.m_numElements;
            varlayout->StorageType = GetStorageType(lay.m_storageType);

            schema->Add(varlayout);
        }
        return schema;
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

        m_eval->StartForwardEvaluation(outputNodeNames);
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
    void ForwardPass(List<ValueBuffer<ElemType>^>^ inputs, List<ValueBuffer<ElemType>^>^ outputs)
    {
        if (m_eval == nullptr)
        {
            throw gcnew ObjectDisposedException("Object has been disposed.");
        }

        try
        {
            Native::ValueRefs<ElemType> stdInputs;
            Native::ValueRefs<ElemType> stdOutputs;
            Native::ValueBuffer<ElemType, Native::VectorRef> vb;

            // Hold gc objects in the stack, while performing native actions
            vector<gcroot<array<ElemType>^>*> pinBuffers;
            vector<gcroot<array<int>^>*> pinIndices;

            // Map the managed space into the native space, results will be written directly into the managed memory space
            // https://msdn.microsoft.com/en-us/library/1dz8byfh.aspx
            TransferVectorsToValueBuffers(inputs, vb, stdInputs, pinBuffers, pinIndices);
            TransferVectorsToValueBuffers(outputs, vb, stdOutputs, pinBuffers, pinIndices);

            try
            {
                m_eval->ForwardPass(stdInputs, stdOutputs);
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

    void TransferVectorsToValueBuffers(List<ValueBuffer<ElemType>^>^ list, Native::ValueBuffer<ElemType, Native::VectorRef>& vb, Native::ValueRefs<ElemType>& valueRefs, vector<gcroot<array<ElemType>^>*>& pinBuffers, vector<gcroot<array<int>^>*>& pinIndices)
    {
        for each (auto item in list)
        {
            int size = item->Size;

            // Buffer is required
            if (item->Buffer == nullptr || item->Buffer->Length == 0)
            {
                throw gcnew CNTKRuntimeException("Invalid buffer (empty) for argument into ForwardPass", String::Empty);
            }

            // gcroot object manages the pointer so that it always corresponds to the correct managed location (even after gc relocation)
            auto pBuf = new gcroot<array<ElemType>^>(item->Buffer);
            pin_ptr<ElemType> pp = &(*pBuf)[0];
            pinBuffers.push_back(pBuf);
            vb.m_buffer.InitFrom(pp, size, size);

            if (item->Indices != nullptr)
            {
                auto pInd = new gcroot<array<int>^>(item->Indices);
                pin_ptr<int> pi = &(*pInd)[0];
                pinIndices.push_back(pInd);
                vb.m_indices.InitFrom(pi, item->Indices->Length, item->Indices->Length);
            }

            if (item->ColIndices != nullptr)
            {
                auto pColInd = new gcroot<array<int>^>(item->ColIndices);
                pin_ptr<int> pci = &(*pColInd)[0];
                pinIndices.push_back(pColInd);
                vb.m_colIndices.InitFrom(pci, item->ColIndices->Length, item->ColIndices->Length);
            }

            valueRefs.push_back(vb);
        }
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
    f.GetSchema(NodeGroup::Specified);
    f.StartForwardEvaluation(nullptr);
    f.ForwardPass(nullptr, nullptr);

    ModelEvaluationExtendedD d;
    d.CreateNetwork("");
    d.GetSchema(NodeGroup::Specified);
    d.StartForwardEvaluation(nullptr);
    d.ForwardPass(nullptr, nullptr);

    VariableSchema sc;
    sc.CreateBuffers<float>();
    sc.CreateBuffers<double>();
}

}}}}}
