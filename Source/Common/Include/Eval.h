//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the EVAL_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// EVAL_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef _WIN32
#if defined(EVAL_EXPORTS)
#define EVAL_API __declspec(dllexport)
#elif defined(EVAL_LOCAL)
#define EVAL_API
#else
#define EVAL_API __declspec(dllimport)
#endif
#else
#define EVAL_API
#endif

#include <map>
#include <vector>
#include <string>
#include <memory>

namespace Microsoft { namespace MSR { namespace CNTK {

template <typename ElemType>
class IEvaluateModelBase 
{
public:
    // 
    // Load a model based on configuration. The syntax is the same as when calling the cntk executable.
    // e.g. "modelFile=model.dat deviceId=0".
    // numCPUThreads can be used to set the thread count of BLAS.
    // 
    virtual void Init(const std::string& config) = 0;

    //
    // Create a network based on an (NDL) network description.
    //
    virtual void CreateNetwork(const std::string& networkDescription) = 0;

    //
    // Free resources
    //
    virtual void Destroy() = 0;
};

// ------------------------------------------------------------------------
// Basic (legacy) interface
// ------------------------------------------------------------------------

enum NodeGroup
{
    nodeInput,  // an input node
    nodeOutput, // an output node
    nodeSpecified
};

// IEvaluateModel - interface used by decoders and other components that need just evaluator functionality in DLL form
// NOTICE: This interface is a public interface for evaluating models in CNTK. 
//         Changes to this interface may affect other projects, such as Argon and LatGen,
//         and therefore need to be communicated with such groups.
template <typename ElemType>
class IEvaluateModel : public IEvaluateModelBase<ElemType> // Evaluate Model Interface
{
public:
    //
    // Retrieves the (flattened) dimensions 
    //
    virtual void GetNodeDimensions(std::map<std::wstring, size_t>& dimensions, NodeGroup nodeGroup) = 0;

    //
    // Allocate resources for a particular output.
    //
    virtual void StartEvaluateMinibatchLoop(const std::wstring& outputNodeName) = 0;
    
    //
    // Evaluate a model in frame mode. This does not support dynamic axes or sparse input data.
    // Given a feature vector of dimension d, the inputs may contain n * d elements. The output will then be computed 
    // for n samples.
    // inputs - map from node name to array of input tensors, flattened to vector
    // outputs - map from node name to output vector, outputs vectors need to be preallocated by caller, sizing will
    // happen during evaluation
    // 
    virtual void Evaluate(std::map<std::wstring, std::vector<ElemType>*>& inputs, std::map<std::wstring, std::vector<ElemType>*>& outputs) = 0;

    //
    // Evaluate - Evaluate using the network without input and provide the outputs
    // outputs - map from node name to output vector, outputs vectors need to be preallocated by caller, sizing will 
    // happen during evaluation
    //
    virtual void Evaluate(std::map<std::wstring, std::vector<ElemType>*>& outputs) = 0;

    //
    // Reset initial state of all Recurrence loops (RNNs) in the model.
    //
    virtual void ResetState() = 0;
};


// GetEval - get a evaluator type from the DLL
// since we have 2 evaluator types based on template parameters, exposes 2 exports
// could be done directly with the templated name, but that requires mangled C++ names
template <typename ElemType>
void EVAL_API GetEval(IEvaluateModel<ElemType>** peval);
extern "C" EVAL_API void GetEvalF(IEvaluateModel<float>** peval);
extern "C" EVAL_API void GetEvalD(IEvaluateModel<double>** peval);

class Plugin;

template <typename ElemType>
class Eval : public IEvaluateModel<ElemType>
{
private:
    IEvaluateModel<ElemType>* m_eval; // evaluation class pointer
    std::shared_ptr<Plugin> m_plugin; 

    void GetEvalClass(const std::string& config);

    // Destroy - cleanup and remove this class. Workaround to ensure that memory allocation / deallocation
    // occur within the DLL boundary.
    // NOTE: this destroys the object, and it can't be used past this point
    virtual void Destroy();

public:
    // EvaluateModel Constructor
    // config - configuration information:
    // deviceId=auto ( can be [0,all,cpu,0:2:3,auto] define accellerators (GPUs) to use, or the CPU
    // modelPath=c:\models\model.dnn (model path, if not specified, must call LoadModel() method before Evaluate()
    // minibatchSize=1024 (minibatch size used during evaluation if < passed data size)
    Eval(const std::string& config);

    virtual ~Eval();

    // CreateNetwork - create a network based on the network description
    // networkDescription - network description
    virtual void CreateNetwork(const std::string& networkDescription);

    // GetNodeDimensions - Get the node dimensions of the specified nodes
    // dimensions - map from name of node to dimension of the node
    // nodeGroup - type of node we are requesting (input/output/specified)
    virtual void GetNodeDimensions(std::map<std::wstring, size_t>& dimensions, NodeGroup nodeGroup);

    // StartEvaluateMinibatchLoop - Prepare network for Evaluate() calls.
    // outputNodeName - name of node that will be evaluated
    virtual void StartEvaluateMinibatchLoop(const std::wstring& outputNodeName);

    // Evaluate - Evaluate using the model with the given inputs and outputs
    // inputs - map from node name to input vector
    // outputs - map from node name to output vector, outputs vectors need to be preallocated by caller, sizing will 
    // happen during evaluation
    virtual void Evaluate(std::map<std::wstring, std::vector<ElemType>*>& inputs, std::map<std::wstring, std::vector<ElemType>*>& outputs);

    // Evaluate - Evaluate using the network without input, and provide the outputs
    // outputs - map from node name to output vector, outputs vectors need to be preallocated by caller, sizing will 
    // happen during evaluation
    virtual void Evaluate(std::map<std::wstring, std::vector<ElemType>*>& outputs);

    virtual void Init(const std::string& config);

    virtual void ResetState();
};


// ------------------------------------------------------------------------
// Extended interface
// ------------------------------------------------------------------------


// Partial instantiation of vector to reduce to one argument.
template <typename ElemType>
using Vector = std::vector<ElemType, std::allocator<ElemType>>;

//
// A buffer to keep data for all samples in a (variable length) sequence 
// from a single input or output.
// This is used for both dense and sparse data.
//
template<typename ElemType, template<typename> class Container = Vector>
struct ValueBuffer
{
    //
    // All elements of a sequence, concatenated.
    // For dense inputs, the number of samples is given by the length of
    // this vector / product of tensor dimensions. E.g. for a tensor of dimension
    // [2,2] and 12 elements in the buffer, the number of samples is 3.
    // For sparse inputs, the number of samples is indicated by the m_colIndices field.
    //
    Container<ElemType> m_buffer;

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
    Container<int> m_indices;

    //
    // Contains numberOfsamples + 1 indices into the buffer. The first entry
    // is always 0. The last entry points after the last element.
    // See http://docs.nvidia.com/cuda/cusparse/#compressed-sparse-column-format-csc
    //
    Container<int> m_colIndices;
};

//
// Helper class that can be used in exchange of a std::vector if the memory is managed externally.
//
template <typename ElemType>
struct VectorRef
{
    ElemType* m_vector;
    size_t m_capacity;   // ElemTypes allocated
    size_t m_size;       // ElemTypes used.

    VectorRef() : m_vector(nullptr), m_capacity(0), m_size(0) {}
    void InitFrom(std::vector<ElemType>& src) { InitFrom(src.data(), src.capacity(), src.size()); }
    void InitFrom(ElemType* data, size_t capacity, size_t size) { m_vector = data; m_capacity = capacity; m_size = size; }
    size_t size() const { return m_size; }
    size_t capacity() const { return m_capacity; }
    ElemType* data() { return m_vector; }
    const ElemType* data() const { return m_vector; }
    ElemType* begin() { return m_vector; }
    ElemType* end() { return m_vector + m_size; }
    void resize(size_t size) { m_size = size; }
    ElemType& operator[](size_t idx) { return m_vector[idx]; }
    const ElemType& operator[](size_t idx) const { return m_vector[idx]; }
};

template <typename ElemType>
using Values = std::vector<ValueBuffer<ElemType, Vector>>;

template <typename ElemType>
using ValueRefs = std::vector<ValueBuffer<ElemType, VectorRef>>;

//
// Meta data
//
struct VariableLayout
{
    enum DataType
    {
        Float32,
        Float64
    };

    enum StorageType
    {
        Undetermined,
        Dense,
        Sparse,
    };

    // Name of the input
    std::wstring m_name;

    DataType m_dataType;

    StorageType m_storageType;

    // Dimension of the tensor, flattened to 1 dimension, for one entry on the dynamic axis.
    // E.g. for a tensor [2,3,*] this would be 6.
    size_t m_numElements;
};

class VariableSchema : public std::vector<VariableLayout>
{
    public:
        template<typename ElemType>
        Values<ElemType> CreateBuffers(const std::vector<size_t>& maxLengths)
        {
            if (maxLengths.size() != size())
                throw std::runtime_error("Expected max lengths for all variables.");

            Values<ElemType> buffers(size());
            for (size_t i = 0; i < size(); ++i)
            {
                buffers[i].m_buffer.reserve(operator[](i).m_numElements * maxLengths[i]);
            }
            return buffers;
        }
};

//
// Extended interface, allowing for sparse input.
// Implementation constraints: 
// - Every output is a single tensor (not a batch), 
// - Outputs must be dense.
// - Output buffer must be preallocated.
//
template <typename ElemType>
class IEvaluateModelExtended : public IEvaluateModelBase<ElemType>
{
public:
    //
    // GetOutputSchema - retrieve information about tensor shapes and memory layout of the outputs for this
    // model.
    //
    virtual VariableSchema GetOutputSchema() const = 0;

    //
    // Allocate internal state for calling ForwardPass(). The call restricts the network (inputs and outputs)
    // to the functions represented by the output name.
    //
    virtual void StartForwardEvaluation(const std::vector<std::wstring>& outputs) = 0;

    //
    // GetVariableLayout - retrieve information about tensor shapes and memory layout of inputs necessary for a
    // particular output. By default this returns all available inputs. After StartForwardEvaluation(), this
    // returns all the inputs necessary to compute the outputs.
    //
    virtual VariableSchema GetInputSchema() const = 0;

    //
    // ForwardPass - Evaluate (perform a forward pass for) a single unit using the model with the given inputs and 
    // outputs.
    // The layout and shape of the data in inputs vector must match the schema returned by GetInputLayouts.
    // Output must be preallocated and sized to avoid memory allocation / deallocation across DLL
    // boundaries.
    // This method is not reentrant, as the forward pass keeps internal state.
    // inputs - vector of input buffers, one for every input as given by GetInputLayouts()
    // outputs - vector of output buffers. Must be sized to fit output schema.
    //
    virtual void ForwardPass(const Values<ElemType>& inputs, Values<ElemType>& output) = 0;

    //
    // Same as above, and
    // resetRNN - flags whether to reset memory cells of RNN. 
    //
    virtual void ForwardPass(const Values<ElemType>& inputs, Values<ElemType>& output, bool resetRNN) = 0;

    //
    // Same as above, but takes references to static arrays instead of std::vector 
    // (e.g. when vectors are manages by .net)
    // 
    virtual void ForwardPass(const ValueRefs<ElemType>& inputs, ValueRefs<ElemType>& output) = 0;

    //
    // Same as above, and
    // resetRNN - flags whether to reset memory cells of RNN. 
    //
    virtual void ForwardPass(const ValueRefs<ElemType>& inputs, ValueRefs<ElemType>& output, bool resetRNN) = 0;
};

template <typename ElemType>
void EVAL_API GetEvalExtended(IEvaluateModelExtended<ElemType>** peval);
extern "C" EVAL_API void GetEvalExtendedF(IEvaluateModelExtended<float>** peval);
extern "C" EVAL_API void GetEvalExtendedD(IEvaluateModelExtended<double>** peval);

} } }
