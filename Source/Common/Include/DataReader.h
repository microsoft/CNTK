//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

// The following ifdef block is the standard way of creating macros which make exporting
// from a DLL simpler. All files within this DLL are compiled with the DATAREADER_EXPORTS
// symbol defined on the command line. This symbol should not be defined on any project
// that uses this DLL. This way any other project whose source files include this file see
// DATAREADER_API functions as being imported from a DLL, whereas this DLL sees symbols
// defined with this macro as being exported.
#ifdef _WIN32
#if defined(DATAREADER_EXPORTS)
#define DATAREADER_API __declspec(dllexport)
#elif defined(DATAREADER_LOCAL)
#define DATAREADER_API
#else
#define DATAREADER_API __declspec(dllimport)
#endif
#else
#define DATAREADER_API
#endif

#include "Basics.h"
#include "Matrix.h"
#include "Sequences.h"
#include "Config.h" // for ConfigParameters
#include "ScriptableObjects.h"
#include <map>
#include <string>
#include <memory>
#include <unordered_set>

// forward-declare these lattice-related types to avoid having to include and pollute everything with lattice-related headers
namespace msra { namespace dbn {  class latticepair; class latticesource; } }
namespace msra { namespace asr {  class simplesenonehmm; } } 

namespace Microsoft { namespace MSR { namespace CNTK {

    // This class contains input stream descriptions, 
    // the network can request less streams than the reader provides.
    // TODO: Should be unified with StreamDescription from the new reader API
    struct InputStreamDescription
    {
        InputStreamDescription(const std::wstring& name, int deviceId, MatrixType matrixType, MatrixFormat format)
            : m_name(name), m_deviceId(deviceId), m_matrixType(matrixType), m_format(format)
        {}

        const std::wstring& GetStreamName() const
        {
            return m_name;
        }

        int GetDeviceId() const
        {
            return m_deviceId;
        }

        MatrixType GetMatrixType() const
        {
            return m_matrixType;
        }

        MatrixFormat GetMatrixFormat() const
        {
            return m_format;
        }

    private:
        // Stream name.
        std::wstring m_name;

        // Device identifier for the resulting matrix of this stream.
        int m_deviceId;

        // Matrix type.
        MatrixType m_matrixType;

        // Matrix format.
        MatrixFormat m_format;
    };

    inline bool operator == (const InputStreamDescription& a, const InputStreamDescription& b)
    {
        return a.GetStreamName() == b.GetStreamName() &&
               a.GetDeviceId() == b.GetDeviceId() &&
               a.GetMatrixType() == b.GetMatrixType() &&
               a.GetMatrixFormat() == b.GetMatrixFormat();
    };
}}}

namespace std
{
    template <> struct hash<Microsoft::MSR::CNTK::InputStreamDescription>
    {
        size_t operator()(const Microsoft::MSR::CNTK::InputStreamDescription& x) const
        {
            // Input name is unique, simply return the hash of the input stream.
            return std::hash<std::wstring>()(x.GetStreamName());
        }
    };
}

namespace Microsoft { namespace MSR { namespace CNTK {

// randomize range set automatically, parameter value for Init()
const size_t randomizeAuto = ((size_t) -1) >> 2;

// don't randomize, parameter value for Init()
const size_t randomizeNone = 0;

// StartMinibatchLoop default parameter, sets number of requested
// frames equal to the constant 3fffffffffffffff computed by ((size_t) -1) >> 2 above.
// We use this constant as a stand in for the total number of frames in the dataset.
const size_t requestDataSize = randomizeAuto;

// this class contains the input data structures to be filled in by the GetMinibatch() call
class StreamMinibatchInputs
{
public:
    struct Input
    {
        /*const*/ MatrixBasePtr matrix;
        /*const*/ MBLayoutPtr pMBLayout;
        /*const*/ TensorShape sampleLayout;

        // constructor
        Input(MatrixBasePtr matrix, MBLayoutPtr pMBLayout, TensorShape sampleLayout) : 
            matrix(matrix), pMBLayout(pMBLayout), sampleLayout(sampleLayout)
        {
            assert(matrix);
        }
        Input() {} // some STL classes need this for general happiness

        // helper for typecasting the matrix pointer
    template<class ElemType>
        Matrix<ElemType>& GetMatrix(const wchar_t* name/*for debugging only*/ = L"(unknown)") const
    {
            assert(matrix);
            auto* matrixp = dynamic_cast<Matrix<ElemType>*>(matrix.get());
        if (!matrixp)
        {
            // print a rather rich error to track down a regression failure
                auto isFloat  = !!dynamic_cast<Matrix<float>*> (matrix.get());
                auto isDouble = !!dynamic_cast<Matrix<double>*>(matrix.get());
                LogicError("GetMatrix<%s>: Attempted to access input stream '%ls' with wrong precision, got %s {%d,%d} instead of %s.",
                    typeid(ElemType).name(), name, typeid(matrix.get()).name(), (int)isFloat, (int)isDouble, typeid(Matrix<ElemType>*).name());
        }
        return *matrixp;
    }
    };

    std::function<std::string(size_t)> m_getKeyById;

private:
    typedef map<std::wstring, Input> MapType;
    MapType inputs;

public:
    void AddInput(const std::wstring& nodeName, const Input& input)
    {
        assert(input.matrix);
        inputs[nodeName] = input;
    }
    void AddInput(const std::wstring& nodeName, const MatrixBasePtr& matrix, const MBLayoutPtr& pMBLayout, const TensorShape& sampleLayout)
    {
        AddInput(nodeName, Input(matrix, pMBLayout, sampleLayout));
    }
    bool HasInput(const std::wstring& nodeName) const { return inputs.find(nodeName) != inputs.end(); }
    const Input& GetInput(const std::wstring& nodeName) const
    {
        auto iter = inputs.find(nodeName);
        if (iter == inputs.end())
            LogicError("GetInputMatrix: Attempted to access non-existent input stream '%ls'", nodeName.c_str());
        return iter->second;
    }
    template<class ElemType>
    Matrix<ElemType>& GetInputMatrix(const std::wstring& nodeName) const
    {
        const auto& input = GetInput(nodeName);
        return input.GetMatrix<ElemType>(nodeName.c_str());
    }
    // iterating
    // TODO: Abstract this.
    MapType::iterator begin() { return inputs.begin(); }
    MapType::iterator end()   { return inputs.end(); }
    MapType::iterator find(const std::wstring& nodeName) { return inputs.find(nodeName); }
    MapType::const_iterator begin() const { return inputs.begin(); }
    MapType::const_iterator end()   const { return inputs.end(); }
    MapType::const_iterator find(const std::wstring& nodeName) const { return inputs.find(nodeName); }
    void clear() { inputs.clear(); }
    // only used by test code:
    void insert(std::pair<wstring, Input> pair) { inputs.insert(pair); }

    // Returns description of required streams for the minibatch.
    std::unordered_set<InputStreamDescription> GetStreamDescriptions() const
    {
        std::unordered_set<InputStreamDescription> streamDescriptions;
        for (auto input = begin(); input != end(); ++input)
        {
            streamDescriptions.insert(
                InputStreamDescription(input->first, input->second.matrix->GetDeviceId(), input->second.matrix->GetMatrixType(), input->second.matrix->GetFormat()));
        }
        return streamDescriptions;
    }
};

// Data Reader interface
// implemented by DataReader and underlying classes
class DATAREADER_API IDataReader
{
public:
    typedef std::string  LabelType;     // surface form of an input token
    typedef unsigned int LabelIdType;   // input token mapped to an integer  --TODO: why not size_t? Does this save space?

    // BUGBUG: We should not have data members in an interace!
    unsigned m_seed;
    size_t mRequestedNumParallelSequences; // number of desired parallel sequences in each minibatch

    virtual void Init(const ConfigParameters& config) = 0;
    virtual void Init(const ScriptableObjects::IConfigRecord& config) = 0;
    virtual void Destroy() = 0;
protected:
    virtual ~IDataReader() { }
public:
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, const std::unordered_set<InputStreamDescription>& /*requiredStreams*/, size_t requestedEpochSamples = requestDataSize)
    {
        StartMinibatchLoop(mbSize, epoch, requestedEpochSamples);
    }

    virtual void StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, const std::unordered_set<InputStreamDescription>& /*requiredStreams*/, size_t requestedEpochSamples = requestDataSize)
    {
        StartDistributedMinibatchLoop(mbSize, epoch, subsetNum, numSubsets, requestedEpochSamples);
    }

    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize) = 0;

    virtual bool SupportsDistributedMBRead() const
    {
        return false;
    };

    // old DataReader architecture
    virtual bool IsLegacyReader() const
    {
        return true;
    };
    
    // Gets current sample position on the global timeline.
    virtual size_t GetCurrentSamplePosition()
    {
        NOT_IMPLEMENTED;
    }

    virtual void StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples = requestDataSize)
    {
        if (SupportsDistributedMBRead() || (numSubsets != 1) || (subsetNum != 0))
        {
            LogicError("This reader does not support distributed reading of mini-batches");
        }

        return StartMinibatchLoop(mbSize, epoch, requestedEpochSamples);
    }

    virtual bool GetMinibatch(StreamMinibatchInputs& matrices) = 0;
    virtual bool GetMinibatch4SE(std::vector<shared_ptr<const msra::dbn::latticepair>>& /*latticeinput*/, vector<size_t>& /*uids*/, vector<size_t>& /*boundaries*/, vector<size_t>& /*extrauttmap*/)
    {
        NOT_IMPLEMENTED;
    };
    virtual bool GetHmmData(msra::asr::simplesenonehmm* /*hmm*/)
    {
        NOT_IMPLEMENTED;
    };

    // TODO: Should be removed when BPTT follows proper minibatch size.
    virtual size_t GetNumParallelSequencesForFixingBPTTMode() = 0;

    //virtual int GetSentenceEndIdFromOutputLabel() { return -1; }
    virtual void SetNumParallelSequences(const size_t sz)
    {
        mRequestedNumParallelSequences = sz;
    }
    //virtual bool RequireSentenceSeg() const { return false; }
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring&)
    {
        NOT_IMPLEMENTED;
    }
    virtual void SetLabelMapping(const std::wstring&, const std::map<LabelIdType, LabelType>&)
    {
        NOT_IMPLEMENTED;
    }
    virtual bool GetData(const std::wstring&, size_t, void*, size_t&, size_t)
    {
        NOT_IMPLEMENTED;
    }
    virtual bool DataEnd()
    {
        NOT_IMPLEMENTED;
    }
    virtual void CopyMBLayoutTo(MBLayoutPtr)
    {
        NOT_IMPLEMENTED;
    }
    virtual void SetRandomSeed(unsigned seed = 0)
    {
        m_seed = seed;
    }
    virtual bool GetProposalObs(StreamMinibatchInputs*, const size_t, vector<size_t>&)
    {
        return false;
    }
    virtual void InitProposals(StreamMinibatchInputs*)
    {
    }
    virtual bool CanReadFor(wstring /* nodeName */) // return true if this reader can output for a node with name nodeName  --TODO: const wstring&
    {
        return false;
    }

    bool GetFrame(StreamMinibatchInputs& /*matrices*/, const size_t /*tidx*/, vector<size_t>& /*history*/)
    {
        NOT_IMPLEMENTED;
    }

    // Workaround for the two-forward-pass sequence and ctc training, which
    // allows processing more utterances at the same time. Only used in
    // Kaldi2Reader.
    // TODO: move this out of the reader.
    virtual bool GetMinibatchCopy(
        std::vector<std::vector<std::pair<wstring, size_t>>>& /*uttInfo*/,
        StreamMinibatchInputs& /*matrices*/,
        MBLayoutPtr /*data copied here*/)
    {
        return false;
    }

    // Workaround for the two-forward-pass sequence and ctc training, which
    // allows processing more utterances at the same time. Only used in
    // Kaldi2Reader.
    // TODO: move this out of the reader.
    virtual bool SetNetOutput(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& /*uttInfo*/,
        const MatrixBase& /*outputs*/,
        const MBLayoutPtr)
    {
        return false;
    }
};
typedef std::shared_ptr<IDataReader> IDataReaderPtr;

// GetReaderX() - get a reader type from the DLL
// The F version gets the 'float' version, and D gets 'double'.
extern "C" DATAREADER_API void GetReaderF(IDataReader** preader);
extern "C" DATAREADER_API void GetReaderD(IDataReader** preader);

// The sole purpose of this base class is to provide backwards compatibility for (old)
// readers that do not support multiple mb layouts.
class DataReaderBase : public IDataReader
{
protected:
    // Verifies that all inputs share the same layout (have the same layout pointer) 
    // and copies the provided layout into the minibatch layout.
    // This method is needed for backwards-compatibility and only meant to be used by old readers!
    void SetMinibatchLayout(StreamMinibatchInputs& minibatch);

    virtual bool TryGetMinibatch(StreamMinibatchInputs& matrices) = 0;
public:
    virtual bool GetMinibatch(StreamMinibatchInputs& matrices) override;
};

// Data Reader class
// interface for clients of the Data Reader
// mirrors the IDataReader interface, except the Init method is private (use the constructor)
class DataReader : public IDataReader, protected Plugin, public ScriptableObjects::Object
{
    vector<wstring> m_ioNames;                          // TODO: why are these needed, why not loop over m_dataReaders?
    map<wstring, IDataReader*> m_dataReaders; // readers

    // Init - Reader Initialize for multiple data sets
    // config - [in] configuration parameters for the datareader
    // Sample format below for CNTKTextFormatReader:
    // # Parameter values for the reader
    // reader=[
    //  # reader to use
    //  readerType="CNTKTextFormatReader"
    //  randomize=false
    //  file="c:\speech\mnist\mnist_test_cntk_text.txt"
    //  input=[
    //    features=[
    //      dim=784
    //      format="dense"
    //    ]
    //    labels=[
    //      dim=10
    //      format="dense"
    //    ]
    //  ]
    //]
    template <class ConfigRecordType>
    void InitFromConfig(const ConfigRecordType&);
    virtual void Init(const ConfigParameters& config) override
    {
        InitFromConfig(config);
    }
    virtual void Init(const ScriptableObjects::IConfigRecord& config) override
    {
        InitFromConfig(config);
    }

    // Destroy - cleanup and remove this class
    // NOTE: this destroys the object, and it can't be used past this point.
    // The reason why this is not just a destructor is that it goes across a DLL boundary.
    virtual void Destroy() override;
public:
    // DataReader Constructor
    // config - [in] configuration parameters for the datareader
    template <class ConfigRecordType>
    DataReader(const ConfigRecordType& config);
    // constructor from Scripting
    DataReader(const ScriptableObjects::IConfigRecordPtr configp)
        : DataReader(*configp)
    {
    }
    virtual ~DataReader();

    size_t GetCurrentSamplePosition() override;

    // StartMinibatchLoop - Startup a minibatch loop
    // mbSize - [in] size of the minibatch (number of frames, etc.)
    // epoch - [in] epoch number for this loop
    // requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize);

    virtual bool SupportsDistributedMBRead() const override;
    virtual bool IsLegacyReader() const override;
    virtual void StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples = requestDataSize) override;

    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, const std::unordered_set<InputStreamDescription>&, size_t requestedEpochSamples = requestDataSize) override;
    virtual void StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, const std::unordered_set<InputStreamDescription>&, size_t requestedEpochSamples = requestDataSize) override;

    // GetMinibatch - Get the next minibatch (features and labels)
    // matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponding matrix,
    //             [out] each matrix resized if necessary containing data.
    // returns - true if there are more minibatches, false if no more minibatchs remain
    virtual bool GetMinibatch(StreamMinibatchInputs& matrices);
    virtual bool GetMinibatch4SE(std::vector<shared_ptr<const msra::dbn::latticepair>>& latticeinput, vector<size_t>& uids, vector<size_t>& boundaries, vector<size_t>& extrauttmap);
    virtual bool GetHmmData(msra::asr::simplesenonehmm* hmm);

    size_t GetNumParallelSequencesForFixingBPTTMode();
    //int GetSentenceEndIdFromOutputLabel();
    //bool RequireSentenceSeg() const override;

    // GetLabelMapping - Gets the label mapping from integer index to label type
    // returns - a map from numeric datatype to native label type
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring& sectionName);

    // SetLabelMapping - Sets the label mapping from integer index to label
    // labelMapping - mapping table from label values to IDs (must be 0-n)
    // note: for tasks with labels, the mapping table must be the same between a training run and a testing run
    virtual void SetLabelMapping(const std::wstring& sectionName, const std::map<LabelIdType, LabelType>& labelMapping);

    // GetData - Gets metadata from the specified section (into CPU memory)
    // sectionName - section name to retrieve data from
    // numRecords - number of records to read
    // data - pointer to data buffer, if NULL, dataBufferSize will be set to size of required buffer to accomidate request
    // dataBufferSize - [in] size of the databuffer in bytes
    //                  [out] size of buffer filled with data
    // recordStart - record to start reading from, defaults to zero (start of data)
    // returns: true if data remains to be read, false if the end of data was reached
    virtual bool GetData(const std::wstring& sectionName, size_t numRecords, void* data, size_t& dataBufferSize, size_t recordStart = 0);

    virtual bool DataEnd();
    // TODO: The return value if this is never used except in loops where we do an &=. It is not clear whether that is a bug or intentionally prevents DataEnd() from being called.
    //       Once this is understood, we can change the return value to void.

    // Gets a copy of the minibatch for the forward computation. This can be
    // useful if some of the computation has to happen in the reader.
    virtual bool GetMinibatchCopy(
        std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        StreamMinibatchInputs& matrices,
        MBLayoutPtr);

    // Sets the neural network output to the reader. This can be useful if some
    // of the computation has to happen in the reader.
    virtual bool SetNetOutput(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        const MatrixBase& outputs,
        const MBLayoutPtr);

    void CopyMBLayoutTo(MBLayoutPtr pMBLayout);

    void SetRandomSeed(int);

    bool GetProposalObs(StreamMinibatchInputs*, const size_t, vector<size_t>&);
    void InitProposals(StreamMinibatchInputs* matrices);
};

}}}
