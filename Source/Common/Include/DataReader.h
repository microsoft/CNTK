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

// forward-declare these lattice-related types to avoid having to include and pollute everything with lattice-related headers
namespace msra { namespace dbn {  class latticepair; class latticesource; } }
namespace msra { namespace asr {  class simplesenonehmm; } } 

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
    typedef map<std::wstring, MatrixBasePtr> MapType;
    MapType matrices;
public:
    void AddInput(const std::wstring& nodeName, const MatrixBasePtr& matrix) { AddInputMatrix(nodeName, matrix); } // use this where entire entry is copied (UCIFastReader::GetMinibatch() async)
    // TODO: GetInput() will return a struct
    // access to matrix entries
    void AddInputMatrix(const std::wstring& nodeName, const MatrixBasePtr& matrix) { matrices[nodeName] = matrix; }
    bool HasInput(const std::wstring& nodeName) const { return matrices.find(nodeName) != matrices.end(); }
    template<class ElemType>
    Matrix<ElemType>& GetInputMatrix(const std::wstring& nodeName) const
    {
        auto iter = matrices.find(nodeName);
        if (iter == matrices.end())
            LogicError("GetInputMatrix: Attempted to access non-existent input stream '%ls'", nodeName.c_str());
        assert(iter->second);
        auto* matrixp = dynamic_cast<Matrix<ElemType>*>(iter->second.get());
        if (!matrixp)
        {
            // print a rather rich error to track down a regression failure
            auto isFloat  = !!dynamic_cast<Matrix<float>*>(iter->second.get());
            auto isDouble = !!dynamic_cast<Matrix<double>*>(iter->second.get());
            LogicError("GetInputMatrix<%s>: Attempted to access input stream '%ls' with wrong precision, got %s {%d,%d} instead of %s.",
                        typeid(ElemType).name(), nodeName.c_str(), typeid(iter->second.get()).name(), (int)isFloat, (int)isDouble, typeid(Matrix<ElemType>*).name());
        }
        return *matrixp;
    }
    // iterating
    // TODO: Abstract this.
    MapType::iterator begin() { return matrices.begin(); }
    MapType::iterator end()   { return matrices.end(); }
    MapType::iterator find(const std::wstring& nodeName) { return matrices.find(nodeName); }
    MapType::const_iterator begin() const { return matrices.begin(); }
    MapType::const_iterator end()   const { return matrices.end(); }
    MapType::const_iterator find(const std::wstring& nodeName) const { return matrices.find(nodeName); }
    void clear() { matrices.clear(); }
    // only used by test code:
    void insert(std::pair<wstring, shared_ptr<MatrixBase>> pair) { matrices.insert(pair); }
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
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize) = 0;

    virtual bool SupportsDistributedMBRead() const
    {
        return false;
    };
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
    virtual size_t GetNumParallelSequences() = 0;
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

// Data Reader class
// interface for clients of the Data Reader
// mirrors the IDataReader interface, except the Init method is private (use the constructor)
class DataReader : public IDataReader, protected Plugin, public ScriptableObjects::Object
{
    vector<wstring> m_ioNames;                          // TODO: why are these needed, why not loop over m_dataReaders?
    map<wstring, IDataReader*> m_dataReaders; // readers

    // Init - Reader Initialize for multiple data sets
    // config - [in] configuration parameters for the datareader
    // Sample format below for UCIReader:
    // # Parameter values for the reader
    // reader=[
    //  # reader to use
    //  readerType="UCIFastReader"
    //  miniBatchMode="partial"
    //  randomize=None
    //  features=[
    //    dim=784
    //    start=1
    //    file="c:\speech\mnist\mnist_test.txt"
    //  ]
    //  labels=[
    //    dim=1
    //      start=0
    //      file="c:\speech\mnist\mnist_test.txt"
    //      labelMappingFile="c:\speech\mnist\labels.txt"
    //      labelDim=10
    //      labelType="category"
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

    // StartMinibatchLoop - Startup a minibatch loop
    // mbSize - [in] size of the minibatch (number of frames, etc.)
    // epoch - [in] epoch number for this loop
    // requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize);

    virtual bool SupportsDistributedMBRead() const override;
    virtual void StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples = requestDataSize) override;

    // GetMinibatch - Get the next minibatch (features and labels)
    // matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponding matrix,
    //             [out] each matrix resized if necessary containing data.
    // returns - true if there are more minibatches, false if no more minibatchs remain
    virtual bool GetMinibatch(StreamMinibatchInputs& matrices);
    virtual bool GetMinibatch4SE(std::vector<shared_ptr<const msra::dbn::latticepair>>& latticeinput, vector<size_t>& uids, vector<size_t>& boundaries, vector<size_t>& extrauttmap);
    virtual bool GetHmmData(msra::asr::simplesenonehmm* hmm);

    size_t GetNumParallelSequences();
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
