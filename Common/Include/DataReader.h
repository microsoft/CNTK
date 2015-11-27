//
// <copyright file="DataReader.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
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
#include "commandArgUtil.h" // for ConfigParameters
#include "ScriptableObjects.h"
#include <map>
#include <string>

// forward-declare these lattice-related types to avoid having to include and pollute everything with lattice-related headers
namespace msra { namespace dbn {
    class latticepair;
    class latticesource;
}}
namespace msra { namespace asr {
    class simplesenonehmm;
}}

namespace Microsoft { namespace MSR { namespace CNTK {

// randomize range set automatically, parameter value for Init()
const size_t randomizeAuto = ((size_t) -1) >> 2;

// don't randomize, parameter value for Init()
const size_t randomizeNone = 0;

// StartMinibatchLoop default parameter, sets number of requested
// frames equal to the constant 3fffffffffffffff computed by ((size_t) -1) >> 2 above.
// We use this constant as a stand in for the total number of frames in the dataset.
const size_t requestDataSize = randomizeAuto;

enum EndDataType
{
    endDataNull, // null values
    endDataEpoch, // end of epoch
    endDataSet, // end of dataset
    endDataSentence, // end of sentence
};

// Data Reader interface
// implemented by DataReader and underlying classes
template<class ElemType>
class DATAREADER_API IDataReader
{
public:
    typedef std::string LabelType;
    typedef unsigned int LabelIdType;
    unsigned m_seed;
    size_t   mBlgSize;  /// number of utterances per minibatch

    virtual void Init(const ConfigParameters & config) = 0;
    virtual void Init(const ScriptableObjects::IConfigRecord & config) = 0;
    virtual void Destroy() = 0;
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples=requestDataSize) = 0;

    virtual bool SupportsDistributedMBRead() const { return false; };
    virtual void StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples = requestDataSize)
    {
        if (SupportsDistributedMBRead() || (numSubsets != 1) || (subsetNum != 0))
        {
            LogicError("This reader does not support distributed reading of mini-batches");
        }

        return StartMinibatchLoop(mbSize, epoch, requestedEpochSamples);
    }

    virtual bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices) = 0;
    virtual bool GetMinibatch4SE(std::vector<shared_ptr<const msra::dbn::latticepair>> & /*latticeinput*/, vector<size_t> &/*uids*/, vector<size_t> &/*boundaries*/, vector<size_t> &/*extrauttmap*/) { NOT_IMPLEMENTED; };
    virtual bool GetHmmData(msra::asr::simplesenonehmm * /*hmm*/) { NOT_IMPLEMENTED; };
    virtual size_t GetNumParallelSequences() = 0; 
    virtual int GetSentenceEndIdFromOutputLabel() { return -1; }
    virtual void SetNumParallelSequences(const size_t sz) { mBlgSize = sz; }
    virtual bool RequireSentenceSeg() const { return false; }
    virtual const std::map<LabelIdType, LabelType>& GetLabelMapping(const std::wstring&) { NOT_IMPLEMENTED; }
    virtual void SetLabelMapping(const std::wstring&, const std::map<LabelIdType, LabelType>&) { NOT_IMPLEMENTED; }
    virtual bool GetData(const std::wstring&, size_t, void*, size_t&, size_t) { NOT_IMPLEMENTED; }
    virtual bool DataEnd(EndDataType) { NOT_IMPLEMENTED; }
    virtual void CopyMBLayoutTo(MBLayoutPtr) { NOT_IMPLEMENTED; }
    virtual void SetRandomSeed(unsigned seed = 0) { m_seed = seed; }
    virtual bool GetProposalObs(std::map<std::wstring, Matrix<ElemType>*>*, const size_t, vector<size_t>&) { return false; }
    virtual void InitProposals(std::map<std::wstring, Matrix<ElemType>*>*) { }
    virtual bool CanReadFor(wstring /* nodeName */) { return false; }

    bool GetFrame(std::map<std::wstring, Matrix<ElemType>*>& /*matrices*/, const size_t /*tidx*/, vector<size_t>& /*history*/) { NOT_IMPLEMENTED; }

    // Workaround for the two-forward-pass sequence and ctc training, which
    // allows processing more utterances at the same time. Only used in
    // Kaldi2Reader.
    // TODO: move this out of the reader.
    virtual bool GetMinibatchCopy(
        std::vector<std::vector<std::pair<wstring, size_t>>>& /*uttInfo*/,
        std::map<std::wstring, Matrix<ElemType>*>& /*matrices*/,
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
        const Matrix<ElemType>& /*outputs*/,
        const MBLayoutPtr)
    {
        return false;
    }
};

// GetReader - get a reader type from the DLL
// since we have 2 reader types based on template parameters, exposes 2 exports
// could be done directly the templated name, but that requires mangled C++ names
template<class ElemType>
void DATAREADER_API GetReader(IDataReader<ElemType>** preader);
extern "C" DATAREADER_API void GetReaderF(IDataReader<float>** preader);
extern "C" DATAREADER_API void GetReaderD(IDataReader<double>** preader);

// Data Reader class
// interface for clients of the Data Reader
// mirrors the IDataReader interface, except the Init method is private (use the constructor)
template<class ElemType>
class DataReader: public IDataReader<ElemType>, protected Plugin, public ScriptableObjects::Object
{
    typedef typename IDataReader<ElemType>::LabelType LabelType;
    typedef typename IDataReader<ElemType>::LabelIdType LabelIdType;
private:
    vector<wstring> m_ioNames;                              // TODO: why are these needed, why not loop over m_dataReaders?
    map<wstring, IDataReader<ElemType> *> m_dataReaders;    // readers

    // Init - Reader Initialize for multiple data sets
    // config - [in] configuration parameters for the datareader
    // Sample format below for UCIReader:
    //# Parameter values for the reader
    //reader=[
    //  # reader to use
    //  readerType=UCIFastReader
    //  miniBatchMode=Partial
    //  randomize=None
    //  features=[
    //    dim=784
    //    start=1
    //    file=c:\speech\mnist\mnist_test.txt
    //  ]
    //  labels=[
    //    dim=1
    //      start=0
    //      file=c:\speech\mnist\mnist_test.txt
    //      labelMappingFile=c:\speech\mnist\labels.txt
    //      labelDim=10
    //      labelType=Category
    //  ]
    //]
    template<class ConfigRecordType> void InitFromConfig(const ConfigRecordType &);
    virtual void Init(const ConfigParameters & config) override { InitFromConfig(config); }
    virtual void Init(const ScriptableObjects::IConfigRecord & config) override { InitFromConfig(config); }

    // Destroy - cleanup and remove this class
    // NOTE: this destroys the object, and it can't be used past this point.
    // The reason why this is not just a destructor is that it goes across a DLL boundary.
    virtual void Destroy() override;

public:
    // DataReader Constructor
    // config - [in] configuration parameters for the datareader
    template<class ConfigRecordType>
    DataReader(const ConfigRecordType& config);
    // constructor from Scripting
    DataReader(const ScriptableObjects::IConfigRecordPtr configp) :
        DataReader(*configp)
    { }
    virtual ~DataReader();

    //StartMinibatchLoop - Startup a minibatch loop 
    // mbSize - [in] size of the minibatch (number of frames, etc.)
    // epoch - [in] epoch number for this loop
    // requestedEpochSamples - [in] number of samples to randomize, defaults to requestDataSize which uses the number of samples there are in the dataset
    virtual void StartMinibatchLoop(size_t mbSize, size_t epoch, size_t requestedEpochSamples = requestDataSize);

    virtual bool SupportsDistributedMBRead() const override;
    virtual void StartDistributedMinibatchLoop(size_t mbSize, size_t epoch, size_t subsetNum, size_t numSubsets, size_t requestedEpochSamples = requestDataSize) override;

    // GetMinibatch - Get the next minibatch (features and labels)
    // matrices - [in] a map with named matrix types (i.e. 'features', 'labels') mapped to the corresponing matrix, 
    //             [out] each matrix resized if necessary containing data. 
    // returns - true if there are more minibatches, false if no more minibatchs remain
    virtual bool GetMinibatch(std::map<std::wstring, Matrix<ElemType>*>& matrices);
    virtual bool GetMinibatch4SE(std::vector<shared_ptr<const msra::dbn::latticepair>> & latticeinput, vector<size_t> &uids, vector<size_t> &boundaries, vector<size_t> &extrauttmap);
    virtual bool GetHmmData(msra::asr::simplesenonehmm * hmm);

    size_t GetNumParallelSequences();
    int GetSentenceEndIdFromOutputLabel();
    bool RequireSentenceSeg() const override;

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

    virtual bool DataEnd(EndDataType endDataType);

    // Gets a copy of the minibatch for the forward computation. This can be
    // useful if some of the computation has to happen in the reader.
    virtual bool GetMinibatchCopy(
        std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        std::map<std::wstring, Matrix<ElemType>*>& matrices,
        MBLayoutPtr);

    // Sets the neural network output to the reader. This can be useful if some
    // of the computation has to happen in the reader.
    virtual bool SetNetOutput(
        const std::vector<std::vector<std::pair<wstring, size_t>>>& uttInfo,
        const Matrix<ElemType>& outputs,
        const MBLayoutPtr);

    void CopyMBLayoutTo(MBLayoutPtr pMBLayout);

    void SetRandomSeed(int);

    bool GetProposalObs(std::map<std::wstring, Matrix<ElemType>*>*, const size_t, vector<size_t>&);
    void InitProposals(std::map<std::wstring, Matrix<ElemType>*>* matrices);

};

}}}
