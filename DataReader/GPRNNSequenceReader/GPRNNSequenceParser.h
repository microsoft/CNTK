// GPRNNSequenceParser.h
//
// <copyright file="GPRNNSequenceParser.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
//

#include <string>
#include <vector>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <map>
#include <memory>
#include <stdint.h>
#include "Platform.h"
#include "DataReader.h"

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

#define MAXSTRING 2048

// SequencePosition, save the ending indexes into the array for a sequence
struct SequencePosition
{
    long inputPos; // max position in the number array for this sequence
    long labelPos; // max position in the label array for this sequence
    unsigned flags; // flags that apply to this sequence
    SequencePosition(long inPos, long labelPos, unsigned flags) :
            inputPos(inPos), labelPos(labelPos), flags(flags)
    {}
};


    // GPRNNSequenceParser - the parser for GPRNN feature extracted format.
template <typename NumType, typename LabelType = wstring>
class GPRNNSequenceParser
{
public:
    using LabelIdType = long;

protected:
    // definition of label and feature dimensions
    size_t m_dimFeatures;

    size_t m_dimLabelsIn;
    wstring m_beginSequenceIn; // starting sequence string (i.e. <s>)
    wstring m_endSequenceIn; // ending sequence string (i.e. </s>)

    size_t m_dimLabelsOut;
    wstring m_beginSequenceOut; // starting sequence string (i.e. 'O')
    wstring m_endSequenceOut; // ending sequence string (i.e. 'O')

    // level of screen output
    int m_traceLevel;

    // sequence state machine variables
    bool m_beginSequence;
    bool m_endSequence;
    wstring m_beginTag;
    wstring m_endTag;

    // file positions/buffer
    FILE * m_pFile;

    BYTE * m_fileBuffer;

    std::vector<std::pair<std::vector<LabelIdType>, std::vector<std::pair<LabelIdType, LabelIdType> > > > *m_inputs; // pointer to vectors of annotation which itself is a vector.
    std::vector<LabelIdType>* m_labels; // pointer to vector to append with labels (may be numeric)
    // FUTURE: do we want a vector to collect string labels in the non string label case? (signifies an error)

public:

    // GPRNNSequenceParser constructor
    GPRNNSequenceParser();
    // setup all the state variables and state tables for state machine
    void Init();

    // Parser destructor
    ~GPRNNSequenceParser();

public:

    // ParseInit - Initialize a parse of a file
    // fileName - path to the file to open
    // dimFeatures - number of features for precomputed features
    // dimLabelsIn - number of lables possible on input
    // dimLabelsOut - number of labels possible on output
    // beginSequenceIn - beginSequence input label
    // endSequenceIn - endSequence input label
    // beginSequenceOut - beginSequence output label
    // endSequenceOut - endSequence output label
    // bufferSize - size of temporary buffer to store reads
    // startPosition - file position on which we should start
    void ParseInit(LPCWSTR fileName, size_t dimFeatures, size_t dimLabelsIn, size_t dimLabelsOut, wstring beginSequenceIn, wstring endSequenceIn, wstring beginSequenceOut, wstring endSequenceOut )
    {
        assert(fileName != NULL);
        m_dimFeatures = dimFeatures;
        m_dimLabelsIn = dimLabelsIn;
        m_beginSequenceIn = beginSequenceIn;
        m_endSequenceIn = endSequenceIn;
        m_dimLabelsOut = dimLabelsOut;
        m_beginSequenceOut = beginSequenceOut;
        m_endSequenceOut = endSequenceOut;

        m_traceLevel = 0;

        m_beginTag = m_beginSequenceIn;
        m_endTag = m_endSequenceIn;

        // if we have a file already open, cleanup
        if (m_pFile != NULL)
            GPRNNSequenceParser<NumType, LabelType>::~GPRNNSequenceParser();

        errno_t err = _wfopen_s( &m_pFile, fileName, L"rb" );
        if (err)
            RuntimeError("GPRNNSequenceParser::ParseInit - error opening file"); 
        int rc = _fseeki64(m_pFile, 0, SEEK_END);
        if (rc)
            RuntimeError("GPRNNSequenceParser::ParseInit - error seeking in file");
    }
};

/// language model sequence parser
typedef struct{
    size_t sLen;
    int sBegin;
    int sEnd;
} stSentenceInfo;

template <typename NumType, typename LabelType = wstring>
class BatchGPRNNSequenceParser : public GPRNNSequenceParser<NumType, LabelType>
{
public:
    wstring mUnkStr; 

public:
    std::wifstream mFile; 
    std::wstring mFileName; 
    vector<stSentenceInfo> mSentenceIndex2SentenceInfo;

public:
    using GPRNNSequenceParser<NumType, LabelType>::m_dimFeatures;
    using GPRNNSequenceParser<NumType, LabelType>::m_dimLabelsIn;
    using GPRNNSequenceParser<NumType, LabelType>::m_beginSequenceIn;
    using GPRNNSequenceParser<NumType, LabelType>::m_endSequenceIn;
    using GPRNNSequenceParser<NumType, LabelType>::m_dimLabelsOut;
    using GPRNNSequenceParser<NumType, LabelType>::m_beginSequenceOut;
    using GPRNNSequenceParser<NumType, LabelType>::m_endSequenceOut;
    using GPRNNSequenceParser<NumType, LabelType>::m_traceLevel;
    using GPRNNSequenceParser<NumType, LabelType>::m_beginTag;
    using GPRNNSequenceParser<NumType, LabelType>::m_endTag;
    using GPRNNSequenceParser<NumType, LabelType>::m_fileBuffer;
    using GPRNNSequenceParser<NumType, LabelType>::m_inputs;
    using GPRNNSequenceParser<NumType, LabelType>::m_labels;
    using GPRNNSequenceParser<NumType, LabelType>::m_beginSequence;
    using GPRNNSequenceParser<NumType, LabelType>::m_endSequence;
    BatchGPRNNSequenceParser() {
    };
    ~BatchGPRNNSequenceParser() {
        mFile.close(); 
    }

    void ParseInit(LPCWSTR fileName, size_t dimLabelsIn, size_t dimLabelsOut, wstring beginSequenceIn, wstring endSequenceIn, wstring beginSequenceOut, wstring endSequenceOut, wstring unkstr = "<UNK>")
    {
        assert(fileName != NULL);
        mFileName = fileName;
        m_dimLabelsIn = dimLabelsIn;
        m_beginSequenceIn = beginSequenceIn;
        m_endSequenceIn = endSequenceIn;
        m_dimLabelsOut = dimLabelsOut;
        m_beginSequenceOut = beginSequenceOut;
        m_endSequenceOut = endSequenceOut;

        m_traceLevel = 0;

        m_beginTag = m_beginSequenceIn;
        m_endTag = m_endSequenceIn;

        mUnkStr = unkstr; 

        mFile.close();
#ifdef __unix__
        mFile.open(ws2s(fileName), wifstream::in);
#else
        mFile.open(fileName, wifstream::in);
#endif
        if (!mFile.good())
            RuntimeError("cannot open file %ls", fileName);
    }

    void ParseReset()
    {
        mFile.close();
#ifdef __unix__
        mFile.open(ws2s(mFileName), wifstream::in);
#else
        mFile.open(mFileName, wifstream::in);
#endif
        if (!mFile.good())
            RuntimeError("cannot open file %ls", mFileName.c_str());
    }

    void AddOneItem(std::vector<long> *labels, std::vector<std::pair<vector<LabelIdType>, std::vector<std::pair<LabelIdType, LabelIdType> > > > *input, std::vector<SequencePosition> *seqPos, long& lineCount,
        long & recordCount, long orgRecordCount, SequencePosition& sequencePositionLast)
    {
        SequencePosition sequencePos((long)input->size(), (long)labels->size(), 1);
        seqPos->push_back(sequencePos);
        sequencePositionLast = sequencePos;

        recordCount = (long)labels->size() - orgRecordCount;
        lineCount++;
    }

    // Parse - Parse the data
    // recordsRequested - number of records requested
    // labels - pointer to vector to return the labels 
    // numbers - pointer to vector to return the numbers 
    // seqPos - pointers to the other two arrays showing positions of each sequence
    // returns - number of records actually read, if the end of file is reached the return value will be < requested records
    // Parse - Parse the data
    // recordsRequested - number of records requested
    // labels - pointer to vector to return the labels 
    // numbers - pointer to vector to return the numbers 
    // seqPos - pointers to the other two arrays showing positions of each sequence
    // returns - number of records actually read, if the end of file is reached the return value will be < requested records
    long Parse(size_t recordsRequested, std::vector<long> *labels, std::vector<std::pair<std::vector<LabelIdType>, std::vector<std::pair<LabelIdType, LabelIdType> > > > *input, std::vector<SequencePosition> *seqPos, bool mAllowMultPassData = false);

};

}}};
