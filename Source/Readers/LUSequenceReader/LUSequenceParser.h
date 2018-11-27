//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// LUSequenceParser.h : Parses the UCI format using a custom state machine (for speed)
//

#pragma once

#include <string>
#include <vector>
#include <assert.h>
#include <fstream>
#include <iostream>
#include <map>
#include <stdint.h>
#include "Platform.h"
#include "DataReader.h"

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

#define MAXSTRING 2048

// SequencePosition, save the ending indexes into the array for a sequence
struct SequencePosition
{
    long inputPos;  // max position in the number array for this sequence
    long labelPos;  // max position in the label array for this sequence
    unsigned flags; // flags that apply to this sequence   --TODO: We really need to know at least what those flags are, if an enum is asking for too much.
    SequencePosition(long inPos, long labelPos, unsigned flags)
        : inputPos(inPos), labelPos(labelPos), flags(flags)
    {
    }
};

// LUSequenceParser - the parser for the UCI format files
// for ultimate speed, this class implements a state machine to read these format files
template <typename NumType, typename LabelType = wstring>
class LUSequenceParser
{
public:
    using LabelIdType = long;

protected:
    // definition of label and feature dimensions
    size_t m_dimFeatures;

    size_t m_dimLabelsIn;
    wstring m_beginSequenceIn; // starting sequence string (i.e. <s>)
    wstring m_endSequenceIn;   // ending sequence string (i.e. </s>)

    size_t m_dimLabelsOut;
    wstring m_beginSequenceOut; // starting sequence string (i.e. 'O')
    wstring m_endSequenceOut;   // ending sequence string (i.e. 'O')

    // level of screen output
    int m_traceLevel;

    // sequence state machine variables
    bool m_beginSequence;
    bool m_endSequence;
    wstring m_beginTag;
    wstring m_endTag;

    // file positions/buffer
    FILE* m_pFile;

    BYTE* m_fileBuffer;

    std::vector<vector<LabelIdType>>* m_inputs; // pointer to vectors to append with numbers
    std::vector<LabelIdType>* m_labels;         // pointer to vector to append with labels (may be numeric)
    // FUTURE: do we want a vector to collect string labels in the non string label case? (signifies an error)

public:
    // LUSequenceParser constructor
    LUSequenceParser();
    // setup all the state variables and state tables for state machine
    void Init();

    // Parser destructor
    ~LUSequenceParser();

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
    void ParseInit(LPCWSTR fileName, size_t dimFeatures, size_t dimLabelsIn, size_t dimLabelsOut, wstring beginSequenceIn, wstring endSequenceIn, wstring beginSequenceOut, wstring endSequenceOut)
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
            LUSequenceParser<NumType, LabelType>::~LUSequenceParser();

        errno_t err = _wfopen_s(&m_pFile, fileName, L"rb");
        if (err)
            RuntimeError("LUSequenceParser::ParseInit - error opening file");
        int rc = _fseeki64(m_pFile, 0, SEEK_END);
        if (rc)
            RuntimeError("LUSequenceParser::ParseInit - error seeking in file");
    }
};

// structure to describe how to find an input sentence in the 'labels' vector which is a concatenation of all
struct SentenceInfo
{
    size_t sLen;
    int sBegin;
    int sEnd;
};

// language-understanding sequence parser
template <typename NumType, typename LabelType = wstring>
class BatchLUSequenceParser : public LUSequenceParser<NumType, LabelType>
{
public:
    wstring mUnkStr;

public:
    std::wifstream mFile;
    std::wstring mFileName;
    vector<SentenceInfo> mSentenceIndex2SentenceInfo;

public:
    using LUSequenceParser<NumType, LabelType>::m_dimFeatures;
    using LUSequenceParser<NumType, LabelType>::m_dimLabelsIn;
    using LUSequenceParser<NumType, LabelType>::m_beginSequenceIn;
    using LUSequenceParser<NumType, LabelType>::m_endSequenceIn;
    using LUSequenceParser<NumType, LabelType>::m_dimLabelsOut;
    using LUSequenceParser<NumType, LabelType>::m_beginSequenceOut;
    using LUSequenceParser<NumType, LabelType>::m_endSequenceOut;
    using LUSequenceParser<NumType, LabelType>::m_traceLevel;
    using LUSequenceParser<NumType, LabelType>::m_beginTag;
    using LUSequenceParser<NumType, LabelType>::m_endTag;
    using LUSequenceParser<NumType, LabelType>::m_fileBuffer;
    using LUSequenceParser<NumType, LabelType>::m_inputs;
    using LUSequenceParser<NumType, LabelType>::m_labels;
    using LUSequenceParser<NumType, LabelType>::m_beginSequence;
    using LUSequenceParser<NumType, LabelType>::m_endSequence;
    BatchLUSequenceParser(){};
    ~BatchLUSequenceParser()
    {
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
        mFile.open(Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(fileName)), wifstream::in);
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
        mFile.open(Microsoft::MSR::CNTK::ToLegacyString(Microsoft::MSR::CNTK::ToUTF8(mFileName)), wifstream::in);
#else
        mFile.open(mFileName, wifstream::in);
#endif
        if (!mFile.good())
            RuntimeError("cannot open file %ls", mFileName.c_str());
    }

    void AddOneItem(std::vector<long>* labels, std::vector<vector<long>>* input, std::vector<SequencePosition>* seqPos, long& lineCount,
                    long& recordCount, long orgRecordCount, SequencePosition& sequencePositionLast)
    {
        SequencePosition sequencePos((long) input->size(), (long) labels->size(), 1);
        seqPos->push_back(sequencePos);
        sequencePositionLast = sequencePos;

        recordCount = (long) labels->size() - orgRecordCount;
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
    long Parse(size_t recordsRequested, std::vector<long>* labels, std::vector<vector<long>>* input, std::vector<SequencePosition>* seqPos, const map<wstring, long>& inputlabel2id, const map<wstring, long>& outputlabel2id, bool mAllowMultPassData = false);
};
}
}
};
