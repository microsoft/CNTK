// LUSequenceParser.h : Parses the UCI format using a custom state machine (for speed)
//
// <copyright file="LUSequenceParser.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
//

#include <string>
#include <vector>
#include <assert.h>
#include <fstream>
#include <map>
#include <stdint.h>
#include "Platform.h"
#include "DataReader.h"

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

#define MAXSTRING 2048
// UCI label location types
enum LabelMode
{
    LabelNone = 0,
    LabelFirst = 1,
    LabelLast = 2,
};

enum ParseMode
{
    ParseNormal = 0,
    ParseLineCount = 1
};

enum SequenceFlags
{
    seqFlagNull = 0,
    seqFlagLineBreak = 1, // line break on the parsed line
    seqFlagEmptyLine = 2, // empty line
    seqFlagStartLabel = 4,
    seqFlagStopLabel = 8
};

// SequencePosition, save the ending indexes into the array for a sequence
struct SequencePosition
{
    size_t inputPos; // max position in the number array for this sequence
    size_t labelPos; // max position in the label array for this sequence
    unsigned flags; // flags that apply to this sequence
    SequencePosition(size_t inPos, size_t labelPos, unsigned flags):
        inputPos(inPos), labelPos(labelPos), flags(flags)
    {}
};

// LUSequenceParser - the parser for the UCI format files
// for ultimate speed, this class implements a state machine to read these format files
template <typename NumType, typename LabelType=int>
class LUSequenceParser
{
protected:
    enum ParseState
    {
        WholeNumber = 0,
        Remainder = 1,
        Exponent = 2,
        Whitespace = 3,
        Sign = 4,
        ExponentSign = 5,
        Period = 6,
        TheLetterE = 7,
        EndOfLine = 8, 
        Label = 9, // any non-number things we run into
        ParseStateMax = 10, // number of parse states
        LineCountEOL = 10,
        LineCountOther = 11,
        AllStateMax = 12
    };

    // type of label processing
    ParseMode m_parseMode;

    // definition of label and feature dimensions
    size_t m_dimFeatures;

    size_t m_dimLabelsIn;
    std::string m_beginSequenceIn; // starting sequence string (i.e. <s>)
    std::string m_endSequenceIn; // ending sequence string (i.e. </s>)

    size_t m_dimLabelsOut;
    std::string m_beginSequenceOut; // starting sequence string (i.e. 'O')
    std::string m_endSequenceOut; // ending sequence string (i.e. 'O')

    // level of screen output
    int m_traceLevel;

    // current state of the state machine
    ParseState m_current_state;

    // state tables
    DWORD *m_stateTable;

    // numeric state machine variables
    double m_partialResult;
    double m_builtUpNumber;
    double m_divider;
    double m_wholeNumberMultiplier;
    double m_exponentMultiplier;

    // label state machine variables
    size_t m_spaceDelimitedStart;
    size_t m_spaceDelimitedMax; // start of the next whitespace sequence (one past the end of the last word)
    int m_numbersConvertedThisLine;
    int m_labelsConvertedThisLine;
    int m_elementsConvertedThisLine;

    // sequence state machine variables
    bool m_beginSequence;
    bool m_endSequence;
    std::string m_beginTag;
    std::string m_endTag;

    // global stats
    int m_totalNumbersConverted;
    int m_totalLabelsConverted;

    // file positions/buffer
    FILE * m_pFile;
    int64_t m_byteCounter;
    int64_t m_fileSize;

    BYTE * m_fileBuffer;
    size_t m_bufferStart;
    size_t m_bufferSize;

    // last label was a string (for last label processing)
    bool m_lastLabelIsString;

    // vectors to append to
    std::vector<vector<LabelType>>* m_inputs; // pointer to vectors to append with numbers
    std::vector<LabelType>* m_labels; // pointer to vector to append with labels (may be numeric)
    // FUTURE: do we want a vector to collect string labels in the non string label case? (signifies an error)

    // SetState for a particular value
    void SetState(int value, ParseState m_current_state, ParseState next_state);

    // SetStateRange - set states transitions for a range of values
    void SetStateRange(int value1, int value2, ParseState m_current_state, ParseState next_state);

    // SetupStateTables - setup state transition tables for each state
    // each state has a block of 256 states indexed by the incoming character
    void SetupStateTables();

    // reset all line state variables
    void PrepareStartLine();

    // reset all number accumulation variables
    void PrepareStartNumber();

    // reset all state variables to start reading at a new position
    void PrepareStartPosition(size_t position);

    // UpdateBuffer - load the next buffer full of data
    // returns - number of records read
    size_t UpdateBuffer();

public:

    // LUSequenceParser constructor
    LUSequenceParser();
    // setup all the state variables and state tables for state machine
    void Init();

    // Parser destructor
    ~LUSequenceParser();

public:
    // SetParseMode - Set the parsing mode
    // mode - set mode to either ParseLineCount, or ParseNormal
    void SetParseMode(ParseMode mode);

    // SetTraceLevel - Set the level of screen output
    // traceLevel - traceLevel, zero means no output, 1 epoch related output, > 1 all output
    void SetTraceLevel(int traceLevel);


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
    void ParseInit(LPCWSTR fileName, size_t dimFeatures, size_t dimLabelsIn, size_t dimLabelsOut, std::string beginSequenceIn="<s>", std::string endSequenceIn="</s>", std::string beginSequenceOut="O", std::string endSequenceOut="O", size_t bufferSize=1024*256, size_t startPosition=0)
    {
        assert(fileName != NULL);
        m_dimFeatures = dimFeatures;
        m_dimLabelsIn = dimLabelsIn;
        m_beginSequenceIn = beginSequenceIn;
        m_endSequenceIn = endSequenceIn;
        m_dimLabelsOut = dimLabelsOut;
        m_beginSequenceOut = beginSequenceOut;
        m_endSequenceOut = endSequenceOut;

        m_parseMode = ParseNormal;
        m_traceLevel = 0;
        m_bufferSize = bufferSize;
        m_bufferStart = startPosition;

        m_beginTag = m_beginSequenceIn;
        m_endTag = m_endSequenceIn;

        // if we have a file already open, cleanup
        if (m_pFile != NULL)
            LUSequenceParser<NumType, LabelType>::~LUSequenceParser();

        errno_t err = _wfopen_s( &m_pFile, fileName, L"rb" );
        if (err)
            RuntimeError("LUSequenceParser::ParseInit - error opening file"); 
        int rc = _fseeki64(m_pFile, 0, SEEK_END);
        if (rc)
            RuntimeError("LUSequenceParser::ParseInit - error seeking in file");

        m_fileBuffer = new BYTE[m_bufferSize];
    }
};

/// language model sequence parser
template <typename NumType, typename LabelType>
class LULUSequenceParser : public LUSequenceParser<NumType, LabelType>
{
protected:
    FILE * mFile; 
    std::wstring mFileName; 

public:
	using LUSequenceParser<NumType, LabelType>::m_dimFeatures;
	using LUSequenceParser<NumType, LabelType>::m_dimLabelsIn;
	using LUSequenceParser<NumType, LabelType>::m_beginSequenceIn;
	using LUSequenceParser<NumType, LabelType>::m_endSequenceIn;
	using LUSequenceParser<NumType, LabelType>::m_dimLabelsOut;
	using LUSequenceParser<NumType, LabelType>::m_beginSequenceOut;
	using LUSequenceParser<NumType, LabelType>::m_endSequenceOut;
	using LUSequenceParser<NumType, LabelType>::m_parseMode;
	using LUSequenceParser<NumType, LabelType>::m_traceLevel;
	using LUSequenceParser<NumType, LabelType>::m_bufferSize;
	using LUSequenceParser<NumType, LabelType>::m_bufferStart;
	using LUSequenceParser<NumType, LabelType>::m_beginTag;
	using LUSequenceParser<NumType, LabelType>::m_endTag;
	using LUSequenceParser<NumType, LabelType>::m_fileBuffer;
	using LUSequenceParser<NumType, LabelType>::m_fileSize;
	using LUSequenceParser<NumType, LabelType>::m_inputs;
	using LUSequenceParser<NumType, LabelType>::m_labels;
	using LUSequenceParser<NumType, LabelType>::m_beginSequence;
	using LUSequenceParser<NumType, LabelType>::m_endSequence;
	using LUSequenceParser<NumType, LabelType>::m_totalNumbersConverted;
    LULUSequenceParser() { 
        mFile = nullptr; 
    };
    ~LULUSequenceParser() { 
        if (mFile) fclose(mFile); 
    }

    void ParseInit(LPCWSTR fileName, size_t dimFeatures, size_t dimLabelsIn, size_t dimLabelsOut, std::string beginSequenceIn="<s>", std::string endSequenceIn="</s>", std::string beginSequenceOut="O", std::string endSequenceOut="O")
    {
        assert(fileName != NULL);
        mFileName = fileName;
        m_dimFeatures = dimFeatures;
        m_dimLabelsIn = dimLabelsIn;
        m_beginSequenceIn = beginSequenceIn;
        m_endSequenceIn = endSequenceIn;
        m_dimLabelsOut = dimLabelsOut;
        m_beginSequenceOut = beginSequenceOut;
        m_endSequenceOut = endSequenceOut;

        m_parseMode = ParseNormal;
        m_traceLevel = 0;
        m_bufferSize = 0;
        m_bufferStart = 0;

        m_beginTag = m_beginSequenceIn;
        m_endTag = m_endSequenceIn;

        m_fileSize = -1;
        m_fileBuffer = NULL;

        if (mFile) fclose(mFile);

        if (_wfopen_s(&mFile, fileName, L"rt") != 0)
            RuntimeError("cannot open file %s", fileName);
    }

    void ParseReset()
    {
        if (mFile) fseek(mFile, 0, SEEK_SET);
    }

    void AddOneItem(std::vector<LabelType> *labels, std::vector<vector<LabelType>> *input, std::vector<SequencePosition> *seqPos, long& lineCount,
        long & recordCount, long orgRecordCount, SequencePosition& sequencePositionLast)
    {
        SequencePosition sequencePos(input->size(), labels->size(),
            m_beginSequence ? seqFlagStartLabel : 0 | m_endSequence ? seqFlagStopLabel : 0 | seqFlagLineBreak);
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
    long Parse(size_t recordsRequested, std::vector<LabelType> *labels, std::vector<vector<LabelType>> *input, std::vector<SequencePosition> *seqPos)
    {
        assert(labels != NULL || m_dimLabelsIn == 0 && m_dimLabelsOut == 0 || m_parseMode == ParseLineCount);

        // transfer to member variables
        m_inputs = input;
        m_labels = labels;

        long TickStart = GetTickCount();
        long recordCount = 0;
        long orgRecordCount = (long)labels->size();
        long lineCount = 0;
        bool bAtEOS = false; /// whether the reader is at the end of sentence position
        SequencePosition sequencePositionLast(0, 0, seqFlagNull);
        /// get line
        char ch2[MAXSTRING];
        while (lineCount < recordsRequested && fgets(ch2, MAXSTRING, mFile) != nullptr)
        {

            string ch = ch2;
            std::vector<string> vstr;
            bool bBlankLine = (trim(ch).length() == 0);
            if (bBlankLine && !bAtEOS && input->size() > 0 && labels->size() > 0)
            {
                AddOneItem(labels, input, seqPos, lineCount, recordCount, orgRecordCount, sequencePositionLast);
                bAtEOS = true;
                continue;
            }

            vstr = sep_string(ch, " ");
            if (vstr.size() < 2)
                continue;

            bAtEOS = false;
            vector<LabelType> vtmp;
            for (size_t i = 0; i < vstr.size() - 1; i++)
            {
                vtmp.push_back(vstr[i]);
            }
            labels->push_back(vstr[vstr.size() - 1]);
            input->push_back(vtmp);
            if ((vstr[vstr.size() - 1] == m_endSequenceOut ||
                /// below is for backward support
                vstr[0] == m_endTag) && input->size() > 0 && labels->size() > 0)
            {
                AddOneItem(labels, input, seqPos, lineCount, recordCount, orgRecordCount, sequencePositionLast);
                bAtEOS = true;
            }

        } // while

        long TickStop = GetTickCount();

        long TickDelta = TickStop - TickStart;

        if (m_traceLevel > 2)
            fprintf(stderr, "\n%d ms, %d numbers parsed\n\n", TickDelta, m_totalNumbersConverted);
        return lineCount;
    }


};

typedef struct{
    size_t sLen;
    int sBegin;
    int sEnd;
} stSentenceInfo; 
/// language model sequence parser
template <typename NumType, typename LabelType>
class LUBatchLUSequenceParser: public LULUSequenceParser<NumType, LabelType>
{
public:
    vector<stSentenceInfo> mSentenceIndex2SentenceInfo;

public:
    LUBatchLUSequenceParser() { };
    ~LUBatchLUSequenceParser() { }

    void ParseInit(LPCWSTR fileName, size_t dimFeatures, size_t dimLabelsIn, size_t dimLabelsOut, std::string beginSequenceIn="<s>", std::string endSequenceIn="</s>", std::string beginSequenceOut="O", std::string endSequenceOut="O");

    // Parse - Parse the data
    // recordsRequested - number of records requested
    // labels - pointer to vector to return the labels 
    // numbers - pointer to vector to return the numbers 
    // seqPos - pointers to the other two arrays showing positions of each sequence
    // returns - number of records actually read, if the end of file is reached the return value will be < requested records
    long Parse(size_t recordsRequested, std::vector<LabelType> *labels, std::vector<vector<LabelType>> *inputs, std::vector<SequencePosition> *seqPos);

};
}}};
