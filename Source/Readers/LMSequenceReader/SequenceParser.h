// SequenceParser.h : Parses the UCI format using a custom state machine (for speed)
//
//
// <copyright file="SequenceParser.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#pragma once

#include <string>
#include <vector>
#include <assert.h>
#include <fstream>
#include <map>
#include <stdint.h>
#include "Basics.h"
#include "fileutil.h"

using namespace std;

#define MAXSTRING 500000
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
    size_t numberPos; // max position in the number array for this sequence
    size_t labelPos;  // max position in the label array for this sequence
    unsigned flags;   // flags that apply to this sequence
    SequencePosition(size_t numPos, size_t labelPos, unsigned flags)
        : numberPos(numPos), labelPos(labelPos), flags(flags)
    {
    }
};

// SequenceParser - the parser for the UCI format files
// for ultimate speed, this class implements a state machine to read these format files
template <typename NumType, typename LabelType = int>
class SequenceParser
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
        Label = 9,          // any non-number things we run into
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
    std::string m_endSequenceIn;   // ending sequence string (i.e. </s>)

    size_t m_dimLabelsOut;
    std::string m_beginSequenceOut; // starting sequence string (i.e. 'O')
    std::string m_endSequenceOut;   // ending sequence string (i.e. 'O')

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
    FILE *m_pFile;
    int64_t m_byteCounter;
    int64_t m_fileSize;

    BYTE *m_fileBuffer;
    size_t m_bufferStart;
    size_t m_bufferSize;

    // last label was a string (for last label processing)
    bool m_lastLabelIsString;

    // vectors to append to
    std::vector<NumType> *m_numbers;  // pointer to vectors to append with numbers
    std::vector<LabelType> *m_labels; // pointer to vector to append with labels (may be numeric)
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
    // SequenceParser constructor
    SequenceParser();
    // setup all the state variables and state tables for state machine
    void Init();

    // Parser destructor
    ~SequenceParser();

private:
    // DoneWithLabel - Called when a string label is found
    void DoneWithLabel();

    // Called when a number is complete
    void DoneWithValue();

    // store label is specialized by LabelType
    void StoreLabel(NumType value);

    // StoreLastLabel - store the last label (for numeric types), tranfers to label vector
    // string label types handled in specialization
    void StoreLastLabel();

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
    void ParseInit(LPCWSTR fileName, size_t dimFeatures, size_t dimLabelsIn, size_t dimLabelsOut, std::string beginSequenceIn = "<s>", std::string endSequenceIn = "</s>", std::string beginSequenceOut = "O", std::string endSequenceOut = "O", size_t bufferSize = 1024 * 256, size_t startPosition = 0)
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
            SequenceParser<NumType, LabelType>::~SequenceParser();

        errno_t err = _wfopen_s(&m_pFile, fileName, L"rb");
        if (err)
            Microsoft::MSR::CNTK::RuntimeError("SequenceParser::ParseInit - error opening file");
        int rc = _fseeki64(m_pFile, 0, SEEK_END);
        if (rc)
            Microsoft::MSR::CNTK::RuntimeError("SequenceParser::ParseInit - error seeking in file");

        m_fileSize = GetFilePosition();
        m_fileBuffer = new BYTE[m_bufferSize];
        SetFilePosition(startPosition);
    }

    // Parse - Parse the data
    // recordsRequested - number of records requested
    // labels - pointer to vector to return the labels
    // numbers - pointer to vector to return the numbers
    // seqPos - pointers to the other two arrays showing positions of each sequence
    // returns - number of records actually read, if the end of file is reached the return value will be < requested records
    long Parse(size_t recordsRequested, std::vector<LabelType> *labels, std::vector<NumType> *numbers, std::vector<SequencePosition> *seqPos)
    {
        assert(numbers != NULL || m_dimFeatures == 0 || m_parseMode == ParseLineCount);
        assert(labels != NULL || m_dimLabelsIn == 0 && m_dimLabelsOut == 0 || m_parseMode == ParseLineCount);

        // transfer to member variables
        m_numbers = numbers;
        m_labels = labels;

        long TickStart = GetTickCount();
        long recordCount = 0;
        long lineCount = 0;
        size_t bufferIndex = m_byteCounter - m_bufferStart;
        SequencePosition sequencePositionLast(0, 0, seqFlagNull);
        while (m_byteCounter < m_fileSize && recordCount < recordsRequested)
        {
            // check to see if we need to update the buffer
            if (bufferIndex >= m_bufferSize)
            {
                UpdateBuffer();
                bufferIndex = m_byteCounter - m_bufferStart;
            }

            char ch = m_fileBuffer[bufferIndex];

            ParseState nextState = (ParseState) m_stateTable[(m_current_state << 8) + ch];

            if (nextState <= Exponent)
            {
                m_builtUpNumber = m_builtUpNumber * 10 + (ch - '0');
                // if we are in the decimal portion of a number increase the divider
                if (nextState == Remainder)
                    m_divider *= 10;
            }

            // only do a test on a state transition
            if (m_current_state != nextState)
            {
                // System.Diagnostics.Debug.WriteLine("Current state = " + m_current_state + ", next state = " + nextState);

                // if the nextState is a label, we don't want to do any number processing, it's a number prefixed string
                if (nextState != Label)
                {
                    // do the numeric processing
                    switch (m_current_state)
                    {
                    case TheLetterE:
                        if (m_divider != 0) // decimal number
                            m_partialResult += m_builtUpNumber / m_divider;
                        else // integer
                            m_partialResult = m_builtUpNumber;
                        m_builtUpNumber = 0;
                        break;
                    case WholeNumber:
                        // could be followed by a remainder, or an exponent
                        if (nextState != TheLetterE)
                            if (nextState != Period)
                                DoneWithValue();
                        if (nextState == Period)
                        {
                            m_partialResult = m_builtUpNumber;
                            m_divider = 1;
                            m_builtUpNumber = 0;
                        }
                        break;
                    case Remainder:
                        // can only be followed by a exponent
                        if (nextState != TheLetterE)
                            DoneWithValue();
                        break;
                    case Exponent:
                        DoneWithValue();
                        break;
                    }
                }

                // label handling
                switch (m_current_state)
                {
                case Label:
                    DoneWithLabel();
                    break;
                case EndOfLine:
                    if (seqPos)
                    {
                        SequencePosition sequencePos(numbers->size(), labels->size(),
                                                     m_beginSequence ? seqFlagStartLabel : 0 | m_endSequence ? seqFlagStopLabel : 0 | seqFlagLineBreak);
                        // add a sequence element to the list
                        seqPos->push_back(sequencePos);
                        sequencePositionLast = sequencePos;
                    }

                    // end of sequence determines record separation
                    if (m_endSequence)
                        recordCount = (long) labels->size();

                    PrepareStartLine();
                    break;
                case Whitespace:
                    // this is the start of the next space delimited entity
                    if (nextState != EndOfLine)
                        m_spaceDelimitedStart = m_byteCounter;
                    break;
                }

                // label handling for next state
                switch (nextState)
                {
                // do sign processing on nextState, since we still have the character handy
                case Sign:
                    if (ch == '-')
                        m_wholeNumberMultiplier = -1;
                    break;
                case ExponentSign:
                    if (ch == '-')
                        m_exponentMultiplier = -1;
                    break;
                // going into whitespace or endOfLine, so end of space delimited entity
                case Whitespace:
                    m_spaceDelimitedMax = m_byteCounter;
                    // hit whitespace and nobody processed anything, so add as label
                    //if (m_elementsConvertedThisLine == elementsProcessed)
                    //    DoneWithLabel();
                    break;
                case EndOfLine:
                    if (m_current_state != Whitespace)
                    {
                        m_spaceDelimitedMax = m_byteCounter;
                        // hit whitespace and nobody processed anything, so add as label
                        //if (m_elementsConvertedThisLine == elementsProcessed)
                        //    DoneWithLabel();
                    }
                // process the label at the end of a line
                //if (m_labelMode == LabelLast && m_labels != NULL)
                //{
                //    StoreLastLabel();
                //}
                // intentional fall-through
                case LineCountEOL:
                    lineCount++; // done with another record
                    if (m_traceLevel > 1)
                    {
                        // print progress dots
                        if (recordCount % 100 == 0)
                        {
                            if (recordCount % 1000 == 0)
                            {
                                if (recordCount % 10000 == 0)
                                {
                                    fprintf(stderr, "#");
                                }
                                else
                                {
                                    fprintf(stderr, "+");
                                }
                            }
                            else
                            {
                                fprintf(stderr, ".");
                            }
                        }
                    }
                    break;
                case LineCountOther:
                    m_spaceDelimitedStart = m_byteCounter;
                    break;
                }
            }

            m_current_state = nextState;

            // move to next character
            m_byteCounter++;
            bufferIndex++;
        } // while

        // at the end of the file we may need to add an additional sequencePosition push
        // this could probably be fixed by taking another pass through the loop above, but this is easier
        if (seqPos)
        {
            SequencePosition sequencePos(numbers->size(), labels->size(),
                                         m_beginSequence ? seqFlagStartLabel : 0 | m_endSequence ? seqFlagStopLabel : 0 | seqFlagLineBreak);
            // add the final sequence element if needed
            if (!(sequencePos.labelPos == sequencePositionLast.labelPos && sequencePos.numberPos == sequencePositionLast.numberPos))
            {
                seqPos->push_back(sequencePos);
            }
        }

        long TickStop = GetTickCount();

        long TickDelta = TickStop - TickStart;

        if (m_traceLevel > 2)
            fprintf(stderr, "\n%ld ms, %d numbers parsed\n\n", TickDelta, m_totalNumbersConverted);
        return lineCount;
    }

    int64_t GetFilePosition();
    void SetFilePosition(int64_t position);

    // HasMoreData - test if the current dataset have more data
    // returns - true if it does, false if not
    bool HasMoreData();
};

// StoreLabel - string version gets last space delimited string and stores in labels vector
template <>
void SequenceParser<float, std::string>::StoreLabel(float finalResult);

// DoneWithLabel - string version stores string label
template <>
void SequenceParser<float, std::string>::DoneWithLabel();

// StoreLastLabel - string version
template <>
void SequenceParser<float, std::string>::StoreLastLabel();

// NOTE: Current code is identical to float, don't know how to specialize with template parameter that only covers one parameter

// StoreLabel - string version gets last space delimited string and stores in labels vector
template <>
void SequenceParser<double, std::string>::StoreLabel(double finalResult);

// DoneWithLabel - string version stores string label
template <>
void SequenceParser<double, std::string>::DoneWithLabel();

// StoreLastLabel - string version
template <>
void SequenceParser<double, std::string>::StoreLastLabel();

/// language model sequence parser
template <typename NumType, typename LabelType>
class LMSequenceParser : public SequenceParser<NumType, LabelType>
{
protected:
    FILE *mFile;
    std::wstring mFileName;

public:
    using SequenceParser<NumType, LabelType>::m_dimFeatures;
    using SequenceParser<NumType, LabelType>::m_dimLabelsIn;
    using SequenceParser<NumType, LabelType>::m_beginSequenceIn;
    using SequenceParser<NumType, LabelType>::m_endSequenceIn;
    using SequenceParser<NumType, LabelType>::m_beginSequenceOut;
    using SequenceParser<NumType, LabelType>::m_endSequenceOut;
    using SequenceParser<NumType, LabelType>::m_parseMode;
    using SequenceParser<NumType, LabelType>::m_traceLevel;
    using SequenceParser<NumType, LabelType>::m_bufferSize;
    using SequenceParser<NumType, LabelType>::m_beginTag;
    using SequenceParser<NumType, LabelType>::m_endTag;
    using SequenceParser<NumType, LabelType>::m_fileSize;
    using SequenceParser<NumType, LabelType>::m_fileBuffer;
    using SequenceParser<NumType, LabelType>::m_numbers;
    using SequenceParser<NumType, LabelType>::m_labels;
    using SequenceParser<NumType, LabelType>::m_beginSequence;
    using SequenceParser<NumType, LabelType>::m_endSequence;
    using SequenceParser<NumType, LabelType>::m_totalNumbersConverted;
    using SequenceParser<NumType, LabelType>::m_dimLabelsOut;
    using SequenceParser<NumType, LabelType>::m_bufferStart;
    LMSequenceParser()
    {
        mFile = nullptr;
    };
    ~LMSequenceParser()
    {
        if (mFile)
            fclose(mFile);
    }

    void ParseInit(LPCWSTR fileName, size_t dimFeatures, size_t dimLabelsIn, size_t dimLabelsOut, std::string beginSequenceIn = "<s>", std::string endSequenceIn = "</s>", std::string beginSequenceOut = "O", std::string endSequenceOut = "O")
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

        if (mFile)
            fclose(mFile);

        if (_wfopen_s(&mFile, fileName, L"rt") != 0)
            Microsoft::MSR::CNTK::Warning("cannot open file %s", fileName);
    }

    void ParseReset()
    {
        if (mFile)
            fseek(mFile, 0, SEEK_SET);
    }

    // Parse - Parse the data
    // recordsRequested - number of records requested
    // labels - pointer to vector to return the labels
    // numbers - pointer to vector to return the numbers
    // seqPos - pointers to the other two arrays showing positions of each sequence
    // returns - number of records actually read, if the end of file is reached the return value will be < requested records
    long Parse(size_t recordsRequested, std::vector<LabelType> *labels, std::vector<NumType> *numbers, std::vector<SequencePosition> *seqPos)
    {
        assert(numbers != NULL || m_dimFeatures == 0 || m_parseMode == ParseLineCount);
        assert(labels != NULL || m_dimLabelsIn == 0 && m_dimLabelsOut == 0 || m_parseMode == ParseLineCount);

        // transfer to member variables
        m_numbers = numbers;
        m_labels = labels;

        long TickStart = GetTickCount();
        long recordCount = 0;
        long orgRecordCount = (long) labels->size();
        long lineCount = 0;
        SequencePosition sequencePositionLast(0, 0, seqFlagNull);
        /// get line
        char ch2[MAXSTRING];
        if (mFile == nullptr)
            Microsoft::MSR::CNTK::RuntimeError("File %ls can not be loaded\n", mFileName.c_str());

        while (recordCount < recordsRequested && fgets(ch2, MAXSTRING, mFile) != nullptr)
        {

            string ch = ch2;
            std::vector<string> vstr;
            vstr = sep_string(ch, " ");
            if (vstr.size() < 3)
                continue;

            for (size_t i = 0; i < vstr.size(); i++)
            {
                labels->push_back(vstr[i]);
            }
            SequencePosition sequencePos(numbers->size(), labels->size(),
                                         m_beginSequence ? seqFlagStartLabel : 0 | m_endSequence ? seqFlagStopLabel : 0 | seqFlagLineBreak);
            // add a sequence element to the list
            seqPos->push_back(sequencePos);
            sequencePositionLast = sequencePos;

            recordCount = (long) labels->size() - orgRecordCount;

            lineCount++;
        } // while

        long TickStop = GetTickCount();

        long TickDelta = TickStop - TickStart;

        if (m_traceLevel > 2)
            fprintf(stderr, "\n%ld ms, %d numbers parsed\n\n", TickDelta, m_totalNumbersConverted);
        return lineCount;
    }
};

typedef struct
{
    size_t sLen;
    size_t sBegin;
    size_t sEnd;
} stSentenceInfo;
/// language model sequence parser
template <typename NumType, typename LabelType>
class LMBatchSequenceParser : public LMSequenceParser<NumType, LabelType>
{
public:
    vector<stSentenceInfo> mSentenceIndex2SentenceInfo;

public:
    LMBatchSequenceParser(){};
    ~LMBatchSequenceParser()
    {
    }

    void ParseInit(LPCWSTR fileName, size_t dimFeatures, size_t dimLabelsIn, size_t dimLabelsOut, std::string beginSequenceIn = "<s>", std::string endSequenceIn = "</s>", std::string beginSequenceOut = "O", std::string endSequenceOut = "O");

    // Parse - Parse the data
    // recordsRequested - number of records requested
    // labels - pointer to vector to return the labels
    // numbers - pointer to vector to return the numbers
    // seqPos - pointers to the other two arrays showing positions of each sequence
    // returns - number of records actually read, if the end of file is reached the return value will be < requested records
    long Parse(size_t recordsRequested, std::vector<LabelType> *labels, std::vector<NumType> *numbers, std::vector<SequencePosition> *seqPos);
};
