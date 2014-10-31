// LUSequenceParser.cpp : Parses the UCI format using a custom state machine (for speed)
//
// <copyright file="LUSequenceParser.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//


#include "stdafx.h"
#include "LUSequenceParser.h"
#include <stdexcept>
#include <fileutil.h>

namespace Microsoft { namespace MSR { namespace CNTK {

// SetState for a particular value
template <typename NumType, typename LabelType>
void LUSequenceParser<NumType, LabelType>::SetState(int value, ParseState m_current_state, ParseState next_state)
{
    DWORD ul = (DWORD)next_state;
    int range_shift = ((int)m_current_state) << 8;
    m_stateTable[range_shift+value] = ul;
}

// SetStateRange - set states transitions for a range of values
template <typename NumType, typename LabelType>
void LUSequenceParser<NumType, LabelType>::SetStateRange(int value1, int value2, ParseState m_current_state, ParseState next_state)
{
    DWORD ul = (DWORD)next_state;
    int range_shift = ((int)m_current_state) << 8;
    for (int value = value1; value <= value2; value++)
    {
        m_stateTable[range_shift+value] = ul;
    }
}

// SetupStateTables - setup state transition tables for each state
// each state has a block of 256 states indexed by the incoming character
template <typename NumType, typename LabelType>
void LUSequenceParser<NumType, LabelType>::SetupStateTables()
{
    //=========================
    // STATE = WHITESPACE
    //=========================

    SetStateRange(0,255, Whitespace, Label);
    SetStateRange('0', '9', Whitespace, WholeNumber);
    SetState('-', Whitespace, Sign);
    SetState('+', Whitespace, Sign);
    // whitespace
    SetState(' ', Whitespace, Whitespace);
    SetState('\t', Whitespace, Whitespace);
    SetState('\r', Whitespace, Whitespace);
    SetState(':', Whitespace, Whitespace); // intepret ':' as white space because it's a divider
    SetState('\n', Whitespace, EndOfLine);

    //=========================
    // STATE = NEGATIVE_SIGN
    //=========================

    SetStateRange( 0, 255, Sign, Label);
    SetStateRange( '0', '9', Sign, WholeNumber);
    // whitespace
    SetState(' ', Sign, Whitespace);
    SetState('\t', Sign, Whitespace);
    SetState('\r', Sign, Whitespace);
    SetState('\n', Sign, EndOfLine);

    //=========================
    // STATE = NUMBER
    //=========================

    SetStateRange( 0, 255, WholeNumber, Label);
    SetStateRange( '0', '9', WholeNumber, WholeNumber);
    SetState('.', WholeNumber, Period);
    SetState('e', WholeNumber, TheLetterE);
    SetState('E', WholeNumber, TheLetterE);
    // whitespace
    SetState(' ', WholeNumber, Whitespace);
    SetState('\t', WholeNumber, Whitespace);
    SetState('\r', WholeNumber, Whitespace);
    SetState(':', WholeNumber, Whitespace); // Add for 1234:0.9 usage in Sequences
    SetState('\n', WholeNumber, EndOfLine);

    //=========================
    // STATE = PERIOD
    //=========================

    SetStateRange(0, 255, Period, Label);
    SetStateRange('0', '9', Period, Remainder);
    // whitespace
    SetState(' ', Period, Whitespace);
    SetState('\t', Period, Whitespace);
    SetState('\r', Period, Whitespace);
    SetState('\n', Period, EndOfLine);

    //=========================
    // STATE = REMAINDER
    //=========================

    SetStateRange(0, 255, Remainder, Label);
    SetStateRange('0', '9', Remainder, Remainder);
    SetState('e', Remainder, TheLetterE);
    SetState('E', Remainder, TheLetterE);
    // whitespace
    SetState(' ', Remainder, Whitespace);
    SetState('\t', Remainder, Whitespace);
    SetState('\r', Remainder, Whitespace);
    SetState(':', Remainder, Whitespace); // Add for 1234:0.9 usage in Sequences
    SetState('\n', Remainder, EndOfLine);

    //=========================
    // STATE = THE_LETTER_E
    //=========================

    SetStateRange(0, 255, TheLetterE, Label);
    SetStateRange('0', '9', TheLetterE, Exponent);
    SetState('-', TheLetterE, ExponentSign);
    SetState('+', TheLetterE, ExponentSign);
    // whitespace
    SetState(' ', TheLetterE, Whitespace);
    SetState('\t', TheLetterE, Whitespace);
    SetState('\r', TheLetterE, Whitespace);
    SetState('\n', TheLetterE, EndOfLine);

    //=========================
    // STATE = EXPONENT_NEGATIVE_SIGN
    //=========================

    SetStateRange(0, 255, ExponentSign, Label);
    SetStateRange('0', '9', ExponentSign, Exponent);
    // whitespace
    SetState(' ', ExponentSign, Whitespace);
    SetState('\t', ExponentSign, Whitespace);
    SetState('\r', ExponentSign, Whitespace);
    SetState('\n', ExponentSign, EndOfLine);

    //=========================
    // STATE = EXPONENT
    //=========================

    SetStateRange(0, 255, Exponent, Label);
    SetStateRange('0', '9', Exponent, Exponent);
    // whitespace
    SetState(' ', Exponent, Whitespace);
    SetState('\t', Exponent, Whitespace);
    SetState('\r', Exponent, Whitespace);
    SetState(':', Exponent, Whitespace);
    SetState('\n', Exponent, EndOfLine);

    //=========================
    // STATE = END_OF_LINE
    //=========================
    SetStateRange(0, 255, EndOfLine, Label);
    SetStateRange( '0', '9', EndOfLine, WholeNumber);
    SetState( '-', EndOfLine, Sign);
    SetState( '\n', EndOfLine, EndOfLine);
    // whitespace
    SetState(' ', EndOfLine, Whitespace);
    SetState('\t', EndOfLine, Whitespace);
    SetState('\r', EndOfLine, Whitespace);


    //=========================
    // STATE = LABEL
    //=========================
    SetStateRange(0, 255, Label, Label);
    SetState('\n', Label, EndOfLine);
    // whitespace
    SetState(' ', Label, Whitespace);
    SetState('\t', Label, Whitespace);
    SetState('\r', Label, Whitespace);
    SetState(':', Label, Whitespace);

    //=========================
    // STATE = LINE_COUNT_EOL
    //=========================
    SetStateRange(0, 255, LineCountEOL, LineCountOther);
    SetState( '\n', LineCountEOL, LineCountEOL);

    //=========================
    // STATE = LINE_COUNT_OTHER
    //=========================
    SetStateRange(0, 255, LineCountOther, LineCountOther);
    SetState('\n', LineCountOther, LineCountEOL);
}


// reset all line state variables
template <typename NumType, typename LabelType>
void LUSequenceParser<NumType, LabelType>::PrepareStartLine()
{
    m_numbersConvertedThisLine = 0;
    m_labelsConvertedThisLine = 0;
    m_elementsConvertedThisLine = 0;
    m_spaceDelimitedStart = m_byteCounter;
    m_spaceDelimitedMax = m_byteCounter;
    m_lastLabelIsString = false;
    m_beginSequence = m_endSequence = false;
}

// reset all number accumulation variables
template <typename NumType, typename LabelType>
void LUSequenceParser<NumType, LabelType>::PrepareStartNumber()
{
    m_partialResult = 0;
    m_builtUpNumber = 0;
    m_divider = 0;
    m_wholeNumberMultiplier = 1;
    m_exponentMultiplier = 1;
}

// reset all state variables to start reading at a new position
template <typename NumType, typename LabelType>
void LUSequenceParser<NumType, LabelType>::PrepareStartPosition(size_t position)
{
    m_current_state = Whitespace;
    m_byteCounter = position;  // must come before PrepareStartLine...
    m_bufferStart = position;

    // prepare state machine for new number and new line
    PrepareStartNumber();
    PrepareStartLine();
    m_totalNumbersConverted = 0;
    m_totalLabelsConverted = 0;
}

// LUSequenceParser constructor
template <typename NumType, typename LabelType>
LUSequenceParser<NumType, LabelType>::LUSequenceParser()
{
    Init();
}

// setup all the state variables and state tables for state machine
template <typename NumType, typename LabelType>
void LUSequenceParser<NumType, LabelType>::Init()
{
    PrepareStartPosition(0);
    m_fileBuffer = NULL;
    m_pFile = NULL;
    m_stateTable = new DWORD[AllStateMax * 256];
    SetupStateTables();
}

// Parser destructor
template <typename NumType, typename LabelType>
LUSequenceParser<NumType, LabelType>::~LUSequenceParser()
{
    delete m_stateTable;
    delete m_fileBuffer;
    if (m_pFile)
        fclose(m_pFile);
}


// UpdateBuffer - load the next buffer full of data
// returns - number of records read
template <typename NumType, typename LabelType>
size_t LUSequenceParser<NumType, LabelType>::UpdateBuffer()
{
    // state machine might want to look back this far, so copy to beginning
    size_t saveBytes = m_byteCounter-m_spaceDelimitedStart;
    assert(saveBytes < m_bufferSize);
    if (saveBytes)
    {
        memcpy_s(m_fileBuffer, m_bufferSize, &m_fileBuffer[m_byteCounter-m_bufferStart-saveBytes], saveBytes);
        m_bufferStart = m_byteCounter-saveBytes;
    }

    // read the next block
    size_t bytesToRead = min(m_bufferSize, m_fileSize-m_bufferStart)-saveBytes;
    size_t bytesRead = fread(m_fileBuffer+saveBytes, 1, bytesToRead, m_pFile);
    if (bytesRead == 0 && ferror(m_pFile))
        RuntimeError("LUSequenceParser::UpdateBuffer - error reading file");
    return bytesRead;
}

template <typename NumType, typename LabelType>
void LUSequenceParser<NumType, LabelType>::SetParseMode(ParseMode mode)
{
    // if already in this mode, nothing to do
    if (m_parseMode == mode)
        return;

    // switching modes
    if (mode == ParseLineCount)
        m_current_state = LineCountOther;
    else
    {
        m_current_state = Whitespace;
        PrepareStartLine();
        PrepareStartNumber();
    }
    m_parseMode = mode;
}

// SetTraceLevel - Set the level of screen output
// traceLevel - traceLevel, zero means no output, 1 epoch related output, > 1 all output
template <typename NumType, typename LabelType>
void LUSequenceParser<NumType, LabelType>::SetTraceLevel(int traceLevel)
{
    m_traceLevel = traceLevel;
}

// NOTE: Current code is identical to float, don't know how to specialize with template parameter that only covers one parameter

#ifdef STANDALONE
int wmain(int argc, wchar_t* argv[])
{
    LUSequenceParser<double, int> parser;
    std::vector<double> values;
    values.reserve(784000*6);
    std::vector<int> labels;
    labels.reserve(60000);
    parser.ParseInit(L"c:\\speech\\mnist\\mnist_train.txt", LabelFirst);
    //parser.ParseInit("c:\\speech\\parseTest.txt", LabelNone);
    int records = 0;
    do
    {
        int recordsRead = parser.Parse(10000, &values, &labels);
        if (recordsRead < 10000)
            parser.SetFilePosition(0);  // go around again
        records += recordsRead;
        values.clear();
        labels.clear();
    }
    while (records < 150000);
    return records;
}
#endif

// instantiate UCI parsers for supported types
template class LUSequenceParser<float, int>;
template class LUSequenceParser<float, float>;
template class LUSequenceParser<float, std::string>;
template class LUSequenceParser<double, int>;
template class LUSequenceParser<double, double>;
template class LUSequenceParser<double, std::string>;

template <typename NumType, typename LabelType>
void LUBatchLUSequenceParser<NumType, LabelType>::ParseInit(LPCWSTR fileName, size_t dimFeatures, size_t dimLabelsIn, size_t dimLabelsOut, std::string beginSequenceIn="<s>", std::string endSequenceIn="</s>", std::string beginSequenceOut="O", std::string endSequenceOut="O")
{
    LULUSequenceParser<NumType, LabelType>::ParseInit(fileName, dimFeatures, dimLabelsIn, dimLabelsOut, beginSequenceIn, endSequenceIn, beginSequenceOut, endSequenceOut);
}

template <typename NumType, typename LabelType>
long LUBatchLUSequenceParser<NumType, LabelType>::Parse(size_t recordsRequested, std::vector<LabelType> *labels, std::vector<vector<LabelType>> *inputs, std::vector<SequencePosition> *seqPos)
{
    long linecnt; 
    linecnt = LULUSequenceParser<NumType, LabelType>::Parse(recordsRequested, labels, inputs, seqPos);

    size_t prvat = 0;    
    size_t i = 0;
    for (auto ptr = seqPos->begin(); ptr != seqPos->end(); ptr++, i++)
    {
        size_t iln = ptr->labelPos - prvat;
        stSentenceInfo stinfo;
        stinfo.sLen = iln;
        stinfo.sBegin = (int)prvat;
        stinfo.sEnd = (int)ptr->labelPos;
        mSentenceIndex2SentenceInfo.push_back(stinfo); 

        prvat = ptr->labelPos;
    }

    assert(mSentenceIndex2SentenceInfo.size() == linecnt);
    return linecnt; 
}

template class LUBatchLUSequenceParser<float, std::string>;
template class LUBatchLUSequenceParser<double, std::string>;
}}}
