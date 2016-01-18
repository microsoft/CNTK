// SequenceParser.cpp : Parses the UCI format using a custom state machine (for speed)
//
//
// <copyright file="SequenceParser.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "stdafx.h"
#include <stdexcept>
#include <stdint.h>
#include "Basics.h"
#include "SequenceParser.h"
#include "fileutil.h"

using namespace Microsoft::MSR::CNTK;

// SetState for a particular value
template <typename NumType, typename LabelType>
void SequenceParser<NumType, LabelType>::SetState(int value, ParseState m_current_state, ParseState next_state)
{
    DWORD ul = (DWORD) next_state;
    int range_shift = ((int) m_current_state) << 8;
    m_stateTable[range_shift + value] = ul;
}

// SetStateRange - set states transitions for a range of values
template <typename NumType, typename LabelType>
void SequenceParser<NumType, LabelType>::SetStateRange(int value1, int value2, ParseState m_current_state, ParseState next_state)
{
    DWORD ul = (DWORD) next_state;
    int range_shift = ((int) m_current_state) << 8;
    for (int value = value1; value <= value2; value++)
    {
        m_stateTable[range_shift + value] = ul;
    }
}

// SetupStateTables - setup state transition tables for each state
// each state has a block of 256 states indexed by the incoming character
template <typename NumType, typename LabelType>
void SequenceParser<NumType, LabelType>::SetupStateTables()
{
    //=========================
    // STATE = WHITESPACE
    //=========================

    SetStateRange(0, 255, Whitespace, Label);
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

    SetStateRange(0, 255, Sign, Label);
    SetStateRange('0', '9', Sign, WholeNumber);
    // whitespace
    SetState(' ', Sign, Whitespace);
    SetState('\t', Sign, Whitespace);
    SetState('\r', Sign, Whitespace);
    SetState('\n', Sign, EndOfLine);

    //=========================
    // STATE = NUMBER
    //=========================

    SetStateRange(0, 255, WholeNumber, Label);
    SetStateRange('0', '9', WholeNumber, WholeNumber);
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
    SetStateRange('0', '9', EndOfLine, WholeNumber);
    SetState('-', EndOfLine, Sign);
    SetState('\n', EndOfLine, EndOfLine);
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
    SetState('\n', LineCountEOL, LineCountEOL);

    //=========================
    // STATE = LINE_COUNT_OTHER
    //=========================
    SetStateRange(0, 255, LineCountOther, LineCountOther);
    SetState('\n', LineCountOther, LineCountEOL);
}

// reset all line state variables
template <typename NumType, typename LabelType>
void SequenceParser<NumType, LabelType>::PrepareStartLine()
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
void SequenceParser<NumType, LabelType>::PrepareStartNumber()
{
    m_partialResult = 0;
    m_builtUpNumber = 0;
    m_divider = 0;
    m_wholeNumberMultiplier = 1;
    m_exponentMultiplier = 1;
}

// reset all state variables to start reading at a new position
template <typename NumType, typename LabelType>
void SequenceParser<NumType, LabelType>::PrepareStartPosition(size_t position)
{
    m_current_state = Whitespace;
    m_byteCounter = position; // must come before PrepareStartLine...
    m_bufferStart = position;

    // prepare state machine for new number and new line
    PrepareStartNumber();
    PrepareStartLine();
    m_totalNumbersConverted = 0;
    m_totalLabelsConverted = 0;
}

// SequenceParser constructor
template <typename NumType, typename LabelType>
SequenceParser<NumType, LabelType>::SequenceParser()
{
    Init();
}

// setup all the state variables and state tables for state machine
template <typename NumType, typename LabelType>
void SequenceParser<NumType, LabelType>::Init()
{
    PrepareStartPosition(0);
    m_fileBuffer = NULL;
    m_pFile = NULL;
    m_stateTable = new DWORD[AllStateMax * 256];
    SetupStateTables();
}

// Parser destructor
template <typename NumType, typename LabelType>
SequenceParser<NumType, LabelType>::~SequenceParser()
{
    delete m_stateTable;
    delete m_fileBuffer;
    if (m_pFile)
        fclose(m_pFile);
}

// DoneWithLabel - Called when a string label is found
template <typename NumType, typename LabelType>
void SequenceParser<NumType, LabelType>::DoneWithLabel()
{
    // if we haven't set the max yet, use the current byte Counter
    if (m_spaceDelimitedMax <= m_spaceDelimitedStart)
        m_spaceDelimitedMax = m_byteCounter;
    {
        std::string label((LPCSTR) &m_fileBuffer[m_spaceDelimitedStart - m_bufferStart], m_spaceDelimitedMax - m_spaceDelimitedStart);
        m_labelsConvertedThisLine++;
        m_elementsConvertedThisLine++;
        m_lastLabelIsString = true;
    }
    PrepareStartNumber();
}

// Called when a number is complete
template <typename NumType, typename LabelType>
void SequenceParser<NumType, LabelType>::DoneWithValue()
{
    // if we are storing it
    if (m_numbers != NULL)
    {
        NumType FinalResult = 0;
        if (m_current_state == Exponent)
        {
            FinalResult = (NumType)(m_partialResult * pow(10.0, m_exponentMultiplier * m_builtUpNumber));
        }
        else if (m_divider != 0)
        {
            FinalResult = (NumType)(m_partialResult + (m_builtUpNumber / m_divider));
        }
        else
        {
            FinalResult = (NumType) m_builtUpNumber;
        }

        FinalResult = (NumType)(FinalResult * m_wholeNumberMultiplier);

        // TODO: In sequence reader we probably don't need to store numbers in labels (we'll see)
        // if it's a label, store in label location instead of number location
        //int index = m_elementsConvertedThisLine;
        //if (m_startLabels <= index && index < m_startLabels + m_dimLabels)
        //{
        //    StoreLabel(FinalResult);
        //}
        //if (m_startFeatures <= index && index < m_startFeatures + m_dimFeatures)
        {
            m_numbers->push_back(FinalResult);
            m_totalNumbersConverted++;
            m_numbersConvertedThisLine++;
            m_elementsConvertedThisLine++;
            m_lastLabelIsString = false;
        }
    }

    PrepareStartNumber();
}

// store label is specialized by LabelType
template <typename NumType, typename LabelType>
void SequenceParser<NumType, LabelType>::StoreLabel(NumType value)
{
    m_labels->push_back((LabelType) value);
    m_totalNumbersConverted++;
    m_numbersConvertedThisLine++;
    m_elementsConvertedThisLine++;
    m_lastLabelIsString = false;
}

// StoreLastLabel - store the last label (for numeric types), tranfers to label vector
// string label types handled in specialization
template <typename NumType, typename LabelType>
void SequenceParser<NumType, LabelType>::StoreLastLabel()
{
    assert(!m_lastLabelIsString); // file format error, last label was a string...
    NumType value = m_numbers->back();
    m_numbers->pop_back();
    m_labels->push_back((LabelType) value);
}

// GetFilePosition - Get the current file position in the text file
// returns current position in the file
template <typename NumType, typename LabelType>
int64_t SequenceParser<NumType, LabelType>::GetFilePosition()
{
    int64_t position = _ftelli64(m_pFile);
    if (position == -1L)
        RuntimeError("SequenceParser::GetFilePosition - error retrieving file position in file");
    return position;
}

// SetFilePosition - Set the current file position from the beginning of the file, and read in the first block of data
// state machine mode will be initialized similar to the beginning of the file
// it is recommneded that only return values from GetFilePosition() known to be the start of a line
// and zero be passed to this function
template <typename NumType, typename LabelType>
void SequenceParser<NumType, LabelType>::SetFilePosition(int64_t position)
{
    int rc = _fseeki64(m_pFile, position, SEEK_SET);
    if (rc)
        RuntimeError("SequenceParser::SetFilePosition - error seeking in file");

    // setup state machine to start at this position
    PrepareStartPosition(position);

    // read in the first buffer of data from this position,  first buffer is expected to be read after a reposition
    UpdateBuffer();

    // FUTURE: in debug we could validate the value is either 0, or the previous character is a '\n'
}

// HasMoreData - test if the current dataset have more data, or just whitespace
// returns - true if it has more data, false if not
template <typename NumType, typename LabelType>
bool SequenceParser<NumType, LabelType>::HasMoreData()
{
    long long byteCounter = m_byteCounter;
    size_t bufferIndex = m_byteCounter - m_bufferStart;

    // test without moving parser state
    for (; byteCounter < m_fileSize; byteCounter++, bufferIndex++)
    {
        // if we reach the end of the buffer, just assume we have more data
        // won't be right 100% of the time, but close enough
        if (bufferIndex >= m_bufferSize)
            return true;

        char ch = m_fileBuffer[bufferIndex];
        ParseState nextState = (ParseState) m_stateTable[(Whitespace << 8) + ch];
        if (!(nextState == Whitespace || nextState == EndOfLine))
            return true;
    }
    return false;
}

// UpdateBuffer - load the next buffer full of data
// returns - number of records read
template <typename NumType, typename LabelType>
size_t SequenceParser<NumType, LabelType>::UpdateBuffer()
{
    // state machine might want to look back this far, so copy to beginning
    size_t saveBytes = m_byteCounter - m_spaceDelimitedStart;
    assert(saveBytes < m_bufferSize);
    if (saveBytes)
    {
        memcpy_s(m_fileBuffer, m_bufferSize, &m_fileBuffer[m_byteCounter - m_bufferStart - saveBytes], saveBytes);
        m_bufferStart = m_byteCounter - saveBytes;
    }

    // read the next block
    size_t bytesToRead = min(m_bufferSize, m_fileSize - m_bufferStart) - saveBytes;
    size_t bytesRead = fread(m_fileBuffer + saveBytes, 1, bytesToRead, m_pFile);
    if (bytesRead == 0 && ferror(m_pFile))
        RuntimeError("SequenceParser::UpdateBuffer - error reading file");
    return bytesRead;
}

template <typename NumType, typename LabelType>
void SequenceParser<NumType, LabelType>::SetParseMode(ParseMode mode)
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
void SequenceParser<NumType, LabelType>::SetTraceLevel(int traceLevel)
{
    m_traceLevel = traceLevel;
}

// StoreLabel - string version gets last space delimited string and stores in labels vector
template <>
void SequenceParser<float, std::string>::StoreLabel(float /*finalResult*/)
{
    // for LabelFirst, Max will not be set yet, but the current byte counter is the Max, so set it
    if (m_spaceDelimitedMax <= m_spaceDelimitedStart)
        m_spaceDelimitedMax = m_byteCounter;
    std::string label((LPCSTR) &m_fileBuffer[m_spaceDelimitedStart - m_bufferStart], m_spaceDelimitedMax - m_spaceDelimitedStart);
    if (!m_beginSequence && !_stricmp(label.c_str(), m_beginTag.c_str()))
        m_beginSequence = true;
    if (!m_endSequence && !_stricmp(label.c_str(), m_endTag.c_str()))
        m_endSequence = true;
    m_labels->push_back(move(label));
    m_labelsConvertedThisLine++;
    m_elementsConvertedThisLine++;
    m_lastLabelIsString = true;
}

// DoneWithLabel - string version stores string label
template <>
void SequenceParser<float, std::string>::DoneWithLabel()
{
    if (m_labels != NULL)
        StoreLabel(0); // store the string label
    PrepareStartNumber();
}

// StoreLastLabel - string version
template <>
void SequenceParser<float, std::string>::StoreLastLabel()
{
    // see if it was already stored as a string label
    if (m_lastLabelIsString)
        return;
    StoreLabel(0);

    // we already stored a numeric version of this label in the numbers array
    // so get rid of that, the user wants it as a string
    m_numbers->pop_back();
    PrepareStartNumber();
}

// NOTE: Current code is identical to float, don't know how to specialize with template parameter that only covers one parameter

// StoreLabel - string version gets last space delimited string and stores in labels vector
template <>
void SequenceParser<double, std::string>::StoreLabel(double /*finalResult*/)
{
    // for LabelFirst, Max will not be set yet, but the current byte counter is the Max, so set it
    if (m_spaceDelimitedMax <= m_spaceDelimitedStart)
        m_spaceDelimitedMax = m_byteCounter;
    std::string label((LPCSTR) &m_fileBuffer[m_spaceDelimitedStart - m_bufferStart], m_spaceDelimitedMax - m_spaceDelimitedStart);
    if (!m_beginSequence && !_stricmp(label.c_str(), m_beginTag.c_str()))
        m_beginSequence = true;
    if (!m_endSequence && !_stricmp(label.c_str(), m_endTag.c_str()))
        m_endSequence = true;
    m_labels->push_back(move(label));
    m_labelsConvertedThisLine++;
    m_elementsConvertedThisLine++;
    m_lastLabelIsString = true;
}

// DoneWithLabel - string version stores string label
template <>
void SequenceParser<double, std::string>::DoneWithLabel()
{
    if (m_labels != NULL)
        StoreLabel(0); // store the string label
    PrepareStartNumber();
}

// StoreLastLabel - string version
template <>
void SequenceParser<double, std::string>::StoreLastLabel()
{
    // see if it was already stored as a string label
    if (m_lastLabelIsString)
        return;
    StoreLabel(0);

    // we already stored a numeric version of this label in the numbers array
    // so get rid of that, the user wants it as a string
    m_numbers->pop_back();
    PrepareStartNumber();
}

#ifdef STANDALONE
int wmain(int argc, wchar_t *argv[])
{
    SequenceParser<double, int> parser;
    std::vector<double> values;
    values.reserve(784000 * 6);
    std::vector<int> labels;
    labels.reserve(60000);
    parser.ParseInit(L"c:\\speech\\mnist\\mnist_train.txt", LabelFirst);
    //parser.ParseInit("c:\\speech\\parseTest.txt", LabelNone);
    int records = 0;
    do
    {
        int recordsRead = parser.Parse(10000, &values, &labels);
        if (recordsRead < 10000)
            parser.SetFilePosition(0); // go around again
        records += recordsRead;
        values.clear();
        labels.clear();
    } while (records < 150000);
    return records;
}
#endif

// instantiate UCI parsers for supported types
template class SequenceParser<float, int>;
template class SequenceParser<float, float>;
template class SequenceParser<float, std::string>;
template class SequenceParser<double, int>;
template class SequenceParser<double, double>;
template class SequenceParser<double, std::string>;

template <typename NumType, typename LabelType>
void LMBatchSequenceParser<NumType, LabelType>::ParseInit(LPCWSTR fileName, size_t dimFeatures, size_t dimLabelsIn, size_t dimLabelsOut, std::string beginSequenceIn /*="<s>"*/, std::string endSequenceIn /*="</s>"*/, std::string beginSequenceOut /*="O"*/, std::string endSequenceOut /*="O"*/)
{
    ::LMSequenceParser<NumType, LabelType>::ParseInit(fileName, dimFeatures, dimLabelsIn, dimLabelsOut, beginSequenceIn, endSequenceIn, beginSequenceOut, endSequenceOut);
}

template <typename NumType, typename LabelType>
long LMBatchSequenceParser<NumType, LabelType>::Parse(size_t recordsRequested, std::vector<LabelType> *labels, std::vector<NumType> *numbers, std::vector<SequencePosition> *seqPos)
{
    long linecnt;
    linecnt = ::LMSequenceParser<NumType, LabelType>::Parse(recordsRequested, labels, numbers, seqPos);

    size_t prvat = 0;
    size_t i = 0;
    for (auto ptr = seqPos->begin(); ptr != seqPos->end(); ptr++, i++)
    {
        size_t iln = ptr->labelPos - prvat;
        stSentenceInfo stinfo;
        stinfo.sLen = iln;
        stinfo.sBegin = prvat;
        stinfo.sEnd = ptr->labelPos;
        mSentenceIndex2SentenceInfo.push_back(stinfo);

        prvat = ptr->labelPos;
    }

    assert(mSentenceIndex2SentenceInfo.size() == linecnt);
    return linecnt;
}

template class LMBatchSequenceParser<float, std::string>;
template class LMBatchSequenceParser<double, std::string>;
