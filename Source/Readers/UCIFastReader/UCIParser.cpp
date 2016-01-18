// UCIParser.cpp : Parses the UCI format using a custom state machine (for speed)
//
//
// <copyright file="UCIParser.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "stdafx.h"
#include "Basics.h"
#include "UCIParser.h"
#include <stdexcept>
#include <stdint.h>

#if WIN32
#define ftell64 _ftelli64
#else
#define ftell64 ftell
#endif

// SetState for a particular value
template <typename NumType, typename LabelType>
void UCIParser<NumType, LabelType>::SetState(int value, ParseState m_current_state, ParseState next_state)
{
    DWORD ul = (DWORD) next_state;
    int range_shift = ((int) m_current_state) << 8;
    m_stateTable[range_shift + value] = ul;
}

// SetStateRange - set states transitions for a range of values
template <typename NumType, typename LabelType>
void UCIParser<NumType, LabelType>::SetStateRange(int value1, int value2, ParseState m_current_state, ParseState next_state)
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
void UCIParser<NumType, LabelType>::SetupStateTables()
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
void UCIParser<NumType, LabelType>::PrepareStartLine()
{
    m_numbersConvertedThisLine = 0;
    m_labelsConvertedThisLine = 0;
    m_elementsConvertedThisLine = 0;
    m_spaceDelimitedStart = m_byteCounter;
    m_spaceDelimitedMax = m_byteCounter;
    m_lastLabelIsString = false;
}

// reset all number accumulation variables
template <typename NumType, typename LabelType>
void UCIParser<NumType, LabelType>::PrepareStartNumber()
{
    m_partialResult = 0;
    m_builtUpNumber = 0;
    m_divider = 0;
    m_wholeNumberMultiplier = 1;
    m_exponentMultiplier = 1;
}

// reset all state variables to start reading at a new position
template <typename NumType, typename LabelType>
void UCIParser<NumType, LabelType>::PrepareStartPosition(size_t position)
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

// UCIParser constructor
template <typename NumType, typename LabelType>
UCIParser<NumType, LabelType>::UCIParser()
{
    Init();
}

// setup all the state variables and state tables for state machine
template <typename NumType, typename LabelType>
void UCIParser<NumType, LabelType>::Init()
{
    PrepareStartPosition(0);
    m_fileBuffer = NULL;
    m_pFile = NULL;
    m_stateTable = new DWORD[AllStateMax * 256];
    SetupStateTables();
}

// Parser destructor
template <typename NumType, typename LabelType>
UCIParser<NumType, LabelType>::~UCIParser()
{
    delete m_stateTable;
    delete m_fileBuffer;
    if (m_pFile)
        fclose(m_pFile);
}

// DoneWithLabel - Called when a string label is found
template <typename NumType, typename LabelType>
void UCIParser<NumType, LabelType>::DoneWithLabel()
{
    // if we haven't set the max yet, use the current byte Counter
    if (m_spaceDelimitedMax <= m_spaceDelimitedStart)
        m_spaceDelimitedMax = m_byteCounter;
    {
        std::string label((LPCSTR) &m_fileBuffer[m_spaceDelimitedStart - m_bufferStart], m_spaceDelimitedMax - m_spaceDelimitedStart);
        fprintf(stderr, "\n** String found in numeric-only file: %s\n", label.c_str());
        m_labelsConvertedThisLine++;
        m_elementsConvertedThisLine++;
        m_lastLabelIsString = true;
    }
    PrepareStartNumber();
}

// Called when a number is complete
template <typename NumType, typename LabelType>
void UCIParser<NumType, LabelType>::DoneWithValue()
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

        // if it's a label, store in label location instead of number location
        int index = m_elementsConvertedThisLine;
        bool stored = false;
        if (m_startLabels <= index && index < m_startLabels + m_dimLabels)
        {
            StoreLabel(FinalResult);
            stored = true;
        }
        if (m_startFeatures <= index && index < m_startFeatures + m_dimFeatures)
        {
            m_numbers->push_back(FinalResult);
            m_totalNumbersConverted++;
            m_numbersConvertedThisLine++;
            m_elementsConvertedThisLine++;
            m_lastLabelIsString = false;
            stored = true;
        }
        // if we haven't stored anything we need to skip the current symbol, so increment
        if (!stored)
        {
            m_elementsConvertedThisLine++;
        }
    }

    PrepareStartNumber();
}

// store label is specialized by LabelType
template <typename NumType, typename LabelType>
void UCIParser<NumType, LabelType>::StoreLabel(NumType value)
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
void UCIParser<NumType, LabelType>::StoreLastLabel()
{
    assert(!m_lastLabelIsString); // file format error, last label was a string...
    NumType value = m_numbers->back();
    m_numbers->pop_back();
    m_labels->push_back((LabelType) value);
}

// ParseInit - Initialize a parse of a file
// fileName - path to the file to open
// startFeatures - column (zero based) where features start
// dimFeatures - number of features
// startLabels - column (zero based) where Labels start
// dimLabels - number of Labels
// bufferSize - size of temporary buffer to store reads
// startPosition - file position on which we should start
template <typename NumType, typename LabelType>
void UCIParser<NumType, LabelType>::ParseInit(LPCWSTR fileName, size_t startFeatures, size_t dimFeatures, size_t startLabels, size_t dimLabels, size_t bufferSize, size_t startPosition)
{
    assert(fileName != NULL);
    m_startLabels = startLabels;
    m_dimLabels = dimLabels;
    m_startFeatures = startFeatures;
    m_dimFeatures = dimFeatures;
    m_parseMode = ParseNormal;
    m_traceLevel = 0;
    m_bufferSize = bufferSize;
    m_bufferStart = startPosition;

    // if we have a file already open, cleanup
    if (m_pFile != NULL)
        UCIParser<NumType, LabelType>::~UCIParser();

    errno_t err = _wfopen_s(&m_pFile, fileName, L"rb");
    if (err)
        RuntimeError("UCIParser::ParseInit - error opening file");
    int rc = _fseeki64(m_pFile, 0, SEEK_END);
    if (rc)
        RuntimeError("UCIParser::ParseInit - error seeking in file");

    m_fileSize = GetFilePosition();
    m_fileBuffer = new BYTE[m_bufferSize];
    SetFilePosition(startPosition);
}

// GetFilePosition - Get the current file position in the text file
// returns current position in the file
template <typename NumType, typename LabelType>
int64_t UCIParser<NumType, LabelType>::GetFilePosition()
{
    int64_t position = ftell64(m_pFile);
    if (position == -1L)
        RuntimeError("UCIParser::GetFilePosition - error retrieving file position in file");
    return position;
}

// SetFilePosition - Set the current file position from the beginning of the file, and read in the first block of data
// state machine mode will be initialized similar to the beginning of the file
// it is recommneded that only return values from GetFilePosition() known to be the start of a line
// and zero be passed to this function
template <typename NumType, typename LabelType>
void UCIParser<NumType, LabelType>::SetFilePosition(int64_t position)
{
    int rc = _fseeki64(m_pFile, position, SEEK_SET);
    if (rc)
        RuntimeError("UCIParser::SetFilePosition - error seeking in file");

    // setup state machine to start at this position
    PrepareStartPosition(position);

    // read in the first buffer of data from this position,  first buffer is expected to be read after a reposition
    UpdateBuffer();

    // FUTURE: in debug we could validate the value is either 0, or the previous character is a '\n'
}

// HasMoreData - test if the current dataset have more data, or just whitespace
// returns - true if it has more data, false if not
template <typename NumType, typename LabelType>
bool UCIParser<NumType, LabelType>::HasMoreData()
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
size_t UCIParser<NumType, LabelType>::UpdateBuffer()
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
        RuntimeError("UCIParser::UpdateBuffer - error reading file");
    return bytesRead;
}

template <typename NumType, typename LabelType>
void UCIParser<NumType, LabelType>::SetParseMode(ParseMode mode)
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
void UCIParser<NumType, LabelType>::SetTraceLevel(int traceLevel)
{
    m_traceLevel = traceLevel;
}

// Parse - Parse the data
// recordsRequested - number of records requested
// numbers - pointer to vector to return the numbers (must be allocated)
// labels - pointer to vector to return the labels (defaults to null)
// returns - number of records actually read, if the end of file is reached the return value will be < requested records
template <typename NumType, typename LabelType>
long UCIParser<NumType, LabelType>::Parse(size_t recordsRequested, std::vector<NumType> *numbers, std::vector<LabelType> *labels)
{
    assert(numbers != NULL || m_dimFeatures == 0 || m_parseMode == ParseLineCount);
    assert(labels != NULL || m_dimLabels == 0 || m_parseMode == ParseLineCount);

    // transfer to member variables
    m_numbers = numbers;
    m_labels = labels;

    long TickStart = GetTickCount();
    long recordCount = 0;
    size_t bufferIndex = m_byteCounter - m_bufferStart;
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
                recordCount++; // done with another record
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
    } // while 1

    long TickStop = GetTickCount();

    long TickDelta = TickStop - TickStart;

    if (m_traceLevel > 2)
        fprintf(stderr, "\n%ld ms, %ld numbers parsed\n\n", TickDelta, m_totalNumbersConverted);
    return recordCount;
}

// StoreLabel - string version gets last space delimited string and stores in labels vector
template <>
void UCIParser<float, std::string>::StoreLabel(float /*finalResult*/)
{
    // for LabelFirst, Max will not be set yet, but the current byte counter is the Max, so set it
    if (m_spaceDelimitedMax <= m_spaceDelimitedStart)
        m_spaceDelimitedMax = m_byteCounter;
    std::string label((LPCSTR) &m_fileBuffer[m_spaceDelimitedStart - m_bufferStart], m_spaceDelimitedMax - m_spaceDelimitedStart);
    m_labels->push_back(move(label));
    m_labelsConvertedThisLine++;
    m_elementsConvertedThisLine++;
    m_lastLabelIsString = true;
}

// DoneWithLabel - string version stores string label
template <>
void UCIParser<float, std::string>::DoneWithLabel()
{
    if (m_labels != NULL)
        StoreLabel(0); // store the string label
    PrepareStartNumber();
}

// StoreLastLabel - string version
template <>
void UCIParser<float, std::string>::StoreLastLabel()
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
void UCIParser<double, std::string>::StoreLabel(double /*finalResult*/)
{
    // for LabelFirst, Max will not be set yet, but the current byte counter is the Max, so set it
    if (m_spaceDelimitedMax <= m_spaceDelimitedStart)
        m_spaceDelimitedMax = m_byteCounter;
    std::string label((LPCSTR) &m_fileBuffer[m_spaceDelimitedStart - m_bufferStart], m_spaceDelimitedMax - m_spaceDelimitedStart);
    m_labels->push_back(move(label));
    m_labelsConvertedThisLine++;
    m_elementsConvertedThisLine++;
    m_lastLabelIsString = true;
}

// DoneWithLabel - string version stores string label
template <>
void UCIParser<double, std::string>::DoneWithLabel()
{
    if (m_labels != NULL)
        StoreLabel(0); // store the string label
    PrepareStartNumber();
}

// StoreLastLabel - string version
template <>
void UCIParser<double, std::string>::StoreLastLabel()
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
    UCIParser<double, int> parser;
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
template class UCIParser<float, int>;
template class UCIParser<float, float>;
template class UCIParser<float, std::string>;
template class UCIParser<double, int>;
template class UCIParser<double, double>;
template class UCIParser<double, std::string>;
