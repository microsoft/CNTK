// UCIParser.h : Parses the UCI format using a custom state machine (for speed)
//
//
// <copyright file="UCIParser.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//

#include "stdafx.h"
#include <string>
#include <vector>
#include <assert.h>
#include <stdint.h>
#include <algorithm>

#ifdef min
#undef min
#endif
#define min std::min

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

// UCIParser - the parser for the UCI format files
// for ultimate speed, this class implements a state machine to read these format files
template <typename NumType, typename LabelType = int>
class UCIParser
{
private:
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
        AllStateMax = 12,
        Error = 12
    };

    // type of label processing
    ParseMode m_parseMode;

    // definition of label and feature locations
    size_t m_startLabels;
    size_t m_dimLabels;
    size_t m_startFeatures;
    size_t m_dimFeatures;

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

    // global stats
    int64_t m_totalNumbersConverted;
    int64_t m_totalLabelsConverted;

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
    // UCIParser constructor
    UCIParser();
    // setup all the state variables and state tables for state machine
    void Init();

    // Parser destructor
    ~UCIParser();

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
    // startFeatures - column (zero based) where features start
    // dimFeatures - number of features
    // startLabels - column (zero based) where Labels start
    // dimLabels - number of Labels
    // bufferSize - size of temporary buffer to store reads
    // startPosition - file position on which we should start
    void ParseInit(LPCWSTR fileName, size_t startFeatures, size_t dimFeatures, size_t startLabels, size_t dimLabels, size_t bufferSize = 1024 * 256, size_t startPosition = 0);

    // Parse - Parse the data
    // recordsRequested - number of records requested
    // numbers - pointer to vector to return the numbers (must be allocated)
    // labels - pointer to vector to return the labels (defaults to null)
    // returns - number of records actually read, if the end of file is reached the return value will be < requested records
    long Parse(size_t recordsRequested, std::vector<NumType> *numbers, std::vector<LabelType> *labels = NULL);

    int64_t GetFilePosition();
    void SetFilePosition(int64_t position);

    // HasMoreData - test if the current dataset have more data
    // returns - true if it does, false if not
    bool HasMoreData();
};

// StoreLabel - string version gets last space delimited string and stores in labels vector
template <>
void UCIParser<float, std::string>::StoreLabel(float finalResult);

// DoneWithLabel - string version stores string label
template <>
void UCIParser<float, std::string>::DoneWithLabel();

// StoreLastLabel - string version
template <>
void UCIParser<float, std::string>::StoreLastLabel();

// NOTE: Current code is identical to float, don't know how to specialize with template parameter that only covers one parameter

// StoreLabel - string version gets last space delimited string and stores in labels vector
template <>
void UCIParser<double, std::string>::StoreLabel(double finalResult);

// DoneWithLabel - string version stores string label
template <>
void UCIParser<double, std::string>::DoneWithLabel();

// StoreLastLabel - string version
template <>
void UCIParser<double, std::string>::StoreLastLabel();
