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
    m_pFile = NULL;
}

// Parser destructor
template <typename NumType, typename LabelType>
LUSequenceParser<NumType, LabelType>::~LUSequenceParser()
{
    if (m_pFile)
        fclose(m_pFile);
}

// instantiate UCI parsers for supported types
template class LUSequenceParser<float, int>;
template class LUSequenceParser<float, float>;
template class LUSequenceParser<float, std::string>;
template class LUSequenceParser<float, std::wstring>;
template class LUSequenceParser<double, int>;
template class LUSequenceParser<double, double>;
template class LUSequenceParser<double, std::string>;
template class LUSequenceParser<double, std::wstring>;

template <class NumType, class LabelType>
long BatchLUSequenceParser<NumType, LabelType>::Parse(size_t recordsRequested, std::vector<long> *labels, std::vector<vector<long>> *input, std::vector<SequencePosition> *seqPos, const map<wstring, long> &inputlabel2id, const map<wstring, long> &outputlabel2id, bool canMultiplePassData)
{
    fprintf(stderr, "BatchLUSequenceParser: Parsing input data...\n");

    // transfer to member variables
    m_inputs = input;
    m_labels = labels;

    long recordCount = 0;
    long orgRecordCount = (long) labels->size();
    long lineCount = 0;
    long tokenCount = 0;
    bool bAtEOS = false; /// whether the reader is at the end of sentence position
    SequencePosition sequencePositionLast(0, 0, 0);

    wstring ch;
    while (lineCount < recordsRequested && mFile.good())
    {
        getline(mFile, ch);
        ch = wtrim(ch);

        if (mFile.eof())
        {
            if (canMultiplePassData)
            {
                ParseReset(); /// restart from the corpus begining
                continue;
            }
            else
                break;
        }

        std::vector<wstring> vstr;
        bool bBlankLine = (ch.length() == 0);
        if (bBlankLine && !bAtEOS && input->size() > 0 && labels->size() > 0)
        {
            AddOneItem(labels, input, seqPos, lineCount, recordCount, orgRecordCount, sequencePositionLast);
            bAtEOS = true;
            continue;
        }

        // got a token
        tokenCount++;

        vstr = wsep_string(ch, L" ");
        if (vstr.size() < 2)
            continue;

        bAtEOS = false;
        vector<long> vtmp;
        for (size_t i = 0; i < vstr.size() - 1; i++)
        {
            if (inputlabel2id.find(vstr[i]) == inputlabel2id.end())
            {
                if (inputlabel2id.find(mUnkStr) == inputlabel2id.end())
                {
                    LogicError("cannot find item %ls and unk str %ls in input label", vstr[i].c_str(), mUnkStr.c_str());
                }
                vtmp.push_back(inputlabel2id.find(mUnkStr)->second);
            }
            else
                vtmp.push_back(inputlabel2id.find(vstr[i])->second);
        }
        if (outputlabel2id.find(vstr[vstr.size() - 1]) == outputlabel2id.end())
        {
            if (outputlabel2id.find(mUnkStr) == outputlabel2id.end())
                LogicError("cannot find item %ls and unk str %ls in output label", vstr[vstr.size() - 1].c_str(), mUnkStr.c_str());
            labels->push_back(outputlabel2id.find(mUnkStr)->second);
        }
        else
            labels->push_back(outputlabel2id.find(vstr[vstr.size() - 1])->second);
        input->push_back(vtmp);
        if ((vstr[vstr.size() - 1] == m_endSequenceOut ||
             /// below is for backward support
             vstr[0] == m_endTag) &&
            input->size() > 0 && labels->size() > 0)
        {
            AddOneItem(labels, input, seqPos, lineCount, recordCount, orgRecordCount, sequencePositionLast);
            bAtEOS = true;
        }

    } // while

    if (sequencePositionLast.inputPos < input->size())
        AddOneItem(labels, input, seqPos, lineCount, recordCount, orgRecordCount, sequencePositionLast);

    int prvat = 0;
    size_t i = 0;
    for (auto ptr = seqPos->begin(); ptr != seqPos->end(); ptr++, i++)
    {
        size_t iln = ptr->labelPos - prvat;
        stSentenceInfo stinfo;
        stinfo.sLen = iln;
        stinfo.sBegin = prvat;
        stinfo.sEnd = (int) ptr->labelPos;
        mSentenceIndex2SentenceInfo.push_back(stinfo);

        prvat = (int) ptr->labelPos;
    }

    fprintf(stderr, "BatchLUSequenceParser: Parsed %ld lines with a total of %ld+%ld tokens.\n", (long) lineCount, (long) (tokenCount - lineCount) /*exclude EOS*/, (long) lineCount);
    return lineCount;
}

template class BatchLUSequenceParser<float, std::string>;
template class BatchLUSequenceParser<double, std::string>;
template class BatchLUSequenceParser<float, std::wstring>;
template class BatchLUSequenceParser<double, std::wstring>;
} } }
