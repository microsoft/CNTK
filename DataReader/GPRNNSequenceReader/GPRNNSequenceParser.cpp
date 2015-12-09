// GPRNNSequenceParser.cpp : Parses the UCI format using a custom state machine (for speed)
//
// <copyright file="GPRNNSequenceParser.cpp" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//


#include "stdafx.h"
#include "GPRNNSequenceParser.h"
#include <stdexcept>
#include <fileutil.h>

namespace Microsoft { namespace MSR { namespace CNTK {

// GPRNNSequenceParser constructor
template <typename NumType, typename LabelType>
GPRNNSequenceParser<NumType, LabelType>::GPRNNSequenceParser()
{
    Init();
}

// setup all the state variables and state tables for state machine
template <typename NumType, typename LabelType>
void GPRNNSequenceParser<NumType, LabelType>::Init()
{
    m_pFile = NULL;
}

// Parser destructor
template <typename NumType, typename LabelType>
GPRNNSequenceParser<NumType, LabelType>::~GPRNNSequenceParser()
{
    if (m_pFile)
        fclose(m_pFile);
}


// instantiate UCI parsers for supported types
template class GPRNNSequenceParser<float, int>;
template class GPRNNSequenceParser<float, float>;
template class GPRNNSequenceParser<float, std::string>;
template class GPRNNSequenceParser<float, std::wstring>;
template class GPRNNSequenceParser<double, int>;
template class GPRNNSequenceParser<double, double>;
template class GPRNNSequenceParser<double, std::string>;
template class GPRNNSequenceParser<double, std::wstring>;

template<class NumType, class LabelType>
long BatchGPRNNSequenceParser<NumType, LabelType>::Parse(size_t recordsRequested, std::vector<long> *labels, std::vector<std::pair<std::vector<LabelIdType>, std::vector<std::pair<LabelIdType, LabelIdType> > > > *input, std::vector<SequencePosition> *seqPos, bool canMultiplePassData)
{
    // transfer to member variables
	m_inputs = input;
	m_labels = labels;

    long recordCount = 0;
    long orgRecordCount = (long)labels->size();
    long lineCount = 0;
    bool bAtEOS = false; /// whether the reader is at the end of sentence position
    SequencePosition sequencePositionLast(0, 0, 0);
    /// get line
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

		if (bBlankLine && bAtEOS)
		{
			continue;
		}

        if (bBlankLine && !bAtEOS && input->size() > 0 && labels->size() > 0)
        {
            AddOneItem(labels, input, seqPos, lineCount, recordCount, orgRecordCount, sequencePositionLast);
            bAtEOS = true;
            continue;
        }

        vstr = wsep_string(ch, L"\t");
		if (vstr.size() < 2 || vstr.size() > 3) {
			LogicError("File Format Requirement: OutputLabel\tWord Feature\tAux Feature");
			continue;
		}


        bAtEOS = false;

		// first column is output label, second column is word context, third context is auxiliary feature
		LabelIdType outputLabel = -1;
		std::vector<LabelIdType> contextWordLabel;
		std::vector<std::pair<LabelIdType, LabelIdType>> auxFeat;

		outputLabel = msra::strfun::toint(vstr[0]);

		if (vstr.size() >= 2){
			std::vector<std::wstring> strContext = msra::strfun::split(vstr[1], L",");
			for (int i = 0; i < strContext.size(); ++i){
				contextWordLabel.push_back(msra::strfun::toint(strContext[i]));
			}
		}

		if (vstr.size() >= 3){
			std::vector<std::wstring> strAux = msra::strfun::split(vstr[2], L",");
			for (std::wstring iter : strAux){

				std::vector<std::wstring> kvpair = msra::strfun::split(iter, L":");
				if (kvpair.size() != 2){
					LogicError("auxiliary feature format error");
				}

				int key = msra::strfun::toint(kvpair[0]);
				int val = msra::strfun::toint(kvpair[1]);

				auxFeat.push_back(std::pair<LabelIdType, LabelIdType>(key, val));
			}
		}

		labels->push_back(outputLabel);
		input->push_back(std::pair<std::vector<LabelIdType>, std::vector<std::pair<LabelIdType, LabelIdType> > >(contextWordLabel, auxFeat));

    } // while

    if (sequencePositionLast.inputPos< input->size())
        AddOneItem(labels, input, seqPos, lineCount, recordCount, orgRecordCount, sequencePositionLast);

    int prvat = 0;
    size_t i = 0;
    for (auto ptr = seqPos->begin(); ptr != seqPos->end(); ptr++, i++)
    {
        size_t iln = ptr->labelPos - prvat;
        stSentenceInfo stinfo;
        stinfo.sLen = iln;
        stinfo.sBegin = prvat;
        stinfo.sEnd = (int)ptr->labelPos;
        mSentenceIndex2SentenceInfo.push_back(stinfo);

        prvat = (int)ptr->labelPos;
    }

    fprintf(stderr, "BatchGPRNNSequenceParser: parse %ld lines\n", lineCount);
    return lineCount;
}


template class BatchGPRNNSequenceParser<float, std::string>;
template class BatchGPRNNSequenceParser<double, std::string>;
template class BatchGPRNNSequenceParser<float, std::wstring>;
template class BatchGPRNNSequenceParser<double, std::wstring>;
}}}
