//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#define __STDC_FORMAT_MACROS
#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS
#include <inttypes.h>
#include "MLFUtils.h"

#pragma warning(disable:4348 4459 4100)
#include <boost/algorithm/string.hpp>
#include <boost/spirit/include/qi.hpp>
#pragma warning(default:4348 4459 4100)

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    void StateTable::ReadStateList(const wstring& stateListPath)
    {
        vector<char> buffer; // buffer owns the characters -- don't release until done
        vector<boost::iterator_range<char*>> lines = ReadLines(stateListPath, buffer);
        size_t index = 0;
        m_silStateMask.reserve(lines.size());

        for (index = 0; index < lines.size(); index++)
        {
            string line(lines[index].begin(), lines[index].end());

            m_stateTable[line] = index;
            m_silStateMask.push_back(IsSilState(line));
        }

        if (index != m_stateTable.size())
            RuntimeError("readstatelist: lines (%d) not equal to statelistmap size (%d)", (int)index, (int)m_stateTable.size());

        if (m_stateTable.size() != m_silStateMask.size())
            RuntimeError("readstatelist: size of statelookuparray (%d) not equal to statelistmap size (%d)", (int)m_silStateMask.size(), (int)m_stateTable.size());

        fprintf(stderr, "total %lu state names in state list %ls\n", (unsigned long)m_stateTable.size(), stateListPath.c_str());

        if (m_stateTable.empty())
            RuntimeError("State list table is not allowed to be empty.");
    }

    vector<boost::iterator_range<char*>> StateTable::ReadLines(const wstring& path, vector<char>& buffer)
    {
        // load it into RAM in one huge chunk
        auto_file_ptr f(fopenOrDie(path, L"rb"));
        size_t len = filesize(f);
        buffer.reserve(len + 1);
        freadOrDie(buffer, len, f);
        buffer.push_back(0); // this makes it a proper C string

        vector<boost::iterator_range<char*>> lines;
        lines.reserve(len / 20);
        auto range = boost::make_iterator_range(buffer.data(), buffer.data() + buffer.size());
        boost::split(lines, range, boost::is_any_of("\r\n"));

        auto end = std::remove_if(lines.begin(), lines.end(), [](const boost::iterator_range<char*>& r) { return r.begin() == r.end(); });
        lines.erase(end, lines.end());
        return lines;
    }

    //////////////////////////////////////////////////////////////////////////////////////////////////////////

    const double MLFFrameRange::htkTimeToFrame = 100000.0;

    void MLFFrameRange::Build(const vector<boost::iterator_range<char*>>& tokens, const unordered_map<string, size_t>& stateTable)
    {
        auto range = ParseFrameRange(tokens);
        size_t uid;
        if (!stateTable.empty()) // state table is given, check the state against the table.
        {
            auto stateName = string(tokens[2].begin(), tokens[2].end());
            auto index = stateTable.find(stateName);
            if (index == stateTable.end())
                RuntimeError("MLFFrameRange: state %s not found in statelist", stateName.c_str());

            uid = index->second; // get state index
        }
        else
        {
            // This is too simplistic for parsing more complex MLF formats.Fix when needed,
            // add support so that it can handle conditions where time instead of frame numer is used.
            if (tokens.size() != 4)
                RuntimeError("MLFFrameRange: currently we only support 4-column format or state list table.");

            if (!boost::spirit::qi::parse(tokens[3].begin(), tokens[3].end(), boost::spirit::qi::int_, uid))
                RuntimeError("MLFFrameRange: cannot parse class id.");
        }

        VerifyAndSaveRange(range, uid);
    }

    void MLFFrameRange::VerifyAndSaveRange(const pair<size_t, size_t>& frameRange, size_t uid)
    {
        if (frameRange.second < frameRange.first)
            RuntimeError("MLFFrameRange Error: end time earlier than start time.");

        m_firstFrame = (unsigned int)frameRange.first;
        m_numFrames = (unsigned int)(frameRange.second - frameRange.first);
        m_classId = (ClassIdType)uid;

        // check for numeric overflow
        if (m_firstFrame != frameRange.first || m_firstFrame + m_numFrames != frameRange.second || m_classId != uid)
            RuntimeError("MLFFrameRange Error: not enough bits for one of the values.");
    }

    pair<size_t, size_t> MLFFrameRange::ParseFrameRange(const vector<boost::iterator_range<char*>>& tokens)
    {
        if (tokens.size() < 2)
            RuntimeError("MLFFrameRange: currently MLF does not support format with less than two columns.");

        double rts = 0;
        if (!boost::spirit::qi::parse(tokens[0].begin(), tokens[0].end(), boost::spirit::qi::double_, rts))
            RuntimeError("MLFFrameRange: cannot parse start frame.");

        double rte = 0;
        if (!boost::spirit::qi::parse(tokens[1].begin(), tokens[1].end(), boost::spirit::qi::double_, rte))
            RuntimeError("MLFFrameRange: cannot parse end frame.");

        // if the difference between two frames is more than htkTimeToFrame, we expect conversion to time
        if (rte - rts >= htkTimeToFrame - 1) // convert time to frame
        {
            return make_pair(
                (size_t)(rts / htkTimeToFrame + 0.5),
                (size_t)(rte / htkTimeToFrame + 0.5));
        }
        else
        {
            return make_pair((size_t)(rts), (size_t)(rte));
        }
    }

    ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

    bool MLFUtteranceParser::Parse(const SequenceDescriptor& sequence, const boost::iterator_range<char*>& sequenceData, vector<MLFFrameRange>& utterance)
    {
        // Split to lines.
        vector<boost::iterator_range<char*>> lines;
        lines.reserve(512);

        boost::split(lines, sequenceData, boost::is_any_of("\r\n"));

        auto end = std::remove_if(lines.begin(), lines.end(),
            [](const boost::iterator_range<char*>& a) { return std::distance(a.begin(), a.end()) == 0; });
        lines.erase(end, lines.end());

        // Start actual parsing of actual entry
        size_t idx = 0;
        string sequenceKey = string(lines[idx].begin(), lines[idx].end());
        idx++;

        // Check that mlf entry has a correct sequence key.
        if (sequenceKey.length() < 3 || sequenceKey[0] != '"' || sequenceKey[sequenceKey.length() - 1] != '"')
        {
            fprintf(stderr, "WARNING: sequence entry (%s)\n", sequenceKey.c_str());
            fprintf(stderr, "Skip current mlf entry from offset '%" PRIu64 "' until offset '%" PRIu64 "'.\n", sequence.m_fileOffsetBytes, sequence.m_fileOffsetBytes + sequence.m_byteSize);
            return false;
        }

        sequenceKey = sequenceKey.substr(1, sequenceKey.length() - 2); // strip quotes

        if (sequenceKey.size() > 2 && sequenceKey[0] == '*' && sequenceKey[1] == '/')
            sequenceKey = sequenceKey.substr(2);

        // Remove extension if specified.
        sequenceKey = sequenceKey.substr(0, sequenceKey.find_last_of("."));

        // determine content line range [s,e)
        size_t s = idx;
        size_t e = lines.size() - 1;

        if (s >= e)
        {
            fprintf(stderr, "WARNING: sequence entry (%s) is empty\n", sequenceKey.c_str());
            return false;
        }

        utterance.resize(e - s);
        vector<boost::iterator_range<char*>> tokens;
        unordered_map<string, size_t> empty;
        for (size_t i = s; i < e; i++)
        {
            tokens.clear();
            boost::split(tokens, lines[i], boost::is_any_of(" "));

            auto& current = utterance[i - s];
            current.Build(tokens, m_states ? m_states->States() : empty);

            // Check that frames are sequential.
            if (i > s)
            {
                const auto& previous = utterance[i - s - 1];
                if (previous.FirstFrame() + previous.NumFrames() != current.FirstFrame())
                {
                    fprintf(stderr, "WARNING: Labels are not in the consecutive order MLF in label set: %s", sequenceKey.c_str());
                    return false;
                }
            }
        }

        if (utterance.front().FirstFrame() != 0)
        {
            fprintf(stderr, "WARNING: Invalid first frame in utterance: %s", sequenceKey.c_str());
            return false;
        }

        return true;
    }

}}}
