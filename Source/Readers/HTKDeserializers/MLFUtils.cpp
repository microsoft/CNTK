//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#define _CRT_SECURE_NO_WARNINGS
#define _SCL_SECURE_NO_WARNINGS
#include "MLFUtils.h"
#include "ReaderUtil.h"

// Disabling some deprecation warnings in boost.
// Classes that we use are not deprecated.
#pragma warning(disable:4348 4459 4100)
#include <boost/spirit/include/qi.hpp>
#pragma warning(default:4348 4459 4100)

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

    inline void EraseEmptyLines(vector<boost::iterator_range<char*>>& lines)
    {
        auto end = std::remove_if(lines.begin(), lines.end(), [](const boost::iterator_range<char*>& r) { return r.empty(); });
        lines.erase(end, lines.end());
    }

    void StateTable::ReadStateList(const wstring& stateListPath)
    {
        vector<char> buffer; // buffer owns the characters -- don't release until done
        vector<boost::iterator_range<char*>> lines = ReadNonEmptyLines(stateListPath, buffer);
        size_t index = 0;
        m_silStateMask.reserve(lines.size());

        for (index = 0; index < lines.size(); index++)
        {
            string line(lines[index].begin(), lines[index].end());

            if (m_stateTable.find(line) != m_stateTable.end())
                RuntimeError("Deduplicate two states with the same name '%s' from the state table '%ls'.", line.c_str(), stateListPath.c_str());

            m_stateTable[line] = index;
            m_silStateMask.push_back(IsSilState(line));
        }

        assert(index == m_stateTable.size());
        fprintf(stderr, "Total (%zu) state names in state list '%ls'\n", m_stateTable.size(), stateListPath.c_str());

        if (m_stateTable.empty())
            RuntimeError("State list table '%ls' is not allowed to be empty.", stateListPath.c_str());
    }

    vector<boost::iterator_range<char*>> StateTable::ReadNonEmptyLines(const wstring& path, vector<char>& buffer)
    {
        // load it into RAM in one huge chunk, not more than a couple 
        // thousand states.
        auto_file_ptr f(fopenOrDie(path, L"rb"));
        size_t len = filesize(f);
        buffer.reserve(len + 1);
        freadOrDie(buffer, len, f);
        buffer.push_back(0); // this makes it a proper C string

        vector<boost::iterator_range<char*>> lines;
        const static std::vector<bool> delim = DelimiterHash({ '\r', '\n' });
        Split(buffer.data(), buffer.data() + buffer.size(), delim, lines);

        EraseEmptyLines(lines);
        return lines;
    }

    const double MLFFrameRange::s_htkTimeToFrame = 100000.0;

    void MLFFrameRange::Build(const vector<boost::iterator_range<char*>>& tokens, const unordered_map<string, size_t>& stateTable, size_t byteOffset)
    {
        auto range = ParseFrameRange(tokens, byteOffset);
        size_t uid;
        if (!stateTable.empty()) // state table is given, check the state against the table.
        {
            auto stateName = string(tokens[2].begin(), tokens[2].end());
            auto index = stateTable.find(stateName);
            if (index == stateTable.end())
                RuntimeError("Offset '%zu': frame range state '%s' is not found in the statelist", byteOffset, stateName.c_str());

            uid = index->second; // get state index
        }
        else
        {
            // This is too simplistic for parsing more complex MLF formats. Fix when needed,
            // add support so that it can handle conditions where time instead of frame number is used.
            if (tokens.size() != 4)
                RuntimeError("Offset '%zu': CNTK supports 4-column format frame range or state list table.", byteOffset);

            if (!boost::spirit::qi::parse(tokens[3].begin(), tokens[3].end(), boost::spirit::qi::int_, uid))
                RuntimeError("Offset '%zu': cannot parse class id of the frame range", byteOffset);
        }

        VerifyAndSaveRange(range, uid, byteOffset);
    }

    void MLFFrameRange::VerifyAndSaveRange(const pair<size_t, size_t>& frameRange, size_t uid, size_t byteOffset)
    {
        if (frameRange.second < frameRange.first)
            RuntimeError("Offset '%zu': frame range end time is earlier than start time.", byteOffset);

        m_firstFrame = (unsigned int)frameRange.first;
        m_numFrames = (unsigned int)(frameRange.second - frameRange.first);
        m_classId = (ClassIdType)uid;

        // check for numeric overflow
        if (m_firstFrame != frameRange.first || m_firstFrame + m_numFrames != frameRange.second)
            RuntimeError("Offset '%zu': not enough bits for one of the frame range values.", byteOffset);

        if(m_classId != uid)
            RuntimeError("Offset '%zu': not enough bits to represent a class id '%zu'.", byteOffset, uid);
    }

    pair<size_t, size_t> MLFFrameRange::ParseFrameRange(const vector<boost::iterator_range<char*>>& tokens, size_t byteOffset)
    {
        if (tokens.size() < 2)
            RuntimeError("Offset '%zu': do not support frame range format with less than two columns.", byteOffset);

        double rts = 0;
        if (!boost::spirit::qi::parse(tokens[0].begin(), tokens[0].end(), boost::spirit::qi::double_, rts))
            RuntimeError("Offset '%zu': cannot parse start frame of range.", byteOffset);

        double rte = 0;
        if (!boost::spirit::qi::parse(tokens[1].begin(), tokens[1].end(), boost::spirit::qi::double_, rte))
            RuntimeError("Offset '%zu': cannot parse end frame of range.", byteOffset);

        // Simulating the old reader behavior.
        // If the difference between two frames is more than s_htkTimeToFrame, we expect conversion to time
        if (rte - rts >= s_htkTimeToFrame - 1) // convert time to frame
        {
            return make_pair(
                (size_t)(rts / s_htkTimeToFrame + 0.5),
                (size_t)(rte / s_htkTimeToFrame + 0.5));
        }
        else
        {
            return make_pair((size_t)(rts), (size_t)(rte));
        }
    }

    // Parses the data into a vector of MLFFrameRanges.
    bool MLFUtteranceParser::Parse(const boost::iterator_range<char*>& sequenceData, vector<MLFFrameRange>& utterance, size_t sequenceOffset)
    {
        // Split to lines.
        vector<boost::iterator_range<char*>> lines;
        lines.reserve(512);

        const static std::vector<bool> delim = DelimiterHash({ '\r', '\n' });
        Split(sequenceData.begin(), sequenceData.end(), delim, lines);
        EraseEmptyLines(lines);

        // Start parsing of actual entry
        size_t idx = 0;
        string sequenceKey = string(lines[idx].begin(), lines[idx].end());
        idx++;

        // Check that mlf entry has a correct sequence key.
        if (sequenceKey.length() < 3 || sequenceKey[0] != '"' || sequenceKey[sequenceKey.length() - 1] != '"')
        {
            fprintf(stderr, "WARNING: skipping sequence entry '%s' due to it being too short or not quoted\n", sequenceKey.c_str());
            return false;
        }

        // strip quotes
        sequenceKey = sequenceKey.substr(1, sequenceKey.length() - 2);

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

            const static std::vector<bool> spaceDelim = DelimiterHash({ ' ' });
            Split(lines[i].begin(), lines[i].end(), spaceDelim, tokens);

            auto& current = utterance[i - s];
            current.Build(tokens, m_states ? m_states->States() : empty, sequenceOffset + std::distance(sequenceData.begin(), lines[i].begin()));

            // Check that frames are sequential.
            if (i > s)
            {
                const auto& previous = utterance[i - s - 1];
                if (previous.FirstFrame() + previous.NumFrames() != current.FirstFrame())
                {
                    fprintf(stderr, "WARNING: Labels are not in the consecutive order MLF in label set for utterance '%s'", sequenceKey.c_str());
                    return false;
                }
            }
        }

        if (utterance.front().FirstFrame() != 0)
        {
            fprintf(stderr, "WARNING: Invalid first frame in utterance '%s'", sequenceKey.c_str());
            return false;
        }

        return true;
    }

}}}
