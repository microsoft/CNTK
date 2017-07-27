//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define __STDC_FORMAT_MACROS
#define _CRT_SECURE_NO_WARNINGS
#include <inttypes.h>
#include <future>
#include "IndexBuilder.h"
#include "ReaderConstants.h"
#include "FileWrapper.h"
#include "EnvironmentUtil.h"
#include <sstream>

namespace CNTK {

using namespace std;

IndexBuilder::IndexBuilder(const FileWrapper& input)
    : m_input(input),
    m_corpus(nullptr),
    m_isCacheEnabled(false),
    m_chunkSize(g_32MB),
    m_bufferSize(g_2MB),
    m_primary(true)
{}

shared_ptr<Index> IndexBuilder::Build()
{
    if (m_isCacheEnabled) 
    {
        auto cacheFilename = GetCacheFilename();
        if (msra::files::fuptodate(cacheFilename, m_input.Filename(), true))
        {
            // cache file is up-to-date, try to reconstruct the index from cache.
            auto index = TryLoadFromCache(cacheFilename, m_chunkSize);

            if (index != nullptr) 
            {
                if (!m_primary) 
                    index->MapSequenceKeyToLocation();
                return index;
            }
        }
    }
    
    auto index = make_shared<Index>(m_chunkSize);
    
    Populate(index);

    if (!m_corpus || m_corpus->IsNumericSequenceKeys() || m_corpus->IsHashingEnabled())
    {
        // For now, we do not cache index if input contains non-numeric sequence ids 
        // and the corpus does not use a (deterministic and stateless) hashing procedure
        // to transform sequence ids into numeric keys.
        WriteIndexCacheAsync(index);
    }

    if (!m_primary)
        index->MapSequenceKeyToLocation();
    return index;
}


void IndexBuilder::WriteIndexCacheAsync(shared_ptr<Index>& index) 
{
    if (!m_isCacheEnabled)
        return;

    if (Microsoft::MSR::CNTK::EnvironmentUtil::GetLocalMPINodeRank() != 0)
        return; // only the main node should write the cache file.
    
    auto cacheFilename = GetCacheFilename();

    // using thread(lambda).detach() as a workaround the blocking
    // async destructor.
    thread([cacheFilename, index]()
    {
        // At this point, it's safe to assume that the previous cache is stale,
        // remove the cache file if it exists (return value is ignored).
        _wunlink(cacheFilename.c_str());

        bool isCacheEnabled = true;
        auto temp = cacheFilename + L".tmp";
        {
            FileWrapper cache(temp, L"wb");
            isCacheEnabled = cache.IsOpen();

            Prefix prefix(s_magic, s_version, index->NumberOfSequences(), uint64_t(sizeof(Prefix)));

            isCacheEnabled = isCacheEnabled && cache.TryWrite(prefix);

            IndexedSequence cachedSequence;
            for (auto& chunk : index->Chunks())
            {
                for (auto& sequence : chunk.Sequences())
                {
                    cachedSequence.SetKey(sequence.m_key)
                        .SetNumberOfSamples(sequence.NumberOfSamples())
                        .SetSize(sequence.SizeInBytes())
                        .SetOffset(chunk.StartOffset() + sequence.OffsetInChunk());

                    isCacheEnabled = isCacheEnabled && cache.TryWrite(cachedSequence);
                }
            }

            isCacheEnabled = isCacheEnabled && cache.TryFlush();
        }
        

        if (isCacheEnabled) 
        {
            try 
            {
                // TODO: add TryRename that does not throw.
                renameOrDie(temp, cacheFilename);
            }
            catch (...) {}
        }
    }).detach();
}

const static size_t s_sequenceSize = sizeof(IndexedSequence);
const static size_t s_numSequencesToBuffer = (g_1MB >> 1) / s_sequenceSize;

/*static*/ shared_ptr<Index> IndexBuilder::TryLoadFromCache(const wstring& cacheFilename, size_t chunkSize)
{
    FileWrapper cache(cacheFilename.c_str(), L"rb");

    if (!cache.IsOpen())
        return nullptr;

    Prefix prefix;
    if (!cache.TryRead(prefix) || prefix.magic != s_magic)
        return nullptr;

    auto index = make_shared<Index>(chunkSize);

    char buffer[s_numSequencesToBuffer * s_sequenceSize];
    for (uint64_t i = 0; i < prefix.totalNumberOfSequences; )
    {
        auto numSequencesToRead = min(s_numSequencesToBuffer, prefix.totalNumberOfSequences - i);

        if (!cache.TryRead(buffer, s_sequenceSize, numSequencesToRead))
            return nullptr;

        for (int j = 0; j < numSequencesToRead; j++) 
            index->AddSequence(*reinterpret_cast<IndexedSequence*>(buffer + j * s_sequenceSize));
        
        i += numSequencesToRead;
    }

    return index;
}

TextInputIndexBuilder::TextInputIndexBuilder(const FileWrapper& input)
    : IndexBuilder(input),
    m_skipSequenceIds(false),
    m_streamPrefix('|'),
    m_mainStream(""),
    m_fileSize(0)
{}

/*virtual*/ wstring TextInputIndexBuilder::GetCacheFilename() /*override*/
{
    // What follows are all the options that affect the outcome of indexing (i.e., specifing a 'main' stream 
    // using the definesMBsize flag will affect the sequence length in terms of number of samples). 
    // We could compute a (SHA1) hash of all these options and add it instead to the filename (+ embed the
    // values themselves into the cache header), but given that there're only a few of them, encoding them
    // into the filename seems like a better option (mainly because it's easy to do that offline in Python).
    wstringstream  wss;
    wss << m_input.Filename() << "."
        << (!m_mainStream.empty() ? wstring(m_mainStream.begin(), m_mainStream.end()) : L"_") << "."
        << (m_skipSequenceIds ? "1" : "0") << "."
        << ((m_corpus && !m_corpus->IsNumericSequenceKeys()) ? "1" : "0") << "."
        << ((m_corpus && m_corpus->IsHashingEnabled()) ? std::to_wstring(CorpusDescriptor::s_hashVersion) : L"0") << "."
        << L"v" << IndexBuilder::s_version << "."
        << L"cache";

    return wss.str();
}

TextInputIndexBuilder::KMP::KMP(const string& value)
    : pattern(value), next(pattern.size())
{
    int j = -1;
    for (int i = 0; i < pattern.size(); i++) 
    {
        if (i == 0)
            next[i] = -1;
        else if (pattern[i] != pattern[j])
            next[i] = j;
        else
            next[i] = next[j];

        while (j >= 0 && pattern[i] != pattern[j])
        {
            j = next[j];
        }
        j++;
    }
}

static const char s_BOM[3] = { '\xEF', '\xBB', '\xBF' };

/*virtual*/ void TextInputIndexBuilder::Populate(shared_ptr<Index>& index) /*override*/
{
    if (!m_mainStream.empty()) 
    {
        m_nfa.reset(new KMP(m_streamPrefix + m_mainStream));
    }

    m_input.CheckIsOpenOrDie();
    
    m_fileSize = m_input.Filesize();

    if (m_fileSize == 0)
        RuntimeError("Input file is empty");

    m_reader.reset(new BufferedFileReader(m_bufferSize, m_input));

    index->Reserve(m_fileSize);

    // skip BOM prefix at the very beginning of the input file if it's there.
    for (char ch : s_BOM) 
    {
        if (!m_reader->Empty() && m_reader->Peek() == ch)
            m_reader->Pop();
        else break;
    }

    if (!isspace(m_streamPrefix))
    {
        // as long as the stream prefix is not a white space, it's safe to skip all leading spaces.
        while (isspace(m_reader->Peek()) && m_reader->Pop()); 
    }

    if (m_reader->Empty())
        RuntimeError("Input file is empty");

    if (m_skipSequenceIds || (!m_reader->Empty() && m_reader->Peek() == m_streamPrefix))
    {
        // Skip sequence id parsing, treat lines as individual sequences
        // In this case the sequences do not have ids, they are assigned corresponding line numbers
        // as ids. Raise an exception if corpus expects symbolic ids.
        if (m_corpus && !m_corpus->IsNumericSequenceKeys())
            RuntimeError("Corpus expects non-numeric sequence keys present but the input file does not have them."
                "Please use the configuration to enable numeric keys instead.");

        PopulateFromLines(index);
    }
    else 
    {
        PopulateImpl(index);
    }
}

void TextInputIndexBuilder::PopulateFromLines(shared_ptr<Index>& index)
{
    IndexedSequence sequence;
    while (!m_reader->Empty())
    {
        size_t offset = m_reader->GetFileOffset();

        if (!FindMainStream())
        { 
            // skip lines that do not contain main stream name.
            m_reader->TryMoveToNextLine();
            continue;
        }

        sequence.SetNumberOfSamples(1).SetOffset(offset).SetKey(m_reader->CurrentLineNumber());

        if (m_reader->TryMoveToNextLine())
        {
            sequence.SetSize(m_reader->GetFileOffset() - offset);
            index->AddSequence(sequence);
        } 
        else  if (offset < m_fileSize)
        {
            // There's a number of characters, not terminated by a newline,
            // add a sequence to the index, parser will have to deal with it.
            sequence.SetSize(m_fileSize - offset);
            index->AddSequence(sequence);
            break;
        }
    }
}

void TextInputIndexBuilder::PopulateImpl(shared_ptr<Index>& index)
{
    IndexedSequence sequence;
    uint32_t numberOfSamples = 0;
    bool foundMainStream = false;
    size_t prevId = 0, nextId = 0, prevOffset = m_reader->GetFileOffset();

    // Go ahead and read the id of the very first sequence.
    if (!TryGetSequenceId(prevId))
    {
        RuntimeError("Expected a sequence id at the offset %zu, none was found.", prevOffset);
    }

    while (!m_reader->Empty())
    {
        if (FindMainStream())
        {
            numberOfSamples++;
            foundMainStream = true;
        }

        m_reader->TryMoveToNextLine(); // ignore whatever is left on this line.

        auto offset = m_reader->GetFileOffset(); // a new line starts at this offset;
        
        if (TryGetSequenceId(nextId) && nextId != prevId)
        {
            // found a new sequence, which starts at the [offset] bytes into the file
            // adding the previous one to the index.
            sequence.SetKey(prevId)
                .SetNumberOfSamples(numberOfSamples)
                .SetOffset(prevOffset)
                .SetSize(offset - prevOffset);

            prevId = nextId;
            prevOffset = offset;
            numberOfSamples = 0;
            
            if (foundMainStream)
                index->AddSequence(sequence);
            foundMainStream = false;
        }
    }

    if (prevOffset < m_fileSize)
    {
        sequence.SetKey(prevId)
            .SetNumberOfSamples(numberOfSamples)
            .SetOffset(prevOffset)
            .SetSize(m_fileSize - prevOffset);
        
        if (foundMainStream)
            index->AddSequence(sequence);
    }
}

inline bool TextInputIndexBuilder::FindMainStream()
{
    if (m_reader->Empty())
        return false;
    
    if (m_mainStream.empty())
        return true;

    auto length = m_nfa->pattern.size();

    int i = 0;
    do  
    {
        char c = m_reader->Peek();
        if (i == length)
        {
            // we found a match, check to see if it's followed by either a space, 
            // a stream prefix or a non-printable character.
            if (isspace(c) || c == m_streamPrefix || c < ' ')
                return true;
            // nope, a false positive (the pattern is contained as a substring in some other stream name), 
            // keep on searching. 
            i = 0;
        }

        while (i >= 0 && c != m_nfa->pattern[i])
            i = m_nfa->next[i];
        i++;

        if (c == g_eol)
            break;
    } while (m_reader->Pop());

    // we hit either the EOL or the EOF, see if we have a match
    return (i == length);
}

inline bool TextInputIndexBuilder::TryGetSequenceId(size_t& id)
{
    if (m_corpus && !m_corpus->IsNumericSequenceKeys())
        return TryGetSymbolicSequenceId(id, m_corpus->KeyToId);

    return TryGetNumericSequenceId(id);
}

inline bool TextInputIndexBuilder::TryGetNumericSequenceId(size_t& id)
{
    if (m_reader->Empty())
        return false;

    bool found = false;
    id = 0;
    do
    {
        char c = m_reader->Peek();
        if (!isdigit(c))
            // Stop as soon as there's a non-digit character
            return found;

        size_t temp = id;
        id = id * 10 + (c - '0');
        if (temp > id)
            RuntimeError("Overflow while reading a numeric sequence id (%zu-bit value).", sizeof(id));
        
        found = true;
    } while (m_reader->Pop());

    // reached EOF without hitting the pipe character,
    // ignore it for now, parser will have to deal with it.
    return false;
}

inline bool TextInputIndexBuilder::TryGetSymbolicSequenceId(size_t& id, function<size_t(const string&)> keyToId)
{
    if (m_reader->Empty())
        return false;

    bool found = false;
    id = 0;
    string key;
    key.reserve(256);
    do
    {
        char c = m_reader->Peek();
        if (isspace(c))
        {
            if (found)
                id = keyToId(key);
            return found;
        }

        key += c;
        found = true;
    } while (m_reader->Pop());

    // reached EOF without hitting the pipe character,
    // ignore it for now, parser will have to deal with it.
    return false;
}

}

