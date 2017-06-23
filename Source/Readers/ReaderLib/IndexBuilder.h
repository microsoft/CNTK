//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include <stdint.h>
#include <vector>
#include "Index.h"
#include "CorpusDescriptor.h"
#include "BufferedFileReader.h"

namespace Microsoft { namespace MSR { namespace CNTK {

class IndexedSequence
{
    size_t key; // Sequence key, uniquely identifies the sequence.
    uint32_t numberOfSamples;
    size_t offset; // offset in file.
    uint32_t size; // size in bytes

    friend class Index;
    friend class ChunkDescriptor;
    
public:
    IndexedSequence& SetKey(size_t value) { key = value; return *this;  }
    
    IndexedSequence& SetNumberOfSamples(uint32_t value) { numberOfSamples = value; return *this; }
    
    IndexedSequence& SetOffset(size_t value) { offset = value; return *this; }
    
    IndexedSequence& SetSize(size_t value)
    {
        size = static_cast<uint32_t>(value);
        if (size != value)
            RuntimeError("Sequence size overflows uint32_t type.");
        return *this;
    }
};


class IndexBuilder 
{
    struct Prefix {
        Prefix() = default;
        Prefix(uint64_t magic, uint64_t version, uint64_t totalNumberOfSequences, uint64_t firstSequenceOffset)
            : magic{ magic }, version{ version },
            totalNumberOfSequences{ totalNumberOfSequences }, firstSequenceOffset{ firstSequenceOffset }
        {}
        uint64_t magic{ 0 };
        uint64_t version{ 0 };
        uint64_t totalNumberOfSequences{ 0 };
        uint64_t firstSequenceOffset{ sizeof(Prefix) }; // this offset is set to the size of prefix for the moment
        // but eventually, this can be used to append additional staff after prefix, without breaking
        // back compat.
    };

public:
    // Reads the input file, building and index of chunks and corresponding
    // sequences. Returns input data index (chunk and sequence metadata);
    shared_ptr<Index> Build();


    IndexBuilder& SetPrimary(bool primary) { m_primary = primary; return *this; }

    IndexBuilder& SetChunkSize(size_t size) { m_chunkSize = size; return *this; }

    IndexBuilder& SetCorpus(CorpusDescriptorPtr corpus) { m_corpus = corpus; return *this; }

    IndexBuilder& SetBufferSize(size_t size) { m_bufferSize = size; return *this; }

protected:
    IndexBuilder(const std::wstring& filename, FILE* file, bool enableCaching = true);

    virtual std::wstring GetCacheFilename() = 0;
    virtual void Populate(shared_ptr<Index>&) {};

    std::wstring m_filename;
    FILE* m_file;
    CorpusDescriptorPtr m_corpus;
    size_t m_bufferSize;
    bool m_primary;
    size_t m_chunkSize;


    static const uint64_t s_version = 1;

private:
    static shared_ptr<Index> TryLoadFromCache(const std::wstring& cacheFilename, size_t chunkSize);
    void WriteIndexCacheAsync(shared_ptr<Index>& index);
    bool m_isCacheEnabled;
    shared_ptr<Index> m_index;

    static const uint64_t s_magic = 0x636e746b5f696478; // 'cntk_idx'
    
    DISABLE_COPY_AND_MOVE(IndexBuilder);
};

// A helper class that does a pass over the input file building up
// an index consisting of sequence and chunk descriptors (which among 
// others specify size and file offset of the respective structure).
// As opposed to the data deserializer, indexer performs almost no parsing 
// and therefore is several magnitudes faster.
class TextInputIndexBuilder : public IndexBuilder
{
public:
    TextInputIndexBuilder(const std::wstring& filename, FILE* file);

    TextInputIndexBuilder& SetSkipSequenceIds(bool skip) { m_skipSequenceIds = skip; return *this; }

    TextInputIndexBuilder& SetMainStream(const std::string& name) { m_mainStream = name; return *this; }

    TextInputIndexBuilder& SetStreamPrefix(char prefix) { m_streamPrefix = prefix; return *this; }

private:
    // Implementation of the Knuth-Morris-Pratt (linear string search without backup)
    // algorithm adopted from http://algs4.cs.princeton.edu/53substring/KMPplus.java.html
    struct KMP
    {
        KMP(const std::string& value);
        std::string pattern;
        std::vector<int> next; // failure funciton table
    };

    virtual std::wstring GetCacheFilename() override;
    virtual void Populate(shared_ptr<Index>& index) override;

    size_t m_fileSize;
    bool m_skipSequenceIds; // true, when input contains one sequence per line 
                           // or when sequence id column was ignored during indexing.
    char m_streamPrefix;

    // Stream that defines the size of the sequence.
    std::string m_mainStream;
    unique_ptr<KMP> m_nfa; 

    unique_ptr<BufferedFileReader> m_reader;

    // Returns true if main stream name if found on the current line.
    bool FindMainStream();

    // Invokes either TryGetNumericSequenceId or TryGetSymbolicSequenceId depending
    // on the specified corpus settings.
    bool TryGetSequenceId(size_t& id);

    // Tries to get numeric sequence id.
    // Throws an exception if a non-numerical is read until the pipe character or 
    // EOF is reached without hitting the pipe character.
    // Returns false if no numerical characters are found preceding the pipe.
    // Otherwise, writes sequence id value to the provided reference, returns true.
    bool TryGetNumericSequenceId(size_t& id);

    // Same as above but for symbolic ids.
    // It reads a symbolic key and converts it to numeric id using provided keyToId function.
    bool TryGetSymbolicSequenceId(size_t& id, std::function<size_t(const std::string&)> keyToId);

    void PopulateImpl(shared_ptr<Index>& index);

    // Parses input line by line, treating each line as an individual sequence.
    // Ignores sequence id information, using the line number instead as the id.
    void PopulateFromLines(shared_ptr<Index>& index);

    DISABLE_COPY_AND_MOVE(TextInputIndexBuilder);
};

}}}
