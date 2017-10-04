//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include <chrono>
#include "stdafx.h"
#include "BufferedFileReader.h"
#include "FileWrapper.h"
#include "Index.h"
#include "Platform.h"
#include "IndexBuilder.h"
#include "ReaderUtil.h"
#include "Common/ReaderTestHelper.h"
#include <boost/algorithm/string/replace.hpp>
#include <boost/algorithm/string.hpp>

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

using namespace ::CNTK;

#define ANY SIZE_MAX

static void CreateTestFile(const std::string& content, const std::wstring filename = L"test.tmp")
{
    _wunlink(filename.c_str());
    auto f = FileWrapper::OpenOrDie(filename, L"w+b");
    f.WriteOrDie(content.c_str(), content.size(), 1);
}

const static std::string s_textData =
    "0\t|a 1 1\t|b 1 1\n"    // 16 characters
    "0\t|a 2 2\t|b 2 2\n"    // 16 
    "0\t|a 3 3\t|b 3 3\n"    // 16
    "0\t|a 4 4\n"            // 9
    "0\t|a 5 5\n"            // 9
    "1\t|b 6 6\t |a 6 6 6\n" // 19
    "1\t|b 7 7\t |a 7 7 7\n" // 19
    "1\t|b 8 8 8 8 8\n"      // 15
    "1\t|b 9\n"              // 7
    "1\t|b 10 10 10 10 10\n";// 20


BOOST_AUTO_TEST_SUITE(BufferedFileReaderTests)

void PeekAndPop(const std::string& content, const std::vector<size_t>& bufferSizes) 
{
    CreateTestFile(content);
    auto testFileSize = content.size();

    for (size_t i : bufferSizes)
    {
        auto f = FileWrapper::OpenOrDie(L"test.tmp", L"rb");
        BOOST_REQUIRE_EQUAL(testFileSize, f.Filesize());

        BufferedFileReader reader(i, f);
        size_t charCount = 0, lineCount = 0;
        for (auto ch : content)
        {
            BOOST_REQUIRE(!reader.Empty());
            BOOST_REQUIRE_EQUAL(lineCount, reader.CurrentLineNumber());
            if (ch == g_eol) lineCount++;
            BOOST_REQUIRE_EQUAL(ch, reader.Peek());
            BOOST_REQUIRE_EQUAL(charCount, reader.GetFileOffset());
            BOOST_REQUIRE_EQUAL(reader.Pop(), ++charCount < testFileSize);
        }
        BOOST_REQUIRE_EQUAL(lineCount, reader.CurrentLineNumber());
        BOOST_REQUIRE_EQUAL(charCount, reader.GetFileOffset());
        BOOST_REQUIRE(reader.Empty());
    }
}

void TryGetNext(const std::string& content, const std::vector<size_t>& bufferSizes)
{
    CreateTestFile(content);
    auto testFileSize = content.size();

    for (size_t i : bufferSizes)
    {
        auto f = FileWrapper::OpenOrDie(L"test.tmp", L"rb");
        BOOST_REQUIRE_EQUAL(testFileSize, f.Filesize());

        BufferedFileReader reader(i, f);
        size_t charCount = 0, lineCount = 0;
        for (auto ch : content)
        {
            BOOST_REQUIRE(!reader.Empty());
            BOOST_REQUIRE_EQUAL(lineCount, reader.CurrentLineNumber());
            if (ch == g_eol) lineCount++;
            char c; 
            BOOST_REQUIRE(reader.TryGetNext(c));
            BOOST_REQUIRE_EQUAL(c, ch);
            BOOST_REQUIRE_EQUAL(++charCount, reader.GetFileOffset());
        }
        BOOST_REQUIRE_EQUAL(lineCount, reader.CurrentLineNumber());
        BOOST_REQUIRE_EQUAL(charCount, reader.GetFileOffset());
        BOOST_REQUIRE(reader.Empty());
    }
}

void SkipLines(const std::string& content, const std::vector<size_t>& bufferSizes)
{
    CreateTestFile(content);
    auto testFileSize = content.size();

    for (size_t i : bufferSizes)
    {
        auto f = FileWrapper::OpenOrDie(L"test.tmp", L"rb");
        BOOST_REQUIRE_EQUAL(testFileSize, f.Filesize());

        BufferedFileReader reader(i, f);
        size_t lineCount = 0;
        for (size_t j = 0; j < content.size() && j != std::string::npos; )
        {
            auto ch = content[j];
            
            BOOST_REQUIRE(!reader.Empty());
            BOOST_REQUIRE_EQUAL(ch, reader.Peek());

            j = content.find("\n", j);

            bool canMoveToNextLine = (j != std::string::npos && j + 1 < content.size());
            BOOST_REQUIRE_EQUAL(canMoveToNextLine, reader.TryMoveToNextLine());

            if (j != std::string::npos) {
                j++;
                lineCount++;
                BOOST_REQUIRE_EQUAL(j, reader.GetFileOffset());
            }
            
            BOOST_REQUIRE_EQUAL(lineCount, reader.CurrentLineNumber());
        }
        BOOST_REQUIRE_EQUAL(lineCount, reader.CurrentLineNumber());
        BOOST_REQUIRE_EQUAL(content.size(), reader.GetFileOffset());
        BOOST_REQUIRE(reader.Empty());
    }
}


void ReadLines(const std::string& content, const std::vector<size_t>& bufferSizes)
{
    CreateTestFile(content);
    auto testFileSize = content.size();

    const static std::vector<bool> delim = DelimiterHash({ '\n' });
    vector<boost::iterator_range<char*>> lines;
    Split(const_cast<char*>(content.data()), const_cast<char*>(content.data()+content.size()), delim, lines);

    for (size_t i : bufferSizes)
    {
        auto f = FileWrapper::OpenOrDie(L"test.tmp", L"rb");
        BOOST_REQUIRE_EQUAL(testFileSize, f.Filesize());

        BufferedFileReader reader(i, f);
        
        BOOST_REQUIRE(!reader.Empty() || content.size() == 0);

        size_t lineCount = 0;
        BOOST_REQUIRE_EQUAL(lineCount, reader.CurrentLineNumber());

        for (string line; reader.TryReadLine(line);)
        {
            if (line != string(lines[lineCount].begin(), lines[lineCount].end()))
                break;
            BOOST_REQUIRE_EQUAL(line, string(lines[lineCount].begin(), lines[lineCount].end()));
            lineCount++;
        }

        BOOST_REQUIRE_EQUAL(reader.CurrentLineNumber() + 1, lines.size());
        
        BOOST_REQUIRE_EQUAL(content.size(), reader.GetFileOffset());
        BOOST_REQUIRE(reader.Empty());
    }
}

BOOST_AUTO_TEST_CASE(Test_peek_and_pop)
{
    for (const auto& str : { "a", "ab", "abc", "abcdefg", "ab cd ef ghi  j",
                             "0\t|a 1 1\t|b 1 1\n1\t|b 10 10 10 10 10" })
        PeekAndPop(str, { 1, 2, 3, 10, 30, 100 });

    for (const auto& str : { "", "\n", "\n\n", "\na\r\n", "ab\rcdef \n123456 \r\n\nA", 
                             "\na\nb\nc defg\n", "\nab cd e\nf \n\n\ngh\ni  j" })
        PeekAndPop(str, { 1, 2, 3, 10, 30, 100});

    PeekAndPop(s_textData, { 1, 2, 3, 7, 19, 33, 71, 139, 144, 145, 146, 147, 150, 300, 1024, g_1MB, g_32MB });
}


BOOST_AUTO_TEST_CASE(Test_get_next_char)
{
    for (const auto& str : { "a", "ab", "abc", "abcdefg", "ab cd ef ghi  j",
        "0\t|a 1 1\t|b 1 1\n1\t|b 10 10 10 10 10" })
        TryGetNext(str, { 1, 2, 3, 10, 30, 100 });

    for (const auto& str : { "", "\n", "\n\n", "\na\r\n", "ab\rcdef \n123456 \r\n\nA",
        "\na\nb\nc defg\n", "\nab cd e\nf \n\n\ngh\ni  j" })
        TryGetNext(str, { 1, 2, 3, 10, 30, 100 });

    PeekAndPop(s_textData, { 1, 2, 3, 7, 19, 33, 71, 139, 144, 145, 146, 147, 150, 300, 1024, g_1MB, g_32MB });
}

BOOST_AUTO_TEST_CASE(Test_line_skipping)
{
    for (const auto& str : { "a","a\n", "\na", "\na\n", "a\nb", "\na\nb", "\n\nabc\n\n", "a\n\nb\nc\nd\ne\nf\ng",
                             "0\t|a 1 1\t|b 1 1\n1\t|b 10 10 10 10 10" })
        SkipLines(str,  { 1, 2, 3, 4, 5, 10, 30, 100 });

    for (const auto& str : { "", "\n", "\n\n", "\n\n\n", "\t\r\n\r",  "\na\r\n", "ab\rcdef \n123456 \r\n\nA",
        "\na\nb\nc defg\n", "\nab cd e\nf \n\n\ngh\ni  j" })
        SkipLines(str, { 1, 2, 3, 10, 20 });

    SkipLines(s_textData, { 1, 2, 3, 7, 19, 33, 71, 139, 144, 145, 146, 147, 150, 300, 1024, g_1MB, g_32MB });
}

BOOST_AUTO_TEST_CASE(Test_line_reading)
{
    for (const auto& str : { "a","a\n", "\na", "\na\n", "a\nb", "\na\nb", "\n\nabc\n\n", "a\n\nb\nc\nd\ne\nf\ng",
        "0\t|a 1 1\t|b 1 1\n1\t|b 10 10 10 10 10" })
        ReadLines(str, { 1, 2, 3, 4, 5, 10, 30, 100 });

    for (const auto& str : { "", "\n", "\n\n", "\n\n\n", "\t\r\n\r",  "\na\r\n", "ab\rcdef \n123456 \r\n\nA",
        "\na\nb\nc defg\n", "\nab cd e\nf \n\n\ngh\ni  j" })
        ReadLines(str, { 1, 2, 3, 10, 20 });

    ReadLines(s_textData, { 1, 2, 3, 7, 19, 33, 71, 139, 144, 145, 146, 147, 150, 300, 1024, g_1MB, g_32MB });
}

BOOST_AUTO_TEST_CASE(Test_set_offset_after_reading_all)
{
    Sleep(5000);
    CreateTestFile(s_textData);

    for (size_t i : {1, 2, 3, 7, 19, 33, 71, 139, 144, 145, 146, 147, 150, 300})
    {
        auto f = FileWrapper::OpenOrDie(L"test.tmp", L"rb");

        BufferedFileReader reader(i, f);
        for (auto ch : s_textData)
        {
            BOOST_REQUIRE(!reader.Empty());
            char c;
            BOOST_REQUIRE(reader.TryGetNext(c));
            BOOST_REQUIRE_EQUAL(c, ch);
        }
        BOOST_REQUIRE(reader.Empty());


        // Reading again from the beginning
        reader.SetFileOffset(0);
        char c;
        BOOST_REQUIRE(reader.TryGetNext(c));
        BOOST_REQUIRE(!reader.Empty());

        // Reading again the last character in the buffer
        if (i < s_textData.size()) {
            reader.SetFileOffset(i - 1);
            BOOST_REQUIRE(reader.TryGetNext(c));
            BOOST_REQUIRE(!reader.Empty());
        }

        // Reading the last character in file and buffer
        if (i == s_textData.size()) {
            reader.SetFileOffset(i - 1);
            BOOST_REQUIRE(reader.TryGetNext(c));
            BOOST_REQUIRE(reader.Empty());
        }
    }
}

BOOST_AUTO_TEST_SUITE_END()


struct IndexBuilderTestsFixture
{
    IndexBuilderTestsFixture()
    {
        RemoveIndexCacheFiles();
    }
};

BOOST_FIXTURE_TEST_SUITE(TextInputIndexBuilderTests, IndexBuilderTestsFixture)

using IndexBuilderPtr = unique_ptr<TextInputIndexBuilder, std::function<void(TextInputIndexBuilder*)>>;

static IndexBuilderPtr GetIndexBuilder(const std::string& input)
{
    static size_t id = 0;
    std::wstring filename = std::to_wstring(id++) +  L".test.tmp";
    CreateTestFile(input, filename);
   
    auto f = FileWrapper::OpenOrDie(filename, L"rb");
   
    auto testFileSize = input.size();
    BOOST_REQUIRE_EQUAL(testFileSize, f.Filesize());

    return unique_ptr<TextInputIndexBuilder, std::function<void(TextInputIndexBuilder*)>>(new TextInputIndexBuilder(f),
        [filename](TextInputIndexBuilder* builder)
    {
        delete builder;
        _wunlink(filename.c_str());
    });
}

static void Check(const shared_ptr<Index>& index, size_t numChunks = ANY, size_t numSeqs = ANY, size_t numSamples = ANY, size_t byteSize = ANY)
{
    BOOST_REQUIRE(index);
    BOOST_REQUIRE(!index->IsEmpty());
    if (numChunks != ANY)
        BOOST_REQUIRE_EQUAL(index->NumberOfChunks(), numChunks);
    if (numSeqs != ANY)
        BOOST_REQUIRE_EQUAL(index->NumberOfSequences(), numSeqs);
    if (numSamples != ANY)
        BOOST_REQUIRE_EQUAL(index->NumberOfSamples(), numSamples);
    if (byteSize != ANY)
        BOOST_REQUIRE_EQUAL(index->SizeInBytes(), byteSize);
}

static void Check(const ChunkDescriptor& chunk, size_t numSeqs = ANY, size_t numSamples = ANY, size_t offset = ANY, size_t byteSize = ANY)
{
    if (numSeqs != ANY)
        BOOST_REQUIRE_EQUAL(chunk.NumberOfSequences(), numSeqs);
    if (numSamples != ANY)
        BOOST_REQUIRE_EQUAL(chunk.NumberOfSamples(), numSamples);
    if (offset != ANY)
        BOOST_REQUIRE_EQUAL(chunk.StartOffset(), offset);
    if (byteSize != ANY)
        BOOST_REQUIRE_EQUAL(chunk.SizeInBytes(), byteSize);
}

static void Check(const SequenceDescriptor& sequence, size_t key = ANY, size_t numSamples = ANY, size_t offsetInChunk = ANY, size_t byteSize = ANY)
{
    if (key != ANY)
        BOOST_REQUIRE_EQUAL(sequence.m_key, key);
    if (numSamples != ANY)
        BOOST_REQUIRE_EQUAL(sequence.NumberOfSamples(), numSamples);
    if (offsetInChunk != ANY)
        BOOST_REQUIRE_EQUAL(sequence.OffsetInChunk(), offsetInChunk);
    if (byteSize != ANY)
        BOOST_REQUIRE_EQUAL(sequence.SizeInBytes(), byteSize);
}

static void CheckIdentical(const shared_ptr<Index>& index1, const shared_ptr<Index>& index2)
{
    BOOST_REQUIRE(index1);
    BOOST_REQUIRE(index2);
    Check(index1, index2->NumberOfChunks(), index2->NumberOfSequences(), index2->NumberOfSamples(), index2->SizeInBytes());
    for (int i = 0; i < index1->NumberOfChunks(); i++)
    {
        auto& chunk1 = (*index1)[i];
        auto& chunk2 = (*index2)[i];
        Check(chunk1, chunk2.NumberOfSequences(), chunk2.NumberOfSamples(), chunk2.StartOffset(), chunk2.SizeInBytes());
        for (int j = 0; j < chunk1.NumberOfSequences(); j++)
        {
            auto& seq1 = chunk1[i];
            auto& seq2 = chunk2[i];
            Check(seq1, seq2.m_key, seq2.NumberOfSamples(), seq2.OffsetInChunk(), seq2.SizeInBytes());
        }
    }
}


BOOST_AUTO_TEST_CASE(Index_exceptions)
{
    BOOST_CHECK_EXCEPTION(
        GetIndexBuilder("")->Build(),
        std::exception,
        [](const std::exception& e) {
        return e.what() == std::string("Input file is empty");
    });

    BOOST_CHECK_EXCEPTION(
        GetIndexBuilder("a")->Build(),
        std::exception,
        [](const std::exception& e) {
        return (string(e.what()).find("Expected a sequence id") != string::npos);
    });

    BOOST_CHECK_EXCEPTION(
        GetIndexBuilder("123")->Build(), // sequence id is not followed by non-digit separator
        std::exception,
        [](const std::exception& e) {
        return (string(e.what()).find("Expected a sequence id") != string::npos);
    });

    BOOST_CHECK_EXCEPTION(
        GetIndexBuilder("18446744073709551616 ")->Build(), // == SIZE_MAX + 1 overflows to 0
        std::exception,
        [](const std::exception& e) {
        return (string(e.what()).find("Overflow") != string::npos);
    });
}

BOOST_AUTO_TEST_CASE(Index_single_line_input)
{
    for (const string& str : { "|", "a", "|a", "1 a", "abc", "abc |d#ef ", "123 |a 567 |b 890", "|a|b|c|d|e 12345679"})
    {
        auto size = str.size();
        auto index = GetIndexBuilder(str)->SetSkipSequenceIds(true).Build();
        Check(index, 1, 1, 1, size);
        Check((*index)[0], 1, 1, 0, size);
        Check((*index)[0][0], 0, 1, 0, size);
    }

    // input begins with a name prefix ('|'), no need to call SetSkipSequenceIds(true), index builder
    // should no try to parse sequence ids.
    for (const string& str : { "|", "|a", "|1 a", "|abc", "|abc |d#ef ", "|123 |a 567 |b 890", "|a|b|c|d|e 12345679" })
    {
        auto size = str.size();
        auto index = GetIndexBuilder(str)->Build();
        Check(index, 1, 1, 1, size);
        Check((*index)[0], 1, 1, 0, size);
        Check((*index)[0][0], 0, 1, 0, size);
    }

    // input begins with a valid (numeric) sequence id followed by a non-digit character
    for (const auto& p : 
    { 
        pair<size_t, string>{1, "1\t"},
        pair<size_t, string>{2, "2 |"},
        pair<size_t, string>{3, "3 |a"},
        pair<size_t, string>{45, "45 |abc"},
        pair<size_t, string>{678, "678 |abc |d#ef "},
        pair<size_t, string>{9, "009 |123 |a 567 |b 890"},
        pair<size_t, string>{99, "0000099|0000099"},
        pair<size_t, string>{SIZE_MAX, "18446744073709551615            "}
    })
    {
        auto key = p.first;
        const auto& str = p.second;
        auto size = str.size();
        auto index = GetIndexBuilder(str)->Build();
        Check(index, 1, 1, 1, size);
        Check((*index)[0], 1, 1, 0, size);
        Check((*index)[0][0], key, 1, 0, size);
    }

    string bom{ '\xEF', '\xBB', '\xBF' };

    // test input that begins with a utf-8 BOM prefix
    for (const string& str : {  "a", " |", " |a", "123 |a" })
    {
        auto offset = 3;
        auto size = boost::trim_copy(str).size();
        auto content = bom + str;
        auto index = GetIndexBuilder(content)->SetSkipSequenceIds(true).Build();
        Check(index, 1, 1, 1, size);
        Check((*index)[0], 1, 1, offset + str.size() - size, size);
        Check((*index)[0][0], 0, 1, 0, size);
    }    

    for (const string& str : { "123 a", "123 |", "123 |a", "123|123" })
    {
        auto offset = 3;
        auto size = str.size();
        auto content = bom + str;
        auto index = GetIndexBuilder(content)->Build();
        Check(index, 1, 1, 1, size);
        Check((*index)[0], 1, 1, offset, size);
        Check((*index)[0][0], 123, 1, 0, size);
    }
}

BOOST_AUTO_TEST_CASE(Index_with_different_buffer_sizes)
{
    auto size = s_textData.size();
    for (const auto& bufferSize : {1,2,5,11,27, 79, 143, 300, 1024})
    {
        auto index = GetIndexBuilder(s_textData)->SetBufferSize(bufferSize).Build();
        Check(index, 1, 2, 10, size);
        Check((*index)[0], 2, 10, 0, size);
        Check((*index)[0][0], 0, 5, 0, 66);
        Check((*index)[0][1], 1, 5, 66, size - 66);
    }
}

BOOST_AUTO_TEST_CASE(Index_with_different_chunk_sizes)
{
    auto size = s_textData.size();
    for (const auto& chunkSize : vector<size_t>{ 1, 11, 27, 65, 66, 67, 80, 145, 146, 147, 300, 1024, g_1MB })
    {
        auto index = GetIndexBuilder(s_textData)->SetChunkSize(chunkSize).Build();

        auto numChunks = chunkSize < size ? 2 : 1;

        Check(index, numChunks, 2, 10, size);

        if (numChunks == 1) 
        {
            Check((*index)[0], 2, 10, 0, size);
            Check((*index)[0][0], 0, 5, 0, 66);
            Check((*index)[0][1], 1, 5, 66, size - 66);
        }
        else 
        {
            Check((*index)[0], 1, 5, 0, 66);
            Check((*index)[0][0], 0, 5, 0, 66);

            Check((*index)[1], 1, 5, 66, size - 66);
            Check((*index)[1][0], 1, 5, 0, size - 66);
        }
    }
}


BOOST_AUTO_TEST_CASE(Index_with_non_empty_main_stream_1)
{
    // this input does not contain a proper main stream name ('|a')
    for (const string& str : { "|b", "1|b", "|ab", "1|ab" "1 | a b |c", "1|abc", "1 |ba", "ab c", "a ", "#a| " })
    {
        auto index = GetIndexBuilder(str)->SetSkipSequenceIds(true).SetMainStream("a").Build();
        BOOST_REQUIRE(index);
        BOOST_REQUIRE(index->IsEmpty());
    }
    for (const string& str : { "1 |b", "12|b", "123 |ab", "123456|ab" "789 | a b |c", "11111|abc", "3431 |ba", "4ab c", "|99a ", "134#a| ", "|bbbb|" })
    {
        auto index = GetIndexBuilder(str)->SetMainStream("a").Build();
        BOOST_REQUIRE(index);
        BOOST_REQUIRE(index->IsEmpty());
    }

    for (const string& str : { "|a |b", "1|a|b", "1 |a |b |c", "1|a|bc", "|a 123", "1 |a|", "|bc|a ", "abc |a    d#ef ", "123 |a 567 |b 890", "|a|b|c|d|e 12345679" })
    {
        auto size = str.size();
        auto index = GetIndexBuilder(str)->SetSkipSequenceIds(true).SetMainStream("a").Build();
        Check(index, 1, 1, 1, size);
        Check((*index)[0], 1, 1, 0, size);
        Check((*index)[0][0], 0, 1, 0, size);
    }

    size_t sequenceId = 0;
    for (const string& str : { "|a |b", "1|a|b", "2 |a |b |c", "3|a|bc", "4|a 123", "5 |a|", "6|bc|a ", "7|abc |a    d#ef ", "8 |a 567 |b 890", "9|a|b|c|d|e 12345679" })
    {
        auto size = str.size();
        auto index = GetIndexBuilder(str)->SetMainStream("a").Build();
        Check(index, 1, 1, 1, size);
        Check((*index)[0], 1, 1, 0, size);
        Check((*index)[0][0], sequenceId++, 1, 0, size);
    }
    

    sequenceId = 0;
    for (const string& str : { " abc |abc ", "1|a abc\t", "2 a b c ab bc a|bc abc ", "3|abc| abc de|abc abc |abc"})
    {
        auto size = str.size();
        auto index = GetIndexBuilder(str)->SetStreamPrefix(' ').SetMainStream("abc").Build();
        Check(index, 1, 1, 1, size);
        Check((*index)[0], 1, 1, 0, size);
        Check((*index)[0][0], sequenceId++, 1, 0, size);
    }

    sequenceId = 0;
    for (const string& str : { " abc | abc ", "1 | a abc\t", "2 a b c ab bc a | bc abc ", "3|abc | abc de|abc abc |abc" })
    {
        auto size = str.size();
        auto index = GetIndexBuilder(str)->SetStreamPrefix(' ').SetMainStream("|").Build();
        Check(index, 1, 1, 1, size);
        Check((*index)[0], 1, 1, 0, size);
        Check((*index)[0][0], sequenceId++, 1, 0, size);
    }
    
    auto size = s_textData.size();
    auto index = GetIndexBuilder(s_textData)->SetMainStream("a").Build();
    Check(index, 1, 2, 7, size);
    Check((*index)[0], 2, 7, 0, size);
    Check((*index)[0][0], 0, 5, 0, 66);
    Check((*index)[0][1], 1, 2, 66, size - 66);

    index = GetIndexBuilder(s_textData)->SetMainStream("b").Build();
    Check(index, 1, 2, 8, size);
    Check((*index)[0], 2, 8, 0, size);
    Check((*index)[0][0], 0, 3, 0, 66);
    Check((*index)[0][1], 1, 5, 66, size - 66);

    string input = boost::replace_all_copy(s_textData, "|", "@| ");
    size = input.size();
    auto firstSeqSize = input.find("\n1\t") + 1;
    index = GetIndexBuilder(input)->SetStreamPrefix('@').SetMainStream("|").Build();
    Check(index, 1, 2, 10, size);
    Check((*index)[0], 2, 10, 0, size);
    Check((*index)[0][0], 0, 5, 0, firstSeqSize);
    Check((*index)[0][1], 1, 5, firstSeqSize, size - firstSeqSize);
}

BOOST_AUTO_TEST_CASE(Index_with_non_empty_main_stream_2)
{
    for (const string input : {"1 |features 1.0\r\n", "1 |features ", "1 |features\n", "1 |features"}) 
    {
        size_t size = input.size();
        auto index = GetIndexBuilder(input)->SetMainStream("features").Build();
        Check(index, 1, 1, 1, size);
        Check((*index)[0], 1, 1, 0, size);
        Check((*index)[0][0], 1, 1, 0, size);
    }

    for (const string input : 
    {
        "1 |features 1.0\r\n"
        "\r\n"
        "2 |features 2.0\r\n",

        "1 |features 1.0\r\n"
        "\r\n \r\n   |\r\n"
        "2 |features\n",

        "1 |features 1.0\r\n"
        "2 |features",

        "1 |features 1.0\n"
        "|\n               \n"
        "2 |features|z"
    })
    {
        size_t size = input.size();
        auto firstSeqSize = input.find("2 |");
        auto index = GetIndexBuilder(input)->SetMainStream("features").Build();
        Check(index, 1, 2, 2, size);
        Check((*index)[0], 2, 2, 0, size);
        Check((*index)[0][0], 1, 1, 0, firstSeqSize);
        Check((*index)[0][1], 2, 1, firstSeqSize, size - firstSeqSize);
    }
}


BOOST_AUTO_TEST_CASE(Index_with_non_numeric_sequence_ids)
{
    auto index = GetIndexBuilder(s_textData)->SetCorpus(std::make_shared<CorpusDescriptor>(true, false)).Build();
    auto size = s_textData.size();
    Check(index, 1, 2, 10, size);
    Check((*index)[0], 2, 10, 0, size);
    Check((*index)[0][0], 0, 5, 0, 66);
    Check((*index)[0][1], 1, 5, 66, size - 66);

    string input = boost::replace_all_copy(s_textData, "0\t|", "abcdefghijklmnopqrstuvwxy1\t|");
    boost::replace_all(input, "1\t|", "abcdefghijklmnopqrstuvwxy2\t|");
    size = input.size();
    auto firstSeqSize = input.find("\nabcdefghijklmnopqrstuvwxy2\t")+1;
    index = GetIndexBuilder(input)->SetCorpus(std::make_shared<CorpusDescriptor>(false, false)).Build();
    Check(index, 1, 2, 10, size);
    Check((*index)[0], 2, 10, 0, size);
    Check((*index)[0][0], 0, 5, 0, firstSeqSize);
    Check((*index)[0][1], 1, 5, firstSeqSize, size - firstSeqSize);

    index = GetIndexBuilder(input)->SetCorpus(std::make_shared<CorpusDescriptor>(false, true)).Build();
    Check(index, 1, 2, 10, size);
    Check((*index)[0], 2, 10, 0, size);

    Check((*index)[0][0], ANY, 5, 0, firstSeqSize);
    Check((*index)[0][1], ANY, 5, firstSeqSize, size - firstSeqSize);
}

BOOST_AUTO_TEST_CASE(Index_with_multi_line_sequences)
{
    size_t count = 0;
    for (const string& str : { "1\n", "1\n2 ", "1\n2\n3|abc", "1 \n2 |a\n3\n4     ", "1|2\n2       \n3|a\n4\t\t\n5\t\tzzzz" })
    {
        auto size = str.size();
        auto index = GetIndexBuilder(str)->Build();
        ++count;
        Check(index, 1, count, count, size);
        Check((*index)[0], count, count, 0, size);
    }

    count = 0;
    for (const string& str : { "1\n", "1\n\n2\n2 ", "1\n\n\n2\n \n \n3|abc\n|abc\n3 |abc", 
                               "1\n1\n1\n1\n1 \n2\n\n\n\n3\n\n3\n3\n4 abc\n4 def\nghj\n4"})
    {
        auto size = str.size();
        auto index = GetIndexBuilder(str)->Build();
        ++count;
        Check(index, 1, count, count*count, size);
        Check((*index)[0], count, count*count, 0, size);
    }

    count = 0;
    for (const string& str : 
    { 
        "1|x\n" // 1 sequence, 2 samples
        "|x ", 

        "1\n"  // 2 sequences, 4 samples
        " |x   \n"
        "2|x\n"
        "2\t\t|x\t\n"
        "|x\n", 

        "1|x\n"  // 3 sequences, 6 samples
        "    |x\n"
        "1   |x\n"
        "0 |y  \n" // this sequence will be ignored as it does not contain the main stream name
        "0 |y   \n" 
        "0  \n"
        "2  |y\n"
        "2  |x\n"
        "2  |y\n"
        "3 |x    \n"
        "|y      |x",

        "0\n\n|x\n"   // 4 sequences, 8 samples
        "1\n1\n\n1\n1\n" // will be ignored
        "2|x\n   |x\n  |x\n   |x\n"
        "3\n\n3\n3|x\n"
        "4|x|x |x\n4|xxxxx\n|x|x\n4" 
    })
    {
        auto size = str.size();
        auto index = GetIndexBuilder(str)->SetMainStream("x").Build();
        ++count;

        Check(index, 1, count, 2*count, ANY);
        Check((*index)[0], count, 2*count, 0, size);
    }
}

BOOST_AUTO_TEST_CASE(Index_non_primary)
{
    auto size = s_textData.size();
    auto index = GetIndexBuilder(s_textData)->SetPrimary(true).Build();
    
    BOOST_REQUIRE(!std::get<0>(index->GetSequenceByKey(0)));
    BOOST_REQUIRE(!std::get<0>(index->GetSequenceByKey(1)));

    index = GetIndexBuilder(s_textData)->SetPrimary(false).Build();

    Check(index, 1, 2, 10, size);
    Check((*index)[0], 2, 10, 0, size);
    Check((*index)[0][0], 0, 5, 0, 66);
    Check((*index)[0][1], 1, 5, 66, size - 66);
    
    auto sequence0 = index->GetSequenceByKey(0);
    BOOST_REQUIRE(std::get<0>(sequence0));
    BOOST_REQUIRE_EQUAL(std::get<1>(sequence0), 0u);
    BOOST_REQUIRE_EQUAL(std::get<2>(sequence0), 0u);

    auto sequence1 = index->GetSequenceByKey(1);
    BOOST_REQUIRE(std::get<0>(sequence1));
    BOOST_REQUIRE_EQUAL(std::get<1>(sequence1), 0u);
    BOOST_REQUIRE_EQUAL(std::get<2>(sequence1), 1u);
}


BOOST_AUTO_TEST_CASE(Index_with_caching)
{
    auto filename = L"test.tmp";
    CreateTestFile(s_textData, filename);
    shared_ptr<Index> index;
    {
        auto f1 = FileWrapper::OpenOrDie(filename, L"rb");
        index = TextInputIndexBuilder(f1).SetCachingEnabled(true).Build();
    }
    // Cache is written out asynchronously in a separate thread, 
    Sleep(1000);  // sleep for a second to give enough time to finish writing.

    FILE* dummy = nullptr;
    TextInputIndexBuilder indexBuilder(FileWrapper(filename, dummy));
        
    auto cachedIndex = indexBuilder.SetCachingEnabled(true).Build();

    _wunlink(filename);
    _wunlink(indexBuilder.GetCacheFilename().c_str());

    CheckIdentical(index, cachedIndex);
}

BOOST_AUTO_TEST_CASE(Index_64MB_with_caching_check_perf)
{
    auto content = s_textData;
    while (content.size() < 64*g_1MB)
    {
        content += content;
    }

    shared_ptr<Index> index, cachedIndex;
    size_t timeToBuildIndexFromFile = 0, timeToBuildIndexFromCache = 0;

    for (int i = 0; i < 3; i++) 
    {
        wstring cacheFilename, filename = std::to_wstring(i) + L".perf.test.tmp";

        CreateTestFile(content, filename);
        {
            auto f1 = FileWrapper::OpenOrDie(filename, L"rb");
            TextInputIndexBuilder indexBuilder(f1);
            indexBuilder.SetCachingEnabled(true).SetChunkSize(g_1MB);
            DWORD start = GetTickCount();
            index = indexBuilder.Build();
            DWORD end = GetTickCount();
            timeToBuildIndexFromFile += (end - start);
        }

        // Cache is written out asynchronously in a separate thread, 
        Sleep(3000);  // sleep for a few seconds to give enough time to finish writing.
        
        {
            FILE* dummy = nullptr;
            TextInputIndexBuilder indexBuilder(FileWrapper(filename, dummy));
            indexBuilder.SetCachingEnabled(true).SetChunkSize(g_1MB);
            cacheFilename = indexBuilder.GetCacheFilename();
            DWORD start = GetTickCount();
            cachedIndex = indexBuilder.Build();
            DWORD end = GetTickCount();

            timeToBuildIndexFromCache += (end - start);
        }

        _wunlink(filename.c_str());
        _wunlink(cacheFilename.c_str());

        CheckIdentical(index, cachedIndex);

        index.reset();
        cachedIndex.reset();
    }
    BOOST_REQUIRE(timeToBuildIndexFromCache < timeToBuildIndexFromFile);
}

BOOST_AUTO_TEST_CASE(Index_1GB_with_caching_check_perf)
{
    if (true)
        // This test is intended to be executed manually and was added only
        // as a reference point to expected indexing timing (with and without caching).
        return;
    
    auto content = s_textData;
    while (content.size() < g_4GB >> 2)
    {
        content += content;
    }

    size_t timeToBuildIndexFromFile = 0, timeToBuildIndexFromCache = 0;

    wstring cacheFilename, filename = L"1gb.perf.test.tmp";

    CreateTestFile(content, filename);
    {
        auto f1 = FileWrapper::OpenOrDie(filename, L"rb");
        TextInputIndexBuilder indexBuilder(f1);
        indexBuilder.SetCachingEnabled(true).SetChunkSize(g_1MB);
        DWORD start = GetTickCount();
        auto index = indexBuilder.Build();
        DWORD end = GetTickCount();
        timeToBuildIndexFromFile += (end - start);
    }

    // Cache is written out asynchronously in a separate thread, 
    Sleep(5000);  // sleep for a few seconds to give enough time to finish writing.

    {
        FILE* dummy = nullptr;
        TextInputIndexBuilder indexBuilder(FileWrapper(filename, dummy));
        indexBuilder.SetCachingEnabled(true).SetChunkSize(g_1MB);
        cacheFilename = indexBuilder.GetCacheFilename();
        DWORD start = GetTickCount();
        auto cachedIndex = indexBuilder.Build();
        DWORD end = GetTickCount();

        timeToBuildIndexFromCache += (end - start);
    }
    
    _wunlink(filename.c_str());
    _wunlink(cacheFilename.c_str());

    BOOST_REQUIRE(2*timeToBuildIndexFromCache < timeToBuildIndexFromFile);
    BOOST_REQUIRE_EQUAL(timeToBuildIndexFromCache, timeToBuildIndexFromFile);
}

BOOST_AUTO_TEST_SUITE_END()

} } } }
