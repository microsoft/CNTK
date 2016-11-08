//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#include "stdafx.h"
#include <algorithm>
#ifdef _WIN32
#include <io.h>
#else // On Linux
#define _dup2 dup2
#define _dup dup
#define _close close
#define _fileno fileno
#endif
#include <cstdio>
#include <boost/scope_exit.hpp>
#include "Common/ReaderTestHelper.h"
#include "TextParser.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK {

// A thin wrapper around CNTK text format reader
template <class ElemType>
class CNTKTextFormatReaderTestRunner
{
    TextParser<ElemType> m_parser;

public:
    ChunkPtr m_chunk;

    CNTKTextFormatReaderTestRunner(const string& filename,
        const vector<StreamDescriptor>& streams, unsigned int maxErrors) :
        m_parser(std::make_shared<CorpusDescriptor>(), wstring(filename.begin(), filename.end()), streams)
    {
        m_parser.SetMaxAllowedErrors(maxErrors);
        m_parser.SetTraceLevel(TextParser<ElemType>::TraceLevel::Info);
        m_parser.SetChunkSize(SIZE_MAX);
        m_parser.SetNumRetries(0);
        m_parser.Initialize();
    }
    // Retrieves a chunk of data.
    void LoadChunk()
    {
        m_chunk = m_parser.GetChunk(0);
    }
};

namespace Test {

// identical to 'sort -o filename filename'
void SortLinesInFile(string filename, size_t expectedNumLines = 1)
{
    vector<string> content;
    content.reserve(expectedNumLines);
    ifstream ifstream(filename);
    
    string line;
    while (getline(ifstream, line))
    {
        content.push_back(line);
    }

    ifstream.close();

    sort(content.begin(), content.end());

    ofstream ofstream(filename);

    copy(content.begin(), content.end(), ostream_iterator<string>(ofstream, "\n"));

    ofstream.close();
}


struct CNTKTextFormatReaderFixture : ReaderFixture
{
    CNTKTextFormatReaderFixture()
        : ReaderFixture("/Data/CNTKTextFormatReader/")
    {
    }
};

BOOST_FIXTURE_TEST_SUITE(ReaderTestSuite, CNTKTextFormatReaderFixture)

BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_Simple_dense)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/Simple_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/Simple_dense_Output.txt",
        "Simple",
        "reader",
        1000, // epoch size
        250,  // mb size
        10,   // num epochs 
        1,
        1,
        0,
        1);
};


BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_MNIST_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/MNIST_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/MNIST_dense_Output.txt",
        "MNIST",
        "reader",
        1000, // epoch size
        1000,  // mb size
        1,   // num epochs
        1,
        1,
        0,
        1);
};

// 1 single sample sequence
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_1x1_1_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/1x1_1_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/1x1_1_dense_Output.txt",
        "1x1",
        "reader",
        1, // epoch size
        1,  // mb size
        1,  // num epochs
        1,
        0, // no labels
        0,
        1);
};

// 1 sequence with 2 samples
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_1x1_2_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/1x1_2_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/1x1_2_dense_Output.txt",
        "1x1",
        "reader",
        2, // epoch size
        1,  // mb size
        3,  // num epochs
        1,
        0, // no labels
        0,
        1);
};

// 1 sequence with 10 samples
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_1x10_dense)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/1x10_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/1x10_dense_Output.txt",
        "1x10",
        "reader",
        10, // epoch size
        10,  // mb size
        1,  // num epochs
        1,
        0, // no labels
        0,
        1);
};

// 10 identical single sample sequences
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_10x1_MI_2_dense)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/10x1_MI_2_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/10x1_MI_2_dense_Output.txt",
        "10x1_MI",
        "reader",
        7, // epoch size
        3, // mb size
        3, // num epochs
        4, // num feature inputs
        3, // num label inputs
        0,
        1);
};

// 10 identical single sample sequences
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_1x10_MI_1_dense)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/10x1_MI_1_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/10x1_MI_1_dense_Output.txt",
        "10x1_MI",
        "reader",
        10, // epoch size
        1, // mb size
        3, // num epochs
        4, // num feature inputs
        3, // num label inputs
        0,
        1);
};

// 10 sequences with 10 samples each (no randomization)
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_10x10_dense)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/10x10_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/10x10_dense_Output.txt",
        "10x10",
        "reader",
        100, // epoch size
        100,  // mb size
        1,  // num epochs
        1,
        0, // no labels
        0,
        1);
};

// 100 identical single sample sequences 
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_100x1_1_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/100x1_1_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/100x1_1_dense_Output.txt",
        "100x1",
        "reader",
        10, // epoch size
        1,  // mb size
        10,  // num epochs
        1,
        1,
        0,
        1);
};

// 100 identical single sample sequences
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_100x1_2_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/100x1_2_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/100x1_2_dense_Output.txt",
        "100x1",
        "reader",
        5,  // epoch size
        3,  // mb size
        4,  // num epochs
        1,
        1,
        0,
        1);
};

// 50 sequences with up to 20 samples each (508 samples in total)
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_50x20_jagged_sequences_dense)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/50x20_jagged_sequences_dense.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/50x20_jagged_sequences_dense_Output.txt",
        "50x20_jagged_sequences",
        "reader",
        508,  // epoch size
        508,  // mb size 
        1,  // num epochs
        1,
        0,
        0,
        1);
};

// 1 single sample sequence
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_1x1_sparse)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/sparse.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/1x1_sparse.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/1x1_sparse_Output.txt",
        "1x1",
        "reader",
        1, // epoch size
        1, // mb size
        1, // num epochs
        1,
        0, // no labels
        0,
        1,
        true);
};

// 1 sequence with 2 samples
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_1x2_sparse)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/sparse.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/1x2_sparse.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/1x2_sparse_Output.txt",
        "1x2",
        "reader",
        2, // epoch size
        2, // mb size
        1, // num epochs
        1,
        0, // no labels
        0,
        1,
        true);
};

// 1 sequence with 10 samples
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_1x10_sparse)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/sparse.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/1x10_sparse.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/1x10_sparse_Output.txt",
        "1x10",
        "reader",
        10, // epoch size
        10, // mb size
        1, // num epochs
        1,
        0, // no labels
        0,
        1,
        true);
};


// 10 sequences with 10 samples each (no randomization)
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_10x10_sparse)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/sparse.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/10x10_sparse.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/10x10_sparse_Output.txt",
        "10x10",
        "reader",
        100, // epoch size
        100, // mb size
        1, // num epochs
        1,
        0, // no labels
        0,
        1,
        true);
};

// 3 sequences with 5 samples for each of 3 input stream (no randomization)
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_3x5_MI_sparse)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/sparse.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/3x5_MI_sparse.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/3x5_MI_sparse_Output.txt",
        "3x5_MI",
        "reader",
        15, // epoch size
        15, // mb size
        1, // num epochs
        3,
        0, // no labels
        0,
        1,
        true);
};

// 20 sequences with 10 samples for each of 3 input stream with
// random number of values in each sample (no randomization)
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_20x10_MI_jagged_samples_sparse)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/sparse.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/20x10_MI_jagged_samples_sparse.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/20x10_MI_jagged_samples_sparse_Output.txt",
        "20x10_MI_jagged_samples",
        "reader",
        200, // epoch size
        200, // mb size
        1, // num epochs
        3,
        0, // no labels
        0,
        1,
        true);
};

// 50 sequences with up to 20 samples each (536 samples in total)
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_50x20_jagged_sequences_sparse)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/sparse.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/50x20_jagged_sequences_sparse.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/50x20_jagged_sequences_sparse_Output.txt",
        "50x20_jagged_sequences",
        "reader",
        564,  // epoch size
        564,  // mb size 
        1,  // num epochs
        1,
        0,
        0,
        1,
        true);
};

// 50 sequences with up to 20 samples each and up to 20 values in each sample 
// (4887 samples in total)
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_100x100_jagged_sparse)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/sparse.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/100x100_jagged_sparse.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/100x100_jagged_sparse_Output.txt",
        "100x100_jagged",
        "reader",
        4887,  // epoch size
        4887,  // mb size 
        1,  // num epochs
        1,
        0,
        0,
        1,
        true);
};


// 1 sequence with 2 samples for each of 3 inputs
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_space_separated)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/edge_cases.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/space_separated.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/space_separated_Output.txt",
        "space_separated",
        "reader",
        2,  // epoch size
        2,  // mb size  
        1,  // num epochs
        3,
        0,
        0,
        1);
};



// 1 sequences with 1 sample/input, the last sequence is not well-formed 
// (trailing '\n' is missing)
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_missing_trailing_newline)
{
    BOOST_REQUIRE_EXCEPTION(
        HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/edge_cases.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/missing_trailing_newline.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/missing_trailing_newline_Output.txt",
        "missing_trailing_newline",
        "reader",
        2,  // epoch size
        2,  // mb size  
        1,  // num epochs
        1,
        0,
        0,
        1),
        std::runtime_error,
        [](std::runtime_error const& ex)
    {
        return string("Reached the maximum number of allowed errors"
            " while reading the input file (missing_trailing_newline.txt).") == ex.what();
    });
};

// 1 sequences with 1 sample/input, the last sequence is not well-formed 
// (trailing '\n' is missing)
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_missing_trailing_newline_ignored)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/edge_cases.cntk",
        // the output file does not contain the last sample from the malformed line
        testDataPath() + "/Control/CNTKTextFormatReader/missing_trailing_newline.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/missing_trailing_newline_Output.txt",
        "missing_trailing_newline_ignored",
        "reader",
        2,  // epoch size
        2,  // mb size  
        1,  // num epochs
        1,
        0,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_blank_lines)
{
    BOOST_REQUIRE_EXCEPTION(
        HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/edge_cases.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/blank_lines.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/blank_lines_Output.txt",
        "blank_lines",
        "reader",
        2,  // epoch size
        2,  // mb size  
        1,  // num epochs
        1,
        0,
        0,
        1),
        std::runtime_error,
        [](std::runtime_error const& ex)
    {
        return string("Reached the maximum number of allowed errors"
            " while reading the input file (contains_blank_lines.txt).") == ex.what();
    });
};


BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_blank_lines_ignored)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/edge_cases.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/blank_lines.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/blank_lines_Output.txt",
        "blank_lines_ignored",
        "reader",
        3,  // epoch size
        3,  // mb size  
        1,  // num epochs
        1,
        0,
        0,
        1);
};

BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_duplicate_inputs) 
{
    BOOST_REQUIRE_EXCEPTION(
        HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/edge_cases.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/duplicate_inputs.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/duplicate_inputs_Output.txt",
        "duplicate_inputs",
        "reader",
        1,  // epoch size
        1,  // mb size  
        1,  // num epochs
        2,
        0,
        0,
        1),
        std::runtime_error,
        [](std::runtime_error const& ex)
    {
        return string("Reached the maximum number of allowed errors"
            " while reading the input file (duplicate_inputs.txt).") == ex.what();
    });
};

// input contains a number of empty sparse samples
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_empty_samples)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/edge_cases.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/empty_samples.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/empty_samples_Output.txt",
        "empty_samples",
        "reader",
        6,  // epoch size
        6,  // mb size  
        1,  // num epochs
        1,
        1,
        0,
        1,
        false, // dense features
        true, // sparse labels
        false); // do not use shared layout
};


// input contains a number of empty sparse samples
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_ref_data_with_escape_sequences)
{
    HelperRunReaderTest<float>(
        testDataPath() + "/Config/CNTKTextFormatReader/edge_cases.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/ref_data_with_escape_sequences.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/ref_data_with_escape_sequences_Output.txt",
        "ref_data_with_escape_sequences",
        "reader",
        9,  // epoch size
        9,  // mb size  
        1,  // num epochs
        1,
        1,
        0,
        1,
        true, // sparse features
        false, // dense labels
        false); // do not use shared layout
};

// input contains a number of empty sparse samples
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_invalid_inputs)
{
    vector<StreamDescriptor> streams(2);
    streams[0].m_alias = "A";
    streams[0].m_name = L"A";
    streams[0].m_storageType = StorageType::dense;
    streams[0].m_sampleDimension = 1;

    streams[1].m_alias = "B";
    streams[1].m_name = L"B";
    streams[1].m_storageType = StorageType::sparse_csc;
    streams[1].m_sampleDimension = 10;

    CNTKTextFormatReaderTestRunner<float> testRunner("invalid_inputs.txt", streams, 99999);

    auto output = testDataPath() + "/Control/CNTKTextFormatReader/invalid_inputs_Output.txt";

    boost::filesystem::remove(output);

    FILE * redirected = fopen(output.c_str(), "w");
    if (redirected == nullptr)
    {
        BOOST_FAIL("Cannot open output file.");
    }

    fflush(stderr);
    // duplicate stderr
    int stderrDup = _dup(2);
    // redirect stderr to the output file
    if (-1 == stderrDup || -1 == _dup2(_fileno(redirected), 2))
    {
        BOOST_FAIL("Cannot redirect stderr.");
    }
    else 
    {
        BOOST_SCOPE_EXIT(stderrDup, redirected)
        {
            fflush(stderr);
            fclose(redirected);
            // restore stderr
            if (-1 == _dup2(stderrDup, 2))
            {
                BOOST_FAIL("Cannot restore stderr.");
            }
            _close(stderrDup);

        } BOOST_SCOPE_EXIT_END
        
        testRunner.LoadChunk();
    }

    auto control = testDataPath() + "/Control/CNTKTextFormatReader/invalid_inputs_Control.txt";

    CheckFilesEquivalent(control, output);
};

// 100 sequences with N samples for each of 3 inputs, where N is chosen at random
// from [1, 100] for each sequence
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_100x100x3)
{
    string outputFile = testDataPath() + "/Control/CNTKTextFormatReader/100x100x3_jagged_sequences_dense_Output.txt";

    HelperReadInAndWriteOut<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        outputFile,
        "100x100x3_randomize_auto",
        "reader",
        5476,  // epoch size = number of samples in the input file
        1000,  // mb size  
        1,  // num epochs
        3,
        0,
        0,
        1, 
        false, // dense features
        false, 
        false); // do not user shared layout

    SortLinesInFile(outputFile, 5476 * 3);

    auto controlFile = testDataPath() + "/Control/CNTKTextFormatReader/100x100x3_jagged_sequences_dense_sorted.txt";
    
    CheckFilesEquivalent(controlFile, outputFile);
};

// 200 sequences with N samples in an input, 
// where N is chosen at random from [1, 200] for each input in a sequence.
BOOST_AUTO_TEST_CASE(CNTKTextFormatReader_200x200x2_seq2seq)
{
    string outputFile = testDataPath() + "/Control/CNTKTextFormatReader/200x200x2_seq2seq_dense_Output.txt";

    HelperReadInAndWriteOut<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        outputFile,
        "200x200x2_seq2seq",
        "reader",
        14237,  // epoch size = number of samples in the input file
        5000,  // mb size  
        1,  // num epochs
        2,
        0,
        0,
        1,
        false, // dense features
        false,
        false); // do not user shared layout

    SortLinesInFile(outputFile, 14237 * 2);

    auto controlFile = testDataPath() + "/Control/CNTKTextFormatReader/200x200x2_seq2seq_dense_sorted.txt";

   CheckFilesEquivalent(controlFile, outputFile);
};

// 5 sequences with up to 10 samples each
BOOST_AUTO_TEST_CASE(CompositeCNTKTextFormatReader_5x5_and_5x10_jagged_minibatch_10)
{
    // Using one file with two streams inside to write the output file.
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/5x10_and_5x5_jagged_Output.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/5x10_and_5x5_jagged_Output.txt",
        "5x10_and_5x5_jagged",
        "reader",
        40,     // epoch size
        10,     // mb size
        3,      // num epochs
        2,
        0,
        0,
        1,
        false,
        false,
        false);

    // Using two files with a stream per file to combine and check against the output written above.
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/5x10_and_5x5_jagged_Output.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/5x10_and_5x5_jagged_composite_Output.txt",
        "5x10_and_5x5_jagged_composite",
        "reader",
        40,     // epoch size
        10,     // mb size
        3,      // num epochs
        2,
        0,
        0,
        1,
        false,
        false,
        false);
};

// 5 sequences with up to 10 samples each
BOOST_AUTO_TEST_CASE(CompositeCNTKTextFormatReader_5x5_and_5x10_jagged_minibatch_21)
{
    // Using one file with two streams inside to write the output file.
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/5x10_and_5x5_jagged_Output.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/5x10_and_5x5_jagged_Output.txt",
        "5x10_and_5x5_jagged",
        "reader",
        413,     // epoch size
        21,     // mb size
        3,      // num epochs
        2,
        0,
        0,
        1,
        false,
        false,
        false);

    // Using two files with a stream per file to combine and check against the output written above.
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/5x10_and_5x5_jagged_Output.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/5x10_and_5x5_jagged_composite_Output.txt",
        "5x10_and_5x5_jagged_composite",
        "reader",
        413,     // epoch size
        21,     // mb size
        3,      // num epochs
        2,
        0,
        0,
        1,
        false,
        false,
        false);
};

BOOST_AUTO_TEST_CASE(CNTKTextFormatReaderNoFirstMinibatchData)
{
    HelperRunReaderTest<double>(
        testDataPath() + "/Config/CNTKTextFormatReader/dense.cntk",
        testDataPath() + "/Control/CNTKTextFormatReader/NonExistent.txt",
        testDataPath() + "/Control/CNTKTextFormatReader/CNTKTextFormatReaderNoFirstMinibatchData_Output.txt",
        "1x2",
        "reader",
        10, // epoch size
        1,  // mb size
        10,  // num epochs
        1,
        0,
        1,
        2);
};

BOOST_AUTO_TEST_SUITE_END()

} } } }
