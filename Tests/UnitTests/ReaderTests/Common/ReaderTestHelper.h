//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include <boost/test/unit_test.hpp>
#include "boost/filesystem.hpp"
#include "DataReader.h"

using namespace Microsoft::MSR::CNTK;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

struct ReaderFixture
{
    // This fixture sets up paths so the tests can assume the right location for finding the configuration
    // file as well as the input data and control data.
    // subPath : an optional sub path (or full path) for the location of data.
    ReaderFixture(string subPath = "", string envVariableErrorMessage = "")
    {
        BOOST_TEST_MESSAGE("Setup fixture");
        m_initialWorkingPath = boost::filesystem::current_path().generic_string();
        BOOST_TEST_MESSAGE("Current working directory: " + m_initialWorkingPath);
        fprintf(stderr, "Current working directory: %s\n", m_initialWorkingPath.c_str());

        boost::filesystem::path path(boost::unit_test::framework::master_test_suite().argv[0]);
        m_parentPath = boost::filesystem::canonical(path.parent_path()).generic_string();
        fprintf(stderr, "Executable path: %s\n", m_parentPath.c_str());

        m_testDataPath = m_parentPath + "/../../../Tests/UnitTests/ReaderTests";
        boost::filesystem::path absTestPath(m_testDataPath);
        absTestPath = boost::filesystem::canonical(absTestPath);
        m_testDataPath = absTestPath.generic_string();

        BOOST_TEST_MESSAGE("Setting test data path to: " + m_testDataPath);
        fprintf(stderr, "Test path: %s\n", m_testDataPath.c_str());

        string newCurrentPath;

        // Determine if a sub-path has been specified and it is not a relative path
        if (subPath.length())
        {
            // Retrieve the full path from the environment variable (if any)
            // Currently limited to a single expansion of an environment variable at the beginning of the string.
            if (subPath[0] == '%')
            {
                auto end = subPath.find_last_of(subPath[0]);
                string environmentVariable = subPath.substr(1, end - 1);

                BOOST_TEST_MESSAGE("Retrieving environment variable: " + environmentVariable);
                fprintf(stderr, "Retrieving environment variable: %s\n", environmentVariable.c_str());

                const char* p = std::getenv(environmentVariable.c_str());
                if (p)
                {
                    newCurrentPath = p + subPath.substr(end + 1);
                }
                else
                {
                    BOOST_TEST_MESSAGE("Invalid environment variable: " + subPath);
                    fprintf(stderr, "Invalid environment variable: %s\n", subPath.c_str());

                    if (!envVariableErrorMessage.empty())
                    {
                        BOOST_TEST_MESSAGE(envVariableErrorMessage);
                        fprintf(stderr, envVariableErrorMessage.c_str());
                    }

                    newCurrentPath = m_testDataPath;
                }
            }
            else if ((subPath[0] == '/' && subPath[1] == '//') || (subPath[0] == '\\' && subPath[1] == '\\'))
            {
                newCurrentPath = subPath;
            }
            else
            {
                newCurrentPath = m_testDataPath + subPath;
            }
        }

        BOOST_TEST_MESSAGE("Setting current path to: " + newCurrentPath);
        fprintf(stderr, "Set current path to: %s\n", newCurrentPath.c_str());
        boost::filesystem::current_path(newCurrentPath);

        BOOST_TEST_MESSAGE("Current working directory is now: " + boost::filesystem::current_path().generic_string());
        fprintf(stderr, "Current working directory is now: %s\n", boost::filesystem::current_path().generic_string().c_str());
    }

    ~ReaderFixture()
    {
        BOOST_TEST_MESSAGE("Teardown fixture");
        BOOST_TEST_MESSAGE("Reverting current path to: " + m_initialWorkingPath);
        fprintf(stderr, "Set current path to: %s\n", m_initialWorkingPath.c_str());
        boost::filesystem::current_path(m_initialWorkingPath);
    }

    // Limits the number of minibatches to read, to reduce time and data file size
    size_t m_maxMiniBatchCount = 10;

    string m_initialWorkingPath;
    string m_testDataPath;
    string m_parentPath;

    string initialPath()
    {
        return m_initialWorkingPath;
    }
    string testDataPath()
    {
        return m_testDataPath;
    }
    string currentPath()
    {
        return boost::filesystem::current_path().generic_string();
    }

    // Helper function to write a matrix (feature or label) to a file
    // matrix       : the matrix to output
    // mblayout     : corresponding mb layout
    // outputFile   : the output stream to write to
    template <class ElemType>
    void OutputMatrix(Matrix<ElemType>& matrix, const MBLayout& layout, ofstream& outputFile)
    {
        if (matrix.GetMatrixType() == MatrixType::SPARSE)
    {
            matrix.SwitchToMatrixType(MatrixType::DENSE, MatrixFormat::matrixFormatDense, true);
        }

        std::unique_ptr<ElemType[]> pItem{matrix.CopyToArray()};
       
        auto numRows = matrix.GetNumRows();
        auto numCols = matrix.GetNumCols();
        auto numParallelSequences = layout.GetNumParallelSequences();

        for (size_t i = 0; i < numCols; i++)
        {
            auto s = i % numParallelSequences;
            auto t = i / numParallelSequences;
            
            if (layout.IsGap(FrameRange(nullptr, t).Sequence(s)))
            {
                continue;
            }

            for (auto j = 0; j < numRows; j++)
        {
                auto idx = i*numRows + j;
                outputFile << pItem[idx] << ((j + 1) == numRows ? "\n" : " ");
            }
        }
    }

    // Helper function to compare files and verify that they are equivalent content-wise 
    // (identical character content ignoring differences in white spaces).
    void CheckFilesEquivalent(
        string filename1,
        string filename2)
    {
        std::ifstream ifstream1(filename1);
        std::ifstream ifstream2(filename2);

        std::istream_iterator<string> beginStream1(ifstream1), endStream1;
        std::istream_iterator<string> beginStream2(ifstream2), endStream2;

        BOOST_CHECK_EQUAL_COLLECTIONS(beginStream1, endStream1, beginStream2, endStream2);
    }



    // Helper function to write the Reader's content to a file.
    // testDataFilePath     : the file path for writing the minibatch data (used for comparing against control data)
    // dataReader       : the DataReader to get minibatches from
    // map              : the map containing the feature and label matrices
    // epochs           : the number of epochs to read
    // mbSize           : the minibatch size
    // epochSize        : the epoch size
    // numFeatureFiles  : the number of feature files used (multi IO)
    // numLabelFiles    : the number of label files used (multi IO)
    // subsetNum        : the subset number for parallel trainings
    // numSubsets       : the number of parallel trainings (set to 1 for single)
    template <class ElemType>
    void HelperWriteReaderContentToFile(
        const string testDataFilePath,
        DataReader& dataReader,
        StreamMinibatchInputs& map,
        size_t epochs,
        size_t mbSize,
        size_t epochSize,
        size_t numFeatureFiles,
        size_t numLabelFiles,
        size_t subsetNum,
        size_t numSubsets)
    {
        // Setup output file
        boost::filesystem::remove(testDataFilePath);
        ofstream outputFile(testDataFilePath, ios::out);

        for (auto epoch = 0; epoch < epochs; epoch++)
        {
            if (numSubsets == 1)
            {
                dataReader.StartMinibatchLoop(mbSize, epoch, epochSize);
            }
            else
            {
                dataReader.StartDistributedMinibatchLoop(mbSize, epoch, subsetNum, numSubsets, epochSize);
            }

            for (auto cnt = 0; dataReader.GetMinibatch(map) && cnt < m_maxMiniBatchCount; cnt++)
            {
                // Process the Feature Matri(x|ces)
                for (auto i = 0; i < numFeatureFiles; i++)
                {
                    wstring name = numFeatureFiles > 1 ? L"features" + std::to_wstring(i + 1) : L"features";
                    auto& layoutPtr = map.GetInput(name).pMBLayout;
                    OutputMatrix(map.GetInputMatrix<ElemType>(name), *layoutPtr, outputFile);
                }

                // Process the Label Matri(x|ces)
                for (auto i = 0; i < numLabelFiles; i++)
                {
                    wstring name = numLabelFiles > 1 ? L"labels" + std::to_wstring(i + 1) : L"labels";
                    auto& layoutPtr = map.GetInput(name).pMBLayout;
                    OutputMatrix(map.GetInputMatrix<ElemType>(name), *layoutPtr, outputFile);
                }
            }
        }

        outputFile.close();
    }


    // Helper function to create and populate input structure.
    // numFeatureFiles      : the number of feature input streams
    // numLabelFiles        : the number of label input streams
    // sparseFeatures       : indicates whether the corresponding matrix type should be set to sparse or not
    // sparseLabels         : same as above, but for labels
    // useSharedLayout      : if false, an individual layout is created for each input

    template <class ElemType>
    shared_ptr<StreamMinibatchInputs> CreateStreamMinibatchInputs(
        size_t numFeatureInputs,
        size_t numLabelInputs,
        bool sparseFeatures = false,
        bool sparseLabels = false,
        bool useSharedLayout = true)
    {
        auto mbInputs = make_shared<StreamMinibatchInputs>();
        std::vector<shared_ptr<Matrix<ElemType>>> features;
        std::vector<shared_ptr<Matrix<ElemType>>> labels;

        // For the time being, use the same layout across all inputs.
        // TODO: add an option to create per-input layouts (once we have test-cases with different layouts)
        MBLayoutPtr pMBLayout = make_shared<MBLayout>(1, 0, L"X");

        for (auto i = 0; i < numFeatureInputs; i++)
        {
            features.push_back(make_shared<Matrix<ElemType>>(0));
            if (sparseFeatures)
            {
                features.back()->SwitchToMatrixType(MatrixType::SPARSE, MatrixFormat::matrixFormatSparseCSC, false);
            }
            wstring name = numFeatureInputs > 1 ? L"features" + std::to_wstring(i + 1) : L"features";
            if (!useSharedLayout)
            {
                pMBLayout = make_shared<MBLayout>(1, 0, name);
            }
            mbInputs->insert(make_pair(name, StreamMinibatchInputs::Input(features[i], pMBLayout, TensorShape())));
        }

        for (auto i = 0; i < numLabelInputs; i++)
        {
            labels.push_back(make_shared<Matrix<ElemType>>(0));
            if (sparseLabels)
            {
                labels.back()->SwitchToMatrixType(MatrixType::SPARSE, MatrixFormat::matrixFormatSparseCSC, false);
            }
            wstring name = numLabelInputs > 1 ? L"labels" + std::to_wstring(i + 1) : L"labels";
            if (!useSharedLayout)
            {
                pMBLayout = make_shared<MBLayout>(1, 0, name);
            }
            
            mbInputs->insert(make_pair(name, StreamMinibatchInputs::Input(labels[i], pMBLayout, TensorShape())));
        }

        return mbInputs;
    }

    // Helper function instantiate a data reader based on the provided configuration.
    // configFileName       : the file name for the config file
    // testSectionName      : the section name for the test inside the config file
    // readerSectionName    : the reader field name in the test section

    shared_ptr<DataReader> GetDataReader(
        const string configFileName,
        const string testSectionName,
        const string readerSectionName)
    {
        std::wstring configFN(configFileName.begin(), configFileName.end());
        std::wstring configFileCommand(L"configFile=" + configFN);

        wchar_t* arg[2]{L"CNTK", &configFileCommand[0]};
        ConfigParameters config;
        const std::string rawConfigString = ConfigParameters::ParseCommandLine(2, arg, config);

        config.ResolveVariables(rawConfigString);
        const ConfigParameters simpleDemoConfig = config(testSectionName);
        const ConfigParameters readerConfig = simpleDemoConfig(readerSectionName);

        return make_shared<DataReader>(readerConfig);
    }

    // Helper function to read in the input dataset and write out the resulting minibatches to a file.
    // configFileName       : the file name for the config file
    // testDataFilePath     : the file path for writing the minibatch data (used for comparing against control data)
    // testSectionName      : the section name for the test inside the config file
    // readerSectionName    : the reader field name in the test section
    // epochSize            : the epoch size
    // mbSize               : the minibatch size
    // epochs               : the number of epochs to read
    // numFeatureFiles      : the number of feature files used (multi IO)
    // numLabelFiles        : the number of label files used (multi IO)
    // subsetNum            : the subset number for parallel trainings
    // numSubsets           : the number of parallel trainings (set to 1 for single)
    // sparseFeatures       : indicates whether the corresponding matrix type should be set to sparse or not
    // sparseLabels         : same as above, but for labels
    // useSharedLayout      : if false, an individual layout is created for each input

    template <class ElemType>
    void HelperReadInAndWriteOut(
        string configFileName,
        string testDataFilePath,
        string testSectionName,
        string readerSectionName,
        size_t epochSize,
        size_t mbSize,
        size_t epochs,
        size_t numFeatureFiles,
        size_t numLabelFiles,
        size_t subsetNum,
        size_t numSubsets,
        bool sparseFeatures = false,
        bool sparseLabels = false,
        bool useSharedLayout = true)
    {
        shared_ptr<StreamMinibatchInputs> inputsPtr =
            CreateStreamMinibatchInputs<ElemType>(numFeatureFiles, numLabelFiles,
            sparseFeatures, sparseLabels, useSharedLayout);

        shared_ptr<DataReader> readerPtr = GetDataReader(configFileName,
            testSectionName, readerSectionName);

        // Perform the data reading
        HelperWriteReaderContentToFile<ElemType>(testDataFilePath, *readerPtr, *inputsPtr,
            epochs, mbSize, epochSize, numFeatureFiles, numLabelFiles, subsetNum, numSubsets);
    }

    // Helper function to run a Reader test.
    // configFileName       : the file name for the config file
    // controlDataFilePath  : the file path for the control data to verify against
    // testDataFilePath     : the file path for writing the minibatch data (used for comparing against control data)
    // testSectionName      : the section name for the test inside the config file
    // readerSectionName    : the reader field name in the test section
    // epochSize            : the epoch size
    // mbSize               : the minibatch size
    // epochs               : the number of epochs to read
    // numFeatureFiles      : the number of feature files used (multi IO)
    // numLabelFiles        : the number of label files used (multi IO)
    // subsetNum            : the subset number for parallel trainings
    // numSubsets           : the number of parallel trainings (set to 1 for single)
    // sparseFeatures       : indicates whether the corresponding matrix type should be set to sparse or not
    // sparseLabels         : same as above, but for labels
    // useSharedLayout      : if false, an individual layout is created for each input

    template <class ElemType>
    void HelperRunReaderTest(
        string configFileName,
        string controlDataFilePath,
        string testDataFilePath,
        string testSectionName,
        string readerSectionName,
        size_t epochSize,
        size_t mbSize,
        size_t epochs,
        size_t numFeatureFiles,
        size_t numLabelFiles,
        size_t subsetNum,
        size_t numSubsets,
        bool sparseFeatures = false,
        bool sparseLabels = false,
        bool useSharedLayout = true)
    {
        HelperReadInAndWriteOut<ElemType>(configFileName, testDataFilePath, testSectionName, readerSectionName,
            epochSize, mbSize, epochs, numFeatureFiles, numLabelFiles, subsetNum,numSubsets,
            sparseFeatures, sparseLabels, useSharedLayout);

        CheckFilesEquivalent(controlDataFilePath, testDataFilePath);
    }

    // Helper function to run a Reader test and catch an expected exception.
    // configFileName       : the file name for the config file
    // testSectionName      : the section name for the test inside the config file
    // readerSectionName    : the reader field name in the test section
    template <class ElemType, class ExceptionType>
    void HelperRunReaderTestWithException(
        string configFileName,
        string testSectionName,
        string readerSectionName)
    {
        BOOST_CHECK_THROW(
            GetDataReader(configFileName,testSectionName, readerSectionName),
            ExceptionType);
    }
};
}

}}}
