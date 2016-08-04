//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "Config.h"
#include "Actions.h"
#include "boost/filesystem.hpp"
#include <boost/test/unit_test_log.hpp>
#include <boost/test/unit_test_suite.hpp>

using namespace Microsoft::MSR::CNTK;
using namespace std;

namespace Microsoft {
namespace MSR {
namespace CNTK {
namespace Test {

struct DataFixture
{
    // This fixture sets up paths so the tests can assume the right location for finding the configuration
    // file as well as the input data and control data.
    // subPath : an optional sub path (or full path) for the location of data.
    DataFixture(std::string subPath = "", string envVariableErrorMessage = "")
    {
        BOOST_TEST_MESSAGE("Setup fixture");
        m_initialWorkingPath = boost::filesystem::current_path().generic_string();
        BOOST_TEST_MESSAGE("Current working directory: " + m_initialWorkingPath);
        fprintf(stderr, "Current working directory: %s\n", m_initialWorkingPath.c_str());

        boost::filesystem::path path(boost::unit_test::framework::master_test_suite().argv[0]);
        m_parentPath = boost::filesystem::canonical(path.parent_path()).generic_string();
        fprintf(stderr, "Executable path: %s\n", m_parentPath.c_str());

#ifdef _WIN32
	// The executable path on Windows is e.g. <cntk>/x64/Debug/Unittests/
        m_testDataPath = m_parentPath + "/../../../Tests/UnitTests/NetworkTests";
#else
	// The executable path on Linux is e.g. <cntk>/build/cpu/release/bin/
        m_testDataPath = m_parentPath + "/../../../../Tests/UnitTests/NetworkTests";
#endif
        boost::filesystem::path absTestPath(m_testDataPath);
        absTestPath = boost::filesystem::canonical(absTestPath);
        m_testDataPath = absTestPath.generic_string();

        BOOST_TEST_MESSAGE("Setting test data path to: " + m_testDataPath);
        fprintf(stderr, "Test path: %s\n", m_testDataPath.c_str());

        string newCurrentPath;

        // Determine if a subpath has been specified and it is not a relative path
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
                        fprintf(stderr, "%s", envVariableErrorMessage.c_str());
                    }

                    newCurrentPath = m_testDataPath;
                }
            }
            else if ((subPath[0] == '/' && subPath[1] == '/') || (subPath[0] == '\\' && subPath[1] == '\\'))
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

    ~DataFixture()
    {
        BOOST_TEST_MESSAGE("Teardown fixture");
        BOOST_TEST_MESSAGE("Reverting current path to: " + m_initialWorkingPath);
        fprintf(stderr, "Set current path to: %s\n", m_initialWorkingPath.c_str());
        boost::filesystem::current_path(m_initialWorkingPath);
    }

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

    // Helper function to run a network test.
    // configFileName       : the file path for the config file
    // controlDataFilePath  : the file path for the control data to verify against
    // testDataFilePath     : the file path for writing the minibatch data (used for comparing against control data)
    template <class ElemType>
    void HelperRunNetworkTest(
        const wstring configFileName,
        const string controlDataFilePath,
        const string testDataFilePath)
    {
        // Setup output file
        boost::filesystem::remove(testDataFilePath);

        ConfigParameters config;
        config.LoadConfigFile(configFileName);

        wstring outFileName = config(L"outputPath", L"");

        ConfigArray command = config(L"command", "write");
        ConfigParameters commandParams(config(command[0]));

        DoWriteOutput<ElemType>(commandParams);

        std::ifstream ifstream1(controlDataFilePath);
        std::ifstream ifstream2(testDataFilePath);

        std::istream_iterator<char> beginStream1(ifstream1);
        std::istream_iterator<char> end;
        std::istream_iterator<char> beginStream2(ifstream2);

        BOOST_CHECK_EQUAL_COLLECTIONS(beginStream1, end, beginStream2, end);
    }
};
}
}
}
}
