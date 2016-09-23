//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include <boost/test/unit_test.hpp>
#include "boost/filesystem.hpp"

using std::string;

namespace Microsoft { namespace MSR { namespace CNTK { namespace Test {

struct EvalFixture
{
    // This fixture sets up paths so the tests can assume the right location for finding the configuration
    // file as well as the input data and control data.
    // subPath : an optional sub path (or full path) for the location of data.
    EvalFixture(string subPath = "", string envVariableErrorMessage = "")
    {
        BOOST_TEST_MESSAGE("Setup fixture");
        m_initialWorkingPath = boost::filesystem::current_path().generic_string();
        BOOST_TEST_MESSAGE("Current working directory: " + m_initialWorkingPath);
        fprintf(stderr, "Current working directory: %s\n", m_initialWorkingPath.c_str());

        boost::filesystem::path path(boost::unit_test::framework::master_test_suite().argv[0]);
        m_parentPath = boost::filesystem::canonical(path.parent_path()).generic_string();
        fprintf(stderr, "Executable path: %s\n", m_parentPath.c_str());

#ifdef _WIN32
        // The executable path on Windows is e.g. <cntk>/x64/Debug/
        m_testDataPath = m_parentPath + "/../../Tests/UnitTests/EvalTests";
#else
        // The executable path on Linux is e.g. <cntk>/build/cpu/release/bin/
        m_testDataPath = m_parentPath + "/../../../../Tests/UnitTests/EvalTests";
#endif
        boost::filesystem::path absTestPath(m_testDataPath);
        absTestPath = boost::filesystem::canonical(absTestPath);
        m_testDataPath = absTestPath.generic_string();

        BOOST_TEST_MESSAGE("Setting test data path to: " + m_testDataPath);
        fprintf(stderr, "Test path: %s\n", m_testDataPath.c_str());

        string newCurrentPath(m_testDataPath);

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
                        fprintf(stderr, "%s\n", envVariableErrorMessage.c_str());
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

    ~EvalFixture()
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
};
}}}}
