#pragma once

#include <boost/test/unit_test.hpp>
#include "boost/filesystem.hpp"

using std::wstring;

struct BSFixture
{
public:
    BSFixture(){
        boost::filesystem::path path(boost::unit_test::framework::master_test_suite().argv[0]);
        wstring parentPath = boost::filesystem::canonical(path.parent_path()).generic_wstring();

#ifdef _WIN32
        // The executable path on Windows is e.g. <cntk>/x64/Debug/Unittests/
        m_testDataPath = parentPath + L"/../../../Tests/UnitTests/BrainScriptTests";
#else
        // The executable path on Linux is e.g. <cntk>/build/cpu/release/bin/
        m_testDataPath = parentPath + L"/../../../../Tests/UnitTests/BrainScriptTests";
#endif

        boost::filesystem::path absTestPath(m_testDataPath);
        absTestPath = boost::filesystem::canonical(absTestPath);
        m_testDataPath = absTestPath.generic_wstring();
    }

    const wstring getDataPath(){
        return m_testDataPath;
    }
private:
    wstring m_testDataPath;
};