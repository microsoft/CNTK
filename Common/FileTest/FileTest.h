//
// <copyright file="FileTest.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#include <string>
#include <vector>
#include "File.h"

namespace Microsoft{ namespace MSR { namespace CNTK {

class FileTest
{
private:
    char m_char;
    wchar_t m_wchar;
    int m_int;
    unsigned m_unsigned;
    long m_long;
    long long m_longlong;
    __int64 m_int64;
    float m_single;
    double m_double;
    size_t m_size_t;
    char* m_str;  // character string, zero terminated
    wchar_t* m_wstr; // wide character string, zero terminated
    std::string m_string; // std string
    std::wstring m_wstring; // std wide string
    std::vector<long> m_vectorLong; // vector of supported type
public:
    FileTest();
    ~FileTest();
    bool operator==(FileTest& test);

    // declare as friend functions so we can access private members
    friend File& operator>>(File& stream, FileTest& test);
    friend File& operator<<(File& stream, FileTest& test);
};

// operator overloading
File& operator>>(File& stream, FileTest& test);
File& operator<<(File& stream, FileTest& test);

}}}