//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// FileTest.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "Basics.h"
#include "fileutil.h"
#include "FileTest.h"
#include "File.h"
#include "Matrix.h"

using namespace Microsoft::MSR::CNTK;

void MatrixFileWriteAndRead()
{
    CPUMatrix<float> M = CPUMatrix<float>::RandomUniform(43, 10, -26.3f, 30.2f);
    CPUMatrix<float> Mcopy(M);
    std::wstring filename(L"c:\\temp\\M.txt");
    File file(filename, fileOptionsUnicode | fileOptionsReadWrite);
    file << M;
    CPUMatrix<float> M1;
    file.SetPosition(0);
    file >> M1;
    if (!Mcopy.IsEqualTo(M1))
        fprintf(stderr, "matrix read/write doesn't pass");
}

// Test the File API
// filename - file to open and read
void TestFileAPI(const TCHAR* filename, int options)
{
    try
    {
        FileTest fileTest;
        File file(filename, options);
        if (options & fileOptionsWrite)
        {
            file << fileTest;
        }
        if (options & fileOptionsRead)
        {
            string str;
            file.SetPosition(0); // rewind to the beginning
            file.GetLine(str);
            file.SetPosition(0); // rewind to the beginning
            file >> fileTest;
            FileTest fileTest2;
            if (fileTest == fileTest2)
                fprintf(stderr, "Success! comparison of serialized classes match");
            else
                fprintf(stderr, "Failure! comparison of serialized classes do not match");
            MatrixFileWriteAndRead();
        }
    }
    catch (exception e)
    {
        fprintf(stderr, "Exception %s\n", e.what());
    }
}

int _tmain(int argc, _TCHAR* argv[])
{
    msra::util::command_line args(argc, argv);
    int options = fileOptionsNull;
    while (args.has(1) && args[0][0] == '-')
    {
        const wchar_t* arg = args.shift();
        switch (arg[1])
        {
        case L't':
        case L'T':
            if (options & fileOptionsType)
            {
                fprintf(stderr, "Only one file type allowed\n");
                goto exit;
            }
            options |= fileOptionsText;
            break;
        case L'u':
        case L'U':
            if (options & fileOptionsType)
            {
                fprintf(stderr, "Only one file type allowed\n");
                goto exit;
            }
            options |= fileOptionsUnicode;
            break;
        case L'b':
        case L'B':
            if (options & fileOptionsType)
            {
                fprintf(stderr, "Only one file type allowed\n");
                goto exit;
            }
            options |= fileOptionsBinary;
            break;
        case L'r':
        case L'R':
            options |= fileOptionsRead;
            break;
        case L'w':
        case L'W':
            options |= fileOptionsWrite;
            break;
        default:
            fprintf(stderr, "invalid option, valid options are:\nFile Type:\n-text > text file (UTF-8)\n-unicode > unicode text file\n-binary > binary file\nOperation:\n-read > read file passed\n-write > write the file given (overwrite if it exists)\n");
            goto exit;
            break;
        }
    }
    if (!(options & fileOptionsType))
    {
        options |= fileOptionsText;
        fprintf(stderr, "No file type specified, using UTF-8 text\n");
    }
    if (!(options & fileOptionsReadWrite))
    {
        options |= fileOptionsReadWrite;
        fprintf(stderr, "No read or write specified, using read/write\n");
    }
    const wchar_t* filename = NULL;
    for (const wchar_t* arg = args.shift(); arg; arg = args.shift())
    {
        filename = arg;
    }
    if (filename == NULL)
    {
        fprintf(stderr, "filename expected after options\n");
        goto exit;
    }
    TestFileAPI(filename, options);
exit:
    return 0;
}

namespace Microsoft { namespace MSR { namespace CNTK {

FileTest::FileTest()
{
    m_char = 'c';
    m_wchar = L'W';
    m_int = 123456;
    m_unsigned = 0xfbadf00d;
    m_long = 0xace;
    m_longlong = 0xaddedbadbeef;
    m_int64 = 0xfeedfacef00d;
    m_size_t = 0xbadfadfacade;
    m_single = 1.23456789e-012f;
    m_double = 9.8765432109876548e-098;
    m_str = new char[80];
    strcpy_s(m_str, 80, "sampleString"); // character string, zero terminated
    m_wstr = new wchar_t[80];
    wcscpy_s(m_wstr, 80, L"wideSampleString");    // wide character string, zero terminated
    m_string.append("std:stringSampleString");    // std string
    m_wstring.append(L"std:wstringSampleString"); // std wide string
    m_vectorLong.push_back(m_int);                // vector of supported type
    m_vectorLong.push_back(m_unsigned);
    m_vectorLong.push_back(m_long);
}

// compare two FileTest objects
bool FileTest::operator==(FileTest& test2)
{
    FileTest& test1 = *this;
    bool compare = true;
    compare = compare && m_char == test2.m_char;
    compare = compare && m_wchar == test2.m_wchar;
    compare = compare && m_int == test2.m_int;
    compare = compare && m_unsigned == test2.m_unsigned;
    compare = compare && m_long == test2.m_long;
    compare = compare && m_longlong == test2.m_longlong;
    compare = compare && m_int64 == test2.m_int64;
    compare = compare && m_size_t == test2.m_size_t;
    compare = compare && m_single == test2.m_single;
    compare = compare && m_double == test2.m_double;
    compare = compare && strcmp(m_str, test2.m_str) == 0;
    compare = compare && wcscmp(m_wstr, test2.m_wstr) == 0;
    compare = compare && m_string == test2.m_string;
    compare = compare && m_wstring == test2.m_wstring;
    compare = compare && m_vectorLong.size() == test2.m_vectorLong.size();
    for (int i = 0; compare && i < m_vectorLong.size(); i++)
    {
        compare = compare && m_vectorLong[i] == test2.m_vectorLong[i];
    }
    return compare;
}

FileTest::~FileTest()
{
    delete m_str;
    m_str = NULL;
    delete m_wstr;
    m_wstr = NULL;
}

File& operator<<(File& stream, FileTest& test)
{
    stream.PutMarker(fileMarkerBeginSection, string("beginFileTest"));
    stream << test.m_char << test.m_wchar;
    stream << test.m_int << test.m_unsigned << test.m_long;
    stream << test.m_longlong;
    stream << test.m_int64;
    stream << test.m_size_t;
    stream << test.m_single << test.m_double;
    stream.WriteString(test.m_str);
    stream.WriteString(test.m_wstr);
    stream << test.m_string << test.m_wstring;
    stream << test.m_vectorLong;
    stream.PutMarker(fileMarkerEndSection, string("endFileTest"));
    return stream;
}

File& operator>>(File& stream, FileTest& test)
{
    stream.GetMarker(fileMarkerBeginSection, string("beginFileTest"));
    stream >> test.m_char >> test.m_wchar;
    stream >> test.m_int;
    stream >> test.m_unsigned;
    stream >> test.m_long;
    stream >> test.m_longlong;
    stream >> test.m_int64;
    stream >> test.m_size_t;
    stream >> test.m_single >> test.m_double;
    delete test.m_str;
    test.m_str = new char[80];
    stream.ReadString(test.m_str, 80);
    delete test.m_wstr;
    test.m_wstr = new wchar_t[80];
    stream.ReadString(test.m_wstr, 80);
    stream >> test.m_string;
    stream >> test.m_wstring;
    stream >> test.m_vectorLong;
    stream.GetMarker(fileMarkerEndSection, string("endFileTest"));
    return stream;
}
} } }
