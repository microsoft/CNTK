//
// <copyright file="File.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once
#include <stdio.h>
#include <string>
#include <vector>
#include <stdint.h>
#ifdef _WIN32
#include <Windows.h>
#endif
#ifdef __unix__
#include <unistd.h>
#endif
#include "fileutil.h"   // for f{ge,pu}t{,Text}()

namespace Microsoft{ namespace MSR { namespace CNTK {

// file options, Type of textfile to use
enum FileOptions
{
    fileOptionsNull = 0, // invalid value
    fileOptionsBinary = 1,  // binary file
    fileOptionsText = 2, // text based file, UTF-8
    fileOptionsUnicode = 4,   // text based file, Unicode
    fileOptionsType = fileOptionsBinary | fileOptionsText | fileOptionsUnicode, // file types
    fileOptionsRead = 8,   // open in read mode
    fileOptionsWrite = 16,  // open in write mode
    fileOptionsSequential = 32,     // optimize for sequential reads (allocates big buffer)
    fileOptionsReadWrite = fileOptionsRead | fileOptionsWrite, // read/write mode
};

// markers used for text files
enum FileMarker
{
    fileMarkerNull = 0, // invalid value
    fileMarkerBeginFile = 1, // begin of file marker
    fileMarkerEndFile = 2, // end of file marker
    fileMarkerBeginList = 3, // Beginning of list marker
    fileMarkerListSeparator = 4, // separate elements of a list
    fileMarkerEndList = 5, // end of line/list marker
    fileMarkerBeginSection = 6, // beginning of section
    fileMarkerEndSection = 7, // end of section
};

// attempt a given operation (lambda) and retry multiple times
// body - the lambda to retry, must be restartable

template<typename FUNCTION> static void attempt(int retries, const FUNCTION & body)
{
    for (int attempt = 1; ; attempt++)
    {
        try
        {
            body();
            if (attempt > 1) fprintf (stderr, "attempt: success after %d retries\n", attempt);
            break;
        }
        catch (const std::exception & e)
        {
            void sleep(size_t ms);
            if (attempt >= retries)
                throw;      // failed N times --give up and rethrow the error
            fprintf (stderr, "attempt: %s, retrying %d-th time out of %d...\n", e.what(), attempt+1, retries);
            // wait a little, then try again
#ifdef _WIN32
            ::Sleep(1000);
#else       // assuming __unix__
            sleep(1);
#endif
        }
    }
}

template<typename FUNCTION> static void attempt (const FUNCTION & body)
{
    static const int retries = 5;
    attempt<FUNCTION> (retries, body);
    //msra::util::attempt<FUNCTION> (retries, body);
}

class File
{
private:
    FILE* m_file;
    size_t m_size;
    int m_options; // FileOptions ored togther
    void Init(const wchar_t* filename, int fileOptions);

public:
    File(const std::wstring& filename, int fileOptions);
    File(const std::string& filename, int fileOptions);
    File(const wchar_t* filename, int fileOptions);
    ~File(void);

    uint64_t GetPosition();
    void SetPosition(uint64_t pos);
    void goToDelimiter(int delim);

    bool IsTextBased();

    size_t Size();
    bool IsUnicodeBOM(bool skip=false);
    bool IsEOF();
    bool IsWhiteSpace(bool skip=false);
    int EndOfLineOrEOF(bool skip=false);

    // TryGetText - for text value, try and get a particular type
    // returns - true if value returned, otherwise false, can't parse
    template <typename T>
    bool TryGetText(T& val)
    {
        assert(IsTextBased());
        return !!ftrygetText(m_file, val);
    }

    void GetLine(std::wstring& str);
    void GetLine(std::string& str);

    // put operator for basic types
    template <typename T>
    File& operator<<(T val)
    {
#ifndef	LINUX
        attempt([=]()
#endif
        {
            if (IsTextBased())
                fputText(m_file, val);
            else
                fput(m_file, val);
        }
#ifndef	LINUX
        );
#endif
        return *this;
    }
    File& operator<<(const std::wstring& val);
    File& operator<<(const std::string& val);  
    File& operator<<(FileMarker marker);
    File& PutMarker(FileMarker marker, size_t count);
    File& PutMarker(FileMarker marker, const std::string& section);
    File& PutMarker(FileMarker marker, const std::wstring& section);

    // put operator for vectors of types
    template <typename T>
    File& operator<<(const std::vector<T>& val)
    {
        this->PutMarker(fileMarkerBeginList, val.size());
        for (int i = 0; i < val.size(); i++)
        {
            *this << val[i] << fileMarkerListSeparator;
        }
        *this << fileMarkerEndList;
        return *this;
    }

    // get operator for basic types
    template <typename T>
    File& operator>>(T& val)
    {
#ifndef	LINUX
        attempt([&]()
#endif
        {
            if (IsTextBased())
                fgetText(m_file, val);
            else
                fget(m_file, val);
        }
#ifndef	LINUX
        );
#endif
        return *this;
    }

    void WriteString(const char* str, int size=0); // zero terminated strings use size=0
    void ReadString(char* str, int size);    // read up to size bytes, or a zero terminator (or space in text mode)
    void WriteString(const wchar_t* str, int size=0); // zero terminated strings use size=0
    void ReadString(wchar_t* str, int size);    // read up to size bytes, or a zero terminator (or space in text mode)
    void ReadChars(std::string& val, size_t cnt, bool reset=false); // read a specified number of characters, and reset read pointer if requested
    void ReadChars(std::wstring& val, size_t cnt, bool reset=false); // read a specified number of characters, and reset read pointer if requested

    File& operator>>(std::wstring& val);
    File& operator>>(std::string& val);
    File& operator>>(FileMarker marker);
    File& GetMarker(FileMarker marker, size_t& count);
    File& GetMarker(FileMarker marker, const std::string& section);
    File& GetMarker(FileMarker marker, const std::wstring& section);
    bool TryGetMarker(FileMarker marker, const std::string& section);
    bool TryGetMarker(FileMarker marker, const std::wstring& section);

    bool IsMarker(FileMarker marker, bool skip = true);

    // get a vector of types
    template <typename T>
    File& operator>>(std::vector<T>& val)
    {
        T element;
        val.clear();
        size_t size=0;
        this->GetMarker(fileMarkerBeginList, size);
        if (size > 0)
        {
            for (int i = 0; i < size; i++)
            {
                // get list separators if not the first element
                if (i > 0)
                    *this >> fileMarkerListSeparator;
                *this >> element;
                val.push_back(element);
            }
            *this >> fileMarkerEndList;
        }
        else
        {
            bool first = true;
            while (!this->IsMarker(fileMarkerEndList))
            {
                if (!first)
                    *this >> fileMarkerListSeparator;
                *this >> element;
                val.push_back(element);
                first = false;
            }
        }
        return *this;
    }
};

}}}
