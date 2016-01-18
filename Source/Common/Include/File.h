//
// <copyright file="File.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "Basics.h"
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
#include "fileutil.h" // for f{ge,pu}t{,Text}()
#include <fstream>    // for LoadMatrixFromTextFile() --TODO: change to using this File class
#include <sstream>

namespace Microsoft { namespace MSR { namespace CNTK {

using namespace std;

// file options, Type of textfile to use
enum FileOptions
{
    fileOptionsNull = 0,                                                        // invalid value
    fileOptionsBinary = 1,                                                      // binary file
    fileOptionsText = 2,                                                        // text based file, UTF-8
    fileOptionsUnicode = 4,                                                     // text based file, Unicode
    fileOptionsType = fileOptionsBinary | fileOptionsText | fileOptionsUnicode, // file types
    fileOptionsRead = 8,                                                        // open in read mode
    fileOptionsWrite = 16,                                                      // open in write mode
    fileOptionsSequential = 32,                                                 // optimize for sequential reads (allocates big buffer)
    fileOptionsReadWrite = fileOptionsRead | fileOptionsWrite,                  // read/write mode
};

// markers used for text files
enum FileMarker
{
    fileMarkerNull = 0,          // invalid value
    fileMarkerBeginFile = 1,     // begin of file marker
    fileMarkerEndFile = 2,       // end of file marker
    fileMarkerBeginList = 3,     // Beginning of list marker
    fileMarkerListSeparator = 4, // separate elements of a list
    fileMarkerEndList = 5,       // end of line/list marker
    fileMarkerBeginSection = 6,  // beginning of section
    fileMarkerEndSection = 7,    // end of section
};

// attempt a given operation (lambda) and retry multiple times
// body - the lambda to retry, must be restartable

template <typename FUNCTION>
static void attempt(int retries, const FUNCTION& body)
{
    for (int attempt = 1;; attempt++)
    {
        try
        {
            body();
            if (attempt > 1)
                fprintf(stderr, "attempt: success after %d retries\n", attempt);
            break;
        }
        catch (const std::exception& e)
        {
#ifdef _WIN32
            void sleep(size_t ms);
#endif
            if (attempt >= retries)
                throw; // failed N times --give up and rethrow the error
            fprintf(stderr, "attempt: %s, retrying %d-th time out of %d...\n", e.what(), attempt + 1, retries);
// wait a little, then try again
#ifdef _WIN32
            ::Sleep(1000);
#else // assuming __unix__
            ::sleep(1);
#endif
        }
    }
}

template <typename FUNCTION>
static void attempt(const FUNCTION& body)
{
    static const int retries = 5;
    attempt<FUNCTION>(retries, body);
    //msra::util::attempt<FUNCTION> (retries, body);
}

class File
{
private:
    std::wstring m_filename;
    FILE* m_file;        // file handle
    bool m_pcloseNeeded; // was opened with popen(), use pclose() when destructing
    bool m_seekable;     // this stream is seekable
    int m_options;       // FileOptions ored togther
    void Init(const wchar_t* filename, int fileOptions);

public:
    File(const std::wstring& filename, int fileOptions);
    File(const std::string& filename, int fileOptions);
    File(const wchar_t* filename, int fileOptions);
    ~File(void);

    void Flush();

    bool CanSeek() const
    {
        return m_seekable;
    }
    size_t Size();
    uint64_t GetPosition();
    void SetPosition(uint64_t pos);
    void SkipToDelimiter(int delim);

    bool IsTextBased();

    bool IsUnicodeBOM(bool skip = false);
    bool IsEOF();
    bool IsWhiteSpace(bool skip = false);
    int EndOfLineOrEOF(bool skip = false);

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
    void GetLines(std::vector<std::wstring>& lines);
    void GetLines(std::vector<std::string>& lines);

    // put operator for basic types
    template <typename T>
    File& operator<<(T val)
    {
#ifndef __CUDACC__ // TODO: CUDA compiler blows up, fix this
        attempt([=]()
#endif
                {
                    if (IsTextBased())
                        fputText(m_file, val);
                    else
                        fput(m_file, val);
                }
#ifndef __CUDACC__
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
#ifndef __CUDACC__ // TODO: CUDA compiler blows up, fix this
        attempt([&]()
#endif
                {
                    if (IsTextBased())
                        fgetText(m_file, val);
                    else
                        fget(m_file, val);
                }
#ifndef __CUDACC__
                );
#endif
        return *this;
    }

    void WriteString(const char* str, int size = 0);                   // zero terminated strings use size=0
    void ReadString(char* str, int size);                              // read up to size bytes, or a zero terminator (or space in text mode)
    void WriteString(const wchar_t* str, int size = 0);                // zero terminated strings use size=0
    void ReadString(wchar_t* str, int size);                           // read up to size bytes, or a zero terminator (or space in text mode)
    void ReadChars(std::string& val, size_t cnt, bool reset = false);  // read a specified number of characters, and reset read pointer if requested
    void ReadChars(std::wstring& val, size_t cnt, bool reset = false); // read a specified number of characters, and reset read pointer if requested

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
        size_t size = 0;
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

    // Read a matrix stored in text format from 'filePath' (whitespace-separated columns, newline-separated rows),
    // and return a flat array containing the contents of this file in column-major format.
    // filePath: path to file containing matrix in text format.
    // numRows/numCols: after this function is called, these parameters contain the number of rows/columns in the matrix.
    // returns: a flat array containing the contents of this file in column-major format
    // NOTE: caller is responsible for deleting the returned buffer once it is finished using it.
    // TODO: change to return a std::vector<ElemType>; solves the ownership issue
    // This function does not quite fit here, but it fits elsewhere even worse. TODO: change to use File class!
    template <class ElemType>
    static vector<ElemType> LoadMatrixFromTextFile(const std::string filePath, size_t& numRows, size_t& numCols)
    {
        size_t r = 0;
        size_t numColsInFirstRow = 0;

        // NOTE: Not using the Microsoft.MSR.CNTK.File API here because it
        // uses a buffer of fixed size, which doesn't allow very long rows.
        // See fileutil.cpp fgetline method (std::string fgetline (FILE * f) { fixed_vector<char> buf (1000000); ... })
        std::ifstream myfile(filePath);

        // load matrix into vector of vectors (since we don't know the size in advance).
        std::vector<std::vector<ElemType>> elements;
        if (myfile.is_open())
        {
            std::string line;
            while (std::getline(myfile, line))
            {
                // Break on empty line.  This allows there to be an empty line at the end of the file.
                if (line == "")
                    break;

                istringstream iss(line);
                ElemType element;
                int numElementsInRow = 0;
                elements.push_back(std::vector<ElemType>());
                while (iss >> element)
                {
                    elements[r].push_back(element);
                    numElementsInRow++;
                }

                if (r == 0)
                    numColsInFirstRow = numElementsInRow;
                else if (numElementsInRow != numColsInFirstRow)
                    RuntimeError("The rows in the provided file do not all have the same number of columns: %s", filePath.c_str());

                r++;
            }
            myfile.close();
        }
        else
            RuntimeError("Unable to open file");

        numRows = r;
        numCols = numColsInFirstRow;

        vector<ElemType> array(numRows * numCols);

        // Perform transpose when copying elements from vectors to ElemType[],
        // in order to store in column-major format.
        for (int i = 0; i < numCols; i++)
        {
            for (int j = 0; j < numRows; j++)
                array[i * numRows + j] = elements[j][i];
        }

        return array;
    }

    operator FILE*() const
    {
        return m_file;
    }
};
} } }
