//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include "File.h"

#ifdef USE_HDFS
#include <hdfs.h>
#endif

namespace Microsoft { namespace MSR { namespace CNTK {

class hdfs_File {
private:
    std::wstring m_filename;
    hdfsFS m_fs;         // hadoop filesystem handle
    hdfsFile m_file;     // file handler
    bool m_seekable;     // this stream is seekable
    int m_options;       // FileOptions ored togther

public:
    hdfs_File(const std::string& filename, int fileOptions);
    ~hdfs_File();

    void Flush();

    bool CanSeek() const { return m_seekable; }
    size_t Size();
    uint64_t GetPosition();
    void SetPosition(uint64_t pos);

    void GetLine(std::string& str);
    void GetLines(std::vector<std::wstring>& lines);
    void GetLines(std::vector<std::string>& lines);

    // static helpers
    // test whether a file exists
    template<class String>
    static bool Exists(const String& filename);

    // make intermediate directories
    template<class String>
    static void MakeIntermediateDirs(const String& filename);

    // determine the directory and naked file name for a given pathname
    static std::wstring DirectoryPathOf(std::wstring path);
    static std::wstring FileNameOf(std::wstring path);

    // put operator for basic types
    template <typename T>
    hdfs_File& operator<<(T val)
    {
        {
            if (IsTextBased())
                fputText(m_file, val);
            else
                fput(m_file, val);
        }
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
        if (IsTextBased())
            fgetText(m_file, val);
        else
            fget(m_file, val);
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

    operator hdfs_File*() const { return this; }

    template <class ElemType>
    static std::vector<ElemType> LoadMatrixFromTextFile(const std::wstring& filePath, size_t& /*out*/ numRows, size_t& /*out*/ numCols);

    template <class ElemType>
    static std::vector<ElemType> LoadMatrixFromStringLiteral(const std::string& literal, size_t& /*out*/ numRows, size_t& /*out*/ numCols);

    // Read a label file.
    template <class LabelType>
    static void LoadLabelFile(const std::wstring& filePath, std::vector<LabelType>& retLabels)
    {
        File file(filePath, fileOptionsRead | fileOptionsText);

        LabelType str;
        retLabels.clear();
        while (!file.IsEOF())
        {
            file.GetLine(str);
            if (str.empty())
                if (file.IsEOF())
                    break;
                else
                    RuntimeError("LoadLabelFile: Invalid empty line in label file.");

            retLabels.push_back(trim(str));
        }
    }
};

}}}
