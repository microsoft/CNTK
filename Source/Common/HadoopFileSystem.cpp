//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include <linux/limits.h>
#include <string>
#include <stdlib.h>

#include "File.h"
#include "HadoopFileSystem.h"

#define HDFS_ERROR -1

using namespace std;

namespace Microsoft { namespace MSR { namespace CNTK {

hdfs_File::hdfs_File(const wstring filename, int fileOptions)
{
#ifdef USE_HDFS
    int loc = filename.find(":");
    string nn = filename.substr(0, loc);
    filename = filename.substr(1);
    loc = filename.find("/");
    int port = stoi(filename.substr(0, loc));
    m_filename = filename.substr(loc + 1);

    m_fs = hdfsConnect(nn, port);
    if (m_fs == nullptr)
    {
        RuntimeError("File: failed to connect to HDFS.")
    }

    const auto reading = !!(fileOptions & fileOptionsRead);
    const auto writing = !!(fileOptions & fileOptionsWrite);
    if (!reading && !writing)
        RuntimeError("File: either fileOptionsRead or fileOptionsWrite must be specified");
    m_seekable = reading ? true : false;
    m_options  = reading ? O_RDONLY : O_WRONLY;
    m_file = hdfsOpenFile(m_fs, m_filename, m_options, 0, 0, 0);
    if (m_file == nullptr)
    {
        RuntimeError("File: failed to open the file on HDFS.")
    }
#else
    RuntimeError("File: CNTK was built without HDFS support.");
#endif
}

#ifdef USE_HDFS
hdfs_File::~hdfs_File()
{
    if (hdfsDiconnect(m_fs) == HDFS_ERROR)
        RuntimeError("File: failed to disconnect with HDFS.");
}

void hdfs_File::Flush()
{
    if (hdfsFlush(m_fs, m_file) == HDFS_ERROR)
    {
        RuntimeError("File: failed to flush file to HDFS.");
    }
}

size_t hdfs_File::Size()
{
    hdfs_FileInfo* fileInfo = hdfsGetPathInfo(m_fs, m_filename);
    if (fileInfo == nullptr)
    {
        RuntimeError("File: failed to get File Info.");
    }
    return (size_t)fileInfo->mSize;
}

uint64_t hdfs_File::GetPosition()
{
    if (!CanSeek())
        RuntimeError("File: attempted to GetPosition() on non-seekable stream");
    return (uint64_t)hdfsTell(m_fs, m_file);
}

void hdfs_File::SetPosition(uint64_t pos)
{
    if (!CanSeek())
        RuntimeError("File: attempted to SetPosition() on non-seekable stream");
    if (hdfsSeek(m_fs, m_file, pos) == HDFS_ERROR)
        RuntimeError("File: failed to seek at the given position %llu", pos);
}

void hdfs_File::GetLine(string& str)
{
    // It is inefficient to read files on HDFS line by line
}

void GetLines(std::vector<std::string>& lines)
{
    // It is inefficient to read files on HDFS line by line
}

template<class String>
static bool Exists(const String& filename)
{
    int loc = filename.find(":");
    filename = filename.substr(1);
    loc = filename.find("/");
    string path = filename.substr(loc + 1);

    return (hdfsExists(m_fs, path) == 0) ? true : false;
}

// make intermediate directories
template<class String>
static void MakeIntermediateDirs(const String& filename)
{
    int loc = filename.find(":");
    filename = filename.substr(1);
    loc = filename.find("/");
    filename = filename.substr(loc + 1);
    loc = filename.find_last_of("/");
    if (hdfsCreateDirectory(m_fs, filename.substr(0, loc).c_str()) == -1)
        RuntimeError("File: failed to create the intermediate directory.");
}

// determine the directory and naked file name for a given pathname
static wstring DirectoryPathOf(wstring path)
{
    int loc = filename.find(L":");
    filename = filename.substr(1);
    loc = filename.find(L"/");
    filename = filename.substr(loc + 1);
    loc = filename.find_last_of(L"/");
    return filename.substr(0, loc);
}
static wstring FileNameOf(wstring path)
{
    auto loc = filename.find(L":");
    filename = filename.substr(1);
    loc = filename.find(L"/");
    filename = filename.substr(loc + 1);
    loc = filename.find_last_of(L"/");
    if (loc != filename.npos)
        return filename.substr(loc + 1);
    return filename;
}

#endif

}}}
