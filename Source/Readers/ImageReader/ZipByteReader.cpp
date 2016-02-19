//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "ByteReader.h"

#ifdef USE_ZIP

namespace Microsoft { namespace MSR { namespace CNTK {

std::string GetZipError(int err)
{
    zip_error_t error;
    zip_error_init_with_code(&error, err);
    std::string errS(zip_error_strerror(&error));
    zip_error_fini(&error);
    return errS;
}

ZipByteReader::ZipByteReader(const std::string& zipPath)
    : m_zip(nullptr)
{
    assert(!zipPath.empty());
    int err = 0;
    m_zip = zip_open(zipPath.c_str(), 0, &err);
    if (ZIP_ER_OK != err)
        RuntimeError("Failed to open %s, zip library error: %s", zipPath.c_str(), GetZipError(err).c_str());
}

ZipByteReader::~ZipByteReader()
{
    if (m_zip != nullptr)
        zip_close(m_zip);
}

void ZipByteReader::Register(size_t seqId, const std::string& path)
{
    struct zip_stat stat;
    zip_stat_init(&stat);
    int err = zip_stat(m_zip, path.c_str(), 0, &stat);
    if (ZIP_ER_OK != err)
        RuntimeError("Failed to get file info of %s, zip library error: %s", path.c_str(), GetZipError(err).c_str());
    m_seqIdToIndex[seqId] = std::make_pair(stat.index, stat.size);
}

cv::Mat ZipByteReader::Read(size_t seqId, const std::string& path)
{
    // Find index of the file in .zip file.
    auto r = m_seqIdToIndex.find(seqId);
    if (r == m_seqIdToIndex.end())
        RuntimeError("Could not find file %s in the zip file, sequence id = %lu", path.c_str(), (long)seqId);

    zip_uint64_t index = std::get<0>((*r).second);
    zip_uint64_t size = std::get<1>((*r).second);

    auto contents = m_workspace.pop_or_create([size]() { return vector<unsigned char>(size); });
    if (contents.size() < size)
        contents.resize(size);
    zip_file *file = zip_fopen_index(m_zip, index, 0);
    assert(nullptr != file);
    if (nullptr == file)
    {
        RuntimeError("Could not open file %s in the zip file, sequence id = %lu, zip library error: %s",
                     path.c_str(), (long)seqId, GetZipError(zip_error_code_zip(zip_get_error(m_zip))).c_str());
    }
    assert(contents.size() >= size);
    zip_uint64_t bytesRead = zip_fread(file, contents.data(), size);
    assert(bytesRead == size);
    if (bytesRead != size)
    {
        zip_fclose(file);
        RuntimeError("Bytes read %lu != expected %lu while reading file %s",
                     (long)bytesRead, (long)size, path.c_str());
    }
    zip_fclose(file);
    cv::Mat img = cv::imdecode(cv::Mat(1, (int)size, CV_8UC1, contents.data()), cv::IMREAD_COLOR);
    m_workspace.push(std::move(contents));
    return img;
}
}}}

#endif