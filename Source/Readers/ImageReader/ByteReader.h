//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include <opencv2/core/mat.hpp>
#include "Config.h"
#ifdef USE_ZIP
#include <zip.h>
#include <unordered_map>
#include <memory>
#include "ConcStack.h"
#endif

namespace CNTK {

using MultiMap = std::map<std::string, std::vector<size_t>>;

class ByteReader
{
public:
    ByteReader() = default;
    virtual ~ByteReader() = default;

    virtual void Register(const MultiMap& sequences) = 0;
    virtual cv::Mat Read(size_t seqId, const std::string& path, bool grayscale) = 0;

    DISABLE_COPY_AND_MOVE(ByteReader);
};

class FileByteReader : public ByteReader
{
public:
    FileByteReader(const std::string& expandDirectory) : m_expandDirectory(expandDirectory)
    {}

    void Register(const MultiMap&) override {}
    cv::Mat Read(size_t seqId, const std::string& path, bool grayscale) override;

    std::string m_expandDirectory;
};


#ifdef USE_FACE_FILE

static const auto FACE_FILE_BYTE_READER_NAME = "FaceFileByteReader";
static const int LANDMARK_POINTS_NUMBER = 28;

static bool isCVReader = false;

class FaceFileByteReader : public ByteReader
{
public:

    FaceFileByteReader() :
        m_rand_flip(true),
        m_flip_landmark_id(27),
        m_landmark_id(27),
        m_batch_img_height(224),
        m_batch_img_width(224),
        m_raw_scale(385.f),
        m_relative_scale(2e-2f),
        m_relative_trans(1e-2f),
        m_bigFilesCount(0),
        m_cacheInfo(5000)
    {
        if (isCVReader)
        {
            m_relative_scale = 0;
            m_relative_trans = 0;
            m_rand_flip = false;
        }
        else
        {
            isCVReader = true;
        }
    }

    void Register(const MultiMap& sequences) override;
    cv::Mat Read(size_t seqId, const std::string& seqPath, bool grayscale) override;

    void SetBigFileId(size_t id)
    {
        if (m_bigFileIds.find(id) != m_bigFileIds.end())
        {
            return;
        }
        m_bigFileIds[id] = m_bigFilesCount++;
    }
    void SetDirectory(std::string str)
    {
        m_directory = str;
    }
    void SetExpandDirectory(std::string str)
    {
        m_expendDirectory = str;
    }

    struct FaceFileInfo
    {
        size_t Offset;
        float Landmarks[LANDMARK_POINTS_NUMBER * 2];
    };

private:

    void CacheFaceFileInfo(vector<FaceFileInfo> &container, const string& faceFile,
        const string& landmarkFile, int pointsCount);
    void CropAndScaleFaceImage(const cv::Mat &input_image, int input_width, int input_height,
        int input_channels, const float *facial_points, int points_num, cv::Mat &output_image);

    bool m_rand_flip;
    int m_flip_landmark_id;
    int m_landmark_id;
    int m_batch_img_height;
    int m_batch_img_width;
    float m_raw_scale;
    float m_relative_scale;
    float m_relative_trans;

    int m_bigFilesCount;
    std::string m_directory;
    std::string m_expendDirectory;

    std::unordered_map<size_t, int> m_bigFileIds;
    std::vector<std::vector<FaceFileInfo>> m_cacheInfo;
};

#ifdef USE_ZIP
class ZipFaceFileReader : public ByteReader
{
public:
    ZipFaceFileReader(const std::string& zipPath);
    ~ZipFaceFileReader();

    void Register(const MultiMap& sequences) override;
    cv::Mat Read(size_t seqId, const std::string& path, bool grayscale) override;

private:
    void CropAndScaleFaceImage(const cv::Mat &input_image, int input_width, int input_height,
        int input_channels, const float *facial_points, int points_num, cv::Mat &output_image);

    bool m_rand_flip;
    int m_flip_landmark_id;
    int m_landmark_id;
    int m_batch_img_height;
    int m_batch_img_width;
    float m_raw_scale;
    float m_relative_scale;
    float m_relative_trans;

    using ZipPtr = zip_t*;
    ZipPtr OpenZip();

    std::mutex m_readLocker;
    std::string m_zipPath;
    ZipPtr m_zipFile;
    std::unordered_map<size_t, std::pair<zip_uint64_t, zip_uint64_t>> m_seqIdToIndex;
    conc_stack<std::vector<unsigned char>> m_workspace;
};
#endif

#endif

#ifdef USE_ZIP
class ZipByteReader : public ByteReader
{
public:
    ZipByteReader(const std::string& zipPath);

    void Register(const std::map<std::string, std::vector<size_t>>& sequences) override;
    cv::Mat Read(size_t seqId, const std::string& path, bool grayscale) override;

private:
    using ZipPtr = std::unique_ptr<zip_t, void(*)(zip_t*)>;
    ZipPtr OpenZip();

    std::string m_zipPath;
    Microsoft::MSR::CNTK::conc_stack<ZipPtr> m_zips;
    std::unordered_map<size_t, std::pair<zip_uint64_t, zip_uint64_t>> m_seqIdToIndex;
    Microsoft::MSR::CNTK::conc_stack<std::vector<unsigned char>> m_workspace;
};
#endif

}
