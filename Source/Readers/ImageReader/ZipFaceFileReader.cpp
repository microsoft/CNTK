//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "ByteReader.h"
#include "ZipDecoder.h"

#ifdef USE_ZIP
#include <File.h>

namespace CNTK {

using namespace Microsoft::MSR::CNTK;

extern std::string GetZipError(int err);

ZipFaceFileReader::ZipFaceFileReader(const std::string& zipPath) :
    m_rand_flip(true),
    m_flip_landmark_id(27),
    m_landmark_id(27),
    m_batch_img_height(224),
    m_batch_img_width(224),
    m_raw_scale(385.f),
    m_relative_scale(2e-2f),
    m_relative_trans(1e-2f),
    m_zipPath(zipPath)
{
    assert(!m_zipPath.empty());

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

ZipFaceFileReader::~ZipFaceFileReader()
{
    assert(m_zipFile != nullptr);
    int err = fclose(m_zipFile);
    assert(0 == err);
#ifdef NDEBUG
    UNUSED(err);
#endif
}

ZipFaceFileReader::ZipPtr ZipFaceFileReader::OpenZip()
{
    auto zip = fopen(m_zipPath.c_str(), "rb");
    if (nullptr == zip)
        RuntimeError("Failed to open %s\n", m_zipPath.c_str());

    return zip;
}

static void Bilinear(const float* pInput, int iWidth, int iHeight, int iChannels, int iStrideW, int iStrideH, int iStrideC,
    float fX, float fY, float* pOutput, int iStrideOutput)
{
    // out of range
    if (fX < 0.f || fX >(float)(iWidth - 1) || fY < 0 || fY >(float)(iHeight - 1))
    {
        for (int i = 0; i < iChannels; i++)
            pOutput[i + iStrideOutput] = uchar(0);
        return;
    }

    // int pixels
    if (fX == (float) (int) (fX) && fY == (float) (int) (fY))
    {
        for (int i = 0; i < iChannels; i++)
            pOutput[i + iStrideOutput] = pInput[iStrideH*(int) fY + iStrideW*(int) fX + iStrideC*i];
        return;
    }

    // otherwise
    int iLeft = (int) floor(fX);
    int iRight = iLeft + 1;
    int iTop = (int) floor(fY);
    int iBottom = iTop + 1;

    float dX0 = fX - iLeft;
    float dX1 = iRight - fX;
    float dY0 = fY - iTop;
    float dY1 = iBottom - fY;

    iRight = iRight >= iWidth ? iWidth - 1 : iRight;
    iBottom = iBottom >= iHeight ? iHeight - 1 : iBottom;

    const auto *v00 = pInput + iTop*iStrideH + iLeft*iStrideW;
    const auto *v01 = pInput + iTop*iStrideH + iRight*iStrideW;
    const auto *v10 = pInput + iBottom*iStrideH + iLeft*iStrideW;
    const auto *v11 = pInput + iBottom*iStrideH + iRight*iStrideW;
    for (int i = 0; i < iChannels; i++)
    {
        pOutput[i + iStrideOutput] = (dX1*dY1*v00[i*iStrideC] + dX1*dY0*v10[i*iStrideC] + dX0*dY0*v11[i*iStrideC] + dX0*dY1*v01[i*iStrideC]);
    }
}

#define CHECK_GT(val1, val2)

void ZipFaceFileReader::CropAndScaleFaceImage(const cv::Mat &input_image, int input_width, int input_height, int input_channels,
    const float *facial_points, int points_num, cv::Mat &output_image)
{
    UNUSED(points_num);

    const int RAND_NUM = 4;
    float uniform_rand[RAND_NUM];
    for (int i = 0; i < RAND_NUM; i++)
        uniform_rand[i] = (rand() / (float) RAND_MAX) * 2.f - 1.f;
    float center_x, center_y, scale_x, scale_y;
    if (m_rand_flip && uniform_rand[0] >= 0)
    {
        CHECK_GT(points_num, m_flip_landmark_id);
        // flip lr
        center_x = facial_points[2 * m_flip_landmark_id] + (float) input_width * m_relative_trans * uniform_rand[1];
        center_y = facial_points[2 * m_flip_landmark_id + 1] + (float) input_height * m_relative_trans * uniform_rand[2];
        int resize_size = (int) ((float) m_raw_scale * (m_relative_scale * uniform_rand[3] + 1) + 0.5f);
        scale_x = -(float) (input_width - 1) / (float) (resize_size - 1);
        scale_y = (float) (input_height - 1) / (float) (resize_size - 1);
    }
    else
    {
        CHECK_GT(points_num, m_landmark_id);
        // original 
        center_x = facial_points[2 * m_landmark_id] + (float) input_width * m_relative_trans * uniform_rand[1];
        center_y = facial_points[2 * m_landmark_id + 1] + (float) input_height * m_relative_trans * uniform_rand[2];
        int resize_size = (int) ((float) m_raw_scale * (m_relative_scale * uniform_rand[3] + 1) + 0.5f);
        scale_x = (float) (input_width - 1) / (float) (resize_size - 1);
        scale_y = (float) (input_height - 1) / (float) (resize_size - 1);
    }

    // get crop and resize image
    auto dst = (float*) output_image.data;
    for (int y = 0; y < m_batch_img_height; y++)
    {
        float y_ori = ((float) y - (float) (m_batch_img_height - 1) / 2.f) * scale_y + center_y;
        for (int x = 0; x < m_batch_img_width; x++)
        {
            float x_ori = ((float) x - (float) (m_batch_img_width - 1) / 2.f) * scale_x + center_x;
            Bilinear((float*) input_image.data, input_width, input_height, input_channels, input_channels, input_width*input_channels, 1,
                x_ori, y_ori, dst, y * m_batch_img_width * 3 + x * 3);
        }
    }
}

void ZipFaceFileReader::Register(const MultiMap& sequences)
{
    m_zipFile = OpenZip();

    auto decoder = new ZipDecoder(m_zipFile);

    size_t numberOfEntries = 0;
    size_t numEntries = decoder->zip_get_entries();
    for (size_t i = 0; i < numEntries; ++i)
    {
        auto sequenceInfo = sequences.find(std::string(decoder->ZipInfo[i].zip_seq_name));
        
        if (sequenceInfo == sequences.end())
        {
            continue;
        }

        for (auto sid : sequenceInfo->second)
            m_seqIdToIndex[sid] = std::make_pair(decoder->ZipInfo[i].zip_seq_index, decoder->ZipInfo[i].zip_seq_size);
        numberOfEntries++;
    }
    delete decoder;

    if (numberOfEntries == sequences.size())
        return;

    // Not all sequences have been found. Let's print them out and throw.
    for (const auto& s : sequences)
    {
        for (const auto& id : s.second)
        {
            if (m_seqIdToIndex.find(id) == m_seqIdToIndex.end())
            {
                fprintf(stderr, "Sequence %s is not found in container %s.\n", s.first.c_str(), m_zipPath.c_str());
                break;
            }
        }
    }

    RuntimeError("Cannot retrieve image data for some sequences. For more detail, please see the log file.");
}

cv::Mat ZipFaceFileReader::Read(size_t seqId, const std::string& path, bool grayscale)
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

    { 
        std::lock_guard<std::mutex> g(m_readLocker);

        _fseeki64(m_zipFile, index, SEEK_SET);
        assert(contents.size() >= size);
        size_t bytesRead = fread(contents.data(), 1, size, m_zipFile);
        assert(bytesRead == size);
        if (bytesRead != size)
        {
            RuntimeError("Bytes read %lu != expected %lu while reading file %s",
                         (long)bytesRead, (long)size, path.c_str());
        }
    }

    float landmarks[LANDMARK_POINTS_NUMBER * 2];
    auto data = contents.data();

    for (int i = 0; i < LANDMARK_POINTS_NUMBER * 2; i++)
    {
        landmarks[i] = *((float*) data);
        data += sizeof(float);
    }

    vector<unsigned char> imageContents(contents.begin() + LANDMARK_POINTS_NUMBER * 2 * sizeof(float) + sizeof(int), contents.end());

    cv::Mat inImg;
    cv::imdecode(imageContents, grayscale ? cv::IMREAD_GRAYSCALE : cv::IMREAD_COLOR).convertTo(inImg, CV_32FC(3));;
    assert(nullptr != inImg.data);

    auto outImg = cv::Mat(m_batch_img_height, m_batch_img_width, CV_32FC(3));
    CropAndScaleFaceImage(inImg, inImg.cols, inImg.rows, inImg.channels(), landmarks, LANDMARK_POINTS_NUMBER, outImg);

    m_workspace.push(std::move(contents));
    return outImg;
}
}

#endif
