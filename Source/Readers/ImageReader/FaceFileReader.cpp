//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include <opencv2/opencv.hpp>
#include "ByteReader.h"

#ifdef USE_FACE_FILE
#include <File.h>

namespace Microsoft { namespace MSR { namespace CNTK {

void FaceFileByteReader::CacheFaceFileInfo(vector<FaceFileInfo> &container, const string& faceFile, const string &landmarkFile, int pointsCount = LANDMARK_POINTS_NUMBER)
{
    FILE *pf = fopen(faceFile.c_str(), "rb");
    std::ifstream fin(landmarkFile.c_str());
    int tmp;
    fin >> tmp;
    while (!feof(pf))
    {
        FaceFileInfo info;

        // read big file
        int n_file_name;
        if (!fread(&n_file_name, sizeof(int), 1, pf)) break;
        (void) fseek(pf, n_file_name, SEEK_CUR);
        info.Offset = ftell(pf);

        int content_length;
        (void) fread(&content_length, sizeof(int), 1, pf);
        (void) fseek(pf, content_length, SEEK_CUR);

        // read points data
        fin >> tmp;
        fin >> tmp;
        for (int i = 0; i < pointsCount * 2; i++)
        {
            fin >> info.Landmarks[i];
        }

        // serialize
        container.push_back(std::move(info));
    }
    (void) fclose(pf);
    fin.close();
}

void FaceFileByteReader::Register(const std::map<std::string, size_t>& sequences)
{
    UNUSED(sequences);

    m_cacheInfo.resize(m_bigFileIds.size());

    std::stringstream tmpNameBuf;
    for (auto &pair : m_bigFileIds)
    {
        tmpNameBuf.str("");
        tmpNameBuf << m_directory << pair.first << ".big";

        CacheFaceFileInfo(m_cacheInfo[pair.second], tmpNameBuf.str(), tmpNameBuf.str() + ".pts");
    }
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

void FaceFileByteReader::CropAndScaleFaceImage(const cv::Mat &input_image, int input_width, int input_height, int input_channels,
    const float *facial_points, int points_num, cv::Mat &output_image)
{
    UNUSED(points_num);

    const int RAND_NUM = 4;
    float uniform_rand[RAND_NUM];
    for (int i = 0; i < RAND_NUM; i++)
        uniform_rand[i] = (rand() / (float)RAND_MAX) * 2.f - 1.f;
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

cv::Mat FaceFileByteReader::Read(size_t seqId, const std::string& path, bool grayscale)
{
    UNUSED(seqId);
    UNUSED(grayscale);

    assert(!path.empty());

    auto atPos = path.find_last_of('@');
    auto bigFilePath = path.substr(0, atPos);
    auto imageId = stoi(path.substr(atPos + 1));

    auto suffixPos = path.find_last_of(".big@");
    auto tmpPath = path.substr(0, suffixPos);
    std::replace(begin(tmpPath), end(tmpPath), '/', '\\');
    auto lastSlashPos = tmpPath.find_last_of('\\');

    size_t bigFileId;
    if (lastSlashPos != std::string::npos)
    {
        bigFileId = stoi(tmpPath.substr(lastSlashPos + 1));
    }
    else
    {
        bigFileId = stoi(tmpPath);
    }

    const auto &info = m_cacheInfo[m_bigFileIds[bigFileId]][imageId-1];

    FILE *pFile = fopen(bigFilePath.c_str(), "rb");
    fseek(pFile, (long) info.Offset, SEEK_SET);
    int content_length;
    fread(&content_length, sizeof(int), 1, pFile);
    vector<int8_t> content(content_length);
    fread(&content[0], content_length, 1, pFile);
    fclose(pFile);

    cv::Mat inImg;
    auto outImg = cv::Mat(m_batch_img_height, m_batch_img_width, CV_32FC(3));
    cv::imdecode(content, cv::IMREAD_COLOR).convertTo(inImg, CV_32FC(3));

    CropAndScaleFaceImage(inImg, inImg.cols, inImg.rows, inImg.channels(), info.Landmarks, LANDMARK_POINTS_NUMBER, outImg);

    // TODO: what about grayscale?

    return outImg;
}

}}}

#endif