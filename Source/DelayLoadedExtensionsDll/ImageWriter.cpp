//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ImageWriter.cpp : Defines the exported functions for the DelayLoadedExtensions DLL.
//

#define IMAGEWRITER_EXPORTS // creating the exports here
#include "ImageWriter.h"
#include <opencv2/opencv.hpp>
#include <opencv2/core/mat.hpp>
//#include <opencv2/imgcodecs.hpp>
//#include <opencv2/imgproc/imgproc_c.h>

namespace Microsoft { namespace MSR { namespace CNTK {

// Ensure OpenCV's imgproc library appears as direct dependency at link
// time so rpath will apply (Linux). // TODO find a better way
void _dummyRefForOpenCVImgProc()
{
    cvThreshHist(0, 0.0);
}

extern "C" IMAGEWRITER_API void EncodeImageAsPNG(void* matrix, ::CNTK::DataType dtype, int height, int width, int depth, std::vector<unsigned char>& buffer)
{
    assert(matrix != nullptr);
    assert(&buffer != nullptr);
    assert(dtype == ::CNTK::DataType::Float || dtype == ::CNTK::DataType::Double);

    int cvDataType = dtype == ::CNTK::DataType::Float ? CV_32FC(depth) : CV_64FC(depth);
    cv::Mat source = cv::Mat(height, width, cvDataType, matrix);
    std::vector<int> parameters = std::vector<int>(2);
    parameters[0] = CV_IMWRITE_PNG_COMPRESSION;
    //parameters[0] = cv::ImwriteFlags::IMWRITE_PNG_COMPRESSION;
    parameters[1] = 3; //default(3)  0-9

    if (!imencode(".png", source, buffer, parameters)) {
        fprintf(stderr, "ImageWriter: PNG encoding failed.");
    }
}

} } }
