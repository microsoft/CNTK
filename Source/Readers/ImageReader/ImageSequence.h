#pragma once

#include <opencv2/core/mat.hpp>
#include "DataDeserializer.h"

namespace Microsoft { namespace MSR { namespace CNTK {

    // Used to keep track of the image. Used in the implementations of open cv transformers 
    // and deserializer. Accessed only using DenseSequenceData interface.
    struct ImageSequenceData : DenseSequenceData
    {
        cv::Mat m_image;

        // In case we do not copy data - we preserve original sequence.
        SequenceDataPtr m_original;
    };
}}}