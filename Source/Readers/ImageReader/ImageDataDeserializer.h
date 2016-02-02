//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once
#include <opencv2/core/mat.hpp>
#include "DataDeserializerBase.h"
#include "Config.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// Image data deserializer based on the OpenCV library.
// The deserializer currently supports two output streams only: a feature and a label stream.
// All sequences consist only of a single sample (image/label).
// For features it uses dense storage format with different layout (dimensions) per sequence.
// For labels it uses the csc sparse storage format.
class ImageDataDeserializer : public DataDeserializerBase
{
public:
    explicit ImageDataDeserializer(const ConfigParameters& config);

    // Description of streams that this data deserializer provides.
    std::vector<StreamDescriptionPtr> GetStreamDescriptions() const override;

    // Get sequences by specified ids. Order of returned sequences corresponds to the order of provided ids.
    std::vector<std::vector<SequenceDataPtr>> GetSequencesById(const std::vector<size_t>& ids) override;

protected:
    void FillSequenceDescriptions(SequenceDescriptions& timeline) const override;

private:
    // Creates a set of sequence descriptions.
    void CreateSequenceDescriptions(std::string mapPath, size_t labelDimension);

    // Image sequence descriptions. Currently, a sequence contains a single sample only.
    struct ImageSequenceDescription : public SequenceDescription
    {
        std::string m_path;
        size_t m_classId;
    };

    // A helper class for generation of type specific labels (currently float/double only).
    class LabelGenerator;
    typedef std::shared_ptr<LabelGenerator> LabelGeneratorPtr;
    LabelGeneratorPtr m_labelGenerator;

    // Sequence descriptions for all input data.
    std::vector<ImageSequenceDescription> m_imageSequences;

    // Buffer to store label data.
    std::vector<SparseSequenceDataPtr> m_labels;

    // Buffer to store feature data.
    std::vector<cv::Mat> m_currentImages;

    // Element type of the feature/label stream (currently float/double only).
    ElementType m_featureElementType;
};

}}}
