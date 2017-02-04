//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "ImageReader.h"
#include "Config.h"
#include "ImageConfigHelper.h"
#include "ImageTransformers.h"
#include "BlockRandomizer.h"
#include "NoRandomizer.h"
#include "ImageDataDeserializer.h"
#include "FramePacker.h"
#include <omp.h>
#include "TransformController.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// TODO: This class should go away eventually.
// TODO: The composition of packer + randomizer + different deserializers in a generic manner is done in the CompositeDataReader.
// TODO: Currently preserving this for backward compatibility with current configs.
ImageReader::ImageReader(const ConfigParameters& config)
    : m_seed(0)
{
    // In the future, deserializers and transformers will be dynamically loaded
    // from external libraries based on the configuration/brain script.
    // We will provide ability to implement the transformer and
    // deserializer interface not only in C++ but in scripting languages as well.

    ImageConfigHelper configHelper(config);
    m_streams = configHelper.GetStreams();
    assert(m_streams.size() == 2);

    int threadCount = configHelper.GetCpuThreadCount();
    if (threadCount > 0)
    {
        omp_set_num_threads(threadCount);
    }

    auto deserializer = std::make_shared<ImageDataDeserializer>(config);

    SequenceEnumeratorPtr randomizer;
    // Request multi-threaded randomizer operation to speed up CPU-intensive image-decoding and transformations.
    const bool multithreadedGetNextSequences = true;
    if (configHelper.ShouldRandomize())
    {
        // We do not do io prefetching, because chunks are single images currently.
        bool ioPrefetch = false;
        randomizer = std::make_shared<BlockRandomizer>(0, 1, deserializer, ioPrefetch, multithreadedGetNextSequences);
    }
    else
    {
        randomizer = std::make_shared<NoRandomizer>(deserializer, multithreadedGetNextSequences);
    }

    // Create transformations for a single feature stream.
    std::wstring featureName = m_streams[configHelper.GetFeatureStreamId()]->m_name;
    ConfigParameters featureStream = config(featureName);

    std::vector<Transformation> transformations;
    transformations.push_back(Transformation{ std::make_shared<CropTransformer>(featureStream), featureName });
    transformations.push_back(Transformation{ std::make_shared<ScaleTransformer>(featureStream), featureName });
    transformations.push_back(Transformation{ std::make_shared<ColorTransformer>(featureStream), featureName });
    transformations.push_back(Transformation{ std::make_shared<IntensityTransformer>(featureStream), featureName });
    transformations.push_back(Transformation{ std::make_shared<MeanTransformer>(featureStream), featureName });

    if (configHelper.GetDataFormat() == CHW)
    {
        transformations.push_back(Transformation{ std::make_shared<TransposeTransformer>(featureStream), featureName });
    }

    // We should always have cast at the end. 
    // It is noop if the matrix element type is already expected by the packer.
    transformations.push_back(Transformation{ std::make_shared<CastTransformer>(featureStream), featureName });

    m_sequenceEnumerator = std::make_shared<TransformController>(transformations, randomizer);
    bool useLocalTimeline = true;
    m_packer = std::make_shared<FramePacker>(
        m_sequenceEnumerator,
        m_streams,
        2 /* number of buffers*/,
        useLocalTimeline);
}

std::vector<StreamDescriptionPtr> ImageReader::GetStreamDescriptions()
{
    assert(!m_streams.empty());
    return m_streams;
}

} } }
