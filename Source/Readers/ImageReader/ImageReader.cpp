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
#include "HeapMemoryProvider.h"
#include "CudaMemoryProvider.h"
#include <opencv2/opencv.hpp>

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

    fprintf(stderr, "cv::useOptimized:%d\n", cv::useOptimized());
    fprintf(stderr, "cv::getNumberOfCPUs:%d\n", cv::getNumberOfCPUs());
    fprintf(stderr, "cv::getNumThreads:%d\n", cv::getNumThreads());

    // We multi-thread across OpenCV, so we let OpenCV only use one thread.
    cv::setNumThreads(1);

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
        // We do not use legacy randomization.
        bool useLegacyRandomization = false;
        // We do not do io prefetching, because chunks are single images currently.
        bool ioPrefetch = false;
        randomizer = std::make_shared<BlockRandomizer>(0, 1, deserializer, ioPrefetch, BlockRandomizer::DecimationMode::sequence, useLegacyRandomization, multithreadedGetNextSequences);
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
    transformations.push_back(Transformation{ std::make_shared<CastTransformer>(featureStream), featureName });

    m_sequenceEnumerator = std::make_shared<TransformController>(transformations, randomizer);

    m_packer = std::make_shared<FramePacker>(
        m_sequenceEnumerator,
        m_streams);
}

std::vector<StreamDescriptionPtr> ImageReader::GetStreamDescriptions()
{
    assert(!m_streams.empty());
    return m_streams;
}

void ImageReader::StartEpoch(const EpochConfiguration& config, const std::map<std::wstring, int>& inputDescriptions)
{
    if (config.m_totalEpochSizeInSamples == 0)
    {
        RuntimeError("Epoch size cannot be 0.");
    }

    if (inputDescriptions.size() != m_requiredInputs.size()
        || !std::equal(inputDescriptions.begin(), inputDescriptions.end(), m_requiredInputs.begin()))
    {
        m_requiredInputs = inputDescriptions;

        // Reallocating memory providers.
        m_memoryProviders.resize(m_streams.size());
        for (size_t i = 0; i < m_streams.size(); ++i)
        {
            int deviceId = m_requiredInputs[m_streams[i]->m_name];
            if (deviceId < 0)
                m_memoryProviders[i] = std::make_shared<HeapMemoryProvider>();
            else
                m_memoryProviders[i] = std::make_shared<CudaMemoryProvider>(deviceId);
        }
    }

    m_sequenceEnumerator->StartEpoch(config);
    m_packer->StartEpoch(config, m_memoryProviders);
}

Minibatch ImageReader::ReadMinibatch()
{
    assert(m_packer != nullptr);
    return m_packer->ReadMinibatch();
}
} } }
