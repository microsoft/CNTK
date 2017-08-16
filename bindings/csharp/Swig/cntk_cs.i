//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// cntk_cs.i -- SWIG Interface file for C#
//

%include "CNTKManagedCommon.i"

%extend CNTK::NDArrayView {
    static NDArrayViewPtr CNTK::NDArrayView::RandomNormalFloat(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device)
    {
        return CNTK::NDArrayView::RandomNormal<float>(shape, mean, stdDev, seed, device);
    }

    static NDArrayViewPtr CNTK::NDArrayView::RandomNormalDouble(const NDShape& shape, double mean, double stdDev, unsigned long seed, const DeviceDescriptor& device)
    {
        return CNTK::NDArrayView::RandomNormal<double>(shape, mean, stdDev, seed, device);
    }

    static NDArrayViewPtr CNTK::NDArrayView::RandomUniformFloat(const NDShape& shape, double rangeStart, double rangeEnd, unsigned long seed, const DeviceDescriptor& device)
    {
        return CNTK::NDArrayView::RandomNormal<float>(shape, rangeStart, rangeEnd, seed, device);
    }

    static NDArrayViewPtr CNTK::NDArrayView::RandomUniformDouble(const NDShape& shape, double rangeStart, double rangeEnd, unsigned long seed, const DeviceDescriptor& device)
    {
        return CNTK::NDArrayView::RandomNormal<double>(shape, rangeStart, rangeEnd, seed, device);
    }
}

%extend CNTK::Constant {
    static CNTK::Constant CNTK::Constant::ScalarFloat(float value, const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::CPUDevice())
    {
        return CNTK::Constant::Scalar<float>(value, device);
    }

    static CNTK::Constant CNTK::Constant::ScalarDouble(double value, const CNTK::DeviceDescriptor& device = CNTK::DeviceDescriptor::CPUDevice())
    {
        return CNTK::Constant::Scalar<double>(value, device);
    }
}

%extend CNTK::MinibatchSource
{
    static std::shared_ptr<MinibatchSource> TextFormatMinibatchSourceInternal(const std::wstring& dataFilePath, const std::vector<CNTK::StreamConfiguration>& streamConfigs,
        unsigned long long epochSize = CNTK::MinibatchSource::InfinitelyRepeat,
        bool randomize = true,
        unsigned long long randomizationWindow = CNTK::MinibatchSource::DefaultRandomizationWindowInChunks,
        bool sampleBasedRandomizationWindow = false)
    {
        return CNTK::TextFormatMinibatchSource(dataFilePath, streamConfigs,
            epochSize, randomize, randomizationWindow, sampleBasedRandomizationWindow);
    }

    static unsigned long long GetFullDataSweep()
    {
        return CNTK::MinibatchSource::FullDataSweep;
    }

    static unsigned long long GetInfinitelyRepeat()
    {
        return CNTK::MinibatchSource::InfinitelyRepeat;
    }

    static unsigned long long GetDefaultRandomizationWindowInChunks()
    {
        return CNTK::MinibatchSource::DefaultRandomizationWindowInChunks;
    }
}

%include "CNTKLibraryInternals.h"
%include "CNTKLibrary.h"

%template(TrainingParameterScheduleDouble) CNTK::TrainingParameterSchedule<double>;
