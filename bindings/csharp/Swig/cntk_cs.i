//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// cntk_cs.i -- SWIG Interface file for C#
//

%include "CNTKManagedCommon.i"

%extend CNTK::NDShape {
    // Swig generated .cxx code narrows size_t to unsigned long therefore special dimension values are lost.
    // For example, InferredDimension (value of -1), when passed to Cpp side with Swig generated code, 
    // becomes 4,294,967,295 instead of 9,223,372,036,854,775,807. This issue exists independent of whether
    // int(32bit) or long(64bit) is used for shape dimension in CSharp API.
    // This method is to bypass Swig generated code to maintain 64 bitness.
    static void CSharp_SizeTVector_AddExt(std::vector< size_t > *vectorSizeT, unsigned long long dim) 
    {
        (*vectorSizeT).push_back(dim);
    }
}

%extend CNTK::MinibatchSourceConfig {
    unsigned long long GetMaxSamples()
    {
        return self->maxSamples;
    }

    void SetMaxSamples(unsigned long long i_maxSamples)
    {
        self->maxSamples = i_maxSamples;
    }

    unsigned long long GetMaxSweeps()
    {
        return self->maxSweeps;
    }

    void SetMaxSweeps(unsigned long long i_maxSweeps)
    {
        self->maxSweeps = i_maxSweeps;
    }
}

%extend CNTK::NDArrayView {
    NDArrayView(const NDShape& viewShape, float *dataBuffer, size_t numBufferElements, const DeviceDescriptor& device, bool readOnly = false)
    {
        if (device.Type() == CNTK::DeviceKind::GPU)
        {
            CNTK::NDArrayView cpuView(viewShape, dataBuffer, numBufferElements, CNTK::DeviceDescriptor::CPUDevice(), readOnly);
            auto gpuView = new CNTK::NDArrayView(cpuView.GetDataType(), cpuView.GetStorageFormat(), viewShape, device);
            gpuView->CopyFrom(cpuView);
            return gpuView;
        }
        else
            return new CNTK::NDArrayView(viewShape, dataBuffer, numBufferElements, device, readOnly);
    }

    NDArrayView(const NDShape& viewShape, double *dataBuffer, size_t numBufferElements, const DeviceDescriptor& device, bool readOnly = false)
    {
        if (device.Type() == CNTK::DeviceKind::GPU)
        {
            CNTK::NDArrayView cpuView(viewShape, dataBuffer, numBufferElements, CNTK::DeviceDescriptor::CPUDevice(), readOnly);
            auto gpuView = new CNTK::NDArrayView(cpuView.GetDataType(), cpuView.GetStorageFormat(), viewShape, device);
            gpuView->CopyFrom(cpuView);
            return gpuView;
        }
        else
            return new CNTK::NDArrayView(viewShape, dataBuffer, numBufferElements, device, readOnly);
    }

    NDArrayView(const NDShape& viewShape, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const float* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly = false)
    {
        return new CNTK::NDArrayView(CNTK::DataType::Float, viewShape, colStarts, rowIndices, nonZeroValues, numNonZeroValues, device, readOnly);
    }

    NDArrayView(const NDShape& viewShape, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const double* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly = false)
    {
        return new CNTK::NDArrayView(CNTK::DataType::Double, viewShape, colStarts, rowIndices, nonZeroValues, numNonZeroValues, device, readOnly);
    }

    NDArrayView(const NDShape& viewShape, const SparseIndexType* colStarts, const SparseIndexType* rowIndices, const int8_t* nonZeroValues, size_t numNonZeroValues, const DeviceDescriptor& device, bool readOnly = false)
    {
        return new CNTK::NDArrayView(CNTK::DataType::Int8, viewShape, colStarts, rowIndices, nonZeroValues, numNonZeroValues, device, readOnly);
    }

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
        return CNTK::NDArrayView::RandomUniform<float>(shape, rangeStart, rangeEnd, seed, device);
    }

    static NDArrayViewPtr CNTK::NDArrayView::RandomUniformDouble(const NDShape& shape, double rangeStart, double rangeEnd, unsigned long seed, const DeviceDescriptor& device)
    {
        return CNTK::NDArrayView::RandomUniform<double>(shape, rangeStart, rangeEnd, seed, device);
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

// Here is the explanation of why we have to put MomentumAsTimeConstantScheduleCS
// in a different file (other than in CNTKLibrary.h) and do things in the following order:
// In order to keep class hierarchy of Swig generated MomentumAsTimeConstantScheduleCS class,
// we have to instantiate its templated base class first (via %template) and then to include
// class declaration of MomentumAsTimeConstantScheduleCS in MomentumAsTimeConstantScheduleCS.h.
// If class declaration is included
// before %template instantiation, MomentumAsTimeConstantScheduleCS will be generated
// without TrainingParameterScheduleDouble being its base class - because TrainingParameterScheduleDouble
// is not created yet.
