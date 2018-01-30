//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"
#include "BestGpu.h"
#include <mutex>
#include <memory>
#include <algorithm>
#include <CPUMatrix.h> // For CPUMatrix::SetNumThreads
#include <thread>
#include "GPUMatrix.h"
#include "Globals.h"
#include "PerformanceProfiler.h"
#include "MPIWrapper.h"
#include "EnvironmentUtil.h"
#include "Basics.h"
#include "ProgressTracing.h"
#include "buildinfo.h"
#include "Constants.h"

extern bool g_shareNodeValueMatrices;
using namespace Microsoft::MSR::CNTK;

namespace CNTK
{
    std::atomic<bool> s_checkedMode(false);
    void SetCheckedMode(bool enable)
    {
        s_checkedMode.store(enable);
    }

    bool GetCheckedMode()
    {
        return s_checkedMode.load();
    }

    namespace Internal
    {

        template <typename E>
        using SparseCSCDataTuple = std::tuple<const E*, const SparseIndexType*, const SparseIndexType*, size_t, NDArrayViewPtr>;

        static std::atomic_ullong s_nextUniqueId = ATOMIC_VAR_INIT(0);
        size_t NewUniqueId()
        {
            return s_nextUniqueId++;
        }
        static std::mutex s_fixedSeedMutex;
        static bool s_fixedRandomSeed = false;
        static std::atomic_ullong s_currentRandomSeed = ATOMIC_VAR_INIT(0);

        unsigned long GetRandomSeed()
        {
            return static_cast<unsigned long>(s_currentRandomSeed.load());
        }

        void SetFixedRandomSeed(unsigned long value)
        {
            std::unique_lock<std::mutex> lock(s_fixedSeedMutex);
            s_currentRandomSeed.store(value);
            s_fixedRandomSeed = true;
        }

        bool IsRandomSeedFixed()
        {
            std::unique_lock<std::mutex> lock(s_fixedSeedMutex);
            return s_fixedRandomSeed;
        }

        void ResetRandomSeed(unsigned long value)
        {
            std::unique_lock<std::mutex> lock(s_fixedSeedMutex);
            s_currentRandomSeed.store(value);
            s_fixedRandomSeed = false;
        }

        // This is used to generate a default seed value for random parameter initializer and also 
        // for stateful nodes (dropout, and both flavors of random sample). The 'perWorkerLocalValue' flag
        // indicates if the generated value should be identical across individual workers in distributed 
        // setting or if each worker should get a different seed value.        
        size_t GenerateRandomSeed(bool perWorkerLocalValue /*= false*/)
        {
            std::unique_lock<std::mutex> lock(s_fixedSeedMutex);
            
            if (s_fixedRandomSeed)
                return s_currentRandomSeed;
            

            if (!perWorkerLocalValue)
                return s_currentRandomSeed++;

            static size_t numWorkers = 1, rank = 0;
            static bool initialized = false;
            if (EnvironmentUtil::GetTotalNumberOfMPINodes() > 1 && !initialized)
            {
                DistributedCommunicatorPtr communicator = MPICommunicator();
                numWorkers = communicator->Workers().size();
                rank = communicator->CurrentWorker().m_globalRank;
                assert(numWorkers > 1);
            }

            initialized = true;
            return (numWorkers * s_currentRandomSeed++) + rank;
        }

        std::atomic<bool> s_reverseTensorShapesInErrorMessages(false);
        void EnableReversingTensorShapesInErrorMessages()
        {
            s_reverseTensorShapesInErrorMessages.store(true);
        }

        bool IsReversingTensorShapesInErrorMessagesEnabled()
        {
            return s_reverseTensorShapesInErrorMessages.load();
        }

        std::atomic<bool> s_alwaysAllowSettingDefaultDevice(false);
        void AlwaysAllowSettingDefaultDevice()
        {
            s_alwaysAllowSettingDefaultDevice.store(true);
        }

        bool IsSettingDefaultDeviceAlwaysAllowed()
        {
            return s_alwaysAllowSettingDefaultDevice.load();
        }

        std::atomic<bool> s_allowRenamingFunctions(false);
        void AllowRenamingFunctions()
        {
            s_allowRenamingFunctions.store(true);
        }

        bool IsRenamingFunctionsAllowed()
        {
            return s_allowRenamingFunctions.load();
        }

        std::atomic<bool> s_disableAutomaticUnpackingOfPackedValues(false);
        void SetAutomaticUnpackingOfPackedValues(bool disable)
        {
            s_disableAutomaticUnpackingOfPackedValues.store(disable);
        }

        bool IsAutomaticUnpackingOfPackedValuesDisabled()
        {
            return s_disableAutomaticUnpackingOfPackedValues.load();
        }

        void EnableForwardValuesSharing()
        {
            Microsoft::MSR::CNTK::Globals::SetShareNodeValueMatrices(/* enable = */ true);
        }

        void DisableForwardValuesSharing()
        {
            Microsoft::MSR::CNTK::Globals::SetShareNodeValueMatrices(/* enable = */ false);
        }

        void EnableGradientAccumulationOptimization()
        {
            Microsoft::MSR::CNTK::Globals::SetGradientAccumulationOptimization(/* enable = */ true);
        }

        void DisableGradientAccumulationOptimization()
        {
            Microsoft::MSR::CNTK::Globals::SetGradientAccumulationOptimization(/* enable = */ false);
        }

        void StartProfiler(const wstring& profilerDir, bool profilerSyncGpu, size_t profilerBufferSize)
        {
#ifndef CNTK_UWP
            std::wstring logSuffix = L"";
            auto mpi = Microsoft::MSR::CNTK::MPIWrapper::GetInstance();
            if (mpi)
            {
                logSuffix = std::to_wstring(mpi->CurrentNodeRank());
            }

            Microsoft::MSR::CNTK::ProfilerInit(
                profilerDir,
                profilerBufferSize,
                logSuffix,
                profilerSyncGpu);
#endif
        }

        void EnableProfiler()
        {
#ifndef CNTK_UWP
            Microsoft::MSR::CNTK::ProfilerEnable(true);
#endif
        }

        void DisableProfiler()
        {
#ifndef CNTK_UWP
            Microsoft::MSR::CNTK::ProfilerEnable(false);
#endif
        }

        void StopProfiler()
        {
#ifndef CNTK_UWP
            Microsoft::MSR::CNTK::ProfilerClose();
#endif
        }

        bool AreEquivalent(const Variable& var1, const Variable& var2, bool allowParameterAndConstantsEquivalence)
        {
            bool areDynamicAxesCompatible = (var1.DynamicAxes().size() == var2.DynamicAxes().size());
            auto numAxes = var1.DynamicAxes().size();
            for (size_t i = 0; areDynamicAxesCompatible && (i < numAxes); ++i)
                areDynamicAxesCompatible = (var1.DynamicAxes()[i].IsOrdered() == var2.DynamicAxes()[i].IsOrdered());

            bool areVarKindsCompatible = (var1.Kind() == var2.Kind()) && (var1.NeedsGradient() == var2.NeedsGradient());


            if (!areVarKindsCompatible && allowParameterAndConstantsEquivalence)
            {
                areVarKindsCompatible = (var1.IsParameter() && var2.IsConstant()) || (var2.IsParameter() && var1.IsConstant());
            }

            return (areVarKindsCompatible &&
                    (var1.GetDataType() == var2.GetDataType()) &&
                    (var1.IsSparse() == var2.IsSparse()) &&
                    (var1.Name() == var2.Name()) &&
                    areDynamicAxesCompatible &&
                    ((var1.Shape() == var2.Shape()) || (AsTensorShape(var1.Shape()) == AsTensorShape(var2.Shape()))));
        }

        bool AreEquivalent(const FunctionPtr& f1, const FunctionPtr& f2, std::unordered_set<std::wstring>& uids)
        {
            if (f1 == f2)
            {
                return true;
            }

            if (uids.find(f1->Uid()) != uids.end())
            {
                return true;
            }
            else
            {
                uids.insert(f1->Uid());
            }

            if (f1->Name() != f2->Name())
            {
                return false;
            }

            if (f1->Attributes() != f2->Attributes())
            {
                return false;
            }

            auto outputs1 = f1->Outputs();
            auto outputs2 = f2->Outputs();

            if (outputs1.size() != outputs2.size())
            {
                return false;
            }

            for (int i = 0; i < outputs1.size(); ++i)
            {
                if (!AreEquivalent(outputs1[i], outputs2[i]))
                {
                    return false;
                }
            }

            auto inputs1 = f1->Inputs();
            auto inputs2 = f2->Inputs();

            if (inputs1.size() != inputs2.size())
            {
                return false;
            }

            for (int i = 0; i < inputs1.size(); ++i)
            {
                if (!AreEquivalent(inputs1[i], inputs2[i]))
                {
                    return false;
                }

                if (inputs1[i].IsOutput() && !AreEquivalent(inputs1[i].Owner(), inputs2[i].Owner(), uids))
                {
                    return false;
                }
            }

            return true;
        }

        bool AreEquivalent(const FunctionPtr& f1, const FunctionPtr& f2)
        {
            std::unordered_set<std::wstring> uids;
            return AreEquivalent(f1, f2, uids);
        }

        template <typename ElementType>
        bool AreEqual(const ElementType* data1, const ElementType* data2, size_t numElements, double relativeTolerance, double absoluteTolerance)
        {
            for (size_t i = 0; i < numElements; ++i)
            {
                auto firstValue = data1[i];
                auto secondValue = data2[i];
                ElementType allowedTolerance = (std::max<ElementType>)(std::abs((ElementType)absoluteTolerance), std::abs(((ElementType)relativeTolerance) * firstValue));
                if (std::abs(firstValue - secondValue) > allowedTolerance)
                    return false;
            }

            return true;
        }

        template <typename ElementType>
        bool AreEqual(const SparseCSCDataTuple<ElementType>& t1, const SparseCSCDataTuple<ElementType>& t2, double relativeTolerance, double absoluteTolerance)
        {
            if (std::get<3>(t1) != std::get<3>(t2))
                return false;
            
            auto nnzCount = std::get<3>(t1);
            auto values1 = std::get<0>(t1);
            auto values2 = std::get<0>(t2);

            for (size_t i = 0; i < nnzCount; ++i)
            {
                auto firstValue = values1[i];
                auto secondValue = values2[i];
                ElementType allowedTolerance = (std::max<ElementType>)(std::abs((ElementType)absoluteTolerance), std::abs(((ElementType)relativeTolerance) * firstValue));
                if (std::abs(firstValue - secondValue) > allowedTolerance)
                    return false;
            }

            auto rowIndices1 = std::get<2>(t1);
            auto rowIndices2 = std::get<2>(t2);

            if (memcmp(rowIndices1, rowIndices2, nnzCount * sizeof(SparseIndexType)) != 0)
                return false;
            
            auto colIndices1 = std::get<1>(t1);
            auto colIndices2 = std::get<1>(t2);

            for (size_t i = 0; i < nnzCount; ++i)
            {
                if (colIndices1[i] != colIndices2[i])
                    return false;
                if (colIndices1[i] == nnzCount)
                    break;
            }

            return true;
        }

        template <typename ElementType>
        std::pair<ElementType*, NDArrayViewPtr> GetCPUDataPtr(const NDArrayView& view) 
        {
            auto deviceType = view.Device().Type();

            if (deviceType == DeviceKind::CPU)
                return{ const_cast<ElementType*>(view.DataBuffer<ElementType>()), nullptr };
            
            if (deviceType == DeviceKind::GPU) 
            {
                auto tempCPUDataView = view.DeepClone(DeviceDescriptor::CPUDevice());
                return{ tempCPUDataView->WritableDataBuffer<ElementType>(), tempCPUDataView };
            }
            
            LogicError("Invalid device type (%u).", (unsigned int)deviceType);
        }

        template <typename ElementType>
        SparseCSCDataTuple<ElementType> GetSparseCSCCPUDataPtr(const NDArrayView& view)
        {
            auto deviceType = view.Device().Type();

            if (deviceType == DeviceKind::CPU)
                return std::tuple_cat(view.SparseCSCDataBuffers<ElementType>(), std::make_tuple(nullptr));

            if (deviceType == DeviceKind::GPU)
            {
                auto tempCPUDataView = view.DeepClone(view.Device());
                tempCPUDataView->ChangeDevice(DeviceDescriptor::CPUDevice());
                auto result = GetSparseCSCCPUDataPtr<ElementType>(*tempCPUDataView);
                std::get<4>(result) = tempCPUDataView;
                return result;
            }

            LogicError("Invalid device type (%u).", (unsigned int)deviceType);
        }

        template <typename ElementType> 
        bool AreEqual(const NDArrayView& view1, const NDArrayView& view2, double relativeTolerance, double absoluteTolerance)
        {
            if (std::addressof(view1) == std::addressof(view2))
            {
                return true;
            }

            if (view1.GetDataType() != view2.GetDataType() ||
                view1.Shape() != view2.Shape() ||
                view1.IsSparse() != view2.IsSparse())
            {
                return false;
            }

            if (!view1.IsSparse()) 
            {
                CNTK::NDArrayViewPtr temp1CpuDataView, temp2CpuDataView;
                ElementType* data1;
                ElementType* data2;
                std::tie(data1, temp1CpuDataView) = GetCPUDataPtr<ElementType>(view1);
                std::tie(data2, temp2CpuDataView) = GetCPUDataPtr<ElementType>(view2);

                size_t numElements = view1.Shape().TotalSize();
                return AreEqual(data1, data2, numElements, relativeTolerance, absoluteTolerance);
            }
            else 
            {
                auto data1 = GetSparseCSCCPUDataPtr<ElementType>(view1);
                auto data2 = GetSparseCSCCPUDataPtr<ElementType>(view2);
                return AreEqual(data1, data2, relativeTolerance, absoluteTolerance);
            }
        }

        bool AreEqual(const NDArrayView& view1, const NDArrayView& view2, double relativeTolerance, double absoluteTolerance)
        {
            if (view1.GetDataType() == DataType::Float)
                return AreEqual<float>(view1, view2, relativeTolerance, absoluteTolerance);

            if (view1.GetDataType() == DataType::Double)
                return AreEqual<double>(view1, view2, relativeTolerance, absoluteTolerance);

            LogicError("AreEqual(NDArrayView): Unknown DataType.");
        }


        std::pair<const MaskKind*, NDMaskPtr> GetCPUDataPtr(const NDMask& mask)
        {
            if (mask.Device() == DeviceDescriptor::CPUDevice())
                return{ mask.DataBuffer(), nullptr };
            else
            {
                auto tempCPUMask = mask.DeepClone(DeviceDescriptor::CPUDevice());
                return{ tempCPUMask->DataBuffer(), tempCPUMask };
            }
        }

        bool AreEqual(const NDMask& mask1, const NDMask& mask2)
        {
            if (mask1.Shape() != mask2.Shape())
                return false;

            NDMaskPtr tempCPUMask1, tempCPUMask2;
            const MaskKind* mask1Data = nullptr;
            const MaskKind* mask2Data = nullptr;
            std::tie(mask1Data, tempCPUMask1) = GetCPUDataPtr(mask1);
            std::tie(mask2Data, tempCPUMask2) = GetCPUDataPtr(mask2);

            size_t numElements = mask1.Shape().TotalSize();
            for (size_t i = 0; i < numElements; ++i)
            {
                if (mask1Data[i] != mask2Data[i])
                    return false;
            }

            return true;
        }

        template <typename ElementType>
        bool AreEqual(const ::CNTK::Value& value1, const ::CNTK::Value& value2, double relativeTolerance, double absoluteTolerance)
        {
            if (std::addressof(value1) == std::addressof(value2))
                return true;

            // If neither of the values have mask, we just compare the Data
            if (!value1.Mask() && !value2.Mask())
                return AreEqual(*value1.Data(), *value2.Data(), relativeTolerance, absoluteTolerance);

            // Both or neither should have masks
            if ((!value1.Mask() && value2.Mask()) || (!value2.Mask() && value1.Mask()) || !AreEqual(*value1.Mask(), *value2.Mask()))
                return false;

            if ((value1.GetDataType() != value2.GetDataType()) || (value1.Shape() != value2.Shape()))
                return false;

            NDMaskPtr tempCPUMask;
            const MaskKind* maskData;
            std::tie(maskData, tempCPUMask) = GetCPUDataPtr(*value1.Mask());

            CNTK::NDArrayViewPtr temp1CpuDataView, temp2CpuDataView;
            ElementType* data1;
            ElementType* data2;
            std::tie(data1, temp1CpuDataView) = GetCPUDataPtr<ElementType>(*value1.Data());
            std::tie(data2, temp2CpuDataView) = GetCPUDataPtr<ElementType>(*value2.Data());

            auto numMaskElements = value1.Mask()->Shape().TotalSize();
            auto numElementsPerMaskUnit = value1.Shape().TotalSize() / numMaskElements;
            for (size_t i = 0; i < numMaskElements; ++i)
            {
                if (maskData[i] != MaskKind::Invalid)
                {
                    if (!AreEqual(data1 + (i * numElementsPerMaskUnit), data2 + (i * numElementsPerMaskUnit), numElementsPerMaskUnit, relativeTolerance, absoluteTolerance))
                        return false;
                }
            }

            return true;
        }

        bool AreEqual(const ::CNTK::Value& value1, const ::CNTK::Value& value2, double relativeTolerance, double absoluteTolerance)
        {
            if (value1.GetDataType() == DataType::Float)
                return AreEqual<float>(value1, value2, relativeTolerance, absoluteTolerance);

            if (value1.GetDataType() == DataType::Double)
                return AreEqual<double>(value1, value2, relativeTolerance, absoluteTolerance);

            LogicError("AreEqual(Value): Unknown DataType.");
        }

        std::atomic<int> s_computationNetworkTraceLevel(0);
        void SetComputationNetworkTraceLevel(int traceLevel)
        {
            s_computationNetworkTraceLevel.store(traceLevel);
        }

        int GetComputationNetworkTraceLevel()
        {
            return s_computationNetworkTraceLevel.load();
        }

        void SetGPUMemoryAllocationTraceLevel(int traceLevel)
        {
            Microsoft::MSR::CNTK::TracingGPUMemoryAllocator::SetTraceLevel(traceLevel);
        }

        void SetMathLibTraceLevel(int traceLevel)
        {
            Microsoft::MSR::CNTK::SetMathLibTraceLevel(traceLevel);
        }

        void ForceDeterministicAlgorithms()
        {
            Microsoft::MSR::CNTK::Globals::ForceDeterministicAlgorithms();
        }

        bool ShouldForceDeterministicAlgorithms()
        {
            return Microsoft::MSR::CNTK::Globals::ShouldForceDeterministicAlgorithms();
        }

        void EnableSynchronousGPUKernelExecution()
        {
            SyncGuard::EnableSync();
        }

        bool IsSynchronousGPUKernelExecutionEnabled()
        {
            return SyncGuard::IsSyncEnabled();
        }

#ifdef CPUONLY
        // CPU SBC aggregation not implemented yet, so fall back to conversion of sparse to dense 
        std::atomic<bool> s_useSparseGradientAggregationInDataParallelSGD(false);
#else
        std::atomic<bool> s_useSparseGradientAggregationInDataParallelSGD(true);
#endif

        void UseSparseGradientAggregationInDataParallelSGD(bool enable)
        {
            s_useSparseGradientAggregationInDataParallelSGD = enable;
        }

        bool ShouldUseSparseGradientAggregationInDataParallelSGD()
        {
            return s_useSparseGradientAggregationInDataParallelSGD;
        }

        static std::atomic<bool> s_threadsAreSet(false);
        bool MaxNumCPUThreadsSet()
        {
            return s_threadsAreSet;
        }

        size_t DefaultPackThresholdSizeInBytes()
        {
            return DEFAULT_PACK_THRESHOLD_SIZE_IN_BYTES;
        }
    }

    std::atomic<TraceLevel> s_traceLevel(TraceLevel::Warning);
    void SetTraceLevel(TraceLevel value)
    {
        using namespace Internal;

        auto previousValue = s_traceLevel.exchange(value);

        if (previousValue == value)
            return;

        if (value == TraceLevel::Info)
        {
            // V1 does not have an intermediate trace level,
            // the logging is either disabled (trace level = 0)
            // or enabled (trace level != 0);
            SetComputationNetworkTraceLevel(int(value));
            SetMathLibTraceLevel(int(value));
        }
        else if (previousValue == TraceLevel::Info)
        {
            SetComputationNetworkTraceLevel(0);
            SetMathLibTraceLevel(0);
        }
    }

    TraceLevel GetTraceLevel()
    {
        return s_traceLevel.load();
    }

    /*static*/ std::mutex DeviceDescriptor::s_mutex;
    /*static*/ bool DeviceDescriptor::s_defaultDeviceFrozen(false);
    /*static*/ std::unique_ptr<DeviceDescriptor> DeviceDescriptor::s_defaultDevice(nullptr);
    /*static*/ std::vector<DeviceDescriptor> DeviceDescriptor::s_excludedDevices;
    /*static*/ std::vector<DeviceDescriptor> DeviceDescriptor::s_allDevices;
    /*static*/ std::vector<GPUProperties> DeviceDescriptor::s_gpuProperties;

    static std::once_flag s_initAllDevicesFlag;

    /*static*/ void DeviceDescriptor::Reset()
    {
        DeviceDescriptor::s_defaultDevice.reset(nullptr);
        DeviceDescriptor::s_defaultDeviceFrozen = false;
        DeviceDescriptor::s_excludedDevices.clear();
    }

    bool DeviceDescriptor::IsLocked() const
    {
        return Microsoft::MSR::CNTK::IsLocked(AsCNTKImplDeviceId(*this));
    }

    /*static*/ DeviceDescriptor DeviceDescriptor::UseDefaultDevice()
    {
        std::unique_lock<std::mutex> lock(s_mutex);

        if (!s_defaultDeviceFrozen && s_defaultDevice == nullptr)
        {
            if (GetTraceLevel() >= TraceLevel::Info) 
            {
                fprintf(stderr, "Auto-selecting process wide default device.\n");
            }

            // This will both initialize the list of available devices and log the device stats
            // (including the info on which devices are compatible and eligible for selection).
            const auto& allDevices = AllDevices();
            UNUSED(allDevices);

            vector<int> excludedIds;
            for (auto device : s_excludedDevices)
            {
                excludedIds.push_back(AsCNTKImplDeviceId(device));
            }

            auto id = Microsoft::MSR::CNTK::GetBestDevice(excludedIds);
            auto selectedDevice = id >= 0 ? DeviceDescriptor::GPUDevice(id) : DeviceDescriptor::CPUDevice();
            s_defaultDevice.reset(new DeviceDescriptor(selectedDevice));
        }

        if (!s_defaultDeviceFrozen)
        {
            fprintf(stderr, "Selected %S as the process wide default device.\n", s_defaultDevice->AsString().c_str());
        }

        s_defaultDeviceFrozen = true;

        return *s_defaultDevice;
    }

    /*static*/ bool DeviceDescriptor::TrySetDefaultDevice(const DeviceDescriptor& newDefaultDevice, bool acquireDeviceLock)
    {
        std::unique_lock<std::mutex> lock(s_mutex);

        if (s_defaultDevice != nullptr && newDefaultDevice == *s_defaultDevice)
            return !acquireDeviceLock || Microsoft::MSR::CNTK::TryLock(AsCNTKImplDeviceId(newDefaultDevice));

        // As a testing backdoor we allow changing the default device even after being "used/frozen"
        if (!Internal::IsSettingDefaultDeviceAlwaysAllowed() && s_defaultDeviceFrozen)
            // TODO: alternatively, print a warning and return false.
        {
            RuntimeError("Process wide default device cannot be changed since it has been frozen by being implicitly used "
                         "as the default device in a CNTK API call; Current default = %S, New default = %S.",
                         s_defaultDevice->AsString().c_str(), newDefaultDevice.AsString().c_str());
        }

        if (std::find(s_excludedDevices.begin(), s_excludedDevices.end(), newDefaultDevice) != s_excludedDevices.end())
            return false;

        if (acquireDeviceLock && !Microsoft::MSR::CNTK::TryLock(AsCNTKImplDeviceId(newDefaultDevice)))
            return false;

        s_defaultDevice.reset(new DeviceDescriptor(newDefaultDevice));

        if (!acquireDeviceLock)
            Microsoft::MSR::CNTK::ReleaseLock();

        return true;
    }

    /*static*/ void DeviceDescriptor::SetExcludedDevices(const std::vector<DeviceDescriptor>& excluded)
    {
        std::unique_lock<std::mutex> lock(s_mutex);
        s_excludedDevices = excluded;
    }
    
    /*static*/ const std::vector<DeviceDescriptor>& DeviceDescriptor::AllDevices()
    {
        using namespace Microsoft::MSR::CNTK;

        std::call_once(s_initAllDevicesFlag, [&]
        {
#ifndef CPUONLY
            auto allGpusData = GetAllGpusData();

            if (GetTraceLevel() >= TraceLevel::Info)
            {
                Internal::PrintGpuInfo(allGpusData);
            }

            for (const auto& gpuData : allGpusData)
            {
                if (gpuData.validity == GpuValidity::Valid)
                {
                    s_allDevices.push_back(DeviceDescriptor((unsigned int) gpuData.deviceId, DeviceKind::GPU));
                    s_gpuProperties.push_back(
                    {
                        (unsigned int)gpuData.deviceId, 
                        gpuData.versionMajor,
                        gpuData.versionMinor,
                        gpuData.cudaCores,
                        gpuData.name,
                        gpuData.totalMemory,
                    });
                }
            }
#endif
            s_allDevices.push_back(DeviceDescriptor::CPUDevice());
        });

        return s_allDevices;
    }

    std::wstring DeviceDescriptor::AsString() const
    {
        std::wstring str = DeviceKindName(Type());
        if (Type() == DeviceKind::GPU)
        {
            auto props = GetGPUProperties(*this);
            std::wstring wname(props.name.begin(), props.name.end());
            str = str + L"[" + std::to_wstring(Id()) + L"] " + wname;
        }
        return str;
    }

    /*static*/ DeviceDescriptor DeviceDescriptor::GPUDevice(unsigned int deviceId) 
    {       
        const auto& allDevices = AllDevices();
       
        if (std::none_of(allDevices.begin(), allDevices.end(), 
            [deviceId](const DeviceDescriptor& device){ return device.Type() == DeviceKind::GPU && device.Id() == deviceId; }))
        {
            InvalidArgument("Specified GPU device id (%u) is invalid.", deviceId);
        }
        return { deviceId, DeviceKind::GPU };
    }

    /*static*/ const GPUProperties& DeviceDescriptor::GetGPUProperties(const DeviceDescriptor& device)
    {
        if (device.Type() == DeviceKind::CPU) 
            InvalidArgument("GPU properties cannot be obtained for a CPU device.");

        // Now, make sure that the device vectores are initialized.
        const auto& allDevices = AllDevices();
        UNUSED(allDevices);

        auto result = std::find_if(s_gpuProperties.begin(), s_gpuProperties.end(),
            [&device](const GPUProperties& props) { return device.Id() == props.deviceId; });

        if (result == s_gpuProperties.end())
            InvalidArgument("Could not find properties for the specified GPU device (id=%u).", device.Id());

        return *result;
    }

    /*static*/ const std::wstring Axis::StaticAxisNamePrefix = L"staticAxisIdx=";

    /*static*/ const int Axis::SentinelStaticAxisIndexValueForDynamicAxes = std::numeric_limits<int>::max();
    /*static*/ const int Axis::SentinelStaticAxisIndexValueForAllStaticAxes = std::numeric_limits<int>::max() - 1;
    /*static*/ const int Axis::SentinelStaticAxisIndexValueForUnknownAxes = std::numeric_limits<int>::max() - 2;
    /*static*/ const int Axis::SentinelEndStaticAxisIndexValue = std::numeric_limits<int>::max() - 3;
    /*static*/ const int Axis::SentinelStaticAxisIndexValueForAllAxes = std::numeric_limits<int>::max() - 4;

    /*static*/ Axis::UniqueDynamicAxesNames Axis::s_uniqueDynamicAxisNames;

    bool Axis::UniqueDynamicAxesNames::RegisterAxisName(const std::wstring& axisName)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        return m_allKnownDynamicAxisNames.insert(axisName).second;
    }

    const std::wstring& Axis::UniqueDynamicAxesNames::NewUniqueDynamicAxisName(const std::wstring& axisNamePrefix)
    {
        std::unique_lock<std::mutex> lock(m_mutex);
        if (m_allKnownDynamicAxisNames.find(axisNamePrefix) == m_allKnownDynamicAxisNames.end())
        {
            m_allKnownDynamicAxisNames.insert(axisNamePrefix);
            return axisNamePrefix;
        }

        for (size_t i = 1;; i++)
        {
            auto newDynamicAxisName = axisNamePrefix + std::to_wstring(i);
            if (m_allKnownDynamicAxisNames.find(newDynamicAxisName) == m_allKnownDynamicAxisNames.end())
            {
                m_allKnownDynamicAxisNames.insert(newDynamicAxisName);
                return *m_allKnownDynamicAxisNames.find(newDynamicAxisName);
            }
        }
    }

    static std::shared_ptr<std::vector<Axis>> s_defaultInputVariableDynamicAxes, s_unknownDynamicAxes;
    static std::once_flag s_initDefaultInputVariableDynamicAxesFlag, s_initUnknownDynamicAxesFlag;

    /*static*/ const std::vector<Axis>& Axis::DefaultInputVariableDynamicAxes() 
    {
      std::call_once(s_initDefaultInputVariableDynamicAxesFlag, []
      {
        s_defaultInputVariableDynamicAxes.reset(new std::vector<Axis>({ Axis::DefaultDynamicAxis(), Axis::DefaultBatchAxis() }));
      });
      return *s_defaultInputVariableDynamicAxes;
    }

    /*static*/ const std::vector<Axis>& Axis::UnknownDynamicAxes()
    {
      std::call_once(s_initUnknownDynamicAxesFlag, []
      {
        s_unknownDynamicAxes.reset(new std::vector<Axis>({ Axis(SentinelStaticAxisIndexValueForUnknownAxes) }));
      });
      return *s_unknownDynamicAxes;
    }

    /*static*/ const Axis& Axis::DefaultDynamicAxis()
    {
        static const Axis s_defaultDynamicAxis(L"defaultDynamicAxis");
        return s_defaultDynamicAxis;
    }

    /*static*/ const Axis& Axis::OperandSequenceAxis()
    {
        static const Axis s_operandSequenceAxis(L"__operandSequenceAxis");
        return s_operandSequenceAxis;
    }

    /*static*/ const Axis& Axis::DefaultBatchAxis()
    {
        static const Axis s_defaultBatchAxis(L"defaultBatchAxis", false);
        return s_defaultBatchAxis;
    }

    /*static*/ const Axis& Axis::AllStaticAxes()
    {
        static const Axis s_allStaticAxes(SentinelStaticAxisIndexValueForAllStaticAxes);
        return s_allStaticAxes;
    }

    /*static*/ const Axis& Axis::AllAxes()
    {
        static const Axis s_allAxes(SentinelStaticAxisIndexValueForAllAxes);
        return s_allAxes;
    }

    void Axis::RegisterAxisName(const std::wstring& axisName)
    {
        s_uniqueDynamicAxisNames.RegisterAxisName(axisName);
    }

    std::wstring Axis::AsString() const
    {
        std::wstringstream wss;
        wss << "Axis('";
        wss << m_name;
        wss << "')";

        return wss.str();
    }

    void SetMaxNumCPUThreads(size_t numCPUThreads)
    {
        Internal::s_threadsAreSet = true;
        Microsoft::MSR::CNTK::CPUMatrix<float>::SetNumThreads((int)numCPUThreads);
    }

    size_t GetMaxNumCPUThreads()
    {
        return Microsoft::MSR::CNTK::CPUMatrix<float>::GetMaxNumThreads();
    }

    static std::atomic<bool> s_defaultUnitGainValue(true);

    bool DefaultUnitGainValue() 
    {
        return s_defaultUnitGainValue;
    }

    void SetDefaultUnitGainValue(bool value) 
    {
        s_defaultUnitGainValue.store(value);
    }

    template <class E>
    __declspec_noreturn void ThrowFormatted(const char* format, ...)
    {
        va_list args;
        va_start(args, format);
        Microsoft::MSR::CNTK::ThrowFormattedVA<E>(format, args);
        va_end(args);
    }

    namespace Internal
    {
        void ExtractCUDAVersion(int version, int& major, int& minor, int& patch_level)
        {
            //e.g. #define CUDNN_VERSION    (CUDNN_MAJOR * 1000 + CUDNN_MINOR * 100 + CUDNN_PATCHLEVEL)
            major = version / 1000;
            minor = (version - major * 1000) / 100;
            patch_level = version % 100;
        }

        void PrintBuiltInfo()
        {
            LOGPRINTF(stderr, "-------------------------------------------------------------------\n");
            LOGPRINTF(stderr, "Build info: \n\n");
            LOGPRINTF(stderr, "\t\tBuilt time: %s %s\n", __DATE__, __TIME__);
            LOGPRINTF(stderr, "\t\tLast modified date: %s\n", __TIMESTAMP__);
#ifdef _BUILDTYPE_
            LOGPRINTF(stderr, "\t\tBuild type: %s\n", _BUILDTYPE_);
#endif
#ifdef _BUILDTARGET_
            LOGPRINTF(stderr, "\t\tBuild target: %s\n", _BUILDTARGET_);
#endif
#ifdef _WITH_1BITSGD_
            LOGPRINTF(stderr, "\t\tWith 1bit-SGD: %s\n", _WITH_1BITSGD_);
#endif
#ifdef _WITH_ASGD_
            LOGPRINTF(stderr, "\t\tWith ASGD: %s\n", _WITH_ASGD_);
#endif
#ifdef _MATHLIB_
            LOGPRINTF(stderr, "\t\tMath lib: %s\n", _MATHLIB_);
#endif
#ifdef _CUDA_PATH_
            int cudaVersion = 0;
            if (cudaRuntimeGetVersion(&cudaVersion) == cudaSuccess)
            {
                int major = 0, minor = 0, patchLevel = 0;
                ExtractCUDAVersion(cudaVersion, major, minor, patchLevel);
                LOGPRINTF(stderr, "\t\tCUDA version: %d.%d.%d\n", major, minor, patchLevel);
            }
#endif
#ifdef _CUDNN_PATH_
            size_t cudnnVersion = GetCUDNNVersion();
            int cudnnMajor = 0, cudnnMinor = 0, cudnnPatchLevel = 0;
            ExtractCUDAVersion(cudnnVersion, cudnnMajor, cudnnMinor, cudnnPatchLevel);
            LOGPRINTF(stderr, "\t\tCUDNN version: %d.%d.%d\n", cudnnMajor, cudnnMinor, cudnnPatchLevel);
#endif
#ifdef _GIT_EXIST
            LOGPRINTF(stderr, "\t\tBuild Branch: %s\n", _BUILDBRANCH_);
            LOGPRINTF(stderr, "\t\tBuild SHA1: %s\n", _BUILDSHA1_);
#endif
#ifdef _MPI_NAME_
            LOGPRINTF(stderr, "\t\tMPI distribution: %s\n", _MPI_NAME_);
#endif
#ifdef _MPI_VERSION_
            LOGPRINTF(stderr, "\t\tMPI version: %s\n", _MPI_VERSION_);
#endif
            LOGPRINTF(stderr, "-------------------------------------------------------------------\n");
        }

        // print gpu info for current gpu devices (e.g. Device[0]: cores = 2496; computeCapability = 5.2; type = "Quadro M4000"; total memory = 8192 MB; free memory = 8192 MB)
        void PrintGpuInfo(const std::vector<Microsoft::MSR::CNTK::GpuData>& gpusData)
        {
#ifndef CPUONLY
            if (gpusData.empty())
            {
                LOGPRINTF(stderr, "No GPUs found\n");
                return;
            }

            LOGPRINTF(stderr, "-------------------------------------------------------------------\n");
            LOGPRINTF(stderr, "GPU info:\n\n");

            for (const GpuData& data : gpusData)
            {
                LOGPRINTF(stderr, "\t\tDevice[%d]: cores = %d; computeCapability = %d.%d; type = \"%s\"; total memory = %lu MB; free memory = %lu MB\n",
                    data.deviceId, data.cudaCores, data.versionMajor, data.versionMinor, data.name.c_str(), (unsigned long)data.totalMemory, (unsigned long)data.freeMemory);
            }
            LOGPRINTF(stderr, "-------------------------------------------------------------------\n");
#endif
        }
    }

    template CNTK_API __declspec_noreturn void ThrowFormatted<std::runtime_error>(const char* format, ...);
    template CNTK_API __declspec_noreturn void ThrowFormatted<std::logic_error>(const char* format, ...);
    template CNTK_API __declspec_noreturn void ThrowFormatted<std::invalid_argument>(const char* format, ...);
}

