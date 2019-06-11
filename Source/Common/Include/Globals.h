//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "ProgressTracing.h"
#include <atomic>

namespace Microsoft { namespace MSR { namespace CNTK {

    // Class containing global configuration for CNTK.
    class Globals
    {
    public:
        static void       ForceDeterministicAlgorithms() {        m_forceDeterministicAlgorithms = true; }
        static bool ShouldForceDeterministicAlgorithms() { return m_forceDeterministicAlgorithms; }

        static void       ForceConstantRandomSeed() {        m_forceConstantRandomSeed = true; }
        static bool ShouldForceConstantRandomSeed() { return m_forceConstantRandomSeed; }

        static void SetGradientAccumulationOptimization(bool enable) { m_optimizeGradientAccumulation = enable; }
        static bool ShouldOptimizeGradientAccumulation() { return m_optimizeGradientAccumulation; }

        // TODO: Currently the flag is set to false. Should be switched to true after more rigorous testing.
        static bool UseV2Aggregator() { return false; }

        static void SetShareNodeValueMatrices(bool enable) { m_enableShareNodeValueMatrices = enable; }
        static bool ShouldEnableShareNodeValueMatrices() { return m_enableShareNodeValueMatrices; }

        static void SetNodeTiming(bool enable) { m_enableNodeTiming = enable; }
        static bool ShouldEnableNodeTiming() { return m_enableNodeTiming; }

        static void SetMPIPackThreshold(std::size_t packThreholdInBytes) { m_mpiPackThresholdInBytes = packThreholdInBytes; }
        static std::size_t GetMPIPackThreshold() { return m_mpiPackThresholdInBytes; }

        static void SetUseBNMomentum(bool useBNMomentum) { m_useBNMomentum = useBNMomentum; }
        static bool GetUseBNMomentum() { return m_useBNMomentum; }

        static void SetBNMomentum(double BNMomentum) { m_BNMomentum = BNMomentum; }
        static double GetBNMomentum() { return m_BNMomentum; }
    private:
        static std::atomic<bool> m_forceDeterministicAlgorithms;
        // The global flag to enable matrices values in forward and backward prop
        static std::atomic<bool> m_enableShareNodeValueMatrices;
        static std::atomic<bool> m_forceConstantRandomSeed;
        static std::atomic<bool> m_optimizeGradientAccumulation;
        static std::atomic<bool> m_enableNodeTiming;
        static std::atomic<std::size_t> m_mpiPackThresholdInBytes;

        // If m_useBNMomentum == true, the BN Momentum will be overwrote as m_BNMomentum, regardless bnTimeConst.
        static std::atomic<bool> m_useBNMomentum;
        static std::atomic<double> m_BNMomentum;
    };
}}}
