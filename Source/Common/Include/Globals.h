//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

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

        static void SetHyperCompressMemory(bool enable) { m_enableHyperCompressMemory = enable; }
        static bool ShouldEnableHyperCompressMemory() { return m_enableHyperCompressMemory; }

    private:
        static std::atomic<bool> m_forceDeterministicAlgorithms;
        // The global flag to enable matrices values in forward and backward prop
        static std::atomic<bool> m_enableShareNodeValueMatrices;
        // The global flag to enable hyper memory compression 
        static std::atomic<bool> m_enableHyperCompressMemory;
        static std::atomic<bool> m_forceConstantRandomSeed;
        static std::atomic<bool> m_optimizeGradientAccumulation;
    };
}}}
