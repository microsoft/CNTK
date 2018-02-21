//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "CNTKLibrary.h"

namespace CNTK
{
    ///
    /// Base class for distributed learners.
    ///
    class DistributedLearnerBase : public DistributedLearner
    {
    public:
        Dictionary CreateCheckpoint() override;

        void RestoreFromCheckpoint(const Dictionary& checkpoint) override;

    protected:
        DistributedLearnerBase(DistributedCommunicatorPtr communicator, LearnerPtr learner, size_t distributeAfterSamples, bool convertSparseToDense=true);

        static void PrepaireZeroGradients(std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues);
        void ConvertToOrdered(const std::unordered_map<Parameter, NDArrayViewPtr>& gradientValues, std::vector<std::pair<Parameter, NDArrayViewPtr>>& result, std::unordered_map<Parameter, NDArrayViewPtr>* convertedGradientValues = nullptr);

        std::vector<std::pair<Parameter, NDArrayViewPtr>> m_gradientBuffer;
        std::vector<Parameter> m_parameters;

        bool m_convertSparseToDense;

        DistributedLearnerBase(const DistributedLearnerBase&) = delete; DistributedLearnerBase& operator=(const DistributedLearnerBase&) = delete; DistributedLearnerBase& operator=(DistributedLearnerBase&&) = delete; DistributedLearnerBase(DistributedLearnerBase&& other) = delete;
    };
}