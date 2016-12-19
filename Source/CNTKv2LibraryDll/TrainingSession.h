//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#pragma once

#include "CNTKLibrary.h"

namespace CNTK
{
    class BasicTrainingSession : public TrainingSession
    {
    public:
        CNTK_API BasicTrainingSession(
            MinibatchSourcePtr trainingSource,
            TrainerPtr trainer,
            const std::unordered_map<Variable, StreamInformation>& modelInputToMinibatchSourceStream,
            const MinibatchSizeSchedule& minibatchSizeSchedule,
            size_t checkpointFrequencyInSamples,
            const std::wstring& checkPointFileName)
            : TrainingSession(trainingSource, trainer, modelInputToMinibatchSourceStream, checkpointFrequencyInSamples, checkPointFileName),
              m_minibatchSizeSchedule(minibatchSizeSchedule)
        {
            if (m_minibatchSizeSchedule.Unit() == MinibatchSizeSchedule::UnitType::Minibatch)
                LogicError("Currently CNTK only supports minibatch size schedule based on samples.");
        }

    protected:
        CNTK_API size_t GetMinibatchSize() override
        {
            return m_minibatchSizeSchedule[Trainer()->TotalNumberOfSamplesSeen()];
        }

    private:
        const MinibatchSizeSchedule m_minibatchSizeSchedule;
    };
}