#pragma once
#ifdef SWIGCSHARP
#include "CNTKLibrary.h"

namespace CNTK
{
    class MomentumAsTimeConstantScheduleCS : public TrainingParameterSchedule<double>
    {
    public:
        MomentumAsTimeConstantScheduleCS(double value)
            : TrainingParameterSchedule<double>::TrainingParameterSchedule(value, UnitType::Sample)
        {
            ConvertToPerSampleValues();
        }

        MomentumAsTimeConstantScheduleCS(const std::vector<double>& schedule, size_t epochSize = FullDataSweep)
            : TrainingParameterSchedule<double>::TrainingParameterSchedule(schedule, UnitType::Sample, epochSize)
        {
            ConvertToPerSampleValues();
        }

        MomentumAsTimeConstantScheduleCS(const std::vector<std::pair<size_t, double>>& schedule, size_t epochSize = FullDataSweep)
            : TrainingParameterSchedule<double>::TrainingParameterSchedule(schedule, UnitType::Sample, epochSize)
        {
            ConvertToPerSampleValues();
        }

    private:
        void ConvertToPerSampleValues()
        {
            for (auto& it : m_schedule)
            {
                double momTC = it.second;
                double momPS = momTC == 0.0 ? 0 : exp(-1.0 / momTC);
                it.second = momPS;
            }
        }
    };
}
#endif
