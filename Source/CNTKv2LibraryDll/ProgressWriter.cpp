//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#include "stdafx.h"
#include "CNTKLibrary.h"
#include "Utils.h"

#include <chrono>

namespace CNTK
{
    class ProgressWriter::Impl
    {
    public:
        Impl(size_t updateWriteFrequency, size_t firstUpdatesToWrite)
            : m_frequency(updateWriteFrequency), m_firstN(firstUpdatesToWrite),
            m_totalUpdates(0), m_totalSummaries(0)
        {
            Reset();
        }

        template<typename OnWriteUpdateFunc>
        void Update(size_t samples, const ValuePtr& accumulatedLoss, const ValuePtr& accumulatedMetric,
                    OnWriteUpdateFunc callback)
        {
            if (samples == 0)
            {
                return;
            }

            m_samples.second += samples;
            m_updates.second++;
            m_totalUpdates++;
            
            if (ShouldWriteUpdate(m_updates.second))
            {
                // Time to output the accumulated updates.
                // Note that we take snapshot of the accumulated loss/metric only when we want to write.
                // We do it this way on purpose, since accumulated loss/metric may be stored on a GPU
                // and we want to minimize the number of GPU->CPU data transfers.
                if (accumulatedLoss)
                {
                    m_loss.second = accumulatedLoss->AsScalar<double>();
                }

                if (accumulatedMetric)
                {
                    m_metric.second = accumulatedMetric->AsScalar<double>();
                }

                callback(m_samples, m_updates, m_loss, m_metric);

                // Reset the window.
                m_loss.first = m_loss.second;
                m_metric.first = m_metric.second;
                m_samples.first = m_samples.second;
                m_updates.first = m_updates.second;
            }
        }

        template<typename OnWriteSummaryFunc>
        void WriteSummary(const ValuePtr& accumulatedLoss, const ValuePtr& accumulatedMetric,
                          OnWriteSummaryFunc callback)
        {
            if (accumulatedLoss && m_samples.second > 0)
            {
                m_loss.second = accumulatedLoss->AsScalar<double>();
            }

            if (accumulatedMetric && m_samples.second > 0)
            {
                m_metric.second = accumulatedMetric->AsScalar<double>();
            }

            m_totalSummaries++;
            auto now = std::chrono::high_resolution_clock::now();
            size_t durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(now - m_lastResetTime).count();

            callback(m_samples.second, m_updates.second, m_totalSummaries, m_loss.second, m_metric.second, durationMs);

            Reset();
        }

        size_t TotalUpdates() const
        {
            return m_totalUpdates;
        }

    private:
        bool ShouldWriteUpdate(size_t update) const
        {
            if (m_frequency == 0)
            {
                // Geometric schedule - write at every 2^(i) steps, with i = 1, 2, 3, ...
                return ((update + 1) & update) == 0;
            }

            // Arithmetic schedule - write at every m_frequency steps or if the update is one of the first m_firstN
            // updates.
            return update % m_frequency == 0 || update <= m_firstN;
        }

        void Reset()
        {
            m_loss = { 0.0, 0.0 };
            m_metric = { 0.0, 0.0 };
            m_samples = { 0, 0 };
            m_updates = { 0, 0 };
            m_lastResetTime = std::chrono::high_resolution_clock::now();
        }

        const size_t m_frequency;
        const size_t m_firstN;

        // (start, end) values in the current window to be reported.
        std::pair<double, double> m_loss;
        std::pair<double, double> m_metric;
        std::pair<size_t, size_t> m_samples;
        std::pair<size_t, size_t> m_updates;

        size_t m_totalUpdates;
        size_t m_totalSummaries;
        std::chrono::time_point<std::chrono::high_resolution_clock> m_lastResetTime;
    };

    ProgressWriter::ProgressWriter(size_t trainingUpdateWriteFrequency, size_t trainingFirstUpdatesToWrite,
                                   size_t testUpdateWriteFrequency, size_t testFirstUpdatesToWrite)
        : m_training(std::make_unique<Impl>(trainingUpdateWriteFrequency, trainingFirstUpdatesToWrite)),
        m_test(std::make_unique<Impl>(testUpdateWriteFrequency, testFirstUpdatesToWrite))
    {
    }

    ProgressWriter::~ProgressWriter()
    {
    }

    void ProgressWriter::UpdateTraining(size_t samples, const ValuePtr& accumulatedLoss,
                                        const ValuePtr& accumulatedMetric)
    {
        m_training->Update(samples, accumulatedLoss, accumulatedMetric,
            [this](const std::pair<size_t, size_t> samples, std::pair<size_t, size_t> updates,
                   const std::pair<double, double> aggregateLoss, std::pair<double, double> aggregateMetric)
            {
                OnWriteTrainingUpdate(samples, updates, aggregateLoss, aggregateMetric);
            });
        OnTrainingUpdateEnd();
    }

    void ProgressWriter::UpdateTest(size_t samples, const ValuePtr& accumulatedMetric)
    {
        m_test->Update(samples, nullptr, accumulatedMetric,
            [this](const std::pair<size_t, size_t> samples, std::pair<size_t, size_t> updates,
                   const std::pair<double, double> /*aggregateLoss*/, std::pair<double, double> aggregateMetric)
            {
                OnWriteTestUpdate(samples, updates, aggregateMetric);
            });
    }

    void ProgressWriter::WriteTrainingSummary(const ValuePtr& accumulatedLoss, const ValuePtr& accumulatedMetric)
    {
        m_training->WriteSummary(
            accumulatedLoss, accumulatedMetric,
            [this](size_t samples, size_t updates, size_t summaries, double aggregateLoss, double aggregateMetric,
                   uint64_t elapsedMs)
            {
                OnWriteTrainingSummary(samples, updates, summaries, aggregateLoss, aggregateMetric, elapsedMs);
            });
    }

    void ProgressWriter::WriteTestSummary(const ValuePtr& accumulatedMetric)
    {
        m_test->WriteSummary(
            nullptr, accumulatedMetric,
            [this](size_t samples, size_t updates, size_t summaries, double /*aggregateLoss*/, double aggregateMetric,
                uint64_t elapsedMs)
            {
                OnWriteTestSummary(samples, updates, summaries, aggregateMetric, elapsedMs);
            });
    }

    size_t ProgressWriter::TotalTrainingUpdates() const
    {
        return m_training->TotalUpdates();
    }

    size_t ProgressWriter::TotalTestUpdates() const
    {
        return m_test->TotalUpdates();
    }
}
