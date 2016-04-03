//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
#pragma once

#include <chrono>
#include "TimerUtility.h"

namespace Microsoft { namespace MSR { namespace CNTK {

// TODO: make this proper C++ functions with variadic templates and a name that reflects their difference to fprintf(stderr) which already implies printing to log
// If the Tracing flag is set, print out a timestamp with no new line at the end
#define PREPENDTS(stream) \
    do \
    { \
        if (ProgressTracing::GetTimestampingFlag()) \
        { \
           std::time_t tt = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now()); \
           char mbstr[30]; \
           if (std::strftime(mbstr, sizeof(mbstr), "%m/%d/%Y %H:%M:%S", std::localtime(&tt))) \
               fprintf(stream, "%s: ", mbstr);  \
        } \
    } while(0)

// Print out a log message.  If the Tracing flag is set, prepend with a timestamp
#define LOGPRINTF(stream, ...) \
    do \
    { \
        PREPENDTS(stream); \
        fprintf(stream, __VA_ARGS__); \
    } while(0)

// ---------------------------------------------------------------------------
// ProgressTracing -- static helper class for logging a progress indicator
//
// This is for use by the cluster management tools for indicating global progress to the user.
//
// This logs to stdout (not stderr) in a specific format, e.g. understood by the Philly cluster. The format is:
//  PROGRESS xx.xx%
//  EVALERR xx.xx%
//
// Specifically, this class handles a two-level progress computation:
//  - outer level: loop over multiple training phases, each running multiple steps (epochs)
//  - inner level in one training phase: loop over multiple steps, *without* knowledge about the other training phases
//
// In order for the inner level to log correctly in the global context, the outer loop
// must inform this class about the total number of steps and the current offset to apply in the inner level.
// ---------------------------------------------------------------------------

/*static*/ class ProgressTracing
{
    bool m_enabled;
    bool m_tracingFlag;
    bool m_timestampFlag;        // TODO: What does this do? TODO: camelCase
    size_t m_totalNumberOfSteps; // total number of epochs in entire training run
    size_t m_currentStepOffset;  // current offset
    Timer m_progressTracingTimer;

    ProgressTracing()
        : m_enabled(false), m_tracingFlag(false), m_timestampFlag(false), m_totalNumberOfSteps(0), m_currentStepOffset(0)
    {
    }

    static ProgressTracing& GetStaticInstance()
    {
        static ProgressTracing us;
        return us;
    } // wrap static state in an accessor, so we won't need a CPP file

public:
    static bool GetTracingFlag()
    {
        return GetStaticInstance().m_tracingFlag;
    }

    static bool GetTimestampingFlag()
    {
        return GetStaticInstance().m_timestampFlag;
        // TODO: timestampFlag or timestampingFlag? (Or timeStampFlag?)
    }

    static void SetTracingFlag()
    {
        auto& us = GetStaticInstance();
        us.m_tracingFlag = true;
    }

    static void SetTimestampingFlag()
    {
        auto& us = GetStaticInstance();
        us.m_timestampFlag = true;
    }

    // call TraceTotalNumberOfSteps() to set the total number of steps
    // Calling this with totalNumberOfSteps>0 will enable progress tracing.
    static void TraceTotalNumberOfSteps(size_t totalNumberOfSteps)
    {
        auto& us = GetStaticInstance();
        us.m_enabled = totalNumberOfSteps > 0;
        if (us.m_enabled)
        {
            us.m_totalNumberOfSteps = totalNumberOfSteps;
            us.m_progressTracingTimer.Start();
        }
    }

    // call SetStepOffset() at start of a multi-epoch training to set the index of the first epoch in that training
    // This value is added to the local epoch index in TraceProgress().
    static void SetStepOffset(size_t currentStepOffset)
    {
        GetStaticInstance().m_currentStepOffset = currentStepOffset;
    }

    // emit the trace message for global progress
    // 'currentStep' will be offset by m_currentStepOffset.
    // This only prints of enough time (10s) has elapsed since last print, and the return value is 'true' if it did print.
    static bool TraceProgressPercentage(size_t epochNumber, double mbProg /*0..100*/, bool isTimerPaced)
    {
        auto& us = GetStaticInstance();
        if (!us.m_enabled)
        {
            return false;
        }

        // compute global progress
        bool needToPrint = us.m_progressTracingTimer.ElapsedSeconds() > 10;
        if (needToPrint || isTimerPaced)
        {
            double epochProg = ((100.0f * (double) (us.m_currentStepOffset + epochNumber)) / (double) us.m_totalNumberOfSteps);
            mbProg = (mbProg * 100.0f) / (double) us.m_totalNumberOfSteps;
            printf("PROGRESS: %.2f%%\n", epochProg + mbProg);
            us.m_progressTracingTimer.Restart();
        }
        return needToPrint;
    }

    // emit a trace message for the train loss value
    // The value is printed in percent.
    static void TraceTrainLoss(double err)
    {
        auto& us = GetStaticInstance();

        if (!us.m_enabled)
        {
            return;
        }

        printf("EVALERR: %.7f%%\n", err);
    }

    // This prints a PROGRESS message with a percentage value of 0 to prevent timeouts on Philly
    // when executing long running non-training operations like PreCompute, CV, Eval, and Write
    static size_t TraceFakeProgress(size_t numIterationsBeforePrintingProgress, size_t numItersSinceLastPrintOfProgress)
    {
        size_t newNumItersSinceLastPrintOfProgress = numItersSinceLastPrintOfProgress;
        if (GetTracingFlag())
        {
            newNumItersSinceLastPrintOfProgress++;
            if (newNumItersSinceLastPrintOfProgress >= numIterationsBeforePrintingProgress)
            {
                printf("PROGRESS: %.2f%%\n", 0.0f);
                newNumItersSinceLastPrintOfProgress = 0;
            }
        }

        return newNumItersSinceLastPrintOfProgress;
    }
};
} } }
