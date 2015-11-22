//
// <copyright file="SGD.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include "TimerUtility.h"

namespace Microsoft { namespace MSR { namespace CNTK {

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
        size_t m_totalNumberOfSteps;        // total number of epochs in entire training run
        size_t m_currentStepOffset;         // current offset
        Timer m_progressTracingTimer;
        ProgressTracing() : m_enabled(false), m_totalNumberOfSteps(0), m_currentStepOffset(0) { }
        static ProgressTracing & GetStaticInstance() { static ProgressTracing us; return us; } // wrap static state in an accessor, so we won't need a CPP file
    public:
        // call TraceTotalNumberOfSteps() to set the total number of steps
        // Calling this with totalNumberOfSteps>0 will enable progress tracing.
        static void TraceTotalNumberOfSteps(size_t totalNumberOfSteps)
        {
            auto & us = GetStaticInstance();
            us.m_enabled = totalNumberOfSteps > 0;
            if (us.m_enabled)
            {
                us.m_totalNumberOfSteps = totalNumberOfSteps;
                us.m_progressTracingTimer.Start();
            }
        }
        // call SetStepOffset() at start of a multi-epoch training to set the index of the first epoch in that training
        // This value is added to the local epoch index in TraceProgress().
        static void SetStepOffset(size_t currentStepOffset) { GetStaticInstance().m_currentStepOffset = currentStepOffset; }
        // emit the trace message for global progress
        // 'currentStep' will be offset by m_currentStepOffset.
        // This only prints of enough time (10s) has elapsed since last print, and the return value is 'true' if it did print.
        static bool TraceProgressPercentage(size_t currentStep, double progressWithinStep/*0..1*/)
        {
            auto & us = GetStaticInstance();
            if (!us.m_enabled)
                return false;
            // in case we are not able to estimate, we will increase as needed
            // BUGBUG: This is a workaround because in BrainScript we cannot estimate the total number of epochs without actually running the actions.
            if (currentStep + 1 > us.m_totalNumberOfSteps)
                us.m_totalNumberOfSteps = currentStep + 1;
            // compute global progress
            bool needToPrint = us.m_progressTracingTimer.ElapsedSeconds() > 0;// 10;
            if (needToPrint)
            {
                size_t globalStep = currentStep + us.m_currentStepOffset;
                double globalStepPartial = (double)globalStep + progressWithinStep;
                double progress = globalStepPartial / us.m_totalNumberOfSteps;
                printf("PROGRESS: %.2f%%\n", 100.0 * progress);
                us.m_progressTracingTimer.Restart();
            }
            return needToPrint;
        }
        // emit a trace message for the error objective
        // The value is printed in percent.
        static void TraceObjectivePercentage(double err)
        {
            auto & us = GetStaticInstance();
            if (!us.m_enabled)
                return;
            printf("EVALERR: %.2f%%\n", 100.0 * err);
        }
    };

}}}
