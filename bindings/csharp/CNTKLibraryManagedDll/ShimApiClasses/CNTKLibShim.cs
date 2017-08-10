//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// CNTKLibShim.cs -- General C# Api methods
//
using System.Collections.Generic;

namespace CNTK
{
    public partial class CNTKLib
    {
        /// <summary>
        /// Create an instance of the CNTK built-in SGD learner.
        /// </summary>
        /// <param name="parameters">Parameters of the learner.</param>
        /// <param name="learningRateSchedule">Learning rate schedule.</param>
        /// <param name="additionalOptions">Additional options.</param>
        /// <returns></returns>
        public static Learner SGDLearner(IList<Parameter> parameters, TrainingParameterScheduleDouble learningRateSchedule, AdditionalLearningOptions additionalOptions)
        {
            ParameterVector parameterVector = Helper.AsParameterVector(parameters);
            return SGDLearner(parameterVector, learningRateSchedule, additionalOptions);
        }
    }
}
