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
        public static Learner SGDLearner(IList<Parameter> parameters, TrainingParameterScheduleDouble learningRateSchedule, AdditionalLearningOptions additionalOptions)
        {
            ParameterVector parameterVector = Helper.AsParameterVector(parameters);
            return SGDLearner(parameterVector, learningRateSchedule, additionalOptions);
        }

        public static Function Combine(IList<Variable> operands, string name)
        {
            VariableVector operandVector = Helper.AsVariableVector(operands);
            return Combine(operandVector, name);
        }
    }
}
