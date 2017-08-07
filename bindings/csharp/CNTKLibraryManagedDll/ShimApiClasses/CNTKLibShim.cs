using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

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
