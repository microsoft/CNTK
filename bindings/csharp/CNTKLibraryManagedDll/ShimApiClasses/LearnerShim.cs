using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK
{
    public partial class Learner
    {
        /// <summary>
        /// Create an instance of the CNTK built-in SGD learner.
        /// </summary>
        /// <param name="parameters">Parameters of the learner.</param>
        /// <param name="learningRateSchedule">Learning rate schedule.</param>
        /// <param name="additionalOptions">Additional options.</param>
        /// <returns></returns>
        public static Learner SGDLearner(IList<Parameter> parameters, TrainingParameterScheduleDouble learningRateSchedule, AdditionalLearningOptions additionalOptions = null)
        {
            ParameterVector parameterVector = Helper.AsParameterVector(parameters);
            if (additionalOptions == null)
            {
                return CNTKLib.SGDLearner(parameterVector, learningRateSchedule);
            }
            else
            {
                return CNTKLib.SGDLearner(parameterVector, learningRateSchedule, additionalOptions);
            }
        }

        public static Learner MomentumSGDLearner(IList<Parameter> parameters, TrainingParameterScheduleDouble learningRateSchedule,
            TrainingParameterScheduleDouble momentumSchedule, bool unitGain, AdditionalLearningOptions additionalOptions = null)
        {
            if (additionalOptions == null)
            {
                additionalOptions = new AdditionalLearningOptions();
            }
            ParameterVector parameterVector = Helper.AsParameterVector(parameters);
            return CNTKLib.MomentumSGDLearner(parameterVector, learningRateSchedule, momentumSchedule, unitGain, additionalOptions);
        }
    }
}
