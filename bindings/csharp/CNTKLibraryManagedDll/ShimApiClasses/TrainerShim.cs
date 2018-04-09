//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// TrainerShim.cs -- C# Api for CNTK Trainer class
//
using System.Collections.Generic;

namespace CNTK
{
    public partial class Trainer
    {
        /// <summary>
        /// construct a trainer
        /// </summary>
        /// <param name="model">model to train</param>
        /// <param name="lossFunction">loss function</param>
        /// <param name="evaluationFunction">evaluation function</param>
        /// <param name="parameterLearners">parameters to train</param>
        /// <param name="progressWriters">training updater</param>
        /// <returns></returns>
        public static Trainer CreateTrainer(Function model, Function lossFunction, Function evaluationFunction, IList<Learner> parameterLearners, 
            ProgressWriterVector progressWriters = null)
        {
            using (LearnerVector learnerVector = Helper.AsLearnerVector(parameterLearners))
                if (progressWriters != null)
                    return CNTKLib.CreateTrainer(model, lossFunction, evaluationFunction, learnerVector, progressWriters);
                else
                    return CNTKLib.CreateTrainer(model, lossFunction, evaluationFunction, learnerVector);
        }

        /// <summary>
        /// train with a minibatch data
        /// </summary>
        /// <param name="arguments">minibatch data as varaible minibatchdata pairs</param>
        /// <param name="computeDevice">device</param>
        /// <returns></returns>
        public bool TrainMinibatch(IDictionary<Variable, MinibatchData> arguments, DeviceDescriptor computeDevice)
        {
            using (UnorderedMapVariableMinibatchData vectorData = Helper.AsUnorderedMapVariableMinibatchData(arguments))
                return _TrainMinibatch(vectorData, computeDevice);
        }

        /// <summary>
        /// train with a minibatch data
        /// </summary>
        /// <param name="arguments">minibatch data as variable value pairs</param>
        /// <param name="computeDevice">device</param>
        /// <returns></returns>
        [System.Obsolete("TrainMinibatch() without isSweepEndInarguments will be deprecated soon. Please TrainMinibatch() with isSweepEndInarguments.", false)]
        public bool TrainMinibatch(IDictionary<Variable, Value> arguments, DeviceDescriptor computeDevice)
        {
            bool isSweepEndInarguments = false;
            using (UnorderedMapVariableValuePtr mapData = Helper.AsUnorderedMapVariableValue(arguments))
                return _TrainMinibatch(mapData, isSweepEndInarguments, computeDevice);
        }

        /// <summary>
        /// train with a minibatch data
        /// </summary>
        /// <param name="arguments">minibatch data as variable value pairs</param>
        /// <param name="isSweepEndInarguments">indicates whether the current minibatch data is the end of one sweep</param>
        /// <param name="computeDevice">device</param>
        /// <returns></returns>
        public bool TrainMinibatch(IDictionary<Variable, Value> arguments, bool isSweepEndInarguments, DeviceDescriptor computeDevice)
        {
            using (UnorderedMapVariableValuePtr mapData = Helper.AsUnorderedMapVariableValue(arguments))
                return _TrainMinibatch(mapData, isSweepEndInarguments, computeDevice);
        }
    }
}
