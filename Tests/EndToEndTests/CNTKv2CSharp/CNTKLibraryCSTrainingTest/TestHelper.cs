//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// TestHelper.cs -- Help functions for CNTK Library C# model training tests.
//
using CNTK;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace CNTK.CNTKLibraryCSTrainingTest
{
    public class TestHelper
    {
        public static Function FullyConnectedLinearLayer(Variable input, int outputDim, DeviceDescriptor device,
            string outputName = "")
        {
            System.Diagnostics.Debug.Assert(input.Shape.Rank == 1);
            int inputDim = input.Shape[0];

            int[] s = { outputDim, inputDim };
            var timesParam = new Parameter((NDShape)s, DataType.Float,
                CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1),
                device, "timesParam");
            var timesFunction = CNTKLib.Times(timesParam, input, "times");

            int[] s2 = { outputDim };
            var plusParam = new Parameter((NDShape)s2, 0.0f, device, "plusParam");
            return CNTKLib.Plus(plusParam, new Variable(timesFunction), outputName);
        }


        public static void SaveAndReloadModel(ref Function function, IList<Variable> variables, DeviceDescriptor device, uint rank = 0)
        {
            string tempModelPath = "feedForward.net" + rank;
            File.Delete(tempModelPath);

            IDictionary<string, Variable> inputVarUids = new Dictionary<string, Variable>();
            IDictionary<string, Variable> outputVarNames = new Dictionary<string, Variable>();

            foreach (var variable in variables)
            {
                if (variable.IsOutput)
                    outputVarNames.Add(variable.Owner.Name, variable);
                else
                    inputVarUids.Add(variable.Uid, variable);
            }

            function.Save(tempModelPath);
            function = Function.Load(tempModelPath, device);

            File.Delete(tempModelPath);

            var inputs = function.Inputs;
            foreach (var inputVarInfo in inputVarUids.ToList())
            {
                var newInputVar = inputs.First(v => v.Uid == inputVarInfo.Key);
                inputVarUids[inputVarInfo.Key] = newInputVar;
            }

            var outputs = function.Outputs;
            foreach (var outputVarInfo in outputVarNames.ToList())
            {
                var newOutputVar = outputs.First(v => v.Owner.Name == outputVarInfo.Key);
                outputVarNames[outputVarInfo.Key] = newOutputVar;
            }
        }

        public static void PrintTrainingProgress(Trainer trainer, int minibatchIdx, int outputFrequencyInMinibatches)
        {
            if ((minibatchIdx % outputFrequencyInMinibatches) == 0 && trainer.PreviousMinibatchSampleCount() != 0)
            {
                double trainLossValue = trainer.PreviousMinibatchLossAverage();
                double evaluationValue = trainer.PreviousMinibatchEvaluationAverage();
                Console.WriteLine($"Minibatch: {minibatchIdx} CrossEntropy loss = {trainLossValue}, Evaluation criterion = {evaluationValue}");
            }
        }
    }
}
