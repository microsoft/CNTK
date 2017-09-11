using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNTK;

using static Microsoft.MachineLearning.Cntk.CntkLib;
namespace CNTK.HighLevelAPI
{
    public class ModelExample
    {
        public void ExampleRun()
        {
            DeviceDescriptor device = DeviceDescriptor.CPUDevice;
            //var x = new InputVariable(new Dim(1, 28, 28));
            //var y = new InputVariable(new Dim(10));

            var model = Model(init: GlorotUniformInitialization(), activation: ReluActivation).
                Convolution2D(filterShape: Dim2(5, 5), numFilters: 8, strides: Dim2(2, 2), pad: true, device: device, name: "first_conv").
                Dropout(dropoutRatio: 0.5).
                Convolution2D(filterShape: Dim2(5, 5), numFilters: 16, strides: Dim2(2, 2), pad: true, device: device, name: "second_conv").
                Dropout(dropoutRatio: 0.5).
                Dense(outputClasses: 10, activation: null, device: device, name: "classify");

            var first_conv = model["first_conv"];
            // System.Console.WriteLine("Number of filters on " + first_conv.Name + " = " + first_conv.NumFilters);

            // TODO: shall be able to create 
            var model2 = Model(activation: ReluActivation).
                Convolution2D(Dim2(3, 3), 5, Dim2(2, 2), true, device, "cvlskdjflsdj");
            var fork1 = model2.Dense(200, ReluActivation, device, "fork1");
            var joined = model2.
                Dense(200, SigmoidActivation, device, "fork2").
                ElementWisePlus(fork1);
        }

        public void ExampleRun2(Variable input, NDShape outputShape, NDShape cellShape, DeviceDescriptor device)
        {
            Variable prevOutput = Variable.PlaceholderVariable(outputShape, input.DynamicAxes);
            Variable prevCellState = Variable.PlaceholderVariable(cellShape, input.DynamicAxes);

            int outputDim = prevOutput.Shape[0];
            int cellDim = prevCellState.Shape[0];

            DataType dataType = DataType.Float;

            Func<int, Parameter> createBiasParam = (dim) => new Parameter(new int[] { dim }, 0.01f, device, "");

            uint seed2 = 1;
            Func<int, Parameter> createProjectionParam = (oDim) =>
                new Parameter(new int[] { oDim, NDShape.InferredDimension }, dataType,
                CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), device);

            Func<int, Parameter> createDiagWeightParam = (dim) =>
                new Parameter(new int[] { dim }, dataType,
                CNTKLib.GlorotUniformInitializer(1.0, 1, 0, seed2++), device);

            Func<Variable> projectInput = () =>
                createBiasParam(cellDim) + (createProjectionParam(cellDim) * input);

            Variable[] combined = { input, prevOutput };

            // Input gate
            Function it =
                CNTKLib.Sigmoid(
                    (Variable)(createProjectionParam(cellDim) * (Function)combined) +
                    CNTKLib.ElementTimes(createDiagWeightParam(cellDim), prevCellState));
            Function bit = CNTKLib.ElementTimes(
                it,
                CNTKLib.Tanh(createProjectionParam(cellDim) * (Function)combined));

            // Forget-me-not gate
            Function ft = CNTKLib.Sigmoid(
                (Variable)(
                        createProjectionParam(cellDim) * (Function)combined) +
                        CNTKLib.ElementTimes(createDiagWeightParam(cellDim), prevCellState));
            Function bft = CNTKLib.ElementTimes(ft, prevCellState);

            Function ct = (Variable)bft + bit;

            // Output gate
            Function ot = CNTKLib.Sigmoid(
                (Variable)(createProjectionParam(cellDim) * (Function)combined) +
                CNTKLib.ElementTimes(createDiagWeightParam(cellDim), ct));
            Function ht = CNTKLib.ElementTimes(ot, CNTKLib.Tanh(ct));

            Function c = ct;
            Function h = (outputDim != cellDim) ? (createProjectionParam(outputDim) * ht) : ht;

            var actualDh = h[-1];
            var actualDc = c[-1];

            // Form the recurrence loop by replacing the dh and dc placeholders with the actualDh and actualDc
            h.ReplacePlaceholders(new Dictionary<Variable, Variable> { { prevOutput, actualDh }, { prevCellState, actualDc } });

            // return new Tuple<Function, Function>(h, c);
            Parameter convParam = null;
            NDShape poolingWindowShape = null;
            Function conv = CNTKLib.Pooling(CNTKLib.Convolution(convParam, input), PoolingType.Average, poolingWindowShape);
            Function resNetNode = CNTKLib.ReLU(CNTKLib.Plus(conv, input));




        }

    }
}
