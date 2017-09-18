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

            var input = CNTKLib.InputVariable(new int[] { 28, 28 }, DataType.Float);
            //  
            var model = Model(input, init: GlorotUniformInitialization(), activation: ReluActivation).
                Convolution2D(filterShape: Dim2(5, 5), numFilters: 8, strides: Dim2(2, 2), pad: new List<bool>() { true, true },
                    bias: true, initBias: 0, activation: ReluActivation, init: GlorotNormalInitialization(), 
                    reductionRank: 0, device: device, name: "first_conv").
                Dropout(dropoutRatio: 0.5).
                Convolution2D(filterShape: Dim2(5, 5), numFilters: 8, strides: Dim2(2, 2), pad: new List<bool>() { true, true },
                    bias: true, initBias: 0, activation: ReluActivation, init: GlorotNormalInitialization(),
                    reductionRank: 0, device: device, name: "first_conv2").
                Dropout(dropoutRatio: 0.5).
                Dense(outputClasses: 10, activation: null, device: device, name: "classify");

            var anotherModel = Model(input, activation: ReluActivation).
                Convolution2D(Dim2(3, 3), 5, ReluActivation, GlorotNormalInitialization(), new List<bool>() { true, true }, Dim2(2, 2),
                    true, 0, 0, device, "anotherModel");
            var fork1 = anotherModel.Dense(200, ReluActivation, device, "fork1");
            var joined = anotherModel.
                Dense(200, SigmoidActivation, device, "fork2").
                ElementWisePlus(fork1);
        }
    }
}
