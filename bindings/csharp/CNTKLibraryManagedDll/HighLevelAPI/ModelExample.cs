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

    public class UnorderedSet : SWIGTYPE_p_std__unordered_setT_CNTK__Variable_t
    {
        public UnorderedSet() : base() { }
    }
    public class UserTimesFunction : Function
    {
        public static Function Create(Variable leftOperand, Variable rightOperand, CNTKDictionary attributes, string name)
        {
            VariableVector models = new VariableVector();
            models.Add(leftOperand);
            models.Add(rightOperand);
            return AsComposite(new UserTimesFunction(models, attributes, name));
        }

        public static Function Create(Variable leftOperand, Variable rightOperand, string name)
        {
            return Create(leftOperand, rightOperand, new CNTKDictionary(), name);
        }

        public UserTimesFunction(VariableVector models, CNTKDictionary attributes, string name) : 
            base(models, attributes, name)
        {

        }

        public override void Backward(BackPropState state, UnorderedMapVariableValuePtr rootGradientValues, UnorderedMapVariableValuePtr backPropagatedGradientValuesForInputs)
        {
            var leftInputVariable = Inputs[0];
            var rightInputVariable = Inputs[1];

            if (!backPropagatedGradientValuesForInputs.ContainsKey(rightInputVariable))
                throw new Exception("UserTimesFunction does not support computing gradient wrt right operand");

            var rightInputValue = state.SavedForwardPropValues()[rightInputVariable];
            var rootGradientValue = rootGradientValues[this.Output];

            TensorView<float> rootGradientDataTV = new TensorView<float>(rootGradientValue.Shape, rootGradientValue.Data, new List<Axis>() { Axis.DefaultBatchAxis() });
            TensorView<float> rightInputDataTV = new TensorView<float>(rightInputValue.Shape, rightInputValue.Data, new List<Axis>() { Axis.DefaultBatchAxis() });

            backPropagatedGradientValuesForInputs[leftInputVariable] = new Value((rootGradientDataTV * rightInputDataTV.Transpose()).Eval().Value());
        }

        protected override BackPropState Forward(VectorValuePtr inputValues, UnorderedMapVariableValuePtr outputs, DeviceDescriptor computeDevice, 
            SWIGTYPE_p_std__unordered_setT_CNTK__Variable_t outputsToRetainBackwardStateFor)
        {
            CalculateForward(inputValues, outputs);

            // Let's save the right input's Value in the BackPropSate to be used in the backward pass for computing gradients
            UnorderedMapVariableValuePtr forwardPropValuesToSave = new UnorderedMapVariableValuePtr();
            forwardPropValuesToSave.Add(new KeyValuePair<Variable, Value>(Inputs[1], inputValues[1]));
            return new BackPropState(this, computeDevice, forwardPropValuesToSave);
        }

        protected override BackPropState Forward(VectorValuePtr inputValues, UnorderedMapVariableValuePtr outputs, DeviceDescriptor computeDevice)
        {
            CalculateForward(inputValues, outputs);

            UnorderedMapVariableValuePtr forwardPropValuesToSave = new UnorderedMapVariableValuePtr();
            return new BackPropState(this, computeDevice, forwardPropValuesToSave);
        }

        private void CalculateForward(VectorValuePtr inputValues, UnorderedMapVariableValuePtr outputs)
        {
            Value leftValue = inputValues[0];
            TensorView<float> leftOperand = new TensorView<float>(new Variable(leftValue.Shape, VariableKind.Parameter, DataType.Float, leftValue.Data, false, 
                Helper.AsAxisVector(new List<Axis>() { Axis.DefaultBatchAxis() }), false, "", ""));
            Value rightValue = inputValues[1];
            TensorView<float> rightOperand = new TensorView<float>(new Variable(rightValue.Shape, VariableKind.Constant, DataType.Float, rightValue.Data, false,
                Helper.AsAxisVector(new List<Axis>() { Axis.DefaultBatchAxis() }), false, "", ""));

            TensorView<float> output = leftOperand * rightOperand;
            outputs[this.Output] = new Value(output.Eval().Value());
        }

        protected override void InferOutputs(VariableVector outputs)
        {
            var leftOperand = Inputs[0];
            var rightOperand = Inputs[1];

            if (leftOperand.Shape.Rank != 2)
                throw new Exception("Left operand must be 2D");

            if (rightOperand.Shape.Rank != 1)
                throw new Exception("Right operand must be 1D");

            if (leftOperand.DynamicAxes.Count() != 0)
                throw new Exception("Left operand must not have dynamic axes (i.e. should not be minibatch data, but be a Parameter of fixed size)");

            outputs.Add(CNTKLib.OutputVariable(new int[]{ leftOperand.Shape[0] }, leftOperand.DataType, Helper.AsAxisVector(rightOperand.DynamicAxes)));
        }

        private uint _CurrentVersion()
        {
            return 0;
        }

        private string _OpName()
        {
            return "UserSimpleFunction";
        }

        private Function _Clone(VariableVector arg0)
        {
            return null;
        }
    }
}
