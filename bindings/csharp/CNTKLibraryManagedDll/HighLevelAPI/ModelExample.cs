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
        }

        protected override BackPropState Forward(VectorValuePtr inputValues, UnorderedMapVariableValuePtr outputs, DeviceDescriptor computeDevice, SWIGTYPE_p_std__unordered_setT_CNTK__Variable_t outputsToRetainBackwardStateFor)
        {
            var leftOperandData = inputValues[0].Data;
            var rightOperandData = inputValues[1].Data;

            // Allocate outputValue if needed
            var outputValue = outputs[this.Output];
            if (outputValue == null)
            {
                var numOutRows = leftOperandData.Shape[0];
                var numOutCols = rightOperandData.Shape[rightOperandData.Shape.Rank - 1];
                outputValue = new Value(new NDArrayView(DataType.Float, new int[]{ numOutRows , numOutCols }, computeDevice));
                outputs[this.Output] = outputValue;
            }

            var outputData = outputValue.Data;
            MatrixMultiply(leftOperandData, rightOperandData, outputData);

            // Let's save the right input's Value in the BackPropSate to be used in the backward pass for computing gradients
            UnorderedMapVariableValuePtr forwardPropValuesToSave = new UnorderedMapVariableValuePtr();
            forwardPropValuesToSave.Add(new KeyValuePair<Variable, Value>(Inputs[1], inputValues[1]));
            return new BackPropState(this, computeDevice, forwardPropValuesToSave);
        }

        protected override BackPropState Forward(VectorValuePtr inputValues, UnorderedMapVariableValuePtr outputs, DeviceDescriptor computeDevice)
        {
            return null;
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

        static Tuple<int, int> GetNumRowsAndCols(NDShape shape, bool transpose = false)
        {
            var numRows = shape[0];
            var numCols = shape[shape.Rank - 1];
            if (transpose)
            {
                var tmp = numRows;
                numRows = numCols;
                numCols = tmp;
            }

            return new Tuple<int, int>(numRows, numCols);
        }

        static int Offset(int rowIdx, int colIdx, NDShape matrixShape, bool transpose = false)
        {
            if (transpose)
            {
                var tmp = rowIdx;
                rowIdx = colIdx;
                colIdx = tmp;
            }

            return (colIdx* matrixShape[0]) + rowIdx;
        }

        static void MatrixMultiply(NDArrayView leftMatrix, NDArrayView rightMatrix, NDArrayView outputMatrix, bool transposeRight = false)
        {
            var tuple = GetNumRowsAndCols(leftMatrix.Shape);
            int leftNumRows = tuple.Item1, leftNumCols = tuple.Item2;

            tuple = GetNumRowsAndCols(rightMatrix.Shape, transposeRight);
            int rightNumRows = tuple.Item1, rightNumCols = tuple.Item2;

            var numOutRows = leftNumRows;
            var K = leftNumCols;
            var numOutCols = rightNumCols;

            if (false == (!leftMatrix.IsSparse && !rightMatrix.IsSparse && !outputMatrix.IsSparse))
            {
                throw new Exception();
            }
            if (false == (K == rightNumRows))
            {
                throw new Exception();
            }

            if (false == ((outputMatrix.Shape[0] == numOutRows) && (outputMatrix.Shape[1] == numOutCols)))
            {
                throw new Exception();
            }

            outputMatrix.SetValue(0.0f);

            // The operands values are in column major layout

            //var leftBuffer = leftMatrix.DataBufferFloat;
            //auto rightBuffer = rightMatrix->DataBuffer<float>();
            //auto outBuffer = outputMatrix->WritableDataBuffer<float>();
            //        for (size_t j = 0; j<numOutCols; ++j)
            //            for (size_t k = 0; k<K; ++k)
            //                for (size_t i = 0; i<numOutRows; ++i)
            //                    outBuffer[Offset(i, j, outputMatrix->Shape())] += leftBuffer[Offset(i, k, leftMatrix->Shape())] * rightBuffer[Offset(k, j, rightMatrix->Shape(), transposeRight)];
        }
    }
}
