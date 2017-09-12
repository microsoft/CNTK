using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using CNTK;

using static Microsoft.MachineLearning.Cntk.CntkLib;

namespace Microsoft.MachineLearning.Cntk
{
    public static class CntkLib
    {
        public static Dim2 Dim2(int a, int b)
        {
            return new Dim2(a, b);
        }
        public static Dim3 Dim3(int a, int b, int c)
        {
            return new Dim3(a, b, c);
        }
        public static Dim4 Dim4(int a, int b, int c, int d)
        {
            return new Dim4(a, b, c, d);
        }

        public delegate Variable AvtivationFunction(Variable input);
        public static AvtivationFunction ReluActivation = a => CNTKLib.ReLU(a);
        public static AvtivationFunction SigmoidActivation = a => CNTKLib.Sigmoid(a);
        public static AvtivationFunction LeakyReluActivation = a => CNTKLib.LeakyReLU(a);

        public static Model Model(InitializationFunction init = null, AvtivationFunction activation = null) { return new ModelImplementation(init: init, activation: activation); }

        public static ZerosInitialization ZerosInitialization() { return new ZerosInitialization(); }
        public static OnesInitialization OnesInitialization() { return new OnesInitialization(); }
        public static GlorotNormalInitialization GlorotNormalInitialization() { return new GlorotNormalInitialization(); }
        public static GlorotUniformInitialization GlorotUniformInitialization() { return new GlorotUniformInitialization(); }
        public static ConstantInitialization ConstantInitialization(double value) { return new ConstantInitialization(value); }

    }

    public class Dim
    {
        public Dim(int a) { }
        public Dim(int a, int b) { }
        public Dim(int a, int b, int c) { }
        public Dim(int a, int b, int c, int d) { }
    }
    public class Dim1
    {
        public Dim1(int a) { }
    }
    public class Dim2
    {
        public Dim2(int a, int b) { }
        public static implicit operator int[](Dim2 dim2)
        {
            return dim2.dims;
        }
        public Dim3 Concat(int d)
        {
            return new Dim3(dims[0], dims[1], d);
        }

        private int[] dims = new int[2]; 

    }
    public class Dim3
    {
        public Dim3(int a, int b, int c) { }
        public static implicit operator int[] (Dim3 dim3)
        {
            return dim3.dims;
        }

        private int[] dims = new int[3];
    }
    public class Dim4
    {
        public Dim4(int a, int b, int c, int d) { }
    }
    public class InputVariable
    {
        public InputVariable(Dim dimension) { }
    }

    public interface InitializationFunction { }
    public class ZerosInitialization : InitializationFunction { }
    public class OnesInitialization : InitializationFunction { }
    public class GlorotNormalInitialization : InitializationFunction { }
    public class GlorotUniformInitialization : InitializationFunction { }
    public class ConstantInitialization : InitializationFunction
    {
        public ConstantInitialization(double value) { }
    }

    public interface Model
    {
        Convolution2D Convolution2D(Dim2 filterShape, int numFilters, Dim2 strides, Boolean pad, DeviceDescriptor device, string name);
        Dense Dense(int outputClasses, AvtivationFunction activation, DeviceDescriptor device, string name);
        MaxPooling MaxPooling(Dim2 filterShape, int strides = 1, bool pad = false, string name = "");
        AveragePooling AveragePooling(Dim2 filterShape, int strides = 1, bool pad = false, string name = "");
        Dropout Dropout(double dropoutRatio, string name);
        Embedding Embedding(NDShape shape, InitializationFunction init, IList<float> weights = null, string name = "");

        Recurrence Recurrence(Function stepfunction, bool goBackwards = false, float initialState = 0,
            bool returnFullState = false, string name = "");

        LSTM LSTM(NDShape shape, NDShape cellShape, AvtivationFunction activation,
            InitializationFunction init, bool usePeepholes = false, float initBias = 0,
            bool enableSelfStabilization = false, string name = "");

        Model this[string name] { get; }
    }

    public class Convolution2D : ModelImplementation
    {
        public Convolution2D(Variable inputVariable, Dim2 filterShape, int numFilters, Dim2 strides, bool pad, DeviceDescriptor device, string name = null)
        {
            Parameter convParams = new Parameter((int[])(filterShape.Concat(numFilters)), DataType.Float,
                Initialization, device);
            this.Function = CNTKLib.Convolution(convParams, inputVariable, (int[])strides);
        }
        public Dim2 FilterShape { get; set; }
        public int NumFilters { get; set; }
        public Dim2 Strides { get; set; }
        public Boolean Pad { get; set; }
    }

    public class AveragePooling : ModelImplementation
    {
        AveragePooling(Dim2 filterShape, int strides = 1, bool pad = false, string name = "")
        {
            throw new NotImplementedException();
        }
    }
    public class MaxPooling : ModelImplementation
    {
        MaxPooling(NDShape filter_shape, int strides = 1, bool pad = false, string name = "")
        {
            throw new NotImplementedException();
        }

    }

    public class Dense : ModelImplementation
    {
        public Dense(Variable input, int outputClasses, AvtivationFunction activation, DeviceDescriptor device, string name = null)
        {
            // TODO: falttern?
            int inputDim = input.Shape[0];

            int[] s = { outputClasses, inputDim };
            var timesParam = new Parameter((NDShape)s, DataType.Float,
                CNTKLib.GlorotUniformInitializer(
                    CNTKLib.DefaultParamInitScale,
                    CNTKLib.SentinelValueForInferParamInitRank,
                    CNTKLib.SentinelValueForInferParamInitRank, 1),
                device, "timesParam");
            var timesFunction = CNTKLib.Times(timesParam, input, "times");

            int[] s2 = { outputClasses };
            var plusParam = new Parameter(s2, 0.0f, device, "plusParam");
            var variable = CNTKLib.Plus(plusParam, timesFunction, name);
            this.Function = activation(variable);
        }
        public int OutputClasses { get; set; }
    }
    public class Dropout : ModelImplementation
    {
        public Dropout(double dropoutRatio, string name = null)
        {

        }
        public double DropoutRatio { get; set; }
    }

    public class Embedding : ModelImplementation
    {

    }

    public class Recurrence : ModelImplementation
    {
        public Recurrence(Function stepfunction, bool goBackwards = false, float initialState = 0,
            bool returnFullState = false, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class RecurrenceFrom : ModelImplementation
    {
        public RecurrenceFrom(Function stepFunction, bool goBackwards = false,
            bool returnFullState = false, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class Fold : ModelImplementation
    {
        Fold(Function folderFunction, bool goBackwards = false, float initialState = 0,
                bool returnFullState = false, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class LSTM : ModelImplementation
    {
        public LSTM(NDShape shape, NDShape cellShape, AvtivationFunction activation,
            InitializationFunction init, bool usePeepholes = false, float initBias = 0, 
            bool enableSelfStabilization = false, string name = "")
        {

        }
    }

    public class GRU : ModelImplementation
    {
        GRU(NDShape shape, NDShape cellShape, AvtivationFunction activation,
            InitializationFunction init,
            float initBias = 0,
            bool enableSelfStabilization = false,
            string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class RNNStep : ModelImplementation
    {
        public RNNStep(NDShape shape, NDShape cellShape, AvtivationFunction activation,
            InitializationFunction init,
            float init_bias = 0,
            bool enable_self_stabilization = false,
            string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class Delay : ModelImplementation
    {
        public Delay(int T = 1, float initialState = 0, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class BatchNormalization : ModelImplementation
    {
        public BatchNormalization(int mapRank = 0,
                       float initScale = 1,
                       int normalizationTimeConstant = 5000,
                       int blendTimeConstant = 0,
                       double epsilon = 0.00001,
                       bool useCntkEngine = false,
                       string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class LayerNormalization : ModelImplementation
    {
        public LayerNormalization(float initialScale = 1, float initialBias = 0,
                double epsilon = 0.00001, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class Stabilizer : ModelImplementation
    {
        public Stabilizer(int steepness = 4, bool enableSelfStabilization = true, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class ModelImplementation : Model
    {
        ModelImplementation(Function function)
        {
            this.Function = function;
        }
        public ModelImplementation(InitializationFunction init = null, AvtivationFunction activation = null) { }
        public Convolution2D Convolution2D(Dim2 filterShape, int numFilters, Dim2 strides, Boolean pad, DeviceDescriptor device, string name = null)
        {
            var conv = new Convolution2D(this.Function, filterShape, numFilters, strides, pad, device, name);
            // somehow connect the new node to the existing model
            return conv;
        }
        public Dense Dense(int outputClasses, AvtivationFunction activation, DeviceDescriptor device, string name = null)
        {
            var dense = new Dense(this.Function, outputClasses, activation, device, name);
            // somehow connect the new node to the existing model
            return dense;
        }
        public Dropout Dropout(double dropoutRatio, string name = null)
        {
            return null;
        }
        public Model ElementWisePlus(Model otherModel)
        {
            return this;
        }
        public Model this[string name]
        {
            get
            {
                return null;
            }
        }
        public string Name { get; set; }
        public CNTKDictionary Initialization { get; set; }

        public Function Function { get; set; }

        public AvtivationFunction Avtivation { get; set; }

        public static implicit operator ModelImplementation(ModelImplementation[] models)
        {
            return new ModelImplementation(Function.Combine(models.Select(m => m.Function.Output).ToList()));
        }

        public ModelImplementation this[int timeOffset]
        {
            get
            {
                Function f = this.Function;
                if (timeOffset > 0)
                {
                    f = CNTKLib.PastValue(f, (uint)Math.Abs(timeOffset));
                }
                else // if (timeOffset > 0)
                {
                    f = CNTKLib.FutureValue(f, (uint)Math.Abs(timeOffset));
                }
                return new ModelImplementation(f);
            }
        }
    }
}
