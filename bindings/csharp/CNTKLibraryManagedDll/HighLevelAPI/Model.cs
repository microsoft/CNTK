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

        public static Model Model(Variable input, InitializationFunction init = null, AvtivationFunction activation = null)
        {
            Model model = new ModelImplementation(init: init, activation: activation)
            {
                Variable = input.ToFunction()
            };
            return model;
        }

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
        Dense Dense(int outputClasses, AvtivationFunction activation, DeviceDescriptor device, string name);

        Convolution2D Convolution2D(Dim2 filterShape, int numFilters, AvtivationFunction activation, 
            InitializationFunction init, IList<bool> pad, Dim2 strides, 
            bool bias, float initBias, int reductionRank, DeviceDescriptor device, string name);

        MaxPooling2D MaxPooling2D(Dim2 filterShape, Dim2 strides, IList<bool> pad, string name = "");

        AveragePooling2D AveragePooling2D(Dim2 filterShape, Dim2 strides, IList<bool> pad, string name = "");

        Dropout Dropout(double dropoutRate, string name = "");

        Embedding Embedding(NDShape shape, InitializationFunction init, IList<float> weights, string name = "");

        Recurrence Recurrence(Function stepfunction, bool goBackwards, float initialState,
            bool returnFullState, string name = "");

        RecurrenceFrom RecurrenceFrom(Function stepFunction, bool goBackwards, bool returnFullState, string name = "");

        Fold Fold(Function folderFunction, bool goBackwards, float initialState, bool returnFullState, string name = "");

        UnfoldFrom UnfoldFrom(Function generatorFunction, Function untilPredicate, int lengthIncrease, string name = "");

        LSTM LSTM(NDShape shape, NDShape cellShape, AvtivationFunction activation,
            bool usePeepholes, InitializationFunction init, float initBias, 
            bool enableSelfStabilization, string name = "");

        GRU GRU(NDShape shape, NDShape cellShape, AvtivationFunction activation,
            InitializationFunction init, float initBias, bool enableSelfStabilization,
            string name = "");

        RNNStep RNNStep(NDShape shape, NDShape cellShape, AvtivationFunction activation,
            InitializationFunction init, float initBias, bool enableSelfStabilization,
            string name = "");

        Delay Delay(int T, float initialState, string name = "");

        BatchNormalization BatchNormalization(int mapRank, float initScale,
            int normalizationTimeConstant, int blendTimeConstant, double epsilon,
            bool useCntkEngine, string name = "");

        LayerNormalization LayerNormalization(float initialScale, float initialBias,
            double epsilon, string name = "");

        Stabilizer Stabilizer(float steepness, bool enableSelfStabilization, string name = "");

        Model this[string name] { get; }
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
            this.Variable = activation(variable);
        }
        public int OutputClasses { get; set; }
    }

    public class Convolution2D : ModelImplementation
    {
        public Convolution2D(Variable input, Dim2 filterShape, int numFilters, AvtivationFunction activation,
            InitializationFunction init, IList<bool> pad, Dim2 strides,
            bool bias, float initBias, int reductionRank, DeviceDescriptor device, string name)
        {
            Parameter convParams = new Parameter((int[])(filterShape.Concat(numFilters)), DataType.Float,
                Initialization, device);
            this.Variable = CNTKLib.Convolution(convParams, input, (int[])strides);
        }
        public Dim2 FilterShape { get; set; }
        public int NumFilters { get; set; }
        public Dim2 Strides { get; set; }
        public Boolean Pad { get; set; }
    }

    public class MaxPooling2D : ModelImplementation
    {
        public MaxPooling2D(Variable input, Dim2 filterShape, Dim2 strides, IList<bool> pad, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class AveragePooling2D : ModelImplementation
    {
        public AveragePooling2D(Variable input, Dim2 filterShape, Dim2 strides, IList<bool> pad, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class Dropout : ModelImplementation
    {
        public Dropout(Variable input, double dropoutRatio, string name = null)
        {

        }
        public double DropoutRatio { get; set; }
    }

    public class Embedding : ModelImplementation
    {
        public Embedding(Variable input, NDShape shape, InitializationFunction init, IList<float> weights = null, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class Recurrence : ModelImplementation
    {
        public Recurrence(Variable input, Function stepfunction, bool goBackwards = false, float initialState = 0,
            bool returnFullState = false, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class RecurrenceFrom : ModelImplementation
    {
        public RecurrenceFrom(Variable input, Function stepFunction, bool goBackwards = false,
            bool returnFullState = false, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class Fold : ModelImplementation
    {
        public Fold(Variable input, Function folderFunction, bool goBackwards = false, float initialState = 0,
                bool returnFullState = false, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class UnfoldFrom : ModelImplementation
    {
        public UnfoldFrom(Variable input, Function generatorFunction, Function untilPredicate, int lengthIncrease, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class LSTM : ModelImplementation
    {
        public LSTM(Variable input, NDShape shape, NDShape cellShape, AvtivationFunction activation,
            InitializationFunction init, bool usePeepholes = false, float initBias = 0, 
            bool enableSelfStabilization = false, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class GRU : ModelImplementation
    {
        public GRU(Variable input, NDShape shape, NDShape cellShape, AvtivationFunction activation,
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
        public RNNStep(Variable input, NDShape shape, NDShape cellShape, AvtivationFunction activation,
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
        public Delay(Variable input, int T = 1, float initialState = 0, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class BatchNormalization : ModelImplementation
    {
        public BatchNormalization(Variable input, int mapRank = 0,
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
        public LayerNormalization(Variable input, float initialScale = 1, float initialBias = 0,
                double epsilon = 0.00001, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class Stabilizer : ModelImplementation
    {
        public Stabilizer(Variable input, float steepness, bool enableSelfStabilization = true, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class ModelImplementation : Model
    {
        ModelImplementation(Function function)
        {
            this.Variable = function;
        }
        public ModelImplementation(InitializationFunction init = null, AvtivationFunction activation = null) { }

        public Dense Dense(int outputClasses, AvtivationFunction activation, DeviceDescriptor device, string name = null)
        {
            return new Dense(this.Variable, outputClasses, activation, device, name);
        }
        public Convolution2D Convolution2D(Dim2 filterShape, int numFilters, AvtivationFunction activation,
            InitializationFunction init, IList<bool> pad, Dim2 strides,
            bool bias, float initBias, int reductionRank, DeviceDescriptor device, string name)
        {
            return new Convolution2D(this.Variable, filterShape, numFilters, activation, 
                init, pad, strides, bias, initBias, reductionRank, device, name);
        }

        public MaxPooling2D MaxPooling2D(Dim2 filterShape, Dim2 strides, IList<bool> pad, string name = "")
        {
            return new MaxPooling2D(this.Variable, filterShape, strides, pad, name);
        }

        public AveragePooling2D AveragePooling2D(Dim2 filterShape, Dim2 strides, IList<bool> pad, string name = "")
        {
            return new AveragePooling2D(this.Variable, filterShape, strides, pad, name);
        }

        public Dropout Dropout(double dropoutRatio, string name = null)
        {
            return new Dropout(this.Variable, dropoutRatio, name);
        }

        public Embedding Embedding(NDShape shape, InitializationFunction init, IList<float> weights = null, string name = "")
        {
            return new Embedding(this.Variable, shape, init, weights, name);
        }

        public Recurrence Recurrence(Function stepfunction, bool goBackwards = false, float initialState = 0,
            bool returnFullState = false, string name = "")
        {
            return new Recurrence(this.Variable, stepfunction, goBackwards, initialState, returnFullState, name);
        }

        public RecurrenceFrom RecurrenceFrom(Function stepFunction, bool goBackwards = false,
            bool returnFullState = false, string name = "")
        {
            return new RecurrenceFrom(this.Variable, stepFunction, goBackwards, returnFullState, name);
        }

        public Fold Fold(Function folderFunction, bool goBackwards = false, float initialState = 0, 
            bool returnFullState = false, string name = "")
        {
            return new Fold(this.Variable, folderFunction, goBackwards, initialState, returnFullState, name);
        }

        public UnfoldFrom UnfoldFrom(Function generatorFunction, Function untilPredicate, int lengthIncrease, string name = "")
        {
            return new UnfoldFrom(this.Variable, generatorFunction, untilPredicate, lengthIncrease, name);
        }

        public LSTM LSTM(NDShape shape, NDShape cellShape, AvtivationFunction activation,
            bool usePeepholes, InitializationFunction init, float initBias,
            bool enableSelfStabilization, string name = "")
        {
            return new LSTM(this.Variable, shape, cellShape, activation, init, usePeepholes, initBias,
            enableSelfStabilization, name);
        }

        public GRU GRU(NDShape shape, NDShape cellShape, AvtivationFunction activation,
            InitializationFunction init,
            float initBias = 0,
            bool enableSelfStabilization = false,
            string name = "")
        {
            return new GRU(this.Variable, shape, cellShape, activation, init,
                initBias, enableSelfStabilization, name);
        }

        public RNNStep RNNStep(NDShape shape, NDShape cellShape, AvtivationFunction activation,
            InitializationFunction init,
            float init_bias = 0,
            bool enable_self_stabilization = false,
            string name = "")
        {
            return new RNNStep(this.Variable, shape, cellShape, activation,
                init, init_bias, enable_self_stabilization, name);
        }

        public Delay Delay(int T = 1, float initialState = 0, string name = "")
        {
            return new Delay(this.Variable, T, initialState, name);
        }

        public BatchNormalization BatchNormalization(int mapRank = 0,
                       float initScale = 1,
                       int normalizationTimeConstant = 5000,
                       int blendTimeConstant = 0,
                       double epsilon = 0.00001,
                       bool useCntkEngine = false,
                       string name = "")
        {
            return new BatchNormalization(this.Variable, mapRank,
                       initScale,
                       normalizationTimeConstant,
                       blendTimeConstant,
                       epsilon,
                       useCntkEngine,
                       name);
        }

        public LayerNormalization LayerNormalization(float initialScale = 1, float initialBias = 0, 
            double epsilon = 0.00001, string name = "")
        {
            return new LayerNormalization(this.Variable, initialScale, initialBias, epsilon, name);
        }

        public Stabilizer Stabilizer(float steepness, bool enableSelfStabilization = true, string name = "")
        {
            return new Stabilizer(this.Variable, steepness, enableSelfStabilization, name);
        }

        /// <summary>
        /// 
        /// </summary>

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

        public Variable Variable { get; set; }

        public AvtivationFunction Avtivation { get; set; }

        public static implicit operator ModelImplementation(ModelImplementation[] models)
        {
            return new ModelImplementation(Function.Combine(models.Select(m => m.Variable).ToList()));
        }

        public ModelImplementation this[int timeOffset]
        {
            get
            {
                Function f = this.Variable;
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
