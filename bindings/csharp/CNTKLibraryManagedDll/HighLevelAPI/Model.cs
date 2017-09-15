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
        Dense Dense(int outputClasses, AvtivationFunction activation, DeviceDescriptor device, string name);

        Convolution2D Convolution2D(Dim2 filterShape, int numFilters, Dim2 strides, Boolean pad, DeviceDescriptor device, string name);

        MaxPooling MaxPooling(Dim2 filterShape, int strides = 1, bool pad = false, string name = "");

        AveragePooling AveragePooling(Dim2 filterShape, int strides = 1, bool pad = false, string name = "");

        Dropout Dropout(double dropoutRatio, string name);

        Embedding Embedding(NDShape shape, InitializationFunction init, IList<float> weights = null, string name = "");

        Recurrence Recurrence(Function stepfunction, bool goBackwards, float initialState,
            bool returnFullState, string name = "");

        RecurrenceFrom RecurrenceFrom(Function stepFunction, bool goBackwards, bool returnFullState, string name = "");

        Fold Fold(Function folderFunction, bool goBackwards, float initialState, bool returnFullState, string name = "");

        UnfoldFrom UnfoldFrom(Function generatorFunction, Function ntilPredicate, int lengthIncrease, string name = "");

        LSTM LSTM(NDShape shape, NDShape cellShape, AvtivationFunction activation, 
            InitializationFunction init, bool usePeepholes, float initBias, 
            bool enableSelfStabilization = false, string name = "");

        GRU GRU(NDShape shape, NDShape cellShape, AvtivationFunction activation,
            InitializationFunction init, float initBias, bool enableSelfStabilization,
            string name = "");

        RNNStep RNNStep(NDShape shape, NDShape cellShape, AvtivationFunction activation,
            InitializationFunction init, float initBias, bool enableSelfStabilization,
            string name = "");

        Delay Delay(int T, float initialState = 0, string name = "");

        BatchNormalization BatchNormalization(int mapRank, float initScale,
            int normalizationTimeConstant, int blendTimeConstant, double epsilon,
            bool useCntkEngine, string name = "");

        LayerNormalization LayerNormalization(float initialScale, float initialBias,
            double epsilon, string name = "");

        Stabilizer Stabilizer(int steepness, bool enableSelfStabilization, string name = "");

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
            this.Function = activation(variable);
        }
        public int OutputClasses { get; set; }
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

    public class MaxPooling : ModelImplementation
    {
        public MaxPooling(Dim2 filterShape, int strides = 1, bool pad = false, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class AveragePooling : ModelImplementation
    {
        public AveragePooling(Dim2 filterShape, int strides = 1, bool pad = false, string name = "")
        {
            throw new NotImplementedException();
        }
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
        public Embedding(NDShape shape, InitializationFunction init, IList<float> weights = null, string name = "")
        {
            throw new NotImplementedException();
        }
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
        public Fold(Function folderFunction, bool goBackwards = false, float initialState = 0,
                bool returnFullState = false, string name = "")
        {
            throw new NotImplementedException();
        }
    }

    public class UnfoldFrom : ModelImplementation
    {
        public UnfoldFrom(Function generatorFunction, Function ntilPredicate, int lengthIncrease, string name = "")
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
            throw new NotImplementedException();
        }
    }

    public class GRU : ModelImplementation
    {
        public GRU(NDShape shape, NDShape cellShape, AvtivationFunction activation,
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

        public Dense Dense(int outputClasses, AvtivationFunction activation, DeviceDescriptor device, string name = null)
        {
            return new Dense(this.Function, outputClasses, activation, device, name);
        }
        public Convolution2D Convolution2D(Dim2 filterShape, int numFilters, Dim2 strides, Boolean pad, DeviceDescriptor device, string name = null)
        {
            return new Convolution2D(this.Function, filterShape, numFilters, strides, pad, device, name);
        }

        public MaxPooling MaxPooling(Dim2 filterShape, int strides = 1, bool pad = false, string name = "")
        {
            return new MaxPooling(filterShape, strides = 1, pad, name);
        }

        public AveragePooling AveragePooling(Dim2 filterShape, int strides = 1, bool pad = false, string name = "")
        {
            return new AveragePooling(filterShape, strides, pad, name);
        }

        public Dropout Dropout(double dropoutRatio, string name = null)
        {
            return new Dropout(dropoutRatio, name);
        }

        public Embedding Embedding(NDShape shape, InitializationFunction init, IList<float> weights = null, string name = "")
        {
            return new Embedding(shape, init, weights, name);
        }

        public Recurrence Recurrence(Function stepfunction, bool goBackwards = false, float initialState = 0,
            bool returnFullState = false, string name = "")
        {
            return new Recurrence(stepfunction, goBackwards, initialState, returnFullState, name);
        }

        public RecurrenceFrom RecurrenceFrom(Function stepFunction, bool goBackwards = false,
            bool returnFullState = false, string name = "")
        {
            return new RecurrenceFrom(stepFunction, goBackwards, returnFullState, name);
        }

        public Fold Fold(Function folderFunction, bool goBackwards = false, float initialState = 0, 
            bool returnFullState = false, string name = "")
        {
            return new Fold(folderFunction, goBackwards, initialState, returnFullState, name);
        }

        public UnfoldFrom UnfoldFrom(Function generatorFunction, Function ntilPredicate, int lengthIncrease, string name = "")
        {
            return new UnfoldFrom(generatorFunction, ntilPredicate, lengthIncrease, name);
        }

        public LSTM LSTM(NDShape shape, NDShape cellShape, AvtivationFunction activation,
            InitializationFunction init, bool usePeepholes = false, float initBias = 0,
            bool enableSelfStabilization = false, string name = "")
        {
            return new LSTM(shape, cellShape, activation, init, usePeepholes, initBias,
            enableSelfStabilization, name);
        }

        public GRU GRU(NDShape shape, NDShape cellShape, AvtivationFunction activation,
            InitializationFunction init,
            float initBias = 0,
            bool enableSelfStabilization = false,
            string name = "")
        {
            return new GRU(shape, cellShape, activation, init,
                initBias, enableSelfStabilization, name);
        }

        public RNNStep RNNStep(NDShape shape, NDShape cellShape, AvtivationFunction activation,
            InitializationFunction init,
            float init_bias = 0,
            bool enable_self_stabilization = false,
            string name = "")
        {
            return new RNNStep(shape, cellShape, activation,
                init, init_bias, enable_self_stabilization, name);
        }

        public Delay Delay(int T = 1, float initialState = 0, string name = "")
        {
            return new Delay(T, initialState, name);
        }

        public BatchNormalization BatchNormalization(int mapRank = 0,
                       float initScale = 1,
                       int normalizationTimeConstant = 5000,
                       int blendTimeConstant = 0,
                       double epsilon = 0.00001,
                       bool useCntkEngine = false,
                       string name = "")
        {
            return new BatchNormalization(mapRank,
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
            return new LayerNormalization(initialScale, initialBias, epsilon, name);
        }

        public Stabilizer Stabilizer(int steepness = 4, bool enableSelfStabilization = true, string name = "")
        {
            return new Stabilizer(steepness, enableSelfStabilization, name);
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
