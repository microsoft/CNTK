using System;
using System.Collections.Generic;
using System.Linq;
using System.Numerics;
using System.Text;
using System.Threading.Tasks;

namespace CNTK.HighLevelAPI
{

    public class TensorView<T> : Variable
    {
        public TensorView(NDShape shape, NDArrayView value, IList<Axis> dynamicAxes, string name = "") :
            base(shape, VariableKind.Constant,
                DataType.Float,
                value,
                false /*needsGradient*/,
                Helper.AsAxisVector(dynamicAxes) /*dynamicAxes*/, 
                false /*isSparse*/, 
                name, 
                "") 
        {

        }

        public TensorView(T[] buffer)
        {

        }

        public TensorView(T[][] buffer)
        {

        }

        public TensorView(Variable variable) : base(variable)
        {

        }

        public static TensorView<T> Zeros(NDShape shape)
        {
            throw new NotImplementedException();
        }

        public static TensorView<T> Ones(NDShape shape)
        {
            throw new NotImplementedException();
        }

        public static TensorView<T> Eye(int dim)
        {
            throw new NotImplementedException();
        }

        public static TensorView<T> RandomNormal(NDShape shape, double mean, double stdDev, uint seed, DeviceDescriptor device)
        {
            return new TensorView<T>(new Constant(NDArrayView.RandomNormal<T>(shape, mean, stdDev, seed, device)));
        }

        public static TensorView<T> RandomUniform(NDShape shape, double rangeStart, double rangeEnd, uint seed, DeviceDescriptor device)
        {
            return new TensorView<T>(new Constant(NDArrayView.RandomUniform<T>(shape, rangeStart, rangeEnd, seed, device)));
        }

        public static TensorView<T> Empty(NDShape shape)
        {
            throw new NotImplementedException();
        }

        public static void Save(string filePath, TensorView<T> tensor)
        {
            tensor.ToFunction().Save(filePath);
        }

        public static byte[] Save(TensorView<T> tensor)
        {
            return tensor.ToFunction().Save();
        }

        public static TensorView<T> Load(string filePath, DeviceDescriptor device)
        {
            return new TensorView<T>(Function.Load(filePath, device));
        }

        public TensorView<T> Load(byte[] buffer, DeviceDescriptor device)
        {
            return new TensorView<T>(Function.Load(buffer, device));
        }

        // Shape: provided via Variable

        public int Rank()
        {
            return this.Shape.Rank;
        }

        public int Size()
        {
            return this.Shape.TotalSize;
        }

        public Constant Eval()
        {
            Function func = this.ToFunction();
            var inputs = func.Inputs;
            Dictionary<Variable, Value> inputValues = new Dictionary<Variable, Value>();
            foreach (Variable input in func.Inputs)
            {
                inputValues.Add(input, new Value(input.Value2()));
            }

            Dictionary<Variable, Value> outputs = new Dictionary<Variable, Value>()
            {
                { func.Output, null}
            };

            func.Evaluate(inputValues, outputs, DeviceDescriptor.CPUDevice);
            return new Constant(outputs[func.Output].Data);
        }

        public static TensorView<T> operator +(TensorView<T> left, TensorView<T> right)
        {
            return new TensorView<T>(CNTKLib.Plus(left, right));
        }

        public static TensorView<T> operator -(TensorView<T> left, TensorView<T> right)
        {
            return new TensorView<T>(CNTKLib.Minus(left, right));
        }

        public static TensorView<T> operator *(TensorView<T> left, TensorView<T> right)
        {
            return new TensorView<T>(CNTKLib.Times(left, right));
        }

        public static TensorView<T> operator /(TensorView<T> left, TensorView<T> right)
        {
            return new TensorView<T>(CNTKLib.ElementDivide(left, right));
        }

        // np.exp(b)
        public TensorView<T> Exp()
        {
            return new TensorView<T>(CNTKLib.Exp(this));
        }

        // np.sqrt(b)
        public TensorView<T> Sqrt()
        {
            return new TensorView<T>(CNTKLib.Sqrt(this));
        }

        // np.sin(a)
        public TensorView<T> Sin()
        {
            return new TensorView<T>(CNTKLib.Sin(this));
        }

        // np.cos(b) 
        public TensorView<T> Cos()
        {
            return new TensorView<T>(CNTKLib.Cos(this));
        }

        // np.log(a)
        public TensorView<T> Log()
        {
            return new TensorView<T>(CNTKLib.Log(this));
        }

        // a == b
        public static bool operator == (TensorView<T> left, TensorView<T> right)
        {
            throw new NotImplementedException();
        }

        public override bool Equals(Object obj)
        {
            // If parameter is null return false.
            if (obj == null)
            {
                return false;
            }

            // If parameter cannot be cast to Point return false.
            TensorView<T> p = obj as TensorView<T>;
            if ((Object)p == null)
            {
                return false;
            }

            // Return true if the fields match:
            return CNTKLib.AreEqual(this, p);
        }

        public bool Equals(TensorView<T> tersor)
        {
            // If parameter is null return false:
            if ((object)tersor == null)
            {
                return false;
            }

            // Return true if the fields match:
            return CNTKLib.AreEqual(this, tersor);
        }

        public override int GetHashCode()
        {
            return base.GetHashCode();
        }

        public static bool operator !=(TensorView<T> left, TensorView<T> right)
        {
            throw new NotImplementedException();
        }

        //  a < 2
        public static bool operator <(TensorView<T> left, TensorView<T> right)
        {
            throw new NotImplementedException();
        }

        public static bool operator >(TensorView<T> left, TensorView<T> right)
        {
            throw new NotImplementedException();
        }

        // a.sum()
        public TensorView<T> Sum(Axis axis)
        {
            return new TensorView<T>(CNTKLib.ReduceSum(this, axis));
        }

        // a.min()
        public TensorView<T> Min(Axis axis)
        {
            return new TensorView<T>(CNTKLib.ReduceMin(this, axis));
        }

        // b.max(axis=0)
        public TensorView<T> Max(Axis axis)
        {
            return new TensorView<T>(CNTKLib.ReduceMax(this, axis));
        }

        // b.cumsum(axis=1)

        // a.mean()
        public TensorView<T> Mean(Axis axis)
        {
            return new TensorView<T>(CNTKLib.ReduceMean(this, axis));
        }

        // b.median()

        // a.corrcoef()

        // np.std(b)


        public TensorView<T> Slice(IList<Axis> axis, IList<int> beginIndex, IList<int> endIndex)
        {
            return new TensorView<T>(CNTKLib.Slice(this, Helper.AsAxisVector(axis), Helper.AsIntVector(beginIndex), Helper.AsIntVector(endIndex))); 
        }

        public TensorView<T> this[Variable indices]
        {
            get
            {
                return new TensorView<T>(CNTKLib.GatherOp(indices, this));
            }
        }

        // i = np.transpose(b) 
        public TensorView<T> Transpose()
        {
            return new TensorView<T>(CNTKLib.Transpose(this));
        }

        // g.reshape(3,-2)
        public TensorView<T> Reshape(NDShape newShape)
        {
            return new TensorView<T>(CNTKLib.Reshape(this, newShape));
        }

        // splice
        public static TensorView<T> Splice(IList<TensorView<T>> tensors, Axis axis)
        {
            return new TensorView<T>(CNTKLib.Splice(Helper.AsVariableVector(tensors.Cast<Variable>().ToList()), axis));
        }

        // TODO:
        // vstack
        // hstack
        // hsplit
        // vsplit
        // flatten
    }
}
