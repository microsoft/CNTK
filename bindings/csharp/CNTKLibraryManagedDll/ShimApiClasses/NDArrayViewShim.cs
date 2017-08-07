namespace CNTK
{
    public partial class NDArrayView
    {
        // Constructor using float dense input.
        public NDArrayView(NDShape viewShape, float[] dataBuffer, DeviceDescriptor device, bool readOnly = false) : this(viewShape, dataBuffer, (uint)dataBuffer.Length, device, readOnly)
        {
        }

        // Constructor using double dense input.
        public NDArrayView(NDShape viewShape, double[] dataBuffer, DeviceDescriptor device, bool readOnly = false) : this(viewShape, dataBuffer, (uint)dataBuffer.Length, device, readOnly)
        {
        }

        // Constructor using float sparse input.
        public NDArrayView(NDShape viewShape, int[] colStarts, int[] rowIndices, float[] nonZeroValues, DeviceDescriptor device, bool readOnly = false) : this(viewShape, colStarts, rowIndices, nonZeroValues, (uint)nonZeroValues.Length, device, readOnly)
        {
            if (rowIndices.Length != nonZeroValues.Length)
            {
                throw new System.ArgumentException("The length of rowIndicies must be same as the length of nonZeroValues.");
            }
            if (viewShape[viewShape.Rank - 1] + 1 != colStarts.Length)
            {
                throw new System.ArgumentException("The length of colStarts does not match the number of rows, i.e. the dimension size of the last rank of viewShape.");
            }
        }

        // Constructor using double sparse input.
        public NDArrayView(NDShape viewShape, int[] colStarts, int[] rowIndices, double[] nonZeroValues, DeviceDescriptor device, bool readOnly = false) : this(viewShape, colStarts, rowIndices, nonZeroValues, (uint)nonZeroValues.Length, device, readOnly)
        {
            if (rowIndices.Length != nonZeroValues.Length)
            {
                throw new System.ArgumentException("The length of rowIndicies must be same as the length of nonZeroValues.");
            }
            if (viewShape[viewShape.Rank - 1] + 1 != colStarts.Length)
            {
                throw new System.ArgumentException("The length of colStarts does not match the number of rows, i.e. the dimension size of the last rank of viewShape.");
            }
        }

        // Property Device.
        public DeviceDescriptor Device
        {
            get { return _Device(); }
        }

        // Property DataType.
        public DataType DataType
        {
            get { return _GetDataType(); }
        }

        // Property Shape.
        public NDShape Shape
        {
            get { return _Shape(); }
        }

        // Property StorageFormat.
        public StorageFormat StorageFormat
        {
            get { return _GetStorageFormat(); }
        }

        // Property IsSparse.
        public bool IsSparse
        {
            get { return _IsSparse(); }
        }

        // Property IsReadOnly.
        public bool IsReadOnly
        {
            get { return _IsReadOnly(); }
        }

        // Returns a slice view.
        public NDArrayView SliceView(System.Collections.Generic.IEnumerable<int> startOffset, System.Collections.Generic.IEnumerable<int> extent, bool readOnly = false)
        {
            var startOffsetVector = Helper.AsSizeTVector(startOffset);

            var extentVector = Helper.AsSizeTVector(extent);

            return _SliceView(startOffsetVector, extentVector, readOnly);
        }

        // Creates a new NDArrayView which is an alias of this NDArrayView.
        public NDArrayView Alias(bool readOnly = false)
        {
            return _Alias(readOnly);
        }

        public static NDArrayView RandomNormal<T>(NDShape shape, double mean, double stdDev, uint seed, DeviceDescriptor device)
        {
            if (device == null)
            {
                device = DeviceDescriptor.UseDefaultDevice();
            }
            if (typeof(T).Equals(typeof(float)))
            {
                return _RandomNormalFloat(shape, mean, stdDev, seed, device);
            }
            else if (typeof(T).Equals(typeof(double)))
            {
                return _RandomNormalDouble(shape, mean, stdDev, seed, device);
            }
            else
            {
                throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }

        public static NDArrayView RandomUniform<T>(NDShape shape, double rangeBegin, double rangeEnd, uint seed, DeviceDescriptor device)
        {
            if (device == null)
            {
                device = DeviceDescriptor.UseDefaultDevice();
            }
            if (typeof(T).Equals(typeof(float)))
            {
                return _RandomUniformFloat(shape, rangeBegin, rangeEnd, seed, device);
            }
            else if (typeof(T).Equals(typeof(double)))
            {
                return _RandomUniformDouble(shape, rangeBegin, rangeEnd, seed, device);
            }
            else
            {
                throw new System.ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }
    }
}
