//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// NDArrayViewShim.cs -- C# Api for CNTK NDArrayView class
//
namespace CNTK
{
    public partial class NDArrayView
    {
        /// <summary>
        /// Constructor using float dense input.
        /// </summary>
        /// <param name="viewShape"></param>
        /// <param name="dataBuffer"></param>
        /// <param name="device"></param>
        /// <param name="readOnly"></param>
        public NDArrayView(NDShape viewShape, float[] dataBuffer, DeviceDescriptor device, bool readOnly = false) : this(viewShape, dataBuffer, (uint)dataBuffer.Length, device, readOnly)
        {
        }

        /// <summary>
        /// Constructor using double dense input.
        /// </summary>
        /// <param name="viewShape"></param>
        /// <param name="dataBuffer"></param>
        /// <param name="device"></param>
        /// <param name="readOnly"></param>
        public NDArrayView(NDShape viewShape, double[] dataBuffer, DeviceDescriptor device, bool readOnly = false) : this(viewShape, dataBuffer, (uint)dataBuffer.Length, device, readOnly)
        {
        }

        /// <summary>
        /// Constructor using float sparse input.
        /// </summary>
        /// <param name="viewShape"></param>
        /// <param name="colStarts"></param>
        /// <param name="rowIndices"></param>
        /// <param name="nonZeroValues"></param>
        /// <param name="device"></param>
        /// <param name="readOnly"></param>
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

        /// <summary>
        /// Constructor using double sparse input.
        /// </summary>
        /// <param name="viewShape"></param>
        /// <param name="colStarts"></param>
        /// <param name="rowIndices"></param>
        /// <param name="nonZeroValues"></param>
        /// <param name="device"></param>
        /// <param name="readOnly"></param>
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

        /// <summary>
        /// Property Device.
        /// </summary>
        public DeviceDescriptor Device
        {
            get { return _Device(); }
        }

        /// <summary>
        /// Property DataType.
        /// </summary>
        public DataType DataType
        {
            get { return _GetDataType(); }
        }

        /// <summary>
        /// Property Shape.
        /// </summary>
        public NDShape Shape
        {
            get { return _Shape(); }
        }

        /// <summary>
        /// Property StorageFormat.
        /// </summary>
        public StorageFormat StorageFormat
        {
            get { return _GetStorageFormat(); }
        }

        /// <summary>
        /// Property IsSparse.
        /// </summary>
        public bool IsSparse
        {
            get { return _IsSparse(); }
        }

        /// <summary>
        /// Property IsReadOnly.
        /// </summary>
        public bool IsReadOnly
        {
            get { return _IsReadOnly(); }
        }

        /// <summary>
        /// Returns a slice view.
        /// </summary>
        /// <param name="startOffset"></param>
        /// <param name="extent"></param>
        /// <param name="readOnly"></param>
        /// <returns></returns>
        public NDArrayView SliceView(System.Collections.Generic.IEnumerable<int> startOffset, System.Collections.Generic.IEnumerable<int> extent, bool readOnly = false)
        {
            var startOffsetVector = Helper.AsSizeTVector(startOffset);

            var extentVector = Helper.AsSizeTVector(extent);

            return _SliceView(startOffsetVector, extentVector, readOnly);
        }

        /// <summary>
        /// Creates a new NDArrayView which is an alias of this NDArrayView.
        /// </summary>
        /// <param name="readOnly"></param>
        /// <returns></returns>
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
