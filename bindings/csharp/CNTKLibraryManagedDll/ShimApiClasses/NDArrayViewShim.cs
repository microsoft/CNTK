//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// NDArrayViewShim.cs -- C# Api for CNTK NDArrayView class
//
using System;
using System.Collections.Generic;

namespace CNTK
{
    public partial class NDArrayView
    {
        /// <summary>
        /// Constructor using float dense input.
        /// </summary>
        /// <param name="viewShape">shape of the data</param>
        /// <param name="dataBuffer">data buffer</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether the data is readonly</param>
        public NDArrayView(NDShape viewShape, float[] dataBuffer, DeviceDescriptor device, bool readOnly = false) : this(viewShape, dataBuffer, (uint)dataBuffer.Length, device, readOnly)
        {
        }

        /// <summary>
        /// Constructor using double dense input.
        /// </summary>
        /// <param name="viewShape">shape of the data</param>
        /// <param name="dataBuffer">data buffer</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether the data is readonly</param>
        public NDArrayView(NDShape viewShape, double[] dataBuffer, DeviceDescriptor device, bool readOnly = false) : this(viewShape, dataBuffer, (uint)dataBuffer.Length, device, readOnly)
        {
        }

        /// <summary>
        /// Constructor using float sparse input.
        /// </summary>
        /// <param name="viewShape">shape of the date</param>
        /// <param name="colStarts">starting colomn</param>
        /// <param name="rowIndices">list of column indices</param>
        /// <param name="nonZeroValues">sparse values</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether the data is readonly</param>
        public NDArrayView(NDShape viewShape, int[] colStarts, int[] rowIndices, float[] nonZeroValues, DeviceDescriptor device, bool readOnly = false) : this(viewShape, colStarts, rowIndices, nonZeroValues, (uint)nonZeroValues.Length, device, readOnly)
        {
            if (rowIndices.Length != nonZeroValues.Length)
            {
                throw new ArgumentException("The length of rowIndicies must be same as the length of nonZeroValues.");
            }
            if (viewShape[viewShape.Rank - 1] + 1 != colStarts.Length)
            {
                throw new ArgumentException("The length of colStarts does not match the number of rows, i.e. the dimension size of the last rank of viewShape.");
            }
        }

        /// <summary>
        /// Constructor using double sparse input.
        /// </summary>
        /// <param name="viewShape">shape of the data</param>
        /// <param name="colStarts">starting column</param>
        /// <param name="rowIndices">list of row indices</param>
        /// <param name="nonZeroValues">sparse data</param>
        /// <param name="device">device</param>
        /// <param name="readOnly">whether the data is readonly</param>
        public NDArrayView(NDShape viewShape, int[] colStarts, int[] rowIndices, double[] nonZeroValues, DeviceDescriptor device, bool readOnly = false) : this(viewShape, colStarts, rowIndices, nonZeroValues, (uint)nonZeroValues.Length, device, readOnly)
        {
            if (rowIndices.Length != nonZeroValues.Length)
            {
                throw new ArgumentException("The length of rowIndicies must be same as the length of nonZeroValues.");
            }
            if (viewShape[viewShape.Rank - 1] + 1 != colStarts.Length)
            {
                throw new ArgumentException("The length of colStarts does not match the number of rows, i.e. the dimension size of the last rank of viewShape.");
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
        public NDArrayView SliceView(IEnumerable<int> startOffset, IEnumerable<int> extent, bool readOnly = false)
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

        /// <summary>
        /// Generate a normal distribution random data NDArrayView 
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="shape">data shape</param>
        /// <param name="mean">mean of the data</param>
        /// <param name="stdDev">standard deviation</param>
        /// <param name="seed">seed</param>
        /// <param name="device">device</param>
        /// <returns></returns>
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
                throw new ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }

        /// <summary>
        /// Generate a uniform distribution random data NDArrayView
        /// </summary>
        /// <typeparam name="T">the data type</typeparam>
        /// <param name="shape">shape of the data</param>
        /// <param name="rangeBegin">low end value</param>
        /// <param name="rangeEnd">high end value</param>
        /// <param name="seed">seed</param>
        /// <param name="device">device</param>
        /// <returns></returns>
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
                throw new ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }
    }
}
