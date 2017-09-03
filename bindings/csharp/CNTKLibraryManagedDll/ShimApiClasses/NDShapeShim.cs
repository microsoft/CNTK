//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// NDShapeShim.cs -- C# Api for CNTK NDShape class
//
using System;
using System.Collections.Generic;

namespace CNTK
{
    public partial class NDShape
    {
        /// <summary>
        /// create a shape
        /// </summary>
        /// <param name="numAxes">number of axes</param>
        /// <param name="dimension">number of dimensions</param>
        public NDShape(int numAxes, int dimension) : this((uint)numAxes, (uint)dimension)
        {
            if (numAxes < 0 || dimension < 0)
            {
                throw new ArgumentException("The parameter numAxes or dimension should not be a negative value");
            }
        }

        /// <summary>
        /// construct a shape
        /// </summary>
        /// <param name="numAxes">number of axes</param>
        public NDShape(int numAxes) : this((uint)numAxes)
        {
            if (numAxes < 0)
            {
                throw new ArgumentException("The parameter numAxes should not be a negative value");
            }
        }

        /// <summary>
        /// implicitly convert a int array to a shape
        /// </summary>
        /// <param name="dim"></param>
        public static implicit operator NDShape(int[] dim)
        {
            return NDShape.CreateNDShape(dim);
        }

        /// <summary>
        /// Property Rank.
        /// </summary>
        public int Rank
        {
            get { return (int)_Rank(); }
        }

        /// <summary>
        /// Property Dimensions.
        /// </summary>
        public IList<int> Dimensions
        {
            get
            {
                var dimList = _Dimensions();
                var retList = new List<int>(dimList.Count);
                foreach (var element in dimList)
                {
                    retList.Add((int)element);
                }
                return retList;
            }
        }

        /// <summary>
        /// Property IsUnknown.
        /// </summary>
        public bool IsUnknown
        {
            get { return _IsUnknown(); }
        }

        /// <summary>
        /// Property HasInferredDimension.
        /// </summary>
        public bool HasInferredDimension
        {
            get { return _HasInferredDimension(); }
        }

        /// <summary>
        /// Property HasFreeDimension.
        /// </summary>
        public bool HasFreeDimension
        {
            get { return _HasFreeDimension(); }
        }

        /// <summary>
        /// Property HasUnboundDimension.
        /// </summary>
        public bool HasUnboundDimension
        {
            get { return _HasUnboundDimension(); }
        }

        /// <summary>
        /// Property TotalSize.
        /// </summary>
        public int TotalSize
        {
            get { return (int)_TotalSize(); }
        }

        /// <summary>
        /// Indexer operator
        /// </summary>
        /// <param name="key"></param>
        /// <returns></returns>
        public int this[int key]
        {
            get { return (int)_DimensionSize((uint)key); }
        }

        /// <summary>
        /// Returns a subshape.
        /// </summary>
        /// <param name="beginAxisId"></param>
        /// <param name="endAxisId"></param>
        /// <returns></returns>
        public NDShape SubShape(int beginAxisId, int endAxisId)
        {
            if (beginAxisId < 0 || endAxisId < 0)
            {
                throw new ArgumentException("The parameter beginAxisId or endAxisId should not be a negative value");
            }
            return _SubShape((uint)beginAxisId, (uint)endAxisId);
        }

        /// <summary>
        /// Returns a subshape.
        /// </summary>
        /// <param name="beginAxisId"></param>
        /// <returns></returns>
        public NDShape SubShape(int beginAxisId = 0)
        {
            if (beginAxisId < 0)
            {
                throw new ArgumentException("The parameter beginAxisId should not be a negative value");
            }
            return _SubShape((uint)beginAxisId);
        }

        /// <summary>
        /// Creates a new NDShape.
        /// </summary>
        /// <param name="dimensions"></param>
        /// <returns></returns>
        public static NDShape CreateNDShape(IEnumerable<int> dimensions)
        {
            var dimVector = new SizeTVector();
            foreach (var element in dimensions)
            {
                if (element < 0 && !IsSpecialDimensionValues(element))
                {
                    throw new ArgumentException("The parameter diemnsions cannot contain a negative value");
                }
                CSharp_SizeTVector_AddExt(dimVector, DimConvertCSToCPP(element));
            }
            return new NDShape(dimVector);
        }

        /// <summary>
        /// Value equality.
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(Object obj)
        {
            // If parameter is null return false.
            if (obj == null)
            {
                return false;
            }

            // If parameter cannot be cast to Point return false.
            NDShape p = obj as NDShape;
            if ((Object)p == null)
            {
                return false;
            }

            // Return true if the fields match:
            return CNTKLib.AreEqual(this, p);
        }

        /// <summary>
        /// Value Equality.
        /// </summary>
        /// <param name="p"></param>
        /// <returns></returns>
        public bool Equals(NDShape p)
        {
            // If parameter is null return false:
            if ((object)p == null)
            {
                return false;
            }

            // Return true if the fields match:
            return CNTKLib.AreEqual(this, p);
        }

        /// <summary>
        /// Returns hash code value.
        /// </summary>
        /// <returns></returns>
        public override int GetHashCode()
        {
            //Todo: another hash function??
            return this._Dimensions().GetHashCode();
        }

        // special constants used to represent the nature of a dimension
        private const ulong InferredDimensionUL = ulong.MaxValue;
        private const ulong FreeDimensionUL = ulong.MaxValue - 2;

        public static readonly int InferredDimension = -1;
        public static readonly int FreeDimension = -3;

        /// <summary>
        /// convert a dimension from ulong to a int. 
        /// CNTK Cpp code use 64bit for dimension. C# use 32bit for dimension. 
        /// </summary>
        /// <param name="dimUL"></param>
        /// <returns></returns>
        internal static int DimConvertCPPToCS(ulong dimUL)
        {
            // down casting keeps the special dimmension values.
            return (int)(uint)dimUL;
        }

        /// <summary>
        /// when converting dimensions form C# to Cpp, we need to 
        /// </summary>
        /// <param name="dim"></param>
        /// <returns></returns>
        internal static ulong DimConvertCSToCPP(int dim)
        {
            // need to maintain dimension values during upcast 
            if (dim == InferredDimension)
            {
                return InferredDimensionUL;
            }
            else if (dim == FreeDimension)
            {
                return FreeDimensionUL;
            }
            else
            {
                return (ulong)dim;
            }
        }

        private static bool IsSpecialDimensionValues(int dim)
        {
            return dim == InferredDimension || dim == FreeDimension;
        }
    }
}
