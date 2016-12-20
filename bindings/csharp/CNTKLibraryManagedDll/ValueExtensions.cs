//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ValueExtendsions.cs -- Define extension methods for Value.
//
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace CNTK
{
    public static class ValueExtensions
    {
        //
        // Copy the data of the Value object into the buffer provided by 'sequences'.
        // The 'sequences' is a list of sequences with variable length. 
        // The number of items contained in the outer list of 'sequences' is the number of sequences in the Value object.
        // Each element of the outer list represents a sequence.
        // Each sequence, represented by List<T>, contains a variable number of samples. 
        // Each sample consits of a fixed number of elements with type of 'T'. The number of elements is determined by the variable shape.
        // The number of samples = the count of elements in List<T> / the count of elements of the sample
        // The shape of the variable should match the shape of the Value object.
        //
        public static void CopyTo<T>(this Value value, Variable sampleVariable, List<List<T>> sequences)
        {
            if (typeof(T).Equals(typeof(float)))
            {
                if (value.GetDataType() != DataType.Float)
                {
                    throw new ArgumentException("The value type does not match the list type.");
                }

                var seqVec = new FloatVectorVector();
                value.CopyToFloat(sampleVariable, seqVec);
                sequences.Clear();
                foreach (var seq in seqVec)
                {
                    var seqList = seq as IEnumerable<T>;
                    if (seqList == null)
                        throw new TypeAccessException("Cannot convert to the value type.");
                    sequences.Add(new List<T>(seqList));
                }
            }
            else if (typeof(T).Equals(typeof(double)))
            {
                if (value.GetDataType() != DataType.Double)
                {
                    throw new ArgumentException("The value type does not match the list type.");
                }

                var seqVec = new DoubleVectorVector();
                value.CopyToDouble(sampleVariable, seqVec);
                sequences.Clear();
                foreach (var seq in seqVec)
                {
                    var seqList = seq as IEnumerable<T>;
                    if (seqList == null)
                        throw new TypeAccessException("Cannot convert to the value type.");
                    sequences.Add(new List<T>(seqList));
                }
            }
            else
            {
                throw new ArgumentException("The value type does not match the list type.");
            }
        }

        //
        // Copy the data of the Value object into the buffer provided by 'sequences'.
        // The 'sequences' is a list of sequences with variable length.
        // The number of items contained in the outer list of 'sequences' is the number of sequences in the Value object.
        // Each element of the outer list represents a sequence.
        // Each sequence, represented by List<uint>, contains a variable number of samples. 
        // Each sample is represented by an index of the OneHot vector. The size of the OneHot vector should match that defined in the variable. 
        // The number of samples = the count of elements in List<uint>.
        //
        public static void CopyTo(this Value value, Variable sampleVariable, List<List<uint>> sequences)
        {
            if (sampleVariable.Shape[0] != sampleVariable.Shape.TotalSize)
            {
                throw new ArgumentException("The sample variable's leading axis dimensionality must equal to the total size of the shape for sparse data");
            }

            var seqVec = new SizeTVectorVector();
            value.CopyTo(sampleVariable, seqVec);

            sequences.Clear();
            foreach(var seq in seqVec)
            {
                sequences.Add(new List<uint>(seq));
            }
            return;
        }
    }
}
