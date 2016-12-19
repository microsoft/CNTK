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
        // Each sample consits of a fixed number of elements with type of 'T'. The number of elements is determined by the sample shape.
        // The number of samples = the count of elements in List<T> / the count of elements of the sample
        // The sampleShape should match the shape of the Value object.
        //
        public static void CopyTo<T>(this Value value, NDShape sampelShape, List<List<T>> sequences)
        {
            if ((value.GetDataType() == DataType.Float) && (!typeof(T).Equals(typeof(float))) || 
                (value.GetDataType() == DataType.Double) && (!typeof(T).Equals(typeof(double))))
            {
                throw new ArgumentException("The value type does not match the list type.");
            }

            throw new Exception("Not implemented yet.");
        }

        //
        // Copy the data of the Value object into the buffer provided by 'sequences'.
        // The 'sequences' is a list of sequences with variable length.
        // The number of items contained in the outer list of 'sequences' is the number of sequences in the Value object.
        // Each element of the outer list represents a sequence.
        // Each sequence, represented by List<uint>, contains a variable number of samples. 
        // Each sample is represented by an index of the OneHot vector. The size of the OneHot vector is vocabularySize. 
        // The number of samples = the count of elements in List<uint>.
        //
        public static void CopyTo<T>(this Value value, uint vocabularySize, List<List<uint>> sequences)
        {
            throw new NotImplementedException("Not implemented");
        }
    }
}
