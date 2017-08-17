//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// Helper.cs -- Common helper functions, used internal only.
//

using System;
using System.Collections.Generic;

namespace CNTK
{
    internal class Helper
    {
        /// <summary>
        /// Helper function to create a FloatVector from IEnumerable.
        /// </summary>
        /// <typeparam name="T">The input data type.</typeparam>
        /// <param name="input">Elements to be stored in the FloatVector.</param>
        /// <returns>A FloatVector containing all elements from the IEnumerable input.</returns>
        internal static FloatVector AsFloatVector<T>(IEnumerable<T> input)
        {
            if (typeof(T).Equals(typeof(float)))
            {
                var inputVector = new FloatVector();
                IEnumerable<float> inputInType = input as IEnumerable<float>;
                if (inputInType == null)
                    throw new ArgumentNullException("The parameter cannot be cast as IEnumerable<float>.");
                foreach (var element in inputInType)
                {
                    inputVector.Add(element);
                }
                return inputVector;
            }
            else
            {
                throw new ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }

        /// <summary>
        /// Helper function to create a DoubleVector from IEnumerable.
        /// </summary>
        /// <typeparam name="T">The input data type.</typeparam>
        /// <param name="input">Elements to be stored in the DoubleVector. </param>
        /// <returns>A DoubleVector containing all elements from the IEnumerable input.</returns>
        internal static DoubleVector AsDoubleVector<T>(IEnumerable<T> input)
        {
            if (typeof(T).Equals(typeof(double)))
            {
                var inputVector = new DoubleVector();
                IEnumerable<double> inputInType = input as IEnumerable<double>;
                if (inputInType == null)
                    throw new ArgumentNullException("The parameter cannot be cast as IEnumerable<double>.");
                foreach (var element in inputInType)
                {
                    inputVector.Add(element);
                }
                return inputVector;
            }
            else
            {
                throw new ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }

        /// <summary>
        /// Helper function to create a SizeTVector from IEnumerable.
        /// </summary>
        /// <param name="input">Elements to be stored in the SizeTVector.</param>
        /// <returns>A SizeTVector containing all elements from the IEnumerable input.</returns>
        internal static SizeTVector AsSizeTVector(IEnumerable<int> input)
        {
            var inputVector = new SizeTVector();
            foreach (var element in input)
            {
                if (element < 0)
                {
                    throw new System.ArgumentException("The argument cannot contain a negative value");
                }
                inputVector.Add((uint)element);
            }
            return inputVector;
        }

        /// <summary>
        /// Helper function to create a BoolVector from IEnumerable.
        /// </summary>
        /// <param name="input">Elements to be stored in the BoolVector </param>
        /// <returns>A BoolVector containing all elements from the IEnumerable input.</returns>
        internal static BoolVector AsBoolVector(IEnumerable<bool> input)
        {
            var inputVector = new BoolVector();
            foreach (var element in input)
            {
                inputVector.Add(element);
            }
            return inputVector;
        }

        internal static StreamConfigurationVector AsStreamConfigurationVector(IList<StreamConfiguration> input)
        {
            StreamConfigurationVector inputVector = new StreamConfigurationVector();
            foreach (var element in input)
            {
                inputVector.Add(element);
            }
            return inputVector;
        }

        internal static UnorderedMapStreamInformationPairNDArrayViewPtrNDArrayViewPtr
            AsUnorderedMapStreamInformationPairNDArrayViewPtrNDArrayViewPtr(
            IDictionary<StreamInformation, Tuple<NDArrayView, NDArrayView>> input)
        {
            UnorderedMapStreamInformationPairNDArrayViewPtrNDArrayViewPtr inputVector = new UnorderedMapStreamInformationPairNDArrayViewPtrNDArrayViewPtr();
            foreach (var element in input)
            {
                inputVector.Add(element.Key, new PairNDArrayViewPtrNDArrayViewPtr(element.Value.Item1, element.Value.Item2));
            }
            return inputVector;
        }

        internal static VariableVector AsVariableVector(IList<Variable> input)
        {
            VariableVector inputVector = new VariableVector();
            foreach (var element in input)
            {
                inputVector.Add(element);
            }
            return inputVector;
        }

        internal static ParameterVector AsParameterVector(IList<Parameter> input)
        {
            ParameterVector inputVector = new ParameterVector();
            foreach (var element in input)
            {
                inputVector.Add(element);
            }
            return inputVector;
        }

        internal static LearnerVector AsLearnerVector(IList<Learner> input)
        {
            LearnerVector inputVector = new LearnerVector();
            foreach (var element in input)
            {
                inputVector.Add(element);
            }
            return inputVector;
        }

        internal static UnorderedMapVariableMinibatchData AsUnorderedMapVariableMinibatchData( 
            IDictionary<Variable, MinibatchData> input)
        {
            UnorderedMapVariableMinibatchData inputVector = new UnorderedMapVariableMinibatchData();
            foreach (var element in input)
            {
                inputVector.Add(element.Key, element.Value);
            }
            return inputVector;
        }

    }
}
