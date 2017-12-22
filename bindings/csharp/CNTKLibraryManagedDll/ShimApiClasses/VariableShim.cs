//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// VariableShim.cs -- C# Api for CNTK Variable class
//
using System;
using System.Collections.Generic;

namespace CNTK
{
    public partial class Variable
    {
        /// <summary>
        /// Property Shape.
        /// </summary>
        public NDShape Shape
        {
            get { return _Shape(); }
        }

        /// <summary>
        /// Property Name.
        /// </summary>
        public string Name
        {
            get { return _Name(); }
        }

        /// <summary>
        /// Property Uid.
        /// </summary>
        public string Uid
        {
            get { return _Uid(); }
        }

        /// <summary>
        /// Property Kind.
        /// </summary>
        public VariableKind Kind
        {
            get { return _Kind(); }
        }

        /// <summary>
        /// Property DataType.
        /// </summary>
        public DataType DataType
        {
            get { return _GetDataType(); }
        }

        /// <summary>
        /// Property DynamicAxes.
        /// </summary>
        public IList<Axis> DynamicAxes
        {
            get
            {
                var axisVector = _DynamicAxes();
                // The CopyTo is to ensure that elements in axisVector live beyond the lifecycle of axisVector.
                var axisArray = new Axis[axisVector.Count];
                axisVector.CopyTo(axisArray);
                var axisList = new List<Axis>(axisArray);
                return axisList;
            }
        }

        /// <summary>
        /// Property IsSparse.
        /// </summary>
        public bool IsSparse
        {
            get { return _IsSparse(); }
        }

        /// <summary>
        /// Property IsInput.
        /// </summary>
        public bool IsInput
        {
            get { return _IsInput(); }
        }

        /// <summary>
        /// Property IsOutput.
        /// </summary>
        public bool IsOutput
        {
            get { return _IsOutput(); }
        }

        /// <summary>
        /// Property IsParameter.
        /// </summary>
        public bool IsParameter
        {
            get { return _IsParameter(); }
        }

        /// <summary>
        /// Property IsConstant.
        /// </summary>
        public bool IsConstant
        {
            get { return _IsConstant(); }
        }

        /// <summary>
        /// Property IsPlaceholder.
        /// </summary>
        public bool IsPlaceholder
        {
            get { return _IsPlaceholder(); }
        }

        /// <summary>
        /// Property Owner.
        /// </summary>
        public Function Owner
        {
            get { return _Owner(); }
        }

        /// <summary>
        /// Property NeedsGradient.
        /// </summary>
        public bool NeedsGradient
        {
            get { return _NeedsGradient(); }
        }

        /// <summary>
        /// Property CurrentValueTimeStamp
        /// </summary>
        public int CurrentValueTimeStamp
        {
            get { return (int)_CurrentValueTimeStamp(); }
        }

        /// <summary>
        /// Value equality.
        /// </summary>
        /// <param name="obj"></param>
        /// <returns>true when equal</returns>
        public override bool Equals(Object obj)
        {
            // If parameter is null return false.
            if (obj == null)
            {
                return false;
            }

            // If parameter cannot be cast to Point return false.
            Variable p = obj as Variable;
            if ((Object)p == null)
            {
                return false;
            }

            // Return true if the fields match:
            return CNTKLib.AreEqual(this, p);
        }

        /// <summary>
        /// Value equality.
        /// </summary>
        /// <param name="p"></param>
        /// <returns>true if equal</returns>
        public bool Equals(Variable p)
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
            // Todo: the hash value in C++ is size_t, but only in C#
            return (int)_GetHashValue();
        }

        /// <summary>
        /// Implicitly construct a Function from a Variable
        /// </summary>
        /// <param name="v"></param>
        public static implicit operator Function(Variable v)
        {
            return v.ToFunction();
        }

        /// <summary>
        /// plus operator of 2 Variables
        /// </summary>
        /// <param name="v1"></param>
        /// <param name="v2"></param>
        /// <returns></returns>
        public static Function operator +(Variable v1, Variable v2)
        {
            return CNTKLib.Plus(v1, v2);
        }

        /// <summary>
        /// times operator of 2 Variables
        /// </summary>
        /// <param name="v1"></param>
        /// <param name="v2"></param>
        /// <returns></returns>
        public static Function operator *(Variable v1, Variable v2)
        {
            return CNTKLib.Times(v1, v2);
        }

        /// <summary>
        /// Create an 'Input' Variable denoting sparse data and specify if gradients are to be computed for this input
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="isSparse"></param>
        /// <param name="dataType"></param>
        /// <param name="needsGradient"></param>
        /// <param name="name"></param>
        /// <param name="dynamicAxes"></param>
        /// <returns></returns>
        public static Variable InputVariable(NDShape shape, DataType dataType, string name = "", IList<Axis> dynamicAxes = null, bool isSparse = false, bool needsGradient = false)
        {
            if (dynamicAxes == null)
                dynamicAxes = Axis.DefaultInputVariableDynamicAxes();
            AxisVector dynamicAxesVector = Helper.AsAxisVector(dynamicAxes);
            return CNTKLib.InputVariable(shape, isSparse, dataType, needsGradient, name, dynamicAxesVector);
        }

        /// <summary>
        /// Create a Placeholder variable to be used as a temporary/placeholder input to a Function.
        /// All placeholder inputs of a Function must be replaced with non-placeholder Variables before Forward evaluation of the Function.
        /// </summary>
        /// <param name="shape"></param>
        /// <param name="dynamicAxes"></param>
        /// <returns></returns>
        public static Variable PlaceholderVariable(NDShape shape, IList<Axis> dynamicAxes)
        {
            AxisVector dynamicAxesVector = Helper.AsAxisVector(dynamicAxes);
            return CNTKLib.PlaceholderVariable(shape, dynamicAxesVector);
        }
    }
}
