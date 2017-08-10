//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// VariableShim.cs -- C# Api for CNTK Variable class
//
namespace CNTK
{
    public partial class Variable
    {
        // Property Shape.
        public NDShape Shape
        {
            get { return _Shape(); }
        }

        // Property Name.
        public string Name
        {
            get { return _Name(); }
        }

        // Property Uid.
        public string Uid
        {
            get { return _Uid(); }
        }

        // Property Kind.
        public VariableKind Kind
        {
            get { return _Kind(); }
        }

        // Property DataType.
        public DataType DataType
        {
            get { return _GetDataType(); }
        }

        // Property DynamicAxes.
        public System.Collections.Generic.IList<Axis> DynamicAxes
        {
            get
            {
                var axisVector = _DynamicAxes();
                // The CopyTo is to ensure that elements in axisVector live beyond the lifecycle of axisVector.
                var axisArray = new Axis[axisVector.Count];
                axisVector.CopyTo(axisArray);
                var axisList = new System.Collections.Generic.List<Axis>(axisArray);
                return axisList;
            }
        }

        // Property IsSparse.
        public bool IsSparse
        {
            get { return _IsSparse(); }
        }

        // Property IsInput.
        public bool IsInput
        {
            get { return _IsInput(); }
        }

        // Property IsOutput.
        public bool IsOutput
        {
            get { return _IsOutput(); }
        }

        // Property IsParameter.
        public bool IsParameter
        {
            get { return _IsParameter(); }
        }

        // Property IsConstant.
        public bool IsConstant
        {
            get { return _IsConstant(); }
        }

        // Property IsPlaceholder.
        public bool IsPlaceholder
        {
            get { return _IsPlaceholder(); }
        }

        // Property Owner.
        public Function Owner
        {
            get { return _Owner(); }
        }

        // Property NeedsGradient.
        public bool NeedsGradient
        {
            get { return _NeedsGradient(); }
        }

        // Property CurrentValueTimeStamp
        public int CurrentValueTimeStamp
        {
            get { return (int)_CurrentValueTimeStamp(); }
        }

        // Value equality.
        public override bool Equals(System.Object obj)
        {
            // If parameter is null return false.
            if (obj == null)
            {
                return false;
            }

            // If parameter cannot be cast to Point return false.
            Variable p = obj as Variable;
            if ((System.Object)p == null)
            {
                return false;
            }

            // Return true if the fields match:
            return CNTKLib.AreEqual(this, p);
        }

        // Value equality.
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

        // Returns hash code value.
        public override int GetHashCode()
        {
            // Todo: the hash value in C++ is size_t, but only in C#
            return (int)_GetHashValue();
        }

        public static implicit operator Function(Variable v)
        {
            return v.ToFunction();
        }

        public static Function operator +(Variable v1, Variable v2)
        {
            return CNTKLib.Plus(v1, v2);
        }

        public static Function operator *(Variable v1, Variable v2)
        {
            return CNTKLib.Times(v1, v2);
        }

        public static Variable InputVariable(NDShape shape, DataType dataType, string name)
        {
            return CNTKLib.InputVariable(shape, dataType, name);
        }
    }
}
