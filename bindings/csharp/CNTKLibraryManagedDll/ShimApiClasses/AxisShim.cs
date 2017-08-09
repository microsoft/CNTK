//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// AxisShim.cs -- C# Api for CNTK Axis class
//
namespace CNTK
{
    public partial class Axis
    {
        // Property Name.
        public string Name
        {
            get { return _Name(); }
        }

        // Property IsStatic.
        public bool IsStatic
        {
            get { return _IsStaticAxis(); }
        }

        // Property IsDynamic.
        public bool IsDynamic
        {
            get { return _IsDynamicAxis(); }
        }

        // Property IsOrdered.
        public bool IsOrdered
        {
            get { return _IsOrdered(); }
        }

        // Returns index of this Axis.
        public int StaticAxisIndex(bool checkStaticAxis = true)
        {
            return _StaticAxisIndex(checkStaticAxis);
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
            Axis p = obj as Axis;
            if ((System.Object)p == null)
            {
                return false;
            }

            // Return true if the fields match:
            return CNTKLib.AreEqual(this, p);
        }

        // Value equality.
        public bool Equals(Axis p)
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
            if (this._IsDynamicAxis())
            {
                return this.Name.GetHashCode();
            }
            else
            {
                return this.StaticAxisIndex(false).GetHashCode();
            }
        }
    }
}
