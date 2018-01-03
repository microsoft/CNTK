//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// AxisShim.cs -- C# Api for CNTK Axis class
//
using System;

namespace CNTK
{
    public partial class Axis
    {
        /// <summary>
        /// Property Name.
        /// </summary>
        public string Name
        {
            get { return _Name(); }
        }

        /// <summary>
        /// Property IsStatic.
        /// </summary>
        public bool IsStatic
        {
            get { return _IsStaticAxis(); }
        }

        /// <summary>
        /// Property IsDynamic.
        /// </summary>
        public bool IsDynamic
        {
            get { return _IsDynamicAxis(); }
        }

        /// <summary>
        /// Property IsOrdered.
        /// </summary>
        public bool IsOrdered
        {
            get { return _IsOrdered(); }
        }

        /// <summary>
        /// Returns index of this Axis.
        /// </summary>
        /// <param name="checkStaticAxis"></param>
        /// <returns>the index of this Axis</returns>
        public int StaticAxisIndex(bool checkStaticAxis = true)
        {
            return _StaticAxisIndex(checkStaticAxis);
        }

        /// <summary>
        /// Value equality.
        /// </summary>
        /// <param name="obj">The object to compare with.</param>
        /// <returns>Returns true if they are equal.</returns>
        public override bool Equals(Object obj)
        {
            // If parameter is null return false.
            if (obj == null)
            {
                return false;
            }

            // If parameter cannot be cast to Point return false.
            Axis p = obj as Axis;
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
        /// <param name="p">The Axis to compare with.</param>
        /// <returns>Returns true if they are equal.</returns>
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

        /// <summary>
        /// Calculates hash code value.
        /// </summary>
        /// <returns>The hash code value.</returns>
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
