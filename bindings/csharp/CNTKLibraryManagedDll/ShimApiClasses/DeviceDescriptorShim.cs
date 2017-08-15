//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// DeviceDescriptorShim.cs -- C# Api for CNTK DeviceDescriptor class
//
namespace CNTK
{
    public partial class DeviceDescriptor
    {
        /// <summary>
        /// Property Id.
        /// </summary>
        public int Id
        {
            get { return (int)_Id(); }
        }

        /// <summary>
        /// Property Type.
        /// </summary>
        public DeviceKind Type
        {
            get { return _Type(); }
        }

        /// <summary>
        /// Property CPUDevice.
        /// </summary>
        public static DeviceDescriptor CPUDevice
        {
            get { return _CPUDevice(); }
        }

        /// <summary>
        /// Returns the GPUDevice with the specific deviceId.
        /// </summary>
        /// <param name="deviceId"></param>
        /// <returns></returns>
        public static DeviceDescriptor GPUDevice(int deviceId)
        {
            if (deviceId < 0)
            {
                throw new System.ArgumentException("The paraemter deviceId should not be a negative value");
            }
            return _GPUDevice((uint)deviceId);
        }

        /// <summary>
        /// Gets all devices.
        /// </summary>
        /// <returns></returns>
        public static System.Collections.Generic.IList<DeviceDescriptor> AllDevices()
        {
            var deviceVector = _AllDevices();
            // The CopyTo is to ensure the elements in the deviceVector can live beyond deviceVector itself.
            var deviceArray = new DeviceDescriptor[deviceVector.Count];
            deviceVector.CopyTo(deviceArray);
            var deviceList = new System.Collections.Generic.List<DeviceDescriptor>(deviceArray);
            return deviceList;
        }

        /// <summary>
        /// Value equality.
        /// </summary>
        /// <param name="obj"></param>
        /// <returns></returns>
        public override bool Equals(System.Object obj)
        {
            // If parameter is null return false.
            if (obj == null)
            {
                return false;
            }

            // If parameter cannot be cast to Point return false.
            DeviceDescriptor p = obj as DeviceDescriptor;
            if ((System.Object)p == null)
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
        /// <returns></returns>
        public bool Equals(DeviceDescriptor p)
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
            return this._Type().GetHashCode();
        }

        /// <summary>
        /// Set devices to be excluded.
        /// </summary>
        /// <param name="excluded"></param>
        public static void SetExcludedDevices(System.Collections.Generic.IEnumerable<DeviceDescriptor> excluded)
        {
            var excludeVector = new DeviceDescriptorVector();
            foreach (var element in excluded)
            {
                excludeVector.Add(element);
            }
            _SetExcludedDevices(excludeVector);
        }
    }
}
