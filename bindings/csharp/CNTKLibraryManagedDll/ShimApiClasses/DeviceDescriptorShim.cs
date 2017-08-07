namespace CNTK
{
    public partial class DeviceDescriptor
    {
        // Property Id.
        public int Id
        {
            get { return (int)_Id(); }
        }

        // Property Type.
        public DeviceKind Type
        {
            get { return _Type(); }
        }

        // Property CPUDevice.
        public static DeviceDescriptor CPUDevice
        {
            get { return _CPUDevice(); }
        }

        // Returns the GPUDevice with the specific deviceId.
        public static DeviceDescriptor GPUDevice(int deviceId)
        {
            if (deviceId < 0)
            {
                throw new System.ArgumentException("The paraemter deviceId should not be a negative value");
            }
            return _GPUDevice((uint)deviceId);
        }

        // Gets all devices.
        public static System.Collections.Generic.IList<DeviceDescriptor> AllDevices()
        {
            var deviceVector = _AllDevices();
            // The CopyTo is to ensure the elements in the deviceVector can live beyond deviceVector itself.
            var deviceArray = new DeviceDescriptor[deviceVector.Count];
            deviceVector.CopyTo(deviceArray);
            var deviceList = new System.Collections.Generic.List<DeviceDescriptor>(deviceArray);
            return deviceList;
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
            DeviceDescriptor p = obj as DeviceDescriptor;
            if ((System.Object)p == null)
            {
                return false;
            }

            // Return true if the fields match:
            return CNTKLib.AreEqual(this, p);
        }

        // Value equality.
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

        // Returns hash code value.
        public override int GetHashCode()
        {
            return this._Type().GetHashCode();
        }

        // Set devices to be excluded.
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
