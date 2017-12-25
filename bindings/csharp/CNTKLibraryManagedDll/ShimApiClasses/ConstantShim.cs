using System;

namespace CNTK
{
    public partial class Constant
    {
        /// <summary>
        /// create a constant
        /// </summary>
        /// <param name="shape">shape of the constant</param>
        /// <param name="initValue">initial value</param>
        /// <param name="device">device</param>
        /// <param name="name">name</param>
        public Constant(NDShape shape, float initValue, DeviceDescriptor device, string name = "") :
            this(shape, DataType.Float, initValue, device, name)
        {

        }
        /// <summary>
        /// Create a scalar constant. The specified value is cast to the specified DataType
        /// </summary>
        /// <typeparam name="T">data type</typeparam>
        /// <param name="value">initial value</param>
        /// <param name="device">device</param>
        /// <returns></returns>
        static public Constant Scalar<T>(T value, DeviceDescriptor device)
        {
            if (device == null)
            {
                device = DeviceDescriptor.CPUDevice;
            }

            if (typeof(T).Equals(typeof(float)))
            {
                return _ScalarFloat((float)(object)value, device);
            }
            else if (typeof(T).Equals(typeof(double)))
            {
                return _ScalarDouble((double)(object)value, device);
            }
            else
            {
                throw new ArgumentException("The data type " + typeof(T).ToString() + " is not supported. Only float or double is supported by CNTK.");
            }
        }
    }
}
