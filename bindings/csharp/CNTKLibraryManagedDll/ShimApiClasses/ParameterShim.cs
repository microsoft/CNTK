using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK
{
    public partial class Parameter
    {
        public Parameter(NDShape shape, float initValue, DeviceDescriptor device, string name) : 
            this(shape, DataType.Float, CNTKLib.ConstantInitializer(initValue), device, name)
        { }

        public Parameter(NDShape shape, double initValue, DeviceDescriptor device, string name) :
            this(shape, DataType.Double, CNTKLib.ConstantInitializer(initValue), device, name)
        { }
    }
}
