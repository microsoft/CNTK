//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// ParameterShim.cs -- C# Api for CNTK Parameter class
//
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
