//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// FunctionExtendsions.cs -- Define extension methods for Function.
//
using System;
using System.Collections.Generic;
using System.Linq;

namespace CNTK
{
    public static class FunctionExtensions
    {
        public static void Evaluate(this Function func, Dictionary<Variable, Value> arguments, Dictionary<Variable, Value> outputs, DeviceDescriptor computeDevice)
        {
            // Evaluate the rootFunction.
            var argMap = new UnorderedMapVariableValuePtr();
            foreach (var p in arguments)
            {
                argMap.Add(p.Key, p.Value);
            }

            var outMap = new UnorderedMapVariableValuePtr();
            foreach (var p in outputs)
            {
                outMap.Add(p.Key, p.Value);
            }

            func.Evaluate(argMap, outMap, computeDevice);

            foreach (var p in outMap)
            {
                outputs[p.Key] = p.Value;
            }
        }


    }
}
