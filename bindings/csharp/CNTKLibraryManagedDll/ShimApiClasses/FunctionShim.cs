//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// FunctionShim.cs -- C# Api for CNTK Function class
//
namespace CNTK
{
    public partial class Function
    {
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

        // Property RootFunction.
        public Function RootFunction
        {
            get { return _RootFunction(); }
        }

        // Property Outputs
        public System.Collections.Generic.IList<Variable> Outputs
        {
            get
            {
                var varVector = _Outputs();
                var varArray = new Variable[varVector.Count];
                // The CopyTo is to ensure that elements in varVector live beyond the lifecycle of varVector.
                varVector.CopyTo(varArray);
                var varList = new System.Collections.Generic.List<Variable>(varArray);
                return varList;
            }
        }

        // Property Output.
        public Variable Output
        {
            get { return _Output(); }
        }

        // Property OpName.
        public string OpName
        {
            get { return _OpName(); }
        }

        // Property IsComposite.
        public bool IsComposite
        {
            get { return _IsComposite(); }
        }

        // Property IsPrimitive.
        public bool IsPrimitive
        {
            get { return _IsPrimitive(); }
        }

        // Property IsBlock.
        public bool IsBlock
        {
            get { return _IsBlock(); }
        }

        // Property CurrentVersion
        public int CurrentVersion
        {
            get { return (int)_CurrentVersion(); }
        }

        // Property Arguments.
        public System.Collections.Generic.IList<Variable> Arguments
        {
            get
            {
                var varVector = _Arguments();
                var varArray = new Variable[varVector.Count];
                // The CopyTo is to ensure that elements in varVector live beyond the lifecycle of varVector.
                varVector.CopyTo(varArray);
                var varList = new System.Collections.Generic.List<Variable>(varArray);
                return varList;
            }
        }

        // Property Inputs.
        public System.Collections.Generic.IList<Variable> Inputs
        {
            get
            {
                var varVector = _Inputs();
                var varArray = new Variable[varVector.Count];
                // The CopyTo is to ensure that elements in varVector live beyond the lifecycle of varVector.
                varVector.CopyTo(varArray);
                var varList = new System.Collections.Generic.List<Variable>(varArray);
                return varList;
            }
        }

        // Creates a new cloned function instance. For C# Eval, default ParameterCloningMethod is share.
        public Function Clone(ParameterCloningMethod parameterCloneMethod = ParameterCloningMethod.Share)
        {
            return _Clone(ParameterCloningMethod.Share);
        }

        // Evaluates the Function using provided inputs.
        public void Evaluate(System.Collections.Generic.IDictionary<Variable, Value> inputs, System.Collections.Generic.IDictionary<Variable, Value> outputs, DeviceDescriptor computeDevice)
        {
            Evaluate(inputs, outputs, false, computeDevice);
        }

        // Evaluates the Function using provided inputs.
        public void Evaluate(System.Collections.Generic.IDictionary<Variable, Value> inputs, System.Collections.Generic.IDictionary<Variable, Value> outputs, bool createPersistentOutputValues, DeviceDescriptor computeDevice)
        {
            // Evaluate the rootFunction.
            var inMap = new UnorderedMapVariableValuePtr();
            var outMap = new UnorderedMapVariableValuePtr();
            foreach (var p in inputs)
            {
                inMap.Add(p.Key, p.Value);
            }

            foreach (var p in outputs)
            {
                outMap.Add(p.Key, p.Value);
            }

            _Evaluate(inMap, outMap, computeDevice);

            foreach (var p in outMap)
            {
                if (createPersistentOutputValues && (outputs[p.Key] == null))
                {
                    outputs[p.Key] = p.Value.DeepClone();
                }
                else
                {
                    // for shared_ptr<Value>, the p.Value returns a copy, so it is safe to use it directly in outputs.
                    outputs[p.Key] = p.Value;
                }
            }
        }

        // Find the function with the specified name.
        public Function FindByName(string name, bool nestedSearchInsideBlockFunction = false)
        {
            return _FindByName(name, nestedSearchInsideBlockFunction);
        }

        // Finds all functions inside this Functions having the specified name.
        public System.Collections.Generic.IList<Function> FindAllWithName(string name, bool nestedSearchInsideBlockFunction = false)
        {
            var funcPtrVector = _FindAllWithName(name, nestedSearchInsideBlockFunction);
            var funcPtrList = new System.Collections.Generic.List<Function>(funcPtrVector.Count);
            for (int i = 0; i < funcPtrVector.Count; i++)
            {
                // for shared_ptr, the funcPtrVector[i] returns a copy, so it is safe to directly use it in return list.
                funcPtrList.Add(funcPtrVector[i]);
            }
            return funcPtrList;
        }

        // Loads a model from file.
        public static Function Load(string filepath, DeviceDescriptor computeDevice)
        {
            return _Load(filepath, computeDevice);
        }

        // Loads a model from memory buffer.
        public static Function Load(byte[] modelBuffer, DeviceDescriptor computeDevice)
        {
            return _Load(modelBuffer, (uint)modelBuffer.Length, computeDevice);
        }

        // Creates a new Function from specified operands.
        public static Function Combine(System.Collections.Generic.IEnumerable<Variable> operands)
        {
            var varVect = new VariableVector();
            foreach (var v in operands)
            {
                varVect.Add(v);
            }
            return CNTKLib.Combine(varVect);
        }

        // Creates a composite function from the rootFunction.
        public static Function AsComposite(Function rootFunction, string name = "")
        {
            return CNTKLib.AsComposite(rootFunction, name);
        }

        // Create a new Function which is the alias of operand.
        public static Function Alias(Variable operand, string name = "")
        {
            return CNTKLib.Alias(operand, name);
        }

        public static implicit operator Variable(Function f)
        {
            return new Variable(f);
        }
    }
}
