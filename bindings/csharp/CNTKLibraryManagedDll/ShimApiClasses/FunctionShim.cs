//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//
// FunctionShim.cs -- C# Api for CNTK Function class
//
using System;
using System.Collections.Generic;

namespace CNTK
{
    public partial class Function
    {
        /// <summary>
        /// Property Name.
        /// </summary>
        public string Name
        {
            get { return _Name(); }
        }

        /// <summary>
        /// Property Uid.
        /// </summary>
        public string Uid
        {
            get { return _Uid(); }
        }

        /// <summary>
        /// Property RootFunction.
        /// </summary>
        public Function RootFunction
        {
            get { return _RootFunction(); }
        }

        /// <summary>
        /// Property Outputs
        /// </summary>
        public IList<Variable> Outputs
        {
            get
            {
                var varVector = _Outputs();
                var varArray = new Variable[varVector.Count];
                // The CopyTo is to ensure that elements in varVector live beyond the lifecycle of varVector.
                varVector.CopyTo(varArray);
                var varList = new List<Variable>(varArray);
                return varList;
            }
        }

        /// <summary>
        /// Property Output.
        /// </summary>
        public Variable Output
        {
            get { return _Output(); }
        }

        /// <summary>
        /// Property OpName.
        /// </summary>
        public string OpName
        {
            get { return _OpName(); }
        }

        /// <summary>
        /// Property IsComposite.
        /// </summary>
        public bool IsComposite
        {
            get { return _IsComposite(); }
        }

        /// <summary>
        /// Property IsPrimitive.
        /// </summary>
        public bool IsPrimitive
        {
            get { return _IsPrimitive(); }
        }

        /// <summary>
        /// Property IsBlock.
        /// </summary>
        public bool IsBlock
        {
            get { return _IsBlock(); }
        }

        /// <summary>
        /// Property CurrentVersion
        /// </summary>
        public int CurrentVersion
        {
            get { return (int)_CurrentVersion(); }
        }

        /// <summary>
        /// Property Arguments.
        /// </summary>
        public IList<Variable> Arguments
        {
            get
            {
                var varVector = _Arguments();
                var varArray = new Variable[varVector.Count];
                // The CopyTo is to ensure that elements in varVector live beyond the lifecycle of varVector.
                varVector.CopyTo(varArray);
                var varList = new List<Variable>(varArray);
                return varList;
            }
        }

        /// <summary>
        /// Property Inputs.
        /// </summary>
        public IList<Variable> Inputs
        {
            get
            {
                var varVector = _Inputs();
                var varArray = new Variable[varVector.Count];
                // The CopyTo is to ensure that elements in varVector live beyond the lifecycle of varVector.
                varVector.CopyTo(varArray);
                var varList = new List<Variable>(varArray);
                return varList;
            }
        }

        /// <summary>
        /// Parameters if the function
        /// </summary>
        /// <returns></returns>
        public IList<Parameter> Parameters()
        {
            ParameterVector parameterVector = _Parameters();
            return Helper.FromParameterVector(parameterVector);
        }

        /// <summary>
        /// Creates a new cloned function instance. For C# Eval, default ParameterCloningMethod is share.
        /// </summary>
        /// <param name="parameterCloneMethod"></param>
        /// <returns></returns>
        public Function Clone(ParameterCloningMethod parameterCloneMethod = ParameterCloningMethod.Share)
        {
            return _Clone(ParameterCloningMethod.Share);
        }

        /// <summary>
        /// Clones 'this' Function. The parameters of the Function are either cloned, shared or frozen as specified by the parameterCloneMethod argument and
        /// any variable replacements requested are applied in the cloned Function instance.
        /// </summary>
        /// <param name="parameterCloneMethod"></param>
        /// <param name="replacements">existing variables to be replaced with new variables.</param>
        /// <returns></returns>
        public Function Clone(ParameterCloningMethod parameterCloneMethod, IDictionary<Variable, Variable> replacements)
        {
            UnorderedMapVariableVariable replacementVector = Helper.AsUnorderedMapVariableVariable(replacements);
            return _Clone(parameterCloneMethod, replacementVector);
        }

        /// <summary>
        /// Evaluates the Function using provided inputs.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        /// <param name="computeDevice"></param>
        public void Evaluate(IDictionary<Variable, Value> inputs, IDictionary<Variable, Value> outputs, DeviceDescriptor computeDevice)
        {
            Evaluate(inputs, outputs, false, computeDevice);
        }

        /// <summary>
        /// Evaluates the Function using provided inputs.
        /// </summary>
        /// <param name="inputs"></param>
        /// <param name="outputs"></param>
        /// <param name="createPersistentOutputValues"></param>
        /// <param name="computeDevice"></param>
        public void Evaluate(IDictionary<Variable, Value> inputs, IDictionary<Variable, Value> outputs, bool createPersistentOutputValues, DeviceDescriptor computeDevice)
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

        /// <summary>
        /// Find the function with the specified name.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="nestedSearchInsideBlockFunction"></param>
        /// <returns></returns>
        public Function FindByName(string name, bool nestedSearchInsideBlockFunction = false)
        {
            return _FindByName(name, nestedSearchInsideBlockFunction);
        }

        /// <summary>
        /// Finds all functions inside this Functions having the specified name.
        /// </summary>
        /// <param name="name"></param>
        /// <param name="nestedSearchInsideBlockFunction"></param>
        /// <returns></returns>
        public IList<Function> FindAllWithName(string name, bool nestedSearchInsideBlockFunction = false)
        {
            var funcPtrVector = _FindAllWithName(name, nestedSearchInsideBlockFunction);
            var funcPtrList = new List<Function>(funcPtrVector.Count);
            for (int i = 0; i < funcPtrVector.Count; i++)
            {
                // for shared_ptr, the funcPtrVector[i] returns a copy, so it is safe to directly use it in return list.
                funcPtrList.Add(funcPtrVector[i]);
            }
            return funcPtrList;
        }

        public byte[] Save()
        {
            UnsignedCharVector vectorBuf = new UnsignedCharVector(); 
            this._Save(vectorBuf);
            byte[] buffer = new byte[vectorBuf.Count];
            vectorBuf.CopyTo(buffer);
            return buffer;
        }

        public void Save(string filepath)
        {
            this._Save(filepath);
        }

        /// <summary>
        /// Loads a model from file.
        /// </summary>
        /// <param name="filepath"></param>
        /// <param name="computeDevice"></param>
        /// <returns></returns>
        public static Function Load(string filepath, DeviceDescriptor computeDevice, ModelFormat format = ModelFormat.CNTKv2)
        {
            return _Load(filepath, computeDevice, format);
        }

        /// <summary>
        /// Loads a model from memory buffer.
        /// </summary>
        /// <param name="modelBuffer"></param>
        /// <param name="computeDevice"></param>
        /// <returns></returns>
        public static Function Load(byte[] modelBuffer, DeviceDescriptor computeDevice, ModelFormat format = ModelFormat.CNTKv2)
        {
            return _Load(modelBuffer, (uint)modelBuffer.Length, computeDevice, format);
        }

        /// <summary>
        /// Creates a composite function from the rootFunction.
        /// </summary>
        /// <param name="rootFunction"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Function AsComposite(Function rootFunction, string name = "")
        {
            return CNTKLib.AsComposite(rootFunction, name);
        }

        /// <summary>
        /// Create a new Function which is the alias of operand.
        /// </summary>
        /// <param name="operand"></param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Function Alias(Variable operand, string name = "")
        {
            return CNTKLib.Alias(operand, name);
        }

        public Function ReplacePlaceholders(IDictionary<Variable, Variable> placeholderReplacements)
        {
            UnorderedMapVariableVariable unorderedMapVariableVariable = Helper.AsUnorderedMapVariableVariable(placeholderReplacements);
            return ReplacePlaceholders(unorderedMapVariableVariable);
        }

        /// <summary>
        /// Implicitly convert a function to a Variable.
        /// </summary>
        /// <param name="f">The function.</param>
        public static implicit operator Variable(Function f)
        {
            return new Variable(f);
        }

        /// <summary>
        /// Create a new Function instance which just combines the outputs of the specified list of 'operands' Functions such that the 'Outputs' of the 
        /// new 'Function' are union of the 'Outputs' of each of the specified 'operands' Functions.
        /// E.g. When creating a classification model, typically the CrossEntropy loss Function and the ClassificationError Function comprise the two roots
        /// of the computation graph which can be "Combine"d to create a single Function with 2 outputs; viz. CrossEntropy loss and ClassificationError output.
        /// </summary>
        /// <param name="operands">variables whose function are to be combined</param>
        /// <param name="name"></param>
        /// <returns></returns>
        public static Function Combine(IList<Variable> operands, string name = "")
        {
            VariableVector operandVector = Helper.AsVariableVector(operands);
            return CNTKLib.Combine(operandVector, name);
        }

    }
}
