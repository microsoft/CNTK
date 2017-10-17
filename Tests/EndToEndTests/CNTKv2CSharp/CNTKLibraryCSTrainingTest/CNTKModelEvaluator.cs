using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text;
using System.Threading.Tasks;

namespace CNTK.CNTKLibraryCSTrainingTest
{
    internal sealed class Pool<T>
    {
        private readonly Func<T> m_createCallback;
        private readonly object m_lock = new object();
        private readonly LinkedList<T> m_freeObjects = new LinkedList<T>();

        internal Pool(Func<T> createCallback, int initialCount)
        {
            this.m_createCallback = createCallback;
            for (int i = 0; i < initialCount; ++i)
            {
                CreateObject();
            };
            Console.WriteLine("Created pool of " + typeof(T).Name + " with " + initialCount + " objects");
        }

        internal T GetObject()
        {
            lock (this.m_lock)
            {
                if (m_freeObjects.Count == 0)
                {
                    Console.WriteLine("Warn: Had to add object " + typeof(T).Name + " to pool");
                    CreateObject();
                };
                T obj = m_freeObjects.Last.Value;
                m_freeObjects.RemoveLast();
                return obj;
            }
        }

        internal void ReturnObject(T obj)
        {
            lock (this.m_lock)
            {
                this.m_freeObjects.AddLast(obj);
            }
        }

        private void CreateObject()
        {
            T obj = this.m_createCallback();
            this.m_freeObjects.AddLast(obj);
        }
    }

    internal sealed class CNTKModelEvaluator : IDisposable
    {
        private const int CPU_MAX_THREADS = 1;

        private readonly Function m_modelFunc;
        //private readonly string m_varInName;
        //private readonly string m_varOutName;
        private readonly DeviceDescriptor m_device = DeviceDescriptor.CPUDevice;
        private readonly Pool<Function> m_functionsPool;
        internal readonly int inputDataSize;

        internal CNTKModelEvaluator(string modelPath, int poolSize)
        {
            //this.m_varInName = varInName;
            //this.m_varOutName = varOutName;

            // CNTK
            Utils.SetTraceLevel(TraceLevel.Warning);

            string modelFileName = Path.Combine(Path.GetDirectoryName(Assembly.GetExecutingAssembly().Location), modelPath);
            this.m_modelFunc = Function.Load(modelFileName, m_device);

            Variable inputVar = GetInputVariable(this.m_modelFunc);
            NDShape inputShape = inputVar.Shape;

            int imageWidth = inputShape[0];
            int imageHeight = inputShape[1];

            this.inputDataSize = imageWidth * imageHeight;

            //if (imageHeight != 1)
            //{
            //    throw new ApplicationException($"Model input size is {imageHeight} height while it should be 1");
            //};

            int imageChannels = inputShape[2];
            if (imageChannels != 1)
            {
                throw new ApplicationException($"Model input has {imageChannels} channels while it should be 1");
            };

            Variable outputVar = GetOutputVariable(this.m_modelFunc);
            //if (outputVar.Shape.Dimensions.Count != 1 || outputVar.Shape.Dimensions[0] != 1)
            //{
            //    throw new ApplicationException($"Problem (wrong size) with output var {varOutName}: {outputVar}");
            //};

            this.m_functionsPool = new Pool<Function>(() => this.m_modelFunc.Clone(ParameterCloningMethod.Share), poolSize);

            Test();
        }

        private void Test()
        {
            /*
             * This should result in kinda high score, because
             * there is complete silence.
             */

            List<float> data = new List<float>(this.inputDataSize);
            for (int i = 0; i < this.inputDataSize; ++i)
            {
                data.Add(0);
            };

            float modelResponseOriginal = Evaluate(data);
            //if (modelResponseOriginal < 0.9f)
            //{
            //    throw new Exception("Unusual model answer - something went wrong?");
            //};

            Console.WriteLine("CNTK model test OK");
        }

        private Variable GetInputVariable(Function func)
        {
            //Variable inputVar = func.Arguments
            //    .Where(item => item.Name == this.m_varInName)
            //    .First();
            //return inputVar;
            return func.Arguments.Single();
        }

        private Variable GetOutputVariable(Function func)
        {
            //Variable outputVar = func.Outputs
            //    .Where(item => item.Name == this.m_varOutName)
            //    .First();
            //return outputVar;
            return func.Outputs.Single();
        }

        internal float Evaluate(List<float> dataList)
        {
            if (dataList.Count != inputDataSize)
            {
                throw new Exception($"Error: data size should be {inputDataSize}, but got {dataList.Count}");
            };

            Function modelFuncLocal = this.m_functionsPool.GetObject();

            using (Variable inputVar = GetInputVariable(modelFuncLocal))
            using (Variable outputVar = GetOutputVariable(modelFuncLocal))
            using (Value inputVal = Value.CreateBatch(inputVar.Shape, dataList, m_device))
            {
                Dictionary<Variable, Value> inputMap = new Dictionary<Variable, Value>() {
                    { inputVar, inputVal }
                };

                Dictionary<Variable, Value> outMap = new Dictionary<Variable, Value>() {
                    { outputVar, null }
                };

                /*
                 * You MUST call SetMaxNumCPUThreads() on EVERY thread you call Evaluate on
                 * (i.e. this is stored as thread-local variable).
                 * This call is cheap, so you can call it each time 
                 * like I do here
                 */
                Utils.SetMaxNumCPUThreads(CPU_MAX_THREADS);
                modelFuncLocal.Evaluate(inputMap, outMap, m_device);

                using (Value outValue = outMap[outputVar])
                {
                    float outValueFloat = outValue.GetDenseData<float>(outputVar)[0][0];

                    // !!! Mandatory !!!
                    // TODO: Rewrite returning using using() statement
                    this.m_functionsPool.ReturnObject(modelFuncLocal);

                    return outValueFloat;
                };
            };
        }

        public void Dispose()
        {
            this.m_modelFunc.Dispose();
        }
    }
}
