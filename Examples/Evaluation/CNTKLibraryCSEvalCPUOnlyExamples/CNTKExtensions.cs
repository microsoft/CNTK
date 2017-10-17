using System.Collections.Generic;
using System.Threading.Tasks;
using CNTK;

namespace CNTKExtension
{
    static class CNTKExtension
    {
        /// <summary>
        /// Launches a task that performs evaluation on the computation graph defined by 'function', using provided 'input'
        /// and stores the results in the 'outputs' map.
        /// It is implemented as an extension method of the class Function.
        /// </summary>
        /// <param name="function"> The function representing the computation graph on which the evaluation is executed.</param>
        /// <param name="inputs"> The map represents input variables and their values.</param>
        /// <param name="outputs"> The map defines output variables. On return, the results are stored in Values of the map.</param>
        /// <param name="computeDevice">T he device on which the computation is executed.</param>
        /// <returns> The task representing the asynchronous operation for the evaluation.</returns>
        public static Task EvaluateAsync(this Function function, IDictionary<Variable, Value> inputs, IDictionary<Variable, Value> outputs, DeviceDescriptor computeDevice)
        {
            return Task.Run(() => function.Evaluate(inputs, outputs, computeDevice));
        }
    }
}