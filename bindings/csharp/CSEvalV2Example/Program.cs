using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CSEvalV2Example
{
    public class Program
    {
        static void Main(string[] args)
        {
            var myFunc = global::Function.LoadModel("01_OneHidden");

            var uid = myFunc.Uid();
            System.Console.WriteLine("Function id:" + (string.IsNullOrEmpty(uid) ? "(empty)" : uid));

            var name = myFunc.Name();

            System.Console.WriteLine("Function Name:" + (string.IsNullOrEmpty(name) ? "(empty)" : name));

            System.Console.WriteLine("Output Variables:");
            var outputList = myFunc.Outputs().ToList();
            foreach (var output in outputList)
            {
                var type = output.GetDataType();
                System.Console.WriteLine("  Id: " + output.Uid() + " Name: " + output.Name() + " Type: " +
                    (type == DataType.Float ? "Float" : (type == DataType.Double) ? "Double" : "Unknown"));
            }
        }
    }
}
