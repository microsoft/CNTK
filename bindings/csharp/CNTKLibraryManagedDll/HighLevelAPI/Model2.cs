using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

// using static CNTK.HighLevelAPI.CntkLib;

namespace CNTK.HighLevelAPI
{
    public class Model2
    {
        public Model2(CNTKDictionary init, Function actication)
        {
            throw new NotImplementedException();
        }

        public Model2(Function inputVariable)
        {
            throw new NotImplementedException();
        }

        public Function Apply(Variable input)
        {
            throw new NotImplementedException();
        }


        public Model2 Convolution2D(NDShape filterShape, int numFilters, NDShape strides, bool pad, string name = "")
        {
            throw new NotImplementedException();
        }

        public Model2 Convolution2D(Variable inputVariable, NDShape filterShape, int numFilters, NDShape strides, bool pad, string name = "")
        {
            throw new NotImplementedException();
        }

        public Model2 Dense(int outputClasses, Function activation, string name = "")
        {
            throw new NotImplementedException();
        }

        public Model2 Dense(Variable inputVariable, int outputClasses, Function activation, string name = "")
        {
            throw new NotImplementedException();
        }

        public Function Function
        {
            get;
            private set;
        }
    }


    public class Example
    { 
    }
}
;