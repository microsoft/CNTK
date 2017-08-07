using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace CNTK
{
    public partial class Trainer
    {
        public static Trainer CreateTrainer(Function model, Function lossFunction, Function evaluationFunction, IList<Learner> parameterLearners, 
            ProgressWriterVector progressWriters = null)
        {
            LearnerVector learnerVector = Helper.AsLearnerVector(parameterLearners);
            if (progressWriters != null)
                return CNTKLib.CreateTrainer(model, lossFunction, evaluationFunction, learnerVector, progressWriters);
            else
                return CNTKLib.CreateTrainer(model, lossFunction, evaluationFunction, learnerVector);
        }

        public bool TrainMinibatch(IDictionary<Variable, MinibatchData> arguments, DeviceDescriptor computeDevice)
        {
            UnorderedMapVariableMinibatchData vectorData = Helper.AsUnorderedMapVariableMinibatchData(arguments);
            return TrainMinibatch(vectorData, computeDevice);
        }
    }
}
