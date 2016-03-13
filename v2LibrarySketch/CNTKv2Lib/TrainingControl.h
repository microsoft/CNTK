#pragma one

#include <unordered_set>
#include "Variable.h"
#include "Value.h"

namespace CNTK
{
    class TrainingControl
    {
    protected:
        TrainingControl(std::unordered_set<Variable> outputSet, std::unordered_set<Variable> gradientSet);

    public:
        virtual void BeforeTrainMinibatch(SGDOptimizer optimizer);

        // By returning false, this function stops the training loop
        virtual bool AfterTrainMinibatch(SGDOptimizer optimizer,
                                         size_t numSamplesInLastMB,
                                         const std::unordered_map<Variable, Value> outputValues,
                                         const std::unordered_map<Variable, Value> gradientValues);

        // Returns the set of output variables, values of which are passed to the AfterTrainMinibatch callback
        std::unordered_set<Variable> OutputSet();

        // Returns the set of variables, gradient values of which are passed to the AfterTrainMinibatch callback
        std::unordered_set<Variable> GradientSet();
    };

    // Builtin training drivers
    TrainingControl BasicTrainingControl(size_t numSamplesToTrainOn, size_t checkpointFrequency, const std::wstring& checkpointFileName);

    // TODO: Additional trainign control objects with facilties for automatic learning rate control, minibatch size control etc.
}
