//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include <functional>
#include "Common.h"
#include "Layers.h"

#include <iostream>
#include <cstdio>

#define let const auto

using namespace CNTK;

using namespace std;

namespace Dynamite {

template<class Base>
class ModelT : public Base
{
public:
    ModelT(const Base& f) : Base(f){}
    // need to think a bit how to store nested NnaryModels
};
typedef ModelT<function<Variable(Variable)>> UnaryModel;
typedef ModelT<function<Variable(Variable,Variable)>> BinaryModel;

UnaryModel Embedding(size_t embeddingDim, const DeviceDescriptor& device)
{
    auto E = Parameter({ embeddingDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device);
    return UnaryModel([=](Variable x)
    {
        return Times(E, x);
    });
}

BinaryModel RNNStep(size_t outputDim, const DeviceDescriptor& device)
{
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device);
    auto R = Parameter({ outputDim, outputDim                  }, DataType::Float, GlorotUniformInitializer(), device);
    auto b = Parameter({ outputDim }, 0.0f, device);
    return [=](Variable prevOutput, Variable input)
    {
        return ReLU(Times(W, input) + Times(R, prevOutput) + b);
    };
}

UnaryModel Linear(size_t outputDim, const DeviceDescriptor& device)
{
    auto W = Parameter({ outputDim, NDShape::InferredDimension }, DataType::Float, GlorotUniformInitializer(), device);
    auto b = Parameter({ outputDim }, 0.0f, device);
    return [=](Variable x) { return Times(W, x) + b; };
}

UnaryModel Sequential(const vector<UnaryModel>& fns)
{
    return [=](Variable x)
    {
        auto arg = Combine({ x });
        for (const auto& f : fns)
            arg = f(arg);
        return arg;
    };
}

struct Batch
{
    static function<vector<Variable>(vector<Variable>)> Map(UnaryModel f)
    {
        return [=](vector<Variable> batch)
        {
            vector<Variable> res;
            res.reserve(batch.size());
            for (const auto& x : batch)
                res.push_back(f(x));
            return res;
        };
    }
};

struct Sequence
{
    const static function<Variable(Variable)> Last;

    static UnaryModel Recurrence(const BinaryModel& stepFunction)
    {
        return [=](Variable x)
        {
            auto dh = PlaceholderVariable();
            auto rec = stepFunction(PastValue(dh), x);
            FunctionPtr(rec)->ReplacePlaceholders({ { dh, rec } });
            return rec;
        };
    }

    static UnaryModel Fold(const BinaryModel& stepFunction)
    {
        auto recurrence = Recurrence(stepFunction);
        return [=](Variable x)
        {
            return Sequence::Last(recurrence(x));
        };
    }

    static function<vector<Variable>(vector<Variable>)> Map(UnaryModel f)
    {
        return Batch::Map(f);
    }

    static function<vector<Variable>(vector<Variable>)> Embedding(size_t embeddingDim, const DeviceDescriptor& device)
    {
        return Map(Dynamite::Embedding(embeddingDim, device));
    }
};
const /*static*/ function<Variable(Variable)> Sequence::Last = [](Variable x) -> Variable { return CNTK::Sequence::Last(x); };

vector<vector<Variable>> FromCNTKMB(vector<ValuePtr> inputs, vector<Variable> variables, const DeviceDescriptor& device) // variables needed for axis info only
// returns vector[numArgs] OF vector[numBatchItems] OF Constant[seqLen,sampleShape]
{
    size_t numSeq = 0;
    for (size_t i = 0; i < inputs.size(); i++)
    {
        // prepare argument i
        let& input    = inputs[i];
        let& variable = variables[i];

        auto sequences = input->UnpackVariableValue(variable, device); // vector[numBatchItems] of NDArrayViews
        if (numSeq == 0)
            numSeq = sequences.size();
        else if (numSeq != sequences.size())
            CNTK::LogicError("inconsistent MB size");
        auto hasAxis = variable.DynamicAxes().size() > 1;

        vector<Variable> arg(numSeq);   // resulting argument
        for (size_t s = 0; s < numSeq; s++)
        {
            auto data = sequences[s]; // NDArrayView
            // convert sparse if needed
            // ... CONTINUE
            // return in correct shape
            if (!hasAxis)
            {
                // slice off sample axis (the last in C++)
                auto dims = data->Shape().Dimensions();
                assert(dims.back() == 1);
                auto newShape = NDShape(vector<size_t>(dims.begin(), dims.end()-1));
                data = data->AsShape(newShape);
            }
            arg[s] = Constant(data);
        }
    }
}

/*
# convert CNTK reader's minibatch to our internal representation
def from_cntk_mb(inputs: tuple, variables: tuple):
    def convert(self, var): # var is for reference to know the axis
        data = self.data
        # unpack MBLayout
        sequences, _ = data.unpack_variable_value(var, True, data.device)
        # turn into correct NDArrayView types
        has_axis = len(var.dynamic_axes) > 1
        def fix_up(data):
            data.__class__ = cntk.core.NDArrayView # data came in as base type
            shape = data.shape
            # map to dense if sparse for now, since we cannot batch sparse due to lack of CUDA kernel
            if data.is_sparse:
                global cached_eyes
                dim = shape[1] # (BUGBUG: won't work for >1D sparse objects)
                if dim not in cached_eyes:
                    eye_np = np.array(np.eye(dim), np.float32)
                    cached_eyes[dim] = cntk.NDArrayView.from_dense(eye_np)
                eye = cached_eyes[dim]
                data = data @ eye
                assert shape == data.shape
            else: # if dense then we clone it so that we won't hold someone else's reference
                data = data.deep_clone()
                data.__class__ = cntk.core.NDArrayView
            item_shape = shape[1:]  # drop a superfluous length dimension
            if has_axis:
                seq = dynamite.Constant(data) # turn into a tensor object
                #return seq
                # BUGBUG: ^^ this fails in a batched __matmul__ of (13,4,2000)
                #         It batched all 4-word sequences for embeddding. Which is bad, it should batch all words for that.
                #         I.e. to allow this, we must repeatedly batch--newly batched ops go into the list again or something. 
                res = [seq[t] for t in range(shape[0])] # slice it
                return res
            else:
                assert shape[0] == 1
                return dynamite.Constant(data[0])
        return [fix_up(seq) for seq in sequences]
    return tuple(convert(inp, var) for inp, var in zip(inputs, variables))
*/

}; // namespace

using namespace Dynamite;

UnaryModel CreateModelFunction(size_t numOutputClasses, size_t embeddingDim, size_t hiddenDim, const DeviceDescriptor& device)
{
    return Sequential({
        Embedding(embeddingDim, device),
        Dynamite::Sequence::Fold(RNNStep(hiddenDim, device)),
        Linear(numOutputClasses, device)
    });
}

function<pair<Variable,Variable>(Variable, Variable)> CreateCriterionFunction(UnaryModel model)
{
    return [=](Variable features, Variable labels)
    {
        auto z = model(features);

        auto loss   = CNTK::CrossEntropyWithSoftmax(z, labels);
        auto metric = CNTK::ClassificationError    (z, labels);

        return make_pair(loss, metric);
    };
}

void TrainSequenceClassifier(const DeviceDescriptor& device, bool useSparseLabels)
{
    const size_t inputDim         = 2000;
    const size_t embeddingDim     = 50;
    const size_t hiddenDim        = 25;
    const size_t numOutputClasses = 5;

    const wstring trainingCTFPath = L"C:/work/CNTK/Tests/EndToEndTests/Text/SequenceClassification/Data/Train.ctf";

    // model and criterion function
    auto model_fn = CreateModelFunction(numOutputClasses, embeddingDim, hiddenDim, device);
    auto criterion_fn = CreateCriterionFunction(model_fn);

    // data
    const wstring featuresName = L"features";
    const wstring labelsName   = L"labels";

    auto minibatchSource = TextFormatMinibatchSource(trainingCTFPath,
    {
        { featuresName, inputDim,         true,  L"x" },
        { labelsName,   numOutputClasses, false, L"y" }
    }, MinibatchSource::FullDataSweep);

    auto featureStreamInfo = minibatchSource->StreamInfo(featuresName);
    auto labelStreamInfo   = minibatchSource->StreamInfo(labelsName);

    // build the graph
    auto features = InputVariable({ inputDim },         true /*isSparse*/, DataType::Float, featuresName);
    auto labels   = InputVariable({ numOutputClasses }, useSparseLabels,   DataType::Float, labelsName, { Axis::DefaultBatchAxis() });

    auto criterion = criterion_fn(features, labels);
    auto loss   = criterion.first;
    auto metric = criterion.second;

    // train
    auto learner = SGDLearner(FunctionPtr(loss)->Parameters(), LearningRatePerSampleSchedule(0.05));
    auto trainer = CreateTrainer(nullptr, loss, metric, { learner });

    const size_t minibatchSize = 200;
    for (size_t i = 0; true; i++)
    {
        auto minibatchData = minibatchSource->GetNextMinibatch(minibatchSize, device);
        if (minibatchData.empty())
            break;

        // Dynamite
        auto args = FromCNTKMB({ minibatchData[featureStreamInfo].data, minibatchData[labelStreamInfo].data }, FunctionPtr(loss)->Arguments(), device);

        // static CNTK
        trainer->TrainMinibatch({ { features, minibatchData[featureStreamInfo] },{ labels, minibatchData[labelStreamInfo] } }, device);
        PrintTrainingProgress(trainer, i, /*outputFrequencyInMinibatches=*/ 1);
    }
}

int main(int argc, char *argv[])
{
    try
    {
        //TrainSequenceClassifier(DeviceDescriptor::GPUDevice(0), true);
        TrainSequenceClassifier(DeviceDescriptor::CPUDevice(), true);
    }
    catch (exception& e)
    {
        fprintf(stderr, "EXCEPTION caught: %s\n", e.what());
    }
}
