//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// experimental/prototypical layers lib in C++

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"
#include "Common.h"

#include <functional>
#include <cstdio>
#include <map>
#include <set>
#include <vector>

#define let const auto

using namespace CNTK;
using namespace std;

namespace Dynamite {

    // slice the last dimension if an NDArrayView (index with index i; then drop the axis)
    // This is used for MB conversion.
    static NDArrayViewPtr Index(NDArrayViewPtr data, NDShapeDimension i)
    {
        auto dims = data->Shape().Dimensions();
        auto startOffset = NDShapeDimensions(dims.size(), 0);
        let& extent = dims;
        if (startOffset.back() != i || extent.back() != 1)
        {
            startOffset.back() = i;
            let lesserExtent = NDShapeDimensions(extent.begin(), extent.end() - 1); // missing extend values default to 1 but do not generate an output axis
            //extent.pop_back(); // missing extend values default to 1 but do not generate an output axis
            data = data->SliceView(startOffset, extent, true); // slice it
            dims = data->Shape().Dimensions();
        }
        else
        {
            let newShape = NDShape(vector<size_t>(dims.begin(), dims.end() - 1));
            data = data->AsShape(newShape); // and drop the final dimension
        }
        return data;
    }

    // helper for converting data to dense
    template<typename ElementType>
    static inline NDArrayViewPtr MakeEye(size_t n, const CNTK::DataType& dataType, const CNTK::DeviceDescriptor& device)
    {
        vector<ElementType> buffer(n*n, 0);
        for (size_t i = 0; i < n; i++)
            buffer[i*n + i] = 1;
        auto eye = make_shared<NDArrayView>(dataType, NDShape{ n, n }, buffer.data(), buffer.size() * sizeof(ElementType), DeviceDescriptor::CPUDevice(), /*readOnly=*/false);
        eye = eye->DeepClone(device);
        return eye;
    }
    static inline NDArrayViewPtr Eye(size_t n, const CNTK::DataType& dataType, const CNTK::DeviceDescriptor& device)
    {
        static map<pair<size_t, CNTK::DataType>, NDArrayViewPtr> cached;
        let key = make_pair(n, dataType);
        auto iter = cached.find(key);
        if (iter != cached.end())
            return iter->second;
        // need to create it
        NDArrayViewPtr eye;  device;
        switch (dataType)
        {
        case DataType::Float:  eye = MakeEye<float>(n, dataType, device);  break;
        case DataType::Double: eye = MakeEye<double>(n, dataType, device); break;
        default: throw logic_error("Eye: Unsupported data type.");
        }
        cached[key] = eye;
        return eye;
    }

    // returns vector[numArgs] OF vector[numBatchItems] OF Constant[seqLen,sampleShape]
    // or no seqLen if isSequence is false for the respective stream
    static inline void FromCNTKMB(vector<vector<Variable>>& res, const vector<ValuePtr>& inputs, const vector<bool>& isSequence, bool inferenceOnly, DataType dataType, const DeviceDescriptor& device) // variables needed for axis info only
    {
        let numArgs = inputs.size();
        res.resize(numArgs);
        for (auto& r : res) // free memory if still held from previous call
            r.clear();
        size_t numSeq = 0;
        size_t med = SIZE_MAX; // hack
        for (size_t i = 0; i < numArgs; i++)
        {
            // prepare argument i
            let& input = inputs[i];
            // UnpackVariableValue() requires an InputVariable for reference, so create one
            // CNTK Readers always return data with 2 axes (length, batch), even for data without sequence axis (the readers don't know).
            // Hence, users must pass in whether a stream is meant to have a sequence axis or not.
            let fullShape = input->Shape();
            let sampleShape = fullShape.SubShape(0, fullShape.Rank() - 2); // input.Shape() includes the dynamic axes shape at the end
            let dynamicAxes = isSequence[i] ? Axis::DefaultInputVariableDynamicAxes() : vector<Axis>{ Axis::DefaultBatchAxis() };
            let variable = CNTK::InputVariable(sampleShape, input->IsSparse(), input->GetDataType(), dynamicAxes);
            auto sequences = input->UnpackVariableValue(variable, device); // -> vector[numBatchItems] of NDArrayViews
            if (numSeq == 0)
                numSeq = sequences.size();
            else if (numSeq != sequences.size())
                CNTK::LogicError("FromCNTKMB: Streams must all have the same number of sequences.");
            auto hasAxis = variable.DynamicAxes().size() > 1;
#if 0       // hack to be able to work with totally homogeneous sequence lengths; as the upper bound
            if (med == SIZE_MAX)
            {
                size_t tot = 0;
                for (size_t s = 0; s < numSeq; s++)
                    tot += sequences[s]->Shape().Dimensions().back();
                tot /= sequences.size(); // av len
                med = 0;
                for (size_t s = 0; s < numSeq; s++)
                {
                    if (abs((int)sequences[s]->Shape().Dimensions().back() - (int)tot) < abs((int)sequences[med]->Shape().Dimensions().back() - (int)tot))
                        med = s;
                }
            }
#endif

            auto& arg = res[i];
            arg.resize(numSeq);   // resulting argument
            for (size_t s = 0; s < numSeq; s++)
            {
                //auto data = sequences[s];      // NDArrayView
                auto data = sequences[med != SIZE_MAX ? med : s];      // NDArrayView
#if 0
                // BUGBUG: This crashes MT.cpp in CPU mode with an A/V! Yikes!
                data = data->DeepClone(); // sometimes we get strange objective values; see if this is the cause. This will release the original matrix.
#endif
                //if (!data->IsSparse())
                //    data = data->DeepClone(); // sometimes we get "cannot resize"--just fishing here   --still happens occasionally
                // return in correct shape
                if (!hasAxis)
                {
                    if (data->Shape().Dimensions().back() != 1)
                        CNTK::LogicError("FromCNTKMB: Streams declared as !isSequence must have a trailing dimension of 1.");
                    data = Index(data, 0); // slice off sample axis (the last in C++)
                    // TODO: This is better done as an AsShape(), no? That would avoid updating the Matrix view
                }
#if 0 // needed for now since PlainTextDeserializer cannot deliver Linear data, and Dynamite metric blows up on Sparse
                if (data->IsSparse())
                {
                    // multiply with  an identity matrix
                    auto eye = Eye(data->Shape()[0], data->GetDataType(), data->Device());
                    data = NDArrayView::MatrixProduct(false, eye, false, data, false, 1.0, 1);
                }
#endif
                auto c = Constant(data, /*isVolatile=*/inferenceOnly);
                if (c.GetDataType() != dataType)
                    c = c.CloneAs(dataType); // note: This is expensive and involves a GPU sync; so only do this for debugging (gradient check)
                arg[s] = c;
                //data = data->AsShape(data->Shape()); // (for debugging)
            }
        }
    }

    // helper to get multiple batches at once so that we can sort and group them for better batching efficiency
    //  - sorting helps auto-batching by reducing the need to rebatch
    //  - grouping into sequences of similar length helps batching parallelism (fewer low-parallelism tails)
    // We sort by stream length, longest first, first stream having highest priority.
    // Finally, the sub-minibatches get random-shuffled, so that we get a random MB sequence w.r.t. length.
    // Returns true unless the end of the data has been reached.
    static inline bool GetSubBatches(vector<vector<vector<Variable>>>& args, const vector<const wchar_t*>& streamNames, size_t numSubMinibatches, size_t shuffleSeed,
                                     const MinibatchSourcePtr& minibatchSource, size_t minibatchSize, size_t numWorkers, size_t thisWorker,
                                     bool inferenceOnly, DataType dataType, const DeviceDescriptor& device)
    {
        // get the big minibatch from CNTK
        // We ask for 'numSubMinibatches' larger size than user target.
        auto minibatchData = minibatchSource->GetNextMinibatch(/*minibatchSizeInSequences=*/ (size_t)0, numSubMinibatches * minibatchSize, numWorkers, thisWorker, device);
        // check for sweepEnd
        if (any_of(minibatchData.begin(), minibatchData.end(), [](const decltype(*minibatchData.begin())& kv) { return kv.second.sweepEnd; }))
            return false;
        // ... may work now. If still not then...: BUGBUG: This ^^ does not work, it just keeps going. How to do this right?

        // convert it to an array of tensors. First into args[0]; later below we will then split it.
        let numStreams = streamNames.size();
        args.resize(numSubMinibatches);
        auto& subBatch0 = args[0];
        vector<ValuePtr> valuePtrs;
        for (let& streamName : streamNames)
            valuePtrs.push_back(minibatchData[minibatchSource->StreamInfo(streamName)].data);
        Dynamite::FromCNTKMB(subBatch0, valuePtrs, /*isSequence[]=*/vector<bool>(numStreams, true), inferenceOnly, dataType, device);
#if 1   // for compat with old loss progressions, don't reorder if no sub-minibatching
        if (numSubMinibatches == 1)
            return true;
#endif

        // gather its statistics
        let& labelSequences = subBatch0.back(); // last stream has the labels
        let numSequences = subBatch0[0].size(); // (get actual # sequences in the batch from first stream)
        size_t numLabels = 0;
        for (size_t i = 0; i < numSequences; i++)
            numLabels += labelSequences[i].size();

        // sort by source length, longest first
        vector<size_t> indices(numSequences); // index array which we sort
        for (size_t k = 0; k < numSequences; k++)
            indices[k] = k;
        sort(indices.begin(), indices.end(), [&](size_t i, size_t j) // longest first, first stream has highest priority
        {
            for (size_t k = 0; k < numStreams; k++)
            {
                let& streamSequences = subBatch0[k];
                let diff = (int)streamSequences[i].size() - (int)streamSequences[j].size();
                if (diff != 0) // if equal then decide by next stream
                    return diff > 0;
            }
            return false;
        });
        vector<Variable> sequencesTemp(numSequences);
        for (size_t k = 0; k < numStreams; k++) // update the actual arrays
        {
            auto& sequences = subBatch0[k];
            for (size_t i = 0; i < numSequences; i++)
                sequencesTemp[i] = move(sequences[indices[i]]);
            for (size_t i = 0; i < numSequences; i++)
                sequences[i] = move(sequencesTemp[i]); // this way we save a malloc; probably makes little sense
        }

        // chunk them
        vector<pair<size_t, size_t>> ranges; // [subMinibatch] -> (beginIndex, endIndex) for sub-chunk
        size_t end = 0; // running index into sequences
        size_t numLabelsConsumed = 0;
        for (size_t j = 0; j < numSubMinibatches; j++)
        {
            let labelsLeft = numLabels - numLabelsConsumed;
            let subBatchesLeft = numSubMinibatches - j;
            let desiredLabels = (labelsLeft * 2 + subBatchesLeft) / subBatchesLeft / 2;
            // find the end where we consume no more than desiredLabels
            let begin = end;
            size_t thisLabelsConsumed = 0;
            for (; end < numSequences; end++)
            {
                let len = labelSequences[end].size();
                if (end > begin && thisLabelsConsumed + len > desiredLabels)
                    break;
                thisLabelsConsumed += len;
            }
            //fprintf(stderr, "range[%d] = [%d,%d)\n", (int)ranges.size(), (int)begin, (int)end), fflush(stderr);
            ranges.push_back(make_pair(begin, end));
            numLabelsConsumed += thisLabelsConsumed;
        }
        if (end != numSequences || numLabelsConsumed != numLabels)
            LogicError("GetSubBatched: somehow didn't use all sequences or labels??");

        // create all sub-batches
        for (size_t j = 1; j < numSubMinibatches; j++)
        {
            args[j].resize(numStreams); // (in case not yet done)
            for (size_t k = 0; k < numStreams; k++)
            {
                let& sequences = subBatch0[k];
                auto& subSequences = args[j][k];
                let& range = ranges[j];
                let numSubSequences = range.second - range.first;
                subSequences.resize(numSubSequences);
                for (size_t i = 0; i < numSubSequences; i++)
                    subSequences[i] = move(sequences[i + range.first]);
            }
        }
        // sub-batch 0 is merely a resize
        for (size_t k = 0; k < numStreams; k++)
        {
            auto& sequences = subBatch0[k];
            let& range = ranges[0];
            sequences.resize(range.second);
        }

        // random-shuffle
        // TODO: Use a local RNG, don't change global state.
        // Note: This must be consistent across multiple workers; they must use the same shuffleSeed.
        srand((unsigned int)shuffleSeed);
        random_shuffle(args.begin(), args.end());

        return true; // true means success, we got data. False means end of data.
    }

}; // namespace
