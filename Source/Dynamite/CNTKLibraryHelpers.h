//
// Copyright (c) Microsoft. All rights reserved.
// Licensed under the MIT license. See LICENSE.md file in the project root for full license information.
//

// experimental/prototypical layers lib in C++

#define _CRT_SECURE_NO_WARNINGS // "secure" CRT not available on all platforms  --add this at the top of all CPP files that give "function or variable may be unsafe" warnings

#include "CNTKLibrary.h"

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

    // subroutine of GetSubBatches()
    // Takes a batch containing the data for 'numBuckets' and sorts and distributes them over the buckets.
    // args[1][1][numSequences][numStreams] -> args[numBuckets][1][numSequences][numStreams]
    static inline void GetSubBatches_CreateMinibatches(vector<vector<vector<vector<Variable>>>>& args,
                                                       vector<vector<Variable>>& streams, // [streamIndex][seqIndex] not const since we modify it in-place for efficiency
                                                       size_t minibatchSize, size_t numBuckets)
    {
        // make space for result
        // We try to not reallocate the existing array, since this is a lot of vectors of vectors of vectors.
        args.resize(numBuckets); // [numBuckets][numPartial][numStreams][numSeq] (will never destruct unless #buckets changes)
        args.front().resize(1); // create space for at least one partial minibatch; but avoid shrinking, to avoid deallocating the vectors

        // gather its statistics
        let& labelSequences = streams.back(); // last stream has the labels
        let totalNumSequences = streams[0].size(); // (get actual # sequences in the multi-minibatch from first stream)
        size_t totalNumLabels = 0;
        for (size_t seqIndex = 0; seqIndex < totalNumSequences; seqIndex++)
            totalNumLabels += labelSequences[seqIndex].size();

        // sort by source length, longest first
        vector<size_t> indices(totalNumSequences); // index array which we sort
        for (size_t seqIndex = 0; seqIndex < totalNumSequences; seqIndex++)
            indices[seqIndex] = seqIndex;
        let numStreams = streams.size();
        sort(indices.begin(), indices.end(), [&](size_t i, size_t j) // longest first, first stream has highest priority
        {
            // TODO: It may be better to sort by max len, since longest may influence memory usage and computation efficiency the most
            for (size_t streamIndex = 0; streamIndex < numStreams; streamIndex++)
            {
                let& streamSequences = streams[streamIndex];
                let diff = (int)streamSequences[i].size() - (int)streamSequences[j].size();
                if (diff != 0) // if equal then decide by next stream
                    return diff > 0;
            }
            return false;
        });
        vector<Variable> sequencesTemp(totalNumSequences);
        for (size_t streamIndex = 0; streamIndex < numStreams; streamIndex++) // update the actual arrays
        {
            auto& sequences = streams[streamIndex];
            for (size_t seqIndex = 0; seqIndex < totalNumSequences; seqIndex++)
                sequencesTemp[seqIndex] = move(sequences[indices[seqIndex]]);
            for (size_t seqIndex = 0; seqIndex < totalNumSequences; seqIndex++)
                sequences[seqIndex] = move(sequencesTemp[seqIndex]); // this way we save a malloc; probably makes little sense
        }

        // chunk them
        vector<pair<size_t, size_t>> ranges; // [subMinibatch] -> (beginIndex, endIndex) for sub-chunk
        size_t end = 0; // running index into sequences
        size_t numLabelsConsumed = 0;
        for (size_t bucketIndex = 0; bucketIndex < numBuckets; bucketIndex++)
        {
            let labelsLeft = totalNumLabels - numLabelsConsumed;
            let numBucketsLeft = numBuckets - bucketIndex;
            let desiredLabels = (labelsLeft * 2 + numBucketsLeft) / numBucketsLeft / 2;
            // find the end where we consume no more than desiredLabels
            let begin = end;
            size_t thisLabelsConsumed = 0;
            for (; end < totalNumSequences; end++)
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
        if (end != totalNumSequences || numLabelsConsumed != totalNumLabels)
            LogicError("GetSubBatched: somehow didn't use all sequences or labels??");

        // implant the data into args[][][][]
        for (size_t bucketIndex = 0; bucketIndex < numBuckets; bucketIndex++)
        {
            args[bucketIndex].resize(1);
            auto& minibatch = args[bucketIndex][0]; // minibatch for bucket [bucketIndex]
            minibatch.resize(numStreams); // (in case not yet done)
            for (size_t streamIndex = 0; streamIndex < numStreams; streamIndex++)
            {
                let& sequences = streams[streamIndex]; // [numSeq]
                auto& minibatchStream = minibatch[streamIndex]; // [numSeq] to be created here
                let& range = ranges[bucketIndex]; // index range for this bucket
                let numSubSequences = range.second - range.first;
                // TODO: can we use assign() with Transform() here?
                minibatchStream.resize(numSubSequences);
                for (size_t i = 0; i < numSubSequences; i++)
                    minibatchStream[i] = move(sequences[i + range.first]);
            }
        }
    }

    // break a minibatch into partial minibatches, each filling maxBatchSizePerWorker tokens
    // partialMinibatches[1][numStreams][numSeq] -> partialMinibatches[numPartial][numStreams][numSeq]
    // Each partial has at least 'numWorkers' entries, to avoid empty partial sub-minibatches.
    static inline void GetSubBatches_CreatePartialMinibatches(vector<vector<vector<Variable>>>& partialMinibatches, // [numPartial][numStreams][numSeq]
                                                              size_t maxBatchSizePerWorker, bool hasPadding, size_t numWorkers)
    {
        auto minibatch = move(partialMinibatches[0]); // [streamIndex][seqIndex] pull out the data
        let numStreams = minibatch.size();
        let numSeq = minibatch[0].size();

        vector<pair<size_t, size_t>> ranges; // sequence index ranges for partial minibatches
        size_t end = 0; // running index into sequences
        vector<size_t> currentTokensPerWorker, maxLenPerWorker;
        while (end < numSeq)
        {
            // find the end where we consume no more than desired
            currentTokensPerWorker.assign(numWorkers, 0);
            maxLenPerWorker.assign(numWorkers, 0);
            // in case of numWorkers > 1, we rnforce that begin is a multiple of numWorkers
            let begin = end;
            if (begin % numWorkers != 0)
                LogicError("begin not a multiple of numWorkers?");
            bool hitMax = false; // gets set once a worker's partition exceeds the max
            for (; end < numSeq; end++) // note: code not nice. Single loop over sequences and worker partitions.
            {
                size_t rank = end % numWorkers;
                auto& currentTokens = currentTokensPerWorker[rank];
                auto& maxLen = maxLenPerWorker[rank];
                size_t newTokens; // number of tokens in this chunk incorporating this sequence
                if (hasPadding)
                {
                    for (let& streamData : minibatch)
                        if (maxLen < streamData[end].size()) // keep track of max. We use max over all streams, to be conservative.
                            maxLen = streamData[end].size();
                    newTokens = maxLen * ((end - begin) / numWorkers + 1);
                }
                else
                    newTokens = currentTokens + minibatch.back()[end].size();
                hitMax |= newTokens > maxBatchSizePerWorker;
                if (end - begin >= numWorkers && hitMax)
                    break;
                currentTokens = newTokens;
            }
            if (end < numSeq)
                end = end / numWorkers * numWorkers; // make it a multiple of this, for splitting across workers
            if (begin == end)
                LogicError("GetSubBatches_CreatePartialMinibatches: empty range??");
            //fprintf(stderr, "range[%d] = [%d,%d)\n", (int)ranges.size(), (int)begin, (int)end), fflush(stderr);
            ranges.push_back(make_pair(begin, end));
            //fprintf(stderr, "[%d:%d, %d -> %d]\n", (int)begin, (int)end, (int)maxLen, (int)currentTokens);
        }

        // enforce numWorkers requirement on last range
        if (ranges.size() == 0)
            LogicError("GetSubBatches_CreatePartialMinibatches: ranges[] empty??");
        if (ranges.back().second - ranges.back().first < numWorkers) // last range too small: merge into previous
        {
            if (ranges.back().second != numSeq)
                LogicError("GetSubBatches_CreatePartialMinibatches: ranges.back() not covering last sequence??");
            ranges.pop_back();
            ranges.back().second = numSeq;
        }

        // if nothing to break then avoid any reallocation
        if (ranges.size() == 1)
        {
            partialMinibatches[0] = move(minibatch); // put it right back as it was
            return;
        }

        // split the data
        partialMinibatches.resize(ranges.size());
        for (size_t partialMinibatchIndex = 0; partialMinibatchIndex < ranges.size(); partialMinibatchIndex++)
        {
            let& range = ranges[partialMinibatchIndex]; // index range for this bucket
            auto& partialMinibatch = partialMinibatches[partialMinibatchIndex]; // data goes here
            partialMinibatch.resize(numStreams);
            for (size_t streamIndex = 0; streamIndex < numStreams; streamIndex++)
            {
                let& sequences = minibatch[streamIndex]; // [numSeq]
                auto& partialSubSequences = partialMinibatch[streamIndex]; // [numSeq] to be created here
                let numSubSequences = range.second - range.first;
                // TODO: can we use assign() with Transform() here?
                partialSubSequences.resize(numSubSequences);
                for (size_t i = 0; i < numSubSequences; i++)
                    partialSubSequences[i] = move(sequences[i + range.first]);
            }
        }
    }

    // sub-sample a set of sequences for a specific one of several workers
    // sequences[numSeq] --> sequences[numSeq / numWorkers]
    static inline void GetSubBatches_StridedSubSample(vector<Variable>& sequences, size_t numWorkers, size_t thisWorker)
    {
        if (numWorkers == 1)
            return;
        let numSeq = sequences.size();
        size_t j = 0;
        for (size_t i = thisWorker; i < numSeq; i += numWorkers)
        {
            if (j != i)
                sequences[j] = move(sequences[i]);
            j++;
        }
        if (j == 0)
            LogicError("GetSubBatches_StridedSubSample: strided sub sample is empty, despite all the efforts??");
        sequences.resize(j); // rest gets dropped
    }

    // helper to get a set of minibatches at once so that we can sort and group them for better batching efficiency
    //  - sorting helps auto-batching by reducing the need to rebatch
    //  - grouping into sequences of similar length helps batching parallelism (fewer low-parallelism tails)
    //  - we also consider the effect of padding as needed for Marian
    // We sort by sequence length, longest first, first stream (source) having highest priority.
    // The minibatches get random-shuffled, so that we get a random MB sequence w.r.t. length.
    // The amount of data to load is specified by maxibatchSize.
    // Returns true unless the end of the data has been reached.
    static inline bool GetSubBatches(vector<vector<vector<vector<Variable>>>>& args, // [minibatchIndex, partialIndex, streamIndex, sequenceIndex]
                                     const vector<const wchar_t*>& streamNames, const MinibatchSourcePtr& minibatchSource, 
                                     size_t minibatchSize, size_t maxibatchSize, size_t maxBatchSizePerWorker, bool hasPadding,
                                     size_t numWorkers, size_t thisWorker,
                                     size_t shuffleSeed, bool inferenceOnly,
                                     DataType dataType, const DeviceDescriptor& device)
    {
        bool splitDataOverWorkersOurselves = true; // true: do it ourselves, don't leave it to the reader
        // ask for a multi-batch, by asking CNTK for a 'numBuckets' larger minibatch
        auto multiMinibatchData = minibatchSource->GetNextMinibatch(/*minibatchSizeInSequences=*/ (size_t)0, maxibatchSize,
                            splitDataOverWorkersOurselves ? 1 : numWorkers,
                            splitDataOverWorkersOurselves ? 0 : thisWorker,
                            device);

        // convert it to an array of tensors, one for each sequence and stream. First into args[0][0]; later below we will then split it.
        vector<ValuePtr> valuePtrs(Transform(streamNames, [&](const wchar_t* streamName) { return multiMinibatchData[minibatchSource->StreamInfo(streamName)].data; }));
        vector<vector<Variable>> multiMinibatchStreams; // [streamIndex][seqIndex]
        Dynamite::FromCNTKMB(multiMinibatchStreams,     // result goes here
                             valuePtrs,                 // Value objects from MinibatchSource, one for each stream
                             /*isSequence[]=*/vector<bool>(streamNames.size(), true), inferenceOnly, dataType, device);
        if (multiMinibatchStreams.empty())
            return false;

        // break into minibatches of similar size
        let numBuckets = max((maxibatchSize + minibatchSize - 1) / minibatchSize, (size_t)1);
        GetSubBatches_CreateMinibatches(args, multiMinibatchStreams, minibatchSize, numBuckets);

        // random-shuffle
        // TODO: Use a local RNG, don't change global state.
        // Note: This must be consistent across multiple workers; they must use the same shuffleSeed.
        srand((unsigned int)shuffleSeed);
        random_shuffle(args.begin(), args.end());

        // in case a minibatch is too large, break it into sub-minibatches
        for (auto& partialBatchSet : args)
            GetSubBatches_CreatePartialMinibatches(partialBatchSet,
                                    maxBatchSizePerWorker,
                                    hasPadding, /*granularity=*/splitDataOverWorkersOurselves ? numWorkers : 1);

        // if we split data ourselves, this is the point
        if (splitDataOverWorkersOurselves)
            for (auto& partialBatchSet : args)
                for (auto& partialBatch : partialBatchSet)
                    for (auto& streamData : partialBatch)
                        GetSubBatches_StridedSubSample(streamData, numWorkers, thisWorker);

        return true; // true means success, we got data. False means end of data.
    }

}; // namespace
