// DataReaderHelper.h -- helper functions that understand both DataReader and ComputationNetwork

#pragma once

#include "Basics.h"
#include "DataReader.h"
#include "ComputationNetwork.h"
#include "MPIWrapper.h"
#include "SpecialPurposeNodes.h"        // for SequenceWithSoftmaxNode
#include <string>
#include <map>
#include <set>

namespace Microsoft { namespace MSR { namespace CNTK {

/*static*/ struct DataReaderHelpers
{
    template <class ElemType>
    static void NotifyChangedNodes(ComputationNetworkPtr net, StreamMinibatchInputs& inputMatrices)
    {
        // reader will have resized input node's m_value directly. Nodes must be notified to do necessary internal state updates from that.
        // TODO: This is a stopgap. SGD will at some point change from sets of matrices to sets of nodes. Then this will become much simpler.
        std::set<MatrixBasePtr> matrices;
        for (const auto& iter : inputMatrices)
            matrices.insert(iter.second.matrix);
        for (auto& node : net->FeatureNodes())
            if (matrices.find(node->As<ComputationNode<ElemType>>()->ValuePtr()) != matrices.end())
                node->NotifyFunctionValuesMBSizeModified();
        for (auto& node : net->LabelNodes())
            if (matrices.find(node->As<ComputationNode<ElemType>>()->ValuePtr()) != matrices.end())
                node->NotifyFunctionValuesMBSizeModified();
    }

    // -------------------------------------------------------------------
    // GetMinibatchIntoNetwork() -- get one minibatch from Reader (this->trainSetDataReader) into Network (this->net)
    // Returns false if no data is read. In that case, no other return value can be expected to contain meaningful values (e.g. actualMBSize will be unchanged).
    // Sets actualMBSize to the number of matrix columns. Note that 0 is a valid value to be returned for actualMBSize, caller must handle that correctly.
    // -------------------------------------------------------------------
    // Note: This will go away with the redesigned reader interface.
    // TODO: callers of this often do ComputationNetwork::BumpEvalTimeStamp(featureNodes) and also for labels; we should eliminate the need for this.
    template <class ElemType>
    static bool GetMinibatchIntoNetwork(IDataReader& trainSetDataReader,
                                        ComputationNetworkPtr net,
                                        ComputationNodeBasePtr criterionNode,
                                        bool useDistributedMBReading,
                                        bool useParallelTrain,
                                        StreamMinibatchInputs& inputMatrices,
                                        size_t& actualMBSize, 
                                        const MPIWrapperPtr& mpi)
    {
        // Reading consists of a sequence of Reader API calls:
        //  - GetMinibatch() --fills the inputMatrices and copies the MBLayout from Reader into inputMatrices
        //  - SetActualMiniBatchSizeFromFeatures()  --tells Network to resize the nodes' buffers
        // with the special twist that in presence of parallelization, there is some decimation involved.

        bool wasDataRead = trainSetDataReader.GetMinibatch(inputMatrices); // fill in the minibatch data into the Input nodes' buffers directly
        // If this returns false, the matrices may contain garbage or not sized to 0 columns.
        // On the other hand, if it returns a 0-column matrix, that would be a perfectly cromulent minibatch (in case of data parallelism with distributed reading).
        // If a passed matrix does not match a reader section, that is an error.

        // if no data read then we are done
        if (!wasDataRead)
            return false;

        // get some additional information when doing sequence training
        // TODO: This should not need to be called in case of wasDataRead == false, since in that case, returned values are invalid.
        if ((criterionNode != nullptr) && (criterionNode->OperationName() == L"SequenceWithSoftmax"))
        {
            auto node = dynamic_pointer_cast<SequenceWithSoftmaxNode<ElemType>>(criterionNode);
            auto latticeinput = node->getLatticePtr();
            auto uids = node->getuidprt();
            auto boundaries = node->getboundaryprt();
            auto extrauttmap = node->getextrauttmap();

            trainSetDataReader.GetMinibatch4SE(*latticeinput, *uids, *boundaries, *extrauttmap);
        }

        if ((criterionNode != nullptr) && (criterionNode->OperationName() == L"CTCwithSoftmax"))
        {
            auto node = dynamic_pointer_cast<CTCwithSoftmaxNode<ElemType>>(criterionNode);

            auto boundaries = node->getboundaryprt();
            auto extrauttmap = node->getextrauttmap();

            trainSetDataReader.GetMinibatch4CTC(*boundaries, *extrauttmap);
        }

        // TODO: move this into shim for the old readers.
        // decimate if needed. Decimation happens in-place.
        // This is only allowed for old readers, which support a single layout for all inputs.
        if (!useDistributedMBReading && useParallelTrain)
        {
            auto& pMBLayout = net->GetMBLayoutPtrOfNetwork();
            
            // Verify that there's indeed a single layout
            for (const auto& iter : inputMatrices)
            {
                assert(iter.second.pMBLayout == pMBLayout);
                // TODO: This must be a runtime check, not an assert().
                UNUSED(iter);
            }

            assert(trainSetDataReader.IsLegacyReader());
            DecimateMinibatchInPlace<ElemType>(inputMatrices, mpi->NumNodesInUse(), mpi->CurrentNodeRank(), pMBLayout);
        }

#if 0   // merge leftover?
        // This will automatically discard a large fraction of the data, useful if the training data is known to be highly correlated
        if (dataDecimationFactor)
        {
            auto& pMBLayout = net->GetMBLayoutPtrOfNetwork();

            // Verify that there's indeed a single layout
            for (const auto& iter : inputMatrices)
            {
                assert(iter.second.pMBLayout == pMBLayout);
                // TODO: This must be a runtime check, not an assert().
                UNUSED(iter);
            }

            DecimateMinibatchInPlace<ElemType>(inputMatrices, dataDecimationFactor, 0, pMBLayout);
        }
#endif

        NotifyChangedNodes<ElemType>(net, inputMatrices);

        // get MB size and tell Network to update its nodes' buffers based on what's in the input matrices
        // Note: Decimation may have reduced this to 0 frames. We still must return 'true'.
        // BUGBUG: This has a definitional problem once we support multiple feature streams with different lenghts.
        // BUGBUG: We should discount gaps.
        actualMBSize = net->DetermineActualMBSizeFromFeatures();

        return true;
    }

    // get StreamMinibatchInputs for a given set of input nodes
    static StreamMinibatchInputs RetrieveInputMatrices(const std::vector<ComputationNodeBasePtr>& inputNodes)
    {
        StreamMinibatchInputs inputMatrices;
        for (auto& node : inputNodes)
            inputMatrices.AddInput(node->NodeName(), node->ValuePtr(), node->GetMBLayout(), node->GetSampleLayout());
        return inputMatrices;
    }


    // -------------------------------------------------------------------
    // DecimateMinibatch - decimate minibatch for parallelization
    // -------------------------------------------------------------------
    // non-inplace decimation , to be used in subminibatch implementation
    // returns a subset of parallel sequences
    template <class ElemType>
    static pair<size_t, size_t> DecimateMinibatch(const StreamMinibatchInputs& MB,    // input matrices
                                                  StreamMinibatchInputs& decimatedMB, // output decimated matrices.
                                                  MBLayoutPtr pMBLayout,              // input MBLayout
                                                  MBLayoutPtr& pDecimateMBLayout,     // output decimated MBLayout (note: cannot work in-place)
                                                  size_t numProcs, size_t rank)
    {
        size_t numParallelSequences = pMBLayout->GetNumParallelSequences();
        size_t nT = pMBLayout->GetNumTimeSteps();

        // decide start column and end column
        size_t st = numParallelSequences *  rank      / numProcs;
        size_t en = numParallelSequences * (rank + 1) / numProcs;
        assert(rank < numProcs);
        en = en > numParallelSequences ? numParallelSequences : en; // TODO: why are these two tests necessary? We should rather test rank
        en = (rank + 1 == numProcs) ? numParallelSequences : en;
        size_t numNewParallelSequence = en - st;

        // begin decimating matrices
        size_t rv = 0;
        for (const auto& it : MB)
        {
            const wstring& name = it.first;
            const auto& input = it.second;
            auto& mat = MB.GetInputMatrix<ElemType>(name);
            size_t numRows = mat.GetNumRows();
            size_t numCols = mat.GetNumCols();
            int deviceId   = mat.GetDeviceId();

            if (rv == 0)
                rv = numCols;
            else if (rv != numCols)
                LogicError("DecimateMinibatch: Inconsistent number of columns among inputs (found %d and %d).", (int) rv, (int) numCols);

            if (nT != numCols / numParallelSequences)
                LogicError("ERROR: MBLayout borked, GetNumTimeSteps() mismatches minibatch number of columns\n");

            auto matrixp = make_shared<Matrix<ElemType>>(deviceId);
            matrixp->AssignRowSliceValuesOf(mat.Reshaped(numRows * numParallelSequences, nT), st * numRows, (en - st) * numRows);
            matrixp->Reshape(numRows, numNewParallelSequence * nT);
            decimatedMB.AddInput(name, matrixp, input.pMBLayout, input.sampleLayout);
            // If we had a RowSlice function, we would like to write in this way
            // decimatedMB[name]->SetValue(mat.Reshaped(nRows*nSequence, nT).RowSlice( st*nRows , (en-st)*nRows).Reshaped(nRows, nNewParallelSequence*nT));
        }
        // decimate MBLayout as well
        pDecimateMBLayout = make_shared<MBLayout>(numNewParallelSequence, nT, L"");
        pDecimateMBLayout->SetAxisName(pMBLayout->GetAxisName());
#if 1
        // now copy over all sequence info records that are inside the range, with adjusted 's'
        const auto& sequences = pMBLayout->GetAllSequences();
        for (const auto& seq : sequences)
        {
            if (seq.s >= st && seq.s < en)
            {
                auto shiftedSeq = seq;
                shiftedSeq.s -= st; // these sequences have shifted up by 'st' sequences
                pDecimateMBLayout->AddSequence(shiftedSeq);
            }
        }
#else
        for (size_t t = 0; t < nT; t++)
            for (size_t id = 0; id < numNewParallelSequence; id++)
                pDecimateMBLayout->Set(id, t, pMBLayout->Get(id + st, t));
#endif

        return pair<size_t, size_t>(st, en);
    }

    // in-place decimation, for use with data-parallel processing
    // returns a subset of parallel sequences
    template <class ElemType>
    static pair<size_t, size_t> DecimateMinibatchInPlace(StreamMinibatchInputs& mb,    // matrix to be decimated
                                                         size_t numprocs, size_t rank, // rank info
                                                         MBLayoutPtr pMBLayout)        // get decimated as well
    {
        if (numprocs == 1)
            return pair<size_t, size_t>(0, pMBLayout->GetNumParallelSequences());
        // no need to do inplace decimation if numproc == 1

        // allocate space for non-inplace decimation
        MBLayoutPtr pDecimatedMBLayout = make_shared<MBLayout>();
        pDecimatedMBLayout->SetAxisName(pMBLayout->GetAxisName());
        StreamMinibatchInputs decimatedMB;
        // call in-place decimation
        pair<size_t, size_t> selected = DecimateMinibatch<ElemType>(mb, decimatedMB, pMBLayout, pDecimatedMBLayout, numprocs, rank);
        // move the data
        for (auto k : mb)
        {
            const auto& name = k.first;
            mb.GetInputMatrix<ElemType>(name).SetValue(decimatedMB.GetInputMatrix<ElemType>(name)); // deep-copy our local one to the output location
        }
        pMBLayout->MoveFrom(pDecimatedMBLayout);
        return selected;
    }

    template<class ElemType>
    static size_t GetNumSubminibatchesNeeded(IDataReader* dataReader,
                                           size_t maxSamplesInRAM,
                                           size_t numSubminibatches,
                                           size_t tunedMBSize)
    {
        if (numSubminibatches > 1) // user-specified maximum number of samples
            return numSubminibatches;

        if (maxSamplesInRAM < SIZE_MAX)
        {
            // into how many pieces would we need to break the minibatch?
            // TODO: The following calculation relies on the ill-devised definition of "minibatch" of the current truncated BPTT implementation. Adapt this once fixed.
            size_t numParallelSequences = dataReader->GetNumParallelSequencesForFixingBPTTMode();
            size_t estimatedMBSize = tunedMBSize * numParallelSequences;
            return (estimatedMBSize + maxSamplesInRAM - 1) / maxSamplesInRAM;
        }

        // The return value of this method decides how many subminibatch needed for the training or 
        // eval process. The current process only starts the subminibatch loop when the calculated
        // subminibatch number is larger than 1. So here returning 0 or 1 shares the same behavior.
        // But the default value should still be 0 which means no subminibatch needed for this case.
        return 0;
    }

    // ===================================================================
    // SubminibatchHelpers -- helper for sub-minibatch implementation
    // TODO: Can this just exist inside SGD.cpp?
    // ===================================================================

    // A sub-minibathc is a part of a minibatch which helps computing large minibatches that cannot load into GPU memory in one forward-backward computation
    // The usage would be :
    //        SubminibatchHelpers sbhelper;
    //        for (;;)
    //        {
    //            size_t nsb=sb.GetMinibatchIntoCache(...);
    //            for (size_t i=0; i<nsb; i++)
    //            {
    //                sbhelper.GetSubMinibatchToNet(i);
    //                net.Evaluate(criterionNodes[0]);
    //                sbhelper.DoneWithCurrentSubMinibatch();
    //            }
    //            UpdateWeights(...);
    //        }

    template <class ElemType>
    class SubminibatchDispatcher
    {
    private:
        typedef std::vector<shared_ptr<const msra::dbn::latticesource::latticepair>> Lattice;
        typedef std::vector<size_t> Uid;
        typedef std::vector<size_t> ExtrauttMap;
        typedef std::vector<size_t> Boundaries;

        typedef std::vector<shared_ptr<const msra::dbn::latticesource::latticepair>>* LatticePtr;
        typedef std::vector<size_t>* UidPtr;
        typedef std::vector<size_t>* ExtrauttMapPtr;
        typedef std::vector<size_t>* BoundariesPtr;
        typedef StreamMinibatchInputs Matrices;

        // member variables served as caching space
        Matrices m_inputMatricesCache;
        MBLayoutPtr m_MBLayoutCache;
        Lattice m_LatticeCache;
        Uid m_uidCache;
        ExtrauttMap m_extrauttmapCache;
        Boundaries m_BoundariesCache;
        shared_ptr<Matrix<ElemType>> m_netCriterionAccumulator;
        shared_ptr<Matrix<ElemType>> m_netEvaluationAccumulator;
        std::map<wstring, vector<shared_ptr<INodeState>>> m_netStates; // m_netStatefulNodes[node][i] caches the state of i-th subminibatch of node
        bool m_hasLattices;

        Matrices m_cachedGradient;
        // we also need to remember where to put into the net
        MBLayoutPtr m_netMBLayoutPtr;
        std::map<wstring, shared_ptr<ComputationNode<ElemType>>> m_LearnableNodePtr;
        // followings are lattice-related
        Matrices m_netInputMatrixPtr;
        LatticePtr m_netLatticePtr;
        UidPtr m_netUidPtr;
        ExtrauttMapPtr m_netExtrauttMapPtr;
        BoundariesPtr m_netBoundariesPtr;
        // we remember the pointer to the learnable Nodes so that we can accumulate the gradient once a sub-minibatch is done

        size_t m_numParallelSequences; // number of paralle sequence in the cached matrix and MBLayout
        size_t m_numSubminibatches;    // how many subminibatches we are going to use ?

        std::vector<shared_ptr<ComputationNode<ElemType>>> m_netCriterionNodes;
        std::vector<shared_ptr<ComputationNode<ElemType>>> m_netEvaluationNodes;
        std::map<wstring, shared_ptr<IStatefulNode>> m_netStatefulNodes; // we need to Export/Import states of stateful nodes when we swtich subminibatches

    private:
        void EnumerateStatefulNodesForRoot(ComputationNetwork& net, ComputationNodeBasePtr root, std::map<wstring, shared_ptr<IStatefulNode>>& statefulNodes)
        {
            for (const auto& node : net.GetAllNodesForRoot(root))
            {
                const auto& name = node->GetName();
                if (statefulNodes.find(name) != statefulNodes.end())
                    continue; // already in the list  --TODO: use insert()
                shared_ptr<IStatefulNode> pNode = dynamic_pointer_cast<IStatefulNode>(node);
                if (pNode) // if it is an IStatefulNode then report it
                    statefulNodes[name] = pNode;
            }
        }

        std::map<wstring, shared_ptr<IStatefulNode>> EnumerateStatefulNode(ComputationNetwork& net,
                                                                           const std::vector<ComputationNodeBasePtr>& criterionNode,
                                                                           const std::vector<ComputationNodeBasePtr>& evaluationNode)
        {
            std::map<wstring, shared_ptr<IStatefulNode>> statefulNodes;
            for (auto& root : criterionNode)
                EnumerateStatefulNodesForRoot(net, root, statefulNodes);
            for (auto& root : evaluationNode)
                EnumerateStatefulNodesForRoot(net, root, statefulNodes);
            return statefulNodes;
        }

    public:
        SubminibatchDispatcher()
            : m_MBLayoutCache(nullptr), m_netLatticePtr(nullptr), m_netExtrauttMapPtr(nullptr), m_netUidPtr(nullptr), m_netBoundariesPtr(nullptr)
        {
        }

        void Init(ComputationNetworkPtr& net,
                  const std::list<ComputationNodeBasePtr>& learnableNodes,
                  const std::vector<ComputationNodeBasePtr>& criterionNodes,
                  const std::vector<ComputationNodeBasePtr>& evaluationNodes)
        {
            m_MBLayoutCache = make_shared<MBLayout>();
            m_netCriterionAccumulator = make_shared<Matrix<ElemType>>(1, 1, net->GetDeviceId());
            m_netEvaluationAccumulator = make_shared<Matrix<ElemType>>(1, evaluationNodes.size(), net->GetDeviceId());
            // remember ptrs to learnable nodes
            for (auto x : learnableNodes)
            {
                shared_ptr<ComputationNode<ElemType>> pLearnableNode = dynamic_pointer_cast<ComputationNode<ElemType>>(x);
                wstring nodename = x->NodeName();
                m_LearnableNodePtr[nodename] = pLearnableNode;
            }
            for (auto& x : criterionNodes)
            {
                m_netCriterionNodes.push_back(dynamic_pointer_cast<ComputationNode<ElemType>>(x));
            }
            for (auto& x : evaluationNodes)
            {
                m_netEvaluationNodes.push_back(dynamic_pointer_cast<ComputationNode<ElemType>>(x));
            }
            m_netCriterionAccumulator->SetValue((ElemType) 0);
            m_netEvaluationAccumulator->SetValue((ElemType) 0);

            // emulate all the nodes, find nodes that have state
            m_netStatefulNodes = EnumerateStatefulNode(*net, criterionNodes, evaluationNodes);
            for (auto x : m_netStatefulNodes)
            {
                wstring name = x.first;
                m_netStates[name] = vector<shared_ptr<INodeState>>();
            }

            // for sequence training
            if (!criterionNodes.empty() && criterionNodes[0]->OperationName() == L"SequenceWithSoftmax")
            {
                auto node = dynamic_pointer_cast<SequenceWithSoftmaxNode<ElemType>>(criterionNodes[0]);
                assert(node);
                m_netLatticePtr = node->getLatticePtr();
                m_netExtrauttMapPtr = node->getextrauttmap();
                m_netUidPtr = node->getuidprt();
                m_netBoundariesPtr = node->getboundaryprt();
                m_hasLattices = true;
            }
            else
            {
                m_netLatticePtr = nullptr;
                m_netExtrauttMapPtr = nullptr;
                m_netUidPtr = nullptr;
                m_netBoundariesPtr = nullptr;
                m_hasLattices = false;
            }
        }

        size_t GetMinibatchIntoCache(IDataReader& trainSetDataReader,
                                     ComputationNetwork& net,
                                     StreamMinibatchInputs& inputMatrices,
                                     size_t requestedSubminibatches)
        {
            // first, remember interface to the net
            // BUGBUG (Issue #95): This will no longer be correct once we have multiple input layouts.
            m_netMBLayoutPtr = net.GetMBLayoutPtrOfNetwork();
            m_netInputMatrixPtr = inputMatrices;

            // second, get data from reader, stored it in cache
            // 1. for each key, allocate the specific matrix on device
            for (auto& pa : inputMatrices)
            {
                const wstring& name = pa.first;
                const auto& input = pa.second;
                auto& M = input.GetMatrix<ElemType>();
                if (m_inputMatricesCache.find(name) == m_inputMatricesCache.end())
                    m_inputMatricesCache.AddInput(name, make_shared<Matrix<ElemType>>(M, M.GetDeviceId()), input.pMBLayout, input.sampleLayout); // deep copy from M
                else
                    m_inputMatricesCache.GetInputMatrix<ElemType>(name).SetValue(M);
            }
            // 2. MBlayout
            m_MBLayoutCache->CopyFrom(net.GetMBLayoutPtrOfNetwork());
            size_t nParallelSequences = m_MBLayoutCache->GetNumParallelSequences();

            // 3. for bits in seq. training
            if (m_hasLattices)
            {
                m_LatticeCache.clear();
                m_uidCache.clear();
                m_extrauttmapCache.clear();
                m_BoundariesCache.clear();

                m_LatticeCache = *m_netLatticePtr;
                m_uidCache = *m_netUidPtr;
                m_extrauttmapCache = *m_netExtrauttMapPtr;
                m_BoundariesCache = *m_netBoundariesPtr;
            }

            // subminibatches are cutted at the parallel sequence level;
            // if #requested subminibatch is larger than #parallel sequence,
            // we cannot split further; instead, each subsequence become a subminibatch
            size_t actualnumSubminibatches = requestedSubminibatches > nParallelSequences ? nParallelSequences : requestedSubminibatches;

            // 4. third, allocate space for accumulated gradient
            for (auto& n : m_LearnableNodePtr)
            {
                auto node = n.second;
                if (node->IsParameterUpdateRequired())
                {
                    wstring nodeName = node->GetName();
                    shared_ptr<ComputationNode<ElemType>> pLearnableNode = node; // TODO: what's this for?
                    const auto& funvalue = pLearnableNode->Value(); // gradient may not be allocated when this function is first called
                    size_t nrow = funvalue.GetNumRows();
                    size_t ncol = funvalue.GetNumCols();
                    if (m_cachedGradient.find(nodeName) == m_cachedGradient.end())
                    {
                        // not allocated yet
                        auto matrixp = make_shared<Matrix<ElemType>>(nrow, ncol, funvalue.GetDeviceId());
                        matrixp->SetValue(0);
                        m_cachedGradient.AddInput(nodeName, matrixp, pLearnableNode->GetMBLayout()/*null*/, pLearnableNode->GetSampleLayout());
                    }
                }
            }
            // 5. for stateful node
            for (auto x : m_netStatefulNodes)
            {
                wstring name = x.first;
                if (m_netStates[name].empty())
                {
                    // this only happens in the first minibatch in an epoch
                    m_netStates[name].resize(actualnumSubminibatches);
                }
            }

            return (m_numSubminibatches = actualnumSubminibatches);
        }

        void DecimateLattices(
            LatticePtr decimatedLattices,         /* output: lattices after decimation*/
            BoundariesPtr decimatedBoundaryPtr,   /* output: boundary after decimation*/
            ExtrauttMapPtr decimatedExtraMapPtr,  /* output: extramap after decimation*/
            UidPtr decimatedUidPtr,               /* output: Uid after decimation*/
            const Lattice lattices,               /* input: lattices to be decimated */
            const Boundaries boundaries,          /* input: boundary to be decimated */
            const ExtrauttMap extraMaps,          /* input: extra map to be decimated */
            const Uid uids,                       /* input: uid to be decimated*/
            pair<size_t, size_t> parallelSeqRange /* input: what parallel sequence range we are looking at */
            )
        {
            size_t parallelSeqStId = parallelSeqRange.first;
            size_t parallelSeqEnId = parallelSeqRange.second;

            decimatedLattices->clear();
            decimatedBoundaryPtr->clear();
            decimatedExtraMapPtr->clear();
            decimatedUidPtr->clear();

            size_t stFrame = 0;
            for (size_t iUtt = 0; iUtt < extraMaps.size(); iUtt++)
            {
                size_t numFramesInThisUtterance = lattices[iUtt]->getnumframes();
                size_t iParallelSeq = extraMaps[iUtt]; // i-th utterance belongs to iParallelSeq-th parallel sequence
                if (iParallelSeq >= parallelSeqStId && iParallelSeq < parallelSeqEnId)
                {
                    // this utterance has been selected
                    decimatedLattices->push_back(lattices[iUtt]);
                    decimatedBoundaryPtr->insert(decimatedBoundaryPtr->end(), boundaries.begin() + stFrame, boundaries.begin() + stFrame + numFramesInThisUtterance);
                    decimatedUidPtr->insert(decimatedUidPtr->end(), uids.begin() + stFrame, uids.begin() + stFrame + numFramesInThisUtterance);
                    decimatedExtraMapPtr->push_back(extraMaps[iUtt] - parallelSeqStId);
                }
                stFrame += numFramesInThisUtterance;
            }
        }

        void GetSubMinibatchToNet(size_t iSubminibatch)
        {
            Matrices decimatedMatrices;
            MBLayoutPtr decimatedLayout;
            pair<size_t, size_t> seqRange = DataReaderHelpers::DecimateMinibatch<ElemType>(m_inputMatricesCache, decimatedMatrices, m_MBLayoutCache, decimatedLayout, m_numSubminibatches, iSubminibatch);
            //  NOTE: deimatedMatrices must be released by caller

            // base on the seqRange, we do the decimation for lattices and related variables
            if (m_hasLattices)
            {
                DecimateLattices(
                    /*output */
                    m_netLatticePtr, m_netBoundariesPtr, m_netExtrauttMapPtr, m_netUidPtr,
                    /*input to be decimated */
                    m_LatticeCache, m_BoundariesCache, m_extrauttmapCache, m_uidCache,
                    /* what range we want ? */
                    seqRange);
            }

            // The following does m_netInputMatrixPtr = decimatedMatrices; with ownership shenanigans.
            for (auto& x : decimatedMatrices)
            {
                const wstring& name = x.first;
                m_netInputMatrixPtr.GetInputMatrix<ElemType>(name).SetValue(decimatedMatrices.GetInputMatrix<ElemType>(name));
            }

            m_netMBLayoutPtr->CopyFrom(decimatedLayout);

            for (auto& x : m_netStatefulNodes)
            {
                const wstring& name = x.first;
                auto& pNode         = x.second;
                if (m_netStates[name][iSubminibatch])
                    pNode->ImportState(std::move(m_netStates[name][iSubminibatch]));
            }
        }

        // TODO: encapsulate it into a destructor? Note: Cannot throw exceptions in destructor.
        void DoneWithCurrentSubMinibatch(size_t iSubminibatch)
        {
            // accumulate gradient here
            for (auto x : m_cachedGradient)
            {
                wstring nodename = x.first;
                if (m_LearnableNodePtr.find(nodename) == m_LearnableNodePtr.end())
                {
                    RuntimeError("ERROR: in DoneWithCurrentSubMinibatch: node %ls not found in LeanrableNode", nodename.c_str());
                }
                shared_ptr<ComputationNode<ElemType>> pNode = m_LearnableNodePtr[nodename];
                m_cachedGradient.GetInputMatrix<ElemType>(nodename) += pNode->Gradient();
                pNode->Gradient().SetValue(0);
            }
            // accumulate criterion value
            if (!m_netCriterionNodes.empty())
            {
                Matrix<ElemType>::AddElementToElement(m_netCriterionNodes[0]->Value(), 0, 0,
                                                      *m_netCriterionAccumulator, 0, 0);
                m_netCriterionNodes[0]->Value().SetValue(0);
            }
            // accumulate evaluation value
            for (size_t i = 0; i < m_netEvaluationNodes.size(); i++)
            {
                Matrix<ElemType>::AddElementToElement(m_netEvaluationNodes[i]->Value(), 0, 0,
                                                      *m_netEvaluationAccumulator, 0, i);
                m_netEvaluationNodes[i]->Value().SetValue(0);
            }

            // Export node state
            for (auto& x : m_netStatefulNodes)
            {
                const wstring& name = x.first;
                m_netStates[name][iSubminibatch] = x.second->ExportState();
            }
        }

        void DoneWithCurrentMinibatch()
        {
            for (auto& x : m_cachedGradient)
            {
                const wstring& name  = x.first;
                auto& accumulategrad = m_cachedGradient.GetInputMatrix<ElemType>(name);

                if (m_LearnableNodePtr.find(name) == m_LearnableNodePtr.end())
                    LogicError("DoneWithCurrentSubMinibatch: Node '%ls' not found in LearnableNode set.", name.c_str());
                m_LearnableNodePtr[name]->Gradient().SetValue(accumulategrad);
                accumulategrad.SetValue(0);
            }
            // also revert net.m_MBLayoutPtr
            m_netMBLayoutPtr->CopyFrom(m_MBLayoutCache);

            if (!m_netCriterionNodes.empty())
            {
                // m_netCriterionNodes[0]->Value().SetValue((ElemType)0);
                Matrix<ElemType>::AddElementToElement(*m_netCriterionAccumulator, 0, 0,
                                                      m_netCriterionNodes[0]->Value(), 0, 0);
            }
            m_netCriterionAccumulator->SetValue(0);

            for (size_t i = 0; i < m_netEvaluationNodes.size(); i++)
            {
                // m_netEvaluationNodes[i]->Value().SetValue((ElemType)0);
                Matrix<ElemType>::AddElementToElement(*m_netEvaluationAccumulator, 0, i,
                                                      m_netEvaluationNodes[i]->Value(), 0, 0);
            }
            m_netEvaluationAccumulator->SetValue(0);
        }
    };
};

}}}
// BUGBUG: If I add a 'x' here, I get an error in ConvolutionEngine.h included from SGD.cpp. Why does ConvolutionEngine.h depend on this header, or whichever is included right before?
