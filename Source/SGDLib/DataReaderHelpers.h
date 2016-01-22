// DataReaderHelper.h -- helper functions that understand both DataReader and ComputationNetwork

#pragma once

#include "Basics.h"
#include "DataReader.h"
#include "ComputationNetwork.h"
#include "MPIWrapper.h"
#include "TrainingCriterionNodes.h"
#include <string>
#include <map>
#include <set>

namespace Microsoft { namespace MSR { namespace CNTK {

/*static*/ struct DataReaderHelpers
{

    // -------------------------------------------------------------------
    // GetMinibatchIntoNetwork() -- get one minibatch from Reader (this->trainSetDataReader) into Network (this->net)
    // Returns false if no data is read. In that case, no other return value can be expected to contain meaningful values (e.g. actualMBSize will be unchanged).
    // Sets actualMBSize to the number of matrix columns. Note that 0 is a valid value to be returned for actualMBSize, caller must handle that correctly.
    // -------------------------------------------------------------------

    // Note: This will go away with the redesigned reader interface.
    // TODO: callers of this often do ComputationNetwork::BumpEvalTimeStamp(featureNodes) and also for labels; we should eliminate the need for this.
    template <class ElemType>
    static bool GetMinibatchIntoNetwork(IDataReader<ElemType>& trainSetDataReader,
                                        ComputationNetworkPtr net,
                                        ComputationNodeBasePtr criterionNode,
                                        bool useDistributedMBReading,
                                        bool useParallelTrain,
                                        std::map<std::wstring, Matrix<ElemType>*>& inputMatrices,
                                        size_t& actualMBSize)
    {
        auto pMBLayout = net->GetMBLayoutPtr();
        // Reading consists of a sequence of Reader API calls:
        //  - GetMinibatch() --fills the inputMatrices
        //  - SetActualMiniBatchSizeFromFeatures()  --tells Network to resize the nodes' buffers
        //  - CopyMBLayoutTo()   --copies the MBLayout from Reader to Network
        //  - VerifyActualNumParallelSequences()  --(refactoring left-over) verify that MBLayout is consistent with #parallel sequences
        // with the special twist that in presence of parallelization, there is some decimation involved.

        bool wasDataRead = trainSetDataReader.GetMinibatch(inputMatrices); // fill in the minibatch data into the Input nodes' buffers directly
        // If this returns false, the matrices may contain garbage or not sized to 0 columns.
        // On the other hand, if it returns a 0-column matrix, that would be a perfectly cromulent minibatch (in case of data parallelism with distributed reading).

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

        // get layout meta-data
        trainSetDataReader.CopyMBLayoutTo(pMBLayout);

        // decimate if needed. Decimation happens in-place.
        if (!useDistributedMBReading && useParallelTrain)
            DecimateMinibatch(inputMatrices, g_mpi->NumNodesInUse(), g_mpi->CurrentNodeRank(), net->GetMBLayoutPtr());

        // reader will have resized input node's m_value directly. Nodes must be notified to do necessary internal state updates from that.
        // TODO: This is a stopgap. SGD will at some point change from sets of matrices to sets of nodes. Then this will become much simpler.
        std::set<Matrix<ElemType>*> matrices;
        for (const auto& iter : inputMatrices)
            matrices.insert(iter.second);
        for (auto& node : net->FeatureNodes())
            if (matrices.find(&node->As<ComputationNode<ElemType>>()->Value()) != matrices.end())
                node->NotifyFunctionValuesMBSizeModified();
        for (auto& node : net->LabelNodes())
            if (matrices.find(&node->As<ComputationNode<ElemType>>()->Value()) != matrices.end())
                node->NotifyFunctionValuesMBSizeModified();

        // get MB size and tell Network to update its nodes' buffers based on what's in the input matrices
        // Note: Decimation may have reduced this to 0 frames. We still must return 'true'.
        // BUGBUG: This has a definitional problem once we support multiple feature streams with different lenghts.
        actualMBSize = net->DetermineActualMBSizeFromFeatures();

        return true;
    }

    // -------------------------------------------------------------------
    // DecimateMinibatch - decimate minibatch for parallelization
    // -------------------------------------------------------------------
    // non-inplace decimation , to be used in subminibatch implementation
    // returns a subset of parallel sequences
    template <class ElemType>
    static pair<size_t, size_t> DecimateMinibatch(const std::map<std::wstring, Matrix<ElemType>*> MB,     // input matrices
                                                  std::map<std::wstring, Matrix<ElemType>*>& decimatedMB, // output decimated matrices.
                                                  MBLayoutPtr pMBLayout,                                  // input MBLayout
                                                  MBLayoutPtr& pDecimateMBLayout,                         // output decimated MBLayout (note: cannot work in-place)
                                                  int numWorker, int rank)
    {
        size_t numParallelSequences = pMBLayout->GetNumParallelSequences();
        size_t nT = pMBLayout->GetNumTimeSteps();

        // decide start column and end column
        size_t st = numParallelSequences * (size_t) rank / numWorker;
        size_t en = numParallelSequences * (size_t)(rank + 1) / numWorker;
        en = en > numParallelSequences ? numParallelSequences : en; // TODO: why are these two tests necessary?
        en = (rank == numWorker - 1) ? numParallelSequences : en;
        size_t numNewParallelSequence = en - st;

        // begin decimate matrices
        size_t rv = 0;
        for (const auto& it : MB)
        {
            wstring name = it.first;
            MSR::CNTK::Matrix<ElemType>& mat = *it.second;
            size_t numRows = mat.GetNumRows();
            size_t numCols = mat.GetNumCols();
            int devID = mat.GetDeviceId();

            if (rv == 0)
                rv = numCols;
            else if (rv != numCols)
                LogicError("DecimateMinibatch: Inconsistent number of columns among inputs (found %d and %d).", (int) rv, (int) numCols);

            if (nT != numCols / numParallelSequences)
                LogicError("ERROR: MBLayout borked, GetNumTimeSteps() mismatches minibatch number of columns\n");

            decimatedMB[name] = new Matrix<ElemType>(devID);
            decimatedMB[name]->AssignRowSliceValuesOf(mat.Reshaped(numRows * numParallelSequences, nT), st * numRows, (en - st) * numRows);
            decimatedMB[name]->Reshape(numRows, numNewParallelSequence * nT);
            // If we had a RowSlice function, we would like to write in this way
            // decimatedMB[name]->SetValue(mat.Reshaped(nRows*nSequence, nT).RowSlice( st*nRows , (en-st)*nRows).Reshaped(nRows, nNewParallelSequence*nT));
        }
        // decimate MBLayout as well
        pDecimateMBLayout = make_shared<MBLayout>(numNewParallelSequence, nT);
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
    static pair<size_t, size_t> DecimateMinibatch(std::map<std::wstring, Matrix<ElemType>*>& mb, // matrix to be decimated
                                                  int numprocs, int rank,                        // rank info
                                                  MBLayoutPtr pMBLayout)                         // get decimated as well
    {
        if (numprocs == 1)
            return pair<size_t, size_t>(0, pMBLayout->GetNumParallelSequences());
        // no need to do inplace decimation if numproc == 1

        // allocate space for non-inplace decimation
        MBLayoutPtr pDecimatedMB = make_shared<MBLayout>();
        std::map<wstring, Matrix<ElemType>*> decimatedMB;
        // call in-place decimation
        pair<size_t, size_t> selected = DecimateMinibatch(mb, decimatedMB, pMBLayout, pDecimatedMB, numprocs, rank);
        // move the data
        for (auto k : mb)
        {
            auto name = k.first;
            k.second->SetValue(*decimatedMB[name]);
            delete decimatedMB[name];
            decimatedMB[name] = nullptr;
        }
        pMBLayout->MoveFrom(pDecimatedMB);
        return selected;
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
        typedef std::map<std::wstring, Matrix<ElemType>*> Matrices;

        // member variables served as caching space
        Matrices m_inputMatricesCache;
        MBLayoutPtr m_MBLayoutCache;
        Lattice m_LatticeCache;
        Uid m_uidCache;
        ExtrauttMap m_extrauttmapCache;
        Boundaries m_BoundariesCache;
        shared_ptr<Matrix<ElemType>> m_NetCriterionAccumulator;
        shared_ptr<Matrix<ElemType>> m_NetEvaluationAccumulator;
        std::map<wstring, vector<shared_ptr<INodeState>>> m_NetStates; // m_NetStatefulNodes[node][i] caches the state of i-th subminibatch of node
        bool m_hasLattices;

        Matrices m_cachedGradient;
        // we also need to remember where to put into the net
        MBLayoutPtr m_NetMBLayoutPtr;
        std::map<wstring, shared_ptr<ComputationNode<ElemType>>> m_LearnableNodePtr;
        // followings are lattice-related
        Matrices m_NetInputMatrixPtr; // TODO: camelCase for all m_Net...
        LatticePtr m_NetLatticePtr;
        UidPtr m_NetUidPtr;
        ExtrauttMapPtr m_NetExtrauttMapPtr;
        BoundariesPtr m_NetBoundariesPtr;
        // we remember the pointer to the learnable Nodes so that we can accumulate the gradient once a sub-minibatch is done

        size_t m_numParallelSequences; // number of paralle sequence in the cached matrix and MBLayout
        size_t m_numSubminibatches;    // how many subminibatches we are going to use ?

        std::vector<shared_ptr<ComputationNode<ElemType>>> m_NetCriterionNodes;
        std::vector<shared_ptr<ComputationNode<ElemType>>> m_NetEvaluationNodes;
        std::map<wstring, shared_ptr<IStatefulNode>> m_NetStatefulNodes; // we need to Export/Import states of stateful nodes when we swtich subminibatches

    private:
        void EnumerateStatefulNodeWithRoot(ComputationNetwork& net, ComputationNodeBasePtr root, std::map<wstring, shared_ptr<IStatefulNode>>& statefulnode)
        {
            const std::list<ComputationNodeBasePtr> evalorder = net.GetEvalOrder(root);
            for (auto& x : evalorder)
            {
                wstring name = x->GetName();
                if (statefulnode.find(name) != statefulnode.end())
                    continue; // already in the list
                shared_ptr<IStatefulNode> pNode = dynamic_pointer_cast<IStatefulNode>(x);
                if (pNode)
                {
                    statefulnode[name] = pNode;
                }
            }
        }

        std::map<wstring, shared_ptr<IStatefulNode>> EnumerateStatefulNode(ComputationNetwork& net,
                                                                           const std::vector<ComputationNodeBasePtr>& criterionNode,
                                                                           const std::vector<ComputationNodeBasePtr>& evaluationNode)
        {
            std::map<wstring, shared_ptr<IStatefulNode>> statefulNodes;
            for (auto& root : criterionNode)
            {
                EnumerateStatefulNodeWithRoot(net, root, statefulNodes);
            }
            for (auto& root : evaluationNode)
            {
                EnumerateStatefulNodeWithRoot(net, root, statefulNodes);
            }
            return statefulNodes;
        }

    public:
        SubminibatchDispatcher()
            : m_MBLayoutCache(nullptr), m_NetLatticePtr(nullptr), m_NetExtrauttMapPtr(nullptr), m_NetUidPtr(nullptr), m_NetBoundariesPtr(nullptr)
        {
        }

        void Init(ComputationNetworkPtr& net,
                  const std::list<ComputationNodeBasePtr>& learnableNodes,
                  const std::vector<ComputationNodeBasePtr>& criterionNodes,
                  const std::vector<ComputationNodeBasePtr>& evaluationNodes)
        {
            m_MBLayoutCache = make_shared<MBLayout>();
            m_NetCriterionAccumulator = make_shared<Matrix<ElemType>>(1, 1, net->GetDeviceId());
            m_NetEvaluationAccumulator = make_shared<Matrix<ElemType>>(1, evaluationNodes.size(), net->GetDeviceId());
            // remember ptrs to learnable nodes
            for (auto x : learnableNodes)
            {
                shared_ptr<ComputationNode<ElemType>> pLearnableNode = dynamic_pointer_cast<ComputationNode<ElemType>>(x);
                wstring nodename = x->NodeName();
                m_LearnableNodePtr[nodename] = pLearnableNode;
            }
            for (auto& x : criterionNodes)
            {
                m_NetCriterionNodes.push_back(dynamic_pointer_cast<ComputationNode<ElemType>>(x));
            }
            for (auto& x : evaluationNodes)
            {
                m_NetEvaluationNodes.push_back(dynamic_pointer_cast<ComputationNode<ElemType>>(x));
            }
            m_NetCriterionAccumulator->SetValue((ElemType) 0);
            m_NetEvaluationAccumulator->SetValue((ElemType) 0);

            // emulate all the nodes, find nodes that have state
            m_NetStatefulNodes = EnumerateStatefulNode(*net, criterionNodes, evaluationNodes);
            for (auto x : m_NetStatefulNodes)
            {
                wstring name = x.first;
                m_NetStates[name] = vector<shared_ptr<INodeState>>();
            }

            // for sequence training
            if (criterionNodes[0]->OperationName() == L"SequenceWithSoftmax")
            {
                auto node = dynamic_pointer_cast<SequenceWithSoftmaxNode<ElemType>>(criterionNodes[0]);
                assert(node);
                m_NetLatticePtr = node->getLatticePtr();
                m_NetExtrauttMapPtr = node->getextrauttmap();
                m_NetUidPtr = node->getuidprt();
                m_NetBoundariesPtr = node->getboundaryprt();
                m_hasLattices = true;
            }
            else
            {
                m_NetLatticePtr = nullptr;
                m_NetExtrauttMapPtr = nullptr;
                m_NetUidPtr = nullptr;
                m_NetBoundariesPtr = nullptr;
                m_hasLattices = false;
            }
        }

        ~SubminibatchDispatcher()
        {
            // TODO: remove these by using shared_ptr

            for (auto x : m_inputMatricesCache)
            {
                delete x.second;
            }

            for (auto x : m_cachedGradient)
            {
                delete x.second;
            }
        }

        size_t GetMinibatchIntoCache(IDataReader<ElemType>& trainSetDataReader,
                                     ComputationNetwork& net,
                                     std::map<std::wstring, Matrix<ElemType>*>& inputMatrices,
                                     size_t requestedSubminibatches)
        {
            // first, remember interface to the net
            m_NetMBLayoutPtr = net.GetMBLayoutPtr();
            m_NetInputMatrixPtr = inputMatrices;

            // second, get data from reader, stored it in cache
            // 1. for each key, allocate the specific matrix on device
            for (auto pa : inputMatrices)
            {
                wstring name = pa.first;
                Matrix<ElemType>* M = pa.second;
                if (m_inputMatricesCache.find(name) == m_inputMatricesCache.end())
                {
                    m_inputMatricesCache[name] = new Matrix<ElemType>(*M, M->GetDeviceId()); // deep copy from M
                }
                else
                {
                    m_inputMatricesCache[name]->SetValue(*M);
                }
            }
            // 2. MBlayout
            m_MBLayoutCache->CopyFrom(net.GetMBLayoutPtr());
            size_t nParallelSequences = m_MBLayoutCache->GetNumParallelSequences();

            // 3. for bits in seq. training
            if (m_hasLattices)
            {
                m_LatticeCache.clear();
                m_uidCache.clear();
                m_extrauttmapCache.clear();
                m_BoundariesCache.clear();

                m_LatticeCache = *m_NetLatticePtr;
                m_uidCache = *m_NetUidPtr;
                m_extrauttmapCache = *m_NetExtrauttMapPtr;
                m_BoundariesCache = *m_NetBoundariesPtr;
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
                    shared_ptr<ComputationNode<ElemType>> pLearnableNode = node;
                    auto funvalue = pLearnableNode->Value(); // gradient may not be allocated when this function is first called
                    size_t nrow = funvalue.GetNumRows();
                    size_t ncol = funvalue.GetNumCols();
                    if (m_cachedGradient.find(nodeName) == m_cachedGradient.end())
                    {
                        // not allocated yet
                        m_cachedGradient[nodeName] = new Matrix<ElemType>(nrow, ncol, funvalue.GetDeviceId());
                        m_cachedGradient[nodeName]->SetValue((ElemType) 0);
                    }
                }
            }
            // 5. for stateful node
            for (auto x : m_NetStatefulNodes)
            {
                wstring name = x.first;
                if (m_NetStates[name].empty())
                {
                    // this only happens in the first minibatch in an epoch
                    m_NetStates[name].resize(actualnumSubminibatches);
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
            pair<size_t, size_t> seqRange = DataReaderHelpers::DecimateMinibatch(m_inputMatricesCache, decimatedMatrices, m_MBLayoutCache, decimatedLayout, m_numSubminibatches, iSubminibatch);
            //  NOTE: deimatedMatrices must be released by caller

            // base on the seqRange, we do the decimation for lattices and related variables
            if (m_hasLattices)
            {
                DecimateLattices(
                    /*output */
                    m_NetLatticePtr, m_NetBoundariesPtr, m_NetExtrauttMapPtr, m_NetUidPtr,
                    /*input to be decimated */
                    m_LatticeCache, m_BoundariesCache, m_extrauttmapCache, m_uidCache,
                    /* what range we want ? */
                    seqRange);
            }

            //m_NetInputMatrixPtr = decimatedMatrices;
            for (auto& x : decimatedMatrices)
            {
                wstring name = x.first;
                m_NetInputMatrixPtr[name]->SetValue(*x.second);
                delete x.second; // TODO: is it safe to delete here ? Yes! SetValue call cuda memcpy so it is a blocking call
                x.second = nullptr;
            }

            m_NetMBLayoutPtr->CopyFrom(decimatedLayout);

            for (auto& x : m_NetStatefulNodes)
            {
                wstring name = x.first;
                shared_ptr<IStatefulNode> pNode = x.second;
                if (m_NetStates[name][iSubminibatch])
                    pNode->ImportState(std::move(m_NetStates[name][iSubminibatch]));
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
                m_cachedGradient[nodename]->operator+=(pNode->Gradient());
                pNode->Gradient().SetValue((ElemType) 0);
            }
            // accumulate criterion value
            Matrix<ElemType>::AddElementToElement(m_NetCriterionNodes[0]->Value(), 0, 0,
                                                  *m_NetCriterionAccumulator, 0, 0);
            m_NetCriterionNodes[0]->Value().SetValue((ElemType) 0);
            // accumulate evaluation value
            for (size_t i = 0; i < m_NetEvaluationNodes.size(); i++)
            {
                Matrix<ElemType>::AddElementToElement(m_NetEvaluationNodes[i]->Value(), 0, 0,
                                                      *m_NetEvaluationAccumulator, 0, i);
                m_NetEvaluationNodes[i]->Value().SetValue((ElemType) 0);
            }

            // Export node state
            for (auto& x : m_NetStatefulNodes)
            {
                wstring name = x.first;
                m_NetStates[name][iSubminibatch] = x.second->ExportState();
            }
        }

        void DoneWithCurrentMinibatch()
        {
            for (auto& x : m_cachedGradient)
            {
                wstring name = x.first;
                Matrix<ElemType>* accumulategrad = x.second;

                if (m_LearnableNodePtr.find(name) == m_LearnableNodePtr.end())
                {
                    // should never happen, remove this code later
                    RuntimeError("ERROR: in DoneWithCurrentSubMinibatch: node %ls not found in LearnableNode", name.c_str());
                }
                m_LearnableNodePtr[name]->Gradient().SetValue(*accumulategrad);
                x.second->SetValue((ElemType) 0);
            }
            // also revert net.m_MBLayoutPtr
            m_NetMBLayoutPtr->CopyFrom(m_MBLayoutCache);

            //m_NetCriterionNodes[0]->Value().SetValue((ElemType)0);
            Matrix<ElemType>::AddElementToElement(*m_NetCriterionAccumulator, 0, 0,
                                                  m_NetCriterionNodes[0]->Value(), 0, 0);
            m_NetCriterionAccumulator->SetValue((ElemType) 0);

            for (size_t i = 0; i < m_NetEvaluationNodes.size(); i++)
            {
                //m_NetEvaluationNodes[i]->Value().SetValue((ElemType)0);
                Matrix<ElemType>::AddElementToElement(*m_NetEvaluationAccumulator, 0, i,
                                                      m_NetEvaluationNodes[i]->Value(), 0, 0);
            }
            m_NetEvaluationAccumulator->SetValue((ElemType) 0);
        }
    };
};
} } }
