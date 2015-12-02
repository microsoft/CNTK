// DataReaderHelper.h -- helper functions that understand both DataReader and ComputationNetwork

#pragma once

#include "Basics.h"
#include "DataReader.h"
#include "ComputationNetwork.h"
#include "MPIWrapper.h"
#include <string>
#include <map>
#include "TrainingCriterionNodes.h"


namespace Microsoft { namespace MSR { namespace CNTK {

    /*static*/ struct DataReaderHelpers
    {

        // -------------------------------------------------------------------
        // GetMinibatchIntoNetwork() -- get one minibatch from Reader (this->trainSetDataReader) into Network (this->net)
        // Returns false if end of epoch has been reached.
        // Sets actualMBSize to the number of matrix columns. Note that 0 is a valid value to be returned for actualMBSize, caller must handle that correctly.
        // -------------------------------------------------------------------

        // Note: This will go away with the redesigned reader interface.
        // TODO: callers of this often do ComputationNetwork::UpdateEvalTimeStamps(featureNodes) and also for labels; we should eliminate the need for this.
        template<class ElemType>
        static bool GetMinibatchIntoNetwork(IDataReader<ElemType>& trainSetDataReader,
            ComputationNetworkPtr net,
            ComputationNodeBasePtr criterionNode,
            bool useDistributedMBReading,
            bool useParallelTrain,
            std::map<std::wstring, Matrix<ElemType>*> & inputMatrices,
            size_t & actualMBSize)
        {
            auto pMBLayout = net->GetMBLayoutPtr();
            // Reading consists of a sequence of Reader API calls:
            //  - GetMinibatch() --fills the inputMatrices
            //  - SetActualMiniBatchSizeFromFeatures()  --tells Network to resize the nodes' buffers
            //  - CopyMBLayoutTo()   --copies the MBLayout from Reader to Network
            //  - VerifyActualNumParallelSequences()  --(refactoring left-over) verify that MBLayout is consistent with #parallel sequences
            // with the special twist that in presence of parallelization, there is some decimation involved.

            // TODO: how is !wasDataRead semantically different from inputMatrices having zero columns?
            // TODO: The reader does not always resize the input matrices to zero when 
            //       no data is read. When it does, 'wasDataRead' can be removed. Will go away with reader redesig.
            bool wasDataRead = trainSetDataReader.GetMinibatch(inputMatrices);      // fill in the minibatch data into the Input nodes' buffers directly
            // reader will have resized input node's m_functionValues directly. Nodes must be notified to do necessary internal state updates from that.
            net->NotifyInputNodesFunctionValuesMBSizeModified();
            size_t readMBSize = net->DetermineActualMBSizeFromFeatures();
            if (readMBSize == 0)
                wasDataRead = false;

            trainSetDataReader.CopyMBLayoutTo(pMBLayout);                           // and layout meta-data

            // verify some DataReader calls that are redundant since the MBLayout refactoring (keep verifying for a while for cosy feeling)
            net->VerifyActualNumParallelSequences(trainSetDataReader.GetNumParallelSequences()); // info already contained in MBLayout

            if ((criterionNode != nullptr) && (criterionNode->OperationName() == L"SequenceWithSoftmax"))
            {
                auto node = dynamic_pointer_cast<SequenceWithSoftmaxNode<ElemType>>(criterionNode);
                auto latticeinput = node->getLatticePtr();
                auto uids = node->getuidprt();
                auto boundaries = node->getboundaryprt();
                auto extrauttmap = node->getextrauttmap();

                trainSetDataReader.GetMinibatch4SE(*latticeinput, *uids, *boundaries, *extrauttmap);
            }

            // did we reach end of epoch?
            if (useDistributedMBReading)
            {
                // In case of distributed reading, the current node needs to continue even with a minibatch size of 0 if any
                // other node in the group has a non-zero size minibatch to process. This is needed to ensure that
                // the gradient aggregation barriers do not get stuck and also to ensure that all nodes update their weights
                // properly using the aggregate gradients from other nodes before moving on to the next epoch even though the current
                // node itself may not have any gradient contribution.
                // TODO: wasDataRead == false means end of epoch, right? Is this state idempotent?
                std::array<int, 1> numNodesWithDataToProcess;
                numNodesWithDataToProcess[0] = wasDataRead ? 1 : 0;
                g_mpi->AllReduce(numNodesWithDataToProcess);

                if (numNodesWithDataToProcess[0] == 0)
                    return false;   // end of epoch
            }
            else if (!wasDataRead)
                return false;       // end of epoch

            // We are not at the end of epoch.
            // Note, however, that in case of parallelization, this MPI rank may have received a share of 0 samples. Calling code, beware.

            // decimate if needed. Decimation happens in-place.
            if (wasDataRead && !useDistributedMBReading && useParallelTrain)
            {
                DecimateMinibatch(inputMatrices, g_mpi->NumNodesInUse(), g_mpi->CurrentNodeRank(), net->GetMBLayoutPtr());
                net->NotifyInputNodesFunctionValuesMBSizeModified(); // need to tell'm again since we modified it again
            }

            // get MB size and tell Network to update its nodes' buffers based on what's in the input matrices
            // Note: Decimation may have reduced this to 0 frames, in which case we must return 'true'.
            actualMBSize = 0;
            if (wasDataRead)    // TODO: what if we call it always?
                actualMBSize = net->DetermineActualMBSizeFromFeatures(); // TODO: don't we know the size from reader? Should this be a check instead?

            return true;
        }

        // -------------------------------------------------------------------
        // DecimateMinibatch - decimate minibatch for parallelization
        // -------------------------------------------------------------------
        // non-inplace decimation , to be used in subminibatch implementation 
        template<class ElemType>
        static void DecimateMinibatch(const std::map<std::wstring, Matrix<ElemType>*> MB,           // input matrices 
            std::map<std::wstring, Matrix<ElemType>*>& decimatedMB,                                 // output decimated matrices. 
                                                                                                    // Caller need to release the memory themselves!!!   TODO: use shared_ptr 
            MBLayoutPtr pMBLayout,                                                                  // input MBLayout 
            MBLayoutPtr& pDecimateMBLayout,                                                         // output decimated MBLayout 
            int numWorker, int rank)
        {
            size_t numParallelSequences = pMBLayout->GetNumParallelSequences();
            size_t nT = pMBLayout->GetNumTimeSteps();

            // decide start column and end column 
            size_t st = numParallelSequences * (size_t)rank / numWorker;
            size_t en = numParallelSequences * (size_t)(rank + 1) / numWorker;
            en = en > numParallelSequences ? numParallelSequences : en;
            en = (rank == numWorker - 1) ? numParallelSequences : en;
            size_t numNewParallelSequence = en - st;

#if 0   // per discussion with Frank and Amit, we remove this warning since the same decimation function is also called for frame-mode
            // warning if needed 
            static bool bWarned = false;
            if (!bWarned && nSequence % numWorker != 0)
            {
                /* give a warning of potential bandwidth wasting */
                fprintf(stderr, "DecimateMinibatch: WARNING: Number of parallel utterances %d not a multiple of number of GPUs %d, GPU usage will be suboptimal.\n",
                    (int)nSequence, (int)numWorker);
                bWarned = true;
            }
#endif 
            // begin decimate matrices 
            size_t rv = 0;
            for (const auto& it : MB)
            {
                wstring name = it.first;
                MSR::CNTK::Matrix<ElemType> & mat = *it.second;
                size_t numRows = mat.GetNumRows();
                size_t numCols = mat.GetNumCols();
                int devID = mat.GetDeviceId();

                if (rv == 0)
                    rv = numCols;
                else if (rv != numCols)
                    LogicError("DecimateMinibatch: Inconsistent number of columns among inputs (found %d and %d).", (int)rv, (int)numCols);

                if (nT != numCols / numParallelSequences)
                    LogicError("ERROR: MBLayout borked, GetNumTimeSteps() mismatches minibatch number of columns\n");

                decimatedMB[name] = new Matrix<ElemType>(devID);
                decimatedMB[name]->AssignRowSliceValuesOf(mat.Reshaped(numRows*numParallelSequences, nT), st*numRows, (en - st)*numRows);
                decimatedMB[name]->Reshape(numRows, numNewParallelSequence*nT);
                // If we have RowSlice function, we would like to write in this way 
                // decimatedMB[name]->SetValue(mat.Reshaped(nRows*nSequence, nT).RowSlice( st*nRows , (en-st)*nRows).Reshaped(nRows, nNewParallelSequence*nT));
            }
            // decimate MBLayout as well 
            pDecimateMBLayout = make_shared<MBLayout>(numNewParallelSequence, nT, true);
            for (size_t t = 0; t < nT; t++) for (size_t id = 0; id < numNewParallelSequence; id++)
                pDecimateMBLayout->Set(id, t, pMBLayout->Get(id + st, t));

        }

      // Inpace decimation 
        template<class ElemType>
        static void DecimateMinibatch(std::map<std::wstring, Matrix<ElemType>*> &mb,    // matrix to be decimated
                                        int numprocs, int rank,                           // rank info
                                        MBLayoutPtr pMBLayout                              // get decimated as well 
                                        )
        {
            if (numprocs == 1)
                return;
            // no need to do inplace decimation if numproc == 1 

            // allocate space for non-inplace decimation 
            MBLayoutPtr pDecimatedMB = make_shared<MBLayout>();
            std::map<wstring, Matrix<ElemType>*>    decimatedMB;
            // call in-place decimation 
            DecimateMinibatch(mb, decimatedMB, pMBLayout, pDecimatedMB, numprocs, rank);
            // move the data 
            for (auto k : mb)
            {
                auto name = k.first;
                k.second->SetValue(*decimatedMB[name]);
                delete decimatedMB[name];
                decimatedMB[name] = nullptr;
            }
            pMBLayout->MoveFrom(pDecimatedMB);
        }
        // SubminibatchHelpers
        // Helper for sub-minibatch implementation
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

        template<class ElemType>
        class SubminibatchDispatcher
        {
        private:
            typedef            std::vector<shared_ptr<const msra::dbn::latticesource::latticepair>>         Lattice;
            typedef            std::vector<size_t>                                                          Uid;
            typedef            std::vector<size_t>                                                          ExtrauttMap;

            typedef            std::vector<shared_ptr<const msra::dbn::latticesource::latticepair>>*        LatticePtr;
            typedef            std::vector<size_t>*                                                         UidPtr;
            typedef            std::vector<size_t>*                                                         ExtrauttMapPtr;
            typedef            std::map<std::wstring, Matrix<ElemType>*>                                    Matrices;


            // member variables served as caching space 
            Matrices                                            m_inputMatricesCache;
            MBLayoutPtr                                         m_MBLayoutCache;
            LatticePtr                                          m_LatticeCache;
            UidPtr                                              m_uidCache;
            ExtrauttMapPtr                                      m_extrauttmapCache;
            shared_ptr<Matrix<ElemType>>                        m_NetCriterionAccumulator;
            shared_ptr<Matrix<ElemType>>                        m_NetEvaluationAccumulator;
            std::map<wstring, vector<shared_ptr<INodeState>>>   m_NetStates;            // m_NetStatefulNodes[node][i] caches the state of i-th subminibatch of node


            Matrices                                            m_CachedGraident;
            // we also need to remember where to put into the net
            MBLayoutPtr                                         m_NetMBLayoutPtr;
            std::map<wstring, shared_ptr<ComputationNode<ElemType>>>    m_LearnableNodePtr;
            // followings are lattice-related 
            Matrices                                            m_NetInputMatrixPtr;
            LatticePtr                                          m_NetLatticePtr;
            UidPtr                                              m_NetUidPtr;
            ExtrauttMapPtr                                      m_NetExtrauttMapPtr;
            // we remember the pointer to the learnable Nodes so that we can accumulate the gradient once a sub-minibatch is done 


            size_t                                              m_numParallelSequences; // number of paralle sequence in the cached matrix and MBLayout 
            size_t                                              m_numSubminibatches;    // how many subminibatches we are going to use ? 

            std::vector<shared_ptr<ComputationNode<ElemType>>>                 m_NetCriterionNodes;
            std::vector<shared_ptr<ComputationNode<ElemType>>>                 m_NetEvaluationNodes;
            std::map<wstring, shared_ptr<IStateFulNode>>                       m_NetStatefulNodes;      // we need to Export/Import states of stateful nodes when we swtich subminibatches 

        private:

            void EnumerateStatefulNodeWithRoot(ComputationNetwork& net, ComputationNodeBasePtr root, std::map<wstring, shared_ptr<IStateFulNode>>& statefulnode)
            {
                std::list<ComputationNodeBasePtr> evalorder = net.GetEvalOrder(root, false);
                for (auto& x : evalorder)
                {
                    wstring name = x->GetName();
                    if (statefulnode.find(name) != statefulnode.end()) continue; // already in the list 
                    shared_ptr<IStateFulNode> pNode = dynamic_pointer_cast<IStateFulNode>(x);
                    if (pNode)
                    {
                        statefulnode[name] = pNode;
                    }
                }
            }
            std::map<wstring, shared_ptr<IStateFulNode>> EnumerateStatefulNode(ComputationNetwork& net,
                const std::vector<ComputationNodeBasePtr>& criterionNode,
                const std::vector<ComputationNodeBasePtr>& evaluationNode)
            {
                std::map<wstring, shared_ptr<IStateFulNode>> statefulnodes;
                for (auto& root : criterionNode)
                {
                    EnumerateStatefulNodeWithRoot(net, root, statefulnodes);
                }
                for (auto& root : evaluationNode)
                {
                    EnumerateStatefulNodeWithRoot(net, root, statefulnodes);
                }
                return statefulnodes;
            }

        public:
            SubminibatchDispatcher() :
                m_MBLayoutCache(nullptr), m_LatticeCache(nullptr), m_uidCache(nullptr), m_extrauttmapCache(nullptr)
            { }

            void Init(ComputationNetworkPtr & net,
                const std::list<ComputationNodeBasePtr>& learnableNodes,
                const std::vector<ComputationNodeBasePtr>& criterionNodes,
                const std::vector<ComputationNodeBasePtr>& evaluationNodes)
            {
                m_MBLayoutCache = make_shared<MBLayout>();
                m_NetCriterionAccumulator = make_shared<Matrix<ElemType>>(1, 1, net->GetDeviceId());
                m_NetEvaluationAccumulator = make_shared<Matrix<ElemType>>(1, evaluationNodes.size(), net->GetDeviceId());
                // remember ptr to  learnableNode 
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
                m_NetCriterionAccumulator->SetValue((ElemType)0);
                m_NetEvaluationAccumulator->SetValue((ElemType)0);

                // emulate all the nodes, find nodes that have state 
                m_NetStatefulNodes = EnumerateStatefulNode(*net, criterionNodes, evaluationNodes);
                for (auto x : m_NetStatefulNodes)
                {
                    wstring name = x.first;
                    m_NetStates[name] = vector<shared_ptr<INodeState>>();
                }
            }

            ~SubminibatchDispatcher()
            {
                // TODO: remove these by using shared_ptr 
                delete m_LatticeCache;
                delete m_uidCache;
                delete m_extrauttmapCache;

                for (auto x : m_inputMatricesCache)
                {
                    delete x.second;
                }

                for (auto x : m_CachedGraident)
                {
                    delete x.second;
                }
            }
            size_t  GetMinibatchIntoCache(IDataReader<ElemType>& trainSetDataReader,
                ComputationNetwork& net,
                std::map<std::wstring, Matrix<ElemType>*> & inputMatrices,
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

                if (m_NetCriterionNodes[0] != nullptr && (m_NetCriterionNodes[0]->OperationName() == L"SequenceWithSoftmax"))
                {
                    // auto node = dynamic_pointer_cast<SequenceWithSoftmaxNode<ElemType>>(criterionNode);
                    NOT_IMPLEMENTED;
                    // TODO: implement this for Sequence training !!!
                }

                // subminibatches are cutted at the parallel sequence level; 
                // if #requested subminibatch is larger than #parallel sequence, 
                // we cannot split further; instead, each subsequence become a subminibatch 
                size_t actualnumSubminibatches = requestedSubminibatches > nParallelSequences ? nParallelSequences : requestedSubminibatches;

                // 3. third, allocate space for accumulated gradient 
                for (auto& n : m_LearnableNodePtr)
                {
                    auto node = n.second;
                    if (node->IsParameterUpdateRequired())
                    {
                        wstring nodeName = node->GetName();
                        shared_ptr<ComputationNode<ElemType>>  pLearnableNode = node;
                        auto funvalue = pLearnableNode->FunctionValues();   // gradient may not be allocated when this function is first called 
                        size_t nrow = funvalue.GetNumRows();
                        size_t ncol = funvalue.GetNumCols();
                        if (m_CachedGraident.find(nodeName) == m_CachedGraident.end())
                        {
                            // not allocated yet 
                            m_CachedGraident[nodeName] = new Matrix<ElemType>(nrow, ncol, funvalue.GetDeviceId());
                            m_CachedGraident[nodeName]->SetValue((ElemType)0);
                        }
                    }
                }
                // 4. for stateful node 
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

            void GetSubMinibatchToNet(size_t iSubminibatch)
            {
                Matrices decimatedMatrices;
                MBLayoutPtr decimatedLayout;
                DataReaderHelpers::DecimateMinibatch(m_inputMatricesCache, decimatedMatrices, m_MBLayoutCache, decimatedLayout, m_numSubminibatches, iSubminibatch);
                //  NOTE: decimatedMatrices must be released by caller

                //m_NetInputMatrixPtr = decimatedMatrices;
                for (auto& x : decimatedMatrices)
                {
                    wstring name = x.first;
                    m_NetInputMatrixPtr[name]->SetValue(*x.second);
                    delete x.second;    // TODO: is it safe to delete here ? Yes! SetValue call cuda memcpy so it is a blocking call  
                    x.second = nullptr;
                }

                m_NetMBLayoutPtr->CopyFrom(decimatedLayout);

                for (auto& x : m_NetStatefulNodes)
                {
                    wstring name = x.first;
                    shared_ptr<IStateFulNode>   pNode = x.second;
                    if (m_NetStates[name][iSubminibatch])
                        pNode->ImportState(m_NetStates[name][iSubminibatch]);
                }
            }
            // TODO: encapsulate it into a destructor !!!   Note: Cannot throw exceptions in destructor.
            void DoneWithCurrentSubMinibatch(size_t iSubminibatch)
            {
                // accumulate gradient here 
                for (auto x : m_CachedGraident)
                {
                    wstring nodename = x.first;
                    if (m_LearnableNodePtr.find(nodename) == m_LearnableNodePtr.end())
                    {
                        RuntimeError("ERROR: in DoneWithCurrentSubMinibatch: node %ls not found in LeanrableNode", nodename.c_str());
                    }
                    shared_ptr<ComputationNode<ElemType>> pNode = m_LearnableNodePtr[nodename];
                    m_CachedGraident[nodename]->operator+=(pNode->GradientValues());
                    pNode->GradientValues().SetValue((ElemType)0);
                }
                // accumulate criterion value 
                Matrix<ElemType>::AddElementToElement(
                    m_NetCriterionNodes[0]->FunctionValues(), 0, 0,
                    *m_NetCriterionAccumulator, 0, 0
                    );
                m_NetCriterionNodes[0]->FunctionValues().SetValue((ElemType)0);
                // accumulate evaluation value 
                for (size_t i = 0; i < m_NetEvaluationNodes.size(); i++)
                {
                    Matrix<ElemType>::AddElementToElement(
                        m_NetEvaluationNodes[i]->FunctionValues(), 0, 0,
                        *m_NetEvaluationAccumulator, 0, i
                        );
                    m_NetEvaluationNodes[i]->FunctionValues().SetValue((ElemType)0);
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
                for (auto& x : m_CachedGraident)
                {
                    wstring name = x.first;
                    Matrix<ElemType>* accumulategrad = x.second;

                    if (m_LearnableNodePtr.find(name) == m_LearnableNodePtr.end())
                    {
                        // should never happen, remove this code later
                        RuntimeError("ERROR: in DoneWithCurrentSubMinibatch: node %ls not found in LearnableNode", name.c_str());
                    }
                    m_LearnableNodePtr[name]->GradientValues().SetValue(*accumulategrad);
                    x.second->SetValue((ElemType)0);
                }
                // also revert net.m_MBLayoutPtr
                m_NetMBLayoutPtr->CopyFrom(m_MBLayoutCache);

                //m_NetCriterionNodes[0]->FunctionValues().SetValue((ElemType)0);
                Matrix<ElemType>::AddElementToElement(
                    *m_NetCriterionAccumulator, 0, 0,
                    m_NetCriterionNodes[0]->FunctionValues(), 0, 0
                    );
                m_NetCriterionAccumulator->SetValue((ElemType)0);

                for (size_t i = 0; i < m_NetEvaluationNodes.size(); i++)
                {
                    //m_NetEvaluationNodes[i]->FunctionValues().SetValue((ElemType)0);
                    Matrix<ElemType>::AddElementToElement(
                        *m_NetEvaluationAccumulator, 0, i,
                        m_NetEvaluationNodes[i]->FunctionValues(), 0, 0
                        );
                }
                m_NetEvaluationAccumulator->SetValue((ElemType)0);
            }
        };
    };
}}}
