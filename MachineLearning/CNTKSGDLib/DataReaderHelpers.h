// DataReaderHelper.h -- helper functions that understand both DataReader and ComputationNetwork

#pragma once

#include "Basics.h"
#include "DataReader.h"
#include "ComputationNetwork.h"
#include "MPIWrapper.h"
#include <string>
#include <map>
#include "TrainingCriterionNodes.h"

//#define SMB_DEBUG

namespace Microsoft { namespace MSR { namespace CNTK {




    /*static*/ struct DataReaderHelpers
    {
        // -------------------------------------------------------------------
        // DecimateMinibatch - decimate minibatch for parallelization
        // -------------------------------------------------------------------
        // non-inplace decimation , to be used in subminibatch implementation 
        template<class ElemType> 
        static void DecimateMinibatch(const std::map<std::wstring, Matrix<ElemType>*> MB,                               // input matrices 
                                      std::map<std::wstring, Matrix<ElemType>*>& decimatedMB,                            // output decimated matrices. 
                                                                                                                        // Caller need to release the memory themselves!!!   TODO: use shared_ptr 
                                      MBLayoutPtr pMBLayout,                                                            // input MBLayout 
                                      MBLayoutPtr& pDecimateMBLayout,                                                    // output decimated MBLayout 
                                      int numWorker, int rank)
        {
            size_t nSequence = pMBLayout->GetNumParallelSequences(); 
            size_t nT = pMBLayout->GetNumTimeSteps(); 

            // decide start column and end column 
            size_t st = nSequence * (size_t)rank / numWorker; 
            size_t en = nSequence * (size_t)(rank + 1) / numWorker; 
            en = en > nSequence ? nSequence : en; 
            en = (rank == numWorker - 1) ? nSequence : en;             
            size_t nNewParallelSequence = en - st;

            // warning if needed 
            static bool bWarned = false; 
            if (!bWarned && nSequence % numWorker != 0)
            {
                /* give a warning of potential bandwidth wasting */
                fprintf(stderr, "DecimateMinibatch: WARNING: Number of parallel utterances %d not a multiple of number of GPUs %d, GPU usage will be suboptimal.\n",
                        (int)nSequence, (int)numWorker);
                bWarned = true;
            }

            // begin decimate matrices 
            size_t rv = 0; 
            for (const auto& it : MB)
            {
                wstring name = it.first;
                MSR::CNTK::Matrix<ElemType> & mat = *it.second; 
                size_t nRows = mat.GetNumRows(); 
                size_t nCols = mat.GetNumCols();
                int devID = mat.GetDeviceId(); 

                if (rv == 0)
                    rv = nCols; 
                else if ( rv != nCols )
                    LogicError("DecimateMinibatch: Inconsistent number of columns among inputs (found %d and %d).", (int)rv, (int)nCols);

                if (nT != nCols / nSequence )
                    LogicError("ERROR: MBLayout borked, GetNumTimeSteps() mismatches minibatch number of columns\n");
                
                decimatedMB[name] = new Matrix<ElemType>(
                    mat.Reshaped(nRows*nSequence, nT).RowSlice( st*nRows , (en-st)*nRows).Reshaped(nRows, nNewParallelSequence*nT), 
                    devID
                ); 
                // NOTE: 
                // Reshaped return a matrix referencing the caller, so the moving constructor will not work here 
                // we need to use devID here to signify compiler that we are going to do a deep copy instead of moving constructor 
            }
            // decimate MBLayout as well 
            pDecimateMBLayout= make_shared<MBLayout>(nNewParallelSequence, nT, true);
            for (size_t t = 0; t < nT; t++) for (size_t id = 0; id < nNewParallelSequence; id++)
                pDecimateMBLayout->Set(id, t, pMBLayout->Get(id + st, t));
            
        }

        // Inpace decimation 
        // We sub-sample the parallel utterances.
        template<class ElemType> 
        static void DecimateMinibatch(std::map<std::wstring, Matrix<ElemType>*> &mb,    // matrix to be decimated
                                      int numprocs, int rank,                           // rank info
                                      MBLayoutPtr pMBLayout)                            // gets decimated as well
        {
            if (numprocs == 1)
                return;

            // For RNN, a input Matrix is organized in the following way: 
            //   | x_t^1  x_t^2 ... x_t^N |  .... | x_{t+T-1}^1 ... x_{t+T-1}^N | 
            //   |<----   block 1    ---->|  .... |<------  block T       ----->| 
            // N is the nSlice (input)
            // The decimation here is to split each block to individual GPUs 
            // So After decimation 
            //   | x_t^{st} ... x_t^{en-1}|  .... | x_{t+T-1}^{st} ... x_{t+T-1}^{en-1} | 
            // Each block now has nSlice/nProcs 
            // 
            // Correspondingly, the MBLayout will be revised 

            size_t nOrigParallelUtts = pMBLayout->GetNumParallelSequences();
            size_t T = pMBLayout->GetNumTimeSteps();

            // decide new parallel utterances
            size_t sent_start = nOrigParallelUtts * (size_t)rank / numprocs;
            size_t sent_end = nOrigParallelUtts * (size_t)(rank + 1) / numprocs;
            static bool warned = false;
            if (nOrigParallelUtts % numprocs != 0 && !warned)
            {
                /* give a warning of potential bandwidth wasting */
                fprintf(stderr, "DecimateMinibatch: WARNING: Number of parallel utterances %d not a multiple of number of GPUs %d, GPU usage will be suboptimal.\n",
                        (int)nOrigParallelUtts, (int)numprocs);
                warned = true;
            }
            size_t newNumParallelSequences = sent_end - sent_start;

            // decimate data
            size_t rv = 0;
            for (auto & it : mb)
            {
                MSR::CNTK::Matrix<ElemType> &mat = *it.second;
                size_t nCols = mat.GetNumCols();

                // assert the cols are even among nodes 
                if (rv == 0)
                    rv = nCols;
                else if (rv != nCols)
                    LogicError("DecimateMinibatch: Inconsistent number of columns among inputs (found %d and %d).", (int)rv, (int)nCols);

                if (T != nCols / nOrigParallelUtts)
                    LogicError("ERROR: MBLayout borked, GetNumTimeSteps() mismatches minibatch number of columns\n");
                if (T * nOrigParallelUtts != nCols) // (should really not happen)
                    LogicError("ERROR: minibatch size %d, but with %d parallel utterances --layout information borked\n", (int)nCols, (int)nOrigParallelUtts);

                if (sent_end == sent_start)
                {
                    // should never happen, print debug info
                    // BUGBUG: Yes, this can happen if we got less parallel sequences than GPUs. But users wouldn't want that, so we should fail.
                    // BUGBUG: This can also happen for a very small minibatch at the end of the epoch.
                    fprintf(stderr, "DecimateMinibatch: WARNING: col_st=col_en=%d, nCol=%d, nBlock=%d, nParaUtts=%d, nGPU=%d--This can happen if #parallel sequences < #GPUs (you'd be leaving a GPU unused)\n",
                            (int)sent_start, (int)nCols, (int)T, (int)nOrigParallelUtts, (int)numprocs);
                }

                // copy the respective columns
                // TODO: not efficient. Instead, use Reshape() and AssignRowSlice...()
                MSR::CNTK::Matrix<ElemType> tmp(mat.GetNumRows(), newNumParallelSequences*T, mat.GetPreferredDeviceId(), mat.GetMatrixType());
                for (size_t t = 0; t < T; t++)
                    tmp.SetColumnSlice(mat.ColumnSlice(t*nOrigParallelUtts + sent_start, newNumParallelSequences), t*newNumParallelSequences, newNumParallelSequences);
                mat.SetValue(tmp);      // update matrix in-place (new matrix has less parallel streams)
                // TODO: ^^ If had Matrix::RowSlice(), this would be simpler.
                //       TODO: But we do have a row-slice assignment function. This could be used.
            }
            // decimate layout
            auto pNewMBLayout = make_shared<MBLayout>(newNumParallelSequences, T, true);
            for (size_t t = 0; t < T; t++) for (size_t id = 0; id < newNumParallelSequences; id++)
                pNewMBLayout->Set(id, t, pMBLayout->Get(id + sent_start, t));
            pMBLayout->MoveFrom(pNewMBLayout);  // update layout in-place
        }

        // -------------------------------------------------------------------
        // GetMinibatchIntoNetwork() -- get one minibatch from Reader (this->trainSetDataReader) into Network (this->net)
        // Returns false if end of epoch has been reached.
        // If not, then actualMBSize is set. Note that 0 is a valid value to be returned for actualMBSize, caller must handle that correctly.
        // -------------------------------------------------------------------

        // Note: Later, a function like this will become part of the reader interface.
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
            // no data is read. When it does, 'wasDataRead' can be removed
            bool wasDataRead = trainSetDataReader.GetMinibatch(inputMatrices);      // fill in the minibatch data into the Input nodes' buffers directly
            // reader will have resized input node's m_functionValues directly. Nodes must be notified to do necessary internal state updates from that.
            net->NotifyInputNodesFunctionValuesMBSizeModified();
            size_t readMBSize = net->DetermineActualMBSizeFromFeatures();
            if (readMBSize == 0)
                wasDataRead = false;

            trainSetDataReader.CopyMBLayoutTo(pMBLayout);                           // and layout meta-data

            // verify some DataReader calls that are redundant since the MBLayout refactoring (keep verifying for a while for cosy feeling)
            net->VerifyActualNumParallelSequences(trainSetDataReader.GetNumParallelSequences()); // info already contained in MBLayout
            //assert(trainSetDataReader.RequireSentenceSeg() == pMBLayout->RequireSentenceSeg()); // this one is redundant, too

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
            // Note: Decimation may have reduced this to 0 frames.
            // TODO: This should be called/callable outside if 'wasDataRead' (GetMinibatch() should fill matrices to empty)
            // TODO: This will go away, as we will do resizing inside EvaluateThisNode(FrameRange()).
            actualMBSize = 0;
            if (wasDataRead)    // TODO: what if we call it always?
                actualMBSize = net->DetermineActualMBSizeFromFeatures(); // TODO: don't we know the size from reader? Should this be a check instead?

            return true;
        }
    };

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

        void EnumerateStatefulNodeWithRoot(ComputationNetwork& net, ComputationNodeBasePtr root,  std::map<wstring, shared_ptr<IStateFulNode>>& statefulnode)
        {
            std::list<ComputationNodeBasePtr> evalorder = net.GetEvalOrder(root, false); 
            for (auto& x : evalorder)
            {
                wstring name = x->GetName(); 
                if (statefulnode.find(name )!=statefulnode.end()) continue; // already in the list 
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
        SubminibatchDispatcher()
            : m_MBLayoutCache(nullptr), m_LatticeCache(nullptr), m_uidCache(nullptr), m_extrauttmapCache(nullptr)
        {
            
        }

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
        size_t  GetMinibatchIntoCache(   IDataReader<ElemType>& trainSetDataReader,
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
                Matrix<ElemType>* M= pa.second; 
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
            for (auto& n: m_LearnableNodePtr)
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
        // TODO: encapsulate it into a destructor !!! 
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
                    m_NetCriterionNodes[0]->FunctionValues() , 0, 0,
                    *m_NetCriterionAccumulator, 0, 0
                    ); 
            m_NetCriterionNodes[0]->FunctionValues().SetValue((ElemType)0);
            // accumulate evaluation value 
            for (size_t i = 0; i < m_NetEvaluationNodes.size(); i++)
            {
                Matrix<ElemType>::AddElementToElement( 
                    m_NetEvaluationNodes[i]->FunctionValues(), 0, 0,
                    *m_NetEvaluationAccumulator,  0, i
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
                    // should never happen , remove this code later
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

#ifdef SMB_DEBUG

        template<class Matrix, class ElemType>
        void WriteMatrix(const Matrix& mat, string filename)
        {
            ElemType* pArray = mat.CopyToArray();
            size_t nRows = mat.GetNumRows();
            size_t nCols = mat.GetNumCols();
            FILE* fp = fopenOrDie(filename, "w");
            for (size_t r = 0; r < nRows; r++)
            {
                for (size_t c = 0; c < nCols; c++)
                {
                    fprintf(fp, "%.9f ", pArray[nRows*c + r]);
                }
                fprintf(fp, "\n");
            }
            fcloseOrDie(fp);
            delete[]pArray;
        }
        void WriteMBLayout(MBLayoutPtr pMBLayout, wstring filename)
        {
            size_t nT = pMBLayout->GetNumTimeSteps();
            size_t nU = pMBLayout->GetNumParallelSequences();

            FILE* fp = fopenOrDie(filename, L"w");
            for (size_t u = 0; u < nU; u++)
            {
                for (size_t t = 0; t < nT; t++)
                {
                    MinibatchPackingFlags flag = pMBLayout->Get(u, t);
                    fprintf(fp, "%d\t", (int)flag);
                }
                fprintf(fp, "\n");
            }
            fcloseOrDie(fp);
        }
        void WriteInputMatriceAndMBLayout(size_t mbID, size_t smbID)
        {
            wstring node = L"features";
            wstring filename = msra::strfun::wstrprintf(L"tmp/%s.%d.%d", node.c_str(), mbID, smbID);
            if (m_NetInputMatrixPtr.find(node) != m_NetInputMatrixPtr.end())
            {
                WriteMatrix<Matrix<ElemType>, ElemType>(*m_NetInputMatrixPtr[node], msra::strfun::wcstombs(filename));
            }
            wstring fn = msra::strfun::wstrprintf(L"tmp/Layout.%d.%d", mbID, smbID);
            WriteMBLayout(m_NetMBLayoutPtr, fn);
        }
        void WriteInputMatriceAndMBLayout(Matrices m, MBLayoutPtr pMBLayout, size_t mbID)
        {
            wstring filename = msra::strfun::wstrprintf(L"tmp/features.%d", mbID);
            wstring fn       = msra::strfun::wstrprintf(L"tmp/layout.%d", mbID);
            if (m.find(L"features") != m.end())
            {
                WriteMatrix<Matrix<ElemType>, ElemType>(*m[L"features"], msra::strfun::wcstombs(filename));
            }
            WriteMBLayout(pMBLayout, fn);
        }

        void WriteGradient(size_t mbID)
        {
            wstring node = L"LSTMoutput1.bias";
            wstring filename = msra::strfun::wstrprintf(L"%s.%d", L"tmp/gradient", mbID);
            if (m_CachedGraident.find(node) != m_CachedGraident.end())
            {
                WriteMatrix<Matrix<ElemType>, ElemType>(*m_CachedGraident[node], msra::strfun::wcstombs(filename));
            }
        }

        void WriteGradient(const Matrix<ElemType>& mat, wstring fn)
        {
            WriteMatrix<Matrix<ElemType>, ElemType>(mat, msra::strfun::wcstombs(fn));
        }
#endif // SMB_DEBUG


    };

}}}
