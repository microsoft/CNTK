#pragma once

#include <Rpc.h>
#pragma comment(lib, "Rpcrt4.lib")

#include <unordered_set>
#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include <algorithm>
#include <WinBase.h>
#include <assert.h>
#include "basetypes.h"

#include <Matrix.h>
#include "PTaskGraphBuilder.h"
#include "ComputationNode.h"

#ifndef NOT_IMPLEMENTED
#define NOT_IMPLEMENTED throw std::exception("Not implemented")
#endif

/// author: Kaisheng Yao
/// kaisheny@microsoft.com

#pragma warning (disable: 4267)

//version number to control how to read and write 
#define CNTK_MODEL_VERSION_1 1
#define CURRENT_CNTK_MODEL_VERSION 1

namespace Microsoft {
	namespace MSR {
		namespace CNTK {

			/**
			* this is a common interface for segmental level evaluation
			* for example, forward backward algorithm
			*/
			template<class ElemType>
			class SegmentalEvaluateNode : public ComputationNode<ElemType>
			{
			private:
				Matrix<ElemType> mAlpha;
				Matrix<ElemType> mBeta;
				Matrix<ElemType> mPostProb;

                Matrix<ElemType> mLogSoftMaxOfRight;
                Matrix<ElemType> mSoftMaxOfRight;

			public:
				SegmentalEvaluateNode(const short deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
                    : ComputationNode(deviceId), mAlpha(deviceId), mBeta(deviceId), mPostProb(deviceId), mLogSoftMaxOfRight(deviceId), mSoftMaxOfRight(deviceId)
				{
					m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
					m_deviceId = deviceId;
					MoveMatricesToDevice(deviceId);
					InitRecurrentNode();
				}

				SegmentalEvaluateNode(File& fstream, const size_t modelVersion, const short deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
                    : ComputationNode(deviceId), mAlpha(deviceId), mBeta(deviceId), mPostProb(deviceId), mLogSoftMaxOfRight(deviceId), mSoftMaxOfRight(deviceId)
				{
					m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
					LoadFromFile(fstream, modelVersion, deviceId);
				}

				virtual const std::wstring OperationName() const { return TypeName(); }
				static const std::wstring TypeName() { return L"SegmentalEvaluateNode"; }

				/// compute posterior probability of label y at position t
				virtual void EvaluateThisNode()
				{
                    EvaluateThisNodeS(mPostProb, mAlpha, mBeta, FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), 
                        Inputs(2)->FunctionValues(), mLogSoftMaxOfRight, mSoftMaxOfRight);
				}

				virtual void EvaluateThisNode(const size_t timeIdxInSeq)
				{
					throw std::logic_error("SegmentalEvaluateNode node should never be in a loop.");
				}

				virtual void ComputeInputPartial(const size_t inputIndex)  //scaled by 2*number of colmns (samples) in the Matrix<ElemType>
				{
					if (inputIndex != 1 && inputIndex != 2)
						throw std::invalid_argument("SegmentalEvaluateNode only takes with respect to input and weight.");

					if (inputIndex == 1)
                        ErrorSignalToPostitionDependentNode(GradientValues(), Inputs(0)->FunctionValues(), 
                            mPostProb, Inputs(inputIndex)->GradientValues());
                    else
						return;
				}

				virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
				{
					throw std::logic_error("SegmentalEvaluateNode node should never be in a loop.");
				}

				static void ErrorSignalToPostitionDependentNode(const Matrix<ElemType> & thisGrd, const Matrix<ElemType>& labls, const Matrix<ElemType>& postProb, Matrix<ElemType>& inputGrd);

				/// compute forward backward algorithm
				static void EvaluateThisNodeS(Matrix<ElemType>& postprob, Matrix<ElemType>& alpha, Matrix<ElemType>& beta, Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues0, const Matrix<ElemType>& inputFunctionValues1, const Matrix<ElemType>& inputFunctionValues2, Matrix<ElemType>& logOfRight, Matrix<ElemType>& softmaxOfRight, const int shift=1);

				/// compute forward backward algorithm
				static void ForwardCompute(Matrix<ElemType>& alpha, const Matrix<ElemType>& beta, Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues0,
					const Matrix<ElemType>& inputFunctionValues1, const Matrix<ElemType>& inputFunctionValues2, const int shift = 1);

				/// compute backward algorithm
				static void BackwardCompute(const Matrix<ElemType>& alpha, Matrix<ElemType>& beta, Matrix<ElemType>& functionValues, const Matrix<ElemType>& inputFunctionValues0,
					const Matrix<ElemType>& inputFunctionValues1, const Matrix<ElemType>& inputFunctionValues2, const int shift = 1);

				/// compute forward backward algorithm
				static void PostProbCompute(Matrix<ElemType>& postprob, const Matrix<ElemType>& alpha, const Matrix<ElemType>& beta , const int shift = 1);

				virtual void Validate()
				{
					PrintSelfBeforeValidation();

					if (m_children.size() != 3)
						throw std::logic_error("SegmentalEvaluateNode requires three inputs.");

					if (!(Inputs(1)->FunctionValues().GetNumRows() == Inputs(2)->FunctionValues().GetNumRows() &&  // position dependent and pair scores have same number of labels
						Inputs(0)->FunctionValues().GetNumRows() == Inputs(1)->FunctionValues().GetNumRows() &&
						Inputs(0)->FunctionValues().GetNumCols() == Inputs(1)->FunctionValues().GetNumCols() && // position dependent and pair scores have the same observation numbers
						Inputs(2)->FunctionValues().GetNumCols() == Inputs(2)->FunctionValues().GetNumRows()))
					{
						throw std::logic_error("The Matrix<ElemType>  dimension in the SegmentalEvaluateNode operation does not match.");
					}

					FunctionValues().Resize(1, 1);
					CopyImageSizeFromInputs();
				}

				virtual void CopyImageSizeFromInputs()
				{
					CopyImageSizeFromInput(0, false);

					m_outputChannels = 1;
					m_outputWidth = 1;
					m_outputHeight = 1;
				}

				/// label : output label vector of [0:T-1]
				/// position_dependent_score : score from position dependent node,
				/// in the R-CRF case, it is the RNN output score before softmax
				/// transition score : score from the transition node, 
				/// in the R-CRF case, it is the transition probability between labels
				virtual void AttachInputs(const ComputationNodePtr label,
					const ComputationNodePtr position_dependent_score,
					const ComputationNodePtr transition_score)
				{
					m_children.resize(3);
					m_children[0] = label;
					m_children[1] = position_dependent_score;
					m_children[2] = transition_score;
				}

				virtual void MoveMatricesToDevice(const short deviceId)
				{
					ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);
				}

				virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring newName, const CopyNodeFlags flags) const
				{
					ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
					SegmentalEvaluateNode<ElemType>* node = (SegmentalEvaluateNode<ElemType>*) nodeP;

				}

				// copy constructor
				SegmentalEvaluateNode(const SegmentalEvaluateNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
					: ComputationNode(node->m_deviceId)
				{
					node->CopyTo(this, newName, flags);
				}

				virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
				{
					const std::wstring& name = (newName == L"") ? NodeName() : newName;

					ComputationNodePtr node = new SegmentalEvaluateNode<ElemType>(this, name, flags);
					return node;
				}

			};

			/**
			* this node does sequence decoding only
            * it corresponds to a decoder
			*/
            template<class ElemType>
            class SegmentalDecodeNode : public ComputationNode<ElemType>
            {
            private:
                Matrix<ElemType> mAlpha;
                Matrix<ElemType> mBacktrace;

            public:
                SegmentalDecodeNode(const short deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
                    : ComputationNode(deviceId), mAlpha(deviceId), mBacktrace(deviceId)
                {
                    m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
                    m_deviceId = deviceId;
                    MoveMatricesToDevice(deviceId);
                    InitRecurrentNode();
                }

                SegmentalDecodeNode(File& fstream, const size_t modelVersion, const short deviceId = AUTOPLACEMATRIX, const std::wstring name = L"")
                    : ComputationNode(deviceId), mAlpha(deviceId), mBacktrace(deviceId)
                {
                    m_nodeName = (name == L"" ? CreateUniqNodeName() : name);
                    LoadFromFile(fstream, modelVersion, deviceId);
                }

                virtual const std::wstring OperationName() const { return TypeName(); }
                static const std::wstring TypeName() { return L"SegmentalDecodeNode"; }

                /// compute posterior probability of label y at position t
                virtual void EvaluateThisNode()
                {
                    EvaluateThisNodeS(mAlpha, mBacktrace, FunctionValues(), Inputs(1)->FunctionValues(),
                        Inputs(2)->FunctionValues());
                }

                virtual void EvaluateThisNode(const size_t timeIdxInSeq)
                {
                    throw std::logic_error("SegmentalDecodeNode node should never be in a loop.");
                }

                virtual void ComputeInputPartial(const size_t inputIndex)  //scaled by 2*number of colmns (samples) in the Matrix<ElemType>
                {
                    throw std::invalid_argument("SegmentalDecodeNode doesn't compute derivatives"); 
                }

                virtual void ComputeInputPartial(const size_t inputIndex, const size_t timeIdxInSeq)
                {
                    throw std::logic_error("SegmentalEvaluateNode node should never be in a loop.");
                }

                /// compute forward backward algorithm
                static void EvaluateThisNodeS(Matrix<ElemType>& alpha, Matrix<ElemType>& backtrace, Matrix<ElemType>& decodepath, const Matrix<ElemType>& inputFunctionValues1, const Matrix<ElemType>& inputFunctionValues2, const int shift = 1);

                /// compute forward backward algorithm
                static void ForwardCompute(Matrix<ElemType>& alpha, 
                    Matrix<ElemType>& backtrace,
                    const Matrix<ElemType>& inputFunctionValues1, const Matrix<ElemType>& inputFunctionValues2, const int shift = 1);

                /// compute backward algorithm
                static void BackwardCompute(const Matrix<ElemType>& alpha, Matrix<ElemType>& decodepath, const Matrix<ElemType>& backtrace, 
                    const int shift = 1);

                virtual void Validate()
                {
                    PrintSelfBeforeValidation();

                    if (m_children.size() != 3)
                        throw std::logic_error("SegmentalDecodeNode requires three inputs.");

                    if (!(Inputs(1)->FunctionValues().GetNumRows() == Inputs(2)->FunctionValues().GetNumRows() &&  // position dependent and pair scores have same number of labels
                        Inputs(2)->FunctionValues().GetNumCols() == Inputs(2)->FunctionValues().GetNumRows()))
                    {
                        throw std::logic_error("The Matrix<ElemType>  dimension in the SegmentalDecodeNode operation does not match.");
                    }

                    CopyImageSizeFromInputs();
                }

                virtual void CopyImageSizeFromInputs()
                {
                    CopyImageSizeFromInput(0, false);

                    m_outputChannels = 1;
                    m_outputWidth = 1;
                    m_outputHeight = 1;
                }

                /// label : output label vector of [0:T-1]
                /// position_dependent_score : score from position dependent node,
                /// in the R-CRF case, it is the RNN output score before softmax
                /// transition score : score from the transition node, 
                /// in the R-CRF case, it is the transition probability between labels
                virtual void AttachInputs(const ComputationNodePtr label,
                    const ComputationNodePtr position_dependent_score,
                    const ComputationNodePtr transition_score)
                {
                    m_children.resize(3);
                    m_children[0] = label;
                    m_children[1] = position_dependent_score;
                    m_children[2] = transition_score;
                }

                virtual void MoveMatricesToDevice(const short deviceId)
                {
                    ComputationNode<ElemType>::MoveMatricesToDevice(deviceId);
                }

                virtual void CopyTo(const ComputationNodePtr nodeP, const std::wstring newName, const CopyNodeFlags flags) const
                {
                    ComputationNode<ElemType>::CopyTo(nodeP, newName, flags);
                    SegmentalDecodeNode<ElemType>* node = (SegmentalDecodeNode<ElemType>*) nodeP;

                }

                // copy constructor
                SegmentalDecodeNode(const SegmentalDecodeNode<ElemType>* node, const std::wstring& newName, const CopyNodeFlags flags)
                    : ComputationNode(node->m_deviceId)
                {
                    node->CopyTo(this, newName, flags);
                }

                virtual ComputationNodePtr Duplicate(const std::wstring& newName, const CopyNodeFlags flags) const
                {
                    const std::wstring& name = (newName == L"") ? NodeName() : newName;

                    ComputationNodePtr node = new SegmentalDecodeNode<ElemType>(this, name, flags);
                    return node;
                }

            };
            
		}
	}
}
