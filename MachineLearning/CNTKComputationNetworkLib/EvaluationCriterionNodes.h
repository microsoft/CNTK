//
// <copyright file="EvaluationCriterionNodes.h" company="Microsoft">
//     Copyright (c) Microsoft Corporation.  All rights reserved.
// </copyright>
//
#pragma once

#include <map>
#include <string>
#include <vector>
#include <stdexcept>
#include <list>
#include <memory>
#include "ComputationNode.h"

namespace Microsoft { namespace MSR { namespace CNTK {
    //note: to save computation the gradient may be scaled by an constant. 

    // -----------------------------------------------------------------------
    // ErrorPredictionNode (label, prediction)    --TODO: is that correct?
    // -----------------------------------------------------------------------

    template<class ElemType>
    class ErrorPredictionNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>
    {
        typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"ErrorPrediction"; }
    public:
        ErrorPredictionNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void ComputeInputPartialNonLooping(size_t /*inputIndex*/) override
        {
            LogicError("%ls operation is used for evaluation only.", OperationName().c_str());
        }

        virtual void /*ComputationNodeNonLooping::*/EvaluateThisNodeNonLooping() override
        {
            FrameRange frameRange(Inputs(0)->GetMBLayout());
            Inputs(0)->ValueSlice(frameRange).VectorMax(*m_maxIndexes0, *m_maxValues, true);
            Inputs(1)->ValueSlice(frameRange).VectorMax(*m_maxIndexes1, *m_maxValues, true, m_topK);
            MaskMissingColumnsToZero(*m_maxIndexes0, Inputs(0)->GetMBLayout(), frameRange);
            MaskMissingColumnsToZero(*m_maxIndexes1, Inputs(1)->GetMBLayout(), frameRange);
            FunctionValues().AssignNumOfDiff(*m_maxIndexes0, *m_maxIndexes1, m_topK > 1);
        #if NANCHECK
            FunctionValues().HasNan("ErrorPrediction");
        #endif
#if DUMPOUTPUT
            FunctionValues().Print("ErrorPredictionNode");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
#if 1
            ValidateBinaryReduce(isFinalValidationPass);
#else
            Base::Validate(isFinalValidationPass);

            ValidateInferBinaryChildrenDims();

            if (isFinalValidationPass)
            {
                if (!(
                      Inputs(0)->GetNumRows() == Inputs(1)->GetNumRows() &&  //match size
                      (Inputs(0)->HasMBLayout() || Inputs(0)->GetNumCols() == Inputs(1)->GetNumCols())
                     ))
                {
                    LogicError("The Matrix dimension in the ErrorPrediction operation does not match.");
                }
            }
            Resize(1,1);
            m_pMBLayout = nullptr;    // this node does not hold mini-batch data
            InferImageDimsFromInputs(); 
#endif

            m_topK = 1;
            // TODO: Make topK a constructor parameter
            if (m_children.size() == 3)
            {
                if (Inputs(2)->GetNumRows() != 1 || Inputs(2)->GetNumCols() != 1)
                    throw std::logic_error("TopK in ErrorPredictionNode must be a scalar value.");
                m_topK = static_cast<int>(Inputs(2)->Get00Element());
            }

            // resize the temporaries to their proper size
            size_t cols = Inputs(0)->GetNumCols();
            m_maxIndexes0->Resize(m_topK, cols);
            m_maxIndexes1->Resize(m_topK, cols);
            m_maxValues->Resize(m_topK, cols);
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            m_outputImageLayout = ImageLayout();
        }

        virtual void CopyTo(ComputationNodeBasePtr nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
        {
            Base::CopyTo(nodeP, newName, flags);
            if (flags & CopyNodeFlags::copyNodeValue)
            {
                auto node = dynamic_pointer_cast<ErrorPredictionNode<ElemType>>(nodeP);
                *node->m_maxIndexes0 = *m_maxIndexes0;
                *node->m_maxIndexes1 = *m_maxIndexes1;
                *node->m_maxValues = *m_maxValues;
            }
        }
        //request matrices needed to do node function value evaluation
        virtual void RequestMatricesBeforeEval(MatrixPool& matrixPool)
        {
            Base::RequestMatricesBeforeEval(matrixPool);
            RequestMatrixFromPool(m_maxIndexes0, matrixPool);
            RequestMatrixFromPool(m_maxIndexes1, matrixPool);
            RequestMatrixFromPool(m_maxValues, matrixPool);
        }

        //release temp matrices that are only used by forward computation
        //don't release matrices that need to be used in the gradient computation
        virtual void ReleaseMatricesAfterEval(MatrixPool& matrixPool)
        {
            Base::ReleaseMatricesAfterEval(matrixPool);
            ReleaseMatrixToPool(m_maxIndexes0, matrixPool);
            ReleaseMatrixToPool(m_maxIndexes1, matrixPool);
            ReleaseMatrixToPool(m_maxValues, matrixPool);
        }
protected:
        virtual bool NodeDoesItsOwnCustomizedMissingColumnsMasking() { return true; }

    private:
        shared_ptr<Matrix<ElemType>> m_maxIndexes0, m_maxIndexes1;
        shared_ptr<Matrix<ElemType>> m_maxValues;
        int m_topK;
    };

    template class ErrorPredictionNode<float>; 
    template class ErrorPredictionNode<double>;

	template<class ElemType>
	class PhoneErrorNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>
	{
		typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
		static const std::wstring TypeName() { return L"PhoneError"; }
	public:
		PhoneErrorNode(DEVICEID_TYPE deviceId, const wstring & name)
			: Base(deviceId, name)
		{
		}

		virtual void ComputeInputPartialNonLooping(size_t /*inputIndex*/) override
		{
			LogicError("%ls operation is used for evaluation only.", OperationName().c_str());
		}


		virtual void EvaluateThisNodeNonLooping() override
		{
			size_t sequenceNum = Inputs(1)->GetNumParallelSequences();
			FrameRange frameRange(Inputs(0)->GetMBLayout());
			Inputs(0)->ValueSlice(frameRange).VectorMax(*m_maxIndexes0, *m_maxValues, true);
			Inputs(1)->ValueSlice(frameRange).VectorMax(*m_maxIndexes1, *m_maxValues, true);

			MaskMissingColumnsToZero(*m_maxIndexes0, Inputs(0)->GetMBLayout(), frameRange);
			MaskMissingColumnsToZero(*m_maxIndexes1, Inputs(1)->GetMBLayout(), frameRange);
			CalErrorphoneWER(FunctionValues(), *m_maxIndexes0, *m_maxIndexes1, sequenceNum, Inputs(0)->GetMBLayout(), Inputs(0)->FunctionValues().GetNumRows(), m_blanknum);
#if NANCHECK
			FunctionValues().HasNan("ErrorPrediction");
#endif
#if DUMPOUTPUT
			FunctionValues().Print("ErrorPredictionNode");
#endif


			/*EvaluateThisNodeS(FunctionValues(), Inputs(0)->FunctionValues(), Inputs(1)->FunctionValues(), m_maxIndexes0, m_maxIndexes1, m_maxValues, shared_from_this(),
			Inputs(0)->GetMBLayout(), sequenceNum, m_blanknum);*/
		}





		virtual void Validate(bool isFinalValidationPass) override
		{
#if 1
			ValidateBinaryReduce(isFinalValidationPass);
#else

			Base::Validate(isFinalValidationPass);

			ValidateInferBinaryChildrenDims();



			if (isFinalValidationPass)
			{
				if (!(
					Inputs(0)->GetNumRows() == Inputs(1)->GetNumRows() &&  //match size
					(Inputs(0)->HasMBLayout() || Inputs(0)->GetNumCols() == Inputs(1)->GetNumCols())
					))
				{
					LogicError("The Matrix dimension in the ErrorPrediction operation does not match.");
				}

			}

			Resize(1, 1);
			m_pMBLayout = nullptr;    // this node does not hold mini-batch data
			InferImageDimsFromInputs();
#endif

			// resize the temporaries to their proper size
			size_t cols = Inputs(0)->GetNumCols();
			m_maxIndexes0->Resize(1, cols);
			m_maxIndexes1->Resize(1, cols);
			m_maxValues->Resize(1, cols);
		}

		virtual void InferImageDimsFromInputs()
		{
			InferImageDimsFromInput(0, false);

			m_outputImageLayout = ImageLayout();
		}





		virtual void CopyTo(ComputationNodeBasePtr  nodeP, const std::wstring& newName, const CopyNodeFlags flags) const override
		{
			Base::CopyTo(nodeP, newName, flags);


			if (flags & CopyNodeFlags::copyNodeValue)
			{
				auto node = dynamic_pointer_cast<PhoneErrorNode<ElemType>>(nodeP);
				node->m_maxIndexes0 = m_maxIndexes0;
				node->m_maxIndexes1 = m_maxIndexes1;
				node->m_maxValues = m_maxValues;
				node->m_blanknum = m_blanknum;
			}
		}

		//request matrices needed to do node function value evaluation
		virtual void RequestMatricesBeforeEval(MatrixPool& matrixPool)
		{
			Base::RequestMatricesBeforeEval(matrixPool);
			RequestMatrixFromPool(m_maxIndexes0, matrixPool);
			RequestMatrixFromPool(m_maxIndexes1, matrixPool);
			RequestMatrixFromPool(m_maxValues, matrixPool);
		}

		//release temp matrices that are only used by forward computation
		//don't release matrices that need to be used in the gradient computation
		virtual void ReleaseMatricesAfterEval(MatrixPool& matrixPool)
		{
			Base::ReleaseMatricesAfterEval(matrixPool);
			ReleaseMatrixToPool(m_maxIndexes0, matrixPool);
			ReleaseMatrixToPool(m_maxIndexes1, matrixPool);
			ReleaseMatrixToPool(m_maxValues, matrixPool);
		}


		void SetBlankNum(const size_t blanknum)
		{
			m_blanknum = blanknum;
		}
	protected:
		virtual bool NodeDoesItsOwnCustomizedMissingColumnsMasking() { return true; }

	private:
		static void CalErrorphoneWER(Matrix<ElemType>& functionvalue, Matrix<ElemType>& refframeseq, Matrix<ElemType> & outputframeseq, size_t samplesInRecurrentStep, MBLayoutPtr pMBLayout,
			size_t allphonenum, size_t blanknum)
		{
			std::vector<size_t> labelseq, outputseq;
			static const int subPen = 10;     /* error penalties */
			static const int delPen = 7;
			static const int insPen = 7;

			Matrix<float> grid;
			Matrix<float> insmatrix;
			Matrix<float> delmatrix;
			Matrix<float> submatrix;

			float d, h, v;
			//size_t fullcolsize = refframeseq.GetNumCols() / samplesInRecurrentStep;
			ElemType wrongPhoneNum = 0.0;
			size_t totalphonenum = 0, totalframenum = 0;
			size_t mbsize = refframeseq.GetNumCols() / samplesInRecurrentStep;
			size_t lastsentend = 0;
			size_t FrameNum = 0;
			size_t j = 0;
			for (size_t nchannel = 0; nchannel < samplesInRecurrentStep; nchannel++)
			{
				//zhaorui for CTC merge
				lastsentend = 0;
				j = 0;
				while (j < mbsize)
				{
					FrameNum = 0;
					for (j = lastsentend; j < mbsize; j++)
					{
						if (pMBLayout->Is(nchannel, j, MinibatchPackingFlags::SequenceEnd))
						{
							FrameNum = j - lastsentend + 1;
							break;
							//validframes.push_back(j + 1);								
						}
					}
					if (FrameNum > 0)
					{
						totalframenum += FrameNum;
						size_t lastid = 65535;
						labelseq.clear();
						//merge same phone id
						for (size_t i = lastsentend; i < FrameNum + lastsentend; i++)
						{
							if (lastid != refframeseq(0, i * samplesInRecurrentStep + nchannel))
							{
								lastid = (size_t)refframeseq(0, i * samplesInRecurrentStep + nchannel);
								labelseq.push_back(lastid);
							}
							/*lastid = (size_t)refframeseq(0, i * samplesInRecurrentStep + nchannel);

							if (lastid != 65535)
							labelseq.push_back(lastid);*/
						}
						outputseq.clear();
						lastid = 65535;
						for (size_t i = lastsentend; i < FrameNum + lastsentend; i++)
						{
							if (lastid != outputframeseq(0, i * samplesInRecurrentStep + nchannel))
							{
								lastid = (size_t)outputframeseq(0, i * samplesInRecurrentStep + nchannel);
								if (lastid < allphonenum - blanknum)  //hard code
									outputseq.push_back(lastid);
							}
						}

						//calcualate phone error rate		
						size_t labelsize = labelseq.size();
						totalphonenum += labelsize;
						size_t outputsize = outputseq.size();
						grid.Resize(labelsize + 1, outputsize + 1);
						insmatrix.Resize(labelsize + 1, outputsize + 1);
						delmatrix.Resize(labelsize + 1, outputsize + 1);
						submatrix.Resize(labelsize + 1, outputsize + 1);
						insmatrix.SetValue(0.0f);
						delmatrix.SetValue(0.0f);
						submatrix.SetValue(0.0f);


						for (size_t i = 0; i < labelsize + 1; i++){
							grid(i, 0) = (float)(i * delPen);
							delmatrix(i, 0) = (float)i;
						}
						fprintf(stderr, "label: ");
						for (size_t i = 0; i < labelsize; i++){
							fprintf(stderr, "%d\t", labelseq[i]);
						}
						fprintf(stderr, "\n");

						fprintf(stderr, "output: ");
						for (size_t i = 0; i < outputsize; i++){
							fprintf(stderr, "%d\t", outputseq[i]);
						}
						fprintf(stderr, "\n");

						for (size_t j = 0; j < outputsize + 1; j++) {
							grid(0, j) = (float)(j * insPen);
							insmatrix(0, j) = (float)j;
						}
						for (size_t i = 1; i < labelsize + 1; i++){
							for (size_t j = 1; j < outputsize + 1; j++) {

								if (labelseq[i - 1] == outputseq[j - 1])
								{
									grid(i, j) = grid(i - 1, j - 1);
									insmatrix(i, j) = insmatrix(i - 1, j - 1);
									delmatrix(i, j) = delmatrix(i - 1, j - 1);
									submatrix(i, j) = submatrix(i - 1, j - 1);

								}
								else
								{
									d = grid(i - 1, j) + (float)delPen; //deletion 
									h = grid(i, j - 1) + (float)insPen;  //insertion
									v = grid(i - 1, j - 1) + (float)subPen; //substitution 
									if (v <= d && v <= h)
									{
										insmatrix(i, j) = insmatrix(i - 1, j - 1);
										delmatrix(i, j) = delmatrix(i - 1, j - 1);

										submatrix(i, j) = submatrix(i - 1, j - 1) + 1.0f;
										grid(i, j) = v;

									}
									else if (d < h)
									{
										insmatrix(i, j) = insmatrix(i - 1, j);
										submatrix(i, j) = submatrix(i - 1, j);
										delmatrix(i, j) = delmatrix(i - 1, j) + 1.0f;
										grid(i, j) = d;
									}
									else
									{
										delmatrix(i, j) = delmatrix(i, j - 1);
										submatrix(i, j) = submatrix(i, j - 1);
										insmatrix(i, j) = insmatrix(i, j - 1) + 1.0f;
										grid(i, j) = h;
									}
								}
							}
						}
						wrongPhoneNum += insmatrix(labelsize, outputsize) + delmatrix(labelsize, outputsize) + submatrix(labelsize, outputsize);
					}
					lastsentend += FrameNum;
				}
			}

			functionvalue(0, 0) = (ElemType)(wrongPhoneNum * totalframenum / totalphonenum);

		}

		shared_ptr<Matrix<ElemType>> m_maxIndexes0, m_maxIndexes1;
		shared_ptr<Matrix<ElemType>> m_maxValues;
		size_t m_blanknum;
	};

	template class PhoneErrorNode<float>;
	template class PhoneErrorNode<double>;

}}}