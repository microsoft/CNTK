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
    // ErrorPredictionNode (label, prediction)   or ErrorPredictionNode (prediction, label)
    // -----------------------------------------------------------------------

    template<class ElemType>
    class ErrorPredictionNode : public ComputationNodeNonLooping/*ComputationNode*/<ElemType>
    {
        typedef ComputationNodeNonLooping<ElemType> Base; UsingComputationNodeMembersBoilerplate;
        static const std::wstring TypeName() { return L"ErrorPrediction"; }
    public:
        DeclareConstructorFromConfig(ErrorPredictionNode);
        ErrorPredictionNode(DEVICEID_TYPE deviceId, const wstring & name) :
            Base(deviceId, name)
        { }

        virtual void BackpropToNonLooping(size_t /*inputIndex*/) override
        {
            LogicError("%ls operation is used for evaluation only.", OperationName().c_str());
        }

        virtual void /*ComputationNodeNonLooping::*/ForwardPropNonLooping() override
        {
            FrameRange fr(Input(0)->GetMBLayout());
            Input(0)->ValueFor(fr).VectorMax(*m_maxIndexes0, *m_maxValues, true);
            Input(1)->ValueFor(fr).VectorMax(*m_maxIndexes1, *m_maxValues, true, m_topK);
            MaskMissingColumnsToZero(*m_maxIndexes0, Input(0)->GetMBLayout(), fr);
            MaskMissingColumnsToZero(*m_maxIndexes1, Input(1)->GetMBLayout(), fr);
            Value().AssignNumOfDiff(*m_maxIndexes0, *m_maxIndexes1, m_topK > 1);
        #if NANCHECK
            Value().HasNan("ErrorPrediction");
        #endif
#if DUMPOUTPUT
            Value().Print("ErrorPredictionNode");
#endif
        }

        virtual void /*ComputationNodeBase::*/Validate(bool isFinalValidationPass) override
        {
            ValidateBinaryReduce(isFinalValidationPass);

            m_topK = 1;
            // TODO: Make topK a constructor parameter
            if (m_inputs.size() == 3)
            {
                if (Input(2)->GetNumRows() != 1 || Input(2)->GetNumCols() != 1)
                    throw std::logic_error("TopK in ErrorPredictionNode must be a scalar value.");
                m_topK = static_cast<int>(Input(2)->Get00Element());
            }
        }

        virtual void UpdateFunctionMBSize() override
        {
            Base::UpdateFunctionMBSize();

            // resize the temporaries to their proper size
            size_t cols = Input(0)->GetNumCols();
            m_maxIndexes0->Resize(m_topK, cols);
            m_maxIndexes1->Resize(m_topK, cols);
            m_maxValues->Resize(m_topK, cols);
        }

        virtual void InferImageDimsFromInputs()
        {
            InferImageDimsFromInput(0, false);

            m_sampleLayout = TensorShape();
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
        virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
        {
            Base::RequestMatricesBeforeForwardProp(matrixPool);
            RequestMatrixFromPool(m_maxIndexes0, matrixPool);
            RequestMatrixFromPool(m_maxIndexes1, matrixPool);
            RequestMatrixFromPool(m_maxValues, matrixPool);
        }

        //release temp matrices that are only used by forward computation
        //don't release matrices that need to be used in the gradient computation
        virtual void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool)
        {
            Base::ReleaseMatricesAfterForwardProp(matrixPool);
            ReleaseMatrixToPool(m_maxIndexes0, matrixPool);
            ReleaseMatrixToPool(m_maxIndexes1, matrixPool);
            ReleaseMatrixToPool(m_maxValues, matrixPool);
        }

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
		DeclareConstructorFromConfig(PhoneErrorNode);
		PhoneErrorNode(DEVICEID_TYPE deviceId, const wstring & name)
			: Base(deviceId, name)
		{
		}

		virtual void BackpropToNonLooping(size_t /*inputIndex*/) override
		{
			LogicError("%ls operation is used for evaluation only.", OperationName().c_str());
		}


		virtual void ForwardPropNonLooping() override
		{
			size_t sequenceNum = Input(1)->GetNumParallelSequences();
			FrameRange frameRange(Input(0)->GetMBLayout());
			Input(0)->ValueFor(frameRange).VectorMax(*m_maxIndexes0, *m_maxValues, true);
			Input(1)->ValueFor(frameRange).VectorMax(*m_maxIndexes1, *m_maxValues, true);

			MaskMissingColumnsToZero(*m_maxIndexes0, Input(0)->GetMBLayout(), frameRange);
			MaskMissingColumnsToZero(*m_maxIndexes1, Input(1)->GetMBLayout(), frameRange);
			CalErrorphoneWER(Value(), *m_maxIndexes0, *m_maxIndexes1, sequenceNum, Input(0)->GetMBLayout(), Input(0)->Value().GetNumRows(), m_blanknum);
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
            ValidateBinaryReduce(isFinalValidationPass);

            m_topK = 1;
            // TODO: Make topK a constructor parameter
            if (m_inputs.size() == 3)
            {
                if (Input(2)->GetNumRows() != 1 || Input(2)->GetNumCols() != 1)
                    throw std::logic_error("TopK in ErrorPredictionNode must be a scalar value.");
                m_topK = static_cast<int>(Input(2)->Get00Element());
            }
        }

        virtual void UpdateFunctionMBSize() override
        {
            Base::UpdateFunctionMBSize();

            // resize the temporaries to their proper size
            size_t cols = Input(0)->GetNumCols();
            m_maxIndexes0->Resize(m_topK, cols);
            m_maxIndexes1->Resize(m_topK, cols);
            m_maxValues->Resize(m_topK, cols);
		}

		virtual void InferImageDimsFromInputs()
		{
			InferImageDimsFromInput(0, false);

			m_sampleLayout = TensorShape();
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
		virtual void RequestMatricesBeforeForwardProp(MatrixPool& matrixPool)
		{
			Base::RequestMatricesBeforeForwardProp(matrixPool);
			RequestMatrixFromPool(m_maxIndexes0, matrixPool);
			RequestMatrixFromPool(m_maxIndexes1, matrixPool);
			RequestMatrixFromPool(m_maxValues, matrixPool);
		}

		//release temp matrices that are only used by forward computation
		//don't release matrices that need to be used in the gradient computation
		virtual void ReleaseMatricesAfterForwardProp(MatrixPool& matrixPool)
		{
			Base::ReleaseMatricesAfterForwardProp(matrixPool);
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
						if (pMBLayout->IsEnd(nchannel, j))
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
                    if (lastsentend < mbsize)
                    {
                        FrameRange fr(pMBLayout, lastsentend);
                        if (pMBLayout->IsGap(fr.Sequence(nchannel)))
                            break;
                    }
				}
			}

			functionvalue(0, 0) = (ElemType)(wrongPhoneNum * totalframenum / totalphonenum);

		}

		shared_ptr<Matrix<ElemType>> m_maxIndexes0, m_maxIndexes1;
		shared_ptr<Matrix<ElemType>> m_maxValues;
		size_t m_blanknum;
        int m_topK;
	};

	template class PhoneErrorNode<float>;
	template class PhoneErrorNode<double>;

}}}
