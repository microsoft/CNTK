#include "ComputationNode.h"
#include "SimpleEvaluator.h"
#include "IComputationNetBuilder.h"
#include "SGD.h"
#include "SegmentalComputationNode.h"

namespace Microsoft {
	namespace MSR {
		namespace CNTK {

            /// pair_scores assumes (i,j) means transition from j to i
			template<class ElemType>
			void SegmentalEvaluateNode<ElemType>::EvaluateThisNodeS(Matrix<ElemType>& postprob, Matrix<ElemType>& alpha, Matrix<ElemType>& beta, Matrix<ElemType>& functionValues, const Matrix<ElemType>& lbls, const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, Matrix<ElemType>& logOfRight, Matrix<ElemType>& softmaxOfRight, const int iStep)
			{
                /// to-do, each slice is for one sentence
                /// to-do, number of slices correspond to number of frames 
                /// this implementation only supports one sentence per minibatch
                
                int nObs = lbls.GetNumCols();

                /// change to other values so can support multiple sentences in each minibatch
                assert(iStep == 1);
				ForwardCompute(alpha, beta, functionValues, lbls, pos_scores, pair_scores, iStep);
				BackwardCompute(alpha, beta, functionValues, lbls, pos_scores, pair_scores, iStep);
				PostProbCompute(postprob, alpha, beta, iStep);

                int firstLbl = -1;
                for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                if (lbls(ik, 0) != 0){
                    firstLbl = ik; break;
                }

                int lastLbl = -1;
                for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                if (lbls(ik, nObs - 1) != 0){
                    lastLbl = ik; break;
                }

                ElemType fAlpha = alpha(lastLbl, nObs - 1);
                ElemType fBeta = beta(firstLbl, 0);

                assert(logOfRight.IsClose(fBeta, fAlpha, 5));

                logOfRight.SetValue(pos_scores);
                functionValues.AssignInnerProductOfMatrices(lbls, logOfRight);

                /// transition score
                ElemType tscore = 0;
                for (int t = 0; t < nObs-1; t++){
                    int i = -1;
                    for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                    if (lbls(ik, t) != 0){
                        i = ik; break;
                    }
                    int j = -1;
                    for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                    if (lbls(ik, t+1) != 0){
                        j = ik; break;
                    }
                    tscore += pair_scores(j, i);
                }

                tscore += functionValues.Get00Element();
                tscore -= fAlpha; 
                functionValues.SetValue(tscore);

                functionValues *= (-1);
            };

			template<class ElemType>
			void SegmentalEvaluateNode<ElemType>::ErrorSignalToPostitionDependentNode(const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& labls, const Matrix<ElemType>& postProb, Matrix<ElemType>& grd)
			{
				Matrix<ElemType>::AddScaledDifference(gradientValues, postProb, labls, grd);
			}

            /**
            postprob: linear scale posterior probability
            */
			template<class ElemType>
			void SegmentalEvaluateNode<ElemType>::PostProbCompute(Matrix<ElemType>& postprob, const Matrix<ElemType>& alpha, 
                const Matrix<ElemType>& beta , const int shift)
			{
                assert(shift == 1);
				int iNumPos = alpha.GetNumCols();
				int iNumLab = alpha.GetNumRows();
                ElemType fSumOld = LZERO;

				postprob.Resize(iNumLab, iNumPos);
				for (int t = 0; t < iNumPos; t++)
				{
					ElemType fSum = LZERO;
					for (int k = 0; k < iNumLab; k++)
						fSum = postprob.LogAdd(fSum, alpha(k, t) + beta(k, t));
                    if (t > 0)
                    {
                        assert(postprob.IsClose(fSumOld , fSum, 5));
                    }
                    fSumOld = fSum;

                    ElemType fAcc = 0.0;
                    for (int k = 0; k < iNumLab; k++)
					{
						ElemType fTmp = alpha(k, t) + beta(k, t);
						fTmp -= fSum;
                        fTmp = exp(fTmp);
                        postprob(k, t) = fTmp;
                        fAcc += fTmp;
					}
                    assert(postprob.IsClose(1.0, fAcc, 2));
				}
			};

			template<class ElemType>
			void SegmentalEvaluateNode<ElemType>::ForwardCompute(Matrix<ElemType>& alpha, const Matrix<ElemType>& beta, 
                Matrix<ElemType>& functionValues,
                const Matrix<ElemType>& lbls,
                const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, const int shift)
			{
                /// to-do, shift more than 1 to support muliple sentences per minibatch
                assert(shift == 1);
				int iNumPos = lbls.GetNumCols();
				int iNumLab = lbls.GetNumRows();

                int firstLbl= -1;
                for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                if (lbls(ik, 0) != 0){
                    firstLbl = ik; break;
                }
                
                alpha.Resize(iNumLab, iNumPos);
                alpha.SetValue(LZERO);
                alpha.SetValue(firstLbl, 0, 0.0);

 				for (int t = 1; t < iNumPos; t++)
				{
					for (int k = 0; k < iNumLab; k++)
					{
						ElemType fTmp = LZERO;
                        for (int j = 0; j < iNumLab; j++)
                        {
                            ElemType fAlpha = alpha(j, t - 1);
                            fTmp = alpha.LogAdd(fTmp, fAlpha + pair_scores(k, j));
                        }
                        fTmp += pos_scores(k, t);
						alpha(k, t) = fTmp;
					}
				}

			};

			template<class ElemType>
			void SegmentalEvaluateNode<ElemType>::BackwardCompute(const Matrix<ElemType>& alpha, Matrix<ElemType>& beta, 
                Matrix<ElemType>& functionValues, const Matrix<ElemType>& lbls,
				const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, const int shift)
			{
                assert(shift == 1);
				int iNumPos = lbls.GetNumCols();
				int iNumLab = lbls.GetNumRows();

                int lastLbl = -1;
                for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                if (lbls(ik, iNumPos - 1) != 0){
                    lastLbl = ik; break;
                }

                beta.Resize(iNumLab, iNumPos);
                beta.SetValue(LZERO);
                beta.SetValue(lastLbl, iNumPos - 1, 0.0);

				for (int t = iNumPos - 2; t >= 0; t--)
				{
					for (int k = 0; k < iNumLab; k++)
					{
						ElemType fTmp = LZERO;
						for (int j = 0; j < iNumLab; j++)
						{
                            ElemType fBeta = beta(j, t + 1);
							fTmp = beta.LogAdd(fTmp, fBeta + pos_scores(j, t + 1) + pair_scores(j, k));
						}
						beta(k, t) = fTmp;
					}
				}

			};

            /// pair_scores assumes (i,j) means transition from j to i
            template<class ElemType>
            void SegmentalDecodeNode<ElemType>::EvaluateThisNodeS(Matrix<ElemType>& alpha, Matrix<ElemType>& backtrace, Matrix<ElemType>& functionValues, const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, const int iStep)
            {
                /// to-do, each slice is for one sentence
                /// to-do, number of slices correspond to number of frames 
                /// this implementation only supports one sentence per minibatch

                int nObs = pos_scores.GetNumCols();

                /// change to other values so can support multiple sentences in each minibatch
                assert(iStep == 1);
                ForwardCompute(alpha, backtrace, pos_scores, pair_scores, iStep);
                BackwardCompute(alpha, functionValues, backtrace, iStep);

            };

            template<class ElemType>
            void SegmentalDecodeNode<ElemType>::ForwardCompute(Matrix<ElemType>& alpha, 
                Matrix<ElemType>& backtrace,
                const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, const int shift)
            {
                /// to-do, shift more than 1 to support muliple sentences per minibatch
                assert(shift == 1);
                int iNumPos = pos_scores.GetNumCols();
                int iNumLab = pos_scores.GetNumRows();

                alpha.Resize(iNumLab, iNumPos);
                backtrace.Resize(iNumLab, iNumPos);
                for (int k = 0; k < iNumLab; k++)
                    backtrace(k, 0) = k;

                for (int t = 1; t < iNumPos; t++)
                {
                    for (int k = 0; k < iNumLab; k++)
                    {
                        ElemType fTmp = LZERO;
                        size_t   iTmp = -1;
                        for (int j = 0; j < iNumLab; j++)
                        {
                            ElemType fAlpha = alpha(j, t - 1) + pair_scores(k, j);
                            if (fAlpha > fTmp){
                                fTmp = fAlpha; 
                                iTmp = j; 
                            }
                        }
                        fTmp += pos_scores(k, t);
                        alpha(k, t) = fTmp;
                        backtrace(k, t) = (ElemType)iTmp;
                    }
                }

            };

            template<class ElemType>
            void SegmentalDecodeNode<ElemType>::BackwardCompute(const Matrix<ElemType>& alpha, Matrix<ElemType>& decodedpath,
                const Matrix<ElemType>& backtrace, const int shift)
            {
                assert(shift == 1);
                int iNumPos = backtrace.GetNumCols();
                int iNumLab = backtrace.GetNumRows();

                decodedpath.Resize(iNumLab, iNumPos);

                ElemType ftmp = LZERO;
                size_t lastlbl; 
                for (int k = 0; k < iNumLab; k++)
                {
                    /// get the highest score
                    if (alpha(k, iNumPos - 1) > ftmp){
                        ftmp = alpha(k, iNumPos - 1);
                        lastlbl = k; 
                    }
                }
                decodedpath(lastlbl, iNumPos - 1) = 1;

                for (int t = iNumPos - 2; t >= 0; t--)
                {
                    lastlbl = backtrace(lastlbl, t);
                    decodedpath(lastlbl, t) = 1;
                }
            };
        }
	}
}