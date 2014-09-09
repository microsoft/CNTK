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

                ElemType fAlpha = LZERO;
                ElemType fBeta = LZERO;

                for (int k = 0; k < lbls.GetNumRows(); k++)
                    fAlpha = logOfRight.LogAdd(fAlpha, alpha(k, nObs - 1));

                logOfRight.SetValue(pos_scores);
                functionValues.AssignInnerProductOfMatrices(lbls, logOfRight);

                /// transition score
                ElemType tscore = 0;
                for (int t = 0; t < nObs - 1; t++){
                    int i = -1;
                    for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                    if (lbls(ik, t) != 0){
                        i = ik; break;
                    }
                    int j = -1;
                    for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                    if (lbls(ik, t + 1) != 0){
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
#define DBG_RCRF
#ifdef DBG_RCRF
                ElemType fSum = LZERO;
                for (int i = 0; i < postProb.GetNumCols(); i++){
                    fSum = 0.0;
                    for (int k = 0; k < postProb.GetNumRows(); k++)
                        fSum += postProb(k, i);
                    grd.IsClose(1.0, fSum, 3);
                }
#endif
				Matrix<ElemType>::AddScaledDifference(gradientValues, postProb, labls, grd);
			}

            template<class ElemType>
            void SegmentalEvaluateNode<ElemType>::ErrorSignalToTransitionNode(const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& labls, const Matrix<ElemType>& alpha, const Matrix<ElemType>& beta, 
                const Matrix<ElemType>& pair_scores, Matrix<ElemType>& grd, const size_t shift)
            {

                grd.Resize(pair_scores.GetNumRows(), pair_scores.GetNumCols());
                for (int t = 0; t < labls.GetNumCols(); t++)
                    TransGrdCompute(labls, alpha, beta, pair_scores, grd, t, shift);
            }

            template<class ElemType>
            void SegmentalEvaluateNode<ElemType>::PostProbCompute(Matrix<ElemType>& postprob, const Matrix<ElemType>& alpha,
                const Matrix<ElemType>& beta, const int shift)
            {
                assert(shift == 1);
                int iNumPos = alpha.GetNumCols();
                int iNumLab = alpha.GetNumRows();

                postprob.Resize(iNumLab, iNumPos);
                postprob.SetValue(beta); 
                postprob.InplaceExp();
            };

            template<class ElemType>
            void SegmentalEvaluateNode<ElemType>::TransGrdCompute(const Matrix<ElemType>& lbls, 
                const Matrix<ElemType>&   alpha,
                const Matrix<ElemType>& beta, 
                const Matrix<ElemType>& pair_scores, 
                Matrix<ElemType>& grd, 
                const size_t tPos, /// position
                const int shift)
            {
                assert(shift == 1);
                int iNumPos = alpha.GetNumCols();
                int iNumLab = alpha.GetNumRows();

                int firstLbl = -1;
                for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                if (lbls(ik, 0) != 0){
                    firstLbl = ik; break;
                }

                Matrix<ElemType> b = beta.ColumnSlice(tPos, 1);
                Matrix<ElemType> a;
                if (tPos > 0) 
                    a = alpha.ColumnSlice(tPos-1, 1);
                for (int i = 0; i < iNumLab; i++){
                    ElemType fTmp = LZERO;
                    for (int j = 0; j < iNumLab; j++){
                        if (tPos == 0){
                            if (i == firstLbl){
                                fTmp = 0; 
                            }
                            else{
                                fTmp = LZERO;
                            }
                        }
                        else{
                            fTmp = a(i, 0);
                        }
                        fTmp += pair_scores(j, i);


                        ElemType fSum = LZERO;
                        for (int k = 0; k < iNumLab; k++){
                            ElemType fTmp2;
                            if (tPos == 0){
                                if (k == firstLbl){
                                    fTmp2 = 0;
                                }
                                else{
                                    fTmp2 = LZERO;
                                }
                            }
                            else{
                                fTmp2 = a(k, 0);
                            }
                            fSum = grd.LogAdd(fSum, fTmp2 + pair_scores(j, k));
                        }
                        
                        fTmp -= fSum; 
                        fTmp += b(j, 0);

                        grd(j, i) += exp(fTmp);
                    }
                }

                /// transition score
                int i = -1;
                if (tPos == 0) i = firstLbl; 
                else {
                    for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                        if (lbls(ik, tPos-1) != 0){
                            i = ik; break;
                        }
                }

                int j = -1;
                for (int ik = 0; ik < lbls.GetNumRows(); ik++){
                    if (lbls(ik, tPos) != 0){
                        j = ik; break;
                    }
                }

                ElemType fTmp = grd(j, i) - 1.0;
                grd(j, i) = fTmp;

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
                
                /// need to have 
                alpha.Resize(iNumLab, iNumPos);

 				for (int t = 0; t < iNumPos; t++)
				{
					for (int k = 0; k < iNumLab; k++)
					{
						ElemType fTmp = LZERO;
                        for (int j = 0; j < iNumLab; j++)
                        {
                            ElemType fAlpha = (j == firstLbl) ? 0.0 : LZERO;
                            if (t > 0)
                                fAlpha = alpha(j, t - 1);
                            fTmp = alpha.LogAdd(fTmp, fAlpha + pair_scores(k, j));
                        }
                        fTmp += pos_scores(k, t);  /// include position dependent score
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
                ElemType fSum; 

                int lastLbl = -1;
                for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                if (lbls(ik, iNumPos - 1) != 0){
                    lastLbl = ik; break;
                }

                beta.Resize(iNumLab, iNumPos);

                for (int t = iNumPos - 1; t >= 0; t--)
                {
//#pragma omp parallel for
                    for (int k = 0; k < iNumLab; k++)
                    {
                        ElemType fTmp = LZERO;
                        if (t == iNumPos - 1)
                        {
                            fSum = LZERO;
                            for (int j = 0; j < iNumLab; j++)
                            {
                                fSum = beta.LogAdd(fSum, alpha(j, t));
                            }

                            fTmp = alpha(k, t) - fSum; 
                            beta(k, t) = fTmp;
                        }
                        else
                        {
                            for (int j = 0; j < iNumLab; j++)
                            {
                                fSum = LZERO;
                                for (int m = 0; m < iNumLab; m++)
                                {
                                    fSum = beta.LogAdd(fSum, alpha(m, t) + pair_scores(j, m));
                                }
                                fTmp = beta.LogAdd(fTmp, beta(j, t + 1) + alpha(k, t) + pair_scores(j, k) - fSum);
                            }
                            beta(k, t) = fTmp;
                        }
                    }
                }

            };

            /// pair_scores assumes (i,j) means transition from j to i
            template<class ElemType>
            void SegmentalDecodeNode<ElemType>::DecideStartEndingOutputLab(const Matrix<ElemType>& lbls, size_t & stt, size_t & stp)
            {
                if (stt != -1 && stp!= -1) 
                    return; /// have computed before

                int iNumPos = lbls.GetNumCols();
                int iNumLab = lbls.GetNumRows();

                int firstLbl = -1;
                for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                if (lbls(ik, 0) != 0){
                    firstLbl = ik; break;
                }

                int lastLbl = -1;
                for (int ik = 0; ik < lbls.GetNumRows(); ik++)
                if (lbls(ik, iNumPos - 1) != 0){
                    lastLbl = ik; break;
                }
                
                stt = firstLbl;
                stp = lastLbl;
            };

            /// pair_scores assumes (i,j) means transition from j to i
            template<class ElemType>
            void SegmentalDecodeNode<ElemType>::EvaluateThisNodeS(Matrix<ElemType>& alpha, Matrix<ElemType>& backtrace, Matrix<ElemType>& functionValues, const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, const size_t stt, const size_t stp, const int iStep)
            {
                /// to-do, each slice is for one sentence
                /// to-do, number of slices correspond to number of frames 
                /// this implementation only supports one sentence per minibatch

                int nObs = pos_scores.GetNumCols();

                /// change to other values so can support multiple sentences in each minibatch
                assert(iStep == 1);
                ForwardCompute(alpha, backtrace, pos_scores, pair_scores, stt, iStep);
                BackwardCompute(alpha, functionValues, backtrace, pair_scores, stp, iStep);

            };

            template<class ElemType>
            void SegmentalDecodeNode<ElemType>::ForwardCompute(Matrix<ElemType>& alpha, 
                Matrix<ElemType>& backtrace,
                const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores, 
                const size_t stt, const int shift)
            {
                /// to-do, shift more than 1 to support muliple sentences per minibatch
                assert(shift == 1);
                int iNumPos = pos_scores.GetNumCols();
                int iNumLab = pos_scores.GetNumRows();
                size_t iTmp;

                /// need to have 
                alpha.Resize(iNumLab, iNumPos);
                backtrace.Resize(iNumLab, iNumPos);

                for (int t = 0; t < iNumPos; t++)
                {
                    for (int k = 0; k < iNumLab; k++)
                    {
                        ElemType fTmp = LZERO;
                        if (t > 1){  
                            for (int j = 0; j < iNumLab; j++)
                            {
                                ElemType fAlpha = alpha(j, t - 1) + pair_scores(k, j);
                                if (fAlpha > fTmp){
                                    fTmp = fAlpha;
                                    iTmp = j;
                                }
                            }
                            fTmp += pos_scores(k, t);  /// include position dependent score
                        }
                        else
                        {
                            /// with constrain that the first word is labeled as a given symbol
                            iTmp = stt;
                            fTmp = 0; 
                            if (t == 1){
                                fTmp = alpha(iTmp, t - 1);
                                fTmp += pair_scores(k, iTmp);
                                fTmp += pos_scores(k, t);
                            }
                            else {
                                fTmp = (k == stt) ? pos_scores(k, t) : LZERO;
                            }
                        }
                        alpha(k, t) = fTmp;
                        backtrace(k, t) = (ElemType)iTmp;
                    }
                }

            };

            template<class ElemType>
            void SegmentalDecodeNode<ElemType>::BackwardCompute(const Matrix<ElemType>& alpha, 
                Matrix<ElemType>& decodedpath,
                const Matrix<ElemType>& backtrace, const Matrix<ElemType>& pair_scores, const size_t stp, const int shift)
            {
                assert(shift == 1);
                int iNumPos = backtrace.GetNumCols();
                int iNumLab = backtrace.GetNumRows();

                decodedpath.Resize(iNumLab, iNumPos);

                ElemType ftmp = LZERO;
                size_t lastlbl = stp; 
                decodedpath(lastlbl, iNumPos - 1) = 1;

                for (int t = iNumPos - 1; t > 0; t--)
                {
                    lastlbl = backtrace(lastlbl, t);
                    decodedpath(lastlbl, t-1) = 1;
                }
            };
        }
	}
}