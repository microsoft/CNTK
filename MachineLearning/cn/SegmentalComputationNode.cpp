#include "ComputationNode.h"
#include "SimpleEvaluator.h"
#include "IComputationNetBuilder.h"
#include "SGD.h"
#include "SegmentalComputationNode.h"

namespace Microsoft {
	namespace MSR {
		namespace CNTK {

			template<class ElemType>
			void SegmentalEvaluateNode<ElemType>::EvaluateThisNodeS(Matrix<ElemType>& postprob, Matrix<ElemType>& alpha, Matrix<ElemType>& beta, Matrix<ElemType>& functionValues, const Matrix<ElemType>& lbls, const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores,
                Matrix<ElemType>& logOfRight)
			{
				ForwardCompute(alpha, beta, functionValues, lbls, pos_scores, pair_scores);
				BackwardCompute(alpha, beta, functionValues, lbls, pos_scores, pair_scores);
				PostProbCompute(postprob, alpha, beta);

                /// to-do, compute the CRF objective function
                logOfRight.SetValue(pos_scores);
                functionValues.AssignInnerProductOfMatrices(lbls, logOfRight);
                functionValues *= (-1);
            };

			template<class ElemType>
			void SegmentalEvaluateNode<ElemType>::ErrorSignalToPostitionDependentNode(const Matrix<ElemType>& gradientValues, const Matrix<ElemType>& labls, const Matrix<ElemType>& postProb, Matrix<ElemType>& grd)
			{
				Matrix<ElemType>::AddScaledDifference(gradientValues, postProb, labls, grd);
			}

			template<class ElemType>
			void SegmentalEvaluateNode<ElemType>::PostProbCompute(Matrix<ElemType>& postprob, const Matrix<ElemType>& alpha, const Matrix<ElemType>& beta)
			{
				int iNumPos = alpha.GetNumCols();
				int iNumLab = alpha.GetNumRows();

				postprob.Resize(iNumLab, iNumPos);
				for (int t = 0; t < iNumPos; t++)
				{
					ElemType fSum = 0;
					for (int k = 0; k < iNumLab; k++)
						fSum = postprob.LogAdd(fSum, alpha(k, t) + beta(k, t));
					for (int k = 0; k < iNumLab; k++)
					{
						ElemType fTmp = alpha(k, t) + beta(k, t);
						fTmp -= fSum;
						postprob(k, t) = fTmp;
					}
				}
			};

			template<class ElemType>
			void SegmentalEvaluateNode<ElemType>::ForwardCompute(Matrix<ElemType>& alpha, const Matrix<ElemType>& beta, Matrix<ElemType>& functionValues, const Matrix<ElemType>& lbls,
				const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores)
			{
				int iNumPos = lbls.GetNumCols();
				int iNumLab = lbls.GetNumRows();

				alpha.Resize(iNumLab, iNumPos);
                Matrix<ElemType> va = alpha.ColumnSlice(0, 1);
                va = 0.0;
				for (int t = 1; t < iNumPos; t++)
				{
					for (int k = 0; k < iNumLab; k++)
					{
						ElemType fTmp = LZERO;
						for (int j = 0; j < iNumLab; j++)
						{
							fTmp = alpha.LogAdd(fTmp, alpha(j, t - 1) + pos_scores(j, t) + pair_scores(j, k));
						}
						alpha(k, t) = fTmp;
					}
				}

			};

			template<class ElemType>
			void SegmentalEvaluateNode<ElemType>::BackwardCompute(const Matrix<ElemType>& alpha, Matrix<ElemType>& beta, Matrix<ElemType>& functionValues, const Matrix<ElemType>& lbls,
				const Matrix<ElemType>& pos_scores, const Matrix<ElemType>& pair_scores)
			{
				int iNumPos = lbls.GetNumCols();
				int iNumLab = lbls.GetNumRows();

				beta.Resize(iNumLab, iNumPos);
                Matrix<ElemType> vb = beta.ColumnSlice(iNumPos - 1, 1);
                vb = 0.0;
				for (int t = iNumPos - 2; t >= 0; t--)
				{
					for (int k = 0; k < iNumLab; k++)
					{
						ElemType fTmp = LZERO;
						for (int j = 0; j < iNumLab; j++)
						{
							fTmp = beta.LogAdd(fTmp, beta(j, t + 1) + pos_scores(j, t + 1) + pair_scores(k, j));
						}
						beta(k, t) = fTmp;
					}
				}

			};

		}
	}
}