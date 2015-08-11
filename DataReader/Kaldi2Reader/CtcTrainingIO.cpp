#include "basetypes.h"
#include "htkfeatio_utils.h"
#include "CtcTrainingIO.h"
#include "kaldi.h"

namespace Microsoft { namespace MSR { namespace CNTK {


    //TODO: To find a good place to copy this code.

  typedef enum {
     kSetZero,
     kUndefined,
     kCopyData
     } MatrixResizeType;


    // a + b, where a and b are assumed to be in the log scale
    template<class ElemType>
    ElemType CtcTrainingIO<ElemType>::AddAB(ElemType a, ElemType b)
    {
      if (a == log_zero_ || b == log_zero_)
        return log_zero_;
      else
        return a + b;
    }

    // a - b, where a and b are assumed to be in the log scale
    template<class ElemType>
    ElemType CtcTrainingIO<ElemType>::SubAB(ElemType a, ElemType b)
    {
      if (a == log_zero_)
        return log_zero_;
      else if (b == log_zero_)
        return log_inf_;
      else
        return a - b;
    }

    // exp(a)
    template<class ElemType>
    ElemType CtcTrainingIO<ElemType>::ExpA(ElemType a)
    {
      if (a <= log_zero_)
        return 0;
      else if (a >= exp_limit_)
        return max_;
      else
        return exp(a);
    }

    // Approximation of  log(a + b) = log(a) + log(1 + b/a), if b < a
    //                              = log(b) + log(1 + a/b), if a < b
    template<class ElemType>
    ElemType CtcTrainingIO<ElemType>::LogAPlusB(ElemType a, ElemType b) // x and y are in log scale and so is the result
    {
        if (b < a)
          return AddAB(a, log(1 + ExpA(SubAB(b, a))));
        else
          return AddAB(b, log(1 + ExpA(SubAB(a, b))));
    }

    // Constructor.
    template<class ElemType>
    CtcTrainingIO<ElemType>::CtcTrainingIO(
        const wstring& labRspecifier,
        const wstring& trainCriterion)
    {
        using namespace msra::asr;
        assert(labRspecifier != L"");
        m_labRspecifier = new kaldi::RandomAccessInt32VectorReader(
            trimmed(fileToStr(toStr(labRspecifier))));
        m_trainCriterion = trainCriterion;

        if (m_trainCriterion != L"ctc")
        {
            LogicError("Supported training criterion is: CTC.\n");
        }
        m_currentUttID = L"";
    }

    // Destructor.
    template<class ElemType>
    CtcTrainingIO<ElemType>::~CtcTrainingIO()
    {
        if (m_labRspecifier != NULL)
        {
            delete m_labRspecifier;
            m_labRspecifier = NULL;
        }
    }

    template<class ElemType>
    void CtcTrainingIO<ElemType>::ComputeCtcLatticeForward(Matrix<ElemType> &alpha,
                                             Matrix<ElemType> &prob,
                                             int row,
                                             std::vector<size_t> &labels) {
        assert(prob.GetNumRows() == alpha.GetNumRows());
        assert(labels.size() == alpha.GetNumCols());

        for (int i=0; i<labels.size(); i++)
        {
          int idxProb = labels[i];

          if (row == 0) {
            if (i < 2)
              alpha(row,i) = prob(row, idxProb);
            else
              alpha(row,i) = log_zero_;
          } else {
            if (i > 1) {
              if (i % 2 == 0 || labels[i-2] == labels[i]) {
                alpha(row,i) = AddAB(prob(row, idxProb), LogAPlusB(alpha(row-1, i-1), alpha(row-1, i)));
              } else {
                ElemType tmp = LogAPlusB(alpha(row-1, i-1), alpha(row-1, i));
                alpha(row,i) = AddAB(prob(row, idxProb), LogAPlusB(alpha(row-1, i-2), tmp));
              }
            } else if (i == 1) {
              alpha(row,i) = AddAB(prob(row, idxProb), LogAPlusB(alpha(row-1, i-1), alpha(row-1, i)));
            } else {
              alpha(row,i) = AddAB(prob(row, idxProb), alpha(row-1, i));
            }
          }
        }
    }

    template<class ElemType>
    void CtcTrainingIO<ElemType>::ComputeCtcLatticeBackward(Matrix<ElemType> &beta,
                                             Matrix<ElemType> &prob,
                                             int row,
                                             std::vector<size_t> &labels) {
        assert(prob.GetNumRows() == beta.GetNumRows());
        assert(labels.size() == beta.GetNumCols());

        for(int i=0; i<labels.size(); i++)
        {
          int idxProb = labels[i];
          int row_num = beta.GetNumRows();
          int dim = beta.GetNumCols();

          if (row == row_num - 1) {
            if (i > dim - 3)
              beta(row,i) = prob(row,idxProb);
            else
              beta(row,i) = log_zero_;
          } else {
           if (i < dim - 2) {
             if (i % 2 == 0 || labels[i+2] == labels[i]) {
               beta(row,i) = AddAB(prob(row,idxProb), LogAPlusB(beta(row+1,i+1), beta(row+1,i)));
             } else {
               ElemType tmp = LogAPlusB(beta(row+1,i+1), beta(row+1,i));
               beta(row,i) = AddAB(prob(row,idxProb), LogAPlusB(beta(row+1,i+2), tmp));
             }
           } else if (i == dim - 2) {
             beta(row,i) = AddAB(prob(row,idxProb), LogAPlusB(beta(row+1,i+1), beta(row+1,i)));
           } else {
             beta(row,i) = AddAB(prob(row,idxProb), beta(row+1,i));
           }
          }
        }
    }

    template<class ElemType>
    void CtcTrainingIO<ElemType>::ComputeCtcError(
                                             Matrix<ElemType> &ctc_err,
                                             Matrix<ElemType> &alpha,
                                             Matrix<ElemType> &beta,
                                             Matrix<ElemType> &log_nnet_out,
                                             std::vector<size_t> &labels,
                                             ElemType pzx) {
      int dim_error_rows = ctc_err.GetNumRows();
      int dim_error_cols = ctc_err.GetNumCols();
      int label_size = labels.size();

      for(int i=0; i<dim_error_rows; i++) {
        for(int j=0; j<dim_error_cols; j++) {
          ElemType err = log_zero_;
          for(int s = 0; s < label_size; s++) {
            if (labels[s] == j) {
              err = LogAPlusB(err, AddAB(alpha(i,s), beta(i,s)));
            }
          }
          ElemType val = ExpA(SubAB(err, AddAB(pzx, ExpA(log_nnet_out(i,j)) == 0? log_zero_ : 2*log_nnet_out(i,j))));
          ctc_err(i,j) = -1.0 * val;
        }
      }
    }

#define DEBUG_UTTERANCE (0)

    template<class ElemType>
    bool CtcTrainingIO<ElemType>::ComputeDerivativeActual(const wstring& uttID,
        const Matrix<ElemType>& logLikelihoodIn,
        Matrix<ElemType>* derivative,
        ElemType* objective)
    {
        //transpose the matrix so that it is in kaldi format
        Matrix<ElemType> log_nnet_out(logLikelihoodIn.Transpose());
        if (log_nnet_out.GetDeviceId() >= 0)
          log_nnet_out.TransferFromDeviceToDevice(log_nnet_out.GetDeviceId(), CPUDEVICE, true, false, false);

        Matrix<ElemType> nnet_out(CPUDEVICE);       // posterior matrix
        nnet_out.Resize(log_nnet_out.GetNumRows(), log_nnet_out.GetNumCols());
        for(int i =0;i<log_nnet_out.GetNumRows();i++) {
          ElemType row_sum=0;
          for(int j=0; j<log_nnet_out.GetNumCols();j++) {
            nnet_out(i,j) = ExpA(log_nnet_out(i,j));
            row_sum = row_sum + nnet_out(i,j);
          }
          for(int j=0; j<log_nnet_out.GetNumCols();j++) {
            nnet_out(i,j) = nnet_out(i,j)/row_sum;
          }
          for(int j=0; j<log_nnet_out.GetNumCols();j++) {
            assert(nnet_out(i,j) >= 0.0);
            log_nnet_out(i,j) = log(nnet_out(i,j));
          }
        }

#if(DEBUG_UTTERANCE)
        FILE * pFile=0;
        pFile = fopen ("debug/posterior.mat.txt","w");
        if (pFile!=NULL)
        {
          for(int i =0;i<nnet_out.GetNumRows();i++) {
            for(int j=0; j<nnet_out.GetNumCols();j++) {
              fprintf(pFile, "%f ", nnet_out(i,j));
            }
            fprintf(pFile, "\n");
          }
          fclose (pFile);
        }
#endif

        std::string uttIDStr = msra::asr::toStr(uttID);

        size_t num_frames = log_nnet_out.GetNumRows();
        size_t num_classes = log_nnet_out.GetNumCols();

        // Check if the label sequence for an utterance is available.
        // and if so read it
        if (!m_labRspecifier->HasKey(uttIDStr))
            RuntimeError("Label not found for utterance %s\n", uttIDStr.c_str());
        const std::vector<int32> label = m_labRspecifier->Value(uttIDStr);

        // label expansion by inserting blank (indexed by 0) at the beginning and end,
        // and between every pair of labels
        size_t len_labels = label.size();
        size_t exp_len_labels = 2*len_labels + 1;

        // this code fills up the label vector with 0
        // Nonspeech phones are assumed to be >= 1
        std::vector<size_t> label_expand;         // the label vector as a matrix
        label_expand.resize(0);
        label_expand.resize(exp_len_labels, 0);
        for (int l = 0; l < len_labels; l++) {
          label_expand[2*l+1] = label[l];
        }

        //define matrices for the forward backward computation
        Matrix<ElemType> alpha(CPUDEVICE);       // alpha matrix
        Matrix<ElemType> beta(CPUDEVICE);        // beta matrix

        alpha.Resize(num_frames, exp_len_labels);
        alpha.SetValue(kSetZero);
        beta.Resize(num_frames, exp_len_labels);
        beta.SetValue(kSetZero);

        for (size_t t = 0; t < num_frames; t++) {
          ComputeCtcLatticeForward(alpha, log_nnet_out, t, label_expand);
        }
        for (int t = (num_frames - 1); t >= 0; t--) {
          ComputeCtcLatticeBackward(beta, log_nnet_out, t, label_expand);
        }
#if(DEBUG_UTTERANCE)
        pFile = fopen ("debug/alpha.mat.txt","w");
        if (pFile!=NULL)
        {
          for(int i =0;i<alpha.GetNumRows();i++) {
            for(int j=0; j<alpha.GetNumCols();j++) {
              fprintf(pFile, "%f ", alpha(i,j));
            }
            fprintf(pFile, "\n");
          }
          fclose (pFile);
        }

        pFile = fopen ("debug/beta.mat.txt","w");
        if (pFile!=NULL)
        {
          for(int i =0;i<beta.GetNumRows();i++) {
            for(int j=0; j<beta.GetNumCols();j++) {
              fprintf(pFile, "%f ", beta(i,j));
            }
            fprintf(pFile, "\n");
          }
          fclose (pFile);
        }
#endif

        // compute the log-likelihood of the label sequence given the inputs logP(z|x)
        ElemType tmp1 = alpha(num_frames-1, exp_len_labels-1);
        ElemType tmp2 = alpha(num_frames-1, exp_len_labels-2);
        ElemType pzx = tmp1 + log(1 + ExpA(tmp2 - tmp1));

        // compute the errors
        Matrix<ElemType> ctc_err(CPUDEVICE);       // error matrix
        ctc_err.Resize(num_frames, num_classes);
        ComputeCtcError(ctc_err, alpha, beta, log_nnet_out, label_expand, pzx);
#if(DEBUG_UTTERANCE)
        printf("\nPzx=%f\n",pzx);
        pFile = fopen ("debug/ctc_error.mat.txt","w");
        if (pFile!=NULL)
        {
          for(int i =0;i<ctc_err.GetNumRows();i++) {
            for(int j=0; j<ctc_err.GetNumCols();j++) {
              fprintf(pFile, "%f ", ctc_err(i,j));
            }
            fprintf(pFile, "\n");
          }
          fclose (pFile);
        }
#endif


        // back-propagate the errors through the softmax layer
        std::vector<ElemType> row_sum;
        row_sum.resize(num_frames, 0);
        for(int i =0;i<ctc_err.GetNumRows();i++) {
          for(int j=0; j<ctc_err.GetNumCols();j++) {
            ctc_err(i,j) = ctc_err(i,j) * nnet_out(i,j);
            row_sum[i] = row_sum[i] + ctc_err(i,j);
          }
        }
        Matrix<ElemType> net_out_tmp(nnet_out);
        for(int i =0;i<net_out_tmp.GetNumRows();i++) {
          ElemType scale = row_sum[i];
          for(int j=0; j<net_out_tmp.GetNumCols();j++) {
            net_out_tmp(i,j) = net_out_tmp(i,j) * scale;
          }
        }

        Matrix<ElemType> diff(ctc_err);
        diff =  diff - net_out_tmp;
        *derivative = diff.Transpose();

#if(DEBUG_UTTERANCE)
        pFile = fopen ("debug/gradient.mat.txt","w");
        if (pFile!=NULL)
        {
          for(int i =0;i<diff.GetNumRows();i++) {
            for(int j=0; j<diff.GetNumCols();j++) {
              fprintf(pFile, "%f ", diff(i,j));
            }
            fprintf(pFile, "\n");
          }
          fclose (pFile);
        }
#endif

        //Set the objective
        *objective = pzx;

        assert(derivative->GetNumCols() == logLikelihoodIn.GetNumCols());

        m_currentUttID = uttID;

        return true;
    }

    template<class ElemType>
    bool CtcTrainingIO<ElemType>::ComputeDerivativeNumerical(const wstring& uttID,
        const Matrix<ElemType>& logLikelihoodIn,
        Matrix<ElemType>* derivative,
        ElemType* objective)
    {
      ElemType eps = 0.00001;
        Matrix<ElemType> diff(CPUDEVICE);
        diff.Resize(logLikelihoodIn.GetNumCols(), logLikelihoodIn.GetNumRows());
        diff.SetValue(kSetZero);

        for(int m=0;m<logLikelihoodIn.GetNumCols();m++) {
          for(int n=0; n<logLikelihoodIn.GetNumRows();n++) {
            ElemType gradElt=0;
            for(int dir=0; dir<2; dir++) {
              //transpose the matrix so that it is in kaldi format
              Matrix<ElemType> log_nnet_out(logLikelihoodIn.Transpose());
              if (log_nnet_out.GetDeviceId() >= 0)
                log_nnet_out.TransferFromDeviceToDevice(log_nnet_out.GetDeviceId(), CPUDEVICE, true, false, false);

              log_nnet_out(m,n) = log_nnet_out(m,n) + ((dir*2) - 1) * eps;

              Matrix<ElemType> nnet_out(CPUDEVICE);       // posterior matrix
              nnet_out.Resize(log_nnet_out.GetNumRows(), log_nnet_out.GetNumCols());
              for(int i =0;i<log_nnet_out.GetNumRows();i++) {
                ElemType row_sum=0;
                for(int j=0; j<log_nnet_out.GetNumCols();j++) {
                  nnet_out(i,j) = ExpA(log_nnet_out(i,j));
                  row_sum = row_sum + nnet_out(i,j);
                }
                for(int j=0; j<log_nnet_out.GetNumCols();j++) {
                  nnet_out(i,j) = nnet_out(i,j)/row_sum;
                }
                for(int j=0; j<log_nnet_out.GetNumCols();j++) {
                  assert(nnet_out(i,j) >= 0.0);
                  log_nnet_out(i,j) = log(nnet_out(i,j));
                }
              }

              std::string uttIDStr = msra::asr::toStr(uttID);

              size_t num_frames = log_nnet_out.GetNumRows();
              size_t num_classes = log_nnet_out.GetNumCols();

              // Check if the label sequence for an utterance is available.
              // and if so read it
              if (!m_labRspecifier->HasKey(uttIDStr))
                  RuntimeError("Label not found for utterance %s\n", uttIDStr.c_str());
              const std::vector<int32> label = m_labRspecifier->Value(uttIDStr);

              // label expansion by inserting blank (indexed by 0) at the beginning and end,
              // and between every pair of labels
              size_t len_labels = label.size();
              size_t exp_len_labels = 2*len_labels + 1;

              // this code fills up the label vector with 0
              // Nonspeech phones are assumed to be >= 1
              std::vector<size_t> label_expand;         // the label vector as a matrix
              label_expand.resize(0);
              label_expand.resize(exp_len_labels, 0);
              for (int l = 0; l < len_labels; l++) {
                label_expand[2*l+1] = label[l];
              }

              //define matrices for the forward backward computation
              Matrix<ElemType> alpha(CPUDEVICE);       // alpha matrix

              alpha.Resize(num_frames, exp_len_labels);
              alpha.SetValue(kSetZero);

              for (size_t t = 0; t < num_frames; t++) {
                ComputeCtcLatticeForward(alpha, log_nnet_out, t, label_expand);
              }

              // compute the log-likelihood of the label sequence given the inputs logP(z|x)
              ElemType tmp1 = alpha(num_frames-1, exp_len_labels-1);
              ElemType tmp2 = alpha(num_frames-1, exp_len_labels-2);
              ElemType pzx = tmp1 + log(1 + ExpA(tmp2 - tmp1));

              gradElt = gradElt + pzx * ((dir*2) - 1);

              m_currentUttID = uttID;
              //Set the objective
              *objective = pzx;
            }
            diff(m,n) = gradElt/(2*eps);
          }
        }

        *derivative = diff.Transpose();

        assert(derivative->GetNumCols() == logLikelihoodIn.GetNumCols());

        return true;
    }

#define PRINT_GRAD  (0)
#define ACTUAL_GRAD (1)

    template<class ElemType>
    bool CtcTrainingIO<ElemType>::ComputeDerivative(const wstring& uttID,
        const Matrix<ElemType>& logLikelihoodIn,
        Matrix<ElemType>* derivative,
        ElemType* objective)
    {

#if(ACTUAL_GRAD)
      bool ret = ComputeDerivativeActual(uttID,
          logLikelihoodIn,
          derivative,
          objective);
#else
      bool ret = ComputeDerivativeNumerical(uttID,
          logLikelihoodIn,
          derivative,
          objective);
#endif

#if(PRINT_GRAD)
      /* BEGIN: print gradients.
       *
       */
      printf("\n\n=====================================================");
      printf("\nPrint (likelihood, gradients).\n");
      printf("\nObjective = %f \n", *objective);
      printf("=====================================================\n");
      Matrix<ElemType> log_nnet_dup(logLikelihoodIn.Transpose());
      if (log_nnet_dup.GetDeviceId() >= 0)
        log_nnet_dup.TransferFromDeviceToDevice(log_nnet_dup.GetDeviceId(), CPUDEVICE, true, false, false);

      for(int i =0;i<derivative->GetNumRows();i++) {
        for(int j=0; j<derivative->GetNumCols();j++) {
          printf("(%f, %f)", logLikelihoodIn(i,j), (*derivative)(i,j));
        }
        printf("\n");
      }
      printf("=====================================================\n\n");

       /*
       END: print gradients.
       */
#endif

      return ret;
    }

    template<class ElemType>
    bool CtcTrainingIO<ElemType>::HasResourceForDerivative(
        const wstring& uttID) const
    {
        if(!m_labRspecifier)
        {
            fprintf(stderr, "WARNING: label reader has not been"
                            " set up yet.\n");
            return false;
        }

        std::string uttIDStr = msra::asr::toStr(uttID);
        if(!m_labRspecifier->HasKey(uttIDStr))
        {
            return false;
        }
        return true;
    }

    template class CtcTrainingIO<float>;
    template class CtcTrainingIO<double>;
}}}
