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

    struct NumericLimits
    {
      static const float log_zero_ = -1e100;
      static const float exp_limit_ = 709.78271289338397;
      static const float log_inf_ = 1e100;
      static const float max_ = 1.7976931348623157e+308;
    };


    // a + b, where a and b are assumed to be in the log scale
    float AddAB(float a, float b)
    {
      if (a == NumericLimits::log_zero_ || b == NumericLimits::log_zero_)
        return NumericLimits::log_zero_;
      else
        return a + b;
    }

    // a - b, where a and b are assumed to be in the log scale
    float SubAB(float a, float b)
    {
      if (a == NumericLimits::log_zero_)
        return NumericLimits::log_zero_;
      else if (b == NumericLimits::log_zero_)
        return NumericLimits::log_inf_;
      else
        return a - b;
    }

    // exp(a)
    float ExpA(float a)
    {
      if (a <= NumericLimits::log_zero_)
        return 0;
      else if (a >= NumericLimits::exp_limit_)
        return NumericLimits::max_;
      else
        return exp(a);
    }

    // Approximation of  log(a + b) = log(a) + log(1 + b/a), if b < a
    //                              = log(b) + log(1 + a/b), if a < b
    float LogAPlusB(float a, float b) // x and y are in log scale and so is the result
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

        if (m_trainCriterion != L"CTC")
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
              alpha(row,i) = NumericLimits::log_zero_;
          } else {
            if (i > 1) {
              if (i % 2 == 0 || labels[i-2] == labels[i]) {
                alpha(row,i) = AddAB(prob(row, idxProb), LogAPlusB(alpha(row-1, i-1), alpha(row-1, i)));
              } else {
                float tmp = LogAPlusB(alpha(row-1, i-1), alpha(row-1, i));
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
              beta(row,i) = NumericLimits::log_zero_;
          } else {
           if (i < dim - 2) {
             if (i % 2 == 0 || labels[i+2] == labels[i]) {
               beta(row,i) = AddAB(prob(row,idxProb), LogAPlusB(beta(row+1,i+1), beta(row+1,i)));
             } else {
               float tmp = LogAPlusB(beta(row+1,i+1), beta(row+1,i));
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
                                             float pzx) {
      int dim_error_rows = ctc_err.GetNumRows();
      int dim_error_cols = ctc_err.GetNumCols();
      int label_size = labels.size();

      for(int i=0; i<dim_error_rows; i++) {
        for(int j=0; j<dim_error_cols; j++) {
          float err = NumericLimits::log_zero_;
          for(int s = 0; s < label_size; s++) {
            if (labels[s] == j) {  //
              err = LogAPlusB(err, AddAB(alpha(i,s), beta(i,s)));
            }
          }
          float val = ExpA(SubAB(err, AddAB(pzx, ExpA(log_nnet_out(i,j)) == 0? NumericLimits::log_zero_ : 2*ExpA(log_nnet_out(i,j)))));
          ctc_err(i,j) = -1.0 * val;
        }
      }
    }

    template<class ElemType>
    bool CtcTrainingIO<ElemType>::ComputeDerivative(const wstring& uttID,
        const Matrix<ElemType>& logLikelihoodIn,
        Matrix<ElemType>* derivative,
        ElemType* objective)
    {
        //transpose the matrix so that it is in kaldi format
        Matrix<ElemType> log_nnet_out(logLikelihoodIn.Transpose());
        if (log_nnet_out.GetDeviceId() >= 0)
          log_nnet_out.TransferFromDeviceToDevice(log_nnet_out.GetDeviceId(), CPUDEVICE, true, false, false);

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

        alpha.Reshape(num_frames, exp_len_labels);
        alpha.SetValue(kSetZero);
        beta.Reshape(num_frames, exp_len_labels);
        beta.SetValue(kSetZero);

        for (size_t t = 0; t < num_frames; t++) {
          ComputeCtcLatticeForward(alpha, log_nnet_out, t, label_expand);
        }
        for (size_t t = (num_frames - 1); t >= 0; t--) {
          ComputeCtcLatticeBackward(beta, log_nnet_out, t, label_expand);
        }

        // compute the log-likelihood of the label sequence given the inputs logP(z|x)
        float tmp1 = alpha(num_frames-1, exp_len_labels-1);
        float tmp2 = alpha(num_frames-1, exp_len_labels-2);
        float pzx = tmp1 + log(1 + ExpA(tmp2 - tmp1));

        // compute the errors
        Matrix<ElemType> ctc_err(CPUDEVICE);       // error matrix
        ctc_err.Reshape(num_frames, num_classes);
        ComputeCtcError(ctc_err, alpha, beta, log_nnet_out, label_expand, pzx);

        // back-propagate the errors through the softmax layer
        Matrix<ElemType> nnet_out(log_nnet_out);       // posterior matrix
        for(int i =0;i<log_nnet_out.GetNumRows();i++) {
          for(int j=0; j<log_nnet_out.GetNumCols();j++) {
            nnet_out(i,j) = ExpA(log_nnet_out(i,j));
          }
        }

        ctc_err.ElementMultiplyWith(nnet_out);
        Matrix<ElemType> row_sum(CPUDEVICE);
        row_sum.Reshape(1, num_frames);
        Matrix<ElemType>::VectorSum(ctc_err, row_sum, false);

        Matrix<ElemType> net_out_tmp(nnet_out);
        net_out_tmp.ColumnElementMultiplyWith(row_sum);

        Matrix<ElemType> diff(ctc_err);
        diff =  net_out_tmp - diff;

        //TODO: this is not the correct posterior format
        //Set the correct posterior format
        *derivative = diff.Transpose();

        //Set the objective
        *objective = logLikelihoodIn.GetNumCols() - pzx;

        assert(derivative->GetNumCols() == logLikelihoodIn.GetNumCols());

        m_currentUttID = uttID;
        return true;
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
