// matrix/matrix-functions.cc

// Copyright 2009-2011  Microsoft Corporation;  Go Vivace Inc.;  Jan Silovsky
//                      Yanmin Qian;  Saarland University;  Johns Hopkins University (Author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.
//
// (*) incorporates, with permission, FFT code from his book
// "Signal Processing with Lapped Transforms", Artech, 1992.

#include "matrix/matrix-functions.h"
#include "matrix/sp-matrix.h"

namespace kaldi {

template<typename Real> void ComplexFt (const VectorBase<Real> &in,
                                     VectorBase<Real> *out, bool forward) {
  int exp_sign = (forward ? -1 : 1);
  KALDI_ASSERT(out != NULL);
  KALDI_ASSERT(in.Dim() == out->Dim());
  KALDI_ASSERT(in.Dim() % 2 == 0);
  int twoN = in.Dim(), N = twoN / 2;
  const Real *data_in = in.Data();
  Real *data_out = out->Data();

  Real exp1N_re, exp1N_im;  //  forward -> exp(-2pi / N), backward -> exp(2pi / N).
  Real fraction = exp_sign * M_2PI / static_cast<Real>(N);  // forward -> -2pi/N, backward->-2pi/N
  ComplexImExp(fraction, &exp1N_re, &exp1N_im);

  Real expm_re = 1.0, expm_im = 0.0;  // forward -> exp(-2pi m / N).

  for (int two_m = 0; two_m < twoN; two_m+=2) {  // For each output component.
    Real expmn_re = 1.0, expmn_im = 0.0;  // forward -> exp(-2pi m n / N).
    Real sum_re = 0.0, sum_im = 0.0;  // complex output for index m (the sum expression)
    for (int two_n = 0; two_n < twoN; two_n+=2) {
      ComplexAddProduct(data_in[two_n], data_in[two_n+1],
                        expmn_re, expmn_im,
                        &sum_re, &sum_im);
      ComplexMul(expm_re, expm_im, &expmn_re, &expmn_im);
    }
    data_out[two_m] = sum_re;
    data_out[two_m + 1] = sum_im;


    if (two_m % 10 == 0) {  // occasionally renew "expm" from scratch to avoid
      // loss of precision.
      int nextm = 1 + two_m/2;
      Real fraction_mult = fraction * nextm;
      ComplexImExp(fraction_mult, &expm_re, &expm_im);
    } else {
      ComplexMul(exp1N_re, exp1N_im, &expm_re, &expm_im);
    }
  }
}

template
void ComplexFt (const VectorBase<float> &in,
                VectorBase<float> *out, bool forward);
template
void ComplexFt (const VectorBase<double> &in,
                VectorBase<double> *out, bool forward);


#define KALDI_COMPLEXFFT_BLOCKSIZE 8192
// This #define affects how we recurse in ComplexFftRecursive.
// We assume that memory-caching happens on a scale at
// least as small as this.


//! ComplexFftRecursive is a recursive function that computes the
//! complex FFT of size N.  The "nffts" arguments specifies how many
//! separate FFTs to compute in parallel (we assume the data for
//! each one is consecutive in memory).  The "forward argument"
//! specifies whether to do the FFT (true) or IFFT (false), although
//! note that we do not include the factor of 1/N (the user should
//! do this if required.  The iterators factor_begin and factor_end
//! point to the beginning and end (i.e. one past the last element)
//! of an array of small factors of N (typically prime factors).
//! See the comments below this code for the detailed equations
//! of the recursion.


template<typename Real>
void ComplexFftRecursive (Real *data, int nffts, int N,
                          const int *factor_begin,
                          const int *factor_end, bool forward,
                          Vector<Real> *tmp_vec) {
  if (factor_begin == factor_end) {
    KALDI_ASSERT(N == 1);
    return;
  }

  {  // an optimization: compute in smaller blocks.
    // this block of code could be removed and it would still work.
    MatrixIndexT size_perblock = N * 2 * sizeof(Real);
    if (nffts > 1 && size_perblock*nffts > KALDI_COMPLEXFFT_BLOCKSIZE) {  // can break it up...
      // Break up into multiple blocks.  This is an optimization.  We make
      // no progress on the FFT when we do this.
      int block_skip = KALDI_COMPLEXFFT_BLOCKSIZE / size_perblock;  // n blocks per call
      if (block_skip == 0) block_skip = 1;
      if (block_skip < nffts) {
        int blocks_left = nffts;
        while (blocks_left > 0) {
          int skip_now = std::min(blocks_left, block_skip);
          ComplexFftRecursive(data, skip_now, N, factor_begin, factor_end, forward, tmp_vec);
          blocks_left -= skip_now;
          data += skip_now * N*2;
        }
        return;
      } // else do the actual algorithm.
    } // else do the actual algorithm.
  }

  int P = *factor_begin;
  KALDI_ASSERT(P > 1);
  int Q = N / P;


  if (P > 1 && Q > 1) {  // Do the rearrangement.   C.f. eq. (8) below.  Transform
    // (a) to (b).
    Real *data_thisblock = data;
    if (tmp_vec->Dim() < (MatrixIndexT)N) tmp_vec->Resize(N);
    Real *data_tmp = tmp_vec->Data();
    for (int thisfft = 0; thisfft < nffts; thisfft++, data_thisblock+=N*2) {
      for (int offset = 0; offset < 2; offset++) {  // 0 == real, 1 == im.
        for (int p = 0; p < P; p++) {
          for (int q = 0; q < Q; q++) {
            int aidx = q*P + p, bidx = p*Q + q;
            data_tmp[bidx] = data_thisblock[2*aidx+offset];
          }
        }
        for (int n = 0;n < P*Q;n++) data_thisblock[2*n+offset] = data_tmp[n];
      }
    }
  }

  {  // Recurse.
    ComplexFftRecursive(data, nffts*P, Q, factor_begin+1, factor_end, forward, tmp_vec);
  }

  int exp_sign = (forward ? -1 : 1);
  Real rootN_re, rootN_im;  // Nth root of unity.
  ComplexImExp(static_cast<Real>(exp_sign * M_2PI / N), &rootN_re, &rootN_im);

  Real rootP_re, rootP_im;  // Pth root of unity.
  ComplexImExp(static_cast<Real>(exp_sign * M_2PI / P), &rootP_re, &rootP_im);

  {  // Do the multiplication
    // could avoid a bunch of complex multiplies by moving the loop over data_thisblock
    // inside.
    if (tmp_vec->Dim() < (MatrixIndexT)(P*2)) tmp_vec->Resize(P*2);
    Real *temp_a = tmp_vec->Data();

    Real *data_thisblock = data, *data_end = data+(N*2*nffts);
    for (; data_thisblock != data_end; data_thisblock += N*2) {  // for each separate fft.
      Real qd_re = 1.0, qd_im = 0.0;  // 1^(q'/N)
      for (int qd = 0; qd < Q; qd++) {
        Real pdQ_qd_re = qd_re, pdQ_qd_im = qd_im;  // 1^((p'Q+q') / N) == 1^((p'/P) + (q'/N))
                                              // Initialize to q'/N, corresponding to p' == 0.
        for (int pd = 0; pd < P; pd++) {  // pd == p'
          {  // This is the p = 0 case of the loop below [an optimization].
            temp_a[pd*2] = data_thisblock[qd*2];
            temp_a[pd*2 + 1] = data_thisblock[qd*2 + 1];
          }
          {  // This is the p = 1 case of the loop below [an optimization]
            // **** MOST OF THE TIME (>60% I think) gets spent here. ***
            ComplexAddProduct(pdQ_qd_re, pdQ_qd_im,
                              data_thisblock[(qd+Q)*2], data_thisblock[(qd+Q)*2 + 1],
                              &(temp_a[pd*2]), &(temp_a[pd*2 + 1]));
          }
          if (P > 2) {
            Real p_pdQ_qd_re = pdQ_qd_re, p_pdQ_qd_im = pdQ_qd_im;  // 1^(p(p'Q+q')/N)
            for (int p = 2; p < P; p++) {
              ComplexMul(pdQ_qd_re, pdQ_qd_im, &p_pdQ_qd_re, &p_pdQ_qd_im);  // p_pdQ_qd *= pdQ_qd.
              int data_idx = p*Q + qd;
              ComplexAddProduct(p_pdQ_qd_re, p_pdQ_qd_im,
                                data_thisblock[data_idx*2], data_thisblock[data_idx*2 + 1],
                                &(temp_a[pd*2]), &(temp_a[pd*2 + 1]));
            }
          }
          if (pd != P-1)
            ComplexMul(rootP_re, rootP_im, &pdQ_qd_re, &pdQ_qd_im);  // pdQ_qd *= (rootP == 1^{1/P})
          // (using 1/P == Q/N)
        }
        for (int pd = 0; pd < P; pd++) {
          data_thisblock[(pd*Q + qd)*2] = temp_a[pd*2];
          data_thisblock[(pd*Q + qd)*2 + 1] = temp_a[pd*2 + 1];
        }
        ComplexMul(rootN_re, rootN_im, &qd_re, &qd_im);  // qd *= rootN.
      }
    }
  }
}

/* Equations for ComplexFftRecursive.
   We consider here one of the "nffts" separate ffts; it's just a question of
   doing them all in parallel.  We also write all equations in terms of
   complex math (the conversion to real arithmetic is not hard, and anyway
   takes place inside function calls).


   Let the input (i.e. "data" at start) be a_n, n = 0..N-1, and
   the output (Fourier transform) be d_k, k = 0..N-1.  We use these letters because
   there will be two intermediate variables b and c.
   We want to compute:

     d_k = \sum_n a_n 1^(kn/N)                                             (1)

   where we use 1^x as shorthand for exp(-2pi x) for the forward algorithm
   and exp(2pi x) for the backward one.

   We factorize N = P Q (P small, Q usually large).
   With p = 0..P-1 and q = 0..Q-1, and also p'=0..P-1 and q'=0..P-1, we let:

    k == p'Q + q'                                                           (2)
    n == qP + p                                                             (3)

   That is, we let p, q, p', q' range over these indices and observe that this way we
   can cover all n, k.  Expanding (1) using (2) and (3), we can write:

      d_k = \sum_{p, q}  a_n 1^((p'Q+q')(qP+p)/N)
          = \sum_{p, q}  a_n 1^(p'pQ/N) 1^(q'qP/N) 1^(q'p/N)                 (4)

   using 1^(PQ/N) = 1 to get rid of the terms with PQ in them.  Rearranging (4),

     d_k =  \sum_p 1^(p'pQ/N) 1^(q'p/N)  \sum_q 1^(q'qP/N) a_n              (5)

   The point here is to separate the index q.  Now we can expand out the remaining
   instances of k and n using (2) and (3):

     d_(p'Q+q') =  \sum_p 1^(p'pQ/N) 1^(q'p/N)  \sum_q 1^(q'qP/N) a_(qP+p)   (6)

   The expression \sum_q varies with the indices p and q'.  Let us define

         C_{p, q'} =  \sum_q 1^(q'qP/N) a_(qP+p)                            (7)

   Here, C_{p, q'}, viewed as a sequence in q', is just the DFT of the points
   a_(qP+p) for q = 1..Q-1.  These points are not consecutive in memory though,
   they jump by P each time.  Let us define b as a rearranged version of a,
   so that

         b_(pQ+q) = a_(qP+p)                                                  (8)

   How to do this rearrangement in place?  In

   We can rearrange (7) to be written in terms of the b's, using (8), so that

         C_{p, q'} =  \sum_q 1^(q'q (P/N)) b_(pQ+q)                            (9)

   Here, the sequence of C_{p, q'} over q'=0..Q-1, is just the DFT of the sequence
   of b_(pQ) .. b_(p(Q+1)-1).  Let's arrange the C_{p, q'} in a single array in
   memory in the same way as the b's, i.e. we define
         c_(pQ+q') == C_{p, q'}.                                                (10)
   Note that we could have written (10) with q in place of q', as there is only
   one index of type q present, but q' is just a more natural variable name to use
   since we use q' elsewhere to subscript c and C.

   Rewriting (9), we have:
         c_(pQ+q')  = \sum_q 1^(q'q (P/N)) b_(pQ+q)                            (11)
    which is the DFT computed by the recursive call to this function [after computing
    the b's by rearranging the a's].  From the c's we want to compute the d's.
    Taking (6), substituting in the sum (7), and using (10) to write it as an array,
    we have:
         d_(p'Q+q') =  \sum_p 1^(p'pQ/N) 1^(q'p/N)  c_(pQ+q')                   (12)
    This sum is independent for different values of q'.  Note that d overwrites c
    in memory.  We compute this in  a direct way, using a little array of size P to
    store the computed d values for one value of q' (we reuse the array for each value
    of q').

    So the overall picture is this:
    We get a call to compute DFT on size N.

    - If N == 1 we return (nothing to do).
    - We factor N = P Q (typically, P is small).
    - Using (8), we rearrange the data in memory so that we have b not a in memory
       (this is the block "do the rearrangement").
       The pseudocode for this is as follows.  For simplicity we use a temporary array.

          for p = 0..P-1
             for q = 0..Q-1
                bidx = pQ + q
                aidx = qP + p
                tmp[bidx] = data[aidx].
             end
          end
          data <-- tmp
        else

        endif


        The reason this accomplishes (8) is that we want pQ+q and qP+p to be swapped
        over for each p, q, and the "if m > n" is a convenient way of ensuring that
        this swapping happens only once (otherwise it would happen twice, since pQ+q
        and qP+p both range over the entire set of numbers 0..N-1).

    - We do the DFT on the smaller block size to compute c from b (this eq eq. (11)).
      Note that this is actually multiple DFTs, one for each value of p, but this
      goes to the "nffts" argument of the function call, which we have ignored up to now.

    -We compute eq. (12) via a loop, as follows
         allocate temporary array e of size P.
         For q' = 0..Q-1:
            for p' = 0..P-1:
               set sum to zero [this will go in e[p']]
               for p = p..P-1:
                  sum += 1^(p'pQ/N) 1^(q'p/N)  c_(pQ+q')
               end
               e[p'] = sum
            end
            for p' = 0..P-1:
               d_(p'Q+q') = e[p']
            end
         end
         delete temporary array e

*/

// This is the outer-layer calling code for ComplexFftRecursive.
// It factorizes the dimension and then calls the FFT routine.
template<typename Real> void ComplexFft(VectorBase<Real> *v, bool forward, Vector<Real> *tmp_in) {
  KALDI_ASSERT(v != NULL);

  if (v->Dim()<=1) return;
  KALDI_ASSERT(v->Dim() % 2 == 0);  // complex input.
  int N = v->Dim() / 2;
  std::vector<int> factors;
  Factorize(N, &factors);
  int *factor_beg = NULL;
  if (factors.size() > 0)
    factor_beg = &(factors[0]);
  Vector<Real> tmp;  // allocated in ComplexFftRecursive.
  ComplexFftRecursive(v->Data(), 1, N, factor_beg, factor_beg+factors.size(), forward, (tmp_in?tmp_in:&tmp));
}

//! Inefficient version of Fourier transform, for testing purposes.
template<typename Real> void RealFftInefficient (VectorBase<Real> *v, bool forward) {
  KALDI_ASSERT(v != NULL);
  MatrixIndexT N = v->Dim();
  KALDI_ASSERT(N%2 == 0);
  if (N == 0) return;
  Vector<Real> vtmp(N*2);  // store as complex.
  if (forward) {
    for (MatrixIndexT i = 0; i < N; i++)  vtmp(i*2) = (*v)(i);
    ComplexFft(&vtmp, forward);  // this is already tested so we can use this.
    v->CopyFromVec( vtmp.Range(0, N) );
    (*v)(1) = vtmp(N);  // Copy the N/2'th fourier component, which is real,
    // to the imaginary part of the 1st complex output.
  } else {
    // reverse the transformation above to get the complex spectrum.
    vtmp(0) = (*v)(0);  // copy F_0 which is real
    vtmp(N) = (*v)(1);  // copy F_{N/2} which is real
    for (MatrixIndexT i = 1; i < N/2; i++) {
      // Copy i'th to i'th fourier component
      vtmp(2*i) = (*v)(2*i);
      vtmp(2*i+1) = (*v)(2*i+1);
      // Copy i'th to N-i'th, conjugated.
      vtmp(2*(N-i)) = (*v)(2*i);
      vtmp(2*(N-i)+1) = -(*v)(2*i+1);
    }
    ComplexFft(&vtmp, forward);  // actually backward since forward == false
    // Copy back real part.  Complex part should be zero.
    for (MatrixIndexT i = 0; i < N; i++)
      (*v)(i) = vtmp(i*2);
  }
}

template void RealFftInefficient (VectorBase<float> *v, bool forward);
template void RealFftInefficient (VectorBase<double> *v, bool forward);

template
void ComplexFft(VectorBase<float> *v, bool forward, Vector<float> *tmp_in);
template
void ComplexFft(VectorBase<double> *v, bool forward, Vector<double> *tmp_in);


// See the long comment below for the math behind this.
template<typename Real> void RealFft (VectorBase<Real> *v, bool forward) {
  KALDI_ASSERT(v != NULL);
  MatrixIndexT N = v->Dim(), N2 = N/2;
  KALDI_ASSERT(N%2 == 0);
  if (N == 0) return;

  if (forward) ComplexFft(v, true);

  Real *data = v->Data();
  Real rootN_re, rootN_im;  // exp(-2pi/N), forward; exp(2pi/N), backward
  int forward_sign = forward ? -1 : 1;
  ComplexImExp(static_cast<Real>(M_2PI/N *forward_sign), &rootN_re, &rootN_im);
  Real kN_re = -forward_sign, kN_im = 0.0;  // exp(-2pik/N), forward; exp(-2pik/N), backward
  // kN starts out as 1.0 for forward algorithm but -1.0 for backward.
  for (MatrixIndexT k = 1; 2*k <= N2; k++) {
    ComplexMul(rootN_re, rootN_im, &kN_re, &kN_im);

    Real Ck_re, Ck_im, Dk_re, Dk_im;
    // C_k = 1/2 (B_k + B_{N/2 - k}^*) :
    Ck_re = 0.5 * (data[2*k] + data[N - 2*k]);
    Ck_im = 0.5 * (data[2*k + 1] - data[N - 2*k + 1]);
    // re(D_k)= 1/2 (im(B_k) + im(B_{N/2-k})):
    Dk_re = 0.5 * (data[2*k + 1] + data[N - 2*k + 1]);
    // im(D_k) = -1/2 (re(B_k) - re(B_{N/2-k}))
    Dk_im =-0.5 * (data[2*k] - data[N - 2*k]);
    // A_k = C_k + 1^(k/N) D_k:
    data[2*k] = Ck_re;  // A_k <-- C_k
    data[2*k+1] = Ck_im;
    // now A_k += D_k 1^(k/N)
    ComplexAddProduct(Dk_re, Dk_im, kN_re, kN_im, &(data[2*k]), &(data[2*k+1]));

    MatrixIndexT kdash = N2 - k;
    if (kdash != k) {
      // Next we handle the index k' = N/2 - k.  This is necessary
      // to do now, to avoid invalidating data that we will later need.
      // The quantities C_{k'} and D_{k'} are just the conjugates of C_k
      // and D_k, so the equations are simple modifications of the above,
      // replacing Ck_im and Dk_im with their negatives.
      data[2*kdash] = Ck_re;  // A_k' <-- C_k'
      data[2*kdash+1] = -Ck_im;
      // now A_k' += D_k' 1^(k'/N)
      // We use 1^(k'/N) = 1^((N/2 - k) / N) = 1^(1/2) 1^(-k/N) = -1 * (1^(k/N))^*
      // so it's the same as 1^(k/N) but with the real part negated.
      ComplexAddProduct(Dk_re, -Dk_im, -kN_re, kN_im, &(data[2*kdash]), &(data[2*kdash+1]));
    }
  }

  {  // Now handle k = 0.
    // In simple terms: after the complex fft, data[0] becomes the sum of real
    // parts input[0], input[2]... and data[1] becomes the sum of imaginary
    // pats input[1], input[3]...
    // "zeroth" [A_0] is just the sum of input[0]+input[1]+input[2]..
    // and "n2th" [A_{N/2}] is input[0]-input[1]+input[2]... .
    Real zeroth = data[0] + data[1],
        n2th = data[0] - data[1];
    data[0] = zeroth;
    data[1] = n2th;
    if (!forward) {
      data[0] /= 2;
      data[1] /= 2;
    }
  }

  if (!forward) {
    ComplexFft(v, false);
    v->Scale(2.0);  // This is so we get a factor of N increase, rather than N/2 which we would
    // otherwise get from [ComplexFft, forward] + [ComplexFft, backward] in dimension N/2.
    // It's for consistency with our normal FFT convensions.
  }
}

template void RealFft (VectorBase<float> *v, bool forward);
template void RealFft (VectorBase<double> *v, bool forward);

/* Notes for real FFTs.
   We are using the same convention as above, 1^x to mean exp(-2\pi x) for the forward transform.
   Actually, in a slight abuse of notation, we use this meaning for 1^x in both the forward and
   backward cases because it's more convenient in this section.

   Suppose we have real data a[0...N-1], with N even, and want to compute its Fourier transform.
   We can make do with the first N/2 points of the transform, since the remaining ones are complex
   conjugates of the first.  We want to compute:
       for k = 0...N/2-1,
       A_k = \sum_{n = 0}^{N-1}  a_n 1^(kn/N)                 (1)

   We treat a[0..N-1] as a complex sequence of length N/2, i.e. a sequence b[0..N/2 - 1].
   Viewed as sequences of length N/2, we have:
       b = c + i d,
   where c = a_0, a_2 ... and d = a_1, a_3 ...

   We can recover the length-N/2 Fourier transforms of c and d by doing FT on b and
   then doing the equations below.  Derivation is marked by (*) in a comment below (search
   for it).  Let B, C, D be the FTs.
   We have
       C_k = 1/2 (B_k + B_{N/2 - k}^*)                                 (z0)
       D_k =-1/2i (B_k - B_{N/2 - k}^*)                                (z1)
so: re(D_k)= 1/2 (im(B_k) + im(B_{N/2-k}))                             (z2)
    im(D_k) = -1/2 (re(B_k) - re(B_{N/2-k}))                             (z3)

    To recover the FT A from C and D, we write, rearranging (1):

       A_k = \sum_{n = 0, 2, ..., N-2} a_n 1^(kn/N)
            +\sum_{n = 1, 3, ..., N-1} a_n 1^(kn/N)
           = \sum_{n = 0, 1, ..., N/2-1} a_n 1^(2kn/N)  + a_{n+1} 1^(2kn/N) 1^(k/N)
           = \sum_{n = 0, 1, ..., N/2-1} c_n 1^(2kn/N)  + d_n  1^(2kn/N) 1^(k/N)
       A_k =  C_k + 1^(k/N) D_k                                              (a0)

    This equation is valid for k = 0...N/2-1, which is the range of the sequences B_k and
    C_k.  We don't use is for k = 0, which is a special case considered below.  For
    1 < k < N/2, it's convenient to consider the pair k, k', where k' = N/2 - k.
    Remember that C_k' = C_k^ *and D_k' = D_k^* [where * is conjugation].  Also,
    1^(N/2 / N) = -1.  So we have:
       A_k' = C_k^* - 1^(k/N) D_k^*                                          (a0b)
    We do (a0) and (a0b) together.



    By symmetry this gives us the Fourier components for N/2+1, ... N, if we want
    them.  However, it doesn't give us the value for exactly k = N/2.  For k = 0 and k = N/2, it
    is easiest to argue directly about the meaning of the A_k, B_k and C_k in terms of
    sums of points.
       A_0 and A_{N/2} are both real, with A_0=\sum_n a_n, and A_1 an alternating sum
       A_1 = a_0 - a_1 + a_2 ...
     It's easy to show that
              A_0 = B_0 + C_0            (a1)
              A_{N/2} = B_0 - C_0.       (a2)
     Since B_0 and C_0 are both real, B_0 is the real coefficient of D_0 and C_0 is the
     imaginary coefficient.

     *REVERSING THE PROCESS*

     Next we want to reverse this process.  We just need to work out C_k and D_k from the
     sequence A_k.  Then we do the inverse complex fft and we get back where we started.
     For 0 and N/2, working from (a1) and (a2) above, we can see that:
          B_0 = 1/2 (A_0 + A_{N/2})                                       (y0)
          C_0 = 1/2 (A_0 + A_{N/2})                                       (y1)
     and we use
         D_0 = B_0 + i C_0
     to get the 1st complex coefficient of D.  This is exactly the same as the forward process
     except with an extra factor of 1/2.

     Consider equations (a0) and (a0b).  We want to work out C_k and D_k from A_k and A_k'.  Remember
     k' = N/2 - k.

     Write down
         A_k     =  C_k + 1^(k/N) D_k        (copying a0)
         A_k'^* =   C_k - 1^(k/N) D_k       (conjugate of a0b)
      So
             C_k =            0.5 (A_k + A_k'^*)                    (p0)
             D_k = 1^(-k/N) . 0.5 (A_k - A_k'^*)                    (p1)
      Next, we want to compute B_k and B_k' from C_k and D_k.  C.f. (z0)..(z3), and remember
      that k' = N/2-k.  We can see
      that
              B_k  = C_k + i D_k                                    (p2)
              B_k' = C_k - i D_k                                    (p3)

     We would like to make the equations (p0) ... (p3) look like the forward equations (z0), (z1),
     (a0) and (a0b) so we can reuse the code.  Define E_k = -i 1^(k/N) D_k.  Then write down (p0)..(p3).
     We have
             C_k  =            0.5 (A_k + A_k'^*)                    (p0')
             E_k  =       -0.5 i   (A_k - A_k'^*)                    (p1')
             B_k  =    C_k - 1^(-k/N) E_k                            (p2')
             B_k' =    C_k + 1^(-k/N) E_k                            (p3')
     So these are exactly the same as (z0), (z1), (a0), (a0b) except replacing 1^(k/N) with
     -1^(-k/N) .  Remember that we defined 1^x above to be exp(-2pi x/N), so the signs here
     might be opposite to what you see in the code.

     MODIFICATION: we need to take care of a factor of two.  The complex FFT we implemented
     does not divide by N in the reverse case.  So upon inversion we get larger by N/2.
     However, this is not consistent with normal FFT conventions where you get a factor of N.
     For this reason we multiply by two after the process described above.

*/


/*
   (*) [this token is referred to in a comment above].

   Notes for separating 2 real transforms from one complex one.  Note that the
   letters here (A, B, C and N) are all distinct from the same letters used in the
   place where this comment is used.
   Suppose we
   have two sequences a_n and b_n, n = 0..N-1.  We combine them into a complex
   number,
      c_n = a_n + i b_n.
   Then we take the fourier transform to get
      C_k = \sum_{n = 0}^{N-1} c_n 1^(n/N) .
   Then we use symmetry.  Define A_k and B_k as the DFTs of a and b.
   We use A_k = A_{N-k}^*, and B_k = B_{N-k}^*, since a and b are real.  Using
      C_k     = A_k    +  i B_k,
      C_{N-k} = A_k^*  +  i B_k^*
              = A_k^*  -  (i B_k)^*
   So:
      A_k     = 1/2  (C_k + C_{N-k}^*)
    i B_k     = 1/2  (C_k - C_{N-k}^*)
->    B_k     =-1/2i (C_k - C_{N-k}^*)
->  re(B_k)   = 1/2 (im(C_k) + im(C_{N-k}))
    im(B_k)   =-1/2 (re(C_k) - re(C_{N-k}))

 */

template<typename Real> void ComputeDctMatrix(Matrix<Real> *M) {
  //KALDI_ASSERT(M->NumRows() == M->NumCols());
  MatrixIndexT K = M->NumRows();
  MatrixIndexT N = M->NumCols();

  KALDI_ASSERT(K > 0);
  KALDI_ASSERT(N > 0);
  Real normalizer = std::sqrt(1.0 / static_cast<Real>(N));  // normalizer for
  // X_0.
  for (MatrixIndexT j = 0; j < N; j++) (*M)(0, j) = normalizer;
  normalizer = std::sqrt(2.0 / static_cast<Real>(N));  // normalizer for other
   // elements.
  for (MatrixIndexT k = 1; k < K; k++)
    for (MatrixIndexT n = 0; n < N; n++)
      (*M)(k, n) = normalizer
          * std::cos( static_cast<double>(M_PI)/N * (n + 0.5) * k );
}


template void ComputeDctMatrix(Matrix<float> *M);
template void ComputeDctMatrix(Matrix<double> *M);


template<typename Real>
void MatrixExponential<Real>::Clear() {
  N_ = 0;
  P_.Resize(0, 0);
  B_.clear();
  powers_.clear();
}

template<typename Real>
void MatrixExponential<Real>::Compute(const MatrixBase<Real> &M,
                                      MatrixBase<Real> *X) {
  // does *X = exp(M)
  KALDI_ASSERT(M.NumRows() == M.NumCols());
  Clear();

  N_ = ComputeN(M);
  MatrixIndexT dim = M.NumRows();
  P_.Resize(dim, dim);
  P_.CopyFromMat(M);
  P_.Scale(std::pow(static_cast<Real>(0.5),
                    static_cast<Real>(N_)));
  // would need to keep this code in sync with ComputeN().
  B_.resize(N_+1);
  B_[0].Resize(dim, dim);
  ComputeTaylor(P_, &(B_[0]));  // set B_[0] = exp(P_)
  for (MatrixIndexT i = 1; i <= N_; i++) {
    // implement the recursion B_[k] = 2 B_[k-1] + B_[k-1]^2.
    B_[i].Resize(dim, dim);  // zeros it.
    B_[i].AddMat(2.0, B_[i-1], kNoTrans);
    B_[i].AddMatMat(1.0, B_[i-1], kNoTrans, B_[i-1], kNoTrans, 1.0);
  }
  KALDI_ASSERT(X->NumRows() == dim && X->NumCols() == dim);
  (*X).CopyFromMat(B_[N_]);  // last one plus the unit matrix is the answer.
  // add in the unit matrix.
  for (MatrixIndexT i = 0; i < dim; i++)
    (*X)(i, i) += 1.0;
};

template<typename Real>
void MatrixExponential<Real>::Compute(const SpMatrix<Real> &M,
                                      SpMatrix<Real> *X) {
  Matrix<Real> Mfull(M), Xfull(M.NumRows(), M.NumCols());
  Compute(Mfull, &Xfull);
  X->CopyFromMat(Xfull);
}


template<typename Real>
MatrixIndexT MatrixExponential<Real>::ComputeN(const MatrixBase<Real> &M) {
  // Computes the power of two we want to use.  Aim to get
  // AScaled.FrobeniusNorm() < 1/10.
  Real norm = M.FrobeniusNorm();
  Real max_norm = 0.1;
  if (norm > 1000) {
    KALDI_WARN << "Trying to compute exponent of very high-norm matrix: norm = "
               << norm;
  }
  MatrixIndexT N = 0;
  while (norm > max_norm) { norm *= 0.5; N++; }
  return N;
}

template<typename Real>
void MatrixExponential<Real>::ComputeTaylor(const MatrixBase<Real> &P, MatrixBase<Real> *B0) {
  KALDI_ASSERT(P.FrobeniusNorm() < 1.001);  // should actually be << 1
  // for this to work fast enough.
  KALDI_ASSERT(P.NumRows() == P.NumCols());
  MatrixIndexT dim = P.NumRows();
  KALDI_ASSERT(B0->NumRows() == dim && B0->NumCols() == dim);
  B0->SetZero();
  MatrixIndexT n = 1, n_factorial = 1, max_iter = 10000;
  Matrix<Real> Pn(P),  // Pn = P^n
      B0cur(dim, dim),
      tmp(dim, dim);  // use B0cur to test whether B0 changed.
  std::vector<Matrix<Real>* > powers_tmp;  // list of stored powers of P, starting
  // from 2 and up to 1 before the last one we used.
  while (n < max_iter) {  // have an arbitrary very large limit on #iters
    B0cur.AddMat(1.0 / n_factorial, Pn);
    if (B0cur.Equal(*B0)) // was no change [already very small]
      break;
    B0->CopyFromMat(B0cur);  // Keep B0 in sync with B0cur.
    tmp.AddMatMat(1.0, P, kNoTrans, Pn, kNoTrans, 0.0);  // tmp = P * P^n
    n++;
    n_factorial *= n;
    Pn.CopyFromMat(tmp);  // copy back to P^n
    powers_tmp.push_back(new Matrix<Real>(tmp));
  }
  if (n == max_iter)
    KALDI_WARN << "Reached maximum iteration computing Taylor expansion of matrix [serious problem]";

  powers_.resize(powers_tmp.size());
  for (MatrixIndexT i = 0;
      i < static_cast<MatrixIndexT>(powers_tmp.size()); i++) {
    powers_[i].Swap(powers_tmp[i]);
    delete powers_tmp[i];
  }
}

template<typename Real>
void MatrixExponential<Real>::Backprop(const MatrixBase<Real> &hX,
                                       MatrixBase<Real> *hM) const {
  MatrixIndexT dim = P_.NumRows();
  KALDI_ASSERT(hX.NumRows() == dim && hX.NumCols() == dim
               && hM->NumRows() == dim && hM->NumCols() == dim);
  Matrix<Real> dB(hX);
  // dB represents the gradient df/dB_[i] for the current
  // value of i, which decreases from N_ to zero (currently it's N_)
  for (MatrixIndexT i = N_-1;
      i != (static_cast<MatrixIndexT>(0)-static_cast<MatrixIndexT>(1));  // i >= 0
      i--) {
    // Propagate back from df/dB_[i+1] to df/dB_[i].
    // Using B_[i+1] = 2 B_[i] + B_[i]*B_[i],
    // df/dB_[i] = 2*(df/dB_[i+1]) +  B_[i]^T (df/dB_[i+1])
    //         + (df/dB_[i+1])B_[i]^T
    // note, here we use the perhaps slightly wrong convention that
    // df/dB_[i] is not transposed, i.e. its i, j'th element is the derivative
    // of f w.r.t. the i, j'th element of B_[i].
    Matrix<Real> prev_dB(dB);

    prev_dB.Scale(2.0);      // the term 2*(df/dB_[i+1])
    // add in the term B_[i](df/dB_[i+1])
    prev_dB.AddMatMat(1.0, B_[i], kTrans, dB, kNoTrans, 1.0);
    // add in the term (df/dB_[i+1])B_[i]
    prev_dB.AddMatMat(1.0, dB, kNoTrans, B_[i], kTrans, 1.0);
    dB.CopyFromMat(prev_dB);
  }
  // currently dB is the gradient df/dB_[0], which is exp(P_) - I.
  // we have to backprop this and we get df/dP_.
  BackpropTaylor(dB, hM);  // at this point, hM is temporarily used to store
  // df/dP_.
  hM->Scale(std::pow(static_cast<Real>(0.5),
                     static_cast<Real>(N_)));  // Since A_Scaled = A * std::pow(0.5, N_).
}


template<typename Real>
void MatrixExponential<Real>::Backprop(const SpMatrix<Real> &hX,
                                       SpMatrix<Real> *hM) const {
  Matrix<Real> hXfull(hX), hMfull(hX.NumRows(), hX.NumCols());
  Backprop(hXfull, &hMfull);
  hM->CopyFromMat(hMfull);
}


template<typename Real>
void MatrixExponential<Real>::BackpropTaylor(const MatrixBase<Real> &hB0,
                                             MatrixBase<Real> *hP) const {
  // Backprop through the Taylor-series computation.
  // the original computation was:
  //  X = \sum_{i = 1}^n (1/i!) P^i
  // Also you can see that X is actually the exponential minus I, since we start
  //   the series from 1; this doesn't affect the derivatives.
  // The variable \hat{B}_0 (hB0) represents df/dX, where f is a scalar function.
  // Note that there is no transpose in our notation for derivative: hB0(i, j) is
  // d/df of E(i, j).
  // Imagine that f is \tr(hB0^T B0) (since this varies linearly with B0 in the same
  // way that the real f does).  We want d/dP (hB0^T B0); call this hP (for \hat{P}).
  //   hP = d/dP  \sum_{i = 1}^n (1/i!) hB0^T P^i
  // Taking each individual term P in this expression and treating the others as constants,
  // and noting that whenever we have something like tr(A B), then B^T is the derivative
  // of this expression w.r.t. A (in our notation),
  //  hP = hB0 + (1/2!) (hB0 P^T + P^T hB0) + (1/3!) (hB0 P^T P^T + P^T hB0 P^T + P^T P^T hB0)
  //          + (1/4!) ....   (1)
  // We can compute this with the following recursion by which we get each of the terms
  // in this series in turn:
  //  hP_1 <-- hB0
  //  hP <-- hP_1
  //  for n = 2 .. infinity:  # we only carry this on for as many terms as we used
  //                          # in the original expansion
  //     hP_n <-- (1/n) hP_{n-1} P^T + (1/n!) (P^T)^(n-1) hP_{n-1}
  //     hP <-- hP + hP_n

  const Matrix<Real> &P(P_);
  MatrixIndexT dim = P.NumRows();
  KALDI_ASSERT(P.NumCols() == dim && hB0.NumRows() == dim && hB0.NumCols() == dim
               && hP->NumRows() == dim && hP->NumCols() == dim);
  hP->SetZero();
  Matrix<Real> hPn1(hB0), 
      hPn(dim, dim);
  hP->AddMat(1.0, hPn1);  // first term in (1):  df/dP += K

  MatrixIndexT n = 2, nfact = 2;
  // Now do n = 2 in comment above (this is special case,
  // since we did not store the 1st power in powers_).
  hPn.AddMatMat(1.0/n, hPn1, kNoTrans, P, kTrans, 0.0);  // hP_n <-- (1/n) hP_{n-1} P^T
  hPn.AddMatMat(1.0/nfact, P, kTrans, hB0, kNoTrans, 1.0);  // hP_n += (1/n!) P^T^(n-1) hB0
  hP->AddMat(1.0, hPn);  // add in second term in (1)
  hPn.Swap(&hPn1);

  for (MatrixIndexT i = 0;
      i < static_cast<MatrixIndexT>(powers_.size()); i++) {
    n++;
    nfact *= n;
    // i corresponds to (n-1)-2, and powers_[i] contains the n-1'th power of P.
    // next line: hP_n <-- (1/n) hP_{n-1} P^T
    hPn.AddMatMat(1.0/n, hPn1, kNoTrans, P, kTrans, 0.0);
    // next line: hP_n += (1/n!) P^T^(n-1) hB0
    hPn.AddMatMat(1.0/nfact, powers_[i], kTrans, hB0, kNoTrans, 1.0);
    hP->AddMat(1.0, hPn);  // add in n'th term in (1)
    hPn.Swap(&hPn1);
  }
}

template class MatrixExponential<float>;
template class MatrixExponential<double>;


template<typename Real>
void ComputePca(const MatrixBase<Real> &X,
                MatrixBase<Real> *U,
                MatrixBase<Real> *A,
                bool print_eigs,
                bool exact) {
  // Note that some of these matrices may be transposed w.r.t. the
  // way it's most natural to describe them in math... it's the rows
  // of X and U that correspond to the (data-points, basis elements).
  MatrixIndexT N = X.NumRows(), D = X.NumCols();
  // N = #points, D = feature dim.
  KALDI_ASSERT(U != NULL && U->NumCols() == D);
  MatrixIndexT G = U->NumRows();  // # of retained basis elements.
  KALDI_ASSERT(A == NULL || (A->NumRows() == N && A->NumCols() == G));
  KALDI_ASSERT(G <= N && G <= D);
  if (D < N) {  // Do conventional PCA.
    SpMatrix<Real> Msp(D);  // Matrix of outer products.
    Msp.AddMat2(1.0, X, kTrans, 0.0);  // M <-- X^T X
    Matrix<Real> Utmp;
    Vector<Real> l;
    if (exact) {
      Utmp.Resize(D, D);
      l.Resize(D);
      //Matrix<Real> M(Msp);
      //M.DestructiveSvd(&l, &Utmp, NULL);
      Msp.Eig(&l, &Utmp);
    } else {
      Utmp.Resize(D, G);
      l.Resize(G);
      Msp.TopEigs(&l, &Utmp);
    }
    SortSvd(&l, &Utmp);
    
    for (MatrixIndexT g = 0; g < G; g++)
      U->Row(g).CopyColFromMat(Utmp, g);
    if (print_eigs)
      KALDI_LOG << (exact ? "" : "Retained ")
                << "PCA eigenvalues are " << l;
    if (A != NULL)
      A->AddMatMat(1.0, X, kNoTrans, *U, kTrans, 0.0);
  } else {  // Do inner-product PCA.
    SpMatrix<Real> Nsp(N);  // Matrix of inner products.
    Nsp.AddMat2(1.0, X, kNoTrans, 0.0);  // M <-- X X^T

    Matrix<Real> Vtmp;
    Vector<Real> l;
    if (exact) {
      Vtmp.Resize(N, N);
      l.Resize(N);
      Matrix<Real> Nmat(Nsp);
      Nmat.DestructiveSvd(&l, &Vtmp, NULL);
    } else {
      Vtmp.Resize(N, G);
      l.Resize(G);
      Nsp.TopEigs(&l, &Vtmp);
    }
    
    MatrixIndexT num_zeroed = 0;
    for (MatrixIndexT g = 0; g < G; g++) {
      if (l(g) < 0.0) {
        KALDI_WARN << "In PCA, setting element " << l(g) << " to zero.";
        l(g) = 0.0;
        num_zeroed++;
      }
    }
    SortSvd(&l, &Vtmp); // Make sure zero elements are last, this
    // is necessary for Orthogonalize() to work properly later.

    Vtmp.Transpose();  // So eigenvalues are the rows.
    
    for (MatrixIndexT g = 0; g < G; g++) {
      Real sqrtlg = sqrt(l(g));
      if (l(g) != 0.0) {
        U->Row(g).AddMatVec(1.0 / sqrtlg, X, kTrans, Vtmp.Row(g), 0.0);
      } else {
        U->Row(g).SetZero();
        (*U)(g, g) = 1.0;  // arbitrary direction.  Will later orthogonalize.
      }
      if (A != NULL)
        for (MatrixIndexT n = 0; n < N; n++)
          (*A)(n, g) = sqrtlg * Vtmp(g, n);
    }
    // Now orthogonalize.  This is mainly useful in
    // case there were zero eigenvalues, but we do it
    // for all of them.
    U->OrthogonalizeRows();
    if (print_eigs)
      KALDI_LOG << "(inner-product) PCA eigenvalues are " << l;
  }
}


template
void ComputePca(const MatrixBase<float> &X,
                MatrixBase<float> *U,
                MatrixBase<float> *A,
                bool print_eigs,
                bool exact);

template
void ComputePca(const MatrixBase<double> &X,
                MatrixBase<double> *U,
                MatrixBase<double> *A,
                bool print_eigs,
                bool exact);


// Added by Dan, Feb. 13 2012. 
// This function does: *plus += max(0, a b^T),
// *minus += max(0, -(a b^T)).
template<typename Real>
void AddOuterProductPlusMinus(Real alpha,
                              const VectorBase<Real> &a,
                              const VectorBase<Real> &b,
                              MatrixBase<Real> *plus, 
                              MatrixBase<Real> *minus) {
  KALDI_ASSERT(a.Dim() == plus->NumRows() && b.Dim() == plus->NumCols()
               && a.Dim() == minus->NumRows() && b.Dim() == minus->NumCols());
  int32 nrows = a.Dim(), ncols = b.Dim(), pskip = plus->Stride() - ncols,
      mskip = minus->Stride() - ncols;
  const Real *adata = a.Data(), *bdata = b.Data();
  Real *plusdata = plus->Data(), *minusdata = minus->Data();

  for (int32 i = 0; i < nrows; i++) {
    const Real *btmp = bdata;
    Real multiple = alpha * *adata;
    if (multiple > 0.0) {
      for (int32 j = 0; j < ncols; j++, plusdata++, minusdata++, btmp++) {
        if (*btmp > 0.0) *plusdata += multiple * *btmp;
        else *minusdata -= multiple * *btmp;
      }
    } else {
      for (int32 j = 0; j < ncols; j++, plusdata++, minusdata++, btmp++) {
        if (*btmp < 0.0) *plusdata += multiple * *btmp;
        else *minusdata -= multiple * *btmp;
      }
    }        
    plusdata += pskip;
    minusdata += mskip;
    adata++;
  }
}

// Instantiate template
template
void AddOuterProductPlusMinus<float>(float alpha,
                                     const VectorBase<float> &a,
                                     const VectorBase<float> &b,
                                     MatrixBase<float> *plus, 
                                     MatrixBase<float> *minus);
template
void AddOuterProductPlusMinus<double>(double alpha,
                                      const VectorBase<double> &a,
                                      const VectorBase<double> &b,
                                      MatrixBase<double> *plus, 
                                      MatrixBase<double> *minus);


} // end namespace kaldi


