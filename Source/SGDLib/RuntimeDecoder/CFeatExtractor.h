#pragma once

#include "CRadix4FFT_float.h"

//
// feature extractor for feature GUID 16kHzStaticStreamE14LFB80
//
// rnnt model feature format:
//      sampling rate: 16kHz
//      window size: 400 (25ms)
//      window advance: 160 (10ms)
//      pre-emphasis: 0.97
//      low frequency cut-off: 0
//      high frequency cut-off: 7690.608442
//      mel frequency bin count: 80
//
class CFeatExtractor
{
public:
    CFeatExtractor()
    {
        // initialize hamming window weights
        float a = (float)(2 * M_PI / (WindowSize - 1));
        for (size_t i = 0; i < WindowSize; i++)
        {
            m_hammingWeights[i] = 0.54f - 0.46f * cosf(a * i);
        }

        // set high freq cut off
        m_khi = f2bin(HiFreqCutOff);

        // mel filter bank weights:
        float mlo = mel(LoFreqCutOff);
        float mhi = mel(HiFreqCutOff);
        float ms = (mhi - mlo) / (FeatDim+1);

        m_centers[0] = mlo;
        for (size_t m = 0; m < FeatDim; m++)
        {
            m_centers[m+1] = mlo + ms * (m + 1);
        }
        m_centers[FeatDim + 1] = mhi;
    }

    void Convert(const float* in_buf, size_t winSize, float* out_buf, size_t featDim)
    {
        rassert_eq1(winSize, WindowSize);
        rassert_eq1(featDim, FeatDim);

        // 1. pre-emphasis
        m_buf[0] = in_buf[0] - PreEmpCoef * in_buf[0]; // note the boundary case; this is what speechlib does
        for (size_t i = WindowSize - 1; i > 0; i--)
        {
            m_buf[i] = in_buf[i] - PreEmpCoef * in_buf[i - 1];
        }

        // 2. apply hamming window
        for (size_t i = 0; i < WindowSize; i++)
        {
            m_buf[i] = m_buf[i] * m_hammingWeights[i];
        }

        // 3. stft
        m_fft.DFT2(m_stft, m_buf);

        // 4. take power
        for (size_t k = 0; k < K; k++)
            m_buf[k] = std::norm(m_stft[k]); // reuse buffer

        // 5. mel filter bank
        // note speechlib triangle is in mel frequency domain, while librosa way is in frequency domain
        // slow implementation, to be optimized later
        for (size_t m = 0; m < FeatDim; m++)
        {
            float o = 0;
            float p = m_centers[m];
            float c = m_centers[m + 1];
            float n = m_centers[m + 2];
            for (size_t k = m_klo; k < m_khi; k++)
            {
                float km = melk(k);
                if (km > p && km < n)
                {
                    o += m_buf[k] * (1.0f - fabs(c - km) / (c - p));
                }
            }

            out_buf[m] = logf(o > 1.0f ? o : 1.0f); 
        }
    }

    static const size_t FeatDim = 80U;
    static const size_t WindowSize = 400U; // 25ms
    static const size_t Advance = 160U; // 10ms
private:
    float mel(float f)
    {
        return 1127.0f * logf(1.0f + f / 700.0f);
    }

    float melk(size_t k)
    {
        return 1127.0f * logf(1.0f + k * SampleRate / (L * 700.0f));
    }

    size_t f2bin(float f)
    {
        return (size_t)((f * L / SampleRate) + 0.5f);  
    }

    const float PreEmpCoef = 0.97f;
    const float LoFreqCutOff = 0;
    const float HiFreqCutOff = 7690.608442f;
    const float SampleRate = 16000.0f;
    static const size_t L = 512U; // fft window size enough to cover signal window
    static const size_t K = L / 2 + 1;

    size_t m_klo = 1U;
    size_t m_khi;
    
    ALIGNED(16) float m_buf[L] = { 0 }; 
    float m_hammingWeights[WindowSize];
    ALIGNED(16) std::complex<float> m_stft[K];
    float m_centers[FeatDim+2];

    CRadix4RFFT_float<L> m_fft;
};
