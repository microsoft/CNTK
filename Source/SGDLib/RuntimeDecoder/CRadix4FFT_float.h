#pragma once

#ifdef _UNIMIC_FOR_ARM
#include "CComplex_simd_neon.h"
#else
#include "CComplex_simd_sse.h"
#endif


//
// Radix-4 FFT (float)
//
template<size_t N>
class CRadix4FFT_float
{
public:
    CRadix4FFT_float()
    {
        for (size_t k = 0; k < Np; k += 4)
        {
            auto kp = rev(k, m);
            rassert_op(kp, <, Np / 4);

            auto kp_16 = (rev_index_t)kp;
            rassert_eq1(kp_16, kp);
            rassert_eq1(rev(kp_16, m), k);

            rev_table_16[k >> 2] = kp_16;
            rassert_eq1((size_t)rev_table_16[k >> 2], kp_16);
        }

        auto w = -2 * M_PI / (double)N;
        for (size_t L = 4; L < Np; L *= 4)
        {
#ifdef __arm__
            auto i0 = 3 * ((L - 1) / 3 - 1);
            for (size_t j = 0; j < L; j += 4)
            {
                for (size_t j1 = j; j1 < j + 4; j1++)
                {
                    auto theta = w * ((j1 * (N / 4 / L)) % N);
                    bu_exp_N[i0 + 3 * j +     (j1 % 4)] = { (float)tan(0.5 * theta), (float)sin(    theta) };
                    bu_exp_N[i0 + 3 * j + 4 + (j1 % 4)] = { (float)tan(      theta), (float)sin(2 * theta) };
                    bu_exp_N[i0 + 3 * j + 8 + (j1 % 4)] = { (float)tan(1.5 * theta), (float)sin(3 * theta) };
                }
            }
#else
            auto i0 = (L - 1) / 3 - 1;
            for (size_t j = 0; j < L; j++)
            {
                auto theta = w * ((j * (N / 4 / L)) % N);
                bu_exp_N[i0 + j] = std::polar(1.0, theta);
            }
#endif
        }
#ifdef _WIN32
#pragma warning(suppress: 6294) // C6294: Ill-defined for-loop: initial condition does not satisfy test. Loop body not executed" when Np == N
#endif
        for (size_t j = 0; j < Np / 2; j++)
            bu_final_exp_N[j] = std::polar(1.0, w * j);
    }

    void DFT2(std::complex<float> out[], const std::complex<float> in[]) const
    {
        // pass 1 special case && bit reversal
        auto dr = b << (2 * (m - 2));

        /*
        auto in_dr = &in[dr];
        auto in_2dr = &in[2 * dr];
        auto in_3dr = &in[3 * dr];
        for (size_t k = 0; k < Np; k += 4)
        {
            auto r0 = b * rev_table_16[k >> 2];

            auto x0 = in[r0];
            auto x1 = in_dr[r0];
            auto x2 = in_2dr[r0];
            auto x3 = in_3dr[r0];

            // 8 additions
            auto A = x0 + x2;
            auto B = x0 - x2;
            auto C = x1 + x3;
            auto D = x1 - x3;

            out[k] = A + C;
            out[k + 1] = { B.real() + D.imag(), B.imag() - D.real() }; // B - i * D
            out[k + 2] = A - C;
            out[k + 3] = { B.real() - D.imag(), B.imag() + D.real() }; // B + i * D
        }
        */

        radix4_fft_first_butterfly(out, in, rev_table_16, Np, b, dr);

        // butterfly
        for (size_t c = 0; c < b; c++)
        {
            std::complex<float>* outp = &out[c * Np];

            for (size_t L = 4; L < Np; L *= 4)
            {
#ifdef __arm__
                auto i0 = 3 * ((L - 1) / 3 - 1);
#else
                auto i0 = (L - 1) / 3 - 1;
#endif
                for (size_t j = 0; j < L; j += 4)
                {
                    /*
                    for (size_t j1 = 0; j1 < 4; j1++)
                    {
                        auto w1 = bu_exp_N[L + j + j1];
                        auto w2 = w1 * w1;
                        auto w3 = w1 * w3;
                        for (size_t k = 0; k < Np; k += 4 * L)
                        {
                            auto st = k + j + j1;
                            // 3 multiplications
                            auto x0 = outp[st];
                            auto x1 = w1 * outp[st + L];
                            auto x2 = w2 * outp[st + 2 * L];
                            auto x3 = w3 * outp[st + 3 * L];

                            // 8 additions
                            auto A = x0 + x2;
                            auto B = x0 - x2;
                            auto C = x1 + x3;
                            auto D = x1 - x3;

                            outp[st] = A + C;
                            outp[st + L] = { B.real() + D.imag(), B.imag() - D.real() }; // B - i * D
                            outp[st + 2 * L] = A - C;
                            outp[st + 3 * L] = { B.real() - D.imag(), B.imag() + D.real() }; // B + i * D
                        }
                    }
                    */

                    radix4_fft_butterfly(j, outp, &bu_exp_N[i0], L, Np);
                }
            }
        }

        // final butterfly if N != 4^p
        if (b > 1)
        {
            // for (size_t j = 0; j < Np; j++)
            // {
            //     auto t = bu_final_exp_N[j] * out[j + Np];
            //     auto u = out[j];
            //     out[j] = u + t;
            //     out[j + Np] = u - t;
            // }

            static_assert(b <= 1 || N == 2 * Np, "N/Np not right");
            static_assert(Np >= 8, "not large enough");
            for (size_t j = 0; j < Np / 2; j += 4)
                radix4_fft_final_butterfly(j, out, bu_final_exp_N, Np);
        }
    }

    void IDFT2(std::complex<float> out[], const std::complex<float> in[]) const
    {
        DFT2(out, in);

        // for (size_t j = 0; j < N; j++)
        //     out[j] /= N;
        auto N_1 = 1.0f / (float)N;
        vector_mul_ps(out, N_1, 0, N);

        std::reverse(&out[1], &out[N]);
    }

private:
    // bit reversal
    static size_t rev(size_t i, size_t m1)
    {
        size_t j = 0;
        for (size_t k = 0; k < m1 - 1; k++)
        {
            j |= ((i & 0x3) << 2 * (m1 - k - 2));
            i >>= 2;
        }

        return j;
    }

    static constexpr size_t get_m(size_t n)
    {
        return n == 0 ? 0 : get_m(n / 4) + 1;
    }

    static constexpr size_t m = get_m(N);
    static constexpr size_t Np = (((size_t)0x1) << (2 * (m - 1)));
    static_assert(N == Np || N == 2 * Np, "wrong N");
    static constexpr size_t b = N / Np; // branch out for two radix-4 fft if N != 4^p

    typedef std::conditional_t<Np / 4 <= 256, uint8_t, uint16_t> rev_index_t;
    ALIGNED(16) rev_index_t rev_table_16[Np / 4];
#ifdef __arm__
    ALIGNED(16) std::complex<float> bu_exp_N[3 * ((Np - 1) / 3 - 1)];
#else
    ALIGNED(16) std::complex<float> bu_exp_N[(Np - 1) / 3 - 1];
#endif
    ALIGNED(16) std::complex<float> bu_final_exp_N[N == Np ? 1 : (N - Np) / 2];
};

//
// FFT for real sequence (float)
//
template<size_t N>
class CRadix4RFFT_float
{
public:
    CRadix4RFFT_float()
    {
        rassert_eq1(N % 4, 0U);
        auto w = -2 * M_PI / (double)N;
        const std::complex<double> J(0, 1);
        for (size_t i = 0; i <= N / 4; i++)
        {
            tx_N[i] = 0.5 - 0.5 * J * std::polar(1.0, w * i);
            conj_tx_N[i] = std::conj(tx_N[i]);
        }
    }

    void DFT2(std::complex<float> out[], const float in[]) const
    {
        m_fft.DFT2(out, (const std::complex<float> *)in);
        out[N / 2] = out[0];

        rfft_convert(out, conj_tx_N, out, N);
    }

    void IDFT2(float out[], const std::complex<float> in[]) const
    {
        rfft_convert(in, tx_N, (std::complex<float>*)m_buf, N);

        m_fft.IDFT2((std::complex<float> *)out, (std::complex<float>*)m_buf);
    }

private:
    CRadix4FFT_float<N / 2> m_fft;
    ALIGNED(16) std::complex<float> conj_tx_N[N / 4 + 1]; 
    ALIGNED(16) std::complex<float> tx_N[N / 4 + 1];
    ALIGNED(16) std::complex<float> m_buf[(N / 2 + 1)];
};
