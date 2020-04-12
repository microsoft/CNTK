#pragma once

#include <smmintrin.h> //requires SSE4.1
#include <future>



    template <class T>
    T* __align(void* p, size_t cnt, size_t ALIGN)
    {
        auto s = sizeof(T) * cnt + ALIGN;
        return (T*) std::align(
            ALIGN,
            sizeof(T) * cnt,
            p,
            s);
    }

enum class MatrixKind
{
    Float = 0,
    Q16 = 1,
    Q8 = 2,
    SSE_Q8 = 3
};

//
// Neural net weight matrix interface.
//
class IMatrix
{
public:
    const uint32_t M;
    const uint32_t N;

    virtual ~IMatrix() {}
    virtual std::future<bool> fread(FILE* fp, bool transpose) = 0;
    virtual std::future<bool> mread(float* buf, bool transpose) = 0;
    virtual void fwrite(FILE* fp) = 0;
    virtual std::unique_ptr<IMatrix> Times(const IMatrix* that) const = 0;

    // C = A * B
    virtual void TimesVector(
        float* C,
        uint32_t C_M,
        uint32_t C_M_Padded,
        const float* B,
        uint32_t B_M,
        uint32_t B_M_Padded) const = 0;

    virtual void RetrieveColumn(
        float* C,
        uint32_t C_M,
        uint32_t j) const = 0;

protected:
    IMatrix(uint32_t m, uint32_t n)
        : M(m),
          N(n)
    {
    }
};

//
// Float weight matrix
//
class CMatrix : public IMatrix
{
public:
    CMatrix(uint32_t m, uint32_t n)
        : IMatrix(m, n),
          M_Padded((m + M_Block - 1) / M_Block * M_Block),
          N_Padded((n + N_Block - 1) / N_Block * N_Block),
          m_store(std::make_unique<uint8_t[]>(sizeof(float) * M_Padded * N_Padded + ALIGN)),
          m_x(__align<float>(m_store.get(), M_Padded * N_Padded, ALIGN))
    {
        rassert_eq(M_Padded % M_Block, 0u);
        rassert_op(M_Padded, >=, M);
        rassert_op(M_Padded - M, <, M_Block);

        rassert_eq(N_Padded % N_Block, 0u);
        rassert_op(N_Padded, >=, N);
        rassert_op(N_Padded - N, <, N_Block);
    }

    static constexpr size_t M_Block = 16;
    const uint32_t M_Padded;

    static constexpr size_t N_Block = 1;
    const uint32_t N_Padded;

    virtual std::future<bool> fread(FILE* fp, bool transpose)
    {
        auto buf = new float[M * N];
        rassert_eq(M * N, ::fread(buf, sizeof(float), M * N, fp));

        return std::async(
            std::launch::async,
            [this, transpose, buf]() {
                for (size_t i = 0; i < M; i++)
                    for (size_t j = 0; j < N; j++)
                        GetElement(i, j) = transpose ? buf[j * M + i] : buf[i * N + j];
                delete[] buf;
                return false;
            });
    }

    virtual std::future<bool> mread(float* buf, bool transpose)
    {
        return std::async(
            std::launch::async,
            [this, transpose, buf]() {
                for (size_t i = 0; i < M; i++)
                    for (size_t j = 0; j < N; j++)
                        GetElement(i, j) = transpose ? buf[j * M + i] : buf[i * N + j];
                return false;
            });
    }
    virtual void fwrite(FILE* fp)
    {
        uint32_t Dims[2] = {M, N};
        rassert_eq(1, ::fwrite(Dims, sizeof(Dims), 1, fp));
        for (uint32_t i = 0; i < M; i++)
            for (uint32_t j = 0; j < N; j++)
                rassert_eq(1, ::fwrite(&GetElement(i, j), sizeof(GetElement(i, j)), 1, fp));
    }

    virtual std::unique_ptr<IMatrix> Times(const IMatrix* that) const
    {
        auto B = dynamic_cast<const CMatrix*>(that);
        rassert_eq(N, B->M);
        auto result = std::make_unique<CMatrix>(M, B->N);

        for (size_t i = 0; i < M; i++)
            for (size_t j = 0; j < B->N; j++)
                for (size_t k = 0; k < N; k++)
                    result->GetElement(i, j) += GetElement(i, k) * B->GetElement(k, j);

        return result;
    }

    // C = A * B
    virtual void TimesVector(
        float* C,
        uint32_t C_M,
        uint32_t C_M_Padded,
        const float* B,
        uint32_t B_M,
        uint32_t B_M_Padded) const
    {
        rassert_eq(M, C_M);
        rassert_op(M_Padded, <=, C_M_Padded);
        rassert_eq(N, B_M);
        rassert_op(N_Padded, <=, B_M_Padded);

        // for (size_t i = 0; i < M; i++)
        // {
        //     float t = 0;
        //     for (size_t k = 0; k < A.N; k++)
        //         t += A[i][k] * B[k];
        //     C[i] = t;
        // }

        static_assert(N_Block == 1, "CMatrix::N_Block != 1");
        static_assert(M_Block == 16, "CMatrix::M_Block != 16");
        for (size_t i = 0; i < M_Padded; i += 16)
        {
            const float* a = GetBlock(i, 0);
            const int* b = (const int*) &B[0];

            auto zero = _mm_setzero_ps();
            auto t0 = zero;
            auto t1 = zero;
            auto t2 = zero;
            auto t3 = zero;
            size_t k = 0;
            for (; k < N_Padded; k += 1)
            {
                auto b_k = b[k];
                if (b_k == 0)
                    continue;

                auto y = _mm_castsi128_ps(_mm_set1_epi32(b_k));

                auto x0 = _mm_load_ps(&a[16 * k + 0]);
                auto x1 = _mm_load_ps(&a[16 * k + 4]);
                auto x2 = _mm_load_ps(&a[16 * k + 8]);
                auto x3 = _mm_load_ps(&a[16 * k + 12]);
                t0 = _mm_add_ps(t0, _mm_mul_ps(x0, y));
                t1 = _mm_add_ps(t1, _mm_mul_ps(x1, y));
                t2 = _mm_add_ps(t2, _mm_mul_ps(x2, y));
                t3 = _mm_add_ps(t3, _mm_mul_ps(x3, y));
            }

            _mm_store_ps(&C[i + 0], t0);
            _mm_store_ps(&C[i + 4], t1);
            _mm_store_ps(&C[i + 8], t2);
            _mm_store_ps(&C[i + 12], t3);
        }
    }

    virtual void RetrieveColumn(
        float* C,
        uint32_t C_M,
        uint32_t j) const
    {
        rassert_eq(M, C_M);
        for (size_t i = 0; i < M; i++)
            C[i] = GetElement(i, j);
    }

    const float* GetBlock(size_t i, size_t j) const
    {
        auto ni = i / M_Block;
        auto nj = j / N_Block;
        return &m_x[(ni * N_Padded + nj * N_Block) * M_Block];
    }

private:
    const float& GetElement(size_t i, size_t j) const
    {
        return ((CMatrix*) this)->GetElement(i, j);
    }

    float& GetElement(size_t i, size_t j)
    {
        auto di = i % M_Block;
        auto dj = j % N_Block;
        auto block = (float*) GetBlock(i, j);
        return block[di * N_Block + dj];
    }

private:
    static constexpr size_t ALIGN = M_Block * sizeof(float);
    std::unique_ptr<uint8_t[]> m_store;
    float* m_x;
};

//
// Quantized weight matrix
// T can be int8_t or int16_t
//
template <class T, size_t _N_Block = 2, size_t _Ny_Block = 8>
class CQMatrix : public IMatrix
{
public:
    CQMatrix(uint32_t m, uint32_t n)
        : IMatrix(m, n),
          M_Padded((m + M_Block - 1) / M_Block * M_Block),
          N_Padded((n + N_Block - 1) / N_Block * N_Block),
          Ny_Padded((n + Ny_Block - 1) / Ny_Block * Ny_Block),
          m_store(std::make_unique<uint8_t[]>(sizeof(T) * M_Padded * N_Padded + ALIGN)),
          m_a_store(std::make_unique<uint8_t[]>(sizeof(float) * M_Padded + SIMD_ALIGN)),
          m_b_store(std::make_unique<uint8_t[]>(sizeof(float) * M_Padded + SIMD_ALIGN)),
          m_x(__align<T>(m_store.get(), M_Padded * N_Padded, ALIGN)),
          m_a(__align<float>(m_a_store.get(), M_Padded, SIMD_ALIGN)),
          m_b(__align<float>(m_b_store.get(), M_Padded, SIMD_ALIGN))
    {
        rassert_eq(M_Padded % M_Block, 0u);
        rassert_op(M_Padded, >=, M);
        rassert_op(M_Padded - M, <, M_Block);

        rassert_eq(N_Padded % N_Block, 0u);
        rassert_op(N_Padded, >=, N);
        rassert_op(N_Padded - N, <, N_Block);

        rassert_eq(Ny_Padded % Ny_Block, 0u);
        rassert_op(Ny_Padded, >=, N);
        rassert_op(Ny_Padded - N, <, Ny_Block);

        rassert_op(N_Padded, <=, Ny_Padded);
    }

    static constexpr size_t M_Block = 16;
    const uint32_t M_Padded;
    static constexpr size_t N_Block = _N_Block;
    const uint32_t N_Padded;
    static constexpr size_t Ny_Block = _Ny_Block;
    const uint32_t Ny_Padded;

    static constexpr size_t ALIGN = M_Block * N_Block * sizeof(T);
    static constexpr size_t SIMD_ALIGN = 16;

    virtual std::future<bool> fread(FILE* fp, bool transpose)
    {
        auto A = new float[M * N];
        if (transpose)
        {
            auto buf = std::make_unique<float[]>(M * N);
            rassert_eq(M * N, ::fread(buf.get(), sizeof(float), M * N, fp));

            for (size_t i = 0; i < M; i++)
                for (size_t j = 0; j < N; j++)
                    A[i * N + j] = buf[j * M + i];
        }
        else
        {
            rassert_eq(M * N, ::fread(A, sizeof(float), M * N, fp));
        }

        return std::async(
            std::launch::async,
            [this, A]() {
                this->init(A);
                delete[] A;
                return false;
            });
    }
    virtual std::future<bool> mread(float* buf, bool transpose)
    {
        auto A = new float[M * N];
        if (transpose)
        {
            for (size_t i = 0; i < M; i++)
                for (size_t j = 0; j < N; j++)
                    A[i * N + j] = buf[j * M + i];
        }
        else
        {
            memcpy(A, buf, sizeof(float) * M * N);
        }

        return std::async(
            std::launch::async,
            [this, A]() {
                this->init(A);
                delete[] A;
                return false;
            });
    }

    virtual void fwrite(FILE* fp)
    {
        fp;
        rfail();
    }

    virtual std::unique_ptr<IMatrix> Times(const IMatrix* that) const
    {
        rfail();
    }

    // C = A * B
    virtual void TimesVector(
        float* C,
        uint32_t C_M,
        uint32_t C_M_Padded,
        const float* B,
        uint32_t B_M,
        uint32_t B_M_Padded) const
    {
        rassert_eq(M, C_M);
        rassert_op(M_Padded, <=, C_M_Padded);
        rassert_eq(N, B_M);
        rassert_op(N_Padded, <=, B_M_Padded);
        rassert_op(Ny_Padded, <=, B_M_Padded);

        typedef int16_t Ty;
        static constexpr float MinTy = std::numeric_limits<Ty>::lowest();
        static constexpr float MaxTy = std::numeric_limits<Ty>::max();

        auto minmax = std::minmax_element(&B[0], &B[N]);
        auto c = std::max(*minmax.second / MaxTy, *minmax.first / MinTy);

        auto c_recip = c == 0.0f ? 1.0f : 1.0f / c;

        #ifdef LINUXRUNTIMECODE
        auto y1 = alloca(Ny_Padded * sizeof(Ty) + SIMD_ALIGN);
        #else
        auto y1 = _alloca(Ny_Padded * sizeof(Ty) + SIMD_ALIGN);
        #endif

        auto y = __align<Ty>(y1, Ny_Padded, SIMD_ALIGN);

        // float sum_B_k = 0;
        // for (size_t k = 0; k < N; k++)
        // {
        //     sum_B_k += B[k];
        //     auto y_k = roundf(B[k] * c_recip);
        //     rassert_op(MinTy, <=, y_k);
        //     rassert_op(y_k, <=, MaxTy);
        //     y[k] = (Ty)y_k;
        //     rassert_eq((float)y[k], y_k);
        // }

        auto cc_recip = _mm_set_ps1(c_recip);
        auto ssB0 = _mm_setzero_ps();
        auto ssB1 = _mm_setzero_ps();
        static_assert(Ny_Block % 8 == 0, "CQMatrix::Ny_Block not multiples of 8");
        rassert_eq(Ny_Padded % 8, uint32_t(0));
        for (size_t k = N; k < Ny_Padded; k++)
            rassert_eq(B[k], 0);
        for (size_t k = 0; k < Ny_Padded; k += 8)
        {
            auto B_k0 = _mm_load_ps(&B[k + 0]);
            auto B_k1 = _mm_load_ps(&B[k + 4]);

            ssB0 = _mm_add_ps(ssB0, B_k0);
            ssB1 = _mm_add_ps(ssB1, B_k1);

            auto y_k0 = _mm_mul_ps(B_k0, cc_recip);
            auto y_k1 = _mm_mul_ps(B_k1, cc_recip);

            y_k0 = _mm_round_ps(y_k0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            y_k1 = _mm_round_ps(y_k1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            #ifdef LINUXRUNTIMECODE
            static_assert(std::is_same<Ty, int16_t>::value, "Ty is not int16_t");
            #else
            static_assert(std::is_same_v<Ty, int16_t>, "Ty is not int16_t");
            #endif
            auto t0 = _mm_cvtps_epi32(y_k0);
            auto t1 = _mm_cvtps_epi32(y_k1);

            _mm_store_si128((__m128i*) &y[k], _mm_packs_epi32(t0, t1));
        }
        auto ss = _mm_hadd_ps(ssB0, ssB1);
        ss = _mm_hadd_ps(ss, ss);
        ss = _mm_hadd_ps(ss, ss);

        constexpr int BITS = 8 * sizeof(T) - 8;
        c *= 1 << BITS;

        // for (size_t i = 0; i < M; i++)
        // {
        //     int32_t t = 0;
        //     for (size_t k = 0; k < N; k++)
        //         t += ((int32_t)GetElement(i, k) * (int32_t)y[k]) >> BITS;
        //     C[i] = m_a[i] * c * t + m_b[i] * sum_B_k;
        // }

        auto cc = _mm_set1_ps(c);
        static_assert(M_Block == 16, "CQMatrix::M_Block != 16");
        for (size_t i = 0; i < M_Padded; i += M_Block)
        {
            static_assert(N_Block % 2 == 0, "CQMatrix::N_Block not multiples of 2");
            const T* x_i = GetBlock(i, 0);

            auto zero = _mm_setzero_si128();
            auto t0 = zero;
            auto t1 = zero;
            auto t2 = zero;
            auto t3 = zero;
            for (size_t k = 0; k < N_Padded; k += 2)
            {
                auto yk2 = *(int*) &y[k]; // 2 x epi16
                if (yk2 == 0)
                    continue;
                auto y_k = _mm_set1_epi32(yk2);

                auto xy0 = _mm_madd_epi16(y_k, load_qmatrix_block(&x_i[M_Block * k + 8 * 0]));
                auto xy1 = _mm_madd_epi16(y_k, load_qmatrix_block(&x_i[M_Block * k + 8 * 1]));
                auto xy2 = _mm_madd_epi16(y_k, load_qmatrix_block(&x_i[M_Block * k + 8 * 2]));
                auto xy3 = _mm_madd_epi16(y_k, load_qmatrix_block(&x_i[M_Block * k + 8 * 3]));

                if (BITS != 0)
                {
                    xy0 = _mm_srai_epi32(xy0, BITS);
                    xy1 = _mm_srai_epi32(xy1, BITS);
                    xy2 = _mm_srai_epi32(xy2, BITS);
                    xy3 = _mm_srai_epi32(xy3, BITS);
                }

                t0 = _mm_add_epi32(t0, xy0);
                t1 = _mm_add_epi32(t1, xy1);
                t2 = _mm_add_epi32(t2, xy2);
                t3 = _mm_add_epi32(t3, xy3);
            }

            store_qmatrix_result(C, i + 4 * 0, cc, t0, ss);
            store_qmatrix_result(C, i + 4 * 1, cc, t1, ss);
            store_qmatrix_result(C, i + 4 * 2, cc, t2, ss);
            store_qmatrix_result(C, i + 4 * 3, cc, t3, ss);
        }
    }

    virtual void RetrieveColumn(
        float* C,
        uint32_t C_M,
        uint32_t j) const
    {
        C;
        C_M;
        j;
        rfail("Not supported\n");
    }

protected:
    virtual void init(const float* A)
    {
        for (size_t i = 0; i < M; i++)
        {
            auto minmax = std::minmax_element(
                &A[i * N],
                &A[i * N + N]);
            m_a[i] = (*minmax.second - *minmax.first) / (MaxT - MinT);
            m_b[i] = (*minmax.first * MaxT - *minmax.second * MinT) / (MaxT - MinT);

            auto a_i_recip = m_a[i] == 0.0f ? 1.0f : 1.0f / m_a[i];
            for (size_t j = 0; j < N; j++)
            {
                auto A_ij = A[i * N + j];
                auto x_ij = roundf((A_ij - m_b[i]) * a_i_recip);
                rassert_op(MinT, <=, x_ij);
                rassert_op(x_ij, <=, MaxT);

                GetElement(i, j) = (T) x_ij;
                rassert_eq((float) GetElement(i, j), x_ij);
            }
        }
    }

    const T* GetBlock(size_t i, size_t j) const
    {
        auto ni = i / M_Block;
        auto nj = j / N_Block;
        return &m_x[(ni * N_Padded + nj * N_Block) * M_Block];
    }

private:
    const T& GetElement(size_t i, size_t j) const
    {
        return ((CQMatrix<T>*) this)->GetElement(i, j);
    }

    T& GetElement(size_t i, size_t j)
    {
        auto di = i % M_Block;
        auto dj = j % N_Block;
        auto block = (T*) GetBlock(i, j);
        return block[di * N_Block + dj];
    }

    static __m128i load_qmatrix_block(const T* addr)
    {

        #ifdef LINUXRUNTIMECODE

        static_assert(
            std::is_same<T, int8_t>::value || std::is_same<T, int16_t>::value,
            "quantization not supported");

        if (std::is_same<T, int16_t>::value)
            return _mm_load_si128((__m128i*) addr);
        #else
        static_assert(
            std::is_same_v<T, int8_t> || std::is_same_v<T, int16_t>,
            "quantization not supported");

        if (std::is_same_v<T, int16_t>)
            return _mm_load_si128((__m128i*) addr);
        #endif
        else
            return _mm_cvtepi8_epi16(_mm_loadl_epi64((__m128i*) addr));
    }

    // C[i] = m_a[i] * c * t + m_b[i] * sum_B_k;
    void store_qmatrix_result(float* C, size_t i, __m128 cc, __m128i tt, __m128 ss) const
    {
        auto m_a_c_t = _mm_mul_ps(
            _mm_mul_ps(_mm_load_ps(&m_a[i]), cc),
            _mm_cvtepi32_ps(tt));
        auto m_b_ss = _mm_mul_ps(_mm_load_ps(&m_b[i]), ss);
        _mm_store_ps(&C[i], _mm_add_ps(m_a_c_t, m_b_ss));
    }

protected:
    // A_ij = a_i * x_ij + b_i;
    static constexpr float MinT = std::numeric_limits<T>::lowest();
    static constexpr float MaxT = std::numeric_limits<T>::max();
    std::unique_ptr<uint8_t[]> m_store;
    std::unique_ptr<uint8_t[]> m_a_store;
    std::unique_ptr<uint8_t[]> m_b_store;
    T* m_x;
    float* m_a;
    float* m_b;
};

typedef CQMatrix<int16_t> CQ16Matrix;
typedef CQMatrix<int8_t> CQ8Matrix;

//
// Quantized weight matrix, specialized for 8-bit and SSE
//
//   A_ij = a_i (x_ij + b_i + dx_ij)
//   B_j  = c_i (y_j  + d   + dy_j )
//
//     sum_j A_ij B_j
//   = a_i c sum_j (x_ij + dx_ij) (y_j + dy_j) +
//     a_i c d sum_j (x_ij + dx_ij) +
//     a_i b_i sum_j B_j
//   ~ a_i (c (sum_j x_ij y_j + d r_i) + b_i sum_j B_j)
//
//   where r_i = sum_j (x_ij + dx_ij)
//
// Key SSE instructions we use:
//
//   - (SSSE3) _mm_maddubs_epi16: used to multiply 8-bit integers from x and y
//   - _mm_sad_epu8: used to horizontally add 16-bit integers
//
// Details about usage of the two instructions:
//
//   - Signed mulplication using _mm_maddubs_epi16
//
//     The instruction does multiplication and addition of two pairs of
//     8-bit unsigned integers with saturation.
//
//     While we can let y_j range from [0, 255), the upper half (128, 255)
//     can potentially saturate _mm_maddubs_epi16 and thus cause significant
//     errors.
//
//     To avoid this, we use 8-bit signed integers for both x_ij and y_j,
//     and use _mm_abs_epi8 to get absolute value of y_j, and use
//     _mm_sign_epi8 to transfer the sign of y_j to x_ij.  See below:
//
//       x1 y1 + x2 y2 = x1 sign(y1) |y1| + x2 sign(y2) |y2|
//
//       auto x  = _mm_load_si128( address of x_ij )
//       auto sy = _mm_load_si128( address of y_j )
//       auto yy = _mm_abs_epi8(sy);
//       auto xy = _mm_maddubs_epi16(yy, _mm_sign_epi8(x, sy));
//
//     We will carefully reuse yy across multiple rows, thus the extra cost is
//     mostly one _mm_sign_epi8 per _mm_maddubs_epi16.
//
//   - 16-bit signed horizontal addition using _mm_sad_epu8
//
//     We add 0x8000 to each signed 16-bit integer and add the result 16-bit integers
//     by high 8-bit and low 8-bit separately.
//
//       sum_i s16_i
//     = sum_i u16_i - 0x8000 * N
//     = 0x100 * sum_i h16_i + sum_i l16_i - 0x8000 * N
//
//     where u16_i = s16_i + 0x8000, and u16_i = 0x100 * h16_i + l16_i
//
class CSSE_Q8Matrix : public CQMatrix<int8_t, 16, 16>
{
public:
    CSSE_Q8Matrix(uint32_t m, uint32_t n)
        : CQMatrix(m, n),
          m_r_store(std::make_unique<uint8_t[]>(sizeof(float) * M_Padded + SIMD_ALIGN)),
          m_r(__align<float>(m_r_store.get(), M_Padded, SIMD_ALIGN))
    {
    }

    // C = A * B
    virtual void TimesVector(
        float* C,
        uint32_t C_M,
        uint32_t C_M_Padded,
        const float* B,
        uint32_t B_M,
        uint32_t B_M_Padded) const
    {
        rassert_eq(M, C_M);
        rassert_op(M_Padded, <=, C_M_Padded);
        rassert_eq(N, B_M);
        rassert_op(N_Padded, <=, B_M_Padded);
        rassert_op(Ny_Padded, <=, B_M_Padded);

        typedef int8_t Ty;
        static constexpr float MinTy = -127; // so that it can be negated safely
        static constexpr float MaxTy = 127;

        float Bmin, Bmax;
        min_max(B, N, Bmin, Bmax);

        auto c_recip = (MaxTy - MinTy) / (Bmax - Bmin);
        auto c = 1.0f / c_recip;
        auto d = (Bmin * MaxTy - Bmax * MinTy) / (Bmax - Bmin);

        // start to add the fix mentioned in Hao's email
        if (Bmax == Bmin)
        {
            c_recip = 1.0f;
            c = 1.0f;
            d = Bmax;
        }
        // end to add the fix mentined in Hao's email: "Re: error if model is not optimized", 3/18/2020
        // TODO: handle werid vector
        rassert_eq(isnormal(c_recip), true);
        rassert_eq(isnormal(c), true);
        rassert_eq(d == 0 || isnormal(d), true);

        #ifdef LINUXRUNTIMECODE
        auto y1 = alloca(Ny_Padded * sizeof(Ty) + SIMD_ALIGN);
        #else
        auto y1 = _alloca(Ny_Padded * sizeof(Ty) + SIMD_ALIGN);
        #endif
        auto y = __align<Ty>(y1, Ny_Padded, SIMD_ALIGN);

        // float s = 0;
        // for (size_t j = 0; j < N; j++)
        // {
        //     s += B[j];
        //     auto y_j = roundf(c_recip * B[j] - d);
        //     rassert_op(MinTy, <=, y_j);
        //     rassert_op(y_j, <=, MaxTy);
        //     y[j] = (Ty)y_j;
        //     rassert_eq((float)y[j], y_j);
        // }
        // auto ss = _mm_set1_ps(s);
        static_assert(Ny_Block == 16, "CQMatrix::Ny_Block != 16");
        rassert_eq(Ny_Padded % 16, uint32_t(0));
        auto ssB0 = _mm_setzero_ps();
        auto ssB1 = _mm_setzero_ps();
        auto ssB2 = _mm_setzero_ps();
        auto ssB3 = _mm_setzero_ps();
        auto dd = _mm_set1_ps(d);
        auto cc_recip = _mm_set1_ps(c_recip);
        for (size_t k = N; k < Ny_Padded; k++)
            rassert_eq(B[k], 0);
        for (size_t k = 0; k < Ny_Padded; k += 16)
        {
            auto B_k0 = _mm_load_ps(&B[k + 0]);
            auto B_k1 = _mm_load_ps(&B[k + 4]);
            auto B_k2 = _mm_load_ps(&B[k + 8]);
            auto B_k3 = _mm_load_ps(&B[k + 12]);

            ssB0 = _mm_add_ps(ssB0, B_k0);
            ssB1 = _mm_add_ps(ssB1, B_k1);
            ssB2 = _mm_add_ps(ssB2, B_k2);
            ssB3 = _mm_add_ps(ssB3, B_k3);

            auto y_k0 = _mm_sub_ps(_mm_mul_ps(cc_recip, B_k0), dd);
            auto y_k1 = _mm_sub_ps(_mm_mul_ps(cc_recip, B_k1), dd);
            auto y_k2 = _mm_sub_ps(_mm_mul_ps(cc_recip, B_k2), dd);
            auto y_k3 = _mm_sub_ps(_mm_mul_ps(cc_recip, B_k3), dd);

            y_k0 = _mm_round_ps(y_k0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            y_k1 = _mm_round_ps(y_k1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            y_k2 = _mm_round_ps(y_k2, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            y_k3 = _mm_round_ps(y_k3, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            #ifdef LINUXRUNTIMECODE
            static_assert(std::is_same<Ty, int8_t>::value, "Ty is not int8_t");
            #else
            static_assert(std::is_same_v<Ty, int8_t>, "Ty is not int8_t");
            #endif
            auto t0 = _mm_cvtps_epi32(y_k0);
            auto t1 = _mm_cvtps_epi32(y_k1);
            auto t2 = _mm_cvtps_epi32(y_k2);
            auto t3 = _mm_cvtps_epi32(y_k3);
            auto t01 = _mm_packs_epi32(t0, t1);
            auto t23 = _mm_packs_epi32(t2, t3);

            _mm_store_si128((__m128i*) &y[k + 0], _mm_packs_epi16(t01, t23));
        }
        auto ssB01 = _mm_hadd_ps(ssB0, ssB1);
        auto ssB23 = _mm_hadd_ps(ssB2, ssB3);
        auto ss = _mm_hadd_ps(ssB01, ssB23);
        ss = _mm_hadd_ps(ss, ss);
        ss = _mm_hadd_ps(ss, ss);

        // auto s = _mm_cvtss_f32(ss);
        // for (size_t i = 0; i < M; i++)
        // {
        //     int32_t t = 0;
        //     for (size_t j = 0; j < N; j++)
        //         t += get_element(i, j) * y[j];
        //     C[i] = m_a[i] * (c * (t + m_r[i] * d) + m_b[i] * s);
        // }
        auto cc = _mm_set1_ps(c);
        static_assert(M_Block == 16, "M_Block != 16");
        static_assert(M_SubBlock == 1, "M_SubBlock != 1");
        for (size_t i = 0; i < M_Padded; i += M_Block)
        {
            constexpr size_t N_MetaBlock = 512;
            static_assert(N_MetaBlock % N_Block == 0, "N_MetaBlock not multiples of N_Block");
            static_assert(N_Block == 16, "N_Block != 16");
            static_assert(N_SubBlock == 16, "N_SubBlock != 16");
            const auto x_i = GetBlock(i, 0);

            auto zero = _mm_setzero_si128();
            auto t0 = zero;
            auto t1 = zero;
            auto t2 = zero;
            auto t3 = zero;
            for (size_t k0 = 0; k0 < N_Padded; k0 += N_MetaBlock)
            {
                auto s0 = zero;
                auto s1 = zero;
                auto s2 = zero;
                auto s3 = zero;

                auto k1 = std::min(k0 + N_MetaBlock, (size_t) N_Padded);
                for (size_t k = k0; k < k1; k += N_Block)
                {
                    auto sy = _mm_load_si128((__m128i*) &y[k + 0]);
                    auto yy = _mm_abs_epi8(sy);

                    s0 = _mm_add_epi16(s0, sad_dp_block((__m128i*) &x_i[M_Block * k + 16 * 0], yy, sy));
                    s1 = _mm_add_epi16(s1, sad_dp_block((__m128i*) &x_i[M_Block * k + 16 * 4], yy, sy));
                    s2 = _mm_add_epi16(s2, sad_dp_block((__m128i*) &x_i[M_Block * k + 16 * 8], yy, sy));
                    s3 = _mm_add_epi16(s3, sad_dp_block((__m128i*) &x_i[M_Block * k + 16 * 12], yy, sy));
                }

                t0 = _mm_add_epi32(t0, sad_cvt_block(s0));
                t1 = _mm_add_epi32(t1, sad_cvt_block(s1));
                t2 = _mm_add_epi32(t2, sad_cvt_block(s2));
                t3 = _mm_add_epi32(t3, sad_cvt_block(s3));
            }

            auto uu = _mm_set1_epi32(0x8000 / 2 * N_Padded);
            t0 = _mm_sub_epi32(t0, uu);
            t1 = _mm_sub_epi32(t1, uu);
            t2 = _mm_sub_epi32(t2, uu);
            t3 = _mm_sub_epi32(t3, uu);

            store_sq8_result(C, i + 4 * 0, cc, dd, ss, t0);
            store_sq8_result(C, i + 4 * 1, cc, dd, ss, t1);
            store_sq8_result(C, i + 4 * 2, cc, dd, ss, t2);
            store_sq8_result(C, i + 4 * 3, cc, dd, ss, t3);
        }
    }

    virtual void RetrieveColumn(
        float* C,
        uint32_t C_M,
        uint32_t j) const
    {
        C;
        C_M;
        j;
        rfail("Not supported\n");
    }

protected:
    virtual void init(const float* A)
    {
        for (size_t i = 0; i < M; i++)
        {
            float Amin, Amax;
            min_max(&A[i * N], N, Amin, Amax);

            float MinT1 = -127; // so that it can be negated safely

            auto a_recip = (MaxT - MinT1) / (Amax - Amin);
            m_a[i] = 1.0f / a_recip;
            m_b[i] = (Amin * MaxT - Amax * MinT1) / (Amax - Amin);

            // start to add the fix simlar as we do Bmax and Bmin
            if (Amax == Amin)
            {
                a_recip = 1.0f;
                m_a[i] = 1.0f;
                m_b[i] = Amax;
            }
            // end to add the fix mentined in Hao's email: "Re: error if model is not optimized", 3/18/2020

            double r = 0;
            for (size_t j = 0; j < N; j++)
            {
                auto A_ij = A[i * N + j];
                auto a_ij = a_recip * A_ij - m_b[i];
                r += a_ij;

                auto x_ij = roundf(a_ij);
                //fprintf(stderr, "amax = %f, amin = %f, a_recip = %f, m_a[i] = %f, m_b[i] = %f, a_ij = %f, x_ij = %f \n",
                //        Amax, Amin, a_recip, m_a[i], m_b[i], a_ij, x_ij);
                
                rassert_op(MinT1, <=, x_ij);
                rassert_op(x_ij, <=, MaxT);

                get_element(i, j) = (int8_t) x_ij;
                rassert_eq((float) get_element(i, j), x_ij);
            }
            m_r[i] = (float) r;

            // TODO: handle werid weight matrix
            rassert_eq(isnormal(a_recip), true);
            rassert_eq(isnormal(m_a[i]), true);
            rassert_eq(m_b[i] == 0 || isnormal(m_b[i]), true);
            rassert_eq(m_r[i] == 0 || isnormal(m_r[i]), true);
        }
    }

private:
    const int8_t& get_element(size_t i, size_t j) const
    {
        return ((CSSE_Q8Matrix*) this)->get_element(i, j);
    }

    int8_t& get_element(size_t i, size_t j)
    {
        static_assert(M_Block % M_SubBlock == 0, "M_Block not multiples of M_SubBlock");
        static_assert(N_Block % N_SubBlock == 0, "N_Block not multiples of N_SubBlock");

        auto si = (i % M_Block) / M_SubBlock;
        auto sj = (j % N_Block) / N_SubBlock;

        auto di = i % M_SubBlock;
        auto dj = j % N_SubBlock;

        auto block = (int8_t*) GetBlock(i, j);
        return block[si * (M_SubBlock * N_Block) +
                     sj * (M_SubBlock * N_SubBlock) +
                     di * N_SubBlock +
                     dj];
    }

    static void min_max(const float* B, size_t N, float& Bmin, float& Bmax)
    {
        // Bmin = std::numeric_limits<float>::infinity();
        // Bmax = -std::numeric_limits<float>::infinity();
        // for (size_t i = 0; i < N; i++)
        // {
        //     if (B[i] < Bmin)
        //         Bmin = B[i];
        //     if (B[i] > Bmax)
        //         Bmax = B[i];
        // }
        auto s = _mm_set1_ps(std::numeric_limits<float>::infinity());
        auto t = _mm_set1_ps(-std::numeric_limits<float>::infinity());
        rassert_eq(N % 4, 0);
        for (size_t i = 0; i < N; i += 4)
        {
            auto B_i = _mm_loadu_ps(&B[i]);
            s = _mm_min_ps(s, B_i);
            t = _mm_max_ps(t, B_i);
        }

        auto u = _mm_movehl_ps(s, s);
        s = _mm_min_ps(s, u);
        u = _mm_movehdup_ps(s);
        s = _mm_min_ps(s, u);
        Bmin = _mm_cvtss_f32(s);

        u = _mm_movehl_ps(t, t);
        t = _mm_max_ps(t, u);
        u = _mm_movehdup_ps(t);
        t = _mm_max_ps(t, u);
        Bmax = _mm_cvtss_f32(t);
    }

    inline static __m128i sad_dp_block(const __m128i* x, __m128i yy, __m128i sy)
    {
        auto xa = _mm_load_si128((__m128i*) &x[0]);
        auto va = sad_dp_line(xa, yy, sy);

        auto xb = _mm_load_si128((__m128i*) &x[1]);
        auto vb = sad_dp_line(xb, yy, sy);

        auto xc = _mm_load_si128((__m128i*) &x[2]);
        auto vc = sad_dp_line(xc, yy, sy);

        auto xd = _mm_load_si128((__m128i*) &x[3]);
        auto vd = sad_dp_line(xd, yy, sy);

        return _mm_packus_epi32(
            _mm_packus_epi32(va, vb),
            _mm_packus_epi32(vc, vd));
    }

    inline static __m128i sad_dp_line(__m128i x, __m128i yy, __m128i sy)
    {
        auto _0x8000 = _mm_set1_epi16((short) 0x8000);
        //auto _0x8000 = _mm_set1_epi16((short) -32768);
        auto _sad_shuf = _mm_set_epi8(
            15, 13, 11, 9, 7, 5, 3, 1,
            14, 12, 10, 8, 6, 4, 2, 0);
        auto zero = _mm_setzero_si128();

        auto xy = _mm_maddubs_epi16(yy, _mm_sign_epi8(x, sy));
        auto t = _mm_xor_si128(xy, _0x8000);
        auto u = _mm_shuffle_epi8(t, _sad_shuf);
        return _mm_sad_epu8(u, zero);
    }

    inline static __m128i sad_cvt_block(__m128i s)
    {
        auto zero = _mm_setzero_si128();
        auto l = _mm_blend_epi16(s, zero, 0xaa);
        auto h = _mm_bsrli_si128(_mm_blend_epi16(s, zero, 0x55), 1);
        return _mm_add_epi32(l, h);
    }

    // C[i] = m_a[i] * (c * (t + m_r[i] * d) + m_b[i] * s);
    inline void store_sq8_result(float* C, size_t i, __m128 cc, __m128 dd, __m128 ss, __m128i t) const
    {
        auto r_dd = _mm_mul_ps(_mm_load_ps(&m_r[i]), dd);
        auto t_r_dd = _mm_add_ps(_mm_cvtepi32_ps(t), r_dd);
        auto cc_t_r_dd = _mm_mul_ps(cc, t_r_dd);

        auto b_ss = _mm_mul_ps(_mm_load_ps(&m_b[i]), ss);
        auto cc_t_r_dd_b_ss = _mm_add_ps(cc_t_r_dd, b_ss);

        auto c = _mm_mul_ps(_mm_load_ps(&m_a[i]), cc_t_r_dd_b_ss);

        _mm_store_ps(&C[i], c);
    }

    std::unique_ptr<uint8_t[]> m_r_store;
    float* m_r;

    // TODO No longer useful
    static constexpr size_t M_SubBlock = 1;
    static constexpr size_t N_SubBlock = 16;
};

static std::unique_ptr<IMatrix> make_unique_matrix(
    uint32_t m,
    uint32_t n,
    MatrixKind kind)
{
    switch (kind)
    {
    case MatrixKind::Float:
        return std::make_unique<CMatrix>(m, n);

    case MatrixKind::Q16:
        return std::make_unique<CQ16Matrix>(m, n);

    case MatrixKind::Q8:
        return std::make_unique<CQ8Matrix>(m, n);

    case MatrixKind::SSE_Q8:
        return std::make_unique<CSSE_Q8Matrix>(m, n);

    default:
        rfail("unknown matrix kind: %d\n", int(kind));
    }
}

static constexpr size_t MaxMatrix_M_Block =
    std::max(CMatrix::M_Block,
             std::max(CQ16Matrix::M_Block,
                      std::max(CQ8Matrix::M_Block,
                               CSSE_Q8Matrix::M_Block)));

static constexpr size_t MaxMatrix_N_Block =
    std::max(CMatrix::N_Block,
             std::max(CQ16Matrix::N_Block,
                      std::max(CQ8Matrix::N_Block,
                               CSSE_Q8Matrix::N_Block)));
