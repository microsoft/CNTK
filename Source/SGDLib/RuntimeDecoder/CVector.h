#pragma once

inline __m128 fast_expf(__m128 x)
{
    // -87.3365 <= x <= 88.7228
    // rassert_op(-87.3365f, <=, x);
    // rassert_op(x, <=, 88.7228f);

    auto C = _mm_set1_ps(12102203.16156148555068672305845f);
    auto D = _mm_set1_epi32(127 << 23);

    auto y = _mm_cvtps_epi32(_mm_mul_ps(C, x));
    auto z = _mm_add_epi32(y, D);

    // 2nd-order correction; relative error < 2.70e-3
    /*
    auto M = _mm_set1_epi32((1 << 23) - 1);
    auto A = _mm_set1_epi32(87);
    auto B = _mm_set1_epi32(-2 * 1024);

    auto d1 = _mm_srai_epi32(_mm_and_si128(y, M), 12);
    auto d2 = _mm_add_epi32(d1, B);
    auto dd = _mm_mullo_epi32(_mm_mullo_epi32(A, d1), d2);
    z = _mm_add_epi32(z, _mm_srai_epi32(dd, 7));
    */

    // 3rd-order correction; relative error < 1.12e-4
    /*
    auto M = _mm_set1_epi32((1 << 23) - 1);
    auto A = _mm_set1_epi32(-32 * 1024);
    auto B = _mm_set1_epi32(32 * 3993 - 7);

    auto d1 = _mm_srai_epi32(_mm_and_si128(y, M), 8);
    auto d2 = _mm_add_epi32(d1, A);
    auto d3 = _mm_add_epi32(d1, B);
    auto dd = _mm_mullo_epi32(
                  _mm_srai_epi32(_mm_mullo_epi32(d1, d2), 15),
                  d3);
    z = _mm_add_epi32(z, _mm_srai_epi32(dd, 11));
    z = _mm_add_epi32(z, _mm_srai_epi32(dd, 13));
    */

    // 4th-order correction; relative error < 9.15e-6
    auto M = _mm_set1_epi32((1 << 23) - 1);
    auto A1 = _mm_set1_epi32(1 - 128 * 1024);
    auto A2 = _mm_set1_epi32(1778);
    auto A3 = _mm_set1_epi32(8596);
    auto A4 = _mm_set1_epi32(20118);

    auto d1 = _mm_srai_epi32(_mm_and_si128(y, M), 6);
    auto d2 = _mm_srai_epi32(_mm_add_epi32(d1, A1), 1);
    auto d12 = _mm_srai_epi32(_mm_mullo_epi32(d1, d2), 15);

    auto d3 = _mm_add_epi32(_mm_srai_epi32(_mm_mullo_epi32(A2, d1), 17), A3);
    auto d34 = _mm_add_epi32(_mm_srai_epi32(_mm_mullo_epi32(d1, d3), 18), A4);

    auto dd = _mm_mullo_epi32(d12, d34);
    z = _mm_add_epi32(z, _mm_srai_epi32(dd, 11));

    return _mm_castsi128_ps(z);
}

float fast_expf(float x)
{
    // -87.3365 <= x <= 88.7228
    // rassert_op(-87.3365f, <=, x);
    // rassert_op(x, <=, 88.7228f);

    constexpr float C = 12102203.16156148555068672305845f;
    constexpr int D = 127 << 23;

    union {
        float f;
        int i;
    } z;

    auto y = (int)(C * x);
    z.i = y + D;

    // 2nd-order correction; relative error < 2.70e-3
    /*
    auto d1 = (y & ((1 << 23) - 1)) >> 12;
    auto d2 = d1 - 2 * 1024;
    z.i += (87 * d1 * d2) >> 7;
    */

    // 3rd-order correction; relative error < 1.12e-4
    /*
    auto d1 = (y & ((1 << 23) - 1)) >> 8;
    auto d2 = d1 - 32 * 1024;
    auto d3 = d1 + 32 * 3993 - 7;
    auto dd = ((d1 * d2) >> 15) * d3;
    z.i += (dd >> 11) + (dd >> 13);
    */

    // 4th-order correction; relative error < 9.15e-6
    auto d1 = (y & ((1 << 23) - 1)) >> 6;
    auto d2 = d1 + (1 - 128 * 1024);
    auto d12 = (d1 * (d2 >> 1)) >> 15;

    auto d3 = ((1778 * d1) >> 17) + 8596;
    auto d34 = ((d1 * d3) >> 18) + 20118;

    auto dd = d12 * d34;
    z.i += (dd >> 11);

    return z.f;
}

inline __m128 fast_sigmoid(__m128 x)
{
    // using 3rd-order correction in fast_expf, max absolute error is about 2.6e-5
    // using 4th-order correction in fast_expf, max absolute error is about 1.6e-6
    auto L = _mm_set1_ps(-87.3365f);
    auto R = _mm_set1_ps( 88.7228f);
    auto One = _mm_set1_ps(1.0f);

    x = _mm_max_ps(x, L);
    x = _mm_min_ps(x, R);

    auto e_x = fast_expf(x);
    return _mm_div_ps(e_x, _mm_add_ps(e_x, One));
}

float fast_sigmoid(float x)
{
    // using 3rd-order correction in fast_expf, max absolute error is about 2.6e-5
    // using 4th-order correction in fast_expf, max absolute error is about 1.6e-6
    if (x < -87.3365f)
        x = -87.3365f;

    if (x > 88.7228f)
        x = 88.7228f;

    auto e_x = fast_expf(x);
    return e_x / (e_x + 1.0f);
}

float sigmoid(float x)
{
    return 1.0f / (1.0f + expf(-x));
}


inline __m128 fast_tanhf(__m128 x)
{
    // using 3rd-order correction in fast_expf, max absolute error is about 5.2e-5
    // using 4th-order correction in fast_expf, max absolute error is about 3.2e-6
    auto x2 = _mm_add_ps(x, x);

    auto L = _mm_set1_ps(-87.3365f);
    auto R = _mm_set1_ps( 88.7228f);
    auto One = _mm_set1_ps(1.0f);

    x2 = _mm_max_ps(x2, L);
    x2 = _mm_min_ps(x2, R);

    auto e_x2 = fast_expf(x2);
    return _mm_div_ps(_mm_sub_ps(e_x2, One), _mm_add_ps(e_x2, One));
}


float fast_tanhf(float x)
{
    // using 3rd-order correction in fast_expf, max absolute error is about 5.2e-5
    // using 4th-order correction in fast_expf, max absolute error is about 3.2e-6
    auto x2 = 2 * x;

    if (x2 < -87.3365f)
        x2 = -87.3365f;

    if (x2 > 88.7228f)
        x2 = 88.7228f;

    auto e_x2 = fast_expf(x2);
    return (e_x2 - 1.0f) / (e_x2 + 1.0f);
}

//
// Vector (Mx1 matrix)
//
class CVector
{
public:
    CVector()
        : M_Padded(0),
          M(0),
          m_x(nullptr)
    {
    }

    CVector(uint32_t m)
        : M_Padded((m + M_Block - 1) / M_Block * M_Block),
          M(m),
          m_heap_x(std::make_unique<uint8_t[]>(sizeof(float) * M_Padded + SIMD_ALIGN)),
          m_x(__align<float>(m_heap_x.get(), M_Padded, SIMD_ALIGN))
    {
        rassert_eq(M_Padded % M_Block, 0u);
        rassert_op(M_Padded, >=, M);
        rassert_op(M_Padded - M, <, M_Block);
    }

#define _stack_CVector(A, m) \
    CVector A( \
        (float*)_alloca(((m) + CVector::M_Block - 1) / CVector::M_Block * CVector::M_Block * sizeof(float) + CVector::SIMD_ALIGN), \
        (m));

    CVector(float* x, uint32_t m)
        : M_Padded((m + M_Block - 1) / M_Block * M_Block),
          M(m),
          m_x(__align<float>(x, M_Padded, SIMD_ALIGN))
    {
        rassert_eq(M_Padded % M_Block, 0u);
        rassert_op(M_Padded, >=, M);
        rassert_op(M_Padded - M, <, M_Block);

        for (size_t i = M; i < M_Padded; i++)
            m_x[i] = 0;
    }

    CVector(CVector&& that)
        : M_Padded(that.M_Padded),
          M(that.M),
          m_heap_x(std::move(that.m_heap_x)),
          m_x(that.m_x)
    {
        (uint32_t&)that.M_Padded = 0;
        (uint32_t&)that.M = 0;
        that.m_x = nullptr;
    }

    void operator=(CVector&& that)
    {
        (uint32_t&)M_Padded = that.M_Padded;
        (uint32_t&)M = that.M;
        m_heap_x = std::move(that.m_heap_x);
        m_x = that.m_x;

        (uint32_t&)that.M_Padded = 0;
        (uint32_t&)that.M = 0;
        that.m_x = nullptr;
    }

    static constexpr size_t M_Block = std::max(MaxMatrix_M_Block, MaxMatrix_N_Block);
    static constexpr size_t SIMD_ALIGN = 16;
    const uint32_t M_Padded;
    const uint32_t M;

    float& operator[](size_t i)
    {
        return m_x[i];
    }

    const float& operator[](size_t i) const
    {
        return m_x[i];
    }

    void Set(const CVector& A)
    {
        rassert_eq(A.M, M);
        for (size_t i = 0; i < M; i++)
            (*this)[i] = A[i];
    }

    float Sum() const
    {
        // float sum = 0;
        // for (size_t i = 0; i < M; i++)
        //    sum += (*this)[i];
        // return sum;
        static_assert(M_Block == 16, "CVector::M_Block is not 16");
        auto t0 = _mm_setzero_ps();
        auto t1 = _mm_setzero_ps();
        auto t2 = _mm_setzero_ps();
        auto t3 = _mm_setzero_ps();
        for (size_t i = 0; i < M_Padded; i += 16)
        {
            t0 = _mm_add_ps(t0, _mm_load_ps(&m_x[i +  0]));
            t1 = _mm_add_ps(t1, _mm_load_ps(&m_x[i +  4]));
            t2 = _mm_add_ps(t2, _mm_load_ps(&m_x[i +  8]));
            t3 = _mm_add_ps(t3, _mm_load_ps(&m_x[i + 12]));
        }

        auto t01 = _mm_add_ps(t0, t1);
        auto t23 = _mm_add_ps(t2, t3);
        auto t = _mm_add_ps(t01, t23);
        t = _mm_hadd_ps(t, t);
        t = _mm_hadd_ps(t, t);
        return _mm_cvtss_f32(t);
    }

    float SqrSum() const
    {
        // float sum2 = 0;
        // for (size_t i = 0; i < M; i++)
        //    sum2 += (*this)[i] * (*this)[i];
        // return sum2;
        static_assert(M_Block == 16, "CVector::M_Block is not 16");
        auto t0 = _mm_setzero_ps();
        auto t1 = _mm_setzero_ps();
        auto t2 = _mm_setzero_ps();
        auto t3 = _mm_setzero_ps();
        for (size_t i = 0; i < M_Padded; i += 16)
        {
            auto x0 = _mm_load_ps(&m_x[i +  0]);
            auto x1 = _mm_load_ps(&m_x[i +  4]);
            auto x2 = _mm_load_ps(&m_x[i +  8]);
            auto x3 = _mm_load_ps(&m_x[i + 12]);

            t0 = _mm_add_ps(t0, _mm_mul_ps(x0, x0));
            t1 = _mm_add_ps(t1, _mm_mul_ps(x1, x1));
            t2 = _mm_add_ps(t2, _mm_mul_ps(x2, x2));
            t3 = _mm_add_ps(t3, _mm_mul_ps(x3, x3));
        }

        auto t01 = _mm_add_ps(t0, t1);
        auto t23 = _mm_add_ps(t2, t3);
        auto t = _mm_add_ps(t01, t23);
        t = _mm_hadd_ps(t, t);
        t = _mm_hadd_ps(t, t);
        return _mm_cvtss_f32(t);
    }

    float Max() const
    {
        // float max = (*this)[0];
        // for (size_t i = 0; i < M; i++)
        //     if (max < (*this)[i])
        //         max = (*this)[i];
        // return max;
        static_assert(M_Block == 16, "CVector::M_Block is not 16");
        auto inf = _mm_set1_ps(-std::numeric_limits<float>::infinity());
        auto t0 = inf;
        auto t1 = inf; 
        auto t2 = inf;
        auto t3 = inf;
        for (size_t i = 0; i < M_Padded; i += 16)
        {
            t0 = _mm_max_ps(t0, _mm_load_ps(&m_x[i +  0]));
            t1 = _mm_max_ps(t1, _mm_load_ps(&m_x[i +  4]));
            t2 = _mm_max_ps(t2, _mm_load_ps(&m_x[i +  8]));
            t3 = _mm_max_ps(t3, _mm_load_ps(&m_x[i + 12]));
        }

        auto t01 = _mm_max_ps(t0, t1);
        auto t23 = _mm_max_ps(t2, t3);
        auto t = _mm_max_ps(t01, t23);

        auto u = _mm_movehl_ps(t, t);
        t = _mm_max_ps(t, u);

        u = _mm_movehdup_ps(t);
        t = _mm_max_ps(t, u);

        return _mm_cvtss_f32(t);
    }

    void SetPlus(const CVector& A, const CVector& B)
    {
        rassert_eq(A.M, M);
        rassert_eq(B.M, M);
        // for (size_t i = 0; i < M; i++)
        //     (*this)[i] = A[i] + B[i];

        static_assert(M_Block == 16, "CVector::M_Block is not 16");
        for (size_t i = 0; i < M_Padded; i += 16)
        {
            auto t0 = _mm_add_ps(_mm_load_ps(&A[i +  0]), _mm_load_ps(&B[i +  0]));
            auto t1 = _mm_add_ps(_mm_load_ps(&A[i +  4]), _mm_load_ps(&B[i +  4]));
            auto t2 = _mm_add_ps(_mm_load_ps(&A[i +  8]), _mm_load_ps(&B[i +  8]));
            auto t3 = _mm_add_ps(_mm_load_ps(&A[i + 12]), _mm_load_ps(&B[i + 12]));

            _mm_store_ps(&(*this)[i +  0], t0);
            _mm_store_ps(&(*this)[i +  4], t1);
            _mm_store_ps(&(*this)[i +  8], t2);
            _mm_store_ps(&(*this)[i + 12], t3);
        }
    }

    void SetTimes(const IMatrix* A, const CVector& B)
    {
        A->TimesVector(&(*this)[0], M, M_Padded, &B[0], B.M, B.M_Padded);
    }

    void SetColumn(const IMatrix* A, uint32_t j)
    {
        A->RetrieveColumn(&(*this)[0], M, j);
    }

    void SetMinus(const CVector& A, float b)
    {
        rassert_eq(A.M, M);
        // for (size_t i = 0; i < M; i++)
        //     (*this)[i] = A[i] - b;

        static_assert(M_Block == 16, "CVector::M_Block is not 16");
        auto bb = _mm_set1_ps(b);
        for (size_t i = 0; i < M_Padded; i += 16)
        {
            auto t0 = _mm_sub_ps(_mm_load_ps(&A[i +  0]), bb);
            auto t1 = _mm_sub_ps(_mm_load_ps(&A[i +  4]), bb);
            auto t2 = _mm_sub_ps(_mm_load_ps(&A[i +  8]), bb);
            auto t3 = _mm_sub_ps(_mm_load_ps(&A[i + 12]), bb);

            _mm_store_ps(&(*this)[i +  0], t0);
            _mm_store_ps(&(*this)[i +  4], t1);
            _mm_store_ps(&(*this)[i +  8], t2);
            _mm_store_ps(&(*this)[i + 12], t3);
        }
    }

    void SetElement(float x)
    {
        for (size_t i = 0; i < M; i++)
            (*this)[i] = x;
    }

    void SetElementSquare(const CVector& A)
    {
        rassert_eq(A.M, M);
        for (size_t i = 0; i < M; i++)
            (*this)[i] = A[i] * A[i];
    }

    void SetElementDivide(const CVector& A, float b)
    {
        rassert_eq(A.M, M);
        // for (size_t i = 0; i < M; i++)
        //     (*this)[i] = A[i] / b;

        static_assert(M_Block == 16, "CVector::M_Block is not 16");
        auto bb = _mm_set1_ps(1.0f / b);
        for (size_t i = 0; i < M_Padded; i += 16)
        {
            auto t0 = _mm_mul_ps(_mm_load_ps(&A[i +  0]), bb);
            auto t1 = _mm_mul_ps(_mm_load_ps(&A[i +  4]), bb);
            auto t2 = _mm_mul_ps(_mm_load_ps(&A[i +  8]), bb);
            auto t3 = _mm_mul_ps(_mm_load_ps(&A[i + 12]), bb);

            _mm_store_ps(&(*this)[i +  0], t0);
            _mm_store_ps(&(*this)[i +  4], t1);
            _mm_store_ps(&(*this)[i +  8], t2);
            _mm_store_ps(&(*this)[i + 12], t3);
        }
    }

    void SetElementTimes(const CVector& A, const CVector& B)
    {
        rassert_eq(A.M, M);
        rassert_eq(B.M, M);
        // for (size_t i = 0; i < M; i++)
        //     (*this)[i] = A[i] * B[i];

        static_assert(M_Block == 16, "CVector::M_Block is not 16");
        for (size_t i = 0; i < M_Padded; i += 16)
        {
            auto t0 = _mm_mul_ps(_mm_load_ps(&A[i +  0]), _mm_load_ps(&B[i +  0]));
            auto t1 = _mm_mul_ps(_mm_load_ps(&A[i +  4]), _mm_load_ps(&B[i +  4]));
            auto t2 = _mm_mul_ps(_mm_load_ps(&A[i +  8]), _mm_load_ps(&B[i +  8]));
            auto t3 = _mm_mul_ps(_mm_load_ps(&A[i + 12]), _mm_load_ps(&B[i + 12]));

            _mm_store_ps(&(*this)[i +  0], t0);
            _mm_store_ps(&(*this)[i +  4], t1);
            _mm_store_ps(&(*this)[i +  8], t2);
            _mm_store_ps(&(*this)[i + 12], t3);
        }
    }

    void SetRowSlice(size_t M0, size_t dM, const CVector& A)
    {
        rassert_eq(dM, M);

        rassert_op(M0 + dM, <=, A.M);

        // for (size_t i = 0; i < M; i++)
        //     (*this)[i] = A[i + M0];
        memcpy(&(*this)[0], &A[M0], M * sizeof(A[0]));
    }

    void SetRowStack(const CVector& A, const CVector& B)
    {
        rassert_eq(A.M + B.M, M);

        // for (size_t i = 0; i < A.M; i++)
        //     (*this)[i] = A[i];
        memcpy(&(*this)[0], &A[0], A.M * sizeof(A[0]));

        // for (size_t i = 0; i < B.M; i++)
        //     (*this)[i + A.M] = B[i];
        memcpy(&(*this)[A.M], &B[0], B.M * sizeof(B[0]));
    }

    void SetSigmoid(const CVector& A)
    {
        rassert_eq(A.M, M);
        // for (size_t i = 0; i < M; i++)
        //     (*this)[i] = fast_sigmoid(A[i]);
        static_assert(M_Block == 16, "CVector::M_Block is not 16");
        for (size_t i = 0; i < M_Padded; i += 16)
        {
            auto t0 = fast_sigmoid(_mm_load_ps(&A[i +  0]));
            auto t1 = fast_sigmoid(_mm_load_ps(&A[i +  4]));
            auto t2 = fast_sigmoid(_mm_load_ps(&A[i +  8]));
            auto t3 = fast_sigmoid(_mm_load_ps(&A[i + 12]));

            _mm_store_ps(&(*this)[i +  0], t0);
            _mm_store_ps(&(*this)[i +  4], t1);
            _mm_store_ps(&(*this)[i +  8], t2);
            _mm_store_ps(&(*this)[i + 12], t3);
        }
    }

    void SetTanh(const CVector& A)
    {
        rassert_eq(A.M, M);
        // for (size_t i = 0; i < M; i++)
        //     (*this)[i] = fast_tanhf(A[i]);
        static_assert(M_Block == 16, "CVector::M_Block is not 16");
        for (size_t i = 0; i < M_Padded; i += 16)
        {
            auto t0 = fast_tanhf(_mm_load_ps(&A[i +  0]));
            auto t1 = fast_tanhf(_mm_load_ps(&A[i +  4]));
            auto t2 = fast_tanhf(_mm_load_ps(&A[i +  8]));
            auto t3 = fast_tanhf(_mm_load_ps(&A[i + 12]));

            _mm_store_ps(&(*this)[i +  0], t0);
            _mm_store_ps(&(*this)[i +  4], t1);
            _mm_store_ps(&(*this)[i +  8], t2);
            _mm_store_ps(&(*this)[i + 12], t3);
        }
    }

    void SetReLU(const CVector& A)
    {
        rassert_eq(A.M, M);
        for (size_t i = 0; i < M; i++)
            (*this)[i] = A[i] > 0 ? A[i] : 0;
    }

    void SetSoftMax(const CVector& A)
    {
        rassert_eq(A.M, M);

        auto max = A.Max();

        float sum = 0.0;
        static constexpr float exp_base = -24;
        static constexpr int exp_gap = 8;
        uint32_t exp_cnts[exp_gap] = { 0 };
        for (size_t i = 0; i < M; i++)
        {
            auto diff = A[i] - max;
            auto n = (int)(diff - exp_base);
            if (n >= exp_gap)
                sum += fast_expf(diff);
            else
                exp_cnts[n < 0 ? 0 : n]++;
        }
        static constexpr float exp_off = 0.541324855f;
        for (int n = 0; n < exp_gap; n++)
            sum += exp_cnts[n] * fast_expf(exp_base + exp_off + n);

        float denorm = max + log(sum);
        SetMinus(A, denorm);
    }

    void Print() const
    {
        for (size_t i = 0; i < M; i++)
            printf(i == 0 ? "%f" : " %f", (*this)[i]);
        printf("\n");
    }

private:
    std::unique_ptr<uint8_t[]> m_heap_x;
    float* m_x;
};
