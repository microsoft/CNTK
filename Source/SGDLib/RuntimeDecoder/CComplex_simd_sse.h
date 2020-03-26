#pragma once

// Disable SSE3 implementation for now. We will dynamically dispatch different SSE versions
// based on ISA availablity on the platform

// #define MAS_SSE3_AVAILABLE

#ifndef MAS_SSE3_AVAILABLE
FORCEINLINE __m128 mm_addsub_ps_sse2(__m128 a, __m128 b)
{
    __m128 d;
    d = a;                           // d = a3a2a1a0 
    a = _mm_sub_ps(a, b);            // a = (a3-b3),(a2-b2),(a1-b1),(a0-b0)
    d = _mm_add_ps(d, b);            // d = (a3+b3),(a2+b2),(a1+b1),(a0+b0)
    a = _mm_shuffle_ps(a, d, _MM_SHUFFLE(3, 1, 2, 0));  // a = (a3+b3),(a1+b1),(a2-b2),(a0-b0)
    a = _mm_shuffle_ps(a, a, _MM_SHUFFLE(3, 1, 2, 0));  // a = (a3+b3),(a2-b2),(a1+b1),(a0-b0)
    return a;
}

FORCEINLINE __m128 mm_hadd_ps_sse2(__m128 a, __m128 b)
{
    __m128 x;
    x = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 3, 0, 1)); // x = a2a3a0a1
    a = _mm_add_ps(a, x);                              // a = a32a32a01a01
    x = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1)); // x = b2b3b0b1
    b = _mm_add_ps(b, x);                              // a = b32b32b01b01
    a = _mm_shuffle_ps(a, b, _MM_SHUFFLE(2,0,2,0));    // a = b32b01a32a01
    return a;
}
#endif

// a * b (2x complex<float>)
FORCEINLINE __m128 complex_mul_ps(__m128 a, __m128 b)
{
#if defined(MAS_SSE3_AVAILABLE)
    auto t1 = _mm_mul_ps(_mm_moveldup_ps(a), b); //SSE3
    b = _mm_shuffle_ps(b, b, 0xb1);
    auto t2 = _mm_mul_ps(_mm_movehdup_ps(a), b);
    auto s = _mm_addsub_ps(t1, t2);
    return s;
#else
    __m128 c;
    c = a;                                             // c = a3a2a1a0
    a = _mm_shuffle_ps(a, a, _MM_SHUFFLE(2, 2, 0, 0)); // a = a2a2a0a0 == _mm_moveldup_ps(a)
    a = _mm_mul_ps(a, b);                              // a = a2b3,a2b2,a0b1,a0b0 == t1
    b = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 3, 0, 1)); // b = b2b3b0b1
    c = _mm_shuffle_ps(c, c, _MM_SHUFFLE(3, 3, 1, 1)); // c = a3a3a1a1 == _mm_movehdup_ps(a)
    c = _mm_mul_ps(c, b);                              // c = a3b2,a3b3,a1b0,a1b1 == t2
    a = mm_addsub_ps_sse2(a, c);
    return a;
#endif
}

// for (L1 <= k < L2)
//     x[k] *= a
inline void vector_mul_ps(
    std::complex<float>* x,
    float a,
    size_t L1,
    size_t L2)
{
    auto a1 = _mm_set1_ps(a);
    for (size_t k = L1 / 2 * 2; k < L2; k += 2)
    {
        auto xk = _mm_loadu_ps((const float*)&x[k]);
        xk = _mm_mul_ps(a1, xk);
        _mm_storeu_ps((float *)&x[k], xk);
    }
}

// for (0 <= k <L)
//     a[k] += alpha * (c[k] * c[k] - a[k])
//     b[k] += alpha * (d[k] * d[k] - b[k])
//     y[k] = x[k] * a[k] / (a[k] + b[k] + delta);
inline void vector_scale_ps(
    std::complex<float>* x,
    std::complex<float>* y,
    float* a,
    float* b,
    const float* c,
    const float* d,
    float alpha,
    float delta,
    size_t L)
{
    size_t k = 0;

    auto a1 = _mm_set1_ps(alpha);
    auto d1 = _mm_set1_ps(delta);
    for (; k + 4 <= L; k += 4)
    {
        auto ak = _mm_loadu_ps(&a[k]);
        auto bk = _mm_loadu_ps(&b[k]);

        auto ck = _mm_loadu_ps(&c[k]);
        auto dk = _mm_loadu_ps(&d[k]);

        auto cc = _mm_mul_ps(ck, ck);
        auto cca = _mm_sub_ps(cc, ak);
        auto acca = _mm_mul_ps(a1, cca);
        ak = _mm_add_ps(ak, acca);
        _mm_storeu_ps(&a[k], ak);

        auto dd = _mm_mul_ps(dk, dk);
        auto ddb = _mm_sub_ps(dd, bk);
        auto addb = _mm_mul_ps(a1, ddb);
        bk = _mm_add_ps(bk, addb);
        _mm_storeu_ps(&b[k], bk);

        auto ab = _mm_add_ps(ak, bk);
        auto abd = _mm_add_ps(ab, d1);
        auto aabd = _mm_div_ps(ak, abd);

        auto xk1 = _mm_loadu_ps((const float*)&x[k]);
        auto s1 = _mm_shuffle_ps(aabd, aabd, _MM_SHUFFLE(1, 1, 0, 0));
        xk1 = _mm_mul_ps(s1, xk1);
        _mm_storeu_ps((float*)&y[k], xk1);

        auto xk2 = _mm_loadu_ps((const float*)&x[k + 2]);
        auto s2 = _mm_shuffle_ps(aabd, aabd, _MM_SHUFFLE(3, 3, 2, 2));
        xk2 = _mm_mul_ps(s2, xk2);
        _mm_storeu_ps((float*)&y[k + 2], xk2);
    }

    while (k < L)
    {
        a[k] += alpha * (c[k] * c[k] - a[k]);
        b[k] += alpha * (d[k] * d[k] - b[k]);
        y[k] = x[k] * a[k] / (a[k] + b[k] + delta);
        k++;
    }
}

// for (L1 <= k < L2)
//     x[k] *= a
inline void float2_vector_mul_ps(
    float* x,
    float a,
    size_t L1,
    size_t L2)
{
    size_t k = L1 / 2 * 2;

    auto a1 = _mm_set1_ps(a);
    for (; k + 4 <= L2; k += 4)
    {
        auto xk = _mm_loadu_ps((const float*)&x[k]);
        xk = _mm_mul_ps(xk, a1);
        _mm_storeu_ps((float *)&x[k], xk);
    }

    for (; k < L2; k += 2)
    {
        auto xk = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)&x[k]));
        xk = _mm_mul_ps(xk, a1);
        _mm_storel_pi((__m64*)&x[k], xk);
    }
}

// for (L1 <= k < L2)
//     x[k] = a[k] * conj(b[k])
inline void vector_conjmul_ps(
    std::complex<float>* x,
    const float* a,
    const std::complex<float>* b,
    size_t L1,
    size_t L2)
{
    size_t k = L1 / 2 * 2;

    for (; k < L2; k += 2)
    {
        auto ak = _mm_set_ps(
            -a[k + 1],
            a[k + 1],
            -a[k],
            a[k]);
        auto bk = _mm_loadu_ps((const float*)&b[k]);

        auto xk = _mm_mul_ps(ak, bk);

        _mm_storeu_ps((float *)&x[k], xk);
    }
}

// for (L1 <= k < L2)
//     x[k] = a[k] * b[k]
inline void float4_vector_mul_ps(
    float* x,
    const float* a,
    const float* b,
    size_t L1,
    size_t L2)
{
    for (size_t j = L1 / 4 * 4; j < L2; j += 4)
    {
        auto aj = _mm_loadu_ps((const float*)&a[j]);
        auto bj = _mm_loadu_ps((const float*)&b[j]);
        auto xj = _mm_mul_ps(aj, bj);

        _mm_storeu_ps((float *)&x[j], xj);
    }
}

// for (0 <= k < L)
//     x[k] += a[k] * b[k]
inline void float_vector_madd_ps(
    float* x,
    const float* a,
    const float* b,
    size_t L
)
{
    size_t k = 0;
    for (; k + 4 <= L; k += 4)
    {
        auto ak = _mm_loadu_ps(&a[k]);
        auto bk = _mm_loadu_ps(&b[k]);
        auto ab = _mm_mul_ps(ak, bk);
        auto xk = _mm_loadu_ps(&x[k]);
        xk = _mm_add_ps(xk, ab);
        _mm_storeu_ps(&x[k], xk);
    }

    while (k < L)
    {
        x[k] += a[k] * b[k];
        k++;
    }
}

// for (0 <= k < L)
//     g = mu * a[k] / (b[k] + delta)
//     x[k] += g * c[k]
inline void float_vector_gd_ps(
    float* x,
    const float* a,
    const float* b,
    const float* c,
    float mu,
    float delta,
    size_t L
)
{
    size_t k = 0;

    auto mu1 = _mm_set1_ps(mu);
    auto d1 = _mm_set1_ps(delta);
    for (; k + 4 <= L; k += 4)
    {
        auto ak = _mm_loadu_ps(&a[k]);
        auto bk = _mm_loadu_ps(&b[k]);
        auto ck = _mm_loadu_ps(&c[k]);
        auto b1 = _mm_add_ps(bk, d1);
        auto ac = _mm_mul_ps(ak, ck);
        auto acm = _mm_mul_ps(mu1, ac);
        auto gd = _mm_div_ps(acm, b1);
        auto xk = _mm_loadu_ps(&x[k]);
        xk = _mm_add_ps(xk, gd);

        _mm_storeu_ps(&x[k], xk);
    }

    while (k < L)
    {
        x[k] += mu * a[k] * c[k] / (b[k] + delta);
        k++;
    }
}

// for (0 <= k < L)
//     sum += x[k] * y[k]
inline float float4_msum_ps(const float* x, const float* y, size_t L)
{
    __m128 result = _mm_set1_ps(0);
    for (size_t i = 0; i < L; i += 4)
    {
        auto xi = _mm_loadu_ps(&x[i]);
        auto yi = _mm_loadu_ps(&y[i]);
        auto xy = _mm_mul_ps(xi, yi);
        result = _mm_add_ps(result, xy);
    }

#ifdef __linux__
    ALIGNED(16) float r[4];
    _mm_storeu_ps(r, result);
    return r[0] + r[1] + r[2] + r[3];
#else
    return result.m128_f32[0] + result.m128_f32[1] + result.m128_f32[2] + result.m128_f32[3];
#endif
}

// for (L1 <= k < L2)
//     x[k] = a[k] + b[k]
inline void float4_vector_add_ps(
    float* x,
    const float* a,
    const float* b,
    size_t L1,
    size_t L2)
{
    for (size_t j = L1 / 4 * 4; j < L2; j += 4)
    {
        auto aj = _mm_loadu_ps((const float*)&a[j]);
        auto bj = _mm_loadu_ps((const float*)&b[j]);
        auto xj = _mm_add_ps(aj, bj);

        _mm_storeu_ps((float *)&x[j], xj);
    }
}

// for (L1 <= k < L2)
//     x[k] = a[k] - b[k]
inline void float_vector_sub_ps(
    float* x,
    const float* a,
    const float* b,
    size_t L1,
    size_t L2)
{
    for (size_t j = L1 / 4 * 4; j < L2; j += 4)
    {
        auto aj = _mm_loadu_ps((const float*)&a[j]);
        auto bj = _mm_loadu_ps((const float*)&b[j]);
        auto xj = _mm_sub_ps(aj, bj);

        _mm_storeu_ps((float *)&x[j], xj);
    }
}

// for (0 <= k < L)
//     x[k] = a[k] - b[k]
inline void float_vector_sub_ps2(
    float* x,
    const float* a,
    const float* b,
    size_t L)
{
    size_t k = 0;
    for (; k + 4 <= L; k += 4)
    {
        auto ak = _mm_loadu_ps((const float*)&a[k]);
        auto bk = _mm_loadu_ps((const float*)&b[k]);
        auto xk = _mm_sub_ps(ak, bk);

        _mm_storeu_ps((float *)&x[k], xk);
    }

    while (k < L)
    {
        x[k] = a[k] - b[k];
        k++;
    }
}

// for (L1 <= k < L2)
//     x[k].real += a[k].real * b[k].real
//     x[k].imag += a[k].imag * b[k].imag
inline void vector_madd_ps(
    std::complex<float>* x,
    const std::complex<float>* a,
    const std::complex<float>* b,
    size_t L1,
    size_t L2)
{
    for (size_t k = L1 / 2 * 2; k < L2; k += 2)
    {
        auto ak = _mm_loadu_ps((const float*)&a[k]);
        auto bk = _mm_loadu_ps((const float*)&b[k]);
        auto ab = _mm_mul_ps(ak, bk);

        auto xk = _mm_loadu_ps((const float*)&x[k]);
        xk = _mm_add_ps(xk, ab);
        _mm_storeu_ps((float *)&x[k], xk);
    }
}

inline float complex_norm(const std::complex<float>& x)
{
    return x.real() * x.real() + x.imag() * x.imag();
}

// for (L1 <= k < L2)
//     x[k] += std::norm(a[k])
inline void vector_normadd_ps(
    float* x,
    const std::complex<float>* a,
    size_t L1,
    size_t L2)
{
    size_t k = L1 / 2 * 2;
    for (; k + 4 <= L2; k += 4)
    {
        auto xk = _mm_loadu_ps((const float*)&x[k]);

        auto ak0 = _mm_loadu_ps((const float*)&a[k]);
        auto ak1 = _mm_loadu_ps((const float*)&a[k + 2]);
#if defined(MAS_SSE3_AVAILABLE)
        auto a2 = _mm_hadd_ps(_mm_mul_ps(ak0, ak0), _mm_mul_ps(ak1, ak1)); //SSE3
#else
        auto a2 = mm_hadd_ps_sse2(_mm_mul_ps(ak0, ak0), _mm_mul_ps(ak1, ak1));
#endif

        xk = _mm_add_ps(xk, a2);

        _mm_storeu_ps((float *)&x[k], xk);
    }

    for (; k < L2; k += 2)
    {
        auto xk = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)&x[k]));
        auto ak = _mm_loadu_ps((const float*)&a[k]);
#if defined(MAS_SSE3_AVAILABLE)
        auto a2 = _mm_hadd_ps(_mm_mul_ps(ak, ak), _mm_mul_ps(ak, ak)); //SSE3
#else
        auto a2 = mm_hadd_ps_sse2(_mm_mul_ps(ak, ak), _mm_mul_ps(ak, ak));
#endif
        xk = _mm_add_ps(xk, a2);

        _mm_storel_pi((__m64*)&x[k], xk);
    }
}

// for (L1 <= k < L2)
//     x[k] += std::norm(a[k])
//     y[k] += w[k] * std::norm(a[k])
inline void vector_normadd2_ps(
    float* x,
    float* y,
    const float* w,
    const std::complex<float>* a,
    size_t L1,
    size_t L2)
{
    size_t k = L1 / 2 * 2;
    for (; k + 4 <= L2; k += 4)
    {
        auto xk = _mm_loadu_ps((const float*)&x[k]);
        auto yk = _mm_loadu_ps((const float*)&y[k]);
        auto wk = _mm_loadu_ps((const float*)&w[k]);

        auto ak0 = _mm_loadu_ps((const float*)&a[k]);
        auto ak1 = _mm_loadu_ps((const float*)&a[k + 2]);
#if defined(MAS_SSE3_AVAILABLE)
        auto a2 = _mm_hadd_ps(_mm_mul_ps(ak0, ak0), _mm_mul_ps(ak1, ak1)); //SSE3
#else
        auto a2 = mm_hadd_ps_sse2(_mm_mul_ps(ak0, ak0), _mm_mul_ps(ak1, ak1));
#endif

        xk = _mm_add_ps(xk, a2);
        yk = _mm_add_ps(yk, _mm_mul_ps(wk, a2));

        _mm_storeu_ps((float *)&x[k], xk);
        _mm_storeu_ps((float *)&y[k], yk);
    }

    for (; k < L2; k += 2)
    {
        auto xk = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)&x[k]));
        auto yk = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)&y[k]));
        auto wk = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)&w[k]));

        auto ak = _mm_loadu_ps((const float*)&a[k]);
#if defined(MAS_SSE3_AVAILABLE)
        auto a2 = _mm_hadd_ps(_mm_mul_ps(ak, ak), _mm_mul_ps(ak, ak)); //SSE3
#else
        auto a2 = mm_hadd_ps_sse2(_mm_mul_ps(ak, ak), _mm_mul_ps(ak, ak));
#endif

        xk = _mm_add_ps(xk, a2);
        yk = _mm_add_ps(yk, _mm_mul_ps(wk, a2));

        _mm_storel_pi((__m64*)&x[k], xk);
        _mm_storel_pi((__m64*)&y[k], yk);
    }
}

// for (L1 <= k < L2)
//     x[k] = a[k] * b[k]
inline void complex_vector_mul_ps(
    std::complex<float>* x,
    const std::complex<float>* a,
    const std::complex<float>* b,
    size_t L1,
    size_t L2)
{
    for (size_t k = L1 / 2 * 2; k < L2; k += 2)
    {
        auto ak = _mm_loadu_ps((const float*)&a[k]);
        auto bk = _mm_loadu_ps((const float*)&b[k]);
        auto xk = complex_mul_ps(ak, bk);
        _mm_storeu_ps((float *)&x[k], xk);
    }
}

// for (L1 <= k < L2)
//     x[k] = a[k] + b[k]
inline void complex_vector_add_ps(
    std::complex<float>* x,
    const std::complex<float>* a,
    const std::complex<float>* b,
    size_t L1,
    size_t L2)
{
    for (size_t k = L1 / 2 * 2; k < L2; k += 2)
    {
        auto ak = _mm_loadu_ps((const float*)&a[k]);
        auto bk = _mm_loadu_ps((const float*)&b[k]);
        auto xk = _mm_add_ps(ak, bk);
        _mm_storeu_ps((float *)&x[k], xk);
    }
}

// for (L1 <= k < L2)
//     x[k] = a[k] + b[k]
inline void complex_vector_sub_ps(
    std::complex<float>* x,
    const std::complex<float>* a,
    const std::complex<float>* b,
    size_t L1,
    size_t L2)
{
    for (size_t k = L1 / 2 * 2; k < L2; k += 2)
    {
        auto ak = _mm_loadu_ps((const float*)&a[k]);
        auto bk = _mm_loadu_ps((const float*)&b[k]);
        auto xk = _mm_sub_ps(ak, bk);
        _mm_storeu_ps((float *)&x[k], xk);
    }
}

// for (L1 <= k < L2)
//     x[k] += a[k] * b[k]
inline void complex_vector_madd_ps(
    std::complex<float>* x,
    const std::complex<float>* a,
    const std::complex<float>* b,
    size_t L1,
    size_t L2)
{
    for (size_t k = L1 / 2 * 2; k < L2; k += 2)
    {
        auto ak = _mm_loadu_ps((const float*)&a[k]);
        auto bk = _mm_loadu_ps((const float*)&b[k]);
        auto ab = complex_mul_ps(ak, bk);

        auto xk = _mm_loadu_ps((const float*)&x[k]);
        xk = _mm_add_ps(xk, ab);
        _mm_storeu_ps((float *)&x[k], xk);
    }
}

// for (L1 <= k < L2)
//     x1[k] += a[k] * b1[k]
//     x2[k] += a[k] * b2[k]
inline void complex_vector2_madd_ps(
    std::complex<float>* x1,
    std::complex<float>* x2,
    const std::complex<float>* a,
    const std::complex<float>* b1,
    const std::complex<float>* b2,
    size_t L1,
    size_t L2)
{
    for (size_t k = L1 / 2 * 2; k < L2; k += 2)
    {
        auto ak = _mm_loadu_ps((const float*)&a[k]);
        auto b1k = _mm_loadu_ps((const float*)&b1[k]);
        auto b2k = _mm_loadu_ps((const float*)&b2[k]);
        auto ab1 = complex_mul_ps(ak, b1k);
        auto ab2 = complex_mul_ps(ak, b2k);

        auto x1k = _mm_loadu_ps((const float*)&x1[k]);
        auto x2k = _mm_loadu_ps((const float*)&x2[k]);
        x1k = _mm_add_ps(x1k, ab1);
        x2k = _mm_add_ps(x2k, ab2);
        _mm_storeu_ps((float *)&x1[k], x1k);
        _mm_storeu_ps((float *)&x2[k], x2k);
    }
}

// for (L1 <= k < L2)
//     x[k] -= a[k] * b[k]
inline void complex_vector_msub_ps(
    std::complex<float>* x,
    const std::complex<float>* a,
    const std::complex<float>* b,
    size_t L1,
    size_t L2)
{
    for (size_t k = L1 / 2 * 2; k < L2; k += 2)
    {
        auto ak = _mm_loadu_ps((const float*)&a[k]);
        auto bk = _mm_loadu_ps((const float*)&b[k]);
        auto ab = complex_mul_ps(ak, bk);

        auto xk = _mm_loadu_ps((const float*)&x[k]);
        xk = _mm_sub_ps(xk, ab);
        _mm_storeu_ps((float *)&x[k], xk);
    }
}

// for (L1 <= k < L2)
//     x1[k] -= a[k] * b1[k]
//     x2[k] -= a[k] * b2[k]
inline void complex_vector2_msub_ps(
    std::complex<float>* x1,
    std::complex<float>* x2,
    const std::complex<float>* a,
    const std::complex<float>* b1,
    const std::complex<float>* b2,
    size_t L1,
    size_t L2)
{
    for (size_t k = L1 / 2 * 2; k < L2; k += 2)
    {
        auto ak = _mm_loadu_ps((const float*)&a[k]);
        auto b1k = _mm_loadu_ps((const float*)&b1[k]);
        auto b2k = _mm_loadu_ps((const float*)&b2[k]);
        auto ab1 = complex_mul_ps(ak, b1k);
        auto ab2 = complex_mul_ps(ak, b2k);

        auto x1k = _mm_loadu_ps((const float*)&x1[k]);
        auto x2k = _mm_loadu_ps((const float*)&x2[k]);
        x1k = _mm_sub_ps(x1k, ab1);
        x2k = _mm_sub_ps(x2k, ab2);
        _mm_storeu_ps((float *)&x1[k], x1k);
        _mm_storeu_ps((float *)&x2[k], x2k);
    }
}

// for (L1 <= k < L2)
//    m[k] = abs(x[k])
inline void complex_vector_abs(
    std::complex<float>* x,
    float* m,
    size_t L1,
    size_t L2)
{
    size_t k = L1 / 2 * 2;

    for (; k + 4 <= L2; k += 4)
    {
        auto xk0 = _mm_loadu_ps((const float*)&x[k]);
        auto xk1 = _mm_loadu_ps((const float*)&x[k + 2]);
#if defined(MAS_SSE3_AVAILABLE)
        auto x2 = _mm_hadd_ps(_mm_mul_ps(xk0, xk0), _mm_mul_ps(xk1, xk1)); //SSE3
#else
        auto x2 = mm_hadd_ps_sse2(_mm_mul_ps(xk0, xk0), _mm_mul_ps(xk1, xk1));
#endif
        auto x3 = _mm_sqrt_ps(x2);
        _mm_storeu_ps((float*)&m[k], x3);
    }

    for (; k < L2; k++)
    {
        m[k] = sqrtf(complex_norm(x[k]));
    }
}

// for (L1 <= k < L2)
//    m[k] += alpha * (x[k] * x[k] - m[k])
inline void float_vector_power(
    float* x,
    float* m,
    float alpha,
    size_t L1,
    size_t L2
)
{
    size_t k = L1 / 2 * 2;

    auto a1 = _mm_set1_ps(alpha);
    for (; k + 4 <= L2; k += 4)
    {
        auto x1 = _mm_loadu_ps(&x[k]);
        auto x2 = _mm_mul_ps(x1, x1);
        auto x3 = _mm_loadu_ps(&m[k]);
        auto x4 = _mm_sub_ps(x2, x3);
        auto xk = _mm_mul_ps(x4, a1);
        auto xt = _mm_add_ps(x3, xk);
        _mm_storeu_ps((float*)&m[k], xt);
    }

    for (; k < L2; k++)
    {
        m[k] += alpha * (x[k] * x[k] - m[k]);
    }
}

template<class T, size_t L1, size_t L2>
class ALIGNED(16) CSimdArray
{
public:
    UNIMIC_ALIGNED_NEW(CSimdArray);

    CSimdArray(bool init = true)
    {
        if (init)
            memset(m_buf, 0, sizeof(m_buf));
    }

    const T& operator[](size_t i) const
    {
        return ((CSimdArray&)*this)[i];
    }

    T& operator[](size_t i)
    {
        // rassert_op(L1, <=, i);
        // rassert_op(i, <, L2);
        // rassert_op(i - L1a, <, _countof(m_x));
        auto x = (T*)m_buf;
        return x[i - L1a];
    }

private:
    static constexpr size_t L1a = L1 / 2 * 2;

    static_assert(L1 <= L2, "invalid bounds");
    static_assert(L1a <= L2 + 1, "invalid bounds");
    ALIGNED(16) char m_buf[((L2 + 1 - L1a) / 2 * 2) * sizeof(T)];
};

// A x = B
// initial values in x; returns solution in x
// !! assumes A[i, i] is real !!
inline void GaussSiedel_1(
    std::complex<float>* x,
    const std::complex<float>* A,
    const std::pair<float, float>* A_diag_1,            // 1 / A[i][i]
    const std::complex<float>* B,
    size_t N,
    size_t N_aligned,
    size_t IterCnt)
{
    auto ss = _mm_castsi128_ps(
        _mm_set_epi32(0x80000000, 0, 0x80000000, 0));

    for (size_t iter = 0; iter < IterCnt; iter++)
    {
        for (size_t i = 0; i < N; i++)
        {
            auto Ai = &A[i * N_aligned];

            auto t1 = _mm_setzero_ps();
            auto t2 = _mm_setzero_ps();
            for (size_t j = 0; j < N; j += 2)
            {
                auto a = _mm_loadu_ps((const float*)&Ai[j]);
                auto b = _mm_loadu_ps((const float*)&x[j]);

                t1 = _mm_add_ps(t1, _mm_mul_ps(a, b));
                a = _mm_shuffle_ps(a, a, 0xb1);
                t2 = _mm_add_ps(t2, _mm_mul_ps(a, b));
            }

            t1 = _mm_xor_ps(ss, t1);
#if defined(MAS_SSE3_AVAILABLE)
            auto t = _mm_hadd_ps(t1, t2); //SSE3
            auto s = _mm_hadd_ps(t, t); //SSE3
#else
            __m128 t, s;
            t = _mm_unpacklo_ps(t1, t2);  // s  = t21t11t20t10
            t1 = _mm_unpackhi_ps(t1, t2); // t1 = t23t13t22t12
            t = _mm_add_ps(t, t1);                            // t = t21+t23,t11+t13,t20+t22,t10+t12
            s = _mm_shuffle_ps(t, t, _MM_SHUFFLE(1, 0, 3, 2)); // s = t20+t22,t10+t12,t21+t23,t11+t13
            s = _mm_add_ps(s, t);                     // s = t20+1+2+3,t10+1+2+3,t20+1+2+3,t10+1+2+3
#endif

            auto d = _mm_castpd_ps(_mm_load_sd((const double*)&B[i]));
            d = _mm_sub_ps(d, s);

            // !! assumes A[i, i] is real !!
            auto a_1 = _mm_castpd_ps(_mm_load_sd((const double*)&A_diag_1[i]));
            d = _mm_mul_ps(d, a_1);

            auto u = _mm_castpd_ps(_mm_load_sd((const double*)&x[i]));
            u = _mm_add_ps(u, d);

            _mm_store_sd((double*)&x[i], _mm_castps_pd(u));
            /*
            if (iter == 0 && i == 1)
            if (x[1] != 0.0f)
            {
            printf("gs1: %d: %g+%gi\n", 0, x[0].real(), x[0].imag());
            printf("gs1: %d: %g+%gi\n", 1, x[1].real(), x[1].imag());
            }
            */
        }
    }
}

// A x = B
// initial values in x; returns solution in x
// !! assumes A[i, i] is real !!
inline void GaussSiedel_2(
    std::complex<float>* x,
    const std::complex<float>* A,
    const std::complex<float>* B,
    size_t N,
    size_t IterCnt)
{
    for (size_t iter = 0; iter < IterCnt; iter++)
    {
        for (size_t i = 0; i < N; i += 2)
        {
            auto Ai0 = &A[i * N];
            auto Ai1 = &A[(i + 1) * N];

            auto t0r = _mm_setzero_ps();
            auto t0i = _mm_setzero_ps();
            auto t1r = _mm_setzero_ps();
            auto t1i = _mm_setzero_ps();
            for (size_t j = 0; j < N; j += 2)
            {
                auto a0 = _mm_loadu_ps((const float*)&Ai0[j]);
                auto a1 = _mm_loadu_ps((const float*)&Ai1[j]);

                auto b = _mm_loadu_ps((const float*)&x[j]);
#if defined(MAS_SSE3_AVAILABLE)
                auto br = _mm_moveldup_ps(b); //SSE3
                auto bi = _mm_movehdup_ps(b); //SSE3
#else
                auto br = _mm_shuffle_ps(b, b, _MM_SHUFFLE(2, 2, 0, 0));
                auto bi = _mm_shuffle_ps(b, b, _MM_SHUFFLE(3, 3, 1, 1));
#endif

                t0r = _mm_add_ps(t0r, _mm_mul_ps(a0, br));
                t0i = _mm_add_ps(t0i, _mm_mul_ps(a0, bi));
                t1r = _mm_add_ps(t1r, _mm_mul_ps(a1, br));
                t1i = _mm_add_ps(t1i, _mm_mul_ps(a1, bi));
            }

            auto tr = _mm_add_ps(_mm_shuffle_ps(t0r, t1r, 0x44), _mm_shuffle_ps(t0r, t1r, 0xee));
            auto ti = _mm_add_ps(_mm_shuffle_ps(t0i, t1i, 0x11), _mm_shuffle_ps(t0i, t1i, 0xbb));
#if defined(MAS_SSE3_AVAILABLE)
            auto s = _mm_addsub_ps(tr, ti); //SSE3
#else
            auto s = mm_addsub_ps_sse2(tr, ti);
#endif

            auto b = _mm_loadu_ps((const float*)&B[i]);
            auto b_s = _mm_sub_ps(b, s);

            // Solving equation for
            // A[i0,i0] A[i0,i1] | dx0 = b_s0
            // A[i1,i0] A[i1,i1] | dx1   b_s1
            // where A[i0,i0], A[i1,i1] are real
            //       A[i0,i1] = conj(A[i1,i0])

            auto A0 = _mm_loadu_ps((const float*)&Ai0[i]);
            auto A1 = _mm_loadu_ps((const float*)&Ai1[i]);

            auto ta = _mm_mul_ps(_mm_shuffle_ps(A1, A0, 0x0a), b_s);
            auto tb = _mm_mul_ps(_mm_shuffle_ps(A0, A1, 0x4e), _mm_shuffle_ps(b_s, b_s, 0x0a));
            auto tc = _mm_mul_ps(_mm_shuffle_ps(A1, A0, 0xb1), _mm_shuffle_ps(b_s, b_s, 0x5f));

            auto dd = _mm_sub_ps(ta, _mm_add_ps(tb, tc));

#if defined(MAS_SSE3_AVAILABLE)
            auto ss = _mm_castsi128_ps(_mm_insert_epi32(_mm_setzero_si128(), 0x80000000, 2)); //SSE4.1
            auto det = _mm_dp_ps(_mm_xor_ps(A0, ss), _mm_shuffle_ps(A1, A1, 0x4e), 0xff); //SSE4.1
#else
            auto ss = _mm_castsi128_ps(_mm_set_epi32(0, 0x80000000, 0, 0));
            A0 = _mm_xor_ps(A0, ss);           //A03 -A02 A01 A00
            A1 = _mm_shuffle_ps(A1, A1, 0x4e); //A11  A10 A13 A12
            A0 = _mm_mul_ps(A0, A1); 
            A1 = _mm_shuffle_ps(A0, A0, _MM_SHUFFLE(2, 3, 0, 1));
            A0 = _mm_add_ps(A0, A1);
            A1 = _mm_shuffle_ps(A0, A0, _MM_SHUFFLE(0, 1, 2, 3));
            auto det = _mm_add_ps(A0, A1);
#endif
            auto d1 = _mm_rcp_ps(det);

            auto dx = _mm_mul_ps(dd, d1);

            /*
            {
            auto A00 = Ai0[i]; auto A01 = Ai0[i+1];
            auto A10 = Ai1[i]; auto A11 = Ai1[i+1];
            ALIGNED(16) std::complex<float> BS[2];
            _mm_storeu_ps((float *)&BS, b_s);
            auto det = A00 * A11 - A01 * A10;
            ALIGNED(16) std::complex<float> dxr[2];
            dxr[0] = (BS[0] * A11 - BS[1] * A01) / det;
            dxr[1] = (A00 * BS[1] - A10 * BS[0]) / det;

            if (0)
            dx = _mm_loadu_ps((const float*)&dxr);
            else
            {
            ALIGNED(16) std::complex<float> dxx[2];
            _mm_storeu_ps((float *)&dxx, dx);

            if (dxr[0] != 0.0f)
            printf("dx0: %g+%gi vs. %g+%gi\n", dxr[0].real(), dxr[0].imag(), dxx[0].real(), dxx[0].imag());
            if (dxr[1] != 0.0f)
            printf("dx1: %g+%gi vs. %g+%gi\n", dxr[1].real(), dxr[1].imag(), dxx[1].real(), dxx[1].imag());
            }
            }
            */

            auto u = _mm_loadu_ps((const float*)&x[i]);
            u = _mm_add_ps(u, dx);

            _mm_storeu_ps((float*)&x[i], u);

            /*
            if (iter == 0 && i < 2)
            {
            if (x[1] != 0.0f)
            {
            printf("gs2: %d: %g+%gi\n", 0, x[0].real(), x[0].imag());
            printf("gs2: %d: %g+%gi\n", 1, x[1].real(), x[1].imag());
            }
            }
            */
        }
    }
}

inline void fft_butterfly(
    std::complex<float> *out,
    const std::complex<float> *bu_exp_N,
    size_t N,
    size_t m)
{
    size_t ib = 1;
    size_t s = 1;
    {
        // size_t len = (((size_t)0x1) << s);
        const size_t len = 2;

        auto ss = _mm_castsi128_ps(
            _mm_set_epi32(0x80000000, 0x80000000, 0, 0));
        for (size_t k = 0; k < N /*- 1*/; k += len)
        {
            //for (size_t j = 0; j < len / 2; j++)
            const size_t j = 0;
            {
                // auto t = exp_N[j*N/len] * out[k + j + len / 2];
                // auto u = out[k + j];
                // out[k + j] = u + t;
                // out[k + j + len / 2] = u - t;

                auto ut = _mm_loadu_ps((const float*)&out[k + j]);
                auto unt = _mm_xor_ps(ut, ss);
                auto tu = _mm_shuffle_ps(ut, ut, 0x4e);

                auto x = _mm_add_ps(unt, tu);
                _mm_storeu_ps((float*)&out[k + j], x);
            }
        }
        ib += len / 2;
    }
    for (s = 2; s < m; s++)
    {
        size_t len = ((size_t)0x1) << s;
        size_t len_2 = len >> 1;
        for (size_t k = 0; k < N /*- 1*/; k += len)
        {
            for (size_t j = 0; j < len_2; j += 2)
            {
                // auto t = exp_N[j*N/len] * out[k + j + len / 2];
                // auto u = out[k + j];
                // out[k + j] = u + t;
                // out[k + j + len / 2] = u - t;

                auto a = _mm_loadu_ps((const float*)&bu_exp_N[ib + j]);
                auto b = _mm_loadu_ps((const float*)&out[k + j + len_2]);
                auto t = complex_mul_ps(a, b);

                auto u = _mm_loadu_ps((const float*)&out[k + j]);

                auto x = _mm_add_ps(u, t);
                _mm_storeu_ps((float *)&out[k + j], x);

                auto y = _mm_sub_ps(u, t);
                _mm_storeu_ps((float *)&out[k + j + len_2], y);
            }
        }
        ib += len_2;
    }
}

inline void rfft_convert(
    const std::complex<float>* in,
    const std::complex<float>* tx_N,
    std::complex<float>* buf,
    size_t N)
{
    // for (size_t i = 0; i <= N / 4; i++)
    // {
    //     auto a = std::conj(in[i]);
    //     auto b = in[N / 2 - i];
    //     auto p = tx_N[i] * (a - b);
    //     buf[i] = std::conj(b + p);
    //     buf[N / 2 - i] = a - p;
    // }

    // TODO: what if N = 4?
    rassert_op(N, >= , 8U);
    auto ss = _mm_castsi128_ps(_mm_set_epi32(0x80000000, 0, 0x80000000, 0));
    for (size_t i = 0; i < N / 4; i += 2)
    {
        auto a = _mm_loadu_ps((const float*)&in[i]);
        a = _mm_xor_ps(ss, a);

        auto b = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*)&in[N / 2 - i]));
        b = _mm_loadh_pi(b, (const __m64*)&in[(N / 2 - 1) - i]);

        auto t = _mm_loadu_ps((const float*)&tx_N[i]);
        auto p = complex_mul_ps(t, _mm_sub_ps(a, b));

        auto x1 = _mm_add_ps(b, p);
        x1 = _mm_xor_ps(ss, x1);
        _mm_storeu_ps((float*)&buf[i], x1);

        auto x2 = _mm_sub_ps(a, p);
        _mm_storel_pi((__m64*)&buf[N / 2 - i], x2);
        _mm_storeh_pi((__m64*)&buf[(N / 2 - 1) - i], x2);
    }

    buf[N / 4] = std::conj(in[N / 4]);
}

struct complex4_t
{
    __m128 val[2];
};


inline complex4_t vld2q_f32(const float* p)
{
    return { {_mm_loadu_ps(p), _mm_loadu_ps(p + 4)} };
}

inline void vst2q_f32(float* p, complex4_t x)
{
    _mm_storeu_ps(p, x.val[0]);
    _mm_storeu_ps(p + 4, x.val[1]);
}

inline __m128 vrev64q_f32(__m128 x)
{
    return _mm_shuffle_ps(x, x, 0xb1);
}

inline complex4_t complex4_mul_ps(complex4_t x, complex4_t y)
{
    return { {complex_mul_ps(x.val[0], y.val[0]),
             complex_mul_ps(x.val[1], y.val[1])} };
}

inline __m128 complex_mulni_ps(__m128 a, __m128 b)
{
#if defined(MAS_SSE3_AVAILABLE)
    auto t1 = _mm_mul_ps(_mm_moveldup_ps(a), b); //SSE3
    b = _mm_shuffle_ps(b, b, 0xb1);
    auto t2 = _mm_mul_ps(_mm_movehdup_ps(a), b); //SSE3
    auto s = vrev64q_f32(_mm_addsub_ps(t2, t1)); //SSE3
    return s;
#else
    __m128 c;
    c = a;                           // c = a3a2a1a0
    a = _mm_shuffle_ps(a, a, 0xa0);  // a = a2a2a0a0 == _mm_moveldup_ps(a)
    a = _mm_mul_ps(a, b);            // a = a2b3,a2b2,a0b1,a0b0 == t1
    b = _mm_shuffle_ps(b, b, 0xb1);  // b = b2b3b0b1
    c = _mm_shuffle_ps(c, c, 0xf5);  // c = a3a3a1a1 == _mm_movehdup_ps(a)
    c = _mm_mul_ps(c, b);            // c = a3b2,a3b3,a1b0,a1b1 == t2
    auto s = vrev64q_f32(mm_addsub_ps_sse2(c, a));
    return s;

#endif
}

inline complex4_t complex4_mulni_ps(complex4_t x, complex4_t y)
{
    return { {complex_mulni_ps(x.val[0], y.val[0]),
             complex_mulni_ps(x.val[1], y.val[1])} };
}

inline complex4_t complex4_add_ps(complex4_t x, complex4_t y)
{
    return { {_mm_add_ps(x.val[0], y.val[0]),
             _mm_add_ps(x.val[1], y.val[1])} };
}

inline complex4_t complex4_sub_ps(complex4_t x, complex4_t y)
{
    return { {_mm_sub_ps(x.val[0], y.val[0]),
             _mm_sub_ps(x.val[1], y.val[1])} };
}

inline __m128 complex_subi_ps(__m128 x, __m128 y)
{
#if defined(MAS_SSE3_AVAILABLE)
    return vrev64q_f32(_mm_addsub_ps(vrev64q_f32(x), y)); //SSE3
#else
    return vrev64q_f32(mm_addsub_ps_sse2(vrev64q_f32(x), y));
#endif
}

inline complex4_t complex4_subi_ps(complex4_t x, complex4_t y)
{
    return { {complex_subi_ps(x.val[0], y.val[0]),
             complex_subi_ps(x.val[1], y.val[1])} };
}

inline __m128 complex_addi_ps(__m128 x, __m128 y)
{
#if defined(MAS_SSE3_AVAILABLE)
    return _mm_addsub_ps(x, vrev64q_f32(y)); //SSE3
#else
    return mm_addsub_ps_sse2(x, vrev64q_f32(y));
#endif
}

inline complex4_t complex4_addi_ps(complex4_t x, complex4_t y)
{
    return { {complex_addi_ps(x.val[0], y.val[0]),
             complex_addi_ps(x.val[1], y.val[1])} };
}

template<class TRevIndex>
inline void radix4_fft_first_butterfly(
    std::complex<float>* out,
    const std::complex<float>* in,
    const TRevIndex* rev_table_16,
    size_t Np,
    size_t b,
    size_t dr)
{
    auto in_dr = &in[dr];
    auto in_2dr = &in[2 * dr];
    auto in_3dr = &in[3 * dr];
    switch (b)
    {
    case 1:
        for (size_t k = 0; k < Np; k += 4)
        {
            auto r0 = b * (size_t)rev_table_16[k >> 2];

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
        break;

    case 2:
        for (size_t k = 0; k < Np; k += 4)
        {
            auto r0 = b * (size_t)rev_table_16[k >> 2];
            {
                auto x0 = _mm_loadu_ps((const float*)&in[r0]);
                auto x1 = _mm_loadu_ps((const float*)&in_dr[r0]);
                auto x2 = _mm_loadu_ps((const float*)&in_2dr[r0]);
                auto x3 = _mm_loadu_ps((const float*)&in_3dr[r0]);

                // 8 additions
                auto A = _mm_add_ps(x0, x2);
                auto B = _mm_sub_ps(x0, x2);
                auto C = _mm_add_ps(x1, x3);
                auto D = _mm_sub_ps(x1, x3);

                auto z0 = _mm_add_ps(A, C);
                auto z1 = complex_subi_ps(B, D);
                auto z2 = _mm_sub_ps(A, C);
                auto z3 = complex_addi_ps(B, D);

                _mm_storeu_ps((float*)&out[k],          _mm_shuffle_ps(z0, z1, 0x44));
                _mm_storeu_ps((float*)&out[k + 2],      _mm_shuffle_ps(z2, z3, 0x44));

                _mm_storeu_ps((float*)&out[k + Np],     _mm_shuffle_ps(z0, z1, 0xee));
                _mm_storeu_ps((float*)&out[k + Np + 2], _mm_shuffle_ps(z2, z3, 0xee));
            }
        }
        break;

    default:
#if defined(_MSC_VER)
        __debugbreak();
#elif defined(__GNUC__)
        raise(SIGTRAP);
#endif
        throw std::runtime_error("unimic_runtime error");
    }
}

inline void radix4_fft_butterfly(
    size_t j,
    std::complex<float>* outp,
    const std::complex<float>* bu_exp_N_i0,
    size_t L,
    size_t Np)
{
    auto w1 = vld2q_f32((const float*)&bu_exp_N_i0[j]);
    auto w2 = complex4_mul_ps(w1, w1);

    for (size_t k = 0; k < Np; k += 4 * L)
    {
        auto st = k + j;

        // 4 multiplications
        // 8 additions

        auto x0 =                     vld2q_f32((const float*)&outp[st]);
        auto x1 =                     vld2q_f32((const float*)&outp[st + L]);
        auto x2 = complex4_mul_ps(w2, vld2q_f32((const float*)&outp[st + 2 * L]));
        auto x3 = complex4_mul_ps(w2, vld2q_f32((const float*)&outp[st + 3 * L]));

        auto A =  complex4_add_ps(x0, x2);
        auto B =  complex4_sub_ps(x0, x2);
        auto C =  complex4_mul_ps(w1, complex4_add_ps(x1, x3));
        auto D =  complex4_mul_ps(w1, complex4_sub_ps(x1, x3));

        vst2q_f32((float*)&outp[st],         complex4_add_ps(A, C));
        vst2q_f32((float*)&outp[st + L],     complex4_subi_ps(B, D));
        vst2q_f32((float*)&outp[st + 2 * L], complex4_sub_ps(A, C));
        vst2q_f32((float*)&outp[st + 3 * L], complex4_addi_ps(B, D));
    }
}

inline void radix4_fft_final_butterfly(
    size_t j,
    std::complex<float>* out,
    const std::complex<float>* bu_final_exp_N,
    size_t Np)
{
    auto a = vld2q_f32((const float*)&bu_final_exp_N[j]);
    auto b = vld2q_f32((const float*)&out[Np + j]);
    auto u = vld2q_f32((const float*)&out[j]);

    auto bR = vld2q_f32((const float*)&out[Np + j + Np / 2]);
    auto uR = vld2q_f32((const float*)&out[j + Np / 2]);

    auto t = complex4_mul_ps(a, b);
    auto tR = complex4_mulni_ps(a, bR);

    vst2q_f32((float *)&out[j],               complex4_add_ps(u, t));
    vst2q_f32((float *)&out[j + Np],          complex4_sub_ps(u, t));
    vst2q_f32((float *)&out[j + Np / 2],      complex4_add_ps(uR, tR));
    vst2q_f32((float *)&out[j + Np + Np / 2], complex4_sub_ps(uR, tR));
}

//static void __builtin_prefetch(void* /*addr*/)
//{
//}

