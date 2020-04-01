/**
 * utils.h
 *
 * Microsoft Confidential
 */

#pragma once

#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <string>
#include <cmath>
#include <vector>
#include <complex>
#include <wchar.h>

#define rassert_op(x, op, y)                                                   \
    do                                                                         \
    {                                                                          \
        auto __rassert_op_x__ = (x);                                           \
        auto __rassert_op_y__ = (y);                                           \
        if (!(__rassert_op_x__ op __rassert_op_y__))                           \
        {                                                                      \
            std::fprintf(stderr,                                               \
                         "rassert_op (line %d of %s):\n%s %s %s: %s vs. %s\n", \
                         __LINE__,                                             \
                         __FILE__,                                             \
                         #x,                                                   \
                         #op,                                                  \
                         #y,                                                   \
                         std::to_string(__rassert_op_x__).c_str(),             \
                         std::to_string(__rassert_op_y__).c_str());            \
            std::abort();                                                      \
        }                                                                      \
    } while (0)

#define rassert_eq(x, y) rassert_op((x), ==, (y))

#define rassert_eq1(x, y) rassert_eq(x, y)

// Run-time failure
// the process aborts
#define rfail(...)                                                          \
    do                                                                      \
    {                                                                       \
        std::fprintf(stderr, "rfail (line %d of %s):", __LINE__, __FILE__); \
        std::fprintf(stderr, " " __VA_ARGS__);                              \
        std::abort();                                                       \
    } while (0)

#define ALIGNED(x) __declspec(align(x))
#define UNIMIC_ALIGNED_NEW(type)                      \
    void *operator new(size_t s)                      \
    {                                                 \
        return _aligned_malloc((s), __alignof(type)); \
    }                                                 \
    void *operator new[](size_t s)                    \
    {                                                 \
        return _aligned_malloc((s), __alignof(type)); \
    }                                                 \
    void operator delete(void *ptr)                   \
    {                                                 \
        _aligned_free(ptr);                           \
    }                                                 \
    void operator delete[](void *ptr)                 \
    {                                                 \
        _aligned_free(ptr);                           \
    }

static bool _getline(FILE *fp, std::wstring &line)
{
    wchar_t buf[4096];

    line.clear();

    for (;;)
    {
        auto ret = fgetws(buf, _countof(buf), fp);

        if (ret == NULL)
        {
            rassert_op(feof(fp), !=, 0);
            rassert_op(ferror(fp), ==, 0);
            return false;
        }

        if (buf[wcslen(buf) - 1] == '\n' ||
            buf[wcslen(buf) - 1] == '\0') // a line is complete
        {
            buf[wcslen(buf) - 1] = '\0';
            line += buf;

            break;
        }
        else
        {
            line += buf;
        }
    }

    return true;
}

inline errno_t _t_strcpy_s(char *dst, size_t cnt, const char *src)
{
    return strcpy_s(dst, cnt, src);
}

#ifndef LINUXRUNTIMECODE
inline errno_t _t_strcpy_s(wchar_t *dst, size_t cnt, const wchar_t *src)
{
    return wcscpy_s(dst, cnt, src);
}
#endif

#ifdef LINUXRUNTIMECODE
inline void _t_strcpy_s(wchar_t *dst, const wchar_t *src)
{
    wcscpy(dst, src);
}
#endif

inline char *_t_strtok_s_noskip(char *str, const char *dlm, char **ctx)
{
    auto q = str == nullptr ? *ctx : str;
    for (auto p = dlm; *p != 0; p++)
        if (*p == *q)
        {
            *q = 0;
            *ctx = q + 1;
            return q;
        }

    return strtok_s(str, dlm, ctx);
}

inline wchar_t *_t_strtok_s_noskip(wchar_t *str, const wchar_t *dlm, wchar_t **ctx)
{
    auto q = str == nullptr ? *ctx : str;
    for (auto p = dlm; *p != 0; p++)
        if (*p == *q)
        {
            *q = 0;
            *ctx = q + 1;
            return q;
        }

    return wcstok_s(str, dlm, ctx);
}

template <typename TChar>
std::vector<std::basic_string<TChar>>
_split(const std::basic_string<TChar> &str, const TChar *delimits)
{
    std::vector<std::basic_string<TChar>> result;

    auto bufSize = str.size() + 1;
    auto buf = std::make_unique<TChar[]>(bufSize);

    #ifdef LINUXRUNTIMECODE
   _t_strcpy_s(buf.get(), str.c_str());
    #else
    rassert_eq(0, _t_strcpy_s(buf.get(), bufSize, str.c_str()));
    #endif
    TChar *token = nullptr;
    TChar *next_token = nullptr;
    token = _t_strtok_s_noskip(buf.get(), delimits, &next_token);
    while (token != nullptr)
    {
        result.push_back(token);
        token = _t_strtok_s_noskip(nullptr, delimits, &next_token);
    }

    return result;
}
