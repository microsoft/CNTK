#pragma once
#include <memory>
#include <type_traits>

#if defined(__GNUC__) && !defined(__cpp_lib_make_unique)
namespace std {

    // make_unique was added in GCC 4.9.0. Requires using -std=c++11.
    template <typename T, typename... Args>
    unique_ptr<T> make_unique(Args &&... args)
    {
        return unique_ptr<T>(new T(forward<Args>(args)...));
    }

    template<bool _Test,
        class _Ty = void>
        using enable_if_t = typename enable_if<_Test, _Ty>::type;

    template<class _Ty>
    using remove_cv_t = typename remove_cv<_Ty>::type;

    template<bool _Test,
        class _Ty1,
        class _Ty2>
        using conditional_t = typename conditional<_Test, _Ty1, _Ty2>::type;

    template<class _Ty>
    using add_pointer_t = typename add_pointer<_Ty>::type;

    template<class _Ty>
    using remove_const_t = typename remove_const<_Ty>::type;
}
#else
using std::make_unique;
#endif

