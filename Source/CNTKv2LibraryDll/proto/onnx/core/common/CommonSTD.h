#pragma once
#include <memory>
#include <type_traits>

// to get make_unique definition
#include "Platform.h"

// to add what is missing in gsl 
#if defined(__GNUC__)
namespace std {
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

