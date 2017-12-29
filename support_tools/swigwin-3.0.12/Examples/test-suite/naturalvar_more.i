%module naturalvar_more

// The instantiation of a template using an enum in the template parameter was not picking up %naturalvar.

// These typemaps will be used if %naturalvar is not working
%typemap(out)    T<Space::E> *te, T<Space::E> *const_te "_should_not_use_this_out_typemap_"
%typemap(varout) T<Space::E> *te, T<Space::E> *const_te "_should_not_use_this_varout_typemap_"
%typemap(out)    Hidden *hidden "_should_not_use_this_out_typemap_"
%typemap(varout) Hidden *hidden "_should_not_use_this_varout_typemap_"

%naturalvar T<Space::E>;
%naturalvar Hidden;

%inline %{
template <typename X> struct T {};
struct K {};
struct Hidden;
namespace Ace {
  int glob;
}
%}
%{
struct Hidden {};
namespace Ace {
  template<typename> struct NoIdea {};
}
%}

%inline %{
namespace Space {
  enum E { E1, E2, E3 };
}
%}

%template(TE) T<Space::E>;

%include <std_string.i>
%include <std_vector.i>
%template(VectorString) std::vector<std::string>;

%inline {
using namespace Space;
struct S {
  T<E> te;
  const T<E> const_te;
  const std::vector<std::string>::value_type const_string_member; // check this resolves to std::string which has a naturalvar
  std::vector<std::string>::value_type string_member; // check this resolves to std::string which has a naturalvar
  Hidden hidden;
  Ace::NoIdea<Hidden> noidea;
  S() : const_te(), const_string_member("initial string value") {}
};
}

