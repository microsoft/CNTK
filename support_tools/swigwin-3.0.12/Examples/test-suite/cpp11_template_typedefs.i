/* This testcase checks whether SWIG correctly handles alias templates. */
%module cpp11_template_typedefs

%inline %{

template<typename T>
using ptr_t = T*;

namespace ns {

template<typename T1, typename T2, int N>
class SomeType {
public:
  using type1_t = T1;
  using type2_t = T2;
  T1 a;
  T2 b;
  constexpr int get_n() { return N; }
};

// Specialization for T1=const char*, T2=bool
template<int N>
class SomeType<const char*, bool, N> {
public:
  using type1_t = const char*;
  using type2_t = bool;
  type1_t a;
  type2_t b;
  constexpr int get_n() { return 3 * N; }
};

// alias templates
template<typename T2>
using TypedefName = SomeType<const char*, T2, 5>;
template<typename T2>
using TypedefNamePtr = ptr_t<SomeType<const char*, T2, 4>>;

// alias template that returns T2 for a SomeType<T1,T2,N> class
template<typename T>
using T2_of = typename T::type2_t;

T2_of<TypedefName<int>> get_SomeType_b(const SomeType<const char*, int, 5>& x) { return x.b; }

template<typename T>
T2_of<TypedefName<T>> get_SomeType_b2(const TypedefName<T>& x) { return x.b; }

} // namespace ns

ns::TypedefName<int> create_TypedefName() { return { "hello", 10}; }
ns::TypedefName<bool> create_TypedefNameBool() { return { "hello", true}; }
ns::TypedefNamePtr<int> identity(ns::TypedefNamePtr<int> a = nullptr) { return a; }

typedef double Val;
template<typename T> struct ListBucket {
};
namespace Alloc {
  template<typename T> struct rebind {
    using other = int;
  };
}

using BucketAllocator1 = typename Alloc::template rebind<ListBucket<Val>>::other;
using BucketAllocator2 = typename Alloc::template rebind<::template ListBucket<double>>::other;

BucketAllocator1 get_bucket_allocator1() { return 1; }
BucketAllocator2 get_bucket_allocator2() { return 2; }
%}

%immutable ns::SomeType::a;

// %template() directives

%template(SomeTypeInt5) ns::SomeType<const char*, int, 5>;
%template(SomeTypeInt4) ns::SomeType<const char*, int, 4>;
%template(SomeTypeBool5) ns::SomeType<const char*, bool, 5>;

%template(ListBucketDouble) ListBucket<Val>;
%template(RebindListBucketDouble) Alloc::rebind<ListBucket<Val>>;

%template() ptr_t<ns::SomeType<const char*, int, 4>>;
%template() ns::TypedefName<int>;
%template() ns::TypedefName<bool>;
%template() ns::TypedefNamePtr<int>;
%template() ns::T2_of<ns::TypedefName<int>>;

%template(get_SomeType_b2) ns::get_SomeType_b2<int>;
