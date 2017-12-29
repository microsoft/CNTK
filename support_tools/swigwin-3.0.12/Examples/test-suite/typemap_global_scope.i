%module typemap_global_scope

// Test global scope operator :: for typemaps. Previously SWIG would not use a typemap that did not specify the global scope
// operator for a type that did have it, and vice-versa.

%typemap(in) SWIGTYPE "_this_will_not_compile_SWIGTYPE_ \"$type\""
%typemap(in) const SWIGTYPE & "_this_will_not_compile_const_SWIGTYPE_REF_ \"$type\""
%typemap(in) enum SWIGTYPE "_this_will_not_compile_enum_SWIGTYPE_ \"$type\""
%typemap(in) const enum SWIGTYPE & "_this_will_not_compile_const_enum_SWIGTYPE_REF_ \"$type\""

/////////////////////////////////////////////////////////////////////
// Structs
/////////////////////////////////////////////////////////////////////

%typemap(in) Test1, ::Test2, Space::Test3, ::Space::Test4 "$1 = $type(); /*in typemap for $type*/"
%typemap(in) const Test1 &, const ::Test2 &, const Space::Test3 &, const ::Space::Test4 & "/*in typemap for $type*/"
%inline %{
struct Test1 {};
struct Test2 {};
namespace Space {
  struct Test3 {};
  struct Test4 {};
}
%}

%inline %{
void test1a(Test1 t, const Test1 &tt) {}
void test1b(::Test1 t, const ::Test1 &tt) {}

void test2a(Test2 t, const Test2 &tt) {}
void test2b(::Test2 t, const ::Test2 &tt) {}

void test3a(Space::Test3 t, const Space::Test3 &tt) {}
void test3b(::Space::Test3 t, const ::Space::Test3 &tt) {}
namespace Space {
  void test3c(Space::Test3 t, const Space::Test3 &tt) {}
  void test3d(::Space::Test3 t, const ::Space::Test3 &tt) {}
  void test3e(Test3 t, const Test3 &tt) {}
}

void test4a(Space::Test4 t, const Space::Test4 &tt) {}
void test4b(::Space::Test4 t, const ::Space::Test4 &tt) {}
namespace Space {
  void test4c(Space::Test4 t, const Space::Test4 &tt) {}
  void test4d(::Space::Test4 t, const ::Space::Test4 &tt) {}
  void test4e(Test4 t, const Test4 &tt) {}
}
%}

/////////////////////////////////////////////////////////////////////
// Templates
/////////////////////////////////////////////////////////////////////

%inline %{
struct XX {};
%}

%typemap(in) TemplateTest1< ::XX >, ::TemplateTest2< ::XX >, Space::TemplateTest3< ::XX >, ::Space::TemplateTest4< ::XX > "$1 = $type(); /* in typemap for $type */"
%typemap(in) const TemplateTest1< XX > &, const ::TemplateTest2< XX > &, const Space::TemplateTest3< XX > &, const ::Space::TemplateTest4< XX > & "/* in typemap for $type */"
%inline %{
template<typename T> struct TemplateTest1 { T m_t; };
template<typename T> struct TemplateTest2 { T m_t; };
namespace Space {
  template<typename T> struct TemplateTest3 { T m_t; };
  template<typename T> struct TemplateTest4 { T m_t; };
}
%}

%template(TemplateTest1XX) TemplateTest1< ::XX >;
%template(TemplateTest2XX) TemplateTest2< ::XX >;
%template(TemplateTest3XX) Space::TemplateTest3< ::XX >;
%template(TemplateTest4XX) Space::TemplateTest4< ::XX >;

%inline %{
void test_template_1a(TemplateTest1< ::XX > t, const TemplateTest1< ::XX > &tt) {}
void test_template_1b(::TemplateTest1< ::XX > t, const ::TemplateTest1< ::XX > &tt) {}

void test_template_2a(TemplateTest2< ::XX > t, const TemplateTest2< ::XX > &tt) {}
void test_template_2b(::TemplateTest2< ::XX > t, const ::TemplateTest2< ::XX > &tt) {}

void test_template_3a(Space::TemplateTest3< ::XX > t, const Space::TemplateTest3< ::XX > &tt) {}
void test_template_3b(::Space::TemplateTest3< ::XX > t, const ::Space::TemplateTest3< ::XX > &tt) {}
namespace Space {
  void test_template_3c(Space::TemplateTest3< ::XX > t, const Space::TemplateTest3< ::XX > &tt) {}
  void test_template_3d(::Space::TemplateTest3< ::XX > t, const ::Space::TemplateTest3< ::XX > &tt) {}
  void test_template_3e(TemplateTest3< ::XX > t, const TemplateTest3< ::XX > &tt) {}
}

void test_template_4a(Space::TemplateTest4< ::XX > t, const Space::TemplateTest4< ::XX > &tt) {}
void test_template_4b(::Space::TemplateTest4< ::XX > t, const ::Space::TemplateTest4< ::XX > &tt) {}
namespace Space {
  void test_template_4c(Space::TemplateTest4< ::XX > t, const Space::TemplateTest4< ::XX > &tt) {}
  void test_template_4d(::Space::TemplateTest4< ::XX > t, const ::Space::TemplateTest4< ::XX > &tt) {}
  void test_template_4e(TemplateTest4< ::XX > t, const TemplateTest4< ::XX > &tt) {}
}
%}

/////////////////////////////////////////////////////////////////////
// Enums
/////////////////////////////////////////////////////////////////////

%typemap(in) Enum1, ::Enum2, Space::Enum3, ::Space::Enum4 "$1 = $1_type(); /*in typemap for $type*/"
%typemap(in) const Enum1 &, const ::Enum2 &, const Space::Enum3 &, const ::Space::Enum4 & "/*in typemap for $type*/"
%inline %{
enum Enum1 { enum_1 };
enum Enum2 { enum_2 };
namespace Space {
  enum Enum3 { enum_3 };
  enum Enum4 { enum_4 };
}
%}

%inline %{
void test_enum_1a(Enum1 t, const Enum1 &tt) {}
void test_enum_1b(::Enum1 t, const ::Enum1 &tt) {}

void test_enum_2a(Enum2 t, const Enum2 &tt) {}
void test_enum_2b(::Enum2 t, const ::Enum2 &tt) {}

void test_enum_3a(Space::Enum3 t, const Space::Enum3 &tt) {}
void test_enum_3b(::Space::Enum3 t, const ::Space::Enum3 &tt) {}
namespace Space {
  void test_enum_3c(Space::Enum3 t, const Space::Enum3 &tt) {}
  void test_enum_3d(::Space::Enum3 t, const ::Space::Enum3 &tt) {}
  void test_enum_3e(Enum3 t, const Enum3 &tt) {}
}

void test_enum_4a(Space::Enum4 t, const Space::Enum4 &tt) {}
void test_enum_4b(::Space::Enum4 t, const ::Space::Enum4 &tt) {}
namespace Space {
  void test_enum_4c(Space::Enum4 t, const Space::Enum4 &tt) {}
  void test_enum_4d(::Space::Enum4 t, const ::Space::Enum4 &tt) {}
  void test_enum_4e(Enum4 t, const Enum4 &tt) {}
}
%}

#if 0
/////////////////////////////////////////////////////////////////////
// Enums with enum specified in typemap
/////////////////////////////////////////////////////////////////////

%typemap(in) enum Mune1, enum ::Mune2, enum Space::Mune3, enum ::Space::Mune4 "/*in typemap for $type*/"
%typemap(in) const enum Mune1 &, const enum ::Mune2 &, const enum Space::Mune3 &, const enum ::Space::Mune4 & "/*in typemap for $type*/"
%inline %{
enum Mune1 { mune_1 };
enum Mune2 { mune_2 };
namespace Space {
  enum Mune3 { mune_3 };
  enum Mune4 { mune_4 };
}
%}

%inline %{
void test_mune_1a(Mune1 t, const Mune1 &tt) {}
void test_mune_1b(::Mune1 t, const ::Mune1 &tt) {}

void test_mune_2a(Mune2 t, const Mune2 &tt) {}
void test_mune_2b(::Mune2 t, const ::Mune2 &tt) {}

void test_mune_3a(Space::Mune3 t, const Space::Mune3 &tt) {}
void test_mune_3b(::Space::Mune3 t, const ::Space::Mune3 &tt) {}
namespace Space {
  void test_mune_3c(Space::Mune3 t, const Space::Mune3 &tt) {}
  void test_mune_3d(::Space::Mune3 t, const ::Space::Mune3 &tt) {}
  void test_mune_3e(Mune3 t, const Mune3 &tt) {}
}

void test_mune_4a(Space::Mune4 t, const Space::Mune4 &tt) {}
void test_mune_4b(::Space::Mune4 t, const ::Space::Mune4 &tt) {}
namespace Space {
  void test_mune_4c(Space::Mune4 t, const Space::Mune4 &tt) {}
  void test_mune_4d(::Space::Mune4 t, const ::Space::Mune4 &tt) {}
  void test_mune_4e(Mune4 t, const Mune4 &tt) {}
}
%}

/////////////////////////////////////////////////////////////////////
// Enums with enum specified in type
/////////////////////////////////////////////////////////////////////

%typemap(in) Nemu1, ::Nemu2, Space::Nemu3, ::Space::Nemu4 "/*in typemap for $type*/"
%typemap(in) const Nemu1 &, const ::Nemu2 &, const Space::Nemu3 &, const ::Space::Nemu4 & "/*in typemap for $type*/"
%inline %{
enum Nemu1 { nemu_1 };
enum Nemu2 { nemu_2 };
namespace Space {
  enum Nemu3 { nemu_3 };
  enum Nemu4 { nemu_4 };
}
%}

%inline %{
void test_nemu_1a(enum Nemu1 t, const enum Nemu1 &tt) {}
void test_nemu_1b(enum ::Nemu1 t, const enum ::Nemu1 &tt) {}

void test_nemu_2a(enum Nemu2 t, const enum Nemu2 &tt) {}
void test_nemu_2b(enum ::Nemu2 t, const enum ::Nemu2 &tt) {}

void test_nemu_3a(enum Space::Nemu3 t, const enum Space::Nemu3 &tt) {}
void test_nemu_3b(enum ::Space::Nemu3 t, const enum ::Space::Nemu3 &tt) {}
namespace Space {
  void test_nemu_3c(enum Space::Nemu3 t, const enum Space::Nemu3 &tt) {}
  void test_nemu_3d(enum ::Space::Nemu3 t, const enum ::Space::Nemu3 &tt) {}
  void test_nemu_3e(enum Nemu3 t, const enum Nemu3 &tt) {}
}

void test_nemu_4a(enum Space::Nemu4 t, const enum Space::Nemu4 &tt) {}
void test_nemu_4b(enum ::Space::Nemu4 t, const enum ::Space::Nemu4 &tt) {}
namespace Space {
  void test_nemu_4c(enum Space::Nemu4 t, const enum Space::Nemu4 &tt) {}
  void test_nemu_4d(enum ::Space::Nemu4 t, const enum ::Space::Nemu4 &tt) {}
  void test_nemu_4e(enum Nemu4 t, const enum Nemu4 &tt) {}
}
%}
#endif
