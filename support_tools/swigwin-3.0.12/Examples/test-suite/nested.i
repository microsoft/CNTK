/*
This testcase tests that nested structs/unions work. Named structs/unions declared within
a struct produced redefinition errors in SWIG 1.3.6 as reported by SF bug #447488.
Also tests reported error when a #define placed in a deeply embedded struct/union.
*/

%module nested


#if defined(SWIGSCILAB)
%rename(OutStNamed) OuterStructNamed;
%rename(InStNamed) OuterStructNamed::InnerStructNamed;
%rename(InUnNamed) OuterStructNamed::Inner_union_named;
#endif

#if defined(SWIG_JAVASCRIPT_V8)

%inline %{
#if __GNUC__ >= 5 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 8)
/* for nested C class wrappers compiled as C++ code */
/* dereferencing type-punned pointer will break strict-aliasing rules [-Werror=strict-aliasing] */
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#endif
%}

#endif

%inline %{

struct TestStruct {
  int a;
};

struct OuterStructNamed {
  struct InnerStructNamed {
    double dd;
  } inner_struct_named;
  union InnerUnionNamed {
    double ee;
    int ff;
  } inner_union_named;
};

%}


#if !defined(SWIGSCILAB)

%inline %{

struct OuterStructUnnamed {
  struct {
    double xx;
  } inner_struct_unnamed;
  union {
    double yy;
    int zz;
  } inner_union_unnamed;
};

typedef struct OuterStruct {
  union {

    struct outer_nested_struct {
      union inner_nested_union {
#define BAD_STYLE 1
        int red;
        struct TestStruct green;
      } InnerNestedUnion;

      struct inner_nested_struct {
        int blue;
      } InnerNestedStruct;
    } OuterNestedStruct;

  } EmbeddedUnion;
} OuterStruct;

%}

#else

%inline %{

struct OutStUnnamed {
  struct {
    double xx;
  } inSt;
  union {
    double yy;
    int zz;
  } inUn;
};

typedef struct OutSt {
  union {

    struct nst_st {
      union in_un {
#define BAD_STYLE 1
        int red;
        struct TestStruct green;
      } InUn;

      struct in_st {
        int blue;
      } InSt;
    } NstdSt;

  } EmbedUn;
} OutSt;

%}

#endif


