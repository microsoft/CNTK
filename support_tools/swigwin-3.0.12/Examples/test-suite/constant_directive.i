%module constant_directive

// %constant and struct

%inline %{
#if defined(_MSC_VER)
  #pragma warning(disable : 4190) // warning C4190: 'result' has C-linkage specified, but returns UDT 'Type1' which is incompatible with C
#endif
struct Type1 {
  Type1(int val = 0) : val(val) {}
  int val;
};
/* Typedefs for const Type and its pointer */
typedef const Type1 Type1Const;
typedef const Type1* Type1Cptr;

/* Typedefs for function pointers returning Type1 */
typedef Type1 (*Type1Fptr)();
typedef Type1 (* const Type1Cfptr)();

/* Function returning an instance of Type1 */
Type1 getType1Instance() { return Type1(111); }
%}

%{
  static Type1 TYPE1_CONSTANT1(1);
  static Type1 TYPE1_CONST2(2);
  static Type1 TYPE1_CONST3(3);
%}

%constant Type1 TYPE1_CONSTANT1;
%constant Type1 TYPE1_CONSTANT2 = TYPE1_CONST2;
%constant Type1 *TYPE1_CONSTANT3 = &TYPE1_CONST3;
/* Typedef'ed types */
%constant Type1Const* TYPE1CONST_CONSTANT1 = &TYPE1_CONSTANT1;
%constant Type1Cptr TYPE1CPTR_CONSTANT1 = &TYPE1_CONSTANT1;
/* Function pointers */
%constant Type1 (*TYPE1FPTR1_CONSTANT1)() = getType1Instance;
%constant Type1 (* const TYPE1CFPTR1_CONSTANT1)() = getType1Instance;
/* Typedef'ed function pointers */
%constant Type1Fptr TYPE1FPTR1DEF_CONSTANT1 = getType1Instance;
%constant Type1Cfptr TYPE1CFPTR1DEF_CONSTANT1 = getType1Instance;
/* Regular constant */
%constant int TYPE_INT = 0;
