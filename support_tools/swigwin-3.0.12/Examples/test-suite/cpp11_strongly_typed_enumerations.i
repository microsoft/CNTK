// This testcase checks whether SWIG produces the correct wrappers for strongly typed enums.

%module cpp11_strongly_typed_enumerations

%warnfilter(SWIGWARN_PARSE_NAMED_NESTED_CLASS) Class1::Struct1;
%warnfilter(SWIGWARN_PARSE_NAMED_NESTED_CLASS) Class2::Struct1;

/* Forward declarations (illegally accepted by SWIG - oh well!) */
enum Enum1 : short;
enum Enum3;
enum ;
enum : unsigned short;

%inline %{
enum class Enum1 {
  Val1,
  Val2,
  Val3 = 13,
  Val4,
  Val5a = 13,
  Val6a
};

enum class Enum2 : short {
  Val1,
  Val2,
  Val3 = 23,
  Val4,
  Val5b = 23,
  Val6b
};
%}

// SWIG should fail this one
enum Enum2 : unsigned short;     // Illegal in C++11, because Enum2 was previously declared with a different type.

%inline %{
/* Forward declarations. */
enum Enum4 : unsigned int;       // Legal in C++11.
enum class Enum5;                // Legal in C++11, because enum class declarations have a default type of "int".
enum class Enum6 : unsigned int; // Legal C++11.

enum Enum4 : unsigned int {
  Val1, Val2, Val3 = 43, Val4
};

enum class Enum5 {
  Val1, Val2, Val3 = 53, Val4
};

enum class Enum6 : unsigned int {
  Val1, Val2, Val3 = 63, Val4
};

typedef enum class Enum7 : unsigned int {
  Val1, Val2, Val3 = 73, Val4
} Enum7td;

// enum inherits from non-primitive type
enum class Enum8 : size_t {
  Val1, Val2, Val3 = 83, Val4
};

template <typename T> struct TType {
  typedef T type_name;
};

enum class Enum10 : TType<int>::type_name {
  Val1, Val2, Val3 = 103, Val4
};

// forward declaration, no definition of enum
enum class Enum11 : int;
struct UseEnum11 {
    Enum11 myenum11;
};

class Class1
{
public:
  enum class Enum12
  {
    Val1 = 1121,
    Val2 = 1122,
    Val3,
    Val4,
    Val5c = 1121,
    Val6c
  };

  enum Enum13
  {
    Val1 = 1131,
    Val2 = 1132,
    Val3,
    Val4,
    Val5d = 1131,
    Val6d
  };

  enum class Enum14
  {
    Val1 = 1141,
    Val2 = 1142,
    Val3,
    Val4,
    Val5e = 1141,
    Val6e
  };

  struct Struct1
  {
    enum class Enum12
    {
      Val1 = 3121,
      Val2 = 3122,
      Val3,
      Val4,
      Val5f = 3121,
      Val6f
    };

    enum Enum13
    {
      Val1 = 3131,
      Val2 = 3132,
      Val3,
      Val4,
    };

    enum class Enum14
    {
      Val1 = 3141,
      Val2 = 3142,
      Val3,
      Val4,
      Val5g = 3141,
      Val6g
    };
  };
  Enum1 class1Test1(Enum1 e) { return e; }
  Enum12 class1Test2(Enum12 e) { return e; }
  Struct1::Enum12 class1Test3(Struct1::Enum12 e) { return e; }
};

class Class2
{
public:
  enum class Enum12
  {
    Val1 = 2121,
    Val2 = 2122,
    Val3,
    Val4,
    Val5h = 2121,
    Val6h
  };

  enum Enum13
  {
    Val1 = 2131,
    Val2 = 2132,
    Val3,
    Val4,
    Val5i = 2131,
    Val6i
  };

  enum class Enum14
  {
    Val1 = 2141,
    Val2 = 2142,
    Val3,
    Val4,
    Val5j = 2141,
    Val6j
  };

  struct Struct1
  {
    enum class Enum12
    {
      Val1 = 4121,
      Val2 = 4122,
      Val3,
      Val4,
      Val5k = 4121,
      Val6k
    };

    enum Enum13
    {
      Val1 = 4131,
      Val2 = 4132,
      Val3,
      Val4,
      Val5l = 4131,
      Val6l
    };

    enum class Enum14
    {
      Val1 = 4141,
      Val2 = 4142,
      Val3,
      Val4,
      Val5m = 4141,
      Val6m
    };
  };
};

Enum1 globalTest1(Enum1 e) { return e; }
Class1::Enum12 globalTest2(Class1::Enum12 e) { return e; }
Class1::Struct1::Enum12 globalTest3(Class1::Struct1::Enum12 e) { return e; }

class PrivateEnumClass {
private:
  enum class Enum {
    PrivateEnum1,
    PrivateEnum2
  };
};
%}

