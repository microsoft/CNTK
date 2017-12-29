%module ruby_naming

%predicate predicateMethod();
%bang bangMethod();

/* This gets mapped to a constant */
%constant int constant1 = 1;

/* This gets mapped to a constant */
#define constant2 2

%immutable TestConstants::constant8;

%inline %{

/* ============  Test Constants Names ============== */

/* This gets mapped to a constant */
#define constant3 3

/* These are all singleton methods */
const int constant4[2] = {10, 20};
const int constant5 = 5;
static const int constant6 = 6;


class TestConstants {
public:
  TestConstants() : constant7(7) {}
  
  /* This gets mapped to a method */
  const int constant7;
  
  /* This gets mapped to a singleton method, but this is not legal C++ */
  static const int constant8;
  
  /* This gets mapped to a method, but this it not legal C++ */
  /*const int constant9 = 9;*/
  
  /* This gets mapped to a constant */
  static const int constant10 = 10;
};

const int TestConstants::constant8 = 8;

const TestConstants * constant11[5];


/* ============  Test Enum ============== */
typedef enum {Red, Green, Blue} Colors;


/* ============  Test Method Names ============== */
class my_class {
public:
	int methodOne()
	{
		return 1;
	}
	
  int MethodTwo()
	{
		return 2;
	}
	
  int Method_THREE()
	{
		return 3;
	}

  int Method44_4()
	{
		return 4;
	}
	
  bool predicateMethod()
	{
		return true;
	}
	
  bool bangMethod()
	{
		return true;
	}
  int begin() 
  {
    return 1;
  }

  int end() 
  {
    return 1;
  }
  
};

%}

%inline 
{
  template <class _Type>
  struct A 
  {
  };
}

%template(A_i) A<int>;
