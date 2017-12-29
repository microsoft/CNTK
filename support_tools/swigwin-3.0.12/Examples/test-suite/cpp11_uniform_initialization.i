/* This testcase checks whether SWIG syntactically correctly parses the initialization syntax using
   {} braces for uniform member initialization. */
%module cpp11_uniform_initialization

%include <std_vector.i>

%template(VectorInt) std::vector<int>;

%inline %{
struct BasicStruct {
 int x;
 double y;
};
 
struct AltStruct {
  AltStruct(int x, double y) : x_{x}, y_{y} {}
  int getX() { return x_; }
  double getY() { return y_; }
 
private:
  int x_;
  double y_;
};
 
BasicStruct var1{5, 3.2}; // only fills the struct components
AltStruct var2{2, 4.3};   // calls the constructor

class MoreInit
{
public:
  int yarray[5] {1,2,3,4,5};
  char *charptr {nullptr};
  std::vector<int> vi {1,2,3,4,5};

  MoreInit() {}

  int more1(std::vector<int> vv = {1,2,3,4}) {
    int sum = 0;
    for (int i : vv)
      sum += i;
    return sum;
  }
};
const int arr1[] = {1,2,3};
const int arr2[]{1,2,3};
const int arr3[][3]{ {1,2,3}, {4,5,6} };
const int arr4[][3] = { {1,2,3}, {4,5,6} };
%}

