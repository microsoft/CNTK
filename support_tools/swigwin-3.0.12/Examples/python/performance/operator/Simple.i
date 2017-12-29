%inline %{
class MyClass {
public:
    MyClass () {}
    ~MyClass () {}
    MyClass& operator+ (int i) { return *this; }
};
%}
