%module variable_replacement

%inline %{

class A {
public:
    int a(int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8, int a9, int a10, int a11, int a12)
    {
        return 0;
    }
};

class B : public A {
};

%}
