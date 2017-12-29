%module example

%rename(Bar) Foo;

%inline %{
typedef struct {
    int x;
} Foo;

%}

