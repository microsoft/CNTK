%module struct_rename

%rename(Bar) Foo;

%inline %{
typedef struct {
    int x;
} Foo;

%}

